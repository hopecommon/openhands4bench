import os
import re
import time
import copy
import asyncio
import random
import json

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, List, Tuple

from verl import DataProto
from .utils import Agent, select_env, is_weird, TaskContext, CallAPI, run_action, count_input_tokens
from .prompts import create_chat
from transformers import AutoTokenizer

AUXILIARY_MODEL_PATH = os.getenv("AUXILIARY_MODEL_PATH", None)
MODEL_PATH = os.getenv("MODEL_PATH", "/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/sjxia/models/glm4.5-air-fp8")
AUXILIARY_MODE = os.getenv("AUXILIARY_MODE", False)
DEBUG_CONTEXT = os.getenv("DEBUG_CONTEXT", "False").lower() in ("true", "1", "t")
VOTING_K = int(os.getenv("VOTING_K", "5"))
KEEP_EARLY_INTERACTION = int(os.getenv("KEEP_EARLY_INTERACTION", "1"))

def print_chat(chat):
    """
    Render a chat transcript as a markdown-like string.
    """
    chat_str = ""
    for turn in chat:
        content = turn.get('content', '')
        role = turn.get('role', '')
        if content is None:
            content = ""
            if 'tool_calls' in turn:
                content = str(turn['tool_calls'])

        if is_weird(str(content)):
            chat_str += '# ' + role + ' **CJK**\n\n' + content + "\n\n---\n\n"
        else:
            chat_str += '# ' + role + '\n\n' + content + "\n\n---\n\n"
    return chat_str


def _stringify_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _format_tool_calls(tool_calls) -> str:
    if not tool_calls:
        return ""
    segments = []
    for idx, call in enumerate(tool_calls, start=1):
        function = call.get('function', {}) or {}
        arguments = function.get('arguments')
        arguments_str = _stringify_field(arguments)
        segments.append(
            f"  [{idx}] id={call.get('id', '')} | type={call.get('type', '')} | "
            f"function={function.get('name', '')} | arguments={arguments_str}"
        )
    return "Tool Calls:\n" + "\n".join(segments)


def _group_turns(messages: List[dict]) -> List[List[dict]]:
    groups = []
    current_group = []
    for msg in messages:
        role = msg.get('role')
        if not current_group:
            current_group.append(msg)
            continue
        last_role = current_group[-1].get('role')
        # Group assistant and immediately following tool calls
        if role == 'tool' and (last_role == 'assistant' or last_role == 'tool'):
            current_group.append(msg)
        else:
            groups.append(current_group)
            current_group = [msg]
    if current_group:
        groups.append(current_group)
    return groups


def format_flat_history(messages, start_index: int = 1, include_tool_content: bool = True) -> str:
    groups = _group_turns(messages)
    formatted_turns = []
    
    for idx, group in enumerate(groups, start=start_index):
        lines = [f"Turn {idx}"]
        
        for msg in group:
            role = msg.get('role', 'unknown')
            reasoning = _stringify_field(msg.get('reasoning_content'))
            content = _stringify_field(msg.get('content'))
            
            lines.append(f"Role: {role}")

            if reasoning:
                lines.append(f"Reasoning: {reasoning}")

            if role == 'tool':
                if include_tool_content:
                    tool_payload = json.dumps({
                        "role": "tool",
                        "name": msg.get('name', ''),
                        "content": content,
                        "tool_call_id": msg.get('tool_call_id', '')
                    }, ensure_ascii=False)
                    lines.append(f"Content: {tool_payload}")
                else:
                    tool_payload = json.dumps({
                        "role": "tool",
                        "name": msg.get('name', ''),
                        "content": "omitted"
                    }, ensure_ascii=False)
                    lines.append(f"Content: {tool_payload}")
            else:
                if content:
                    lines.append(f"Content: {content}")

            tool_calls_text = _format_tool_calls(msg.get('tool_calls'))
            if tool_calls_text:
                lines.append(tool_calls_text)
            
            # Add spacing between messages in the same turn
            lines.append("")
        
        # Remove trailing empty line if exists
        if lines and lines[-1] == "":
            lines.pop()

        formatted_turns.append("\n".join(lines))
    return "\n\n".join(formatted_turns)


def format_structured_history(messages, start_turn: int = 1, include_tool_content: bool = True) -> Tuple[str, int]:
    """
    Formats history specifically for the summary prompt.
    Returns:
        - Formatted history string
        - Number of messages in the "Recent Turns" section
    """
    summary_marker = "For this question, you have already made the following progress"
    
    problem_statement_msgs = []
    previous_summary_content = None
    conversation_msgs = []
    
    summary_msg_index = -1
    # Identify structure
    for i, msg in enumerate(messages):
        content = msg.get('content', '')
        if content and summary_marker in str(content) and msg.get('role') == 'user':
            summary_msg_index = i
            break
            
    if summary_msg_index != -1:
        # Found a previous summary injection
        # Structure: [Problem Statement] ... [Empty Asst] [Summary User Msg] [New Turns...]
        # We assume the msg before summary is the Empty Asst.
        limit_idx = max(0, summary_msg_index - 1)
        problem_statement_msgs = messages[:limit_idx]

        # Explicitly remove trailing empty assistant message if it exists (safety check)
        if problem_statement_msgs and problem_statement_msgs[-1].get('role') == 'assistant' and not problem_statement_msgs[-1].get('content'):
            problem_statement_msgs = problem_statement_msgs[:-1]
        
        full_summary_text = messages[summary_msg_index]['content']
        # Extract the core summary. 
        # Expected format: "... summarized as follow:\n\n{summary_block}\n\nNow continue..."
        # summary_block starts with "Summary of previous progress:\n"
        
        try:
            start_marker = "Summary of previous progress:\n"
            end_marker = "\n\nNow continue"
            start_idx = full_summary_text.find(start_marker)
            end_idx = full_summary_text.find(end_marker, start_idx)
            
            if start_idx != -1 and end_idx != -1:
                previous_summary_content = full_summary_text[start_idx + len(start_marker) : end_idx].strip()
            else:
                previous_summary_content = full_summary_text # Fallback
        except:
            previous_summary_content = full_summary_text

        conversation_msgs = messages[summary_msg_index + 1:]
    else:
        # No previous summary (Base case)
        # Find first assistant message to split Prompt vs Conversation
        first_asst_idx = -1
        for i, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                first_asst_idx = i
                break
        
        if first_asst_idx != -1:
            problem_statement_msgs = messages[:first_asst_idx]
            conversation_msgs = messages[first_asst_idx:]
        else:
            problem_statement_msgs = messages
            conversation_msgs = []

    sections = []
    
    # Section 1
    ps_text = []
    for msg in problem_statement_msgs:
        role = msg.get('role', '')
        content = _stringify_field(msg.get('content'))
        ps_text.append(f"[{role}]: {content}")
    sections.append("## Section 1: Problem Statement & User Instruction\n" + "\n\n".join(ps_text))
    
    # Section 2
    if previous_summary_content:
        sections.append("## Section 2: Previous Summary\n" + previous_summary_content)
    
    # Section 3
    # Use format_flat_history for consistency
    conv_text = format_flat_history(conversation_msgs, start_index=start_turn, include_tool_content=include_tool_content)
    
    recent_turns_title = "## Section 3: Recent Conversation Turns" if previous_summary_content else "## Section 2: Recent Conversation Turns"
    sections.append(recent_turns_title + "\n" + conv_text)
    
    # Return count of logical turns
    return "\n\n".join(sections), len(_group_turns(conversation_msgs))


async def simple_llm_call(llm_client, prompt_text, tokenizer, config, agent_size: Optional[int] = None, use_auxiliary: bool = False, timeout: int = 120):
    try:
        temp_history = [{'role': 'user', 'content': prompt_text}]
        temp_agent = Agent(llm_client, temp_history, tokenizer, config, prompt_turn=1, tools=None)
        response, _ = await asyncio.wait_for(temp_agent.step(auxiliary_client=use_auxiliary), timeout=timeout)
        content = response.content if response else ""
        return content
    except asyncio.TimeoutError:
        print(f"[DynamicContext] LLM Call timed out after {timeout}s.")
        return None
    except Exception as e:
        size_note = agent_size if agent_size is not None else "unknown"
        print(f"[DynamicContext] LLM Call failed: {e} agent_size={size_note}")
        return None

# Modified prompt to include 3 specific sections
BLOCK_SUMMARY_PROMPT = """
You are a technical assistant.
Your task is to summarize the progress of a conversation based on the provided history.

The history consists of:
1. Problem Statement & User Instruction
2. Previous Summary (if any)
3. Recent Conversation Turns

**Requirements:**
You must parse the "Recent Conversation Turns" and output the following 3 strictly separated XML sections:

1. **<history_detail>**:
   - If a Previous Summary exists, fold it into a single item like `[Turn 1-3]`, consolidating the context while retaining key details.
   - Parse the "Recent Conversation Turns" (strictly **excluding the very last turn**) by summarizing each turn individually to retain necessary details, using the format `[Turn X]: ...`.

2. **<history_summary>**:
   - Provide a concise summary of the content in `<history_detail>` (i.e., **excluding the last turn**).
   - Focus on what has been accomplished so far.

3. **<future_direction>**:
   - Focus **only on the very last turn** of the conversation.
   - Based on this last turn, suggest the immediate next steps or direction for the agent.

**Input:**
{history}

Respond strictly in XML format as follows:
<history_detail>
[Turn X]: ...
[Turn Y]: ...
</history_detail>

<history_summary>
...
</history_summary>

<future_direction>
...
</future_direction>
"""

# Patterns for extracting the 3 sections
HISTORY_DETAIL_PATTERN = re.compile(r'<history_detail>(.*?)</history_detail>', re.IGNORECASE | re.DOTALL)
HISTORY_SUMMARY_PATTERN = re.compile(r'<history_summary>(.*?)</history_summary>', re.IGNORECASE | re.DOTALL)
FUTURE_DIRECTION_PATTERN = re.compile(r'<future_direction>(.*?)</future_direction>', re.IGNORECASE | re.DOTALL)
REASONING_PATTERN = re.compile(r'<reasoning>(.*?)</reasoning>', re.IGNORECASE | re.DOTALL)

REPLACEMENT_JUDGE_PROMPT = """
You are a Context Manager.
Decide if the current raw conversation history should be replaced by the condensed narrative to save context window space.

{history}

## Policy
Replace context if:
1. A distinct sub-task has been completed.
2. The model is trapped (stuck in a loop or unable to progress).
3. The model has attempted other methods.

Respond strictly in XML:
<reasoning>
Briefly explain your decision based on the policy.
</reasoning>
<decision>YES or NO</decision>
"""

class DynamicContextManager:
    """Manages narrative updates and context replacement decisions."""

    def __init__(self, llm_client, tokenizer, config):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.config = config

    async def summarize_segment(self, messages: List[dict], agent_size: int, start_turn: int = 1) -> Tuple[str, int]:
        # Use new formatting logic for summary
        history_text, recent_count = format_structured_history(messages, start_turn=start_turn, include_tool_content=False)
        
        if DEBUG_CONTEXT:
            print(f"\n[DynamicContext Debug] Summarizing segment with {len(messages)} messages.")
            print(f"[DynamicContext Debug] Formatted History Input:\n{history_text}\n" + "-"*50)

        prompt = BLOCK_SUMMARY_PROMPT.format(history=history_text)
        
        summary_final_text = ""
        
        for attempt in range(3):
            # Use main model for summary
            response = await simple_llm_call(self.llm_client, prompt, self.tokenizer, self.config, agent_size=agent_size, use_auxiliary=False, timeout=300)
            if response:
                detail_match = HISTORY_DETAIL_PATTERN.search(response)
                summary_match = HISTORY_SUMMARY_PATTERN.search(response)
                future_match = FUTURE_DIRECTION_PATTERN.search(response)
                
                if detail_match and summary_match and future_match:
                    h_detail = detail_match.group(1).strip()
                    h_summary = summary_match.group(1).strip()
                    f_future = future_match.group(1).strip()
                    
                    # Construct the composite summary string
                    summary_final_text = (
                        f"## History Details\n{h_detail}\n\n"
                        f"## History Summary\n{h_summary}\n\n"
                        f"## Future Direction\n{f_future}"
                    )
                    break
                else:
                     print(f"[DynamicContext] Summary parse failed (attempt {attempt+1}/3). Response: {response[:100]}...")
            else:
                 print(f"[DynamicContext] Summary failed (empty response) (attempt {attempt+1}/3).")
        
        if not summary_final_text:
             print("[DynamicContext] Failed to summarize segment. Returning raw text description.")
             return "Summary generation failed.", 0
        
        if DEBUG_CONTEXT:
            print(f"[DynamicContext Debug] Generated Summary:\n{summary_final_text}\n" + "="*50 + "\n")
             
        return summary_final_text, recent_count

    async def judge_replacement(self, recent_messages: List[dict], agent_size: int, voting_count: int = 5) -> Tuple[bool, str]:
        recent_text, _ = format_structured_history(recent_messages, start_turn=1, include_tool_content=False)
        prompt = REPLACEMENT_JUDGE_PROMPT.format(
            history=recent_text
        )
        
        valid_votes = 0
        yes_votes = 0
        reasoning_list = []
        # Allow some retries for parse failures
        max_attempts = voting_count * 3
        
        import math
        threshold = math.ceil(voting_count / 2)
        
        # First run parallel attempts
        parallel_tasks = [
            simple_llm_call(self.llm_client, prompt, self.tokenizer, self.config, agent_size=agent_size, use_auxiliary=AUXILIARY_MODE, timeout=60)
            for _ in range(voting_count)
        ]
        parallel_responses = await asyncio.gather(*parallel_tasks)
        
        for i, response in enumerate(parallel_responses):
            if response:
                reasoning_match = REASONING_PATTERN.search(response)
                reasoning_content = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
                
                if "<decision>YES</decision>" in response:
                    valid_votes += 1
                    yes_votes += 1
                    reasoning_list.append(f"[Vote {valid_votes} YES]: {reasoning_content}")
                elif "<decision>NO</decision>" in response:
                    valid_votes += 1
                    reasoning_list.append(f"[Vote {valid_votes} NO]: {reasoning_content}")
                else:
                    print(f"[DynamicContext] Judge replacement parse failed (parallel attempt {i+1}). Response: {response[:100]}...")
            else:
                 print(f"[DynamicContext] Judge replacement failed (empty response) (parallel attempt {i+1}).")

        # If not enough votes, continue serially
        attempts_used = voting_count
        while valid_votes < voting_count and attempts_used < max_attempts:
            attempts_used += 1
            
            # Use auxiliary model for judgment if available
            response = await simple_llm_call(self.llm_client, prompt, self.tokenizer, self.config, agent_size=agent_size, use_auxiliary=AUXILIARY_MODE, timeout=60)
            if response:
                reasoning_match = REASONING_PATTERN.search(response)
                reasoning_content = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
                
                if "<decision>YES</decision>" in response:
                    valid_votes += 1
                    yes_votes += 1
                    reasoning_list.append(f"[Vote {valid_votes} YES]: {reasoning_content}")
                elif "<decision>NO</decision>" in response:
                    valid_votes += 1
                    reasoning_list.append(f"[Vote {valid_votes} NO]: {reasoning_content}")
                else:
                    print(f"[DynamicContext] Judge replacement parse failed (serial attempt {attempts_used}). Response: {response[:100]}...")
            else:
                 print(f"[DynamicContext] Judge replacement failed (empty response) (serial attempt {attempts_used}).")
        
        if valid_votes < voting_count:
            print(f"[DynamicContext] Warning: Only collected {valid_votes}/{voting_count} valid votes. Making decision based on available votes.")
            if valid_votes == 0:
                return False, "No valid votes collected."
            # Adjust threshold for partial results? Or stick to original absolute count?
            # Let's stick to majority of VALID votes if we time out
            threshold = math.ceil(valid_votes / 2)
        # threshold = valid_votes
        if DEBUG_CONTEXT:
            print(f"[Threshold count] {yes_votes}/{threshold}")
        final_decision = (yes_votes >= threshold)
        consolidated_reasoning = "\n".join(reasoning_list)
        
        return final_decision, consolidated_reasoning

class ContextManagementLoop:
    def __init__(self, manager: DynamicContextManager):
        self.manager = manager
        self._tasks = set()

    def start(self):
        pass

    async def stop(self):
        if not self._tasks:
            return
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

def format_messages_for_tokens(agent_instance, tokenizer, tools):
    return count_input_tokens(tokenizer, agent_instance.messages(), tools=tools)


async def process_item(
        item: DataProto,
        context: TaskContext,
        LLMClass=CallAPI,
) -> DataProto:
    os.environ["no_proxy"] = ""
    if AUXILIARY_MODE:
        auxiliary_tokenizer = AutoTokenizer.from_pretrained(AUXILIARY_MODEL_PATH, trust_remote_code=True)
    else:
        auxiliary_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    ability = item.non_tensor_batch['ability'][0]

    EnvClass = select_env(ability, config)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) or getattr(config.plugin, "workflow", "search")

    user_prompt, tools = create_chat(env.instance_info['problem_statement'], workflow, item)

    max_turn = agent_config.get("max_turn", 64)
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    max_traj = getattr(config.plugin, "max_traj", None)

    host = context.server_host
    port = context.server_port

    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn, tools=tools)
    current = 'main'
    session_start_time = time.time()
    iteration = 0
    mask_rollout = True
    session_message = []
    session_message.extend(user_prompt)

    
    context_manager = DynamicContextManager(llm_client, auxiliary_tokenizer, config)
    # context_loop = ContextManagementLoop(context_manager) # Not needed if we don't have background narrative tasks
    
    # Store futures as (iteration_index, future_object, msgs_snapshot)
    judgment_tasks_buffer = []
    
    current_turn_count = 0
    # Track turns for the current branch to manage early interaction trust
    current_branch_turns = 0
    just_replaced = False

    async def judge_job(full_messages, agent_size, turn_idx):
        voting_k = VOTING_K
        should_replace, reasoning = await context_manager.judge_replacement(full_messages, agent_size, voting_count=voting_k)
        if DEBUG_CONTEXT:
            print(f"[DynamicContext Debug] Turn: {turn_idx} | Replacement: {should_replace}")
            print(f"Reasoning: {reasoning}")
        return should_replace, reasoning

    async def _apply_summary_and_replacement(trigger_iter, summary_text, msgs_to_summarize, reasoning=None):
        nonlocal current, session_message, agent, current_branch_turns
        if DEBUG_CONTEXT:
            print(f"[DynamicContext] Applying replacement triggered by iteration {trigger_iter}")
        
        if len(agent) >= max_session:
            print('[SESSION] Context overflow and no session budget left (replacement skipped).')
            return False

        # Create new agent branch
        current = current + '+'
        
        # Rollback session_message to match the end of msgs_to_summarize
        # This removes turns that were executed after the snapshot or explicitly excluded
        if msgs_to_summarize:
            last_kept_msg = msgs_to_summarize[-1]
            # Search backwards for the matching message
            match_index = -1
            for i in range(len(session_message) - 1, -1, -1):
                # We use content comparison as objects might differ (copy vs original)
                # But strict dict equality usually works for simple structures
                if session_message[i] == last_kept_msg:
                    match_index = i
                    break
            
            if match_index != -1:
                # Truncate session_message to keep the matching message and discard subsequent ones
                del session_message[match_index + 1:]
                print(f"[DynamicContext] Rolled back session_message to index {match_index}.")
            else:
                print("[DynamicContext] Warning: Could not find matching message for rollback in session_message.")

        # Construct the new state
        summary_block = f"Summary of previous progress:\n{summary_text}"
        next_session_prompt = (
            f"For this question, you have already made the following progress, "
            f"summarized as follow:\n\n{summary_block}\n\nNow continue work on it.")

        # Initialize new agent with original prompt
        agent[current] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn, tools=tools)

        # Apply the pattern: Assistant(empty) -> User(Summary)
        # This simulates a "pre-fill" or a transition where the assistant acknowledges the reset/summary
        empty_asst_msg = {'role': 'assistant', 'content': ""}
        summary_user_msg = {'role': 'user', 'content': next_session_prompt}

        agent[current].append(empty_asst_msg)
        agent[current].append(summary_user_msg)
        
        # Append last 2 rounds (messages) of the previous session to the new agent
        if msgs_to_summarize:
            last_two_msgs = msgs_to_summarize[-2:]
            for msg in last_two_msgs:
                agent[current].append(msg)

        # Update session_message (Global Log) - PRESERVE HISTORY
        # Re-add user_prompt to signify a new "session" context in the log, mirroring agent state
        session_message.extend(user_prompt)
        
        # Append reasoning to session_message if provided
        if reasoning:
            reasoning_msg = {'role': 'user', 'content': f"<judgement_reasoning>\n{reasoning}\n</judgement_reasoning>"}
            session_message.append(reasoning_msg)

        session_message.append(empty_asst_msg)
        session_message.append(summary_user_msg)
        
        # Sync last 2 messages to session_message
        if msgs_to_summarize:
            last_two_msgs = msgs_to_summarize[-2:]
            for msg in last_two_msgs:
                session_message.append(msg)
        
        # Reset current branch turn counter since we started a new context
        current_branch_turns = 0
        
        return True

    while True:
        # Check context length
        max_len = config.context_length
        input_length = format_messages_for_tokens(agent[current], tokenizer, tools)
        context_limit_reached = (max_len - input_length < 512)
        
        # 1. Check Judgments & Context Limit
        # Only process judgments if NOT just replaced to avoid double-triggers on restart
        if (iteration > 0 or context_limit_reached) and not just_replaced:
            trigger_iter = -1
            msgs_to_summarize = []
            trigger_reasoning = None
            
            # Check accumulated judgments
            if judgment_tasks_buffer:
                try:
                    # judgment_tasks_buffer contains tuples: (iter_num, future, snapshot)
                    tasks = [t[1] for t in judgment_tasks_buffer]
                    results = await asyncio.gather(*tasks)
                    
                    # Find first YES
                    for idx, (is_yes, reasoning) in enumerate(results):
                        if is_yes:
                            trigger_iter = judgment_tasks_buffer[idx][0]
                            # Use full snapshot (Do NOT truncate [:-2] as user requested)
                            # We want the summary to include the "future direction" of the last turn
                            msgs_to_summarize = judgment_tasks_buffer[idx][2]
                            trigger_reasoning = reasoning
                            break
                            
                except Exception as e:
                    print(f"[DynamicContext] Await judgment buffer failed: {e}")
                finally:
                    judgment_tasks_buffer = [] # Clear buffer after checking

            if trigger_iter != -1 or context_limit_reached:
                if context_limit_reached and trigger_iter == -1:
                    print(f"[DynamicContext] Force replacement due to context limit (input_length={input_length}).")
                    trigger_iter = iteration
                    # Use full current messages (Do NOT truncate [:-2])
                    msgs_to_summarize = agent[current].messages()

                # Ensure we have enough content to summarize (more than just prompt)
                if len(msgs_to_summarize) > prompt_turn:
                    # Use current_turn_count + 1 as start
                    summary_text, summarized_count = await context_manager.summarize_segment(
                        msgs_to_summarize, 
                        len(agent), 
                        start_turn=current_turn_count + 1
                    )
                    
                    if not await _apply_summary_and_replacement(trigger_iter, summary_text, msgs_to_summarize, reasoning=trigger_reasoning):
                        if context_limit_reached:
                             print('[SESSION] Context overflow and replacement failed. Stopping.')
                             break
                    else:
                        iteration += 1
                        # Update turn count
                        current_turn_count += summarized_count
                        if DEBUG_CONTEXT:
                            print(f"[DynamicContext] Updated global turn count to {current_turn_count}")
                        just_replaced = True
                else:
                    print(f"[DynamicContext] Not enough messages to summarize for trigger iter {trigger_iter}. Skipping.")
        elif just_replaced:
            just_replaced = False

        if iteration >= max_turn:
            break
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            break

        iteration += 1

        response, finish_reason = await agent[current].step()
        if response is None:
            continue
        
        # Increment turn count for the current branch
        current_branch_turns += 1

        response_dump = response.model_dump()
        session_message.append(response_dump)
        
        if finish_reason == 'tool_calls' and response.tool_calls:
            tool_messages, is_finish = await run_action(env, response.tool_calls)
            for tool_msg in tool_messages:
                agent[current].append(tool_msg)
                session_message.append(tool_msg)
            if is_finish:
                break
        
        # 2. Start Async Judgment
        # Only perform judge if we are past the early interaction trusted window
        if current_branch_turns > KEEP_EARLY_INTERACTION:
            # We judge based on the full history available to the agent so far
            # Capture snapshot NOW
            current_full_messages = copy.deepcopy(agent[current].messages())
            
            j_task = asyncio.create_task(
                judge_job(current_full_messages, len(agent), iteration)
            )
            judgment_tasks_buffer.append((iteration, j_task, current_full_messages))

    # Cleanup
    try:
        if judgment_tasks_buffer:
             tasks = [t[1] for t in judgment_tasks_buffer]
             await asyncio.gather(*tasks)
    except:
        pass
    
    env.stats['session_time'] = time.time() - session_start_time
    print('[TASK] Task Finish, Start Reward')
    
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent[current].messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}
        
    outs = []
    
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    env.stats['total_token'] = len(tokenizer.encode(print_chat(session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(agent) >= max_session)

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    if getattr(config.plugin, "must_finish", None):
        if not is_finish:
            score = ('', 0)

    for name in agent if is_train else ['main']:
        out = await agent[name].dataproto()
        messages = agent[name].messages()

        out = await env.update_dataproto(out, item, messages, score, reward_dict,
                                         tag=name, metrics=agent[name].get_metrics())
        out.batch['is_overlong'] = torch.Tensor([mask_rollout])
        session_message_str = print_chat(session_message)
        out.non_tensor_batch['message_str'] = np.array([session_message_str], dtype=object)
        out.non_tensor_batch['raw_messages'] = np.array([session_message], dtype=object)
        meta_info = f"N: {len(agent)} | {name}"
        out.non_tensor_batch['meta_info'] = np.array([meta_info], dtype=object)

        if env.predicted_answer:
            out.non_tensor_batch['predicted_answer'] = np.array([env.predicted_answer[0]], dtype=object)
        else:
            out.non_tensor_batch['predicted_answer'] = np.array([''], dtype=object)
        out.non_tensor_batch['label_answer'] = np.array([env.label_answer], dtype=object)
        outs.append(copy.deepcopy(out))

    if max_traj is not None and len(outs) > max_traj:
        idx = [0] + sorted(random.sample(range(1, len(outs)), k=max_traj - 1))
        outs = [outs[i] for i in idx]

    try:
        res = DataProto.concat(outs)
        return res
    except Exception as e:
        print(f"[DataProto Error] {e}")
        return