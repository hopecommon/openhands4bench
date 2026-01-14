# IMPORTANT: LEGACY V0 CODE
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# V1 replacement for this module lives in the Software Agent SDK.
import hashlib
import os
import sys
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.codeact_agent.tools.think import ThinkTool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.config.condenser_config import ToolResponseDiscardCondenserConfig
from openhands.core.exceptions import LLMContextWindowExceedError
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import (
    AgentFinishAction,
    CondensationAction,
    CondensationRequestAction,
    MessageAction,
)
from openhands.events.event import Event
from openhands.events.observation import AgentCondensationObservation, Observation
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager
from openhands.utils.context_snapshot import (
    CONTEXT_SNAPSHOT_PENDING_KEY,
    CONTEXT_SNAPSHOT_VERSION,
    next_snapshot_id,
)


class CodeActAgent(Agent):
    VERSION = '2.2'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents' **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/OpenHands/OpenHands/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self.config.resolved_system_prompt_filename,
            )

        return self._prompt_manager

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # For these models, we use short tool descriptions ( < 1024 tokens)
        # to avoid hitting the OpenAI token limit for tool descriptions.
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-4', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        if self.llm is not None:
            # For historical reasons, previously OpenAI enforces max function description length of 1k characters
            # https://community.openai.com/t/function-call-description-max-length/529902
            # But it no longer seems to be an issue recently
            # https://community.openai.com/t/was-the-character-limit-for-schema-descriptions-upgraded/1225975
            # Tested on GPT-5 and longer description still works. But we still keep the logic to be safe for older models.
            use_short_tool_desc = any(
                model_substr in self.llm.config.model
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools = []
        if self.config.enable_cmd:
            tools.append(create_cmd_run_tool(use_short_description=use_short_tool_desc))
        if self.config.enable_think:
            tools.append(ThinkTool)
        if self.config.enable_finish:
            tools.append(FinishTool)
        if self.config.enable_condensation_request:
            tools.append(CondensationRequestTool)
        if self.config.enable_browsing:
            if sys.platform == 'win32':
                logger.warning('Windows runtime does not support browsing yet')
            else:
                tools.append(BrowserTool)
        if self.config.enable_jupyter:
            tools.append(IPythonTool)
        if self.config.enable_plan_mode:
            # In plan mode, we use the task_tracker tool for task management
            tools.append(create_task_tracker_tool(use_short_tool_desc))
        if self.config.enable_llm_editor:
            tools.append(LLMBasedFileEditTool)
        elif self.config.enable_editor:
            tools.append(
                create_str_replace_editor_tool(
                    use_short_description=use_short_tool_desc,
                    runtime_type=self.config.runtime,
                )
            )
        return tools

    def reset(self) -> None:
        """Resets the CodeAct Agent's internal state."""
        super().reset()
        # Only clear pending actions, not LLM metrics
        self.pending_actions.clear()

    @staticmethod
    def _is_discard_all_config(config: AgentConfig) -> bool:
        strategy = config.context_strategy
        if strategy:
            normalized = strategy.strip().lower().replace('-', '_')
            if normalized in ('discard_all', 'tool_response_discard'):
                return True
        return isinstance(config.condenser, ToolResponseDiscardCondenserConfig)

    @staticmethod
    def _find_last_discard_all_index(history: list[Event]) -> int | None:
        for index in range(len(history) - 1, -1, -1):
            event = history[index]
            if isinstance(event, CondensationAction):
                metadata = event.metadata or {}
                strategy = str(metadata.get('strategy', '')).strip().lower()
                if strategy in ('discard_all', 'tool_response_discard'):
                    return index
        return None

    @staticmethod
    def _has_unredacted_tool_outputs(events: list[Event]) -> bool:
        for event in events:
            if isinstance(event, Observation) and event.tool_call_metadata is not None:
                if event.content != 'omitted':
                    return True
        return False

    def _should_abort_discard_all(self, state: State) -> bool:
        if not self._is_discard_all_config(self.config):
            return False
        discard_index = self._find_last_discard_all_index(state.history)
        if discard_index is None:
            return False
        if self._has_unredacted_tool_outputs(state.history[discard_index + 1 :]):
            return False
        return True

    def step(self, state: State) -> 'Action':
        """Performs one step using the CodeAct Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        - CondensationAction(...) - condense conversation history by forgetting specified events and optionally providing a summary
        - FileReadAction(path, ...) - read file content from specified path
        - FileEditAction(path, ...) - edit file using LLM-based (deprecated) or ACI-based editing
        - AgentThinkAction(thought) - log agent's thought/reasoning process
        - CondensationRequestAction() - request condensation of conversation history
        - BrowseInteractiveAction(browser_actions) - interact with browser using specified actions
        - MCPAction(name, arguments) - interact with MCP server tools
        """
        # Continue with pending actions if any
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        # Track which event IDs have been forgotten/condensed
        forgotten_event_ids: set[int] = set()
        match self.condenser.condensed_history(state):
            case View(events=events, forgotten_event_ids=forgotten_ids):
                condensed_history = events
                forgotten_event_ids = forgotten_ids

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(
            condensed_history, initial_user_message, forgotten_event_ids
        )
        token_count: int | None = None
        context_limit = self.config.context_window_limit_tokens
        if context_limit:
            try:
                token_count = self.llm.get_token_count(messages)
            except Exception as e:
                logger.warning(f'Failed to estimate token count: {e}')
            else:
                if token_count > context_limit:
                    logger.warning(
                        'Context window limit exceeded: %s tokens > %s tokens',
                        token_count,
                        context_limit,
                    )
                    if self.config.enable_history_truncation:
                        if self._should_abort_discard_all(state):
                            raise LLMContextWindowExceedError(
                                f'Context window ({token_count} > {context_limit}) still exceeded after discard_all; terminating.'
                            )
                        return CondensationRequestAction(
                            context_strategy=self.config.context_strategy,
                            token_count=token_count,
                            context_limit=context_limit,
                            trigger='context_window_limit',
                        )
                    raise LLMContextWindowExceedError(
                        f'Conversation exceeds configured context window limit ({token_count} > {context_limit}).'
                    )
        snapshot: dict | None = None
        if self.config.save_context_snapshots:
            snapshot = self._build_context_snapshot(
                state=state,
                condensed_history=condensed_history,
                forgotten_event_ids=forgotten_event_ids,
                messages=messages,
                token_count=token_count,
            )

        params: dict = {
            'messages': messages,
        }
        params['tools'] = check_tools(self.tools, self.llm.config)
        params['extra_body'] = {
            'metadata': state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        response = self.llm.completion(**params)
        if snapshot is not None:
            state.extra_data[CONTEXT_SNAPSHOT_PENDING_KEY] = snapshot
        logger.debug(f'Response from LLM: {response}')
        actions = self.response_to_actions(response)
        logger.debug(f'Actions after response_to_actions: {actions}')
        for action in actions:
            self.pending_actions.append(action)
        return self.pending_actions.popleft()

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Finds the initial user message action from the full history."""
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break

        if initial_user_message is None:
            # This should not happen in a valid conversation
            logger.error(
                f'CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history.'
            )
            # Depending on desired robustness, could raise error or create a dummy action
            # and log the error
            raise ValueError(
                'Initial user message not found in history. Please report this issue.'
            )
        return initial_user_message

    def _get_messages(
        self,
        events: list[Event],
        initial_user_message: MessageAction,
        forgotten_event_ids: set[int],
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand. It handles both regular
        message flow and function-calling scenarios.

        The method performs the following steps:
        1. Checks for SystemMessageAction in events, adds one if missing (legacy support)
        2. Processes events (Actions and Observations) into messages, including SystemMessageAction
        3. Handles tool calls and their responses in function-calling mode
        4. Manages message role alternation (user/assistant/tool)
        5. Applies caching for specific LLM providers (e.g., Anthropic)
        6. Adds environment reminders for non-function-calling mode

        Args:
            events: The list of events to convert to messages

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption, including:
                - System message with prompt (from SystemMessageAction)
                - Action messages (from both user and assistant)
                - Observation messages (including tool responses)
                - Environment reminders (in non-function-calling mode)

        Note:
            - In function-calling mode, tool calls and their responses are carefully tracked
              to maintain proper conversation flow
            - Messages from the same role are combined to prevent consecutive same-role messages
            - For Anthropic models, specific messages are cached according to their documentation
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process events (including SystemMessageAction)
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            forgotten_event_ids=forgotten_event_ids,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _build_context_snapshot(
        self,
        state: State,
        condensed_history: list[Event],
        forgotten_event_ids: set[int],
        messages: list[Message],
        token_count: int | None,
    ) -> dict:
        snapshot_id = next_snapshot_id(state)
        condensed_event_ids = [
            event.id for event in condensed_history if event.id != Event.INVALID_ID
        ]
        omitted_tool_response_event_ids = [
            event.id
            for event in condensed_history
            if isinstance(event, Observation)
            and event.tool_call_metadata is not None
            and event.content == 'omitted'
            and event.id != Event.INVALID_ID
        ]

        summary: str | None = None
        summary_offset: int | None = None
        condensation_action_id: int | None = None
        for event in reversed(state.history):
            if isinstance(event, CondensationAction):
                condensation_action_id = event.id
                summary_offset = event.summary_offset
                if event.summary:
                    summary = event.summary
                break

        if summary is None:
            for event in condensed_history:
                if isinstance(event, AgentCondensationObservation):
                    summary = event.content
                    break

        snapshot: dict = {
            'version': CONTEXT_SNAPSHOT_VERSION,
            'snapshot_id': snapshot_id,
            'created_at': datetime.now().isoformat(),
            'session_id': state.session_id,
            'agent': self.name,
            'context_strategy': self.config.context_strategy,
            'context_window_limit_tokens': self.config.context_window_limit_tokens,
            'max_message_chars': self.llm.config.max_message_chars,
            'vision_is_active': self.llm.vision_is_active(),
            'condensed_event_ids': condensed_event_ids,
            'forgotten_event_ids': sorted(forgotten_event_ids),
            'omitted_tool_response_event_ids': omitted_tool_response_event_ids,
        }

        if condensation_action_id is not None:
            snapshot['condensation_action_id'] = condensation_action_id
        if summary is not None:
            snapshot['summary'] = summary
        if summary_offset is not None:
            snapshot['summary_offset'] = summary_offset
        if token_count is not None:
            snapshot['token_count'] = token_count

        if self.config.save_context_prompt:
            formatted_messages, system_prompt_hash, system_prompt_content = (
                self._format_snapshot_messages(messages)
            )
            snapshot['messages'] = formatted_messages
            if system_prompt_hash:
                snapshot['system_prompt_hash'] = system_prompt_hash
            if system_prompt_content:
                snapshot['system_prompt_content'] = system_prompt_content

        return snapshot

    def _format_snapshot_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict], str | None, str | None]:
        formatted_messages = self.llm.format_messages_for_llm(messages)
        system_prompt_hash: str | None = None
        system_prompt_content: str | None = None
        for message in formatted_messages:
            if message.get('role') != 'system':
                continue
            content = message.get('content', '')
            system_prompt_content = self._extract_text_from_content(content)
            if system_prompt_content:
                system_prompt_hash = hashlib.sha256(
                    system_prompt_content.encode('utf-8')
                ).hexdigest()
                message['content'] = self._replace_system_prompt_content(
                    content, system_prompt_hash
                )
            break
        return formatted_messages, system_prompt_hash, system_prompt_content

    def _extract_text_from_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if 'text' in item:
                        parts.append(str(item.get('text', '')))
            return '\n'.join(parts)
        return ''

    def _replace_system_prompt_content(
        self, content: object, prompt_hash: str
    ) -> str | list[dict]:
        placeholder = f'<system_prompt:{prompt_hash}>'
        if isinstance(content, list):
            return [{'type': 'text', 'text': placeholder}]
        return placeholder

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return codeact_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )
