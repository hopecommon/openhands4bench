# IMPORTANT: LEGACY V0 CODE
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

import json

from litellm import (
    ModelResponse,
)

from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    CondensationRequestTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.security_utils import RISK_LEVELS
from openhands.core.exceptions import (
    FunctionCallNotExistsError,
    FunctionCallValidationError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    ActionSecurityRisk,
    AgentDelegateAction,
    AgentFinishAction,
    AgentThinkAction,
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
    TaskTrackingAction,
)
from openhands.events.action.agent import CondensationRequestAction
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.tool import ToolCallMetadata
from openhands.llm.tool_names import TASK_TRACKER_TOOL_NAME


def combine_thought(action: Action, thought: str) -> Action:
    if not hasattr(action, 'thought'):
        return action
    if action.thought:
        # Keep tool-provided thought intact; reasoning/content stay on the message.
        return action
    if thought:
        action.thought = thought
    return action


def _extract_text_from_content(content: object) -> str:
    """Best-effort extraction of display text from an OpenAI-compatible content field."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(str(item.get('text', '')))
            elif isinstance(item, str):
                parts.append(item)
        return ''.join(parts)
    if isinstance(content, dict):
        # Some providers may return {"type": "text", "text": "..."}
        if content.get('type') == 'text' and 'text' in content:
            return str(content.get('text', ''))
    return str(content)


def _extract_reasoning_from_message(message: object) -> tuple[bool, str]:
    """Extract provider-specific reasoning content.

    Returns:
        (present, value)

    Where `present` means the provider returned a reasoning field key/attr at all
    (even if the value is empty or null). This lets us preserve empty strings in
    trajectories when the field exists but is empty.
    """
    if message is None:
        return False, ''

    # Prefer the explicit OpenAI-compatible name when present.
    priority_attrs = ('reasoning_content', 'reasoning', 'thinking', 'thought')

    def _normalize(v: object) -> str:
        if v is None:
            return ''
        if isinstance(v, str):
            return v
        return str(v)

    # Attribute-style access (LiteLLM Message object)
    for attr in priority_attrs:
        try:
            has_attr = hasattr(message, attr)
        except Exception:
            has_attr = False
        if has_attr:
            try:
                value = getattr(message, attr, None)
            except Exception:
                value = None
            return True, _normalize(value)

    # Provider-specific fields (common LiteLLM/OpenAI-compatible extension)
    try:
        psf = getattr(message, 'provider_specific_fields', None)
    except Exception:
        psf = None
    if isinstance(psf, dict):
        if 'reasoning_content' in psf:
            return True, _normalize(psf.get('reasoning_content'))

    # Mapping-style access (dict-like)
    if hasattr(message, 'get'):
        try:
            for key in priority_attrs:
                try:
                    # type: ignore[attr-defined]
                    has_key = key in message
                except Exception:
                    has_key = False
                if has_key:
                    # type: ignore[attr-defined]
                    value = message.get(key)
                    return True, _normalize(value)

            # Nested provider_specific_fields for dict-like messages
            try:
                # type: ignore[attr-defined]
                has_psf = 'provider_specific_fields' in message
            except Exception:
                has_psf = False
            if has_psf:
                # type: ignore[attr-defined]
                psf2 = message.get('provider_specific_fields')
                if isinstance(psf2, dict) and 'reasoning_content' in psf2:
                    return True, _normalize(psf2.get('reasoning_content'))
        except Exception:
            pass

    return False, ''


def set_security_risk(action: Action, arguments: dict) -> None:
    """Set the security risk level for the action."""

    # Set security_risk attribute if provided
    if 'security_risk' in arguments:
        if arguments['security_risk'] in RISK_LEVELS:
            if hasattr(action, 'security_risk'):
                action.security_risk = getattr(
                    ActionSecurityRisk, arguments['security_risk']
                )
        else:
            logger.warning(f'Invalid security_risk value: {arguments["security_risk"]}')


def response_to_actions(
    response: ModelResponse, mcp_tool_names: list[str] | None = None
) -> list[Action]:
    actions: list[Action] = []
    assert len(response.choices) == 1, 'Only one choice is supported for now'
    choice = response.choices[0]
    assistant_msg = choice.message
    reasoning_present, reasoning_text = _extract_reasoning_from_message(assistant_msg)
    if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls:
        # Prefer provider-reported reasoning content if present; otherwise fall back to content text.
        content_text = _extract_text_from_content(getattr(assistant_msg, 'content', None))
        thought_parts: list[str] = []
        if reasoning_text:
            thought_parts.append(reasoning_text)
        if content_text and content_text != reasoning_text:
            thought_parts.append(content_text)
        thought = '\n'.join(thought_parts)

        # Process each tool call to OpenHands action
        for i, tool_call in enumerate(assistant_msg.tool_calls):
            action: Action
            logger.debug(f'Tool call in function_calling.py: {tool_call}')
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.decoder.JSONDecodeError as e:
                raise FunctionCallValidationError(
                    f'Failed to parse tool call arguments: {tool_call.function.arguments}'
                ) from e

            # ================================================
            # CmdRunTool (Bash)
            # ================================================

            if tool_call.function.name == create_cmd_run_tool()['function']['name']:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                # convert is_input to boolean
                is_input = arguments.get('is_input', 'false') == 'true'
                action = CmdRunAction(command=arguments['command'], is_input=is_input)

                # Set hard timeout if provided
                if 'timeout' in arguments:
                    try:
                        action.set_hard_timeout(float(arguments['timeout']))
                    except ValueError as e:
                        raise FunctionCallValidationError(
                            f"Invalid float passed to 'timeout' argument: {arguments['timeout']}"
                        ) from e
                set_security_risk(action, arguments)

            # ================================================
            # IPythonTool (Jupyter)
            # ================================================
            elif tool_call.function.name == IPythonTool['function']['name']:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.function.name}'
                    )
                action = IPythonRunCellAction(code=arguments['code'])
                set_security_risk(action, arguments)

            # ================================================
            # AgentDelegateAction (Delegation to another agent)
            # ================================================
            elif tool_call.function.name == 'delegate_to_browsing_agent':
                action = AgentDelegateAction(
                    agent='BrowsingAgent',
                    inputs=arguments,
                )

            # ================================================
            # AgentFinishAction
            # ================================================
            elif tool_call.function.name == FinishTool['function']['name']:
                action = AgentFinishAction(
                    final_thought=arguments.get('message', ''),
                )

            # ================================================
            # LLMBasedFileEditTool (LLM-based file editor, deprecated)
            # ================================================
            elif tool_call.function.name == LLMBasedFileEditTool['function']['name']:
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.function.name}'
                    )
                if 'content' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "content" in tool call {tool_call.function.name}'
                    )
                action = FileEditAction(
                    path=arguments['path'],
                    content=arguments['content'],
                    start=arguments.get('start', 1),
                    end=arguments.get('end', -1),
                    impl_source=arguments.get(
                        'impl_source', FileEditSource.LLM_BASED_EDIT
                    ),
                )
            elif (
                tool_call.function.name
                == create_str_replace_editor_tool()['function']['name']
            ):
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                if 'path' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "path" in tool call {tool_call.function.name}'
                    )
                path = arguments['path']
                command = arguments['command']
                other_kwargs = {
                    k: v for k, v in arguments.items() if k not in ['command', 'path']
                }

                if command == 'view':
                    action = FileReadAction(
                        path=path,
                        impl_source=FileReadSource.OH_ACI,
                        view_range=other_kwargs.get('view_range', None),
                    )
                else:
                    if 'view_range' in other_kwargs:
                        # Remove view_range from other_kwargs since it is not needed for FileEditAction
                        other_kwargs.pop('view_range')

                    # Filter out unexpected arguments
                    valid_kwargs_for_editor = {}
                    # Get valid parameters from the str_replace_editor tool definition
                    str_replace_editor_tool = create_str_replace_editor_tool()
                    valid_params = set(
                        str_replace_editor_tool['function']['parameters'][
                            'properties'
                        ].keys()
                    )

                    for key, value in other_kwargs.items():
                        if key in valid_params:
                            # security_risk is valid but should NOT be part of editor kwargs
                            if key != 'security_risk':
                                valid_kwargs_for_editor[key] = value
                        else:
                            raise FunctionCallValidationError(
                                f'Unexpected argument {key} in tool call {tool_call.function.name}. Allowed arguments are: {valid_params}'
                            )

                    action = FileEditAction(
                        path=path,
                        command=command,
                        impl_source=FileEditSource.OH_ACI,
                        **valid_kwargs_for_editor,
                    )

                set_security_risk(action, arguments)
            # ================================================
            # AgentThinkAction
            # ================================================
            elif tool_call.function.name == ThinkTool['function']['name']:
                action = AgentThinkAction(thought=arguments.get('thought', ''))

            # ================================================
            # CondensationRequestAction
            # ================================================
            elif tool_call.function.name == CondensationRequestTool['function']['name']:
                action = CondensationRequestAction()

            # ================================================
            # BrowserTool
            # ================================================
            elif tool_call.function.name == BrowserTool['function']['name']:
                if 'code' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "code" in tool call {tool_call.function.name}'
                    )
                action = BrowseInteractiveAction(browser_actions=arguments['code'])
                set_security_risk(action, arguments)

            # ================================================
            # TaskTrackingAction
            # ================================================
            elif tool_call.function.name == TASK_TRACKER_TOOL_NAME:
                if 'command' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "command" in tool call {tool_call.function.name}'
                    )
                if arguments['command'] == 'plan' and 'task_list' not in arguments:
                    raise FunctionCallValidationError(
                        f'Missing required argument "task_list" for "plan" command in tool call {tool_call.function.name}'
                    )

                raw_task_list = arguments.get('task_list', [])
                if not isinstance(raw_task_list, list):
                    raise FunctionCallValidationError(
                        f'Invalid format for "task_list". Expected a list but got {type(raw_task_list)}.'
                    )

                # Normalize task_list to ensure it's always a list of dictionaries
                normalized_task_list = []
                for i, task in enumerate(raw_task_list):
                    if isinstance(task, dict):
                        # Task is already in correct format, ensure required fields exist
                        normalized_task = {
                            'id': task.get('id', f'task-{i + 1}'),
                            'title': task.get('title', 'Untitled task'),
                            'status': task.get('status', 'todo'),
                            'notes': task.get('notes', ''),
                        }
                    else:
                        # Unexpected format, raise validation error
                        logger.warning(
                            f'Unexpected task format in task_list: {type(task)} - {task}'
                        )
                        raise FunctionCallValidationError(
                            f'Unexpected task format in task_list: {type(task)}. Each task should be a dictionary.'
                        )
                    normalized_task_list.append(normalized_task)

                action = TaskTrackingAction(
                    command=arguments['command'],
                    task_list=normalized_task_list,
                )

            # ================================================
            # MCPAction (MCP)
            # ================================================
            elif mcp_tool_names and tool_call.function.name in mcp_tool_names:
                action = MCPAction(
                    name=tool_call.function.name,
                    arguments=arguments,
                )
            else:
                raise FunctionCallNotExistsError(
                    f'Tool {tool_call.function.name} is not registered. (arguments: {arguments}). Please check the tool name and retry with an existing tool.'
                )

            # We only add thought to the first action
            if i == 0:
                action = combine_thought(action, thought)
            # Add metadata for tool calling
            action.tool_call_metadata = ToolCallMetadata(
                tool_call_id=tool_call.id,
                function_name=tool_call.function.name,
                model_response=response,
                total_calls_in_response=len(assistant_msg.tool_calls),
            )
            actions.append(action)
    else:
        content_text = _extract_text_from_content(getattr(assistant_msg, 'content', None))
        actions.append(
            MessageAction(
                content=content_text,
                wait_for_response=True,
                reasoning=(reasoning_text if reasoning_present else None),
            )
        )

    # Add response id to actions
    # This will ensure we can match both actions without tool calls (e.g. MessageAction)
    # and actions with tool calls (e.g. CmdRunAction, IPythonRunCellAction, etc.)
    # with the token usage data
    for action in actions:
        action.response_id = response.id

    assert len(actions) >= 1
    return actions
