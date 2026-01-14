# IMPORTANT: LEGACY V0 CODE
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from logging import DEBUG
from typing import Any

from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import ModelResponse

from openhands.core.logger import llm_prompt_logger, llm_response_logger
from openhands.core.logger import openhands_logger as logger

MESSAGE_SEPARATOR = '\n\n----------\n\n'


class DebugMixin:
    def log_prompt(self, messages: list[dict[str, Any]] | dict[str, Any]) -> None:
        if not logger.isEnabledFor(DEBUG):
            # Don't use memory building message string if not logging.
            return
        if not messages:
            logger.debug('No completion messages!')
            return

        messages = messages if isinstance(messages, list) else [messages]
        debug_message = MESSAGE_SEPARATOR.join(
            self._format_message_content(msg)
            for msg in messages
            if msg['content'] is not None
        )

        if debug_message:
            llm_prompt_logger.debug(debug_message)
        else:
            logger.debug('No completion messages!')

    def log_response(self, resp: ModelResponse) -> None:
        if not logger.isEnabledFor(DEBUG):
            # Don't use memory building message string if not logging.
            return
        msg = resp['choices'][0].get('message', {})

        reasoning_back = ''
        if isinstance(msg, dict):
            reasoning_back = (
                msg.get('reasoning_content')
                or msg.get('reasoning')
                or msg.get('thinking')
                or msg.get('thought')
                or ''
            )
        else:
            # LiteLLM may return a Message-like object
            for attr in ('reasoning_content', 'reasoning', 'thinking', 'thought'):
                try:
                    val = getattr(msg, attr, None)
                except Exception:
                    val = None
                if isinstance(val, str) and val.strip():
                    reasoning_back = val
                    break

        content_back = ''
        if isinstance(msg, dict):
            content_back = msg.get('content') or ''
        else:
            try:
                content_back = getattr(msg, 'content', '') or ''
            except Exception:
                content_back = ''

        message_back: str = ''
        if reasoning_back:
            message_back += f'[reasoning]\n{reasoning_back}\n'
        message_back += content_back

        tool_calls: list[ChatCompletionMessageToolCall] = []
        if isinstance(msg, dict):
            tool_calls = msg.get('tool_calls', [])
        else:
            tool_calls = getattr(msg, 'tool_calls', []) or []
        if tool_calls:
            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments
                message_back += f'\nFunction call: {fn_name}({fn_args})'

        if message_back:
            llm_response_logger.debug(message_back)

    def _format_message_content(self, message: dict[str, Any]) -> str:
        content = message['content']
        if isinstance(content, list):
            return '\n'.join(
                self._format_content_element(element) for element in content
            )
        return str(content)

    def _format_content_element(self, element: dict[str, Any] | Any) -> str:
        if isinstance(element, dict):
            if 'text' in element:
                return str(element['text'])
            if (
                self.vision_is_active()
                and 'image_url' in element
                and 'url' in element['image_url']
            ):
                return str(element['image_url']['url'])
        return str(element)

    # This method should be implemented in the class that uses DebugMixin
    def vision_is_active(self) -> bool:
        raise NotImplementedError
