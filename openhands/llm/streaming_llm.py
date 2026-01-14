# IMPORTANT: LEGACY V0 CODE
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import asyncio
from functools import partial
from typing import Any, Callable

import datetime
import json
import os
import uuid

from openhands.core.exceptions import UserCancelledError
from openhands.core.logger import openhands_logger as logger
from openhands.llm.async_llm import LLM_RETRY_EXCEPTIONS, AsyncLLM
from openhands.llm.model_features import get_features


class StreamingLLM(AsyncLLM):
    """Streaming LLM class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._async_streaming_completion = partial(
            self._call_acompletion,
            model=self.config.model,
            api_key=self.config.api_key.get_secret_value()
            if self.config.api_key
            else None,
            base_url=self.config.base_url,
            api_version=self.config.api_version,
            custom_llm_provider=self.config.custom_llm_provider,
            max_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            drop_params=self.config.drop_params,
            stream=True,  # Ensure streaming is enabled
        )

        async_streaming_completion_unwrapped = self._async_streaming_completion

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
        )
        async def async_streaming_completion_wrapper(*args: Any, **kwargs: Any) -> Any:
            messages: list[dict[str, Any]] | dict[str, Any] = []

            # some callers might send the model and messages directly
            # litellm allows positional args, like completion(model, messages, **kwargs)
            # see llm.py for more details
            if len(args) > 1:
                messages = args[1] if len(args) > 1 else args[0]
                kwargs['messages'] = messages

                # remove the first args, they're sent in kwargs
                args = args[2:]
            elif 'messages' in kwargs:
                messages = kwargs['messages']

            # ensure we work with a list of messages
            messages = messages if isinstance(messages, list) else [messages]

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            # Set reasoning effort for models that support it, only if explicitly provided
            if (
                get_features(self.config.model).supports_reasoning_effort
                and self.config.reasoning_effort is not None
            ):
                kwargs['reasoning_effort'] = self.config.reasoning_effort

            self.log_prompt(messages)

            capture_raw_response = os.environ.get(
                'OPENHANDS_CAPTURE_RAW_RESPONSE', ''
            ).strip().lower() in ('1', 'true', 'yes')
            preserve_reasoning_field = os.environ.get(
                'OPENHANDS_PRESERVE_REASONING_FIELD', 'true'
            ).strip().lower() not in ('0', 'false', 'no')
            litellm_logging_obj = kwargs.get('litellm_logging_obj')
            if (
                (capture_raw_response or preserve_reasoning_field)
                and litellm_logging_obj is None
            ):
                from litellm.litellm_core_utils.cached_imports import (
                    get_litellm_logging_class,
                )

                Logging = get_litellm_logging_class()
                litellm_logging_obj = Logging(
                    model=self.config.model,
                    messages=messages,
                    stream=True,
                    call_type='completion',
                    litellm_call_id=str(uuid.uuid4()),
                    start_time=datetime.datetime.now(),
                    function_id=str(uuid.uuid4()),
                    log_raw_request_response=(capture_raw_response or preserve_reasoning_field),
                )
                kwargs['litellm_logging_obj'] = litellm_logging_obj

            try:
                # Directly call and await litellm_acompletion
                resp = await async_streaming_completion_unwrapped(*args, **kwargs)

                # For streaming we iterate over the chunks
                async for chunk in resp:
                    if litellm_logging_obj is not None:
                        original_response = litellm_logging_obj.model_call_details.get(
                            'original_response'
                        )
                        raw_response = original_response
                        if isinstance(original_response, str):
                            try:
                                raw_response = json.loads(original_response)
                            except Exception:
                                raw_response = original_response

                    # Check for cancellation before yielding the chunk
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        raise UserCancelledError(
                            'LLM request cancelled due to CANCELLED state'
                        )

                    # When the stream finishes, backfill reasoning content and raw response.
                    is_final = (
                        chunk.get('choices', [{}])[0].get('finish_reason')
                        is not None
                    )
                    if is_final and raw_response is not None:
                        if capture_raw_response:
                            chunk['original_response'] = raw_response
                        if preserve_reasoning_field and isinstance(raw_response, dict):
                            raw_message = raw_response.get('choices', [{}])[0].get(
                                'message', {}
                            )
                            if (
                                isinstance(raw_message, dict)
                                and 'reasoning_content' in raw_message
                            ):
                                raw_reasoning = raw_message.get('reasoning_content')
                                reasoning_value = (
                                    '' if raw_reasoning is None else raw_reasoning
                                )
                                delta = chunk.get('choices', [{}])[0].get('delta')
                                if isinstance(delta, dict):
                                    current_reasoning = delta.get('reasoning_content')
                                    should_set = False
                                    if 'reasoning_content' not in delta:
                                        should_set = True
                                    elif current_reasoning in (None, ''):
                                        should_set = True
                                    elif reasoning_value != '' and current_reasoning != reasoning_value:
                                        should_set = True

                                    if should_set:
                                        delta['reasoning_content'] = reasoning_value

                    # with streaming, it is "delta", not "message"!
                    message_back = chunk['choices'][0]['delta'].get('content', '')
                    if message_back:
                        self.log_response(message_back)
                    self._post_completion(chunk)

                    yield chunk

            except UserCancelledError:
                logger.debug('LLM request cancelled by user.')
                raise
            except Exception as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                # sleep for 0.1 seconds to allow the stream to be flushed
                if kwargs.get('stream', False):
                    await asyncio.sleep(0.1)

        self._async_streaming_completion = async_streaming_completion_wrapper

    @property
    def async_streaming_completion(self) -> Callable:
        """Decorator for the async litellm acompletion function with streaming."""
        return self._async_streaming_completion
