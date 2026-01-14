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

from litellm import acompletion as litellm_acompletion

from openhands.core.exceptions import UserCancelledError
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import (
    LLM,
    LLM_RETRY_EXCEPTIONS,
)
from openhands.llm.model_features import get_features
from openhands.utils.shutdown_listener import should_continue


class AsyncLLM(LLM):
    """Asynchronous LLM class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._async_completion = partial(
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
            seed=self.config.seed,
        )

        async_completion_unwrapped = self._async_completion

        @self.retry_decorator(
            num_retries=self.config.num_retries,
            retry_exceptions=LLM_RETRY_EXCEPTIONS,
            retry_min_wait=self.config.retry_min_wait,
            retry_max_wait=self.config.retry_max_wait,
            retry_multiplier=self.config.retry_multiplier,
        )
        async def async_completion_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for the litellm acompletion function that adds logging and cost tracking."""
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

            # Set reasoning effort for models that support it, only if explicitly provided
            if (
                get_features(self.config.model).supports_reasoning_effort
                and self.config.reasoning_effort is not None
            ):
                kwargs['reasoning_effort'] = self.config.reasoning_effort

            # ensure we work with a list of messages
            messages = messages if isinstance(messages, list) else [messages]

            # if we have no messages, something went very wrong
            if not messages:
                raise ValueError(
                    'The messages list is empty. At least one message is required.'
                )

            self.log_prompt(messages)

            capture_raw_response = os.environ.get(
                'OPENHANDS_CAPTURE_RAW_RESPONSE', ''
            ).strip().lower() in ('1', 'true', 'yes')
            preserve_reasoning_field = os.environ.get(
                'OPENHANDS_PRESERVE_REASONING_FIELD', 'true'
            ).strip().lower() not in ('0', 'false', 'no')
            litellm_logging_obj = None
            if capture_raw_response or preserve_reasoning_field:
                from litellm.litellm_core_utils.cached_imports import (
                    get_litellm_logging_class,
                )

                Logging = get_litellm_logging_class()
                litellm_logging_obj = Logging(
                    model=self.config.model,
                    messages=messages,
                    stream=bool(kwargs.get('stream', False)),
                    call_type='completion',
                    litellm_call_id=str(uuid.uuid4()),
                    start_time=datetime.datetime.now(),
                    function_id=str(uuid.uuid4()),
                    log_raw_request_response=(capture_raw_response or preserve_reasoning_field),
                )
                kwargs['litellm_logging_obj'] = litellm_logging_obj

            async def check_stopped() -> None:
                while should_continue():
                    if (
                        hasattr(self.config, 'on_cancel_requested_fn')
                        and self.config.on_cancel_requested_fn is not None
                        and await self.config.on_cancel_requested_fn()
                    ):
                        return
                    await asyncio.sleep(0.1)

            stop_check_task = asyncio.create_task(check_stopped())

            try:
                # Directly call and await litellm_acompletion
                resp = await async_completion_unwrapped(*args, **kwargs)

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
                    if raw_response is not None and capture_raw_response:
                        try:
                            resp['original_response'] = raw_response
                        except Exception:
                            pass

                    if preserve_reasoning_field and isinstance(raw_response, dict):
                        try:
                            raw_message = raw_response.get('choices', [{}])[0].get(
                                'message', {}
                            )
                        except Exception:
                            raw_message = {}
                        if (
                            isinstance(raw_message, dict)
                            and 'reasoning_content' in raw_message
                        ):
                            raw_reasoning = raw_message.get('reasoning_content')
                            reasoning_value = (
                                '' if raw_reasoning is None else raw_reasoning
                            )
                            try:
                                msg = resp.get('choices', [{}])[0].get('message', {})
                            except Exception:
                                msg = {}
                            if isinstance(msg, dict):
                                current_reasoning = msg.get('reasoning_content')

                                # Raw is authoritative when present, but avoid overwriting a
                                # non-empty value with an empty one.
                                should_set = False
                                if 'reasoning_content' not in msg:
                                    should_set = True
                                elif current_reasoning in (None, ''):
                                    should_set = True
                                elif reasoning_value != '' and current_reasoning != reasoning_value:
                                    should_set = True

                                if should_set:
                                    msg['reasoning_content'] = reasoning_value
                                psf = msg.get('provider_specific_fields') or {}
                                if isinstance(psf, dict):
                                    if (
                                        'reasoning_content' not in psf
                                        or psf.get('reasoning_content') in (None, '')
                                    ):
                                        psf['reasoning_content'] = msg.get(
                                            'reasoning_content'
                                        )
                                    msg['provider_specific_fields'] = psf

                # log the LLM response (best-effort)
                try:
                    self.log_response(resp)
                except Exception:
                    pass

                # log costs and tokens used
                self._post_completion(resp)

                # We do not support streaming in this method, thus return resp
                return resp

            except UserCancelledError:
                logger.debug('LLM request cancelled by user.')
                raise
            except Exception as e:
                logger.error(f'Completion Error occurred:\n{e}')
                raise

            finally:
                await asyncio.sleep(0.1)
                stop_check_task.cancel()
                try:
                    await stop_check_task
                except asyncio.CancelledError:
                    pass

        self._async_completion = async_completion_wrapper

    async def _call_acompletion(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper for the litellm acompletion function."""
        # Used in testing?
        return await litellm_acompletion(*args, **kwargs)

    @property
    def async_completion(self) -> Callable:
        """Decorator for the async litellm acompletion function."""
        return self._async_completion
