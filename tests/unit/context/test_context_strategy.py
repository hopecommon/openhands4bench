import json

from litellm import ModelResponse

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.config.utils import apply_context_strategy
from openhands.core.config.condenser_config import (
    LLMSummarizingCondenserConfig,
    NoOpCondenserConfig,
    ToolResponseDiscardCondenserConfig,
)
from openhands.events.tool import ToolCallMetadata
from openhands.events.observation.commands import CmdOutputObservation
from openhands.memory.condenser.impl.tool_response_discard_condenser import (
    ToolResponseDiscardCondenser,
)
from openhands.memory.view import View


def _create_mock_response() -> ModelResponse:
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'execute_bash',
                                'arguments': json.dumps({'command': 'ls'}),
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def test_apply_context_strategy_react():
    config = OpenHandsConfig()
    agent_config = config.get_agent_config()
    agent_config.context_strategy = 'react'

    apply_context_strategy(config)

    assert isinstance(agent_config.condenser, NoOpCondenserConfig)
    assert agent_config.enable_history_truncation is False


def test_apply_context_strategy_summary():
    config = OpenHandsConfig()
    agent_config = config.get_agent_config()
    agent_config.context_strategy = 'summary'

    apply_context_strategy(config)

    assert isinstance(agent_config.condenser, LLMSummarizingCondenserConfig)
    assert agent_config.enable_history_truncation is True
    assert (
        agent_config.condenser.llm_config.model
        == config.get_llm_config_from_agent_config(agent_config).model
    )


def test_apply_context_strategy_discard_all():
    config = OpenHandsConfig()
    agent_config = config.get_agent_config()
    agent_config.context_strategy = 'discard_all'

    apply_context_strategy(config)

    assert isinstance(agent_config.condenser, ToolResponseDiscardCondenserConfig)
    assert agent_config.enable_history_truncation is True


def test_tool_response_discard_condenser_redacts():
    tool_metadata = ToolCallMetadata(
        function_name='execute_bash',
        tool_call_id='mock-tool-call-id',
        model_response=_create_mock_response(),
        total_calls_in_response=1,
    )

    tool_obs = CmdOutputObservation(content='secret output', command='ls')
    tool_obs.tool_call_metadata = tool_metadata
    non_tool_obs = CmdOutputObservation(content='keep output', command='pwd')

    view = View(events=[tool_obs, non_tool_obs], unhandled_condensation_request=True)
    condenser = ToolResponseDiscardCondenser()
    condensation = condenser.get_condensation(view)

    assert tool_obs.content == 'omitted'
    assert non_tool_obs.content == 'keep output'
    assert condensation.action.forgotten_event_ids == []
