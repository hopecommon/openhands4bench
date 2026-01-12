from __future__ import annotations

from openhands.core.config.condenser_config import ToolResponseDiscardCondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import CondensationAction
from openhands.events.observation.observation import Observation
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import Condensation, RollingCondenser, View


class ToolResponseDiscardCondenser(RollingCondenser):
    """Condenser that redacts tool response observations on request.

    This is intended for context strategies that drop tool outputs once the
    context window is exceeded (triggered by a condensation request).
    """

    def get_condensation(self, view: View) -> Condensation:
        redacted_count = 0
        for event in view.events:
            if isinstance(event, Observation) and event.tool_call_metadata is not None:
                event.content = 'omitted'
                redacted_count += 1

        total_events = len(view.events)
        discard_ratio = (redacted_count / total_events) if total_events else 0.0

        self.add_metadata('strategy', 'discard_all')
        self.add_metadata('trigger', 'condensation_request')
        self.add_metadata('total_events', total_events)
        self.add_metadata('redacted_tool_responses', redacted_count)
        self.add_metadata('discard_ratio', discard_ratio)

        logger.info(
            'ToolResponseDiscardCondenser: redacted %s tool responses out of %s events',
            redacted_count,
            total_events,
        )

        action = CondensationAction(
            forgotten_event_ids=[],
            metadata={
                'strategy': 'discard_all',
                'trigger': 'condensation_request',
                'total_events': total_events,
                'redacted_tool_responses': redacted_count,
                'discard_ratio': discard_ratio,
            },
        )
        return Condensation(action=action)

    def should_condense(self, view: View) -> bool:
        return view.unhandled_condensation_request

    @classmethod
    def from_config(
        cls,
        _config: ToolResponseDiscardCondenserConfig,
        llm_registry: LLMRegistry,
    ) -> ToolResponseDiscardCondenser:
        return ToolResponseDiscardCondenser()


ToolResponseDiscardCondenser.register_config(ToolResponseDiscardCondenserConfig)
