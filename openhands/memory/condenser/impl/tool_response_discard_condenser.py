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
        """Discard tool responses only before the last assistant action.
        
        This matches the reference implementation's behavior:
        - Find the last assistant action in the view
        - Only discard tool responses BEFORE that last assistant action
        - Keep tool responses AFTER the last assistant action (recent context)
        """
        from openhands.events.action.action import Action
        
        # Find the index of the last assistant action
        last_assistant_idx = -1
        for idx, event in enumerate(view.events):
            if isinstance(event, Action) and event.source and event.source.value == 'agent':
                last_assistant_idx = idx
        
        # If no assistant action found or it's the first event, nothing to discard
        if last_assistant_idx <= 0:
            logger.info('ToolResponseDiscardCondenser: No assistant actions found or assistant is first event, nothing to discard')
            action = CondensationAction(
                forgotten_event_ids=[],
                metadata={
                    'strategy': 'discard_all',
                    'trigger': 'condensation_request',
                    'total_events': len(view.events),
                    'total_tool_responses': 0,
                    'redacted_tool_responses': 0,
                    'already_omitted_tool_responses': 0,
                    'discard_ratio': 0.0,
                },
            )
            return Condensation(action=action)
        
        # Only process events BEFORE the last assistant action
        redacted_count = 0
        already_omitted_count = 0
        total_tool_responses = 0
        
        for event in view.events[:last_assistant_idx]:
            if isinstance(event, Observation) and event.tool_call_metadata is not None:
                total_tool_responses += 1
                if event.content == 'omitted':
                    already_omitted_count += 1
                    continue
                event.content = 'omitted'
                redacted_count += 1

        total_events = len(view.events)
        discard_ratio = (redacted_count / total_events) if total_events else 0.0

        self.add_metadata('strategy', 'discard_all')
        self.add_metadata('trigger', 'condensation_request')
        self.add_metadata('total_events', total_events)
        self.add_metadata('total_tool_responses', total_tool_responses)
        self.add_metadata('redacted_tool_responses', redacted_count)
        self.add_metadata('already_omitted_tool_responses', already_omitted_count)
        self.add_metadata('discard_ratio', discard_ratio)

        logger.info(
            'ToolResponseDiscardCondenser: redacted %s tool responses (%s already omitted) out of %s events',
            redacted_count,
            already_omitted_count,
            total_events,
        )

        action = CondensationAction(
            forgotten_event_ids=[],
            metadata={
                'strategy': 'discard_all',
                'trigger': 'condensation_request',
                'total_events': total_events,
                'total_tool_responses': total_tool_responses,
                'redacted_tool_responses': redacted_count,
                'already_omitted_tool_responses': already_omitted_count,
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
