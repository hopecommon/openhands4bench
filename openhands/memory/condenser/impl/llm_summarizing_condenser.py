from __future__ import annotations

from openhands.core.config.condenser_config import LLMSummarizingCondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action.agent import CondensationAction
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)


class LLMSummarizingCondenser(RollingCondenser):
    """A condenser that summarizes forgotten events.

    Maintains a condensed history and forgets old events when it grows too large,
    keeping a special summarization event after the prefix that summarizes all previous summarizations
    and newly forgotten events.
    """

    def __init__(
        self,
        llm: LLM,
        max_size: int = 100,
        keep_first: int = 1,
        max_event_length: int = 10_000,
        trigger_on_max_size: bool = True,
    ):
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if max_size < 1:
            raise ValueError(f'max_size ({max_size}) cannot be non-positive')

        self.max_size = max_size
        self.keep_first = keep_first
        self.max_event_length = max_event_length
        self.trigger_on_max_size = trigger_on_max_size
        self.llm = llm

        super().__init__()

    def _truncate(self, content: str) -> str:
        """Truncate the content to fit within the specified maximum event length."""
        return truncate_content(content, max_chars=self.max_event_length)

    def get_condensation(self, view: View) -> Condensation:
        head = view[: self.keep_first]
        target_size = self.max_size // 2
        # Number of events to keep from the tail -- target size, minus however many
        # prefix events from the head, minus one for the summarization event
        events_from_tail = target_size - len(head) - 1

        summary_event = AgentCondensationObservation('No events summarized')
        if len(view) > self.keep_first and isinstance(
            view[self.keep_first], AgentCondensationObservation
        ):
            summary_event = view[self.keep_first]

        # Identify events to be forgotten (those not in head or tail)
        tail_stop = -events_from_tail if events_from_tail > 0 else None

        forgotten_events = [
            event
            for event in view[self.keep_first : tail_stop]
            if not isinstance(event, AgentCondensationObservation)
        ]

        # When condensation is requested (typically due to token limits), the view
        # can still be "small" in terms of event count (e.g., a single huge tool
        # output). In that case the size-based slice above can yield no events to
        # forget. We must still handle the request and attempt to reduce context.
        if not forgotten_events and view.unhandled_condensation_request:
            candidates = [
                event
                for event in view[self.keep_first :]
                if not isinstance(event, AgentCondensationObservation)
            ]
            if candidates:
                forgotten_events = [max(candidates, key=lambda e: len(str(e)))]

        # If there is still nothing to forget (e.g., only a system prompt), handle
        # the request with a no-op condensation instead of raising.
        if not forgotten_events:
            trigger = (
                'condensation_request'
                if view.unhandled_condensation_request
                else 'max_size'
            )
            self.add_metadata('strategy', 'summary')
            self.add_metadata('trigger', trigger)
            self.add_metadata('forgotten_event_count', 0)
            self.add_metadata('discard_ratio', 0.0)

            return Condensation(
                action=CondensationAction(
                    forgotten_event_ids=[],
                    summary=None,
                    summary_offset=None,
                    metadata={
                        'strategy': 'summary',
                        'trigger': trigger,
                        'summary_length': 0,
                        'forgotten_event_count': 0,
                        'discard_ratio': 0.0,
                        'note': 'no_forgotten_events',
                    },
                )
            )

        # Construct prompt for summarization
        prompt = """You are maintaining a context-aware state summary for an interactive agent.
You will be given a list of events corresponding to actions taken by the agent, and the most recent previous summary if one exists.
If the events being summarized contain ANY task-tracking, you MUST include a TASK_TRACKING section to maintain continuity.
When referencing tasks make sure to preserve exact task IDs and statuses.

Track:

USER_CONTEXT: (Preserve essential user requirements, goals, and clarifications in concise form)

TASK_TRACKING: {Active tasks, their IDs and statuses - PRESERVE TASK IDs}

COMPLETED: (Tasks completed so far, with brief results)
PENDING: (Tasks that still need to be done)
CURRENT_STATE: (Current variables, data structures, or relevant state)

For code-specific tasks, also include:
CODE_STATE: {File paths, function signatures, data structures}
TESTS: {Failing cases, error messages, outputs}
CHANGES: {Code edits, variable updates}
DEPS: {Dependencies, imports, external calls}
VERSION_CONTROL_STATUS: {Repository state, current branch, PR status, commit history}

PRIORITIZE:
1. Adapt tracking format to match the actual task type
2. Capture key user requirements and goals
3. Distinguish between completed and pending tasks
4. Keep all sections concise and relevant

SKIP: Tracking irrelevant details for the current task type

Example formats:

For code tasks:
USER_CONTEXT: Fix FITS card float representation issue
COMPLETED: Modified mod_float() in card.py, all tests passing
PENDING: Create PR, update documentation
CODE_STATE: mod_float() in card.py updated
TESTS: test_format() passed
CHANGES: str(val) replaces f"{val:.16G}"
DEPS: None modified
VERSION_CONTROL_STATUS: Branch: fix-float-precision, Latest commit: a1b2c3d

For other tasks:
USER_CONTEXT: Write 20 haikus based on coin flip results
COMPLETED: 15 haikus written for results [T,H,T,H,T,H,T,T,H,T,H,T,H,T,H]
PENDING: 5 more haikus needed
CURRENT_STATE: Last flip: Heads, Haiku count: 15/20"""

        prompt += '\n\n'

        # Add the previous summary if it exists. We'll always have a summary
        # event, but the types aren't precise enought to guarantee that it has a
        # message attribute.
        summary_event_content = self._truncate(
            summary_event.message if summary_event.message else ''
        )
        prompt += f'<PREVIOUS SUMMARY>\n{summary_event_content}\n</PREVIOUS SUMMARY>\n'

        prompt += '\n\n'

        # Add all events that are being forgotten. We use the string
        # representation defined by the event, and truncate it if necessary.
        for forgotten_event in forgotten_events:
            event_content = self._truncate(str(forgotten_event))
            prompt += f'<EVENT id={forgotten_event.id}>\n{event_content}\n</EVENT>\n'

        prompt += 'Now summarize the events using the rules above.'

        messages = [Message(role='user', content=[TextContent(text=prompt)])]

        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
            extra_body={'metadata': self.llm_metadata},
        )
        summary = response.choices[0].message.content

        total_events = len(view)
        discard_ratio = (
            (len(forgotten_events) / total_events) if total_events else 0.0
        )
        trigger = (
            'condensation_request'
            if view.unhandled_condensation_request
            else 'max_size'
        )

        self.add_metadata('response', response.model_dump())
        self.add_metadata('metrics', self.llm.metrics.get())
        self.add_metadata('strategy', 'summary')
        self.add_metadata('trigger', trigger)
        self.add_metadata('summary_length', len(summary))
        self.add_metadata('forgotten_event_count', len(forgotten_events))
        self.add_metadata('discard_ratio', discard_ratio)

        summary_preview = summary
        if len(summary_preview) > 1000:
            summary_preview = summary_preview[:1000] + '...(truncated)'
        logger.info(
            'LLMSummarizingCondenser: summarized %s events into %s chars',
            len(forgotten_events),
            len(summary),
        )
        logger.info('LLMSummarizingCondenser summary: %s', summary_preview)

        return Condensation(
            action=CondensationAction(
                forgotten_events_start_id=min(event.id for event in forgotten_events),
                forgotten_events_end_id=max(event.id for event in forgotten_events),
                summary=summary,
                summary_offset=self.keep_first,
                metadata={
                    'strategy': 'summary',
                    'trigger': trigger,
                    'summary_length': len(summary),
                    'forgotten_event_count': len(forgotten_events),
                    'discard_ratio': discard_ratio,
                },
            )
        )

    def should_condense(self, view: View) -> bool:
        if view.unhandled_condensation_request:
            return True
        if not self.trigger_on_max_size:
            return False
        return len(view) > self.max_size

    @classmethod
    def from_config(
        cls, config: LLMSummarizingCondenserConfig, llm_registry: LLMRegistry
    ) -> LLMSummarizingCondenser:
        # This condenser cannot take advantage of prompt caching. If it happens
        # to be set, we'll pay for the cache writes but never get a chance to
        # save on a read.
        llm_config = config.llm_config.model_copy()
        llm_config.caching_prompt = False
        llm = llm_registry.get_llm('condenser', llm_config)

        return LLMSummarizingCondenser(
            llm=llm,
            max_size=config.max_size,
            keep_first=config.keep_first,
            max_event_length=config.max_event_length,
            trigger_on_max_size=config.trigger_on_max_size,
        )


LLMSummarizingCondenser.register_config(LLMSummarizingCondenserConfig)
