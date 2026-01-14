from __future__ import annotations

import re

from openhands.core.config.condenser_config import Mem1CondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action.agent import CondensationAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)


class Mem1Condenser(RollingCondenser):
    """A condenser that maintains Mem1-style persistent memory in <think> tags."""

    def __init__(
        self,
        llm: LLM,
        max_size: int | None = None,
        keep_first: int = 1,
        keep_last: int = 4,
        max_event_length: int = 10_000,
    ):
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if keep_last < 0:
            raise ValueError(f'keep_last ({keep_last}) cannot be negative')
        if max_size is not None and max_size < 1:
            raise ValueError(f'max_size ({max_size}) cannot be non-positive')
        if (
            max_size is not None
            and max_size <= keep_first + keep_last + 1
        ):
            raise ValueError(
                'max_size must be greater than keep_first + keep_last + 1'
            )

        self.llm = llm
        self.max_size = max_size
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.max_event_length = max_event_length

        super().__init__()

    def _truncate(self, content: str) -> str:
        return truncate_content(content, max_chars=self.max_event_length)

    def _normalize_think_block(self, summary: str) -> str:
        match = re.search(r'<think>(.*?)</think>', summary, re.DOTALL)
        if match:
            return f"<think>{match.group(1).strip()}</think>"
        return f"<think>{summary.strip()}</think>"

    def _strip_think_block(self, summary: str) -> str:
        match = re.search(r'<think>(.*?)</think>', summary, re.DOTALL)
        if match:
            return match.group(1).strip()
        return summary.strip()

    def get_condensation(self, view: View) -> Condensation:
        events = list(view)
        total_events = len(events)

        keep_indices = set(range(min(self.keep_first, total_events)))
        if self.keep_last > 0:
            tail_start = max(total_events - self.keep_last, 0)
            keep_indices.update(range(tail_start, total_events))

        event_ids_to_forget = set()
        events_to_summarize = []

        for index, event in enumerate(events):
            if isinstance(event, AgentCondensationObservation):
                continue
            if isinstance(event, SystemMessageAction):
                continue

            events_to_summarize.append(event)
            if index in keep_indices:
                continue
            else:
                event_ids_to_forget.add(event.id)

        summary_event = (
            events[self.keep_first]
            if self.keep_first < total_events
            and isinstance(events[self.keep_first], AgentCondensationObservation)
            else AgentCondensationObservation('No previous memory')
        )
        previous_summary = self._strip_think_block(
            summary_event.message if summary_event.message else ''
        )

        prompt = """You are updating a persistent memory summary for a long-horizon agent.
At each step, produce a concise, cumulative memory inside <think>...</think>.
The events below will be discarded after this update, so the <think> memory is
the only persistent state. It must integrate the previous memory with the new
events below. Include essential details needed to continue the task (user goals,
constraints, decisions, tool results, file paths, errors, TODOs). Remove
redundancy and irrelevant details.

Output ONLY the updated <think>...</think> block and nothing else."""

        prompt += '\n\n'
        prompt += (
            f'<PREVIOUS_THINK>\n{self._truncate(previous_summary)}\n</PREVIOUS_THINK>\n'
        )
        prompt += '\n<information>\n'
        for event in events_to_summarize:
            event_content = self._truncate(str(event))
            prompt += f'<EVENT id={event.id}>\n{event_content}\n</EVENT>\n'
        prompt += '</information>\n'
        prompt += 'Return the updated <think> summary now.'

        messages = [Message(role='user', content=[TextContent(text=prompt)])]

        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
            extra_body={'metadata': self.llm_metadata},
        )
        summary = response.choices[0].message.content
        summary = self._normalize_think_block(summary)

        discard_ratio = (
            (len(event_ids_to_forget) / total_events) if total_events else 0.0
        )
        trigger = (
            'condensation_request'
            if view.unhandled_condensation_request
            else 'max_size'
        )

        self.add_metadata('response', response.model_dump())
        self.add_metadata('metrics', self.llm.metrics.get())
        self.add_metadata('strategy', 'mem1')
        self.add_metadata('trigger', trigger)
        self.add_metadata('summary_length', len(summary))
        self.add_metadata('forgotten_event_count', len(event_ids_to_forget))
        self.add_metadata('discard_ratio', discard_ratio)
        self.add_metadata('keep_first', self.keep_first)
        self.add_metadata('keep_last', self.keep_last)

        summary_preview = summary
        if len(summary_preview) > 1000:
            summary_preview = summary_preview[:1000] + '...(truncated)'
        logger.info(
            'Mem1Condenser: summarized %s events into %s chars',
            len(event_ids_to_forget),
            len(summary),
        )
        logger.info('Mem1Condenser summary: %s', summary_preview)

        forgotten_ids = sorted(event_ids_to_forget)
        if not forgotten_ids:
            action = CondensationAction(
                forgotten_event_ids=[],
                summary=summary,
                summary_offset=self.keep_first,
                metadata={
                    'strategy': 'mem1',
                    'trigger': trigger,
                    'summary_length': len(summary),
                    'forgotten_event_count': 0,
                    'discard_ratio': discard_ratio,
                    'keep_first': self.keep_first,
                    'keep_last': self.keep_last,
                },
            )
            return Condensation(action=action)

        contiguous = (
            forgotten_ids[-1] - forgotten_ids[0] == len(forgotten_ids) - 1
        )
        if contiguous:
            action = CondensationAction(
                forgotten_events_start_id=forgotten_ids[0],
                forgotten_events_end_id=forgotten_ids[-1],
                summary=summary,
                summary_offset=self.keep_first,
                metadata={
                    'strategy': 'mem1',
                    'trigger': trigger,
                    'summary_length': len(summary),
                    'forgotten_event_count': len(forgotten_ids),
                    'discard_ratio': discard_ratio,
                    'keep_first': self.keep_first,
                    'keep_last': self.keep_last,
                },
            )
        else:
            action = CondensationAction(
                forgotten_event_ids=forgotten_ids,
                summary=summary,
                summary_offset=self.keep_first,
                metadata={
                    'strategy': 'mem1',
                    'trigger': trigger,
                    'summary_length': len(summary),
                    'forgotten_event_count': len(forgotten_ids),
                    'discard_ratio': discard_ratio,
                    'keep_first': self.keep_first,
                    'keep_last': self.keep_last,
                },
            )

        return Condensation(action=action)

    def should_condense(self, view: View) -> bool:
        return view.unhandled_condensation_request or (
            self.max_size is not None and len(view) > self.max_size
        )

    @classmethod
    def from_config(
        cls, config: Mem1CondenserConfig, llm_registry: LLMRegistry
    ) -> Mem1Condenser:
        llm_config = config.llm_config.model_copy()
        llm_config.caching_prompt = False
        llm = llm_registry.get_llm('condenser', llm_config)

        return Mem1Condenser(
            llm=llm,
            max_size=config.max_size,
            keep_first=config.keep_first,
            keep_last=config.keep_last,
            max_event_length=config.max_event_length,
        )


Mem1Condenser.register_config(Mem1CondenserConfig)
