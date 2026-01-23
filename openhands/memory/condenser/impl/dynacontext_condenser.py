from __future__ import annotations

import json
import math
import re

from openhands.core.config.condenser_config import DynaContextCondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action.action import Action
from openhands.events.action.agent import CondensationAction, CondensationRequestAction
from openhands.events.action.message import MessageAction
from openhands.events.event import EventSource
from openhands.events.serialization.event import event_to_dict, truncate_content
from openhands.llm.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import Condensation, RollingCondenser, View

SUMMARY_PATTERN = re.compile(r'<summary>(.*?)</summary>', re.IGNORECASE | re.DOTALL)
REASONING_PATTERN = re.compile(
    r'<reasoning>(.*?)</reasoning>', re.IGNORECASE | re.DOTALL
)
DECISION_PATTERN = re.compile(
    r'<decision>\s*(YES|NO)\s*</decision>', re.IGNORECASE | re.DOTALL
)

BLOCK_SUMMARY_PROMPT = """
You are a technical assistant.
Summarize the following conversation segment into a concise narrative.
Focus on the actions taken, tools used, and key results obtained.
Preserve important information needed for future steps.

Segment:
{history}

Respond strictly in XML:
<summary>
... (The summary content) ...
</summary>
""".strip()

REPLACEMENT_JUDGE_PROMPT = """
You are a Context Manager.
Decide if the current raw conversation history should be replaced by the condensed narrative to save context window space.

{history}

## Policy
Replace context if:
1. A distinct sub-task has been completed.
2. The model is trapped (stuck in a loop or unable to progress).
3. The model has attempted other methods.

Respond strictly in XML:
<reasoning>
Briefly explain your decision based on the policy.
</reasoning>
<decision>YES or NO</decision>
""".strip()


class DynaContextCondenser(RollingCondenser):
    """A condenser that uses judge voting to decide when to summarize context."""

    def __init__(
        self,
        llm: LLM,
        judge_llm: LLM,
        voting_k: int = 3,
        keep_first: int = 1,
        keep_last: int = 0,
        early_turns: int = 1,
        max_event_length: int = 10_000,
        exclude_tail_max: int = 0,
    ):
        if voting_k < 1:
            raise ValueError(f'voting_k ({voting_k}) must be >= 1')
        if keep_first < 0:
            raise ValueError(f'keep_first ({keep_first}) cannot be negative')
        if keep_last < 0:
            raise ValueError(f'keep_last ({keep_last}) cannot be negative')
        if early_turns < 0:
            raise ValueError(f'early_turns ({early_turns}) cannot be negative')
        if exclude_tail_max < 0:
            raise ValueError(
                f'exclude_tail_max ({exclude_tail_max}) cannot be negative'
            )

        self.llm = llm
        self.judge_llm = judge_llm
        self.voting_k = voting_k
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.early_turns = early_turns
        self.max_event_length = max_event_length
        self.exclude_tail_max = exclude_tail_max

        self._last_seen_event_id = -1
        self._turns_since_reset = 0
        self._pending_judge_metadata: dict | None = None
        self._last_skip_reason: str | None = None
        self._last_judge_metadata: dict | None = None

        super().__init__()

    def _truncate(self, content: str) -> str:
        return truncate_content(content, max_chars=self.max_event_length)

    def _format_events_for_manager(
        self,
        events: list,
        include_tool_content: bool = True,
        start_index: int = 1,
    ) -> str:
        formatted_turns: list[str] = []
        for idx, event in enumerate(events, start=start_index):
            data = event_to_dict(event)
            if (
                not include_tool_content
                and event.tool_call_metadata is not None
                and 'content' in data
            ):
                data['content'] = 'omitted'
            payload = self._truncate(json.dumps(data, ensure_ascii=False))
            formatted_turns.append(f'Turn {idx}\n{payload}')
        return '\n\n'.join(formatted_turns)

    def _count_new_agent_turns(self, events: list) -> None:
        latest_seen = self._last_seen_event_id
        turns_added = 0
        for event in events:
            if event.id <= latest_seen:
                continue
            if isinstance(event, Action):
                if event.source and event.source.value == 'agent':
                    if isinstance(event, CondensationRequestAction):
                        continue
                    turns_added += 1
        if events:
            self._last_seen_event_id = max(self._last_seen_event_id, events[-1].id)
        self._turns_since_reset += turns_added

    def _should_judge(self) -> bool:
        return self._turns_since_reset > self.early_turns

    def _tail_exclusion_count(self, events: list) -> int:
        count = 0
        for event in reversed(events):
            if count >= self.exclude_tail_max:
                break
            if isinstance(event, MessageAction) and event.source == EventSource.USER:
                break
            if (
                isinstance(event, Action)
                and event.source
                and event.source.value == 'agent'
            ) or event.tool_call_metadata is not None:
                count += 1
                continue
            break
        return count

    def _summary_candidates(
        self, events: list, *, hard_trigger: bool
    ) -> list:
        total_events = len(events)
        keep_indices = set(range(min(self.keep_first, total_events)))
        if self.keep_last > 0:
            tail_start = max(total_events - self.keep_last, 0)
            keep_indices.update(range(tail_start, total_events))

        exclude_tail = 0 if hard_trigger else self._tail_exclusion_count(events)
        summary_cutoff = max(total_events - exclude_tail, 0)
        candidates: list = []
        for index, event in enumerate(events):
            if index in keep_indices:
                continue
            if index < summary_cutoff:
                candidates.append(event)
        return candidates

    def _judge_replacement(self, events: list) -> tuple[bool, dict]:
        history_text = self._format_events_for_manager(
            events, include_tool_content=False, start_index=1
        )
        prompt = REPLACEMENT_JUDGE_PROMPT.format(history=history_text)
        messages = [Message(role='user', content=[TextContent(text=prompt)])]
        formatted = self.judge_llm.format_messages_for_llm(messages)

        valid_votes = 0
        yes_votes = 0
        no_votes = 0
        reasoning_list: list[str] = []
        judge_votes: list[dict] = []

        max_attempts = self.voting_k * 3
        attempts_used = 0
        threshold = math.ceil(self.voting_k / 2)

        def _one_vote() -> tuple[str | None, str | None, str | None]:
            """Return (decision, reasoning, error_type)."""
            try:
                response = self.judge_llm.completion(
                    messages=formatted,
                    extra_body={'metadata': self.llm_metadata},
                )
            except Exception as exc:
                logger.warning('DynaContext: Judge call failed: %s', exc)
                return None, None, 'exception'

            content = response.choices[0].message.content if response else ''
            if not content:
                return None, None, 'empty'

            decision_match = DECISION_PATTERN.search(content)
            if not decision_match:
                return None, None, 'parse_fail'

            decision = decision_match.group(1).strip().upper()
            reasoning_match = REASONING_PATTERN.search(content)
            reasoning = (
                reasoning_match.group(1).strip()
                if reasoning_match
                else 'No reasoning provided.'
            )
            return decision, reasoning, None

        # Fire a first synchronous concurrent batch. If the pool fails, fall back to serial.
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.voting_k) as pool:
                futures = [pool.submit(_one_vote) for _ in range(self.voting_k)]
                for fut in as_completed(futures):
                    attempts_used += 1
                    decision, reasoning, error_type = fut.result()

                    if decision is None:
                        judge_votes.append(
                            {
                                'decision': None,
                                'parsed_ok': False,
                                'reasoning_present': False,
                                'error_type': error_type,
                            }
                        )
                        continue

                    judge_votes.append(
                        {
                            'decision': decision,
                            'parsed_ok': True,
                            'reasoning_present': reasoning != 'No reasoning provided.',
                            'error_type': None,
                            'reasoning': reasoning,
                        }
                    )
                    valid_votes += 1
                    if decision == 'YES':
                        yes_votes += 1
                        reasoning_list.append(f'[Vote {valid_votes} YES]: {reasoning}')
                    else:
                        no_votes += 1
                        reasoning_list.append(f'[Vote {valid_votes} NO]: {reasoning}')

                    if yes_votes >= threshold or no_votes >= threshold:
                        break

        except Exception as exc:
            logger.warning(
                'DynaContext: concurrent judge voting failed; falling back to serial: %s',
                exc,
            )

        # If concurrency produced insufficient valid votes (parse failures/timeouts),
        # keep going serially until we collect enough or exhaust attempts.
        while valid_votes < self.voting_k and attempts_used < max_attempts:
            attempts_used += 1
            decision, reasoning, error_type = _one_vote()

            if decision is None:
                judge_votes.append(
                    {
                        'decision': None,
                        'parsed_ok': False,
                        'reasoning_present': False,
                        'error_type': error_type,
                    }
                )
                continue

            judge_votes.append(
                {
                    'decision': decision,
                    'parsed_ok': True,
                    'reasoning_present': reasoning != 'No reasoning provided.',
                    'error_type': None,
                    'reasoning': reasoning,
                }
            )
            valid_votes += 1
            if decision == 'YES':
                yes_votes += 1
                reasoning_list.append(f'[Vote {valid_votes} YES]: {reasoning}')
            else:
                no_votes += 1
                reasoning_list.append(f'[Vote {valid_votes} NO]: {reasoning}')

            if yes_votes >= threshold or no_votes >= threshold:
                break

        if valid_votes == 0:
            return False, {
                'valid_votes': 0,
                'yes_votes': 0,
                'no_votes': 0,
                'voting_k': self.voting_k,
                'decision': False,
                'reasoning': 'No valid votes collected.',
                'attempts_used': attempts_used,
                'threshold': threshold,
                'judge_votes': judge_votes,
            }

        if valid_votes < self.voting_k:
            threshold = math.ceil(valid_votes / 2)
        decision = yes_votes >= threshold
        reasoning = '\n'.join(reasoning_list)
        return decision, {
            'valid_votes': valid_votes,
            'yes_votes': yes_votes,
            'no_votes': no_votes,
            'voting_k': self.voting_k,
            'decision': decision,
            'reasoning': reasoning,
            'threshold': threshold,
            'attempts_used': attempts_used,
            'judge_votes': judge_votes,
        }

    def _summarize_segment(self, events: list) -> str:
        history_text = self._format_events_for_manager(
            events, include_tool_content=False, start_index=1
        )
        prompt = BLOCK_SUMMARY_PROMPT.format(history=history_text)
        messages = [Message(role='user', content=[TextContent(text=prompt)])]
        formatted = self.llm.format_messages_for_llm(messages)

        summary_content = ''
        for attempt in range(3):
            try:
                response = self.llm.completion(
                    messages=formatted,
                    extra_body={'metadata': self.llm_metadata},
                )
            except Exception as exc:
                logger.warning('DynaContext: Summary call failed: %s', exc)
                continue
            content = response.choices[0].message.content if response else ''
            if not content:
                continue
            match = SUMMARY_PATTERN.search(content)
            if match:
                summary_content = match.group(1).strip()
                break
            logger.info(
                'DynaContext: Summary parse failed (attempt %s/3).',
                attempt + 1,
            )

        if not summary_content:
            logger.warning('DynaContext: Failed to summarize segment.')
            return 'Summary generation failed.'
        return summary_content

    def should_condense(self, view: View) -> bool:
        events = list(view)
        self._count_new_agent_turns(events)
        if view.unhandled_condensation_request:
            self._last_skip_reason = None
            return True
        if not self._should_judge():
            self._last_skip_reason = 'early_turns'
            self._last_judge_metadata = None
            return False
        decision, metadata = self._judge_replacement(events)
        self._last_judge_metadata = metadata
        if not metadata.get('valid_votes'):
            self._last_skip_reason = 'no_valid_votes'
        elif not decision:
            self._last_skip_reason = 'judge_no_majority'
        else:
            self._last_skip_reason = None
        self._pending_judge_metadata = metadata if decision else None
        if not decision:
            return False
        return len(
            self._summary_candidates(events, hard_trigger=False)
        ) > self.keep_first

    def get_condensation(self, view: View) -> Condensation:
        events = list(view)
        total_events = len(events)
        keep_indices = set(range(min(self.keep_first, total_events)))
        if self.keep_last > 0:
            tail_start = max(total_events - self.keep_last, 0)
            keep_indices.update(range(tail_start, total_events))

        event_ids_to_forget: set[int] = set()
        hard_trigger = view.unhandled_condensation_request
        if hard_trigger:
            self._pending_judge_metadata = None
        tail_exclusion_count = (
            0 if hard_trigger else self._tail_exclusion_count(events)
        )
        summary_candidates = self._summary_candidates(
            events, hard_trigger=hard_trigger
        )

        for index, event in enumerate(events):
            if index in keep_indices:
                continue

            # Semantics: never forget the initial user instruction event.
            # keep_first is defined as "保留首条用户任务描述"; however, index-based slicing
            # includes the system message at index 0. We therefore treat the first user
            # message action (if present) as part of the preserved prefix.
            if (
                index == 1
                and isinstance(event, MessageAction)
                and event.source == EventSource.USER
            ):
                continue

            event_ids_to_forget.add(event.id)

        if len(summary_candidates) <= self.keep_first:
            if hard_trigger:
                forget_candidates = [
                    event.id for event in events if event.id in event_ids_to_forget
                ]
                if forget_candidates:
                    forget_ids = [forget_candidates[0]]
                else:
                    fallback_id = None
                    for index, event in enumerate(events):
                        if index >= self.keep_first:
                            fallback_id = event.id
                            break
                    if fallback_id is not None:
                        forget_ids = [fallback_id]
                    else:
                        forget_ids = []
                self.add_metadata('strategy', 'dynacontext')
                self.add_metadata('trigger', 'condensation_request')
                self.add_metadata('summary_length', 0)
                self.add_metadata('forgotten_event_count', len(forget_ids))
                self.add_metadata('discard_ratio', 0.0)
                self.add_metadata('hard_trigger', True)
                self.add_metadata('tail_exclusion_count', tail_exclusion_count)
                self.add_metadata('summary_candidate_count', len(summary_candidates))
                self.add_metadata(
                    'summary_candidate_event_ids',
                    [event.id for event in summary_candidates],
                )
                action = CondensationAction(
                    forgotten_event_ids=forget_ids,
                    summary=None,
                    summary_offset=None,
                    metadata={
                        'strategy': 'dynacontext',
                        'trigger': 'condensation_request',
                        'summary_length': 0,
                        'forgotten_event_count': len(forget_ids),
                        'discard_ratio': 0.0,
                        'hard_trigger': True,
                        'tail_exclusion_count': tail_exclusion_count,
                        'summary_candidate_count': len(summary_candidates),
                        'summary_candidate_event_ids': [
                            event.id for event in summary_candidates
                        ],
                        'note': 'minimal_forget',
                    },
                )
                self._turns_since_reset = 0
                self._pending_judge_metadata = None
                return Condensation(action=action)
            return Condensation(
                action=CondensationAction(
                    forgotten_event_ids=[],
                    summary=None,
                    summary_offset=None,
                    metadata={
                        'strategy': 'dynacontext',
                        'trigger': 'judge',
                        'summary_length': 0,
                        'forgotten_event_count': 0,
                        'discard_ratio': 0.0,
                        'note': 'no_summarizable_events',
                    },
                )
            )

        summary = self._summarize_segment(summary_candidates)

        discard_ratio = (
            (len(event_ids_to_forget) / total_events) if total_events else 0.0
        )
        trigger = (
            'condensation_request'
            if view.unhandled_condensation_request
            else 'judge'
        )

        self.add_metadata('strategy', 'dynacontext')
        self.add_metadata('trigger', trigger)
        self.add_metadata('summary_length', len(summary))
        self.add_metadata('forgotten_event_count', len(event_ids_to_forget))
        self.add_metadata('discard_ratio', discard_ratio)
        self.add_metadata('keep_first', self.keep_first)
        self.add_metadata('keep_last', self.keep_last)
        self.add_metadata('early_turns', self.early_turns)
        self.add_metadata('voting_k', self.voting_k)
        self.add_metadata('turns_since_reset', self._turns_since_reset)
        self.add_metadata('hard_trigger', hard_trigger)
        self.add_metadata('tail_exclusion_count', tail_exclusion_count)
        self.add_metadata('summary_candidate_count', len(summary_candidates))
        self.add_metadata(
            'summary_candidate_event_ids',
            [event.id for event in summary_candidates],
        )
        if trigger == 'judge' and self._pending_judge_metadata is not None:
            self.add_metadata('judge_metadata', self._pending_judge_metadata)
            self._pending_judge_metadata = None

        summary_preview = summary
        if len(summary_preview) > 1000:
            summary_preview = summary_preview[:1000] + '...(truncated)'
        logger.info(
            'DynaContextCondenser: summarized %s events into %s chars',
            len(event_ids_to_forget),
            len(summary),
        )
        logger.info('DynaContextCondenser summary: %s', summary_preview)

        forgotten_ids = sorted(event_ids_to_forget)
        if not forgotten_ids:
            action = CondensationAction(
                forgotten_event_ids=[],
                summary=summary,
                summary_offset=self.keep_first,
                metadata={
                    'strategy': 'dynacontext',
                    'trigger': trigger,
                    'summary_length': len(summary),
                    'forgotten_event_count': 0,
                    'discard_ratio': discard_ratio,
                    'keep_first': self.keep_first,
                    'keep_last': self.keep_last,
                },
            )
            self._turns_since_reset = 0
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
                    'strategy': 'dynacontext',
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
                    'strategy': 'dynacontext',
                    'trigger': trigger,
                    'summary_length': len(summary),
                    'forgotten_event_count': len(forgotten_ids),
                    'discard_ratio': discard_ratio,
                    'keep_first': self.keep_first,
                    'keep_last': self.keep_last,
                },
            )

        self._turns_since_reset = 0
        return Condensation(action=action)

    def get_debug_state(self) -> dict:
        return {
            'turns_since_reset': self._turns_since_reset,
            'early_turns': self.early_turns,
            'should_judge': self._should_judge(),
            'last_skip_reason': self._last_skip_reason,
            'last_judge_metadata': self._last_judge_metadata,
        }

    @classmethod
    def from_config(
        cls, config: DynaContextCondenserConfig, llm_registry: LLMRegistry
    ) -> DynaContextCondenser:
        llm_config = config.llm_config.model_copy()
        llm_config.caching_prompt = False
        llm = llm_registry.get_llm('condenser', llm_config)

        judge_config = (
            config.judge_llm_config.model_copy()
            if config.judge_llm_config is not None
            else llm_config.model_copy()
        )
        judge_config.caching_prompt = False
        judge_llm = llm_registry.get_llm('condenser_judge', judge_config)

        return DynaContextCondenser(
            llm=llm,
            judge_llm=judge_llm,
            voting_k=config.voting_k,
            keep_first=config.keep_first,
            keep_last=config.keep_last,
            early_turns=config.early_turns,
            max_event_length=config.max_event_length,
            exclude_tail_max=config.exclude_tail_max,
        )


DynaContextCondenser.register_config(DynaContextCondenserConfig)
