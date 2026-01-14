from __future__ import annotations

from openhands.controller.state.state import State

CONTEXT_SNAPSHOT_PENDING_KEY = 'pending_context_snapshot'
CONTEXT_SNAPSHOT_COUNTER_KEY = 'context_snapshot_counter'
CONTEXT_SNAPSHOT_VERSION = 1


def next_snapshot_id(state: State) -> int:
    current = state.extra_data.get(CONTEXT_SNAPSHOT_COUNTER_KEY, 0)
    try:
        current_id = int(current)
    except (TypeError, ValueError):
        current_id = 0
    snapshot_id = current_id + 1
    state.extra_data[CONTEXT_SNAPSHOT_COUNTER_KEY] = snapshot_id
    return snapshot_id
