# Context Management Progress

## Step 1/5: Code Survey (2025-02-14)

### Key Files & Functions
- `openhands/agenthub/codeact_agent/codeact_agent.py`: `CodeActAgent.step()` calls `self.condenser.condensed_history(state)`; if a `CondensationAction` is returned, it is emitted directly, otherwise it builds messages via `ConversationMemory.process_events`.
- `openhands/memory/condenser/condenser.py`: Core condenser interface + registry; `Condenser.condensed_history()` writes metadata into `State.extra_data`.
- `openhands/memory/view.py`: Builds `View` from events, injects summaries, tracks `forgotten_event_ids`, and detects unhandled condensation requests.
- `openhands/memory/condenser/impl/llm_summarizing_condenser.py`: LLM-based summarization strategy (existing “Summary”-like behavior).
- `openhands/memory/condenser/impl/conversation_window_condenser.py`: Window-based truncation that keeps essential early events and recent history.
- `openhands/controller/agent_controller.py`: Catches context window errors; if `enable_history_truncation` is true, emits `CondensationRequestAction` to trigger condensation.
- `openhands/core/config/agent_config.py` + `openhands/core/config/utils.py`: Configuration of condensers via `config.toml`; defaults to an LLM summarizing condenser when enabled.

### Call Chain (Prompt Assembly)
1. `AgentController._step()` -> `CodeActAgent.step(state)`
2. `Condenser.condensed_history(state)` -> `View` or `CondensationAction`
3. `ConversationMemory.process_events(...)` -> `list[Message]`
4. `LLM.completion(messages=...)`

### Candidate Extension Points
- Condenser selection (`AgentConfig.condenser` + `Condenser.from_config`)
- Condenser implementations in `openhands/memory/condenser/impl`
- Context window error handling (emit `CondensationRequestAction` in controller)

### Notes / Open Questions
- Current “summary” and “truncate” behaviors already exist as condensers; new strategies should likely map onto or extend this condenser layer.
- Need to confirm if V1 (app_server + SDK) path is in scope for NL2RepoBench; current entry point appears to be legacy V0 `CodeActAgent`.

### Next Step
- Define strategy interface + configuration entry points (env/config/CLI), and map baseline strategies onto existing or new condensers.

## Step 2/5: Strategy Interface Design (2025-02-14)

### Proposed Interface
- Add `context_strategy: str | None` to `AgentConfig` (e.g., `react`, `summary`, `discard_all`, `mem1`, `folding`, `dynacontext`).
- Use a small registry/helper (e.g., `apply_context_strategy(config, agent_name)`) to map strategy name -> `AgentConfig` mutations:
  - Set `agent_config.condenser` to a specific `CondenserConfig` implementation.
  - Toggle `agent_config.enable_history_truncation` for ReAct baseline.

### Configuration Entry Points
- **Environment variable:** `AGENT_CONTEXT_STRATEGY` (via existing env loader on `AgentConfig`).
- **config.toml:** `[agent] context_strategy = "summary"` (or per-agent override under `[agent.<name>]`).
- **CLI:** add `--context-strategy` to headless/cli parsers; apply to default agent in `setup_config_from_args`.

### Strategy Mapping (initial)
- `react`: `NoOpCondenserConfig`, `enable_history_truncation = false` (context overflow -> fail).
- `summary`: `LLMSummarizingCondenserConfig` (existing summarizer), `enable_history_truncation = true`.
- `discard_all`: new condenser that forgets tool response observations when condensation is requested.
- Others (`mem1`, `folding`, `dynacontext`, `strategy-x`): placeholder mapping + future config params.

### Logging Plan
- Log selected strategy + key parameters at config finalize / agent creation.
- Each condenser writes metadata (strategy name, trigger, forgotten counts, summary length) to `State.extra_data` and logs key events.

### Next Step
- Implement the strategy registry + `discard_all` condenser, wire CLI/config/env, and add logging.

## Step 3/5: Implementation (2025-02-14)

### Changes Applied
- Added `context_strategy` to `AgentConfig` and CLI `--context-strategy` override.
- Added `context_window_limit_tokens` to `AgentConfig` plus CLI/env/config support.
- Implemented `apply_context_strategy(...)` to map strategy names to condenser configs and history-truncation settings.
- Added `ToolResponseDiscardCondenser` + config type `tool_response_discard` (redacts tool responses as "omitted").
- Added summary logging + metadata (summary length, discard ratio) in `LLMSummarizingCondenser`.

### Files Touched
- `openhands/core/config/agent_config.py` (new field)
- `openhands/core/config/arg_utils.py` (CLI arg)
- `openhands/core/config/utils.py` (strategy application)
- `openhands/core/config/condenser_config.py` (new config type)
- `openhands/memory/condenser/impl/tool_response_discard_condenser.py` (new condenser)
- `openhands/memory/condenser/impl/__init__.py` (registration)
- `openhands/memory/condenser/impl/llm_summarizing_condenser.py` (logging + metadata)
- `config.template.toml` (config hint)

### Notes / Open Questions
- `discard_all` currently redacts tool response content on condensation request; message structure is preserved but content is replaced.
- `summary` can now be configured to skip event-count triggering and only respond to condensation requests via `trigger_on_max_size = false`.

### Next Step
- Validate strategy switching via logs and document final usage details.

## Step 4/5: Mem1 Strategy Integration (2025-02-14)

### Changes Applied
- Added a Mem1-style condenser that maintains a persistent <think> memory block and keeps a small tail window.
- Wired the new `mem1` context strategy into config parsing and strategy mapping.
- Added config/template hints for `mem1` usage.

### Files Touched
- `openhands/memory/condenser/impl/mem1_condenser.py`
- `openhands/core/config/condenser_config.py`
- `openhands/core/config/utils.py`
- `openhands/memory/condenser/impl/__init__.py`
- `config.template.toml`
- `tests/unit/memory/condenser/test_condenser.py`

### Notes / Open Questions
- Default `keep_last` is tuned for a small rolling window; can be overridden via `[condenser]` if needed.
- Strategy currently triggers on condensation requests and optional `max_size` threshold.
