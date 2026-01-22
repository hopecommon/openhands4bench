# DynaContext 实现记录（OpenHands 评测版）

本文记录 OpenHands 中 DynaContext 策略的实现细节、与 `refer/dynamic_context_agent_dev.py` 的对照、以及为评测对齐做出的取舍。目标是便于后续二次开发与结果复现，而不是优化交互体验。

---

## 1. 目标与约束

- 目标：在 OpenHands 框架中复现 DynaContext 的**策略意图**，用于评测不同上下文策略效果。
- 约束：
  - OpenHands Condenser 为**同步**机制，仅返回 `CondensationAction`；不存在 refer 中的“分支/回滚/重放”语义。
  - 评测为主（headless），对齐优先于交互体验。

---

## 2. 触发机制对齐

### refer
- **策略触发（Judge）**：每回合投票判断是否替换。
- **硬触发（Context Limit）**：`context_length - input_length < 512` 时强制压缩。

### OpenHands 实现
- **策略触发（Judge）**：
  - `should_condense` 中进行投票判断，若 YES 则触发。
  - `early_turns` 控制早期不触发判定（默认 1）。
- **硬触发（Context Limit）**：
  - 当 `CondensationRequestAction` 出现时，`should_condense` 直接返回 True。
  - 避免“反复 request → 不压缩 → 自旋”。

---

## 3. Summary 范围对齐

### refer
- `msgs_to_summarize = snapshot[:-2]`
- 实际总结范围：**初始 prompt 之后的全部历史**，但排除“当前回合尾巴 2 条”（通常是 agent/tool 半成品）。

### OpenHands 实现
- **keep_first = 1**：保留首条用户任务描述，不纳入 summary（与 refer 一致）。
- **keep_last = 0**：不保留尾部历史（headless 假设下符合评测对齐）。
- **尾部排除规则（软触发）**：
  - 仅排除“尾部 agent/tool 事件”，最多 2 条（模拟 refer 的 `[:-2]` 语义）。
  - 一旦遇到 `MessageAction` 且 `source == USER`，立即停止排除。
  - 遇到非 agent/tool 事件也停止排除。
- **硬触发（Context Limit）**：
  - 不排尾巴，避免误伤最新 user 指令。
  - Summary 覆盖“首条 user 之后的全部历史”。

---

## 4. Judge 投票与失败策略

### refer
- `K` 次并发投票（`asyncio.gather`），失败补票。

### OpenHands 实现
- 同步串行投票，最多 `3K` 次，**提前终止**：
  - 若 YES 或 NO 已达到多数阈值即停止。
- Judge 调用失败 **直接视为 NO**（不回退主模型）。
- 投票 metadata 缓存，**仅在真正触发 condensation 时写入**，避免噪音。

> 注：OpenHands Condenser 为同步流程，暂未引入异步并发以避免事件循环冲突。

---

## 5. 硬触发“至少遗忘一段”兜底

### 问题
硬触发时若 “没有可 summary 的内容”，可能出现反复 `CondensationRequestAction` → 不压缩 → 再请求的自旋。

### 解决
在硬触发且 `summary_candidates` 过少时：
- 强制遗忘 **至少 1 个 event**（从 `keep_first` 后开始），避免死循环。
- 若仍无可遗忘事件，则允许返回空遗忘（极端边界）。

---

## 6. 观测与追踪字段（快照/消息）

为便于科研验证与回溯，本实现补充了以下观测数据：

### Context Snapshot（V0）
- `condenser_meta_latest`：最近一次 condensation 的元数据（包含 dynacontext 的投票/触发/候选范围等）。

### LLM Messages（llm_messages.json）
- `context_strategy`：会话级策略名。
- `condenser_meta_latest`：最近一次 condensation 的元数据。
- `dynacontext_state`：最后一次判定状态（turns_since_reset/should_judge/last_skip_reason/last_judge_metadata）。

### DynaContext Condenser Metadata（condensation meta）
- `trigger`：`judge` 或 `condensation_request`
- `hard_trigger`：是否由硬触发导致
- `judge_metadata`：投票结果（`valid_votes/yes_votes/no_votes/threshold/reasoning`）
- `judge_votes`：逐票记录（decision/parsed_ok/reasoning_present/error_type/reasoning）
- `tail_exclusion_count`：本次 summary 排除的尾部事件数量
- `summary_candidate_count` / `summary_candidate_event_ids`
- `keep_first` / `keep_last` / `early_turns` / `voting_k`
- `turns_since_reset`：用于解释 early_turns gate

> 说明：在无头评测默认设置下（`keep_last=0`），被尾部排除的事件会被**丢弃**（不保留、不总结），这是为了避免“当前回合半成品/可能错误路径”污染后续摘要与决策。

---

## 7. 配置映射

- `context_strategy = "dynacontext"` 会映射到 `DynaContextCondenserConfig`。
- Judge 模型配置：
  - 若存在 `[llm.dynacontext_judge]`，优先使用；
  - 否则回退主模型配置。

---

## 8. 与 refer 的关键差异

- **无分支/回滚重放**：OpenHands 直接通过 `CondensationAction` 替换历史。
- **尾部排除按事件类型**：避免误删最新 user，软触发时最多排除 2 条 agent/tool 尾巴。
- **硬触发强制缩短**：防止 request 自旋。

这些差异是为评测对齐与框架约束所做的取舍。

---

## 9. 默认参数

- `keep_first = 1`
- `keep_last = 0`
- `voting_k = 5`
- `early_turns = 1`
- `exclude_tail_max = 2`（内部常量）

---

## 10. 相关代码位置

- `openhands/memory/condenser/impl/dynacontext_condenser.py`
- `openhands/core/config/condenser_config.py`
- `openhands/core/config/utils.py`
- `config.template.toml`
- `tests/unit/memory/condenser/test_condenser.py`
