# V0 Context Snapshot (决策上下文快照) 开发文档与使用指南

本文档说明 V0 旧链路下的“决策上下文快照”机制，帮助你追踪每一步 LLM 决策时真实携带的上下文，并为上下文策略（summary / discard_all 等）提供可落盘的溯源数据。

## 目标与范围

- 目标：记录“LLM 实际看到的上下文”与“上下文策略效果”，用于定位、复盘、评估与改进。
- 范围：仅覆盖 V0 旧链路（`openhands/agenthub/codeact_agent` + `openhands/controller`）。
- 不包含：V1 app_server 的会话存储与事件导出（该链路另行设计）。

## 机制概览

每次 LLM 调用前：
1. 由 `CodeActAgent` 在构造完 `messages` 后生成 snapshot（含事件引用、summary、omitted 列表等）。
2. snapshot 暂存到 `State.extra_data` 中。

每次 LLM 调用后：
1. `AgentController` 在 action 入库事件后写入 snapshot 文件。
2. 写入位置在 file_store 目录中，与 events 同级。

## 配置项

在 `config.toml` 的 `[agent]` 段落中：

```toml
[agent]
save_context_snapshots = true
save_context_prompt = false
```

- `save_context_snapshots`：是否写入快照（默认 false）。
- `save_context_prompt`：是否保存格式化后的 prompt messages（默认 false）。

模板位置：`config.template.toml`。

## 存储路径与文件结构

快照文件路径（与 events 同级）：

```
sessions/<sid>/context_snapshots/snapshot_000001.json
```

带 user_id 的路径：

```
users/<user_id>/conversations/<sid>/context_snapshots/snapshot_000001.json
```

当 `save_context_prompt = true` 时，system prompt 会被去重保存：

```
sessions/<sid>/context_snapshots/system_prompt_<hash>.txt
```

在 snapshot 中，system prompt 会被替换为：

```
<system_prompt:<hash>>
```

## Snapshot 字段说明

核心字段（必有）：
- `version`：快照版本号（当前为 1）
- `snapshot_id`：递增 ID
- `created_at` / `saved_at`
- `session_id`
- `agent`
- `context_strategy`
- `context_window_limit_tokens`
- `max_message_chars`
- `vision_is_active`
- `condensed_event_ids`：本次进入上下文的事件 ID 列表
- `forgotten_event_ids`：被 condenser 忘记的事件 ID
- `omitted_tool_response_event_ids`：被标记为 `omitted` 的工具响应事件 ID

与 summary 相关字段（可选）：
- `condensation_action_id`
- `summary`
- `summary_offset`

与 action 关联字段：
- `action_event_id`
- `action_type`

可选 prompt 字段（`save_context_prompt = true`）：
- `messages`：格式化后的 LLM messages
- `system_prompt_hash`

## 如何确认 discard_all 结果

当 `discard_all` 触发时：
- 被“省略”的工具响应事件，其内容会被写为 `omitted`。
- snapshot 里的 `omitted_tool_response_event_ids` 会列出这些事件的 ID。
- 若开启 `save_context_prompt`，可以在 `messages` 中看到工具输出是否已变为 `omitted`。

## 如何确认 summary 内容

当 summary 触发时：
- snapshot 中会包含 `summary`（从 `CondensationAction.summary` 抽取）。
- 如果没有 `CondensationAction`，会尝试从 `AgentCondensationObservation` 中提取。

## 示例（简化）

```json
{
  "version": 1,
  "snapshot_id": 12,
  "created_at": "2025-01-01T10:00:00",
  "saved_at": "2025-01-01T10:00:01",
  "session_id": "abc123",
  "agent": "CodeActAgent",
  "context_strategy": "discard_all",
  "max_message_chars": 8000,
  "condensed_event_ids": [1, 2, 5, 6, 7],
  "forgotten_event_ids": [3, 4],
  "omitted_tool_response_event_ids": [6],
  "summary": "<think>...</think>",
  "summary_offset": 1,
  "action_event_id": 9,
  "action_type": "CmdRunAction"
}
```

## 与传统 trajectory/events 的关系

- events/trajectory 记录“发生了什么”。
- context snapshot 记录“当时 LLM 看到了什么”。
- 两者可以通过 `action_event_id` 和 `condensed_event_ids` 关联。

## 设计注意事项

- snapshot 默认不保存完整 prompt，防止体积过大。
- system prompt 单独去重存储（hash 引用），避免重复写入。
- snapshot 仅写入 file_store，不影响事件流与 trajectory 导出。

## 限制与可扩展点

- 当前仅覆盖 V0 `CodeActAgent` 路径，其他 agent 需单独接入。
- `omitted_tool_response_event_ids` 记录的是“在本次上下文内为 omitted 的事件”。
- 如果需要“相对上一轮新增 omitted 列表”，可在 snapshot 间做 diff 或扩展逻辑。

## 快速排查建议

1. 确认配置：
   - `save_context_snapshots = true`
2. 运行一次任务后检查目录：
   - `~/.openhands/sessions/<sid>/context_snapshots/`
3. 若需要完整上下文确认：
   - 将 `save_context_prompt = true`
   - 查看 `messages` 与 `system_prompt_<hash>.txt`
