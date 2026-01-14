# OpenHands LLM Messages 捕获机制总结

## 目标

在 OpenHands V0 无头模式下，捕获并保存完整的 LLM messages 列表，格式严格对齐参考文件 `refer/results_20260109_141906.json` 中的 `messages` 字段。

## 核心理解

### V0 架构特点

1. **Event-Driven**：V0 的核心是 EventStream，记录的是 Action/Observation events
2. **Messages 是临时的**：`role=tool` 等消息只在组装 LLM prompt 时由 `ConversationMemory.process_events()` 临时生成
3. **不在轨迹中**：原始轨迹中看不到 `role=tool` 是正常的，因为轨迹记录的是 Events 不是 Messages

### 消息流转过程

```
User Input (CLI)
  → MessageAction (source=USER) 
    → EventStream
      → AgentController.on_event()
        → Agent.step()
          → ConversationMemory.process_events() 【生成 messages 列表】
            → LLM.completion(messages)
              → ModelResponse (tool_calls + reasoning_content)
                → response_to_actions()
                  → Actions (with ToolCallMetadata)
                    → EventStream
                      → Runtime.on_event()
                        → run_action()
                          → Observation (with tool_call_metadata)
                            → EventStream (下一轮循环)
```

## 目标格式分析

参考文件中的 messages 格式：

### System Message
```json
{
  "role": "system",
  "content": "system prompt text..."
}
```

### User Message
```json
{
  "role": "user",
  "content": "user query text..."
}
```

### Assistant Message (带 tool_calls)
```json
{
  "content": null,
  "refusal": null,
  "role": "assistant",
  "annotations": null,
  "audio": null,
  "function_call": null,
  "tool_calls": [
    {
      "id": "call_89b7fd353e4a4bd8bf2ac51f",
      "function": {
        "arguments": "{\"query\": \"...\"}",
        "name": "search"
      },
      "type": "function",
      "index": -1
    }
  ],
  "reasoning_content": "reasoning text if available..."
}
```

### Tool Message
```json
{
  "role": "tool",
  "tool_call_id": "call_89b7fd353e4a4bd8bf2ac51f",
  "name": "search",
  "content": "tool output..."
}
```

## 已完成的改动

### 1. `openhands/utils/result_messages.py` (新文件)

转换函数，将 OpenHands 内部的 `Message` 对象转换为目标格式：

- `messages_to_results_format()`: 主转换函数
- `_content_to_string()`: 处理多模态内容
- `_tool_calls_to_results()`: 转换 tool_calls 格式

关键点：
- Assistant 有 tool_calls 时，content 设为 `null`
- Tool calls 的 `index` 固定为 `-1`（参考格式要求）
- 保留 `reasoning_content` 字段

### 2. `openhands/memory/conversation_memory.py`

在 `_process_observation()` 方法中添加 `_extract_reasoning()` 函数，从多个可能的来源提取 reasoning：

```python
def _extract_reasoning(assistant_msg: Any) -> str | None:
    candidates: list[Any] = [
        getattr(assistant_msg, 'reasoning_content', None),
        getattr(assistant_msg, 'reasoning', None),
    ]
    provider_fields = getattr(assistant_msg, 'provider_specific_fields', None)
    if isinstance(provider_fields, dict):
        candidates.append(provider_fields.get('reasoning_content'))
        candidates.append(provider_fields.get('reasoning'))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
    return None
```

在生成 assistant message 时附加 reasoning：

```python
Message(
    role='assistant',
    content=[TextContent(text=content)],
    tool_calls=assistant_msg.tool_calls,
    reasoning=_extract_reasoning(assistant_msg),  # 新增
)
```

### 3. `openhands/storage/locations.py`

添加新函数获取 messages 文件路径：

```python
def get_conversation_llm_messages_filename(sid: str, user_id: str | None = None) -> str:
    """Session-level LLM messages transcript in the benchmark results-format."""
    return f'{get_conversation_dir(sid, user_id)}llm_messages.json'
```

### 4. `openhands/controller/agent_controller.py`

在 `close()` 方法末尾添加逻辑，保存 messages：

```python
try:
    save_messages = os.environ.get('OPENHANDS_SAVE_LLM_MESSAGES', '1').strip().lower()
    if (
        save_messages not in ('0', 'false', 'no')
        and not self.is_delegate
        and self.file_store is not None
    ):
        initial_user = self.get_first_user_message(events=self.state.history)
        conversation_memory = getattr(self.agent, 'conversation_memory', None)
        llm = getattr(self.agent, 'llm', None)
        if initial_user is not None and conversation_memory is not None and llm is not None:
            messages = conversation_memory.process_events(
                condensed_history=list(self.state.history),
                initial_user_action=initial_user,
                forgotten_event_ids=set(),
                max_message_chars=getattr(llm.config, 'max_message_chars', None),
                vision_is_active=bool(getattr(llm, 'vision_is_active', lambda: False)()),
            )
            payload = {'messages': messages_to_results_format(messages)}
            path = get_conversation_llm_messages_filename(self.id, self.user_id)
            self.file_store.write(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )
except Exception as e:
    logger.warning(f'Failed to dump llm_messages.json: {e}')
```

## 待完成

### 缺失的 `get_first_user_message()` 方法

需要在 `AgentController` 中添加此方法，用于从事件历史中提取第一个用户消息。

实现逻辑：
1. 遍历事件历史
2. 找到第一个 `MessageAction` 且 `source == EventSource.USER`
3. 返回该 MessageAction

## 环境变量控制

- `OPENHANDS_SAVE_LLM_MESSAGES`: 控制是否保存 messages
  - `'1'` (默认): 保存
  - `'0'`, `'false'`, `'no'`: 不保存

## 输出位置

保存路径：`{conversation_dir}/llm_messages.json`

格式：
```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": null, "tool_calls": [...], "reasoning_content": "..." },
    { "role": "tool", "tool_call_id": "...", "name": "...", "content": "..." },
    ...
  ]
}
```

## 与轨迹 (trajectory) 的区别

| 维度 | Trajectory (Events) | LLM Messages |
|------|-------------------|--------------|
| 记录对象 | Events (Action/Observation) | Chat Messages |
| role=tool | 不存在，只有 Observation | 存在 |
| tool response | Observation.content + metadata | Message(role='tool') |
| 时机 | 实时记录到 EventStream | 会话结束时重建 |
| 用途 | 完整事件历史 | LLM prompt 重建 |

## 关键注意事项

1. **不要修改 EventStream 的记录逻辑**：轨迹保持原样
2. **Messages 是重建的**：通过 `ConversationMemory.process_events()` 从 Events 重建
3. **Reasoning 提取**：需要从多个可能的字段提取（不同 provider 可能不同）
4. **只在 close() 保存**：避免频繁写入，一次性保存完整会话
5. **容错处理**：即使保存失败也不影响主流程
