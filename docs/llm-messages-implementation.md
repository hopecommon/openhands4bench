# OpenHands LLM Messages 捕获实现总结

## 目标

在 OpenHands V0 无头模式下，捕获并保存完整的 LLM messages 列表，格式严格对齐参考文件 `refer/results_20260109_141906.json` 中的 `messages` 字段。

## 参考格式分析

### 消息类型统计（来自参考文件第一个 result）
- **system**: 1 条
- **user**: 1 条  
- **assistant**: 10 条
- **tool**: 10 条
- **总计**: 22 条消息

### 各角色消息格式

#### 1. System Message
```json
{
  "role": "system",
  "content": "system prompt text..."
}
```
- 仅包含 `role` 和 `content` 两个字段

#### 2. User Message
```json
{
  "role": "user",
  "content": "user query text..."
}
```
- 仅包含 `role` 和 `content` 两个字段

#### 3. Assistant Message（带 tool_calls）
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

**关键特征：**
- 有 tool_calls 时，`content` 为 `null`
- `tool_calls[].index` 固定为 `-1`
- `reasoning_content` 包含推理过程（如果有）
- 固定包含所有字段，即使为 `null`

#### 4. Tool Message
```json
{
  "role": "tool",
  "tool_call_id": "call_89b7fd353e4a4bd8bf2ac51f",
  "name": "search",
  "content": "tool output..."
}
```

**关键特征：**
- `tool_call_id` 对应之前 assistant message 中的 tool call ID
- `name` 是工具名称
- `content` 是工具的返回结果

---

## 实现方案

### 核心原理

V0 架构中：
1. **EventStream 记录 Events**（Action/Observation），不是 Messages
2. **Messages 是临时的**，仅在组装 LLM prompt 时由 `ConversationMemory.process_events()` 生成
3. **需要重建**：在会话结束时调用 `process_events()` 重新生成完整的 messages 列表

### 消息流转过程

```
User Input (CLI)
  → MessageAction (source=USER) 
    → EventStream
      → AgentController.on_event()
        → Agent.step()
          → ConversationMemory.process_events() ⟵ 生成 messages 列表
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

---

## 代码修改

### 1. 新增文件：`openhands/utils/result_messages.py`

**功能：** 将 OpenHands 内部的 `Message` 对象转换为参考格式

```python
def messages_to_results_format(messages: list[Message]) -> list[dict[str, Any]]
```

**关键实现：**
- `_content_to_string()`: 处理多模态内容（文本、图片URL）
- `_tool_calls_to_results()`: 转换 tool_calls，确保 `index=-1`
- Assistant 有 tool_calls 时，将 `content` 设为 `null`
- 保留 `reasoning_content` 字段

**示例输出：**
```python
# System/User message
{"role": "system", "content": "..."}

# Assistant with tool_calls
{
    "content": null,
    "refusal": null,
    "role": "assistant",
    "annotations": null,
    "audio": null,
    "function_call": null,
    "tool_calls": [...],
    "reasoning_content": "..."
}

# Tool message
{
    "role": "tool",
    "tool_call_id": "call_xxx",
    "name": "search",
    "content": "..."
}
```

### 2. 修改：`openhands/memory/conversation_memory.py`

在 `_process_observation()` 方法中添加 reasoning 提取逻辑：

```python
def _extract_reasoning(assistant_msg: Any) -> str | None:
    """从多个可能的来源提取 reasoning"""
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

### 3. 修改：`openhands/storage/locations.py`

添加新函数获取 messages 文件路径：

```python
def get_conversation_llm_messages_filename(sid: str, user_id: str | None = None) -> str:
    """Session-level LLM messages transcript in the benchmark results-format."""
    return f'{get_conversation_dir(sid, user_id)}llm_messages.json'
```

### 4. 修改：`openhands/controller/agent_controller.py`

#### 导入必要模块

```python
from openhands.storage.locations import (
    get_conversation_llm_messages_filename,
    ...
)
from openhands.utils.result_messages import messages_to_results_format
```

#### 添加公共方法

```python
def get_first_user_message(self, events: list[Event]) -> MessageAction | None:
    """Public wrapper for _first_user_message, used by message logging."""
    return self._first_user_message(events)
```

#### 在 `close()` 方法末尾添加保存逻辑

```python
async def close(self, set_stop_state: bool = True) -> None:
    if set_stop_state:
        await self.set_agent_state_to(AgentState.STOPPED)

    # 保存 LLM messages 到文件（可选功能）
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
                # 重建 messages 列表
                messages = conversation_memory.process_events(
                    condensed_history=list(self.state.history),
                    initial_user_action=initial_user,
                    forgotten_event_ids=set(),
                    max_message_chars=getattr(llm.config, 'max_message_chars', None),
                    vision_is_active=bool(getattr(llm, 'vision_is_active', lambda: False)()),
                )
                
                # 转换为参考格式并保存
                payload = {'messages': messages_to_results_format(messages)}
                path = get_conversation_llm_messages_filename(self.id, self.user_id)
                self.file_store.write(
                    path,
                    json.dumps(payload, ensure_ascii=False, indent=2),
                )
    except Exception as e:
        # 容错处理：即使失败也不影响主流程
        logger.warning(f'Failed to dump llm_messages.json: {e}')

    # 原有的 close 逻辑
    self.state_tracker.close(self.event_stream)
    ...
```

---

## 环境变量控制

- **`OPENHANDS_SAVE_LLM_MESSAGES`**：控制是否保存 messages
  - `'1'`（默认）：保存
  - `'0'`, `'false'`, `'no'`：不保存

---

## 输出格式

### 文件位置
保存路径：`{conversation_dir}/llm_messages.json`

### 文件格式
```json
{
  "messages": [
    {
      "role": "system",
      "content": "..."
    },
    {
      "role": "user",
      "content": "..."
    },
    {
      "content": null,
      "refusal": null,
      "role": "assistant",
      "annotations": null,
      "audio": null,
      "function_call": null,
      "tool_calls": [
        {
          "id": "call_xxx",
          "function": {
            "arguments": "{\"query\": \"...\"}",
            "name": "search"
          },
          "type": "function",
          "index": -1
        }
      ],
      "reasoning_content": "..."
    },
    {
      "role": "tool",
      "tool_call_id": "call_xxx",
      "name": "search",
      "content": "..."
    },
    ...
  ]
}
```

---

## 与 Trajectory 的区别

| 维度 | Trajectory (Events) | LLM Messages |
|------|---------------------|--------------|
| **记录对象** | Events (Action/Observation) | Chat Messages |
| **role=tool** | ❌ 不存在，只有 Observation | ✅ 存在 |
| **tool response** | Observation.content + metadata | Message(role='tool') |
| **记录时机** | 实时记录到 EventStream | 会话结束时重建 |
| **用途** | 完整事件历史 | LLM prompt 重建 |
| **持久化** | trajectory.json | llm_messages.json |

---

## 设计要点

### 1. 为什么 Trajectory 中没有 role=tool？
因为 V0 的轨迹记录的是 **Events（事件）**，不是 **Messages（消息）**：
- Action → MessageAction, AgentDelegate, CmdRun, ...
- Observation → CmdOutput, AgentDelegate, ...
- **没有 "ToolMessage" 这个 Event 类型**

### 2. role=tool 从哪来？
`ConversationMemory.process_events()` 在处理 Observation 时：
- 检查 `observation.tool_call_metadata`
- 如果存在，生成一个 `Message(role='tool')`
- 这个 Message **只存在于内存中**，用于构建 LLM prompt

### 3. 为什么要在 close() 时保存？
- **避免频繁写入**：只在会话结束时保存一次
- **保证完整性**：确保所有事件都已处理完毕
- **容错设计**：即使保存失败也不影响主流程

### 4. Reasoning 的处理
不同 LLM provider 可能在不同字段存储 reasoning：
- `reasoning_content`
- `reasoning`
- `provider_specific_fields['reasoning_content']`
- `provider_specific_fields['reasoning']`

因此使用 `_extract_reasoning()` 从多个来源尝试提取。

---

## 使用示例

### 启动无头模式运行
```bash
# 默认开启 messages 保存
poetry run python openhands/core/main.py ...

# 显式关闭
OPENHANDS_SAVE_LLM_MESSAGES=0 poetry run python openhands/core/main.py ...
```

### 查看生成的文件
```bash
cat ~/.openhands/sessions/<session_id>/llm_messages.json
```

### 验证格式
```python
import json

with open('llm_messages.json') as f:
    data = json.load(f)

messages = data['messages']
print(f"Total messages: {len(messages)}")

for msg in messages:
    role = msg['role']
    print(f"{role}: {list(msg.keys())}")
```

---

## 注意事项

1. **不要修改 EventStream 的记录逻辑**
   - 轨迹（trajectory）保持原样
   - Messages 是额外的导出功能

2. **Messages 是重建的**
   - 通过 `ConversationMemory.process_events()` 从 Events 重建
   - 不是实时记录的

3. **Reasoning 提取**
   - 需要从多个可能的字段提取
   - 不同 provider 可能不同

4. **只在 close() 保存**
   - 避免频繁写入
   - 一次性保存完整会话

5. **容错处理**
   - 即使保存失败也不影响主流程
   - 使用 `try-except` 包裹

6. **Delegate Agent**
   - 不保存子 agent 的 messages（`is_delegate=True`）
   - 只保存根 agent 的完整对话

---

## 测试验证

参考文件分析脚本：`analyze_reference.py`

```bash
python3.12 analyze_reference.py
```

输出示例：
```
Total results: 100
First result has 22 messages

Message role distribution:
  assistant: 10
  system: 1
  tool: 10
  user: 1
```

---

## 后续工作

当前实现已完成 `messages` 字段的捕获和格式化。其他字段（如 `instance_id`, `model`, `cost` 等）由其他脚本填充。

### 完整的 result 对象结构
```json
{
  "instance_id": "...",
  "model": "...",
  "cost": 0.0,
  "messages": [...],  // ← 本实现负责
  "output": "...",
  "metadata": {...}
}
```

### 集成流程
1. OpenHands 运行生成 `llm_messages.json`
2. 后处理脚本读取并合并到最终的 `results_*.json`
3. 添加其他元数据（instance_id, cost, 等）

---

## 总结

✅ **已完成：**
- 创建 `result_messages.py` 转换模块
- 修改 `conversation_memory.py` 提取 reasoning
- 修改 `agent_controller.py` 在 close() 时保存
- 添加 `get_conversation_llm_messages_filename()` 辅助函数
- 添加 `get_first_user_message()` 公共方法
- 支持环境变量控制
- 格式严格对齐参考文件

✅ **格式验证：**
- System/User: ✓ 简单的 role + content
- Assistant: ✓ 完整的字段集（content, tool_calls, reasoning_content, 等）
- Tool: ✓ role + tool_call_id + name + content
- Tool calls: ✓ index 固定为 -1
- Reasoning: ✓ 从多个来源提取

✅ **设计原则：**
- 不修改原有轨迹记录逻辑
- 通过 process_events() 重建 messages
- 容错处理，不影响主流程
- 环境变量控制，灵活开关

🎯 **目标达成：** 
现在可以在 OpenHands V0 无头模式运行后，自动生成严格对齐参考格式的 `llm_messages.json` 文件！
