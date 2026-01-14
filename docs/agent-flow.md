# OpenHands V0 Agent 调用流程（聚焦 V0）

本文件基于本仓库的 V0（legacy）实现梳理 Agent 调用链路与消息/工具调用处理流程。V1 已迁移到外部 SDK，不在本文范围内。

## 范围与核心概念

V0 里有两套“消息/事件”体系：
- **Event（Action/Observation）**：系统内部与 UI/轨迹记录的原子事件，带 `EventSource`（USER/AGENT/ENVIRONMENT）。`openhands/events/event.py`
- **LLM Message**：仅在构建 prompt/调用模型时生成，带 LLM 的 `role`（system/user/assistant/tool）。`openhands/core/message.py` `openhands/memory/conversation_memory.py`

因此：UI/轨迹里看不到 `role=tool` 是正常的，因为 `role=tool` 仅在 prompt 组装阶段产生。

## 端到端主流程（Web 会话，V0）

1. **前端 -> WebSocket**
   - 前端通过 Socket.IO 发送用户输入。
   - 连接时服务端会回放历史事件。`openhands/server/listen_socket.py`

2. **ConversationManager 分发**
   - 将用户输入转发给对应会话。`openhands/server/conversation_manager/standalone_conversation_manager.py`

3. **Session 接收并写入 EventStream**
   - `session.dispatch` 将 data 反序列化为 Event（通常是 MessageAction），并写入 EventStream（source=USER）。`openhands/server/session/session.py`

4. **AgentSession 启动与组件装配**
   - 创建 Runtime、Memory、Controller；并将初始 MessageAction 送入 EventStream。`openhands/server/session/agent_session.py`

5. **Controller 注入 System Message**
   - Controller 初始化时调用 `Agent.get_system_message()` 创建 SystemMessageAction，写入 EventStream。`openhands/controller/agent_controller.py` `openhands/controller/agent.py`

6. **EventStream 广播**
   - EventStream 将新事件投递给订阅者：Controller、Runtime、Memory 等。`openhands/events/stream.py`

7. **Memory 处理 RecallAction**
   - Controller 收到用户 MessageAction 后发起 RecallAction；Memory 监听后回写 RecallObservation（ENVIRONMENT）。`openhands/controller/agent_controller.py` `openhands/memory/memory.py`

8. **Controller 调用 Agent.step**
   - Controller 在 `_step` 中调用 `agent.step(state)`，生成下一步 Action。`openhands/controller/agent_controller.py`

9. **Agent 组装 LLM messages 并调用 LLM**
   - `CodeActAgent.step` 调用 ConversationMemory 生成 LLM messages，调用 `LLM.completion`。`openhands/agenthub/codeact_agent/codeact_agent.py` `openhands/memory/conversation_memory.py` `openhands/llm/llm.py`

10. **LLM 响应 -> Action**
    - `response_to_actions` 将 LLM tool calls 或文本回复转换为具体 Action，并写入 ToolCallMetadata。`openhands/agenthub/codeact_agent/function_calling.py`

11. **Runtime 执行 Action -> Observation**
    - Runtime 执行 Action 产生 Observation，并把 `tool_call_metadata` 复制到 Observation。`openhands/runtime/base.py`

12. **Session 将事件发给前端**
    - Session 收到 Action/Observation 事件后发送到 UI。`openhands/server/session/session.py`

## 关键路径：Event -> LLM Message

### EventSource 与 LLM role 的区别
- EventSource 是事件的来源：USER/AGENT/ENVIRONMENT。`openhands/events/event.py`
- LLM role 是 prompt 里的角色：system/user/assistant/tool。`openhands/core/message.py`
- 二者相互独立。

### MessageAction 与 Observation 的映射
ConversationMemory 将事件转换为 LLM Message：
- `MessageAction(source=USER)` -> role=user
- `MessageAction(source=AGENT)` -> role=assistant
- `SystemMessageAction` -> role=system
- `Observation` -> 默认 role=user（环境反馈）
- **Observation 带 `tool_call_metadata` 时 -> role=tool**

实现细节：`openhands/memory/conversation_memory.py`

## Tool Call 处理机制

### 1) LLM 返回 tool_calls -> Action
`response_to_actions` 解析 tool_calls 并生成 Action：
- 例如：CmdRunAction、IPythonRunCellAction、FileEditAction、MCPAction 等。
- 每个 Action 会附带 ToolCallMetadata（tool_call_id、function_name、原始 model_response）。  
`openhands/agenthub/codeact_agent/function_calling.py` `openhands/events/tool.py`

### 2) Runtime 执行 Action -> Observation
Runtime 执行 Action 并产生 Observation：
- Observation 会继承 Action 的 `tool_call_metadata`。  
`openhands/runtime/base.py`

### 3) Observation -> role=tool
在下一轮 prompt 组装时：
- 如果 Observation 带 `tool_call_metadata`，ConversationMemory 生成 role=tool 的 Message，包含 tool_call_id 与 name。  
`openhands/memory/conversation_memory.py`

### 4) Native 与非 Native 工具调用
- **Native tool calling**：LLM 直接返回 `tool_calls`，系统使用 role=tool 消息与 tool_call_id 对齐。
- **非 native tool calling（mock）**：LLM 输入输出被转换成“文本协议”形式，工具调用与结果被嵌入普通文本里；此时不会出现 role=tool。
  - 转换发生在 `LLM.completion` 内部，使用 `convert_fncall_messages_to_non_fncall_messages` / `convert_non_fncall_messages_to_fncall_messages`。  
  `openhands/llm/llm.py` `openhands/llm/fn_call_converter.py`

## 为什么轨迹里看不到 role=tool / tool response

1. **轨迹基于 Event，而不是 LLM Message**
   - 轨迹序列化走 `event_to_trajectory`，记录的是 Action/Observation。  
     `openhands/events/serialization/event.py` `openhands/controller/state/state_tracker.py`

2. **role=tool 只在 prompt 构建阶段出现**
   - ConversationMemory 只在生成 LLM messages 时才创建 role=tool。  
     `openhands/memory/conversation_memory.py`

3. **Tool response 在轨迹里表现为 Observation**
   - 例如 CmdOutputObservation、FileReadObservation、MCPObservation 等，带 `tool_call_metadata`。  
     `openhands/events/observation/*`

4. **上下文压缩会“抹掉”工具输出**
   - 若启用 discard_all/tool_response_discard 策略，tool 输出 Observation 会被替换为 `content="omitted"`。  
     `openhands/memory/condenser/impl/tool_response_discard_condenser.py`

5. **非 native tool calling 不会生成 role=tool**
   - 工具调用和结果都被写成普通文本，因此轨迹里也只看到普通 Observation/MessageAction。  
     `openhands/llm/fn_call_converter.py`

## UI 与事件发送

Session 会把 Action/Observation 事件发送给前端：
- USER/AGENT 的 Action、Observation 会被转为 `oh_event` 推送。  
`openhands/server/session/session.py`

注意：这仍是 Event 层数据，前端并不会显示 LLM role=tool 的消息。

## 如何重建“LLM 真实看到的 messages”

如果需要对齐 LLM 实际 prompt：
1. 从 EventStore/轨迹读取事件列表。
2. 使用 ConversationMemory.process_events 构建 LLM messages。
3. 用 `LLM.format_messages_for_llm` 得到最终传给模型的 payload。

关键文件：
- `openhands/events/event_store.py`
- `openhands/memory/conversation_memory.py`
- `openhands/llm/llm.py`

## 关键文件速查

- `openhands/server/listen_socket.py` WebSocket 入口与事件回放
- `openhands/server/session/session.py` 前端数据 -> EventStream
- `openhands/server/session/agent_session.py` runtime/memory/controller 初始化
- `openhands/controller/agent_controller.py` Agent 主循环与状态机
- `openhands/agenthub/codeact_agent/codeact_agent.py` LLM 调用与 message 组装
- `openhands/agenthub/codeact_agent/function_calling.py` tool_calls -> Action
- `openhands/memory/conversation_memory.py` Event -> LLM Message
- `openhands/llm/llm.py` LLM 调用与 function calling 模式
- `openhands/llm/fn_call_converter.py` 非 native 工具调用转换
- `openhands/runtime/base.py` Action 执行 -> Observation
- `openhands/events/serialization/event.py` 轨迹序列化
