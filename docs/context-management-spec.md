# OpenHands Context Management 改造 Spec（面向 AI Agent）

## 目的

- 在 OpenHands 中实现“可切换的上下文管理策略”，用于评测不同策略对 NL2RepoBench 的影响。
- 保持与现有 NL2RepoBench 流程兼容，支持多机与可复现部署（后续以镜像发布为主）。
- **本次实现以评测为目标**：策略落地优先“对齐参考脚本/论文实现意图”，而非优化 CLI 交互体验。仅在架构限制下做必要适配，并在文档中明确差异。

## 项目背景（高层认知）

- NL2RepoBench(具体代码目录位于 ../NL2RepoBench) 以 OpenHands headless 方式运行，通过 config.toml + Docker 启动并执行任务。
- 任务描述来自 start.md，执行结束后会做 Docker 化测试并产出结果。
- 我们将修改 OpenHands，使其支持多种 context strategy（具体如： ReAct / Discard All / Summary / Mem1 / Folding Agent / DynaContext）。之后会有更具体详细的说明。
- 评测目标是“策略差异”与“模型差异”的可控对比。
- **默认运行模式**：无头任务模式（headless），仅在起始注入一次用户任务描述；之后由 Agent 自主执行直到完成或报错。

———

## 改造目标

1. 新增上下文策略选择机制
- 优先支持：环境变量、配置文件项（config.toml）、CLI 参数三选一或多入口统一。
- 约定一个统一入口，如 context_strategy 字段或 CONTEXT_STRATEGY 环境变量。
2. 逐步实现每个策略
- ReAct(baseline)：超出设置的上下文限制后直接报错退出，视为失败
- Discard All：达到设置的上下文限制后触发，会自动去除所有对话历史中的tool response结果
- Summary：达到设置的上下文限制后触发，由模型进行总结后，去除旧内容，用总结后的内容来代替继续进行指导完成（可能是当前 OpenHands 的默认方式，但也可能需要修改来对齐我们的版本，具体待定）
- Mem 1：已实现为 Mem1 风格总结（维护累积的 <think> 记忆块）
- Folding Agent：需要查阅具体论文，暂时待定
- DynaContext：我们本次重点实现的对象，需要查阅相关具体文档，待定
- strategy-X：新策略（例如强化裁剪/汇总/检索等）
- 后续可扩展为 N 种，目前先聚焦实现已有的几种和等待补充完善一些方法的文档指导
3. 可追踪性
- 日志中必须记录：所选策略、关键参数、触发点、摘要内容、裁剪比例等。
- 需要能在运行后复盘策略效果（日志或轨迹可定位）。

———

## 当前实现状态（V0 headless 入口）

### 入口与优先级
- 环境变量：`AGENT_CONTEXT_STRATEGY`
- config.toml：`[agent] context_strategy = "summary"`（或 `[agent.<name>]` 覆盖）
- CLI：`--context-strategy summary`

CLI 覆盖同名字段；未设置时沿用原有 condenser 逻辑，不改变默认行为。

### 最大上下文限制（显式参数）
- 环境变量：`AGENT_CONTEXT_WINDOW_LIMIT_TOKENS`
- config.toml：`[agent] context_window_limit_tokens = 128000`
- CLI：`--context-window-limit-tokens 128000`

超过限制时会触发对应策略；`react` 会直接报错退出，`summary`/`discard_all` 会触发 condensation。

### 已实现策略映射
- `react`：`NoOpCondenser` + `enable_history_truncation = false`（上下文超限直接退出或者声明某个error flag可以被捕获）
- `summary`：`LLMSummarizingCondenser`（LLM 总结替换旧内容；可通过 condenser 参数控制触发条件）
- `discard_all`：`ToolResponseDiscardCondenser`（在触发 condensation request 时，将工具响应的内容替换为 "omitted"）
- `mem1`：`Mem1Condenser`（更新 <think> 记忆块并保留少量最近事件）

> 其他策略（`mem1` / `folding` / `dynacontext` / `strategy-x`）为待实现占位，后续补文档与实现。

提示：当 `context_strategy = "summary"` 时，如需覆盖默认阈值或仅在 condensation request 时触发，请通过 `[condenser]` 自定义 `max_size` 与 `trigger_on_max_size`。

### 使用示例
- 环境变量：
  - `AGENT_CONTEXT_STRATEGY=summary openhands -t "..."`
  - `AGENT_CONTEXT_STRATEGY=mem1 openhands -t "..."`
- config.toml：
  - `[agent]`
  - `context_strategy = "discard_all"`
  - `context_strategy = "mem1"`
  - `[condenser]`
  - `type = "llm"`
  - `trigger_on_max_size = false`
  - `max_size = 100`
- CLI：
  - `openhands --context-strategy react -t "..."`
  - `openhands --context-strategy mem1 -t "..."`

### 可追踪性（日志与元数据）
- Condenser 会在日志中输出策略名、触发点与裁剪统计。
- 事件元数据记录在 `State.extra_data["condenser_meta"]` 中（可用于复盘策略效果）。

## AI Agent 代码阅读与理解流程（先读再改）

1. 定位上下文管理入口
- 找 “消息拼接 / prompt 组装 / conversation history / memory / summarizer / condenser” 等关键模块。
- 关键词搜索建议：context, memory, conversation, history, condense, summarize, truncate, messages.
2. 梳理调用链
- 找从“任务执行 / Agent loop”到 “prompt 生成”的调用路径。
- 用 1 张简单的流程图或列表记录“入口 -> 处理 -> 输出”。
3. 明确可插拔点
- 看是否已有策略类、配置项、hook、接口（比如 Condenser, Memory, PromptBuilder）。
- 如果已有扩展机制，优先复用。
4. 明确输入/输出
- 输入：消息历史、系统提示、工具调用记录、上下文长度限制
- 输出：传给 LLM 的最终 message list / prompt

## 记录方式（必须执行）

- 在 docs/notes/context-management-progress.md 维护：
- 当前步骤（例：Step 2/5: locate prompt builder）
- 发现的关键文件和函数
- 修改记录（文件路径 + 变更要点）
- 遇到的问题与下一步
- 保持“每个开发阶段”一次更新，时间戳可选。

———

## 阶段计划（建议）

阶段 0：对齐目标

- 明确策略需求：策略命名、输入参数、预期效果
- 确认输出：日志中必须包含哪些字段

阶段 1：代码调研

- 完成入口定位 & 调用链梳理
- 输出：docs/notes/context-management-progress.md 中的流程说明

阶段 2：设计

- 决定策略接口形式（类/函数/策略枚举）
- 决定参数来源（env / config / CLI）
- 产出：docs/notes/context-management-progress.md 中的“接口设计”部分

阶段 3：实现

- 实现 StrategyRegistry 或 ContextStrategy 抽象
- 接入现有 prompt builder / conversation assembler
- 增加日志输出（策略名、关键统计）

阶段 4：验证

- 写最小测试或在本地跑 1 个任务（手工也可）
- 验证策略切换生效（日志可见）
- 记录问题/修复

阶段 5：整理

- 更新 README 或 docs/context-management-spec.md
- 列出如何使用策略（env/config/CLI）

———

## 风险点与注意事项

- 不要破坏默认行为（baseline 必须一致）
- 大量裁剪或摘要可能影响指令完整性，优先记录裁剪内容
- 确保日志不会过度膨胀（限制长度）
- 如果 OpenHands 已有 Condenser/Memory 机制，务必复用而不是重造

———

## 交付物清单

- 新策略机制代码（策略接口 + 至少一个新策略）
- 文档：
- docs/context-management-spec.md（使用说明）
- docs/notes/context-management-progress.md（过程记录）
- 日志支持：运行时输出策略选择与关键统计

———

## AI Agent 执行准则

- 先调研再改代码；每阶段至少一次记录
- 修改尽量小且可回退（避免一次性大改）
- 若不确定结构，先提交“定位报告”再改
