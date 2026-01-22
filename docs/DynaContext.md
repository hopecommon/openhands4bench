下面这份实现把 “DynaContext” 做成了一个**可中断、可投票决策、可回滚再压缩**的上下文管理环：当对话变长或判断“阶段性任务已完成/模型卡住”时，用摘要替换一段原始历史，从而把上下文窗口腾出来，但又尽量保留继续工作的必要信息。代码在 dynamic_context_agent_dev.py。

**机制拆解（它到底做了什么）**
- **两件事分开做：摘要 vs 是否替换**
  - `summarize_segment(...)`：把一段 history 格式化后喂给模型，要求严格输出 `<summary>...</summary>`，最多重试 3 次，失败则降级为 “Summary generation failed.”。
  - `judge_replacement(...)`：把“最近消息”交给模型，要求严格输出 `<reasoning>...` + `<decision>YES/NO</decision>`，用**多次投票**做鲁棒决策。
- **投票判定（减少单次 LLM 不稳定）**
  - 并发发起 `voting_count` 次判断（默认 `VOTING_K=5`），统计 YES 票数；阈值是 $\lceil K/2 \rceil$。
  - 若解析失败导致有效票不足，会继续串行补票；仍不足则按“有效票多数”重新计算阈值。
- **触发条件有两个：策略触发 + 硬上限触发**
  - 策略触发：每回合（过了“早期信任窗口”）异步启动 `judge_job(...)`，结果放进 `judgment_tasks_buffer`。
  - 硬上限触发：用 `count_input_tokens(...)` 估算 token，若 `context_length - input_length < 512`，直接强制压缩（即使投票还没 YES）。
- **关键细节：它不是“截断”，而是“回滚 + 新分支重放 + 摘要注入”**
  - 一旦触发，会选定 `msgs_to_summarize`（注意这里刻意做了 `snapshot[:-2]`，也就是**把当时最近两条去掉**，通常是为了避开“当前回合的 assistant/tool 交互尾巴”，减少把半成品塞进摘要）。
  - `_apply_summary_and_replacement(...)` 会在全局日志 `session_message` 上做一次**回滚对齐**：在 `session_message` 里从后往前找最后一条“要保留的消息”，截掉之后的内容，避免“投票触发点”和“真正应用摘要时点”之间产生漂移。
  - 然后创建一个新 agent 分支（`current = current + '+'`），并把摘要作为**新的 user 消息**注入：
    - 先 append 一个空 assistant：`{'role': 'assistant', 'content': ""}`
    - 再 append 摘要 user：`{'role': 'user', 'content': next_session_prompt}`
    这相当于模拟一次“上下文重置后，用户把已完成进度摘要给你，你继续做”。
- **“早期交互保留”窗口（避免过早压缩）**
  - `KEEP_EARLY_INTERACTION` 默认 1：`current_branch_turns > KEEP_EARLY_INTERACTION` 才开始投票判断。换句话说，新分支刚启动的前几轮不轻易压缩，避免摘要过早导致丢信息。
- **工具调用与摘要隔离**
  - `format_history_for_manager(... include_tool_content=False)`：给摘要/判定用的 history 不包含工具返回正文（只保留结构化信息）。这样更省 token、更不容易把冗长 tool 输出塞进摘要，但代价是可能丢失关键证据（例如搜索结果细节）。

---

## OpenHands 评测实现差异说明

本仓库中的 OpenHands 实现侧重“**策略意图对齐**”，但由于架构不同，无法 100% 复刻 refer 脚本的执行形态：
- **无回滚/分支重放语义**：OpenHands 使用 Condenser 返回 `CondensationAction` 直接替换历史，不做 session 回滚或分支重建。
- **尾部排除按事件类型**：为避免丢失最新 user 指令，非硬触发时仅排除尾部的 agent/tool 事件；硬触发时不排除尾部。
- **硬触发必定缩短**：在 context-limit 触发时，强制遗忘至少一个事件，避免反复 CondensationRequest 自旋。

这些差异是为评测验证所做的取舍，而非交互体验优化。
