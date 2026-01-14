# Mem1 策略实现说明

## 目标与对齐点
本实现将 Mem1 的“持续记忆”思想映射为 OpenHands 的 condenser：当触发凝缩时，用 LLM 更新一个累积的 `<think>...</think>` 记忆块，使其成为后续决策的主要上下文，并丢弃中间历史。核心对齐点：
- 记忆是“累积、简洁、可增长”的 `<think>` 内容。
- 新记忆必须融合“上一轮记忆 + 新信息”，并明确提示“旧事件即将被丢弃”。
- 记忆内保留关键信息（用户目标、约束、关键决策、工具结果、文件路径、错误、TODO）。

## 与官方实现的对应关系
参考官方仓库 `Mem1/gen_data/data_process/*.py` 中的提示规范：要求模型在每轮将关键信息写入 `<think>` 作为唯一持久记忆，并在新的观察信息后更新该记忆。我们做了以下等价映射：
- **官方做法**：每轮输出 `<think>`，并将历史上下文清空，只保留记忆。
- **本实现**：在“上下文超限”或设置的 `max_size` 触发时进行记忆更新；保留任务起始与少量近期事件（`keep_first` + `keep_last`），并将中间事件压缩为 `<think>`。

**差异说明（需审核）**：
- 这里的记忆更新触发点是“凝缩触发”，不是“每轮强制更新”。如果希望更接近官方“每轮压缩”的形态，可设置较小的 `max_size` 以提高更新频率。
- 未引入 Mem1 训练模型与检索模块，仅复用当前 LLM 进行记忆更新。

## 实现位置与入口
- 核心实现：`openhands/memory/condenser/impl/mem1_condenser.py`
- 配置结构：`openhands/core/config/condenser_config.py`（新增 `Mem1CondenserConfig`）
- 策略映射：`openhands/core/config/utils.py`（`context_strategy=mem1`）
- CLI 提示：`openhands/core/config/arg_utils.py`
- 模板提示：`config.template.toml`

## 核心算法（逐步）
1. **保留范围**：保留 `keep_first`（通常是任务起始）+ `keep_last`（近期上下文），系统消息不参与记忆融合。
2. **准备记忆更新输入**：
   - 提取上一轮 `<think>`（如不存在则为空）。
   - 将需要被丢弃的事件序列化为 `<EVENT id=...>`，放入 `<information>` 块。
3. **提示词**：要求模型输出“仅包含 `<think>...</think>` 的更新记忆”，并强调旧事件会被丢弃。
4. **产出与注入**：
   - 若输出缺失 `<think>` 标签，会自动补齐。
   - 生成 `CondensationAction`，携带 `summary=<think>...</think>` 与 `summary_offset=keep_first`。

## 关键参数
- `keep_first`：保留的前缀事件数，默认 1。
- `keep_last`：保留的最新事件数，默认 4（更接近“当前观察 + 记忆”）。
- `max_size`：可选的长度阈值；若未设置则仅在上下文超限时触发。
- `max_event_length`：单事件文本裁剪长度，避免提示过大。

## 日志与轨迹中的可观测点
### 日志
触发后会输出：
- `Mem1Condenser: summarized ...`（事件数与摘要长度）
- `Mem1Condenser summary: ...`（摘要预览）

### Condenser 元数据
在 `State.extra_data["condenser_meta"]` 中记录：
- `strategy=mem1`
- `trigger=condensation_request|max_size`
- `summary_length` / `forgotten_event_count` / `discard_ratio`
- `keep_first` / `keep_last`

### 轨迹 (trajectory)
`CondensationAction` 会被序列化到轨迹中，包含：
- `summary` 字段（完整 `<think>...</think>`）
- `metadata` 字段（策略与统计信息）

> 注意：`AgentCondensationObservation` 仅在内存视图中插入，不会作为事件写入轨迹；因此应以 `CondensationAction.summary` 为准。

## 验证建议
1. **单测**：
   - `tests/unit/memory/condenser/test_condenser.py::test_mem1_condenser_from_config`
2. **功能验证（建议）**：
   - 设置较小阈值触发凝缩：
     ```toml
     [agent]
     context_strategy = "mem1"

     [condenser]
     type = "mem1"
     llm_config = "condenser"
     max_size = 12
     keep_last = 2
     ```
   - 启动任务并保存轨迹（`save_trajectory_path`）。
   - 在轨迹中查找 `action=condensation` 的条目，检查 `summary` 是否为 `<think>...</think>`，以及 `metadata.strategy == "mem1"`。
3. **日志验证**：
   - 观察控制台输出是否包含 `Mem1Condenser:` 前缀日志。

## 可选改进点（如需更贴近官方）
- 将 `max_size` 设为较小值，近似“每轮更新记忆”。
- 进一步定制提示，加入官方的“结构顺序”和“必须总结 `<information>`”等硬约束。
