# A 同学交付计划（按最新版要求修订）

## 最新基线的正确解读

组长提供的是端到端任务成功率，不是单独的视觉准确率：

| Task | Original | Spatial | Color | 需要区分的问题 |
|---|---:|---:|---:|---|
| Task 1 | 100% | 0% | 0% | Color 明确暴露旧视觉精确颜色依赖；Spatial 还可能包含 Planner 坐标硬编码 |
| Task 2 | 100% | 66.7% | 0% | 感知颜色鲁棒性需修复；剩余空间失败需由 B 侧根据符号图排查 |
| Task 3 | 100% | 33.3% | 0% | 同上，多房间记忆/目标顺序不属于 A 侧 |
| Task 4 | 100% | 33.3% | 0% | A 侧保证桥、gap、switch 等可见；机制策略属于 B 侧 |
| Task 5 | 0% | 0% | 0% | 原图也失败，主要是综合策略未完成，不能归因给颜色感知 |

因此本轮 A 侧交付标准是：证明 `SymbolMap` 在 original/spatial/color 像素上是否正确，并保持既有接口；不修改 `planner.py`、`fsm.py`、`agent.py`、`vision_policy.py` 或学生策略文件。

## 已完成

- 安装 Python 3.12.10，创建仓库级 `.venv`，执行 `pip install -e .`。
- 更新 `submissions/vision.py`，支持正式五种颜色变体的自动识别。
- 对灰度/高对比观测增加不依赖地图状态的形状模板匹配和缓存。
- 保持 `detect(frame) -> SymbolMap`、`VisionState.observe(...)` 与 `SymbolMap` 字段不变。
- 扩展 `submissions/vision_benchmark.py`，可交叉测试 `--obs-variants` 和 `--map-variants`，并继续以 grid 作为仅测试期 oracle。
- 完成 5 tasks × 6 observations 的初始帧检查，全部逐格一致。
- 完成 5 tasks × 3 spatial variants 的初始帧检查，全部逐格一致。
- 完成 Task 1 六种观测的端到端单 seed 回归，全部成功。
- 完成 Task 2 灰度/高对比诊断：灰度成功；高对比已完成击杀、开箱和拿钥匙，但未完成出口阶段，需 A/B 联合做只读轨迹归因。
- 修复颜色鲁棒改动对原色动态实体覆盖语义的回归；Task 3/4 原图单 seed 已恢复旧版通关结果，且空间/颜色初始识别无退化。

## 下一轮 A 侧验收

1. 对五个任务、五种颜色变体运行至少 200 步、多 seed 的视觉 benchmark，重点统计移动怪物。
2. 与 B 同学冻结 `SymbolMap`：确认是否需要显式的宝箱 loot 图标类别；未确认前不擅自改接口。
3. 让 B 同学用现有 Planner 跑正式 `--robustness-suite`。A 只分析其中由错误符号图导致的失败轨迹。
4. Day 6 输出视觉 benchmark JSON/CSV、代表性截图和报告中的感知章节。
5. Day 7 只修视觉 bug，不新增规划功能。

## 正式测评命令

```powershell
.\.venv\Scripts\python.exe utils\evaluate_policy.py `
  --policy submissions\student_policy.py `
  --info-mode safe `
  --robustness-suite `
  --num-envs 100 `
  --json-out results\robustness_suite_eval.json
```

按老师最新版文档，报告必须分别列出每个 Task 的 `original`、`spatial`、`color` 成功率，并记录实际命令、seed、episodes、代码版本及是否覆盖 `max_steps`/`action_repeat`。视觉 benchmark 只能解释错误来源，不能替代正式成功率。
