# A 视觉模块说明（新版测评）

## 责任边界

A 同学只负责把正式测评提供的 RGB `obs` 转为 `SymbolMap`，以及感知层的离线 oracle 测试。路径规划、Goal/FSM、动作下发和 Task 5 策略属于 B 同学，不在本模块修改范围内。

最新版 `utils/evaluate_policy.py` 直接传入 shape 为 `(128, 160, 3)` 的 `uint8` 数组，而不是旧文档示意的 `obs["frame"]`。视觉模块同时保留字典输入兼容层，但正式入口以原始数组为准。

```python
from submissions.vision import VisionState, detect

symbol_map = detect(obs)

vision = VisionState()
symbol_map = vision.observe(obs, reward=last_reward)
```

运行时只读取像素。`info`、地图 JSON、结构化 grid、对象坐标和测评变体名称均不进入检测路径；grid 只在 `vision_smoke.py` 和 `vision_benchmark.py` 中作为本地 oracle。

## 输出契约

坐标统一为 `(x, y)`，`grid[y, x]`；地图为 10 × 8，每格 16 × 16 像素。

| Code | Cell |
|---:|---|
| 0 | empty |
| 1 | wall |
| 2 | player |
| 3 | monster |
| 4 | chest（包括带 key/gold/heal/sword 图标的宝箱） |
| 5 | exit / door |
| 6 | trap |
| 7 | button |
| 8 | npc |
| 9 | gap / abyss |
| 10 | bridge |
| 11 | switch |

`SymbolMap` 继续提供 `blocked_tiles()`、`danger_tiles()` 和 `passable_tiles()`；接口没有因本次修复而改变。

## 新版颜色鲁棒方案

正式 color 阶段循环 `grayscale`、`dark`、`bright`、`high_contrast`、`inverted`。旧实现只统计原始 RGB 精确值，因此五种变体全部失效。

新版实现分两层：

1. 从像素本身判断确定性颜色变换，不依赖 evaluator 传递变体名称。
2. `dark`、`bright`、`inverted` 使用对应调色板计数；`grayscale` 和 `high_contrast` 已丢失部分颜色信息，改用渲染形状模板做逐 tile 最近匹配。

模板只由公开 renderer 的几何绘制规则生成，不读取当前任务地图或运行时状态。模板结果按 `(x, y, variant)` 缓存，并将候选堆叠后用 NumPy 向量化比较。

## 测试方法与当前结论

```powershell
.\.venv\Scripts\python.exe submissions\vision_smoke.py --steps 80
.\.venv\Scripts\python.exe submissions\vision_benchmark.py --steps 200
.\.venv\Scripts\python.exe submissions\vision_benchmark.py `
  --steps 0 --obs-variants default `
  --map-variants spatial_a spatial_b spatial_c
```

2026-07-12，seed 0：

- 五个原始任务的初始帧：逐格完全对齐。
- 五个任务 × 五种颜色变体的初始帧：逐格完全对齐。
- 五个任务 × 三种空间地图变体的初始帧：逐格完全对齐。
- Task 1 使用现有集成策略分别跑六种观测，均 1/1 成功；这是回归样本，不是正式 100 轮成绩。
- Task 2 单 seed 回归中，灰度 1/1 成功；高对比完成击杀、开箱和拿钥匙，但 500 步内未到达出口。该失败尚不能仅凭事件统计判定为视觉或 Planner 问题。
- 100 步灰度随机 rollout 的单次本机耗时约 2.9 秒，Task 1 静态准确率 0.9994，玩家精确率 0.9505。

JSON 样本保存在 `results/vision_color_seed0.json`、`results/vision_spatial_seed0.json`、`results/task1_color_after_vision.json` 和 `results/task2_lossy_color_after_vision.json`。正式报告仍应按新版要求运行每 task 100 episodes 的 `--robustness-suite`，并把策略成功率与视觉准确率分开报告。

## 已知限制

- 玩家和怪物在像素移动中会跨 tile，动态位置与 grid oracle 可能短暂相差一格；`VisionState` 会保留静态地图，但动态实体仍应视为近似位置。
- 高对比变体会把多种颜色压成相同的 0/255 通道，怪物移动帧比静态物体更难区分；当前初始检测准确，但仍需长 rollout 和多 seed 统计。
- 视觉层不会修复空间变体中的策略硬编码。若 `SymbolMap` 正确而通关失败，应交给 Planner/FSM 侧处理。
- 视觉层不直接判断宝箱内的 key/sword 等语义；这些信息由宝箱图标和正式允许的 inventory 共同供上层使用，接口冻结前如需细分类别，应由 A/B 双方评审后新增字段。

## 2026-07-13 原色回归修复

颜色鲁棒版本曾把动态实体候选限制为 `EMPTY`，导致玩家经过 exit、bridge 等非空 tile 时被错放到相邻格，进而破坏多房间切换。修复后：

- 原色及 dark/bright/inverted 恢复原有动态定位行为。
- grayscale/high_contrast 仅对真正有颜色混淆风险的动态候选做限制；玩家仍可覆盖 exit、bridge、trap、button、switch 和 gap。
- Task 3 seed 0 恢复为 543 步成功，Task 4 seed 0 恢复为 1699 步成功。
- 新增 Task 3 向西穿越出口的 scripted smoke regression。
- 5 tasks × 3 spatial variants × default/grayscale/high_contrast 的初始帧全部逐格一致。
