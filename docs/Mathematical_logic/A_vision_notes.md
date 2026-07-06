# A 视觉模块说明

A 同学当前负责的是只基于图像帧的感知层：

```python
from submissions.vision import VisionState, detect

symbol_map = detect(obs)  # obs 是 RGB frame，shape 为 (128, 160, 3)

vision = VisionState()
symbol_map = vision.observe(obs, reward=last_reward)
```

检测器不读取 `info`、地图 JSON、物体坐标，也不读取环境的结构化 `grid` observation。结构化网格只在 `submissions/vision_smoke.py` 中作为本地调试 oracle 使用。

## 输出约定

`detect(frame)` 返回 `SymbolMap`：

- `grid`：shape 为 `(8, 10)` 的 `uint8` 数组
- `player`：`(x, y) | None`
- `monsters`、`chests`、`exits`、`walls`、`traps`、`buttons`、`switches`、`npcs`、`gaps`、`bridges`：由 `(x, y)` 组成的 tuple
- `blocked_tiles()`：墙、箱子、NPC、缺口
- `danger_tiles()`：陷阱、怪物
- `passable_tiles(avoid_danger=True)`：供 planner/BFS/A* 使用的候选可通行格子

`normalize_agent_observation(obs, reward, inventory)` 用于最终策略的输入归一化。它可以接收原始像素数组，也可以接收 README 风格、包含 `frame` 的字典观测，并且只向后传递允许使用的输入：frame、reward、显式 inventory。

Cell code 与 NesyLink 的 debug grid 保持一致：

| Code | Cell |
|---:|---|
| 0 | empty |
| 1 | wall |
| 2 | player |
| 3 | monster |
| 4 | chest |
| 5 | exit |
| 6 | trap |
| 7 | button |
| 8 | npc |
| 9 | gap |
| 10 | bridge |
| 11 | switch |

## 当前方法

地图大小是 10 x 8，每个格子是 16 x 16 像素。检测器会把 RGB frame 切成格子，并通过 renderer 的精确颜色计数对每个格子分类。这个方法刻意保持简单、可解释，也方便后续在报告或形式化描述中作为“图像到符号抽象”的假设说明。

目前公开 Task1-Task5 的初始帧都能和环境 debug grid 精确对齐。

对于移动过程中的帧，集成 planner 时应优先使用 `VisionState.observe(...)`，不要只用无状态的 `detect(...)`。

## 已知限制

像素级移动时，玩家和怪物的精灵图可能跨在两个格子之间。因此单帧检测器在接近 tile 边界的少数中间帧上，可能和 debug grid 不完全一致。规划时建议采用以下做法之一：

1. 使用 `VisionState.observe(frame)` 在多帧之间保留静态地图记忆。
2. Executor 重复移动动作，直到玩家接近 tile 对齐后再重新规划。
3. 在移动过程中，把玩家/怪物位置当成近似位置处理。

## 本地检查

```bash
python submissions/vision_smoke.py --steps 80
python submissions/vision_benchmark.py --steps 200
```

这些脚本会运行本地 oracle 对比，只用于调试，不应出现在最终策略的推理路径中。

最新 benchmark 表格和两天压缩交付 checklist 见 `docs/Mathematical_logic/A_delivery_plan.md`。
