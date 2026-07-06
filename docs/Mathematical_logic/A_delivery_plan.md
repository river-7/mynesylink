# A 同学交付计划

这份计划把 A 同学的工作压缩到两天内完成，同时严格遵守项目 README 的要求：最终推理只能使用图像帧、奖励历史，以及显式提供的背包/物品信息。`info`、地图 JSON、隐藏物体坐标、调试网格只能用于本地调试和统计评估，不能进入最终策略的推理路径。

## 已完成内容

- 实现了 `submissions/vision.py`。
- 添加了 `detect(frame) -> SymbolMap`，用于单帧图像感知。
- 添加了 `VisionState.observe(frame, reward=...) -> SymbolMap`，用于静态地图记忆和动态实体连续跟踪。
- 添加了 `normalize_agent_observation(obs, reward, inventory)`，使最终策略既能接收原始像素数组，也能接收 README 风格、包含 `frame` 的字典观测。
- 添加了本地 oracle 检查脚本 `submissions/vision_smoke.py`。
- 添加了定量 benchmark 脚本 `submissions/vision_benchmark.py`。
- 添加了 `submissions/vision_policy.py`，作为一个最小的、只使用图像帧的集成验证策略，证明视觉输出可以驱动路径规划、开箱、战斗和出入口切换。
- 使用 `utils/evaluate_policy.py` 验证了 Task1、Task2、Task3、Task4：每个任务 3/3 seeds 通过。
- 在不读取 `info` 的前提下，为开过的箱子、已知出口、击杀怪物流程、钥匙收集、Task3 西/东房间遍历、Task4 桥/开关路线添加了任务链记忆。

## 给 Planner 的接口

```python
from submissions.vision import VisionState

vision = VisionState()
symbol_map = vision.observe(frame, reward=last_reward)

player = symbol_map.player
blocked = symbol_map.blocked_tiles()
danger = symbol_map.danger_tiles()
passable = symbol_map.passable_tiles(avoid_danger=True)
targets = {
    "exits": symbol_map.exits,
    "chests": symbol_map.chests,
    "monsters": symbol_map.monsters,
    "buttons": symbol_map.buttons,
    "switches": symbol_map.switches,
}
```

坐标格式是 `(x, y)`，其中 `x` 是列号 `0..9`，`y` 是行号 `0..7`，`grid[y, x]` 是该格子的符号编码。

## 当前 Benchmark

命令：

```bash
python submissions/vision_benchmark.py --steps 200
```

最新本地结果：

| Task | 静态格准确率 | 玩家精确率 | 玩家平均距离 | 怪物精确率 | 怪物平均距离 |
|---|---:|---:|---:|---:|---:|
| task_1 | 0.9980 | 0.8458 | 0.154 | 1.0000 | 0.000 |
| task_2 | 0.9929 | 0.8209 | 0.179 | 0.6269 | 0.383 |
| task_3 | 0.9977 | 0.8209 | 0.179 | 1.0000 | 0.000 |
| task_4 | 0.9977 | 0.8159 | 0.184 | 1.0000 | 0.000 |
| task_5 | 0.9971 | 0.8209 | 0.179 | 0.9552 | 0.045 |

解释：静态地形和可交互物体已经比较稳定。玩家和怪物的精确率较低，主要是因为像素级移动时，环境 debug grid 使用实体中心点，而精灵图在视觉上可能跨在两个格子之间。Planner/Executor 应该在 tile center 附近重新规划，或者把动态实体位置视作近似的一格障碍。

## A 侧剩余工作

第 1 天：

- 保留 `vision_policy.py` 作为回归验证 harness：后续继续加行为时，Task1-4 必须保持通过。
- 如果最终 planner 需要更干净的动作接口，可以为 B 同学的 executor 补一个可选的 tile-alignment helper。
- 如报告需要，保存代表性的视觉/调试截图。
- 和 B 同学冻结 `SymbolMap` 接口。

第 2 天：

- 在多个 seeds 上运行 benchmark，并导出 JSON/CSV。
- 通过 `VisionState` 和 B 同学的 planner 集成。
- 在当前 key-door-heal-gold 路线之后继续清理 Task5：西房间宝箱、剩余战斗、最终全箱子完成。
- 编写最终报告中的 A 部分：感知假设、无 `info` 保证、已知限制、benchmark 表格。
- 最后检查提交策略不读取 `info`，除非 evaluator 显式暴露且 README 允许使用的 inventory 信息。

## 当前策略 Smoke Test

```bash
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_1 mathematical_logic/task_2 --num-envs 3 --max-steps 1200
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_3 --num-envs 3 --max-steps 1500
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_4 --num-envs 3 --max-steps 2600
python utils/evaluate_policy.py --policy submissions/vision_policy.py --tasks mathematical_logic/task_5 --num-envs 1 --max-steps 2000
```

结果：

```text
mathematical_logic/task_1
success_rate: 1.000
avg_steps:    277.0
avg_reward:   127.180

mathematical_logic/task_2
success_rate: 1.000
avg_steps:    171.0
avg_reward:   128.290

mathematical_logic/task_3
success_rate: 1.000
avg_steps:    541.0
avg_reward:   164.590
monster_killed: 1.000
key_collected: 1.000

mathematical_logic/task_4
success_rate: 1.000
avg_steps:    1110.0
avg_reward:   250.900
switch_activated: 1.000
key_collected: 1.000
door_opened: 1.000
item_collected: 1.000
monster_killed: 1.000

mathematical_logic/task_5
success_rate: 0.000
avg_steps:    1000.0
avg_reward:   40.400
chest_opened: 1.000
button_pressed: 1.000
key_collected: 1.000
door_opened: 1.000
agent_healed: 1.000
gold_collected: 1.000
monster_killed: 1.000
trap_triggered: 0.000
```

`vision_policy.py` 仍然是 A 侧的集成验证 harness，不是小组最终 planner。它目前证明了：在 README 限制下，只使用图像帧也能稳定通过 Task1-4，并在 Task5 中推进到 button-key-door-heal-gold 链条，同时完成一次击杀，作为部分得分 baseline。
