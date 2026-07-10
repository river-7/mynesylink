import Planner
import Transition

/-!
  Day 5 路径、计划与执行模型。

  成员 C 在 Day5 只负责定义 Path、Plan、Execution，并给出 `executePlan`。
  后续的 ValidPathSafe、ExecutePlanSafe、PlannerSound 等证明属于成员 D。
-/

namespace NesyLink

/-- 路径是 tile 坐标序列。 -/
abbrev Path := List Coord

/-- 相邻路径点之间应由某个移动动作连接。 -/
def ConsecutiveMove (fromPos toPos : Coord) : Prop :=
  ∃ action, isMoveAction action ∧ toPos = nextCoord fromPos action

/-- 路径合法性的基础定义：相邻坐标必须对应一次移动。 -/
def ValidPath : Path → Prop
  | [] => True
  | [_] => True
  | fromPos :: toPos :: rest =>
      ConsecutiveMove fromPos toPos ∧ ValidPath (toPos :: rest)

/-- 路径安全性的基础定义：路径中的每个点都是安全 tile。 -/
def PathSafe (m : Map) (path : Path) : Prop :=
  ∀ p, p ∈ path → SafeTile m p

/-- 执行记录：从 start 执行 plan 后到达 finalState。 -/
structure Execution where
  start : State
  plan : Plan
  finalState : State

/-- 依次执行一个 action plan。 -/
noncomputable def executePlan (s : State) : Plan → State
  | [] => s
  | action :: rest => executePlan (move s action) rest

noncomputable def runExecution (s : State) (plan : Plan) : Execution :=
  {
    start := s,
    plan := plan,
    finalState := executePlan s plan
  }

noncomputable def executeTaskState (ts : TaskState) : State :=
  executePlan ts.state ts.actionQueue

noncomputable def executePlannedTask (planner : PlannerModel) (ts : TaskState) : State :=
  let planned := planTask planner ts
  executeTaskState planned

end NesyLink
