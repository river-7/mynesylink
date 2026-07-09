import Goal

/-!
  Day 4 Planner/FSM 模型。

  本文件定义 Python FSM 对应的 Lean 抽象：
  `TaskState -> Goal -> Plan`。这里只建立模型，不证明 ActionMask
  正确性；ActionMask 证明属于成员 D。
-/

namespace NesyLink

/-- Python 侧 action queue / plan 的 Lean 抽象。 -/
abbrev Plan := List Action

/-- Day4 FSM 阶段，对应 PDF 中的 SEARCH_KEY 等阶段。 -/
inductive FsmPhase where
  | searchKey
  | openDoor
  | searchSword
  | killMonster
  | openChest
  | exitRoom
  | done
  deriving BEq, Repr

instance : ToString FsmPhase where
  toString := reprStr

/-- Planner 运行时任务状态。 -/
structure TaskState where
  state : State
  phase : FsmPhase
  goal : CurrentGoal
  actionQueue : Plan

def emptyPlan : Plan := []

def TaskState.withGoal (ts : TaskState) (goal : CurrentGoal) : TaskState :=
  { ts with goal := goal }

def TaskState.withPlan (ts : TaskState) (plan : Plan) : TaskState :=
  { ts with actionQueue := plan }

def hasAliveMonsterB (objects : List Object) : Bool :=
  objects.any (fun obj => obj.isAliveMonster)

def hasClosedChestB (objects : List Object) : Bool :=
  objects.any (fun obj => obj.isClosedChest)

def hasClosedDoorB (objects : List Object) : Bool :=
  objects.any (fun obj => obj.isClosedDoor)

def phaseGoalKind : FsmPhase → GoalKind
  | FsmPhase.searchKey => GoalKind.collectKey
  | FsmPhase.openDoor => GoalKind.openDoor
  | FsmPhase.searchSword => GoalKind.reach
  | FsmPhase.killMonster => GoalKind.killMonster
  | FsmPhase.openChest => GoalKind.openChest
  | FsmPhase.exitRoom => GoalKind.exitRoom
  | FsmPhase.done => GoalKind.explore

def goalForPhase (phase : FsmPhase) (target : Option Coord) : CurrentGoal :=
  { kind := phaseGoalKind phase, target := target }

/--
FSM 阶段推进规则。

它抽象对应 Python fsm 的任务链：
SEARCH_KEY -> OPEN_DOOR -> SEARCH_SWORD -> KILL_MONSTER -> OPEN_CHEST。
-/
def nextPhase (ts : TaskState) : FsmPhase :=
  match ts.phase with
  | FsmPhase.searchKey =>
      if ts.state.inventory.keys > 0 then FsmPhase.openDoor else FsmPhase.searchKey
  | FsmPhase.openDoor =>
      if hasClosedDoorB ts.state.objects then FsmPhase.openDoor else FsmPhase.searchSword
  | FsmPhase.searchSword =>
      if ts.state.inventory.hasSword then FsmPhase.killMonster else FsmPhase.searchSword
  | FsmPhase.killMonster =>
      if hasAliveMonsterB ts.state.objects then FsmPhase.killMonster else FsmPhase.openChest
  | FsmPhase.openChest =>
      if hasClosedChestB ts.state.objects then FsmPhase.openChest else FsmPhase.exitRoom
  | FsmPhase.exitRoom => FsmPhase.done
  | FsmPhase.done => FsmPhase.done

def syncGoalWithPhase (ts : TaskState) : TaskState :=
  let phase' := nextPhase ts
  { ts with
    phase := phase',
    goal := goalForPhase phase' ts.goal.target }

/-- Planner 模型：给定 TaskState 与 Goal，输出一段 Action plan。 -/
abbrev PlannerModel := TaskState → CurrentGoal → Plan

def planForGoal (planner : PlannerModel) (ts : TaskState) (goal : CurrentGoal) : Plan :=
  planner ts goal

def planTask (planner : PlannerModel) (ts : TaskState) : TaskState :=
  let ts' := syncGoalWithPhase ts
  let plan := planForGoal planner ts' ts'.goal
  ts'.withPlan plan

/-- 最小空 planner，占位用于接口联调；真实 Python planner 可替换该模型。 -/
def emptyPlanner : PlannerModel :=
  fun _ _ => emptyPlan

end NesyLink
