import State

/-!
  Day 4 目标谓词。

  本文件只负责定义 Goal Predicate：GoalReached、MonsterKilled、
  ChestOpened、DoorOpened。它们描述 Python FSM/Planner 需要追踪的
  可验证目标状态。
-/

namespace NesyLink

/-- 某个位置存在指定类型的对象。 -/
def ObjectAt (s : State) (kind : ObjectKind) (p : Coord) : Prop :=
  ∃ obj, obj ∈ s.objects ∧ obj.kind = kind ∧ obj.pos = p

/-- 指定位置的怪物已经被击败。 -/
def MonsterKilled (s : State) (p : Coord) : Prop :=
  ∃ obj,
    obj ∈ s.objects ∧
    obj.kind = ObjectKind.monster ∧
    obj.pos = p ∧
    obj.monsterState = MonsterState.defeated

/-- 指定位置的宝箱已经打开。 -/
def ChestOpened (s : State) (p : Coord) : Prop :=
  ∃ obj,
    obj ∈ s.objects ∧
    obj.kind = ObjectKind.chest ∧
    obj.pos = p ∧
    obj.chestState = ChestState.opened

/-- 指定位置的门已经打开。 -/
def DoorOpened (s : State) (p : Coord) : Prop :=
  ∃ obj,
    obj ∈ s.objects ∧
    obj.kind = ObjectKind.door ∧
    obj.pos = p ∧
    obj.doorState = DoorState.open

/-- 当前状态是否满足给定高层目标。 -/
def GoalReached (s : State) (goal : CurrentGoal) : Prop :=
  match goal.kind, goal.target with
  | GoalKind.explore, _ => True
  | GoalKind.reach, some p => s.player = p
  | GoalKind.reach, none => False
  | GoalKind.collectKey, _ => HasKey s
  | GoalKind.openDoor, some p => DoorOpened s p
  | GoalKind.openDoor, none => False
  | GoalKind.killMonster, some p => MonsterKilled s p
  | GoalKind.killMonster, none => False
  | GoalKind.openChest, some p => ChestOpened s p
  | GoalKind.openChest, none => False
  | GoalKind.exitRoom, some p => s.player = p
  | GoalKind.exitRoom, none => False

end NesyLink
