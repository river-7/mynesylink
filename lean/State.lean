import Move
import Object
import Inventory

/-!
  Day 3 符号状态层。

  本文件把 Day 2 的移动模型从单个玩家坐标提升为一个小型游戏状态抽象。
  `map` 字段表示视觉模块输出后的 `SymbolMap`，`objects` 表示房间中的
  符号对象，inventory 和 memory 则表示 planner/executor 接口会显式维护
  的状态。
-/

namespace NesyLink

/-- Task1/Task2 原型 planner 使用的高层目标。 -/
inductive GoalKind where
  | explore
  | reach
  | collectKey
  | openDoor
  | killMonster
  | openChest
  | exitRoom
  deriving BEq, Repr

instance : ToString GoalKind where
  toString := reprStr

/-- 当前符号目标。对于非位置型目标，target 可以为空。 -/
structure CurrentGoal where
  kind : GoalKind
  target : Option Coord
  deriving BEq, Repr

/-- planner 选出具体目标之前使用的默认探索目标。 -/
def CurrentGoal.explore : CurrentGoal :=
  { kind := GoalKind.explore, target := none }

/-- Planner 记忆。允许重复记录，表示该坐标被再次观测到。 -/
structure Memory where
  visited : List Coord
  deriving BEq, Repr

def Memory.empty : Memory :=
  { visited := [] }

def Memory.remember (memory : Memory) (p : Coord) : Memory :=
  { memory with visited := p :: memory.visited }

/-- Lean 定义与 Python planner 契约共享的 Day 3 状态抽象。 -/
structure State where
  map : Map
  objects : List Object
  player : Coord
  facing : Direction
  hp : Nat
  inventory : Inventory
  memory : Memory
  currentGoal : CurrentGoal

def Alive (s : State) : Prop :=
  s.hp > 0

def PlayerSafe (s : State) : Prop :=
  SafeState s.map s.player

def StateInvariant (s : State) : Prop :=
  Alive s ∧ PlayerSafe s

def HasKey (s : State) : Prop :=
  s.inventory.keys > 0

def HasCoin (s : State) : Prop :=
  s.inventory.coins > 0

def HasSword (s : State) : Prop :=
  s.inventory.hasSword = true

def HasShield (s : State) : Prop :=
  s.inventory.hasShield = true

def Visited (s : State) (p : Coord) : Prop :=
  p ∈ s.memory.visited

def State.objectsAt (s : State) (p : Coord) : List Object :=
  s.objects.filter (fun obj => obj.pos == p)

def State.hasObjectKind (s : State) (kind : ObjectKind) : Bool :=
  s.objects.any (fun obj => obj.kind == kind)

def actionDirection? : Action → Option Direction
  | Action.moveUp => some Direction.up
  | Action.moveDown => some Direction.down
  | Action.moveLeft => some Direction.left
  | Action.moveRight => some Direction.right
  | _ => none

def updateFacing (old : Direction) (a : Action) : Direction :=
  match actionDirection? a with
  | some dir => dir
  | none => old

/--
单个符号动作对应的状态级转移。

移动部分使用 Day 2 的 `next_state` 安全语义。会改变物品栏或对象状态的
事件在 `Transition.lean` 中作为 `attack` 与 `interact` 定义。
-/
noncomputable def nextState (s : State) (a : Action) : State :=
  let player' := next_state s.map s.player a
  { s with
    player := player',
    facing := updateFacing s.facing a,
    memory := s.memory.remember player' }

def collectKey (s : State) : State :=
  { s with inventory := s.inventory.addKey }

def spendKey (s : State) : State :=
  { s with inventory := s.inventory.spendKey }

def collectCoin (amount : Nat) (s : State) : State :=
  { s with inventory := s.inventory.addCoin amount }

/-- 兼容旧命名：金币在新计划中按 coin 建模。 -/
def collectGold (amount : Nat) (s : State) : State :=
  collectCoin amount s

def equipSword (s : State) : State :=
  { s with inventory := s.inventory.equipSword }

def equipShield (s : State) : State :=
  { s with inventory := s.inventory.equipShield }

def takeDamage (amount : Nat) (s : State) : State :=
  { s with hp := s.hp - amount }

def heal (amount : Nat) (s : State) : State :=
  { s with hp := s.hp + amount }

def setGoal (goal : CurrentGoal) (s : State) : State :=
  { s with currentGoal := goal }

def clearGoal (s : State) : State :=
  { s with currentGoal := CurrentGoal.explore }

theorem nextState_map_eq (s : State) (a : Action) :
    (nextState s a).map = s.map := by
  simp [nextState]

theorem nextState_player_eq (s : State) (a : Action) :
    (nextState s a).player = next_state s.map s.player a := by
  simp [nextState]

theorem nextState_hp_eq (s : State) (a : Action) :
    (nextState s a).hp = s.hp := by
  simp [nextState]

theorem nextState_remembers_player (s : State) (a : Action) :
    Visited (nextState s a) (nextState s a).player := by
  simp [Visited, nextState, Memory.remember]

theorem nextState_player_eq_self_of_not_allowed
    {s : State} {a : Action}
    (h : ¬ MoveAllowed s.map s.player a) :
    (nextState s a).player = s.player := by
  simp [nextState, next_state_eq_self_of_not_allowed h]

theorem allowed_nextState_player_safe
    {s : State} {a : Action}
    (h : MoveAllowed s.map s.player a) :
    PlayerSafe (nextState s a) := by
  unfold PlayerSafe
  simp [nextState]
  exact allowed_move_preserves_next_state h

theorem allowed_nextState_preserves_invariant
    {s : State} {a : Action}
    (hinv : StateInvariant s)
    (h : MoveAllowed s.map s.player a) :
    StateInvariant (nextState s a) := by
  constructor
  · simpa [Alive, nextState] using hinv.1
  · exact allowed_nextState_player_safe h

theorem collectKey_hasKey (s : State) :
    HasKey (collectKey s) := by
  simp [HasKey, collectKey, Inventory.addKey]

theorem equipSword_hasSword (s : State) :
    HasSword (equipSword s) := by
  simp [HasSword, equipSword, Inventory.equipSword]

theorem equipShield_hasShield (s : State) :
    HasShield (equipShield s) := by
  simp [HasShield, equipShield, Inventory.equipShield]

theorem takeDamage_hp_le (amount : Nat) (s : State) :
    (takeDamage amount s).hp ≤ s.hp := by
  simp [takeDamage]

theorem heal_hp_ge (amount : Nat) (s : State) :
    s.hp ≤ (heal amount s).hp := by
  simp [heal]

theorem setGoal_currentGoal (goal : CurrentGoal) (s : State) :
    (setGoal goal s).currentGoal = goal := by
  rfl

theorem clearGoal_currentGoal (s : State) :
    (clearGoal s).currentGoal = CurrentGoal.explore := by
  rfl

end NesyLink
