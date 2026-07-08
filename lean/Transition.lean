import State

/-!
  Day 3 状态转移定义。

  CD 时间安排要求 Transition 只定义 move、attack、interact，暂时不证明。
  这里把移动、攻击、交互分别建模为 State 到 State 的符号更新。
-/

namespace NesyLink

def sameCoordB (a b : Coord) : Bool :=
  a.x == b.x && a.y == b.y

def coordDistance (a b : Coord) : Nat :=
  let dx := if a.x ≤ b.x then b.x - a.x else a.x - b.x
  let dy := if a.y ≤ b.y then b.y - a.y else a.y - b.y
  dx + dy

def adjacentB (a b : Coord) : Bool :=
  coordDistance a b == 1

def reachableForInteractB (player target : Coord) : Bool :=
  sameCoordB player target || adjacentB player target

def frontCoord (player : Coord) : Direction → Coord
  | Direction.up => { player with y := player.y - 1 }
  | Direction.down => { player with y := player.y + 1 }
  | Direction.left => { player with x := player.x - 1 }
  | Direction.right => { player with x := player.x + 1 }

def targetInFrontB (s : State) (target : Object) : Bool :=
  sameCoordB (frontCoord s.player s.facing) target.pos

def replaceObject (oldObj newObj : Object) : List Object → List Object
  | [] => []
  | obj :: rest =>
      if obj == oldObj then
        newObj :: rest
      else
        obj :: replaceObject oldObj newObj rest

/-- move：复用 Day3 `State.nextState`，其移动坐标来自 Day2 `next_state`。 -/
noncomputable def move (s : State) (a : Action) : State :=
  nextState s a

def canAttackB (s : State) (target : Object) : Bool :=
  s.inventory.hasSword &&
  target.isAliveMonster &&
  targetInFrontB s target

/-- 攻击动作的前置条件：有剑，并且面向方向的下一格是活怪物。 -/
def AttackPreconditionSpec (s : State) (target : Object) : Prop :=
  s.inventory.hasSword = true ∧
  target.isAliveMonster = true ∧
  targetInFrontB s target = true

/--
若符号执行层允许 attack，则攻击前置条件成立。

这覆盖课程要求中的核心条件：玩家拥有武器；同时目标格是怪物。
-/
theorem AttackPrecondition
    {s : State} {target : Object}
    (h : canAttackB s target = true) :
    AttackPreconditionSpec s target := by
  simp [AttackPreconditionSpec, canAttackB] at h ⊢
  exact ⟨h.1.1, h.1.2, h.2⟩

/-- attack：有剑且目标怪物相邻时，将怪物标记为 defeated。 -/
def attack (s : State) (target : Object) : State :=
  if canAttackB s target then
    { s with objects := replaceObject target target.defeatMonster s.objects }
  else
    s

def canInteractB (s : State) (target : Object) : Bool :=
  targetInFrontB s target

/-- 交互对象的具体前置条件；门需要钥匙，其余可交互对象只要求类型匹配。 -/
def InteractObjectPrecondition (s : State) (target : Object) : Prop :=
  target.isClosedChest = true ∨
  (target.isClosedDoor = true ∧ HasKey s) ∨
  (target.kind == ObjectKind.button) = true ∨
  (target.kind == ObjectKind.sword) = true

/-- 交互动作的前置条件：目标在前方一格，且对象类型与交互规则匹配。 -/
def InteractPreconditionSpec (s : State) (target : Object) : Prop :=
  targetInFrontB s target = true ∧
  InteractObjectPrecondition s target

theorem InteractChestPrecondition
    {s : State} {target : Object}
    (h : canInteractB s target && target.isClosedChest = true) :
    InteractPreconditionSpec s target := by
  simp [InteractPreconditionSpec, InteractObjectPrecondition, canInteractB] at h ⊢
  exact ⟨h.1, Or.inl h.2⟩

theorem InteractDoorPrecondition
    {s : State} {target : Object}
    (h : canInteractB s target && target.isClosedDoor && (s.inventory.keys > 0) = true) :
    InteractPreconditionSpec s target := by
  simp [InteractPreconditionSpec, InteractObjectPrecondition, canInteractB, HasKey] at h ⊢
  exact ⟨h.1.1, Or.inr (Or.inl ⟨h.1.2, h.2⟩)⟩

theorem InteractButtonPrecondition
    {s : State} {target : Object}
    (h : canInteractB s target && (target.kind == ObjectKind.button) = true) :
    InteractPreconditionSpec s target := by
  simp [InteractPreconditionSpec, InteractObjectPrecondition, canInteractB] at h ⊢
  exact ⟨h.1, Or.inr (Or.inr (Or.inl h.2))⟩

theorem InteractSwordPrecondition
    {s : State} {target : Object}
    (h : canInteractB s target && (target.kind == ObjectKind.sword) = true) :
    InteractPreconditionSpec s target := by
  simp [InteractPreconditionSpec, InteractObjectPrecondition, canInteractB] at h ⊢
  exact ⟨h.1, Or.inr (Or.inr (Or.inr h.2))⟩

/--
统一的 interact 前置条件定理：若某个具体交互分支的布尔守卫为真，
则前方存在对应可交互对象；若对象是门，则物品栏中必须有钥匙。
-/
theorem InteractPrecondition
    {s : State} {target : Object}
    (h :
      (canInteractB s target && target.isClosedChest = true) ∨
      (canInteractB s target && target.isClosedDoor && (s.inventory.keys > 0) = true) ∨
      (canInteractB s target && (target.kind == ObjectKind.button) = true) ∨
      (canInteractB s target && (target.kind == ObjectKind.sword) = true)) :
    InteractPreconditionSpec s target := by
  rcases h with hChest | hDoor | hButton | hSword
  · exact InteractChestPrecondition hChest
  · exact InteractDoorPrecondition hDoor
  · exact InteractButtonPrecondition hButton
  · exact InteractSwordPrecondition hSword

def interactChest (s : State) (target : Object) : State :=
  if canInteractB s target && target.isClosedChest then
    { collectKey s with objects := replaceObject target target.openChest s.objects }
  else
    s

def interactDoor (s : State) (target : Object) : State :=
  if canInteractB s target && target.isClosedDoor && (s.inventory.keys > 0) then
    { spendKey s with objects := replaceObject target target.openDoor s.objects }
  else
    s

def interactButton (s : State) (target : Object) : State :=
  if canInteractB s target && target.kind == ObjectKind.button then
    { s with objects := replaceObject target target.pressButton s.objects }
  else
    s

def interactSword (s : State) (target : Object) : State :=
  if canInteractB s target && target.kind == ObjectKind.sword then
    equipSword s
  else
    s

/--
interact：按对象类型进行交互。

目前覆盖 Task1/Task2/Task4 需要的宝箱、门、按钮和剑；墙、陷阱、
怪物没有直接交互效果。
-/
def interact (s : State) (target : Object) : State :=
  match target.kind with
  | ObjectKind.chest => interactChest s target
  | ObjectKind.door => interactDoor s target
  | ObjectKind.button => interactButton s target
  | ObjectKind.sword => interactSword s target
  | _ => s

end NesyLink
