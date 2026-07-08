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
  adjacentB s.player target.pos

/-- attack：有剑且目标怪物相邻时，将怪物标记为 defeated。 -/
def attack (s : State) (target : Object) : State :=
  if canAttackB s target then
    { s with objects := replaceObject target target.defeatMonster s.objects }
  else
    s

def canInteractB (s : State) (target : Object) : Bool :=
  reachableForInteractB s.player target.pos

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
