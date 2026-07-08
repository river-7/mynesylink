import Base

/-!
  Day 3 对象模型。

  本文件只定义环境对象及其基础状态，用来覆盖 Task1、Task2、Task4 中
  需要描述的 Wall、Chest、Monster、Trap、Button、Sword、Door。
-/

namespace NesyLink

/-- 关卡中的对象种类。 -/
inductive ObjectKind where
  | wall
  | chest
  | monster
  | trap
  | button
  | sword
  | door
  deriving BEq, Repr

instance : ToString ObjectKind where
  toString := reprStr

/-- 门的抽象状态。 -/
inductive DoorState where
  | open
  | closed
  | locked
  deriving BEq, Repr

instance : ToString DoorState where
  toString := reprStr

/-- 宝箱的抽象状态。 -/
inductive ChestState where
  | closed
  | opened
  deriving BEq, Repr

instance : ToString ChestState where
  toString := reprStr

/-- 怪物的抽象状态。 -/
inductive MonsterState where
  | alive
  | defeated
  deriving BEq, Repr

instance : ToString MonsterState where
  toString := reprStr

/-- 按钮的抽象状态。 -/
inductive ButtonState where
  | released
  | pressed
  deriving BEq, Repr

instance : ToString ButtonState where
  toString := reprStr

/-- 单个符号对象，位置使用 NesyLink tile 坐标。 -/
structure Object where
  kind : ObjectKind
  pos : Coord
  doorState : DoorState
  chestState : ChestState
  monsterState : MonsterState
  buttonState : ButtonState
  deriving BEq, Repr

def Object.wall (p : Coord) : Object :=
  {
    kind := ObjectKind.wall,
    pos := p,
    doorState := DoorState.closed,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.chest (p : Coord) : Object :=
  {
    kind := ObjectKind.chest,
    pos := p,
    doorState := DoorState.closed,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.monster (p : Coord) : Object :=
  {
    kind := ObjectKind.monster,
    pos := p,
    doorState := DoorState.closed,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.trap (p : Coord) : Object :=
  {
    kind := ObjectKind.trap,
    pos := p,
    doorState := DoorState.closed,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.button (p : Coord) : Object :=
  {
    kind := ObjectKind.button,
    pos := p,
    doorState := DoorState.closed,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.sword (p : Coord) : Object :=
  {
    kind := ObjectKind.sword,
    pos := p,
    doorState := DoorState.closed,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.door (p : Coord) (state : DoorState) : Object :=
  {
    kind := ObjectKind.door,
    pos := p,
    doorState := state,
    chestState := ChestState.closed,
    monsterState := MonsterState.alive,
    buttonState := ButtonState.released
  }

def Object.isWall (obj : Object) : Bool :=
  obj.kind == ObjectKind.wall

def Object.isClosedChest (obj : Object) : Bool :=
  obj.kind == ObjectKind.chest && obj.chestState == ChestState.closed

def Object.isAliveMonster (obj : Object) : Bool :=
  obj.kind == ObjectKind.monster && obj.monsterState == MonsterState.alive

def Object.isTrap (obj : Object) : Bool :=
  obj.kind == ObjectKind.trap

def Object.isClosedDoor (obj : Object) : Bool :=
  obj.kind == ObjectKind.door && obj.doorState != DoorState.open

def Object.blocksMove (obj : Object) : Bool :=
  obj.isWall || obj.isClosedChest || obj.isClosedDoor

def Object.isDangerous (obj : Object) : Bool :=
  obj.isTrap || obj.isAliveMonster

def Object.openChest (obj : Object) : Object :=
  { obj with chestState := ChestState.opened }

def Object.defeatMonster (obj : Object) : Object :=
  { obj with monsterState := MonsterState.defeated }

def Object.pressButton (obj : Object) : Object :=
  { obj with buttonState := ButtonState.pressed }

def Object.openDoor (obj : Object) : Object :=
  { obj with doorState := DoorState.open }

/-- 对象到 Day1 `Cell` 抽象的对应关系。 -/
def Object.toCell (obj : Object) : Cell :=
  match obj.kind with
  | ObjectKind.wall => Cell.wall
  | ObjectKind.chest => Cell.chest
  | ObjectKind.monster => Cell.monster
  | ObjectKind.trap => Cell.trap
  | ObjectKind.button => Cell.button
  | ObjectKind.sword => Cell.empty
  | ObjectKind.door => Cell.exit

end NesyLink
