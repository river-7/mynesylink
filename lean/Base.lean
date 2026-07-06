/-!
  Day 1 base definitions for the Zelda-like maze model.

  This file intentionally contains only static data types:
  coordinates, directions, actions, and symbolic cell categories.

  The definitions align with the public environment API, the Python grid
  observation semantics, and the symbol layer extracted in `vision.py`.
-/

namespace NesyLink

/-- A coordinate written as `(x, y)`, matching Python-side tile positions. -/
structure Coord where
  x : Nat
  y : Nat
  deriving BEq, Repr

instance : ToString Coord where
  toString := reprStr

/-- Cardinal directions used by movement and facing. -/
inductive Direction where
  | up
  | down
  | left
  | right
  deriving BEq, Repr

instance : ToString Direction where
  toString := reprStr

/--
The discrete action interface exposed by the environment.

The public API encodes actions as:
- `0`: wait
- `1`: move up
- `2`: move down
- `3`: move left
- `4`: move right
- `5`: trigger slot A
- `6`: trigger slot B
-/
inductive Action where
  | wait
  | moveUp
  | moveDown
  | moveLeft
  | moveRight
  | buttonA
  | buttonB
  deriving BEq, Repr

instance : ToString Action where
  toString := reprStr

/--
Symbolic cell categories used by the public grid observation and by the vision
layer's extracted `SymbolMap`.

These constructors intentionally model the post-perception symbolic layer
rather than the RGB recognition procedure itself.
-/
inductive Cell where
  | empty
  | wall
  | player
  | monster
  | chest
  | exit
  | trap
  | button
  | npc
  | gap
  | bridge
  | switch
  deriving BEq, Repr

instance : ToString Cell where
  toString := reprStr

end NesyLink
