import Base

/-!
  Day 2 move-safety layer.

  This file formalizes the tile-level safety contract used by the Python
  planner/action mask.  It intentionally models the symbolic map obtained after
  vision, not the RGB detector itself.
-/

namespace NesyLink

/-- Public NesyLink task rooms are 10 columns wide. -/
def gridWidth : Nat := 10

/-- Public NesyLink task rooms are 8 rows high. -/
def gridHeight : Nat := 8

/-- A symbolic map is enough for movement safety: each coordinate has a cell. -/
structure SymbolMap where
  cellAt : Coord → Cell

/-- Sprint Day 2 name for the symbolic room map abstraction. -/
abbrev Map := SymbolMap

/-- Tile coordinates inside the playable room. -/
def InBounds (p : Coord) : Prop :=
  p.x < gridWidth ∧ p.y < gridHeight

/--
Tiles that block movement in the Python symbolic layer:
walls, closed/visible chests, NPCs, and dynamic gaps.
-/
def IsBlockedCell : Cell → Prop
  | Cell.wall => True
  | Cell.chest => True
  | Cell.npc => True
  | Cell.gap => True
  | _ => False

/-- Hazardous but not necessarily blocking tiles. -/
def IsDangerCell : Cell → Prop
  | Cell.trap => True
  | Cell.monster => True
  | _ => False

def Blocked (m : SymbolMap) (p : Coord) : Prop :=
  IsBlockedCell (m.cellAt p)

def Dangerous (m : SymbolMap) (p : Coord) : Prop :=
  IsDangerCell (m.cellAt p)

/-- Movement-only safety: in bounds and not physically blocked. -/
def Traversable (m : SymbolMap) (p : Coord) : Prop :=
  InBounds p ∧ ¬ Blocked m p

/-- Planner safety: traversable and not a known hazard. -/
def SafeTile (m : SymbolMap) (p : Coord) : Prop :=
  Traversable m p ∧ ¬ Dangerous m p

/-- The safety invariant tracked for the player position. -/
def SafeState (m : SymbolMap) (player : Coord) : Prop :=
  SafeTile m player

def isMoveAction : Action → Prop
  | Action.moveUp => True
  | Action.moveDown => True
  | Action.moveLeft => True
  | Action.moveRight => True
  | _ => False

/-- Tile-level moves, separated from non-movement button actions. -/
inductive Move where
  | up
  | down
  | left
  | right
  deriving BEq, Repr

instance : ToString Move where
  toString := reprStr

/-- Embed a tile move into the public environment action type. -/
def Move.toAction : Move → Action
  | Move.up => Action.moveUp
  | Move.down => Action.moveDown
  | Move.left => Action.moveLeft
  | Move.right => Action.moveRight

/-- Candidate coordinate after one tile move. Boundary underflow stays put. -/
def nextCoord (p : Coord) : Action → Coord
  | Action.moveUp => { p with y := p.y - 1 }
  | Action.moveDown => { p with y := p.y + 1 }
  | Action.moveLeft => { p with x := p.x - 1 }
  | Action.moveRight => { p with x := p.x + 1 }
  | _ => p

/-- Candidate coordinate for the movement-only API. -/
def nextCoordByMove (p : Coord) : Move → Coord
  | Move.up => { p with y := p.y - 1 }
  | Move.down => { p with y := p.y + 1 }
  | Move.left => { p with x := p.x - 1 }
  | Move.right => { p with x := p.x + 1 }

/-- A movement command is allowed exactly when its target tile is safe. -/
def MoveAllowedByMove (m : Map) (player : Coord) (mv : Move) : Prop :=
  SafeTile m (nextCoordByMove player mv)

/-- The action-mask contract: only safe movement actions are allowed. -/
def MoveAllowed (m : SymbolMap) (player : Coord) (a : Action) : Prop :=
  isMoveAction a ∧ SafeTile m (nextCoord player a)

/--
Minimal deterministic next-position semantics for Day 2:
allowed movement goes to its target; every other action leaves the player in
place.  Later files can refine this into a full game state transition.
-/
noncomputable def nextPlayer (m : SymbolMap) (player : Coord) (a : Action) : Coord := by
  classical
  exact if _h : MoveAllowed m player a then nextCoord player a else player

/-- Sprint Day 2 `next_state` interface for the tile-level player position. -/
noncomputable def next_state (m : Map) (player : Coord) (a : Action) : Coord :=
  nextPlayer m player a

theorem nextCoordByMove_eq_nextCoord
    (player : Coord) (mv : Move) :
    nextCoordByMove player mv = nextCoord player mv.toAction := by
  cases mv <;> rfl

theorem moveAllowedByMove_toAction
    {m : Map} {player : Coord} {mv : Move}
    (h : MoveAllowedByMove m player mv) :
    MoveAllowed m player mv.toAction := by
  constructor
  · cases mv <;> simp [Move.toAction, isMoveAction]
  · unfold MoveAllowedByMove at h
    rw [← nextCoordByMove_eq_nextCoord]
    exact h

theorem safeTile_inBounds
    {m : SymbolMap} {p : Coord}
    (h : SafeTile m p) :
    InBounds p := by
  exact h.1.1

theorem safeTile_not_blocked
    {m : SymbolMap} {p : Coord}
    (h : SafeTile m p) :
    ¬ Blocked m p := by
  exact h.1.2

theorem safeTile_not_dangerous
    {m : SymbolMap} {p : Coord}
    (h : SafeTile m p) :
    ¬ Dangerous m p := by
  exact h.2

theorem moveAllowed_is_move
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    isMoveAction a := by
  exact h.1

theorem moveAllowed_target_safe
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    SafeTile m (nextCoord player a) := by
  exact h.2

theorem moveAllowed_target_inBounds
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    InBounds (nextCoord player a) := by
  exact safeTile_inBounds h.2

theorem moveAllowed_target_not_blocked
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    ¬ Blocked m (nextCoord player a) := by
  exact safeTile_not_blocked h.2

theorem moveAllowed_target_not_dangerous
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    ¬ Dangerous m (nextCoord player a) := by
  exact safeTile_not_dangerous h.2

theorem nextPlayer_eq_nextCoord_of_allowed
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    nextPlayer m player a = nextCoord player a := by
  unfold nextPlayer
  simp [h]

theorem nextPlayer_eq_self_of_not_allowed
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : ¬ MoveAllowed m player a) :
    nextPlayer m player a = player := by
  unfold nextPlayer
  simp [h]

theorem next_state_eq_nextCoord_of_allowed
    {m : Map} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    next_state m player a = nextCoord player a := by
  unfold next_state
  exact nextPlayer_eq_nextCoord_of_allowed h

theorem next_state_eq_self_of_not_allowed
    {m : Map} {player : Coord} {a : Action}
    (h : ¬ MoveAllowed m player a) :
    next_state m player a = player := by
  unfold next_state
  exact nextPlayer_eq_self_of_not_allowed h

theorem allowed_move_preserves_safe_state
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    SafeState m (nextPlayer m player a) := by
  rw [nextPlayer_eq_nextCoord_of_allowed h]
  exact moveAllowed_target_safe h

theorem allowed_move_preserves_inBounds
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    InBounds (nextPlayer m player a) := by
  exact safeTile_inBounds (allowed_move_preserves_safe_state h)

theorem allowed_move_does_not_enter_blocked
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    ¬ Blocked m (nextPlayer m player a) := by
  exact safeTile_not_blocked (allowed_move_preserves_safe_state h)

theorem allowed_move_does_not_enter_danger
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    ¬ Dangerous m (nextPlayer m player a) := by
  exact safeTile_not_dangerous (allowed_move_preserves_safe_state h)

theorem allowed_move_preserves_next_state
    {m : Map} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    SafeState m (next_state m player a) := by
  unfold next_state
  exact allowed_move_preserves_safe_state h

theorem allowed_move_next_state_inBounds
    {m : Map} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    InBounds (next_state m player a) := by
  exact safeTile_inBounds (allowed_move_preserves_next_state h)

theorem allowed_move_next_state_not_blocked
    {m : Map} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    ¬ Blocked m (next_state m player a) := by
  exact safeTile_not_blocked (allowed_move_preserves_next_state h)

theorem allowed_move_next_state_not_dangerous
    {m : Map} {player : Coord} {a : Action}
    (h : MoveAllowed m player a) :
    ¬ Dangerous m (next_state m player a) := by
  exact safeTile_not_dangerous (allowed_move_preserves_next_state h)

/--
Soundness statement for a Python-style action mask.  If every action emitted by
the mask satisfies `MoveAllowed`, then every emitted action preserves safety.
-/
def ActionMaskSound (mask : SymbolMap → Coord → Action → Prop) : Prop :=
  ∀ {m player a}, mask m player a → MoveAllowed m player a

theorem actionMask_emits_safe_next_state
    {mask : SymbolMap → Coord → Action → Prop}
    (hsound : ActionMaskSound mask)
    {m : SymbolMap} {player : Coord} {a : Action}
    (hmask : mask m player a) :
    SafeState m (nextPlayer m player a) := by
  exact allowed_move_preserves_safe_state (hsound hmask)

end NesyLink
