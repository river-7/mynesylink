import Move

/-!
  Action mask soundness proof.

  This file formalizes the movement slice of Python `get_action_mask()`:
  for each directional action, the mask allows the action exactly when the
  destination tile is in bounds, not blocked, and not dangerous.  Non-movement
  actions are handled by the interaction/attack precondition proofs, so this
  mask does not emit them for the `MoveAllowed` theorem.
-/

namespace NesyLink

def inBoundsB (p : Coord) : Bool :=
  p.x < gridWidth && p.y < gridHeight

def blockedCellB : Cell → Bool
  | Cell.wall => true
  | Cell.chest => true
  | Cell.npc => true
  | Cell.gap => true
  | _ => false

def dangerCellB : Cell → Bool
  | Cell.trap => true
  | Cell.monster => true
  | _ => false

def safeTileB (m : SymbolMap) (p : Coord) : Bool :=
  inBoundsB p && !blockedCellB (m.cellAt p) && !dangerCellB (m.cellAt p)

/--
Lean model of the movement part of `submissions.agent.get_action_mask()`.

The Python list also contains slots for `NOOP`, attack, and secondary action.
For the `MoveAllowed` contract we emit only directional actions, because
`MoveAllowed` is intentionally a movement-safety predicate.
-/
def ActionMask (m : SymbolMap) (player : Coord) : Action → Bool
  | Action.moveUp => safeTileB m (nextCoord player Action.moveUp)
  | Action.moveDown => safeTileB m (nextCoord player Action.moveDown)
  | Action.moveLeft => safeTileB m (nextCoord player Action.moveLeft)
  | Action.moveRight => safeTileB m (nextCoord player Action.moveRight)
  | _ => false

theorem inBoundsB_eq_true_iff (p : Coord) :
    inBoundsB p = true ↔ InBounds p := by
  simp [inBoundsB, InBounds]

theorem blockedCellB_eq_false_iff (c : Cell) :
    blockedCellB c = false ↔ ¬ IsBlockedCell c := by
  cases c <;> simp [blockedCellB, IsBlockedCell]

theorem dangerCellB_eq_false_iff (c : Cell) :
    dangerCellB c = false ↔ ¬ IsDangerCell c := by
  cases c <;> simp [dangerCellB, IsDangerCell]

theorem safeTileB_eq_true_iff (m : SymbolMap) (p : Coord) :
    safeTileB m p = true ↔ SafeTile m p := by
  simp [
    safeTileB,
    SafeTile,
    Traversable,
    Blocked,
    Dangerous,
    inBoundsB_eq_true_iff,
    blockedCellB_eq_false_iff,
    dangerCellB_eq_false_iff
  ]

theorem actionMask_allowed_is_move
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : ActionMask m player a = true) :
    isMoveAction a := by
  cases a <;> simp [ActionMask, isMoveAction] at h ⊢

theorem actionMask_allowed_target_safe
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : ActionMask m player a = true) :
    SafeTile m (nextCoord player a) := by
  cases a <;> simp [ActionMask, safeTileB_eq_true_iff] at h ⊢
  · exact h
  · exact h
  · exact h
  · exact h

/--
ActionMaskSound（动作掩码可靠性）：
如果某个动作被 Lean 里的 `ActionMask` 允许，则它一定满足 `MoveAllowed`。
-/
theorem ActionMaskProof
    {m : SymbolMap} {player : Coord} {a : Action}
    (h : ActionMask m player a = true) :
    MoveAllowed m player a := by
  exact ⟨actionMask_allowed_is_move h, actionMask_allowed_target_safe h⟩

theorem ActionMaskSound_proof :
    ActionMaskSound (fun m player a => ActionMask m player a = true) := by
  intro m player a h
  exact ActionMaskProof h

end NesyLink
