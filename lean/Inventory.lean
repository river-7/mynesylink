import Base

/-!
  Day 3 物品栏模型。

  CD 时间安排中要求 Inventory 覆盖 key、sword、coin。这里把钥匙和金币
  建模为计数，把剑建模为是否已经持有。
-/

namespace NesyLink

/-- 物品栏可追踪的物品种类。 -/
inductive ItemKind where
  | key
  | sword
  | coin
  deriving BEq, Repr

instance : ToString ItemKind where
  toString := reprStr

/-- Agent 可显式使用的物品栏状态。 -/
structure Inventory where
  keys : Nat
  coins : Nat
  hasSword : Bool
  hasShield : Bool
  deriving BEq, Repr

def Inventory.empty : Inventory :=
  { keys := 0, coins := 0, hasSword := false, hasShield := false }

def Inventory.addKey (inv : Inventory) : Inventory :=
  { inv with keys := inv.keys + 1 }

def Inventory.spendKey (inv : Inventory) : Inventory :=
  { inv with keys := inv.keys - 1 }

def Inventory.addCoin (amount : Nat) (inv : Inventory) : Inventory :=
  { inv with coins := inv.coins + amount }

def Inventory.equipSword (inv : Inventory) : Inventory :=
  { inv with hasSword := true }

def Inventory.equipShield (inv : Inventory) : Inventory :=
  { inv with hasShield := true }

def Inventory.hasKey (inv : Inventory) : Prop :=
  inv.keys > 0

def Inventory.hasCoin (inv : Inventory) : Prop :=
  inv.coins > 0

end NesyLink
