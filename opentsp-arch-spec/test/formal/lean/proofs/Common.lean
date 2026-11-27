-- Shared facts about the generated model at the exported profile. `iq_cap` is an instantiated
-- constant (mini and ref both give 320 units); avoiding its numeral in a proof does not prove
-- transfer to every admitted custom capacity. That transfer is a separate obligation.
import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions

namespace Proofs

/-- `ring` folds an index below twice the capacity back into the ring. -/
theorem ring_spec (x : Nat) : ring x = if x ≥ iq_cap then x - iq_cap else x := by
  unfold ring
  by_cases h : x ≥ iq_cap
  · simp [h]
  · simp [h]

theorem vectorUpdate_self (V : Vector Nat 145) (q x : Nat) (hq : q < 145) : (vectorUpdate V q x)[q]! = x := by
  simp [vectorUpdate, hq, Vector.set!]

end Proofs
