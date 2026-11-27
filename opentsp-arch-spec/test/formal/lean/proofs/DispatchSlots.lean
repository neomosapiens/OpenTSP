-- Producer invariant of the actual classifier, required by the bank grant proof.
import Proofs.Common
set_option autoImplicit false
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs

namespace Proofs.DispatchSlots

def HasThread (t : BitVec 1) (i : Option instruction) : Prop :=
  match i with | none => True | some insn => ref_thread insn = t

def SlotThreads (r : cycle_recs) : Prop :=
  HasThread 0#1 r.refs[0]! ∧ HasThread 1#1 r.refs[1]! ∧
  HasThread 0#1 r.refs[2]! ∧ HasThread 1#1 r.refs[3]!

theorem ref_path_range (i : instruction) (p : Nat) (h : ref_path i = some p) : p=0 ∨ p=1 := by
  cases i <;> simp [ref_path] at h
  all_goals first | simp_all | split at h <;> simp_all

theorem place_preserves_threads (r : cycle_recs) (i : instruction) (h : SlotThreads r) :
    SlotThreads (place r i).1 := by
  cases hc : agt_config i with
  | some config =>
    simp only [place,hc]
    split <;> simpa [SlotThreads] using h
  | none =>
    cases hp : ref_path i with
    | none =>
      cases i <;> simp only [place,hc,hp]
      all_goals first
        | simpa [SlotThreads] using h
        | split <;> simpa [SlotThreads] using h
    | some p =>
      have paths := ref_path_range i p hp
      have small : (ref_thread i).toNat < 2 := by simpa using (ref_thread i).isLt
      have threads : ref_thread i = 0#1 ∨ ref_thread i = 1#1 := by
        have digits : (ref_thread i).toNat = 0 ∨ (ref_thread i).toNat = 1 := by omega
        rcases digits with h0|h1
        · left; apply BitVec.eq_of_toNat_eq; simpa using h0
        · right; apply BitVec.eq_of_toNat_eq; simpa using h1
      rcases paths with h0|h1 <;> rcases threads with t0|t1 <;> subst p
      all_goals simp_all [place,bit_index]
      all_goals split
      all_goals simp_all [SlotThreads,HasThread,vectorUpdate,Vector.set!]

theorem place_all_preserves_threads (r : cycle_recs) (xs : List (Result instruction diag)) (ok : Bool)
    (h : SlotThreads r) : SlotThreads (place_all r xs ok).1 := by
  induction xs generalizing r ok with
  | nil => simpa [place_all] using h
  | cons item rest ih =>
    cases item with
    | Err d => simpa [place_all] using ih r false h
    | Ok i =>
      have hp := place_preserves_threads r i h
      simpa [place_all] using ih (place r i).1 (ok && (place r i).2) hp

theorem no_recs_threads (n : Nat) : SlotThreads {no_recs with n := n} := by
  simp [SlotThreads,HasThread,no_recs,vectorInit]

/-- The production classifier establishes the exact input invariant required by bank_rule,
    for arbitrary parcels (including invalid instructions and slot collisions), without a
    list-length bound or a surrogate parser. -/
theorem classify_items_threads (q : Nat) (items : List (Result instruction diag)) :
    SlotThreads (classify_items q items) := by
  cases items with
  | nil => simp [classify_items,SlotThreads,HasThread,no_recs,vectorInit]
  | cons item rest =>
    cases rest with
    | nil =>
      cases item with
      | Err d => simp [classify_items,SlotThreads,HasThread,no_recs,vectorInit]
      | Ok i =>
        simp only [classify_items]
        split
        · exact no_recs_threads 0
        · have hp := place_preserves_threads {no_recs with n := items_len [.Ok i]} i (no_recs_threads 0)
          dsimp only at *
          split
          · exact hp
          · exact no_recs_threads 0
    | cons second rest =>
      simp only [classify_items]
      split
      · exact no_recs_threads 0
      · have hp := place_all_preserves_threads {no_recs with n := items_len (item::second::rest)}
            (item::second::rest) true (no_recs_threads 0)
        simpa only [SlotThreads] using hp

end Proofs.DispatchSlots
