import Proofs.EventTimedView
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventTimedViewProofs
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace EventBarrierHistoryProofs

def BornBy (starts : Vector (Option Nat) 145) (t : Nat) : Prop :=
  ∀i (hi:i<145) a, starts[i]=some a → a≤t

theorem born_none (t : Nat) : BornBy (vectorInit none) t := by
  intro i hi a h
  simp [vectorInit] at h

theorem born_later (starts : Vector (Option Nat) 145) (c t : Nat)
    (h : BornBy starts c) (hct : c≤t) : BornBy starts t := by
  intro i hi a ha
  have hv:=h i hi a ha
  omega

@[simp] theorem get_set_bounded {α : Type} (v : Vector α 145) (k i : Nat) (hi:i<145) (hk:k<145) (x : α) :
    (vectorUpdate v k x)[i]=(if i=k then x else v[i]) := by
  change (v.setIfInBounds k x)[i]=_
  rw [Vector.getElem_setIfInBounds]
  by_cases h:i=k <;> simp_all [eq_comm]

theorem born_write (starts : Vector (Option Nat) 145) (k c t : Nat)
    (h : BornBy starts t) (hk:k<145) (hct:c≤t) :
    BornBy (vectorUpdate starts k (some c)) t := by
  intro i hi a ha
  rw [get_set_bounded starts k i hi hk (some c)] at ha
  split at ha
  · simp only [Option.some.injEq] at ha
    omega
  · exact h i hi a ha

theorem born_clear (starts : Vector (Option Nat) 145) (k t : Nat)
    (h : BornBy starts t) (hk:k<145) :
    BornBy (vectorUpdate starts k none) t := by
  intro i hi a ha
  rw [get_set_bounded starts k i hi hk none] at ha
  split at ha
  · contradiction
  · exact h i hi a ha

#print axioms born_none
#print axioms born_later
#print axioms get_set_bounded
#print axioms born_write
#print axioms born_clear
end EventBarrierHistoryProofs
