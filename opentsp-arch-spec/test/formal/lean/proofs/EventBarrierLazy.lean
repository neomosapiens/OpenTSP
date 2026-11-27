import Proofs.EventBoundaries
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace EventBarrierLazyProofs
abbrev Entry := Option Nat × Option Nat

def retire (cutoff : Nat) (e : Entry) : Entry :=
  match e.2 with | some r=>if r≤cutoff then (none,none) else e | none=>e

def eligible (e : Entry) : Bool :=
  match e with | (some _,none)=>true | _=>false

def notify (n delay : Nat) (e : Entry) : Entry :=
  match e with
  | (some s,none)=>if s≤n then (some s,some (n+delay)) else e
  | _=>e

theorem retire_window (cutoff t : Nat) (h : cutoff≤t) (e : Entry) :
    in_barrier_window (retire cutoff e).1 (retire cutoff e).2 t=
      in_barrier_window e.1 e.2 t := by
  rcases e with ⟨start,release⟩
  cases release with
  | none=>rfl
  | some r=>
    by_cases hr:r≤cutoff
    · have ht:¬t<r:=by omega
      cases start <;> simp [retire,hr,in_barrier_window,ht]
    · simp [retire,hr]

theorem retire_eligible (cutoff : Nat) (e : Entry) :
    eligible (retire cutoff e)=eligible e := by
  rcases e with ⟨start,release⟩
  cases release with
  | none=>rfl
  | some r=>
    by_cases hr:r≤cutoff
    all_goals cases start <;> simp [retire,hr,eligible]

theorem notify_retire_commute (cutoff n delay : Nat) (hn : cutoff<n) (e : Entry) :
    retire cutoff (notify n delay e)=notify n delay (retire cutoff e) := by
  rcases e with ⟨s,r⟩
  cases s <;> cases r <;> simp [retire,notify]
  all_goals repeat' split
  all_goals simp_all
  all_goals omega

theorem retire_idempotent (c : Nat) (e : Entry) : retire c (retire c e)=retire c e := by
  rcases e with ⟨start,release⟩
  cases release with
  | none=>rfl
  | some r=>
    by_cases hr:r≤c <;> simp [retire,hr]

theorem retire_composes (a b : Nat) (hab : a ≤ b) (e : Entry) :
    retire b (retire a e)=retire b e := by
  rcases e with ⟨start,release⟩
  cases release with
  | none=>rfl
  | some r=>
    by_cases ha:r≤a
    all_goals by_cases hb:r ≤ b
    all_goals simp [retire,ha,hb]
    omega

#print axioms retire_window
#print axioms retire_eligible
#print axioms notify_retire_commute
#print axioms retire_idempotent
#print axioms retire_composes
end EventBarrierLazyProofs
