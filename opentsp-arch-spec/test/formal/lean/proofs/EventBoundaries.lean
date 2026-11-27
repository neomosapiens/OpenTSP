import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace EventBoundaryProofs

def floor (base : Nat) (ends : List Nat) (t : Nat) : Nat :=
  match ends with
  | [] => base
  | x::xs => floor (if x≤t ∧ base<x then x else base) xs t

theorem floor_ge (base t : Nat) (ends : List Nat) : base≤floor base ends t := by
  induction ends generalizing base with
  | nil => simp [floor]
  | cons x xs ih =>
    simp only [floor]
    have h := ih (if x≤t ∧ base<x then x else base)
    by_cases hc : x≤t ∧ base<x <;> simp [hc] at h ⊢ <;> omega

theorem floor_le (base t : Nat) (ends : List Nat) (ht : base≤t) :
    floor base ends t≤t := by
  induction ends generalizing base with
  | nil => simpa [floor] using ht
  | cons x xs ih =>
    simp only [floor]
    apply ih
    split <;> omega

theorem floor_member (base t : Nat) (ends : List Nat) :
    floor base ends t=base ∨ floor base ends t∈ends := by
  induction ends generalizing base with
  | nil => simp [floor]
  | cons x xs ih =>
    simp only [floor]
    have h := ih (if x≤t ∧ base<x then x else base)
    rcases h with h|h
    · rw [h]
      split <;> simp_all
    · exact Or.inr (List.mem_cons_of_mem x h)

theorem floor_captures (base t x : Nat) (ends : List Nat)
    (hx : x∈ends) (ht : x≤t) : x≤floor base ends t := by
  induction ends generalizing base with
  | nil => simp at hx
  | cons y ys ih =>
    simp only [floor]
    rcases List.mem_cons.mp hx with hxy|hrest
    · subst y
      have h := floor_ge (if x≤t ∧ base<x then x else base) t ys
      by_cases hc : x≤t ∧ base<x <;> simp [hc] at h ⊢ <;> omega
    · exact ih _ hrest

theorem floor_threshold (base t x : Nat) (ends : List Nat)
    (hb : base≤t) (hx : x∈ends) :
    (floor base ends t<x ↔ t<x) := by
  have hlo := floor_le base t ends hb
  have hcap := floor_captures base t x ends hx
  omega

theorem barrier_monotone_without_end (sync release : Option Nat) (a t : Nat)
    (hat : a≤t)
    (endpoint : ∀r, release=some r → r≤t → r≤a) :
    in_barrier_window sync release a=true →
      in_barrier_window sync release t=true := by
  cases sync with
  | none => simp [in_barrier_window]
  | some s =>
    cases release with
    | none => simp [in_barrier_window]; omega
    | some r =>
      have he := endpoint r rfl
      simp [in_barrier_window]
      omega

theorem candidate_barrier_monotone (sync release : Option Nat) (base t : Nat)
    (ends : List Nat) (hb : base≤t)
    (hend : ∀r,release=some r → r∈ends) :
    in_barrier_window sync release (floor base ends t)=true →
      in_barrier_window sync release t=true := by
  apply barrier_monotone_without_end sync release (floor base ends t) t
    (floor_le base t ends hb)
  intro r hr ht
  exact floor_captures base t r ends (hend r hr) ht

-- Earlier candidate views can only remove positive blocking flags.
def ViewLE (a b : hold_view) : Prop :=
  (a.vparked=true → b.vparked=true) ∧
  (a.vhost=true → b.vhost=true) ∧
  (a.vbusy=true → b.vbusy=true) ∧
  a.vrepeat=b.vrepeat ∧ a.vpoll=b.vpoll ∧
  (∀p : Nat, (a.vp[p]!).pbusy=true → (b.vp[p]!).pbusy=true) ∧
  (∀p : Nat, (a.vp[p]!).ptail=true → (b.vp[p]!).ptail=true) ∧
  (∀p : Nat, (a.vp[p]!).pstate=(b.vp[p]!).pstate)

theorem view_paths_monotone (a b : hold_view) (h : ViewLE a b) :
    view_paths_busy a=true → view_paths_busy b=true := by
  rcases h with ⟨_,_,_,_,_,hp,ht,hs⟩
  have hp0 := hp 0; have hp1 := hp 1
  have ht0 := ht 0; have ht1 := ht 1
  have hs0 := hs 0; have hs1 := hs 1
  simp only [view_paths_busy, Bool.or_eq_true]
  grind

theorem hold_decide_monotone (a b : hold_view) (h : ViewLE a b)
    (own whole cfg : Bool) (path : Option Nat) :
    hold_decide a own whole path cfg=true → hold_decide b own whole path cfg=true := by
  have hv := view_paths_monotone a b h
  rcases h with ⟨hpark,hhost,hbusy,hrep,hpoll,hp,ht,hs⟩
  cases path with
  | none =>
    simp only [hold_decide, Bool.or_eq_true, Bool.and_eq_true] at *
    grind
  | some p =>
    have hpb := hp p; have hpt := ht p; have hps := hs p
    simp only [hold_decide, Bool.or_eq_true, Bool.and_eq_true] at *
    grind

#print axioms view_paths_monotone
#print axioms hold_decide_monotone

#print axioms floor_ge
#print axioms floor_le
#print axioms floor_member
#print axioms floor_captures
#print axioms floor_threshold
#print axioms barrier_monotone_without_end
#print axioms candidate_barrier_monotone
end EventBoundaryProofs
