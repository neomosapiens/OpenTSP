-- Production next-event arithmetic; complete skip/trace equivalence is not yet established.
import Proofs.Common
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
set_option autoImplicit false
namespace EventProofs

def Future (c : Nat) (e : Int) : Prop := e < 0 ∨ (c : Int) < e

theorem earliest_select (a b : Int) : earliest a b = a ∨ earliest a b = b := by
  unfold earliest
  split
  · exact Or.inr rfl
  · split
    · exact Or.inl rfl
    · split
      · exact Or.inl rfl
      · exact Or.inr rfl

theorem earliest_future (c : Nat) (a b : Int)
    (ha : Future c a) (hb : Future c b) : Future c (earliest a b) := by
  rcases earliest_select a b with h | h
  · simpa [h] using ha
  · simpa [h] using hb

theorem earliest_nonnegative_min (a b : Int) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    earliest a b = min a b := by
  simp [earliest, Int.min_def]
  omega

abbrev St := SequentialState RegisterType trivialChoiceSource

def Post {α : Type} (P : α → Prop) (action : SailM α) : Prop :=
  ∀ st a st', action.run st = EStateM.Result.ok a st' → P a

theorem post_any {α : Type} (action : SailM α) : Post (fun _ => True) action := by
  intro _ _ _ _
  trivial

theorem post_pure {α : Type} (P : α → Prop) (a : α) (ha : P a) : Post P (pure a) := by
  intro st b st' h
  simp [EStateM.run, pure, EStateM.pure] at h
  rcases h with ⟨rfl, rfl⟩
  exact ha

theorem post_bind {α β : Type} (P : α → Prop) (Q : β → Prop)
    (action : SailM α) (next : α → SailM β)
    (ha : Post P action) (hn : ∀ a, P a → Post Q (next a)) :
    Post Q (action >>= next) := by
  intro st b st' h
  simp only [EStateM.run, bind, EStateM.bind] at h
  cases hs : action st with
  | error error state => simp [hs] at h
  | ok a state =>
    simp only [hs] at h
    exact hn a (ha st a state hs) state b st' h

theorem post_bind_any {α β : Type} (P : β → Prop) (action : SailM α) (next : α → SailM β)
    (hn : ∀ a, Post P (next a)) : Post P (action >>= next) := by
  exact post_bind (fun _ => True) P action next (post_any action) (fun a _ => hn a)

theorem post_forIn {β : Type} (range : IntRange) (P : β → Prop) (b : β)
    (f : Int → β → SailM (ForInStep β)) (hb : P b)
    (hf : ∀ i x, P x → Post (fun step => match step with | .done y => P y | .yield y => P y) (f i x)) :
    Post P (forIn range b f) := by
  change Post P (IntRange.forIn' range b (fun i _ x => f i x))
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun x i hs => P x → Post P (IntRange.forIn'.loop range (fun i _ x => f i x) x i hs))
  · intro x i hs hi ih hx
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply post_bind _ P _ _ (hf i x hx)
    intro step hstep
    cases step with
    | done y => exact post_pure P y hstep
    | yield y => exact ih y hstep
  · intro x i hs hi hx
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact post_pure P x hx
  · exact hb

#print axioms post_any
#print axioms post_pure
#print axioms post_bind
#print axioms post_bind_any
#print axioms post_forIn

attribute [irreducible] Post

def leave_window (park release : Option Nat) (t : Nat) : Nat :=
  match park, release with
  | some a, some b => if a ≤ t ∧ t < b then b else t
  | _, _ => t

theorem leave_window_mono (park release : Option Nat) (t : Nat) :
    t ≤ leave_window park release t := by
  cases park <;> cases release <;> simp [leave_window]
  split <;> omega

theorem unparked_from_mono (q t : Nat) :
    Post (fun u => t ≤ u) (unparked_from q t) := by
  simp only [unparked_from, bind_pure, pure_bind]
  apply post_forIn [0:1:1]i (fun u => t ≤ u) t
  · exact Nat.le_refl t
  · intro pass u hu
    apply post_bind (fun v => t ≤ v) _ _ _
    · apply post_forIn [0:1:1]i (fun v => t ≤ v) u
      · exact hu
      · intro j v hv
        apply post_bind_any
        intro (parks : Vector (Option Nat) 145)
        apply post_bind_any
        intro (releases : Vector (Option Nat) 145)
        cases hp : parks[if j == 0 then q else 144]! <;>
          cases hr : releases[if j == 0 then q else 144]! <;>
          simp only [hp, hr]
        all_goals first | exact post_pure _ _ hv | skip
        all_goals
          apply post_bind (fun x : Nat => t ≤ x) _ _ _
          · split
            · apply post_pure
              simp_all
              omega
            · exact post_pure _ _ hv
          · intro x hx
            exact post_pure _ _ hx
    · intro v hv
      exact post_pure _ _ hv

def FromBase (base : Nat) (event : Int) : Prop := event < 0 ∨ (base : Int) ≤ event

theorem earliest_helpers (a b : Int) : earliest_int a b = earliest a b := by rfl

theorem earliest_base (base : Nat) (a b : Int)
    (ha : FromBase base a) (hb : FromBase base b) : FromBase base (earliest_int a b) := by
  rw [earliest_helpers]
  rcases earliest_select a b with h | h
  · simpa [h] using ha
  · simpa [h] using hb

theorem earliest_free_lower (q base : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) (candidates : List Nat)
    (best : Int) (hbest : FromBase base best) :
    Post (FromBase base) (earliest_free q base r items candidates best) := by
  induction candidates generalizing best with
  | nil => exact post_pure _ _ hbest
  | cons time rest ih =>
    simp only [earliest_free]
    apply post_bind (FromBase base) _ _ _
    · apply post_bind_any
      intro held
      split
      · apply post_pure
        apply earliest_base base best time hbest
        right
        simp_all [FromBase]
      · exact post_pure _ _ hbest
    · intro next hn
      exact ih next hn

theorem parcel_dispatchable_lower (q base : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) :
    Post (FromBase base) (parcel_dispatchable q base r items) := by
  unfold parcel_dispatchable
  apply post_bind_any
  intro candidates
  apply post_bind_any
  intro candidates
  apply post_bind_any
  intro candidates
  apply post_bind_any
  intro candidates
  exact earliest_free_lower q base r items candidates (-1) (by left; omega)

theorem post_imp {α : Type} (P Q : α → Prop) (action : SailM α)
    (hp : Post P action) (h : ∀ a, P a → Q a) : Post Q action := by
  unfold Post at *
  intro st a st' ha
  exact h a (hp st a st' ha)

theorem parcel_event_future (q c : Nat) :
    Post (Future c) (parcel_event q c) := by
  simp only [parcel_event, bind_assoc, pure_bind]
  apply post_bind_any
  intro lengths
  split
  · apply post_pure; left; omega
  · apply post_bind_any
    intro p
    split
    · apply post_pure; left; omega
    · apply post_bind_any
      intro early
      apply post_imp _ _ _ (parcel_dispatchable_lower q _ _ _)
      intro event he
      rcases he with h | h
      · exact Or.inl h
      · right
        have hh := Nat.le_max_left (c + 1) early
        omega

theorem control_event_future (c : Nat) : Post (Future c) (control_event c) := by
  simp only [control_event, bind_assoc, pure_bind]
  apply post_bind_any
  intro lengths
  split
  · apply post_pure; left; omega
  · apply post_bind_any
    intro p
    split
    · apply post_bind_any
      intro early
      apply post_pure
      right
      have hh := Nat.le_max_left (c + 1) early
      omega
    · apply post_pure; left; omega

#print axioms parcel_dispatchable_lower
#print axioms post_imp
#print axioms parcel_event_future
#print axioms control_event_future
#print axioms earliest_helpers
#print axioms earliest_base
#print axioms earliest_free_lower
#print axioms unparked_from_mono
#print axioms leave_window_mono
#print axioms earliest_select
#print axioms earliest_future
#print axioms earliest_nonnegative_min

theorem future_succ (c : Nat) : Future c (c + 1) := by right; omega

theorem future_max (c n : Nat) : Future c (max (c + 1) n) := by
  right
  have hh := Nat.le_max_left (c + 1) n
  omega

theorem future_ring (c : Nat) (k : Int) (size : Int) (hsize : 0 < size) :
    Future c ((c : Int) + 1 + (k - ((c : Int) + 1)) % size) := by
  right
  have hh := Int.emod_nonneg (k - ((c : Int) + 1)) (by omega : size ≠ 0)
  omega

theorem future_max_int (c : Nat) (n : Int) : Future c (max ((c : Int)+1) n) := by
  right
  have hh := Int.le_max_left ((c : Int)+1) n
  omega

theorem model_next_event_future (c : Nat) : Post (Future c) (model_next_event c) := by
  simp only [model_next_event, bind_assoc, pure_bind, bind_pure]
  apply post_bind (Future c)
  · apply post_forIn [0:143:1]i (Future c) (-1) _ (by left; omega)
    intro k best hb
    apply post_bind_any
    intro values
    split
    · apply post_pure
      exact earliest_future c _ _ hb (future_max_int c _)
    · exact post_pure _ _ hb
  · intro b1 h1
    apply post_bind (Future c)
    · apply post_forIn [0:87:1]i (Future c) b1 _ h1
      intro k best hb
      apply post_bind_any
      intro values
      split
      · apply post_pure
        exact earliest_future c _ _ hb (future_max_int c _)
      · exact post_pure _ _ hb
    · intro b2 h2
      apply post_bind (Future c)
      · apply post_forIn [0:127:1]i (Future c) b2 _ h2
        intro k best hb
        apply post_bind_any
        intro values
        split
        · exact post_pure _ _ hb
        · apply post_pure
          exact earliest_future c _ _ hb (future_ring c k 128 (by decide))
      · intro b3 h3
        apply post_bind (Future c)
        · apply post_forIn [0:15:1]i (Future c) b3 _ h3
          intro k best hb
          apply post_bind_any
          intro values
          split
          · exact post_pure _ _ hb
          · apply post_pure
            exact earliest_future c _ _ hb (future_ring c k 16 (by decide))
        · intro b4 h4
          apply post_bind (Future c)
          · apply post_forIn [0:144:1]i (Future c) b4 _ h4
            intro k best hb
            apply post_bind_any
            intro values
            split
            · apply post_pure
              exact earliest_future c _ _ hb (future_max_int c _)
            · exact post_pure _ _ hb
          · intro b5 h5
            apply post_bind (Future c)
            · exact control_event_future c
            · intro b6 h6
              apply post_forIn [0:143:1]i (Future c) (earliest b5 b6) _ (earliest_future c _ _ h5 h6)
              intro q best hb
              apply post_bind (Future c)
              · apply post_forIn [0:1:1]i (Future c) best _ hb
                intro p best hb
                apply post_bind_any
                intro a
                split
                · apply post_pure
                  exact earliest_future c _ _ hb (future_succ c)
                · apply post_bind_any
                  intro held
                  apply post_bind (Future c)
                  · split
                    · apply post_bind (fun u => max a.next (c+1) ≤ u) _ _ _ (unparked_from_mono q.toNat _)
                      intro u hu
                      apply post_pure
                      apply earliest_future c _ _ hb
                      right
                      have hh := Nat.le_max_right a.next (c+1)
                      omega
                    · exact post_pure _ _ hb
                  · intro x hx
                    exact post_pure _ _ hx
              · intro after_agt h_agt
                apply post_bind_any
                intro repeats
                apply post_bind (Future c)
                · split
                  · apply post_bind_any
                    intro held
                    split
                    · apply post_bind (fun u => max _ (c+1) ≤ u) _ _ _ (unparked_from_mono q.toNat _)
                      intro u hu
                      apply post_pure
                      apply earliest_future c _ _ h_agt
                      right
                      omega
                    · exact post_pure _ _ h_agt
                  · exact post_pure _ _ h_agt
                · intro after_repeat h_repeat
                  apply post_bind (Future c) _ _ _ (parcel_event_future q.toNat c)
                  intro ev he
                  apply post_pure
                  exact earliest_future c _ _ h_repeat he

#print axioms future_max_int
#print axioms future_succ
#print axioms future_max
#print axioms future_ring
#print axioms model_next_event_future
end EventProofs
