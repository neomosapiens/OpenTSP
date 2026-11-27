-- Production IQ preservation, with the external ring's actual admitted argument domain.
import Proofs.IqPush
import Proofs.IqPop
set_option autoImplicit false
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs

namespace Proofs.Frontend

/-- The architectural ring invariant includes every data queue and control queue144. -/
def IqBounds (s : St) : Prop :=
  ∃ H L : Vector Nat 145,
    s.regs.get? Register.iq_head = some H ∧ s.regs.get? Register.iq_len = some L ∧
    ∀ q : Nat, q < 145 → H[q]! < iq_cap ∧ L[q]! ≤ iq_cap

private theorem update_other {α : Type} [Inhabited α] {n : Nat}
    (V : Vector α n) (q r : Nat) (x : α) (_hq : q < n) (hr : r < n) (hne : r ≠ q) :
    (vectorUpdate V q x)[r]! = V[r]! := by
  simp [vectorUpdate, Vector.set!, hr, Array.getElem_setIfInBounds_ne, Ne.symm hne]

/-- The actual push either refuses with its entire state unchanged, or invokes unit/tag
    stores only at an admitted queue+slot and preserves all145 rings' bounds. -/
theorem iq_push_preserves_bounds (q : Nat) (hq : q < 145) (u : BitVec 32) (tag : Int)
    (s : St) (hs : IqBounds s)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv recU recT) :
    ∃ b after, (iq_push_tagged q u tag).run s = EStateM.Result.ok b after ∧ IqBounds after := by
  obtain ⟨H,L,hH,hL,hbounds⟩ := hs
  by_cases full : iq_cap ≤ L[q]!
  · exact ⟨false, s, iq_push_full q u tag s L hL full, H,L,hH,hL,hbounds⟩
  · have room : L[q]! < iq_cap := by omega
    obtain ⟨after,hr,regs,hu,ht⟩ := iq_push_room q u tag s recU recT env H L hH hL hq (hbounds q hq).1 room
    refine ⟨true,after,hr,H,vectorUpdate L q (L[q]!+1),?_,?_,?_⟩
    · rw [regs]; simp [Std.ExtDHashMap.get?_insert, hH]
    · rw [regs]; simp
    · intro r hr
      constructor
      · exact (hbounds r hr).1
      · by_cases same : r=q
        · subst r
          rw [vectorUpdate_self L q (L[q]!+1) hq]
          omega
        · rw [update_other L q r (L[q]!+1) hq hr same]
          exact (hbounds r hr).2

/-- The actual pop preserves the ring invariant whenever it removes only existing units.
    The assertion is about the real generated transition, not a length-only surrogate. -/
theorem iq_pop_preserves_bounds (q n : Nat) (hq : q < 145) (s : St) (hs : IqBounds s)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L) (hn : n ≤ L[q]!) :
    ∃ after, (iq_pop q n).run s = EStateM.Result.ok () after ∧ IqBounds after := by
  obtain ⟨H0,L0,hH0,hL0,hbounds⟩ := hs
  have eqH : H0=H := by simpa [hH] using hH0.symm
  have eqL : L0=L := by simpa [hL] using hL0.symm
  subst H0; subst L0
  refine ⟨_,iq_pop_run q n s H L hH hL,
    vectorUpdate H q (ring (H[q]!+n)),vectorUpdate L q (L[q]!-n),?_,?_,?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · intro r hr
    by_cases same : r=q
    · subst r
      rw [vectorUpdate_self H q _ hq,vectorUpdate_self L q _ hq]
      constructor
      · rw [ring_spec]
        have hhead := (hbounds q hq).1
        have hlen := (hbounds q hq).2
        split <;> omega
      · have hlen := (hbounds q hq).2
        omega
    · rw [update_other H q r _ hq hr same,update_other L q r _ hq hr same]
      exact hbounds r hr

end Proofs.Frontend

namespace Proofs.Frontend

/-- Partial-effect safety includes error states; it does not assert callback termination. -/
def Preserves {α : Type} (action : SailM α) : Prop :=
  ∀ s : St, IqBounds s → match action.run s with
    | .ok _ after => IqBounds after
    | .error _ after => IqBounds after

theorem preserves_pure {α : Type} (a : α) : Preserves (pure a) := by
  intro s hs
  exact hs

theorem preserves_ite {α : Type} (p : Prop) [Decidable p] (a b : SailM α)
    (ha : Preserves a) (hb : Preserves b) : Preserves (if p then a else b) := by
  split <;> assumption

theorem preserves_bind {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : Preserves a) (hf : ∀ x, Preserves (f x)) : Preserves (a >>= f) := by
  intro s hs
  have h := ha s hs
  cases he : a.run s with
  | ok x after =>
    have hp : IqBounds after := by simpa [he] using h
    have hf' := hf x after hp
    simpa [EStateM.run, bind, EStateM.bind, show a s = .ok x after from he] using hf'
  | error error after =>
    simpa [EStateM.run, bind, EStateM.bind, show a s = .error error after from he] using h

theorem preserves_read (r : Register) : Preserves (Out.readReg r) := by
  intro s hs
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind, get, getThe,
    MonadStateOf.get, EStateM.get]
  cases s.regs.get? r <;> exact hs

theorem preserves_other_write (r : Register) (v : RegisterType r)
    (hh : r ≠ Register.iq_head) (hl : r ≠ Register.iq_len) : Preserves (Out.writeReg r v) := by
  intro s hs
  obtain ⟨H,L,hH,hL,hbounds⟩ := hs
  simp [Out.writeReg, PreSail.writeReg, EStateM.run, modify, modifyGet,
    MonadStateOf.modifyGet, EStateM.modifyGet]
  exact ⟨H,L,by simpa [Std.ExtDHashMap.get?_insert, hh, Ne.symm hh] using hH,
    by simpa [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl] using hL,hbounds⟩

theorem preserves_forIn {β : Type} (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, Preserves (f i hi value)) : Preserves (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => Preserves (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply preserves_bind _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact preserves_pure value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact preserves_pure value

theorem preserves_push (q : Nat) (hq : q < 145) (u : BitVec 32) (tag : Int)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv recU recT) : Preserves (iq_push_tagged q u tag) := by
  intro s hs
  obtain ⟨b,after,h,hi⟩ := iq_push_preserves_bounds q hq u tag s hs recU recT env
  simpa [h] using hi

/-- Updating text-word integrity metadata preserves every queue's ring bounds. -/
theorem overlay_preserves_bounds (q w : Nat) (e : BitVec 10) : Preserves (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn, ForIn'.forIn', instForInOfForIn']
  repeat' first
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

/-- The complete production delivery function preserves all queue bounds, including
    padding, not-ready and empty payloads, overflow/refusal, tag writes and overlay updates.
    The invariant holds also if a model read errors; this is not a termination theorem. -/
theorem iq_deliver_preserves_bounds (q c : Nat) (hq : q < 145)
    (recU : St → Option (Nat × Nat × BitVec 32)) (recT : St → Option (Nat × Nat × Int))
    (env : IqEnv recU recT) : Preserves (iq_deliver q c) := by
  unfold iq_deliver
  dsimp only [ForIn.forIn, ForIn'.forIn', instForInOfForIn']
  repeat' first
    | exact preserves_push q hq _ _ recU recT env
    | exact overlay_preserves_bounds _ _ _
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

end Proofs.Frontend
