import Proofs.Initialized
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep
namespace Proofs.StorageProduct
/-- Product relation with a concrete external-store witness. This is preserved by whole
    actions, unlike initialized Bounds alone. The chosen WritePorts interpretation and
    successful allocator/GMP boundary remain explicit premises. -/
def Good (cap : Nat) (s : St) : Prop :=
  ∃ c : CView, Represents cap c s ∧ Proofs.Initialized.Bounds cap s

theorem push_rep (cap q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts cap) (s : St) (bounds : Proofs.Initialized.Bounds cap s)
    (c : CView) (rep : Represents cap c s) :
    ∃ b after c', (pushC cap q u tag).run s = .ok b after ∧ Represents cap c' after := by
  obtain ⟨H,L,hH,hL,hb⟩ := bounds
  by_cases full : cap≤L[q]!
  · exact ⟨false,s,c,iq_push_full q u tag s L hL full,rep⟩
  · have room : L[q]!<cap := by omega
    let slot := ringC cap (H[q]!+L[q]!)
    have hslot : slot<cap := by
      dsimp [slot]
      rw [Proofs.Capacity.ring_spec]
      have hh := (hb q hq).1
      split <;> omega
    let t1 := writeCell s (145*cap) 0 (q*cap+slot) u.toNat
    let t2 := writeCell t1 (145*cap) 1 (q*cap+slot) (zigzag tag)
    let after : St := {t2 with regs := t2.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1))}
    have e1 : iq_unit_write q slot u s = .ok () t1 := ports.unit q slot u s hq hslot
    have e2 : iq_tag_write q slot tag t1 = .ok () t2 := ports.tag q slot tag t1 hq hslot
    have r1 : t1.regs=s.regs := rfl
    have r2 : t2.regs=s.regs := rfl
    have hc : ((H[q]! : Int)+(L[q]! : Int)).toNat = H[q]!+L[q]! := by omega
    refine ⟨true,after,cTagWrite (cUnitWrite c q slot u) q slot tag,?_,?_⟩
    · simp [pushC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,hL,hH,
        EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,
        EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,full,hc,slot,e1,e2,r1,r2,after]
    · change Represents cap (cTagWrite (cUnitWrite c q slot u) q slot tag) t2
      exact tag_represents cap (cUnitWrite c q slot u) t1
        (unit_represents cap c s rep q slot u hq hslot) q slot tag hq hslot

variable {cap : Nat}
def Safe (cap : Nat) {α : Type} (action : SailM α) : Prop :=
  ∀ s : St, (Good cap) s → match action.run s with
    | .ok _ after => (Good cap) after
    | .error _ after => (Good cap) after

theorem preserves_pure {α : Type} (a : α) : (Safe cap) (pure a) := by
  intro s hs
  exact hs

theorem preserves_ite {α : Type} (p : Prop) [Decidable p] (a b : SailM α)
    (ha : (Safe cap) a) (hb : (Safe cap) b) : (Safe cap) (if p then a else b) := by
  split <;> assumption

theorem preserves_bind {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : (Safe cap) a) (hf : ∀ x, (Safe cap) (f x)) : (Safe cap) (a >>= f) := by
  intro s hs
  have h := ha s hs
  cases he : a.run s with
  | ok x after =>
    have hp : (Good cap) after := by simpa [he] using h
    have hf' := hf x after hp
    simpa [EStateM.run, bind, EStateM.bind, show a s = .ok x after from he] using hf'
  | error error after =>
    simpa [EStateM.run, bind, EStateM.bind, show a s = .error error after from he] using h

theorem preserves_read (r : Register) : (Safe cap) (Out.readReg r) := by
  intro s hs
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind, get, getThe,
    MonadStateOf.get, EStateM.get]
  cases s.regs.get? r <;> exact hs

theorem preserves_other_write (r : Register) (v : RegisterType r)
    (hh : r ≠ Register.iq_head) (hl : r ≠ Register.iq_len) : Safe cap (Out.writeReg r v) := by
  intro s hs
  obtain ⟨c,rep,bounds⟩ := hs
  have hb := Proofs.Initialized.preserves_other_write (cap:=cap) r v hh hl s bounds
  simp [Out.writeReg,PreSail.writeReg,EStateM.run,modify,modifyGet,
    MonadStateOf.modifyGet,EStateM.modifyGet] at hb ⊢
  exact ⟨c,rep,hb⟩

theorem preserves_forIn {β : Type} (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, (Safe cap) (f i hi value)) : (Safe cap) (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => (Safe cap) (IntRange.forIn'.loop range f value i hs))
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


theorem push_safe (q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts cap) : Safe cap (pushC cap q u tag) := by
  intro s hs
  obtain ⟨c,rep,bounds⟩ := hs
  obtain ⟨b,t,he,hb⟩ := Proofs.Initialized.push_preserves cap q hq u tag ports s bounds
  obtain ⟨b',t',c',he',hr⟩ := push_rep cap q hq u tag ports s bounds c rep
  rw [he] at he'
  cases he'
  simpa [he] using (show Good cap t from ⟨c',hr,hb⟩)

theorem overlay_safe (q w : Nat) (e : BitVec 10) : Safe cap (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

theorem deliver_safe (rate q c : Nat) (hq : q<145) (ports : Proofs.Initialized.WritePorts cap) :
    Safe cap (deliverC cap rate q c) := by
  unfold deliverC
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact push_safe q hq _ _ ports
    | exact overlay_safe _ _ _
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x

theorem actual_push_safe (q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts iq_cap) : Safe iq_cap (iq_push_tagged q u tag) := by
  simpa only [push_production] using push_safe (cap := iq_cap) q hq u tag ports

theorem image_len_bind_safe {α : Type} (q : Nat) (hq : q<145)
    (length : Nat → Nat) (image : Proofs.Reset.ImageEnv length) (f : Nat → SailM α)
    (hf : Safe iq_cap (f (length q))) : Safe iq_cap ((iq_image_len q) >>= f) := by
  intro s hs
  have he := image.len_frame q s hq
  have h := hf s hs
  change iq_image_len q s = .ok (length q) s at he
  simpa [EStateM.run,bind,EStateM.bind,he] using h

theorem image_unit_safe (q i : Nat) (hq : q<145)
    (length : Nat → Nat) (hi : i<length q) (image : Proofs.Reset.ImageEnv length) :
    Safe iq_cap (iq_image_unit q i) := by
  intro s hs
  obtain ⟨u,he⟩ := image.unit_frame q i s hq hi
  simpa [he] using hs

theorem reset_tail_safe (ports : Proofs.Initialized.WritePorts iq_cap) (length : Nat → Nat)
    (image : Proofs.Reset.ImageEnv length) : Safe iq_cap Proofs.Reset.resetTail := by
  unfold Proofs.Reset.resetTail
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact actual_push_safe _ (by simp only [Membership.mem] at *; simp at *; omega) _ _ ports
    | with_reducible apply image_len_bind_safe _ (by simp only [Membership.mem] at *; simp at *; omega) length image
    | exact image_unit_safe _ _ (by simp only [Membership.mem] at *; simp at *; omega) length (by simp only [Membership.mem] at *; simp at *; omega) image
    | with_reducible apply preserves_read
    | with_reducible apply preserves_pure
    | with_reducible apply preserves_other_write <;> decide
    | with_reducible apply preserves_ite
    | with_reducible apply preserves_forIn
    | with_reducible apply preserves_bind
    | split
    | intro x


/-- Whole actual reset preserves represented backing storage and establishes initialized
    logical occupancy, including returned error states and repeated reset with stale cells. -/
theorem reset_product (s : St) (hcap : 0<iq_cap) (c : CView) (rep : Represents iq_cap c s)
    (ports : Proofs.Initialized.WritePorts iq_cap) (length : Nat → Nat)
    (image : Proofs.Reset.ImageEnv length) :
    match (iq_reset ()).run s with
    | .ok _ after => Good iq_cap after
    | .error _ after => Good iq_cap after := by
  rw [Proofs.Reset.reset_split]
  let s0 : St := {s with regs := (s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145)}
  have hzero : Good iq_cap s0 := ⟨c,rep,Proofs.Initialized.zero_bounds iq_cap hcap s⟩
  have ht := reset_tail_safe ports length image s0 hzero
  change (match Proofs.Reset.resetTail.run s0 with
    | .ok _ after => Good iq_cap after
    | .error _ after => Good iq_cap after)
  cases he : Proofs.Reset.resetTail.run s0 <;> simpa only [he] using ht

/-- Any live read after a represented whole action has the actual indexed C read's
    initialization/domain preconditions, with unit and tag identity for its witness. -/
theorem live_read (cap : Nat) (s : St) (good : Good cap s) (q slot : Nat)
    (live : Proofs.Storage.Live cap s q slot) :
    ∃ c, Represents cap c s ∧ q<145 ∧ slot<cap ∧ c.initialized q slot ∧
      units cap s q slot=c.unit q slot ∧ tags cap s q slot=c.tag q slot := by
  obtain ⟨c,rep,bounds⟩ := good
  obtain ⟨hq,hs,hi,hu⟩ := Proofs.Initialized.represented_live_read cap c s rep bounds q slot live
  exact ⟨c,rep,hq,hs,hi,hu,represented_tag cap c s rep q slot hq hs⟩
end Proofs.StorageProduct
