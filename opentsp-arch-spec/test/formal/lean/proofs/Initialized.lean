import Proofs.StorageRep
import Proofs.Reset
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep

namespace Proofs.Initialized

/-- The named trusted port interpretation, not a theorem equating opaque declarations
    with C. Concrete C transfer is required only on represented states and initialized
    occupied reads; behavior on malformed/unreachable abstract states is a total extension. -/
structure WritePorts (cap : Nat) : Prop where
  unit : ∀ q slot u (s : St), q<145 → slot<cap →
    (iq_unit_write q slot u).run s =
      .ok () (writeCell s (145*cap) 0 (q*cap+slot) u.toNat)
  tag : ∀ q slot value (s : St), q<145 → slot<cap →
    (iq_tag_write q slot value).run s =
      .ok () (writeCell s (145*cap) 1 (q*cap+slot) (zigzag value))

def Bounds (cap : Nat) (s : St) : Prop :=
  ∃ H L : Vector Nat 145, s.regs.get? Register.iq_head = some H ∧
    s.regs.get? Register.iq_len = some L ∧
    ∀ q, q<145 → H[q]!<cap ∧ L[q]!≤cap ∧
      ∀ i, i<L[q]! → Initialized cap s q (ringC cap (H[q]! + i))

private theorem update_other {α : Type} [Inhabited α] {n : Nat}
    (V : Vector α n) (q r : Nat) (x : α) (hr : r<n) (hne : r≠q) :
    (vectorUpdate V q x)[r]! = V[r]! := by
  simp [vectorUpdate,Vector.set!,hr,Array.getElem_setIfInBounds_ne,Ne.symm hne]

theorem zero_bounds (cap : Nat) (hcap : 0<cap) (s : St) :
    Bounds cap {s with regs := (s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145)} := by
  refine ⟨vectorInit 0,vectorInit 0,?_,?_,?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · intro q hq
    simp [vectorInit,hq]
    exact hcap

theorem push_preserves (cap q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : WritePorts cap) (s : St) (hs : Bounds cap s) :
    ∃ b after, (pushC cap q u tag).run s = .ok b after ∧ Bounds cap after := by
  obtain ⟨H,L,hH,hL,hb⟩ := hs
  by_cases full : cap≤L[q]!
  · exact ⟨false,s,iq_push_full q u tag s L hL full,H,L,hH,hL,hb⟩
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
    have he : (pushC cap q u tag).run s = .ok true after := by
      simp [pushC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,hL,hH,
        EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,
        EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,full,hc,slot,e1,e2,r1,r2,after]
    refine ⟨true,after,he,H,vectorUpdate L q (L[q]!+1),?_,?_,?_⟩
    · simp [after,r2,Std.ExtDHashMap.get?_insert,hH]
    · simp [after]
    · intro r hr
      constructor
      · exact (hb r hr).1
      constructor
      · by_cases eq : r=q
        · subst r; rw [vectorUpdate_self L q _ hq]; omega
        · rw [update_other L q r _ hr eq]
          exact (hb r hr).2.1
      · intro i hi
        have lift_old : ∀ j, j<L[r]! → Initialized cap after r (ringC cap (H[r]! + j)) := by
          intro j hj
          change Initialized cap t2 r (ringC cap (H[r]! + j))
          apply (initialized_cross cap t1 (q*cap+slot) (zigzag tag) r _).2
          apply initialized_preserved cap s q slot u.toNat r _ hq hslot hr
          · rw [Proofs.Capacity.ring_spec]
            have hh := (hb r hr).1
            have hl := (hb r hr).2.1
            split <;> omega
          · exact (hb r hr).2.2 j hj
        by_cases eq : r=q
        · subst r
          rw [vectorUpdate_self L q _ hq] at hi
          by_cases old : i<L[q]!
          · exact lift_old i old
          · have last : i=L[q]! := by omega
            subst i
            change Initialized cap t2 q slot
            apply (initialized_cross cap t1 (q*cap+slot) (zigzag tag) q slot).2
            exact initialized_written cap s q slot u.toNat hq hslot
        · rw [update_other L q r _ hr eq] at hi
          exact lift_old i hi

theorem pop_preserves (cap q n : Nat) (hq : q<145) (s : St) (hs : Bounds cap s)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L) (hn : n≤L[q]!) :
    ∃ after, (popC cap q n).run s = .ok () after ∧ Bounds cap after := by
  obtain ⟨H0,L0,hH0,hL0,hb⟩ := hs
  have eH : H0=H := by simpa [hH] using hH0.symm
  have eL : L0=L := by simpa [hL] using hL0.symm
  subst H0; subst L0
  let after : St := {s with regs := (s.regs.insert Register.iq_head (vectorUpdate H q (ringC cap (H[q]!+n)))).insert Register.iq_len (vectorUpdate L q (L[q]!-n))}
  refine ⟨after,iq_pop_run q n s H L hH hL,
    vectorUpdate H q (ringC cap (H[q]!+n)),vectorUpdate L q (L[q]!-n),?_,?_,?_⟩
  · simp [after,Std.ExtDHashMap.get?_insert]
  · simp [after]
  · intro r hr
    by_cases eq : r=q
    · subst r
      rw [vectorUpdate_self H q _ hq,vectorUpdate_self L q _ hq]
      constructor
      · rw [Proofs.Capacity.ring_spec]
        have hh := (hb q hq).1
        have hl := (hb q hq).2.1
        split <;> omega
      constructor
      · have hl := (hb q hq).2.1
        omega
      · intro i hi
        change Initialized cap s q (ringC cap (ringC cap (H[q]!+n) + i))
        rw [Proofs.Storage.ring_shift cap H[q]! L[q]! n i (hb q hq).1 (hb q hq).2.1 hn hi]
        simpa [Nat.add_assoc] using (hb q hq).2.2 (n + i) (by omega)
    · rw [update_other H q r _ hr eq,update_other L q r _ hr eq]
      exact hb r hr

variable {cap : Nat}
def Safe (cap : Nat) {α : Type} (action : SailM α) : Prop :=
  ∀ s : St, (Bounds cap) s → match action.run s with
    | .ok _ after => (Bounds cap) after
    | .error _ after => (Bounds cap) after

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
    have hp : (Bounds cap) after := by simpa [he] using h
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
    (hh : r ≠ Register.iq_head) (hl : r ≠ Register.iq_len) : (Safe cap) (Out.writeReg r v) := by
  intro s hs
  obtain ⟨H,L,hH,hL,hbounds⟩ := hs
  simp [Out.writeReg, PreSail.writeReg, EStateM.run, modify, modifyGet,
    MonadStateOf.modifyGet, EStateM.modifyGet]
  exact ⟨H,L,by simpa [Std.ExtDHashMap.get?_insert, hh, Ne.symm hh] using hH,
    by simpa [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl] using hL,hbounds⟩

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
    (ports : WritePorts cap) : Safe cap (pushC cap q u tag) := by
  intro s hs
  obtain ⟨b,after,he,ha⟩ := push_preserves cap q hq u tag ports s hs
  simpa [he] using ha

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

theorem deliver_safe (rate q c : Nat) (hq : q<145) (ports : WritePorts cap) :
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
    (ports : WritePorts iq_cap) : Safe iq_cap (iq_push_tagged q u tag) := by
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

theorem reset_tail_safe (ports : WritePorts iq_cap) (length : Nat → Nat)
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

/-- Complete actual reset establishes initialized logical occupancy even after restart:
    old external cells may remain; zero logical length makes them initially unobservable,
    and every newly live image unit is written before length increases. -/
theorem reset_initializes (s : St) (hcap : 0<iq_cap)
    (ports : WritePorts iq_cap) (length : Nat → Nat) (image : Proofs.Reset.ImageEnv length) :
    match (iq_reset ()).run s with
    | .ok _ after => Bounds iq_cap after
    | .error _ after => Bounds iq_cap after := by
  rw [Proofs.Reset.reset_split]
  let s0 : St := {s with regs := (s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145)}
  have hzero : Bounds iq_cap s0 := zero_bounds iq_cap hcap s
  have ht := reset_tail_safe ports length image s0 hzero
  change (match Proofs.Reset.resetTail.run s0 with
    | .ok _ after => Bounds iq_cap after
    | .error _ after => Bounds iq_cap after)
  cases he : Proofs.Reset.resetTail.run s0 <;> simpa only [he] using ht

theorem live_initialized (cap : Nat) (s : St) (hs : Bounds cap s)
    (q slot : Nat) (live : Proofs.Storage.Live cap s q slot) :
    Initialized cap s q slot := by
  obtain ⟨H,L,hH,hL,hb⟩ := hs
  obtain ⟨H0,L0,hH0,hL0,hq,_,_,i,hi,he⟩ := live
  have eH : H0=H := by simpa [hH] using hH0.symm
  have eL : L0=L := by simpa [hL] using hL0.symm
  subst H0; subst L0
  rw [he]
  exact (hb q hq).2.2 i hi

/-- The concrete C unit-read precondition is discharged for every live model slot:
    its index is in the admitted array and its cell was initialized. -/
theorem represented_live_read (cap : Nat) (c : CView) (s : St)
    (rep : Represents cap c s) (hs : Bounds cap s)
    (q slot : Nat) (live : Proofs.Storage.Live cap s q slot) :
    q<145 ∧ slot<cap ∧ c.initialized q slot ∧ units cap s q slot=c.unit q slot := by
  obtain ⟨hq,hslot⟩ := Proofs.Storage.live_domain cap s q slot live
  have hi := live_initialized cap s hs q slot live
  have ci := (rep.2 q slot hq hslot).1.mpr hi
  exact ⟨hq,hslot,ci,represented_unit cap c s rep q slot hq hslot ci⟩

end Proofs.Initialized
