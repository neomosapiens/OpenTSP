import Proofs.TagCoverage
import Proofs.TextCheckProduct
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
set_option autoImplicit false
namespace Proofs.TaggedCheck
open Proofs.TagCoverage
open Proofs.ParcelFrame
def Keep (A : Nat → Prop) (H L : Vector Nat 145) (s : St) : Prop :=
  Proofs.TextCheckProduct.Keep H L s ∧ TagBounds iq_cap s ∧ Allocation iq_cap A s ∧ MachineInitProofs.Complete s

abbrev IrqFrame := Proofs.TextCheckProduct.IrqFrame

variable (A : Nat → Prop)
variable {H L : Vector Nat 145}
def Safe {α : Type} (action : SailM α) : Prop :=
  ∀ s : St, Keep A H L s → match action.run s with
    | .ok _ after => Keep A H L after
    | .error _ after => Keep A H L after
theorem preserves_pure {α : Type} (a : α) : (Safe A (H:=H) (L:=L)) (pure a) := by
  intro s hs
  exact hs

theorem preserves_ite {α : Type} (p : Prop) [Decidable p] (a b : SailM α)
    (ha : (Safe A (H:=H) (L:=L)) a) (hb : (Safe A (H:=H) (L:=L)) b) : (Safe A (H:=H) (L:=L)) (if p then a else b) := by
  split <;> assumption

theorem preserves_bind {α β : Type} (a : SailM α) (f : α → SailM β)
    (ha : (Safe A (H:=H) (L:=L)) a) (hf : ∀ x, (Safe A (H:=H) (L:=L)) (f x)) : (Safe A (H:=H) (L:=L)) (a >>= f) := by
  intro s hs
  have h := ha s hs
  cases he : a.run s with
  | ok x after =>
    have hp : (Keep A H L) after := by simpa [he] using h
    have hf' := hf x after hp
    simpa [EStateM.run, bind, EStateM.bind, show a s = .ok x after from he] using hf'
  | error error after =>
    simpa [EStateM.run, bind, EStateM.bind, show a s = .error error after from he] using h

theorem preserves_read (r : Register) : (Safe A (H:=H) (L:=L)) (Out.readReg r) := by
  intro s hs
  simp [Out.readReg, PreSail.readReg, EStateM.run, bind, EStateM.bind, get, getThe,
    MonadStateOf.get, EStateM.get]
  cases s.regs.get? r <;> exact hs

theorem preserves_other_write (r : Register) (v : RegisterType r)
    (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) : Safe A (H:=H) (L:=L) (Out.writeReg r v) := by
  intro s hs
  obtain ⟨⟨good,hH,hL⟩,tags,allocation,complete⟩ := hs
  have hg := Proofs.StorageProduct.preserves_other_write (cap:=iq_cap) r v hh hl s good
  simp [Out.writeReg,PreSail.writeReg,EStateM.run,modify,modifyGet,
    MonadStateOf.modifyGet,EStateM.modifyGet] at hg ⊢
  exact ⟨⟨hg,by simpa [Std.ExtDHashMap.get?_insert,hh,Ne.symm hh] using hH,
    by simpa [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl] using hL⟩,
    tag_bounds_other_write iq_cap s tags r v hh hl,
    allocation_same_memory iq_cap A s _ allocation rfl,MachineInitProofs.complete_insert s r v complete⟩

theorem preserves_forIn {β : Type} (range : IntRange) (b : β)
    (f : (i : Int) → i ∈ range → β → SailM (ForInStep β))
    (hf : ∀ i hi value, (Safe A (H:=H) (L:=L)) (f i hi value)) : (Safe A (H:=H) (L:=L)) (IntRange.forIn' range b f) := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun value i hs => (Safe A (H:=H) (L:=L)) (IntRange.forIn'.loop range f value i hs))
  · intro value i hs hi ih
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi]
    apply preserves_bind A _ _ (hf i hi value)
    intro step
    cases step with
    | done value => exact preserves_pure A value
    | yield value => exact ih value
  · intro value i hs hi
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hi]
    exact preserves_pure A value


theorem frame_safe {α : Type} (a : SailM α)
    (hf : ∀ s, Keep A H L s → FrameAt s a) : Safe A (H:=H) (L:=L) a := by
  intro s hs
  have h := hf s hs
  cases he : a.run s with
  | ok x after =>
    have eq : after=s := by simpa [FrameAt,he] using h
    simpa [he,eq] using hs
  | error error after =>
    have eq : after=s := by simpa [FrameAt,he] using h
    simpa [he,eq] using hs

theorem unit_write_safe (q slot : Nat) (u : BitVec 32) (hq : q<145) (hslot : slot<iq_cap)
    (ports : Proofs.Initialized.WritePorts iq_cap) :
    Safe A (H:=H) (L:=L) (iq_unit_write q slot u) := by
  intro s hs
  obtain ⟨⟨⟨c,rep,bounds⟩,hH,hL⟩,tags,allocation,complete⟩ := hs
  have he := ports.unit q slot u s hq hslot
  let t := writeCell s (145* iq_cap) 0 (q* iq_cap+slot) u.toNat
  have ht : Proofs.Initialized.Bounds iq_cap t := by
    obtain ⟨H0,L0,hH0,hL0,hb⟩ := bounds
    refine ⟨H0,L0,hH0,hL0,?_⟩
    intro r hr
    refine ⟨(hb r hr).1,(hb r hr).2.1,?_⟩
    intro i hi
    apply initialized_preserved iq_cap s q slot u.toNat r _ hq hslot hr
    · rw [Proofs.Capacity.ring_spec]
      have hh := (hb r hr).1
      have hl := (hb r hr).2.1
      split <;> omega
    · exact (hb r hr).2.2 i hi
  have keep : Keep A H L t := ⟨⟨⟨cUnitWrite c q slot u,unit_represents iq_cap c s rep q slot u hq hslot,ht⟩,hH,hL⟩,
    tag_bounds_unit_write iq_cap s tags (q * iq_cap+slot) u.toNat,
    unit_keeps_allocation iq_cap A s allocation (q * iq_cap+slot) u.toNat,complete⟩
  simpa [he] using keep

theorem view_safe (q pos : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hi : pos<L[q]!) (ports : ReadPorts iq_cap) :
    Safe A (H:=H) (L:=L) (iq_view q pos) := by
  apply frame_safe A
  intro s hs
  exact view_frame s q pos ports H hs.1.2.1 L hs.1.2.2 hs.1.1 hq hh hl hi

theorem word_safe (q pos : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hi : pos≤iq_cap) (ports : ReadPorts iq_cap) :
    Safe A (H:=H) (L:=L) (iq_unit_word q pos) := by
  apply frame_safe A
  intro s hs
  exact unit_word_frame s q pos ports H hs.1.2.1 hq hh hi

theorem irq_safe (c n : Nat) (hn : n<16) (irq : IrqFrame) :
    Safe A (H:=H) (L:=L) (host_irq_callback c n) := by
  intro s hs
  have frame := irq.irq c n s hn
  have transfer (after : St) (regs : after.regs=s.regs) (mem : after.mem=s.mem) :
      Keep A H L after := by
    refine ⟨(Proofs.TextCheckProduct.keep_same_fields H L s after regs mem).mpr hs.1, ?_, ?_, ?_⟩
    · simpa only [TagBounds, TagAt, Present, regs, mem] using hs.2.1
    · exact allocation_same_memory iq_cap A s after hs.2.2.1 mem
    · simpa only [MachineInitProofs.Complete, regs] using hs.2.2.2
  cases run : (host_irq_callback c n).run s with
  | ok value after =>
    simp only [run] at frame ⊢
    exact transfer after frame.1 frame.2
  | error error after =>
    simp only [run] at frame ⊢
    exact transfer after frame.1 frame.2

theorem host_msix_safe (c : Nat) (irq : IrqFrame) :
    Safe A (H:=H) (L:=L) (host_msix c 0) := by
  unfold host_msix host_deliver
  repeat' first
    | exact irq_safe A c 0 (by decide) irq
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_bind A
    | split
    | intro x

theorem ras_safe (c : Nat) (q : Int) (kind : Nat) (loc : BitVec 43) (sl pos : Nat) (irq : IrqFrame) :
    Safe A (H:=H) (L:=L) (ras_record c q kind loc sl pos) := by
  unfold ras_record
  repeat' first
    | exact host_msix_safe A c irq
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_bind A
    | split
    | intro x

theorem ras_ecc_safe (c q kind : Nat) (unc isw : Bool) (id addr sl pos : Nat) (irq : IrqFrame) :
    Safe A (H:=H) (L:=L) (ras_ecc_event c q kind unc isw id addr sl pos) := by
  unfold ras_ecc_event
  exact ras_safe A c q _ _ sl pos irq

theorem head_bind_safe {α : Type} (f : Vector Nat 145 → SailM α)
    (hf : Safe A (H:=H) (L:=L) (f H)) :
    Safe A (H:=H) (L:=L) ((Out.readReg Register.iq_head) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_head s = .ok H s := by
    simp [Out.readReg,PreSail.readReg,hs.1.2.1,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,he] using hf s hs

theorem length_bind_safe {α : Type} (f : Vector Nat 145 → SailM α)
    (hf : Safe A (H:=H) (L:=L) (f L)) :
    Safe A (H:=H) (L:=L) ((Out.readReg Register.iq_len) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_len s = .ok L s := by
    simp [Out.readReg,PreSail.readReg,hs.1.2.2,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,he] using hf s hs

theorem peek_safe (q pos : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hi : pos<L[q]!) (ports : ReadPorts iq_cap) :
    Safe A (H:=H) (L:=L) (iq_peek q pos) := by
  apply frame_safe A
  intro s hs
  exact peek_frame s q pos ports H hs.1.2.1 L hs.1.2.2 hs.1.1 hq hh hl hi

theorem tag_read_safe (q slot : Nat) (hq : q<145) (hslot : slot<iq_cap) (ports : ReadPorts iq_cap) :
    Safe A (H:=H) (L:=L) (iq_tag_read q slot) := by
  apply frame_safe A
  intro s _
  obtain ⟨value,he⟩ := ports.tag q slot s hq hslot
  simp [FrameAt,he]

theorem overlay_read_safe (q w : Nat) :
    Safe A (H:=H) (L:=L) (iq_word_ov q w) := by
  apply frame_safe A
  intro s _
  exact word_overlay_frame s q w

theorem overlay_write_safe (q w : Nat) (e : BitVec 10) :
    Safe A (H:=H) (L:=L) (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_forIn A
    | with_reducible apply preserves_bind A
    | split
    | intro x


theorem length_map_bind_safe {α β : Type} (g : Vector Nat 145 → β) (f : β → SailM α)
    (hf : Safe A (H:=H) (L:=L) (f (g L))) :
    Safe A (H:=H) (L:=L) ((do let v ← Out.readReg Register.iq_len; pure (g v)) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_len s = .ok L s := by
    simp [Out.readReg,PreSail.readReg,hs.1.2.2,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he] using hf s hs

theorem head_map_bind_safe {α β : Type} (g : Vector Nat 145 → β) (f : β → SailM α)
    (hf : Safe A (H:=H) (L:=L) (f (g H))) :
    Safe A (H:=H) (L:=L) ((do let v ← Out.readReg Register.iq_head; pure (g v)) >>= f) := by
  intro s hs
  have he : Out.readReg Register.iq_head s = .ok H s := by
    simp [Out.readReg,PreSail.readReg,hs.1.2.1,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  simpa [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he] using hf s hs

theorem ring_int_domain (h : Nat) (i : Int) (hh : h<iq_cap)
    (hi0 : 0≤i) (hi : i≤iq_cap) : ring ((h:Int) + i).toNat<iq_cap := by
  have he : ((h:Int) + i).toNat=h + i.toNat := by omega
  rw [he]
  exact ring_domain h i.toNat hh (by omega)

theorem word_check_safe (q c : Nat) (w : Int) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    Safe A (H:=H) (L:=L) (iq_word_check q c w) := by
  unfold iq_word_check
  simp only [bind_assoc,pure_bind,bind_pure]
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact ras_ecc_safe A _ _ _ _ _ _ _ _ _ irq
    | exact overlay_read_safe A _ _
    | exact overlay_write_safe A _ _ _
    | with_reducible apply head_map_bind_safe A
    | with_reducible apply length_map_bind_safe A
    | with_reducible apply head_bind_safe A
    | with_reducible apply length_bind_safe A
    | exact peek_safe A q _ hq hh hl (by simp only [Membership.mem] at *; simp at *; omega) reads
    | exact unit_write_safe A q _ _ hq (by
        simp only [Membership.mem] at *
        simp at *
        apply ring_int_domain H[q]! _ hh <;> omega) writes
    | exact tag_read_safe A q _ hq (by
        simp only [Membership.mem] at *
        simp at *
        apply ring_int_domain H[q]! _ hh <;> omega) reads
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_forIn A
    | with_reducible apply preserves_bind A
    | split
    | intro x


/-- Full actual text checking/repair preserves the represented initialized FIFO
    shape and the exact captured head/length, even on partial failure paths. -/
theorem text_check_safe (q c units : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hu : units≤L[q]!) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    Safe A (H:=H) (L:=L) (iq_text_check q c units) := by
  unfold iq_text_check
  simp only [bind_assoc,pure_bind,bind_pure]
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact word_check_safe A q c _ hq hh hl reads writes irq
    | exact view_safe A q _ hq hh hl (by simp_all <;> omega) reads
    | exact word_safe A q _ hq hh (by simp_all <;> omega) reads
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_forIn A
    | with_reducible apply preserves_bind A
    | split
    | intro x


def popHead (H : Vector Nat 145) (q units : Nat) : Vector Nat 145 :=
  vectorUpdate H q (Proofs.Capacity.ringC iq_cap (H[q]!+units))
def popLength (L : Vector Nat 145) (q units : Nat) : Vector Nat 145 :=
  vectorUpdate L q (L[q]!-units)

theorem pop_product (q units : Nat) (hq : q<145) (hu : units≤L[q]!)
    (s : St) (keep : Keep A H L s) :
    ∃ after, (iq_pop q units).run s=.ok () after ∧ Keep A (popHead H q units) (popLength L q units) after := by
  obtain ⟨⟨⟨view,rep,bounds⟩,hH,hL⟩,tags,allocation,complete⟩ := keep
  let after : St := {s with regs := ((s.regs.insert Register.iq_head (popHead H q units)).insert Register.iq_len (popLength L q units))}
  have run : (Proofs.Capacity.popC iq_cap q units).run s=.ok () after :=
    Proofs.Capacity.iq_pop_run q units s H L hH hL
  obtain ⟨t,ht,hb⟩ := Proofs.Initialized.pop_preserves iq_cap q units hq s bounds H L hH hL hu
  rw [run] at ht
  cases ht
  obtain ⟨t,ht,tag_bounds⟩ := pop_tag_bounds iq_cap q units hq s tags H L hH hL hu
  rw [run] at ht
  cases ht
  refine ⟨after,?_,⟨⟨view,rep,hb⟩,?_,?_⟩,tag_bounds,?_,?_⟩
  · simpa only [Proofs.Capacity.pop_production] using run
  · simp [after,Std.ExtDHashMap.get?_insert]
  · simp [after,Std.ExtDHashMap.get?_insert]
  · exact allocation_same_memory iq_cap A s after allocation rfl
  · exact MachineInitProofs.complete_insert _ Register.iq_len _
      (MachineInitProofs.complete_insert s Register.iq_head _ complete)

theorem checked_pop_product (q c units : Nat) (hq : q<145) (hh : H[q]!<iq_cap)
    (hl : L[q]!≤iq_cap) (hu : units≤L[q]!) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame)
    (s : St) (keep : Keep A H L s) :
    match (do iq_text_check q c units; iq_pop q units).run s with
    | .ok _ after => Keep A (popHead H q units) (popLength L q units) after
    | .error _ after => Keep A H L after := by
  have hc := text_check_safe A q c units hq hh hl hu reads writes irq s keep
  cases check : (iq_text_check q c units).run s with
  | error error after =>
    have old : Keep A H L after := by simpa [check] using hc
    change iq_text_check q c units s=.error error after at check
    simpa [EStateM.run,bind,EStateM.bind,check] using old
  | ok u after =>
    cases u
    have checked : Keep A H L after := by simpa [check] using hc
    obtain ⟨done,pop,result⟩ := pop_product A q units hq hu after checked
    change iq_text_check q c units s=.ok () after at check
    change iq_pop q units after=.ok () done at pop
    simpa [EStateM.run,bind,EStateM.bind,check,pop] using result

theorem report_safe (q c : Nat) (d : diag) : Safe A (H:=H) (L:=L) (report q c d) := by
  unfold report diag_report diag_record
  repeat' first
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_bind A
    | split
    | intro x

theorem barrier_sync_safe (q c : Nat) : Safe A (H:=H) (L:=L) (barrier_sync q c) := by
  unfold barrier_sync
  repeat' first
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_bind A
    | split
    | intro x

theorem barrier_notify_safe (c : Nat) : Safe A (H:=H) (L:=L) (barrier_notify c) := by
  unfold barrier_notify
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_ite A
    | with_reducible apply preserves_forIn A
    | with_reducible apply preserves_bind A
    | split
    | intro x

end Proofs.TaggedCheck
