import Proofs.TagCoverage
import Proofs.ParcelFrame
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep
open Proofs.IqReach Proofs.OrderedImage Proofs.OrderedDelivery Proofs.TagCoverage
set_option autoImplicit false
set_option maxRecDepth 6000
set_option maxHeartbeats 1000000
namespace Proofs.AllocationHistory
abbrev State := SequentialState RegisterType trivialChoiceSource

/-- Allocation history retains old physical arrays even when the logical IQ is
    empty. A newly nonempty queue allocates through its real paired tag store. -/
def Dynamic (cap : Nat) (initial : Nat → Prop) (s : State) : Prop :=
  ∃ L : Vector Nat 145, s.regs.get? Register.iq_len=some L ∧
    Allocation cap (fun q => initial q ∨ 0<L[q]!) s

def Ready (cap : Nat) (base : State) (initial : Nat → Prop) (s : State) : Prop :=
  TaggedReady cap base s ∧ Dynamic cap initial s

theorem allocation_memory (cap : Nat) (A : Nat → Prop) (s t : State)
    (h : Allocation cap A s) (same : t.mem=s.mem) : Allocation cap A t := by
  simpa only [Allocation,TagAt,Present,same] using h

theorem dynamic_frame (cap : Nat) (A : Nat → Prop) (s t : State)
    (h : Dynamic cap A s) (len : t.regs.get? Register.iq_len=s.regs.get? Register.iq_len)
    (memory : t.mem=s.mem) : Dynamic cap A t := by
  obtain ⟨L,hL,alloc⟩ := h
  exact ⟨L,len.trans hL,allocation_memory cap _ s t alloc memory⟩

private theorem update_other {α : Type} [Inhabited α] {n : Nat}
    (V : Vector α n) (q r : Nat) (x : α) (_hq : q<n) (hr : r<n) (hne : r≠q) :
    (vectorUpdate V q x)[r]! = V[r]! := by
  simp [vectorUpdate,Vector.set!,hr,Array.getElem_setIfInBounds_ne,Ne.symm hne]

theorem dynamic_push (cap q : Nat) (hq : q<145) (A : Nat → Prop)
    (u : BitVec 32) (tag : Int) (ports : Proofs.Initialized.WritePorts cap)
    (s : State) (tagged : TagBounds cap s) (dynamic : Dynamic cap A s) :
    ∃ b after, (pushC cap q u tag).run s=.ok b after ∧ Dynamic cap A after := by
  obtain ⟨H,L,hH,hL,bounds⟩ := tagged
  obtain ⟨L0,hL0,allocation⟩ := dynamic
  have eq : L0=L := by rw [hL] at hL0; cases hL0; rfl
  subst L0
  by_cases full : cap≤L[q]!
  · exact ⟨false,s,iq_push_full q u tag s L hL full,L,hL,allocation⟩
  · have room : L[q]!<cap := by omega
    let slot := ringC cap (H[q]!+L[q]!)
    have hslot : slot<cap := by
      dsimp [slot]
      rw [Proofs.Capacity.ring_spec]
      have hh := (bounds q hq).1
      split <;> omega
    let t1 := writeCell s (145*cap) 0 (q*cap+slot) u.toNat
    let t2 := writeCell t1 (145*cap) 1 (q*cap+slot) (zigzag tag)
    let after : State := {t2 with regs:=t2.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1))}
    have e1 : iq_unit_write q slot u s=.ok () t1 := ports.unit q slot u s hq hslot
    have e2 : iq_tag_write q slot tag t1=.ok () t2 := ports.tag q slot tag t1 hq hslot
    have r1 : t1.regs=s.regs := rfl
    have r2 : t2.regs=s.regs := rfl
    have cast : ((H[q]! : Int)+(L[q]! : Int)).toNat=H[q]!+L[q]! := by omega
    have run : (pushC cap q u tag).run s=.ok true after := by
      simp [pushC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,hL,hH,
        EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,
        EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,full,cast,
        slot,e1,e2,r1,r2,after]
    have first := unit_keeps_allocation cap (fun r => A r ∨ 0<L[r]!) s allocation
      (q*cap+slot) u.toNat
    have second := tag_store_allocates cap (fun r => A r ∨ 0<L[r]!) t1 first q slot (zigzag tag) hq hslot
    refine ⟨true,after,run,vectorUpdate L q (L[q]!+1),?_,?_⟩
    · simp [after]
    · intro r hr
      have pair := second r hr
      change ((A r ∨ 0<(vectorUpdate L q (L[q]!+1))[r]!) ↔ _) 
      change (allocate (fun r => A r ∨ 0<L[r]!) q r ↔ _) at pair
      by_cases eq : r=q
      · subst r
        simpa [allocate,vectorUpdate_self _ _ _ hq] using pair
      · simpa [allocate,eq,update_other L q r _ hq hr eq] using pair

theorem ready_read (cap : Nat) (base : State) (A : Nat → Prop) (r : Register) :
    Total (Ready cap base A) (Out.readReg r) := by
  intro s hs
  obtain ⟨value,after,run,tagged⟩ := tagged_read cap base r s hs.1
  have eq : after=s := by
    have frame := Proofs.ParcelFrame.read_frame s r
    simpa [Proofs.ParcelFrame.FrameAt,run] using frame
  subst after
  exact ⟨value,s,run,tagged,hs.2⟩

theorem ready_write (cap : Nat) (base : State) (A : Nat → Prop)
    (r : Register) (v : RegisterType r) (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) :
    Total (Ready cap base A) (Out.writeReg r v) := by
  intro s hs
  obtain ⟨u,t,run,tagged⟩ := tagged_write cap base r v hh hl s hs.1
  change EStateM.Result.ok () {s with regs:=s.regs.insert r v}=.ok u t at run
  cases run
  refine ⟨(),_,rfl,tagged,dynamic_frame cap A s _ hs.2 ?_ rfl⟩
  simp [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl]

theorem ready_push (cap : Nat) (base : State) (A : Nat → Prop) (q : Nat) (hq : q<145)
    (u : BitVec 32) (tag : Int) (ports : Proofs.Initialized.WritePorts cap) :
    Total (Ready cap base A) (pushC cap q u tag) := by
  intro s hs
  obtain ⟨b,t,run,tagged⟩ := tagged_push cap base q hq u tag ports s hs.1
  obtain ⟨b',t',run',dynamic⟩ := dynamic_push cap q hq A u tag ports s hs.1.2 hs.2
  rw [run] at run'
  cases run'
  exact ⟨b,t,run,tagged,dynamic⟩

theorem ready_overlay (cap : Nat) (base : State) (A : Nat → Prop) (q w : Nat) (e : BitVec 10) :
    Total (Ready cap base A) (iq_word_ov_set q w e) := by
  intro s hs
  obtain ⟨u,t,run,tagged⟩ := tagged_overlay cap base q w e s hs.1
  have start : OverlayFrame s s := ⟨rfl,rfl,(by intros; rfl),hs.1.1.2.1 Register.iq_wov⟩
  obtain ⟨u',t',run',frame⟩ := overlay_total s q w e s start
  rw [run] at run'
  cases run'
  exact ⟨u,t,run,tagged,dynamic_frame cap A s t hs.2
    (frame.2.2.1 Register.iq_len (by decide) (by decide)) frame.1⟩

theorem actual_delivery_allocation (cap rate q c : Nat) (hq : q<145) (base : State)
    (A : Nat → Prop) (ports : Proofs.Initialized.WritePorts cap) :
    Total (Ready cap base A) (deliverC cap rate q c) := by
  unfold deliverC
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact ready_push cap base A q hq _ _ ports
    | exact ready_overlay cap base A _ _ _
    | exact ready_read cap base A _
    | apply ready_write <;> decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

theorem image_len_bind {α : Type} (cap : Nat) (base : State) (A : Nat → Prop) (q : Nat) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (f : Nat → SailM α) (hf : Total (Ready cap base A) (f (length q))) :
    Total (Ready cap base A) (iq_image_len q >>= f) := by
  intro s hs
  obtain ⟨v,t,he,ht⟩ := hf s hs
  have read : iq_image_len q s=.ok (length q) s := values.len q s hq
  refine ⟨v,t,?_,ht⟩
  simpa [EStateM.run,bind,EStateM.bind,read] using he

theorem image_unit_total (cap : Nat) (base : State) (A : Nat → Prop) (q i : Nat) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (hi : i<length q) : Total (Ready cap base A) (iq_image_unit q i) := by
  intro s hs
  exact ⟨image q i,s,values.unit q i s hq hi,hs⟩

theorem reset_tail_total (cap : Nat) (base : State) (A : Nat → Prop)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length) :
    Total (Ready cap base A) (resetTailC cap) := by
  unfold resetTailC nativeImageLoop imageStep
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact ready_push cap base A _ (by simp only [Membership.mem] at *; simp at *; omega) _ _ ports
    | with_reducible apply image_len_bind cap base A _ (by simp only [Membership.mem] at *; simp at *; omega) image length values
    | exact image_unit_total cap base A _ _ (by simp only [Membership.mem] at *; simp at *; omega)
        image length values (by simp only [Membership.mem] at *; simp at *; omega)
    | exact ready_read cap base A _
    | apply ready_write <;> decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

/-- The complete actual IQ reset retains old physical allocations, and adds exactly
    those forced by real image paired stores; padding/refusal performs no tag store. -/
theorem actual_reset_allocation (cap : Nat) (hcap : 0<cap) (base s : State)
    (view : CView) (rep : Represents cap view s) (complete : MachineInitProofs.Complete s)
    (foreign : ForeignFrame base s) (A : Nat → Prop) (allocated : Allocation cap A s)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length) :
    ∃ after, (resetC cap).run s=.ok () after ∧ Ready cap base A after := by
  let initial : State := {s with regs := ((s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145))}
  have good : Proofs.StorageProduct.Good cap initial :=
    ⟨view,rep,Proofs.Initialized.zero_bounds cap hcap s⟩
  have cp : MachineInitProofs.Complete initial :=
    MachineInitProofs.complete_insert _ Register.iq_len _ (MachineInitProofs.complete_insert s Register.iq_head _ complete)
  have tags : TaggedReady cap base initial := ⟨⟨good,cp,foreign⟩,zero_tag_bounds cap hcap s⟩
  have dyn : Dynamic cap A initial := by
    refine ⟨vectorInit 0,by simp [initial],?_⟩
    intro q hq
    have zero : (vectorInit 0 : Vector Nat 145)[q]! = 0 := by simp [vectorInit,hq]
    change (A q ∨ 0<(vectorInit 0 : Vector Nat 145)[q]!) ↔ ∃ slot, slot<cap ∧ TagAt cap s q slot
    rw [zero]
    simpa using allocated q hq
  obtain ⟨v,after,run,result⟩ := reset_tail_total cap base A ports image length values initial ⟨tags,dyn⟩
  refine ⟨after,?_,result⟩
  have run' : resetTailC cap initial=.ok () after := run
  simpa [resetC,Out.writeReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,
    modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,initial,run'] using run

/-- A new producer phase starts from the allocator projection proved by the preceding
    phase. This rebase is essential: consuming a last item never frees its tag array. -/
theorem from_allocation (cap : Nat) (base : State) (A : Nat → Prop) (s : State)
    (tags : TaggedReady cap base s) (allocated : Allocation cap A s) : Ready cap base A s := by
  refine ⟨tags,?_⟩
  obtain ⟨H,L,hH,hL,bounds⟩ := tags.2
  refine ⟨L,hL,?_⟩
  intro q hq
  constructor
  · intro h
    apply (allocated q hq).mp
    rcases h with old | live
    · exact old
    · exact occupied_queue_allocated cap A s ⟨H,L,hH,hL,bounds⟩ allocated L hL q hq live
  · intro h
    exact Or.inl ((allocated q hq).mpr h)

theorem expose_boundary (cap : Nat) (base : State) (A : Nat → Prop) (s : State)
    (h : Ready cap base A s) :
    ∃ L : Vector Nat 145, s.regs.get? Register.iq_len=some L ∧
      Allocation cap (fun q => A q ∨ 0<L[q]!) s ∧
      (∀ q, A q → (A q ∨ 0<L[q]!)) := by
  obtain ⟨tags,L,hL,allocated⟩ := h
  exact ⟨L,hL,allocated,fun _ old => Or.inl old⟩

/-- Reset/delivery phases expose a definite monotone A0→A1 map. Consumer
    phases retain A1 exactly; they do not reuse A0∨currentLength after a pop. -/
def Extends (before after : Nat → Prop) : Prop := ∀ q, before q → after q

theorem extends_trans (A B C : Nat → Prop) (ab : Extends A B) (bc : Extends B C) :
    Extends A C := fun q h => bc q (ab q h)

end Proofs.AllocationHistory
