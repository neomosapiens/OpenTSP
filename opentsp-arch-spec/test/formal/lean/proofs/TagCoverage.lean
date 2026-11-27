import Proofs.OrderedFetch
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep
open Proofs.OrderedImage Proofs.OrderedDelivery Proofs.IqReach
set_option autoImplicit false
set_option maxRecDepth 4000
namespace Proofs.TagCoverage
open Out.Register

/-- Presence of a written tag, separate from the total -1 fallback tag value. -/
def TagAt (cap : Nat) (s : St) (q slot : Nat) : Prop :=
  ∃ value, Present s (145*cap) 1 (q*cap+slot) value

def TagBounds (cap : Nat) (s : St) : Prop :=
  ∃ H L : Vector Nat 145, s.regs.get? iq_head=some H ∧ s.regs.get? iq_len=some L ∧
    ∀ q, q<145 → H[q]!<cap ∧ L[q]!≤cap ∧
      ∀ i, i<L[q]! → TagAt cap s q (ringC cap (H[q]! + i))

theorem tag_unit_cross (cap : Nat) (s : St) (slot value q j : Nat) :
    TagAt cap (writeCell s (145*cap) 0 slot value) q j ↔ TagAt cap s q j := by
  unfold TagAt
  constructor
  · rintro ⟨v,hv⟩
    exact ⟨v,(present_write_cross s (145*cap) 0 1 slot (q*cap+j) value v (by decide) (by decide) (by decide)).1 hv⟩
  · rintro ⟨v,hv⟩
    exact ⟨v,(present_write_cross s (145*cap) 0 1 slot (q*cap+j) value v (by decide) (by decide) (by decide)).2 hv⟩

theorem tag_store_written (cap : Nat) (s : St) (q slot value : Nat)
    (hq : q<145) (hs : slot<cap) :
    TagAt cap (writeCell s (145*cap) 1 (q*cap+slot) value) q slot := by
  refine ⟨value,?_⟩
  rw [present_write s (145*cap) 1 1 (q*cap+slot) (q*cap+slot) value value
    (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap q slot hq hs)]
  simp

theorem tag_store_preserved (cap : Nat) (s : St) (q slot value r j : Nat)
    (hq : q<145) (hs : slot<cap) (hr : r<145) (hj : j<cap) (old : TagAt cap s r j) :
    TagAt cap (writeCell s (145*cap) 1 (q*cap+slot) value) r j := by
  rcases old with ⟨v,hv⟩
  by_cases eq : q*cap+slot=r*cap+j
  · refine ⟨value,?_⟩
    rw [present_write s (145*cap) 1 1 (q*cap+slot) (r*cap+j) value value
      (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj)]
    simp [eq]
  · refine ⟨v,?_⟩
    rw [present_write s (145*cap) 1 1 (q*cap+slot) (r*cap+j) value v
      (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj)]
    simpa [eq] using hv

private theorem update_other {α : Type} [Inhabited α] {n : Nat}
    (V : Vector α n) (q r : Nat) (x : α) (hr : r<n) (hne : r≠q) :
    (vectorUpdate V q x)[r]! = V[r]! := by
  simp [vectorUpdate,Vector.set!,hr,Array.getElem_setIfInBounds_ne,Ne.symm hne]

theorem zero_tag_bounds (cap : Nat) (hcap : 0<cap) (s : St) :
    TagBounds cap {s with regs := (s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145)} := by
  refine ⟨vectorInit 0,vectorInit 0,?_,?_,?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · intro q hq
    simp [vectorInit,hq]
    exact hcap

theorem push_tag_bounds (cap q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts cap) (s : St) (hs : TagBounds cap s) :
    ∃ b after, (pushC cap q u tag).run s = .ok b after ∧ TagBounds cap after := by
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
        have lift_old : ∀ j, j<L[r]! → TagAt cap after r (ringC cap (H[r]! + j)) := by
          intro j hj
          change TagAt cap t2 r (ringC cap (H[r]! + j))
          have target : ringC cap (H[r]! + j)<cap := by
            rw [Proofs.Capacity.ring_spec]
            have hh := (hb r hr).1
            have hl := (hb r hr).2.1
            split <;> omega
          exact tag_store_preserved cap t1 q slot (zigzag tag) r _ hq hslot hr target
            ((tag_unit_cross cap s (q*cap+slot) u.toNat r _).2 ((hb r hr).2.2 j hj))
        by_cases eq : r=q
        · subst r
          rw [vectorUpdate_self L q _ hq] at hi
          by_cases old : i<L[q]!
          · exact lift_old i old
          · have last : i=L[q]! := by omega
            subst i
            change TagAt cap t2 q slot
            exact tag_store_written cap t1 q slot (zigzag tag) hq hslot
        · rw [update_other L q r _ hr eq] at hi
          exact lift_old i hi

def TaggedReady (cap : Nat) (base s : St) : Prop := Ready cap base s ∧ TagBounds cap s

theorem tag_bounds_frame (cap : Nat) (s t : St) (h : TagBounds cap s)
    (hh : t.regs.get? iq_head=s.regs.get? iq_head)
    (hl : t.regs.get? iq_len=s.regs.get? iq_len) (hm : t.mem=s.mem) : TagBounds cap t := by
  obtain ⟨H,L,hH,hL,hb⟩ := h
  refine ⟨H,L,hh.trans hH,hl.trans hL,?_⟩
  simpa only [TagAt,Present,hm] using hb

theorem tag_bounds_other_write (cap : Nat) (s : St) (h : TagBounds cap s)
    (r : Register) (v : RegisterType r) (hh : r≠iq_head) (hl : r≠iq_len) :
    TagBounds cap {s with regs:=s.regs.insert r v} := by
  apply tag_bounds_frame cap s _ h
  · simp [Std.ExtDHashMap.get?_insert,hh,Ne.symm hh]
  · simp [Std.ExtDHashMap.get?_insert,hl,Ne.symm hl]
  · rfl

theorem tagged_read (cap : Nat) (base : St) (r : Register) :
    Total (TaggedReady cap base) (Out.readReg r) :=
  total_read _ r (fun s hs => hs.1.2.1 r)

theorem tagged_write (cap : Nat) (base : St) (r : Register) (v : RegisterType r)
    (hh : r≠iq_head) (hl : r≠iq_len) :
    Total (TaggedReady cap base) (Out.writeReg r v) := by
  intro s hs
  obtain ⟨u,t,he,ht⟩ := ready_write cap base r v hh hl s hs.1
  change EStateM.Result.ok () {s with regs:=s.regs.insert r v}=.ok u t at he
  cases he
  exact ⟨(),_,rfl,ht,tag_bounds_other_write cap s hs.2 r v hh hl⟩

theorem tagged_push (cap : Nat) (base : St) (q : Nat) (hq : q<145)
    (u : BitVec 32) (tag : Int) (ports : Proofs.Initialized.WritePorts cap) :
    Total (TaggedReady cap base) (pushC cap q u tag) := by
  intro s hs
  obtain ⟨b,t,he,hr⟩ := ready_push cap base q hq u tag ports s hs.1
  obtain ⟨b',t',he',tags⟩ := push_tag_bounds cap q hq u tag ports s hs.2
  rw [he] at he'
  cases he'
  exact ⟨b,t,he,hr,tags⟩

theorem tagged_overlay (cap : Nat) (base : St) (q w : Nat) (e : BitVec 10) :
    Total (TaggedReady cap base) (iq_word_ov_set q w e) := by
  intro s hs
  obtain ⟨u,t,he,hr⟩ := ready_overlay cap base q w e s hs.1
  have start : OverlayFrame s s := ⟨rfl,rfl,(by intros; rfl),hs.1.2.1 iq_wov⟩
  obtain ⟨u',t',he',frame⟩ := overlay_total s q w e s start
  rw [he] at he'
  cases he'
  exact ⟨u,t,he,hr,tag_bounds_frame cap s t hs.2
    (frame.2.2.1 iq_head (by decide) (by decide))
    (frame.2.2.1 iq_len (by decide) (by decide)) frame.1⟩

theorem tagged_delivery (cap rate q c : Nat) (hq : q<145) (base : St)
    (ports : Proofs.Initialized.WritePorts cap) :
    Total (TaggedReady cap base) (deliverC cap rate q c) := by
  unfold deliverC
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact tagged_push cap base q hq _ _ ports
    | exact tagged_overlay cap base _ _ _
    | exact tagged_read cap base _
    | apply tagged_write <;> decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

theorem tagged_image_len_bind {α : Type} (cap : Nat) (base : St) (q : Nat) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (f : Nat → SailM α) (hf : Total (TaggedReady cap base) (f (length q))) :
    Total (TaggedReady cap base) (iq_image_len q >>= f) := by
  intro s hs
  obtain ⟨v,t,he,ht⟩ := hf s hs
  have read : iq_image_len q s=.ok (length q) s := values.len q s hq
  refine ⟨v,t,?_,ht⟩
  simpa [EStateM.run,bind,EStateM.bind,read] using he

theorem tagged_image_unit (cap : Nat) (base : St) (q i : Nat) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (hi : i<length q) : Total (TaggedReady cap base) (iq_image_unit q i) := by
  intro s hs
  exact ⟨image q i,s,values.unit q i s hq hi,hs⟩

theorem tagged_reset_tail (cap : Nat) (base : St)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length) :
    Total (TaggedReady cap base) (resetTailC cap) := by
  unfold resetTailC nativeImageLoop imageStep
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact tagged_push cap base _ (by simp only [Membership.mem] at *; simp at *; omega) _ _ ports
    | with_reducible apply tagged_image_len_bind cap base _ (by simp only [Membership.mem] at *; simp at *; omega) image length values
    | exact tagged_image_unit cap base _ _ (by simp only [Membership.mem] at *; simp at *; omega)
        image length values (by simp only [Membership.mem] at *; simp at *; omega)
    | exact tagged_read cap base _
    | apply tagged_write <;> decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

theorem tagged_reset (cap : Nat) (hcap : 0<cap) (base s : St) (view : CView)
    (rep : Represents cap view s) (complete : MachineInitProofs.Complete s) (foreign : ForeignFrame base s)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length) :
    ∃ after, (resetC cap).run s=.ok () after ∧ TaggedReady cap base after := by
  let initial : St := {s with regs := ((s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145))}
  have good : Proofs.StorageProduct.Good cap initial :=
    ⟨view,rep,Proofs.Initialized.zero_bounds cap hcap s⟩
  have cp : MachineInitProofs.Complete initial :=
    MachineInitProofs.complete_insert _ Register.iq_len _ (MachineInitProofs.complete_insert s Register.iq_head _ complete)
  have ready : TaggedReady cap base initial := ⟨⟨good,cp,foreign⟩,zero_tag_bounds cap hcap s⟩
  obtain ⟨v,after,he,ha⟩ := tagged_reset_tail cap base ports image length values initial ready
  refine ⟨after,?_,ha⟩
  have he' : resetTailC cap initial=.ok () after := he
  simpa [resetC,Out.writeReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,
    modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,initial,he'] using he

/-- The exact ordered image reset and cross-resource/typed-register invariant hold
    for the SAME execution and SAME initial concrete store witness. -/
theorem tagged_reset_exact (cap : Nat) (hcap : 0<cap) (base s : St) (view : CView)
    (rep : Represents cap view s) (complete : MachineInitProofs.Complete s) (foreign : ForeignFrame base s)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length)
    (admit : ∀ q, q<145 → length q≤cap) :
    ∃ after, (resetC cap).run s=.ok () after ∧ TaggedReady cap base after ∧
      OuterInv cap view image length (initialIQState s) 145 after ∧
      ∀ q, q<145 → content cap (resetStore cap view image length 145) q 0
        (prefixPairs (image q) (length q)).length=prefixPairs (image q) (length q) := by
  obtain ⟨after,he,hr⟩ := tagged_reset cap hcap base s view rep complete foreign ports image length values
  obtain ⟨other,he',ho,hc⟩ := reset_exact cap hcap image length admit values ports view s rep
  rw [he] at he'
  cases he'
  exact ⟨after,he,hr,ho,hc⟩


theorem pop_tag_bounds (cap q n : Nat) (hq : q<145) (s : St) (hs : TagBounds cap s)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head = some H)
    (hL : s.regs.get? Register.iq_len = some L) (hn : n≤L[q]!) :
    ∃ after, (popC cap q n).run s = .ok () after ∧ TagBounds cap after := by
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
        change TagAt cap s q (ringC cap (ringC cap (H[q]!+n) + i))
        rw [Proofs.Storage.ring_shift cap H[q]! L[q]! n i (hb q hq).1 (hb q hq).2.1 hn hi]
        simpa [Nat.add_assoc] using (hb q hq).2.2 (n + i) (by omega)
    · rw [update_other H q r _ hr eq,update_other L q r _ hr eq]
      exact hb r hr


/-- Physical allocation is a separate trusted C projection. Logical total tags
    alone do not establish a GMP array. The exact C tag-write allocator realizes
    this relation, retained across repeated logical reset and paired stores. -/
def Allocation (cap : Nat) (allocated : Nat → Prop) (s : St) : Prop :=
  ∀ q, q<145 → (allocated q ↔ ∃ slot, slot<cap ∧ TagAt cap s q slot)

def allocate (allocated : Nat → Prop) (q : Nat) : Nat → Prop :=
  fun r => r=q ∨ allocated r

theorem unit_keeps_allocation (cap : Nat) (allocated : Nat → Prop) (s : St)
    (h : Allocation cap allocated s) (slot value : Nat) :
    Allocation cap allocated (writeCell s (145*cap) 0 slot value) := by
  intro q hq
  simpa only [tag_unit_cross] using h q hq

theorem tag_store_other (cap : Nat) (s : St) (q slot value r j : Nat)
    (hq : q<145) (hs : slot<cap) (hr : r<145) (hj : j<cap) (different : q≠r) :
    TagAt cap (writeCell s (145*cap) 1 (q*cap+slot) value) r j ↔ TagAt cap s r j := by
  have diff : q*cap+slot≠r*cap+j := by
    intro eq
    exact different (flat_injective cap q r slot j hs hj eq).1
  unfold TagAt
  apply exists_congr
  intro v
  rw [present_write s (145*cap) 1 1 (q*cap+slot) (r*cap+j) value v
    (by decide) (by decide) (slot_bound cap q slot hq hs) (slot_bound cap r j hr hj)]
  simp [diff]

theorem tag_store_allocates (cap : Nat) (allocated : Nat → Prop) (s : St)
    (h : Allocation cap allocated s) (q slot value : Nat) (hq : q<145) (hs : slot<cap) :
    Allocation cap (allocate allocated q) (writeCell s (145*cap) 1 (q*cap+slot) value) := by
  intro r hr
  by_cases eq : r=q
  · subst r
    constructor
    · intro _
      exact ⟨slot,hs,tag_store_written cap s q slot value hq hs⟩
    · intro _
      exact Or.inl rfl
  · have diff : q≠r := Ne.symm eq
    have same : (∃ j, j<cap ∧ TagAt cap (writeCell s (145*cap) 1 (q*cap+slot) value) r j) ↔
        ∃ j, j<cap ∧ TagAt cap s r j := by
      constructor
      · rintro ⟨j,hj,hv⟩
        exact ⟨j,hj,(tag_store_other cap s q slot value r j hq hs hr hj diff).1 hv⟩
      · rintro ⟨j,hj,hv⟩
        exact ⟨j,hj,(tag_store_other cap s q slot value r j hq hs hr hj diff).2 hv⟩
    simpa [Allocation,allocate,eq] using (h r hr).trans same.symm

theorem occupied_queue_allocated (cap : Nat) (allocated : Nat → Prop) (s : St)
    (tags : TagBounds cap s) (physical : Allocation cap allocated s)
    (L : Vector Nat 145) (hL : s.regs.get? iq_len=some L)
    (q : Nat) (hq : q<145) (occupied : 0<L[q]!) : allocated q := by
  obtain ⟨H,L0,hH,hL0,hb⟩ := tags
  have same : L0=L := by simpa [hL] using hL0.symm
  subst L0
  have ht := (hb q hq).2.2 0 occupied
  have hh := (hb q hq).1
  apply (physical q hq).2
  refine ⟨ringC cap (H[q]!+0),?_,ht⟩
  rw [Proofs.Capacity.ring_spec]
  split <;> omega

theorem ring_agrees_with_c_modulo (cap head index : Nat) (hh : head<cap) (hi : index<cap) :
    ringC cap (head+ index)=(head+ index)%cap := by
  rw [Proofs.Capacity.ring_spec]
  by_cases full : cap≤head+ index
  · simp only [full,ite_true]
    rw [Nat.mod_eq_sub_mod full,Nat.mod_eq_of_lt (by omega : head+ index-cap<cap)]
  · simp only [full,ite_false]
    rw [Nat.mod_eq_of_lt (by omega : head+ index<cap)]

/-- Admitted iq_depth<=4096 means capacity<=1024; native uint64 head+i cannot
    wrap before the C modulo. This is existing profile admission, not a new cap. -/
theorem native_observer_index_no_wrap (cap head count index : Nat)
    (admitted : cap≤1024) (hh : head<cap) (hc : count≤cap) (hi : index<count) :
    head+ index<2048 ∧ head+ index<18446744073709551616 := by omega

theorem tagged_delivery_prefix (cap rate q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (base s : St) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145)
    (seen : Seen cap view H L F s) (complete : MachineInitProofs.Complete s)
    (foreign : ForeignFrame base s) (tags : TagBounds cap s) :
    let result := finishPlan (planLoop cap L[q]! c (rate+list_len F[q]!+1) (startPlan rate F[q]!))
    ∃ after, (deliverC cap rate q c).run s=.ok (result.pushed,result.lost) after ∧
      TaggedReady cap base after ∧
      Seen cap (historyStore cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
      Prefix (cap-L[q]!) result := by
  dsimp only
  have start : TaggedReady cap base s := ⟨⟨⟨view,seen.rep,seen.bounds⟩,complete,foreign⟩,tags⟩
  obtain ⟨v,after,he,ht⟩ := tagged_delivery cap rate q c hq base ports s start
  obtain ⟨other,he',hr,hs,hp⟩ := delivery_prefix_ready cap rate q c hq ports base s view H L F seen complete foreign
  rw [he] at he'
  cases he'
  exact ⟨after,he,ht,hs,hp⟩

theorem tagged_fetch (cap q t : Nat) (hq : q<145) (base : St)
    (a b : Proofs.OrderedFetch.Vec) (ova ovb : Proofs.OrderedFetch.Overlay)
    (observer : Proofs.OrderedFetch.FetchObserver) :
    Total (TaggedReady cap base) (iq_fetch_ov q t a b ova ovb) := by
  intro s hs
  obtain ⟨F,hF⟩ := hs.1.2.1 fill_fifo
  obtain ⟨P,hP⟩ := hs.1.2.1 iq_prod
  by_cases full : iq_staging_depth≤F[q]!.length
  · exact ⟨false,s,Proofs.OrderedFetch.fetch_full_no_effect q t a b ova ovb s F hF full,hs⟩
  · obtain ⟨observed,he,hr,hm,fetch⟩ := Proofs.OrderedFetch.fetch_room_exact q t hq a b ova ovb observer s F P hF hP (by omega)
    have observed_ready := Proofs.OrderedFetch.ready_same_parts cap base s observed hs.1 hr hm
    have observed_tags := tag_bounds_frame cap s observed hs.2 (by rw [hr]) (by rw [hr]) hm
    have prod_tags := tag_bounds_other_write cap observed observed_tags iq_prod
      (vectorUpdate P q (P[q]!+8*Out.Functions.superlanes)) (by decide) (by decide)
    have final_tags := tag_bounds_other_write cap _ prod_tags fill_fifo
      (vectorUpdate F q (F[q]!++[Proofs.OrderedFetch.fetchedSegment t (P[q]!:Int) a b ova ovb])) (by decide) (by decide)
    exact ⟨true,_,fetch,Proofs.OrderedFetch.fetched_state_ready cap base observed observed_ready F P q t a b ova ovb,final_tags⟩

theorem full_reset_tagged (hcap : 0<iq_cap) (base s : St) (view : CView)
    (rep : Represents iq_cap view s) (complete : MachineInitProofs.Complete s)
    (foreign : ForeignFrame base s) (ports : Proofs.Initialized.WritePorts iq_cap)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (admit : ∀ q, q<145 → length q≤iq_cap) :
    ∃ after middle, (Out.Functions.reset ()).run s=.ok () after ∧ TaggedReady iq_cap base after ∧
      machineResetPrefix.run s=.ok () middle ∧
      OuterInv iq_cap view image length (initialIQState middle) 145 after := by
  obtain ⟨v,middle,he,hcomplete⟩ := machine_prefix_complete s complete
  obtain ⟨other,he',hmem⟩ := machine_prefix_memory s
  rw [he] at he'
  cases he'
  have rep' := represents_memory iq_cap view s middle hmem rep
  have foreign' : ForeignFrame base middle := by
    intro key hk
    exact (congrArg (fun mem => mem.get? key) hmem).trans (foreign key hk)
  obtain ⟨after,hr,ready,inv,contents⟩ := tagged_reset_exact iq_cap hcap base middle view
    rep' hcomplete foreign' ports image length values admit
  refine ⟨after,middle,?_,ready,he,inv⟩
  rw [machine_reset_split,reset_production]
  change machineResetPrefix s=.ok () middle at he
  change resetC iq_cap middle=.ok () after at hr
  simpa [EStateM.run,bind,EStateM.bind,he] using hr

theorem initialized_full_reset_tagged (hcap : 0<iq_cap) (s : St) (view : CView)
    (rep : Represents iq_cap view s) (ports : Proofs.Initialized.WritePorts iq_cap)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (admit : ∀ q, q<145 → length q≤iq_cap) :
    ∃ initialized after, (sail_model_init ()).run s=.ok () initialized ∧
      (Out.Functions.reset ()).run initialized=.ok () after ∧ TaggedReady iq_cap s after := by
  have foreign : ForeignFrame s s := fun _ _ => rfl
  obtain ⟨initialized,hi,rep',complete,foreign'⟩ := actual_init_rep_complete iq_cap s s view rep foreign
  obtain ⟨after,middle,hr,ready,hp,inv⟩ := full_reset_tagged hcap s initialized view
    rep' complete foreign' ports image length values admit
  exact ⟨initialized,after,hi,hr,ready⟩

theorem default_tag_value_is_not_tag_presence (cap q slot : Nat) (s : St) :
    tags cap {s with mem:=∅} q slot = -1 ∧ ¬TagAt cap {s with mem:=∅} q slot := by
  constructor
  · simp [tags,empty_read]
  · simp [TagAt,Present]

theorem tag_bounds_unit_write (cap : Nat) (s : St) (tags : TagBounds cap s)
    (slot value : Nat) : TagBounds cap (writeCell s (145*cap) 0 slot value) := by
  obtain ⟨H,L,hH,hL,hb⟩ := tags
  refine ⟨H,L,hH,hL,?_⟩
  intro q hq
  refine ⟨(hb q hq).1,(hb q hq).2.1,?_⟩
  intro i hi
  exact (tag_unit_cross cap s slot value q _).2 ((hb q hq).2.2 i hi)

theorem allocation_same_memory (cap : Nat) (allocated : Nat → Prop) (s t : St)
    (h : Allocation cap allocated s) (hm : t.mem=s.mem) : Allocation cap allocated t := by
  simpa only [Allocation,TagAt,Present,hm] using h

end Proofs.TagCoverage
