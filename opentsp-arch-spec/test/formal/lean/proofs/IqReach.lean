import Proofs.MachineInit
import Proofs.OrderedDelivery
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep
open Proofs.OrderedImage Proofs.OrderedDelivery
set_option autoImplicit false
set_option maxRecDepth 4000
namespace Proofs.IqReach

/-- The IQ interpretation occupies fields0/1. Resource SRAM data/ECC may use
    fields2/3 of the SAME mem map, with its own counts and bounds. -/
def ForeignFrame (before after : St) : Prop :=
  ∀ key, 2≤key%4 → after.mem.get? key=before.mem.get? key

theorem writeCell_other_key (s : St) (count field slot value key : Nat)
    (hf : field<2) (hk : 2≤key%4) :
    (writeCell s count field slot value).mem.get? key=s.mem.get? key := by
  have ne : cellKey count field slot value≠key := by
    intro eq
    have h := key_field count field slot value hf
    rw [eq] at h
    omega
  have nf : ¬key%4=field := by omega
  simp only [writeCell,Std.ExtHashMap.get?_eq_getElem?,
    Std.ExtHashMap.getElem?_insert,Std.ExtHashMap.getElem?_filter']
  simp [ne,Ne.symm ne,nf]
  cases s.mem[key]? <;> rfl

theorem writeCell_foreign (base s : St) (h : ForeignFrame base s)
    (count field slot value : Nat) (hf : field<2) :
    ForeignFrame base (writeCell s count field slot value) := by
  intro key hk
  exact (writeCell_other_key s count field slot value key hf hk).trans (h key hk)

def Ready (cap : Nat) (base s : St) : Prop :=
  Proofs.StorageProduct.Good cap s ∧ MachineInitProofs.Complete s ∧ ForeignFrame base s

theorem ready_read (cap : Nat) (base : St) (r : Register) :
    Total (Ready cap base) (Out.readReg r) :=
  total_read _ r (fun s hs => hs.2.1 r)

theorem ready_write (cap : Nat) (base : St) (r : Register) (v : RegisterType r)
    (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) :
    Total (Ready cap base) (Out.writeReg r v) := by
  apply total_write
  intro s hs
  refine ⟨?_,MachineInitProofs.complete_insert s r v hs.2.1,hs.2.2⟩
  have good := Proofs.StorageProduct.preserves_other_write (cap:=cap) r v hh hl s hs.1
  simpa [Out.writeReg,PreSail.writeReg,EStateM.run,modify,modifyGet,
    MonadStateOf.modifyGet,EStateM.modifyGet] using good

theorem overlay_complete (q w : Nat) (e : BitVec 10) :
    Total MachineInitProofs.Complete (iq_word_ov_set q w e) := by
  unfold iq_word_ov_set
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact total_read _ _ (fun s hs => hs _)
    | apply total_write <;> exact fun s hs => MachineInitProofs.complete_insert s _ _ hs
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

theorem ready_overlay (cap : Nat) (base : St) (q w : Nat) (e : BitVec 10) :
    Total (Ready cap base) (iq_word_ov_set q w e) := by
  intro s hs
  obtain ⟨v,t,he,hcomplete⟩ := overlay_complete q w e s hs.2.1
  have start : OverlayFrame s s := ⟨rfl,rfl,(by intros; rfl),hs.2.1 Register.iq_wov⟩
  obtain ⟨u,t',he',frame⟩ := overlay_total s q w e s start
  rw [he] at he'
  cases he'
  have good := Proofs.StorageProduct.overlay_safe (cap:=cap) q w e s hs.1
  simp only [he] at good
  refine ⟨v,t,he,good,hcomplete,?_⟩
  intro key hk
  exact (congrArg (fun mem => mem.get? key) frame.1).trans (hs.2.2 key hk)

theorem ready_push (cap : Nat) (base : St) (q : Nat) (hq : q<145)
    (u : BitVec 32) (tag : Int) (ports : Proofs.Initialized.WritePorts cap) :
    Total (Ready cap base) (pushC cap q u tag) := by
  intro s hs
  obtain ⟨view,rep,bounds⟩ := hs.1
  obtain ⟨H,L,hH,hL,hb⟩ := bounds
  by_cases full : cap≤L[q]!
  · exact ⟨false,s,Proofs.Capacity.iq_push_full q u tag s L hL full,hs⟩
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
    have e1 : iq_unit_write q slot u s=.ok () t1 := ports.unit q slot u s hq hslot
    have e2 : iq_tag_write q slot tag t1=.ok () t2 := ports.tag q slot tag t1 hq hslot
    have r1 : t1.regs=s.regs := rfl
    have r2 : t2.regs=s.regs := rfl
    have hc : ((H[q]! : Int)+(L[q]! : Int)).toNat=H[q]!+L[q]! := by omega
    have run : (pushC cap q u tag).run s=.ok true after := by
      simp [pushC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,hL,hH,
        EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,
        EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,full,hc,
        slot,e1,e2,r1,r2,after]
    have good := Proofs.StorageProduct.push_safe (cap:=cap) q hq u tag ports s hs.1
    simp only [run] at good
    have complete : MachineInitProofs.Complete after := by
      simpa only [after,t2,t1,writeCell] using
        MachineInitProofs.complete_insert s Register.iq_len (vectorUpdate L q (L[q]!+1)) hs.2.1
    have foreign : ForeignFrame base after :=
      writeCell_foreign base t1 (writeCell_foreign base s hs.2.2 _ _ _ _ (by decide)) _ _ _ _ (by decide)
    exact ⟨true,after,run,good,complete,foreign⟩

/-- Complete actual delivery preserves ALL typed-register presence and non-IQ
    memory namespaces; it does not assume representation implies completeness. -/
theorem delivery_ready (cap rate q c : Nat) (hq : q<145) (base : St)
    (ports : Proofs.Initialized.WritePorts cap) :
    Total (Ready cap base) (deliverC cap rate q c) := by
  unfold deliverC
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact ready_push cap base q hq _ _ ports
    | exact ready_overlay cap base _ _ _
    | exact ready_read cap base _
    | apply ready_write <;> decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x


theorem image_len_bind_ready {α : Type} (cap : Nat) (base : St) (q : Nat) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (f : Nat → SailM α) (hf : Total (Ready cap base) (f (length q))) :
    Total (Ready cap base) (iq_image_len q >>= f) := by
  intro s hs
  obtain ⟨v,t,he,ht⟩ := hf s hs
  have read : iq_image_len q s=.ok (length q) s := values.len q s hq
  refine ⟨v,t,?_,ht⟩
  simpa [EStateM.run,bind,EStateM.bind,read] using he

theorem image_unit_ready (cap : Nat) (base : St) (q i : Nat) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (hi : i<length q) : Total (Ready cap base) (iq_image_unit q i) := by
  intro s hs
  exact ⟨image q i,s,values.unit q i s hq hi,hs⟩

theorem reset_tail_ready (cap : Nat) (base : St)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length) :
    Total (Ready cap base) (resetTailC cap) := by
  unfold resetTailC nativeImageLoop imageStep
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact ready_push cap base _ (by simp only [Membership.mem] at *; simp at *; omega) _ _ ports
    | with_reducible apply image_len_bind_ready cap base _ (by simp only [Membership.mem] at *; simp at *; omega) image length values
    | exact image_unit_ready cap base _ _ (by simp only [Membership.mem] at *; simp at *; omega)
        image length values (by simp only [Membership.mem] at *; simp at *; omega)
    | exact ready_read cap base _
    | apply ready_write <;> decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

theorem reset_ready (cap : Nat) (hcap : 0<cap) (base s : St) (view : CView)
    (rep : Represents cap view s) (complete : MachineInitProofs.Complete s) (foreign : ForeignFrame base s)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length) :
    ∃ after, (resetC cap).run s=.ok () after ∧ Ready cap base after := by
  let initial : St := {s with regs := ((s.regs.insert Register.iq_head (vectorInit 0 : Vector Nat 145)).insert Register.iq_len (vectorInit 0 : Vector Nat 145))}
  have good : Proofs.StorageProduct.Good cap initial :=
    ⟨view,rep,Proofs.Initialized.zero_bounds cap hcap s⟩
  have cp : MachineInitProofs.Complete initial :=
    MachineInitProofs.complete_insert _ Register.iq_len _ (MachineInitProofs.complete_insert s Register.iq_head _ complete)
  have ready : Ready cap base initial := ⟨good,cp,foreign⟩
  obtain ⟨v,after,he,ha⟩ := reset_tail_ready cap base ports image length values initial ready
  refine ⟨after,?_,ha⟩
  have he' : resetTailC cap initial=.ok () after := he
  simpa [resetC,Out.writeReg,PreSail.writeReg,EStateM.run,bind,EStateM.bind,
    modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,initial,he'] using he

/-- The exact ordered image reset and cross-resource/typed-register invariant hold
    for the SAME execution and SAME initial concrete store witness. -/
theorem reset_exact_ready (cap : Nat) (hcap : 0<cap) (base s : St) (view : CView)
    (rep : Represents cap view s) (complete : MachineInitProofs.Complete s) (foreign : ForeignFrame base s)
    (ports : Proofs.Initialized.WritePorts cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (values : ImageValues image length)
    (admit : ∀ q, q<145 → length q≤cap) :
    ∃ after, (resetC cap).run s=.ok () after ∧ Ready cap base after ∧
      OuterInv cap view image length (initialIQState s) 145 after ∧
      ∀ q, q<145 → content cap (resetStore cap view image length 145) q 0
        (prefixPairs (image q) (length q)).length=prefixPairs (image q) (length q) := by
  obtain ⟨after,he,hr⟩ := reset_ready cap hcap base s view rep complete foreign ports image length values
  obtain ⟨other,he',ho,hc⟩ := reset_exact cap hcap image length admit values ports view s rep
  rw [he] at he'
  cases he'
  exact ⟨after,he,hr,ho,hc⟩

/-- Lifting whole ordered delivery adds no new storage witness or memory clearing. -/
theorem delivery_prefix_ready (cap rate q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (base s : St) (view : CView)
    (H L : Vector Nat 145) (F : Vector (List fill_seg) 145)
    (seen : Seen cap view H L F s) (complete : MachineInitProofs.Complete s)
    (foreign : ForeignFrame base s) :
    let result := finishPlan (planLoop cap L[q]! c (rate+list_len F[q]!+1) (startPlan rate F[q]!))
    ∃ after, (deliverC cap rate q c).run s=.ok (result.pushed,result.lost) after ∧
      Ready cap base after ∧
      Seen cap (historyStore cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate F q result.fifo) after ∧
      Prefix (cap-L[q]!) result := by
  dsimp only
  have start : Ready cap base s := ⟨⟨view,seen.rep,seen.bounds⟩,complete,foreign⟩
  obtain ⟨v,after,he,hr⟩ := delivery_ready cap rate q c hq base ports s start
  obtain ⟨other,he',hs,hp,hc⟩ := delivery_prefix_exact cap rate q c hq ports view H L F s seen
  rw [he] at he'
  cases he'
  exact ⟨after,he,hr,hs,hp⟩


theorem represents_memory (cap : Nat) (view : CView) (s t : St) (hmem : t.mem=s.mem)
    (rep : Represents cap view s) : Represents cap view t := by
  simpa only [Represents,Consistent,Present,Initialized,readValue,tags,hmem] using rep

theorem actual_init_memory (s : St) :
    ∃ after, (sail_model_init ()).run s=.ok () after ∧ after.mem=s.mem := by
  exact ⟨_,rfl,rfl⟩

theorem actual_init_rep_complete (cap : Nat) (base s : St) (view : CView)
    (rep : Represents cap view s) (foreign : ForeignFrame base s) :
    ∃ after, (sail_model_init ()).run s=.ok () after ∧
      Represents cap view after ∧ MachineInitProofs.Complete after ∧ ForeignFrame base after := by
  obtain ⟨after,he,hcomplete⟩ := MachineInitProofs.actual_init_complete s
  obtain ⟨other,he',hmem⟩ := actual_init_memory s
  rw [he] at he'
  cases he'
  refine ⟨after,he,represents_memory cap view s after hmem rep,hcomplete,?_⟩
  intro key hk
  exact (congrArg (fun mem => mem.get? key) hmem).trans (foreign key hk)

open Out.Register
def machineResetPrefix : SailM Unit := do
  (srf_reset ())
  (mxm_reset ())
  (sxm_reset ())
  Out.writeReg pending (vectorInit [])
  Out.writeReg execute_ring (vectorInit [])
  Out.writeReg queue_busy_until (vectorInit 0)
  Out.writeReg path_busy_until (vectorInit 0)
  Out.writeReg local_fetch_bank (vectorInit none)
  Out.writeReg local_fetches (vectorInit none)
  Out.writeReg local_fetch_busy (vectorInit 0)
  Out.writeReg stream_fetches (vectorInit none)
  Out.writeReg path_tail_until (vectorInit 0)
  Out.writeReg queue_last_at (vectorInit 0)
  Out.writeReg agts (vectorInit agt_idle)
  Out.writeReg agts_w (vectorInit agt_idle)
  Out.writeReg agt_records (vectorInit 0)
  (mem_reset ())
  (masks_reset ())
  Out.writeReg parked_since (vectorInit none)
  Out.writeReg release_at (vectorInit none)
  Out.writeReg last_notify none
  Out.writeReg queue_last (vectorInit none)
  Out.writeReg queue_repeat (vectorInit none)
  Out.writeReg queue_poll (vectorInit none)
  Out.writeReg last_tick 0
  Out.writeReg scrub_next 0
  Out.writeReg scrub_ptr (vectorInit 0)
  Out.writeReg scrub_pending (vectorInit none)
  Out.writeReg sram_revision (vectorInit 0)
  (c2c_reset ())
  (host_reset ())


theorem machine_reset_split :
    Out.Functions.reset () = (do machineResetPrefix; iq_reset ()) := by
  simp only [Out.Functions.reset,machineResetPrefix,bind_assoc]

theorem machine_prefix_memory (s : St) :
    ∃ after, machineResetPrefix.run s=.ok () after ∧ after.mem=s.mem := by
  exact ⟨_,rfl,rfl⟩

theorem machine_prefix_complete :
    Total MachineInitProofs.Complete machineResetPrefix := by
  intro s hs
  refine ⟨(),_,rfl,?_⟩
  repeat' apply MachineInitProofs.complete_insert
  exact hs


/-- The actual full reset prefix is retained; IQ identity does not replace SRF,
    MXM, masks, host, link or scheduler initialization with a logger. -/
theorem machine_reset_exact_ready (hcap : 0<iq_cap) (base s : St) (view : CView)
    (rep : Represents iq_cap view s) (complete : MachineInitProofs.Complete s)
    (foreign : ForeignFrame base s) (ports : Proofs.Initialized.WritePorts iq_cap)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (admit : ∀ q, q<145 → length q≤iq_cap) :
    ∃ after middle, (Out.Functions.reset ()).run s=.ok () after ∧ Ready iq_cap base after ∧
      machineResetPrefix.run s=.ok () middle ∧
      OuterInv iq_cap view image length (initialIQState middle) 145 after ∧
      ∀ q, q<145 → content iq_cap (resetStore iq_cap view image length 145) q 0
        (prefixPairs (image q) (length q)).length=prefixPairs (image q) (length q) := by
  obtain ⟨v,middle,he,hcomplete⟩ := machine_prefix_complete s complete
  obtain ⟨other,he',hmem⟩ := machine_prefix_memory s
  rw [he] at he'
  cases he'
  have rep' := represents_memory iq_cap view s middle hmem rep
  have foreign' : ForeignFrame base middle := by
    intro key hk
    exact (congrArg (fun mem => mem.get? key) hmem).trans (foreign key hk)
  obtain ⟨after,hr,ready,inv,contents⟩ := reset_exact_ready iq_cap hcap base middle view
    rep' hcomplete foreign' ports image length values admit
  refine ⟨after,middle,?_,ready,he,inv,contents⟩
  rw [machine_reset_split,reset_production]
  change machineResetPrefix s=.ok () middle at he
  change resetC iq_cap middle=.ok () after at hr
  simpa [EStateM.run,bind,EStateM.bind,he] using hr

theorem initialized_machine_reset (hcap : 0<iq_cap) (s : St) (view : CView)
    (rep : Represents iq_cap view s) (ports : Proofs.Initialized.WritePorts iq_cap)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (values : ImageValues image length)
    (admit : ∀ q, q<145 → length q≤iq_cap) :
    ∃ initialized after middle,
      (sail_model_init ()).run s=.ok () initialized ∧
      (Out.Functions.reset ()).run initialized=.ok () after ∧ Ready iq_cap s after ∧
      machineResetPrefix.run initialized=.ok () middle ∧
      OuterInv iq_cap view image length (initialIQState middle) 145 after ∧
      ∀ q, q<145 → content iq_cap (resetStore iq_cap view image length 145) q 0
        (prefixPairs (image q) (length q)).length=prefixPairs (image q) (length q) := by
  have foreign : ForeignFrame s s := fun _ _ => rfl
  obtain ⟨initialized,hi,rep',complete,foreign'⟩ := actual_init_rep_complete iq_cap s s view rep foreign
  obtain ⟨after,middle,hr,ready,hp,inv,contents⟩ := machine_reset_exact_ready hcap s initialized view
    rep' complete foreign' ports image length values admit
  exact ⟨initialized,after,middle,hi,hr,ready,hp,inv,contents⟩


def iqResetRegs : List Register :=
  [iq_head,iq_len,iq_last,iq_dry,iq_active,fill_fifo,iq_prod,iq_wov,iq_ecc_live]

theorem reset_other_registers (cap : Nat) (view : CView) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (before after : St)
    (inv : OuterInv cap view image length (initialIQState before) 145 after)
    (r : Register) (outside : r∉iqResetRegs) : after.regs.get? r=before.regs.get? r := by
  simp [iqResetRegs] at outside
  rcases outside with ⟨hh,hl,ht,hd,ha,hf,hp,hw,he⟩
  rw [inv.frame r hl hp]
  simp [initialIQState,setReg,Std.ExtDHashMap.get?_insert,
    hh,hl,ht,hd,ha,hf,hp,hw,he,Ne.symm hh,Ne.symm hl,Ne.symm ht,Ne.symm hd,
    Ne.symm ha,Ne.symm hf,Ne.symm hp,Ne.symm hw,Ne.symm he]

def Keeps {α : Type} (r : Register) (before : St) (action : SailM α) : Prop :=
  match action.run before with
  | .ok _ after => after.regs.get? r=before.regs.get? r
  | .error _ after => after.regs.get? r=before.regs.get? r

theorem push_keeps (cap q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts cap) (s : St) (good : Proofs.StorageProduct.Good cap s)
    (r : Register) (hr : r≠iq_len) : Keeps r s (pushC cap q u tag) := by
  obtain ⟨view,rep,bounds⟩ := good
  obtain ⟨H,L,hH,hL,hb⟩ := bounds
  by_cases full : cap≤L[q]!
  · have run := Proofs.Capacity.iq_push_full q u tag s L hL full
    simp only [Keeps,run]
  · obtain ⟨after,run,_,_,regs,_⟩ := push_room_exact cap q hq u tag ports s view rep
      ⟨H,L,hH,hL,hb⟩ H L hH hL (hb q hq).1 (by omega)
    simp [Keeps,run,regs,Std.ExtDHashMap.get?_insert,hr,Ne.symm hr]

def RReady (cap : Nat) (base original : St) (r : Register) (s : St) : Prop :=
  Ready cap base s ∧ s.regs.get? r=original.regs.get? r

theorem rready_read (cap : Nat) (base original : St) (r dst : Register) :
    Total (RReady cap base original r) (Out.readReg dst) :=
  total_read _ dst (fun s hs => hs.1.2.1 dst)

theorem rready_write (cap : Nat) (base original : St) (r dst : Register) (v : RegisterType dst)
    (hh : dst≠iq_head) (hl : dst≠iq_len) (hr : r≠dst) :
    Total (RReady cap base original r) (Out.writeReg dst v) := by
  apply total_write
  intro s hs
  obtain ⟨value,t,he,ht⟩ := ready_write cap base dst v hh hl s hs.1
  have actual : (Out.writeReg dst v).run s=.ok () {s with regs:=s.regs.insert dst v} := rfl
  rw [actual] at he
  cases he
  refine ⟨ht,?_⟩
  simpa [Std.ExtDHashMap.get?_insert,hr,Ne.symm hr] using hs.2

theorem rready_push (cap : Nat) (base original : St) (r : Register) (hr : r≠iq_len)
    (q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int) (ports : Proofs.Initialized.WritePorts cap) :
    Total (RReady cap base original r) (pushC cap q u tag) := by
  intro s hs
  obtain ⟨value,t,he,ht⟩ := ready_push cap base q hq u tag ports s hs.1
  have hk := push_keeps cap q hq u tag ports s hs.1.1 r hr
  simp only [Keeps,he] at hk
  exact ⟨value,t,he,ht,hk.trans hs.2⟩

theorem rready_overlay (cap : Nat) (base original : St) (r : Register)
    (hw : r≠iq_wov) (hecc : r≠iq_ecc_live) (q w : Nat) (e : BitVec 10) :
    Total (RReady cap base original r) (iq_word_ov_set q w e) := by
  intro s hs
  obtain ⟨value,t,he,ht⟩ := ready_overlay cap base q w e s hs.1
  have start : OverlayFrame s s := ⟨rfl,rfl,(by intros; rfl),hs.1.2.1 iq_wov⟩
  obtain ⟨v,t',he',frame⟩ := overlay_total s q w e s start
  rw [he] at he'
  cases he'
  exact ⟨value,t,he,ht,(frame.2.2.1 r hw hecc).trans hs.2⟩

def iqDeliveryRegs : List Register := [iq_len,fill_fifo,iq_active,iq_wov,iq_ecc_live]

theorem delivery_rready (cap rate q c : Nat) (hq : q<145)
    (base original : St) (r : Register) (outside : r∉iqDeliveryRegs)
    (ports : Proofs.Initialized.WritePorts cap) :
    Total (RReady cap base original r) (deliverC cap rate q c) := by
  simp [iqDeliveryRegs] at outside
  rcases outside with ⟨hl,hf,ha,hw,he⟩
  unfold deliverC
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact rready_push cap base original r hl q hq _ _ ports
    | exact rready_overlay cap base original r hw he _ _ _
    | exact rready_read cap base original r _
    | apply rready_write <;> first | assumption | decide
    | with_reducible apply total_pure
    | with_reducible apply total_ite
    | with_reducible apply total_forIn
    | with_reducible apply total_bind
    | split
    | intro x

/-- Covers both result constructors. Under the proved reachable Ready invariant the
    real action succeeds; absent-register/errors are not silently admitted as C runs. -/
theorem delivery_other_registers (cap rate q c : Nat) (hq : q<145)
    (ports : Proofs.Initialized.WritePorts cap) (base s : St) (ready : Ready cap base s)
    (r : Register) (outside : r∉iqDeliveryRegs) : Keeps r s (deliverC cap rate q c) := by
  have start : RReady cap base s r s := ⟨ready,rfl⟩
  obtain ⟨v,t,he,ht⟩ := delivery_rready cap rate q c hq base s r outside ports s start
  simpa only [Keeps,he] using ht.2

end Proofs.IqReach
