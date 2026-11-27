import Proofs.IqReach
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep
open Proofs.OrderedImage Proofs.OrderedDelivery Proofs.IqReach
set_option autoImplicit false
set_option maxRecDepth 8000
set_option maxHeartbeats 800000
open Out.Register
namespace Proofs.OrderedFetch

abbrev Vec := BitVec (8*(16*Out.Functions.superlanes))
abbrev Overlay := BitVec (10*Out.Functions.superlanes)

def words (v : Vec) : List (BitVec 32) :=
  (List.range (4*Out.Functions.superlanes)).map (fun k => BitVec.slice v (32*k) 32)

def overlays (ova ovb : Overlay) : List (BitVec 10) :=
  (List.range Out.Functions.superlanes).map (ecc_ov_get ova) ++
  (List.range Out.Functions.superlanes).map (ecc_ov_get ovb)

def descRange (n : Nat) : IntRange := {start := (n:Int)-1, stop := 0, step := -1}

def descBody {m : Type → Type} [Monad m] {α : Type} (n : Nat) (f : Int → α)
    (i : Int) (_hi : i∈descRange n) (xs : List α) : m (ForInStep (List α)) :=
  pure (.yield (f i::xs))

theorem desc_loop_eq {m : Type → Type} [Monad m] [LawfulMonad m] {α : Type}
    (n count : Nat) (hc : count≤n) (f : Int → α) (rest : List α) :
    IntRange.forIn'.loop (descRange n) (descBody (m:=m) n f) rest ((count:Int)-1)
      (by simp [descRange]) = pure ((List.range count).map (fun i => f (i:Int))++rest) := by
  induction count generalizing rest with
  | zero =>
    have hm : ¬(-1:Int)∈descRange n := by
      simp only [Membership.mem,descRange]
      simp
    rw [IntRange.forIn'.loop.eq_def]
    simp only [Int.natCast_zero,Int.zero_sub,dif_neg hm,List.range_zero,List.map_nil,List.nil_append]
  | succ count ih =>
    have hm : ((count+1:Nat):Int)-1∈descRange n := by
      simp only [Membership.mem,descRange]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hm,descBody,pure_bind]
    have hcount : count≤n := by omega
    have hih := ih hcount (f (count:Int)::rest)
    simpa only [descRange,Int.natCast_add,Int.natCast_one,Int.add_sub_cancel,
      Int.add_neg_one,List.range_succ,List.map_append,List.map_singleton,
      List.append_assoc,List.singleton_append] using hih

def nativeDesc {m : Type → Type} [Monad m] {α : Type}
    (n : Nat) (f : Int → α) (rest : List α) : m (List α) := do
  let mut xs := rest
  for i in [((n:Int)-1):0:-1]i do
    xs := f i::xs
  pure xs

theorem native_desc_eq {m : Type → Type} [Monad m] [LawfulMonad m] {α : Type}
    (n : Nat) (f : Int → α) (rest : List α) :
    nativeDesc (m:=m) n f rest=pure ((List.range n).map (fun i => f (i:Int))++rest) := by
  unfold nativeDesc
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  simp only [bind_pure,pure_bind]
  exact desc_loop_eq n n (by omega) f rest

theorem desc_forIn_eq {m : Type → Type} [Monad m] [LawfulMonad m] {α : Type}
    (n : Nat) (f : Int → α) (rest : List α) :
    (forIn ({start := (n:Int)-1, stop := 0, step := -1} : IntRange) rest (fun i xs => pure (ForInStep.yield (f i::xs))) : m (List α)) =
      pure ((List.range n).map (fun i => f (i:Int))++rest) := by
  have h := native_desc_eq (m:=m) n f rest
  simpa only [nativeDesc,pure_bind,bind_pure] using h

theorem actual_vector_units (v : Vec) (rest : List (BitVec 32)) :
    vec_units v rest=words v++rest := by
  have source : vec_units v rest =
      nativeDesc (m:=Id) (4*Out.Functions.superlanes) (fun i => BitVec.slice v (((32:Int) * i).toNat) 32) rest := by rfl
  rw [source,native_desc_eq]
  change (List.range (4*Out.Functions.superlanes)).map (fun k => BitVec.slice v (((32:Int)*(k:Int)).toNat) 32) ++ rest = words v ++ rest
  simp only [List.map_map,Function.comp_def,←Int.natCast_mul,Int.toNat_natCast]
  rfl

theorem fill_append_eq (xs : List fill_seg) (s : fill_seg) :
    fill_append xs s=xs++[s] := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa only [fill_append,List.cons_append] using congrArg (List.cons x) ih

theorem fill_len_eq (xs : List fill_seg) : fill_len xs=xs.length := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [fill_len,ih,List.length_cons]; omega

def fetchC (q t : Nat) (a b : Vec) (ova ovb : Overlay) : SailM Bool := do
  let fifo ← Out.readReg fill_fifo
  if iq_staging_depth≤(fifo[q]!).length then pure false
  else
    let P ← Out.readReg iq_prod
    let base := P[q]!
    observe_site "fetch" t q 0 (8*Out.Functions.superlanes) base
    Out.writeReg iq_prod (vectorUpdate (← Out.readReg iq_prod) q (base+8*Out.Functions.superlanes))
    Out.writeReg fill_fifo (vectorUpdate (← Out.readReg fill_fifo) q
      ((← Out.readReg fill_fifo)[q]! ++ [{ready := t + iq_fetch_latency, units := words a++words b, base := base, taken := 0, ovw := overlays ova ovb}]))
    pure true

theorem actual_fetch (q t : Nat) (a b : Vec) (ova ovb : Overlay) :
    iq_fetch_ov q t a b ova ovb=fetchC q t a b ova ovb := by
  have nat_step (x n : Nat) : ((x:Int)+(8:Int)*(n:Int)).toNat=x+8*n := by omega
  simp only [iq_fetch_ov,fetchC,fill_len_eq,actual_vector_units,fill_append_eq,List.append_nil]
  simp only [pure_bind,desc_forIn_eq]
  simp [overlays,Int.toNat_add,Int.toNat_mul,nat_step]

/-- A named small trusted C observer frame. The callback may update external
    output; its actual fetch arm reads no IQ cells and writes no Sail register or
    IQ/SRAM storage. This premise does not prove the bytes/order of erased sinks. -/
structure FetchObserver : Prop where
  frame : ∀ (t q : Nat) (base : Int) (s : St), q<145 →
    ∃ observed, (observe_site "fetch" t q 0 (8*Out.Functions.superlanes) base).run s=.ok () observed ∧
      observed.regs=s.regs ∧ observed.mem=s.mem

def fetchedSegment (t : Nat) (base : Int) (a b : Vec) (ova ovb : Overlay) : fill_seg :=
  {ready := t + iq_fetch_latency, units := words a++words b,
    base := base, taken := 0, ovw := overlays ova ovb}

def fetchedState (observed : St) (F : Vector (List fill_seg) 145) (P : Vector Nat 145)
    (q t : Nat) (a b : Vec) (ova ovb : Overlay) : St :=
  {observed with regs := ((observed.regs.insert iq_prod (vectorUpdate P q (P[q]!+8*Out.Functions.superlanes))).insert fill_fifo (vectorUpdate F q (F[q]!++[fetchedSegment t (P[q]!:Int) a b ova ovb])))}

theorem fetched_payload_order (t : Nat) (base : Int) (a b : Vec) (ova ovb : Overlay) :
    (fetchedSegment t base a b ova ovb).units =
      (List.range (4*Out.Functions.superlanes)).map (fun k => BitVec.slice a (32*k) 32) ++
      (List.range (4*Out.Functions.superlanes)).map (fun k => BitVec.slice b (32*k) 32) ∧
    (fetchedSegment t base a b ova ovb).units.length=8*Out.Functions.superlanes := by
  constructor
  · rfl
  · simp only [fetchedSegment,words,List.length_append,List.length_map,List.length_range]
    omega

theorem fetch_full_no_effect (q t : Nat) (a b : Vec) (ova ovb : Overlay)
    (s : St) (F : Vector (List fill_seg) 145)
    (hF : s.regs.get? fill_fifo=some F) (full : iq_staging_depth≤F[q]!.length) :
    (iq_fetch_ov q t a b ova ovb).run s=.ok false s := by
  rw [actual_fetch]
  simp [fetchC,Out.readReg,PreSail.readReg,hF,EStateM.run,bind,EStateM.bind,
    pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,full]

theorem fetch_room_exact (q t : Nat) (hq : q<145) (a b : Vec) (ova ovb : Overlay)
    (observer : FetchObserver) (s : St) (F : Vector (List fill_seg) 145) (P : Vector Nat 145)
    (hF : s.regs.get? fill_fifo=some F) (hP : s.regs.get? iq_prod=some P)
    (room : F[q]!.length<iq_staging_depth) :
    ∃ observed, (observe_site "fetch" t q 0 (8*Out.Functions.superlanes) (P[q]!:Int)).run s=.ok () observed ∧
      observed.regs=s.regs ∧ observed.mem=s.mem ∧
      (iq_fetch_ov q t a b ova ovb).run s=.ok true (fetchedState observed F P q t a b ova ovb) := by
  obtain ⟨observed,he,hr,hm⟩ := observer.frame t q (P[q]!:Int) s hq
  have call : observe_site "fetch" t q 0 (8*Out.Functions.superlanes) (P[q]!:Int) s=.ok () observed := he
  refine ⟨observed,he,hr,hm,?_⟩
  have notfull : ¬iq_staging_depth≤F[q]!.length := by omega
  have hp' : observed.regs.get? iq_prod=some P := by rw [hr,hP]
  have hf' : observed.regs.get? fill_fifo=some F := by rw [hr,hF]
  rw [actual_fetch]
  simp [fetchC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,hF,hP,hp',hf',
    EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
    modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,notfull,call,
    fetchedState,fetchedSegment,Std.ExtDHashMap.get?_insert]

theorem ready_same_parts (cap : Nat) (base s t : St) (hs : Ready cap base s)
    (hr : t.regs=s.regs) (hm : t.mem=s.mem) : Ready cap base t := by
  obtain ⟨⟨view,rep,bounds⟩,complete,foreign⟩ := hs
  refine ⟨⟨view,represents_memory cap view s t hm rep,?_⟩,?_,?_⟩
  · simpa only [Proofs.Initialized.Bounds,Proofs.StorageRep.Initialized,Present,hr,hm] using bounds
  · simpa only [MachineInitProofs.Complete,hr] using complete
  · simpa only [ForeignFrame,hm] using foreign

theorem fetched_state_ready (cap : Nat) (base observed : St) (ready : Ready cap base observed)
    (F : Vector (List fill_seg) 145) (P : Vector Nat 145) (q t : Nat)
    (a b : Vec) (ova ovb : Overlay) :
    Ready cap base (fetchedState observed F P q t a b ova ovb) := by
  obtain ⟨x,s1,h1,r1⟩ := ready_write cap base iq_prod
    (vectorUpdate P q (P[q]!+8*Out.Functions.superlanes)) (by decide) (by decide) observed ready
  change EStateM.Result.ok () {observed with regs := observed.regs.insert iq_prod (vectorUpdate P q (P[q]!+8*Out.Functions.superlanes))} = .ok x s1 at h1
  cases h1
  obtain ⟨y,s2,h2,r2⟩ := ready_write cap base fill_fifo
    (vectorUpdate F q (F[q]!++[fetchedSegment t (P[q]!:Int) a b ova ovb]))
    (by decide) (by decide) _ r1
  change EStateM.Result.ok () (fetchedState observed F P q t a b ova ovb) = .ok y s2 at h2
  cases h2
  exact r2

theorem actual_fetch_ready (cap q t : Nat) (hq : q<145) (base : St)
    (a b : Vec) (ova ovb : Overlay) (observer : FetchObserver) :
    Total (Ready cap base) (iq_fetch_ov q t a b ova ovb) := by
  intro state hs
  obtain ⟨F,hF⟩ := hs.2.1 fill_fifo
  obtain ⟨P,hP⟩ := hs.2.1 iq_prod
  by_cases full : iq_staging_depth≤F[q]!.length
  · exact ⟨false,state,fetch_full_no_effect q t a b ova ovb state F hF full,hs⟩
  · obtain ⟨observed,he,hr,hm,fetch⟩ := fetch_room_exact q t hq a b ova ovb observer state F P hF hP (by omega)
    exact ⟨true,_,fetch,fetched_state_ready cap base observed
      (ready_same_parts cap base state observed hs hr hm) F P q t a b ova ovb⟩

theorem seen_same_parts (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s t : St) (seen : Seen cap view H L F s)
    (hr : t.regs=s.regs) (hm : t.mem=s.mem) : Seen cap view H L F t := by
  refine ⟨represents_memory cap view s t hm seen.rep,?_,?_,?_,?_,?_,?_⟩
  · simpa only [Proofs.Initialized.Bounds,Proofs.StorageRep.Initialized,Present,hr,hm] using seen.bounds
  · simpa only [hr] using seen.head
  · simpa only [hr] using seen.len
  · simpa only [hr] using seen.fifo
  · simpa only [hr] using seen.active
  · simpa only [hr] using seen.wov

theorem seen_producer (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (s : St) (seen : Seen cap view H L F s)
    (P : Vector Nat 145) :
    Seen cap view H L F {s with regs:=s.regs.insert iq_prod P} := by
  refine ⟨seen.rep,?_,?_,?_,?_,?_,?_⟩
  · exact Proofs.Initialized.preserves_other_write (cap:=cap) iq_prod P (by decide) (by decide) s seen.bounds
  · simpa [Std.ExtDHashMap.get?_insert] using seen.head
  · simpa [Std.ExtDHashMap.get?_insert] using seen.len
  · simpa [Std.ExtDHashMap.get?_insert] using seen.fifo
  · obtain ⟨A,hA⟩ := seen.active
    exact ⟨A,by simpa [Std.ExtDHashMap.get?_insert] using hA⟩
  · obtain ⟨W,hW⟩ := seen.wov
    exact ⟨W,by simpa [Std.ExtDHashMap.get?_insert] using hW⟩

theorem fetched_state_seen (cap : Nat) (view : CView) (H L : Vector Nat 145)
    (F : Vector (List fill_seg) 145) (P : Vector Nat 145) (observed : St)
    (seen : Seen cap view H L F observed) (q t : Nat) (a b : Vec) (ova ovb : Overlay) :
    Seen cap view H L
      (vectorUpdate F q (F[q]!++[fetchedSegment t (P[q]!:Int) a b ova ovb]))
      (fetchedState observed F P q t a b ova ovb) := by
  exact seen_fifo cap view H L F
    (vectorUpdate F q (F[q]!++[fetchedSegment t (P[q]!:Int) a b ova ovb])) _
    (seen_producer cap view H L F observed seen
      (vectorUpdate P q (P[q]!+8*Out.Functions.superlanes)))

/-- The actual producer either refuses with no effect, or advances its original
    unit index by the entire unfiltered payload. Padding never compresses tags. -/
theorem actual_producer_step (cap q t : Nat) (hq : q<145) (base s : St)
    (ready : Ready cap base s) (observer : FetchObserver) (a b : Vec) (ova ovb : Overlay)
    (P : Vector Nat 145) (hP : s.regs.get? iq_prod=some P) :
    ∃ accepted after, (iq_fetch_ov q t a b ova ovb).run s=.ok accepted after ∧
      Ready cap base after ∧
      after.regs.get? iq_prod=some
        (if accepted then vectorUpdate P q (P[q]!+8*Out.Functions.superlanes) else P) := by
  obtain ⟨F,hF⟩ := ready.2.1 fill_fifo
  by_cases full : iq_staging_depth≤F[q]!.length
  · exact ⟨false,s,fetch_full_no_effect q t a b ova ovb s F hF full,ready,hP⟩
  · obtain ⟨observed,he,hr,hm,fetch⟩ := fetch_room_exact q t hq a b ova ovb observer s F P hF hP (by omega)
    refine ⟨true,_,fetch,fetched_state_ready cap base observed
      (ready_same_parts cap base s observed ready hr hm) F P q t a b ova ovb,?_⟩
    simp [fetchedState,Std.ExtDHashMap.get?_insert]

/-- Same concrete CView witness through actual enqueue followed by actual delivery.
    The control/tick caller must separately establish when these actions are invoked. -/
theorem actual_fetch_delivery_prefix (cap rate q t c : Nat) (hq : q<145)
    (base s : St) (view : CView) (H L P : Vector Nat 145) (F : Vector (List fill_seg) 145)
    (seen : Seen cap view H L F s) (complete : MachineInitProofs.Complete s)
    (foreign : ForeignFrame base s) (hP : s.regs.get? iq_prod=some P)
    (observer : FetchObserver) (ports : Proofs.Initialized.WritePorts cap)
    (a b : Vec) (ova ovb : Overlay) (room : F[q]!.length<iq_staging_depth) :
    let G := vectorUpdate F q (F[q]!++[fetchedSegment t (P[q]!:Int) a b ova ovb])
    let result := finishPlan (planLoop cap L[q]! c (rate+list_len G[q]!+1) (startPlan rate G[q]!))
    ∃ observed middle after,
      (observe_site "fetch" t q 0 (8*Out.Functions.superlanes) (P[q]!:Int)).run s=.ok () observed ∧
      middle=fetchedState observed F P q t a b ova ovb ∧
      (do let accepted ← iq_fetch_ov q t a b ova ovb
          let result ← deliverC cap rate q c
          pure (accepted,result)).run s=.ok (true,(result.pushed,result.lost)) after ∧
      Ready cap base after ∧
      Seen cap (historyStore cap view q H[q]! L[q]! result.accepted) H
        (vectorUpdate L q (L[q]!+result.accepted.length)) (vectorUpdate G q result.fifo) after ∧
      Prefix (cap-L[q]!) result := by
  dsimp only
  obtain ⟨observed,observe,hr,hm,fetch⟩ := fetch_room_exact q t hq a b ova ovb observer s F P seen.fifo hP room
  let middle := fetchedState observed F P q t a b ova ovb
  have start : Ready cap base s := ⟨⟨view,seen.rep,seen.bounds⟩,complete,foreign⟩
  have middle_ready := fetched_state_ready cap base observed
    (ready_same_parts cap base s observed start hr hm) F P q t a b ova ovb
  have middle_seen := fetched_state_seen cap view H L F P observed
    (seen_same_parts cap view H L F s observed seen hr hm) q t a b ova ovb
  obtain ⟨after,deliver,after_ready,after_seen,prefix_ok⟩ :=
    delivery_prefix_ready cap rate q c hq ports base middle view H L _ middle_seen middle_ready.2.1 middle_ready.2.2
  refine ⟨observed,middle,after,observe,rfl,?_,after_ready,after_seen,prefix_ok⟩
  change iq_fetch_ov q t a b ova ovb s=.ok true middle at fetch
  change deliverC cap rate q c middle=.ok _ after at deliver
  simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,fetch,deliver]

/-- Actual producer calls since the most recent reset. Intervening machine actions
    require their own producer/Ready frame before joining this history. -/
inductive FetchHistory (q : Nat) : St → St → Nat → Prop
  | initial (s : St) : FetchHistory q s s 0
  | fetch {s middle after : St} {count : Nat} (history : FetchHistory q s middle count)
      (t : Nat) (a b : Vec) (ova ovb : Overlay) (accepted : Bool)
      (run : (iq_fetch_ov q t a b ova ovb).run middle=.ok accepted after) :
      FetchHistory q s after (count + if accepted then 1 else 0)

theorem producer_history (cap q : Nat) (hq : q<145) (observer : FetchObserver)
    (base s after : St) (ready : Ready cap base s) (P : Vector Nat 145)
    (hP : s.regs.get? iq_prod=some P) (count : Nat) (history : FetchHistory q s after count) :
    Ready cap base after ∧
      after.regs.get? iq_prod=some (vectorUpdate P q (P[q]!+count*(8*Out.Functions.superlanes))) := by
  induction history with
  | initial =>
    refine ⟨ready,?_⟩
    simpa only [Nat.zero_mul,Nat.add_zero,vector_update_value P q hq] using hP
  | @fetch middle after count history t a b ova ovb accepted run ih =>
    obtain ⟨prior_ready,prior_prod⟩ := ih
    let prior := vectorUpdate P q (P[q]!+count*(8*Out.Functions.superlanes))
    obtain ⟨accepted',actual,he,ha,hprod⟩ := actual_producer_step cap q t hq base middle prior_ready observer a b ova ovb prior prior_prod
    rw [run] at he
    cases he
    refine ⟨ha,?_⟩
    cases accepted
    · simpa only [Bool.false_eq_true,ite_false,Nat.add_zero,prior] using hprod
    · simpa only [ite_true,prior,vectorUpdate_self _ _ _ hq,vector_update_twice P q _ _ hq,
        Nat.add_mul,Nat.one_mul,Nat.add_assoc] using hprod

/-- The C observer's fetch index/unit-offset formula recovers the original
    producer index for every in-payload unit; no tag-width narrowing is used. -/
theorem fetch_identity_coordinates (image_base count payload unit : Nat)
    (positive : 0<payload) (within : unit<payload) :
    (image_base+count*payload+unit- image_base)/payload=count ∧
    (image_base+count*payload+unit- image_base)%payload=unit := by
  have sub : image_base+count*payload+unit- image_base=count*payload+unit := by omega
  rw [sub]
  constructor
  · rw [Nat.add_comm,Nat.add_mul_div_right _ _ positive,Nat.div_eq_of_lt within]
    simp
  · simp [Nat.add_mod,Nat.mod_eq_of_lt within]

theorem distinct_fetch_payloads (image_base first second payload i j : Nat)
    (positive : 0<payload) (hi : i<payload) (hj : j<payload) (different : first≠second) :
    image_base+first*payload+ i ≠image_base+second*payload+ j := by
  intro equal
  have a := (fetch_identity_coordinates image_base first payload i positive hi).1
  have b := (fetch_identity_coordinates image_base second payload j positive hj).1
  rw [equal] at a
  exact different (a.symm.trans b)

end Proofs.OrderedFetch
