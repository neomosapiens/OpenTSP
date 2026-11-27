import Proofs.StorageProduct
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Capacity Proofs.StorageRep
set_option autoImplicit false
namespace Proofs.OrderedImage

def content (cap : Nat) (c : CView) (q h len : Nat) :=
  Proofs.Storage.contents cap c.unit c.tag q h len

def appendStore (cap : Nat) (c : CView) (q h len : Nat) (u : BitVec 32) (tag : Int) : CView :=
  cTagWrite (cUnitWrite c q (ringC cap (h+len)) u) q (ringC cap (h+len)) tag

theorem append_units (cap : Nat) (c : CView) (q h len : Nat) (u : BitVec 32) (tag : Int) :
    (appendStore cap c q h len u tag).unit =
      Proofs.Storage.put c.unit q (ringC cap (h+len)) u := by
  funext r j
  simp [appendStore,cTagWrite,cUnitWrite,Proofs.Storage.put,eq_comm,and_comm]

theorem append_tags (cap : Nat) (c : CView) (q h len : Nat) (u : BitVec 32) (tag : Int) :
    (appendStore cap c q h len u tag).tag =
      Proofs.Storage.put c.tag q (ringC cap (h+len)) tag := by
  funext r j
  simp [appendStore,cTagWrite,cUnitWrite,Proofs.Storage.put,eq_comm,and_comm]

theorem append_content (cap : Nat) (c : CView) (q h len : Nat) (u : BitVec 32) (tag : Int)
    (hh : h<cap) (room : len<cap) :
    content cap (appendStore cap c q h len u tag) q h (len+1) =
      content cap c q h len ++ [(u,tag)] := by
  unfold content
  rw [append_units,append_tags]
  exact Proofs.Storage.contents_append cap q h len hh room c.unit c.tag u tag

/-- Specific initial/final store history, retaining the SAME CView witness and
    indexed unit/tag updates. Values outside initialized cells are ghost extensions;
    no claim about allocator garbage bytes is made. -/
theorem push_room_exact (cap q : Nat) (hq : q<145) (u : BitVec 32) (tag : Int)
    (ports : Proofs.Initialized.WritePorts cap) (s : St) (c : CView)
    (rep : Represents cap c s) (bounds : Proofs.Initialized.Bounds cap s)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (hh : H[q]!<cap) (room : L[q]!<cap) :
    ∃ after, (pushC cap q u tag).run s = .ok true after ∧
      Represents cap (appendStore cap c q H[q]! L[q]! u tag) after ∧
      Proofs.Initialized.Bounds cap after ∧
      after.regs=s.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1)) ∧
      content cap (appendStore cap c q H[q]! L[q]! u tag) q H[q]! (L[q]!+1) =
        content cap c q H[q]! L[q]! ++ [(u,tag)] := by
  let slot := ringC cap (H[q]!+L[q]!)
  have hslot : slot<cap := by
    dsimp [slot]
    rw [Proofs.Capacity.ring_spec]
    split <;> omega
  let t1 := writeCell s (145*cap) 0 (q*cap+slot) u.toNat
  let t2 := writeCell t1 (145*cap) 1 (q*cap+slot) (zigzag tag)
  let after : St := {t2 with regs := t2.regs.insert Register.iq_len (vectorUpdate L q (L[q]!+1))}
  have e1 : iq_unit_write q slot u s = .ok () t1 := ports.unit q slot u s hq hslot
  have e2 : iq_tag_write q slot tag t1 = .ok () t2 := ports.tag q slot tag t1 hq hslot
  have r1 : t1.regs=s.regs := rfl
  have r2 : t2.regs=s.regs := rfl
  have hc : ((H[q]! : Int)+(L[q]! : Int)).toNat=H[q]!+L[q]! := by omega
  have full : ¬cap≤L[q]! := by omega
  have he : (pushC cap q u tag).run s = .ok true after := by
    simp [pushC,Out.readReg,Out.writeReg,PreSail.readReg,PreSail.writeReg,hL,hH,
      EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,
      EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,full,hc,slot,e1,e2,r1,r2,after]
  have ha : Proofs.Initialized.Bounds cap after := by
    obtain ⟨b,t,ht,hb⟩ := Proofs.Initialized.push_preserves cap q hq u tag ports s bounds
    rw [he] at ht
    cases ht
    exact hb
  refine ⟨after,he,?_,ha,?_,append_content cap c q H[q]! L[q]! u tag hh room⟩
  · change Represents cap (cTagWrite (cUnitWrite c q slot u) q slot tag) t2
    exact tag_represents cap (cUnitWrite c q slot u) t1
      (unit_represents cap c s rep q slot u hq hslot) q slot tag hq hslot
  · simp [after,r2]


def iterUnit (start count : Nat) (f : Nat → SailM Unit) : SailM Unit :=
  match count with
  | 0 => pure ()
  | count+1 => do
    f start
    iterUnit (start+1) count f

def rangeBody (n : Nat) (f : Nat → SailM Unit)
    (i : Int) (_hi : i ∈ ({stop := (n:Int)-1} : IntRange)) (_u : Unit) :
    SailM (ForInStep Unit) := do
  f i.toNat
  pure (ForInStep.yield ())

/-- Kernel equality for the real lean-sail inclusive iterator, for arbitrary stateful
    body actions. This retains all errors and partial state effects. -/
theorem native_loop_eq (n start count : Nat) (hn : start+count=n) (f : Nat → SailM Unit) :
    IntRange.forIn'.loop ({stop := (n:Int)-1} : IntRange) (rangeBody n f) ()
      (start:Int) (by simp) = iterUnit start count f := by
  induction count generalizing start with
  | zero =>
    have hnot : ¬(start:Int) ∈ ({stop := (n:Int)-1} : IntRange) := by
      simp only [Membership.mem]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hnot,iterUnit]
  | succ count ih =>
    have hmem : (start:Int) ∈ ({stop := (n:Int)-1} : IntRange) := by
      simp only [Membership.mem]
      simp
      omega
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hmem,rangeBody,iterUnit,bind_assoc,pure_bind]
    have hnext : start+1+count=n := by omega
    have hih := ih (start+1) hnext
    simpa only [Int.natCast_add,Int.natCast_add_one] using
      congrArg (fun next : SailM Unit => f start >>= fun _ => next) hih

def nativeIter (n : Nat) (f : Nat → SailM Unit) : SailM Unit := do
  let mut u : Unit := ()
  for i in [0:((n:Int)-1):1]i do
    let () := u
    u ← f i.toNat
  pure u

theorem native_iter_eq (n : Nat) (f : Nat → SailM Unit) :
    nativeIter n f = iterUnit 0 n f := by
  unfold nativeIter
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  simp only [bind_pure,pure_bind]
  apply Eq.trans (b := IntRange.forIn' ({stop := (n:Int)-1} : IntRange) () (rangeBody n f))
  · rfl
  · exact native_loop_eq n 0 n (by omega) f


/-- Image values are immutable, with exactly the allocation/length domain exposed by C.
    Successful allocator/exclusive ownership and primitive store interpretation stay
    at the named trusted boundary; actual model loops must establish every bound. -/
structure ImageValues (image : Nat → Nat → BitVec 32) (length : Nat → Nat) : Prop where
  len : ∀ q s, q<145 → (iq_image_len q).run s = .ok (length q) s
  unit : ∀ q i s, q<145 → i<length q → (iq_image_unit q i).run s = .ok (image q i) s

def prefixPairs (image : Nat → BitVec 32) (count : Nat) : List (BitVec 32 × Int) :=
  (List.range count).filterMap fun i => if image i=iq_padding then none else some (image i,(i:Int))

theorem prefix_zero (image : Nat → BitVec 32) : prefixPairs image 0 = [] := by
  simp [prefixPairs]

theorem prefix_succ (image : Nat → BitVec 32) (i : Nat) :
    prefixPairs image (i+1) = prefixPairs image i ++
      (if image i=iq_padding then [] else [(image i,(i:Int))]) := by
  simp [prefixPairs,List.range_succ,List.filterMap_append]
  split <;> simp_all

theorem prefix_length (image : Nat → BitVec 32) (count : Nat) :
    (prefixPairs image count).length≤count := by
  induction count with
  | zero => simp [prefix_zero]
  | succ i ih =>
    rw [prefix_succ]
    split <;> simp_all <;> omega

def loadStore (cap : Nat) (c : CView) (q : Nat) (image : Nat → BitVec 32) : Nat → CView
  | 0 => c
  | i+1 => if image i=iq_padding then loadStore cap c q image i
    else appendStore cap (loadStore cap c q image i) q 0
      (prefixPairs image i).length (image i) (i:Int)

def imageStep (cap q : Nat) (i : Int) : SailM Unit := do
  let u ← iq_image_unit q i.toNat
  if u != iq_padding then
    let _ ← pushC cap q u i
    pure ()
  else pure ()

def nativeImageLoop (cap q n : Nat) : SailM Unit := do
  let mut u : Unit := ()
  for i in [0:((n:Int)-1):1]i do
    let () := u
    u ← imageStep cap q i
  pure u

theorem native_image_loop_eq (cap q n : Nat) :
    nativeImageLoop cap q n = iterUnit 0 n (fun i => imageStep cap q (i:Int)) := by
  rw [← native_iter_eq n (fun i => imageStep cap q (i:Int))]
  unfold nativeImageLoop nativeIter
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  simp only [bind_pure,pure_bind]
  congr 1
  funext i hi u
  have hnonneg : 0≤i := by
    simp only [Membership.mem] at hi
    simp at hi
    omega
  have he : (i.toNat:Int)=i := by omega
  simp only [he]

theorem vector_update_twice (V : Vector Nat 145) (q x y : Nat) (hq : q<145) :
    vectorUpdate (vectorUpdate V q x) q y = vectorUpdate V q y := by
  simp [vectorUpdate,Vector.set!,hq]



theorem vector_update_value (V : Vector Nat 145) (q : Nat) (hq : q<145) :
    vectorUpdate V q V[q]! = V := by
  cases V
  simp_all [vectorUpdate,Vector.set!,Array.setIfInBounds]

theorem store_content (cap : Nat) (hcap : 0<cap) (c : CView) (q count : Nat)
    (image : Nat → BitVec 32) (within : count≤cap) :
    content cap (loadStore cap c q image count) q 0 (prefixPairs image count).length =
      prefixPairs image count := by
  induction count with
  | zero => simp [loadStore,prefix_zero,content,Proofs.Storage.contents]
  | succ i ih =>
    have hi : i≤cap := by omega
    have ih' := ih hi
    have room : (prefixPairs image i).length<cap := by
      have hp := prefix_length image i
      omega
    rw [loadStore,prefix_succ]
    by_cases padding : image i=iq_padding
    · simp [padding,ih']
    · simp only [padding,if_false,List.length_append,List.length_singleton]
      rw [append_content cap (loadStore cap c q image i) q 0 _ _ _ hcap room,ih']

structure LoadInv (cap q : Nat) (c : CView) (image : Nat → BitVec 32)
    (base : St) (L0 : Vector Nat 145) (i : Nat) (s : St) : Prop where
  rep : Represents cap (loadStore cap c q image i) s
  bounds : Proofs.Initialized.Bounds cap s
  length : s.regs.get? Register.iq_len=some (vectorUpdate L0 q (prefixPairs image i).length)
  frame : ∀ r, r≠Register.iq_len → s.regs.get? r=base.regs.get? r

theorem image_step_exact (cap q n i : Nat) (hcap : 0<cap) (hq : q<145)
    (hn : n≤cap) (hi : i<n) (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
    (himage : length q=n) (images : ImageValues image length)
    (ports : Proofs.Initialized.WritePorts cap) (c : CView) (base s : St)
    (H L0 : Vector Nat 145) (hH : base.regs.get? Register.iq_head=some H) (hh : H[q]! = 0)
    (inv : LoadInv cap q c (image q) base L0 i s) :
    ∃ after, (imageStep cap q (i:Int)).run s = .ok () after ∧
      LoadInv cap q c (image q) base L0 (i+1) after := by
  have read := images.unit q i s hq (by omega)
  change iq_image_unit q i s = .ok (image q i) s at read
  have hcurH : s.regs.get? Register.iq_head=some H := by
    rw [inv.frame Register.iq_head (by decide),hH]
  by_cases padding : image q i=iq_padding
  · refine ⟨s,?_,?_⟩
    · simp [imageStep,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,read,padding]
    · refine ⟨?_,inv.bounds,?_,inv.frame⟩
      · simpa [loadStore,padding] using inv.rep
      · simpa [prefix_succ,padding] using inv.length
  · have room : (prefixPairs (image q) i).length<cap := by
      have hp := prefix_length (image q) i
      omega
    let L := vectorUpdate L0 q (prefixPairs (image q) i).length
    have hL : s.regs.get? Register.iq_len=some L := inv.length
    have lq : L[q]! = (prefixPairs (image q) i).length := by
      exact Proofs.vectorUpdate_self L0 q _ hq
    obtain ⟨after,push,rep,bounds,regs,contents⟩ :=
      push_room_exact cap q hq (image q i) (i:Int) ports s
        (loadStore cap c q (image q) i) inv.rep inv.bounds H L hcurH hL
        (by omega) (by omega)
    refine ⟨after,?_,?_,bounds,?_,?_⟩
    · have push' : pushC cap q (image q i) (i:Int) s = .ok true after := push
      simp [imageStep,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,read,padding,push']
    · simpa [loadStore,padding,hh,lq] using rep
    · rw [regs]
      simp only [Std.ExtDHashMap.get?_insert]
      rw [lq]
      simp only [L,prefix_succ,padding,if_false,List.length_append,List.length_singleton]
      rw [vector_update_twice L0 q _ _ hq]
      simp
    · intro r hr
      rw [regs]
      simpa [Std.ExtDHashMap.get?_insert,hr,Ne.symm hr] using inv.frame r hr


theorem iterate_image_exact (count : Nat) (cap q n : Nat) (hcap : 0<cap) (hq : q<145)
    (hn : n≤cap) (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
    (himage : length q=n) (images : ImageValues image length)
    (ports : Proofs.Initialized.WritePorts cap) (c : CView) (base : St)
    (H L0 : Vector Nat 145) (hH : base.regs.get? Register.iq_head=some H) (hh : H[q]! = 0) :
    ∀ start s, start+count≤n → LoadInv cap q c (image q) base L0 start s →
      ∃ after, (iterUnit start count (fun i => imageStep cap q (i:Int))).run s = .ok () after ∧
        LoadInv cap q c (image q) base L0 (start+count) after := by
  induction count with
  | zero =>
    intro start s hi inv
    exact ⟨s,rfl,by simpa using inv⟩
  | succ count ih =>
    intro start s hi inv
    obtain ⟨middle,step,nextInv⟩ := image_step_exact cap q n start hcap hq hn (by omega)
      image length himage images ports c base s H L0 hH hh inv
    obtain ⟨after,rest,lastInv⟩ := ih (start+1) middle (by omega) nextInv
    refine ⟨after,?_,?_⟩
    · have step' : imageStep cap q (start:Int) s = .ok () middle := step
      have rest' : iterUnit (start+1) count (fun i => imageStep cap q (i:Int)) middle = .ok () after := rest
      simp [iterUnit,EStateM.run,bind,EStateM.bind,step',rest']
    · simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using lastInv

/-- Complete native inclusive image loop, retaining a specific CView store history:
    padding occupies no logical slot, while each retained unit keeps its original
    image index as its tag. Stale non-live backing cells are neither zeroed nor read. -/
theorem native_image_load (cap q n : Nat) (hcap : 0<cap) (hq : q<145) (hn : n≤cap)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
    (himage : length q=n) (images : ImageValues image length)
    (ports : Proofs.Initialized.WritePorts cap) (c : CView) (s : St)
    (rep : Represents cap c s) (bounds : Proofs.Initialized.Bounds cap s)
    (H L0 : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L0) (hh : H[q]! = 0) (empty : L0[q]! = 0) :
    ∃ after, (nativeImageLoop cap q n).run s = .ok () after ∧
      Represents cap (loadStore cap c q (image q) n) after ∧
      Proofs.Initialized.Bounds cap after ∧
      after.regs.get? Register.iq_len=some (vectorUpdate L0 q (prefixPairs (image q) n).length) ∧
      content cap (loadStore cap c q (image q) n) q 0 (prefixPairs (image q) n).length =
        prefixPairs (image q) n ∧
      (∀ r, r≠Register.iq_len → after.regs.get? r=s.regs.get? r) := by
  have initial : LoadInv cap q c (image q) s L0 0 s := by
    refine ⟨by simpa [loadStore] using rep,bounds,?_,?_⟩
    · have he : vectorUpdate L0 q 0=L0 := by simpa [empty] using vector_update_value L0 q hq
      simpa [prefix_zero,he] using hL
    · intro r hr; rfl
  obtain ⟨after,run,inv⟩ := iterate_image_exact n cap q n hcap hq hn image length himage images
    ports c s H L0 hH hh 0 s (by omega) initial
  simp only [Nat.zero_add] at inv
  refine ⟨after,?_,inv.rep,inv.bounds,inv.length,store_content cap hcap c q n (image q) hn,inv.frame⟩
  simpa only [native_image_loop_eq] using run


open Out.Register
def resetTailC (cap : Nat) : SailM Unit := do
  Out.writeReg iq_last (vectorInit (Neg.neg 1))
  Out.writeReg iq_dry (vectorInit false)
  Out.writeReg iq_active (vectorInit 0)
  Out.writeReg fill_fifo (vectorInit [])
  Out.writeReg iq_prod (vectorInit 0)
  Out.writeReg iq_wov (vectorInit [])
  Out.writeReg iq_ecc_live false
  let loop_q_lower := 0
  let loop_q_upper := 144
  let mut loop_vars := ()
  for q in [loop_q_lower:loop_q_upper:1]i do
    let () := loop_vars
    loop_vars ← do
      let n ← do (iq_image_len q)
      nativeImageLoop cap q n
      Out.writeReg iq_prod (vectorUpdate (← Out.readReg iq_prod) q (4 *i (Int.ediv (n +i 3) 4)))
  (pure loop_vars)


theorem bind_if {α β : Type} (p : Prop) [Decidable p] (a b : SailM α) (f : α → SailM β) :
    ((if p then a else b) >>= f) = (if p then a >>= f else b >>= f) := by
  by_cases h : p <;> simp [h]

theorem reset_tail_bridge : Proofs.Reset.resetTail = resetTailC iq_cap := by
  simp only [Proofs.Reset.resetTail,resetTailC,nativeImageLoop,imageStep,Proofs.Capacity.push_production,bind_if,bind_assoc,pure_bind,bind_pure,Prod.eta]


theorem append_content_other (cap : Nat) (c : CView) (q r h len start count : Nat)
    (u : BitVec 32) (tag : Int) (different : r≠q) :
    content cap (appendStore cap c q h len u tag) r start count =
      content cap c r start count := by
  unfold content Proofs.Storage.contents
  apply List.map_congr_left
  intro i hi
  simp [appendStore,cUnitWrite,cTagWrite,Ne.symm different,different]

theorem store_content_other (cap : Nat) (c : CView) (q r start count upto : Nat)
    (image : Nat → BitVec 32) (different : r≠q) :
    content cap (loadStore cap c q image upto) r start count=content cap c r start count := by
  induction upto with
  | zero => rfl
  | succ i ih =>
    unfold loadStore
    split
    · exact ih
    · rw [append_content_other cap _ q r _ _ start count _ _ different,ih]

def resetStore (cap : Nat) (c : CView) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) : Nat → CView
  | 0 => c
  | q+1 => loadStore cap (resetStore cap c image length q) q (image q) (length q)

theorem reset_store_content (cap : Nat) (hcap : 0<cap) (c : CView)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (upto : Nat)
    (admit : ∀ q, q<upto → length q≤cap) :
    ∀ q, q<upto →
      content cap (resetStore cap c image length upto) q 0 (prefixPairs (image q) (length q)).length =
        prefixPairs (image q) (length q) := by
  induction upto with
  | zero => intro q hq; omega
  | succ r ih =>
    intro q hq
    by_cases same : q=r
    · subst q
      exact store_content cap hcap (resetStore cap c image length r) r (length r)
        (image r) (admit r (by omega))
    · have bound : q<r := by omega
      rw [resetStore,store_content_other cap _ r q 0 _ _ _ same]
      exact ih (fun j hj => admit j (by omega)) q bound

def imageQueue (cap q : Nat) : SailM Unit := do
  let n ← iq_image_len q
  nativeImageLoop cap q n
  Out.writeReg Register.iq_prod (vectorUpdate (← Out.readReg Register.iq_prod) q
    (4 *i (Int.ediv (n +i 3) 4)))

def resetMeta : SailM Unit := do
  Out.writeReg iq_last (vectorInit (Neg.neg 1))
  Out.writeReg iq_dry (vectorInit false)
  Out.writeReg iq_active (vectorInit 0)
  Out.writeReg fill_fifo (vectorInit [])
  Out.writeReg iq_prod (vectorInit 0)
  Out.writeReg iq_wov (vectorInit [])
  Out.writeReg iq_ecc_live false

theorem reset_tail_iteration (cap : Nat) :
    resetTailC cap = (do resetMeta; iterUnit 0 145 (imageQueue cap)) := by
  rw [← native_iter_eq 145 (imageQueue cap)]
  simp only [resetTailC,resetMeta,nativeIter,imageQueue,bind_assoc,pure_bind,bind_pure,Prod.eta]
  rfl


theorem vector_update_other (V : Vector Nat 145) (q r value : Nat) (hr : r<145) (hne : r≠q) :
    (vectorUpdate V q value)[r]! = V[r]! := by
  simp [vectorUpdate,Vector.set!,hr,Array.getElem_setIfInBounds_ne,Ne.symm hne]

def alignedImageEnd (n : Nat) : Nat := (4 * (Int.ediv ((n:Int)+3) 4)).toNat

structure OuterInv (cap : Nat) (c : CView) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (base : St) (upto : Nat) (s : St) : Prop where
  rep : Represents cap (resetStore cap c image length upto) s
  bounds : Proofs.Initialized.Bounds cap s
  head : s.regs.get? iq_head=some (vectorInit 0 : Vector Nat 145)
  len : ∃ L : Vector Nat 145, s.regs.get? iq_len=some L ∧
    ∀ q, q<145 → L[q]! = if q<upto then (prefixPairs (image q) (length q)).length else 0
  prod : ∃ P : Vector Nat 145, s.regs.get? iq_prod=some P ∧
    ∀ q, q<145 → P[q]! = if q<upto then alignedImageEnd (length q) else 0
  frame : ∀ r, r≠iq_len → r≠iq_prod → s.regs.get? r=base.regs.get? r

def setReg (s : St) (r : Register) (v : RegisterType r) : St :=
  {s with regs := s.regs.insert r v}

def initialIQState (s : St) : St :=
  let s := setReg s iq_head (vectorInit 0)
  let s := setReg s iq_len (vectorInit 0)
  let s := setReg s iq_last (vectorInit (-1))
  let s := setReg s iq_dry (vectorInit false)
  let s := setReg s iq_active (vectorInit 0)
  let s := setReg s fill_fifo (vectorInit [])
  let s := setReg s iq_prod (vectorInit 0)
  let s := setReg s iq_wov (vectorInit [])
  setReg s iq_ecc_live false

def resetPrefix : SailM Unit := do
  Out.writeReg iq_head (vectorInit 0)
  Out.writeReg iq_len (vectorInit 0)
  resetMeta

theorem reset_prefix_run (s : St) :
    resetPrefix.run s = .ok () (initialIQState s) := by rfl

theorem initial_outer (cap : Nat) (hcap : 0<cap) (c : CView) (s : St)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat) (rep : Represents cap c s) :
    OuterInv cap c image length (initialIQState s) 0 (initialIQState s) := by
  have head : (initialIQState s).regs.get? iq_head=some (vectorInit 0 : Vector Nat 145) := by
    simp [initialIQState,setReg,Std.ExtDHashMap.get?_insert]
  have len : (initialIQState s).regs.get? iq_len=some (vectorInit 0 : Vector Nat 145) := by
    simp [initialIQState,setReg,Std.ExtDHashMap.get?_insert]
  refine ⟨rep,?_,head,⟨vectorInit 0,len,?_⟩,⟨vectorInit 0,?_,?_⟩,?_⟩
  · exact ⟨vectorInit 0,vectorInit 0,head,len,fun q hq => by simp [vectorInit,hq,hcap]⟩
  · intro q hq; simp [vectorInit,hq]
  · simp [initialIQState,setReg,Std.ExtDHashMap.get?_insert]
  · intro q hq; simp [vectorInit,hq]
  · intro r hr hp; rfl


theorem queue_image_exact (cap q : Nat) (hcap : 0<cap) (hq : q<145)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
    (admit : length q≤cap) (images : ImageValues image length)
    (ports : Proofs.Initialized.WritePorts cap) (c : CView) (base s : St)
    (inv : OuterInv cap c image length base q s) :
    ∃ after, (imageQueue cap q).run s = .ok () after ∧
      OuterInv cap c image length base (q+1) after := by
  obtain ⟨L,hL,hLs⟩ := inv.len
  obtain ⟨P,hP,hPs⟩ := inv.prod
  have empty : L[q]! = 0 := by simpa using hLs q hq
  obtain ⟨middle,load,rep,bounds,len,contents,frame⟩ := native_image_load cap q (length q)
    hcap hq admit image length rfl images ports (resetStore cap c image length q)
    s inv.rep inv.bounds (vectorInit 0) L inv.head hL (by simp [vectorInit,hq]) empty
  have hPt : middle.regs.get? iq_prod=some P := by
    rw [frame iq_prod (by decide),hP]
  let P' := vectorUpdate P q (alignedImageEnd (length q))
  let after := setReg middle iq_prod P'
  have write : (Out.writeReg iq_prod P').run middle = .ok () after := by rfl
  have afterBounds : Proofs.Initialized.Bounds cap after := by
    have hb := Proofs.Initialized.preserves_other_write (cap:=cap) iq_prod P'
      (by decide) (by decide) middle bounds
    simpa [write] using hb
  refine ⟨after,?_,rep,afterBounds,?_,?_,?_,?_⟩
  · have read : iq_image_len q s = .ok (length q) s := images.len q s hq
    have load' : nativeImageLoop cap q (length q) s = .ok () middle := load
    simp [imageQueue,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,read,load',
      Out.readReg,PreSail.readReg,hPt,get,getThe,MonadStateOf.get,EStateM.get,
      Out.writeReg,PreSail.writeReg,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,
      P',alignedImageEnd,after,setReg]
  · change (middle.regs.insert iq_prod P').get? iq_head=some (vectorInit 0 : Vector Nat 145)
    simpa [Std.ExtDHashMap.get?_insert,frame iq_head (by decide)] using inv.head
  · refine ⟨vectorUpdate L q (prefixPairs (image q) (length q)).length,?_,?_⟩
    · simpa [after,setReg,Std.ExtDHashMap.get?_insert] using len
    · intro r hr
      by_cases same : r=q
      · subst r
        rw [Proofs.vectorUpdate_self _ _ _ hq]
        simp
      · rw [vector_update_other L q r _ hr same,hLs r hr]
        have he : r<q+1 ↔ r<q := by omega
        simp only [he]
  · refine ⟨P',by simp [after,setReg],?_⟩
    intro r hr
    dsimp only [P']
    by_cases same : r=q
    · subst r
      rw [Proofs.vectorUpdate_self _ _ _ hq]
      simp
    · rw [vector_update_other P q r _ hr same,hPs r hr]
      have he : r<q+1 ↔ r<q := by omega
      simp only [he]
  · intro r hl hp
    change (middle.regs.insert iq_prod P').get? r=base.regs.get? r
    simp only [Std.ExtDHashMap.get?_insert]
    simpa [hp,Ne.symm hp,frame r hl] using inv.frame r hl hp


theorem iterate_queues_exact (count : Nat) (cap : Nat) (hcap : 0<cap)
    (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
    (admit : ∀ q, q<145 → length q≤cap) (images : ImageValues image length)
    (ports : Proofs.Initialized.WritePorts cap) (c : CView) (base : St) :
    ∀ start s, start+count≤145 → OuterInv cap c image length base start s →
      ∃ after, (iterUnit start count (imageQueue cap)).run s = .ok () after ∧
        OuterInv cap c image length base (start+count) after := by
  induction count with
  | zero =>
    intro start s hi inv
    exact ⟨s,rfl,by simpa using inv⟩
  | succ count ih =>
    intro start s hi inv
    have hq : start<145 := by omega
    obtain ⟨middle,step,nextInv⟩ := queue_image_exact cap start hcap hq image length
      (admit start hq) images ports c base s inv
    obtain ⟨after,rest,lastInv⟩ := ih (start+1) middle (by omega) nextInv
    refine ⟨after,?_,?_⟩
    · have step' : imageQueue cap start s = .ok () middle := step
      have rest' : iterUnit (start+1) count (imageQueue cap) middle = .ok () after := rest
      simp [iterUnit,EStateM.run,bind,EStateM.bind,step',rest']
    · simpa [Nat.add_assoc,Nat.add_comm,Nat.add_left_comm] using lastInv

def resetC (cap : Nat) : SailM Unit := do
  Out.writeReg iq_head (vectorInit 0)
  Out.writeReg iq_len (vectorInit 0)
  resetTailC cap

theorem resetC_iteration (cap : Nat) :
    resetC cap = (do resetPrefix; iterUnit 0 145 (imageQueue cap)) := by
  simp only [resetC,reset_tail_iteration,resetPrefix,bind_assoc]

theorem reset_production : iq_reset () = resetC iq_cap := by
  rw [Proofs.Reset.reset_split,reset_tail_bridge]
  rfl

/-- Whole reset, all145 queues and one retained CView store history. The length
    admission premise is the existing image-loader bound, not a new input limit. -/
theorem reset_exact (cap : Nat) (hcap : 0<cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (admit : ∀ q, q<145 → length q≤cap)
    (images : ImageValues image length) (ports : Proofs.Initialized.WritePorts cap)
    (c : CView) (s : St) (rep : Represents cap c s) :
    ∃ after, (resetC cap).run s = .ok () after ∧
      OuterInv cap c image length (initialIQState s) 145 after ∧
      ∀ q, q<145 →
        content cap (resetStore cap c image length 145) q 0 (prefixPairs (image q) (length q)).length =
          prefixPairs (image q) (length q) := by
  have initial := initial_outer cap hcap c s image length rep
  obtain ⟨after,run,inv⟩ := iterate_queues_exact 145 cap hcap image length admit images ports
    c (initialIQState s) 0 (initialIQState s) (by omega) initial
  simp only [Nat.zero_add] at inv
  refine ⟨after,?_,inv,reset_store_content cap hcap c image length 145 admit⟩
  rw [resetC_iteration]
  have startRun : resetPrefix s = .ok () (initialIQState s) := reset_prefix_run s
  have run' : iterUnit 0 145 (imageQueue cap) (initialIQState s) = .ok () after := run
  simp [EStateM.run,bind,EStateM.bind,startRun,run']

theorem actual_reset_exact (hcap : 0<iq_cap) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (admit : ∀ q, q<145 → length q≤iq_cap)
    (images : ImageValues image length) (ports : Proofs.Initialized.WritePorts iq_cap)
    (c : CView) (s : St) (rep : Represents iq_cap c s) :
    ∃ after, (iq_reset ()).run s = .ok () after ∧
      OuterInv iq_cap c image length (initialIQState s) 145 after ∧
      ∀ q, q<145 →
        content iq_cap (resetStore iq_cap c image length 145) q 0 (prefixPairs (image q) (length q)).length =
          prefixPairs (image q) (length q) := by
  simpa only [reset_production] using reset_exact iq_cap hcap image length admit images ports c s rep


/-- The reset epoch fields are explicit, including queued fills and old ECC overlays.
    Logical empty/loaded occupancy does not require clearing stale external cells. -/
theorem reset_epoch_fields (cap : Nat) (c : CView) (image : Nat → Nat → BitVec 32)
    (length : Nat → Nat) (s after : St)
    (inv : OuterInv cap c image length (initialIQState s) 145 after) :
    after.regs.get? iq_last=some (vectorInit (-1)) ∧
    after.regs.get? iq_dry=some (vectorInit false) ∧
    after.regs.get? iq_active=some (vectorInit 0) ∧
    after.regs.get? fill_fifo=some (vectorInit []) ∧
    after.regs.get? iq_wov=some (vectorInit []) ∧
    after.regs.get? iq_ecc_live=some false := by
  refine ⟨?_,?_,?_,?_,?_,?_⟩
  all_goals rw [inv.frame _ (by decide) (by decide)]
  all_goals simp [initialIQState,setReg,Std.ExtDHashMap.get?_insert]

end Proofs.OrderedImage
