import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
namespace Proofs.OverlayHistory
def lookup (w : Nat) : List ov_ent → BitVec 10
 | [] => 0#10
 | ent::rest => if ent.ov_k=w then ent.ov_v else lookup w rest
/-- A key has one value; this also permits identical duplicate entries, which
    cannot change a lookup when the actual setter reverses retained entries. -/
def Coherent (xs : List ov_ent) : Prop :=
 ∀ a, a∈xs → ∀ b, b∈xs → a.ov_k=b.ov_k → a.ov_v=b.ov_v
def update (w : Nat) (e : BitVec 10) (xs : List ov_ent) : List ov_ent :=
 let kept:=(xs.filter (fun x=>x.ov_k != w)).reverse
 if e=0#10 then kept else {ov_k:=w,ov_v:=e}::kept

theorem coherent_nil : Coherent [] := by simp [Coherent]
theorem coherent_sub (xs ys : List ov_ent) (h : Coherent xs)
 (subset : ∀ a,a∈ys → a∈xs) : Coherent ys := by
 intro a ha b hb same
 exact h a (subset a ha) b (subset b hb) same

theorem lookup_zero (w : Nat) (xs : List ov_ent)
 (none : ∀ a,a∈xs → a.ov_k≠w) : lookup w xs=0#10 := by
 induction xs with
 | nil => rfl
 | cons a rest ih =>
   have head:=none a (by simp)
   simp only [lookup,if_neg head]
   exact ih (fun b hb=>none b (by simp [hb]))

theorem lookup_member (w : Nat) (xs : List ov_ent) (h : Coherent xs)
 (a : ov_ent) (member : a∈xs) (key : a.ov_k=w) : lookup w xs=a.ov_v := by
 induction xs with
 | nil => simp at member
 | cons head rest ih =>
   by_cases same : head.ov_k=w
   · simp only [lookup,if_pos same]
     exact h head (by simp) a member (by omega)
   · simp only [lookup,if_neg same]
     have tail : a∈rest := by
       rcases List.mem_cons.mp member with eq|mem
       · subst a;exact False.elim (same key)
       · exact mem
     exact ih (coherent_sub (head::rest) rest h (by intro a ha;simp [ha])) tail

theorem coherent_update (w : Nat) (e : BitVec 10) (xs : List ov_ent)
 (h : Coherent xs) : Coherent (update w e xs) := by
 have kept : Coherent ((xs.filter (fun x=>x.ov_k != w)).reverse) :=
   coherent_sub xs _ h (by intro a ha;exact (show a∈xs ∧ a.ov_k≠w by simpa using ha).1)
 unfold update
 split
 · exact kept
 · intro a ha b hb same
   rcases List.mem_cons.mp ha with rfl|ha
   · rcases List.mem_cons.mp hb with rfl|hb
     · rfl
     · have other := (show b∈xs ∧ b.ov_k≠w by simpa using hb).2
       simp only [bne_iff_ne] at other
       exact False.elim (other same.symm)
   · rcases List.mem_cons.mp hb with rfl|hb
     · have other := (show a∈xs ∧ a.ov_k≠w by simpa using ha).2
       simp only [bne_iff_ne] at other
       exact False.elim (other same)
     · exact kept a ha b hb same

theorem lookup_update (w key : Nat) (e : BitVec 10) (xs : List ov_ent)
 (h : Coherent xs) : lookup key (update w e xs)=if key=w then e else lookup key xs := by
 have kept : Coherent ((xs.filter (fun x=>x.ov_k != w)).reverse) :=
   coherent_sub xs _ h (by intro a ha;exact (show a∈xs ∧ a.ov_k≠w by simpa using ha).1)
 have notkey : ∀ a,a∈((xs.filter (fun x=>x.ov_k != w)).reverse) → a.ov_k≠w := by
   intro a ha
   exact by simpa only [bne_iff_ne] using (show a∈xs ∧ a.ov_k≠w by simpa using ha).2
 by_cases eq : key=w
 · subst key
   unfold update
   split
   · rename_i zero
     rw [lookup_zero w _ notkey]
     simp [zero]
   · simp [lookup]
 · have equal : lookup key ((xs.filter (fun x=>x.ov_k != w)).reverse)=lookup key xs := by
     by_cases found : ∃ a,a∈xs ∧ a.ov_k=key
     · obtain ⟨a,member,ka⟩:=found
       have remains : a∈((xs.filter (fun x=>x.ov_k != w)).reverse) := by
         simp [member,ka,eq]
       exact (lookup_member key _ kept a remains ka).trans (lookup_member key xs h a member ka).symm
     · have none : ∀ a,a∈xs → a.ov_k≠key := by
         intro a ha same
         exact found ⟨a,ha,same⟩
       rw [lookup_zero key xs none,lookup_zero key _ (fun a ha=>none a
         (show a∈xs ∧ a.ov_k≠w by simpa using ha).1)]
   unfold update
   split
   · simpa [eq] using equal
   · simpa [lookup,eq,Ne.symm eq] using equal

def iterate {α : Type} (f : α → α) : Nat → α → α
 | 0,x => x
 | n+1,x => iterate f n (f x)
theorem range_pure_loop {α : Type} (f : α → α) (range : IntRange)
 (one : range.step=1) (value : α) (i : Int) (hs : (i-range.start)%range.step=0)
 (start : range.start≤i) :
 IntRange.forIn'.loop (m:=SailM) range (fun _ _ v=>pure (.yield (f v))) value i hs =
 pure (iterate f (range.stop - i+1).toNat value) := by
 induction value,i,hs using IntRange.forIn'.loop.induct range with
 | case1 value i hs hi ih =>
   have inside:=hi
   simp [Membership.mem,one] at inside
   rw [IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,pure_bind]
   have n : (range.stop - i+1).toNat=(range.stop-(i+range.step)+1).toNat+1 := by omega
   rw [n]
   simpa only [iterate] using ih (f value) (by omega)
 | case2 value i hs hi =>
   have outside:=hi
   simp [Membership.mem,one] at outside
   have n : (range.stop - i+1).toNat=0 := by omega
   rw [IntRange.forIn'.loop.eq_1]
   simp [hi,n,iterate]

theorem range_pure {α : Type} (f : α → α) (range : IntRange) (one : range.step=1) (value : α) :
 IntRange.forIn' (m:=SailM) range value (fun _ _ v=>pure (.yield (f v))) =
 pure (iterate f (range.stop-range.start+1).toNat value) := by
 unfold IntRange.forIn'
 exact range_pure_loop f range one value range.start (by simp) (by omega)

theorem actual_list_len {α : Type} (xs : List α) : list_len xs=xs.length := by
 induction xs with
 | nil => rfl
 | cons x xs ih => simp only [list_len,ih,List.length_cons];omega

def getStep (w : Nat) (s : BitVec 10 × Bool × List ov_ent) : BitVec 10 × Bool × List ov_ent :=
 let (e,go,xs):=s
 if go then match xs with
 | [] => (e,false,xs)
 | ent::rest => if ent.ov_k=w then (ent.ov_v,false,xs) else (e,go,rest)
 else s

theorem get_stopped (w n : Nat) (e : BitVec 10) (xs : List ov_ent) :
 iterate (getStep w) n (e,false,xs)=(e,false,xs) := by
 induction n with
 | zero => rfl
 | succ n ih => simpa [iterate,getStep] using ih

theorem get_count (w : Nat) (xs : List ov_ent) :
 (iterate (getStep w) xs.length (0#10,true,xs)).1=lookup w xs := by
 induction xs with
 | nil => rfl
 | cons ent rest ih =>
   by_cases key : ent.ov_k=w
   · simp [iterate,getStep,key,get_stopped,lookup]
   · simpa [iterate,getStep,key,lookup] using ih

def putStep (w : Nat) (s : List ov_ent × List ov_ent) : List ov_ent × List ov_ent :=
 let (out,xs):=s
 match xs with
 | [] => s
 | ent::rest => (if ent.ov_k≠w then ent::out else out,rest)

theorem put_count (w : Nat) (xs out : List ov_ent) :
 iterate (putStep w) xs.length (out,xs)=((xs.filter (fun x=>x.ov_k != w)).reverse++out,[]) := by
 induction xs generalizing out with
 | nil => rfl
 | cons ent rest ih =>
   by_cases key : ent.ov_k=w
   · simpa [iterate,putStep,key] using ih out
   · simpa [iterate,putStep,key,List.reverse_cons,List.append_assoc] using ih (ent::out)


def getter (q w : Nat) : SailM (BitVec 10) := do
 let xs := (← Out.readReg Register.iq_wov)[q]!
 let n:=list_len ((← Out.readReg Register.iq_wov)[q]!)
 let result ← IntRange.forIn' {start:=0,stop:=(n:Int)-1,step:=1,step_pos:=by decide}
   (0#10,true,xs) (fun _ _ v=>pure (.yield (getStep w v)))
 pure result.1

theorem actual_getter_factor (q w : Nat) : iq_word_ov q w=getter q w := by
 simp only [iq_word_ov,getter,getStep,ForIn.forIn,ForIn'.forIn',instForInOfForIn',
   beq_iff_eq,bind_assoc,pure_bind,bind_pure,Prod.eta]
 rfl

theorem actual_getter (s : SequentialState RegisterType trivialChoiceSource)
 (W : Vector (List ov_ent) 145) (present : s.regs.get? Register.iq_wov=some W)
 (q w : Nat) : (iq_word_ov q w).run s=.ok (lookup w W[q]!) s := by
 rw [actual_getter_factor]
 have hr : Out.readReg Register.iq_wov s=.ok W s := by
   simp [Out.readReg,PreSail.readReg,present,EStateM.run,bind,EStateM.bind,
    get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure]
 simp only [getter,EStateM.run,bind,EStateM.bind,hr]
 have rangeEq : ((list_len W[q]! :Int)-1-0+1).toNat=W[q]!.length := by
   rw [actual_list_len]
   omega
 have rp := range_pure (getStep w) {start:=0,stop:=(list_len W[q]!:Int)-1,step:=1,step_pos:=by decide} rfl (0#10,true,W[q]!)
 rw [rp]
 simp only [rangeEq,pure_bind]
 simp only [pure,EStateM.pure,get_count]


def setter (q w : Nat) (e : BitVec 10) : SailM Unit := do
 let xs:=(← Out.readReg Register.iq_wov)[q]!
 let n:=list_len ((← Out.readReg Register.iq_wov)[q]!)
 let pair ← IntRange.forIn' {start:=0,stop:=(n:Int)-1,step:=1,step_pos:=by decide}
   ([],xs) (fun _ _ v=>pure (.yield (putStep w v)))
 let out ← if e≠0#10 then do
   Out.writeReg Register.iq_ecc_live true
   pure ({ov_k:=w,ov_v:=e}::pair.1)
  else pure pair.1
 Out.writeReg Register.iq_wov (vectorUpdate (← Out.readReg Register.iq_wov) q out)

theorem bind_ite {α β : Type} (p : Prop) [Decidable p] (a b : SailM α) (f : α → SailM β) :
 (if p then a else b) >>= f = if p then a >>= f else b >>= f := by split <;> rfl

theorem actual_setter_factor (q w : Nat) (e : BitVec 10) : iq_word_ov_set q w e=setter q w e := by
 simp only [iq_word_ov_set,setter,putStep,ForIn.forIn,ForIn'.forIn',instForInOfForIn',
   bne_iff_ne,bind_assoc,pure_bind,bind_pure,Prod.eta,bind_ite]
 rfl

theorem actual_setter (s : SequentialState RegisterType trivialChoiceSource)
 (W : Vector (List ov_ent) 145) (present : s.regs.get? Register.iq_wov=some W)
 (q w : Nat) (e : BitVec 10) :
 let middle:=if e=0#10 then s else {s with regs:=s.regs.insert Register.iq_ecc_live true}
 (iq_word_ov_set q w e).run s=.ok ()
   {middle with regs:=middle.regs.insert Register.iq_wov (vectorUpdate W q (update w e W[q]!))} := by
 rw [actual_setter_factor]
 have rp := range_pure (putStep w) {start:=0,stop:=(list_len W[q]!:Int)-1,step:=1,step_pos:=by decide} rfl ([],W[q]!)
 have rangeEq : ((list_len W[q]! :Int)-1-0+1).toNat=W[q]!.length := by
   rw [actual_list_len]
   omega
 simp only [rangeEq] at rp
 have hr : Out.readReg Register.iq_wov s=.ok W s := by
   simp [Out.readReg,PreSail.readReg,present,EStateM.run,bind,EStateM.bind,
    get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure]
 dsimp only
 simp only [setter,EStateM.run,bind,EStateM.bind,hr]
 rw [rp]
 simp only [pure,EStateM.pure,put_count,List.append_nil]
 by_cases zero : e=0#10
 · simp [Std.ExtDHashMap.get?_insert,zero,update,Out.writeReg,PreSail.writeReg,Out.readReg,PreSail.readReg,present,
     EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
     set,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,MonadStateOf.set,EStateM.set]
 · simp [Std.ExtDHashMap.get?_insert,zero,update,Out.writeReg,PreSail.writeReg,Out.readReg,PreSail.readReg,present,
     EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
     set,modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,MonadStateOf.set,EStateM.set]

theorem vector_update_get {α : Type} [Inhabited α] (W : Vector α 145)
 (q r : Nat) (value : α) (hq : q<145) (hr : r<145) :
 (vectorUpdate W q value)[r]! = if q=r then value else W[r]! := by
 simp [vectorUpdate,Vector.set!,hq,hr,getElem!_pos,Vector.getElem_set,Array.getElem_setIfInBounds]

theorem actual_clear (s : SequentialState RegisterType trivialChoiceSource)
 (W : Vector (List ov_ent) 145) (present : s.regs.get? Register.iq_wov=some W)
 (coherent : ∀ q,q<145 → Coherent W[q]!) (q w : Nat) (hq : q<145) :
 let after:={s with regs:=s.regs.insert Register.iq_wov (vectorUpdate W q (update w (0#10) W[q]!))}
 (iq_word_ov_set q w (0#10)).run s=.ok () after ∧
 ∀ r,r<145 → ∀ key,(iq_word_ov r key).run after=
   .ok (if r=q ∧ key=w then 0#10 else lookup key W[r]!) after := by
 dsimp only
 refine ⟨by simpa using actual_setter s W present q w (0#10),?_⟩
 intro r hr key
 have has : (s.regs.insert Register.iq_wov (vectorUpdate W q (update w (0#10) W[q]!))).get?
    Register.iq_wov=some (vectorUpdate W q (update w (0#10) W[q]!)) := by
   simp [Std.ExtDHashMap.get?_insert]
 have get:=actual_getter {s with regs:=s.regs.insert Register.iq_wov (vectorUpdate W q (update w (0#10) W[q]!))} (vectorUpdate W q (update w (0#10) W[q]!)) has r key
 rw [vector_update_get W q r _ hq hr] at get
 by_cases same : q=r
 · subst r
   simpa [lookup_update w key (0#10) W[q]! (coherent q hq)] using get
 · simpa [same,Ne.symm same] using get
end Proofs.OverlayHistory




