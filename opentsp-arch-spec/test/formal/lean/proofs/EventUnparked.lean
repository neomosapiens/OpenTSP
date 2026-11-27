import Proofs.EventTimedView
import Proofs.SlotAdmission
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventBoundaryProofs EventTimedViewProofs
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace EventUnparkedProofs
abbrev St := SequentialState RegisterType trivialChoiceSource

def leave (v : Values) (k u : Nat) : Nat :=
  match (v .parked_since)[k]!, (v .release_at)[k]! with
  | some s,some r=>if s≤u ∧ u<r then r else u
  | _,_=>u

theorem leave_le (v : Values) (k u t : Nat) (hu : u≤t)
    (hf : in_barrier_window (v .parked_since)[k]! (v .release_at)[k]! t=false) :
    leave v k u≤t := by
  unfold leave
  cases hs:(v .parked_since)[k]!
  all_goals cases hr:(v .release_at)[k]!
  all_goals simp_all [in_barrier_window]
  split <;> omega

def pass (v : Values) (q u : Nat) := leave v 144 (leave v q u)
def unpark (v : Values) (q u : Nat) := pass v q (pass v q u)

theorem unpark_le (v : Values) (q u t : Nat) (hu : u≤t)
    (hq : in_barrier_window (v .parked_since)[q]! (v .release_at)[q]! t=false)
    (hc : in_barrier_window (v .parked_since)[144]! (v .release_at)[144]! t=false) :
    unpark v q u≤t := by
  unfold unpark pass
  exact leave_le v 144 _ t (leave_le v q _ t
    (leave_le v 144 _ t (leave_le v q u t hu hq) hc) hq) hc

def passAction (q u : Nat) : SailM Nat :=
 forIn [0:1:1]i u (fun (j:Int) old=>do
  let k:=if j==0 then q else 144
  let ps←Out.readReg .parked_since
  let ra←Out.readReg .release_at
  match (ps[k]!,ra[k]!) with
  | (some a,some b)=>do
    let next←if ((a ≤b old) && (old <b b)) then pure b else pure old
    pure (.yield next)
  | _=>pure (.yield old))

theorem pass_run (s : St) (v : Values) (hv : Present s v) (q u : Nat) :
 (passAction q u).run s=.ok (pass v q u) s := by
 change (∀r,s.regs.get? r=some (v r)) at hv
 have hpure : IntRange.forIn' (m:=Id) {start:=0,stop:=1,step:=1} u
   (fun j _ old=>pure (.yield (leave v (if j==0 then q else 144) old)))=pass v q u := by
   rw [ResourceScanProofs.actual_indexed_range_fold _ _ _ rfl]
   simp only [ResourceScanProofs.indexedFold]
   rfl
 rw [←hpure]
 unfold passAction
 try simp only [pure_bind,bind_pure]
 change (IntRange.forIn' (m:=SailM) {start:=0,stop:=1,step:=1} u _).run s=_
 unfold IntRange.forIn'
 apply SlotAdmissionProofs.query_loop_run
 intro j hj old
 cases hs:(v .parked_since)[if j==0 then q else 144]!
 all_goals cases hr:(v .release_at)[if j==0 then q else 144]!
 all_goals simp_all [leave,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
   get,getThe,MonadStateOf.get,EStateM.get,hv]
 all_goals first | rfl | (split <;> rfl)

@[simp] theorem choose_pure_bind {α β : Type} (cond : Prop) [Decidable cond]
    (x y : α) (f : α→SailM β) :
    ((if cond then pure x else pure y) >>= f)=(if cond then f x else f y) := by
  by_cases h:cond <;> simp [h]

theorem actual_unpark (s : St) (v : Values) (hv : Present s v) (q u : Nat) :
    (unparked_from q u).run s=.ok (unpark v q u) s := by
  have hpure : IntRange.forIn' (m:=Id) {start:=0,stop:=1,step:=1} u
    (fun _ _ old=>pure (.yield (pass v q old)))=unpark v q u := by
    rw [ResourceScanProofs.actual_indexed_range_fold _ _ _ rfl]
    rfl
  rw [←hpure]
  unfold unparked_from
  simp only [pure_bind,bind_pure]
  change (IntRange.forIn' (m:=SailM) {start:=0,stop:=1,step:=1} u _).run s=_
  unfold IntRange.forIn'
  apply SlotAdmissionProofs.query_loop_run
  intro j hj old
  have h:passAction q old s=.ok (pass v q old) s:=pass_run s v hv q old
  have hmap : ((do let y←passAction q old;pure (.yield y)) : SailM (ForInStep Nat)).run s=
      .ok (.yield (pass v q old)) s := by
    simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,h]
  simpa only [passAction,pure_bind,bind_pure,choose_pure_bind] using hmap

def stuckValue (v : Values) (k : Nat) : Bool :=
  match (v .parked_since)[k]!, (v .release_at)[k]! with
  | some _,none=>true
  | _,_=>false

def heldValue (v : Values) (q : Nat) : Bool := stuckValue v q || stuckValue v 144

theorem actual_held (s : St) (v : Values) (hv : Present s v) (q : Nat) :
    (held q).run s=.ok (heldValue v q) s := by
  change (∀r,s.regs.get? r=some (v r)) at hv
  have hpure:IntRange.forIn' (m:=Id) {start:=0,stop:=1,step:=1} false
      (fun j _ old=>pure (.yield (old || stuckValue v (if j==0 then q else 144))))=heldValue v q := by
    rw [ResourceScanProofs.actual_indexed_range_fold _ _ _ rfl]
    simp [ResourceScanProofs.indexedFold,heldValue]
  rw [←hpure]
  unfold held
  simp only [pure_bind,bind_pure]
  change (IntRange.forIn' (m:=SailM) {start:=0,stop:=1,step:=1} false _).run s=_
  unfold IntRange.forIn'
  apply SlotAdmissionProofs.query_loop_run
  intro j hj old
  cases hs:(v .parked_since)[if j==0 then q else 144]!
  all_goals cases hr:(v .release_at)[if j==0 then q else 144]!
  all_goals simp_all [stuckValue,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
    get,getThe,MonadStateOf.get,EStateM.get,hv]
  all_goals cases old <;> rfl

theorem stuck_false_of_unparked (v : Values) (k t : Nat)
    (starts : ∀a,(v .parked_since)[k]! = some a → a≤t)
    (hf : in_barrier_window (v .parked_since)[k]! (v .release_at)[k]! t=false) :
    stuckValue v k=false := by
  cases hs:(v .parked_since)[k]!
  all_goals cases hr:(v .release_at)[k]!
  all_goals simp_all [stuckValue,in_barrier_window]

theorem actual_unparked_captures (s : St) (v : Values) (hv : Present s v) (q base t : Nat)
    (hb : base≤t)
    (hq : in_barrier_window (v .parked_since)[q]! (v .release_at)[q]! t=false)
    (hc : in_barrier_window (v .parked_since)[144]! (v .release_at)[144]! t=false) :
    ∃u,(unparked_from q base).run s=.ok u s ∧ u≤t := by
  exact ⟨unpark v q base,actual_unpark s v hv q base,unpark_le v q base t hb hq hc⟩

#print axioms actual_held
#print axioms stuck_false_of_unparked
#print axioms actual_unparked_captures

#print axioms leave_le
#print axioms unpark_le
#print axioms actual_unpark
end EventUnparkedProofs
