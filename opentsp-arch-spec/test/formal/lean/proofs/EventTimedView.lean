import Proofs.EventBoundaries
import Proofs.MachineInit
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventBoundaryProofs
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace EventTimedViewProofs
abbrev St := SequentialState RegisterType trivialChoiceSource
abbrev Values := (r : Register) → RegisterType r
def Present (s : St) (v : Values) : Prop := ∀ r, s.regs.get? r=some (v r)

def atTime (v : Values) (q c : Nat) : hold_view := {
  vparked:=in_barrier_window (v .parked_since)[q]! (v .release_at)[q]! c ||
    ((q != 144) && in_barrier_window (v .parked_since)[144]! (v .release_at)[144]! c)
  vhost:=in_barrier_window (v .hw_since)[q]! (v .hw_release)[q]! c
  vbusy:=decide (c<(v .queue_busy_until)[q]!)
  vrepeat:=((v .queue_repeat)[q]! != none)
  vpoll:=(match (v .queue_poll)[q]! with | some _=>true | none=>false)
  vp:=path_views
    {pbusy:=decide (c<(v .path_busy_until)[2*q]!),ptail:=decide (c<(v .path_tail_until)[2*q]!),pstate:=(v .agts)[q]!.state}
    {pbusy:=decide (c<(v .path_busy_until)[2*q+1]!),ptail:=decide (c<(v .path_tail_until)[2*q+1]!),pstate:=(v .agts_w)[q]!.state}}

theorem complete_values (s : St) (hc : MachineInitProofs.Complete s) :
    ∃v,Present s v := by
  classical
  exact ⟨fun r=>Classical.choose (hc r),fun r=>Classical.choose_spec (hc r)⟩

theorem actual_view (s : St) (v : Values) (hv : Present s v) (q c : Nat) :
    (hold_view_of q c).run s=.ok (atTime v q c) s := by
  change (∀r, s.regs.get? r=some (v r)) at hv
  by_cases hq:q=144
  all_goals cases hpoll:(v .queue_poll)[q]!
  all_goals simp [atTime,hold_view_of,queue_parked,host_parked,parked_one,polling,path_views,path_view_of,path_busy,path_tail,agt_get,
    Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hv,hq,hpoll]
  all_goals rfl

def ends (v : Values) (q base : Nat) : List Nat :=
  [base,(v .queue_busy_until)[q]!, (v .path_busy_until)[2*q]!, (v .path_busy_until)[2*q+1]!,
    (v .path_tail_until)[2*q]!, (v .path_tail_until)[2*q+1]!] ++
    [(v .release_at)[q]!, (v .release_at)[144]!, (v .hw_release)[q]!].filterMap id

theorem end_release (v : Values) (q base : Nat) (x : Nat)
    (h : (v .release_at)[q]! = some x) : x∈ends v q base := by
  simp_all [ends]

theorem end_control_release (v : Values) (q base : Nat) (x : Nat)
    (h : (v .release_at)[144]! = some x) : x∈ends v q base := by
  simp_all [ends]

theorem end_host_release (v : Values) (q base : Nat) (x : Nat)
    (h : (v .hw_release)[q]! = some x) : x∈ends v q base := by
  simp_all [ends]

@[simp] theorem paths_zero (r w : path_view) : (path_views r w)[0]! = r := by rfl
@[simp] theorem paths_one (r w : path_view) : (path_views r w)[1]! = w := by rfl

@[simp] theorem paths_zero_get (r w : path_view) : (path_views r w)[0] = r := by rfl
@[simp] theorem paths_one_get (r w : path_view) : (path_views r w)[1] = w := by rfl

theorem candidate_view_monotone (v : Values) (q base t : Nat) (hb : base≤t) :
    ViewLE (atTime v q (floor base (ends v q base) t)) (atTime v q t) := by
  let es:=ends v q base
  let k:=floor base es t
  have hqrel := candidate_barrier_monotone (v .parked_since)[q]! (v .release_at)[q]!
    base t es hb (fun x hx=>end_release v q base x hx)
  have hcrel := candidate_barrier_monotone (v .parked_since)[144]! (v .release_at)[144]!
    base t es hb (fun x hx=>end_control_release v q base x hx)
  have hhrel := candidate_barrier_monotone (v .hw_since)[q]! (v .hw_release)[q]!
    base t es hb (fun x hx=>end_host_release v q base x hx)
  have hqb:(v .queue_busy_until)[q]!∈es:=by simp [es,ends]
  have hp0:(v .path_busy_until)[2*q]!∈es:=by simp [es,ends]
  have hp1:(v .path_busy_until)[2*q+1]!∈es:=by simp [es,ends]
  have ht0:(v .path_tail_until)[2*q]!∈es:=by simp [es,ends]
  have ht1:(v .path_tail_until)[2*q+1]!∈es:=by simp [es,ends]
  have hbusy:=floor_threshold base t _ es hb hqb
  have hb0:=floor_threshold base t _ es hb hp0
  have hb1:=floor_threshold base t _ es hb hp1
  have htail0:=floor_threshold base t _ es hb ht0
  have htail1:=floor_threshold base t _ es hb ht1
  refine ⟨?_,?_,?_,rfl,rfl,?_,?_,?_⟩
  · simp only [atTime, Bool.or_eq_true, Bool.and_eq_true] at *
    grind
  · exact hhrel
  · simpa only [atTime,decide_eq_true_eq] using hbusy.mp
  · intro p
    cases p with
    | zero => simpa [atTime,es] using hb0.mp
    | succ p =>
      cases p with
      | zero => simpa [atTime,es] using hb1.mp
      | succ p => simp [atTime,path_views,vectorInit,vectorUpdate,getElem!_def]
  · intro p
    cases p with
    | zero => simpa [atTime,es] using htail0.mp
    | succ p =>
      cases p with
      | zero => simpa [atTime,es] using htail1.mp
      | succ p => simp [atTime,path_views,vectorInit,vectorUpdate,getElem!_def]
  · intro p
    cases p with
    | zero => simp [atTime]
    | succ p =>
      cases p with
      | zero => simp [atTime]
      | succ p => simp [atTime,path_views,vectorInit,vectorUpdate,getElem!_def]

#print axioms candidate_view_monotone

#print axioms complete_values
#print axioms actual_view
#print axioms end_release
#print axioms end_control_release
#print axioms end_host_release
end EventTimedViewProofs
