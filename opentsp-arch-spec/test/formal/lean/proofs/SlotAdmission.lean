-- Selected real parcel checks, including every reference slot.
import Proofs.HoldAdmission
import Proofs.ResourceScan
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SlotAdmissionProofs
open HoldAdmissionProofs
abbrev St:=ResourceFrameProofs.St

theorem query_loop_run {α:Type} (range:IntRange)
 (body:(k:Int)→k∈range→α→SailM (ForInStep α)) (f:Int→α→α)
 (s:St) (hb:∀k hk a,(body k hk a).run s=.ok (.yield (f k a)) s)
 (state:α) (i:Int) (hs:(i-range.start)%range.step=0) :
 (IntRange.forIn'.loop (m:=SailM) range body state i hs).run s=
 .ok (IntRange.forIn'.loop (m:=Id) range (fun k _ a=>pure (.yield (f k a))) state i hs) s := by
 induction state,i,hs using IntRange.forIn'.loop.induct range with
 | case1 state i hs hi ih =>
   rw [IntRange.forIn'.loop.eq_1,IntRange.forIn'.loop.eq_1]
   simp only [hi,dite_true,pure_bind]
   have hh:=hb i hi state
   simpa only [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,show body i hi state s=.ok (.yield (f i state)) s from hh] using ih (f i state)
 | case2 state i hs hi =>
   rw [IntRange.forIn'.loop.eq_1,IntRange.forIn'.loop.eq_1]
   simp [hi,EStateM.run,pure,EStateM.pure]

def pathFor (k:Int):Nat:=if (k≥b2) then 1 else 0

def refsValue (q:Nat) (r:cycle_recs) (v:hold_view) (h:Bool) : Bool:=
 IntRange.forIn' (m:=Id) {start:=0,stop:=3,step:=1} h
  (fun k _ old=>pure (.yield (match r.refs[k]! with
   | some i=>if checkValue q i (some (pathFor k)) false v then true else old
   | none=>old)))

def refsAction (q c:Nat) (r:cycle_recs) (h:Bool):SailM Bool:=do
 let mut held:=h
 for k in [0:3:1]i do
  match r.refs[k]! with
  | some i=>if ←hold_check q c i (some (pathFor k)) false then held:=true
  | none=>pure ()
 pure held

theorem refs_run (s:St) (q c:Nat) (r:cycle_recs) (v:hold_view)
 (hv:(hold_view_of q c).run s=.ok v s) (h:Bool):
 (refsAction q c r h).run s=.ok (refsValue q r v h) s := by
 unfold refsAction refsValue
 simp only [pure_bind,bind_pure]
 unfold IntRange.forIn'
 apply query_loop_run
 intro k hk old
 cases hr:r.refs[k]! with
 | none=>simp [hr,EStateM.run,pure,EStateM.pure]
 | some i=>
   have hc:=actual_check_from_view s q c i (some (pathFor k)) false v hv
   cases hvv:checkValue q i (some (pathFor k)) false v
   all_goals simp [hr,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,show hold_check q c i (some (pathFor k)) false s=.ok (checkValue q i (some (pathFor k)) false v) s from hc,hvv]

def fetchPath (i:instruction):Option Nat:=match i with | .IFETCHM _=>some 0 | _=>none

def fetchValue (q:Nat) (r:cycle_recs) (v:hold_view):Bool:=
 match r.ifetch with
 | some i=>checkValue q i (fetchPath i) false v
 | none=>false

def cfgValue (q:Nat) (r:cycle_recs) (v:hold_view) (p:Nat):Bool:=
 match r.cfg with
 | some i=>checkValue q i (some p) false v
 | none=>false

def slotValue (q:Nat) (r:cycle_recs) (v:hold_view) (p:Nat):Bool:=
 refsValue q r v (cfgValue q r v p || fetchValue q r v)

def slotsAction (q c:Nat) (r:cycle_recs):SailM Bool:=do
 let held←(match r.ifetch with
  | some i=>do
    if ←hold_check q c i (fetchPath i) false then pure true else pure false
  | none=>pure false)
 let held←(match r.cfg with
  | some i=>do
    if ←hold_check q c i (some (←cfg_path q)) false then pure true else pure held
  | none=>pure held)
 refsAction q c r held

section
attribute [local irreducible] refsAction IntRange.forIn'
theorem actual_slots_source (q c:Nat) (r:cycle_recs) (items:List (Result instruction diag))
 (hn:r.n≠0) (hok:r.ok=true) (hs:r.single=none):
 parcel_held q c r items=slotsAction q c r := by
 unfold parcel_held slotsAction refsAction pathFor fetchPath
 simp only [hn,hok,hs,beq_iff_eq,Out.Functions.not,Bool.not_true,Bool.false_eq_true,if_false,if_true,pure_bind,bind_pure]
 congr 1
 funext h
 congr 1
 funext h
 congr 1
 funext k old
 cases hr:r.refs[k]! <;> simp only [hr,pure_bind,bind_pure,bind_assoc]
 congr 1
 funext check
 cases check <;> rfl
end

theorem slots_run (s:St) (q c:Nat) (r:cycle_recs) (v:hold_view) (p:Nat)
 (hv:(hold_view_of q c).run s=.ok v s) (hp:(cfg_path q).run s=.ok p s):
 (slotsAction q c r).run s=.ok (slotValue q r v p) s := by
 have hc:∀i path,hold_check q c i path false s=.ok (checkValue q i path false v) s:=by
  intro i path;exact actual_check_from_view s q c i path false v hv
 have cp:cfg_path q s=.ok p s:=hp
 cases hf:r.ifetch with
 | none=>
   cases hg:r.cfg with
   | none=>simpa [slotsAction,slotValue,fetchValue,cfgValue,hf,hg,pure_bind] using refs_run s q c r v hv false
   | some i=>
     have hh:=refs_run s q c r v hv (checkValue q i (some p) false v)
     cases hhcheck:checkValue q i (some p) false v
     all_goals simpa [slotsAction,slotValue,fetchValue,cfgValue,hf,hg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hc,cp,hhcheck] using hh
 | some f=>
   cases hg:r.cfg with
   | none=>
     have hh:=refs_run s q c r v hv (checkValue q f (fetchPath f) false v)
     cases hhcheck:checkValue q f (fetchPath f) false v
     all_goals simpa [slotsAction,slotValue,fetchValue,cfgValue,hf,hg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hc,hhcheck] using hh
   | some i=>
     have hh:=refs_run s q c r v hv (checkValue q i (some p) false v || checkValue q f (fetchPath f) false v)
     cases hfc:checkValue q f (fetchPath f) false v <;> cases hgc:checkValue q i (some p) false v
     all_goals simpa [slotsAction,slotValue,fetchValue,cfgValue,hf,hg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hc,cp,hfc,hgc] using hh

def refValue (q:Nat) (r:cycle_recs) (v:hold_view) (k:Int):Bool:=
 match r.refs[k]! with
 | some i=>checkValue q i (some (pathFor k)) false v
 | none=>false

theorem refs_value_closed (q:Nat) (r:cycle_recs) (v:hold_view) (h:Bool):
 refsValue q r v h = (h || refValue q r v 0 || refValue q r v 1 || refValue q r v 2 || refValue q r v 3) := by
 unfold refsValue
 rw [ResourceScanProofs.actual_indexed_range_fold _ _ _ rfl]
 change ResourceScanProofs.indexedFold _ (0:Int) (4:Nat) h=_
 have step_or:∀k:Int,∀old:Bool,
  (match r.refs[k]! with
   | some i=>if checkValue q i (some (pathFor k)) false v then true else old
   | none=>old)=(old || refValue q r v k):=by
  intro k old
  cases hr:r.refs[k]! with
  | none=> simp only [refValue,hr,Bool.or_false]
  | some i=>
    simp only [refValue,hr]
    cases checkValue q i (some (pathFor k)) false v <;> cases old <;> rfl
 simp only [ResourceScanProofs.indexedFold,step_or]
 rfl

theorem selected_checks_false (q:Nat) (r:cycle_recs) (v:hold_view) (p:Nat)
 (ha:slotValue q r v p=false):
 fetchValue q r v=false ∧ cfgValue q r v p=false ∧ ∀k:Int,0≤k→k≤3→refValue q r v k=false := by
 rw [slotValue,refs_value_closed] at ha
 have hs:=Bool.or_eq_false_iff.mp ha
 have hs2:=Bool.or_eq_false_iff.mp hs.1
 have hs1:=Bool.or_eq_false_iff.mp hs2.1
 have hs0:=Bool.or_eq_false_iff.mp hs1.1
 have hp:=Bool.or_eq_false_iff.mp hs0.1
 refine ⟨hp.2,hp.1,?_⟩
 intro k hk hl
 have hk4:k=0∨k=1∨k=2∨k=3:=by omega
 rcases hk4 with hk|hk|hk|hk
 · simpa [hk] using hs0.2
 · simpa [hk] using hs1.2
 · simpa [hk] using hs2.2
 · simpa [hk] using hs.2

theorem actual_parcel_selected_admitted (s:St) (hc:MachineInitProofs.Complete s)
 (q c:Nat) (r:cycle_recs) (items:List (Result instruction diag))
 (hn:r.n≠0) (hok:r.ok=true) (hs:r.single=none)
 (ha:(parcel_held q c r items).run s=.ok false s):
 (∀i,r.ifetch=some i→(hold_check q c i (fetchPath i) false).run s=.ok false s) ∧
 (∃p,(cfg_path q).run s=.ok p s ∧
  (∀i,r.cfg=some i→(hold_check q c i (some p) false).run s=.ok false s)) ∧
 (∀k:Int,0≤k→k≤3→∀i,r.refs[k]! = some i→(hold_check q c i (some (pathFor k)) false).run s=.ok false s) := by
 obtain ⟨v,hv⟩:=actual_hold_view_defined s hc q c
 obtain ⟨p,hp⟩:=actual_cfg_path_defined s hc q
 have hr:=slots_run s q c r v p hv hp
 rw [←actual_slots_source q c r items hn hok hs] at hr
 rw [ha] at hr
 have hvalue:slotValue q r v p=false:=by
  have he:=congrArg (fun res=>match res with | .ok b _=>b | .error _ _=>true) hr
  exact he.symm
 obtain ⟨hf,hg,hrefs⟩:=selected_checks_false q r v p hvalue
 refine ⟨?_,⟨p,hp,?_⟩,?_⟩
 · intro i hi
   have hh:=actual_check_from_view s q c i (fetchPath i) false v hv
   simp only [fetchValue,hi] at hf
   simpa only [hf] using hh
 · intro i hi
   have hh:=actual_check_from_view s q c i (some p) false v hv
   simp only [cfgValue,hi] at hg
   simpa only [hg] using hh
 · intro k hk hl i hi
   have hvv:=hrefs k hk hl
   simp only [refValue,hi] at hvv
   simpa only [hvv] using actual_check_from_view s q c i (some (pathFor k)) false v hv

end SlotAdmissionProofs
