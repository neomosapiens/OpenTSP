-- Actual admitted checks remain admitted across the real intervening execution graph.
import Proofs.HoldExecutionFrames
import Proofs.HoldQueryRel
import Proofs.ResourceComplete
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace HoldAdmissionProofs
open ResourceFrameProofs
abbrev St:=ResourceFrameProofs.St

theorem frame_agree (s t:St) (h:HoldExecutionFrameProofs.RView t=HoldExecutionFrameProofs.RView s) :
 HoldQueryRel.Agree s t := by
 intro r hr
 have hm:r∈HoldExecutionFrameProofs.HoldRegs:=by
  simpa [HoldExecutionFrameProofs.HoldRegs,HoldQueryRel.HoldRegs] using hr
 have he:=congrArg (fun f=>f r) h
 simpa only [HoldExecutionFrameProofs.RView,hm,ite_true] using he.symm

theorem hold_after_action {α:Type} (action:SailM α)
 (hf:Keeps HoldExecutionFrameProofs.RView action)
 (s:St) (q c:Nat) (i:instruction) (path:Option Nat)
 (ha:(hold_check q c i path false).run s=.ok false s) :
 match action.run s with
 | .ok _ after=>(hold_check q c i path false).run after=.ok false after
 | .error _ after=>(hold_check q c i path false).run after=.ok false after := by
 have hp:=hf s
 cases hrun:action.run s
 all_goals simp only [hrun] at hp
 all_goals exact HoldQueryRel.successful_value_transfer _ (HoldQueryRel.hold_check_query q c i path false) s _ (frame_agree s _ hp) false ha

def unitVerdict (q:Nat) (i:instruction) : Option diag :=
 if !(issued_by i (issuer_of q).unit) then
  some (match i with | .ILLEGAL32 _=>.DEC_OPCODE | .ILLEGAL64 _=>.DEC_OPCODE | _=>.QUE_UNIT)
 else none

theorem actual_dispatch_admitted (s:St) (q c:Nat) (i:instruction) (path:Option Nat)
 (ha:(hold_check q c i path false).run s=.ok false s) :
 (dispatch_check q c i path).run s=.ok (unitVerdict q i) s := by
 have hh:hold_check q c i path false s=.ok false s:=ha
 cases hunit:issued_by i (issuer_of q).unit
 all_goals simp [dispatch_check,unitVerdict,Out.Functions.not,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hh,hunit]
 all_goals rfl

theorem actual_execution_keeps_grant (ports:HoldExecutionFrameProofs.Ports)
 (s:St) (q c:Nat) (i:instruction) (path:Option Nat)
 (ha:(hold_check q c i path false).run s=.ok false s)
 (producerq:Nat) (producer:instruction) (c0 k:Nat) (off soff:Int) :
 match (run_element producerq producer c0 k off soff).run s with
 | .ok _ after=>(dispatch_check q c i path).run after=.ok (unitVerdict q i) after
 | .error _ after=>(dispatch_check q c i path).run after=.ok (unitVerdict q i) after := by
 have hp:=hold_after_action (run_element producerq producer c0 k off soff)
  (HoldExecutionFrameProofs.run_element_frame ports _ _ _ _ _ _) s q c i path ha
 cases hrun:(run_element producerq producer c0 k off soff).run s
 all_goals simp only [hrun] at hp
 all_goals exact actual_dispatch_admitted _ q c i path hp

theorem actual_report_keeps_grant (ports:HoldExecutionFrameProofs.Ports)
 (s:St) (q c:Nat) (i:instruction) (path:Option Nat)
 (ha:(hold_check q c i path false).run s=.ok false s)
 (reportq reportc:Nat) (d:diag) :
 match (diag_report reportc reportq d).run s with
 | .ok _ after=>(dispatch_check q c i path).run after=.ok (unitVerdict q i) after
 | .error _ after=>(dispatch_check q c i path).run after=.ok (unitVerdict q i) after := by
 have hp:=hold_after_action (diag_report reportc reportq d)
  (HoldExecutionFrameProofs.diag_report_frame ports _ _ _) s q c i path ha
 cases hrun:(diag_report reportc reportq d).run s
 all_goals simp only [hrun] at hp
 all_goals exact actual_dispatch_admitted _ q c i path hp

def checkValue (q:Nat) (i:instruction) (path:Option Nat) (strict:Bool) (v:hold_view) : Bool:=
 let own:=match i with | .SYNC () =>true | .NOTIFY () =>true | _=>false
 let fetch:=!strict && (match i with | .IFETCHM _=>true | .IFETCH _=>true | _=>false)
 hold_decide v own (fetch && decide (q<=87)) path (agt_config i != none)

theorem actual_check_from_view (s:St) (q c:Nat) (i:instruction) (path:Option Nat) (strict:Bool)
 (v:hold_view) (hv:(hold_view_of q c).run s=.ok v s) :
 (hold_check q c i path strict).run s=.ok (checkValue q i path strict v) s := by
 have h:hold_view_of q c s=.ok v s:=hv
 simp [hold_check,checkValue,Out.Functions.not,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,h]
 all_goals cases i <;> rfl

theorem actual_hold_view_defined (s:St) (hc:MachineInitProofs.Complete s) (q c:Nat) :
 ∃v,(hold_view_of q c).run s=.ok v s := by
 obtain ⟨r,hr⟩:=hc Register.agts
 obtain ⟨w,hw⟩:=hc Register.agts_w
 obtain ⟨pb,hpb⟩:=hc Register.path_busy_until
 obtain ⟨pt,hpt⟩:=hc Register.path_tail_until
 obtain ⟨qb,hqb⟩:=hc Register.queue_busy_until
 obtain ⟨qr,hqr⟩:=hc Register.queue_repeat
 obtain ⟨qp,hqp⟩:=hc Register.queue_poll
 obtain ⟨ps,hps⟩:=hc Register.parked_since
 obtain ⟨ra,hra⟩:=hc Register.release_at
 obtain ⟨hs,hhs⟩:=hc Register.hw_since
 obtain ⟨hx,hhx⟩:=hc Register.hw_release
 let v:hold_view:={
  vparked:=in_barrier_window ps[q]! ra[q]! c || ((q != 144) && in_barrier_window ps[144]! ra[144]! c)
  vhost:=in_barrier_window hs[q]! hx[q]! c
  vbusy:=decide (c<qb[q]!)
  vrepeat:=(qr[q]! != none)
  vpoll:=(match qp[q]! with | some _=>true | none=>false)
  vp:=path_views {pbusy:=decide (c<pb[2*q]!),ptail:=decide (c<pt[2*q]!),pstate:=r[q]!.state}
    {pbusy:=decide (c<pb[2*q+1]!),ptail:=decide (c<pt[2*q+1]!),pstate:=w[q]!.state}}
 refine ⟨v,?_⟩
 by_cases hq:q=144
 all_goals cases hpoll:qp[q]!
 all_goals simp [v,hold_view_of,queue_parked,host_parked,parked_one,polling,path_views,path_view_of,path_busy,path_tail,agt_get,
  Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
  hr,hw,hpb,hpt,hqb,hqr,hqp,hps,hra,hhs,hhx,hq,hpoll]
 all_goals rfl

theorem actual_cfg_path_defined (s:St) (hc:MachineInitProofs.Complete s) (q:Nat) :
 ∃p,(cfg_path q).run s=.ok p s := by
 obtain ⟨r,hr⟩:=hc Register.agts
 obtain ⟨w,hw⟩:=hc Register.agts_w
 obtain ⟨recs,hrecs⟩:=hc Register.agt_records
 refine ⟨cfg_path_pick (r[q]!.state==.AGT_CONFIG) (w[q]!.state==.AGT_CONFIG) recs[2*q]! recs[2*q+1]!,?_⟩
 simp [cfg_path,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
  get,getThe,MonadStateOf.get,EStateM.get,hr,hw,hrecs]
 all_goals rfl

end HoldAdmissionProofs
