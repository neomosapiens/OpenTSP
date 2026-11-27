-- Actual execution/report transitions preserve the same retained-writer history.
import Proofs.SrfValueFrames
import Proofs.SrfFaultHistory
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfHistoryFrameProofs
open ResourceFrameProofs SrfCommittedHistoryProofs SrfCommitJournalProofs
abbrev St:=ResourceFrameProofs.St

theorem history_frame {α:Type} (action:SailM α) (h:Histories) (st:St)
 (hs:HasRows st h) (hf:Keeps SrfValueFrameProofs.RView action):
 match action.run st with
 | .ok _ after=>HasRows after h
 | .error _ after=>HasRows after h:=by
 have hp:=hf st
 cases hr:action.run st
 all_goals simp only [hr,SrfValueFrameProofs.RView] at hp
 all_goals constructor
 all_goals first
  | exact (congrArg Prod.fst hp).trans hs.1
  | exact (congrArg Prod.snd hp).trans hs.2

theorem actual_reset_empty_history (st:St):
 ∃after,(srf_reset ()).run st=.ok () after ∧ HasRows after ([],[]):=by
 refine ⟨_,rfl,?_⟩
 constructor
 all_goals simp [rowValue,bankHistory,emptyBank,Std.ExtDHashMap.get?_insert]

theorem actual_report_preserves_history (ports:SrfValueFrameProofs.Ports)
 (st:St) (h:Histories) (hs:HasRows st h) (q c:Nat) (d:diag):
 match (report q c d).run st with
 | .ok _ after=>HasRows after h
 | .error _ after=>HasRows after h:=by
 have hh:=history_frame _ h st hs (SrfValueFrameProofs.report_frame ports q c d)
 cases hr:(report q c d).run st
 all_goals simpa only [hr] using hh

theorem actual_error_report_preserves_history (ports:SrfValueFrameProofs.Ports)
 (st:St) (h:Histories) (hs:HasRows st h) (r:row) (g c:Nat) (prio:Int)
 (xs:BitVec Out.Functions.superlanes) (sp:Vector (BitVec Out.Functions.superlanes) 30):
 match (srf_report r g c prio xs sp).run st with
 | .ok _ after=>HasRows after h
 | .error _ after=>HasRows after h:=by
 have hh:=history_frame _ h st hs (SrfValueFrameProofs.srf_report_frame ports r g c prio xs sp)
 cases hr:(srf_report r g c prio xs sp).run st
 all_goals simpa only [hr] using hh

theorem actual_execute_preserves_history (ports:SrfValueFrameProofs.Ports)
 (st:St) (h:Histories) (hs:HasRows st h) (e:exec_element):
 match (execute_element e).run st with
 | .ok _ after=>HasRows after h
 | .error _ after=>HasRows after h:=by
 have hh:=history_frame _ h st hs (SrfValueFrameProofs.execute_element_frame ports e)
 cases hr:(execute_element e).run st
 all_goals simpa only [hr] using hh

end SrfHistoryFrameProofs
