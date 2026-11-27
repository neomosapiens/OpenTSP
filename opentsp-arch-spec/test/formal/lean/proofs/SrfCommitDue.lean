-- Actual due-commit caller, including ordered diagnostics and pending-slot clearing.
import Proofs.SrfHistoryFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfCommitDueProofs
open ResourceFrameProofs SrfCommittedHistoryProofs SrfCommitJournalProofs SrfHistoryFrameProofs
abbrev St:=ResourceFrameProofs.St

def commitTail (c:Nat) (hits:Vector Nat 145):SailM Unit:=do
 for q in [0:144:1]i do
  for _j in [1:hits[q]!:1]i do
   report q.toNat c .SRF_COLLIDE
 Out.writeReg Register.pending (vectorUpdate (←Out.readReg Register.pending) (tmod_int (c:Int) 128).toNat [])

theorem actual_due_source (c:Nat):
 commit_due c=(do
  let all:=(←Out.readReg Register.pending)[(tmod_int (c:Int) 128).toNat]!
  let hits←commit_apply c all all 0 (vectorInit 0)
  commitTail c hits):=by
 unfold commit_due commitTail
 simp only [pure_bind,bind_pure,bind_assoc]
 rfl

theorem actual_tail_frame (ports:SrfValueFrameProofs.Ports) (c:Nat) (hits:Vector Nat 145):
 Keeps SrfValueFrameProofs.RView (commitTail c hits):=by
 unfold commitTail
 repeat' first
  | exact pure_keeps _ _
  | exact read_keeps _ _
  | exact SrfValueFrameProofs.write_frame _ _ (by decide) (by decide)
  | exact SrfValueFrameProofs.report_frame ports _ _ _
  | reduce_resource_beta
  | apply forIn_keeps
  | apply bind_keeps
  | intro

theorem actual_commit_due_journal (ports:SrfValueFrameProofs.Ports)
 (c:Nat) (st:St) (h:Histories) (hh:HasRows st h) (pending:Vector (List commit) 128)
 (hp:st.regs.get? Register.pending=some pending):
 match (commit_due c).run st with
 | .ok _ after=>HasRows after (journal c pending[c%128]! pending[c%128]! 0 h)
 | .error _ after=>HasRows after (journal c pending[c%128]! pending[c%128]! 0 h):=by
 let all:=pending[c%128]!
 obtain ⟨counts,applied,ha,hs⟩:=actual_commit_apply_journal c all all 0 (vectorInit 0) st h hh
 have hread:Out.readReg Register.pending st=.ok pending st:=by
  simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,
   get,getThe,MonadStateOf.get,EStateM.get,hp]
 have hm:(tmod_int (c:Int) 128).toNat=c%128:=by
  change ((c:Int).tmod 128).toNat=c%128
  rw [Int.tmod_eq_emod_of_nonneg (by omega)]
  rfl
 have ha0:commit_apply c pending[c%128]! pending[c%128]! 0 (vectorInit 0) st=.ok counts applied:=ha
 have he:(commit_due c).run st=(commitTail c counts).run applied:=by
  rw [actual_due_source]
  simp [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,hread,hm,ha0]
 rw [he]
 have hf:=history_frame (commitTail c counts) (journal c all all 0 h) applied hs (actual_tail_frame ports c counts)
 cases hr:(commitTail c counts).run applied
 all_goals simpa only [hr] using hf

end SrfCommitDueProofs
