import Proofs.SrfCommitPolicy
import Proofs.SrfCommitDue
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfPolicyHistoryProofs
open SrfCommitPolicyProofs SrfCommitJournalProofs SrfCommittedHistoryProofs

/-- A history step justified by the independent cardinality/priority/merge policy.
    The relation contains no call to commit_wins or commit_scan. -/
inductive PolicyJournal (c : Nat) (all : List commit) : List commit → Nat → Histories → Histories → Prop where
 | nil (i : Nat) (h : Histories) : PolicyJournal c all [] i h h
 | cons (e : commit) (xs : List commit) (i : Nat) (h final : Histories)
   (out : Bool × Bool × vec × ecc_ov)
   (policy : GroupPolicy e i all out)
   (tail : PolicyJournal c all xs (i+1)
     (if out.2.1 then record e.r (selectedWrite c e out.2.2.1 out.2.2.2) h else h) final) :
   PolicyJournal c all (e::xs) i h final

theorem actual_journal_independent (c : Nat) (all xs : List commit) (i : Nat) (h : Histories)
 (domain : CommitDomain all) : PolicyJournal c all xs i h (journal c all xs i h) := by
 induction xs generalizing i h with
 | nil => exact .nil i h
 | cons e xs ih =>
   have hp:=actual_commit_wins_policy e i all domain
   rcases hw:commit_wins e i all with ⟨collision,win,value,ov⟩
   rw [hw] at hp
   have ht:=ih (i+1) (if win then record e.r (selectedWrite c e value ov) h else h)
   simpa only [journal,hw] using PolicyJournal.cons e xs i h _ (collision,win,value,ov) hp ht

/-- Same ACTUAL afterstate has the retained bank histories and a policy-valid
    journal; no independent afterstate or abstract whole-callee preservation. -/
theorem actual_apply_policy_history (c : Nat) (all xs : List commit) (i : Nat)
 (hits : Vector Nat 145) (st : ResourceFrameProofs.St) (h : Histories)
 (rows : HasRows st h) (domain : CommitDomain all) :
 ∃ counts after hnext,(commit_apply c all xs i hits).run st=.ok counts after ∧
 HasRows after hnext ∧ PolicyJournal c all xs i h hnext := by
 obtain ⟨counts,after,run,rowsAfter⟩:=actual_commit_apply_journal c all xs i hits st h rows
 exact ⟨counts,after,journal c all xs i h,run,rowsAfter,actual_journal_independent c all xs i h domain⟩

theorem actual_due_policy_history (ports : SrfValueFrameProofs.Ports)
 (c : Nat) (st : ResourceFrameProofs.St) (h : Histories) (rows : HasRows st h)
 (pending : Vector (List commit) 128) (hp : st.regs.get? Register.pending=some pending)
 (domain : CommitDomain pending[c%128]!) :
 let after:=match (commit_due c).run st with | .ok _ after=>after | .error _ after=>after
 ∃ hnext,HasRows after hnext ∧ PolicyJournal c pending[c%128]! pending[c%128]! 0 h hnext := by
 have hr:=SrfCommitDueProofs.actual_commit_due_journal ports c st h rows pending hp
 have hj:=actual_journal_independent c pending[c%128]! pending[c%128]! 0 h domain
 cases run:(commit_due c).run st <;>
  exact ⟨journal c pending[c%128]! pending[c%128]! 0 h,by simpa only [run] using hr,hj⟩

#print axioms actual_due_policy_history
end SrfPolicyHistoryProofs
