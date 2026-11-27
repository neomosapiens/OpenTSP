import Out
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Out.Register Out.diag
set_option autoImplicit false
set_option maxHeartbeats 2000000
namespace Proofs.FailureEffects
abbrev St := SequentialState RegisterType trivialChoiceSource

/-- Diagnostic is a returned value, so actual retained state is passed to the
    reporting continuation. No rollback or source-state replacement occurs. -/
theorem returned_state_is_retained {α β : Type} (a : SailM α) (f : α → SailM β)
 (s t : St) (x : α) (h : a.run s = .ok x t) :
 (a >>= f).run s = (f x).run t := by
 simp only [EStateM.run, bind, EStateM.bind]
 change a s = _ at h
 simp only [h]

/-- A genuine execution failure retains its partial state and suppresses the
    continuation, independently of the state before execution. -/
theorem thrown_state_is_retained {α β : Type} (a : SailM α) (f : α → SailM β)
 (s t : St) (e : Sail.Error exception) (h : a.run s = .error e t) :
 (a >>= f).run s = .error e t := by
 simp only [EStateM.run, bind, EStateM.bind]
 change a s = _ at h
 simp only [h]

def issuePrefix (e : exec_element) : SailM Unit := do
 Out.writeReg issue_cycle e.c0
 Out.writeReg issue_unit (issuer_of e.q)
 Out.writeReg issue_queue e.q
 Out.writeReg issue_element e.k
 Out.writeReg issue_addr_offset e.off
 Out.writeReg issue_stream_offset e.soff
 Out.writeReg issue_thread 0#1
 Out.writeReg issue_lat (latency_of_instr e.instr)
 observe_site (entry_of e.instr) (← exec_cycle ()) e.q e.k 0 e.c0
 ecc_begin ()

def reportResult (e : exec_element) (result : ExecutionResult) : SailM Unit := do
 match result with
 | .Issued () =>
   if (← Out.readReg ecc_link_bad) != BitVec.zero Functions.superlanes then
     report e.q (← exec_cycle ()) LNK_UNC
   if (← Out.readReg ecc_bad) != BitVec.zero Functions.superlanes then
     report e.q (← exec_cycle ()) ECC_UNC
 | .Diagnostic d =>
   report e.q (← exec_cycle ()) d
   if (← Out.readReg ecc_after_diag) && ((← Out.readReg ecc_bad) != BitVec.zero Functions.superlanes) then
     report e.q (← exec_cycle ()) ECC_UNC

/-- The complete actual execution stage, not a replacement execution function. -/
theorem actual_execute_factor (e : exec_element) :
 execute_element e = (do
   issuePrefix e
   let result ← execute e.instr
   reportResult e result) := by
 simp only [execute_element, issuePrefix, reportResult, bind_assoc]
 rfl

/-- IRQ count is updated before invoking the real external IRQ effect. The
    external callback is retained, not replaced with a trace-only logger. -/
theorem actual_host_deliver_factor (c n : Nat) :
 host_deliver c n = (do
   Out.writeReg csr_irq_count ((← Out.readReg csr_irq_count) + csr_value 1)
   host_irq_callback c n) := rfl

theorem actual_diag_selection (c q : Nat) (d : diag) :
 diag_report c q d =
 (if diag_is_data q d then diag_record c q (diag_number d) else pure ()) := by
 rfl

-- This last equality concerns normal Lean state only: pure output callbacks
-- are erased there. Ordered external output uses the separate source-connected
-- callback/trace interpretation, never this equality as a trace erasure claim.
end Proofs.FailureEffects
