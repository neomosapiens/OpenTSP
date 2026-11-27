-- Actual RAS reporting may change its own registers and external IRQ effects;
-- only unrelated resource registers are framed.
import Proofs.ResourceFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace ResourceRasFrameProofs
open ResourceFrameProofs

def RasWrites : List Register := [Register.ras_mbe_count, Register.ras_sbe_count,
  Register.ras_err_status, Register.ras_first_valid, Register.ras_first0, Register.ras_first1,
  Register.ras_log0, Register.ras_log1, Register.ras_log_n,
  Register.csr_irq_cause, Register.csr_irq_count]

theorem actual_ras_record_other_reg (r : Register)
    (h : ∀ w, w ∈ RasWrites → r≠w) (c : Nat) (q : Int) (kind : Nat)
    (loc : BitVec 43) (sl pos : Nat) (port : RegKeeps r (host_irq_callback c 0)) :
    RegKeeps r (ras_record c q kind loc sl pos) := by
  unfold RegKeeps ras_record host_msix host_deliver
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | (apply unrelated_register_write; apply h; simp [RasWrites])
    | apply ite_keeps
    | apply bind_keeps
    | intro

theorem actual_scrub_report_other_reg (r : Register)
    (h : ∀ w, w ∈ RasWrites → r≠w) (c slice addr sl pos : Nat) (unc : Bool)
    (port : RegKeeps r (host_irq_callback c 0)) :
    RegKeeps r (ras_scrub_event c slice unc addr sl pos) := by
  unfold ras_scrub_event
  exact actual_ras_record_other_reg r h c slice _ _ sl pos port
theorem actual_ras_record_keeps_view {β : Type} (view : St → β)
    (writes : ∀ w, w ∈ RasWrites → ∀ v : RegisterType w, Keeps view (Out.writeReg w v))
    (c : Nat) (q : Int) (kind : Nat) (loc : BitVec 43) (sl pos : Nat)
    (port : Keeps view (host_irq_callback c 0)) : Keeps view (ras_record c q kind loc sl pos) := by
  unfold ras_record host_msix host_deliver
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | (apply writes; simp [RasWrites])
    | apply ite_keeps
    | apply bind_keeps
    | intro

theorem actual_scrub_report_keeps_view {β : Type} (view : St → β)
    (writes : ∀ w, w ∈ RasWrites → ∀ v : RegisterType w, Keeps view (Out.writeReg w v))
    (c slice addr sl pos : Nat) (unc : Bool)
    (port : Keeps view (host_irq_callback c 0)) : Keeps view (ras_scrub_event c slice unc addr sl pos) := by
  unfold ras_scrub_event
  exact actual_ras_record_keeps_view view writes c slice _ _ sl pos port

end ResourceRasFrameProofs
