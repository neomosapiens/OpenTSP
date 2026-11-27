import Proofs.EventQuiet
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions EventProofs EventQuietProofs
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace EventStageQuietProofs
-- Actual end-of-cycle phase, including final writes: empty staged lists leave
-- register/memory state unchanged and do not invoke report callbacks.
theorem actual_empty_flush (s : St) (c : Nat)
    (hm : s.regs.get? Register.mxm_stage_list = some [])
    (hs : s.regs.get? Register.sxm_stage_list = some []) :
    (unit_state_flush c).run s = .ok () s := by
  have im := insert_existing s Register.mxm_stage_list [] hm
  have is := insert_existing s Register.sxm_stage_list [] hs
  simp only [unit_state_flush,mxm_bad,sxm_bad,q_in,mxm_apply_ok,sxm_apply_ok,
    unit_forIn,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
    EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,
    MonadStateOf.get,EStateM.get,modify,modifyGet,MonadStateOf.modifyGet,
    EStateM.modifyGet,hm,hs,im,is]
  rw [unit_forIn _ _ (by intro i x; cases x; rfl)]
  simp [pure,EStateM.pure,im,is]
#print axioms actual_empty_flush
end EventStageQuietProofs
