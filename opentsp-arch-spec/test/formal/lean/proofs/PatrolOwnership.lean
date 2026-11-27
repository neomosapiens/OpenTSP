-- Exact ownership-query transport through unrelated slices' real patrol effects.
import Proofs.PatrolCycleFrames
import Proofs.ResourceInvariant
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace PatrolOwnershipProofs
open ResourceFrameProofs PatrolCycleFrameProofs ResourceInvariantProofs

def Ownership (s : St) := (s.regs.get? Register.path_busy_until,
  s.regs.get? Register.queue_last,s.regs.get? Register.queue_last_at)

def ownershipValue (q m : Nat) (s : St) : Option Bool := do
  let Q ← s.regs.get? Register.queue_last
  let last ← match Q[q]! with
    | none => some none
    | some _ => (s.regs.get? Register.queue_last_at).map (fun L => some L[q]!)
  let H ← s.regs.get? Register.path_busy_until
  some (scrub_slot_taken m (entry_latency "Read" .T_MEM).d_skew H[2*q]! H[2*q+1]! last)

theorem actual_ownership_run (q m : Nat) (s : St) :
    (scrub_issued q m).run s = match ownershipValue q m s with
      | some b => .ok b s
      | none => .error Error.Unreachable s := by
  have hi : (2*(q:Int)).toNat=2*q := by omega
  have hi1 : (2*(q:Int)+1).toNat=2*q+1 := by omega
  simp only [scrub_issued,ownershipValue]
  cases hQ : s.regs.get? Register.queue_last with
  | none => simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,get,getThe,
      MonadStateOf.get,EStateM.get,hQ]; rfl
  | some Q =>
    cases hq : Q[q]! <;>
      cases hH : s.regs.get? Register.path_busy_until <;>
      cases hL : s.regs.get? Register.queue_last_at <;>
      simp [Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,get,getThe,
        MonadStateOf.get,EStateM.get,pure,EStateM.pure,hQ,hq,hH,hL,hi,hi1]
    all_goals rfl

theorem ownership_transport (q m : Nat) (s t : St) (h : Ownership t=Ownership s) :
    ownershipValue q m t=ownershipValue q m s := by
  have hH := congrArg Prod.fst h
  have hQ := congrArg (fun x => x.2.1) h
  have hL := congrArg (fun x => x.2.2) h
  simp only [Ownership] at hH hQ hL
  simp only [ownershipValue,hH,hQ,hL]

theorem ownership_keeps_of_registers {α : Type} (action : SailM α)
    (hH : RegKeeps Register.path_busy_until action)
    (hQ : RegKeeps Register.queue_last action)
    (hL : RegKeeps Register.queue_last_at action) : Keeps Ownership action := by
  exact pair_keeps _ _ action hH (pair_keeps _ _ action hQ hL)

theorem actual_cycle_ownership (m : Nat)
    (hH : RawRegisterPorts Register.path_busy_until)
    (hQ : RawRegisterPorts Register.queue_last)
    (hL : RawRegisterPorts Register.queue_last_at) : Keeps Ownership (scrub_cycle m) := by
  apply ownership_keeps_of_registers
  · exact actual_patrol_cycle_unrelated_reg _ (by decide +kernel) hH m
  · exact actual_patrol_cycle_unrelated_reg _ (by decide +kernel) hQ m
  · exact actual_patrol_cycle_unrelated_reg _ (by decide +kernel) hL m
end PatrolOwnershipProofs
