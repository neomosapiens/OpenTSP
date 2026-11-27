-- Actual full patrol-cycle no-effect theorem, lifted through its production inclusive loop.
import Proofs.Resources
set_option autoImplicit false
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.Resources

namespace Proofs.Patrol

theorem intRange_identity (range : IntRange) (s : State)
    (f : (i : Int) → i ∈ range → Unit → SailM (ForInStep Unit))
    (hf : ∀ i hi, (f i hi ()).run s = EStateM.Result.ok (.yield ()) s) :
    (IntRange.forIn' range () f).run s = EStateM.Result.ok () s := by
  unfold IntRange.forIn'
  apply IntRange.forIn'.loop.induct range
    (motive := fun b i hs => (IntRange.forIn'.loop range f b i hs).run s = EStateM.Result.ok () s)
  · intro b i hs hi ih
    cases b
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hi, EStateM.run, bind, EStateM.bind]
    have hfi := hf i hi
    change f i hi () s = EStateM.Result.ok (.yield ()) s at hfi
    rw [hfi]
    exact ih ()
  · intro b i hs hi
    cases b
    rw [IntRange.forIn'.loop.eq_def]
    simp [hi, EStateM.run, pure, EStateM.pure]

/-- When every MEM slice's actual ownership query says software owns the cycle,
    the complete production patrol cycle returns unchanged state. No SRAM callback
    assumptions are needed: no patrol read, repair write, report or revision change runs. -/
theorem scrub_cycle_software_no_effect (m : Nat) (s : State)
    (P : Vector (Option scrub_write) 88) (R : Vector Nat 88)
    (hP : s.regs.get? Register.scrub_pending = some P)
    (hR : s.regs.get? Register.sram_revision = some R)
    (hOwned : ∀ q : Nat, q < 88 → (scrub_issued q m).run s = EStateM.Result.ok true s) :
    (scrub_cycle m).run s = EStateM.Result.ok () s := by
  unfold scrub_cycle
  simp only [bind_pure]
  apply intRange_identity
  intro i hi
  have hiNat : i.toNat < 88 := by
    simp only [Membership.mem] at hi
    simp at hi
    omega
  simp [Out.readReg, PreSail.readReg, hP, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get]
  cases h : P[i]! <;> simp [EStateM.bind, EStateM.get, EStateM.pure, hR]
  all_goals
    have ho := hOwned i.toNat hiNat
    change scrub_issued i.toNat m s = EStateM.Result.ok true s at ho
    rw [ho]
  all_goals simp [scrub_arbitrate, EStateM.bind, EStateM.pure]

end Proofs.Patrol
