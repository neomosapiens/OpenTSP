-- Lean counterpart of the SMT obligation p_iq_pop_conservation (model/props/props.sail), proved about the real generated `iq_pop`.
import Proofs.Common
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs

/-- What `iq_pop` does to the register file: head and length of queue q are updated, nothing else is read or written. -/
theorem iq_pop_run (q n : Nat) (s : SequentialState RegisterType trivialChoiceSource) (H L : Vector Nat 145)
    (hH : s.regs.get? Register.iq_head = some H) (hL : s.regs.get? Register.iq_len = some L) :
    (iq_pop q n).run s = EStateM.Result.ok ()
      { s with regs := (s.regs.insert Register.iq_head (vectorUpdate H q (ring (H[q]! + n)))).insert Register.iq_len
                (vectorUpdate L q (L[q]! - n)) } := by
  simp [iq_pop, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hH, hL, EStateM.run, bind,
    EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get, modify, modifyGet,
    MonadStateOf.modifyGet, EStateM.modifyGet, Std.ExtDHashMap.get?_insert]
  have h : ((H[q]! : Int) + (n : Int)).toNat = H[q]! + n := by omega
  simpa [h]

/-- The lengths add up, the head stays inside the ring, and the surviving logical index denotes the same physical slot. -/
theorem iq_pop_conservation (q n : Nat) (hq : q < 145)
    (s : SequentialState RegisterType trivialChoiceSource) (H L : Vector Nat 145)
    (hH : s.regs.get? Register.iq_head = some H) (hL : s.regs.get? Register.iq_len = some L)
    (hhead : H[q]! < iq_cap) (hold : L[q]! ≤ iq_cap) (hpop : n ≤ L[q]!) :
    ∃ s', (iq_pop q n).run s = EStateM.Result.ok () s' ∧
      ∃ H' L', s'.regs.get? Register.iq_head = some H' ∧ s'.regs.get? Register.iq_len = some L' ∧
        L'[q]! + n = L[q]! ∧ H'[q]! < iq_cap ∧
        ∀ i, i < L'[q]! → ring (H'[q]! + i) = ring (H[q]! + n + i) := by
  refine ⟨_, iq_pop_run q n s H L hH hL, vectorUpdate H q (ring (H[q]! + n)), vectorUpdate L q (L[q]! - n), ?_, ?_, ?_, ?_, ?_⟩
  · simp [Std.ExtDHashMap.get?_insert]
  · simp
  · rw [vectorUpdate_self _ _ _ hq]; omega
  · rw [vectorUpdate_self _ _ _ hq, ring_spec]
    repeat' split
    all_goals omega
  · intro i hi
    simp only [vectorUpdate_self _ _ _ hq] at hi ⊢
    simp only [ring_spec]
    repeat' split
    all_goals omega
