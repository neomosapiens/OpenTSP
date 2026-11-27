-- Concrete generated production SRF register transitions. Trace callbacks remain an explicit external trust boundary.
import Proofs.Common
set_option autoImplicit false
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
namespace Proofs.SrfState
abbrev State := SequentialState RegisterType trivialChoiceSource

theorem reset_frame (s : State) :
    (srf_reset ()).run s = EStateM.Result.ok ()
      {s with regs := ((((s.regs.insert Register.srf_reps (vectorInit empty_rep)).insert
        Register.srf_rep_n 0).insert Register.srf_e
        (vectorInit (vectorInit (vectorInit empty_write)))).insert Register.srf_w
        (vectorInit (vectorInit (vectorInit empty_write))))} := by
  simp [srf_reset, Out.writeReg, PreSail.writeReg, EStateM.run, bind, EStateM.bind,
    modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet]

theorem get_stream_e_run (s : State) (E : Vector srf_stream 32) (i : Nat) (_hi : i < 32)
    (hE : s.regs.get? Register.srf_e = some E) :
    (get_stream .RowE i).run s = EStateM.Result.ok E[i]! s := by
  simp [get_stream, Out.readReg, PreSail.readReg, hE, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get]

theorem get_stream_w_run (s : State) (W : Vector srf_stream 32) (i : Nat) (_hi : i < 32)
    (hW : s.regs.get? Register.srf_w = some W) :
    (get_stream .RowW i).run s = EStateM.Result.ok W[i]! s := by
  simp [get_stream, Out.readReg, PreSail.readReg, hW, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get]

theorem set_stream_e_frame (s : State) (E : Vector srf_stream 32) (i : Nat) (v : srf_stream) (_hi : i < 32)
    (hE : s.regs.get? Register.srf_e = some E) :
    (set_stream .RowE i v).run s = EStateM.Result.ok ()
      {s with regs := s.regs.insert Register.srf_e (vectorUpdate E i v)} := by
  simp [set_stream, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hE,
    EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get,
    EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet]

theorem set_stream_w_frame (s : State) (W : Vector srf_stream 32) (i : Nat) (v : srf_stream) (_hi : i < 32)
    (hW : s.regs.get? Register.srf_w = some W) :
    (set_stream .RowW i v).run s = EStateM.Result.ok ()
      {s with regs := s.regs.insert Register.srf_w (vectorUpdate W i v)} := by
  simp [set_stream, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg, hW,
    EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get,
    EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet]

/-- Exact public write transition, preserving every unrelated model register. The
    pure trace callback's external host observation is not proved by this statement. -/
theorem write_e_frame (st : State) (E : Vector srf_stream 32) (i g c : Nat)
    (_hi : i < 32) (_hg : g < 44) (v : vec) (ov : ecc_ov)
    (hE : st.regs.get? Register.srf_e = some E) :
    (srf_write_ov .RowE i g c v ov).run st =
      EStateM.Result.ok (srf_slot_collides E[i]![(slot_of ((g : Int) - c))]! g ((g : Int) - c))
        {st with regs := st.regs.insert Register.srf_e (vectorUpdate E i
          (vectorUpdate E[i]! (slot_of ((g : Int) - c))
            (srf_slot_write E[i]![(slot_of ((g : Int) - c))]! g ((g : Int) - c) v ov)))} := by
  simp [srf_write_ov, get_stream, set_stream, Out.readReg, Out.writeReg,
    PreSail.readReg, PreSail.writeReg, hE, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get,
    modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet]

theorem write_w_frame (st : State) (W : Vector srf_stream 32) (i g c : Nat)
    (_hi : i < 32) (_hg : g < 44) (v : vec) (ov : ecc_ov)
    (hW : st.regs.get? Register.srf_w = some W) :
    (srf_write_ov .RowW i g c v ov).run st =
      EStateM.Result.ok (srf_slot_collides W[i]![(slot_of ((g : Int) - c))]! g ((g : Int) - c))
        {st with regs := st.regs.insert Register.srf_w (vectorUpdate W i
          (vectorUpdate W[i]! (slot_of ((g : Int) - c))
            (srf_slot_write W[i]![(slot_of ((g : Int) - c))]! g ((g : Int) - c) v ov)))} := by
  simp [srf_write_ov, get_stream, set_stream, Out.readReg, Out.writeReg,
    PreSail.readReg, PreSail.writeReg, hW, EStateM.run, bind, EStateM.bind,
    pure, EStateM.pure, get, getThe, MonadStateOf.get, EStateM.get,
    modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet]

/-- Actual slot replacement makes exactly the chained value/overlay visible. -/
theorem slot_write_chain (slot : srf_slot) (g : Nat) (hg : g < 44)
    (d : Int) (v : vec) (ov : ecc_ov) :
    srf_slot_chain_ov (srf_slot_write slot g d v ov) g d = (true, v, ov) := by
  simp [srf_slot_chain_ov, srf_slot_write, vectorUpdate, hg, Vector.set!]

/-- Actual E-row chained read is register-preserving and has no callback premise. -/
theorem chain_e_run (st : State) (E : Vector srf_stream 32) (i g : Nat) (t : Int) (_hi : i < 32) (_hg : g < 44)
    (hE : st.regs.get? Register.srf_e = some E) :
    (srf_read_chain_ov .RowE i g t).run st = EStateM.Result.ok
      (let (found, value, overlay) := srf_slot_chain_ov E[i]![slot_of ((g : Int) - t + 1)]! g ((g : Int) - t + 1)
       if found then (.SRF_FOUND, value, overlay) else (.SRF_NONE, BitVec.zero _, ecc_ov_zero)) st := by
  have hget := get_stream_e_run st E i _hi hE
  change get_stream .RowE i st = EStateM.Result.ok E[i]! st at hget
  simp only [srf_read_chain_ov, EStateM.run, bind, EStateM.bind]
  rw [hget]
  simp only [pure, EStateM.pure]
  generalize hval : srf_slot_chain_ov E[i]![slot_of ((g : Int) - t + 1)]! g ((g : Int) - t + 1) = vals
  rcases vals with ⟨found, value, overlay⟩
  cases found <;> rfl

/-- Complete public write -> chained read next cycle, with collision result and
    exact post-write state retained. All other stream/register contents are arbitrary. -/
theorem write_e_then_chain (st : State) (E : Vector srf_stream 32) (i g c : Nat)
    (hi : i < 32) (hg : g < 44) (v : vec) (ov : ecc_ov)
    (hE : st.regs.get? Register.srf_e = some E) :
    ∃ after,
      (srf_write_ov .RowE i g c v ov).run st = EStateM.Result.ok
        (srf_slot_collides E[i]![slot_of ((g : Int) - c)]! g ((g : Int) - c)) after ∧
      (srf_read_chain_ov .RowE i g ((c : Int) + 1)).run after =
        EStateM.Result.ok (.SRF_FOUND, v, ov) after := by
  let d : Int := (g : Int) - c
  let updated := vectorUpdate E i (vectorUpdate E[i]! (slot_of d)
    (srf_slot_write E[i]![slot_of d]! g d v ov))
  let after : State := {st with regs := st.regs.insert Register.srf_e updated}
  refine ⟨after, write_e_frame st E i g c hi hg v ov hE, ?_⟩
  have hreg : after.regs.get? Register.srf_e = some updated := by
    simp [after]
  rw [chain_e_run after updated i g ((c : Int) + 1) hi hg hreg]
  have hd : (g : Int) - ((c : Int) + 1) + 1 = d := by dsimp [d]; omega
  simp only [hd]
  have hslot : slot_of d < 64 := by
    have h0 := Int.emod_nonneg d (by decide : (64 : Int) ≠ 0)
    have h1 := Int.emod_lt_of_pos d (by decide : (0 : Int) < 64)
    change (d % 64).toNat < 64
    omega
  have hlook : srf_slot_chain_ov updated[i]![slot_of d]! g d = (true, v, ov) := by
    simpa [updated, vectorUpdate, hi, hslot, Vector.set!] using
      slot_write_chain E[i]![slot_of d]! g hg d v ov
  rw [hlook]
  rfl

theorem chain_w_run (st : State) (W : Vector srf_stream 32) (i g : Nat) (t : Int) (_hi : i < 32) (_hg : g < 44)
    (hW : st.regs.get? Register.srf_w = some W) :
    (srf_read_chain_ov .RowW i g t).run st = EStateM.Result.ok
      (let (found, value, overlay) := srf_slot_chain_ov W[i]![slot_of ((g : Int) - t + 1)]! g ((g : Int) - t + 1)
       if found then (.SRF_FOUND, value, overlay) else (.SRF_NONE, BitVec.zero _, ecc_ov_zero)) st := by
  have hget := get_stream_w_run st W i _hi hW
  change get_stream .RowW i st = EStateM.Result.ok W[i]! st at hget
  simp only [srf_read_chain_ov, EStateM.run, bind, EStateM.bind]
  rw [hget]
  simp only [pure, EStateM.pure]
  generalize hval : srf_slot_chain_ov W[i]![slot_of ((g : Int) - t + 1)]! g ((g : Int) - t + 1) = vals
  rcases vals with ⟨found, value, overlay⟩
  cases found <;> rfl

theorem write_w_then_chain (st : State) (W : Vector srf_stream 32) (i g c : Nat)
    (hi : i < 32) (hg : g < 44) (v : vec) (ov : ecc_ov)
    (hW : st.regs.get? Register.srf_w = some W) :
    ∃ after,
      (srf_write_ov .RowW i g c v ov).run st = EStateM.Result.ok
        (srf_slot_collides W[i]![slot_of ((g : Int) - c)]! g ((g : Int) - c)) after ∧
      (srf_read_chain_ov .RowW i g ((c : Int) + 1)).run after =
        EStateM.Result.ok (.SRF_FOUND, v, ov) after := by
  let d : Int := (g : Int) - c
  let updated := vectorUpdate W i (vectorUpdate W[i]! (slot_of d)
    (srf_slot_write W[i]![slot_of d]! g d v ov))
  let after : State := {st with regs := st.regs.insert Register.srf_w updated}
  refine ⟨after, write_w_frame st W i g c hi hg v ov hW, ?_⟩
  have hreg : after.regs.get? Register.srf_w = some updated := by
    simp [after]
  rw [chain_w_run after updated i g ((c : Int) + 1) hi hg hreg]
  have hd : (g : Int) - ((c : Int) + 1) + 1 = d := by dsimp [d]; omega
  simp only [hd]
  have hslot : slot_of d < 64 := by
    have h0 := Int.emod_nonneg d (by decide : (64 : Int) ≠ 0)
    have h1 := Int.emod_lt_of_pos d (by decide : (0 : Int) < 64)
    change (d % 64).toNat < 64
    omega
  have hlook : srf_slot_chain_ov updated[i]![slot_of d]! g d = (true, v, ov) := by
    simpa [updated, vectorUpdate, hi, hslot, Vector.set!] using
      slot_write_chain W[i]![slot_of d]! g hg d v ov
  rw [hlook]
  rfl

#print axioms reset_frame
#print axioms get_stream_e_run
#print axioms get_stream_w_run
#print axioms set_stream_e_frame
#print axioms set_stream_w_frame
#print axioms write_e_frame
#print axioms write_w_frame
#print axioms slot_write_chain
#print axioms chain_e_run
#print axioms write_e_then_chain
#print axioms chain_w_run
#print axioms write_w_then_chain
end Proofs.SrfState
