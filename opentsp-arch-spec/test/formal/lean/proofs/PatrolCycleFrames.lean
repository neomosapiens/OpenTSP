-- Full real patrol cycle frames for every unrelated architectural register.
import Proofs.PatrolFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions Out.Register Out.scrub_action
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace PatrolCycleFrameProofs
open ResourceFrameProofs ResourceRasFrameProofs PatrolFrameProofs

def PatrolWrites : List Register := [Register.scrub_ptr,Register.scrub_pending,Register.sram_revision] ++ RasWrites

/-- Primitive-only register frames. This is not a callback contract for scrub,
    execute, reporting or any other model function. It asserts no C value/time
    correctness or totality for an invalid address. -/
structure RawRegisterPorts (r : Register) : Prop where
  readData : ∀ s a, RegKeeps r (read_sram_word s a)
  readEcc : ∀ s a, RegKeeps r (read_sram_ecc s a)
  writeData : ∀ s a v, RegKeeps r (raw_write_sram_word s a v)
  writeEcc : ∀ s a v, RegKeeps r (raw_write_sram_ecc s a v)
  irq : ∀ c, RegKeeps r (host_irq_callback c 0)

theorem patrol_ne (r w : Register) (h : r∉PatrolWrites) (hw : w∈PatrolWrites) : r≠w := by
  intro he
  exact h (he ▸ hw)

theorem ras_subset (w : Register) (hw : w∈RasWrites) : w∈PatrolWrites := by
  exact List.mem_append_right _ hw

theorem actual_ownership_query_frame (r : Register) (slice m : Nat) : RegKeeps r (scrub_issued slice m) := by
  unfold RegKeeps scrub_issued
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | apply bind_keeps
    | split
    | intro

theorem actual_take_unrelated_reg (r : Register) (hne : r≠Register.scrub_pending)
    (slice : Nat) : RegKeeps r (scrub_take slice) := by
  unfold RegKeeps scrub_take
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | exact unrelated_register_write _ _ _ hne
    | apply ite_keeps
    | apply bind_keeps
    | split
    | intro

theorem actual_slice_unrelated_reg (r : Register) (h : r∉PatrolWrites) (ports : RawRegisterPorts r)
    (slice m : Nat) : RegKeeps r (scrub_slice slice m) := by
  apply actual_scrub_slice_keeps_view
  · intro w hw value
    exact unrelated_register_write r w value (patrol_ne r w h (ras_subset w hw))
  · exact ports.readData slice
  · exact ports.readEcc slice
  · exact ports.irq m
  · intro a w ov
    exact actual_save_repair_other_reg r (patrol_ne r Register.scrub_pending h (by simp [PatrolWrites])) slice a w ov
  · intro a
    exact actual_move_pointer_other_reg r (patrol_ne r Register.scrub_ptr h (by simp [PatrolWrites])) slice a

theorem actual_patrol_cycle_unrelated_reg (r : Register) (h : r∉PatrolWrites) (ports : RawRegisterPorts r)
    (m : Nat) : RegKeeps r (scrub_cycle m) := by
  unfold RegKeeps scrub_cycle
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | exact actual_ownership_query_frame r _ _
    | exact actual_slice_unrelated_reg r h ports _ _
    | exact actual_take_unrelated_reg r (patrol_ne r Register.scrub_pending h (by simp [PatrolWrites])) _
    | exact write_sram_word_other_reg r (patrol_ne r Register.sram_revision h (by simp [PatrolWrites])) _ _ _ (ports.writeData _ _ _)
    | exact write_sram_ecc_other_reg r (patrol_ne r Register.sram_revision h (by simp [PatrolWrites])) _ _ _ (ports.writeEcc _ _ _)
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | split
    | intro

theorem actual_patrol_cycle_scrub_next (ports : RawRegisterPorts Register.scrub_next) (m : Nat) :
    RegKeeps Register.scrub_next (scrub_cycle m) :=
  actual_patrol_cycle_unrelated_reg Register.scrub_next (by decide +kernel) ports m

end PatrolCycleFrameProofs
