-- A protected slice's actual pending/pointer/revision projections while others act.
import Proofs.PatrolOwnership
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace PatrolProtectedProofs
open ResourceFrameProofs ResourceInvariantProofs ResourceRasFrameProofs PatrolFrameProofs PatrolCycleFrameProofs PatrolOwnershipProofs

def Cells (q : Nat) (s : St) :=
  (FieldView Register.scrub_pending (fun v => v[q]!) s,
   FieldView Register.scrub_ptr (fun v => v[q]!) s,
   FieldView Register.sram_revision (fun v => v[q]!) s)

def Protected (q : Nat) (s : St) := (Ownership s,Cells q s)

theorem protected_keeps {α : Type} (q : Nat) (action : SailM α)
    (hO : Keeps Ownership action)
    (hP : Keeps (FieldView Register.scrub_pending (fun v => v[q]!)) action)
    (hA : Keeps (FieldView Register.scrub_ptr (fun v => v[q]!)) action)
    (hR : Keeps (FieldView Register.sram_revision (fun v => v[q]!)) action) :
    Keeps (Protected q) action :=
  pair_keeps _ _ action hO (pair_keeps _ _ action hP (pair_keeps _ _ action hA hR))

theorem protected_write (q : Nat) (r : Register) (v : RegisterType r)
    (h : r∉[Register.path_busy_until,Register.queue_last,Register.queue_last_at,
      Register.scrub_pending,Register.scrub_ptr,Register.sram_revision]) :
    Keeps (Protected q) (Out.writeReg r v) := by
  apply protected_keeps
  · apply ownership_keeps_of_registers <;> exact unrelated_register_write _ _ _ (by intro he; subst r; simp_all)
  all_goals apply reg_keeps_field; exact unrelated_register_write _ _ _ (by intro he; subst r; simp_all)

theorem actual_take_other_pending (q slice : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice) :
    Keeps (FieldView Register.scrub_pending (fun v => v[q]!)) (scrub_take slice) := by
  intro state
  cases hP : state.regs.get? Register.scrub_pending with
  | none =>
    have hm := read_missing_run state Register.scrub_pending hP
    change Out.readReg Register.scrub_pending state=.error Error.Unreachable state at hm
    simp only [scrub_take,EStateM.run,bind,EStateM.bind]
    rw [hm]
  | some P =>
    have hc := vector_update_other 88 q slice P (none : Option scrub_write) hq hs hne
    cases hp : P[slice]! with
    | none =>
      simpa [FieldView,scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
        EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
        modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hP,hp] using congrArg some hc
    | some w =>
      cases hR : state.regs.get? Register.sram_revision with
      | none =>
        have he : (scrub_take slice).run state=.error Error.Unreachable
            {state with regs:=state.regs.insert Register.scrub_pending (vectorUpdate P slice none)} := by
          simp [scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
            EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
            modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hP,hp,hR]
          rfl
        rw [he]
        simpa [FieldView,Std.ExtDHashMap.get?_insert,hP] using congrArg some hc
      | some R =>
        by_cases hr : w.revision=R[slice]!
        all_goals simpa [FieldView,scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
          EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
          modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hP,hp,hR,hr] using congrArg some hc

/-- No whole scrub helper is a premise: all nonlocal writes are discharged against
    their real registers/cells and only actual external primitive frames remain. -/
theorem actual_other_slice (q slice m : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice)
    (data : ∀ a, Keeps (Protected q) (read_sram_word slice a))
    (ecc : ∀ a, Keeps (Protected q) (read_sram_ecc slice a))
    (irq : Keeps (Protected q) (host_irq_callback m 0)) :
    Keeps (Protected q) (scrub_slice slice m) := by
  apply actual_scrub_slice_keeps_view
  · intro r hr v
    apply protected_write
    simp only [RasWrites,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hr
    rcases hr with h|h|h|h|h|h|h|h|h|h|h <;> subst r <;> decide
  · exact data
  · exact ecc
  · exact irq
  · intro a w ov
    apply protected_keeps
    · apply ownership_keeps_of_registers <;> exact actual_save_repair_other_reg _ (by decide) _ _ _ _
    · exact actual_save_repair_other_cell q slice a hq hs hne w ov
    · apply reg_keeps_field; exact actual_save_repair_other_reg _ (by decide) _ _ _ _
    · apply reg_keeps_field; exact actual_save_repair_other_reg _ (by decide) _ _ _ _
  · intro a
    apply protected_keeps
    · apply ownership_keeps_of_registers <;> exact actual_move_pointer_other_reg _ (by decide) _ _
    · apply reg_keeps_field; exact actual_move_pointer_other_reg _ (by decide) _ _
    · exact actual_move_pointer_other_cell q slice a hq hs hne
    · apply reg_keeps_field; exact actual_move_pointer_other_reg _ (by decide) _ _
theorem actual_other_take (q slice : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice) :
    Keeps (Protected q) (scrub_take slice) := by
  apply protected_keeps
  · apply ownership_keeps_of_registers <;> exact actual_take_unrelated_reg _ (by decide) slice
  · exact actual_take_other_pending q slice hq hs hne
  · apply reg_keeps_field; exact actual_take_unrelated_reg _ (by decide) slice
  · apply reg_keeps_field; exact actual_take_unrelated_reg _ (by decide) slice

theorem map_keeps {α β γ : Type} (view : St → β) (f : β → γ) (action : SailM α)
    (h : Keeps view action) : Keeps (fun s => f (view s)) action := by
  intro s
  have hv := h s
  cases hr : action.run s <;> simp only [hr] at hv <;> simp only [hr] <;> exact congrArg f hv

theorem word_other_field {β : Type} (r : Register) (pick : RegisterType r → β)
    (hne : r≠Register.sram_revision) (slice a : Nat) (w : vec)
    (port : Keeps (FieldView r pick) (raw_write_sram_word slice a w)) :
    Keeps (FieldView r pick) (write_sram_word slice a w) := by
  unfold write_sram_word
  repeat' first
    | exact port
    | exact read_keeps _ _
    | exact unrelated_register_field_write _ _ _ _ hne
    | apply bind_keeps
    | intro

theorem actual_other_word (q slice a : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice) (w : vec)
    (port : Keeps (Protected q) (raw_write_sram_word slice a w)) :
    Keeps (Protected q) (write_sram_word slice a w) := by
  have hO : Keeps Ownership (raw_write_sram_word slice a w) := map_keeps (Protected q) (fun x => x.1) _ port
  apply protected_keeps
  · apply ownership_keeps_of_registers
    · exact write_sram_word_other_reg _ (by decide) _ _ _ (map_keeps Ownership (fun x => x.1) _ hO)
    · exact write_sram_word_other_reg _ (by decide) _ _ _ (map_keeps Ownership (fun x => x.2.1) _ hO)
    · exact write_sram_word_other_reg _ (by decide) _ _ _ (map_keeps Ownership (fun x => x.2.2) _ hO)
  · exact word_other_field _ _ (by decide) _ _ _ (map_keeps (Protected q) (fun x => x.2.1) _ port)
  · exact word_other_field _ _ (by decide) _ _ _ (map_keeps (Protected q) (fun x => x.2.2.1) _ port)
  · exact actual_word_write_other_revision_cell q slice a hq hs hne w
      (map_keeps (Protected q) (fun x => x.2.2.2) _ port)
theorem ecc_other_field {β : Type} (r : Register) (pick : RegisterType r → β)
    (hne : r≠Register.sram_revision) (slice a : Nat) (w : ecc_ov)
    (port : Keeps (FieldView r pick) (raw_write_sram_ecc slice a w)) :
    Keeps (FieldView r pick) (write_sram_ecc slice a w) := by
  unfold write_sram_ecc
  repeat' first
    | exact port
    | exact read_keeps _ _
    | exact unrelated_register_field_write _ _ _ _ hne
    | apply bind_keeps
    | intro

theorem actual_other_ecc (q slice a : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice) (w : ecc_ov)
    (port : Keeps (Protected q) (raw_write_sram_ecc slice a w)) :
    Keeps (Protected q) (write_sram_ecc slice a w) := by
  have hO : Keeps Ownership (raw_write_sram_ecc slice a w) := map_keeps (Protected q) (fun x => x.1) _ port
  apply protected_keeps
  · apply ownership_keeps_of_registers
    · exact write_sram_ecc_other_reg _ (by decide) _ _ _ (map_keeps Ownership (fun x => x.1) _ hO)
    · exact write_sram_ecc_other_reg _ (by decide) _ _ _ (map_keeps Ownership (fun x => x.2.1) _ hO)
    · exact write_sram_ecc_other_reg _ (by decide) _ _ _ (map_keeps Ownership (fun x => x.2.2) _ hO)
  · exact ecc_other_field _ _ (by decide) _ _ _ (map_keeps (Protected q) (fun x => x.2.1) _ port)
  · exact ecc_other_field _ _ (by decide) _ _ _ (map_keeps (Protected q) (fun x => x.2.2.1) _ port)
  · exact actual_ecc_write_other_revision_cell q slice a hq hs hne w
      (map_keeps (Protected q) (fun x => x.2.2.2) _ port)
end PatrolProtectedProofs
