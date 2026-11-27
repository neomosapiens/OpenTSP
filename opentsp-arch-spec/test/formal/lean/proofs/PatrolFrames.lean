-- Actual patrol per-slice effects; other slices remain free to act.
import Proofs.ResourceRasFrames
import Proofs.Patrol
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions Out.Register Out.ecc_class
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 100000
namespace PatrolFrameProofs
open ResourceFrameProofs ResourceRasFrameProofs
def scrubLane (s m a : Nat) (g : Int) (acc : Bool × ecc_ov × vec) : SailM (Bool × ecc_ov × vec) := do
  let (fixed,ov,w) := acc
  let e := (ecc_ov_get ov g)
  let (k, d2, pos) := (ecc_correct (BitVec.slice w (128 *i g) 128) e)
  let (fixed, ov, w) ← (( do
    match k with
    | .ECC_OK => (pure (fixed, ov, w))
    | .ECC_CORRECTED =>
      (do
        let w : vec := (Sail.BitVec.updateSubrange w ((128 *i g) +i 127) (128 *i g) d2)
        let ov : ecc_ov := (ecc_ov_set ov g (e &&& 0b1000000000#10))
        let fixed : Bool := true
        (ras_scrub_event m s false a g pos)
        (pure (fixed, ov, w)))
    | .ECC_UNCORRECTABLE =>
      (do
        (ras_scrub_event m s true a g pos)
        (pure (fixed, ov, w))) ) : SailM
    (Bool × ecc_ov × vec) )
  (pure (fixed, ov, w))

def saveRepair (s a : Nat) (w : vec) (ov : ecc_ov) : SailM Unit := do
  writeReg scrub_pending (vectorUpdate (← readReg scrub_pending) s
    (some
      { addr := a
        data := w
        ov := ov
        revision := ← (pure (GetElem?.getElem! (← readReg sram_revision) s)) }))

def movePointer (s a : Nat) : SailM Unit := do
  writeReg scrub_ptr (vectorUpdate (← readReg scrub_ptr) s (Int.emod (a +i 1) Out.Functions.words_per_slice))

def scrubSliceFactory (s : Nat) (m : Nat) (upper : Int) : SailM Unit := do
  let a ← (( do (pure (GetElem?.getElem! (← readReg scrub_ptr) s)) ) : SailM Nat )
  let w ← (( do (read_sram_word s a) ) : SailM vec )
  let ov ← (( do (read_sram_ecc s a) ) : SailM ecc_ov )
  let fixed : Bool := false
  if ((ov != ecc_ov_zero) : Bool)
  then
    (do
      let (fixed, ov, w) ← (( do
        let loop_g_lower := 0
        let loop_g_upper := upper
        let mut loop_vars := (fixed, ov, w)
        for g in [loop_g_lower:loop_g_upper:1]i do
          let (fixed, ov, w) := loop_vars
          loop_vars ← do
            scrubLane s m a g (fixed,ov,w)
        (pure loop_vars) ) : SailM (Bool × ecc_ov × vec) )
      if (fixed : Bool)
      then
        saveRepair s a w ov
      else (pure ()))
  else (pure ())
  movePointer s a

section SourceBinding
attribute [local irreducible] IntRange.forIn' Out.Functions.ras_scrub_event

theorem actual_scrub_slice_source (s m : Nat) :
    scrub_slice s m=scrubSliceFactory s m (Out.Functions.superlanes -i 1) := by
  simp only [scrub_slice,scrubSliceFactory,scrubLane,saveRepair,movePointer,pure_bind,bind_pure,bind_assoc]
  rfl
end SourceBinding

theorem actual_scrub_lane_keeps_view {β : Type} (view : St → β)
    (writes : ∀ w, w ∈ RasWrites → ∀ v : RegisterType w, Keeps view (Out.writeReg w v))
    (s m a : Nat) (g : Int) (acc : Bool × ecc_ov × vec)
    (port : Keeps view (host_irq_callback m 0)) : Keeps view (scrubLane s m a g acc) := by
  unfold scrubLane
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | (apply actual_scrub_report_keeps_view _ writes; exact port)
    | apply ite_keeps
    | apply bind_keeps
    | split
    | intro

theorem actual_scrub_slice_keeps_view {β : Type} (view : St → β)
    (writes : ∀ w, w ∈ RasWrites → ∀ v : RegisterType w, Keeps view (Out.writeReg w v))
    (s m : Nat) (data : ∀ a, Keeps view (read_sram_word s a))
    (ecc : ∀ a, Keeps view (read_sram_ecc s a))
    (irq : Keeps view (host_irq_callback m 0))
    (save : ∀ a w ov, Keeps view (saveRepair s a w ov))
    (move : ∀ a, Keeps view (movePointer s a)) : Keeps view (scrub_slice s m) := by
  rw [actual_scrub_slice_source]
  unfold scrubSliceFactory
  repeat' first
    | assumption
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | exact data _
    | exact ecc _
    | exact save _ _ _
    | exact move _
    | exact actual_scrub_lane_keeps_view view writes _ _ _ _ _ irq
    | apply forIn_keeps
    | apply ite_keeps
    | apply bind_keeps
    | split
    | intro

theorem actual_move_pointer_other_cell (q slice a : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice) :
    Keeps (FieldView Register.scrub_ptr (fun v => v[q]!)) (movePointer slice a) := by
  unfold movePointer
  apply update_field_keeps
  intro old
  exact vector_update_other 88 q slice old _ hq hs hne

theorem actual_save_repair_other_cell (q slice a : Nat) (hq : q<88) (hs : slice<88) (hne : q≠slice)
    (w : vec) (ov : ecc_ov) :
    Keeps (FieldView Register.scrub_pending (fun v => v[q]!)) (saveRepair slice a w ov) := by
  let value : SailM (Option scrub_write) := do
    let revisions ← Out.readReg Register.sram_revision
    pure (some {addr:=a,data:=w,ov:=ov,revision:=revisions[slice]!})
  have hv : Keeps (fun s : St => s) value := by
    apply bind_keeps _ _ _ (read_keeps _ _)
    intro revisions
    exact pure_keeps _ _
  have h := read_value_update_field_keeps Register.scrub_pending (fun v => v[q]!) value
    (fun old x => vectorUpdate old slice x) hv
    (fun old x => vector_update_other 88 q slice old x hq hs hne)
  simpa only [saveRepair,value,pure_bind,bind_pure,bind_assoc] using h

theorem actual_save_repair_other_reg (r : Register) (hne : r≠Register.scrub_pending)
    (slice a : Nat) (w : vec) (ov : ecc_ov) : RegKeeps r (saveRepair slice a w ov) := by
  unfold RegKeeps saveRepair
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | exact unrelated_register_write _ _ _ hne
    | apply bind_keeps
    | intro

theorem actual_move_pointer_other_reg (r : Register) (hne : r≠Register.scrub_ptr)
    (slice a : Nat) : RegKeeps r (movePointer slice a) := by
  unfold RegKeeps movePointer
  apply bind_keeps _ _ _ (read_keeps _ _)
  intro old
  exact unrelated_register_write _ _ _ hne

end PatrolFrameProofs
