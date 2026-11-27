-- Reachable patrol pointer and saved-address invariants for actual primitive calls.
import Proofs.PatrolMemory
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace PatrolValidityProofs
open ResourceFrameProofs ResourceInvariantProofs PatrolFrameProofs ResourceRasFrameProofs PatrolCycleFrameProofs
abbrev State := ResourceFrameProofs.St

def PtrOk (v : Option (Vector Nat 88)) : Prop := match v with
  | none => True
  | some p => ∀q,q<88 → p[q]!<Out.Functions.words_per_slice

def RepairOk (v : Option (Vector (Option scrub_write) 88)) : Prop := match v with
  | none => True
  | some p => ∀q,q<88 → ∀w,p[q]! = some w → w.addr<Out.Functions.words_per_slice

def View (s : State) := (s.regs.get? Register.scrub_ptr,s.regs.get? Register.scrub_pending)
def ValidView (v : Option (Vector Nat 88) × Option (Vector (Option scrub_write) 88)) := PtrOk v.1 ∧ RepairOk v.2
def Valid (s : State) := ValidView (View s)

theorem unrelated_valid (r : Register) (value : RegisterType r)
    (hr : r≠Register.scrub_ptr) (hp : r≠Register.scrub_pending) : Inv Valid (Out.writeReg r value) := by
  apply keeps_pred View ValidView
  exact pair_keeps _ _ _ (unrelated_register_write _ _ _ (Ne.symm hr)) (unrelated_register_write _ _ _ (Ne.symm hp))

theorem write_ptr_valid (p : Vector Nat 88) (hp : ∀q,q<88 → p[q]!<Out.Functions.words_per_slice) :
    Inv Valid (Out.writeReg Register.scrub_ptr p) := by
  intro s hs
  change Valid {s with regs:=s.regs.insert Register.scrub_ptr p}
  simpa [Valid,ValidView,View,PtrOk,Std.ExtDHashMap.get?_insert] using And.intro hp hs.2

theorem write_repair_valid (p : Vector (Option scrub_write) 88)
    (hp : ∀q,q<88 → ∀w,p[q]! = some w → w.addr<Out.Functions.words_per_slice) :
    Inv Valid (Out.writeReg Register.scrub_pending p) := by
  intro s hs
  change Valid {s with regs:=s.regs.insert Register.scrub_pending p}
  simpa [Valid,ValidView,View,RepairOk,Std.ExtDHashMap.get?_insert] using And.intro hs.1 hp

theorem update_ptr_bound (p : Vector Nat 88) (slice a : Nat) (hs : slice<88)
    (hp : ∀q,q<88 → p[q]!<Out.Functions.words_per_slice) :
    ∀q,q<88 → (vectorUpdate p slice (Int.emod ((a:Int)+1) Out.Functions.words_per_slice).toNat)[q]!<Out.Functions.words_per_slice := by
  intro q hq
  by_cases he : q=slice
  · subst q
    have h0:=Int.emod_nonneg ((a:Int)+1) (by decide : (Out.Functions.words_per_slice:Int)≠0)
    have hl:=Int.emod_lt_of_pos ((a:Int)+1) (by decide : (0:Int)<Out.Functions.words_per_slice)
    have hv : (Int.emod ((a:Int)+1) (Out.Functions.words_per_slice:Int)).toNat<Out.Functions.words_per_slice := by
      change (((a:Int)+1) % (Out.Functions.words_per_slice:Int)).toNat<Out.Functions.words_per_slice
      exact (Int.toNat_lt h0).mpr hl
    simpa [vectorUpdate,Vector.set!,hs] using hv
  · rw [vector_update_other 88 q slice p _ hq hs he]
    exact hp q hq

theorem actual_pointer_valid (slice a : Nat) (hs : slice<88) : Inv Valid (movePointer slice a) := by
  unfold movePointer
  apply read_bind_bounded Valid Register.scrub_ptr (fun p => ∀q,q<88 → p[q]!<Out.Functions.words_per_slice)
  · intro s hv p hp
    have h:=hv.1
    simpa [Valid,ValidView,View,PtrOk,hp] using h
  · intro p hp
    exact write_ptr_valid _ (update_ptr_bound p slice a hs hp)

theorem update_repair_bound (p : Vector (Option scrub_write) 88) (slice : Nat)
    (hs : slice<88) (old : ∀q,q<88 → ∀w,p[q]! = some w → w.addr<Out.Functions.words_per_slice)
    (value : Option scrub_write) (hv : ∀w,value=some w → w.addr<Out.Functions.words_per_slice) :
    ∀q,q<88 → ∀w,(vectorUpdate p slice value)[q]! = some w → w.addr<Out.Functions.words_per_slice := by
  intro q hq w hw
  by_cases he : q=slice
  · subst q
    have hself : (vectorUpdate p slice value)[slice]! = value := by simp [vectorUpdate,Vector.set!,hs]
    exact hv w (hself ▸ hw)
  · rw [vector_update_other 88 q slice p value hq hs he] at hw
    exact old q hq w hw
theorem actual_save_valid (slice a : Nat) (hs : slice<88)
    (ha : a<Out.Functions.words_per_slice) (w : vec) (ov : ecc_ov) :
    Inv Valid (saveRepair slice a w ov) := by
  unfold saveRepair
  simp only [pure_bind]
  apply read_bind_bounded Valid Register.scrub_pending (fun p => ∀q,q<88 → ∀w,p[q]! = some w → w.addr<Out.Functions.words_per_slice)
  · intro s hv p hp
    have h:=hv.2
    simpa [Valid,ValidView,View,RepairOk,hp] using h
  · intro p hp
    apply bind_inv _ _ _ (read_inv _ _)
    intro revisions
    apply write_repair_valid
    apply update_repair_bound p slice hs hp
    intro v hv
    cases hv
    exact ha

theorem actual_clear_valid (slice : Nat) (hs : slice<88) :
    Inv Valid (do let p ← Out.readReg Register.scrub_pending; Out.writeReg Register.scrub_pending (vectorUpdate p slice none)) := by
  apply read_bind_bounded Valid Register.scrub_pending (fun p => ∀q,q<88 → ∀w,p[q]! = some w → w.addr<Out.Functions.words_per_slice)
  · intro s hv p hp
    have h:=hv.2
    simpa [Valid,ValidView,View,RepairOk,hp] using h
  · intro p hp
    apply write_repair_valid
    apply update_repair_bound p slice hs hp
    intro w hw
    cases hw

theorem actual_take_valid (slice : Nat) (hs : slice<88) : Inv Valid (scrub_take slice) := by
  unfold scrub_take
  simp only [pure_bind]
  apply bind_inv _ _ _ (read_inv _ _)
  intro old
  rw [←bind_assoc]
  apply bind_inv _ _ _ (actual_clear_valid slice hs)
  intro u
  split
  · apply bind_inv _ _ _ (read_inv _ _)
    intro r
    split <;> exact pure_inv _ _
  · exact pure_inv _ _
structure Ports (m : Nat) : Prop where
  data : ∀slice,slice<88 → ∀a,a<Out.Functions.words_per_slice → Keeps View (read_sram_word slice a)
  ecc : ∀slice,slice<88 → ∀a,a<Out.Functions.words_per_slice → Keeps View (read_sram_ecc slice a)
  putData : ∀slice,slice<88 → ∀a,a<Out.Functions.words_per_slice → ∀w,Keeps View (raw_write_sram_word slice a w)
  putEcc : ∀slice,slice<88 → ∀a,a<Out.Functions.words_per_slice → ∀w,Keeps View (raw_write_sram_ecc slice a w)
  irq : Keeps View (host_irq_callback m 0)

theorem ras_write_view (r : Register) (hr : r∈RasWrites) (v : RegisterType r) :
    Keeps View (Out.writeReg r v) := by
  apply pair_keeps
  all_goals
    apply unrelated_register_write
    simp only [RasWrites,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with h|h|h|h|h|h|h|h|h|h|h <;> subst r <;> decide

theorem actual_lane_valid (slice m a : Nat) (g : Int) (state : Bool × ecc_ov × vec)
    (ports : Ports m) : Inv Valid (scrubLane slice m a g state) := by
  apply keeps_pred View ValidView
  exact actual_scrub_lane_keeps_view View ras_write_view slice m a g state ports.irq

theorem actual_slice_valid (slice m : Nat) (hs : slice<88) (ports : Ports m) :
    Inv Valid (scrub_slice slice m) := by
  rw [actual_scrub_slice_source]
  unfold scrubSliceFactory
  simp only [pure_bind,bind_assoc]
  apply read_bind_bounded Valid Register.scrub_ptr (fun p => ∀q,q<88 → p[q]!<Out.Functions.words_per_slice)
  · intro s hv p hp
    have h:=hv.1
    simpa [Valid,ValidView,View,PtrOk,hp] using h
  · intro p hp
    have ha:=hp slice hs
    repeat' first
      | exact pure_inv _ _
      | exact actual_save_valid slice _ hs ha _ _
      | exact actual_pointer_valid slice _ hs
      | exact actual_lane_valid _ _ _ _ _ ports
      | exact keeps_pred View ValidView _ (ports.data slice hs _ ha)
      | exact keeps_pred View ValidView _ (ports.ecc slice hs _ ha)
      | apply forIn_inv
      | apply bind_inv
      | split
      | intro
theorem actual_take_origin (slice : Nat) (s after : State) (w : scrub_write)
    (hRun : (scrub_take slice).run s=.ok (some w) after) :
    ∃p,s.regs.get? Register.scrub_pending=some p ∧ p[slice]! = some w := by
  cases hp : s.regs.get? Register.scrub_pending with
  | none =>
    have hm:=read_missing_run s Register.scrub_pending hp
    change Out.readReg Register.scrub_pending s=.error Error.Unreachable s at hm
    simp only [scrub_take,EStateM.run,bind,EStateM.bind,hm] at hRun
    cases hRun
  | some p =>
    cases hold : p[slice]! with
    | none =>
      simp [scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
        EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
        modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hp,hold] at hRun
    | some v =>
      cases hr : s.regs.get? Register.sram_revision with
      | none =>
        have he : (scrub_take slice).run s=.error Error.Unreachable
            {s with regs:=s.regs.insert Register.scrub_pending (vectorUpdate p slice none)} := by
          simp [scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
            EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
            modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hp,hold,hr]
          rfl
        rw [he] at hRun
        cases hRun
      | some revisions =>
        by_cases he : v.revision=revisions[slice]!
        · have hw : v=w := by
            simp [scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
              EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
              modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hp,hold,hr,he] at hRun
            exact hRun.1
          exact ⟨p,rfl,hw ▸ hold⟩
        · simp [scrub_take,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
            EStateM.run,bind,EStateM.bind,get,getThe,MonadStateOf.get,EStateM.get,pure,EStateM.pure,
            modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,Std.ExtDHashMap.get?_insert,hp,hold,hr,he] at hRun

theorem actual_take_bound (slice : Nat) (hslice : slice<88) (s after : State) (w : scrub_write)
    (hv : Valid s) (hr : (scrub_take slice).run s=.ok (some w) after) :
    w.addr<Out.Functions.words_per_slice := by
  obtain ⟨p,hp,hw⟩:=actual_take_origin slice s after w hr
  have h:=hv.2
  simp only [Valid,ValidView,View,hp,RepairOk] at h
  exact h slice hslice w hw

theorem take_bind_bounded {α : Type} (slice : Nat) (hslice : slice<88)
    (f : Option scrub_write → SailM α)
    (hn : Inv Valid (f none))
    (hf : ∀w,w.addr<Out.Functions.words_per_slice → Inv Valid (f (some w))) :
    Inv Valid (scrub_take slice >>= f) := by
  intro s hs
  have hi:=actual_take_valid slice hslice s hs
  cases hr : (scrub_take slice).run s with
  | error e after =>
    simp only [hr] at hi
    change scrub_take slice s=.error e after at hr
    simpa only [EStateM.run,bind,EStateM.bind,hr] using hi
  | ok value after =>
    simp only [hr] at hi
    have hnext : match (f value).run after with
      | .ok _ final => Valid final
      | .error _ final => Valid final := by
      cases value with
      | none => exact hn after hi
      | some w => exact hf w (actual_take_bound slice hslice s after w hs hr) after hi
    change scrub_take slice s=.ok value after at hr
    simpa only [EStateM.run,bind,EStateM.bind,hr] using hnext
theorem word_valid (slice m a : Nat) (w : vec) (hs : slice<88)
    (ha : a<Out.Functions.words_per_slice) (ports : Ports m) :
    Inv Valid (write_sram_word slice a w) := by
  apply keeps_pred View ValidView
  unfold write_sram_word
  repeat' first
    | exact read_keeps _ _
    | exact ports.putData _ hs _ ha _
    | (apply pair_keeps <;> exact unrelated_register_write _ _ _ (by decide))
    | apply bind_keeps
    | intro

theorem ecc_valid (slice m a : Nat) (w : ecc_ov) (hs : slice<88)
    (ha : a<Out.Functions.words_per_slice) (ports : Ports m) :
    Inv Valid (write_sram_ecc slice a w) := by
  apply keeps_pred View ValidView
  unfold write_sram_ecc
  repeat' first
    | exact read_keeps _ _
    | exact ports.putEcc _ hs _ ha _
    | (apply pair_keeps <;> exact unrelated_register_write _ _ _ (by decide))
    | apply bind_keeps
    | intro

section
attribute [local irreducible] Out.Functions.scrub_issued Out.Functions.scrub_slice Out.Functions.scrub_take Out.Functions.write_sram_word Out.Functions.write_sram_ecc
theorem actual_cycle_valid (m : Nat) (ports : Ports m) : Inv Valid (scrub_cycle m) := by
  unfold scrub_cycle
  apply bind_inv
  · apply forIn_inv
    intro i hi value
    have hib : 0≤i ∧ i≤87 := by simpa using hi.1
    have hi88 : i.toNat<88 := by omega
    repeat' first
      | exact pure_inv _ _
      | exact read_inv _ _
      | exact actual_slice_valid _ _ hi88 ports
      | exact actual_take_valid _ hi88
      | exact keeps_pred View ValidView _ (PatrolOwnedCycleProofs.query_keeps View _ _)
      | exact word_valid _ m _ _ hi88 (by simp_all) ports
      | exact ecc_valid _ m _ _ hi88 (by simp_all) ports
      | apply take_bind_bounded _ hi88
      | apply bind_inv
      | split
      | intro
  · intro x
    exact pure_inv _ _
end

theorem actual_init_valid (s : State) :
    ∃after,(sail_model_init ()).run s=.ok () after ∧ Valid after := by
  refine ⟨_,rfl,?_⟩
  constructor
  · simp only [Valid,ValidView,View,PtrOk,Std.ExtDHashMap.get?_insert,vectorInit]
    intro q hq
    simp [hq,Out.Functions.words_per_slice]
  · simp only [Valid,ValidView,View,RepairOk,Std.ExtDHashMap.get?_insert,vectorInit]
    intro q hq w hw
    simpa [hq] using hw
end PatrolValidityProofs
