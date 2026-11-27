-- Whole real patrol preserves an owned slice's external SRAM witness while others act.
import Proofs.SramStorage
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace PatrolMemoryProofs
open ResourceFrameProofs ResourceInvariantProofs ResourceRasFrameProofs PatrolFrameProofs PatrolCycleFrameProofs PatrolOwnershipProofs PatrolProtectedProofs SramStorageProofs
abbrev State := ResourceFrameProofs.St

def View (words q : Nat) (s : State) := (Ownership s,SliceView words q s)

theorem write_view (words q : Nat) (r : Register) (v : RegisterType r)
    (h : r∉[Register.path_busy_until,Register.queue_last,Register.queue_last_at]) :
    Keeps (View words q) (Out.writeReg r v) := by
  apply pair_keeps
  · apply ownership_keeps_of_registers <;> exact unrelated_register_write _ _ _ (by intro he; subst r; simp_all)
  · exact write_register_preserves_memory words q r v

theorem actual_repair_view (words q slice a : Nat) (w : vec) (ov : ecc_ov) :
    Keeps (View words q) (saveRepair slice a w ov) := by
  unfold saveRepair
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | exact write_view words q _ _ (by decide)
    | apply bind_keeps
    | intro

theorem actual_pointer_view (words q slice a : Nat) : Keeps (View words q) (movePointer slice a) := by
  unfold movePointer
  apply bind_keeps _ _ _ (read_keeps _ _)
  intro v
  exact write_view words q _ _ (by decide)

theorem actual_take_view (words q slice : Nat) : Keeps (View words q) (scrub_take slice) := by
  unfold scrub_take
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | exact write_view words q _ _ (by decide)
    | apply bind_keeps
    | split
    | intro

structure Ports (words q m : Nat) : Prop where
  data : ∀ slice,slice<88 → slice≠q → ∀ a,Keeps (View words q) (read_sram_word slice a)
  ecc : ∀ slice,slice<88 → slice≠q → ∀ a,Keeps (View words q) (read_sram_ecc slice a)
  putData : ∀ slice,slice<88 → slice≠q → ∀ a w,Keeps (View words q) (raw_write_sram_word slice a w)
  putEcc : ∀ slice,slice<88 → slice≠q → ∀ a w,Keeps (View words q) (raw_write_sram_ecc slice a w)
  irq : Keeps (View words q) (host_irq_callback m 0)

theorem actual_slice_view (words q slice m : Nat) (hs : slice<88) (hne : slice≠q)
    (ports : Ports words q m) : Keeps (View words q) (scrub_slice slice m) := by
  apply actual_scrub_slice_keeps_view
  · intro r hr v
    apply write_view
    simp only [RasWrites,List.mem_cons,List.not_mem_nil,or_false] at hr
    rcases hr with h|h|h|h|h|h|h|h|h|h|h <;> subst r <;> decide
  · exact ports.data _ hs hne
  · exact ports.ecc _ hs hne
  · exact ports.irq
  · exact actual_repair_view words q slice
  · exact actual_pointer_view words q slice

theorem actual_word_view (words q slice m a : Nat) (w : vec) (hs : slice<88) (hne : slice≠q)
    (ports : Ports words q m) : Keeps (View words q) (write_sram_word slice a w) := by
  unfold write_sram_word
  repeat' first
    | exact read_keeps _ _
    | exact write_view words q _ _ (by decide)
    | exact ports.putData _ hs hne _ _
    | apply bind_keeps
    | intro

theorem actual_ecc_view (words q slice m a : Nat) (w : ecc_ov) (hs : slice<88) (hne : slice≠q)
    (ports : Ports words q m) : Keeps (View words q) (write_sram_ecc slice a w) := by
  unfold write_sram_ecc
  repeat' first
    | exact read_keeps _ _
    | exact write_view words q _ _ (by decide)
    | exact ports.putEcc _ hs hne _ _
    | apply bind_keeps
    | intro

theorem owned_query_bind {α : Type} (words q m : Nat) (base : State)
    (held : ownershipValue q m base=some true) (f : Bool → SailM α)
    (hf : Inv (fun s => View words q s=View words q base) (f true)) :
    Inv (fun s => View words q s=View words q base) (scrub_issued q m >>= f) := by
  intro s hs
  have hO : Ownership s=Ownership base := congrArg Prod.fst hs
  have hv := (ownership_transport q m base s hO).trans held
  have hr := actual_ownership_run q m s
  simp only [hv] at hr
  change scrub_issued q m s=.ok true s at hr
  simpa only [EStateM.run,bind,EStateM.bind,hr] using hf s hs

section
attribute [local irreducible] Out.Functions.scrub_issued Out.Functions.scrub_slice Out.Functions.scrub_take
theorem actual_owned_memory_cycle (words q m : Nat) (hq : q<88) (base : State)
    (held : ownershipValue q m base=some true) (ports : Ports words q m) :
    Inv (fun s => View words q s=View words q base) (scrub_cycle m) := by
  unfold scrub_cycle
  apply bind_inv
  · apply forIn_inv
    intro i hi value
    have hib : 0≤i ∧ i≤87 := by simpa using hi.1
    have hi88 : i.toNat<88 := by omega
    by_cases he : i.toNat=q
    · simp only [he]
      repeat' first
        | exact pure_inv _ _
        | exact read_inv _ _
        | (apply owned_query_bind words q m base held; simp [scrub_arbitrate,Out.Functions.not])
        | apply bind_inv
        | split
        | intro
    · apply keeps_inv
      repeat' first
        | exact pure_keeps _ _
        | exact read_keeps _ _
        | exact PatrolOwnedCycleProofs.query_keeps _ _ _
        | exact actual_slice_view words q i.toNat m hi88 he ports
        | exact actual_take_view words q _
        | exact actual_word_view words q i.toNat m _ _ hi88 he ports
        | exact actual_ecc_view words q i.toNat m _ _ hi88 he ports
        | apply bind_keeps
        | split
        | intro
  · intro x
    exact pure_inv _ _
end
end PatrolMemoryProofs
