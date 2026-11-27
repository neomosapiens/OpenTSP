-- Whole production patrol, preserving an owned slice while other slices act.
import Proofs.PatrolProtected
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace PatrolOwnedCycleProofs
open ResourceFrameProofs ResourceInvariantProofs PatrolCycleFrameProofs PatrolOwnershipProofs PatrolProtectedProofs

def Same (q : Nat) (base : St) (s : St) : Prop := Protected q s=Protected q base

theorem owned_query_bind {α : Type} (q m : Nat) (base : St)
    (held : ownershipValue q m base=some true) (f : Bool → SailM α)
    (hf : Inv (Same q base) (f true)) : Inv (Same q base) (scrub_issued q m >>= f) := by
  intro s hs
  have hO : Ownership s=Ownership base := congrArg Prod.fst hs
  have hv : ownershipValue q m s=some true := (ownership_transport q m base s hO).trans held
  have hr := actual_ownership_run q m s
  simp only [hv] at hr
  change scrub_issued q m s=.ok true s at hr
  simpa only [EStateM.run,bind,EStateM.bind,hr] using hf s hs

structure Ports (q m : Nat) : Prop where
  data : ∀ slice, slice<88 → slice≠q → ∀ a, Keeps (Protected q) (read_sram_word slice a)
  ecc : ∀ slice, slice<88 → slice≠q → ∀ a, Keeps (Protected q) (read_sram_ecc slice a)
  putData : ∀ slice, slice<88 → slice≠q → ∀ a w, Keeps (Protected q) (raw_write_sram_word slice a w)
  putEcc : ∀ slice, slice<88 → slice≠q → ∀ a w, Keeps (Protected q) (raw_write_sram_ecc slice a w)
  irq : Keeps (Protected q) (host_irq_callback m 0)

theorem query_keeps {β : Type} (view : St → β) (q m : Nat) : Keeps view (scrub_issued q m) := by
  unfold scrub_issued
  repeat' first
    | exact pure_keeps _ _
    | exact read_keeps _ _
    | apply bind_keeps
    | split
    | intro

section
attribute [local irreducible] Out.Functions.scrub_issued Out.Functions.scrub_slice Out.Functions.scrub_take
theorem actual_owned_cycle (q m : Nat) (hq : q<88) (base : St)
    (held : ownershipValue q m base=some true) (ports : Ports q m) :
    Inv (Same q base) (scrub_cycle m) := by
  unfold scrub_cycle
  apply bind_inv
  · apply forIn_inv
    intro i hi value
    have hib : 0≤i ∧ i≤87 := by simpa using hi.1
    have hi0 : 0≤i := hib.1
    have hi88 : i.toNat<88 := by
      have hil : i≤87 := hib.2
      omega
    by_cases he : i.toNat=q
    · simp only [he]
      repeat' first
        | exact pure_inv _ _
        | exact read_inv _ _
        | (apply owned_query_bind q m base held; simp [scrub_arbitrate,Out.Functions.not])
        | apply bind_inv
        | split
        | intro
    · apply keeps_inv (Protected q) (Protected q base)
      repeat' first
        | exact pure_keeps _ _
        | exact read_keeps _ _
        | exact query_keeps _ _ _
        | exact actual_other_slice q i.toNat m hq hi88 (Ne.symm he) (ports.data _ hi88 he) (ports.ecc _ hi88 he) ports.irq
        | exact actual_other_take q i.toNat hq hi88 (Ne.symm he)
        | exact actual_other_word q i.toNat _ hq hi88 (Ne.symm he) _ (ports.putData _ hi88 he _ _)
        | exact actual_other_ecc q i.toNat _ hq hi88 (Ne.symm he) _ (ports.putEcc _ hi88 he _ _)
        | apply bind_keeps
        | split
        | intro
  · intro x
    exact pure_inv _ _
end
end PatrolOwnedCycleProofs
