import Proofs.TaggedCheck
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
open Proofs.ParcelFrame Proofs.TagCoverage Proofs.TaggedCheck
open Out.Register Out.diag
set_option autoImplicit false
set_option maxRecDepth 6000
set_option maxHeartbeats 1000000
namespace Proofs.ControlConsumption
attribute [local irreducible] Proofs.TaggedCheck.Safe

/-- The direct C observer reads GMP slots only for a nonempty count. `A` is the
    concrete allocated-tag projection; unlike a total -1 read value, it is named
    separately and is supplied by the proved paired-store allocation relation. -/
structure DirectObserver (A : Nat → Prop) : Prop where
  frame : ∀ (event : String) (c q head units : Nat) (detail : Int) (s : St),
    Allocation iq_cap A s → q<145 → units≤iq_cap → (0<units → head<iq_cap ∧ A q) →
    ∃ after, (observe_site event c q head units detail).run s=.ok () after ∧
      after.regs=s.regs ∧ after.mem=s.mem

def consumeParcel (c : Nat) (p : parcel) : SailM (Option parcel) := do
  observe_site "dispatch" c 144 (GetElem?.getElem! (← Out.readReg iq_head) 144) p.units
    (← parcel_earliest 144 p)
  Proofs.TextCheckProduct.checkedPop 144 c p
  pure (some p)

def controlHead (c : Nat) : SailM (Option parcel) := do
  if (((GetElem?.getElem! (← Out.readReg iq_len) 144) >b 0) : Bool) then
    let p ← parcel_scan 144
    if ((p.complete && ((← parcel_earliest 144 p) ≤b c)) : Bool) then consumeParcel c p
    else pure none
  else pure none

def controlTail (c : Nat) (p : parcel) : SailM Unit := do
  Out.writeReg iq_last (vectorUpdate (← Out.readReg iq_last) 144 c)
  Out.writeReg iq_active (vectorUpdate (← Out.readReg iq_active) 144 c)
  match p.items with
  | (x :: rest) =>
    (do
      match rest with
      | [] =>
        (do
          match x with
          | .Err d => (report 144 c d)
          | .Ok i =>
            (do
              match i with
              | .SYNC () => (barrier_sync 144 c)
              | .NOTIFY () => (barrier_notify c)
              | .CONFIG _ => (pure ())
              | _ => (report 144 c QUE_UNIT)))
      | _ =>
        (do
          let loop_k_lower := 1
          let loop_k_upper := p.n
          let mut loop_vars := ()
          for k in [loop_k_lower:loop_k_upper:1]i do
            let () := loop_vars
            loop_vars ← do (report 144 c QUE_BUSY)
          (pure loop_vars)))
  | [] => (pure ())

theorem actual_control_factor (c : Nat) : Out.Functions.control c =
    (do let p ← controlHead c
        match p with | none => pure () | some p => controlTail c p) := by
  rw [Proofs.TextCheckProduct.control_checked_pop_bridge]
  simp only [Proofs.TextCheckProduct.controlUsingCheckedPop,controlHead,consumeParcel,controlTail,
    Proofs.OrderedImage.bind_if,bind_assoc,pure_bind]
  all_goals rfl

theorem earliest_frame (s : St) (q : Nat) (p : parcel) : FrameAt s (parcel_earliest q p) := by
  unfold parcel_earliest
  apply bind_frame _ _ _ (read_frame s iq_last)
  intro last
  exact pure_frame s _

theorem keep_same_parts (A : Nat → Prop) (H L : Vector Nat 145) (s t : St)
    (keep : Keep A H L s) (hr : t.regs=s.regs) (hm : t.mem=s.mem) : Keep A H L t := by
  obtain ⟨⟨⟨view,rep,bounds⟩,hH,hL⟩,tags,allocation,complete⟩ := keep
  refine ⟨⟨⟨view,Proofs.IqReach.represents_memory iq_cap view s t hm rep,?_⟩,?_,?_⟩,?_,?_,?_⟩
  · simpa only [Proofs.Initialized.Bounds,Proofs.StorageRep.Initialized,Present,hr,hm] using bounds
  · simpa only [hr] using hH
  · simpa only [hr] using hL
  · exact tag_bounds_frame iq_cap s t tags (by rw [hr]) (by rw [hr]) hm
  · exact allocation_same_memory iq_cap A s t allocation hm
  · simpa only [MachineInitProofs.Complete,hr] using complete

theorem direct_dispatch_domain (A : Nat → Prop) (H L : Vector Nat 145)
    (s : St) (keep : Keep A H L s) (q units : Nat) (hq : q<145)
    (hh : H[q]!<iq_cap) (bound : units≤L[q]!) (positive : 0<units) :
    H[q]!<iq_cap ∧ A q := by
  exact ⟨hh,occupied_queue_allocated iq_cap A s keep.2.1 keep.2.2.1 L keep.1.2.2 q hq (by omega)⟩

theorem tail_safe (A : Nat → Prop) (H L : Vector Nat 145) (c : Nat) (p : parcel) :
    Safe A (H:=H) (L:=L) (controlTail c p) := by
  unfold controlTail
  dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
  repeat' first
    | exact report_safe A _ _ _
    | exact barrier_sync_safe A _ _
    | exact barrier_notify_safe A _
    | with_reducible apply preserves_read A
    | with_reducible apply preserves_pure A
    | with_reducible apply preserves_other_write A <;> decide
    | with_reducible apply preserves_forIn A
    | with_reducible apply preserves_bind A
    | split
    | intro x


def HeadOutcome (A : Nat → Prop) (H L : Vector Nat 145) (value : Option parcel) (after : St) : Prop :=
  match value with
  | none => Keep A H L after
  | some p => p.units≤L[144]! ∧ Keep A (popHead H 144 p.units) (popLength L 144 p.units) after

def HeadSafeAt (A : Nat → Prop) (H L : Vector Nat 145) (s : St) (action : SailM (Option parcel)) : Prop :=
  match action.run s with
  | .ok value after => HeadOutcome A H L value after
  | .error _ after => Keep A H L after

theorem head_bind_frame {α : Type} (A : Nat → Prop) (H L : Vector Nat 145)
    (s : St) (keep : Keep A H L s) (a : SailM α) (f : α → SailM (Option parcel))
    (frame : FrameAt s a) (next : ∀ x, HeadSafeAt A H L s (f x)) :
    HeadSafeAt A H L s (a >>= f) := by
  cases run : a.run s with
  | error error after =>
    have eq : after=s := by simpa [FrameAt,run] using frame
    subst after
    change a s=.error error s at run
    simpa [HeadSafeAt,EStateM.run,bind,EStateM.bind,run] using keep
  | ok x after =>
    have eq : after=s := by simpa [FrameAt,run] using frame
    subst after
    change a s=.ok x s at run
    simpa [HeadSafeAt,EStateM.run,bind,EStateM.bind,run] using next x

theorem head_bind_scan (A : Nat → Prop) (H L : Vector Nat 145) (s : St)
    (keep : Keep A H L s) (hh : H[144]!<iq_cap) (hl : L[144]!≤iq_cap)
    (reads : ReadPorts iq_cap) (f : parcel → SailM (Option parcel))
    (next : ∀ p, p.units≤L[144]! → HeadSafeAt A H L s (f p)) :
    HeadSafeAt A H L s (parcel_scan 144 >>= f) := by
  have bound := Proofs.ParcelBound.scan_bound s 144 reads H L keep.1.2.1 keep.1.2.2 keep.1.1 (by decide) hh hl
  cases run : (parcel_scan 144).run s with
  | error error after =>
    have eq : after=s := by simpa [Proofs.ParcelBound.SpecAt,run] using bound
    subst after
    change parcel_scan 144 s=.error error s at run
    simpa [HeadSafeAt,EStateM.run,bind,EStateM.bind,run] using keep
  | ok p after =>
    have pair : after=s ∧ p.units≤L[144]! := by simpa [Proofs.ParcelBound.SpecAt,run] using bound
    obtain ⟨eq,bound⟩ := pair
    subst after
    change parcel_scan 144 s=.ok p s at run
    simpa [HeadSafeAt,EStateM.run,bind,EStateM.bind,run] using next p bound

theorem observed_check_pop (A : Nat → Prop) (H L : Vector Nat 145) (s : St)
    (keep : Keep A H L s) (p : parcel) (bound : p.units≤L[144]!) (c : Nat) (detail : Int)
    (hh : H[144]!<iq_cap) (hl : L[144]!≤iq_cap) (observer : DirectObserver A)
    (reads : ReadPorts iq_cap) (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    HeadSafeAt A H L s (do
      observe_site "dispatch" c 144 H[144]! p.units detail
      Proofs.TextCheckProduct.checkedPop 144 c p
      pure (some p)) := by
  obtain ⟨observed,he,hr,hm⟩ := observer.frame "dispatch" c 144 H[144]! p.units detail s keep.2.2.1
    (by decide) (by omega) (direct_dispatch_domain A H L s keep 144 p.units (by decide) hh bound)
  have kept := keep_same_parts A H L s observed keep hr hm
  have checked := checked_pop_product A 144 c p.units (by decide) hh hl bound reads writes irq observed kept
  change (match (Proofs.TextCheckProduct.checkedPop 144 c p).run observed with
    | .ok _ after => Keep A (popHead H 144 p.units) (popLength L 144 p.units) after
    | .error _ after => Keep A H L after) at checked
  change observe_site "dispatch" c 144 H[144]! p.units detail s=.ok () observed at he
  simp at he
  cases run : (Proofs.TextCheckProduct.checkedPop 144 c p).run observed with
  | error error after =>
    have old : Keep A H L after := by simpa [run] using checked
    change Proofs.TextCheckProduct.checkedPop 144 c p observed=.error error after at run
    simpa [HeadSafeAt,EStateM.run,bind,EStateM.bind,he,run] using old
  | ok u after =>
    cases u
    have result : Keep A (popHead H 144 p.units) (popLength L 144 p.units) after := by simpa [run] using checked
    change Proofs.TextCheckProduct.checkedPop 144 c p observed=.ok () after at run
    simpa [HeadSafeAt,HeadOutcome,EStateM.run,bind,EStateM.bind,pure,EStateM.pure,he,run] using And.intro bound result

theorem consume_safe (A : Nat → Prop) (H L : Vector Nat 145) (s : St)
    (keep : Keep A H L s) (p : parcel) (bound : p.units≤L[144]!) (c : Nat)
    (hh : H[144]!<iq_cap) (hl : L[144]!≤iq_cap) (observer : DirectObserver A)
    (reads : ReadPorts iq_cap) (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    HeadSafeAt A H L s (consumeParcel c p) := by
  have head : Out.readReg iq_head s=.ok H s := by
    simp [Out.readReg,PreSail.readReg,keep.1.2.1,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get]
  have next := head_bind_frame A H L s keep (parcel_earliest 144 p)
    (fun detail => do
      observe_site "dispatch" c 144 H[144]! p.units detail
      Proofs.TextCheckProduct.checkedPop 144 c p
      pure (some p)) (earliest_frame s 144 p)
    (fun detail => observed_check_pop A H L s keep p bound c detail hh hl observer reads writes irq)
  simpa [consumeParcel,HeadSafeAt,EStateM.run,bind,EStateM.bind,head] using next

theorem control_head_safe (A : Nat → Prop) (H L : Vector Nat 145) (s : St)
    (keep : Keep A H L s) (c : Nat) (hh : H[144]!<iq_cap) (hl : L[144]!≤iq_cap)
    (observer : DirectObserver A) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    HeadSafeAt A H L s (controlHead c) := by
  unfold controlHead
  apply head_bind_frame A H L s keep _ _ (read_frame s iq_len)
  intro length
  split
  · apply head_bind_scan A H L s keep hh hl reads
    intro p bound
    apply head_bind_frame A H L s keep _ _ (earliest_frame s 144 p)
    intro earliest
    split
    · exact consume_safe A H L s keep p bound c hh hl observer reads writes irq
    · simpa [HeadSafeAt,HeadOutcome] using keep
  · simpa [HeadSafeAt,HeadOutcome] using keep

def ControlPost (A : Nat → Prop) (H L : Vector Nat 145) (after : St) : Prop :=
  Keep A H L after ∨ ∃ p : parcel, p.units≤L[144]! ∧
    Keep A (popHead H 144 p.units) (popLength L 144 p.units) after

/-- Complete actual control: callback domains are established before observing;
    error paths before pop preserve the old IQ, and any consumed parcel is bounded
    by the SAME scan's captured length. All real barrier/report tail effects remain. -/
theorem actual_control_consumption (A : Nat → Prop) (H L : Vector Nat 145) (s : St)
    (keep : Keep A H L s) (c : Nat) (hh : H[144]!<iq_cap) (hl : L[144]!≤iq_cap)
    (observer : DirectObserver A) (reads : ReadPorts iq_cap)
    (writes : Proofs.Initialized.WritePorts iq_cap) (irq : IrqFrame) :
    match (Out.Functions.control c).run s with
    | .ok _ after => ControlPost A H L after
    | .error _ after => ControlPost A H L after := by
  have hs := control_head_safe A H L s keep c hh hl observer reads writes irq
  rw [actual_control_factor]
  cases run : (controlHead c).run s with
  | error error after =>
    have old : Keep A H L after := by simpa [HeadSafeAt,run] using hs
    change controlHead c s=.error error after at run
    simpa [EStateM.run,bind,EStateM.bind,run] using Or.inl old
  | ok value middle =>
    cases value with
    | none =>
      have old : Keep A H L middle := by simpa [HeadSafeAt,HeadOutcome,run] using hs
      change controlHead c s=.ok none middle at run
      simpa [EStateM.run,bind,EStateM.bind,pure,EStateM.pure,run] using Or.inl old
    | some p =>
      have pair : p.units≤L[144]! ∧ Keep A (popHead H 144 p.units) (popLength L 144 p.units) middle := by
        simpa [HeadSafeAt,HeadOutcome,run] using hs
      have tail_spec := tail_safe A (popHead H 144 p.units) (popLength L 144 p.units) c p
      unfold Proofs.TaggedCheck.Safe at tail_spec
      have tail := tail_spec middle pair.2
      change controlHead c s=.ok (some p) middle at run
      cases rt : (controlTail c p).run middle with
      | ok u after =>
        have final : Keep A (popHead H 144 p.units) (popLength L 144 p.units) after := by simpa [rt] using tail
        simpa [EStateM.run,bind,EStateM.bind,run,show controlTail c p middle=.ok u after from rt] using Or.inr ⟨p,pair.1,final⟩
      | error error after =>
        have final : Keep A (popHead H 144 p.units) (popLength L 144 p.units) after := by simpa [rt] using tail
        simpa [EStateM.run,bind,EStateM.bind,run,show controlTail c p middle=.error error after from rt] using Or.inr ⟨p,pair.1,final⟩

end Proofs.ControlConsumption
