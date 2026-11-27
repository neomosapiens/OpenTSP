import Proofs.QueueConsumption
import Proofs.AllocationHistory
import Proofs.HistoryFrames
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs Proofs.StorageRep
open Proofs.TagCoverage Proofs.TaggedCheck Proofs.OrderedDelivery Proofs.IqReach
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace Proofs.HistoryComposition
abbrev State := ResourceFrameProofs.St

/-- Every history boundary carries its actual allocation and captured logical
    vectors. The allocator witness is retained across consumers and rebased only
    from the preceding proved physical projection, never reconstructed from a
    possibly emptied logical length. -/
def Ready (s : State) : Prop := ∃ A H L, Keep A H L s

theorem captured_bounds (A : Nat → Prop) (H L : Vector Nat 145) (s : State)
    (keep : Keep A H L s) : ∀ q, q<145 → H[q]!<iq_cap ∧ L[q]!≤iq_cap := by
  obtain ⟨H0,L0,hH,hL,bounds⟩ := keep.2.1
  have eH : H0=H := by rw [keep.1.2.1] at hH; cases hH; rfl
  have eL : L0=L := by rw [keep.1.2.2] at hL; cases hL; rfl
  subst H0; subst L0
  exact fun q hq => ⟨(bounds q hq).1,(bounds q hq).2.1⟩

theorem producer_boundary (A : Nat → Prop) (H L : Vector Nat 145) (s : State)
    (keep : Keep A H L s) : Proofs.AllocationHistory.Ready iq_cap s A s := by
  apply Proofs.AllocationHistory.from_allocation iq_cap s A s
  · exact ⟨⟨keep.1.1,keep.2.2.2,fun _ _ => rfl⟩,keep.2.1⟩
  · exact keep.2.2.1

theorem consumer_boundary (base : State) (A : Nat → Prop) (s : State)
    (ready : Proofs.AllocationHistory.Ready iq_cap base A s) : Ready s := by
  obtain ⟨L,hL,allocated,_⟩ := Proofs.AllocationHistory.expose_boundary iq_cap base A s ready
  obtain ⟨H0,L0,hH,hL0,bounds⟩ := ready.1.2
  have eq : L0=L := by rw [hL] at hL0; cases hL0; rfl
  subst L0
  exact ⟨(fun q => A q ∨ 0<L[q]!),H0,L,⟨ready.1.1.1,hH,hL⟩,
    ready.1.2,allocated,ready.1.1.2.1⟩

theorem actual_delivery_boundary (ports : Proofs.Initialized.WritePorts iq_cap)
    (q c : Nat) (hq : q<145) : Total Ready (iq_deliver q c) := by
  intro s state
  obtain ⟨A,H,L,keep⟩ := state
  have start := producer_boundary A H L s keep
  obtain ⟨value,after,run,good⟩ := Proofs.AllocationHistory.actual_delivery_allocation
    iq_cap Out.Functions.iq_rate q c hq s A ports s start
  exact ⟨value,after,by simpa only [Proofs.Capacity.deliver_production] using run,
    consumer_boundary s A after good⟩

structure Ports : Prop where
  resource : ResourceExecuteFrameProofs.Ports
  complete : ResourceCompleteProofs.Ports
  observer : ∀ A, Proofs.ControlConsumption.DirectObserver A
  reads : Proofs.ParcelFrame.ReadPorts iq_cap
  writes : Proofs.Initialized.WritePorts iq_cap
  irq : Proofs.TaggedCheck.IrqFrame

theorem actual_queue_boundary (ports : Ports) (q c : Nat) (hq : q<144) :
    ResourceInvariantProofs.Inv Ready (tick_queue q c) := by
  intro s state
  obtain ⟨A,H,L,keep⟩ := state
  have bounds := captured_bounds A H L s keep q (by omega)
  have result := Proofs.QueueConsumption.actual_tick_queue_consumption A H L
    ports.resource ports.complete (ports.observer A) ports.reads ports.writes ports.irq
    s keep q c hq bounds.1 bounds.2
  cases run : (tick_queue q c).run s with
  | ok value after =>
    have good : Proofs.QueueConsumption.QueuePost A H L q after := by simpa [run] using result
    rcases good with old | ⟨p,_,changed⟩
    · exact ⟨A,H,L,old⟩
    · exact ⟨A,_,_,changed⟩
  | error error after =>
    have good : Proofs.QueueConsumption.QueuePost A H L q after := by simpa [run] using result
    rcases good with old | ⟨p,_,changed⟩
    · exact ⟨A,H,L,old⟩
    · exact ⟨A,_,_,changed⟩

theorem actual_control_boundary (ports : Ports) (c : Nat) :
    ResourceInvariantProofs.Inv Ready (Out.Functions.control c) := by
  intro s state
  obtain ⟨A,H,L,keep⟩ := state
  have bounds := captured_bounds A H L s keep 144 (by decide)
  have result := Proofs.ControlConsumption.actual_control_consumption A H L s keep c
    bounds.1 bounds.2 (ports.observer A) ports.reads ports.writes ports.irq
  cases run : (Out.Functions.control c).run s with
  | ok value after =>
    have good : Proofs.ControlConsumption.ControlPost A H L after := by simpa [run] using result
    rcases good with old | ⟨p,_,changed⟩
    · exact ⟨A,H,L,old⟩
    · exact ⟨A,_,_,changed⟩
  | error error after =>
    have good : Proofs.ControlConsumption.ControlPost A H L after := by simpa [run] using result
    rcases good with old | ⟨p,_,changed⟩
    · exact ⟨A,H,L,old⟩
    · exact ⟨A,_,_,changed⟩

/-- Actual operations can be interleaved on arbitrary queues. Producer boundaries
    supply fresh allocation witnesses; consumers retain their incoming allocator,
    including when their final logical item is consumed. -/
inductive Action where
 | deliver (q c : Nat) (bound : q<145)
 | control (c : Nat)
 | queue (q c : Nat) (bound : q<144)

def action : Action → SailM Unit
 | .deliver q c _ => iq_deliver q c >>= fun _ => pure ()
 | .control c => Out.Functions.control c
 | .queue q c _ => tick_queue q c

theorem total_inv {α : Type} (a : SailM α) (h : Total Ready a) : ResourceInvariantProofs.Inv Ready a := by
 intro s hs
 obtain ⟨v,t,run,good⟩ := h s hs
 simpa [run] using good

theorem action_ready (ports : Ports) (a : Action) : ResourceInvariantProofs.Inv Ready (action a) := by
 cases a with
 | deliver q c bound =>
   exact ResourceInvariantProofs.bind_inv _ _ _
     (total_inv _ (actual_delivery_boundary ports.writes q c bound))
     (fun _ => ResourceInvariantProofs.pure_inv _ _)
 | control c => exact actual_control_boundary ports c
 | queue q c bound => exact actual_queue_boundary ports q c bound

def history : List Action → SailM Unit
 | [] => pure ()
 | a::rest => action a >>= fun _ => history rest

theorem actual_interleaved_history (ports : Ports) (actions : List Action) :
    ResourceInvariantProofs.Inv Ready (history actions) := by
 induction actions with
 | nil => exact ResourceInvariantProofs.pure_inv _ _
 | cons a rest ih => exact ResourceInvariantProofs.bind_inv _ _ _ (action_ready ports a) (fun _ => ih)

open ResourceInvariantProofs ResourceFrameProofs

theorem frame_inv {α : Type} (a : SailM α)
 (frame : Keeps Proofs.HistoryFrames.View a)
 (full : Inv MachineInitProofs.Complete a) : Inv Ready a := by
 intro s state
 obtain ⟨A,H,L,keep⟩:=state
 have hf:=frame s
 have hc:=full s keep.2.2.2
 cases run:a.run s with
 | ok value after =>
   have eq : Proofs.HistoryFrames.View after=Proofs.HistoryFrames.View s := by simpa [run] using hf
   have cp : MachineInitProofs.Complete after := by simpa [run] using hc
   have old := Proofs.QueueConsumption.keep_projection A H L s after keep
     (congrArg Prod.fst eq) (congrArg (fun v=>v.2.1) eq) (congrArg (fun v=>v.2.2) eq) cp
   exact ⟨A,H,L,old⟩
 | error error after =>
   have eq : Proofs.HistoryFrames.View after=Proofs.HistoryFrames.View s := by simpa [run] using hf
   have cp : MachineInitProofs.Complete after := by simpa [run] using hc
   have old := Proofs.QueueConsumption.keep_projection A H L s after keep
     (congrArg Prod.fst eq) (congrArg (fun v=>v.2.1) eq) (congrArg (fun v=>v.2.2) eq) cp
   exact ⟨A,H,L,old⟩

theorem other_write_inv (r : Register) (v : RegisterType r)
 (hh : r≠Register.iq_head) (hl : r≠Register.iq_len) : Inv Ready (Out.writeReg r v) := by
 intro s state
 obtain ⟨A,H,L,keep⟩:=state
 have kept:=preserves_other_write A (H:=H) (L:=L) r v hh hl s keep
 exact ⟨A,H,L,kept⟩

theorem scan_inv (ports : Ports) (q : Nat) (hq : q<145) : Inv Ready (parcel_scan q) := by
 intro s state
 obtain ⟨A,H,L,keep⟩:=state
 have bounds:=captured_bounds A H L s keep q hq
 have hs:=Proofs.ParcelBound.scan_bound s q ports.reads H L keep.1.2.1 keep.1.2.2
   keep.1.1 hq bounds.1 bounds.2
 cases run:(parcel_scan q).run s with
 | ok p after =>
   have eq : after=s := (show after=s ∧ p.units≤L[q]! from by simpa [Proofs.ParcelBound.SpecAt,run] using hs).1
   subst after
   exact ⟨A,H,L,keep⟩
 | error error after =>
   have eq : after=s := by simpa [Proofs.ParcelBound.SpecAt,run] using hs
   subst after
   exact ⟨A,H,L,keep⟩

theorem earliest_inv (q : Nat) (p : parcel) : Inv Ready (parcel_earliest q p) := by
 intro s state
 have frame:=Proofs.ControlConsumption.earliest_frame s q p
 cases run:(parcel_earliest q p).run s with
 | ok v after =>
   have eq : after=s := by simpa [Proofs.ParcelFrame.FrameAt,run] using frame
   subst after; exact state
 | error e after =>
   have eq : after=s := by simpa [Proofs.ParcelFrame.FrameAt,run] using frame
   subst after; exact state

attribute [local irreducible] Ready ResourceInvariantProofs.Inv

theorem actual_enqueue_boundary (ports : Ports) (q c : Nat) (hq : q<145) : Inv Ready (enqueue q c) := by
 unfold enqueue
 dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
 repeat' first
  | exact pure_inv _ _
  | exact read_inv _ _
  | exact earliest_inv _ _
  | exact scan_inv ports q hq
  | exact total_inv _ (actual_delivery_boundary ports.writes q c hq)
  | exact frame_inv _ (Proofs.HistoryFrames.free_at_frame ports.resource _ _) (ResourceCompleteProofs.free_at_complete ports.complete _ _)
  | exact frame_inv _ (Proofs.HistoryFrames.project_frame _ (ResourceExecuteFrameProofs.report_frame ports.resource _ _ _)) (ResourceCompleteProofs.report_complete ports.complete _ _ _)
  | apply other_write_inv <;> decide
  | apply bind_inv
  | split
  | intro


theorem local_fetch_complete_boundary (ports : Ports) (c : Nat) : Inv Ready (local_fetch_complete c) :=
 frame_inv _ (Proofs.HistoryFrames.local_fetch_complete_frame ports.resource c) (ResourceCompleteProofs.local_fetch_complete_complete ports.complete c)

theorem barrier_retire_boundary (ports : Ports) (c : Nat) : Inv Ready (barrier_retire c) :=
 frame_inv _ (Proofs.HistoryFrames.barrier_retire_frame ports.resource c) (ResourceCompleteProofs.barrier_retire_complete ports.complete c)

theorem commit_due_boundary (ports : Ports) (c : Nat) : Inv Ready (commit_due c) :=
 frame_inv _ (Proofs.HistoryFrames.commit_due_frame ports.resource c) (ResourceCompleteProofs.commit_due_complete ports.complete c)

theorem execute_due_boundary (ports : Ports) (c : Nat) : Inv Ready (execute_due c) :=
 frame_inv _ (Proofs.HistoryFrames.execute_due_frame ports.resource c) (ResourceCompleteProofs.execute_due_complete ports.complete c)

theorem stream_fetch_complete_boundary (ports : Ports) (c : Nat) : Inv Ready (stream_fetch_complete c) :=
 frame_inv _ (Proofs.HistoryFrames.stream_fetch_complete_frame ports.resource c) (ResourceCompleteProofs.stream_fetch_complete_complete ports.complete c)

theorem unit_state_flush_boundary (ports : Ports) (c : Nat) : Inv Ready (unit_state_flush c) :=
 frame_inv _ (Proofs.HistoryFrames.unit_state_flush_frame ports.resource c) (ResourceCompleteProofs.unit_state_flush_complete ports.complete c)

theorem scrub_run_boundary (ports : Ports) (c : Int) : Inv Ready (scrub_run c) :=
 frame_inv _ (Proofs.HistoryFrames.scrub_run_frame ports.resource c) (ResourceCompleteProofs.scrub_run_complete ports.complete c)


/-- The complete production tick invokes only proved phase transitions. In
    particular enqueue may grow the allocator, and each control/data consumer
    captures the resulting allocator/head/length before consuming. -/
theorem actual_tick_boundary (ports : Ports) (c : Nat) : Inv Ready (tick c) := by
 unfold tick
 dsimp only [ForIn.forIn,ForIn'.forIn',instForInOfForIn']
 repeat' first
  | exact pure_inv _ _
  | apply other_write_inv <;> decide
  | exact scrub_run_boundary ports _
  | exact local_fetch_complete_boundary ports _
  | exact barrier_retire_boundary ports _
  | exact commit_due_boundary ports _
  | exact execute_due_boundary ports _
  | exact stream_fetch_complete_boundary ports _
  | exact unit_state_flush_boundary ports _
  | exact actual_control_boundary ports _
  | apply actual_enqueue_boundary ports <;> (simp only [Membership.mem] at *; simp at *; omega)
  | apply actual_queue_boundary ports <;> (simp only [Membership.mem] at *; simp at *; omega)
  | apply forIn_inv
  | apply bind_inv
  | split
  | intro

def ticks : List Nat → SailM Unit
 | [] => pure ()
 | c::rest => tick c >>= fun _ => ticks rest

theorem actual_tick_history (ports : Ports) (cycles : List Nat) : Inv Ready (ticks cycles) := by
 induction cycles with
 | nil => exact pure_inv _ _
 | cons c rest ih => exact bind_inv _ _ _ (actual_tick_boundary ports c) (fun _=>ih)

/-- Reset starts from the concrete previous physical tag projection. It does not
    infer allocator emptiness from logical lengths, nor discard old tag arrays. -/
theorem actual_reset_from_complete (hcap : 0<iq_cap) (s : State) (view : CView)
 (rep : Represents iq_cap view s) (complete : MachineInitProofs.Complete s)
 (ports : Proofs.Initialized.WritePorts iq_cap)
 (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
 (values : Proofs.OrderedImage.ImageValues image length) :
 ∃ after,(Out.Functions.reset ()).run s=.ok () after ∧ Ready after := by
 obtain ⟨v,middle,hprefix,cp⟩:=Proofs.IqReach.machine_prefix_complete s complete
 obtain ⟨other,hprefix',mem⟩:=Proofs.IqReach.machine_prefix_memory s
 rw [hprefix] at hprefix'
 cases hprefix'
 let A : Nat → Prop := fun q=>∃slot,slot<iq_cap ∧ TagAt iq_cap s q slot
 have allocated : Allocation iq_cap A s := fun _ _ => Iff.rfl
 have am : Allocation iq_cap A middle := Proofs.AllocationHistory.allocation_memory iq_cap A s middle allocated mem
 have rm := Proofs.IqReach.represents_memory iq_cap view s middle mem rep
 obtain ⟨after,run,good⟩:=Proofs.AllocationHistory.actual_reset_allocation iq_cap hcap middle middle
   view rm cp (fun _ _=>rfl) A am ports image length values
 refine ⟨after,?_,consumer_boundary middle A after good⟩
 rw [Proofs.IqReach.machine_reset_split,Proofs.OrderedImage.reset_production]
 change Proofs.IqReach.machineResetPrefix s=.ok () middle at hprefix
 change Proofs.OrderedImage.resetC iq_cap middle=.ok () after at run
 simpa [EStateM.run,bind,EStateM.bind,hprefix] using run

/-- Actual initialization, complete reset and arbitrary finite production tick
    histories preserve the concrete queue/allocation invariant on both success
    and partial-error exits. No whole-caller invariant is assumed. -/
theorem actual_initialized_reset_tick_history (ports : Ports) (hcap : 0<iq_cap)
 (s : State) (view : CView) (rep : Represents iq_cap view s)
 (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
 (values : Proofs.OrderedImage.ImageValues image length) (cycles : List Nat) :
 match (sail_model_init () >>= fun _=>Out.Functions.reset () >>= fun _=>ticks cycles).run s with
 | .ok _ after => Ready after
 | .error _ after => Ready after := by
 obtain ⟨initialized,hi,ri,cp,foreign⟩:=Proofs.IqReach.actual_init_rep_complete iq_cap s s view rep (fun _ _=>rfl)
 obtain ⟨reset,hr,ready⟩:=actual_reset_from_complete hcap initialized view ri cp ports.writes image length values
 have invariant:=actual_tick_history ports cycles
 unfold ResourceInvariantProofs.Inv at invariant
 have result:=invariant reset ready
 change sail_model_init () s=.ok () initialized at hi
 change Out.Functions.reset () initialized=.ok () reset at hr
 simp only [EStateM.run,bind,EStateM.bind,hi,hr]
 cases run:ticks cycles reset <;> simpa [ResourceInvariantProofs.Inv,EStateM.run,run] using result
end Proofs.HistoryComposition

