-- Joint actual init/reset/tick history invariant; concrete foreign ports remain explicit.
import Proofs.HistoryComposition
import Proofs.AgtTimedHistory
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
namespace Proofs.CombinedMachineInvariant
abbrev State := ResourceFrameProofs.St
abbrev Ports := Proofs.HistoryComposition.Ports

def Core (time : Nat) (s : State) : Prop :=
 MachineInitProofs.Complete s ∧ Proofs.HistoryComposition.Ready s ∧ AgtTimeBoundProofs.Bound time s

def postState {α : Type} (r : EStateM.Result (Error exception) State α) : State :=
 match r with | .ok _ s => s | .error _ s => s

theorem actual_init_reset_core (ports : Ports) (hcap : 0<Out.Functions.iq_cap)
 (s : State) (view : Proofs.StorageRep.CView) (rep : Proofs.StorageRep.Represents Out.Functions.iq_cap view s)
 (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
 (values : Proofs.OrderedImage.ImageValues image length) :
 ∃ after, (sail_model_init () >>= fun _ => Out.Functions.reset ()).run s = .ok () after ∧ Core 0 after := by
 obtain ⟨initialized,hi,ri,cp,foreign⟩ := Proofs.IqReach.actual_init_rep_complete Out.Functions.iq_cap s s view rep (fun _ _ => rfl)
 obtain ⟨same,hs,hb⟩ := AgtTimeClosureProofs.actual_initialized_bound (cycle:=0) s
 rw [hi] at hs
 cases hs
 obtain ⟨after,hr,ready⟩ := Proofs.HistoryComposition.actual_reset_from_complete hcap initialized view ri cp ports.writes image length values
 have complete := ResourceCompleteProofs.reset_complete ports.complete () initialized cp
 have bound := AgtTimeClosureProofs.reset_bound (cycle:=0) ports.complete () initialized hb
 change Out.Functions.reset () initialized = .ok () after at hr
 change sail_model_init () s = .ok () initialized at hi
 simp only [EStateM.run,hr] at complete bound
 refine ⟨after,?_,complete,ready,bound⟩
 simp only [EStateM.run,bind,EStateM.bind,hi,hr]

theorem actual_tick_core (ports : Ports) (old next : Nat) (s : State)
 (core : Core old s) (mono : old≤next) : Core next (postState ((tick next).run s)) := by
 have complete := ResourceCompleteProofs.tick_complete ports.complete next s core.1
 have ready := Proofs.HistoryComposition.actual_tick_boundary ports next s core.2.1
 have bound := AgtTimedHistoryProofs.actual_tick_bound ports.complete s old next core.2.2 mono
 cases hr : (tick next).run s <;>
  simp only [hr,postState,AgtTimedHistoryProofs.ResultState] at complete ready bound ⊢ <;>
  exact ⟨complete,ready,bound⟩

def OrderedCycles : Nat → List Nat → Prop
 | _,[] => True
 | old,c::rest => old≤c ∧ OrderedCycles c rest

-- Stops at the first actual partial-error exit; the returned time is visited, never a future ghost time.
def timedRun (old : Nat) (cycles : List Nat) (s : State) : Nat × EStateM.Result (Error exception) State Unit :=
 match cycles with
 | [] => (old,.ok () s)
 | c::rest => match (tick c).run s with
   | .ok _ after => timedRun c rest after
   | .error err after => (c,.error err after)

theorem timed_result_is_production (old : Nat) (cycles : List Nat) (s : State) :
 (timedRun old cycles s).2 = (Proofs.HistoryComposition.ticks cycles).run s := by
 induction cycles generalizing old s with
 | nil => rfl
 | cons c rest ih =>
   simp only [timedRun,Proofs.HistoryComposition.ticks,EStateM.run,bind,EStateM.bind]
   cases hr : tick c s with
   | ok value after => simpa [hr] using ih c after
   | error err after => rfl

theorem actual_finite_history_core (ports : Ports) (old : Nat) (cycles : List Nat) (s : State)
 (core : Core old s) (ordered : OrderedCycles old cycles) :
 Core (timedRun old cycles s).1 (postState (timedRun old cycles s).2) := by
 induction cycles generalizing old s with
 | nil => exact core
 | cons c rest ih =>
   have next := actual_tick_core ports old c s core ordered.1
   cases hr : (tick c).run s with
   | ok value after =>
     simp only [hr,postState] at next
     simpa only [timedRun,hr] using ih c after next ordered.2
   | error err after =>
     simpa only [timedRun,hr,postState] using next

theorem actual_initialized_reset_finite_history_core (ports : Ports) (hcap : 0<Out.Functions.iq_cap)
 (s : State) (view : Proofs.StorageRep.CView) (rep : Proofs.StorageRep.Represents Out.Functions.iq_cap view s)
 (image : Nat → Nat → BitVec 32) (length : Nat → Nat)
 (values : Proofs.OrderedImage.ImageValues image length) (cycles : List Nat) (ordered : OrderedCycles 0 cycles) :
 ∃ resetState, (sail_model_init () >>= fun _ => Out.Functions.reset ()).run s = .ok () resetState ∧
 Core (timedRun 0 cycles resetState).1 (postState ((Proofs.HistoryComposition.ticks cycles).run resetState)) := by
 obtain ⟨resetState,run,core⟩ := actual_init_reset_core ports hcap s view rep image length values
 refine ⟨resetState,run,?_⟩
 rw [← timed_result_is_production 0 cycles resetState]
 exact actual_finite_history_core ports 0 cycles resetState core ordered

end Proofs.CombinedMachineInvariant
