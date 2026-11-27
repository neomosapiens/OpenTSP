-- Normal initialized/reset/finite-cycle patrol callback-domain composition.
import Proofs.ResourceReach
import Proofs.ResourcePresence
import Proofs.MachineInit
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
namespace ResourceReadyProofs
open ResourceFrameProofs ResourceInvariantProofs

def Ready (s : ResourceFrameProofs.St) := PatrolValidityProofs.Valid s ∧ ResourcePresenceProofs.Present s

theorem inv_and {α : Type} (p q : ResourceFrameProofs.St → Prop) (a : SailM α)
 (hp : Inv p a) (hq : Inv q a) : Inv (fun s => p s ∧ q s) a := by
 intro s hs
 have h1:=hp s hs.1
 have h2:=hq s hs.2
 cases hx:a.run s <;> simpa [hx] using And.intro h1 h2

theorem actual_init_ready (s : ResourceFrameProofs.St) :
 ∃after,(sail_model_init ()).run s=.ok () after ∧ Ready after := by
 obtain ⟨a,ha,hv⟩:=PatrolValidityProofs.actual_init_valid s
 obtain ⟨b,hb,hn,hp,hr⟩:=MachineInitProofs.actual_init_scrub s
 rw [ha] at hb
 cases hb
 exact ⟨a,ha,hv,⟨_,hp⟩,⟨_,hr⟩⟩

theorem actual_reset_ready (v : ResourceReachProofs.Ports) (p : ResourcePresenceProofs.Ports) :
 Inv Ready (reset ()) := inv_and _ _ _ (ResourceReachProofs.reset_valid v ()) (ResourcePresenceProofs.reset_present p ())

theorem actual_tick_ready (v : ResourceReachProofs.Ports) (p : ResourcePresenceProofs.Ports) (c : Nat) :
 Inv Ready (tick c) := inv_and _ _ _ (ResourceReachProofs.tick_valid v c) (ResourcePresenceProofs.tick_present p c)

def tickHistory : List Nat → SailM Unit
 | [] => pure ()
 | c::cs => tick c >>= fun _ => tickHistory cs

theorem actual_finite_history_ready (v : ResourceReachProofs.Ports) (p : ResourcePresenceProofs.Ports) (cs : List Nat) :
 Inv Ready (tickHistory cs) := by
 induction cs with
 | nil => exact pure_inv _ _
 | cons c cs ih => exact bind_inv _ _ _ (actual_tick_ready v p c) (fun _ => ih)

theorem actual_reset_history_ready (v : ResourceReachProofs.Ports) (p : ResourcePresenceProofs.Ports) (cs : List Nat) :
 Inv Ready (reset () >>= fun _ => tickHistory cs) :=
 bind_inv _ _ _ (actual_reset_ready v p) (fun _ => actual_finite_history_ready v p cs)

theorem actual_initialized_lifecycle (v : ResourceReachProofs.Ports) (p : ResourcePresenceProofs.Ports) (cs : List Nat)
 (s : ResourceFrameProofs.St) :
 match (sail_model_init () >>= fun _ => reset () >>= fun _ => tickHistory cs).run s with
 | .ok _ after => Ready after
 | .error _ after => Ready after := by
 obtain ⟨init,hi,hr⟩:=actual_init_ready s
 have hh:=actual_reset_history_ready v p cs init hr
 simp only [EStateM.run,bind,EStateM.bind,show sail_model_init () s=.ok () init from hi]
 cases hx : reset () init with
 | ok u after =>
   cases ht : tickHistory cs after <;> simpa [ResourceInvariantProofs.Inv,EStateM.run,bind,EStateM.bind,hx,ht] using hh
 | error err after => simpa [ResourceInvariantProofs.Inv,EStateM.run,bind,EStateM.bind,hx] using hh

theorem ready_pointer (s : ResourceFrameProofs.St) (h : Ready s) :
 ∃ptr,s.regs.get? Register.scrub_ptr=some ptr ∧ ∀q,q<88 → ptr[q]!<Out.Functions.words_per_slice := by
 obtain ⟨ptr,hptr⟩:=h.2.1
 change s.regs.get? Register.scrub_ptr=some ptr at hptr
 refine ⟨ptr,hptr,?_⟩
 have hv:=h.1.1
 simpa [PatrolValidityProofs.Valid,PatrolValidityProofs.ValidView,PatrolValidityProofs.View,PatrolValidityProofs.PtrOk,hptr] using hv

theorem ready_saved_address (s : ResourceFrameProofs.St) (h : Ready s) :
 ∃pending,s.regs.get? Register.scrub_pending=some pending ∧ ∀q,q<88 → ∀w,pending[q]! = some w → w.addr<Out.Functions.words_per_slice := by
 obtain ⟨p,hp⟩:=h.2.2
 change s.regs.get? Register.scrub_pending=some p at hp
 refine ⟨p,hp,?_⟩
 have hv:=h.1.2
 simpa [PatrolValidityProofs.Valid,PatrolValidityProofs.ValidView,PatrolValidityProofs.View,PatrolValidityProofs.RepairOk,hp] using hv
end ResourceReadyProofs
