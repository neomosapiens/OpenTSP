-- Reset-reachable histories of actual tick/reset transitions at monotonic times.
import Proofs.AgtTimeClosure
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace AgtTimedHistoryProofs
open ResourceFrameProofs ResourceInvariantProofs AgtTimeBoundProofs AgtTimeClosureProofs
abbrev St:=ResourceFrameProofs.St

theorem bound_monotone (old next:Nat) (h:old≤next) (s:St) (hs:Bound old s):Bound next s:=by
 constructor
 all_goals intro xs hx i hi
 · have ha:=hs.1 xs hx i hi
   exact ⟨ha.1,Nat.le_trans ha.2 h⟩
 · have ha:=hs.2 xs hx i hi
   exact ⟨ha.1,Nat.le_trans ha.2 h⟩

def ResultState {α:Type} (result:EStateM.Result (Error exception) St α):St:=
 match result with | .ok _ s=>s | .error _ s=>s

theorem actual_tick_bound (ports:Ports) (s:St) (old next:Nat)
 (hs:Bound old s) (ht:old≤next):Bound next (ResultState ((tick next).run s)):=by
 have hh:=tick_bound (cycle:=next) ports s (bound_monotone old next ht s hs)
 cases hr:(tick next).run s <;> simp only [hr] at hh <;> exact hh

theorem actual_reset_bound (ports:Ports) (s:St) (time:Nat)
 (hs:Bound time s):Bound time (ResultState ((reset ()).run s)):=by
 have hh:=reset_bound (cycle:=time) ports () s hs
 cases hr:(reset ()).run s <;> simp only [hr] at hh <;> exact hh

inductive ActualHistory : Nat→St→Prop where
 | initialized (before:St) : ActualHistory 0 (ResultState ((sail_model_init ()).run before))
 | reset {time:Nat} {before:St} : ActualHistory time before→
    ActualHistory time (ResultState ((reset ()).run before))
 | tick {old next:Nat} {before:St}:ActualHistory old before→old≤next→
    ActualHistory next (ResultState ((tick next).run before))

theorem actual_history_bound (ports:Ports) {time:Nat} {s:St}
 (history:ActualHistory time s):Bound time s:=by
 induction history with
 | initialized before=>
   obtain ⟨after,hr,hb⟩:=actual_initialized_bound (cycle:=0) before
   simpa only [hr,ResultState] using hb
 | reset _ ih=>exact actual_reset_bound ports _ _ ih
 | tick _ ht ih=>exact actual_tick_bound ports _ _ _ ih ht

theorem actual_get_bound {time:Nat} (s:St) (hs:Bound time s) (q p:Nat) (a:agt)
 (hget:(agt_get q p).run s=.ok a s):At time a:=by
 by_cases hp:p=0
 · cases hr:s.regs.get? Register.agts with
   | none=>
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     cases hget
   | some arr=>
     have ha:=array_get_at arr (hs.1 arr hr) q
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     simpa [hget] using ha
 · cases hr:s.regs.get? Register.agts_w with
   | none=>
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     cases hget
   | some arr=>
     have ha:=array_get_at arr (hs.2 arr hr) q
     simp [agt_get,hp,Out.readReg,PreSail.readReg,EStateM.run,bind,EStateM.bind,
      pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,hr] at hget
     simpa [hget] using ha

theorem actual_history_producer_origin (ports:Ports) {time:Nat} {s:St}
 (history:ActualHistory time s) (q p:Nat) (a:agt)
 (hget:(agt_get q p).run s=.ok a s):
 a.t0+(max ((time:Int)-a.t0) 0).toNat=time:=by
 have ha:=actual_get_bound s (actual_history_bound ports history) q p a hget
 exact ExecuteProducerProofs.agt_reference_origin a.t0 time (Nat.le_trans ha.1 ha.2)

end AgtTimedHistoryProofs
