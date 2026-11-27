-- Actual execute-ring producer/consumer transitions and non-aliasing ownership.
import Proofs.ExecuteProducer
import Proofs.ExecuteRingFrames
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace ExecutionHistoryProofs
open ExecuteProducerProofs
abbrev St:=ResourceFrameProofs.St
abbrev Ring:=Vector (List exec_element) 16

def ReadyAt (c:Nat) (r:Ring) : Prop :=
 ∀ k:Nat,k<16 → ∀ e,e∈r[k]! → c<=target e ∧ target e<c+16 ∧ target e%16=k ∧ e.q<145

def FutureAt (c:Nat) (r:Ring) : Prop :=
 ∀ k:Nat,k<16 → ∀ e,e∈r[k]! → c<target e ∧ target e<c+16 ∧ target e%16=k ∧ e.q<145

theorem current_bucket_exact (c:Nat) (r:Ring) (h:ReadyAt c r) (e:exec_element) (he:e∈r[c%16]!) :
 target e=c ∧ e.q<145 := by
 have hh:=h (c%16) (Nat.mod_lt c (by decide)) e he
 have hm1:=Nat.mod_add_div c 16
 have hm2:=Nat.mod_add_div (target e) 16
 constructor
 · omega
 · exact hh.2.2.2

def clear (r:Ring) (c:Nat) : Ring:=vectorUpdate r (c%16) []

theorem after_clear_future (c:Nat) (r:Ring) (h:ReadyAt c r) : FutureAt c (clear r c) := by
 intro k hk e he
 by_cases same:k=c%16
 · subst k
   have hi:c%16<16:=Nat.mod_lt c (by decide)
   simp [clear,vectorUpdate,Vector.set!,hi] at he
 · have hget:(clear r c)[k]! =r[k]! := by
    exact ResourceFrameProofs.vector_update_other 16 k (c%16) r [] hk (Nat.mod_lt c (by decide)) same
   rw [hget] at he
   have hh:=h k hk e he
   have hn:target e≠c:=by intro eq; apply same; simpa [eq] using hh.2.2.1.symm
   exact ⟨by omega,hh.2⟩

def clearedState (s:St) (r:Ring) (c:Nat) : St:=
 {s with regs:=s.regs.insert Register.execute_ring (clear r c)}

theorem actual_execute_due_run (s:St) (r:Ring)
 (hr:s.regs.get? Register.execute_ring=some r) (c:Nat) :
 (execute_due c).run s=(execute_all r[c%16]!).run (clearedState s r c) := by
 have hm:(tmod_int (c:Int) 16).toNat=c%16:=by simpa using actual_mod16 c 0 0
 simp [execute_due,clearedState,clear,hm,Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
 EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
 modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,hr]
 change execute_all r[(tmod_int (c:Int) 16).toNat]! {s with regs:=s.regs.insert Register.execute_ring (vectorUpdate r (tmod_int (c:Int) 16).toNat [])} = _
 rw [hm]

theorem actual_due_clears_before_execution (ports:ExecuteRingFrameProofs.Ports)
 (s:St) (r:Ring) (hr:s.regs.get? Register.execute_ring=some r) (c:Nat) :
 match (execute_due c).run s with
 | .ok _ after=>after.regs.get? Register.execute_ring=some (clear r c)
 | .error _ after=>after.regs.get? Register.execute_ring=some (clear r c) := by
 rw [actual_execute_due_run s r hr c]
 have hf:=ExecuteRingFrameProofs.actual_execute_all_frame ports r[c%16]! (clearedState s r c)
 cases hrun:(execute_all r[c%16]!).run (clearedState s r c)
 all_goals simp only [hrun,ExecuteRingFrameProofs.RView] at hf
 all_goals simpa [clearedState,Std.ExtDHashMap.get?_insert] using hf

def push (r:Ring) (e:exec_element) : Ring:=
 vectorUpdate r (target e%16) (r[target e%16]!++[e])

theorem push_future (c:Nat) (r:Ring) (h:FutureAt c r) (e:exec_element)
 (he:c<target e ∧ target e<c+16 ∧ e.q<145) : FutureAt c (push r e) := by
 intro k hk old hold
 have hi:target e%16<16:=Nat.mod_lt _ (by decide)
 by_cases same:k=target e%16
 · subst k
   have hg:(push r e)[target e%16]! =r[target e%16]!++[e] := by simp [push,vectorUpdate,Vector.set!,hi]
   rw [hg] at hold
   rcases List.mem_append.mp hold with original|inserted
   · exact h _ hi old original
   · have ht:old=e:=List.mem_singleton.mp inserted
     subst old
     exact ⟨he.1,he.2.1,rfl,he.2.2⟩
 · have hg:(push r e)[k]! =r[k]! :=
    ResourceFrameProofs.vector_update_other 16 k (target e%16) r _ hk hi same
   rw [hg] at hold
   exact h k hk old hold

theorem actual_delayed_enqueue_future (s:St) (r:Ring) (hr:s.regs.get? Register.execute_ring=some r)
 (q:Nat) (i:instruction) (c0 k c:Nat) (off soff:Int)
 (hq:q<145) (horigin:c0+k=c) (hpos:0<(latency_of_instr i).d_skew)
 (hmax:(latency_of_instr i).d_skew<=15) (h:FutureAt c r) :
 ∃after newRing,(run_element q i c0 k off soff).run s=.ok () after ∧
 after.regs.get? Register.execute_ring=some newRing ∧ FutureAt c newRing := by
 let e:=element q i c0 k off soff
 let nr:=push r e
 let after:St:={s with regs:=s.regs.insert Register.execute_ring nr}
 have hh:=delayed_horizon q i c0 k c off soff horigin hpos hmax
 have hn:(latency_of_instr i).d_skew≠0:=Nat.ne_of_gt hpos
 have ht:target e=c+(latency_of_instr i).d_skew:=by dsimp [e,target,element]; omega
 have hf:FutureAt c nr:=push_future c r h e ⟨hh.1,Nat.lt_of_le_of_lt hh.2 (by omega:c+15<c+16),hq⟩
 refine ⟨after,nr,?_,?_,hf⟩
 · simpa only [after,nr,push,e,target,element] using actual_delayed_enqueue s r hr q i c0 k off soff hn
 · simp [after,Std.ExtDHashMap.get?_insert]

theorem future_next_ready (c:Nat) (r:Ring) (h:FutureAt c r) : ReadyAt (c+1) r := by
 intro k hk e he
 have hh:=h k hk e he
 exact ⟨by omega,by omega,hh.2.2⟩

theorem empty_ready (c:Nat) : ReadyAt c (vectorInit []) := by
 intro k hk e he
 simp [vectorInit,hk] at he

end ExecutionHistoryProofs
