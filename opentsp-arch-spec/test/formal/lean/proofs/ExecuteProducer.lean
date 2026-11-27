-- Actual delayed execution insertion, with the real instruction/queue/offset origin.
import Proofs.ResourceReadLoops
import Proofs.SrfLifetime
open Sail Sail.ConcurrencyInterfaceV1 Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace ExecuteProducerProofs
open ResourceReadLoopProofs

def element (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) : exec_element :=
 {q:=q,instr:=i,c0:=c,k:=k,off:=off,soff:=soff}
def target (e : exec_element) : Nat := e.c0+e.k+(latency_of_instr e.instr).d_skew

theorem actual_append (xs : List exec_element) (e : exec_element) : append_element xs e=xs++[e] := by
 induction xs with
 | nil => rfl
 | cons h t ih => simp [append_element,ih]

theorem actual_mod16 (c k skew : Nat) :
 (tmod_int ((c:Int)+k+skew) 16).toNat=(c+k+skew)%16 := by
 change (((c:Int)+k+skew).tmod 16).toNat=(c+k+skew)%16
 have he:(c:Int)+k+skew=((c+k+skew:Nat):Int):=by omega
 rw [he,Int.tmod_eq_emod_of_nonneg (by omega)]
 rfl

theorem actual_run_element_source (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) :
 run_element q i c k off soff=
 (if (latency_of_instr i).d_skew=0 then execute_element (element q i c k off soff)
  else do
   let r ← Out.readReg Register.execute_ring
   let r' ← Out.readReg Register.execute_ring
   Out.writeReg Register.execute_ring
    (vectorUpdate r (tmod_int ((c:Int)+k+(latency_of_instr i).d_skew) 16).toNat
     (r'[(tmod_int ((c:Int)+k+(latency_of_instr i).d_skew) 16).toNat]!++[element q i c k off soff]))) := by
 unfold run_element element
 simp only [pure_bind,bind_pure,actual_append,beq_iff_eq]
 rfl

theorem actual_immediate_real_execute (q : Nat) (i : instruction) (c k : Nat) (off soff : Int)
 (h:(latency_of_instr i).d_skew=0) :
 run_element q i c k off soff=execute_element (element q i c k off soff) := by
 rw [actual_run_element_source,if_pos h]

theorem actual_delayed_enqueue (s : ResourceFrameProofs.St) (r : Vector (List exec_element) 16)
 (hr:s.regs.get? Register.execute_ring=some r)
 (q : Nat) (i : instruction) (c k : Nat) (off soff : Int)
 (h:(latency_of_instr i).d_skew≠0) :
 (run_element q i c k off soff).run s=.ok ()
  {s with regs:=(s.regs.insert Register.execute_ring
    (vectorUpdate r ((c+k+(latency_of_instr i).d_skew)%16)
     (r[(c+k+(latency_of_instr i).d_skew)%16]!++[element q i c k off soff])))} := by
 rw [actual_run_element_source,if_neg h]
 simp [Out.readReg,PreSail.readReg,Out.writeReg,PreSail.writeReg,
 EStateM.run,bind,EStateM.bind,pure,EStateM.pure,get,getThe,MonadStateOf.get,EStateM.get,
 modify,modifyGet,MonadStateOf.modifyGet,EStateM.modifyGet,hr,actual_mod16]

theorem actual_origin_fields (q : Nat) (i : instruction) (c k : Nat) (off soff : Int) :
 let e:=element q i c k off soff
 e.q=q ∧ e.instr=i ∧ e.c0=c ∧ e.k=k ∧ e.off=off ∧ e.soff=soff ∧
 target e=c+k+(latency_of_instr i).d_skew := by
 exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩

theorem delayed_horizon (q : Nat) (i : instruction) (c k now : Nat) (off soff : Int)
 (hissue:c+k=now) (h0:0<(latency_of_instr i).d_skew) (hb:(latency_of_instr i).d_skew≤15) :
 now<target (element q i c k off soff) ∧ target (element q i c k off soff)≤now+15 := by
 change now<c+k+(latency_of_instr i).d_skew ∧ c+k+(latency_of_instr i).d_skew≤now+15
 rw [←hissue]
 exact ⟨Nat.lt_add_of_pos_right h0,Nat.add_le_add_left hb (c+k)⟩

theorem same_bucket_unique16 (edge a b : Nat)
 (ha:edge≤a ∧ a<edge+16) (hb:edge ≤ b ∧ b<edge+16) (hmod:a%16=b%16) : a=b := by
 have h1:=Nat.mod_add_div a 16
 have h2:=Nat.mod_add_div b 16
 omega

theorem same_bucket_unique128 (edge a b : Nat)
 (ha:edge≤a ∧ a<edge+128) (hb:edge ≤ b ∧ b<edge+128) (hmod:a%128=b%128) : a=b := by
 have h1:=Nat.mod_add_div a 128
 have h2:=Nat.mod_add_div b 128
 omega

theorem pending_source_horizon (dispatch elem skew func now : Nat)
 (he:dispatch+elem+skew=now) (hpos:skew<func) (hmax:func≤63) :
 now<dispatch+elem+func ∧ dispatch+elem+func≤now+63 := by omega

theorem agt_reference_origin (t0 now : Nat) (h:t0≤now) :
 t0+(max ((now:Int)-t0) 0).toNat=now := by omega
end ExecuteProducerProofs
