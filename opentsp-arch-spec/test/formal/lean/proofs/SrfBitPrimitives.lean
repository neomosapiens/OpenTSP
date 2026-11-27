-- Bit meaning of the actual native one-bit update, for every width and valid bit.
import Proofs.SrfMerge
open Sail Out Out.Functions
set_option autoImplicit false
namespace SrfBitPrimitiveProofs

theorem bit_outside {N : Nat} (x : BitVec N) (j : Nat) (hj:N≤j) : x.getLsbD j=false := by
 cases h:x.getLsbD j with
 | false => rfl
 | true => have hn:=BitVec.lt_of_getLsbD h; omega

theorem actual_update_bit {N : Nat} (x : BitVec N) (k j : Nat) (b : BitVec 1) (hj:j<N) :
 (Sail.BitVec.update x k b).getLsbD j=(if j=k then b.getLsbD 0 else x.getLsbD j) := by
 by_cases same:j=k
 · subst j
   simp [Sail.BitVec.update,Sail.BitVec.updateSubrange',BitVec.zeroExtend_eq_setWidth,hj]
 by_cases below:j<k
 · simp [Sail.BitVec.update,Sail.BitVec.updateSubrange',BitVec.zeroExtend_eq_setWidth,hj,below,same]
 · have hk:k<j:=by omega
   have hsub:0<j-k:=by omega
   have hwide:j-k<N:=by omega
   have hb:b.getLsbD (j-k)=false:=bit_outside b _ (by omega)
   simp [Sail.BitVec.update,Sail.BitVec.updateSubrange',BitVec.zeroExtend_eq_setWidth,hj,show ¬j<k from below,same,hwide,show ¬j-k<1 from by omega,hb]
   omega

theorem access_one {N : Nat} (x : BitVec N) (k : Nat) (hk:k<N) :
 ((Sail.BitVec.access x k)==1#1)=true ↔ x.getLsbD k=true := by
 simp only [Sail.BitVec.access]
 simp only [getElem!_pos x k hk,←BitVec.getLsbD_eq_getElem hk]
 cases h:x.getLsbD k <;> decide
end SrfBitPrimitiveProofs
