-- Spare reports preserve every ECC/check bit in every superlane.
import Proofs.SrfCommittedHistory
import Proofs.SrfBitPrimitives
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace SrfPoisonFlowProofs
open SrfBitPrimitiveProofs SrfMergeProofs ResourceScanProofs

theorem actual_set_bit (ov:ecc_ov) (g j:Nat) (e:BitVec 10)
 (hg:g<Out.Functions.superlanes) (hj:j<10*Out.Functions.superlanes):
 (ecc_ov_set ov g e).getLsbD j=
 (if 10*g≤j ∧ j<10*g+10 then e.getLsbD (j-10*g) else ov.getLsbD j):=by
 change ((ov &&& ~~~((BitVec.ofNat (10*Out.Functions.superlanes) 1023) <<< (10*g))) |||
  ((e.setWidth (10*Out.Functions.superlanes)) <<< (10*g))).getLsbD j=_
 erw [BitVec.getLsbD_or,BitVec.getLsbD_and,BitVec.getLsbD_not]
 erw [BitVec.getLsbD_shiftLeft,BitVec.getLsbD_shiftLeft]
 erw [BitVec.getLsbD_setWidth]
 have hmask:(BitVec.ofNat (10*Out.Functions.superlanes) 1023)=
  (BitVec.allOnes 10).setWidth (10*Out.Functions.superlanes):=by rfl
 erw [hmask,BitVec.getLsbD_setWidth,BitVec.getLsbD_allOnes]
 have hidx:j-10*g<10*Out.Functions.superlanes:=by omega
 simp only [Out.Functions.superlanes] at hg hj hidx ⊢
 by_cases hlo:j<10*g
 · have hnon:¬10*g≤j:=by omega
   simp [hlo,hnon,hj]
 · have hnon:10*g≤j:=by omega
   have hnotlo:¬j<10*g:=by omega
   by_cases hlen:j-10*g<10
   · have hb:j<10*g+10:=by omega
     repeat' erw [show decide (j<10*g)=false from decide_eq_false (by omega : ¬j<10*g)]
     simp_all
   · have hb:¬j<10*g+10:=by omega
     repeat' erw [show decide (j<10*g)=false from decide_eq_false (by omega : ¬j<10*g)]
     repeat' erw [show decide (j-10*g<10)=false from decide_eq_false (by omega : ¬j-10*g<10)]
     repeat' erw [show decide (j<10*g+10)=false from decide_eq_false (by omega : ¬j<10*g+10)]
     simp_all [Bool.and_comm]


def spareStep (m:BitVec Out.Functions.superlanes) (x:Int) (ov:ecc_ov):ecc_ov:=
 if (Sail.BitVec.access m x)==1#1 then ecc_ov_set ov x (ecc_ov_get ov x ||| 512#10) else ov

theorem actual_spare_source (ov:ecc_ov) (m:BitVec Out.Functions.superlanes):
 srf_ov_spare ov m=
 if m != (BitVec.zero Out.Functions.superlanes) then
  IntRange.forIn' (m:=Id) {start:=0,stop:=(Out.Functions.superlanes:Int)-1,step:=1} ov
   (fun x _ state=>pure (.yield (spareStep m x state)))
 else ov:=by rfl

theorem actual_get_bit (ov:ecc_ov) (g b:Nat) (hb:b<10):
 (ecc_ov_get ov g).getLsbD b=ov.getLsbD (10*g+b):=by
 change (ov.extractLsb' (10*g) 10).getLsbD b=_
 simp [BitVec.getLsbD_extractLsb',hb,Nat.add_comm]

theorem spare_pattern_low (b:Nat) (hb:b<9):(512#10).getLsbD b=false:=by
 have he:b=0∨b=1∨b=2∨b=3∨b=4∨b=5∨b=6∨b=7∨b=8:=by omega
 rcases he with h|h|h|h|h|h|h|h|h
 all_goals subst b;decide +kernel

theorem spare_step_check_bits (m:BitVec Out.Functions.superlanes) (x:Int) (ov:ecc_ov)
 (lane b:Nat) (hx:0≤x) (hxn:x<Out.Functions.superlanes)
 (hl:lane<Out.Functions.superlanes) (hb:b<9):
 (spareStep m x ov).getLsbD (10*lane+b)=ov.getLsbD (10*lane+b):=by
 unfold spareStep
 split
 · have hi:x.toNat<Out.Functions.superlanes:=by omega
   have hj:10*lane+b<10*Out.Functions.superlanes:=by omega
   have he:(x.toNat:Int)=x:=by omega
   have ht:=actual_set_bit ov x.toNat (10*lane+b) (ecc_ov_get ov x ||| 512#10) hi hj
   simp only [he] at ht
   erw [ht]
   by_cases same:x.toNat=lane
   · have hei:x=(lane:Int):=by omega
     have hlo:10*x.toNat≤10*lane+b:=by omega
     have hhi:10*lane+b<10*x.toNat+10:=by omega
     have hd:10*lane+b-10*x.toNat=b:=by omega
     simp only [hlo,hhi,and_self,if_true,hd]
     erw [BitVec.getLsbD_or]
     rw [hei,actual_get_bit _ _ _ (by omega),spare_pattern_low b hb]
     simp
   · have hnot:¬(10*x.toNat≤10*lane+b ∧ 10*lane+b<10*x.toNat+10):=by omega
     rw [if_neg hnot]
 · rfl

theorem spare_fold_check_bits (m:BitVec Out.Functions.superlanes)
 (start:Int) (n:Nat) (ov:ecc_ov) (lane b:Nat)
 (hstart:0≤start) (hend:start+n≤Out.Functions.superlanes)
 (hl:lane<Out.Functions.superlanes) (hb:b<9):
 (indexedFold (spareStep m) start n ov).getLsbD (10*lane+b)=ov.getLsbD (10*lane+b):=by
 induction n generalizing start ov with
 | zero=>rfl
 | succ n ih=>
   rw [indexedFold]
   rw [ih (start+1) (spareStep m start ov) (by omega) (by omega)]
   exact spare_step_check_bits m start ov lane b hstart (by omega) hl hb

theorem actual_spare_preserves_check_bits (ov:ecc_ov) (m:BitVec Out.Functions.superlanes)
 (lane b:Nat) (hl:lane<Out.Functions.superlanes) (hb:b<9):
 (srf_ov_spare ov m).getLsbD (10*lane+b)=ov.getLsbD (10*lane+b):=by
 rw [actual_spare_source]
 split
 · rw [actual_indexed_range_fold _ _ _ rfl]
   exact spare_fold_check_bits m 0 _ ov lane b (by decide) (by simp) hl hb
 · rfl

end SrfPoisonFlowProofs
