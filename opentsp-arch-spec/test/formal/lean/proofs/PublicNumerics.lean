-- Complete public special/cutoff/cast policies and actual ISA dispatch bridges.
import Proofs.ReduceSemantics
open Sail Out Out.Functions Out.fclass
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
set_option exponentiation.threshold 100000
namespace PublicNumericProofs

def fpPublicSpec (op : Int) (f : fmt) (a : fval) : Option (BitVec 32) :=
  match a.cls with
  | FNaN => some (canon_nan f)
  | FZero => some (if op=21 then zero_bits f a.neg else if op=22 then fp_of_int f 1
      else if op=23 then inf_bits f a.neg else inf_bits f true)
  | FInf => some (if op=21 then fp_of_int f (if a.neg then -1 else 1)
      else if op=22 then (if a.neg then zero_bits f false else inf_bits f false)
      else if op=23 then (if a.neg then canon_nan f else zero_bits f false)
      else if a.neg then canon_nan f else inf_bits f false)
  | FFinite =>
    if (op=23 ∨ op=24) ∧ a.neg=true then some (canon_nan f)
    else if op=24 ∧ a.mag=pow2i (-a.e) ∧ a.e≤0 then some (zero_bits f false)
    else if op=22 ∧ 8<a.e+ ilog2 a.mag then some (if a.neg then zero_bits f false else inf_bits f false)
    else if op=21 ∧ 5<a.e+ ilog2 a.mag then some (fp_of_int f (if a.neg then -1 else 1))
    else tr_eval op f a

theorem complete_fp_public_policy (op : Int) (f : fmt) (x : BitVec 32) :
    fp_tr_value op f x=fpPublicSpec op f (fdecode f x) := by
  cases hc : (fdecode f x).cls <;> cases hn : (fdecode f x).neg <;>
    simp [fp_tr_value,fpPublicSpec,hc,hn,and_assoc]

def i8PublicSpec (op v : Int) : Option Int :=
  if v=0 then some (if op=22 then 1 else if op=23 then 127 else if op=24 then -128 else 0)
  else if (op=23 ∨ op=24) ∧ v<0 then some 0
  else if op=22 ∧ 5<v then some 127
  else if op=22 ∧ v < -1 then some 0
  else if op=23 ∧ v=4 then some 0
  else tr_eval_int op {cls:=FFinite,neg:=decide (v<0),mag:=if v<0 then -v else v,e:=0}

theorem complete_i8_public_policy (op v : Int) : i8_tr_value op v=i8PublicSpec op v := by
  simp [i8_tr_value,i8PublicSpec,and_assoc]

theorem exp_public_cutoff (f : fmt) (x : BitVec 32) (hc : (fdecode f x).cls=FFinite)
    (hcut : 8 < (fdecode f x).e+ ilog2 (fdecode f x).mag) :
    fp_tr_value 22 f x=some (if (fdecode f x).neg then zero_bits f false else inf_bits f false) := by
  rw [complete_fp_public_policy]
  simp [fpPublicSpec,hc,hcut]

theorem tanh_public_cutoff (f : fmt) (x : BitVec 32) (hc : (fdecode f x).cls=FFinite)
    (hcut : 5 < (fdecode f x).e+ ilog2 (fdecode f x).mag) :
    fp_tr_value 21 f x=some (fp_of_int f (if (fdecode f x).neg then -1 else 1)) := by
  rw [complete_fp_public_policy]
  simp [fpPublicSpec,hc,hcut]

/-- Actual cutoff integer test implies the exact dyadic magnitude lower bound
    at every common unit; no approximate host log/exp filter is used. -/
theorem cutoff_dyadic_lower (mag e cutoff base : Int) (hm : 1 ≤ mag)
    (hcap : mag < pow2i 16384) (hc : cutoff ≤ ilog2 mag+e)
    (he : 0 ≤ e+base) (hcb : 0 ≤ cutoff+base) :
    pow2i (cutoff+base) ≤ mag*pow2i (e+base) := by
  have hp := BinadeProofs.ilog2_binade mag hm hcap
  have hs := IntervalProofs.pow2i_positive (e+base) he
  have hi := Int.mul_le_mul_of_nonneg_right hp.2.1 (by omega : 0 ≤ pow2i (e+base))
  have hpow := NumericalProofs.pow2i_add_nonnegative (ilog2 mag) (e+base) hp.1 he
  have hmono := GridBoundProofs.pow2_mono (cutoff+base) (ilog2 mag+(e+base)) hcb (by omega)
  rw [hpow] at hmono
  exact Int.le_trans hmono hi

theorem i8_zero_policy (op : Int) :
    i8_tr_value op 0=some (if op=22 then 1 else if op=23 then 127 else if op=24 then -128 else 0) := by
  simp [i8_tr_value]

theorem i8_exp_high (v : Int) (h : 5 < v) : i8_tr_value 22 v=some 127 := by
  simp [i8_tr_value,show v≠0 by omega,h]

theorem i8_exp_low (v : Int) (h : v < -1) : i8_tr_value 22 v=some 0 := by
  simp [i8_tr_value,show v≠0 by omega,show ¬v>5 by omega,h]

theorem i8_negative_domain (op v : Int) (ho : op=23 ∨ op=24) (hv : v<0) :
    i8_tr_value op v=some 0 := by
  rcases ho with h|h <;> subst op <;> simp [i8_tr_value,show v≠0 by omega,hv]

theorem rsqrt_only_integer_half_tie (v n : Int) (hv : 1 ≤ v) (hn : 0 ≤ n)
    (he : v*((2*n+1)*(2*n+1))=4) : n=0 ∧ v=4 := by
  by_cases hz : n=0
  · subst n; simp only [Int.mul_zero,Int.zero_add,Int.one_mul] at he; omega
  · have hh : 3 ≤ 2*n+1 := by omega
    have hs1 := Int.mul_le_mul_of_nonneg_right hh (by decide : (0:Int) ≤ 3)
    have hs2 := Int.mul_le_mul_of_nonneg_left hh (show 0 ≤ 2*n+1 by omega)
    have hsquare : 9 ≤ (2*n+1)*(2*n+1) := Int.le_trans hs1 hs2
    have hvprod := Int.mul_le_mul_of_nonneg_right hv (show 0 ≤ (2*n+1)*(2*n+1) by omega)
    omega

theorem i8_rsqrt_tie_policy : i8_tr_value 23 4=some 0 := by decide +kernel

theorem cutoff_exact_rational_budgets :
    (8:Nat)^6>127*3^6 ∧ (2:Nat)^512>2^150 ∧ (2:Nat)^128>2^26 := by decide +kernel

def applySpec (op : vxm_lane_op) (a b m : BitVec 32) : SailM (BitVec 32) :=
  match op with
  | .L_BIN (o,t) => pure (k_binary o t a b)
  | .L_UN (o,t) => if 21≤o ∧ o≤24 then
      (if is_float t then fp_transcendental o (fmt_of t) a else i8_transcendental o a)
      else pure (k_unary o t a)
  | .L_MASK t => pure (if k_nonzero t b then a else 0)
  | .L_SELECT t => pure (if k_nonzero t m then a else b)
  | .L_CMP (o,t) => pure (k_cmp o t a b)
  | .L_CONV o => pure (k_convert o a)
  | .L_CVT (st,dt) => pure (k_convert_any st dt a)

theorem complete_apply_dispatch (op : vxm_lane_op) (a b m : BitVec 32) :
    Out.Functions.apply op a b m=applySpec op a b m := by
  cases op <;> simp [Out.Functions.apply,applySpec]
  all_goals split <;> rfl

theorem actual_float_trans_returns (op : Int) (t : eltype) (a b m r : BitVec 32)
    (ho : 21≤op ∧ op≤24) (ht : is_float t=true)
    (hr : fp_tr_value op (fmt_of t) a=some r) :
    Out.Functions.apply (.L_UN (op,t)) a b m=pure r := by
  rw [complete_apply_dispatch]
  simp [applySpec,ho.1,ho.2,ht,SignedRoundingProofs.fp_transcendental_success op (fmt_of t) a r hr]

theorem actual_i8_trans_returns (op : Int) (t : eltype) (a b m : BitVec 32) (r : Int)
    (ho : 21≤op ∧ op≤24) (ht : is_float t=false)
    (hr : i8_tr_value op (s_of_bits a 8)=some r) :
    Out.Functions.apply (.L_UN (op,t)) a b m=pure (bits_of_s (sat_s r 8) 8) := by
  rw [complete_apply_dispatch]
  simp [applySpec,ho.1,ho.2,ht,SignedRoundingProofs.i8_transcendental_success op a r hr]

theorem actual_generic_convert_dispatch (st dt : eltype) (d : streamref) (a : vsrc) :
    execute_VXM_CVT st dt d a=
      vxm_lanes (.L_CVT (st,dt)) d a none none (eltype_width st) (eltype_width dt) 0#1 := by rfl

theorem actual_legacy_convert_dispatch (c : vxm_conv) (t : eltype) (d : streamref) (a : vsrc) :
    execute_VXM_CONV c t d a=
      let (st,dt) := conv_types c
      vxm_lanes (.L_CONV (BitVec.toNatInt (encdec_vxm_conv_forwards c))) d a none none
        (eltype_width st) (eltype_width dt) 0#1 := by rfl

theorem actual_binary_dispatch (op : vxm_binop) (t : eltype) (d : streamref) (a b : vsrc) (g : BitVec 1) :
    execute_VXM_BIN op t d a b g = vxm_lanes (.L_BIN (BitVec.toNatInt (encdec_vxm_binop_forwards op),t))
      d a (some b) none (eltype_width t) (eltype_width t) g := by rfl

theorem actual_unary_dispatch (op : vxm_unop) (t : eltype) (d : streamref) (a : vsrc) (g : BitVec 1) :
    execute_VXM_UN op t d a g = vxm_lanes (.L_UN (BitVec.toNatInt (encdec_vxm_unop_forwards op),t))
      d a none none (eltype_width t) (eltype_width t) g := by rfl

theorem actual_comparison_dispatch (mode : cmp_mode) (t : eltype) (d : streamref) (a b : vsrc) (g : BitVec 1) :
    execute_VXM_CMP mode t d a b g = vxm_lanes (.L_CMP (BitVec.toNatInt (encdec_cmp_mode_forwards mode),t))
      d a (some b) none (eltype_width t) (eltype_width t) g := by rfl

def fpConvertSpec (dst : fmt) (a : fval) : BitVec 32 :=
  match a.cls with
  | FNaN => canon_nan dst
  | FInf => inf_bits dst a.neg
  | FZero => zero_bits dst a.neg
  | FFinite => fround dst a.neg a.mag a.e

theorem complete_float_conversion_policy (src dst : fmt) (x : BitVec 32) :
    fp_convert src dst x=fpConvertSpec dst (fdecode src x) := by
  cases hc : (fdecode src x).cls <;> simp [fp_convert,fpConvertSpec,hc]

theorem actual_legacy_conversion_domains (c : vxm_conv) :
    let op := BitVec.toNatInt (encdec_vxm_conv_forwards c)
    25 ≤ op ∧ op ≤ 32 ∧ (conv_types c).1=conv_src op ∧ (conv_types c).2=conv_dst op := by
  cases c <;> simp [encdec_vxm_conv_forwards,conv_types,conv_src,conv_dst,BitVec.toNatInt]

theorem f16_dot_complete (wlo whi xlo xhi : vec) :
    f16_dot wlo whi xlo xhi=DotSemanticProofs.dotResultSpec
      (DotSemanticProofs.packDot (DotSemanticProofs.indexedFold
        (DotSemanticProofs.pairSemanticStep FMT16 48 wlo whi xlo xhi) 0 Out.Functions.vlanes.toNat
        (true,false,false,false,0))) 48 := by
  simp only [f16_dot,f16_exact,dsum_round,DotSemanticProofs.pair_exact_complete,
    DotSemanticProofs.complete_dot_final_class_table]

theorem bf16_dot_complete (wlo whi xlo xhi : vec) :
    bf16_dot wlo whi xlo xhi=DotSemanticProofs.dotResultSpec
      (DotSemanticProofs.packDot (DotSemanticProofs.indexedFold
        (DotSemanticProofs.pairSemanticStep FMT_BF16 266 wlo whi xlo xhi) 0 Out.Functions.vlanes.toNat
        (true,false,false,false,0))) 266 := by
  simp only [bf16_dot,DotSemanticProofs.pair_exact_complete,DotSemanticProofs.complete_dot_final_class_table]

theorem fp8_dot_complete (f : fmt) (wr xv : vec) :
    fp8_dot f wr xv=DotSemanticProofs.dotResultSpec
      (DotSemanticProofs.packDot (DotSemanticProofs.indexedFold
        (DotSemanticProofs.fp8SemanticStep f wr xv) 0 Out.Functions.vlanes.toNat (true,false,false,false,0))) 48 := by
  simp only [fp8_dot,dsum_round,DotSemanticProofs.fp8_exact_complete,DotSemanticProofs.complete_dot_final_class_table]

end PublicNumericProofs
