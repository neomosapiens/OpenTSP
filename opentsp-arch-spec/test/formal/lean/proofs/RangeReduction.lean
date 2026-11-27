-- Actual production range-reduction arithmetic. These finite constant facts
-- do not assert the analytic ln(2) enclosure or global Ziv convergence.
import Proofs.Enclosure
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 1000000
set_option maxHeartbeats 20000000
namespace RangeReductionProofs

def SafeLn2 (p : Int) : Prop :=
  let S := pow2i p
  let v := ln2_iv p
  0 < S ∧ 0 < v.1 ∧ 2*S ≤ 3*v.1 ∧ 0 ≤ v.2-v.1 ∧
    v.1 + 2 + 2048*(v.2-v.1) ≤ S

theorem ln2_safe_64 : SafeLn2 64 := by
  unfold SafeLn2
  decide +kernel

theorem ln2_safe_128 : SafeLn2 128 := by
  unfold SafeLn2
  decide +kernel

theorem ln2_safe_256 : SafeLn2 256 := by
  unfold SafeLn2
  decide +kernel

theorem ln2_safe_512 : SafeLn2 512 := by
  unfold SafeLn2
  decide +kernel

theorem ln2_safe_1024 : SafeLn2 1024 := by
  unfold SafeLn2
  decide +kernel

theorem ln2_safe_2048 : SafeLn2 2048 := by
  unfold SafeLn2
  decide +kernel

theorem ln2_safe_4096 : SafeLn2 4096 := by
  unfold SafeLn2
  decide +kernel

/-- The actual signed midpoint is inside every integer interval, with distance
    at most its width from either endpoint. -/
theorem midpoint_bounds (a b : Int) (hab : a ≤ b) :
    a ≤ (a+b).tdiv 2 ∧ (a+b).tdiv 2 ≤ b := by
  have hr0 := Int.emod_nonneg (a+b) (by decide : (2:Int) ≠ 0)
  have hr1 := Int.emod_lt_of_pos (a+b) (by decide : (0:Int) < 2)
  have hd := Int.ediv_mul_add_emod (a+b) 2
  by_cases h : 0 ≤ a+b ∨ (2:Int) ∣ a+b
  · rw [Int.tdiv_eq_ediv]; simp only [h, ite_true]; omega
  · rw [Int.tdiv_eq_ediv]; simp only [h, ite_false]; change a ≤ (a+b)/2+1 ∧ (a+b)/2+1 ≤ b; omega

/-- The reduced-input expressions used by exp_iv, including its sign-dependent
    multiplication of the ln2 endpoints. -/
def reduction (x : ival) : Int × Int × Int :=
  let v := ln2_iv x.p
  let mid := (x.lo+x.hi).tdiv 2
  let n := idiv_floor (mid+v.1.tdiv 2) v.1
  (if n ≥ 0 then x.lo-n*v.2 else x.lo-n*v.1,
   if n ≥ 0 then x.hi-n*v.1 else x.hi-n*v.2, n)

theorem exp_iv_production_reduction (x : ival) :
    let r := reduction x
    exp_iv x = ((exp_small r.1 x.p).1, (exp_small r.2.1 x.p).2, r.2.2) := by
  simp [exp_iv, reduction]

/-- Integer range-reduction estimate, independent of any real-function identity.
    Its premises are explicit and are separately tied to actual ln2 constants. -/
theorem reduction_integer_bound (a b S L H : Int)
    (_hS : 0 < S) (hL : 0 < L) (hscale : 2*S ≤ 3*L)
    (hW : 0 ≤ H-L) (hsafe : L+2+2048*(H-L) ≤ S)
    (hab : a ≤ b) (hwidth : b-a ≤ 1)
    (ha : -512*S ≤ a) (hb : b ≤ 512*S) :
    let m := (a+b).tdiv 2
    let n := idiv_floor (m+L.tdiv 2) L
    let rlo := if n ≥ 0 then a-n*H else a-n*L
    let rhi := if n ≥ 0 then b-n*L else b-n*H
    2*(rlo.natAbs:Int) ≤ S ∧ 2*(rhi.natAbs:Int) ≤ S := by
  dsimp only
  let m := (a+b).tdiv 2
  let n := idiv_floor (m+L.tdiv 2) L
  let h := L.tdiv 2
  have hm := midpoint_bounds a b hab
  have hf := IntervalProofs.floor_bounds (m+h) L hL
  have hh : 0 ≤ 2*h ∧ 2*h ≤ L ∧ L ≤ 2*h+1 := by
    dsimp [h]
    rw [Int.tdiv_eq_ediv_of_nonneg (by omega : 0 ≤ L)]
    have hr0 := Int.emod_nonneg L (by decide : (2:Int) ≠ 0)
    have hr1 := Int.emod_lt_of_pos L (by decide : (0:Int) < 2)
    have hd := Int.ediv_mul_add_emod L 2
    have hd0 := Int.ediv_nonneg (by omega : 0 ≤ L) (by decide : (0:Int) ≤ 2)
    omega
  change n*L ≤ m+h ∧ m+h < (n+1)*L at hf
  have hLmax : L ≤ S := by omega
  have hn : -1024 ≤ n ∧ n ≤ 1024 := by
    constructor
    · by_cases hn : -1024 ≤ n
      · exact hn
      exfalso
      have hc : n+1 ≤ -1024 := by omega
      have hp := Int.mul_le_mul_of_nonneg_right hc (by omega : 0 ≤ L)
      have he : (n+1)*L = n*L+L := by grind
      dsimp [m] at hf
      omega
    · by_cases hn : n ≤ 1024
      · exact hn
      exfalso
      have hc : 1025 ≤ n := by omega
      have hp := Int.mul_le_mul_of_nonneg_right hc (by omega : 0 ≤ L)
      dsimp [m] at hf
      omega
  have hprodlo := Int.mul_le_mul_of_nonneg_right hn.1 hW
  have hprodhi := Int.mul_le_mul_of_nonneg_right hn.2 hW
  have hring : n*H = n*L+n*(H-L) := by grind
  have hmnear : a ≤ m ∧ m ≤ b := hm
  have hfloor : n*L ≤ m+h ∧ m+h < n*L+L := by grind
  have hlo : -S ≤ 2*(if n ≥ 0 then a-n*H else a-n*L) ∧
      2*(if n ≥ 0 then a-n*H else a-n*L) ≤ S := by
    split <;> omega
  have hhi : -S ≤ 2*(if n ≥ 0 then b-n*L else b-n*H) ∧
      2*(if n ≥ 0 then b-n*L else b-n*H) ≤ S := by
    split <;> omega
  have abs_bound (r : Int) (hr : -S ≤ 2*r ∧ 2*r ≤ S) : 2*(r.natAbs:Int) ≤ S := by
    by_cases hr0 : 0 ≤ r
    · rw [Int.natAbs_of_nonneg hr0]; omega
    · have he : (r.natAbs:Int) = -r := by omega
      rw [he]; omega
  exact ⟨abs_bound _ hlo, abs_bound _ hhi⟩

/-- Complete enumeration of the actual Ziv precision schedule, not a sample
    of inputs: each constant lemma is reduced by the Lean kernel. -/
theorem schedule_ln2_safe (p : Int)
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096) :
    SafeLn2 p := by
  rcases hp with h|h|h|h|h|h|h
  all_goals subst p
  · exact ln2_safe_64
  · exact ln2_safe_128
  · exact ln2_safe_256
  · exact ln2_safe_512
  · exact ln2_safe_1024
  · exact ln2_safe_2048
  · exact ln2_safe_4096

theorem exp_iv_reduced_half_domain (x : ival)
    (hp : x.p=64 ∨ x.p=128 ∨ x.p=256 ∨ x.p=512 ∨ x.p=1024 ∨ x.p=2048 ∨ x.p=4096)
    (hab : x.lo ≤ x.hi) (hw : x.hi-x.lo ≤ 1)
    (ha : -512*pow2i x.p ≤ x.lo) (hb : x.hi ≤ 512*pow2i x.p) :
    2*((reduction x).1.natAbs:Int) ≤ pow2i x.p ∧
    2*((reduction x).2.1.natAbs:Int) ≤ pow2i x.p := by
  have hs := schedule_ln2_safe x.p hp
  dsimp [SafeLn2] at hs
  exact reduction_integer_bound x.lo x.hi (pow2i x.p)
    (ln2_iv x.p).1 (ln2_iv x.p).2 hs.1 hs.2.1 hs.2.2.1
    hs.2.2.2.1 hs.2.2.2.2 hab hw ha hb

/-- Signed floor/ceil conversion loses no more than one scaled integer unit. -/
theorem floor_ceil_width (a b : Int) (hb : 0 < b) :
    idiv_floor a b ≤ idiv_ceil a b ∧ idiv_ceil a b - idiv_floor a b ≤ 1 := by
  rw [IntervalProofs.idiv_floor_correct a b hb]
  simp only [idiv_ceil, Int.zero_sub, IntervalProofs.idiv_floor_correct (-a) b hb, Int.neg_ediv]
  rw [Int.sign_eq_one_of_pos hb]
  split <;> omega

theorem fx_of_width (neg : Bool) (mag e p : Int) :
    (fx_of neg mag e p).lo ≤ (fx_of neg mag e p).hi ∧
    (fx_of neg mag e p).hi-(fx_of neg mag e p).lo ≤ 1 := by
  by_cases hs : 0 ≤ e+p
  · cases neg <;> simp [fx_of, hs]
  · have hd := IntervalProofs.pow2i_positive (-(e+p)) (by omega)
    have hw := floor_ceil_width (if neg then -mag else mag) (pow2i (-(e+p))) hd
    cases neg <;> simpa [fx_of, hs] using hw

theorem floor_ceil_preserve_bounds (a d B : Int) (hd : 0 < d)
    (ha : -B*d ≤ a) (hb : a ≤ B*d) :
    -B ≤ idiv_floor a d ∧ idiv_ceil a d ≤ B := by
  have hf := IntervalProofs.floor_bounds a d hd
  have hc := IntervalProofs.ceil_bounds a d hd
  constructor
  · by_cases h : -B ≤ idiv_floor a d
    · exact h
    have hp := Int.mul_le_mul_of_nonneg_right (show idiv_floor a d+1 ≤ -B by omega) (by omega : 0 ≤ d)
    omega
  · by_cases h : idiv_ceil a d ≤ B
    · exact h
    have hp := Int.mul_le_mul_of_nonneg_right (show B ≤ idiv_ceil a d-1 by omega) (by omega : 0 ≤ d)
    omega

/-- Transfer a magnitude/binade bound to actual fixed-point endpoints at any
    nonnegative working precision, including signed fractional conversion. -/
theorem fx_of_scaled_bound (neg : Bool) (mag e p : Int)
    (hp : 0 ≤ p) (hm0 : 0 ≤ mag) (he : e ≤ 8)
    (hm : mag ≤ pow2i (9-e)) :
    -512*pow2i p ≤ (fx_of neg mag e p).lo ∧
    (fx_of neg mag e p).hi ≤ 512*pow2i p := by
  have hp0 := IntervalProofs.pow2i_positive p hp
  have he0 := IntervalProofs.pow2i_positive (9-e) (by omega)
  have h9 : pow2i 9 = 512 := by decide
  have hp9 : pow2i (9+p) = 512*pow2i p := by
    rw [NumericalProofs.pow2i_add_nonnegative 9 p (by decide) hp, h9]
  by_cases hs : 0 ≤ e+p
  · have hpow := NumericalProofs.pow2i_add_nonnegative (9-e) (e+p) (by omega) hs
    have heq : 9-e+(e+p) = 9+p := by omega
    rw [heq, hp9] at hpow
    have hsp := IntervalProofs.pow2i_positive (e+p) hs
    have hmul := Int.mul_le_mul_of_nonneg_right hm (by omega : 0 ≤ pow2i (e+p))
    have hmpos := Int.mul_nonneg hm0 (by omega : 0 ≤ pow2i (e+p))
    rw [←hpow] at hmul
    cases neg <;> simp [fx_of, hs] <;> omega
  · have hd := IntervalProofs.pow2i_positive (-(e+p)) (by omega)
    have hpow := NumericalProofs.pow2i_add_nonnegative (9+p) (-(e+p)) (by omega) (by omega)
    have heq : 9+p+ -(e+p) = 9-e := by omega
    rw [heq, hp9] at hpow
    have hmB : mag ≤ (512*pow2i p)*pow2i (-(e+p)) := by omega
    have hnum : -(512*pow2i p)*pow2i (-(e+p)) ≤ (if neg then -mag else mag) ∧
        (if neg then -mag else mag) ≤ (512*pow2i p)*pow2i (-(e+p)) := by
      cases neg <;> dsimp <;> rw [Int.neg_mul] <;> omega
    have hb := floor_ceil_preserve_bounds (if neg then -mag else mag)
      (pow2i (-(e+p))) (512*pow2i p) hd hnum.1 hnum.2
    cases neg <;> simpa [fx_of, hs, Int.neg_mul] using hb

/-- End-to-end actual fx_of -> exp_iv half-domain transfer. The remaining
    decoded-input premises are the exact contracts checked by the domains SMT
    suite on every input carrier for each admitted format and both callers. -/
theorem decoded_exp_caller_half_domain (neg : Bool) (mag e p : Int)
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096)
    (hm0 : 0 ≤ mag) (he : e ≤ 8) (hm : mag ≤ pow2i (9-e)) :
    2*((reduction (fx_of neg mag e p)).1.natAbs:Int) ≤ pow2i p ∧
    2*((reduction (fx_of neg mag e p)).2.1.natAbs:Int) ≤ pow2i p := by
  have hp0 : 0 ≤ p := by omega
  have hwidth := fx_of_width neg mag e p
  have hbound := fx_of_scaled_bound neg mag e p hp0 hm0 he hm
  have hprecision : (fx_of neg mag e p).p = p := by
    simp [fx_of]; split <;> rfl
  have hc := exp_iv_reduced_half_domain (fx_of neg mag e p)
    (by simpa [hprecision] using hp) hwidth.1 hwidth.2
    (by simpa [hprecision] using hbound.1) (by simpa [hprecision] using hbound.2)
  simpa [hprecision] using hc

/-- Source-bound dispatch: every finite public Exp evaluation which reaches
    tr_eval satisfies exactly the cutoff used by the SMT caller contracts. -/
theorem exp_public_dispatch (f : fmt) (x : BitVec 32)
    (hc : (fdecode f x).cls = fclass.FFinite) :
    fp_tr_value 22 f x =
      if (fdecode f x).e + ilog2 (fdecode f x).mag > 8 then
        Option.some (if (fdecode f x).neg then zero_bits f false else inf_bits f false)
      else tr_eval 22 f (fdecode f x) := by
  simp [fp_tr_value, hc]

theorem tanh_public_dispatch (f : fmt) (x : BitVec 32)
    (hc : (fdecode f x).cls = fclass.FFinite) :
    fp_tr_value 21 f x =
      if (fdecode f x).e + ilog2 (fdecode f x).mag > 5 then
        Option.some (fp_of_int f (if (fdecode f x).neg then -1 else 1))
      else tr_eval 21 f (fdecode f x) := by
  simp [fp_tr_value, hc]

/-- The post-processing of the actual TanH interval, separated solely to expose
    the precise exp_iv argument to the caller-domain composition. -/
def tanhPost (neg : Bool) (p : Int) (v : Int × Int × Int) : Int × Int × Int :=
  let (lo,hi,n) := v
  let elo := if n ≥ 0 then lo*pow2i n else idiv_floor lo (pow2i (-n))
  let ehi := if n ≥ 0 then hi*pow2i n else idiv_ceil hi (pow2i (-n))
  let one := pow2i p
  let big := pow2i (2*p)
  let tlo := big - idiv_ceil (2*big*one) (elo+one)
  let thi := big - idiv_floor (2*big*one) (ehi+one)
  let lo2 := if neg then -thi else tlo
  let hi2 := if neg then -tlo else thi
  (lo2-1, hi2+1, -2*p)

theorem tr_interval_exp_argument (a : fval) (p : Int) :
    tr_interval 22 a p =
      let (lo,hi,n) := exp_iv (fx_of a.neg a.mag a.e p)
      (lo,hi,n-p) := by rfl

theorem tr_interval_tanh_argument (a : fval) (p : Int) :
    tr_interval 21 a p =
      tanhPost a.neg p (exp_iv (fx_of false a.mag (a.e+1) p)) := by
  simp [tr_interval, tanhPost, Int.neg_mul]

/-- All public i8 carriers decode to the ISA signed range; this does not
    restrict arbitrary direct calls to the internal integer helper. -/
theorem s_of_bits_i8_range (x : BitVec 32) :
    -128 ≤ s_of_bits x 8 ∧ s_of_bits x 8 ≤ 127 := by
  have h256 : pow2i 8 = 256 := by decide
  have h128 : pow2i (8-1) = 128 := by decide
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have ht : (BitVec.toNatInt x).tmod 256 = (BitVec.toNatInt x)%256 :=
    Int.tmod_eq_emod_of_nonneg hx
  have hm0 := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (256:Int) ≠ 0)
  have hm1 := Int.emod_lt_of_pos (BitVec.toNatInt x) (by decide : (0:Int) < 256)
  simp only [s_of_bits, h256, h128, ht]
  split <;> simp_all <;> omega

/-- Exact source binding of the integer helper input and all its shortcut
    cases. The range theorem below is required only at the public ISA caller. -/
def i8Input (v : Int) : fval :=
  { cls := fclass.FFinite, neg := decide (v<0), mag := if v<0 then -v else v, e := 0 }

theorem i8_public_dispatch (op v : Int) :
    i8_tr_value op v =
      if v=0 then some (if op=22 then 1 else if op=23 then 127 else if op=24 then -128 else 0)
      else if (op=23 ∨ op=24) ∧ v<0 then some 0
      else if op=22 ∧ v>5 then some 127
      else if op=22 ∧ v < -1 then some 0
      else if op=23 ∧ v=4 then some 0
      else tr_eval_int op (i8Input v) := by
  simp [i8_tr_value, i8Input]

theorem i8_caller_half_domain (v p : Int) (doubled : Bool)
    (hv : -128 ≤ v ∧ v ≤ 127)
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096) :
    let a := i8Input v
    let e := a.e + if doubled then 1 else 0
    2*((reduction (fx_of (if doubled then false else a.neg) a.mag e p)).1.natAbs:Int) ≤ pow2i p ∧
    2*((reduction (fx_of (if doubled then false else a.neg) a.mag e p)).2.1.natAbs:Int) ≤ pow2i p := by
  dsimp only
  apply decoded_exp_caller_half_domain _ _ _ _ hp
  · dsimp [i8Input]; split <;> omega
  · cases doubled <;> simp [i8Input]
  · have hm : (i8Input v).mag ≤ 128 := by dsimp [i8Input]; split <;> omega
    cases doubled
    · change (i8Input v).mag ≤ 512; omega
    · change (i8Input v).mag ≤ 256; omega

end RangeReductionProofs
