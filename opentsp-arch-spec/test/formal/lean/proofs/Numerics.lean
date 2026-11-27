-- Properties of the actual Sail-generated arbitrary-precision numerical code.
-- These are decomposition/exception/rounding-decision obligations, not a claim
-- that the separate transcendental enclosure or Ziv cap obligations are proved.
import Out.Fp
import Out.Transcendental

open Sail Out Out.Functions Out.fclass

set_option autoImplicit false

namespace NumericalProofs

@[simp] theorem cls_FZero_FZero : (FZero == FZero) = true := by rfl
@[simp] theorem cls_FZero_FFinite : (FZero == FFinite) = false := by rfl
@[simp] theorem cls_FZero_FInf : (FZero == FInf) = false := by rfl
@[simp] theorem cls_FZero_FNaN : (FZero == FNaN) = false := by rfl
@[simp] theorem cls_FFinite_FZero : (FFinite == FZero) = false := by rfl
@[simp] theorem cls_FFinite_FFinite : (FFinite == FFinite) = true := by rfl
@[simp] theorem cls_FFinite_FInf : (FFinite == FInf) = false := by rfl
@[simp] theorem cls_FFinite_FNaN : (FFinite == FNaN) = false := by rfl
@[simp] theorem cls_FInf_FZero : (FInf == FZero) = false := by rfl
@[simp] theorem cls_FInf_FFinite : (FInf == FFinite) = false := by rfl
@[simp] theorem cls_FInf_FInf : (FInf == FInf) = true := by rfl
@[simp] theorem cls_FInf_FNaN : (FInf == FNaN) = false := by rfl
@[simp] theorem cls_FNaN_FZero : (FNaN == FZero) = false := by rfl
@[simp] theorem cls_FNaN_FFinite : (FNaN == FFinite) = false := by rfl
@[simp] theorem cls_FNaN_FInf : (FNaN == FInf) = false := by rfl
@[simp] theorem cls_FNaN_FNaN : (FNaN == FNaN) = true := by rfl

def signedMag (neg : Bool) (mag : Int) : Int := if neg then -mag else mag

def alignedSum (a b : fval) (subtract : Bool) : Int :=
  let e := min a.e b.e
  signedMag a.neg (a.mag * pow2i (a.e - e)) +
    signedMag (if subtract then !b.neg else b.neg) (b.mag * pow2i (b.e - e))

def roundSigned (f : fmt) (s e : Int) : BitVec 32 :=
  if s = 0 then zero_bits f false
  else if s < 0 then fround f true (-s) e else fround f false s e

/-- For any format, finite addition/subtraction forms the exact signed aligned
    integer once, then calls the production rounding core once. -/
theorem add_finite_decomposition (f : fmt) (a b : fval) (subtract : Bool)
    (ha : a.cls = FFinite) (hb : b.cls = FFinite) :
    fadd_core f a b subtract = roundSigned f (alignedSum a b subtract) (min a.e b.e) := by
  have hm : (if a.e < b.e then a.e else b.e) = min a.e b.e := by omega
  simp [fadd_core, ha, hb, alignedSum, roundSigned, signedMag, hm]

/-- The public operation reaches the above theorem through the actual decoder,
    rather than a separately reimplemented arithmetic function. -/
theorem fp_add_finite_decomposition (f : fmt) (x y : BitVec 32)
    (hx : (fdecode f x).cls = FFinite) (hy : (fdecode f y).cls = FFinite) :
    fp_add f x y = roundSigned f (alignedSum (fdecode f x) (fdecode f y) false)
      (min (fdecode f x).e (fdecode f y).e) := by
  exact add_finite_decomposition f _ _ false hx hy

theorem fp_sub_finite_decomposition (f : fmt) (x y : BitVec 32)
    (hx : (fdecode f x).cls = FFinite) (hy : (fdecode f y).cls = FFinite) :
    fp_sub f x y = roundSigned f (alignedSum (fdecode f x) (fdecode f y) true)
      (min (fdecode f x).e (fdecode f y).e) := by
  exact add_finite_decomposition f _ _ true hx hy

/-- Any nonzero finite product uses exact arbitrary-precision significand
    multiplication and exponent addition before one rounding call. -/
theorem fp_mul_finite_decomposition (f : fmt) (x y : BitVec 32)
    (hx : (fdecode f x).cls = FFinite) (hy : (fdecode f y).cls = FFinite) :
    fp_mul f x y = fround f ((fdecode f x).neg != (fdecode f y).neg)
      ((fdecode f x).mag * (fdecode f y).mag)
      ((fdecode f x).e + (fdecode f y).e) := by
  cases hn : (fdecode f x).neg <;> cases hm : (fdecode f y).neg <;>
    simp [fp_mul, hx, hy, neq_bool, hn, hm]

theorem add_nan_left (f : fmt) (a b : fval) (sub : Bool) (h : a.cls = FNaN) :
    fadd_core f a b sub = canon_nan f := by
  simp [fadd_core, h]

theorem add_nan_right (f : fmt) (a b : fval) (sub : Bool) (h : b.cls = FNaN) :
    fadd_core f a b sub = canon_nan f := by
  simp [fadd_core, h]

theorem add_signed_zeros (f : fmt) (a b : fval) (sub : Bool)
    (ha : a.cls = FZero) (hb : b.cls = FZero) :
    fadd_core f a b sub = zero_bits f (a.neg && (if sub then !b.neg else b.neg)) := by
  simp [fadd_core, ha, hb]

theorem add_opposite_infinities (f : fmt) (a b : fval)
    (ha : a.cls = FInf) (hb : b.cls = FInf) (hs : a.neg != b.neg) :
    fadd_core f a b false = canon_nan f := by
  cases hn : a.neg <;> cases hm : b.neg <;> simp_all [fadd_core]

theorem mul_nan_left (f : fmt) (x y : BitVec 32) (h : (fdecode f x).cls = FNaN) :
    fp_mul f x y = canon_nan f := by
  simp [fp_mul, h]

theorem mul_nan_right (f : fmt) (x y : BitVec 32) (h : (fdecode f y).cls = FNaN) :
    fp_mul f x y = canon_nan f := by
  simp [fp_mul, h]

theorem mul_inf_zero (f : fmt) (x y : BitVec 32)
    (hx : (fdecode f x).cls = FInf) (hy : (fdecode f y).cls = FZero) :
    fp_mul f x y = canon_nan f := by
  simp [fp_mul, hx, hy]

theorem mul_zero_inf (f : fmt) (x y : BitVec 32)
    (hx : (fdecode f x).cls = FZero) (hy : (fdecode f y).cls = FInf) :
    fp_mul f x y = canon_nan f := by
  simp [fp_mul, hx, hy]

/-- An exact finite cancellation is positive zero under RNE. -/
theorem add_exact_cancellation (f : fmt) (a b : fval) (sub : Bool)
    (ha : a.cls = FFinite) (hb : b.cls = FFinite)
    (hs : alignedSum a b sub = 0) : fadd_core f a b sub = zero_bits f false := by
  rw [add_finite_decomposition f a b sub ha hb, hs]
  simp [roundSigned]

/-- Sail's integer exponent is a natural exponent on nonnegative inputs. -/
theorem pow2i_nonnegative (n : Int) (hn : 0 ≤ n) :
    pow2i n = (2 : Int) ^ n.toNat := by
  by_cases hz : n ≤ 0
  · have he : n = 0 := by omega
    subst n
    simp [pow2i]
  · simp [pow2i, hz]
    rfl

/-- Integer scaling is exact; no bounded machine integer axiom is used. -/
theorem pow2i_add_nonnegative (a b : Int) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    pow2i (a + b) = pow2i a * pow2i b := by
  rw [pow2i_nonnegative (a + b) (by omega), pow2i_nonnegative a ha,
    pow2i_nonnegative b hb, Int.toNat_add ha hb, Int.pow_add]

/-- Changing a dyadic operand's exponent to a common lower exponent preserves
    its exact integer value after multiplication by any common denominator. -/
theorem align_operand_exact (mag e lo base : Int) (he : lo ≤ e) (hb : 0 ≤ lo + base) :
    (mag * pow2i (e - lo)) * pow2i (lo + base) = mag * pow2i (e + base) := by
  rw [Int.mul_assoc, ← pow2i_add_nonnegative (e - lo) (lo + base) (by omega) hb]
  congr 2
  omega

theorem signed_scale (n : Bool) (m s : Int) :
    signedMag n m * s = signedMag n (m * s) := by
  cases n <;> simp [signedMag, Int.neg_mul]

/-- The exact aligned sum used by the production finite add/subtract theorem
    represents precisely the sum of the two source dyadics. It covers all
    exponent differences, cancellation and signs; `base` is any denominator
    large enough to make both integer scalings nonnegative. -/
theorem aligned_sum_exact (a b : fval) (sub : Bool) (base : Int)
    (hb : 0 ≤ min a.e b.e + base) :
    alignedSum a b sub * pow2i (min a.e b.e + base) =
      signedMag a.neg (a.mag * pow2i (a.e + base)) +
      signedMag (if sub then !b.neg else b.neg) (b.mag * pow2i (b.e + base)) := by
  unfold alignedSum
  rw [Int.add_mul]
  rw [signed_scale, signed_scale]
  rw [align_operand_exact a.mag a.e (min a.e b.e) base (by omega) hb,
      align_operand_exact b.mag b.e (min a.e b.e) base (by omega) hb]

/-- XOR of operand signs is the sign of the exact product. -/
theorem product_sign_exact (na nb : Bool) (ma mb : Int) :
    signedMag (na != nb) (ma * mb) = signedMag na ma * signedMag nb mb := by
  cases na <;> cases nb <;> simp [signedMag, Int.neg_mul, Int.mul_neg]

/-- A class-table specification, distinct from the production conditional tree. -/
def mulClassSpec (f : fmt) (a b : fval) : BitVec 32 :=
  let neg := a.neg != b.neg
  match a.cls, b.cls with
  | FNaN, _ | _, FNaN => canon_nan f
  | FInf, FZero | FZero, FInf => canon_nan f
  | FInf, _ | _, FInf => inf_bits f neg
  | FZero, _ | _, FZero => zero_bits f neg
  | FFinite, FFinite => fround f neg (a.mag * b.mag) (a.e + b.e)

/-- All sixteen source class pairs (and all signs) are covered, not only
    selected NaN/zero witnesses. Rounding correctness is a separate obligation. -/
theorem fp_mul_class_table (f : fmt) (x y : BitVec 32) :
    fp_mul f x y = mulClassSpec f (fdecode f x) (fdecode f y) := by
  cases hx : (fdecode f x).cls <;> cases hy : (fdecode f y).cls <;>
    cases hn : (fdecode f x).neg <;> cases hm : (fdecode f y).neg <;>
    simp [fp_mul, mulClassSpec, hx, hy, hn, hm, neq_bool]

theorem add_zero_finite (f : fmt) (a b : fval) (sub : Bool)
    (ha : a.cls = FZero) (hb : b.cls = FFinite) (hm : 0 < b.mag) :
    fadd_core f a b sub = fround f (if sub then !b.neg else b.neg) b.mag b.e := by
  cases hs : sub <;> cases hn : b.neg <;>
    simp [fadd_core, ha, hb, hs, hn, pow2i]
  all_goals split <;> rename_i h <;> simp_all
  all_goals omega

theorem add_finite_zero (f : fmt) (a b : fval) (sub : Bool)
    (ha : a.cls = FFinite) (hb : b.cls = FZero) (hm : 0 < a.mag) :
    fadd_core f a b sub = fround f a.neg a.mag a.e := by
  cases hn : a.neg <;> simp [fadd_core, ha, hb, hn, pow2i]
  all_goals split <;> rename_i h <;> simp_all
  all_goals omega

/-- RNE lattice decision in exact integers. Here2*h is the spacing and r is
    the exact remainder; the production fround uses this same decision after
    division. This establishes the mathematical decision, not the still-open
    generic decode/packing theorem. -/
def rneChoice (q r h : Int) : Int :=
  if h < r ∨ (r = h ∧ q.tmod 2 = 1) then q + 1 else q

theorem rne_choice_half_ulp (q r h : Int) (hh : 0 < h)
    (hr0 : 0 ≤ r) (hr : r < 2 * h) :
    -h ≤ q * (2 * h) + r - rneChoice q r h * (2 * h) ∧
      q * (2 * h) + r - rneChoice q r h * (2 * h) ≤ h := by
  by_cases hs : h < r ∨ (r = h ∧ q.tmod 2 = 1)
  · simp [rneChoice, hs, Int.add_mul]
    omega
  · simp [rneChoice, hs]
    omega

end NumericalProofs
