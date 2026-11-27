-- Complete scalar class tables and ordering policies over actual source.
import Proofs.CarrierDomains
open Sail Out Out.Functions Out.fclass
set_option autoImplicit false
namespace ScalarSemanticProofs
open NumericalProofs

def addClassSpec (f : fmt) (a b : fval) (sub : Bool) : BitVec 32 :=
  let bn := if sub then !b.neg else b.neg
  match a.cls,b.cls with
  | FNaN,_ | _,FNaN => canon_nan f
  | FInf,FInf => if a.neg=bn then inf_bits f a.neg else canon_nan f
  | FInf,_ => inf_bits f a.neg
  | _,FInf => inf_bits f bn
  | FZero,FZero => zero_bits f (a.neg && bn)
  | FZero,FFinite => roundSigned f (signedMag bn b.mag) b.e
  | FFinite,FZero => roundSigned f (signedMag a.neg a.mag) a.e
  | FFinite,FFinite => roundSigned f (alignedSum a b sub) (min a.e b.e)

theorem add_complete_class_table (f : fmt) (a b : fval) (sub : Bool) :
    fadd_core f a b sub=addClassSpec f a b sub := by
  have h0 : pow2i 0=1 := by decide
  have hmin : (if a.e< b.e then a.e else b.e)=min a.e b.e := by omega
  cases ha : a.cls <;> cases hb : b.cls <;> cases hs : sub <;>
    cases hn : a.neg <;> cases hm : b.neg <;>
    simp [fadd_core,addClassSpec,roundSigned,signedMag,alignedSum,ha,hb,hs,hn,hm,h0,hmin]

theorem public_add_class_table (f : fmt) (x y : BitVec 32) :
    fp_add f x y=addClassSpec f (fdecode f x) (fdecode f y) false :=
  add_complete_class_table f _ _ false

theorem public_sub_class_table (f : fmt) (x y : BitVec 32) :
    fp_sub f x y=addClassSpec f (fdecode f x) (fdecode f y) true :=
  add_complete_class_table f _ _ true

def compareInt (a b : Int) : Int := if a< b then -1 else if a> b then 1 else 0

def finiteCompare (a b : fval) : Int :=
  let base := min a.e b.e
  compareInt (signedMag a.neg (a.mag*pow2i (a.e-base)))
    (signedMag b.neg (b.mag*pow2i (b.e-base)))

theorem finite_compare_exact (f : fmt) (x y : BitVec 32)
    (ha : (fdecode f x).cls=FFinite) (hb : (fdecode f y).cls=FFinite)
    (hma : 0 < (fdecode f x).mag) (hmb : 0 < (fdecode f y).mag) :
    fcmp f x y=finiteCompare (fdecode f x) (fdecode f y) := by
  have hmin : (if (fdecode f x).e<(fdecode f y).e then (fdecode f x).e else (fdecode f y).e)=
      min (fdecode f x).e (fdecode f y).e := by omega
  have hpa := PositivityProofs.pow2i_any_positive ((fdecode f x).e-min (fdecode f x).e (fdecode f y).e)
  have hpb := PositivityProofs.pow2i_any_positive ((fdecode f y).e-min (fdecode f x).e (fdecode f y).e)
  have hva := Int.mul_pos hma hpa
  have hvb := Int.mul_pos hmb hpb
  cases hn : (fdecode f x).neg <;> cases hm : (fdecode f y).neg <;>
    simp [fcmp,finiteCompare,compareInt,signedMag,ha,hb,hn,hm,hmin]
  all_goals repeat' first | split | omega

theorem compare_int_range (a b : Int) : -1 ≤ compareInt a b ∧ compareInt a b ≤ 1 := by
  simp [compareInt]; split <;> (first | omega | (split <;> omega))

theorem compare_int_antisymmetric (a b : Int) : compareInt a b= -compareInt b a := by
  simp [compareInt]; repeat' first | split | omega

/-- maxNum/minNum policy: a single NaN returns the numeric operand; two NaNs
    produce canonical NaN; equal numeric values prefer positive or negative zero respectively. -/
theorem max_num_policy (f : fmt) (x y : BitVec 32) :
    fp_max f x y=
      if is_nan f x then (if is_nan f y then canon_nan f else y)
      else if is_nan f y then x
      else if 0<fcmp f x y then x else if fcmp f x y<0 then y
      else if (fdecode f x).neg then y else x := by
  cases hx : is_nan f x <;> cases hy : is_nan f y <;> simp [fp_max,hx,hy]

theorem min_num_policy (f : fmt) (x y : BitVec 32) :
    fp_min f x y=
      if is_nan f x then (if is_nan f y then canon_nan f else y)
      else if is_nan f y then x
      else if fcmp f x y<0 then x else if 0<fcmp f x y then y
      else if (fdecode f x).neg then x else y := by
  cases hx : is_nan f x <;> cases hy : is_nan f y <;> simp [fp_min,hx,hy]

def comparisonClassSpec (a b : fval) : Int :=
  match a.cls,b.cls with
  | FNaN,_ | _,FNaN => 0
  | FZero,FZero => 0
  | FInf,FInf => if a.neg=b.neg then 0 else if a.neg then -1 else 1
  | FInf,_ => if a.neg then -1 else 1
  | _,FInf => if b.neg then 1 else -1
  | FZero,FFinite => if b.neg then 1 else -1
  | FFinite,FZero => if a.neg then -1 else 1
  | FFinite,FFinite => finiteCompare a b

theorem complete_non_nan_comparison (f : fmt) (x y : BitVec 32)
    (ha : (fdecode f x).cls≠FNaN) (hb : (fdecode f y).cls≠FNaN)
    (hma : (fdecode f x).cls=FFinite → 0 < (fdecode f x).mag)
    (hmb : (fdecode f y).cls=FFinite → 0 < (fdecode f y).mag) :
    fcmp f x y=comparisonClassSpec (fdecode f x) (fdecode f y) := by
  have hmin : (if (fdecode f x).e < (fdecode f y).e then (fdecode f x).e else (fdecode f y).e)=
      min (fdecode f x).e (fdecode f y).e := by omega
  have hva : (fdecode f x).cls=FFinite → 0 < (fdecode f x).mag*pow2i ((fdecode f x).e-min (fdecode f x).e (fdecode f y).e) := by
    intro h; exact Int.mul_pos (hma h) (PositivityProofs.pow2i_any_positive _)
  have hvb : (fdecode f y).cls=FFinite → 0 < (fdecode f y).mag*pow2i ((fdecode f y).e-min (fdecode f x).e (fdecode f y).e) := by
    intro h; exact Int.mul_pos (hmb h) (PositivityProofs.pow2i_any_positive _)
  cases hx : (fdecode f x).cls <;> cases hy : (fdecode f y).cls <;>
    cases hn : (fdecode f x).neg <;> cases hm : (fdecode f y).neg <;>
    simp_all [fcmp,comparisonClassSpec,finiteCompare,compareInt,signedMag,hmin]
  all_goals repeat first | split | omega

end ScalarSemanticProofs
