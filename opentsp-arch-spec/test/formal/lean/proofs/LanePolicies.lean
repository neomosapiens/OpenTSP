-- Actual legal type/op domains and logical/comparison/conversion policies.
import Proofs.LaneArithmetic
import Out.VxmInsts
open Sail Out Out.Functions
set_option autoImplicit false
namespace LanePolicyProofs

def FloatBinary (op : Int) : Prop := op=0 ∨ op=3 ∨ op=6 ∨ op=9 ∨ op=10
def BoolBinary (op : Int) : Prop := op=11 ∨ op=12 ∨ op=13

theorem actual_type_classes (t : eltype) :
    (nbits_of t=8 ∨ nbits_of t=16 ∨ nbits_of t=32) ∧
    (is_bool t=true → is_float t=false ∧ is_unsigned t=false) ∧
    (is_unsigned t=true → is_float t=false ∧ is_bool t=false) := by
  cases t <;> decide +kernel

theorem actual_binary_guards (op : vxm_binop) (t : eltype) :
    binop_legal op t=true →
      let n := BitVec.toNatInt (encdec_vxm_binop_forwards op)
      (is_float t=true → FloatBinary n) ∧
      (is_bool t=true → BoolBinary n) ∧ (0 ≤ n ∧ n ≤ 13) := by
  unfold FloatBinary BoolBinary
  cases op <;> cases t <;> decide +kernel

theorem actual_unary_guards (op : vxm_unop) (t : eltype) :
    unop_legal op t=true →
      let n := BitVec.toNatInt (encdec_vxm_unop_forwards op)
      (n=14 ∨ n=15 ∨ n=16 ∨ n=20 ∨ n=21 ∨ n=22 ∨ n=23 ∨ n=24) ∧
      (21 ≤ n → (t==eltype.I8)=true ∨ (t==eltype.F16)=true ∨ (t==eltype.F32)=true) ∧
      (is_bool t=true → n=14) ∧ (is_unsigned t=true → n=14) := by
  cases op <;> cases t <;> decide +kernel

theorem actual_comparison_guards (mode : cmp_mode) (t : eltype) :
    cmp_legal mode t=true →
      let n := BitVec.toNatInt (encdec_cmp_mode_forwards mode)
      (0 ≤ n ∧ n ≤ 5) ∧ (is_bool t=true → n=0 ∨ n=1) := by
  cases mode <;> cases t <;> decide +kernel

def maskedTruth (t : eltype) (x : BitVec 32) : Bool := (x &&& all_ones t) != 0

def boolBinarySpec (op : Int) (t : eltype) (x y : BitVec 32) : BitVec 32 :=
  let a := maskedTruth t x
  let b := maskedTruth t y
  match op with
  | 11 => if a && b then all_ones t else 0
  | 12 => if a || b then all_ones t else 0
  | 13 => if a  !=  b then all_ones t else 0
  | _ => 0

theorem bool_binary_complete (op : Int) (t : eltype) (x y : BitVec 32)
    (ht : is_bool t=true) (hop : BoolBinary op) :
    k_binary op t x y=boolBinarySpec op t x y := by
  rcases hop with h|h|h <;> subst op <;>
    simp [k_binary,boolBinarySpec,maskedTruth,bool_bits,ht,neq_bool]

theorem bool_unary_not (t : eltype) (x : BitVec 32) (ht : is_bool t=true) :
    k_unary 14 t x=if maskedTruth t x then 0 else all_ones t := by
  have hf := (actual_type_classes t).2.1 ht
  simp [k_unary,ht,hf.1,maskedTruth,bool_bits]

def cmpPredicate (mode a b : Int) : Bool :=
  match mode with
  | 0 => a==b
  | 1 => a != b
  | 2 => a <b b
  | 3 => a ≤b b
  | 4 => a >b b
  | _ => a ≥b b

theorem integer_comparison_complete (mode : Int) (t : eltype) (x y : BitVec 32)
    (ht : is_float t=false ∧ is_bool t=false) (hm : 0 ≤ mode ∧ mode ≤ 5) :
    k_cmp mode t x y=if cmpPredicate mode (int_of t x) (int_of t y) then all_ones t else 0 := by
  have hc : mode=0 ∨ mode=1 ∨ mode=2 ∨ mode=3 ∨ mode=4 ∨ mode=5 := by omega
  rcases hc with h|h|h|h|h|h <;> subst mode <;> simp [k_cmp,cmpPredicate,ht.1,ht.2]

theorem bool_comparison_complete (mode : Int) (t : eltype) (x y : BitVec 32)
    (ht : is_bool t=true) (hm : mode=0 ∨ mode=1) :
    k_cmp mode t x y=
      if (if mode=0 then maskedTruth t x==maskedTruth t y else maskedTruth t x != maskedTruth t y)
      then all_ones t else 0 := by
  rcases hm with h|h <;> subst mode <;> simp [k_cmp,maskedTruth,ht,neq_bool]

theorem float_comparison_complete (mode : Int) (t : eltype) (x y : BitVec 32)
    (ht : is_float t=true ∧ is_bool t=false) (hm : 0 ≤ mode ∧ mode ≤ 5) :
    k_cmp mode t x y=
      if (if is_nan (fmt_of t) x || is_nan (fmt_of t) y then mode==1
          else cmpPredicate mode (fcmp (fmt_of t) x y) 0)
      then all_ones t else 0 := by
  have hc : mode=0 ∨ mode=1 ∨ mode=2 ∨ mode=3 ∨ mode=4 ∨ mode=5 := by omega
  rcases hc with h|h|h|h|h|h <;> subst mode <;> simp [k_cmp,cmpPredicate,ht.1,ht.2]

theorem float_binary_complete (op : Int) (t : eltype) (x y : BitVec 32)
    (ht : is_float t=true ∧ is_bool t=false) (hop : FloatBinary op) :
    k_binary op t x y=
      match op with
      | 0 => fp_add (fmt_of t) x y
      | 3 => fp_sub (fmt_of t) x y
      | 6 => fp_mul (fmt_of t) x y
      | 9 => fp_max (fmt_of t) x y
      | _ => fp_min (fmt_of t) x y := by
  rcases hop with h|h|h|h|h <;> subst op <;> simp [k_binary,ht.1,ht.2]

theorem convert_to_bool (src dst : eltype) (x : BitVec 32) (hd : is_bool dst=true) :
    k_convert_any src dst x=
      if (if is_float src then !(fdecode (fmt_of src) x).cls==fclass.FZero else maskedTruth src x)
      then all_ones dst else 0 := by
  simp [k_convert_any,hd,maskedTruth,Out.Functions.not]

theorem float_to_unsigned_source (src dst : eltype) (x : BitVec 32)
    (hs : is_float src=true) (hd : is_unsigned dst=true) :
    k_convert_any src dst x=bits_t dst (sat_t dst
      (max (int_of_fp (fmt_of src) x (nbits_of dst+1)) 0)) := by
  have hdtype := (actual_type_classes dst).2.2 hd
  have htype : is_float src=true → is_bool src=false := by cases src <;> decide +kernel
  have hstype := htype hs
  simp [k_convert_any,hs,hd,hdtype.1,hdtype.2,hstype]

theorem convert_to_float_complete (src dst : eltype) (x : BitVec 32) (hd : is_float dst=true) :
    k_convert_any src dst x=
      if is_float src then fp_convert (fmt_of src) (fmt_of dst) x
      else fp_of_int (fmt_of dst) (if is_bool src then (if maskedTruth src x then 1 else 0) else int_of src x) := by
  have ht : is_float dst=true → is_bool dst=false := by cases dst <;> decide +kernel
  have hdb := ht hd
  cases hs : is_float src <;> simp [k_convert_any,hd,hdb,hs,maskedTruth]

theorem integer_to_integer_conversion (src dst : eltype) (x : BitVec 32)
    (hs : is_float src=false ∧ is_bool src=false)
    (hd : is_float dst=false ∧ is_bool dst=false) :
    k_convert_any src dst x=bits_t dst (sat_t dst (int_of src x)) := by
  simp [k_convert_any,hs.1,hs.2,hd.1,hd.2]

theorem bool_to_integer_conversion (src dst : eltype) (x : BitVec 32)
    (hs : is_bool src=true) (hd : is_float dst=false ∧ is_bool dst=false) :
    k_convert_any src dst x=bits_t dst (sat_t dst (if maskedTruth src x then 1 else 0)) := by
  have hsf := (actual_type_classes src).2.1 hs
  simp [k_convert_any,hs,hsf.1,hd.1,hd.2,maskedTruth]

theorem float_to_signed_source (src dst : eltype) (x : BitVec 32)
    (hs : is_float src=true)
    (hd : is_float dst=false ∧ is_bool dst=false ∧ is_unsigned dst=false) :
    k_convert_any src dst x=bits_t dst (sat_s (int_of_fp (fmt_of src) x (nbits_of dst)) (nbits_of dst)) := by
  have htype : is_float src=true → is_bool src=false := by cases src <;> decide +kernel
  have hsb := htype hs
  simp [k_convert_any,hs,hsb,hd.1,hd.2.1,hd.2.2,sat_t]

def legacyConvertSpec (op : Int) (x : BitVec 32) : BitVec 32 :=
  match op with
  | 25 => fp_of_int FMT32 (s_of_bits x 8)
  | 26 => bits_of_s (int_of_fp FMT32 x 8) 8
  | 27 => fp_of_int FMT16 (s_of_bits x 8)
  | 28 => bits_of_s (int_of_fp FMT16 x 8) 8
  | 29 => bits_of_s (s_of_bits x 8) 32
  | 30 => bits_of_s (sat_s (s_of_bits x 32) 8) 8
  | 31 => fp_convert FMT16 FMT32 x
  | 32 => fp_convert FMT32 FMT16 x
  | _ => 0

theorem legacy_conversion_complete (op : Int) (x : BitVec 32) (ho : 25 ≤ op ∧ op ≤ 32) :
    k_convert op x=legacyConvertSpec op x := by
  have h : op=25 ∨ op=26 ∨ op=27 ∨ op=28 ∨ op=29 ∨ op=30 ∨ op=31 ∨ op=32 := by omega
  rcases h with h|h|h|h|h|h|h|h <;> subst op <;> simp [k_convert,legacyConvertSpec]

end LanePolicyProofs
