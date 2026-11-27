-- Exact encoded sign operations for every admitted floating-point format.
import Proofs.LanePolicies
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
namespace FpSignProofs

def clearLane (width : Int) (x : BitVec 32) : BitVec 32 :=
  get_slice_int 32 (BitVec.toNatInt x%pow2i (width-1)) 0

def toggleLane (width : Int) (x : BitVec 32) : BitVec 32 :=
  get_slice_int 32 ((BitVec.toNatInt x+pow2i (width-1))%pow2i width) 0

def Supported (f : fmt) : Prop :=
  f=FMT32 ∨ f=FMT16 ∨ f=FMT_BF16 ∨ f=FMT_E4M3 ∨ f=FMT_E5M2

theorem actual_fmt_supported (t : eltype) : Supported (fmt_of t) := by
  cases t <;> simp [Supported,fmt_of]

theorem transform_values_generic (width : Int) (x : BitVec 32) (hw : 1 ≤ width ∧ width ≤ 32) :
    BitVec.toNatInt (clearLane width x)=BitVec.toNatInt x%pow2i (width-1) ∧
    BitVec.toNatInt (toggleLane width x)=(BitVec.toNatInt x+pow2i (width-1))%pow2i width := by
  have hH := IntervalProofs.pow2i_positive (width-1) (by omega)
  have hM := IntervalProofs.pow2i_positive width (by omega)
  have h32 : pow2i 32=4294967296 := by decide +kernel
  have hc := GridBoundProofs.pow2_mono width 32 (by omega) hw.2
  have hcH := GridBoundProofs.pow2_mono (width-1) width (by omega) (by omega)
  have h0 := Int.emod_nonneg (BitVec.toNatInt x) (by omega : pow2i (width-1)≠0)
  have h1 := Int.emod_lt_of_pos (BitVec.toNatInt x) hH
  have h2 := Int.emod_nonneg (BitVec.toNatInt x+pow2i (width-1)) (by omega : pow2i width≠0)
  have h3 := Int.emod_lt_of_pos (BitVec.toNatInt x+pow2i (width-1)) hM
  exact ⟨LaneArithmeticProofs.get_slice32_unsigned_exact _ (by omega),
    LaneArithmeticProofs.get_slice32_unsigned_exact _ (by omega)⟩

theorem transform_values8 (x : BitVec 32) :
    BitVec.toNatInt (clearLane 8 x)=BitVec.toNatInt x%128 ∧
    BitVec.toNatInt (toggleLane 8 x)=(BitVec.toNatInt x+128)%256 := by
  have h := transform_values_generic 8 x (by decide)
  have hH : pow2i (8-1)=128 := by decide +kernel
  have hM : pow2i 8=256 := by decide +kernel
  simpa only [hH,hM] using h

theorem transform_values16 (x : BitVec 32) :
    BitVec.toNatInt (clearLane 16 x)=BitVec.toNatInt x%32768 ∧
    BitVec.toNatInt (toggleLane 16 x)=(BitVec.toNatInt x+32768)%65536 := by
  have h := transform_values_generic 16 x (by decide)
  have hH : pow2i (16-1)=32768 := by decide +kernel
  have hM : pow2i 16=65536 := by decide +kernel
  simpa only [hH,hM] using h

theorem transform_values32 (x : BitVec 32) :
    BitVec.toNatInt (clearLane 32 x)=BitVec.toNatInt x%2147483648 ∧
    BitVec.toNatInt (toggleLane 32 x)=(BitVec.toNatInt x+2147483648)%4294967296 := by
  have h := transform_values_generic 32 x (by decide)
  have hH : pow2i (32-1)=2147483648 := by decide +kernel
  have hM : pow2i 32=4294967296 := by decide +kernel
  simpa only [hH,hM] using h

theorem transform_values (x : BitVec 32) :
    BitVec.toNatInt (clearLane 8 x)=BitVec.toNatInt x%128 ∧
    BitVec.toNatInt (toggleLane 8 x)=(BitVec.toNatInt x+128)%256 ∧
    BitVec.toNatInt (clearLane 16 x)=BitVec.toNatInt x%32768 ∧
    BitVec.toNatInt (toggleLane 16 x)=(BitVec.toNatInt x+32768)%65536 ∧
    BitVec.toNatInt (clearLane 32 x)=BitVec.toNatInt x%2147483648 ∧
    BitVec.toNatInt (toggleLane 32 x)=(BitVec.toNatInt x+2147483648)%4294967296 :=
  ⟨(transform_values8 x).1,(transform_values8 x).2,(transform_values16 x).1,
    (transform_values16 x).2,(transform_values32 x).1,(transform_values32 x).2⟩

def SignFields (f : fmt) (x : BitVec 32) : Prop :=
  let sign := field x (f.width-1) 1
  0 ≤ sign ∧ sign ≤ 1 ∧
  field (clearLane f.width x) (f.width-1) 1=0 ∧
  field (toggleLane f.width x) (f.width-1) 1=1-sign ∧
  field (clearLane f.width x) f.mbits f.ebits=field x f.mbits f.ebits ∧
  field (toggleLane f.width x) f.mbits f.ebits=field x f.mbits f.ebits ∧
  field (clearLane f.width x) 0 f.mbits=field x 0 f.mbits ∧
  field (toggleLane f.width x) 0 f.mbits=field x 0 f.mbits

theorem e4_sign_fields (x : BitVec 32) : SignFields FMT_E4M3 x := by
  have hv := transform_values x
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hclear := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (128:Int)≠0)
  have htoggle := Int.emod_nonneg (BitVec.toNatInt x+128) (by decide : (256:Int)≠0)
  simp only [SignFields,FMT_E4M3]
  simp only [field,hv.1,hv.2.1]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hclear,Int.tdiv_eq_ediv_of_nonneg htoggle]
  simp [pow2i]
  omega

theorem e5_sign_fields (x : BitVec 32) : SignFields FMT_E5M2 x := by
  have hv := transform_values x
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hclear := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (128:Int)≠0)
  have htoggle := Int.emod_nonneg (BitVec.toNatInt x+128) (by decide : (256:Int)≠0)
  simp only [SignFields,FMT_E5M2]
  simp only [field,hv.1,hv.2.1]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hclear,Int.tdiv_eq_ediv_of_nonneg htoggle]
  simp [pow2i]
  omega

theorem f16_sign_fields (x : BitVec 32) : SignFields FMT16 x := by
  have hv := transform_values x
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hclear := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (32768:Int)≠0)
  have htoggle := Int.emod_nonneg (BitVec.toNatInt x+32768) (by decide : (65536:Int)≠0)
  simp only [SignFields,FMT16]
  simp only [field,hv.2.2.1,hv.2.2.2.1]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hclear,Int.tdiv_eq_ediv_of_nonneg htoggle]
  simp [pow2i]
  omega

theorem bf16_sign_fields (x : BitVec 32) : SignFields FMT_BF16 x := by
  have hv := transform_values x
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hclear := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (32768:Int)≠0)
  have htoggle := Int.emod_nonneg (BitVec.toNatInt x+32768) (by decide : (65536:Int)≠0)
  simp only [SignFields,FMT_BF16]
  simp only [field,hv.2.2.1,hv.2.2.2.1]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hclear,Int.tdiv_eq_ediv_of_nonneg htoggle]
  simp [pow2i]
  omega

theorem f32_sign_fields (x : BitVec 32) : SignFields FMT32 x := by
  have hv := transform_values x
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hclear := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (2147483648:Int)≠0)
  have htoggle := Int.emod_nonneg (BitVec.toNatInt x+2147483648) (by decide : (4294967296:Int)≠0)
  simp only [SignFields,FMT32]
  simp only [field,hv.2.2.2.2.1,hv.2.2.2.2.2]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hclear,Int.tdiv_eq_ediv_of_nonneg htoggle]
  simp [pow2i]
  omega

theorem all_sign_fields (f : fmt) (x : BitVec 32) (hf : Supported f) : SignFields f x := by
  rcases hf with h|h|h|h|h <;> subst f
  · exact f32_sign_fields x
  · exact f16_sign_fields x
  · exact bf16_sign_fields x
  · exact e4_sign_fields x
  · exact e5_sign_fields x

theorem decoder_sign_value (f : fmt) (x : BitVec 32) :
    (fdecode f x).neg=(field x (f.width-1) 1==1) := by
  simp [fdecode]
  repeat first | split | rfl

theorem decoded_sign_congruence (f : fmt) (x y : BitVec 32) (neg : Bool)
    (hs : (field y (f.width-1) 1==1)=neg)
    (he : field y f.mbits f.ebits=field x f.mbits f.ebits)
    (hf : field y 0 f.mbits=field x 0 f.mbits) :
    fdecode f y={ fdecode f x with neg := neg } := by
  simp only [fdecode]
  rw [he,hf,hs]
  repeat first | split | rfl

theorem abs_source_non_nan (f : fmt) (x : BitVec 32) (hn : is_nan f x=false) :
    fp_abs f x=clearLane f.width x := by
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  simp [fp_abs,hn,clearLane,Int.tmod_eq_emod_of_nonneg hx]

theorem neg_source_non_nan (f : fmt) (x : BitVec 32) (hn : is_nan f x=false) :
    fp_neg f x=toggleLane f.width x := by
  have hx : 0 ≤ BitVec.toNatInt x+pow2i (f.width-1) := by
    have h0 : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
    have hp := PositivityProofs.pow2i_any_positive (f.width-1)
    omega
  simp [fp_neg,hn,toggleLane,Int.tmod_eq_emod_of_nonneg hx]

theorem abs_decoded_semantics (f : fmt) (x : BitVec 32) (hf : Supported f)
    (hn : is_nan f x=false) :
    fdecode f (fp_abs f x)={ fdecode f x with neg := false } := by
  have h := all_sign_fields f x hf
  dsimp only [SignFields] at h
  rw [abs_source_non_nan f x hn]
  apply decoded_sign_congruence f x (clearLane f.width x) false _ h.2.2.2.2.1 h.2.2.2.2.2.2.1
  rw [h.2.2.1]
  decide

theorem neg_decoded_semantics (f : fmt) (x : BitVec 32) (hf : Supported f)
    (hn : is_nan f x=false) :
    fdecode f (fp_neg f x)={ fdecode f x with neg := !(fdecode f x).neg } := by
  have h := all_sign_fields f x hf
  dsimp only [SignFields] at h
  rw [neg_source_non_nan f x hn]
  apply decoded_sign_congruence f x (toggleLane f.width x) (!(fdecode f x).neg) _ h.2.2.2.2.2.1 h.2.2.2.2.2.2.2
  rw [h.2.2.2.1,decoder_sign_value]
  have hs : field x (f.width-1) 1=0 ∨ field x (f.width-1) 1=1 := by omega
  rcases hs with hs|hs <;> rw [hs] <;> decide

theorem sign_ops_canonical_nan (f : fmt) (x : BitVec 32) (hn : is_nan f x=true) :
    fp_abs f x=canon_nan f ∧ fp_neg f x=canon_nan f := by simp [fp_abs,fp_neg,hn]

theorem float_sign_unary_source (t : eltype) (x : BitVec 32) (ht : is_float t=true) :
    k_unary 15 t x=fp_neg (fmt_of t) x ∧ k_unary 16 t x=fp_abs (fmt_of t) x := by
  simp [k_unary,ht]

theorem float_relu_source (t : eltype) (x : BitVec 32) (ht : is_float t=true) :
    k_unary 20 t x=
      let a := fdecode (fmt_of t) x
      if a.cls==fclass.FNaN then canon_nan (fmt_of t)
      else if a.neg || a.cls==fclass.FZero then zero_bits (fmt_of t) false
      else x &&& all_ones t := by
  simp [k_unary,ht]

end FpSignProofs
