-- Integer lane wrapping/saturation semantics and complete legal opcode tables.
import Proofs.ScalarSemantics
open Sail Out Out.Functions
set_option autoImplicit false
namespace LaneArithmeticProofs

def signedModulo (v bits : Int) : Int :=
  let u := v%pow2i bits
  if u ≥ pow2i (bits-1) then u-pow2i bits else u

theorem wrap_s_mathematical (v bits : Int) (hb : 1 ≤ bits) :
    wrap_s v bits=signedModulo v bits := by
  have hp := IntervalProofs.pow2i_positive bits (by omega)
  have hc := CallerBoundProofs.corrected_tmod v (pow2i bits) hp
  simp only [wrap_s,signedModulo]
  simp only at hc
  have he : (if ((v.tmod (pow2i bits)<b 0):Bool) then v.tmod (pow2i bits)+pow2i bits else v.tmod (pow2i bits))=
      v%pow2i bits := by simpa using hc
  rw [he]
  simp

theorem wrap_s_modulo (v bits : Int) (hb : 1 ≤ bits) :
    wrap_s v bits%pow2i bits=v%pow2i bits := by
  rw [wrap_s_mathematical v bits hb]
  simp only [signedModulo]
  split
  · rw [Int.sub_emod]; simp
  · simp

theorem unsigned_saturation_nearest (v bits z : Int) (hb : 1 ≤ bits)
    (hz : 0 ≤ z ∧ z < pow2i bits) :
    let c := if v<0 then 0 else if v ≥ pow2i bits then pow2i bits-1 else v
    (v-c).natAbs ≤ (v-z).natAbs := by
  dsimp only
  split <;> (first | omega | (split <;> omega))

theorem sat_t_range (t : eltype) (v : Int) :
    if is_unsigned t then 0 ≤ sat_t t v ∧ sat_t t v < pow2i (nbits_of t)
    else -pow2i (nbits_of t-1) ≤ sat_t t v ∧ sat_t t v < pow2i (nbits_of t-1) := by
  have hn := CallerBoundProofs.nbits_of_domain t
  have hp := IntervalProofs.pow2i_positive (nbits_of t) (by omega)
  cases hu : is_unsigned t
  · have h := SignedRoundingProofs.sat_s_range v (nbits_of t) hn.1
    simp only [sat_t,hu,Bool.false_eq_true,ite_false]
    exact ⟨h.1,by omega⟩
  · simp [sat_t,hu]
    split <;> (first | omega | (split <;> omega))

theorem sat_t_nearest (t : eltype) (v z : Int)
    (hz : if is_unsigned t then 0 ≤ z ∧ z < pow2i (nbits_of t)
      else -pow2i (nbits_of t-1) ≤ z ∧ z < pow2i (nbits_of t-1)) :
    (v-sat_t t v).natAbs ≤ (v-z).natAbs := by
  have hn := CallerBoundProofs.nbits_of_domain t
  cases hu : is_unsigned t
  · simp only [hu,Bool.false_eq_true,ite_false] at hz ⊢
    simpa [sat_t,hu] using SignedRoundingProofs.sat_s_nearest v (nbits_of t) z hn.1 (by omega)
  · simp only [hu,ite_true] at hz ⊢
    simpa [sat_t,hu] using unsigned_saturation_nearest v (nbits_of t) z hn.1 hz

def integerBinarySpec (op : Int) (t : eltype) (x y : BitVec 32) : BitVec 32 :=
  let a := int_of t x
  let b := int_of t y
  match op with
  | 0 | 2 => bits_t t (wrap_t t (a+b))
  | 1 => bits_t t (sat_t t (a+b))
  | 3 | 5 => bits_t t (wrap_t t (a-b))
  | 4 => bits_t t (sat_t t (a-b))
  | 6 | 8 => bits_t t (wrap_t t (a*b))
  | 7 => bits_t t (sat_t t (a*b))
  | 9 => bits_t t (max a b)
  | 10 => bits_t t (min a b)
  | 11 => (x &&& y) &&& all_ones t
  | 12 => (x ||| y) &&& all_ones t
  | 13 => (x ^^^ y) &&& all_ones t
  | _ => 0

theorem integer_binary_complete (op : Int) (t : eltype) (x y : BitVec 32)
    (ht : is_float t=false ∧ is_bool t=false) (hop : 0 ≤ op ∧ op ≤ 13) :
    k_binary op t x y=integerBinarySpec op t x y := by
  have hcases : op=0 ∨ op=1 ∨ op=2 ∨ op=3 ∨ op=4 ∨ op=5 ∨ op=6 ∨ op=7 ∨
      op=8 ∨ op=9 ∨ op=10 ∨ op=11 ∨ op=12 ∨ op=13 := by omega
  rcases hcases with h|h|h|h|h|h|h|h|h|h|h|h|h|h <;> subst op <;>
    simp [k_binary,integerBinarySpec,ht.1,ht.2,Int.max_def,Int.min_def]
  all_goals repeat first | split | rfl | omega
  all_goals congr 1 <;> omega

def integerUnarySpec (op : Int) (t : eltype) (x : BitVec 32) : BitVec 32 :=
  let a := int_of t x
  match op with
  | 14 => (~~~x) &&& all_ones t
  | 15 => bits_t t (wrap_t t (-a))
  | 16 => bits_t t (wrap_t t (if a<0 then -a else a))
  | 20 => bits_t t (max a 0)
  | _ => 0

theorem integer_unary_complete (op : Int) (t : eltype) (x : BitVec 32)
    (ht : is_float t=false ∧ is_bool t=false)
    (hop : op=14 ∨ op=15 ∨ op=16 ∨ op=20) :
    k_unary op t x=integerUnarySpec op t x := by
  rcases hop with h|h|h|h <;> subst op <;>
    simp [k_unary,integerUnarySpec,ht.1,ht.2,Int.max_def]
  repeat first | split | rfl | omega
  all_goals congr 1 <;> omega

/-- At signed minimum, two's-complement Neg and Abs intentionally wrap back
    to the signed minimum; they are not silently saturated to signed maximum. -/
theorem signed_minimum_wrap (bits : Int) (hb : 1 ≤ bits) :
    wrap_s (pow2i (bits-1)) bits= -pow2i (bits-1) := by
  have hp := RoundingProofs.positive_power_even bits (by omega)
  have hh := IntervalProofs.pow2i_positive (bits-1) (by omega)
  rw [wrap_s_mathematical _ _ hb]
  simp only [signedModulo]
  have he : pow2i (bits-1)%pow2i bits=pow2i (bits-1) :=
    Int.emod_eq_of_lt (by omega) (by rw [hp.1]; omega)
  rw [he]
  simp only [ge_iff_le,Int.le_refl,ite_true]
  rw [hp.1]
  omega

def unsignedClamp (v bits : Int) : Int :=
  if v<0 then 0 else if v ≥ pow2i bits then pow2i bits-1 else v

theorem unsigned_float_extra_bit_clamp (v bits : Int) (hb : 1 ≤ bits) :
    unsignedClamp (max (sat_s v (bits+1)) 0) bits=unsignedClamp v bits := by
  have hp := IntervalProofs.pow2i_positive bits (by omega)
  have he : bits+1-1=bits := by omega
  simp [sat_s,he,unsignedClamp,Int.max_def]
  all_goals repeat first | split | omega
  all_goals omega

theorem get_slice32_unsigned_exact (v : Int) (hv : 0 ≤ v ∧ v<4294967296) :
    BitVec.toNatInt (get_slice_int 32 v 0)=v := by
  simp [get_slice_int,BitVec.extractLsb'_toNat,BitVec.toNat_ofInt,BitVec.toNatInt]
  omega

theorem signed_integer_roundtrip (v bits : Int) (hb : 1 ≤ bits ∧ bits ≤ 32)
    (hv : -pow2i (bits-1) ≤ v ∧ v < pow2i (bits-1)) :
    s_of_bits (bits_of_s v bits) bits=v := by
  let H := pow2i (bits-1)
  let M := pow2i bits
  have hH : 0 < H := IntervalProofs.pow2i_positive _ (by omega)
  have hM : M=2*H := (RoundingProofs.positive_power_even bits (by omega)).1
  have hcap := GridBoundProofs.pow2_mono bits 32 (by omega) hb.2
  have h32 : pow2i 32=4294967296 := by decide
  let u := if v<0 then v+M else v
  have hu : 0 ≤ u ∧ u<M := by dsimp [u,M,H] at *; split <;> omega
  have hbv : BitVec.toNatInt (bits_of_s v bits)=u := by
    unfold bits_of_s
    simpa [M,u] using get_slice32_unsigned_exact u (by change 0 ≤ u ∧ u<4294967296; dsimp [M] at hu; omega)
  have ht := Int.tmod_eq_emod_of_nonneg (b:=M) hu.1
  have hmod : u%M=u := Int.emod_eq_of_lt hu.1 hu.2
  simp only [s_of_bits,hbv]
  change (if ((u.tmod M ≥b H):Bool) then u.tmod M-M else u.tmod M)=v
  rw [show (u.tmod M)=u by rw [ht,hmod]]
  by_cases hn : v<0
  · have hvalue : u=v+M := by simp [u,hn]
    have hhalf : H ≤ u := by dsimp [H] at *; omega
    simp [hhalf]
    omega
  · have hvalue : u=v := by simp [u,hn]
    have hhalf : ¬H ≤ u := by dsimp [H] at *; omega
    simp [hhalf]
    omega

theorem unsigned_integer_roundtrip (t : eltype) (v : Int) (hu : is_unsigned t=true)
    (hv : 0 ≤ v ∧ v<pow2i (nbits_of t)) : int_of t (bits_t t v)=v := by
  have hn := CallerBoundProofs.nbits_of_domain t
  have hpow := GridBoundProofs.pow2_mono (nbits_of t) 32 (by omega) hn.2
  have h32 : pow2i 32=4294967296 := by decide
  have hc := get_slice32_unsigned_exact v (by omega)
  have ht := Int.tmod_eq_emod_of_nonneg (b:=pow2i (nbits_of t)) hv.1
  simp [bits_t,int_of,hu,hc,ht,Int.emod_eq_of_lt hv.1 hv.2]

theorem signed_typed_roundtrip (t : eltype) (v : Int) (hu : is_unsigned t=false)
    (hv : -pow2i (nbits_of t-1) ≤ v ∧ v<pow2i (nbits_of t-1)) :
    int_of t (bits_t t v)=v := by
  have hn := CallerBoundProofs.nbits_of_domain t
  simpa [int_of,bits_t,hu] using signed_integer_roundtrip v (nbits_of t) hn hv

theorem wrap_t_mathematical (t : eltype) (v : Int) :
    wrap_t t v=if is_unsigned t then v%pow2i (nbits_of t) else signedModulo v (nbits_of t) := by
  have hn := CallerBoundProofs.nbits_of_domain t
  cases hu : is_unsigned t
  · simp [wrap_t,hu,wrap_s_mathematical v (nbits_of t) hn.1]
  · simp [wrap_t,hu]
    rfl

theorem wrap_t_range (t : eltype) (v : Int) :
    if is_unsigned t then 0 ≤ wrap_t t v ∧ wrap_t t v<pow2i (nbits_of t)
    else -pow2i (nbits_of t-1) ≤ wrap_t t v ∧ wrap_t t v<pow2i (nbits_of t-1) := by
  have hn := CallerBoundProofs.nbits_of_domain t
  cases hu : is_unsigned t
  · simpa [wrap_t,hu] using CallerBoundProofs.wrap_s_range v (nbits_of t) hn.1
  · have hp := IntervalProofs.pow2i_positive (nbits_of t) (by omega)
    simp [wrap_t,hu]
    exact ⟨Int.emod_nonneg v (by omega),Int.emod_lt_of_pos v hp⟩

theorem wrapped_lane_decode_exact (t : eltype) (v : Int) :
    int_of t (bits_t t (wrap_t t v))=wrap_t t v := by
  have h := wrap_t_range t v
  cases hu : is_unsigned t
  · apply signed_typed_roundtrip t _ hu
    simpa [hu] using h
  · apply unsigned_integer_roundtrip t _ hu
    simpa [hu] using h

theorem saturated_lane_decode_exact (t : eltype) (v : Int) :
    int_of t (bits_t t (sat_t t v))=sat_t t v := by
  have h := sat_t_range t v
  cases hu : is_unsigned t
  · apply signed_typed_roundtrip t _ hu
    simpa [hu] using h
  · apply unsigned_integer_roundtrip t _ hu
    simpa [hu] using h

end LaneArithmeticProofs
