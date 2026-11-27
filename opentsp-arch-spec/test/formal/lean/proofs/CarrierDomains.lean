-- Transfer exhaustive canonical lane certificates to every 32-bit carrier.
import Proofs.EndpointBounds
open Sail Out Out.Functions
set_option autoImplicit false
namespace CarrierDomainProofs

def canonicalLane (width : Nat) (x : BitVec 32) : BitVec 32 :=
  get_slice_int 32 (x.toNat % 2^width) 0

theorem fdecode_field_congruence (f : fmt) (x y : BitVec 32)
    (hs : field x (f.width-1) 1=field y (f.width-1) 1)
    (he : field x f.mbits f.ebits=field y f.mbits f.ebits)
    (hf : field x 0 f.mbits=field y 0 f.mbits) : fdecode f x=fdecode f y := by
  simp only [fdecode]
  rw [hs,he,hf]

theorem canonical_lane_values (x : BitVec 32) :
    BitVec.toNatInt (canonicalLane 8 x)=BitVec.toNatInt x%256 ∧
    BitVec.toNatInt (canonicalLane 16 x)=BitVec.toNatInt x%65536 := by
  simp [canonicalLane,get_slice_int,BitVec.extractLsb'_toNat,BitVec.toNat_ofInt,BitVec.toNatInt]
  omega

theorem field_low8 (x : BitVec 32) :
    field x 7 1=field (canonicalLane 8 x) 7 1 ∧
    field x 3 4=field (canonicalLane 8 x) 3 4 ∧
    field x 2 5=field (canonicalLane 8 x) 2 5 ∧
    field x 0 3=field (canonicalLane 8 x) 0 3 ∧
    field x 0 2=field (canonicalLane 8 x) 0 2 := by
  have hc := (canonical_lane_values x).1
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hm := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (256:Int)≠0)
  simp only [field,hc]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hm]
  simp [pow2i]
  omega

theorem field_low16 (x : BitVec 32) :
    field x 15 1=field (canonicalLane 16 x) 15 1 ∧
    field x 10 5=field (canonicalLane 16 x) 10 5 ∧
    field x 7 8=field (canonicalLane 16 x) 7 8 ∧
    field x 0 10=field (canonicalLane 16 x) 0 10 ∧
    field x 0 7=field (canonicalLane 16 x) 0 7 := by
  have hc := (canonical_lane_values x).2
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hm := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (65536:Int)≠0)
  simp only [field,hc]
  simp only [Int.tdiv_eq_ediv_of_nonneg hx,Int.tdiv_eq_ediv_of_nonneg hm]
  simp [pow2i]
  omega

theorem decoded_narrow_carrier (x : BitVec 32) :
    fdecode FMT16 x=fdecode FMT16 (canonicalLane 16 x) ∧
    fdecode FMT_BF16 x=fdecode FMT_BF16 (canonicalLane 16 x) ∧
    fdecode FMT_E4M3 x=fdecode FMT_E4M3 (canonicalLane 8 x) ∧
    fdecode FMT_E5M2 x=fdecode FMT_E5M2 (canonicalLane 8 x) := by
  have h8 := field_low8 x
  have h16 := field_low16 x
  exact ⟨fdecode_field_congruence _ _ _ h16.1 h16.2.1 h16.2.2.2.1,
    fdecode_field_congruence _ _ _ h16.1 h16.2.2.1 h16.2.2.2.2,
    fdecode_field_congruence _ _ _ h8.1 h8.2.1 h8.2.2.2.1,
    fdecode_field_congruence _ _ _ h8.1 h8.2.2.1 h8.2.2.2.2⟩

theorem fp_tr_decode_congruence (op : Int) (f : fmt) (x y : BitVec 32)
    (h : fdecode f x=fdecode f y) : fp_tr_value op f x=fp_tr_value op f y := by
  unfold fp_tr_value
  rw [h]

theorem signed_i8_domain (x : BitVec 32) : -128 ≤ s_of_bits x 8 ∧ s_of_bits x 8 ≤ 127 := by
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have ht := Int.tmod_eq_emod_of_nonneg (b:=256) hx
  have h0 := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (256:Int)≠0)
  have h1 := Int.emod_lt_of_pos (BitVec.toNatInt x) (by decide : (0:Int)<256)
  simp [s_of_bits,pow2i,Int.tmod_eq_emod_of_nonneg hx]
  split <;> omega

theorem canonical_narrow_ranges (x : BitVec 32) :
    0 ≤ BitVec.toNatInt (canonicalLane 8 x) ∧ BitVec.toNatInt (canonicalLane 8 x)<256 ∧
    0 ≤ BitVec.toNatInt (canonicalLane 16 x) ∧ BitVec.toNatInt (canonicalLane 16 x)<65536 := by
  have h8 := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (256:Int)≠0)
  have h8u := Int.emod_lt_of_pos (BitVec.toNatInt x) (by decide : (0:Int)<256)
  have h16 := Int.emod_nonneg (BitVec.toNatInt x) (by decide : (65536:Int)≠0)
  have h16u := Int.emod_lt_of_pos (BitVec.toNatInt x) (by decide : (0:Int)<65536)
  rw [(canonical_lane_values x).1,(canonical_lane_values x).2]
  exact ⟨h8,h8u,h16,h16u⟩

theorem fp_tr_narrow_carrier (op : Int) (x : BitVec 32) :
    fp_tr_value op FMT16 x=fp_tr_value op FMT16 (canonicalLane 16 x) ∧
    fp_tr_value op FMT_BF16 x=fp_tr_value op FMT_BF16 (canonicalLane 16 x) ∧
    fp_tr_value op FMT_E4M3 x=fp_tr_value op FMT_E4M3 (canonicalLane 8 x) ∧
    fp_tr_value op FMT_E5M2 x=fp_tr_value op FMT_E5M2 (canonicalLane 8 x) := by
  have h := decoded_narrow_carrier x
  exact ⟨fp_tr_decode_congruence _ _ _ _ h.1,fp_tr_decode_congruence _ _ _ _ h.2.1,
    fp_tr_decode_congruence _ _ _ _ h.2.2.1,fp_tr_decode_congruence _ _ _ _ h.2.2.2⟩

end CarrierDomainProofs
