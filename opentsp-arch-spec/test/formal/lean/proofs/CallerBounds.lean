-- Coarse complete caller-domain bounds for the actual bounded ilog2 helper.
import Proofs.SignedRounding
import Out.MxmInsts
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 100000
set_option exponentiation.threshold 100000
namespace CallerBoundProofs
open GridBoundProofs NumericalProofs

theorem pow2_any_upper (e b : Int) (hb : 0 ≤ b) (he : e ≤ b) :
    pow2i e ≤ pow2i b := by
  by_cases hp : 0 ≤ e
  · exact pow2_mono e b hp he
  · have hB := IntervalProofs.pow2i_positive b hb
    have hz : pow2i e=1 := by simp [pow2i,show e ≤ 0 by omega]
    rw [hz]; omega

theorem nonnegative_product_bound (a b A B : Int) (ha : 0 ≤ a ∧ a ≤ A)
    (hb : 0 ≤ b ∧ b ≤ B) : 0 ≤ a*b ∧ a*b ≤ A*B := by
  have h1 := Int.mul_le_mul_of_nonneg_right ha.2 hb.1
  have h2 := Int.mul_le_mul_of_nonneg_left hb.2 (by omega : 0 ≤ A)
  exact ⟨Int.mul_nonneg ha.1 hb.1,by omega⟩

theorem decoded_product_bound (a b ae be u adjustment : Int)
    (ha : 0 ≤ a ∧ a < pow2i 24) (hb : 0 ≤ b ∧ b < pow2i 24)
    (hea : ae ≤ 120) (heb : be ≤ 120) (hu : u ≤ 266) (hadj : adjustment ≤ 14) :
    0 ≤ a*b*pow2i (ae+be+u+adjustment) ∧
      a*b*pow2i (ae+be+u+adjustment) ≤ pow2i 568 := by
  have h1 := nonnegative_product_bound a b (pow2i 24) (pow2i 24) ⟨ha.1,by omega⟩ ⟨hb.1,by omega⟩
  have hp := NumericalProofs.pow2i_add_nonnegative 24 24 (by decide) (by decide)
  rw [show (24:Int)+24=48 by decide] at hp
  rw [←hp] at h1
  have he := pow2_any_upper (ae+be+u+adjustment) 520 (by decide) (by omega)
  have hz := PositivityProofs.pow2i_any_positive (ae+be+u+adjustment)
  have h2 := nonnegative_product_bound (a*b) (pow2i (ae+be+u+adjustment))
    (pow2i 48) (pow2i 520) h1 ⟨by omega,he⟩
  have hf := NumericalProofs.pow2i_add_nonnegative 48 520 (by decide) (by decide)
  rw [show (48:Int)+520=568 by decide] at hf
  rw [hf]
  exact h2

theorem finite_multiply_magnitude_bound (a b : Int)
    (ha : 0 ≤ a ∧ a < pow2i 24) (hb : 0 ≤ b ∧ b < pow2i 24) :
    0 ≤ a*b ∧ a*b < pow2i 49 := by
  have h := nonnegative_product_bound a b (pow2i 24) (pow2i 24) ⟨ha.1,by omega⟩ ⟨hb.1,by omega⟩
  have hp := NumericalProofs.pow2i_add_nonnegative 24 24 (by decide) (by decide)
  rw [show (24:Int)+24=48 by decide] at hp
  rw [←hp] at h
  have h49 := RoundingProofs.positive_power_even 49 (by decide)
  rw [show (49:Int)-1=48 by decide] at h49
  have h48 := IntervalProofs.pow2i_positive 48 (by decide)
  exact ⟨h.1,by omega⟩

theorem aligned_component_bound (a ae base : Int)
    (ha : 0 ≤ a ∧ a < pow2i 24) (he : ae ≤ 120) (hl : -149 ≤ base)
    (hb : base ≤ ae) :
    0 ≤ a*pow2i (ae-base) ∧ a*pow2i (ae-base) ≤ pow2i 293 := by
  have hp := pow2_mono (ae-base) 269 (by omega) (by omega)
  have hz := IntervalProofs.pow2i_positive (ae-base) (by omega)
  have h := nonnegative_product_bound a (pow2i (ae-base)) (pow2i 24) (pow2i 269)
    ⟨ha.1,by omega⟩ ⟨by omega,hp⟩
  have he := NumericalProofs.pow2i_add_nonnegative 24 269 (by decide) (by decide)
  rw [show (24:Int)+269=293 by decide] at he
  rw [he]
  exact h

theorem aligned_sum_magnitude_bound (a b : fval) (sub : Bool)
    (ha : 0 ≤ a.mag ∧ a.mag < pow2i 24 ∧ -149 ≤ a.e ∧ a.e ≤ 120)
    (hb : 0 ≤ b.mag ∧ b.mag < pow2i 24 ∧ -149 ≤ b.e ∧ b.e ≤ 120) :
    -pow2i 294 ≤ alignedSum a b sub ∧ alignedSum a b sub ≤ pow2i 294 := by
  have hA := aligned_component_bound a.mag a.e (min a.e b.e) ⟨ha.1,ha.2.1⟩ ha.2.2.2 (by omega) (by omega)
  have hB := aligned_component_bound b.mag b.e (min a.e b.e) ⟨hb.1,hb.2.1⟩ hb.2.2.2 (by omega) (by omega)
  have hp := RoundingProofs.positive_power_even 294 (by decide)
  rw [show (294:Int)-1=293 by decide] at hp
  cases a.neg <;> cases b.neg <;> cases sub <;> simp [alignedSum,signedMag] <;> omega

theorem nbits_of_domain (t : eltype) : 1 ≤ nbits_of t ∧ nbits_of t ≤ 32 := by
  cases t <;> decide +kernel

theorem scalar_integer_decode_bound (t : eltype) (x : BitVec 32) :
    -pow2i 32 ≤ int_of t x ∧ int_of t x < pow2i 32 := by
  have hn := nbits_of_domain t
  have hx : 0 ≤ BitVec.toNatInt x := by simp [BitVec.toNatInt]
  have hpow := IntervalProofs.pow2i_positive (nbits_of t) (by omega)
  have hhalf := IntervalProofs.pow2i_positive (nbits_of t-1) (by omega)
  have hfull := RoundingProofs.positive_power_even (nbits_of t) (by omega)
  have htop := pow2_mono (nbits_of t) 32 (by omega) (by omega)
  have h0 := Int.emod_nonneg (BitVec.toNatInt x) (by omega : pow2i (nbits_of t)≠0)
  have h1 := Int.emod_lt_of_pos (BitVec.toNatInt x) hpow
  have ht := Int.tmod_eq_emod_of_nonneg (b:=pow2i (nbits_of t)) hx
  simp [int_of,s_of_bits,ht]
  split <;> (first | omega | (split <;> omega))


theorem corrected_tmod (v m : Int) (hm : 0 < m) :
    (if v.tmod m < 0 then v.tmod m+m else v.tmod m)=v%m := by
  have h0 := Int.emod_nonneg v (by omega : m≠0)
  have h1 := Int.emod_lt_of_pos v hm
  have habs : (m.natAbs:Int)=m := by omega
  by_cases h : 0 ≤ v ∨ m ∣ v
  · rw [Int.tmod_eq_emod]
    simp [h,show ¬v%m < 0 by omega]
  · have hn : v%m≠0 := by intro h; exact (by omega : ¬m ∣ v) (Int.dvd_of_emod_eq_zero h)
    rw [Int.tmod_eq_emod]
    simp [h,habs,show v%m-m < 0 by omega]

theorem wrap_s_range (v bits : Int) (hb : 1 ≤ bits) :
    -pow2i (bits-1) ≤ wrap_s v bits ∧ wrap_s v bits < pow2i (bits-1) := by
  have hm := IntervalProofs.pow2i_positive bits (by omega)
  have hh := IntervalProofs.pow2i_positive (bits-1) (by omega)
  have hp := RoundingProofs.positive_power_even bits (by omega)
  have h0 := Int.emod_nonneg v (by omega : pow2i bits≠0)
  have h1 := Int.emod_lt_of_pos v hm
  have hcorr := corrected_tmod v (pow2i bits) hm
  unfold wrap_s
  simp only
  have he : (if (((v.tmod (pow2i bits)) <b 0):Bool) then v.tmod (pow2i bits)+pow2i bits else v.tmod (pow2i bits))=v%pow2i bits := by simpa using hcorr
  rw [he]
  by_cases hg : pow2i (bits-1) ≤ v%pow2i bits <;> simp [hg] <;> omega

theorem wlane_finite_add_source (a b : wlane) (ha : a.wsp=0) (hb : b.wsp=0) :
    wlane_add a b={wv:=wrap_s (a.wv+b.wv) 128,wsp:=0,wnz:=a.wnz && b.wnz} := by
  simp [wlane_add,ha,hb]

theorem wlane_finite_add_bound (a b : wlane) (ha : a.wsp=0) (hb : b.wsp=0) :
    -pow2i 127 ≤ (wlane_add a b).wv ∧ (wlane_add a b).wv < pow2i 127 := by
  rw [wlane_finite_add_source a b ha hb]
  exact wrap_s_range (a.wv+b.wv) 128 (by decide)

theorem wlane_finite_read_source (w : wlane) (h : w.wsp=0) :
    wlane_read w=
      if w.wv=0 then zero_bits FMT32 w.wnz
      else if w.wv<0 then fround FMT32 true (-w.wv) (-48)
      else fround FMT32 false w.wv (-48) := by simp [wlane_read,h]

theorem dsum_finite_read_source (d : dsum) (u : Int)
    (hn : d.dnan=false) (hp : d.dpinf=false) (hm : d.dninf=false) :
    dsum_round_u d u=
      if d.dsum=0 then zero_bits FMT32 d.dnz
      else if d.dsum<0 then fround FMT32 true (-d.dsum) (-u)
      else fround FMT32 false d.dsum (-u) := by simp [dsum_round_u,hn,hp,hm]

theorem fp_convert_finite_source (src dst : fmt) (x : BitVec 32)
    (hc : (fdecode src x).cls=fclass.FFinite) :
    fp_convert src dst x=fround dst (fdecode src x).neg (fdecode src x).mag (fdecode src x).e := by
  simp [fp_convert,hc]

theorem fp_of_integer_source (f : fmt) (v : Int) :
    fp_of_int f v=fround f (decide (v<0)) (if v<0 then -v else v) 0 := by
  by_cases hv : v<0 <;> simp [fp_of_int,hv]


def magnitudeOf (v : Int) : Int := if v<0 then -v else v

theorem magnitude_of_absolute (v : Int) : magnitudeOf v=(v.natAbs:Int) := by
  dsimp [magnitudeOf]
  split <;> omega

theorem signed_bound_to_ilog_domain (v bits : Int) (hb : 0 ≤ bits) (hbits : bits+1 ≤ 16384)
    (hv : -pow2i bits ≤ v ∧ v ≤ pow2i bits) :
    0 ≤ magnitudeOf v ∧ magnitudeOf v < pow2i 16384 := by
  have hp := RoundingProofs.positive_power_even (bits+1) (by omega)
  rw [show bits+1-1=bits by omega] at hp
  have hm := pow2_mono (bits+1) 16384 (by omega) hbits
  have hpos := IntervalProofs.pow2i_positive bits hb
  dsimp [magnitudeOf]
  split <;> omega

theorem nat_bound_to_ilog_domain (v : Int) (bits : Nat) (hbits : bits+1 ≤ 16384)
    (hv : v.natAbs ≤ 2^bits) :
    0 ≤ magnitudeOf v ∧ magnitudeOf v < pow2i 16384 := by
  have hp : pow2i (bits:Int)=((2^bits:Nat):Int) := by
    rw [NumericalProofs.pow2i_nonnegative (bits:Int) (by omega)]
    simp [Int.natCast_pow]
  have hc : (v.natAbs:Int) ≤ ((2^bits:Nat):Int) := Int.ofNat_le.mpr hv
  apply signed_bound_to_ilog_domain v (bits:Int) (by omega) (by omega)
  rw [←hp] at hc
  omega


def WideBound (w : wlane) : Prop := -pow2i 600 ≤ w.wv ∧ w.wv ≤ pow2i 600

theorem wide_zero_bound : WideBound wlane_zero := by
  have h := IntervalProofs.pow2i_positive 600 (by decide)
  simpa [WideBound,wlane_zero] using And.intro (by omega : -pow2i 600 ≤ 0) (by omega : 0 ≤ pow2i 600)

theorem dsum_wide_bound (d : dsum) (h : -pow2i 600 ≤ d.dsum ∧ d.dsum ≤ pow2i 600) :
    WideBound (dsum_wide d) := by
  have hp := IntervalProofs.pow2i_positive 600 (by decide)
  unfold WideBound
  simp [dsum_wide]
  split
  · change -pow2i 600 ≤ 0 ∧ 0 ≤ pow2i 600; omega
  · split
    · change -pow2i 600 ≤ 0 ∧ 0 ≤ pow2i 600; omega
    · split
      · change -pow2i 600 ≤ 0 ∧ 0 ≤ pow2i 600; omega
      · exact h

theorem wide_add_preserves_bound (a b : wlane) (ha : WideBound a) (hb : WideBound b) :
    WideBound (wlane_add a b) := by
  have hp := IntervalProofs.pow2i_positive 600 (by decide)
  have hw := wrap_s_range (a.wv+b.wv) 128 (by decide)
  rw [show (128:Int)-1=127 by decide] at hw
  have hmono := pow2_mono 127 600 (by decide) (by decide)
  unfold WideBound at *
  simp [wlane_add]
  split
  · change -pow2i 600 ≤ 0 ∧ 0 ≤ pow2i 600; omega
  · split
    · split
      · change -pow2i 600 ≤ wrap_s (a.wv+b.wv) 128 ∧ wrap_s (a.wv+b.wv) 128 ≤ pow2i 600
        omega
      · exact hb
    · exact ha

theorem wide_read_magnitude_domain (w : wlane) (h : WideBound w) :
    0 ≤ magnitudeOf w.wv ∧ magnitudeOf w.wv < pow2i 16384 :=
  signed_bound_to_ilog_domain w.wv 600 (by decide) (by decide) h

end CallerBoundProofs
