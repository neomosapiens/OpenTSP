-- Nearest rounding against every finite dyadic format candidate.
import Proofs.GridBounds
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxHeartbeats 1000000
namespace FormatNearestProofs
open RoundingProofs GridBoundProofs

theorem scaled_nearest (a r d scale : Int) (hs : 0 < scale)
    (h : ∀ z : Int, (a-r*d).natAbs ≤ (a-z*d).natAbs) (z : Int) :
    (a*scale-r*(d*scale)).natAbs ≤ (a*scale-z*(d*scale)).natAbs := by
  have he1 : a*scale-r*(d*scale)=(a-r*d)*scale := by grobner
  have he2 : a*scale-z*(d*scale)=(a-z*d)*scale := by grobner
  rw [he1,he2,Int.natAbs_mul,Int.natAbs_mul]
  exact Nat.mul_le_mul_right scale.natAbs (h z)

theorem below_boundary_distance (a r d boundary value : Int)
    (h : ∀ z : Int, (a-r*d).natAbs ≤ (a-z*d).natAbs)
    (hb : boundary ≤ a) (hv : value ≤ boundary) (k : Int) (hk : boundary=k*d) :
    (a-r*d).natAbs ≤ (a-value).natAbs := by
  have hn := h k
  rw [←hk] at hn
  have hd : (a-boundary).natAbs ≤ (a-value).natAbs := by omega
  exact Nat.le_trans hn hd

theorem grid_nearest_common_unit (mag e q unit : Int) (hm : 0 ≤ mag)
    (hu : unit ≤ e ∧ unit ≤ q) (z : Int) :
    (mag*pow2i (e-unit)-gridRound mag e q*pow2i (q-unit)).natAbs ≤
    (mag*pow2i (e-unit)-z*pow2i (q-unit)).natAbs := by
  by_cases hq : e < q
  · have h := gridRound_nearest mag e q hm hq
    have hs := scaled_nearest mag (gridRound mag e q) (pow2i (q-e)) (pow2i (e-unit))
      (IntervalProofs.pow2i_positive _ (by omega)) h z
    have hp := NumericalProofs.pow2i_add_nonnegative (q-e) (e-unit) (by omega) (by omega)
    rw [show q-e+(e-unit)=q-unit by omega] at hp
    rw [←hp] at hs
    exact hs
  · have he : e ≥ q := by omega
    have hg : gridRound mag e q=mag*pow2i (e-q) := by simp [gridRound,he]
    have hp := NumericalProofs.pow2i_add_nonnegative (e-q) (q-unit) (by omega) (by omega)
    rw [show e-q+(q-unit)=e-unit by omega] at hp
    have hz : mag*pow2i (e-unit)-gridRound mag e q*pow2i (q-unit)=0 := by
      rw [hg,hp]; grobner
    rw [hz]; simp

/-- An arbitrary finite candidate in this format has exponent at least emin
    (including subnormals at emin) and coefficient below 2^(mbits+1).
    The theorem quantifies over ALL such candidates, including candidates with
    another exponent; it is not merely nearest on the selected lattice. -/
theorem nearest_every_finite_candidate (f : fmt) (mag e unit m ex : Int)
    (hm : 1 ≤ mag) (hcap : mag < pow2i 16384) (hb : 0 ≤ f.mbits)
    (hu : unit ≤ e ∧ unit ≤ f.emin-f.mbits)
    (hex : f.emin ≤ ex) (hcoef : 0 ≤ m ∧ m < pow2i (f.mbits+1)) :
    let ee := if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e
    let q := ee-f.mbits
    (mag*pow2i (e-unit)-gridRound mag e q*pow2i (q-unit)).natAbs ≤
      (mag*pow2i (e-unit)-m*pow2i (ex-f.mbits-unit)).natAbs := by
  let ee := if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e
  let q := ee-f.mbits
  have hee : f.emin ≤ ee := by dsimp [ee]; split <;> omega
  have hqu : unit ≤ q := by dsimp [q]; omega
  have hnearest := grid_nearest_common_unit mag e q unit (by omega) ⟨hu.1,hqu⟩
  change (mag*pow2i (e-unit)-gridRound mag e q*pow2i (q-unit)).natAbs ≤
    (mag*pow2i (e-unit)-m*pow2i (ex-f.mbits-unit)).natAbs
  by_cases hx : ee ≤ ex
  · have hp := NumericalProofs.pow2i_add_nonnegative (ex-ee) (q-unit) (by omega) (by omega)
    rw [show ex-ee+(q-unit)=ex-f.mbits-unit by dsimp [q]; omega] at hp
    have hn := hnearest (m*pow2i (ex-ee))
    rw [hp]
    have he : m*(pow2i (ex-ee)*pow2i (q-unit))=(m*pow2i (ex-ee))*pow2i (q-unit) := by grobner
    rw [he]
    exact hn
  · have hnormal : ee=ilog2 mag+e := by
      have hn : ¬ilog2 mag+e<f.emin := by
        intro h
        have he : ee=f.emin := by simp [ee,h]
        omega
      simp [ee,hn]
    have hlog := BinadeProofs.ilog2_binade mag hm hcap
    have hinput := Int.mul_le_mul_of_nonneg_right hlog.2.1
      (show 0 ≤ pow2i (e-unit) from Int.le_of_lt (IntervalProofs.pow2i_positive _ (by omega)))
    have hp := NumericalProofs.pow2i_add_nonnegative (ilog2 mag) (e-unit) hlog.1 (by omega)
    rw [show ilog2 mag+(e-unit)=ee-unit by omega] at hp
    rw [←hp] at hinput
    have hB := NumericalProofs.pow2i_add_nonnegative f.mbits (q-unit) hb (by omega)
    rw [show f.mbits+(q-unit)=ee-unit by dsimp [q]; omega] at hB
    have hval := Int.mul_lt_mul_of_pos_right hcoef.2
      (IntervalProofs.pow2i_positive (ex-f.mbits-unit) (by omega))
    have hvpow := NumericalProofs.pow2i_add_nonnegative (f.mbits+1) (ex-f.mbits-unit) (by omega) (by omega)
    rw [show f.mbits+1+(ex-f.mbits-unit)=ex+1-unit by omega] at hvpow
    rw [←hvpow] at hval
    have hmon := pow2_mono (ex+1-unit) (ee-unit) (by omega) (by omega)
    apply below_boundary_distance _ _ _ (pow2i (ee-unit)) _ hnearest hinput (by omega)
      (pow2i f.mbits) hB

theorem negative_candidate_distance (a r d value : Int) (ha : 0 ≤ a) (hv : 0 ≤ value)
    (h : ∀ z : Int, (a-r*d).natAbs ≤ (a-z*d).natAbs) :
    (a-r*d).natAbs ≤ (a-(-value)).natAbs := by
  have hn := h 0
  simp only [Int.zero_mul,Int.sub_zero] at hn
  have hother : a.natAbs ≤ (a-(-value)).natAbs := by omega
  exact Nat.le_trans hn hother

theorem sign_isometry (a b : Int) : (-a-(-b)).natAbs=(a-b).natAbs := by
  have he : -a-(-b)= -(a-b) := by omega
  rw [he,Int.natAbs_neg]

theorem strict_below_boundary (a r d boundary value k : Int)
    (h : ∀ z : Int, (a-r*d).natAbs ≤ (a-z*d).natAbs)
    (hb : boundary ≤ a) (hv : value < boundary) (hk : boundary=k*d) :
    (a-r*d).natAbs < (a-value).natAbs := by
  have hn := h k
  rw [←hk] at hn
  have hd : (a-boundary).natAbs < (a-value).natAbs := by omega
  exact Nat.lt_of_le_of_lt hn hd

/-- Any deterministic nearest selector on an ordered common candidate set is
    monotone, even at midpoint ties. Both selected outputs belong to the set. -/
theorem nearest_selector_monotone (F : Int → Int) (x y : Int) (hxy : x ≤ y)
    (hx : (x-F x).natAbs ≤ (x-F y).natAbs)
    (hy : (y-F y).natAbs ≤ (y-F x).natAbs) : F x ≤ F y := by
  by_cases h : F x ≤ F y
  · exact h
  · have he : x=y := by omega
    subst y
    omega

/-- Equality of the rounded endpoint values extends throughout the enclosure
    once monotonicity of the deterministic nearest selector is established. -/
theorem endpoint_agreement (F : Int → Int) (lo hi x : Int)
    (hmono : ∀ a b : Int, a ≤ b → F a ≤ F b)
    (hx : lo ≤ x ∧ x ≤ hi) (he : F lo=F hi) : F x=F lo := by
  have hl := hmono lo x hx.1
  have hh := hmono x hi hx.2
  omega

/-- Source carry preserves the even significand decision, including an exact
    power-of-two carry. The admitted formats all have mbits>=2. -/
theorem normalization_tie_even (mbits m ee : Int) (hb : 1 ≤ mbits) (hm : m%2=0) :
    (normalizeCarry mbits m ee).2%2=0 := by
  by_cases hc : m=pow2i (mbits+1)
  · have hp := RoundingProofs.positive_power_even (mbits+1) (by omega)
    rw [show mbits+1-1=mbits by omega] at hp
    have hd : (pow2i (mbits+1)).tdiv 2=pow2i mbits := by
      rw [hp.1,Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0)]
    have he := RoundingProofs.positive_power_even mbits (by omega)
    simp only [normalizeCarry,hc,ite_true,hd]
    rw [he.1,Int.mul_emod]
    simp
  · simp [normalizeCarry,hc,hm]

/-- Clamping an over-range nearest output to the largest finite value is
    nearest among all finite candidates, whenever input itself exceeds it. -/
theorem finite_saturation_nearest (a ceiling z : Int) (ha : ceiling ≤ a) (hz : z ≤ ceiling) :
    (a-ceiling).natAbs ≤ (a-z).natAbs := by omega

/-- Actual round_iv compares two uses of the same source rounding function. -/
theorem round_iv_source (f : fmt) (lo hi sh : Int) :
    round_iv f lo hi sh=
      let F := fun v : Int => if v<0 then fround f true (-v) sh else fround f false v sh
      if F lo=F hi then some (F lo) else none := by
  simp [round_iv]

/-- Nonnegative RNE to zero includes the exact half-minimum-subnormal tie. -/
theorem rne_zero_iff (q rem half : Int) (hq : 0 ≤ q) (hh : 0 < half)
    (hr : 0 ≤ rem ∧ rem < 2*half) :
    NumericalProofs.rneChoice q rem half=0 ↔ q=0 ∧ rem ≤ half := by
  have hmod : (0:Int).tmod 2=0 := by decide
  by_cases hz : q=0
  · subst q; simp [NumericalProofs.rneChoice,hmod]
  · simp [NumericalProofs.rneChoice]; split <;> omega

/-- The coefficient at a binade carry is even. This states the full threshold
    decision before IEEE encoding, including exact midpoint rounding upward. -/
theorem rne_even_threshold (q rem half limit : Int) (hqn : 0 ≤ q) (hh : 0 < half)
    (hr : 0 ≤ rem ∧ rem < 2*half) (hl : limit%2=0) :
    limit ≤ NumericalProofs.rneChoice q rem half ↔
      limit ≤ q ∨ (q=limit-1 ∧ half ≤ rem) := by
  by_cases hq : q=limit-1
  · have hodd : q%2=1 := by omega
    have ht : q.tmod 2=1 := by rw [Int.tmod_eq_emod_of_nonneg (b:=2) hqn]; exact hodd
    by_cases hrh : half ≤ rem
    · have hc : half < rem ∨ (rem=half ∧ q.tmod 2=1) := by omega
      rw [NumericalProofs.rneChoice,if_pos hc]
      omega
    · have hc : ¬(half < rem ∨ (rem=half ∧ q.tmod 2=1)) := by omega
      rw [NumericalProofs.rneChoice,if_neg hc]
      omega
  · simp [NumericalProofs.rneChoice]
    split <;> omega

theorem nearest_above_ceiling (a chosen ceiling : Int) (hc : ceiling < chosen)
    (hn : (a-chosen).natAbs ≤ (a-ceiling).natAbs) : ceiling < a := by omega

theorem round_zero_sign (f : fmt) (neg : Bool) (ee : Int) (hb : 0 ≤ f.mbits) :
    packRounded f neg 0 ee=zero_bits f neg := by
  have hp := IntervalProofs.pow2i_positive (f.mbits+1) (by omega)
  simp [packRounded,show (0:Int)≠pow2i (f.mbits+1) by omega]

theorem signed_zeros_distinct :
    zero_bits FMT32 true≠zero_bits FMT32 false ∧
    zero_bits FMT16 true≠zero_bits FMT16 false ∧
    zero_bits FMT_BF16 true≠zero_bits FMT_BF16 false ∧
    zero_bits FMT_E4M3 true≠zero_bits FMT_E4M3 false ∧
    zero_bits FMT_E5M2 true≠zero_bits FMT_E5M2 false := by decide +kernel

theorem quotient_threshold (q rem half limit : Int) (hh : 0 < half)
    (hr : 0 ≤ rem ∧ rem < 2*half) :
    (limit ≤ q ∨ (q=limit-1 ∧ half ≤ rem)) ↔
      limit*(2*half)-half ≤ q*(2*half)+rem := by
  by_cases hq : limit ≤ q
  · have hm := Int.mul_le_mul_of_nonneg_right hq (by omega : 0 ≤ 2*half)
    omega
  · by_cases hq1 : q=limit-1
    · have he : q*(2*half)=limit*(2*half)-2*half := by rw [hq1]; grobner
      rw [he]; omega
    · have hq2 : q+1 ≤ limit-1 := by omega
      have hm := Int.mul_le_mul_of_nonneg_right hq2 (by omega : 0 ≤ 2*half)
      have he1 : (q+1)*(2*half)=q*(2*half)+2*half := by grobner
      have he2 : (limit-1)*(2*half)=limit*(2*half)-2*half := by grobner
      rw [he1,he2] at hm
      omega

/-- Actual production rounding crosses an even coefficient exactly at its
    half-ulp boundary, with equality choosing the even upper coefficient. -/
theorem actual_even_half_ulp_threshold (mag e q limit : Int) (hm : 0 ≤ mag)
    (hq : e < q) (hl : limit%2=0) :
    limit ≤ gridRound mag e q ↔
      limit*pow2i (q-e)-(pow2i (q-e)).tdiv 2 ≤ mag := by
  have hp := positive_power_even (q-e) (by omega)
  let half := pow2i (q-e-1)
  have hd : pow2i (q-e)=2*half := hp.1
  have hh : 0 < half := hp.2
  let lower := mag.tdiv (2*half)
  let rem := mag-lower*(2*half)
  have hlo : 0 ≤ lower := Int.tdiv_nonneg hm (by omega)
  have hdiv := Int.tdiv_eq_ediv_of_nonneg (b:=2*half) hm
  have he := Int.ediv_mul_add_emod mag (2*half)
  have hr0 := Int.emod_nonneg mag (by omega : 2*half≠0)
  have hr1 := Int.emod_lt_of_pos mag (by omega : 0 < 2*half)
  have hr : 0 ≤ rem ∧ rem < 2*half := by dsimp [rem,lower]; rw [hdiv]; omega
  have hg : gridRound mag e q=NumericalProofs.rneChoice lower rem half := by
    simp [gridRound,show ¬e ≥ q by omega,hd,NumericalProofs.rneChoice,lower,rem,
      Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0)]
  have hchoice := rne_even_threshold lower rem half limit hlo hh hr hl
  have hthreshold := quotient_threshold lower rem half limit hh hr
  rw [hg,hchoice,hthreshold,hd]
  have hhdiv : (2*half).tdiv 2=half := Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0)
  rw [hhdiv]
  have heq : lower*(2*half)+rem=mag := by dsimp [rem]; omega
  rw [heq]

theorem e4_saturation_literal (neg : Bool) :
    inf_bits FMT_E4M3 neg=get_slice_int 32 ((if neg then 128 else 0)+126) 0 := by
  cases neg <;> decide +kernel

theorem packed_lsb_parity (code : Int) :
    ((get_slice_int 32 code 0).toNat:Int)%2=code%2 := by
  simp only [get_slice_int,BitVec.extractLsb'_toNat,BitVec.toNat_ofInt]
  omega

end FormatNearestProofs
