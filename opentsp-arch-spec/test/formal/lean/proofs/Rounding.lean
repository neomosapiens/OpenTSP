-- Generic nearest-even lattice arithmetic tied to the actual fround pipeline.
-- Format-grid selection and complete bit packing remain separate obligations.
import Proofs.Enclosure
open Sail Out Out.Functions
set_option autoImplicit false
namespace RoundingProofs

def gridRound (mag e q : Int) : Int :=
  if ((e ≥b q) : Bool) then mag *i pow2i (e -i q)
  else
    let d := pow2i (q -i e)
    let lo := Int.tdiv mag d
    let rem := mag -i (lo *i d)
    let half := Int.tdiv d 2
    if (((rem >b half) || ((rem == half) && (((Int.tmod lo 2) == 1) : Bool))) : Bool)
    then lo +i 1 else lo

def packRounded (f : fmt) (neg : Bool) (m ee : Int) : BitVec 32 :=
  let (ex, m) : (Int × Int) :=
    if ((m == (pow2i (f.mbits +i 1))) : Bool)
    then (ee +i 1, Int.tdiv m 2) else (ee,m)
  if ((m == 0) : Bool) then zero_bits f neg
  else if ((m <b (pow2i f.mbits)) : Bool) then
    get_slice_int 32 ((if neg then pow2i (f.width -i 1) else 0) +i m) 0
  else if ((f.ieee && ((ex +i f.bias) ≥b ((pow2i f.ebits) -i 1))) : Bool)
    then inf_bits f neg
  else if (((Out.Functions.not f.ieee) && (((ex +i f.bias) >b ((pow2i f.ebits) -i 1)) ||
      (((ex +i f.bias) == ((pow2i f.ebits) -i 1)) &&
      ((m -i (pow2i f.mbits)) == ((pow2i f.mbits) -i 1))))) : Bool)
    then inf_bits f neg
  else get_slice_int 32 (((if neg then pow2i (f.width -i 1) else 0) +i
    ((ex +i f.bias) *i (pow2i f.mbits))) +i (m -i (pow2i f.mbits))) 0

theorem fround_production_pipeline (f : fmt) (neg : Bool) (mag e : Int) :
    fround f neg mag e =
      if ((mag == 0) : Bool) then zero_bits f neg else
        let ee := if (((ilog2 mag +i e) <b f.emin) : Bool) then f.emin else ilog2 mag +i e
        packRounded f neg (gridRound mag e (ee -i f.mbits)) ee := by
  rfl

/-- A half-spacing error bound makes the chosen lattice point nearest among
    every integer point, not only an adjacent candidate or a sample. -/
theorem nearest_from_half (a r h : Int) (hh : 0<h)
    (hb : -h ≤ a-r*(2*h) ∧ a-r*(2*h) ≤ h) (z : Int) :
    (a-r*(2*h)).natAbs ≤ (a-z*(2*h)).natAbs := by
  have hchosen : ((a-r*(2*h)).natAbs:Int) ≤ h := by omega
  by_cases he : z=r
  · subst z; omega
  by_cases hz : z<r
  · have hm := Int.mul_le_mul_of_nonneg_right (show z+1 ≤ r by omega) (by omega : 0 ≤ 2*h)
    have hr : (z+1)*(2*h)=z*(2*h)+2*h := by grind
    have hother : h ≤ a-z*(2*h) := by omega
    have habs : a-z*(2*h) ≤ ((a-z*(2*h)).natAbs:Int) := Int.le_natAbs
    omega
  · have hm := Int.mul_le_mul_of_nonneg_right (show r+1 ≤ z by omega) (by omega : 0 ≤ 2*h)
    have hr : (r+1)*(2*h)=r*(2*h)+2*h := by grind
    have hother : a-z*(2*h) ≤ -h := by omega
    have habs : -(a-z*(2*h)) ≤ ((a-z*(2*h)).natAbs:Int) := by
      have hx : -(a-z*(2*h)) ≤ ((-(a-z*(2*h))).natAbs:Int) := Int.le_natAbs
      simpa using hx
    omega

theorem positive_power_even (k : Int) (hk : 0<k) :
    pow2i k = 2*pow2i (k-1) ∧ 0<pow2i (k-1) := by
  have h := NumericalProofs.pow2i_add_nonnegative 1 (k-1) (by decide) (by omega)
  have he : 1+(k-1)=k := by omega
  rw [he] at h
  have htwo : pow2i 1=2 := by decide
  rw [htwo] at h
  exact ⟨h,IntervalProofs.pow2i_positive (k-1) (by omega)⟩

theorem gridRound_nearest (mag e q : Int) (hm : 0≤mag) (hq : e<q) (z : Int) :
    (mag-gridRound mag e q*pow2i (q-e)).natAbs ≤
      (mag-z*pow2i (q-e)).natAbs := by
  have hp := positive_power_even (q-e) (by omega)
  let h := pow2i (q-e-1)
  have he : pow2i (q-e)=2*h := hp.1
  have hh : 0<h := hp.2
  have ht := Int.tdiv_eq_ediv_of_nonneg (b:=2*h) hm
  have hd := Int.ediv_mul_add_emod mag (2*h)
  have hr0 := Int.emod_nonneg mag (by omega : 2*h≠0)
  have hr1 := Int.emod_lt_of_pos mag (by omega : 0<2*h)
  have hr : mag-mag.tdiv (2*h)*(2*h)=mag%(2*h) := by rw [ht]; omega
  have hc := NumericalProofs.rne_choice_half_ulp (mag.tdiv (2*h)) (mag%(2*h)) h hh hr0 hr1
  have hgrid : gridRound mag e q = NumericalProofs.rneChoice (mag.tdiv (2*h)) (mag%(2*h)) h := by
    simp only [gridRound, show ¬e≥q by omega, ite_false, he, Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0)]
    rw [hr]
    simp [NumericalProofs.rneChoice]
  rw [he,hgrid]
  apply nearest_from_half _ _ h hh _ z
  have hsum : mag.tdiv (2*h)*(2*h)+mag%(2*h)=mag := by rw [ht]; exact hd
  rw [hsum] at hc
  exact hc

/-- Exactly halfway cases choose an even lattice index. -/
theorem rne_choice_tie_even (q h : Int) (hq : 0≤q) :
    (NumericalProofs.rneChoice q h h)%2=0 := by
  have hmod0 := Int.emod_nonneg q (by decide : (2:Int)≠0)
  have hmod1 := Int.emod_lt_of_pos q (by decide : (0:Int)<2)
  have ht := Int.tmod_eq_emod_of_nonneg (b:=2) hq
  by_cases ho : q%2=1
  · have hh : (q+1)%2=0 := by rw [Int.add_emod]; simp [ho]
    simp [NumericalProofs.rneChoice,ht,ho,hh]
  · have he : q%2=0 := by omega
    simp [NumericalProofs.rneChoice,ht,he]

theorem gridRound_tie_even (mag e q : Int) (hm : 0≤mag) (hq : e<q)
    (htie : mag-mag.tdiv (pow2i (q-e))*pow2i (q-e) = (pow2i (q-e)).tdiv 2) :
    gridRound mag e q %2=0 := by
  have hd := IntervalProofs.pow2i_positive (q-e) (by omega)
  have hlo := Int.tdiv_nonneg hm (by omega : 0≤pow2i (q-e))
  have hc := rne_choice_tie_even (mag.tdiv (pow2i (q-e))) ((pow2i (q-e)).tdiv 2) hlo
  simpa [gridRound,show ¬e≥q by omega,htie,NumericalProofs.rneChoice] using hc

/-- The source carry/renormalization divides an exactly even significand by2;
    it therefore preserves the represented dyadic when the exponent rises1. -/
theorem normalization_carry_exact (mbits : Int) (hb : 0≤mbits) :
    (pow2i (mbits+1)).tdiv 2 * 2 = pow2i (mbits+1) := by
  have hp := positive_power_even (mbits+1) (by omega)
  rw [show mbits+1-1=mbits by omega] at hp
  rw [hp.1,Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0),Int.mul_comm]

/-- The actual production bit result is the declared pack of a lattice point
    nearest to every integer alternative; this does not yet identify pack with
    the full independent IEEE encoding/overflow definition. -/
theorem fround_nearest_witness (f : fmt) (neg : Bool) (mag e : Int)
    (hm : 0<mag)
    (hq : e < (if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e)-f.mbits) :
    let ee := if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e
    let q := ee-f.mbits
    let r := gridRound mag e q
    fround f neg mag e = packRounded f neg r ee ∧
      ∀ z : Int, (mag-r*pow2i (q-e)).natAbs ≤ (mag-z*pow2i (q-e)).natAbs := by
  dsimp only
  constructor
  · rw [fround_production_pipeline]
    simp [show mag≠0 by omega]
  · intro z
    exact gridRound_nearest _ _ _ (by omega) hq z

end RoundingProofs
