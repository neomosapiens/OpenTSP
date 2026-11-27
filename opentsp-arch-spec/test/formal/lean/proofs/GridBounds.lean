-- Actual rounding-grid range, normalization and overflow obligations.
import Proofs.Binade
import Proofs.Rounding
open Sail Out Out.Functions
set_option autoImplicit false
namespace GridBoundProofs
open RoundingProofs

theorem pow2_mono (a b : Int) (ha : 0 ≤ a) (hab : a ≤ b) : pow2i a ≤ pow2i b := by
  have hp := NumericalProofs.pow2i_add_nonnegative a (b-a) ha (by omega)
  have he : a+(b-a)=b := by omega
  rw [he] at hp
  have hf := IntervalProofs.pow2i_positive (b-a) (by omega)
  have hs := IntervalProofs.pow2i_positive a ha
  have hm := Int.mul_le_mul_of_nonneg_left (show 1 ≤ pow2i (b-a) by omega) (by omega : 0 ≤ pow2i a)
  rw [hp]
  simpa using hm

theorem grid_division_bounds (mag e q : Int) (hm : 0 ≤ mag) (hq : e < q) :
    0 ≤ gridRound mag e q ∧ mag.tdiv (pow2i (q-e)) ≤ gridRound mag e q ∧
      gridRound mag e q ≤ mag.tdiv (pow2i (q-e))+1 := by
  have hd := IntervalProofs.pow2i_positive (q-e) (by omega)
  have hl := Int.tdiv_nonneg hm (by omega : 0 ≤ pow2i (q-e))
  simp [gridRound,show ¬e ≥ q by omega]
  split <;> omega

theorem grid_range (mag p e emin mbits : Int)
    (hm : 1 ≤ mag) (hp : 0 ≤ p) (hb : 0 ≤ mbits)
    (hlo : pow2i p ≤ mag) (hhi : mag < pow2i (p+1)) :
    let ee := max (p+e) emin
    let q := ee-mbits
    0 ≤ gridRound mag e q ∧ gridRound mag e q ≤ pow2i (mbits+1) ∧
      (ee=p+e → pow2i mbits ≤ gridRound mag e q) := by
  let ee := max (p+e) emin
  let q := ee-mbits
  change 0 ≤ gridRound mag e q ∧ gridRound mag e q ≤ pow2i (mbits+1) ∧
      (ee=p+e → pow2i mbits ≤ gridRound mag e q)
  have hee : p+e ≤ ee := by dsimp [ee]; omega
  have hq : q=ee-mbits := rfl
  by_cases hd : e < q
  · have hg := grid_division_bounds mag e q (by omega) hd
    have hD := IntervalProofs.pow2i_positive (q-e) (by omega)
    have hpow := NumericalProofs.pow2i_add_nonnegative (q-e) (mbits+1) (by omega) (by omega)
    have hmono := pow2_mono (p+1) ((q-e)+(mbits+1)) (by omega) (by omega)
    have hbig : mag < pow2i (mbits+1)*pow2i (q-e) := by
      rw [Int.mul_comm,←hpow]
      omega
    have hdiv : mag.tdiv (pow2i (q-e)) < pow2i (mbits+1) := by
      rw [Int.tdiv_eq_ediv_of_nonneg (by omega : 0 ≤ mag)]
      exact Int.ediv_lt_of_lt_mul hD hbig
    refine ⟨hg.1,by omega,?_⟩
    intro hn
    have hfactor := NumericalProofs.pow2i_add_nonnegative mbits (q-e) hb (by omega)
    have he : mbits+(q-e)=p := by omega
    rw [he] at hfactor
    have hlo' : pow2i mbits*pow2i (q-e) ≤ mag := by rw [←hfactor]; exact hlo
    have hdivlo : pow2i mbits ≤ mag.tdiv (pow2i (q-e)) := by
      rw [Int.tdiv_eq_ediv_of_nonneg (by omega : 0 ≤ mag)]
      exact Int.le_ediv_of_mul_le hD hlo'
    omega
  · have hge : e ≥ q := by omega
    have hs := IntervalProofs.pow2i_positive (e-q) (by omega)
    have hgrid : gridRound mag e q=mag*pow2i (e-q) := by simp [gridRound,hge]
    have hpow := NumericalProofs.pow2i_add_nonnegative (p+1) (e-q) (by omega) (by omega)
    have hmono := pow2_mono ((p+1)+(e-q)) (mbits+1) (by omega) (by omega)
    have hprod := Int.mul_lt_mul_of_pos_right hhi hs
    have hupper : mag*pow2i (e-q) < pow2i (mbits+1) := by rw [hpow] at hmono; omega
    rw [hgrid]
    refine ⟨Int.mul_nonneg (by omega) (by omega),by omega,?_⟩
    intro hn
    have hfactor := NumericalProofs.pow2i_add_nonnegative p (e-q) hp (by omega)
    have he : p+(e-q)=mbits := by omega
    rw [he] at hfactor
    have hmul := Int.mul_le_mul_of_nonneg_right hlo (by omega : 0 ≤ pow2i (e-q))
    rw [←hfactor] at hmul
    exact hmul

theorem actual_grid_range (f : fmt) (mag e : Int) (hm : 1 ≤ mag)
    (hcap : mag < pow2i 16384) (hb : 0 ≤ f.mbits) :
    let ee := if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e
    let q := ee-f.mbits
    0 ≤ gridRound mag e q ∧ gridRound mag e q ≤ pow2i (f.mbits+1) ∧
      (ee=ilog2 mag+e → pow2i f.mbits ≤ gridRound mag e q) := by
  have hp := BinadeProofs.ilog2_binade mag hm hcap
  have he : (if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e)=
      max (ilog2 mag+e) f.emin := by omega
  rw [he]
  exact grid_range mag (ilog2 mag) e f.emin f.mbits hm hp.1 hb hp.2.1 hp.2.2


theorem binade_unique (mag p b : Int) (hp : 0 ≤ p) (hb : 0 ≤ b)
    (hpL : pow2i p ≤ mag) (hpU : mag < pow2i (p+1))
    (hbL : pow2i b ≤ mag) (hbU : mag < pow2i (b+1)) : p=b := by
  by_cases h : p=b
  · exact h
  · by_cases hlt : p < b
    · have hm := pow2_mono (p+1) b (by omega) (by omega)
      omega
    · have hm := pow2_mono (b+1) p (by omega) (by omega)
      omega

def packCanonical (f : fmt) (neg : Bool) (m ex : Int) : BitVec 32 :=
  if m=0 then zero_bits f neg
  else if m < pow2i f.mbits then
    get_slice_int 32 ((if neg then pow2i (f.width-1) else 0)+m) 0
  else if f.ieee && decide (ex+f.bias ≥ pow2i f.ebits-1) then inf_bits f neg
  else if !f.ieee && decide (ex+f.bias > pow2i f.ebits-1 ∨
      (ex+f.bias=pow2i f.ebits-1 ∧ m-pow2i f.mbits=pow2i f.mbits-1)) then inf_bits f neg
  else get_slice_int 32 ((if neg then pow2i (f.width-1) else 0)+
      (ex+f.bias)*pow2i f.mbits+(m-pow2i f.mbits)) 0

theorem pack_no_carry (f : fmt) (neg : Bool) (m ex : Int)
    (hm : m < pow2i (f.mbits+1)) :
    packRounded f neg m ex=packCanonical f neg m ex := by
  simp [packRounded,packCanonical,Out.Functions.not,show m≠pow2i (f.mbits+1) by omega]

theorem canonical_normal_reduction (f : fmt) (neg : Bool) (m ex : Int)
    (hb : 0 ≤ f.mbits) (hbU : f.mbits+1 ≤ 16384) (he : f.emin ≤ ex)
    (hmL : pow2i f.mbits ≤ m) (hmU : m < pow2i (f.mbits+1)) :
    fround f neg m (ex-f.mbits)=packCanonical f neg m ex := by
  have hL := IntervalProofs.pow2i_positive f.mbits hb
  have hcap := pow2_mono (f.mbits+1) 16384 (by omega) hbU
  have hp := BinadeProofs.ilog2_binade m (by omega) (by omega)
  have hlog := binade_unique m (ilog2 m) f.mbits hp.1 hb hp.2.1 hp.2.2 hmL hmU
  have hee : (if ilog2 m+(ex-f.mbits)<f.emin then f.emin else ilog2 m+(ex-f.mbits))=ex := by rw [hlog]; omega
  rw [fround_production_pipeline]
  simp [show m≠0 by omega,hee]
  have hq : ex-(f.mbits)=ex-f.mbits := rfl
  have hzero : pow2i 0=1 := by decide
  have hg : gridRound m (ex-f.mbits) (ex-f.mbits)=m := by simp [gridRound,hzero]
  rw [hg]
  exact pack_no_carry f neg m ex hmU

theorem canonical_subnormal_reduction (f : fmt) (neg : Bool) (m : Int)
    (hb : 0 ≤ f.mbits) (hbU : f.mbits+1 ≤ 16384)
    (hm : 0 ≤ m) (hmU : m < pow2i f.mbits) :
    fround f neg m (f.emin-f.mbits)=packCanonical f neg m f.emin := by
  by_cases hz : m=0
  · simp [fround_production_pipeline,packCanonical,hz]
  · have hcap := pow2_mono f.mbits 16384 hb (by omega)
    have hp := BinadeProofs.ilog2_binade m (by omega) (by omega)
    have hlog : ilog2 m < f.mbits := by
      by_cases h : ilog2 m < f.mbits
      · exact h
      · have hmon := pow2_mono f.mbits (ilog2 m) hb (by omega)
        omega
    have hee : (if ilog2 m+(f.emin-f.mbits)<f.emin then f.emin else ilog2 m+(f.emin-f.mbits))=f.emin := by omega
    rw [fround_production_pipeline]
    simp [hz,hee]
    have hzero : pow2i 0=1 := by decide
    have hg : gridRound m (f.emin-f.mbits) (f.emin-f.mbits)=m := by simp [gridRound,hzero]
    rw [hg]
    apply pack_no_carry
    have hmon := pow2_mono f.mbits (f.mbits+1) hb (by omega)
    omega


def normalizeCarry (mbits m ee : Int) : Int × Int :=
  if m=pow2i (mbits+1) then (ee+1,m.tdiv 2) else (ee,m)

theorem carry_source (f : fmt) (neg : Bool) (m ee : Int) :
    packRounded f neg m ee=
      let (ex,r) := normalizeCarry f.mbits m ee
      packCanonical f neg r ex := by
  simp [packRounded,normalizeCarry,packCanonical,Out.Functions.not]

theorem normalized_canonical_range (mbits m ee emin : Int) (hb : 0 ≤ mbits)
    (hm : 0 ≤ m ∧ m ≤ pow2i (mbits+1)) (he : emin ≤ ee)
    (hsub : m < pow2i mbits → ee=emin) :
    let pair := normalizeCarry mbits m ee
    0 ≤ pair.2 ∧ pair.2 < pow2i (mbits+1) ∧ emin ≤ pair.1 ∧
      (pair.2 < pow2i mbits → pair.1=emin) ∧
      m=pair.2*pow2i (pair.1-ee) := by
  have hL := IntervalProofs.pow2i_positive mbits hb
  have hp := RoundingProofs.positive_power_even (mbits+1) (by omega)
  rw [show mbits+1-1=mbits by omega] at hp
  have hzero : pow2i 0=1 := by decide
  have hone : pow2i 1=2 := by decide
  by_cases hc : m=pow2i (mbits+1)
  · have hdiv : (pow2i (mbits+1)).tdiv 2=pow2i mbits := by
      rw [hp.1,Int.mul_tdiv_cancel_left _ (by decide : (2:Int)≠0)]
    simp only [normalizeCarry,hc,ite_true,hdiv]
    simp only [show ee+1-ee=1 by omega,hone]
    refine ⟨by omega,by omega,by omega,?_,?_⟩
    · intro h; omega
    · rw [hp.1]; grobner
  · simp only [normalizeCarry,hc,ite_false]
    simp only [Int.sub_self,hzero,Int.mul_one]
    exact ⟨hm.1,by omega,he,hsub,True.intro⟩

theorem actual_normalized_range (f : fmt) (mag e : Int) (hm : 1 ≤ mag)
    (hcap : mag < pow2i 16384) (hb : 0 ≤ f.mbits) :
    let ee := if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e
    let m := gridRound mag e (ee-f.mbits)
    let pair := normalizeCarry f.mbits m ee
    0 ≤ pair.2 ∧ pair.2 < pow2i (f.mbits+1) ∧ f.emin ≤ pair.1 ∧
      (pair.2 < pow2i f.mbits → pair.1=f.emin) ∧
      m=pair.2*pow2i (pair.1-ee) := by
  dsimp only
  have hg := actual_grid_range f mag e hm hcap hb
  dsimp only at hg
  apply normalized_canonical_range _ _ _ _ hb ⟨hg.1,hg.2.1⟩
  · split <;> omega
  · intro h
    by_cases hbranch : ilog2 mag+e<f.emin
    · simp [hbranch]
    · have hee : (if ilog2 mag+e<f.emin then f.emin else ilog2 mag+e)=ilog2 mag+e := by simp [hbranch]
      have hl := hg.2.2 hee
      omega

theorem canonical_ieee_overflow (f : fmt) (neg : Bool) (m ex : Int)
    (hb : 0 ≤ f.mbits) (hm : pow2i f.mbits ≤ m) (hf : f.ieee=true)
    (hx : pow2i f.ebits-1 ≤ ex+f.bias) :
    packCanonical f neg m ex=inf_bits f neg := by
  have hp := IntervalProofs.pow2i_positive f.mbits hb
  simp [packCanonical,hf,show m≠0 by omega,show ¬m<pow2i f.mbits by omega,hx]

theorem canonical_finite_only_saturation (f : fmt) (neg : Bool) (m ex : Int)
    (hb : 0 ≤ f.mbits) (hm : pow2i f.mbits ≤ m) (hf : f.ieee=false)
    (hx : pow2i f.ebits-1 < ex+f.bias ∨
      (ex+f.bias=pow2i f.ebits-1 ∧ m-pow2i f.mbits=pow2i f.mbits-1)) :
    packCanonical f neg m ex=inf_bits f neg := by
  have hp := IntervalProofs.pow2i_positive f.mbits hb
  simp [packCanonical,hf,show m≠0 by omega,show ¬m<pow2i f.mbits by omega,hx]

end GridBoundProofs
