-- Rational Taylor error invariants over the exact production recurrence.
-- Standard analytic series/remainder identities are a separate, explicit leg.
import Proofs.Intervals

open Sail Out Out.Functions
set_option autoImplicit false

namespace EnclosureProofs

/-- An exact integer identity used to transfer the local production remainder
    and the preceding rational-term error to the next rational term. -/
theorem term_error_decomposition (t u numerator y d scale k : Int) :
    u * (d * scale * k) - numerator * y =
      (u * (scale * k) - t * y) * d + (t * d - numerator) * y := by
  grind

/-- The actual two-divide recurrence preserves an error of less than two
    fixed-point units against the exact Taylor term, for every signed input and
    positive arbitrary-size denominator/scale/index. -/
theorem term_error_step (term numerator y : Int) (d scale k : Nat)
    (hd : 0 < d) (hs : 0 < scale) (hk : 1 ≤ k)
    (hy : 2 * y.natAbs ≤ scale)
    (hp : (term * (d : Int) - numerator).natAbs < 2 * d) :
    let next := ((term * y).tdiv (scale : Int)).tdiv (k : Int)
    (next * ((d * scale * k : Nat) : Int) - numerator * y).natAbs < 2 * (d * scale * k) := by
  let next := ((term * y).tdiv (scale : Int)).tdiv (k : Int)
  have hr0 := IntervalProofs.taylor_step_remainder_bound term y (scale : Int) (k : Int)
    (by omega) (by omega)
  have heq : next * ((scale : Int) * (k : Int)) - term * y =
      -(term * y - ((scale : Int) * (k : Int)) * next) := by grind
  have hr : (next * ((scale : Int) * (k : Int)) - term * y).natAbs < scale * k := by
    rw [heq, Int.natAbs_neg]
    simpa [next, Int.natAbs_mul] using hr0
  have hfirst := Nat.mul_lt_mul_of_pos_right hr hd
  have hsecond : (term * (d : Int) - numerator).natAbs * y.natAbs ≤ d * scale := by
    calc
      _ ≤ (2 * d) * y.natAbs := Nat.mul_le_mul_right _ (Nat.le_of_lt hp)
      _ = d * (2 * y.natAbs) := by ac_rfl
      _ ≤ d * scale := Nat.mul_le_mul_left d hy
  have hbudget : (scale * k) * d + d * scale ≤ 2 * (d * scale * k) := by
    have hlarge : d * scale ≤ d * scale * k := by
      simpa using Nat.mul_le_mul_left (d * scale) hk
    have hc : (scale * k) * d = d * scale * k := by ac_rfl
    rw [hc]
    omega
  have hpoly := term_error_decomposition term next numerator y (d : Int) (scale : Int) (k : Int)
  calc
    _ ≤ (next * ((scale : Int) * (k : Int)) - term * y).natAbs * d +
        (term * (d : Int) - numerator).natAbs * y.natAbs := by
      simp only [Int.natCast_mul]
      rw [hpoly]
      simpa [Int.natAbs_mul] using Int.natAbs_add_le
        ((next * ((scale : Int) * (k : Int)) - term * y) * (d : Int))
        ((term * (d : Int) - numerator) * y)
    _ < (scale * k) * d + d * scale := Nat.add_lt_add_of_lt_of_le hfirst hsecond
    _ ≤ 2 * (d * scale * k) := hbudget

theorem sum_error_step (sum next oldSum numerator y : Int) (d scale n : Nat)
    (hp : (sum * (d : Int) - oldSum).natAbs ≤ 2 * n * d)
    (ht : (next * ((d * scale * (n + 1) : Nat) : Int) - numerator * y).natAbs <
      2 * (d * scale * (n + 1))) :
    ((sum + next) * ((d * scale * (n + 1) : Nat) : Int) -
      (oldSum * ((scale * (n + 1) : Nat) : Int) + numerator * y)).natAbs <
        2 * (n + 1) * (d * scale * (n + 1)) := by
  let newD := d * scale * (n + 1)
  have hpoly : (sum + next) * (newD : Int) -
      (oldSum * ((scale * (n + 1) : Nat) : Int) + numerator * y) =
      (sum * (d : Int) - oldSum) * ((scale * (n + 1) : Nat) : Int) +
        (next * (newD : Int) - numerator * y) := by
    simp only [newD, Int.natCast_mul, Int.natCast_add]
    grind
  have hold := Nat.mul_le_mul_right (scale * (n + 1)) hp
  have hold' : (sum * (d : Int) - oldSum).natAbs * (scale * (n + 1)) ≤ 2 * n * newD := by
    calc
      _ ≤ (2 * n * d) * (scale * (n + 1)) := hold
      _ = 2 * n * newD := by simp only [newD]; ac_rfl
  calc
    _ ≤ (sum * (d : Int) - oldSum).natAbs * (scale * (n + 1)) +
        (next * (newD : Int) - numerator * y).natAbs := by
      rw [hpoly]
      simpa [Int.natAbs_mul] using Int.natAbs_add_le
        ((sum * (d : Int) - oldSum) * ((scale * (n + 1) : Nat) : Int))
        (next * (newD : Int) - numerator * y)
    _ < 2 * n * newD + 2 * newD := Nat.add_lt_add_of_le_of_lt hold' ht
    _ = 2 * (n + 1) * newD := by grind

/-- Exact rational Taylor data after term n. The represented partial sum is
    sumNumerator/denominator, and the last term numerator/denominator. -/
structure TaylorData where
  denominator : Nat
  numerator : Int
  sumNumerator : Int

def taylorData (scale : Nat) (y : Int) : Nat → TaylorData
  | 0 => ⟨1, scale, scale⟩
  | n + 1 =>
    let old := taylorData scale y n
    ⟨old.denominator * scale * (n + 1), old.numerator * y,
      old.sumNumerator * ((scale * (n + 1) : Nat) : Int) + old.numerator * y⟩

theorem taylor_denominator_positive (scale : Nat) (y : Int) (hs : 0 < scale) (n : Nat) :
    0 < (taylorData scale y n).denominator := by
  induction n with
  | zero => change 0 < 1; decide
  | succ n ih => exact Nat.mul_pos (Nat.mul_pos ih hs) (Nat.succ_pos n)

def TaylorInvariant (scale : Nat) (y : Int) (state : IntervalProofs.ExpState) : Prop :=
  ∃ n : Nat, state.2.1 = ((n + 1 : Nat) : Int) ∧ state.1 = 2 * ((n + 1 : Nat) : Int) ∧
    (state.2.2.2 * ((taylorData scale y n).denominator : Int) -
      (taylorData scale y n).numerator).natAbs < 2 * (taylorData scale y n).denominator ∧
    (state.2.2.1 * ((taylorData scale y n).denominator : Int) -
      (taylorData scale y n).sumNumerator).natAbs ≤ 2 * n * (taylorData scale y n).denominator

theorem taylor_invariant_initial (scale : Nat) (y : Int) :
    TaylorInvariant scale y (2, 1, scale, scale) := by
  refine ⟨0, rfl, rfl, ?_, ?_⟩ <;> simp [taylorData]

theorem taylor_invariant_step (scale : Nat) (y : Int) (hs : 0 < scale)
    (hy : 2 * y.natAbs ≤ scale) (state : IntervalProofs.ExpState)
    (hi : TaylorInvariant scale y state) :
    TaylorInvariant scale y (IntervalProofs.expStep (scale : Int) y state) := by
  rcases state with ⟨err, k, sum, term⟩
  by_cases hz : term = 0
  · simpa [IntervalProofs.expStep, hz] using hi
  · have hg : term > 0 ∨ term < 0 := by omega
    rcases hi with ⟨n, hk, he, ht, hsum⟩
    dsimp at hk he ht hsum
    have hd := taylor_denominator_positive scale y hs n
    have ht' := term_error_step term (taylorData scale y n).numerator y
      (taylorData scale y n).denominator scale (n + 1) hd hs (by omega) hy ht
    have hsum' := sum_error_step sum (((term * y).tdiv (scale : Int)).tdiv ((n + 1 : Nat) : Int))
      (taylorData scale y n).sumNumerator (taylorData scale y n).numerator y
      (taylorData scale y n).denominator scale n hsum ht'
    refine ⟨n + 1, ?_, ?_, ?_, ?_⟩
    · simp [IntervalProofs.expStep, hg]
      omega
    · simp [IntervalProofs.expStep, hg]
      omega
    · simpa [IntervalProofs.expStep, hg, taylorData, hk] using ht'
    · exact Nat.le_of_lt (by simpa [IntervalProofs.expStep, hg, taylorData, hk] using hsum')

theorem repeat_taylor_invariant (scale : Nat) (y : Int) (hs : 0 < scale)
    (hy : 2 * y.natAbs ≤ scale) (steps : Nat) (state : IntervalProofs.ExpState)
    (hi : TaylorInvariant scale y state) :
    TaylorInvariant scale y (IntervalProofs.repeatStep (IntervalProofs.expStep (scale : Int) y) steps state) := by
  induction steps generalizing state with
  | zero => exact hi
  | succ steps ih =>
    exact ih (IntervalProofs.expStep (scale : Int) y state)
      (taylor_invariant_step scale y hs hy state hi)

theorem exp_state_taylor_invariant (y : Int) (p : Nat) (hy : 2 * y.natAbs ≤ 2 ^ p) :
    TaylorInvariant (2 ^ p) y (IntervalProofs.expState y (p : Int)) := by
  have hone : pow2i (p : Int) = ((2 ^ p : Nat) : Int) := by
    rw [NumericalProofs.pow2i_nonnegative (p : Int) (by omega)]
    simp
  rw [IntervalProofs.exp_state_iteration, hone]
  exact repeat_taylor_invariant (2 ^ p) y (Nat.pow_pos (by decide)) hy _ _
    (taylor_invariant_initial _ _)

theorem exp_step_index_lower (state : IntervalProofs.ExpState) (one y lower : Int)
    (h : lower ≤ state.2.1) : lower ≤ (IntervalProofs.expStep one y state).2.1 := by
  rcases state with ⟨err, k, sum, term⟩
  by_cases hz : term = 0
  · simpa [IntervalProofs.expStep, hz] using h
  · have hg : term > 0 ∨ term < 0 := by omega
    simp [IntervalProofs.expStep, hg]
    dsimp at h
    omega

theorem repeat_index_lower (state : IntervalProofs.ExpState) (one y lower : Int) (steps : Nat)
    (h : lower ≤ state.2.1) :
    lower ≤ (IntervalProofs.repeatStep (IntervalProofs.expStep one y) steps state).2.1 := by
  induction steps generalizing state with
  | zero => exact h
  | succ steps ih => exact ih _ (exp_step_index_lower state one y lower h)

theorem exp_state_index_at_least_two (y : Int) (p : Nat) :
    2 ≤ (IntervalProofs.expState y (p : Int)).2.1 := by
  have hs := IntervalProofs.pow2i_positive (p : Int) (by omega)
  rw [IntervalProofs.exp_state_iteration]
  have hcount : ((p : Int) + 1).toNat = p + 1 := by omega
  rw [hcount, IntervalProofs.repeatStep]
  apply repeat_index_lower
  simp [IntervalProofs.expStep, show pow2i (p : Int) > 0 by omega]

theorem interval_margin_from_error (sum err numerator : Int) (d n : Nat)
    (he : err = 2 * ((n + 1 : Nat) : Int))
    (hs : (sum * (d : Int) - numerator).natAbs ≤ 2 * n * d) :
    (sum - err + 2) * (d : Int) ≤ numerator ∧ numerator ≤ (sum + err - 2) * (d : Int) := by
  let delta := sum * (d : Int) - numerator
  have hu : delta ≤ (delta.natAbs : Int) := Int.le_natAbs
  have hl' : -delta ≤ ((-delta).natAbs : Int) := Int.le_natAbs
  simp only [Int.natAbs_neg] at hl'
  have hb : (delta.natAbs : Int) ≤ 2 * (n : Int) * (d : Int) := Int.ofNat_le.mpr hs
  change err = 2 * ((n : Int) + 1) at he
  have h1 : (sum - err + 2) * (d : Int) = sum * (d : Int) - 2 * (n : Int) * (d : Int) := by
    rw [he]
    grind
  have h2 : (sum + err - 2) * (d : Int) = sum * (d : Int) + 2 * (n : Int) * (d : Int) := by
    rw [he]
    grind
  rw [h1, h2]
  dsimp only [delta] at hu hl' hb
  omega

/-- Complete integer-side certificate for the actual exp_small result. The
    represented exact Taylor partial sum is at least two fixed-point units
    inside the returned interval, and the last exact term has magnitude <2.
    The only remaining mathematical bridge is the standard analytic Taylor
    identity/geometric tail for |Y/2^p|<=1/2 (not assumed as a program property). -/
theorem exp_small_partial_sum_certificate (y : Int) (p : Nat)
    (hy : 2 * y.natAbs ≤ 2 ^ p) :
    ∃ n : Nat, 1 ≤ n ∧
      let data := taylorData (2 ^ p) y n
      0 < data.denominator ∧ data.numerator.natAbs < 2 * data.denominator ∧
      ((exp_small y (p : Int)).1 + 2) * (data.denominator : Int) ≤ data.sumNumerator ∧
        data.sumNumerator ≤ ((exp_small y (p : Int)).2 - 2) * (data.denominator : Int) := by
  let state := IntervalProofs.expState y (p : Int)
  have hi := exp_state_taylor_invariant y p hy
  rcases hi with ⟨n, hk, he, ht, hsum⟩
  have hindex := exp_state_index_at_least_two y p
  have hn : 1 ≤ n := by omega
  have hz : state.2.2.2 = 0 := by
    by_cases hp : p = 0
    · subst p
      exact IntervalProofs.exp_small_term_zero_p0 y (by simpa using hy)
    · have hp1 : 1 ≤ p := by omega
      apply IntervalProofs.exp_small_term_zero y p hp1
      have heq : p - 1 + 1 = p := by omega
      have hpow := Nat.pow_succ 2 (p - 1)
      change 2 ^ (p - 1 + 1) = 2 ^ (p - 1) * 2 at hpow
      rw [heq] at hpow
      rw [hpow] at hy
      omega
  have hterm : (taylorData (2 ^ p) y n).numerator.natAbs <
      2 * (taylorData (2 ^ p) y n).denominator := by
    change (state.2.2.2 * _ - _).natAbs < _ at ht
    simpa [hz] using ht
  have hsource : exp_small y (p : Int) = (state.2.2.1 - state.1, state.2.2.1 + state.1) :=
    IntervalProofs.exp_small_production_loop y (p : Int)
  have hpositive := taylor_denominator_positive (2 ^ p) y (Nat.pow_pos (by decide)) n
  have hmargin := interval_margin_from_error state.2.2.1 state.1
    (taylorData (2 ^ p) y n).sumNumerator (taylorData (2 ^ p) y n).denominator n he hsum
  refine ⟨n, hn, hpositive, hterm, ?_, ?_⟩
  · rw [hsource]
    exact hmargin.1
  · rw [hsource]
    exact hmargin.2

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => factorial n * (n + 1)

theorem taylor_denominator_formula (scale : Nat) (y : Int) (n : Nat) :
    (taylorData scale y n).denominator = scale ^ n * factorial n := by
  induction n with
  | zero => simp [taylorData, factorial]
  | succ n ih => simp [taylorData, factorial, ih, Nat.pow_succ]; ac_rfl

theorem taylor_numerator_formula (scale : Nat) (y : Int) (n : Nat) :
    (taylorData scale y n).numerator = (scale : Int) * y ^ n := by
  induction n with
  | zero => simp [taylorData]
  | succ n ih => simp [taylorData, ih, Int.pow_succ, Int.mul_assoc]

end EnclosureProofs
