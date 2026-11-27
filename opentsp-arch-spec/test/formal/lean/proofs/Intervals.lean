-- Exact production fixed-point enclosure arithmetic. Analytic exp/log/sqrt
-- remainder facts and the global Ziv-cap certificate are separate obligations.
import Proofs.Numerics

open Sail Out Out.Functions
set_option autoImplicit false

namespace IntervalProofs

/-- The production truncating-division correction is mathematical floor for
    every signed numerator and positive denominator. -/
theorem idiv_floor_correct (a b : Int) (hb : 0 < b) : idiv_floor a b = a / b := by
  have hmod0 := Int.emod_nonneg a (by omega : b ≠ 0)
  have hmodlt := Int.emod_lt_of_pos a hb
  have hdecomp := Int.ediv_mul_add_emod a b
  by_cases h : 0 ≤ a ∨ b ∣ a
  · have ht : a.tdiv b = a / b := by simp [Int.tdiv_eq_ediv, h]
    have hc : a / b * b ≤ a := Int.ediv_mul_le a (by omega)
    simp [idiv_floor, ht, show ¬a / b * b > a by omega]
  · have hnotdvd : ¬b ∣ a := by omega
    have hrem : a % b ≠ 0 := by
      intro hz
      have hd : b ∣ a := Int.dvd_of_emod_eq_zero hz
      contradiction
    have ht : a.tdiv b = a / b + 1 := by
      simp [Int.tdiv_eq_ediv, h, Int.sign_eq_one_of_pos hb]
    have hc : a < (a / b + 1) * b := by
      rw [Int.add_mul, Int.one_mul]
      omega
    simp [idiv_floor, ht, hc]

theorem floor_bounds (a b : Int) (hb : 0 < b) :
    idiv_floor a b * b ≤ a ∧ a < (idiv_floor a b + 1) * b := by
  rw [idiv_floor_correct a b hb]
  have hmod0 := Int.emod_nonneg a (by omega : b ≠ 0)
  have hmodlt := Int.emod_lt_of_pos a hb
  have hdecomp := Int.ediv_mul_add_emod a b
  rw [Int.add_mul, Int.one_mul]
  omega

theorem ceil_bounds (a b : Int) (hb : 0 < b) :
    (idiv_ceil a b - 1) * b < a ∧ a ≤ idiv_ceil a b * b := by
  have hf := floor_bounds (-a) b hb
  simp only [Int.add_mul, Int.one_mul] at hf
  simp only [idiv_ceil, Int.zero_sub]
  rw [Int.sub_mul, Int.neg_mul, Int.one_mul]
  omega

theorem pow2i_positive (n : Int) (hn : 0 ≤ n) : 0 < pow2i n := by
  rw [NumericalProofs.pow2i_nonnegative n hn]
  exact Int.pow_pos (by decide : (0 : Int) < 2)

/-- Exact input conversion when no fractional bits are discarded. -/
theorem fx_of_exact (neg : Bool) (mag e p : Int) (hs : 0 ≤ e + p) :
    (fx_of neg mag e p).lo = NumericalProofs.signedMag neg (mag * pow2i (e + p)) ∧
    (fx_of neg mag e p).hi = NumericalProofs.signedMag neg (mag * pow2i (e + p)) := by
  cases neg <;> simp [fx_of, hs, NumericalProofs.signedMag]

/-- In the fractional case the actual production endpoints enclose the exact
    signed input rational, proved after multiplication by its positive common
    denominator. No float sampling or real-number approximation is involved. -/
theorem fx_of_fraction_enclosure (neg : Bool) (mag e p : Int) (hs : e + p < 0) :
    let d := pow2i (-(e + p))
    (fx_of neg mag e p).lo * d ≤ NumericalProofs.signedMag neg mag ∧
      NumericalProofs.signedMag neg mag ≤ (fx_of neg mag e p).hi * d := by
  have hd := pow2i_positive (-(e + p)) (by omega)
  have hf := floor_bounds (NumericalProofs.signedMag neg mag) (pow2i (-(e + p))) hd
  have hc := ceil_bounds (NumericalProofs.signedMag neg mag) (pow2i (-(e + p))) hd
  cases neg <;> simp [fx_of, show ¬0 ≤ e + p by omega, NumericalProofs.signedMag] at *
  all_goals omega

abbrev ExpState := Int × Int × Int × Int

/-- Naming the actual generated recurrence permits invariant proofs. The next
    theorem binds this named step to the whole production loop by definitional
    equality, rather than assuming that a copied algorithm models production. -/
def expStep (one y : Int) (state : ExpState) : ExpState :=
  let (err, k, sum, term) := state
  if term > 0 || term < 0 then
    let t := (term * y).tdiv one
    let t := t.tdiv k
    (err + 2, k + 1, sum + t, t)
  else state

def expState (y p : Int) : ExpState := Id.run do
  let one := pow2i p
  let mut state : ExpState := (2, 1, one, one)
  for _step in [1:p + 1:1]i do
    state := expStep one y state
  return state

theorem exp_small_production_loop (y p : Int) :
    let state := expState y p
    exp_small y p = (state.2.2.1 - state.1, state.2.2.1 + state.1) := by
  rfl

/-- Exact truncating arithmetic halves the magnitude at each nonzero Taylor
    step whenever the actual reduced input is within one half. -/
theorem taylor_step_halves (term y k : Int) (half : Nat) (hh : 0 < half)
    (hy : y.natAbs ≤ half) :
    (((term * y).tdiv (2 * (half : Int))).tdiv k).natAbs ≤ term.natAbs / 2 := by
  rw [Int.natAbs_tdiv, Int.natAbs_tdiv, Int.natAbs_mul]
  have hden : (2 * (half : Int)).natAbs = 2 * half := by simp [Int.natAbs_mul]
  rw [hden]
  apply Nat.le_trans (Nat.div_le_self _ _)
  calc
    term.natAbs * y.natAbs / (2 * half) ≤ term.natAbs * half / (2 * half) :=
      Nat.div_le_div_right (Nat.mul_le_mul_left _ hy)
    _ = term.natAbs / 2 := Nat.mul_div_mul_right _ _ hh

theorem exp_step_halves (state : ExpState) (y : Int) (half : Nat)
    (hh : 0 < half) (hy : y.natAbs ≤ half) :
    (expStep (2 * (half : Int)) y state).2.2.2.natAbs ≤ state.2.2.2.natAbs / 2 := by
  rcases state with ⟨err, k, sum, term⟩
  by_cases hz : term = 0
  · simp [expStep, hz]
  · have hg : term > 0 ∨ term < 0 := by omega
    simpa [expStep, hg] using taylor_step_halves term y k half hh hy

def repeatStep {α : Type} (f : α → α) : Nat → α → α
  | 0, state => state
  | n + 1, state => repeatStep f n (f state)

theorem repeat_halving_bound (f : ExpState → ExpState)
    (hf : ∀ state, (f state).2.2.2.natAbs ≤ state.2.2.2.natAbs / 2)
    (n : Nat) (state : ExpState) :
    (repeatStep f n state).2.2.2.natAbs ≤ state.2.2.2.natAbs / 2 ^ n := by
  induction n generalizing state with
  | zero => simp [repeatStep]
  | succ n ih =>
    simp only [repeatStep]
    apply Nat.le_trans (ih (f state))
    calc
      (f state).2.2.2.natAbs / 2 ^ n ≤ (state.2.2.2.natAbs / 2) / 2 ^ n :=
        Nat.div_le_div_right (hf state)
      _ = state.2.2.2.natAbs / 2 ^ (n + 1) := by
        rw [Nat.div_div_eq_div_mul, Nat.pow_succ]
        rw [Nat.mul_comm 2 (2 ^ n)]

/-- The library's actual inclusive integer loop, with a state-only body, is
    precisely a natural-number iteration. This supplies the loop transfer. -/
theorem unit_range_loop_iterate {α : Type} (range : IntRange) (f : α → α)
    (state : α) (i : Int) (hs : (i - range.start) % range.step = 0)
    (hstep : range.step = 1) (hstart : range.start ≤ i) :
    IntRange.forIn'.loop (m := Id) range (fun _ _ x => pure (.yield (f x))) state i hs =
      repeatStep f (range.stop - i + 1).toNat state := by
  induction state, i, hs using IntRange.forIn'.loop.induct range with
  | case1 state i hs hi ih =>
    have hmem := hi
    simp [Membership.mem, hstep] at hmem
    rw [IntRange.forIn'.loop.eq_1]
    simp only [hi, dite_true, pure_bind]
    rw [ih (f state) (by omega)]
    have hn : (range.stop - i + 1).toNat =
        (range.stop - (i + range.step) + 1).toNat + 1 := by omega
    rw [hn, repeatStep]
  | case2 state i hs hi =>
    have hmem := hi
    simp [Membership.mem, hstep] at hmem
    have hn : (range.stop - i + 1).toNat = 0 := by omega
    rw [IntRange.forIn'.loop.eq_1]
    simp [hi, hn, repeatStep]
    rfl

theorem unit_range_iterate {α : Type} (range : IntRange) (f : α → α)
    (state : α) (hstep : range.step = 1) :
    IntRange.forIn' (m := Id) range state (fun _ _ x => pure (.yield (f x))) =
      repeatStep f (range.stop - range.start + 1).toNat state := by
  unfold IntRange.forIn'
  exact unit_range_loop_iterate range f state range.start (by simp) hstep (by omega)

theorem exp_state_iteration (y p : Int) :
    expState y p = repeatStep (expStep (pow2i p) y) (p + 1).toNat (2, 1, pow2i p, pow2i p) := by
  change IntRange.forIn' (m := Id) {start := 1, stop := p + 1, step := 1}
      (2, 1, pow2i p, pow2i p) (fun _ _ x => pure (.yield (expStep (pow2i p) y x))) = _
  simpa using unit_range_iterate {start := 1, stop := p + 1, step := 1}
    (expStep (pow2i p) y) (2, 1, pow2i p, pow2i p) rfl

/-- The repaired production loop reaches an exactly zero computed term for
    every admitted precision and every reduced argument, not only the p=4096
    regression witnesses. The analytic remainder enclosure remains separate. -/
theorem exp_small_term_zero (y : Int) (p : Nat) (hp : 1 ≤ p)
    (hy : y.natAbs ≤ 2 ^ (p - 1)) :
    (expState y (p : Int)).2.2.2 = 0 := by
  have hh : 0 < 2 ^ (p - 1) := Nat.pow_pos (by decide)
  have hpstep : p = (p - 1) + 1 := by omega
  have hone : pow2i (p : Int) = 2 * ((2 ^ (p - 1) : Nat) : Int) := by
    rw [NumericalProofs.pow2i_nonnegative (p : Int) (by omega)]
    simp only [Int.toNat_natCast]
    calc
      (2 : Int) ^ p = (2 : Int) ^ ((p - 1) + 1) := by rw [← hpstep]
      _ = (2 : Int) ^ (p - 1) * 2 := Int.pow_succ _ _
      _ = 2 * ((2 ^ (p - 1) : Nat) : Int) := by
        simp only [Int.natCast_pow]
        rw [Int.mul_comm]
        rfl
  rw [exp_state_iteration]
  have hcount : ((p : Int) + 1).toNat = p + 1 := by omega
  rw [hcount]
  have hb := repeat_halving_bound (expStep (pow2i (p : Int)) y)
    (fun state => by rw [hone]; exact exp_step_halves state y _ hh hy)
    (p + 1) (2, 1, pow2i (p : Int), pow2i (p : Int))
  have hmag : (pow2i (p : Int)).natAbs = 2 ^ p := by
    rw [NumericalProofs.pow2i_nonnegative (p : Int) (by omega)]
    simp
  have hlt : 2 ^ p < 2 ^ (p + 1) := by
    have hpos : 0 < 2 ^ p := Nat.pow_pos (by decide)
    rw [Nat.pow_succ]
    omega
  simp only [hmag, Nat.div_eq_of_lt hlt] at hb
  exact Int.natAbs_eq_zero.mp (Nat.eq_zero_of_le_zero hb)

/-- Nested truncation by positive scale/factorial denominators has exactly
    one truncation error, not two unrelated real approximations. -/
theorem tdiv_tdiv_nat (a : Int) (b c : Nat) :
    (a.tdiv (b : Int)).tdiv (c : Int) = a.tdiv ((b * c : Nat) : Int) := by
  cases a with
  | ofNat n =>
    change Int.ofNat ((n / b) / c) = Int.ofNat (n / (b * c))
    rw [Nat.div_div_eq_div_mul]
  | negSucc n =>
    change ((-((n + 1 : Nat) : Int)).tdiv (b : Int)).tdiv (c : Int) =
      (-((n + 1 : Nat) : Int)).tdiv ((b * c : Nat) : Int)
    rw [Int.neg_tdiv, Int.neg_tdiv, Int.neg_tdiv]
    congr 1
    change Int.ofNat (((n + 1) / b) / c) = Int.ofNat ((n + 1) / (b * c))
    rw [Nat.div_div_eq_div_mul]

theorem tdiv_tdiv_positive (a b c : Int) (hb : 0 < b) (hc : 0 < c) :
    (a.tdiv b).tdiv c = a.tdiv (b * c) := by
  have hbcast : (b.toNat : Int) = b := Int.toNat_of_nonneg (by omega)
  have hccast : (c.toNat : Int) = c := Int.toNat_of_nonneg (by omega)
  have h := tdiv_tdiv_nat a b.toNat c.toNat
  simpa [Int.natCast_mul, hbcast, hccast] using h

theorem tdiv_remainder_bound (a d : Int) (hd : 0 < d) :
    (a - d * a.tdiv d).natAbs < d.natAbs := by
  rw [← Int.tmod_def, Int.natAbs_tmod]
  apply Nat.mod_lt
  have hne : d ≠ 0 := by omega
  exact Nat.pos_of_ne_zero (by intro h; apply hne; exact Int.natAbs_eq_zero.mp h)

/-- Formal local error bound for the actual two-divide Taylor recurrence,
    as an exact integer remainder under the common denominator one*k. -/
theorem taylor_step_remainder_bound (term y one k : Int) (ho : 0 < one) (hk : 0 < k) :
    (term * y - (one * k) * ((term * y).tdiv one).tdiv k).natAbs < (one * k).natAbs := by
  rw [tdiv_tdiv_positive (term * y) one k ho hk]
  exact tdiv_remainder_bound (term * y) (one * k) (Int.mul_pos ho hk)

theorem exp_step_counters (state : ExpState) (one y : Int)
    (he : state.1 = 2 * state.2.1) (hk : 1 ≤ state.2.1) :
    (expStep one y state).1 = 2 * (expStep one y state).2.1 ∧
      1 ≤ (expStep one y state).2.1 := by
  rcases state with ⟨err, k, sum, term⟩
  by_cases hz : term = 0
  · simp [expStep, hz] at *
    omega
  · have hg : term > 0 ∨ term < 0 := by omega
    simp [expStep, hg] at *
    omega

theorem repeat_counter_invariant (n : Nat) (state : ExpState) (one y : Int)
    (he : state.1 = 2 * state.2.1) (hk : 1 ≤ state.2.1) :
    (repeatStep (expStep one y) n state).1 =
        2 * (repeatStep (expStep one y) n state).2.1 ∧
      1 ≤ (repeatStep (expStep one y) n state).2.1 := by
  induction n generalizing state with
  | zero => exact ⟨he, hk⟩
  | succ n ih =>
    have h := exp_step_counters state one y he hk
    exact ih (expStep one y state) h.1 h.2

theorem exp_state_counters (y p : Int) :
    (expState y p).1 = 2 * (expState y p).2.1 ∧ 1 ≤ (expState y p).2.1 := by
  rw [exp_state_iteration]
  exact repeat_counter_invariant (p + 1).toNat (2, 1, pow2i p, pow2i p) (pow2i p) y rfl (by change (1 : Int) ≤ 1; decide)

theorem repeat_fixedpoint {α : Type} (f : α → α) (state : α)
    (hf : f state = state) (n : Nat) : repeatStep f n state = state := by
  induction n with
  | zero => rfl
  | succ n ih => simp [repeatStep, hf, ih]

theorem exp_zero_argument_state (p : Int) (hp : 0 ≤ p) :
    expState 0 p = (4, 2, pow2i p, 0) := by
  rw [exp_state_iteration]
  have hn : (p + 1).toNat = p.toNat + 1 := by omega
  rw [hn, repeatStep]
  have hfirst : expStep (pow2i p) 0 (2, 1, pow2i p, pow2i p) = (4, 2, pow2i p, 0) := by
    have hpos := pow2i_positive p hp
    simp [expStep, show pow2i p > 0 by omega]
  rw [hfirst]
  exact repeat_fixedpoint _ _ (by simp [expStep]) _

/-- Zero is included, including the p=0 helper contract boundary. -/
theorem exp_small_zero_argument (p : Int) (hp : 0 ≤ p) :
    exp_small 0 p = (pow2i p - 4, pow2i p + 4) := by
  rw [exp_small_production_loop, exp_zero_argument_state p hp]

theorem exp_small_term_zero_p0 (y : Int) (hy : 2 * y.natAbs ≤ 1) :
    (expState y 0).2.2.2 = 0 := by
  have hz : y = 0 := Int.natAbs_eq_zero.mp (by omega)
  rw [hz, exp_zero_argument_state 0 (by decide)]

end IntervalProofs
