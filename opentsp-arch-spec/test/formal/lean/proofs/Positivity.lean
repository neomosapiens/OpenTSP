-- Actual production Exp endpoint positivity, needed before TanH inversion.
import Proofs.RangeReduction
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 1000000
set_option exponentiation.threshold 1000000
namespace PositivityProofs
open IntervalProofs

theorem tdiv_nonpos (a b : Int) (ha : a ≤ 0) (hb : 0 ≤ b) : a.tdiv b ≤ 0 := by
  have h := Int.tdiv_nonneg (show 0 ≤ -a by omega) hb
  rw [Int.neg_tdiv] at h
  omega

theorem step_index_upper (state : ExpState) (one y : Int) :
    (expStep one y state).2.1  ≤  state.2.1+1 := by
  rcases state with ⟨err,k,sum,term⟩
  by_cases hz : term=0
  · simp [expStep,hz]; omega
  · have hg : term > 0 ∨ term < 0 := by omega
    simp [expStep,hg]

theorem repeat_index_upper (steps : Nat) (state : ExpState) (one y : Int) :
    (repeatStep (expStep one y) steps state).2.1  ≤  state.2.1+(steps:Int) := by
  induction steps generalizing state with
  | zero =>  simp [repeatStep]
  | succ steps ih =>
    have h := step_index_upper state one y
    have hi := ih (expStep one y state)
    simp only [repeatStep]
    omega

theorem exp_index_upper (y : Int) (p : Nat) :
    (expState y p).2.1  ≤  (p:Int)+2 := by
  rw [exp_state_iteration]
  have hn : ((p:Int)+1).toNat=p+1 := by omega
  rw [hn]
  have h := repeat_index_upper (p+1) (2,1,pow2i p,pow2i p) (pow2i p) y
  dsimp at h
  omega

theorem positive_step_sum (state : ExpState) (one y base : Int)
    (hone : 0 < one) (hy : 0 ≤ y) (hk : 1 ≤ state.2.1)
    (ht : 0 ≤ state.2.2.2) (hs : base ≤ state.2.2.1) :
    1 ≤ (expStep one y state).2.1 ∧ 0 ≤ (expStep one y state).2.2.2 ∧
      base ≤ (expStep one y state).2.2.1 := by
  rcases state with ⟨err,k,sum,term⟩
  dsimp at hk ht hs
  by_cases hz : term=0
  · simpa [expStep,hz] using And.intro hk (And.intro ht hs)
  · have hg : term > 0 ∨ term < 0 := by omega
    have ht1 := Int.tdiv_nonneg (Int.mul_nonneg ht hy) (by omega : 0 ≤ one)
    have ht2 := Int.tdiv_nonneg ht1 (by omega : 0 ≤ k)
    simp [expStep,hg]
    omega

theorem positive_repeat_sum (steps : Nat) (state : ExpState) (one y base : Int)
    (hone : 0 < one) (hy : 0 ≤ y) (hk : 1 ≤ state.2.1)
    (ht : 0 ≤ state.2.2.2) (hs : base ≤ state.2.2.1) :
    base ≤ (repeatStep (expStep one y) steps state).2.2.1 := by
  induction steps generalizing state with
  | zero =>  exact hs
  | succ steps ih =>
    have h := positive_step_sum state one y base hone hy hk ht hs
    exact ih (expStep one y state) h.1 h.2.1 h.2.2

/-- Alternating integer terms preserve the lower first-partial-sum bound;
    truncation cannot reverse a term sign or increase its magnitude. -/
theorem negative_step_sum (state : ExpState) (y base : Int) (half : Nat)
    (hh : 0 < half) (hy : y ≤ 0) (hym : y.natAbs ≤ half) (hk : 1 ≤ state.2.1)
    (hs : base+max state.2.2.2 0  ≤  state.2.2.1) :
    1 ≤ (expStep (2*(half:Int)) y state).2.1 ∧
    base+max (expStep (2*(half:Int)) y state).2.2.2 0  ≤
      (expStep (2*(half:Int)) y state).2.2.1 := by
  rcases state with ⟨err,k,sum,term⟩
  dsimp at hk hs
  by_cases hz : term=0
  · simpa [expStep,hz] using And.intro hk hs
  · have hg : term > 0 ∨ term < 0 := by omega
    let t := ((term*y).tdiv (2*(half:Int))).tdiv k
    have hm := taylor_step_halves term y k half hh hym
    have habs : t.natAbs ≤ term.natAbs := Nat.le_trans hm (Nat.div_le_self _ _)
    by_cases ht : 0 ≤ term
    · have ht1 := tdiv_nonpos (term*y) (2*(half:Int)) (Int.mul_nonpos_of_nonneg_of_nonpos ht hy) (by omega : 0 ≤ 2*(half:Int))
      have ht2 := tdiv_nonpos ((term*y).tdiv (2*(half:Int))) k ht1 (by omega : 0 ≤ k)
      have htm : -term ≤ t := by dsimp [t] at *; omega
      have hs0 : base+term ≤ sum := by simpa [Int.max_eq_left ht] using hs
      simp [expStep,hg]
      change 1 ≤ k+1 ∧ base+max t 0 ≤ sum+t
      have ht0 : t ≤ 0 := ht2
      rw [Int.max_eq_right ht0]
      omega
    · have ht0 : term ≤ 0 := by omega
      have ht1 := Int.tdiv_nonneg (Int.mul_nonneg_of_nonpos_of_nonpos ht0 hy) (by omega : 0 ≤ 2*(half:Int))
      have ht2 := Int.tdiv_nonneg ht1 (by omega : 0 ≤ k)
      have hs0 : base ≤ sum := by simpa [Int.max_eq_right ht0] using hs
      simp [expStep,hg]
      change 1 ≤ k+1 ∧ base+max t 0 ≤ sum+t
      have ht0 : 0 ≤ t := ht2
      rw [Int.max_eq_left ht0]
      omega

theorem negative_repeat_sum (steps : Nat) (state : ExpState) (y base : Int) (half : Nat)
    (hh : 0 < half) (hy : y ≤ 0) (hym : y.natAbs ≤ half) (hk : 1 ≤ state.2.1)
    (hs : base+max state.2.2.2 0 ≤ state.2.2.1) :
    base+max (repeatStep (expStep (2*(half:Int)) y) steps state).2.2.2 0  ≤
      (repeatStep (expStep (2*(half:Int)) y) steps state).2.2.1 := by
  induction steps generalizing state with
  | zero =>  exact hs
  | succ steps ih =>
    have h := negative_step_sum state y base half hh hy hym hk hs
    exact ih (expStep (2*(half:Int)) y state) h.1 h.2

theorem exp_sum_lower (y : Int) (p : Nat) (hp : 1 ≤ p)
    (hy : 2*y.natAbs ≤ 2^p) :
    pow2i p ≤ 2*(expState y p).2.2.1 := by
  have honepos := pow2i_positive (p:Int) (by omega)
  by_cases hsign : 0 ≤ y
  · have h := positive_repeat_sum ((p:Int)+1).toNat
      (2,1,pow2i p,pow2i p) (pow2i p) y (pow2i p) honepos hsign
      (by change (1:Int) ≤ 1; decide) (by dsimp; omega) (by dsimp; omega)
    rw [exp_state_iteration]
    omega
  · let half : Nat := 2^(p-1)
    have hhalf : 0 < half := Nat.pow_pos (by decide)
    have hpow : 2^p = 2*half := by
      have hsucc := Nat.pow_succ 2 (p-1)
      have he : p-1+1=p := by omega
      change 2^(p-1+1)=2^(p-1)*2 at hsucc
      rw [he] at hsucc
      dsimp [half]
      omega
    have hone : pow2i (p:Int)=2*(half:Int) := by
      rw [NumericalProofs.pow2i_nonnegative (p:Int) (by omega)]
      simp only [Int.toNat_natCast]
      have hcast := congrArg (fun n : Nat => (n:Int)) hpow
      simpa only [Int.natCast_pow,Int.natCast_mul] using hcast
    have hym : y.natAbs ≤ half := by omega
    have hy0 : y ≤ 0 := by omega
    have hylo : -(half:Int) ≤ y := by omega
    have hf : expStep (pow2i p) y (2,1,pow2i p,pow2i p) =
        (4,2,pow2i p+y,y) := by
      simp [expStep,show pow2i (p:Int)>0 by omega,
        Int.mul_tdiv_cancel_left y (by omega : pow2i (p:Int)≠0)]
    have hneg := negative_repeat_sum p (4,2,pow2i p+y,y) y
      (pow2i p+y) half hhalf hy0 hym (by change (1:Int) ≤ 2; decide)
      (by simp [Int.max_eq_right hy0])
    rw [←hone] at hneg
    have hn : ((p:Int)+1).toNat=p+1 := by omega
    rw [exp_state_iteration,hn,repeatStep,hf]
    have hmax : 0 ≤ max (repeatStep (expStep (pow2i p) y) p
        (4,2,pow2i p+y,y)).2.2.2 0 := Int.le_max_right _ _
    omega

theorem schedule_margin (p : Nat)
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096) :
    4*(p+2) < pow2i (p:Int) := by
  rcases hp with h|h|h|h|h|h|h
  all_goals subst p; decide

/-- Entirely integer positivity: alternating truncated terms stay above their
    first partial sum; this uses no real exponential lower-bound assumption. -/
theorem exp_small_positive (y : Int) (p : Nat)
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096)
    (hy : 2*y.natAbs ≤ 2^p) :
    0 < (exp_small y p).1 ∧ (exp_small y p).1 ≤ (exp_small y p).2 := by
  have hp1 : 1 ≤ p := by omega
  have hsum := exp_sum_lower y p hp1 hy
  have hk := exp_index_upper y p
  have hc := exp_state_counters y (p:Int)
  have hm := schedule_margin p hp
  rw [exp_small_production_loop]
  dsimp only
  omega

theorem exp_iv_positive (x : ival) (p : Nat) (hxp : x.p=(p:Int))
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096)
    (hlo : 2*(RangeReductionProofs.reduction x).1.natAbs ≤ 2^p)
    (hhi : 2*(RangeReductionProofs.reduction x).2.1.natAbs ≤ 2^p) :
    0 < (exp_iv x).1 ∧ 0 < (exp_iv x).2.1 := by
  have hl := exp_small_positive (RangeReductionProofs.reduction x).1 p hp hlo
  have hh := exp_small_positive (RangeReductionProofs.reduction x).2.1 p hp hhi
  rw [RangeReductionProofs.exp_iv_production_reduction]
  dsimp only
  rw [hxp]
  exact ⟨hl.1,by omega⟩

theorem pow2i_any_positive (n : Int) : 0 < pow2i n := by
  by_cases hn : 0 ≤ n
  · exact pow2i_positive n hn
  · simp [pow2i,show n ≤ 0 by omega]

theorem floor_ceil_nonnegative (a d : Int) (ha : 0 ≤ a) (hd : 0 < d) :
    0 ≤ idiv_floor a d ∧ 0 ≤ idiv_ceil a d := by
  have hf : 0 ≤ idiv_floor a d := by
    rw [idiv_floor_correct a d hd]
    exact Int.ediv_nonneg ha (by omega)
  have hw := RangeReductionProofs.floor_ceil_width a d hd
  omega

theorem scaled_exp_nonnegative (lo hi n : Int) (hlo : 0 ≤ lo) (hhi : 0 ≤ hi) :
    let elo := if n ≥ 0 then lo*pow2i n else idiv_floor lo (pow2i (-n))
    let ehi := if n ≥ 0 then hi*pow2i n else idiv_ceil hi (pow2i (-n))
    0 ≤ elo ∧ 0 ≤ ehi := by
  dsimp only
  by_cases hn : 0 ≤ n
  · simp only [hn,ite_true]
    exact ⟨Int.mul_nonneg hlo (by have h := pow2i_any_positive n; omega),
      Int.mul_nonneg hhi (by have h := pow2i_any_positive n; omega)⟩
  · simp only [hn,ite_false]
    exact ⟨(floor_ceil_nonnegative lo _ hlo (pow2i_any_positive (-n))).1,
      (floor_ceil_nonnegative hi _ hhi (pow2i_any_positive (-n))).2⟩

theorem tanh_denominators_positive (lo hi n p : Int)
    (hlo : 0 ≤ lo) (hhi : 0 ≤ hi) :
    let elo := if n ≥ 0 then lo*pow2i n else idiv_floor lo (pow2i (-n))
    let ehi := if n ≥ 0 then hi*pow2i n else idiv_ceil hi (pow2i (-n))
    0 < elo+pow2i p ∧ 0 < ehi+pow2i p := by
  have h := scaled_exp_nonnegative lo hi n hlo hhi
  have hp := pow2i_any_positive p
  dsimp only at h ⊢
  omega

end PositivityProofs
