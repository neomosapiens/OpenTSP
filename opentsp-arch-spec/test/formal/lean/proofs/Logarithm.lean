-- Actual positive atanh-series Log arithmetic and finite-loop domain.
import Proofs.Reciprocal
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 1000000
set_option maxHeartbeats 1000000
set_option exponentiation.threshold 1000000
namespace LogarithmProofs
open IntervalProofs

abbrev LogState := Int × Int × Int

def logStep (one z2 : Int) (state : LogState) : LogState :=
  let (j,sum,t) := state
  let (j,sum,t) : (Int × Int × Int) :=
    if (((t >b 0) && (j <b 2000)) : Bool) then
      let sum := sum +i Int.tdiv t ((2 *i j) +i 1)
      let t := Int.tdiv (t *i z2) one
      let j := j +i 1
      (j,sum,t)
    else (j,sum,t)
  (j,sum,t)

def logStateCount (z z2 one upper : Int) : LogState := Id.run do
  let mut state : LogState := (0,0,z)
  for _step in [0:upper:1]i do
    state := logStep one z2 state
  return state

abbrev logState (z z2 one : Int) : LogState := logStateCount z z2 one 1999

theorem log_state_count_iteration (z z2 one upper : Int) :
    logStateCount z z2 one upper = repeatStep (logStep one z2) (upper+1).toNat (0,0,z) := by
  change IntRange.forIn' (m := Id) {start := 0,stop := upper,step := 1}
    (0,0,z) (fun _ _ state => pure (.yield (logStep one z2 state))) = _
  simpa using unit_range_iterate {start := 0,stop := upper,step := 1}
    (logStep one z2) (0,0,z) rfl

theorem log_state_iteration (z z2 one : Int) :
    logState z z2 one = repeatStep (logStep one z2) 2000 (0,0,z) :=
  log_state_count_iteration z z2 one 1999

def logFactory (mag : Int) (e : Int) (p : Int) (upper : Int) : (Int × Int) := Id.run do
  let b := (ilog2 mag)
  let k := (b +i e)
  let one := (pow2i (p +i 8))
  let num := ((mag -i (pow2i b)) *i one)
  let den := (mag +i (pow2i b))
  let z := (Int.tdiv num den)
  let z2 := (Int.tdiv (z *i z) one)
  let t : Int := z
  let sum : Int := 0
  let j : Int := 0
  let (j, sum, t) ← (( do
    let loop__step_lower := 0
    let loop__step_upper := upper
    let mut loop_vars := (j, sum, t)
    for _step in [loop__step_lower:loop__step_upper:1]i do
      let (j, sum, t) := loop_vars
      loop_vars :=
        let (j, sum, t) : (Int × Int × Int) :=
          if (((t >b 0) && (j <b 2000)) : Bool)
          then
            (let sum : Int := (sum +i (Int.tdiv t ((2 *i j) +i 1)))
            let t : Int := (Int.tdiv (t *i z2) one)
            let j : Int := (j +i 1)
            (j, sum, t))
          else (j, sum, t)
        (j, sum, t)
    (pure loop_vars) ) : Id (Int × Int × Int) )
  let lnm := (2 *i sum)
  let errm := (4 *i (j +i 3))
  let (l2lo, l2hi) := (ln2_iv (p +i 8))
  let lo :=
    (((if ((k ≥b 0) : Bool)
        then (k *i l2lo)
        else (k *i l2hi)) +i lnm) -i errm)
  let hi :=
    (((if ((k ≥b 0) : Bool)
        then (k *i l2hi)
        else (k *i l2lo)) +i lnm) +i errm)
  (pure ((idiv_floor lo 256), (idiv_ceil hi 256)))


theorem log_iv_factory_source (mag e p : Int) :
    log_iv mag e p=logFactory mag e p 1999 := by rfl

theorem log_factory_decomposition (mag e p upper : Int) :
    logFactory mag e p upper =
      let b := ilog2 mag
      let k := b+e
      let one := pow2i (p+8)
      let z := ((mag-pow2i b)*one).tdiv (mag+pow2i b)
      let z2 := (z*z).tdiv one
      let (j,sum,_t) := logStateCount z z2 one upper
      let lnm := 2*sum
      let err := 4*(j+3)
      let (l2lo,l2hi) := ln2_iv (p+8)
      (idiv_floor ((if ((k ≥b 0):Bool) then k*l2lo else k*l2hi)+lnm-err) 256,
       idiv_ceil ((if ((k ≥b 0):Bool) then k*l2hi else k*l2lo)+lnm+err) 256) := by
  set_option maxRecDepth 1000 in rfl

theorem log_iv_production (mag e p : Int) :
    log_iv mag e p =
      let b := ilog2 mag
      let k := b+e
      let one := pow2i (p+8)
      let z := ((mag-pow2i b)*one).tdiv (mag+pow2i b)
      let z2 := (z*z).tdiv one
      let (j,sum,_t) := logState z z2 one
      let lnm := 2*sum
      let err := 4*(j+3)
      let (l2lo,l2hi) := ln2_iv (p+8)
      (idiv_floor ((if ((k ≥b 0):Bool) then k*l2lo else k*l2hi)+lnm-err) 256,
       idiv_ceil ((if ((k ≥b 0):Bool) then k*l2hi else k*l2lo)+lnm+err) 256) := by
  rw [log_iv_factory_source,log_factory_decomposition]


theorem normalized_ratio (mag c : Int) (hc : 0 < c)
    (hlo : c ≤ mag) (hhi : mag ≤ 2*c) :
    0 ≤ mag-c ∧ 0 < mag+c ∧ 3*(mag-c) ≤ mag+c := by omega

theorem nonnegative_tdiv_bounds (a b : Int) (ha : 0 ≤ a) (hb : 0 < b) :
    0 ≤ a.tdiv b ∧ 0 ≤ a-a.tdiv b*b ∧ a-a.tdiv b*b < b := by
  rw [Int.tdiv_eq_ediv_of_nonneg ha]
  have hq := Int.ediv_nonneg ha (by omega : 0 ≤ b)
  have hr0 := Int.emod_nonneg a (by omega : b≠0)
  have hr1 := Int.emod_lt_of_pos a hb
  have he := Int.ediv_mul_add_emod a b
  omega

/-- Actual normalized integer z has the one-third domain used by the series. -/
theorem normalized_z_domain (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) :
    let z := (A*S).tdiv D
    0 ≤ z ∧ 3*z ≤ S ∧ 0 ≤ A*S-z*D ∧ A*S-z*D < D := by
  have hb := nonnegative_tdiv_bounds (A*S) D (Int.mul_nonneg hA (by omega)) hD
  let z := (A*S).tdiv D
  change 0 ≤ z ∧ 0 ≤ A*S-z*D ∧ A*S-z*D < D at hb
  have hmul := Int.mul_le_mul_of_nonneg_right hr (by omega : 0 ≤ S)
  have hcross : (3*z)*D ≤ S*D := by
    have he1 : (3*z)*D=3*(z*D) := by grobner
    have he2 : (3*A)*S=3*(A*S) := by grobner
    rw [he1]
    rw [he2] at hmul
    have he3 : S*D=D*S := by grobner
    rw [he3]
    omega
  have hz := Int.le_of_mul_le_mul_right hcross hD
  exact ⟨hb.1,hz,hb.2.1,hb.2.2⟩

/-- The actual squared coefficient is nonnegative and at most one ninth. -/
theorem squared_coefficient_domain (z S : Int) (hz : 0 ≤ z)
    (hS : 0 < S) (hthird : 3*z ≤ S) :
    let z2 := (z*z).tdiv S
    0 ≤ z2 ∧ 9*z2 ≤ S := by
  have hb := nonnegative_tdiv_bounds (z*z) S (Int.mul_nonneg hz hz) hS
  let z2 := (z*z).tdiv S
  change 0 ≤ z2 ∧ 0 ≤ z*z-z2*S ∧ z*z-z2*S < S at hb
  have h1 := Int.mul_le_mul_of_nonneg_right hthird hz
  have h2 := Int.mul_le_mul_of_nonneg_right hthird (by omega : 0 ≤ S)
  have hcross : (9*z2)*S ≤ S*S := by
    have he1 : (3*z)*z=3*(z*z) := by grobner
    have he2 : (3*z)*S=3*(S*z) := by grobner
    have he3 : (9*z2)*S=9*(z2*S) := by grobner
    rw [he1] at h1
    rw [he2] at h2
    rw [he3]
    omega
  exact ⟨hb.1,Int.le_of_mul_le_mul_right hcross hS⟩

def uncappedStep (one z2 : Int) (state : LogState) : LogState :=
  let (j,sum,t) := state
  if t > 0 then (j+1,sum+t.tdiv (2*j+1),(t*z2).tdiv one) else state

theorem uncapped_index_upper (state : LogState) (one z2 : Int) :
    (uncappedStep one z2 state).1 ≤ state.1+1 := by
  rcases state with ⟨j,sum,t⟩
  by_cases ht : t > 0
  · simp [uncappedStep,ht]
  · simp [uncappedStep,ht]; omega

/-- The cap cannot bypass any of the actual 2000 iterations: the index rises
    at most once per step from zero. This is not a successful Ziv-cap claim. -/
theorem capped_uncapped_iteration (steps : Nat) (state : LogState) (one z2 : Int)
    (hbound : state.1+(steps:Int) ≤ 2000) :
    repeatStep (logStep one z2) steps state =
      repeatStep (uncappedStep one z2) steps state := by
  induction steps generalizing state with
  | zero => rfl
  | succ steps ih =>
    have hj : state.1 < 2000 := by omega
    have hstep : logStep one z2 state=uncappedStep one z2 state := by
      rcases state with ⟨j,sum,t⟩
      dsimp at hj
      simp [logStep,uncappedStep,hj]
    simp only [repeatStep,hstep]
    apply ih
    have hu := uncapped_index_upper state one z2
    omega

theorem product_div_eight (term z2 : Int) (half : Nat) (hh : 0 < half)
    (hb : z2.natAbs ≤ half) :
    ((term*z2).tdiv (8*(half:Int))).natAbs ≤ term.natAbs/8 := by
  rw [Int.natAbs_tdiv,Int.natAbs_mul]
  have hd : (8*(half:Int)).natAbs=8*half := by simp [Int.natAbs_mul]
  rw [hd]
  calc
    term.natAbs*z2.natAbs/(8*half) ≤ term.natAbs*half/(8*half) :=
      Nat.div_le_div_right (Nat.mul_le_mul_left _ hb)
    _ = term.natAbs/8 := Nat.mul_div_mul_right _ _ hh

theorem uncapped_term_shrink (state : LogState) (z2 : Int) (half : Nat)
    (hh : 0 < half) (hb : 0 ≤ z2) (hbm : z2.natAbs ≤ half)
    (ht : 0 ≤ state.2.2) :
    0 ≤ (uncappedStep (8*(half:Int)) z2 state).2.2 ∧
      (uncappedStep (8*(half:Int)) z2 state).2.2.natAbs ≤ state.2.2.natAbs/8 := by
  rcases state with ⟨j,sum,t⟩
  dsimp at ht
  by_cases hz : t=0
  · simp [uncappedStep,hz]
  · have hp : t > 0 := by omega
    have hn := Int.tdiv_nonneg (Int.mul_nonneg ht hb) (by omega : 0 ≤ 8*(half:Int))
    have hm := product_div_eight t z2 half hh hbm
    simpa [uncappedStep,hp] using And.intro hn hm

theorem uncapped_repeat_shrink (steps : Nat) (state : LogState) (z2 : Int) (half : Nat)
    (hh : 0 < half) (hb : 0 ≤ z2) (hbm : z2.natAbs ≤ half)
    (ht : 0 ≤ state.2.2) :
    0 ≤ (repeatStep (uncappedStep (8*(half:Int)) z2) steps state).2.2 ∧
      (repeatStep (uncappedStep (8*(half:Int)) z2) steps state).2.2.natAbs ≤
        state.2.2.natAbs/8^steps := by
  induction steps generalizing state with
  | zero => simpa [repeatStep] using ht
  | succ steps ih =>
    have hs := uncapped_term_shrink state z2 half hh hb hbm ht
    have hi := ih (uncappedStep (8*(half:Int)) z2 state) hs.1
    simp only [repeatStep]
    refine ⟨hi.1,Nat.le_trans hi.2 ?_⟩
    calc
      (uncappedStep (8*(half:Int)) z2 state).2.2.natAbs/8^steps ≤
          (state.2.2.natAbs/8)/8^steps := Nat.div_le_div_right hs.2
      _ = state.2.2.natAbs/8^(steps+1) := by
        rw [Nat.div_div_eq_div_mul,Nat.pow_succ]
        rw [Nat.mul_comm 8 (8^steps)]

/-- Every actual Log-series run at a Ziv precision has a zero computed tail
    after its actual 2000 iterations, for the complete normalized domain. -/
theorem log_computed_term_zero (z z2 : Int) (p : Nat) (hp : p ≤ 4096)
    (hz : 0 ≤ z) (hzm : z ≤ pow2i ((p:Int)+8))
    (hb : 0 ≤ z2) (hbm : 9*z2 ≤ pow2i ((p:Int)+8)) :
    (logState z z2 (pow2i ((p:Int)+8))).2.2=0 := by
  let half : Nat := 2^(p+5)
  have hh : 0 < half := Nat.pow_pos (by decide)
  have hone : pow2i ((p:Int)+8)=8*(half:Int) := by
    rw [NumericalProofs.pow2i_nonnegative ((p:Int)+8) (by omega)]
    have hn : ((p:Int)+8).toNat=p+8 := by omega
    rw [hn]
    have he : p+8=3+(p+5) := by omega
    rw [he,Int.pow_add]
    simp [half,Int.natCast_pow]
  have hbmN : z2.natAbs ≤ half := by omega
  have hpowlt : 2^(p+8) < 8^2000 := by
    have hbnd : p+8 < 3*2000 := by omega
    have h := Nat.pow_lt_pow_of_lt (by decide : 1 < (2:Nat)) hbnd
    have he : (8:Nat)^2000=2^(3*2000) := by rw [Nat.pow_mul]
    rw [he]
    exact h
  have hzmag : z.natAbs ≤ 2^(p+8) := by
    have hm : (pow2i ((p:Int)+8)).natAbs=2^(p+8) := by
      rw [NumericalProofs.pow2i_nonnegative ((p:Int)+8) (by omega)]
      simp [show ((p:Int)+8).toNat=p+8 by omega]
    omega
  rw [log_state_iteration,capped_uncapped_iteration 2000 _ _ _ (by change (0:Int)+2000 ≤ 2000; decide)]
  rw [hone]
  have hs := uncapped_repeat_shrink 2000 (0,0,z) z2 half hh hb hbmN hz
  have hlt : z.natAbs < 8^2000 := Nat.lt_of_le_of_lt hzmag hpowlt
  simp only [Nat.div_eq_of_lt hlt] at hs
  exact Int.natAbs_eq_zero.mp (Nat.eq_zero_of_le_zero hs.2)

/-- The rounded squared coefficient underestimates the exact rational square
    by strictly fewer than two fixed-point units, uniformly over all inputs. -/
theorem squared_coefficient_error (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) :
    let z := (A*S).tdiv D
    let b := (z*z).tdiv S
    0 ≤ S*(A*A)-b*(D*D) ∧ S*(A*A)-b*(D*D) < 2*(D*D) := by
  let z := (A*S).tdiv D
  let b := (z*z).tdiv S
  change 0 ≤ S*(A*A)-b*(D*D) ∧ S*(A*A)-b*(D*D) < 2*(D*D)
  have hz := normalized_z_domain A D S hA hD hS hr
  change 0 ≤ z ∧ 3*z ≤ S ∧ 0 ≤ A*S-z*D ∧ A*S-z*D < D at hz
  have hb := nonnegative_tdiv_bounds (z*z) S (Int.mul_nonneg hz.1 hz.1) hS
  change 0 ≤ b ∧ 0 ≤ z*z-b*S ∧ z*z-b*S < S at hb
  have hZD : 0 ≤ z*D := Int.mul_nonneg hz.1 (by omega)
  have hAS : 0 ≤ A*S := Int.mul_nonneg hA (by omega)
  have hzd : z*D ≤ A*S := by omega
  have hsquare : (z*D)*(z*D) ≤ (A*S)*(A*S) :=
    Int.le_trans (Int.mul_le_mul_of_nonneg_right hzd hZD)
      (Int.mul_le_mul_of_nonneg_left hzd hAS)
  have hF : 0 < D*D := Int.mul_pos hD hD
  have hloScaled : (b*(D*D))*S ≤ (S*(A*A))*S := by
    have h1 := Int.mul_le_mul_of_nonneg_right (show b*S ≤ z*z by omega) (by omega : 0 ≤ D*D)
    have he1 : (b*(D*D))*S=(b*S)*(D*D) := by grobner
    have he2 : (S*(A*A))*S=(A*S)*(A*S) := by grobner
    have he3 : (z*z)*(D*D)=(z*D)*(z*D) := by grobner
    rw [he1,he2]
    rw [he3] at h1
    omega
  have hlo := Int.le_of_mul_le_mul_right hloScaled hS
  have hsum : A*S+z*D ≤ S*D := by
    have h2 := Int.mul_le_mul_of_nonneg_right (show 2*A ≤ D by omega) (by omega : 0 ≤ S)
    have he : (2*A)*S=2*(A*S) := by grobner
    rw [he] at h2
    have hc : D*S=S*D := by grobner
    rw [hc] at h2
    omega
  have hfirst : (A*S-z*D)*(A*S+z*D) ≤ S*(D*D) := by
    calc
      (A*S-z*D)*(A*S+z*D) ≤ (A*S-z*D)*(S*D) :=
        Int.mul_le_mul_of_nonneg_left hsum hz.2.2.1
      _ ≤ D*(S*D) := Int.mul_le_mul_of_nonneg_right (by omega : A*S-z*D ≤ D)
        (Int.mul_nonneg (by omega) (by omega))
      _ = S*(D*D) := by grobner
  have hsecond : (z*z-b*S)*(D*D) < S*(D*D) :=
    Int.mul_lt_mul_of_pos_right hb.2.2 hF
  have he : (S*(A*A)-b*(D*D))*S =
      (A*S-z*D)*(A*S+z*D)+(z*z-b*S)*(D*D) := by grobner
  have huScaled : (S*(A*A)-b*(D*D))*S < (2*(D*D))*S := by
    rw [he]
    have he2 : (2*(D*D))*S=S*(D*D)+S*(D*D) := by grobner
    rw [he2]
    omega
  exact ⟨by omega,Int.lt_of_mul_lt_mul_right huScaled (by omega)⟩

/-- One-sided rational error invariant for the actual positive power recurrence.
    The 17/9 bound leaves strict margin below two units at every step. -/
theorem power_term_error_step (t N D C F b S : Int)
    (ht : 0 ≤ t) (hD : 0 < D) (hF : 0 < F) (hS : 0 < S)
    (hC : 0 ≤ C) (hb : 0 ≤ b) (hratio : 9*C ≤ F)
    (hsize : 3*N ≤ S*D)
    (hold : 0 ≤ N-t*D ∧ N-t*D < 2*D)
    (hcoef : 0 ≤ S*C-b*F ∧ S*C-b*F < 2*F) :
    let u := (t*b).tdiv S
    0 ≤ u ∧ 0 ≤ N*C-u*(D*F) ∧ N*C-u*(D*F) < 2*(D*F) := by
  let u := (t*b).tdiv S
  change 0 ≤ u ∧ 0 ≤ N*C-u*(D*F) ∧ N*C-u*(D*F) < 2*(D*F)
  let E := N-t*D
  let U := t*D
  let Q := S*C-b*F
  let R := t*b-u*S
  let B := S*D*F
  have htdiv := nonnegative_tdiv_bounds (t*b) S (Int.mul_nonneg ht hb) hS
  change 0 ≤ u ∧ 0 ≤ R ∧ R < S at htdiv
  have hE : 0 ≤ E ∧ E ≤ 2*D := by dsimp [E]; omega
  have hQ : 0 ≤ Q ∧ Q ≤ 2*F := by dsimp [Q]; omega
  have hU : 0 ≤ U := Int.mul_nonneg ht (by omega)
  have hU3 : 3*U ≤ S*D := by dsimp [U]; omega
  have hDF : 0 < D*F := Int.mul_pos hD hF
  have hB : 0 < B := Int.mul_pos (Int.mul_pos hS hD) hF
  have hfirst : 9*E*S*C ≤ 2*B := by
    calc
      9*E*S*C = (E*S)*(9*C) := by grobner
      _ ≤ (E*S)*F := Int.mul_le_mul_of_nonneg_left hratio
        (Int.mul_nonneg hE.1 (by omega))
      _ = E*(S*F) := by grobner
      _ ≤ (2*D)*(S*F) := Int.mul_le_mul_of_nonneg_right hE.2
        (Int.mul_nonneg (by omega) (by omega))
      _ = 2*B := by grobner
  have hsecond : 9*U*Q ≤ 6*B := by
    calc
      9*U*Q ≤ 9*U*(2*F) := Int.mul_le_mul_of_nonneg_left hQ.2 (by omega)
      _ = (3*U)*(6*F) := by grobner
      _ ≤ (S*D)*(6*F) := Int.mul_le_mul_of_nonneg_right hU3 (by omega)
      _ = 6*B := by grobner
  have hthird : 9*R*D*F ≤ 9*B := by
    calc
      9*R*D*F = R*(9*D*F) := by grobner
      _ ≤ S*(9*D*F) := Int.mul_le_mul_of_nonneg_right (by omega : R ≤ S)
        (by have h := Int.mul_nonneg (by omega : 0 ≤ 9*D) (by omega : 0 ≤ F); exact h)
      _ = 9*B := by grobner
  have he : (N*C-u*(D*F))*S=E*S*C+U*Q+R*D*F := by
    dsimp [E,U,Q,R]
    grobner
  have he9 : (9*(N*C-u*(D*F)))*S=9*E*S*C+9*U*Q+9*R*D*F := by
    have h := congrArg (fun x:Int => 9*x) he
    grobner
  have hscaled : (9*(N*C-u*(D*F)))*S ≤ (17*(D*F))*S := by
    rw [he9]
    have hr : (17*(D*F))*S=17*B := by grobner
    rw [hr]
    omega
  have hupper := Int.le_of_mul_le_mul_right hscaled hS
  have hnonneg : 0 ≤ (N*C-u*(D*F))*S := by
    rw [he]
    have h1 := Int.mul_nonneg (Int.mul_nonneg hE.1 (by omega : 0 ≤ S)) hC
    have h2 := Int.mul_nonneg hU hQ.1
    have h3 := Int.mul_nonneg (Int.mul_nonneg htdiv.2.1 (by omega : 0 ≤ D)) (by omega : 0 ≤ F)
    omega
  have hlower : 0 ≤ N*C-u*(D*F) :=
    Int.le_of_mul_le_mul_right (by simpa using hnonneg) hS
  exact ⟨htdiv.1,hlower,by omega⟩


theorem tr_interval_log_argument (a : fval) (p : Int) :
    tr_interval 24 a p =
      let (lo,hi) := log_iv a.mag a.e p
      (lo,hi,-p) := by simp [tr_interval]

end LogarithmProofs
