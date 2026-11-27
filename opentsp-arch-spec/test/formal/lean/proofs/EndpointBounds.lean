-- Actual transcendental endpoint magnitude transfer to the bounded ilog2.
import Proofs.DotBounds
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
set_option exponentiation.threshold 100000
namespace EndpointBoundProofs
open IntervalProofs

theorem exp_step_sum_growth (state : ExpState) (y : Int) (half : Nat)
    (hh : 0 < half) (hy : y.natAbs ≤ half) :
    (expStep (2*(half:Int)) y state).2.2.1.natAbs ≤ state.2.2.1.natAbs+state.2.2.2.natAbs := by
  rcases state with ⟨err,k,sum,term⟩
  by_cases hz : term=0
  · simp [expStep,hz]
  · have hg : term > 0 ∨ term < 0 := by omega
    have ht := taylor_step_halves term y k half hh hy
    have h := Nat.le_trans ht (Nat.div_le_self term.natAbs 2)
    have hs := Int.natAbs_add_le sum (((term*y).tdiv (2*(half:Int))).tdiv k)
    have hb := Nat.add_le_add_left h sum.natAbs
    simpa [expStep,hg] using Nat.le_trans hs hb

theorem exp_repeat_sum_bound (steps : Nat) (state : ExpState) (y : Int) (half S : Nat)
    (hh : 0 < half) (hy : y.natAbs ≤ half) (ht : state.2.2.2.natAbs ≤ S) :
    (repeatStep (expStep (2*(half:Int)) y) steps state).2.2.1.natAbs ≤
      state.2.2.1.natAbs+steps*S := by
  induction steps generalizing state with
  | zero => simp [repeatStep]
  | succ steps ih =>
    have hterm := exp_step_halves state y half hh hy
    have hnext : (expStep (2*(half:Int)) y state).2.2.2.natAbs ≤ S :=
      Nat.le_trans hterm (Nat.le_trans (Nat.div_le_self _ _) ht)
    have hsum := exp_step_sum_growth state y half hh hy
    have h := ih (expStep (2*(half:Int)) y state) hnext
    simp only [repeatStep]
    rw [Nat.add_mul]
    simp only [Nat.one_mul]
    omega

theorem exp_state_sum_bound (y : Int) (p : Nat) (hp : 1 ≤ p) (hy : 2*y.natAbs ≤ 2^p) :
    (expState y p).2.2.1.natAbs ≤ (p+2)*2^p := by
  let half : Nat := 2^(p-1)
  have hh : 0 < half := Nat.pow_pos (by decide)
  have hpow : 2^p=2*half := by
    have h := Nat.pow_succ 2 (p-1)
    have he : p-1+1=p := by omega
    change 2^(p-1+1)=2^(p-1)*2 at h
    rw [he] at h
    dsimp [half]; omega
  have hone : pow2i (p:Int)=2*(half:Int) := by
    rw [NumericalProofs.pow2i_nonnegative (p:Int) (by omega)]
    simp only [Int.toNat_natCast]
    have h := congrArg (fun n : Nat => (n:Int)) hpow
    simpa only [Int.natCast_pow,Int.natCast_mul] using h
  have hnat : (pow2i (p:Int)).natAbs=2^p := by
    rw [NumericalProofs.pow2i_nonnegative (p:Int) (by omega)]
    simp
  have h := exp_repeat_sum_bound (p+1) (2,1,pow2i p,pow2i p) y half (2^p) hh
    (by omega) (by rw [hnat]; exact Nat.le_refl _)
  rw [exp_state_iteration]
  have hc : ((p:Int)+1).toNat=p+1 := by omega
  rw [hc]
  rw [←hone] at h
  simp only [hnat] at h
  have he : 2^p+(p+1)*2^p=(p+2)*2^p := by
    calc
      _ = (1+(p+1))*2^p := by
        simpa only [Nat.one_mul] using (Nat.add_mul 1 (p+1) (2^p)).symm
      _ = (p+2)*2^p := by rw [show 1+(p+1)=p+2 by omega]
  rw [he] at h
  exact h

theorem exp_small_endpoint_budget (y : Int) (p : Nat) (hp : 1 ≤ p)
    (hy : 2*y.natAbs ≤ 2^p) :
    (exp_small y p).1.natAbs ≤ (p+2)*2^p+2*(p+2) ∧
      (exp_small y p).2.natAbs ≤ (p+2)*2^p+2*(p+2) := by
  have hs := exp_state_sum_bound y p hp hy
  have hc := exp_state_counters y p
  have hk := PositivityProofs.exp_index_upper y p
  have herr : (expState y p).1.natAbs ≤ 2*(p+2) := by omega
  have hlo := Int.natAbs_add_le (expState y p).2.2.1 (-(expState y p).1)
  simp only [Int.natAbs_neg] at hlo
  have hhi := Int.natAbs_add_le (expState y p).2.2.1 (expState y p).1
  rw [exp_small_production_loop]
  dsimp only
  rw [Int.sub_eq_add_neg]
  exact ⟨Nat.le_trans hlo (Nat.add_le_add hs herr),Nat.le_trans hhi (Nat.add_le_add hs herr)⟩

theorem exp_small_endpoint_coarse (y : Int) (p : Nat) (hp : 1 ≤ p) (hcap : p ≤ 4096)
    (hy : 2*y.natAbs ≤ 2^p) :
    (exp_small y p).1.natAbs < 2^8192 ∧ (exp_small y p).2.natAbs < 2^8192 := by
  have h := exp_small_endpoint_budget y p hp hy
  have hpow : (2:Nat)^p ≤ 2^4096 := Nat.pow_le_pow_of_le (by decide) hcap
  have hm := Nat.mul_le_mul (show p+2 ≤ 4098 by omega) (Nat.add_le_add_right hpow 2)
  rw [Nat.mul_add] at hm
  have hc : 4098*(2^4096+2) < (2:Nat)^8192 := by decide +kernel
  omega

theorem exp_iv_endpoint_coarse (x : ival) (p : Nat) (hxp : x.p=(p:Int))
    (hp : 1 ≤ p) (hcap : p ≤ 4096)
    (hlo : 2*(RangeReductionProofs.reduction x).1.natAbs ≤ 2^p)
    (hhi : 2*(RangeReductionProofs.reduction x).2.1.natAbs ≤ 2^p) :
    (exp_iv x).1.natAbs < 2^8192 ∧ (exp_iv x).2.1.natAbs < 2^8192 := by
  have hlo := exp_small_endpoint_coarse (RangeReductionProofs.reduction x).1 p hp hcap hlo
  have hhi := exp_small_endpoint_coarse (RangeReductionProofs.reduction x).2.1 p hp hcap hhi
  have hs := RangeReductionProofs.exp_iv_production_reduction x
  dsimp only at hs
  rw [hxp] at hs
  rw [hs]
  exact ⟨hlo.1,hhi.2⟩

open LogarithmProofs

theorem log_step_growth (state : LogState) (z2 : Int) (half : Nat)
    (hh : 0 < half) (hb : z2.natAbs ≤ half) :
    (logStep (8*(half:Int)) z2 state).2.2.natAbs ≤ state.2.2.natAbs ∧
    (logStep (8*(half:Int)) z2 state).2.1.natAbs ≤ state.2.1.natAbs+state.2.2.natAbs ∧
    state.1 ≤ (logStep (8*(half:Int)) z2 state).1 ∧
    (logStep (8*(half:Int)) z2 state).1 ≤ state.1+1 := by
  rcases state with ⟨j,sum,t⟩
  by_cases hc : t > 0 ∧ j < 2000
  · have ht := product_div_eight t z2 half hh hb
    have hsum := Int.natAbs_add_le sum (t.tdiv (2*j+1))
    have hdiv : (t.tdiv (2*j+1)).natAbs ≤ t.natAbs := by
      rw [Int.natAbs_tdiv]; exact Nat.div_le_self _ _
    have he : (((t >b 0) && (j <b 2000)):Bool)=true := by simp [hc.1,hc.2]
    simp only [logStep,he,ite_true]
    exact ⟨Nat.le_trans ht (Nat.div_le_self _ _),Nat.le_trans hsum (Nat.add_le_add_left hdiv _),by omega,by omega⟩
  · have he : (((t >b 0) && (j <b 2000)):Bool)=false := by simp; omega
    simp [logStep,he]; omega

theorem log_repeat_growth (steps : Nat) (state : LogState) (z2 : Int) (half S : Nat)
    (hh : 0 < half) (hb : z2.natAbs ≤ half) (ht : state.2.2.natAbs ≤ S) :
    (repeatStep (logStep (8*(half:Int)) z2) steps state).2.1.natAbs ≤ state.2.1.natAbs+steps*S ∧
    state.1 ≤ (repeatStep (logStep (8*(half:Int)) z2) steps state).1 ∧
    (repeatStep (logStep (8*(half:Int)) z2) steps state).1 ≤ state.1+(steps:Int) := by
  induction steps generalizing state with
  | zero => simp [repeatStep]
  | succ steps ih =>
    have hg := log_step_growth state z2 half hh hb
    have hi := ih (logStep (8*(half:Int)) z2 state) (Nat.le_trans hg.1 ht)
    simp only [repeatStep]
    rw [Nat.add_mul]
    simp only [Nat.one_mul]
    have hn : ((steps+1:Nat):Int)=(steps:Int)+1 := by omega
    omega

theorem log_state_budget (z z2 : Int) (p : Nat) (hz : z.natAbs ≤ 2^(p+8))
    (hb : 0 ≤ z2 ∧ 9*z2 ≤ pow2i ((p:Int)+8)) :
    (logState z z2 (pow2i ((p:Int)+8))).2.1.natAbs ≤ 2000*2^(p+8) ∧
    0 ≤ (logState z z2 (pow2i ((p:Int)+8))).1 ∧
    (logState z z2 (pow2i ((p:Int)+8))).1 ≤ 2000 := by
  let half : Nat := 2^(p+5)
  have hh : 0 < half := Nat.pow_pos (by decide)
  have hone : pow2i ((p:Int)+8)=8*(half:Int) := by
    rw [NumericalProofs.pow2i_nonnegative ((p:Int)+8) (by omega)]
    rw [show ((p:Int)+8).toNat=p+8 by omega]
    rw [show p+8=3+(p+5) by omega,Int.pow_add]
    simp [half,Int.natCast_pow]
  have hbm : z2.natAbs ≤ half := by rw [hone] at hb; omega
  have h := log_repeat_growth 2000 (0,0,z) z2 half (2^(p+8)) hh hbm hz
  rw [log_state_iteration,hone]
  simpa using h

theorem floor_ceil_abs_le (a b : Int) (hb : 1 ≤ b) :
    (idiv_floor a b).natAbs ≤ a.natAbs ∧ (idiv_ceil a b).natAbs ≤ a.natAbs := by
  have hf := floor_bounds a b (by omega)
  have hc := ceil_bounds a b (by omega)
  have hflo : -((a.natAbs:Int)) ≤ idiv_floor a b := by
    by_cases h : -((a.natAbs:Int)) ≤ idiv_floor a b
    · exact h
    · exfalso
      have hm := Int.mul_le_mul_of_nonpos_left (show idiv_floor a b+1 ≤ 0 by omega) hb
      omega
  have hfhi : idiv_floor a b ≤ (a.natAbs:Int) := by
    by_cases h : idiv_floor a b ≤ (a.natAbs:Int)
    · exact h
    · exfalso
      have hm := Int.mul_le_mul_of_nonneg_left hb (show 0 ≤ idiv_floor a b by omega)
      omega
  have hclo : -((a.natAbs:Int)) ≤ idiv_ceil a b := by
    by_cases h : -((a.natAbs:Int)) ≤ idiv_ceil a b
    · exact h
    · exfalso
      have hm := Int.mul_le_mul_of_nonpos_left (show idiv_ceil a b ≤ 0 by omega) hb
      omega
  have hchi : idiv_ceil a b ≤ (a.natAbs:Int) := by
    by_cases h : idiv_ceil a b ≤ (a.natAbs:Int)
    · exact h
    · exfalso
      have hm := Int.mul_le_mul_of_nonneg_left hb (show 0 ≤ idiv_ceil a b-1 by omega)
      omega
  omega

theorem shifted_ln2_size (p : Int) (hp : ZivDomainProofs.Active p) :
    (ln2_iv (p+8)).1.natAbs ≤ (pow2i (p+8)).natAbs ∧
    (ln2_iv (p+8)).2.natAbs ≤ (pow2i (p+8)).natAbs := by
  rcases hp with h|h|h|h|h|h|h <;> subst p <;> decide +kernel

theorem log_post_budget (k j sum L H : Int) (S : Nat)
    (hk : k.natAbs ≤ 256) (hj : 0 ≤ j ∧ j ≤ 2000)
    (hs : sum.natAbs ≤ 2000*S) (hL : L.natAbs ≤ S) (hH : H.natAbs ≤ S) :
    (idiv_floor ((if ((k ≥b 0):Bool) then k*L else k*H)+2*sum-4*(j+3)) 256).natAbs ≤ 4256*S+8012 ∧
    (idiv_ceil ((if ((k ≥b 0):Bool) then k*H else k*L)+2*sum+4*(j+3)) 256).natAbs ≤ 4256*S+8012 := by
  have hl : (if ((k ≥b 0):Bool) then k*L else k*H).natAbs ≤ 256*S := by
    split <;> simp only [Int.natAbs_mul] <;> exact Nat.mul_le_mul hk (by assumption)
  have hh : (if ((k ≥b 0):Bool) then k*H else k*L).natAbs ≤ 256*S := by
    split <;> simp only [Int.natAbs_mul] <;> exact Nat.mul_le_mul hk (by assumption)
  have hsum : (2*sum).natAbs ≤ 4000*S := by
    simp only [Int.natAbs_mul]; change 2*sum.natAbs ≤ 4000*S; have h := Nat.mul_le_mul_left 2 hs; omega
  have herr : (4*(j+3)).natAbs ≤ 8012 := by
    have : 0 ≤ 4*(j+3) ∧ 4*(j+3) ≤ 8012 := by omega
    omega
  have hb1 := Int.natAbs_add_le (if ((k ≥b 0):Bool) then k*L else k*H) (2*sum)
  have hb2 := Int.natAbs_add_le (if ((k ≥b 0):Bool) then k*H else k*L) (2*sum)
  have hb3 := Int.natAbs_sub_le ((if ((k ≥b 0):Bool) then k*L else k*H)+2*sum) (4*(j+3))
  have hb4 := Int.natAbs_add_le ((if ((k ≥b 0):Bool) then k*H else k*L)+2*sum) (4*(j+3))
  have hf := floor_ceil_abs_le ((if ((k ≥b 0):Bool) then k*L else k*H)+2*sum-4*(j+3)) 256 (by decide)
  have hc := floor_ceil_abs_le ((if ((k ≥b 0):Bool) then k*H else k*L)+2*sum+4*(j+3)) 256 (by decide)
  omega

def logPost (k : Int) (state : LogState) (ln : Int × Int) : Int × Int :=
  let (j,sum,_t) := state
  let (L,H) := ln
  (idiv_floor ((if ((k ≥b 0):Bool) then k*L else k*H)+2*sum-4*(j+3)) 256,
   idiv_ceil ((if ((k ≥b 0):Bool) then k*H else k*L)+2*sum+4*(j+3)) 256)

theorem log_post_source (mag e p : Int) :
    log_iv mag e p=
      let c := pow2i (ilog2 mag)
      let S := pow2i (p+8)
      let z := ((mag-c)*S).tdiv (mag+c)
      let z2 := (z*z).tdiv S
      logPost (ilog2 mag+e) (logState z z2 S) (ln2_iv (p+8)) := by
  rw [log_iv_production]
  rfl

theorem log_post_state_budget (k : Int) (state : LogState) (ln : Int × Int) (S : Nat)
    (hk : k.natAbs ≤ 256) (hj : 0 ≤ state.1 ∧ state.1 ≤ 2000)
    (hs : state.2.1.natAbs ≤ 2000*S) (hL : ln.1.natAbs ≤ S) (hH : ln.2.natAbs ≤ S) :
    (logPost k state ln).1.natAbs ≤ 4256*S+8012 ∧ (logPost k state ln).2.natAbs ≤ 4256*S+8012 := by
  rcases state with ⟨j,sum,t⟩
  rcases ln with ⟨L,H⟩
  exact log_post_budget k j sum L H S hk hj hs hL hH

set_option maxHeartbeats 20000 in
theorem log_iv_endpoint_coarse (mag e p : Int) (hm : 1 ≤ mag ∧ mag < 16777216)
    (he : -149 ≤ e ∧ e ≤ 120) (hp : ZivDomainProofs.Active p) :
    (log_iv mag e p).1.natAbs < 2^8192 ∧ (log_iv mag e p).2.natAbs < 2^8192 := by
  have hpp : 0 ≤ p ∧ p ≤ 4096 := by dsimp [ZivDomainProofs.Active] at hp; omega
  have hmcap : mag < pow2i 16384 := by
    have h24 : pow2i 24=16777216 := by decide
    have hmon := GridBoundProofs.pow2_mono 24 16384 (by decide) (by decide)
    omega
  have hb := BinadeProofs.ilog2_binade mag hm.1 hmcap
  have hb24 : ilog2 mag < 24 := by
    by_cases h : ilog2 mag < 24
    · exact h
    · have hmon := GridBoundProofs.pow2_mono 24 (ilog2 mag) (by decide) (by omega)
      have h24 : pow2i 24=16777216 := by decide
      omega
  let c := pow2i (ilog2 mag)
  let S := pow2i (p+8)
  let z := ((mag-c)*S).tdiv (mag+c)
  let z2 := (z*z).tdiv S
  have hc : 0 < c := IntervalProofs.pow2i_positive _ hb.1
  have htwoc : pow2i (ilog2 mag+1)=2*c := by
    have h := NumericalProofs.pow2i_add_nonnegative (ilog2 mag) 1 hb.1 (by decide)
    have ht : pow2i 1=2 := by decide
    rw [ht] at h; change _=2*c; rw [h]; omega
  have hn := normalized_ratio mag c hc hb.2.1 (by rw [htwoc] at hb; omega)
  have hS : 0 < S := IntervalProofs.pow2i_positive _ (by omega)
  have hz := normalized_z_domain (mag-c) (mag+c) S hn.1 hn.2.1 hS hn.2.2
  change 0 ≤ z ∧ 3*z ≤ S ∧ _ at hz
  have hz2 := squared_coefficient_domain z S hz.1 hS hz.2.1
  change 0 ≤ z2 ∧ 9*z2 ≤ S at hz2
  have hsnat : S.natAbs=2^(p.toNat+8) := by
    dsimp [S]; rw [NumericalProofs.pow2i_nonnegative _ (by omega)]
    simp [show (p+8).toNat=p.toNat+8 by omega]
  have hstate := log_state_budget z z2 p.toNat (by rw [←hsnat]; omega) (by simpa [show (p.toNat:Int)=p by omega] using hz2)
  have hpow : pow2i ((p.toNat:Int)+8)=S := by dsimp [S]; congr 1; omega
  rw [hpow] at hstate
  have hl := shifted_ln2_size p hp
  change (ln2_iv (p+8)).1.natAbs ≤ S.natAbs ∧ (ln2_iv (p+8)).2.natAbs ≤ S.natAbs at hl
  have hpost := log_post_state_budget (ilog2 mag+e) (logState z z2 S)
    (ln2_iv (p+8)) (2^(p.toNat+8)) (by omega) hstate.2 hstate.1
    (by rw [←hsnat]; exact hl.1) (by rw [←hsnat]; exact hl.2)
  have hbound : 4256*(2^(p.toNat+8))+8012 < (2:Nat)^8192 := by
    have hm := Nat.pow_le_pow_of_le (by decide : 1 < (2:Nat)) (show p.toNat+8 ≤ 4104 by omega)
    have hc : 4256*(2^4104)+8012 < (2:Nat)^8192 := by decide +kernel
    omega
  have hsrc : log_iv mag e p=logPost (ilog2 mag+e) (logState z z2 S) (ln2_iv (p+8)) :=
    log_post_source mag e p
  rw [hsrc]
  exact ⟨Nat.lt_of_le_of_lt hpost.1 hbound,Nat.lt_of_le_of_lt hpost.2 hbound⟩

theorem positive_division_budget (N D C : Int) (hN : 0 ≤ N) (hD : 0 < D)
    (hC : 0 ≤ C) (hprod : N ≤ C*D) :
    0 ≤ idiv_floor N D ∧ idiv_floor N D ≤ C ∧
    0 ≤ idiv_ceil N D ∧ idiv_ceil N D ≤ C := by
  have hn := PositivityProofs.floor_ceil_nonnegative N D hN hD
  have hf := floor_bounds N D hD
  have hc := ceil_bounds N D hD
  have hfl : idiv_floor N D ≤ C := Int.le_of_mul_le_mul_right (Int.le_trans hf.1 hprod) hD
  have hcl : idiv_ceil N D ≤ C := by
    by_cases h : idiv_ceil N D ≤ C
    · exact h
    · have hm := Int.mul_le_mul_of_nonneg_right (show C ≤ idiv_ceil N D-1 by omega) (by omega : 0 ≤ D)
      omega
  exact ⟨hn.1,hfl,hn.2,hcl⟩

theorem tanh_post_endpoint_budget (lo hi n p : Int) (neg : Bool)
    (hlo : 0 ≤ lo) (hhi : 0 ≤ hi) :
    (RangeReductionProofs.tanhPost neg p (lo,hi,n)).1.natAbs ≤ (pow2i (2*p)).natAbs+1 ∧
    (RangeReductionProofs.tanhPost neg p (lo,hi,n)).2.1.natAbs ≤ (pow2i (2*p)).natAbs+1 := by
  let S := pow2i p
  let B := pow2i (2*p)
  let u := if n ≥ 0 then lo*pow2i n else idiv_floor lo (pow2i (-n))
  let v := if n ≥ 0 then hi*pow2i n else idiv_ceil hi (pow2i (-n))
  have hpos := PositivityProofs.scaled_exp_nonnegative lo hi n hlo hhi
  change 0 ≤ u ∧ 0 ≤ v at hpos
  have hS : 0 < S := PositivityProofs.pow2i_any_positive p
  have hB : 0 < B := PositivityProofs.pow2i_any_positive (2*p)
  have hN : 0 ≤ 2*B*S := Int.mul_nonneg (by omega) (by omega)
  have hup : 2*B*S ≤ (2*B)*(u+S) := by
    have h := Int.mul_le_mul_of_nonneg_left (show S ≤ u+S by omega) (show 0 ≤ 2*B by omega)
    exact h
  have hvp : 2*B*S ≤ (2*B)*(v+S) := by
    exact Int.mul_le_mul_of_nonneg_left (show S ≤ v+S by omega) (show 0 ≤ 2*B by omega)
  have hu := positive_division_budget (2*B*S) (u+S) (2*B) hN (by omega) (by omega) hup
  have hv := positive_division_budget (2*B*S) (v+S) (2*B) hN (by omega) (by omega) hvp
  cases neg
  · change (B - idiv_ceil (2*B*S) (u+S)-1).natAbs ≤ B.natAbs+1 ∧
      (B - idiv_floor (2*B*S) (v+S)+1).natAbs ≤ B.natAbs+1
    omega
  · change (-(B - idiv_floor (2*B*S) (v+S))-1).natAbs ≤ B.natAbs+1 ∧
      (-(B - idiv_ceil (2*B*S) (u+S))+1).natAbs ≤ B.natAbs+1
    omega

theorem tanh_post_endpoint_coarse (lo hi n p : Int) (neg : Bool)
    (hlo : 0 ≤ lo) (hhi : 0 ≤ hi) (hp : 0 ≤ p ∧ p ≤ 4096) :
    (RangeReductionProofs.tanhPost neg p (lo,hi,n)).1.natAbs < 2^16384 ∧
    (RangeReductionProofs.tanhPost neg p (lo,hi,n)).2.1.natAbs < 2^16384 := by
  have h := tanh_post_endpoint_budget lo hi n p neg hlo hhi
  have hm := GridBoundProofs.pow2_mono (2*p) 8192 (by omega) (by omega)
  have hc : (pow2i 8192).natAbs+1 < (2:Nat)^16384 := by decide +kernel
  have hb : (pow2i (2*p)).natAbs ≤ (pow2i 8192).natAbs := by
    have h1 := IntervalProofs.pow2i_positive (2*p) (by omega)
    have h2 := IntervalProofs.pow2i_positive 8192 (by decide)
    omega
  omega

theorem certified_root_scale_lower (n s B : Int) (hB : 1 ≤ B) (hs : 0 ≤ s)
    (hn : B*B ≤ n) (hupper : n < (s+1)*(s+1)) : B ≤ s := by
  by_cases h : B ≤ s
  · exact h
  · have hle : s+1 ≤ B := by omega
    have hm1 := Int.mul_le_mul_of_nonneg_left hle (show 0 ≤ s+1 by omega)
    have hm2 := Int.mul_le_mul_of_nonneg_right hle (show 0 ≤ B by omega)
    have hm : (s+1)*(s+1) ≤ B*B := Int.le_trans hm1 hm2
    omega

theorem root_post_endpoint_budget (mag e p : Int) (hm : 1 ≤ RootEnclosureProofs.doubledMagnitude mag e)
    (hp : 0 ≤ p)
    (hs : 0 ≤ isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p)))
    (hroot : RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p) <
      (isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p))+1)*
      (isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p))+1)) :
    (RootEnclosureProofs.rootPost mag e p).1.natAbs ≤ (pow2i (3*p)).natAbs ∧
    (RootEnclosureProofs.rootPost mag e p).2.1.natAbs ≤ (pow2i (3*p)).natAbs+1 := by
  let B := pow2i p
  let s := isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p))
  let N := pow2i (4*p)
  let C := pow2i (3*p)
  have hB : 0 < B := IntervalProofs.pow2i_positive p hp
  have hC : 0 < C := IntervalProofs.pow2i_positive (3*p) (by omega)
  have hN : 0 < N := IntervalProofs.pow2i_positive (4*p) (by omega)
  have hpow := RootEnclosureProofs.root_scale_powers p hp
  have hn : B*B ≤ RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p) := by
    have hm2 := Int.mul_le_mul_of_nonneg_right hm (show 0 ≤ pow2i (2*p) from Int.le_of_lt (IntervalProofs.pow2i_positive _ (by omega)))
    have he : B*B=pow2i (2*p) := hpow.1.symm
    rw [he]
    simpa using hm2
  have hslow : B ≤ s := certified_root_scale_lower _ s B (by omega) hs hn hroot
  have hNC : N=C*B := by
    have h := NumericalProofs.pow2i_add_nonnegative (3*p) p (by omega) hp
    rw [show 3*p+p=4*p by omega] at h
    exact h
  have hNs : N ≤ C*s := by rw [hNC]; exact Int.mul_le_mul_of_nonneg_left hslow (by omega)
  have hNs1 : N ≤ C*(s+1) := by
    exact Int.le_trans hNs (Int.mul_le_mul_of_nonneg_left (show s ≤ s+1 by omega) (by omega))
  have hdiv := nonnegative_tdiv_bounds N s (by omega) (by omega)
  have hdiv1 := nonnegative_tdiv_bounds N (s+1) (by omega) (by omega)
  have hu : N.tdiv s ≤ C := Int.le_of_mul_le_mul_right (by omega : N.tdiv s*s ≤ C*s) (by omega)
  have hl : N.tdiv (s+1) ≤ C := Int.le_of_mul_le_mul_right (by omega : N.tdiv (s+1)*(s+1) ≤ C*(s+1)) (by omega)
  change (N.tdiv (s+1)).natAbs ≤ C.natAbs ∧ (N.tdiv s+1).natAbs ≤ C.natAbs+1
  omega

theorem root_post_endpoint_coarse (mag e p : Int) (hm : 1 ≤ RootEnclosureProofs.doubledMagnitude mag e)
    (hp : 0 ≤ p ∧ p ≤ 4096)
    (hs : 0 ≤ isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p)))
    (hroot : RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p) <
      (isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p))+1)*
      (isqrt (RootEnclosureProofs.doubledMagnitude mag e*pow2i (2*p))+1)) :
    (RootEnclosureProofs.rootPost mag e p).1.natAbs < 2^16384 ∧
    (RootEnclosureProofs.rootPost mag e p).2.1.natAbs < 2^16384 := by
  have h := root_post_endpoint_budget mag e p hm hp.1 hs hroot
  have hm := GridBoundProofs.pow2_mono (3*p) 12288 (by omega) (by omega)
  have hc : (pow2i 12288).natAbs+1 < (2:Nat)^16384 := by decide +kernel
  have hb : (pow2i (3*p)).natAbs ≤ (pow2i 12288).natAbs := by
    have h1 := IntervalProofs.pow2i_positive (3*p) (by omega)
    have h2 := IntervalProofs.pow2i_positive 12288 (by decide)
    omega
  omega

theorem tr_tanh_endpoint_coarse (a : fval) (p : Nat)
    (hp : p=64 ∨ p=128 ∨ p=256 ∨ p=512 ∨ p=1024 ∨ p=2048 ∨ p=4096)
    (hlo : 2*(RangeReductionProofs.reduction (fx_of false a.mag (a.e+1) p)).1.natAbs ≤ 2^p)
    (hhi : 2*(RangeReductionProofs.reduction (fx_of false a.mag (a.e+1) p)).2.1.natAbs ≤ 2^p) :
    (tr_interval 21 a p).1.natAbs < 2^16384 ∧ (tr_interval 21 a p).2.1.natAbs < 2^16384 := by
  have hxp : (fx_of false a.mag (a.e+1) p).p=(p:Int) := by simp [fx_of]; split <;> rfl
  have hpos := PositivityProofs.exp_iv_positive (fx_of false a.mag (a.e+1) p) p hxp hp hlo hhi
  rcases hexp : exp_iv (fx_of false a.mag (a.e+1) p) with ⟨lo,hi,n⟩
  rw [hexp] at hpos
  change 0 < lo ∧ 0 < hi at hpos
  rw [RangeReductionProofs.tr_interval_tanh_argument,hexp]
  exact tanh_post_endpoint_coarse lo hi n p a.neg (by omega) (by omega) (by omega)

theorem tr_exp_endpoint_coarse (a : fval) (p : Nat) (hp : 1 ≤ p ∧ p ≤ 4096)
    (hlo : 2*(RangeReductionProofs.reduction (fx_of a.neg a.mag a.e p)).1.natAbs ≤ 2^p)
    (hhi : 2*(RangeReductionProofs.reduction (fx_of a.neg a.mag a.e p)).2.1.natAbs ≤ 2^p) :
    (tr_interval 22 a p).1.natAbs < 2^8192 ∧ (tr_interval 22 a p).2.1.natAbs < 2^8192 := by
  have hxp : (fx_of a.neg a.mag a.e p).p=(p:Int) := by simp [fx_of]; split <;> rfl
  have h := exp_iv_endpoint_coarse (fx_of a.neg a.mag a.e p) p hxp hp.1 hp.2 hlo hhi
  rcases hexp : exp_iv (fx_of a.neg a.mag a.e p) with ⟨lo,hi,n⟩
  rw [hexp] at h
  rw [RangeReductionProofs.tr_interval_exp_argument,hexp]
  exact h

theorem tr_log_endpoint_coarse (a : fval) (p : Int) (hm : 1 ≤ a.mag ∧ a.mag < 16777216)
    (he : -149 ≤ a.e ∧ a.e ≤ 120) (hp : ZivDomainProofs.Active p) :
    (tr_interval 24 a p).1.natAbs < 2^8192 ∧ (tr_interval 24 a p).2.1.natAbs < 2^8192 := by
  have h := log_iv_endpoint_coarse a.mag a.e p hm he hp
  simpa [tr_interval] using h

theorem tr_rsqrt_endpoint_coarse (a : fval) (p : Int) (hm : 1 ≤ a.mag ∧ a.mag < 16777216)
    (hp : 0 ≤ p ∧ p ≤ 4096)
    (hs : 0 ≤ isqrt (RootEnclosureProofs.doubledMagnitude a.mag a.e*pow2i (2*p)))
    (hroot : RootEnclosureProofs.doubledMagnitude a.mag a.e*pow2i (2*p) <
      (isqrt (RootEnclosureProofs.doubledMagnitude a.mag a.e*pow2i (2*p))+1)*
      (isqrt (RootEnclosureProofs.doubledMagnitude a.mag a.e*pow2i (2*p))+1)) :
    (tr_interval 23 a p).1.natAbs < 2^16384 ∧ (tr_interval 23 a p).2.1.natAbs < 2^16384 := by
  rw [RootEnclosureProofs.tr_interval_rsqrt_source]
  exact root_post_endpoint_coarse a.mag a.e p (RootEnclosureProofs.root_argument_domain a.mag a.e hm).1 hp hs hroot

end EndpointBoundProofs
