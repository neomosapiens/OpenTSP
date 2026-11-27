-- Exact rational power and partial-sum invariants for actual Log recurrence.
import Proofs.Logarithm
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000
namespace LogSeriesProofs
open IntervalProofs LogarithmProofs

structure SeriesData where
  pden : Int
  pnum : Int
  sden : Int
  snum : Int

def seriesData (A D S : Int) : Nat → SeriesData
  | 0 => ⟨D,S*A,1,0⟩
  | n+1 =>
    let old := seriesData A D S n
    let k : Int := 2*(n:Int)+1
    ⟨old.pden*(D*D),old.pnum*(A*A),old.sden*(old.pden*k),
      old.snum*(old.pden*k)+old.pnum*old.sden⟩

theorem ratio_square_bound (A D : Int) (hA : 0 ≤ A) (hr : 3*A ≤ D) :
    0 ≤ A*A ∧ 9*(A*A) ≤ D*D := by
  have hD : 0 ≤ D := by omega
  have h1 := Int.mul_le_mul_of_nonneg_right hr hA
  have h2 := Int.mul_le_mul_of_nonneg_right hr hD
  have he1 : (3*A)*A=3*(A*A) := by grobner
  have he2 : (3*A)*D=3*(D*A) := by grobner
  rw [he1] at h1
  rw [he2] at h2
  exact ⟨Int.mul_nonneg hA hA,by omega⟩

theorem series_data_properties (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) (n : Nat) :
    0 < (seriesData A D S n).pden ∧ 0 < (seriesData A D S n).sden ∧
    0 ≤ (seriesData A D S n).pnum ∧
      3*(seriesData A D S n).pnum ≤ S*(seriesData A D S n).pden := by
  induction n with
  | zero =>
    have hm := Int.mul_le_mul_of_nonneg_right hr (by omega : 0 ≤ S)
    have he1 : (3*A)*S=3*(S*A) := by grobner
    have he2 : D*S=S*D := by grobner
    rw [he1,he2] at hm
    exact ⟨hD,by change (0:Int) < 1; decide,Int.mul_nonneg (by omega) hA,hm⟩
  | succ n ih =>
    have hs := ratio_square_bound A D hA hr
    have hF : 0 < D*D := Int.mul_pos hD hD
    have hk : 0 < 2*(n:Int)+1 := by omega
    have hsize : 3*((seriesData A D S n).pnum*(A*A)) ≤
        S*((seriesData A D S n).pden*(D*D)) := by
      calc
        3*((seriesData A D S n).pnum*(A*A)) =
            (3*(seriesData A D S n).pnum)*(A*A) := by grobner
        _ ≤ (S*(seriesData A D S n).pden)*(A*A) :=
          Int.mul_le_mul_of_nonneg_right ih.2.2.2 hs.1
        _ ≤ (S*(seriesData A D S n).pden)*(D*D) :=
          Int.mul_le_mul_of_nonneg_left (by omega : A*A ≤ D*D)
            (Int.mul_nonneg (by omega) (by omega))
        _ = S*((seriesData A D S n).pden*(D*D)) := by grobner
    exact ⟨Int.mul_pos ih.1 hF,Int.mul_pos ih.2.1 (Int.mul_pos ih.1 hk),
      Int.mul_nonneg ih.2.2.1 hs.1,hsize⟩

/-- Dividing an integer term by the positive odd index adds at most one unit;
    the exact integer remainder makes the total strictly below two units. -/
theorem divided_term_error (t N D k : Int) (ht : 0 ≤ t) (hD : 0 < D)
    (hk : 1 ≤ k) (he : 0 ≤ N-t*D ∧ N-t*D < 2*D) :
    let v := t.tdiv k
    0 ≤ v ∧ 0 ≤ N-v*(D*k) ∧ N-v*(D*k) < 2*(D*k) := by
  let v := t.tdiv k
  change 0 ≤ v ∧ 0 ≤ N-v*(D*k) ∧ N-v*(D*k) < 2*(D*k)
  have hr := nonnegative_tdiv_bounds t k ht (by omega)
  change 0 ≤ v ∧ 0 ≤ t-v*k ∧ t-v*k < k at hr
  have hrem := Int.mul_le_mul_of_nonneg_right
    (show t-v*k ≤ k-1 by omega) (by omega : 0 ≤ D)
  have hnonneg := Int.mul_nonneg hr.2.1 (by omega : 0 ≤ D)
  have hid : N-v*(D*k)=(N-t*D)+(t-v*k)*D := by grobner
  have htop : (k-1)*D+2*D=(k+1)*D := by grobner
  have hbound := Int.mul_le_mul_of_nonneg_right (show k+1 ≤ 2*k by omega) (by omega : 0 ≤ D)
  have heq : (2*k)*D=2*(D*k) := by grobner
  rw [heq] at hbound
  rw [hid]
  exact ⟨hr.1,by omega,by omega⟩

/-- Exact common-denominator accumulation of one divided term. -/
theorem partial_sum_error_step (sum v oldNum termNum oldDen termDen k : Int) (n : Nat)
    (hD : 0 < oldDen) (hT : 0 < termDen) (hk : 0 < k)
    (hold : 0 ≤ oldNum-sum*oldDen ∧ oldNum-sum*oldDen ≤ 2*(n:Int)*oldDen)
    (hterm : 0 ≤ termNum-v*(termDen*k) ∧ termNum-v*(termDen*k) < 2*(termDen*k)) :
    let den := oldDen*(termDen*k)
    let num := oldNum*(termDen*k)+termNum*oldDen
    0 ≤ num-(sum+v)*den ∧ num-(sum+v)*den ≤ 2*((n+1:Nat):Int)*den := by
  let E := oldNum-sum*oldDen
  let Q := termNum-v*(termDen*k)
  have hTk : 0 < termDen*k := Int.mul_pos hT hk
  have h1 := Int.mul_le_mul_of_nonneg_right hold.2 (by omega : 0 ≤ termDen*k)
  have h2 := Int.mul_le_mul_of_nonneg_right (show Q ≤ 2*(termDen*k) by dsimp [Q]; omega) (by omega : 0 ≤ oldDen)
  have hn1 := Int.mul_nonneg hold.1 (by omega : 0 ≤ termDen*k)
  have hn2 := Int.mul_nonneg hterm.1 (by omega : 0 ≤ oldDen)
  have hid : oldNum*(termDen*k)+termNum*oldDen-(sum+v)*(oldDen*(termDen*k)) =
      E*(termDen*k)+Q*oldDen := by dsimp [E,Q]; grobner
  have htop : 2*((n+1:Nat):Int)*(oldDen*(termDen*k)) =
      (2*(n:Int)*oldDen)*(termDen*k)+(2*(termDen*k))*oldDen := by
    have hn : ((n+1:Nat):Int)=(n:Int)+1 := by omega
    rw [hn]
    grobner
  dsimp only
  rw [hid,htop]
  dsimp [E,Q] at *
  omega


def SeriesInvariant (A D S : Int) (state : LogState) : Prop :=
  ∃ n : Nat, state.1=(n:Int) ∧ 0 ≤ state.2.2 ∧
    0 ≤ (seriesData A D S n).pnum-state.2.2*(seriesData A D S n).pden ∧
    (seriesData A D S n).pnum-state.2.2*(seriesData A D S n).pden < 2*(seriesData A D S n).pden ∧
    0 ≤ (seriesData A D S n).snum-state.2.1*(seriesData A D S n).sden ∧
    (seriesData A D S n).snum-state.2.1*(seriesData A D S n).sden ≤
      2*(n:Int)*(seriesData A D S n).sden

theorem series_invariant_initial (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) :
    SeriesInvariant A D S (0,0,(A*S).tdiv D) := by
  have hz := normalized_z_domain A D S hA hD hS hr
  refine ⟨0,rfl,hz.1,?_,?_,?_,?_⟩
  · simpa [seriesData,Int.mul_comm] using hz.2.2.1
  · change S*A-((A*S).tdiv D)*D < 2*D
    have he : S*A=A*S := by grobner
    rw [he]; omega
  · simp [seriesData]
  · simp [seriesData]

theorem series_invariant_step (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) (state : LogState)
    (hi : SeriesInvariant A D S state) :
    SeriesInvariant A D S
      (logStep S ((((A*S).tdiv D)*((A*S).tdiv D)).tdiv S) state) := by
  let z := (A*S).tdiv D
  let b := (z*z).tdiv S
  change SeriesInvariant A D S (logStep S b state)
  have hz := normalized_z_domain A D S hA hD hS hr
  have hb := squared_coefficient_domain z S hz.1 hS hz.2.1
  have hc := squared_coefficient_error A D S hA hD hS hr
  have hratio := ratio_square_bound A D hA hr
  rcases state with ⟨j,sum,t⟩
  rcases hi with ⟨n,hj,ht,herr0,herr1,hs0,hs1⟩
  dsimp only at hj ht herr0 herr1 hs0 hs1
  by_cases hg : t > 0 ∧ j < 2000
  · have hnlt : (n:Int) < 2000 := by omega
    have hd := series_data_properties A D S hA hD hS hr n
    have hnext := power_term_error_step t (seriesData A D S n).pnum
      (seriesData A D S n).pden (A*A) (D*D) b S ht hd.1
      (Int.mul_pos hD hD) hS hratio.1 hb.1 hratio.2 hd.2.2.2
      ⟨herr0,herr1⟩ hc
    have hv := divided_term_error t (seriesData A D S n).pnum
      (seriesData A D S n).pden (2*j+1) ht hd.1 (by omega) ⟨herr0,herr1⟩
    have hsum := partial_sum_error_step sum (t.tdiv (2*j+1))
      (seriesData A D S n).snum (seriesData A D S n).pnum
      (seriesData A D S n).sden (seriesData A D S n).pden (2*j+1) n
      hd.2.1 hd.1 (by omega) ⟨hs0,hs1⟩ hv.2
    refine ⟨n+1,?_,?_,?_,?_,?_,?_⟩
    · simp [logStep,hg]; omega
    · simpa [logStep,hg] using hnext.1
    · simpa [logStep,hg,seriesData,hj,hnlt] using hnext.2.1
    · simpa [logStep,hg,seriesData,hj,hnlt] using hnext.2.2
    · simpa [logStep,hg,seriesData,hj,hnlt] using hsum.1
    · simpa [logStep,hg,seriesData,hj,hnlt] using hsum.2
  · simpa [logStep,hg] using (show SeriesInvariant A D S (j,sum,t) from
      ⟨n,hj,ht,herr0,herr1,hs0,hs1⟩)

theorem repeat_series_invariant (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) (steps : Nat) (state : LogState)
    (hi : SeriesInvariant A D S state) :
    SeriesInvariant A D S
      (repeatStep (logStep S ((((A*S).tdiv D)*((A*S).tdiv D)).tdiv S)) steps state) := by
  induction steps generalizing state with
  | zero => exact hi
  | succ steps ih => exact ih _ (series_invariant_step A D S hA hD hS hr state hi)

theorem actual_log_series_invariant (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) :
    let z := (A*S).tdiv D
    let b := (z*z).tdiv S
    SeriesInvariant A D S (logState z b S) := by
  dsimp only
  rw [log_state_iteration]
  exact repeat_series_invariant A D S hA hD hS hr _ _
    (series_invariant_initial A D S hA hD hS hr)



def SeriesCertificate (A D S : Int) (state : LogState) : Prop :=
  ∃ n : Nat, state.1=(n:Int) ∧ state.2.2=0 ∧
      0 < (seriesData A D S n).pden ∧ 0 < (seriesData A D S n).sden ∧
      0 ≤ (seriesData A D S n).pnum ∧
      (seriesData A D S n).pnum < 2*(seriesData A D S n).pden ∧
      (2*state.2.1-4*(state.1+3))*(seriesData A D S n).sden ≤
        2*(seriesData A D S n).snum ∧
      2*(seriesData A D S n).snum ≤
        (2*state.2.1+4*(state.1+3)-12)*(seriesData A D S n).sden

theorem certificate_of_invariant (A D S : Int) (hA : 0 ≤ A) (hD : 0 < D)
    (hS : 0 < S) (hr : 3*A ≤ D) (state : LogState)
    (hi : SeriesInvariant A D S state) (hzero : state.2.2=0) :
    SeriesCertificate A D S state := by
  rcases hi with ⟨n,hj,ht,he0,he1,hs0,hs1⟩
  have hd := series_data_properties A D S hA hD hS hr n
  have hlo : (2*state.2.1-4*(state.1+3))*(seriesData A D S n).sden =
      2*(state.2.1*(seriesData A D S n).sden)-
        (4*(n:Int)+12)*(seriesData A D S n).sden := by rw [hj]; grobner
  have hhi : (2*state.2.1+4*(state.1+3)-12)*(seriesData A D S n).sden =
      2*(state.2.1*(seriesData A D S n).sden)+
        4*(n:Int)*(seriesData A D S n).sden := by rw [hj]; grobner
  have hbound : 4*(n:Int)*(seriesData A D S n).sden =
      2*(2*(n:Int)*(seriesData A D S n).sden) := by grobner
  have hn : 0 ≤ 4*(n:Int)+12 := by omega
  have hnprod := Int.mul_nonneg hn (by omega : 0 ≤ (seriesData A D S n).sden)
  refine ⟨n,hj,hzero,hd.1,hd.2.1,hd.2.2.1,?_,?_,?_⟩
  · rw [hzero] at he1; simpa using he1
  · rw [hlo]; omega
  · rw [hhi]; omega


theorem actual_log_series_certificate (A D : Int) (p : Nat) (hp : p ≤ 4096)
    (hA : 0 ≤ A) (hD : 0 < D) (hr : 3*A ≤ D) :
    let S := pow2i ((p:Int)+8)
    let z := (A*S).tdiv D
    let b := (z*z).tdiv S
    SeriesCertificate A D S (logState z b S) := by
  dsimp only
  have hS : 0 < pow2i ((p:Int)+8) := pow2i_positive _ (by omega)
  have hz := normalized_z_domain A D (pow2i ((p:Int)+8)) hA hD hS hr
  have hb := squared_coefficient_domain ((A*pow2i ((p:Int)+8)).tdiv D)
    (pow2i ((p:Int)+8)) hz.1 hS hz.2.1
  apply certificate_of_invariant A D (pow2i ((p:Int)+8)) hA hD hS hr
  · exact actual_log_series_invariant A D (pow2i ((p:Int)+8)) hA hD hS hr
  · exact log_computed_term_zero _ _ p hp hz.1 (by omega) hb.1 hb.2


theorem series_power_closed_form (A D S : Int) (n : Nat) :
    (seriesData A D S n).pden=D^(2*n+1) ∧
      (seriesData A D S n).pnum=S*A^(2*n+1) := by
  induction n with
  | zero => simp [seriesData,Int.pow_one]
  | succ n ih =>
    have he : 2*(n+1)+1=((2*n+1)+1)+1 := by omega
    change (seriesData A D S n).pden*(D*D)=D^(2*(n+1)+1) ∧
      (seriesData A D S n).pnum*(A*A)=S*A^(2*(n+1)+1)
    rw [ih.1,ih.2,he,Int.pow_succ,Int.pow_succ]
    constructor <;> grobner

end LogSeriesProofs
