-- Source/domain transfer and reciprocal algebra for the actual RSqrt interval.
-- The bounded native root-floor certificate is an explicitly separate leg.
import Proofs.LogSeries
open Sail Out Out.Functions
set_option autoImplicit false
namespace RootEnclosureProofs

def evenExponent (e : Int) := if e.tmod 2=0 then e else e-1
def doubledMagnitude (mag e : Int) := if e.tmod 2=0 then mag else mag*2

def rootPost (mag e p : Int) : Int × Int × Int :=
  let e2 := evenExponent e
  let m2 := doubledMagnitude mag e
  let s := isqrt (m2*pow2i (2*p))
  let num := pow2i (4*p)
  (num.tdiv (s+1),num.tdiv s+1,-3*p-e2.tdiv 2)

theorem tr_interval_rsqrt_source (a : fval) (p : Int) :
    tr_interval 23 a p=rootPost a.mag a.e p := by
  simp [tr_interval,rootPost,evenExponent,doubledMagnitude,Int.neg_mul]

theorem root_argument_domain (mag e : Int) (hm : 1 ≤ mag ∧ mag < 16777216) :
    1 ≤ doubledMagnitude mag e ∧ doubledMagnitude mag e < 33554432 := by
  dsimp [doubledMagnitude]; split <;> omega

theorem i8_root_argument_domain (v e : Int) (hv : 1 ≤ v ∧ v ≤ 128) :
    1 ≤ doubledMagnitude v e ∧ doubledMagnitude v e < 33554432 := by
  apply root_argument_domain; omega

theorem truncated_even_iff (e : Int) : e.tmod 2=0 ↔ e%2=0 := by
  have h0 := Int.emod_nonneg e (by decide : (2:Int)≠0)
  have h1 := Int.emod_lt_of_pos e (by decide : (0:Int)<2)
  rw [Int.tmod_eq_emod]
  split <;> simp_all <;> omega

theorem even_exponent_exact (e : Int) :
    (evenExponent e).tdiv 2*2=evenExponent e ∧
      (evenExponent e=e ∨ evenExponent e=e-1) := by
  have h0 := Int.emod_nonneg e (by decide : (2:Int)≠0)
  have h1 := Int.emod_lt_of_pos e (by decide : (0:Int)<2)
  have he : evenExponent e%2=0 := by
    dsimp [evenExponent]
    split
    · exact (truncated_even_iff e).mp (by assumption)
    · have hn : e%2≠0 := by intro h; exact (by assumption : ¬e.tmod 2=0) ((truncated_even_iff e).mpr h)
      have hr : e%2=1 := by omega
      omega
  have hd : 2 ∣ evenExponent e := Int.dvd_iff_emod_eq_zero.mpr he
  have hr := Int.ediv_mul_add_emod (evenExponent e) 2
  rw [Int.tdiv_eq_ediv_of_dvd hd]
  constructor
  · omega
  · dsimp [evenExponent]; split <;> omega

theorem magnitude_scale_exact (mag e : Int) :
    doubledMagnitude mag e=mag*pow2i (e-evenExponent e) := by
  have h0 : pow2i 0=1 := by decide
  have h1 : pow2i 1=2 := by decide
  dsimp [evenExponent,doubledMagnitude]
  split
  · simp [h0]
  · have he : e-(e-1)=1 := by omega
    rw [he,h1]

theorem root_input_positive (mag e p : Int)
    (hm : 1 ≤ mag ∧ mag < 16777216) (hp : 0 ≤ p) :
    0 < doubledMagnitude mag e*pow2i (2*p) := by
  have h := root_argument_domain mag e hm
  exact Int.mul_pos (by omega) (IntervalProofs.pow2i_positive _ (by omega))

theorem floor_root_positive (n s : Int) (hn : 0 < n) (hs : 0 ≤ s)
    (hupper : n < (s+1)*(s+1)) : 0 < s := by
  by_cases h : 0 < s
  · exact h
  · have he : s=0 := by omega
    rw [he] at hupper
    omega

/-- The production reciprocal endpoints enclose N/r whenever the actual root
    floor s brackets positive r. This is exact rational cross-product algebra;
    the same ordered-real argument applies to sqrt(n). -/
theorem root_reciprocal_rational (N s a d : Int) (hN : 0 < N) (hs : 0 < s)
    (hd : 0 < d) (hlo : s*d ≤ a) (hhi : a ≤ (s+1)*d) :
    let lower := N.tdiv (s+1)
    let upper := N.tdiv s+1
    lower*a ≤ N*d ∧ N*d ≤ upper*a := by
  let lower := N.tdiv (s+1)
  let upper := N.tdiv s+1
  have hl := LogarithmProofs.nonnegative_tdiv_bounds N (s+1) (by omega) (by omega)
  have hu := LogarithmProofs.nonnegative_tdiv_bounds N s (by omega) hs
  have hl0 : 0 ≤ lower := hl.1
  have hu0 : 0 ≤ upper := by dsimp [upper]; omega
  have hlN : lower*(s+1) ≤ N := by dsimp [lower]; omega
  have huN : N ≤ upper*s := by
    have he : upper*s=(N.tdiv s)*s+s := by dsimp [upper]; grobner
    rw [he]; omega
  constructor
  · calc
      lower*a ≤ lower*((s+1)*d) := Int.mul_le_mul_of_nonneg_left hhi hl0
      _ = (lower*(s+1))*d := by grobner
      _ ≤ N*d := Int.mul_le_mul_of_nonneg_right hlN (by omega)
  · calc
      N*d ≤ (upper*s)*d := Int.mul_le_mul_of_nonneg_right huN (by omega)
      _ = upper*(s*d) := by grobner
      _ ≤ upper*a := Int.mul_le_mul_of_nonneg_left hlo hu0


theorem native_magnitude_representation (mag e : Int)
    (hm : 1 ≤ mag ∧ mag < 16777216) :
    ((doubledMagnitude mag e).toNat:Int)=doubledMagnitude mag e ∧
      (doubledMagnitude mag e).toNat < 33554432 := by
  have h := root_argument_domain mag e hm
  omega

theorem root_scale_powers (p : Int) (hp : 0 ≤ p) :
    pow2i (2*p)=pow2i p*pow2i p ∧
    pow2i (4*p)=pow2i (2*p)*pow2i (2*p) ∧
    pow2i (3*p)=pow2i p*pow2i (2*p) := by
  have h2 := NumericalProofs.pow2i_add_nonnegative p p hp hp
  have h4 := NumericalProofs.pow2i_add_nonnegative (2*p) (2*p) (by omega) (by omega)
  have h3 := NumericalProofs.pow2i_add_nonnegative p (2*p) hp (by omega)
  have e2 : p+p=2*p := by omega
  have e4 : 2*p+2*p=4*p := by omega
  have e3 : p+2*p=3*p := by omega
  rw [e2] at h2
  rw [e4] at h4
  rw [e3] at h3
  exact ⟨h2,h4,h3⟩

end RootEnclosureProofs
