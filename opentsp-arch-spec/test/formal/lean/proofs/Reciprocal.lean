-- Exact integer cross-products for the actual TanH reciprocal enclosure.
import Proofs.Positivity
open Sail Out Out.Functions
set_option autoImplicit false
namespace ReciprocalProofs

/-- For every nonnegative rational E=a/d bracketed by [u/S,v/S], the actual
    floor/ceil reciprocal formula encloses B*(E-1)/(E+1). Positivity is proved
    separately from actual Exp output; it is never inferred from this goal. -/
theorem tanh_reciprocal_rational (S B u v a d : Int)
    (hS : 0 < S) (hB : 0 < B) (hu : 0 ≤ u) (hv : 0 ≤ v)
    (ha : 0 ≤ a) (hd : 0 < d)
    (hlo : u*d ≤ a*S) (hhi : a*S ≤ v*d) :
    let lower := B - idiv_ceil (2*B*S) (u+S)
    let upper := B - idiv_floor (2*B*S) (v+S)
    lower*(a+d) ≤ B*(a-d) ∧ B*(a-d) ≤ upper*(a+d) := by
  let l := idiv_ceil (2*B*S) (u+S)
  let h := idiv_floor (2*B*S) (v+S)
  have huS : 0 < u+S := by omega
  have hvS : 0 < v+S := by omega
  have hnum : 0 ≤ 2*B*S := Int.mul_nonneg (by omega) (by omega)
  have hl0 : 0 ≤ l := (PositivityProofs.floor_ceil_nonnegative _ _ hnum huS).2
  have hh0 : 0 ≤ h := (PositivityProofs.floor_ceil_nonnegative _ _ hnum hvS).1
  have hl := (IntervalProofs.ceil_bounds (2*B*S) (u+S) huS).2
  have hh := (IntervalProofs.floor_bounds (2*B*S) (v+S) hvS).1
  change 2*B*S ≤ l*(u+S) at hl
  change h*(v+S) ≤ 2*B*S at hh
  have hscaledlo : (u+S)*d ≤ S*(a+d) := by
    have he1 : (u+S)*d = u*d+S*d := by grind
    have he2 : S*(a+d) = a*S+S*d := by grind
    rw [he1,he2]
    omega
  have hscaledhi : S*(a+d) ≤ (v+S)*d := by
    have he1 : (v+S)*d = v*d+S*d := by grind
    have he2 : S*(a+d) = a*S+S*d := by grind
    rw [he1,he2]
    omega
  have hmullo : (2*B*d)*S ≤ (l*(a+d))*S := by
    calc
      (2*B*d)*S = (2*B*S)*d := by grind
      _ ≤ (l*(u+S))*d := Int.mul_le_mul_of_nonneg_right hl (by omega)
      _ = l*((u+S)*d) := by grind
      _ ≤ l*(S*(a+d)) := Int.mul_le_mul_of_nonneg_left hscaledlo hl0
      _ = (l*(a+d))*S := by grind
  have hmulhi : (h*(a+d))*S ≤ (2*B*d)*S := by
    calc
      (h*(a+d))*S = h*(S*(a+d)) := by grind
      _ ≤ h*((v+S)*d) := Int.mul_le_mul_of_nonneg_left hscaledhi hh0
      _ = (h*(v+S))*d := by grind
      _ ≤ (2*B*S)*d := Int.mul_le_mul_of_nonneg_right hh (by omega)
      _ = (2*B*d)*S := by grind
  have hcl := Int.le_of_mul_le_mul_right hmullo hS
  have hch := Int.le_of_mul_le_mul_right hmulhi hS
  dsimp only
  change (B-l)*(a+d) ≤ B*(a-d) ∧ B*(a-d) ≤ (B-h)*(a+d)
  have hel : (B-l)*(a+d) = B*(a-d)+(2*B*d-l*(a+d)) := by grind
  have heh : (B-h)*(a+d) = B*(a-d)+(2*B*d-h*(a+d)) := by grind
  rw [hel,heh]
  omega

/-- The complete actual TanH post-processing, including exponent rescaling,
    odd-sign reversal and the final one-unit outward margin. -/
theorem tanhPost_rational_enclosure (lo hi n p a d : Int) (neg : Bool)
    (hlo : 0 ≤ lo) (hhi : 0 ≤ hi) (ha : 0 ≤ a) (hd : 0 < d)
    (hElow : (if n ≥ 0 then lo*pow2i n else idiv_floor lo (pow2i (-n)))*d ≤ a*pow2i p)
    (hEhigh : a*pow2i p ≤ (if n ≥ 0 then hi*pow2i n else idiv_ceil hi (pow2i (-n)))*d) :
    (RangeReductionProofs.tanhPost neg p (lo,hi,n)).1*(a+d) ≤
        (if neg then -(pow2i (2*p)*(a-d)) else pow2i (2*p)*(a-d)) ∧
      (if neg then -(pow2i (2*p)*(a-d)) else pow2i (2*p)*(a-d)) ≤
        (RangeReductionProofs.tanhPost neg p (lo,hi,n)).2.1*(a+d) := by
  let S := pow2i p
  let B := pow2i (2*p)
  let u := if n ≥ 0 then lo*pow2i n else idiv_floor lo (pow2i (-n))
  let v := if n ≥ 0 then hi*pow2i n else idiv_ceil hi (pow2i (-n))
  let lower := B - idiv_ceil (2*B*S) (u+S)
  let upper := B - idiv_floor (2*B*S) (v+S)
  have hpos := PositivityProofs.scaled_exp_nonnegative lo hi n hlo hhi
  change 0 ≤ u ∧ 0 ≤ v at hpos
  have h := tanh_reciprocal_rational S B u v a d
    (PositivityProofs.pow2i_any_positive p) (PositivityProofs.pow2i_any_positive (2*p))
    hpos.1 hpos.2 ha hd hElow hEhigh
  change lower*(a+d) ≤ B*(a-d) ∧ B*(a-d) ≤ upper*(a+d) at h
  have hD : 0 < a+d := by omega
  have he1 : (lower-1)*(a+d)=lower*(a+d)-(a+d) := by grind
  have he2 : (upper+1)*(a+d)=upper*(a+d)+(a+d) := by grind
  have he3 : (-upper-1)*(a+d)= -(upper*(a+d))-(a+d) := by grind
  have he4 : (-lower+1)*(a+d)= -(lower*(a+d))+(a+d) := by grind
  cases neg
  · change (lower-1)*(a+d) ≤ B*(a-d) ∧ B*(a-d) ≤ (upper+1)*(a+d)
    rw [he1,he2]
    omega
  · change (-upper-1)*(a+d) ≤ -(B*(a-d)) ∧ -(B*(a-d)) ≤ (-lower+1)*(a+d)
    rw [he3,he4]
    omega

end ReciprocalProofs
