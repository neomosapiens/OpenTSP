-- Independent exact-series certificates for every actual ln2 precision.
-- The analytic ln2 = 2*atanh(1/3) identity and geometric-series limit are
-- explicit standard mathematical bridge facts, not Lean axioms introduced here.
import Proofs.RangeReduction
open Sail Out Out.Functions
set_option autoImplicit false
set_option maxRecDepth 1000000
set_option maxHeartbeats 20000000
namespace Ln2CertificateProofs

/-- Independent evaluation: one exact floor of each whole rational term,
    64 extra precision bits, not the production nested-truncation recurrence. -/
def independentLower (precision terms : Nat) : Nat :=
  (List.range terms).foldl
    (fun sum k => sum + (2 * 2^precision) / ((2*k+1)*3^(2*k+1))) 0

def Certificate (p : Nat) : Prop :=
  let P := p+64
  let N := (p+70)/3+1
  let L := independentLower P N
  let v := ln2_iv p
  v.1 * (2^P:Int) ≤ (L:Int)*(2^p:Int) ∧
    ((L+N+1:Nat):Int)*(2^p:Int) ≤ v.2*(2^P:Int)

/-- Every floor in the independent certificate has the exact signed/rational
    error bound used in its mathematical interpretation. -/
theorem term_floor_bound (P k : Nat) :
    let numerator := 2*2^P
    let denominator := (2*k+1)*3^(2*k+1)
    let q := numerator / denominator
    q*denominator ≤ numerator ∧ numerator < (q+1)*denominator := by
  dsimp only
  have hd : 0 < (2*k+1)*3^(2*k+1) := Nat.mul_pos (by omega) (Nat.pow_pos (by decide))
  have hm := Nat.mod_lt (2*2^P) hd
  have he := Nat.div_add_mod (2*2^P) ((2*k+1)*3^(2*k+1))
  rw [Nat.mul_comm ((2*k+1)*3^(2*k+1)) (2*2^P / ((2*k+1)*3^(2*k+1)))] at he
  have hr : (2*2^P / ((2*k+1)*3^(2*k+1)) + 1) * ((2*k+1)*3^(2*k+1)) =
      (2*2^P / ((2*k+1)*3^(2*k+1))) * ((2*k+1)*3^(2*k+1)) + ((2*k+1)*3^(2*k+1)) := by grind
  omega

theorem tail_precision_budget (p : Nat) : p+64 < 3*((p+70)/3+1) := by omega

theorem ln2_certificate_64 : Certificate 64 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_128 : Certificate 128 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_256 : Certificate 256 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_512 : Certificate 512 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_1024 : Certificate 1024 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_2048 : Certificate 2048 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_4096 : Certificate 4096 := by
  unfold Certificate
  decide +kernel

/-- Exact power inequality behind the independent positive-series tail bound.
    It is valid for every precision, not only the seven kernel evaluations. -/
theorem geometric_tail_budget (p : Nat) :
    3*2^(p+64) < 3^(2*((p+70)/3+1)+1) := by
  let N := (p+70)/3+1
  have hn : p+64 < 3*N := tail_precision_budget p
  have hpow := Nat.pow_lt_pow_of_lt (by decide : 1 < (2:Nat)) hn
  have hbase := Nat.pow_le_pow_left (show (8:Nat) ≤ 9 by decide) N
  have h8 : (8:Nat)^N = 2^(3*N) := by
    rw [Nat.pow_mul]
  have h3 : (3:Nat)^(2*N+1) = 3*9^N := by
    rw [Nat.pow_add, Nat.pow_mul]; simp [Nat.mul_comm]
  rw [h8] at hbase
  rw [h3]
  omega

theorem ln2_certificate_72 : Certificate 72 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_136 : Certificate 136 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_264 : Certificate 264 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_520 : Certificate 520 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_1032 : Certificate 1032 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_2056 : Certificate 2056 := by
  unfold Certificate
  decide +kernel

theorem ln2_certificate_4104 : Certificate 4104 := by
  unfold Certificate
  decide +kernel

end Ln2CertificateProofs
