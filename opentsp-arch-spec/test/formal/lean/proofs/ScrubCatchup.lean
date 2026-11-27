-- Exact effect-order decomposition of the production patrol catch-up loop.
import Proofs.Patrol
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions
set_option autoImplicit false
open Register
namespace ScrubCatchupProofs

def repeated {m : Type → Type} [Monad m] (action : m Unit) : Nat → m Unit
  | 0 => pure ()
  | n+1 => action >>= fun _ => repeated action n

theorem repeated_add {m : Type → Type} [Monad m] [LawfulMonad m]
    (action : m Unit) (n k : Nat) :
    repeated action (n+k) = (repeated action n >>= fun _ => repeated action k) := by
  induction n with
  | zero => simp [repeated]
  | succ n ih => simp [Nat.succ_add,repeated,ih,bind_assoc]

theorem inclusive_repeated {m : Type → Type} [Monad m] [LawfulMonad m]
    (action : m Unit) (upper : Int) :
    (forIn ({start := 1, stop := upper, step := 1, step_pos := by decide} : IntRange) () (fun _ _ => action >>= fun _ => pure (.yield ()))) =
      repeated action upper.toNat := by
  change IntRange.forIn' _ () _ = _
  unfold IntRange.forIn'
  suffices h : ∀ (b : Unit) (i : Int) (hs : (i - 1) % 1 = 0), 1 ≤ i →
      IntRange.forIn'.loop ({start := 1, stop := upper, step := 1, step_pos := by decide} : IntRange)
        (fun _ _ _ => action >>= fun _ => pure (.yield ())) b i hs =
          repeated action (upper - i + 1).toNat by
    simpa using h () 1 (by simp) (by omega)
  apply IntRange.forIn'.loop.induct ({start := 1, stop := upper, step := 1, step_pos := by decide} : IntRange)
  · intro b i hs hmem ih hlo
    cases b
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_pos hmem,bind_assoc,pure_bind]
    have hm : 1 ≤ i ∧ i ≤ upper := by simpa [Membership.mem] using hmem
    have hi : i ≤ upper := hm.2
    have hn : (upper - i + 1).toNat = (upper - (i + 1) + 1).toNat+1 := by omega
    rw [hn,repeated]
    congr 1
    funext x
    cases x
    exact ih () (by dsimp; omega)
  · intro b i hs hmem hlo
    cases b
    rw [IntRange.forIn'.loop.eq_def]
    simp only [dif_neg hmem]
    have hi : upper < i := by simp [Membership.mem] at hmem;omega
    have hz : (upper - i + 1).toNat=0 := by omega
    simp [hz,repeated]

def patrol_step : SailM Unit := do
  scrub_cycle (← Out.readReg scrub_next)
  Out.writeReg scrub_next ((← Out.readReg scrub_next) + scrub_period)

theorem scrub_run_production (upto : Int) :
    scrub_run upto = (do
      let n ← Out.readReg scrub_next
      repeated patrol_step (scrub_due_count n scrub_period upto).toNat) := by
  simp only [scrub_run, patrol_step, bind_assoc, pure_bind, bind_pure]
  congr 1
  funext n
  have hu : ∀ y : Unit, y = () := fun y => Subsingleton.elim y ()
  simpa [patrol_step, bind_assoc, hu] using
    inclusive_repeated (m := SailM) patrol_step (scrub_due_count n scrub_period upto)

theorem due_count_nonnegative (n p : Nat) (upto : Int) (ht : 0 ≤ upto) :
    0 ≤ scrub_due_count n p upto := by
  unfold scrub_due_count
  split
  · rename_i h
    have hp : 0 < p ∧ n ≤ upto.toNat := by simpa using h
    have hn : (n : Int) ≤ upto := by omega
    have hq := Int.ediv_nonneg (by omega : 0 ≤ upto - (n : Int))
      (by omega : 0 ≤ (p : Int))
    change 0 ≤ (upto - (n : Int)) / (p : Int) + 1
    omega
  · omega

theorem due_count_frontier (n p : Nat) (upto : Int) (hp : 0 < p) :
    upto < (n : Int) + (scrub_due_count n p upto).toNat * (p : Int) := by
  by_cases ht : upto < 0
  · have hn := Int.natCast_nonneg n
    have hk := Int.natCast_nonneg (scrub_due_count n p upto).toNat
    have hm := Int.mul_nonneg hk (Int.natCast_nonneg p)
    omega
  · have ht : 0 ≤ upto := by omega
    unfold scrub_due_count
    split
    · rename_i h
      have hb : 0 < p ∧ n ≤ upto.toNat := by simpa using h
      have hn : (n : Int) ≤ upto := by omega
      have hq := Int.ediv_nonneg (by omega : 0 ≤ upto - (n : Int))
        (by omega : 0 ≤ (p : Int))
      have hcast : ((((upto - (n : Int)) / (p : Int) + 1).toNat : Nat) : Int) =
        (upto - (n : Int)) / (p : Int) + 1 := Int.toNat_of_nonneg (by omega)
      change upto < (n : Int) + (((upto - (n : Int)) / (p : Int) + 1).toNat : Int) * (p : Int)
      rw [hcast, Int.add_mul]
      have hbound := (Int.ediv_le_iff_le_mul (by omega : 0 < (p : Int))).mp
        (Int.le_refl ((upto - (n : Int)) / (p : Int)))
      omega
    · rename_i h
      have hb : ¬ (0 < p ∧ n ≤ upto.toNat) := by simpa using h
      change upto < (n : Int) + 0 * (p : Int)
      omega

theorem due_count_last (n p : Nat) (upto : Int) (ht : 0 ≤ upto)
    (hk : 0 < (scrub_due_count n p upto).toNat) :
    (n : Int) + ((scrub_due_count n p upto).toNat - 1 : Nat) * (p : Int) ≤ upto := by
  unfold scrub_due_count at *
  split at hk
  · rename_i h
    simp only [if_pos h]
    have hb : 0 < p ∧ n ≤ upto.toNat := by simpa using h
    have hn : (n : Int) ≤ upto := by omega
    have hq := Int.ediv_nonneg (by omega : 0 ≤ upto - (n : Int))
      (by omega : 0 ≤ (p : Int))
    change (n : Int) + (((((upto - (n : Int)) / (p : Int) + 1).toNat - 1 : Nat) : Int) * (p : Int)) ≤ upto
    have hcast : ((((upto - (n : Int)) / (p : Int) + 1).toNat - 1 : Nat) : Int) =
      (upto - (n : Int)) / (p : Int) := by omega
    rw [hcast]
    have hbound := Int.ediv_mul_le (upto - (n : Int)) (by omega : (p : Int) ≠ 0)
    omega
  · simp at hk

theorem negative_count_zero (n p : Nat) (upto : Int) (ht : upto < 0) :
    (scrub_due_count n p upto).toNat = 0 := by
  unfold scrub_due_count
  split
  · rename_i h
    have hb : 0 < p ∧ n ≤ upto.toNat := by simpa using h
    have hn : n = 0 := by omega
    have hq : (upto - (n : Int)) / (p : Int) < 0 :=
      (Int.ediv_lt_iff_lt_mul (by omega : 0 < (p : Int))).mpr (by omega)
    change ((upto - (n : Int)) / (p : Int) + 1).toNat = 0
    omega
  · rfl

theorem scrub_run_negative (upto : Int) (ht : upto < 0) :
    scrub_run upto = (do
      let _ ← Out.readReg scrub_next
      pure ()) := by
  simp only [scrub_run_production, negative_count_zero _ _ upto ht, repeated]

#print axioms repeated_add
#print axioms inclusive_repeated
#print axioms scrub_run_production
#print axioms due_count_nonnegative
#print axioms due_count_frontier
#print axioms due_count_last
#print axioms negative_count_zero
#print axioms scrub_run_negative
end ScrubCatchupProofs
