-- Lean counterparts of p_barrier_window, p_barrier_open_ended and p_barrier_release (model/props/props.sail), about the real
-- generated `in_barrier_window`, and the arms the SMT obligations leave to the function's last clause: a queue with no Sync is never in a
-- window (barrier_closed), whatever the release is.
import Proofs.Common
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs

theorem barrier_window (s n c : Nat) :
    in_barrier_window (some s) (some (n + barrier_release_latency)) c = decide (s ≤ c ∧ c < n + barrier_release_latency) := by
  simp [in_barrier_window]

theorem barrier_open_ended (s c : Nat) :
    in_barrier_window (some s) none c = decide (s ≤ c) := by
  simp [in_barrier_window]

/-- No Sync, no window: with a release or without one, at every cycle. -/
theorem barrier_closed (release : Option Nat) (c : Nat) :
    in_barrier_window none release c = false := by
  cases release <;> simp [in_barrier_window]

/-- The window closes exactly `release_latency` cycles after Notify. -/
theorem barrier_release (s n : Nat) (h : s < n + barrier_release_latency) :
    (n + barrier_release_latency = 0 ∨
      in_barrier_window (some s) (some (n + barrier_release_latency)) (n + barrier_release_latency - 1) = true) ∧
    in_barrier_window (some s) (some (n + barrier_release_latency)) (n + barrier_release_latency) = false := by
  constructor
  · right
    simp [in_barrier_window]
    omega
  · simp [in_barrier_window]
