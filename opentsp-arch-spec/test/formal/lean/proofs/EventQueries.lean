-- Production scheduler query frames under bounded initialized frontend ports.
import Proofs.ParcelBound
import Proofs.EventFrames
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs
open Proofs.ParcelFrame
set_option autoImplicit false
set_option maxHeartbeats 500000
namespace EventQueryProofs

macro "query_frame" : tactic => `(tactic|
  repeat' first
  | exact pure_frame _ _
  | exact read_frame _ _
  | apply bind_frame
  | (intro x)
  | apply forIn_frame
  | split)

theorem agt_get_frame (s : St) (q p : Nat) : FrameAt s (agt_get q p) := by
  simp only [agt_get]
  split <;> query_frame

theorem hold_view_frame (s : St) (q c : Nat) : FrameAt s (hold_view_of q c) := by
  simp only [hold_view_of, queue_parked, host_parked, parked_one, polling, path_views, path_view_of,
    agt_get, bind_assoc, pure_bind, bind_pure]
  query_frame

theorem hold_check_frame (s : St) (q c : Nat) (i : instruction) (path : Option Nat) (strict : Bool) :
    FrameAt s (hold_check q c i path strict) := by
  simp only [hold_check, bind_assoc, pure_bind, bind_pure]
  apply bind_frame _ _ _ (hold_view_frame s q c)
  intro view
  query_frame

#print axioms agt_get_frame
#print axioms hold_view_frame
#print axioms hold_check_frame

theorem strict_held_frame (s : St) (q c : Nat) (items : List (Result instruction diag)) (held : Bool) :
    FrameAt s (strict_held q c items held) := by
  induction items generalizing held with
  | nil => exact pure_frame _ _
  | cons x xs ih =>
    simp only [strict_held, bind_assoc, pure_bind, bind_pure]
    apply bind_frame
    · cases x <;> exact hold_check_frame _ _ _ _ _ _
    · intro next
      exact ih _

theorem parcel_held_frame (s : St) (q c : Nat) (r : cycle_recs) (items : List (Result instruction diag)) :
    FrameAt s (parcel_held q c r items) := by
  simp only [parcel_held, cfg_path, agt_get, bind_assoc, pure_bind, bind_pure]
  repeat' first
  | exact strict_held_frame _ _ _ _ _
  | exact hold_check_frame _ _ _ _ _ _
  | exact pure_frame _ _
  | exact read_frame _ _
  | apply bind_frame
  | apply forIn_frame
  | (intro x)
  | split_action_match
  | split

#print axioms strict_held_frame
#print axioms parcel_held_frame

theorem earliest_free_frame (s : St) (q base : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) (cands : List Nat) (best : Int) :
    FrameAt s (earliest_free q base r items cands best) := by
  induction cands generalizing best with
  | nil => exact pure_frame _ _
  | cons t rest ih =>
    simp only [earliest_free, bind_assoc, pure_bind, bind_pure]
    repeat' first
    | exact ih _
    | exact parcel_held_frame _ _ _ _ _
    | exact pure_frame _ _
    | apply bind_frame
    | (intro x)
    | split

theorem parcel_dispatchable_frame (s : St) (q base : Nat) (r : cycle_recs)
    (items : List (Result instruction diag)) : FrameAt s (parcel_dispatchable q base r items) := by
  simp only [parcel_dispatchable, bind_assoc, pure_bind, bind_pure]
  repeat' first
  | exact earliest_free_frame _ _ _ _ _ _ _
  | exact pure_frame _ _
  | exact read_frame _ _
  | apply bind_frame
  | (intro x)
  | split_action_match
  | split

theorem parcel_earliest_frame (s : St) (q : Nat) (p : parcel) : FrameAt s (parcel_earliest q p) := by
  simp only [parcel_earliest, bind_assoc, pure_bind, bind_pure]
  query_frame

#print axioms earliest_free_frame
#print axioms parcel_dispatchable_frame
#print axioms parcel_earliest_frame

theorem unparked_frame (s : St) (q t : Nat) : FrameAt s (unparked_from q t) := by
  simp only [unparked_from, bind_assoc, pure_bind, bind_pure]
  query_frame

theorem parcel_event_frame (s : St) (q c : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hq : q<144) (hh : H[q]!<iq_cap) (hl : L[q]!≤iq_cap) : FrameAt s (parcel_event q c) := by
  simp only [parcel_event, bind_assoc, pure_bind, bind_pure]
  apply bind_frame _ _ _ (read_frame _ _)
  intro lengths
  split
  · exact pure_frame _ _
  · apply bind_frame _ _ _ (Proofs.ParcelBound.scan_frame s q ports H L hH hL good (by omega) hh hl)
    intro p
    split
    · exact pure_frame _ _
    · apply bind_frame _ _ _ (parcel_earliest_frame _ _ _)
      intro early
      exact parcel_dispatchable_frame _ _ _ _ _

theorem control_event_frame (s : St) (c : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (hh : H[144]!<iq_cap) (hl : L[144]!≤iq_cap) : FrameAt s (control_event c) := by
  simp only [control_event, bind_assoc, pure_bind, bind_pure]
  apply bind_frame _ _ _ (read_frame _ _)
  intro lengths
  split
  · exact pure_frame _ _
  · apply bind_frame _ _ _ (Proofs.ParcelBound.scan_frame s 144 ports H L hH hL good (by decide) hh hl)
    intro p
    split
    · apply bind_frame _ _ _ (parcel_earliest_frame _ _ _)
      intro early
      exact pure_frame _ _
    · exact pure_frame _ _

#print axioms unparked_frame
#print axioms parcel_event_frame
#print axioms control_event_frame

theorem model_next_event_frame (s : St) (c : Nat) (ports : ReadPorts iq_cap)
    (H L : Vector Nat 145) (hH : s.regs.get? Register.iq_head=some H)
    (hL : s.regs.get? Register.iq_len=some L) (good : Proofs.StorageProduct.Good iq_cap s)
    (heads : ∀ q, q<145 → H[q]!<iq_cap) (lengths : ∀ q, q<145 → L[q]!≤iq_cap) :
    FrameAt s (model_next_event c) := by
  simp only [model_next_event, bind_assoc, pure_bind, bind_pure]
  apply bind_frame
  · query_frame
  · intro best0
    apply bind_frame
    · query_frame
    · intro best1
      apply bind_frame
      · query_frame
      · intro best2
        apply bind_frame
        · query_frame
        · intro best3
          apply bind_frame
          · query_frame
          · intro best4
            apply bind_frame _ _ _ (control_event_frame s c ports H L hH hL good (heads 144 (by decide)) (lengths 144 (by decide)))
            intro control
            apply forIn_frame
            intro q hq best
            have qbound : q.toNat < 144 := by
              have hi : 0 ≤ q ∧ q ≤ 143 := by simpa [Membership.mem] using hq
              omega
            apply bind_frame
            · repeat' first
              | exact unparked_frame _ _ _
              | exact agt_get_frame _ _ _
              | exact read_frame _ _
              | exact pure_frame _ _
              | apply bind_frame
              | apply forIn_frame
              | (intro x)
              | split_action_match
              | split
            · intro after_agt
              apply bind_frame _ _ _ (read_frame _ _)
              intro repeats
              apply bind_frame
              · repeat' first
                | exact unparked_frame _ _ _
                | exact read_frame _ _
                | exact pure_frame _ _
                | apply bind_frame
                | apply forIn_frame
                | (intro x)
                | split_action_match
                | split
              · intro after_repeat
                apply bind_frame _ _ _ (parcel_event_frame s q.toNat c ports H L hH hL good qbound (heads _ (by omega)) (lengths _ (by omega)))
                intro event
                exact pure_frame _ _
#print axioms model_next_event_frame
end EventQueryProofs
