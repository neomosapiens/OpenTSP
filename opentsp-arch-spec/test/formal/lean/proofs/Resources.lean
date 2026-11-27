-- Production-state resource reservation and patrol exclusion. No execution callback is rebound.
import Proofs.Common
set_option autoImplicit false
open Sail Sail.ConcurrencyInterfaceV1 PreSail Out Out.Functions Proofs

namespace Proofs.Resources

abbrev State := SequentialState RegisterType trivialChoiceSource

/-- Exact production reserve transition for a non-MEM queue, including its complete register frame. -/
theorem reserve_nonmem_run (q c : Nat) (hq : 87 < q) (s : State) (H : Vector Nat 288)
    (hH : s.regs.get? Register.path_busy_until = some H) (i : instruction) :
    (reserve_local_fetch q i c).run s = EStateM.Result.ok ()
      {s with regs := s.regs.insert Register.path_busy_until (vectorUpdate H (2*q) (c+2))} := by
  have hn : ¬ q ≤ 87 := by omega
  have h2 : (2 * (q : Int)).toNat = 2*q := by omega
  have hc : ((c : Int) + 2).toNat = c+2 := by omega
  simp [h2, hc, reserve_local_fetch, fetchm_hold_until, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg,
    hH, hn, EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get,
    EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet]

/-- Exact production reservation for every MEM queue. The second-beat note is determined by
    the actual fetchm_bank_note, and all unrelated state is preserved. -/
theorem reserve_mem_run (q c : Nat) (hq : q ≤ 87) (s : State) (H : Vector Nat 288)
    (M : Vector Nat 88) (B : Vector (Option bank_note) 144)
    (hH : s.regs.get? Register.path_busy_until = some H)
    (hM : s.regs.get? Register.mem_segment = some M)
    (hB : s.regs.get? Register.local_fetch_bank = some B) (i : instruction) :
    (reserve_local_fetch q i c).run s = EStateM.Result.ok ()
      {s with regs := match fetchm_bank_note i c M[q]! with
        | some n => (s.regs.insert Register.path_busy_until (vectorUpdate H (2*q) (c+2))).insert
            Register.local_fetch_bank (vectorUpdate B q (some n))
        | none => s.regs.insert Register.path_busy_until (vectorUpdate H (2*q) (c+2))} := by
  have h2 : (2 * (q : Int)).toNat = 2*q := by omega
  have hc : ((c : Int) + 2).toNat = c+2 := by omega
  simp [h2, hc, reserve_local_fetch, fetchm_hold_until, Out.readReg, Out.writeReg, PreSail.readReg, PreSail.writeReg,
    hH, hM, hq, EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get,
    EStateM.get, modify, modifyGet, MonadStateOf.modifyGet, EStateM.modifyGet,
    Std.ExtDHashMap.get?_insert]
  split <;> simp_all [EStateM.bind, EStateM.get, EStateM.pure, EStateM.modifyGet, Std.ExtDHashMap.get?_insert]

/-- Both SRAM beats of an IfetchM reservation exclude patrol, at any Nat cycle and skew.
    This is unbounded arithmetic, not the SMT backend's finite integer encoding. -/
theorem fetchm_patrol_taken (c skew beat busy1 : Nat) (last : Option Nat) (hb : beat < 2) :
    scrub_slot_taken (c + skew + beat) skew (fetchm_hold_until c) busy1 last = true := by
  simp [scrub_slot_taken, fetchm_hold_until]
  omega

/-- The exact production arbiter suppresses all patrol activity in a software-owned slot,
    regardless of presence, saved repair or its freshness. -/
theorem patrol_software_priority (present saved fresh : Bool) :
    scrub_arbitrate present true saved fresh = .SCRUB_SKIP := by
  simp [scrub_arbitrate]

/-- A successful real run_element is followed by the actual reservation, for every state it
    returns. No assumption that run_element is a logger, pure, or register-preserving. -/
theorem repeat_ifetchm_after_success (q c : Nat) (a : BitVec Out.Functions.addr_bits) (s mid : State)
    (hRun : (run_element q (.IFETCHM (.SramAddr a)) c 0 0 0).run s = EStateM.Result.ok () mid) :
    (repeat_issue q (.IFETCHM (.SramAddr a)) c).run s =
      (reserve_local_fetch q (.IFETCHM (.SramAddr a)) c).run mid := by
  simp [repeat_issue, EStateM.run, bind, EStateM.bind] at *
  rw [hRun]

/-- Failure propagation is explicit: a failing execute stage does not run the reservation.
    This prevents the successful composition theorem being read as totality of callbacks. -/
theorem repeat_ifetchm_after_failure (q c : Nat) (a : BitVec Out.Functions.addr_bits) (s mid : State) (e : Error exception)
    (hRun : (run_element q (.IFETCHM (.SramAddr a)) c 0 0 0).run s = EStateM.Result.error e mid) :
    (repeat_issue q (.IFETCHM (.SramAddr a)) c).run s = EStateM.Result.error e mid := by
  simp [repeat_issue, EStateM.run, bind, EStateM.bind] at *
  rw [hRun]

/-- The real patrol ownership query returns true, with a complete state frame, at both
    reserved IfetchM beats. queue_last may contain any instruction or be absent. -/
theorem scrub_issued_reserved_run (q c beat : Nat) (hb : beat < 2) (s : State)
    (H : Vector Nat 288) (Q : Vector (Option instruction) 144) (L : Vector Nat 144)
    (hH : s.regs.get? Register.path_busy_until = some H)
    (hQ : s.regs.get? Register.queue_last = some Q)
    (hL : s.regs.get? Register.queue_last_at = some L)
    (hBusy : H[2*q]! = c+2) :
    (scrub_issued q (c + (entry_latency "Read" timing_class.T_MEM).d_skew + beat)).run s =
      EStateM.Result.ok true s := by
  have h2 : (2 * (q : Int)).toNat = 2*q := by omega
  have hI : H[(2 * (q : Int))]! = H[2*q]! := by
    change H[(2 * (q : Int)).toNat]! = H[2*q]!
    rw [h2]
  have ht : ((c : Int) + (entry_latency "Read" timing_class.T_MEM).d_skew + beat).toNat =
      c + (entry_latency "Read" timing_class.T_MEM).d_skew + beat := by omega
  simp [scrub_issued, Out.readReg, PreSail.readReg, hQ,
    EStateM.run, bind, EStateM.bind, pure, EStateM.pure, get, getThe, MonadStateOf.get,
    EStateM.get]
  cases h : Q[q]! <;> simp [EStateM.bind, EStateM.get, EStateM.pure, hL, hH, hI, hBusy,
    scrub_slot_taken, ht]
  all_goals omega

/-- Compose the exact state transition and the actual patrol query, rather than leaving a
    manual link between reservation and exclusion. Arbitrary queue_last contents survive. -/
theorem reserve_mem_excludes_patrol (q c beat : Nat) (hq : q ≤ 87) (hb : beat < 2)
    (s : State) (H : Vector Nat 288) (M : Vector Nat 88)
    (B : Vector (Option bank_note) 144) (Q : Vector (Option instruction) 144) (L : Vector Nat 144)
    (hH : s.regs.get? Register.path_busy_until = some H)
    (hM : s.regs.get? Register.mem_segment = some M)
    (hB : s.regs.get? Register.local_fetch_bank = some B)
    (hQ : s.regs.get? Register.queue_last = some Q)
    (hL : s.regs.get? Register.queue_last_at = some L) (i : instruction) :
    ∃ after, (reserve_local_fetch q i c).run s = EStateM.Result.ok () after ∧
      (scrub_issued q (c + (entry_latency "Read" timing_class.T_MEM).d_skew + beat)).run after =
        EStateM.Result.ok true after := by
  refine ⟨_, reserve_mem_run q c hq s H M B hH hM hB i, ?_⟩
  apply scrub_issued_reserved_run q c beat hb _ (vectorUpdate H (2*q) (c+2)) Q L
  · split <;> simp [Std.ExtDHashMap.get?_insert]
  · split <;> simp [Std.ExtDHashMap.get?_insert, hQ]
  · split <;> simp [Std.ExtDHashMap.get?_insert, hL]
  · have hidx : 2*q < 288 := by omega
    simp [vectorUpdate, hidx, Vector.set!]

/-- A complete successful real Repeat iteration reserves both beats against actual patrol
    ownership, irrespective of the execute stage's returned state and queue history. -/
theorem repeat_ifetchm_excludes_patrol (q c beat : Nat) (hq : q ≤ 87) (hb : beat < 2)
    (a : BitVec Out.Functions.addr_bits) (s mid : State) (H : Vector Nat 288) (M : Vector Nat 88)
    (B : Vector (Option bank_note) 144) (Q : Vector (Option instruction) 144) (L : Vector Nat 144)
    (hRun : (run_element q (.IFETCHM (.SramAddr a)) c 0 0 0).run s = EStateM.Result.ok () mid)
    (hH : mid.regs.get? Register.path_busy_until = some H)
    (hM : mid.regs.get? Register.mem_segment = some M)
    (hB : mid.regs.get? Register.local_fetch_bank = some B)
    (hQ : mid.regs.get? Register.queue_last = some Q)
    (hL : mid.regs.get? Register.queue_last_at = some L) :
    ∃ after, (repeat_issue q (.IFETCHM (.SramAddr a)) c).run s = EStateM.Result.ok () after ∧
      (scrub_issued q (c + (entry_latency "Read" timing_class.T_MEM).d_skew + beat)).run after =
        EStateM.Result.ok true after := by
  obtain ⟨after, hr, hp⟩ := reserve_mem_excludes_patrol q c beat hq hb mid H M B Q L hH hM hB hQ hL
    (.IFETCHM (.SramAddr a))
  refine ⟨after, ?_, hp⟩
  rw [repeat_ifetchm_after_success q c a s mid hRun]
  exact hr

/-- Exact second-beat stamp and address, including the segment/slice-end no-note cases. -/
theorem fetchm_second_beat_note (a : BitVec Out.Functions.addr_bits) (c seg : Nat) :
    fetchm_bank_note (.IFETCHM (.SramAddr a)) c seg =
      if a.toNat + 1 < seg ∧ a.toNat + 1 < Out.Functions.words_per_slice then
        some {bn_at := c+1, bn_addr := a.toNat+1} else none := by
  simp only [fetchm_bank_note, BitVec.toNatInt,Out.Functions.addr_bits,BitVec.setWidth_eq]
  split <;> split <;> simp_all
  all_goals omega

end Proofs.Resources
