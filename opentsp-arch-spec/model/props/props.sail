// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Proof obligations, stated over the model's own functions (not restated rules). Proved with
// `sail -smt --smt-auto` and z3 on both shipped profiles (tools/model_prove.sh).
// Every $property must be proved; every $counterexample must be refuted.
//
// Scope: the SMT backend handles these functions directly. A proof of the full
// encdec32/64/128 round trip (P-ENC-1/2) over every clause did not finish in 15 minutes on
// the whole project; that obligation is covered empirically instead, by the catalogue
// cross-check (tools/isa_db_check.sh), which runs every layout's field values through the
// real encdec mapping. The queue map (144 values) is checked exhaustively by
// unit_tests/test_slices.sail, which is equivalent and much cheaper than an SMT query.

// Proof environment. The MSI-X transport consumes the emitted notification;
// these obligations check the chip-visible count, mask and cause. No PCIe
// delivery guarantee follows from this environment assumption.
function host_irq_callback(_, _) = ()
function ecc_event_callback(_, _, _, _, _, _, _, _, _) = ()

// ---- Encoding sub-mappings (P-ENC-1 for the field encoders shared by every layout) ----

$property
function p_eltype_roundtrip(b : bits(5)) -> bool =
  (unsigned(b) > 12 & b != 0b10000) | encdec_eltype(encdec_eltype(b)) == b

$property
function p_stream_fields(s : bits(6)) -> bool = {
  let r = Streamref(s);
  stream_id(r) == unsigned(s[4 .. 0]) & (stream_row(r) == RowW) == (s[5] == bitone)
}

// ---- Operand alignment (OTSP-CON-ALIGN; the same check guards reads and writes) ----

// An aligned w-stream operand stays on its row and inside streams 0..31 (a sanity check:
// this is the second conjunct of aligned(), operands.sail).
$property
function p_aligned_in_range(s : bits(6), k : bits(2)) -> bool = {
  let w : {1, 2, 4} = if k == 0b00 then 1 else if k == 0b01 then 2 else 4;
  not(aligned(Streamref(s), w)) | stream_id(Streamref(s)) + w <= 32
}

// Every 4-aligned stream is also 2- and 1-aligned.
$property
function p_aligned_nested(s : bits(6)) -> bool =
  not(aligned(Streamref(s), 4)) | (aligned(Streamref(s), 2) & aligned(Streamref(s), 1))

// ---- SRF ring (P-SRF-3): live diagonals never share a slot ----
// A value at gap g with commit cycle c has diagonal g - c. A read at gap G in cycle t (its
// execute cycle, dispatch + d_skew) only needs diagonals committed in the last 44 cycles; the
// property keeps a margin of d_skew (0..15, enforced by config/schema.json). For every
// such margin two distinct diagonals that
// close together map to distinct slots of the 64-slot ring (emod_int(_, 64) is slot_of in
// core/streams.sail, which is private). This is arithmetic about the ring, not about
// srf_write/srf_read: the SMT backend cannot translate srf_write (its trace callback has
// no generator), so the reads and writes are covered by the Sail unit tests `test_srf_*` and the Python oracle instead.

$property
function p_srf_live_window(d : bits(12), k : bits(6), sk : bits(4)) -> bool = {
  let a = signed(d);
  unsigned(k) == 0 | unsigned(k) > 44 + unsigned(sk) | emod_int(a, 64) != emod_int(a + unsigned(k), 64)
}

// ---- SRAM addressing: only words_per_slice words exist ----

$property
function p_sram_index_bound(a : bits(addr_bits)) -> bool =
  match sram_index(SramAddr(a)) {
    Some(i) => i < words_per_slice & i == unsigned(a),
    None()  => unsigned(a) >= words_per_slice,
  }

// ---- Barrier (P-BAR-1) ----

private function bar_clear() -> unit = {
  // Element by element: vector_init() of a register inside a property gives invalid SMT.
  foreach (k from 0 to 144) { parked_since[k] = None(); release_at[k] = None() };
  last_notify = None()
}

// Behaviour over the barrier's state-changing functions (postlude/tick.sail): a Sync on the
// control channel with no later Notify parks every data queue at and after its dispatch,
// whatever Notify came before it. Non-vacuous: with a seeded bug (barrier_sync parks from
// c + 1) the solver finds a counterexample.
$property
function p_barrier_cc_parks_all(k : bits(7), s : bits(8), dc : bits(10), n : bits(7)) -> bool = {
  bar_clear();
  let q = unsigned(k);
  barrier_notify_test(unsigned(n));
  barrier_sync_test(144, unsigned(s) + 128);
  queue_parked(q, unsigned(s) + 128 + unsigned(dc))
}

// The upper 16 data queues are a separate obligation: together these cover
// every queue 0..143 without an expensive guarded 8-bit vector selection.
$property
function p_barrier_cc_parks_upper(k : bits(4), s : bits(8), dc : bits(10), n : bits(7)) -> bool = {
  bar_clear();
  barrier_notify_test(unsigned(n));
  barrier_sync_test(144, unsigned(s) + 128);
  queue_parked(128 + unsigned(k), unsigned(s) + 128 + unsigned(dc))
}

// Sanity checks of the pure window function (they restate in_barrier_window's arms, so
// they guard its form, not the behaviour above).

$property
function p_barrier_window(s : bits(16), n : bits(16), c : bits(17)) -> bool = {
  let r : nat = unsigned(n) + barrier_release_latency;
  in_barrier_window(Some(unsigned(s)), Some(r), unsigned(c)) == (unsigned(s) <= unsigned(c) & unsigned(c) < r)
}

// The window closes exactly barrier.release_latency cycles after Notify, and a Sync with
// no Notify parks every later cycle.
$property
function p_barrier_release(s : bits(8), n : bits(16)) -> bool = {
  let r : nat = unsigned(n) + barrier_release_latency;
  let sy : nat = unsigned(s);
  not(sy < r) | ((r == 0 | in_barrier_window(Some(sy), Some(r), max(r - 1, 0))) & not(in_barrier_window(Some(sy), Some(r), r)))
}

$property
function p_barrier_open_ended(s : bits(16), c : bits(17)) -> bool =
  in_barrier_window(Some(unsigned(s)), None(), unsigned(c)) == (unsigned(s) <= unsigned(c))

// ---- Sanity: these must be refuted ----

$counterexample
function c_everything_aligned(s : bits(6)) -> bool = aligned(Streamref(s), 4)

// The ring is finite: diagonals 64 apart share a slot, so the bound in p_srf_live_window
// is what keeps them apart.
$counterexample
function c_srf_ring_distinct_at_64(d : bits(12)) -> bool = emod_int(signed(d), 64) != emod_int(signed(d) + 64, 64)

$counterexample
function c_never_parked(c : bits(16)) -> bool = not(in_barrier_window(Some(10), Some(20 + barrier_release_latency), unsigned(c)))

// All protected bit positions, arbitrary original data. ecc_flip_delta is the
// physical flip in the model's overlay representation; bit 137 is the spare.
$property
function p_ecc_single(d : bits(128), index : bits(8)) -> bool = {
  let n = unsigned(index);
  let mask : bits(128) = sail_shiftleft(get_slice_int(128, 1, 0), n);
  let damaged = if n < 128 then d ^ mask else d;
  let (k, recovered, _) = ecc_correct(damaged, ecc_flip_delta(n));
  n > 136 | (k == ECC_CORRECTED & recovered == d)
}

$property
function p_ecc_double(x : bits(8), y : bits(8)) -> bool = {
  let a = unsigned(x);
  let b = unsigned(y);
  let (k, _) = ecc_classify(ecc_flip_delta(a) ^ ecc_flip_delta(b));
  a >= b | b > 136 | k == ECC_UNCORRECTABLE
}

// The actual profile validator implies source availability for every aligned
// unit, even away from the two endpoints used by the implementation.
$property
function p_fetch_unit_available(ns : bits(5), ls : bits(6), rs : bits(4), bs : bits(1),
                                ks : bits(5), u : bits(2)) -> bool = {
  let sl = 1 + unsigned(ns);               // 1 .. 32 superlanes
  let rate = 4 * (1 + unsigned(rs));
  let latency = 1 + unsigned(ls);
  let beat = unsigned(bs);
  let tile = unsigned(ks);
  let byte_offset = 16 * (sl * beat + tile) + 4 * unsigned(u);
  tile >= sl | not(fetch_profile_causal(sl, latency, rate)) |
    1 + latency + byte_offset / rate > beat + tile
}

$counterexample
function c_last_tile_can_always_feed_fast_iq() -> bool = fetch_profile_causal(20, 1, 64)

// Arbitrary legal ring head, occupancy, and number actually removed, including
// a held queue (n=0). The surviving logical index denotes the same physical slot.
$property
function p_iq_pop_conservation(h : bits(10), len : bits(11), n : bits(11), idx : bits(11)) -> bool = {
  let head = unsigned(h);
  let old = unsigned(len);
  let pop = unsigned(n);
  let i = unsigned(idx);
  if head >= iq_cap | old > iq_cap | pop > old then true else {
    iq_head[0] = head;
    iq_len[0] = old;
    iq_pop(0, pop);
    iq_len[0] + pop == old & iq_head[0] < iq_cap &
      (i >= iq_len[0] | ring(iq_head[0] + i) == ring(head + pop + i))
  }
}

// A newer per-queue Sync is not released by an older Notify. A later Notify
// releases it at the configured boundary, including latency zero.
$property
function p_barrier_two_generations(s : bits(8), gap : bits(6)) -> bool = {
  bar_clear();
  let start = unsigned(s);
  barrier_sync_test(3, start);
  barrier_notify_test(start + 1);
  let second = start + 2 + barrier_release_latency;
  barrier_sync_test(3, second);
  let wake = second + 1 + unsigned(gap);
  let waited = queue_parked(3, wake - 1);
  barrier_notify_test(wake);
  waited & not(queue_parked(3, wake + barrier_release_latency))
}

// One uncorrectable class exercises the counter and
// sticky-bit transitions. Seed the counter below or at saturation, log at the
// last slot or already full, and interrupt delivery masked or enabled.
$property
function p_ras_reporting(x : bits(32), full : bool, enabled : bool) -> bool = {
  ras_err_status = 0x0000000000000000;
  ras_err_mask = 0x0000000000000002;
  ras_mbe_count = 0x00000000 @ x;
  ras_log_n = if full then ras_log_depth else ras_log_depth - 1;
  ras_first_valid = false;
  csr_irq_cause = 0x0000000000000000;
  csr_irq_mask = if enabled then 0x0000000000000001 else 0x0000000000000000;
  csr_irq_count = 0x0000000000000000;
  ras_ecc_event(10, 3, 0, true, false, 0, 0, 0, 3);
  let recorded = ras_first_valid & ras_err_status[1] == bitone &
    ((ras_err_status[9] == bitone) == full) & ras_log_n == ras_log_depth &
    unsigned(ras_mbe_count) == min(unsigned(x) + 1, 4294967295);
  recorded & csr_irq_cause[0] == bitone & unsigned(csr_irq_count) == (if enabled then 1 else 0)
}

$property
function p_hostwait_consumes_once(start : bits(16), gap : bits(12)) -> bool = {
  let s = unsigned(start);
  hw_since[128] = None();
  hw_since[129] = None();
  hw_release[128] = None();
  hw_release[129] = None();
  csr_doorbell[0] = true;
  host_wait(128, s, 0);
  host_wait(129, s, 0);
  not(host_parked(128, s + 1)) & host_parked(129, s + 1 + unsigned(gap)) & not(csr_doorbell[0])
}

$property
function p_fp8_decode(x : bits(8), ieee : bool) -> bool = {
  let f = if ieee then FMT_E5M2 else FMT_E4M3;
  let decoded = fdecode(f, 0x000000 @ x);
  let exponent = if ieee then unsigned(x[6 .. 2]) else unsigned(x[6 .. 3]);
  let fraction = if ieee then unsigned(x[1 .. 0]) else unsigned(x[2 .. 0]);
  let special = if ieee then exponent == 31 else (exponent == 15 & fraction == 7);
  let class_ok = if special then (if ieee & fraction == 0 then decoded.cls == FInf else decoded.cls == FNaN)
    else if exponent == 0 & fraction == 0 then decoded.cls == FZero
    else decoded.cls == FFinite & decoded.mag == fraction + (if exponent == 0 then 0 else if ieee then 4 else 8) &
         decoded.e == max(exponent, 1) - (if ieee then 17 else 10);
  decoded.neg == (x[7] == bitone) & class_ok
}

// Instrument the external ring-store contract to inspect the slot, value and
// identity written by the actual append operation, including calls on a full IQ.
register proof_slot : int
register proof_tag_slot : int
register proof_data : bits(32)
register proof_tag : int
function iq_unit_write(_, slot, u) = { proof_slot = slot; proof_data = u }
function iq_tag_write(_, slot, tag) = { proof_tag_slot = slot; proof_tag = tag }

$property
function p_iq_append(h : bits(10), l : bits(11), payload : bits(32), source : bits(40)) -> bool = {
  let head = unsigned(h);
  let occ = unsigned(l);
  if head >= iq_cap | occ > iq_cap then true else {
    iq_head[0] = head;
    iq_len[0] = occ;
    proof_slot = -1;
    proof_tag_slot = -1;
    let ok = iq_push_tagged(0, payload, unsigned(source));
    let slot = emod_int(head + occ, iq_cap);
    iq_head[0] == head & (if occ == iq_cap then
      not(ok) & iq_len[0] == occ & proof_slot == -1 & proof_tag_slot == -1
    else ok & iq_len[0] == occ + 1 & proof_slot == slot & proof_tag_slot == slot &
      proof_data == payload & proof_tag == unsigned(source))
  }
}

$property
function p_fetch_sample_lifetime(t : bits(16), data : vec, first_ov : ecc_ov, next_ov : ecc_ov) -> bool = {
  let c = unsigned(t);
  stream_fetch_arm(88, Streamref(0b000000), c, data, first_ov);
  let early = stream_fetch_take(88, c);
  let ready = stream_fetch_take(88, c + 1);
  // A subsequent fetch may start on the completion edge.
  stream_fetch_arm(88, Streamref(0b100001), c + 1, zero_vec, next_ov);
  match (early, ready, stream_fetch_take(88, c + 1), stream_fetch_take(88, c + 2), stream_fetch_take(88, c + 3)) {
    (None(), Some(a), None(), Some(b), None()) =>
      a.ready == c + 1 & a.first == data & a.ov == first_ov & b.ov == next_ov &
      stream_id(a.source) == 0 & stream_id(b.source) == 1,
    _ => false,
  }
}

function raw_write_sram_ecc(_, _, _) = ()
function raw_write_sram_word(_, _, _) = ()

$property
function p_patrol_writeback(rev : bits(32), newer_data : bool, newer_code : bool) -> bool = {
  let r = unsigned(rev);
  sram_revision[0] = r;
  scrub_pending[0] = Some(struct { revision = r, addr = 3, data = zero_vec, ov = ecc_ov_zero });
  if newer_data then write_sram_word(0, 4, zero_vec);
  if newer_code then write_sram_ecc(0, 3, ecc_ov_zero);
  let repair = scrub_take(0);
  let consumed : bool = match scrub_take(0) { None() => true, _ => false };
  consumed & (match repair { None() => newer_data | newer_code,
    Some(w) => not(newer_data | newer_code) & w.addr == 3 & w.revision == r })
}

// One reporting step for each architectural event class, for all reachable
// SBE/MBE counter values and both edges of the finite log. A prior sticky
// status prevents redelivery until cleared (the overflow cause is masked).
$property
function p_ras_event_classes(code : bits(4), prior : bool, full : bool, x : bits(32), c : bits(16)) -> bool = {
  let k = 1 + min(8, unsigned(code));
  let mask : bits(64) = sail_shiftleft(0x0000000000000001, k - 1);
  ras_err_status = if prior then mask else 0x0000000000000000;
  ras_err_mask = mask;
  ras_mbe_count = 0x00000000 @ x; ras_sbe_count = 0x00000000 @ x;
  ras_log_n = if full then ras_log_depth else 0;
  ras_first_valid = false; csr_irq_mask = 0x0000000000000001;
  csr_irq_cause = 0x0000000000000000; csr_irq_count = 0x0000000000000000;
  ras_record(unsigned(c), 88, k, sail_zeros(43), 3, 8);
  let uncorrectable = k == 2 | k == 4 | k == 6 | k == 8;
  let saturated = min(4294967295, unsigned(x) + 1);
  let recorded = ras_first_valid & unsigned(ras_first0[63 .. 52]) == 88 &
                 unsigned(ras_first0[51 .. 48]) == k & unsigned(ras_first0[47 .. 0]) == unsigned(c);
  let expected_count : nat = if full then ras_log_depth else 1;
  let log_ok = ras_log_n == expected_count & (full | (ras_log0[0] == ras_first0 & ras_log1[0] == ras_first1));
  recorded & log_ok & and_vec(ras_err_status, mask) == mask & ((ras_err_status[9] == bitone) == full) &
    unsigned(ras_mbe_count) == (if uncorrectable then saturated else unsigned(x)) &
    unsigned(ras_sbe_count) == (if not(uncorrectable) & k != 9 then saturated else unsigned(x)) &
    unsigned(csr_irq_count) == (if prior then 0 else 1)
}

$property
function p_fetch_word_poison(m : bits(superlanes), k : bits(5)) -> bool = {
  let g = unsigned(k);
  if g >= superlanes then true
  else {
    let extracted = sail_shiftright(ecc_poison_ov(m), 10 * g);
    let mask = sail_shiftright(m, g);
    extracted[9 .. 0] == (if mask[0] == bitone then 0b0000000011 else 0b0000000000)
  }
}
