// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the instruction front end (assembly.md sec.
// 7: slices/cu/cu_iq.sail, postlude/iq.sail and the head-parcel logic of postlude/tick.sail).
// Diagnostics and lints need the harness (the unit-test stubs ignore them): the end-to-end behaviour
// (the lint OTS-W-IQ-001, the diagnostics OTS-E-IQ-002 to OTS-E-IQ-004, OTS-E-QUE-001 on parcels, DEC-003, the control channel,
// waiting behind a Repeat or a loop) is in test/directed/frontend.py.

// A unit with the bundle bit and the slice tag in place.
private function cu_unit(bundle : bool, i : instruction) -> bits(32) =
  (if bundle then 0x80000000 else 0x00000000) | encdec32(i)

private function nop_u(n : nat) -> bits(32) = encdec32(NOP(get_slice_int(16, n, 0)))
private function setvl_u(n : nat) -> bits(32) = encdec32(SETVL(get_slice_int(6, n, 0)))

private function fill_pending(q : range(0, 144)) -> bool = match fill_fifo[q] { [||] => false, _ => true }

// Load the units into queue q's IQ (as the initial image does).
private function load_units(q : range(0, 144), xs : list(bits(32))) -> unit = {
  var ys = xs;
  var go : bool = true;
  while go do {
    match ys {
      [||] => go = false,
      u :: rest => { let _ = iq_push(q, u); ys = rest }
    }
  }
}

// The ring: units in order, across the wrap, with the capacity enforced.
$[test]
function test_iq_ring() -> unit = {
  reset();
  assert(iq_cap == iq_depth / 4 & iq_rate == iq_fetch_rate / 4);
  foreach (k from 0 to (iq_cap - 1)) { assert(iq_push(7, get_slice_int(32, k, 0))) };
  assert(iq_len[7] == iq_cap);
  assert(not(iq_push(7, 0x12345678)));                   // full: the unit is lost
  assert(iq_len[7] == iq_cap & iq_peek(7, 0) == 0x00000000 & iq_peek(7, 5) == 0x00000005);
  iq_pop(7, 3);
  assert(iq_len[7] == iq_cap - 3 & iq_peek(7, 0) == 0x00000003);
  assert(iq_push(7, 0xAAAA0001) & iq_push(7, 0xAAAA0002));   // wraps
  assert(iq_len[7] == iq_cap - 1 & iq_peek(7, iq_cap - 4) == get_slice_int(32, iq_cap - 1, 0) & iq_peek(7, iq_cap - 3) == 0xAAAA0001 & iq_peek(7, iq_cap - 2) == 0xAAAA0002);
  assert(iq_len[8] == 0 & iq_len[144] == 0);             // queues are independent
  reset();
  assert(iq_len[7] == 0)
}

// Enqueue: a fetch executed at t delivers from t + icu.fetch_latency, icu.fetch_bytes_per_cycle
// bytes per cycle, in FIFO order across fetches; padding is dropped but takes its place in the
// rate; the units of a vector are its lanes, lane 0 first, little-endian.
$[test]
function test_iq_fill() -> unit = {
  reset();
  var a : vec = zero_vec;
  var b : vec = zero_vec;
  a[7 .. 0] = 0x11; a[15 .. 8] = 0x22; a[23 .. 16] = 0x33; a[31 .. 24] = 0x44;
  a[63 .. 32] = 0xFFFFFFFF;                              // padding
  b[7 .. 0] = 0x55;
  let _ = iq_fetch(3, 10, a, b);
  let t = 10 + iq_fetch_latency;
  let (p0, l0) = iq_deliver(3, t - 1);
  assert(not(p0) & not(l0) & iq_len[3] == 0);            // not ready yet
  let (p1, l1) = iq_deliver(3, t);
  assert(p1 & not(l1));
  // The first unit is lanes 0..3; the second (padding) is dropped but used a place.
  assert(iq_peek(3, 0) == 0x44332211);
  assert(iq_len[3] == (if iq_rate >= 2 then iq_rate - 1 else 1));
  // A second fetch queues behind the first.
  let _ = iq_fetch(3, 10, b, a);
  var c : nat = t + 1;
  var guard : nat = 0;
  while fill_pending(3) & guard < 1000 do { let _ = iq_deliver(3, c); c = c + 1; guard = guard + 1 };
  assert(guard < 1000);
  // Both payloads are in, in order: 2 x 2 vectors of vlanes bytes, minus the two padding units.
  assert(iq_len[3] == min(vlanes - 2, iq_cap));
  reset()
}

// A full IQ loses the arriving units and says so.
$[test]
function test_iq_overflow() -> unit = {
  reset();
  foreach (k from 1 to iq_cap) { assert(iq_push(9, 0x00000001)) };
  let _ = iq_fetch(9, 0, zero_vec, zero_vec);                    // all-zero units are not padding
  let (p, l) = iq_deliver(9, iq_fetch_latency);
  assert(not(p) & l & iq_len[9] == iq_cap);
  reset()
}

// The head parcel (sec. 7.5): leading NOPs are a timing prefix, a parcel ends at the first
// instruction with bundle bit 0, a two-unit instruction needs both units, and a NOP with the
// bundle bit, or after an instruction with it, is a member of the parcel.
$[test]
function test_parcel_scan() -> unit = {
  reset();
  assert(not(parcel_scan(5).complete));                  // empty IQ
  load_units(5, [| nop_u(3), nop_u(4) |]);
  assert(not(parcel_scan(5).complete));                  // only a prefix
  load_units(5, [| setvl_u(5) |]);
  let p = parcel_scan(5);
  assert(p.complete & p.nops == 7 & p.n == 1 & p.units == 3);
  assert(parcel_earliest(5, p) == 7);                    // last = -1: sum of the prefix
  iq_last[5] = 100;
  assert(parcel_earliest(5, parcel_scan(5)) == 108);
  reset();
  // A bundle: two instructions, the first with the bundle bit.
  load_units(6, [| cu_unit(true, SYNC()), setvl_u(2), setvl_u(3) |]);
  let q = parcel_scan(6);
  assert(q.complete & q.n == 2 & q.units == 2 & q.nops == 0);
  // A NOP inside the parcel is a member (it breaks the slot rule), not a prefix.
  reset();
  load_units(6, [| cu_unit(true, NOP(0x0002)), setvl_u(2) |]);
  let r = parcel_scan(6);
  assert(r.complete & r.n == 2 & r.nops == 0);
  reset();
  load_units(6, [| cu_unit(true, SETVL(0b000010)), nop_u(2) |]);
  let s = parcel_scan(6);
  assert(s.complete & s.n == 2 & s.nops == 0);
  // A bundle that continues past the text is not complete.
  reset();
  load_units(6, [| cu_unit(true, SYNC()) |]);
  assert(not(parcel_scan(6).complete));
  // A 64-bit instruction is two units; the first alone is not complete.
  reset();
  let rd = encdec64(MEM_DIRECT(MEM_READ, sram_addr_of(1), Streamref(0b000000), 0b0));
  load_units(6, [| rd[63 .. 32] |]);
  assert(not(parcel_scan(6).complete));
  load_units(6, [| rd[31 .. 0] |]);
  let t = parcel_scan(6);
  assert(t.complete & t.n == 1 & t.units == 2);
  // A tag-6 unit is one undecodable instruction.
  reset();
  load_units(6, [| 0x60000000, setvl_u(2) |]);
  let u = parcel_scan(6);
  assert(u.complete & u.n == 1 & u.units == 1 & (match u.items { [|Err(DEC_OPCODE)|] => true, _ => false }));
  reset()
}

// The dispatch tests below use the VXM queues 100 and 101, which exist in every profile
// (a MEM queue of an absent slice reports OTS-E-QUE-002 and does nothing).
// Dispatch in order from the text: a NOP prefix delays the next parcel (earliest = last + 1 +
// the sum of the NOPs), and a held head parcel (parked queue) waits with everything behind it.
$[test]
function test_frontend_dispatch() -> unit = {
  reset();
  load_units(100, [| nop_u(3), setvl_u(5), setvl_u(6) |]);
  tick(0); tick(1); tick(2);
  assert(iq_len[100] == 3);
  tick(3);
  assert(queue_vl[100] == min(5, superlanes) & iq_len[100] == 1 & iq_last[100] == 3);
  tick(4);
  assert(queue_vl[100] == min(6, superlanes) & iq_len[100] == 0 & iq_last[100] == 4);
  assert(model_quiescent());
  reset();
  // Behind a Sync the queue is parked: the next parcel waits; a Notify on another queue
  // releases it barrier_release_latency cycles later.
  load_units(100, [| cu_unit(false, SYNC()), setvl_u(4) |]);
  load_units(101, [| nop_u(5), cu_unit(false, NOTIFY()) |]);
  tick(0);
  assert(queue_parked(100, 1) & iq_len[100] == 1);
  tick(1);
  assert(iq_len[100] == 1);
  assert(model_next_event(1) == 5);                      // the Notify at 5; the Sync's head parcel is held
  tick(5);
  assert(iq_len[101] == 0 & iq_len[100] == 1);
  let rel = 5 + barrier_release_latency;
  assert(model_next_event(5) == rel);
  tick(rel);
  assert(queue_vl[100] == min(4, superlanes) & iq_len[100] == 0);
  assert(model_quiescent());
  reset()
}

// A whole-queue instruction (design 7.6: an AGT reference issued in cycle c also holds
// the queue for it) due in the cycle the configuration of a path ends (state Config,
// with a reference due that cycle or not) waits for the loop and its tail; it must not
// dispatch next to a reference of its own cycle (dispatch_check would fault).
$[test]
function test_frontend_whole_queue_open_config() -> unit = {
  let rd = encdec64(MEM_DIRECT(MEM_READ, sram_addr_of(1), Streamref(0b000001), 0b0));
  var cd : nat = 1;
  while cd <= 3 do {
    // Read at 0, Countdown at 1, SetVL due at 2: the references are at 0 and 1 + cd (two default
    // iterations); the path is held until last + 1 + cd = 2 + 2 cd (design 7.6).
    reset();
    let cdw = encdec64(MEM_COUNTDOWN(get_slice_int(16, cd, 0)));
    load_units(0, [| rd[63 .. 32], rd[31 .. 0], cdw[63 .. 32], cdw[31 .. 0], setvl_u(4) |]);
    tick(0); tick(1);
    assert(agts[0].state == AGT_CONFIG & iq_len[0] == 1);
    tick(2);
    assert(iq_len[0] == 1);                                // the SetVL waits, with no fault
    let free = 2 + 2 * cd;
    assert(model_next_event(2) == 4);                       // the reference of cycle 4 (cd 3), or the end of the tail (cd 1)
    var t : nat = 3;
    while t < free do { tick(t); assert(iq_len[0] == 1); t = t + 1 };
    tick(free);
    assert(iq_len[0] == 0 & iq_last[0] == free);
    cd = cd + 2
  };
  reset()
}

// A parcel that is not complete or not due is a pending event only when it can dispatch.
$[test]
function test_frontend_events() -> unit = {
  reset();
  assert(model_quiescent() & model_next_event(0) == -1);
  load_units(100, [| nop_u(40), setvl_u(2) |]);
  assert(not(model_quiescent()) & model_next_event(0) == 40);
  // An unreleased Sync holds what is behind it: nothing to wait for.
  reset();
  load_units(100, [| cu_unit(false, SYNC()), setvl_u(2) |]);
  tick(0);
  assert(model_quiescent() & model_next_event(0) == -1);
  // Fetched text that is on its way is an event, then one every cycle.
  reset();
  let _ = iq_fetch(100, 20, zero_vec, zero_vec);
  assert(not(model_quiescent()) & model_next_event(20) == 20 + iq_fetch_latency);
  reset()
}

// The staging FIFO holds icu.fetch_staging_depth payloads per queue; a fetch that finds it full
// fetches nothing (and the instruction reports OTS-E-IQ-003); a payload that has entered the IQ
// no longer counts; the other queues are independent.
$[test]
function test_iq_staging_bound() -> unit = {
  reset();
  foreach (k from 1 to iq_staging_depth) { assert(iq_fetch(5, 10, zero_vec, zero_vec)) };
  assert(fill_len(fill_fifo[5]) == iq_staging_depth);
  assert(not(iq_fetch(5, 10, zero_vec, zero_vec)));
  assert(fill_len(fill_fifo[5]) == iq_staging_depth);          // the refused fetch left no trace
  assert(iq_fetch(6, 10, zero_vec, zero_vec) & iq_fetch(144, 10, zero_vec, zero_vec));
  // Drain queue 5 cycle by cycle: a head payload leaves the FIFO with its last unit.
  var t : nat = 10 + iq_fetch_latency;
  while fill_len(fill_fifo[5]) == iq_staging_depth do { let _ = iq_deliver(5, t); t = t + 1 };
  assert(fill_len(fill_fifo[5]) == iq_staging_depth - 1);
  assert(iq_fetch(5, t, zero_vec, zero_vec));
  reset()
}

// The end-of-run check (model_finish): text that cannot form a parcel is stranded; a trailing NOP
// prefix is not.
$[test]
function test_iq_stranded() -> unit = {
  reset();
  assert(not(iq_stranded(100)));
  load_units(100, [| nop_u(5), nop_u(7) |]);
  assert(not(iq_stranded(100)));                                  // a NOP with no successor is no truncation
  reset();
  load_units(100, [| cu_unit(true, SETVL(0b000010)) |]);
  assert(iq_stranded(100));                                       // the last instruction has the bundle bit
  reset();
  load_units(100, [| nop_u(3), cu_unit(true, SETVL(0b000010)) |]);
  assert(iq_stranded(100));
  reset();
  load_units(100, [| 0x10000000 |]);                              // the first unit of a MEM instruction only
  assert(iq_stranded(100));
  reset();
  load_units(100, [| 0x20000000 |]);                              // ... of a VXM instruction
  assert(iq_stranded(100));
  reset();
  load_units(144, [| cu_unit(true, NOTIFY()) |]);
  assert(iq_stranded(144) & not(iq_stranded(100)));
  reset()
}

// Seed the production counter at a reachable aligned payload base, bypassing
// millions of earlier all-padding fetches. Use real staging, delivery and decode.
$[test]
function test_text_word_epoch_boundary() -> unit = {
  reset();
  iq_prod[88] = 4294967296;
  let text : vec = sail_zero_extend(0x52000001, 8 * vlanes);
  assert(iq_fetch_ov(88, 0, text, zero_vec, ecc_ov_set(ecc_ov_zero, 0, ecc_poison_sl), ecc_ov_zero));
  model_inject_unit(88, iq_fetch_latency, 0x52000002);
  assert(fill_len(fill_fifo[88]) == 2, "a host instruction cannot inherit a fetched source tag");
  let _ = iq_deliver(88, iq_fetch_latency);
  assert(iq_text_unc(88, 0, 1), "uncorrectable fetched text must retain its integrity identity across 2^32 units")
}
