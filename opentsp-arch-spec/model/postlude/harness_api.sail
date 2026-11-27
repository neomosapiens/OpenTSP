// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Queries for the harness (c_emulator/tsp_sim.c): when the run is finished and how far it
// may skip ahead. They read the cycle driver's state (postlude/tick.sail) and change
// nothing. Cycle-by-cycle and event-skipping harnesses preserve architectural effects
// and ordered effect output when they deliver the same external events at the specified
// cycles. Optional observe_site records describe visited edges: skipped idle edges produce
// no wait samples, so the complete O stream need not be identical.
// The harness owns the initial IQ/SRAM load and timed host, DMA, link and fault inputs.
// Its next-event frontier must include those inputs as well as the model's own events;
// model quiescence alone does not authorize skipping a pending external action.

// Nothing is left to happen: no commit is pending, no fetched text is on its way, and no
// queue has a head parcel that can still dispatch, Repeat iterations or AGT references left
// (queues parked by a Sync that no Notify has released are ignored: they never resume).
function model_quiescent() -> bool = model_idle()

// The first cycle at or after t at which queue q (or the control channel) is not
// parked by a Sync whose release cycle is already known.
private function unparked_from(q : queue_id, t : nat) -> nat = {
  var u : nat = t;
  foreach (pass from 0 to 1) {
    foreach (j from 0 to 1) {
      let k : range(0, 144) = if j == 0 then q else 144;
      match (parked_since[k], release_at[k]) {
        (Some(s), Some(r)) => if s <= u & u < r then u = r,
        _ => (),
      }
    }
  };
  u
}

// A queue whose pending Sync has no Notify yet (it waits for a record, not for time).
private function held(q : queue_id) -> bool = {
  var h : bool = false;
  foreach (j from 0 to 1) {
    let k : range(0, 144) = if j == 0 then q else 144;
    match (parked_since[k], release_at[k]) { (Some(_), None()) => h = true, _ => () }
  };
  h
}

private function earliest(a : int, b : int) -> int =
  if a < 0 then b else if b < 0 then a else if a < b then a else b

// After tick(c): the earliest cycle t > c at which tick(t) can do something (-1: none). The
// candidates are a pending commit, an element waiting to execute (dispatch + d_skew), fetched
// text that enters an IQ (the head of a fill FIFO, from its ready cycle on, then every cycle),
// the head parcel of a queue that can dispatch (not before its earliest cycle, and when
// nothing holds it any more: parcel_event), an open AGT configuration on either dispatch path
// (it ends at the first cycle in which the queue dispatches no configuration instruction, so
// every cycle counts), the next AGT reference and the next Repeat iteration (delayed to the
// release of a known barrier window; work held by an unreleased Sync waits for a Notify).
function model_next_event(c : nat) -> int = {
  var best : int = -1;
  foreach (q from 0 to 143) {
    match stream_fetches[q] { Some(f) => best = earliest(best, max(c + 1, f.ready)), None() => () }
  };
  foreach (q from 0 to 87) {
    match local_fetches[q] { Some(f) => best = earliest(best, max(c + 1, f.ready)), None() => () }
  };
  foreach (k from 0 to 127) {
    match pending[k] {
      [||] => (),
      _ => best = earliest(best, c + 1 + emod_int(k - (c + 1), 128)),
    }
  };
  foreach (k from 0 to 15) {
    match execute_ring[k] {
      [||] => (),
      _ => best = earliest(best, c + 1 + emod_int(k - (c + 1), 16)),
    }
  };
  foreach (q from 0 to 144) {
    match fill_fifo[q] {
      seg :: _ => best = earliest(best, max(c + 1, seg.ready)),
      [||] => (),
    }
  };
  best = earliest(best, control_event(c));
  foreach (q from 0 to 143) {
    foreach (p from 0 to 1) {
      let a = agt_get(q, p);
      if a.state == AGT_CONFIG then best = earliest(best, c + 1)
      else if a.state == AGT_RUN & not(held(q)) then best = earliest(best, unparked_from(q, max(a.next, c + 1)))
    };
    match queue_repeat[q] {
      Some(r) => if not(held(q)) then best = earliest(best, unparked_from(q, max(r.next, c + 1))),
      None() => (),
    };
    best = earliest(best, parcel_event(q, c))
  };
  best
}

// The next epoch boundary at which a queue's DeskewRecv can find its vector (-1: none): the boundary at or after the
// arrival of the oldest vector the harness knows on the link and after the last tick; when the queue is parked by a
// barrier at that boundary, the first boundary from the cycle the barrier ends (none while it has no release yet). A queue
// whose link has no vector waits for one: the harness has the event when the vector is delivered (c2c_next_arrival). The
// harness calls it between ticks, so the answer is the earliest cycle at which tick can do something because of a poll.
function model_poll_event() -> int = {
  var best : int = -1;
  foreach (q from 0 to 143) {
    match queue_poll[q] {
      Some(p) => {
        let a : int = deskewrecv_next_arrival(p.link);
        if a >= 0 & not(held(q)) then {
          let an : nat = if a >= 0 then a else 0;
          let lo : int = max(an, last_tick + 1);
          let lon : nat = if lo >= 0 then lo else 0;
          var b : nat = next_boundary(lon);
          if queue_parked(q, b) then b = next_boundary(unparked_from(q, b));
          best = earliest(best, b)
        }
      },
      None() => (),
    }
  };
  best
}

// The end-of-run report, called once by the harness after model_quiescent holds: each queue
// whose IQ still holds text that cannot dispatch (iq_stranded) and that is not parked at a
// Sync no Notify releases (such a queue never resumes by design) raises OTS-E-IQ-004, in
// ascending queue order with the control channel last, at the cycle of the queue's last IQ
// activity (a unit entering the IQ or a dispatch; 0 for an initial image that never moved).
// The cycle does not depend on which cycles the harness ticked.
private function hw_for_good(k : range(0, 144)) -> bool =
  match (hw_since[k], hw_release[k]) { (Some(_), None()) => true, _ => false }

// A queue whose DeskewRecv has not found its vector (the harness has no event for it, or the run would not have ended).
private function poll_for_good(k : range(0, 144)) -> bool = {
  let q : queue_id = if k <= 143 then k else 0;
  k <= 143 & polling(q)
}

private function parked_for_good(k : range(0, 144)) -> bool =
  match (parked_since[k], release_at[k]) { (Some(_), None()) => true, _ => hw_for_good(k) | poll_for_good(k) }

function model_finish() -> unit =
  foreach (q from 0 to 144) {
    if iq_len[q] > 0 & not(parked_for_good(q) | parked_for_good(144)) & iq_stranded(q) then
      diag_report(iq_active[q], q, IQ_STRANDED)
  }


// The host's side of the chip (core/csr.sail; the harness runs the host's script). Each call
// takes the cycle the host acts in; the harness ticks that cycle afterwards.

// A host read of a CSR (the BAR).
function model_csr_read(addr : bits(12)) -> bits(64) = csr_read(unsigned(addr))

// A host write of a CSR at cycle c.
function model_csr_write(addr : bits(12), v : bits(64), c : nat) -> unit = csr_write_chip(unsigned(addr), v, c)

// The packet types of a link (slices/c2c/c2c_insts.sail): whether a packet of the type (c2c_ptype_code)
// enters the receive FIFO, and how many cycles it occupies a link. The harness's control unit
// consumes csr and ctrl packets itself and sends them only in a gap of the program's schedule.
function model_ptype_queued(code : range(0, 3)) -> bool = c2c_ptype_queued(match code { 0 => PKT_DATA, 1 => PKT_CSR, 2 => PKT_SYNC, _ => PKT_CTRL })
function model_packet_cycles(code : range(0, 3)) -> nat = c2c_packet_cycles(match code { 0 => PKT_DATA, 1 => PKT_CSR, 2 => PKT_SYNC, _ => PKT_CTRL })

// Link FEC (microarchitecture.md sec. 17): flit `flit` of a vector that arrived on link l in cycle c was corrected (unc false) or
// detected uncorrectable (unc true); the harness calls this in the arrival cycle, once per flit, before the tick. The link counter
// counts it and the event goes to the error registers (core/csr.sail ras_link_event, sec. 16).
function model_link_fec(c : nat, l : range(0, 127), flit : range(0, 127), unc : bool) -> unit = {
  if unc then csr_link_unc[l] = add_bits(csr_link_unc[l], 0x0000000000000001)
  else csr_link_corr[l] = add_bits(csr_link_corr[l], 0x0000000000000001);
  ras_link_event(c, l, flit, unc)
}

// The flits of a link transfer: a header flit, vector_flits - 1 payload flits of 8 bytes, a tail flit; the superlanes a flit covers
// (bit g = superlane g): a payload flit its 8 lanes, the header and the tail flit the whole vector.
function model_link_flits() -> nat = link_flits
function model_link_flit_mask(f : range(0, 127)) -> range(0, 4294967295) = link_flit_mask(f)

// `F mxm c plane lane bit` (microarchitecture.md sec. 18): bit `bit` (0..95) of the exact sum of lane `lane` of the ABC that
// executes on `plane` in cycle c flips. The harness calls this before the tick of c; the ABC consumes it. After the tick,
// `model_mxm_pending` is the plane of a fault no ABC consumed (an input error), or -1.
function model_inject_mxm(pl : range(0, 3), lane : nat, bit : range(0, 95)) -> unit = mxm_inject(pl, lane, bit)
function model_mxm_pending() -> int = match mxm_fault_list { [||] => -1, f :: _ => f.mf_pl }

// `H start c`: the queues start dispatching at cycle c, as if every queue had last dispatched
// in cycle c - 1. The default start is cycle 0 (iq_last = -1).
function model_set_start(c : nat) -> unit = iq_last = vector_init(c - 1)

private function fill_units_append(xs : list(bits(32)), u : bits(32)) -> list(bits(32)) =
  match xs { [||] => [| u |], x :: rest => x :: fill_units_append(rest, u) }

private function fill_push_unit(xs : list(fill_seg), c : nat, u : bits(32)) -> list(fill_seg) =
  match xs {
    [||] => [| struct { ready = c, units = [| u |], base = -1, taken = 0, ovw = [||] } |],
    x :: [||] => if x.base < 0 & x.ready == c then [| { x with units = fill_units_append(x.units, u) } |] else [| x, struct { ready = c, units = [| u |], base = -1, taken = 0, ovw = [||] } |],
    x :: rest => x :: fill_push_unit(rest, c, u),
  }

// `H inject c queue units`: one unit of host text for queue q, entering the IQ from cycle c on
// through Enqueue (step 1 of tick) with the overflow rule of the IQ (OTS-E-IQ-002). The units
// of one line follow each other in order.
function model_inject_unit(q : range(0, 144), c : nat, u : bits(32)) -> unit = fill_fifo[q] = fill_push_unit(fill_fifo[q], c, u)


// `F sram c slice addr superlane bit`: a bit of a stored SRAM word flips (core/ecc.sail): a data
// bit changes the stored data, a check bit or the spare bit the stored overlay. The stored
// word keeps the flip until a write replaces it.
function model_flip_sram(s : mem_slice, a : nat, sl : range(0, 31), n : range(0, 137)) -> unit = {
  if a < words_per_slice & sl < superlanes then {
    if n < 128 then write_sram_word(s, a, read_sram_word(s, a) ^ sail_shiftleft(get_slice_int(8 * vlanes, 1, 0), 128 * sl + n));
    let ov = read_sram_ecc(s, a);
    write_sram_ecc(s, a, ecc_ov_set(ov, sl, ecc_ov_get(ov, sl) ^ ecc_flip_delta(n)))
  }
}

// `F iq c queue word bit`: a bit of text word `word` of the queue (counted from 0 over the queue's text in production
// order) flips while the word is in the IQ: a data bit (0..127) changes the stored unit and the overlay, a check bit
// or the spare bit the overlay. False (an input error of the harness) when no unit of the word is in the IQ.
function model_flip_iq(q : range(0, 144), w : nat, n : range(0, 137)) -> bool = {
  var present : bool = false;
  foreach (i from 0 to (iq_len[q] - 1)) {
    let t = iq_tag_read(q, ring(iq_head[q] + i));
    if t != iq_no_word & ediv_int(t, 4) == w then present = true
  };
  if present then {
    if n < 128 then {
      let want = 4 * w + ediv_int(n, 32);
      foreach (i from 0 to (iq_len[q] - 1)) {
        if iq_tag_read(q, ring(iq_head[q] + i)) == want then
          iq_unit_write(q, ring(iq_head[q] + i), iq_peek(q, i) ^ sail_shiftleft(get_slice_int(32, 1, 0), emod_int(n, 32)))
      }
    };
    iq_word_ov_set(q, w, iq_word_ov(q, w) ^ ecc_flip_delta(n));
    iq_ecc_live = true
  };
  present
}

// `F srf c row stream gap superlane bit`: a bit of the value in a stream register flips.
function model_flip_srf(c : nat, w : bool, s : range(0, 31), g : gap, sl : range(0, 31), n : range(0, 137)) -> unit =
  srf_flip(c, if w then RowW else RowE, s, g, sl, n)

// The scrub cycles before cycle c, run before the harness acts at the start of c (an `F` flip, a host write): they saw the state
// of the earlier cycles (microarchitecture.md sec. 15.4).
function model_scrub_before(c : nat) -> unit = scrub_run(c - 1)
