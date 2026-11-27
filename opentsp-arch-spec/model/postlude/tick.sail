// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The cycle driver. The TSP has no fetch/decode/execute loop of its
// own: 144 queues issue and results land at later cycles. Every queue (and the control
// channel, 144) has an instruction queue (IQ, slices/cu/cu_iq.sail) of text units that the
// host loads before cycle 0 (the initial image) and that Ifetch refills; the driver takes
// each queue's head parcel (postlude/iq.sail), dispatches it when it is due and nothing
// holds it, and calls tick(c) for the cycles at which something can happen
// (harness_api.sail model_next_event).
//
// One cycle:
//   0. fetched text due at c enters the IQs (a full IQ loses it: OTS-E-IQ-002);
//   1. commits scheduled for cycle c enter the stream register file;
//   2. the instructions dispatched earlier whose execute cycle (dispatch + d_skew) is c execute;
//   3. the control channel (queue 0xFFFF) dispatches, then data queues 0..143 in order.
// A read in cycle t only sees values committed in cycles before t: a value committed at
// gap g in cycle c is in register g + 1 from cycle c + 1, so step 1 at cycle c makes it
// visible at gap g + 1 in cycle c + 1, never earlier. Results therefore do not depend on
// queue order, and the barrier is a property of the DISPATCH cycle.
//
// Dispatch is in order (design sec. 7.6; ISCA 2020 Table I: Sync "parks at the head of the
// instruction dispatch queue"): a head parcel that is not due, not complete, or held stays at
// the head and everything behind it waits. Nothing is rejected for waiting: there is no
// "queue busy" or "barrier" diagnostic.

// Barrier (ISCA 2020 sec. III-A2: Sync "parks at the head of the instruction dispatch
// queue to await barrier notification"; Notify "releases the pending barrier operations";
// 35 cycles from Notify to the Sync retiring). Causal: a Sync that DISPATCHES parks its
// queue (on the control channel: every data queue) from its dispatch cycle; a Notify that
// dispatches at N on any queue releases every Sync dispatched at or before N, at
// N + barrier_release_latency. A Sync or Notify that does not dispatch (a decode error, a
// parcel that breaks the slot rule) has no effect. A parked queue issues nothing: its head
// parcel, its AGT references and its Repeat iterations wait (they resume, shifted, when the
// queue is released); a Sync or Notify at its head is not held by the barrier.
//
// parked_since[q]: dispatch cycle of the queue's pending Sync, if any; release_at[q]: the
// cycle its pending Sync retires, once a Notify has been seen.
register parked_since : vector(145, option(nat)) = vector_init(None())
register release_at : vector(145, option(nat)) = vector_init(None())
// Dispatch cycle of the latest Notify: a Sync dispatched later in that same cycle (on a
// higher-numbered queue) is released by it too, so the result does not depend on the
// order in which the queues of one cycle are visited.
register last_notify : option(nat) = None()

// The barrier window as a pure function of one Sync and its release cycle (proved in props/).
function in_barrier_window(sync : option(nat), release : option(nat), c : nat) -> bool =
  match (sync, release) {
    (Some(s), Some(r)) => s <= c & c < r,
    (Some(s), None())  => s <= c,
    _                  => false,
  }

private function parked_one(k : range(0, 144), c : nat) -> bool = in_barrier_window(parked_since[k], release_at[k], c)

// Queue q is parked by HostWait at cycle c (core/csr.sail): from its dispatch until the cycle
// before the host's ring takes effect. Unlike a barrier park it holds Sync and Notify too.
function host_parked(q : range(0, 144), c : nat) -> bool = in_barrier_window(hw_since[q], hw_release[q], c)

// Queue q (or the control channel, 144) is parked at cycle c.
function queue_parked(q : range(0, 144), c : nat) -> bool = parked_one(q, c) | (q != 144 & parked_one(144, c))

function barrier_parked(c : nat) -> bool = parked_one(144, c)

// A Sync dispatched on queue k at cycle c.
private function barrier_sync(k : range(0, 144), c : nat) -> unit = {
  parked_since[k] = Some(c);
  release_at[k] = if last_notify == Some(c) then Some(c + barrier_release_latency) else None()
}

// A Notify dispatched at cycle n releases every Sync dispatched at or before n.
private function barrier_notify(n : nat) -> unit = {
  last_notify = Some(n);
  foreach (k from 0 to 144) {
    match (parked_since[k], release_at[k]) {
      (Some(s), None()) => if s <= n then release_at[k] = Some(n + barrier_release_latency),
      _                 => (),
    }
  }
}

// Retire Syncs whose release cycle has passed (so a later Sync can park again).
private function barrier_retire(c : nat) -> unit =
  foreach (k from 0 to 144) {
    match release_at[k] {
      Some(r) => if c >= r then { parked_since[k] = None(); release_at[k] = None() },
      None()  => (),
    };
    match hw_release[k] {
      Some(r) => if c >= r then { hw_since[k] = None(); hw_release[k] = None() },
      None()  => (),
    }
  }

// Test hooks for the barrier unit tests.
function barrier_sync_test(k : range(0, 144), c : nat) -> unit = barrier_sync(k, c)
function barrier_notify_test(n : nat) -> unit = barrier_notify(n)

// Per-queue busy-until cycle (instructions that hold their queue for several cycles: NOP n
// holds it n cycles; a Repeat iteration holds it through its cycle). Index 144 is the
// control channel.
register queue_busy_until : vector(145, nat) = vector_init(0)

// Per-path busy-until cycle of a MEM queue (index 2q + path; mem_insts.sail "Dispatch
// paths"): a reference, or an AGT reference, holds its path through its cycle.
register path_busy_until : vector(288, nat) = vector_init(0)
// Second read bank at its dispatch-relative cycle; it cannot be canceled retroactively.
struct bank_note = { bn_at : nat, bn_addr : nat }
register local_fetch_bank : vector(144, option(bank_note)) = vector_init(None())

function path_busy(q : queue_id, p : mem_path, c : nat) -> bool = c < path_busy_until[2 * q + p]

// The trailing waits of a finished AGT loop. After its last reference the loop still waits
// out the countdown of every level it has, so the loop (and its path) ends at last + 1 + the sum of those countdowns.
// The path stays held until then: a new reference on it, or a whole-queue instruction, waits
// at the head of its queue.
register path_tail_until : vector(288, nat) = vector_init(0)

function path_tail(q : queue_id, p : mem_path, c : nat) -> bool = c < path_tail_until[2 * q + p]
// Port reservation, as pure functions of the issue cycle. A reference holds its path through its own cycle; an IfetchM holds
// the R path through the cycle of its second read; the bank of word a + 1 is taken in that cycle when the word is in the
// queue's segment and the slice (seg is the segment length of the queue, 0 when the queue has no segment).
function path_hold_until(c : nat) -> nat = c + 1
function fetchm_hold_until(c : nat) -> nat = c + 2
function fetchm_bank_note(i : instruction, c : nat, seg : nat) -> option(bank_note) =
  match i {
    IFETCHM(SramAddr(a)) => if unsigned(a) + 1 < seg & unsigned(a) + 1 < words_per_slice
      then Some(struct { bn_at = c + 1, bn_addr = unsigned(a) + 1 }) else None(),
    _ => None(),
  }

private function hold_path(q : queue_id, p : mem_path, c : nat) -> unit = path_busy_until[2 * q + p] = path_hold_until(c)

// ---------------------------------------------------------------------------------------
// Address generation table (AGT), one per queue (HC34 p.25; the loop and configuration
// rules are opentsp's model). A loopable MEM instruction (Read, Write, Gather, Scatter) is reference 0 at
// its dispatch cycle t. Countdown/Step/StreamStep/Iterations records on the same queue in the
// following consecutive cycles configure up to four nested loop levels, inner to outer; a
// level exists once one of its fields is written. Within a level the fields go Countdown ->
// Step -> StreamStep -> Iterations; a write that does not advance in that order starts the
// next level, and fields not received keep the defaults (countdown 0, step 0,
// 2 references). An explicit Countdown below 1 counts as 1.
// Iterations n makes n references; n below 1 counts as 1.
// Timing: a reference takes one cycle and every countdown wait is unconditional (only the
// address step depends on whether the level advances). So the
// next reference issues 1 + the sum of the countdowns of the levels that wrapped and of the
// level that advances after the previous one, with the advancing level's step added. The
// loop runs while its configuration is still arriving (the first inner loop iteration
// may complete before an iterations record arrives): a configuration record dispatches before the AGT step of its cycle, and each
// reference uses the fields received so far. When every level received so far is
// exhausted while the configuration is still open, the AGT waits for the next record
// (opentsp decision); the configuration ends at the first cycle without a configuration
// record, and the loop ends when no level can advance after that.

struct agt_level = { countdown : nat, step : int, sstep : int, iters : nat, idx : nat }

private let agt_default_level : agt_level = struct { countdown = 0, step = 0, sstep = 0, iters = 2, idx = 0 }

enum agt_state = {AGT_IDLE, AGT_CONFIG, AGT_RUN}

struct agt = {
  state  : agt_state,
  instr  : instruction,
  t0     : nat,                    // dispatch cycle of reference 0
  levels : vector(4, agt_level),
  level  : range(0, 3),            // level being configured
  pos    : int,                    // last field written in that level (-1: none)
  offset : int,                    // address offset of the last reference
  soffset : int,                   // stream id offset of the last reference (StreamStep)
  last   : nat,                    // cycle of the last reference
  next   : nat,                    // cycle of the next reference (valid in AGT_RUN)
}

let agt_idle : agt = struct {
  state = AGT_IDLE, instr = ILLEGAL32(0x00000000), t0 = 0, levels = vector_init(agt_default_level),
  level = 0, pos = -1, offset = 0, soffset = 0, last = 0, next = 0,
}

// One AGT per dispatch path: agts is the R path's (and the only one of a
// non-MEM queue, which never uses it), agts_w the W path's.
register agts : vector(144, agt) = vector_init(agt_idle)
register agts_w : vector(144, agt) = vector_init(agt_idle)
// Configuration records each path's open configuration has received (index 2q + path).
register agt_records : vector(288, nat) = vector_init(0)

function agt_get(q : queue_id, p : mem_path) -> agt = if p == 0 then agts[q] else agts_w[q]
function agt_set(q : queue_id, p : mem_path, a : agt) -> unit = if p == 0 then agts[q] = a else agts_w[q] = a

// Repeat n, d (ISCA 2020 Table I: "Repeat the previous instruction n times, with d cycles
// between iterations"): re-issue the queue's previous instruction n more times with d cycles
// between iterations, so the spacing is d + 1, counted from that instruction's own issue
// cycle p: iteration k at p + k (d + 1). The first iteration is never before the
// Repeat's own cycle r: it issues at max(p + d + 1, r), the rest d + 1 apart. The queue is
// busy meanwhile. (NOP N is a delay of N cycles between parcels: its spacing is N + 1.)
struct repeat_state = { instr : instruction, left : nat, gap : nat, next : nat }
register queue_last : vector(144, option(instruction)) = vector_init(None())
// The cycle the queue last issued queue_last (its dispatch, or its latest Repeat iteration).
register queue_last_at : vector(144, nat) = vector_init(0)
register queue_repeat : vector(144, option(repeat_state)) = vector_init(None())

// DeskewRecv (slices/c2c/c2c_insts.sail; the polling loop at program start, ISCA 2022 sec. 3.2): a queue whose DeskewRecv has
// dispatched at `since` and has not yet found its vector holds every later instruction, like a Repeat that is running. At
// each epoch boundary after `since` (cycle mod c2c.hac_period = 0) the queue is looked at in `poll_queue`; when a vector has
// arrived on the link, the poll becomes a Receive dispatched in that cycle.
struct poll_state = { chip : bits(9), link : bits(7), s : streamref, since : nat }
register queue_poll : vector(144, option(poll_state)) = vector_init(None())

private function agt_field_pos(f : agt_field) -> int =
  match f { AGT_COUNTDOWN => 0, AGT_STEP => 1, AGT_STREAMSTEP => 2, AGT_ITERATIONS => 3 }

// Apply one configuration write to the AGT a: the new AGT, and false (with a unchanged) when it would need a fifth level. A pure
// function of the AGT and the write.
function agt_configure_pure(a0 : agt, f : agt_field, v : bits(16)) -> (agt, bool) = {
  var a = a0;
  let fp = agt_field_pos(f);
  var ok : bool = true;
  if fp <= a.pos then {
    let lv = a.level;
    if lv >= 3 then ok = false else { a.level = lv + 1; a.pos = -1 }
  };
  if ok then {
    var l = a.levels[a.level];
    match f {
      AGT_COUNTDOWN  => l.countdown = max(unsigned(v), 1),
      AGT_STEP       => l.step = signed(v),
      AGT_STREAMSTEP => l.sstep = unsigned(v),
      AGT_ITERATIONS => l.iters = max(unsigned(v), 1),
    };
    a.levels[a.level] = l;
    a.pos = fp
  };
  if ok then (a, true) else (a0, false)
}

// Apply one configuration write; false when it would need a fifth level.
private function agt_configure(q : queue_id, p : mem_path, f : agt_field, v : bits(16)) -> bool = {
  let (a, ok) = agt_configure_pure(agt_get(q, p), f, v);
  if ok then agt_set(q, p, a);
  ok
}

// Test hooks (the unit tests are a separate module and cannot see private functions).
function agt_configure_test(q : queue_id, f : agt_field, v : bits(16)) -> bool = agt_configure(q, 0, f, v)

// The loop counters of the next reference after the one at counters `levels`, over the
// first n levels (the levels received so far); None() when none of them can advance.
// Returns the delay from the previous reference (its Issue cycle, plus Wait(countdown) of
// every level that wrapped and of the level that advances) and the step and stream step of
// the advancing level.
private function agt_advance(levels : vector(4, agt_level), n : nat) -> option((vector(4, agt_level), nat, int, int)) = {
  var ls = levels;
  var res : option((vector(4, agt_level), nat, int, int)) = None();
  var done : bool = false;
  var wait : nat = 1;
  foreach (k from 0 to 3) {
    if not(done) & k < n then {
      var l = ls[k];
      wait = wait + l.countdown;
      if l.idx + 1 < l.iters then {
        l.idx = l.idx + 1;
        ls[k] = l;
        res = Some((ls, wait, l.step, l.sstep));
        done = true
      } else {
        l.idx = 0;
        ls[k] = l
      }
    }
  };
  res
}

function agt_advance_test(levels : vector(4, agt_level), n : nat) -> option((vector(4, agt_level), nat, int, int)) = agt_advance(levels, n)

// Number of levels received so far.
private function agt_nlevels(a : agt) -> nat = if a.pos < 0 then 0 else a.level + 1

// With the configuration of an AGT closed (AGT_RUN): the AGT with the cycle of its next reference, or the idle AGT and the cycle
// until which the path stays held when no level can advance (the trailing waits of every level the loop has). An AGT that is not
// running is unchanged. A pure function of the AGT.
function agt_refresh_pure(a : agt) -> (agt, option(nat)) =
  if a.state == AGT_RUN then match agt_advance(a.levels, agt_nlevels(a)) {
    None() => {
      var tail : nat = a.last + 1;
      foreach (k from 0 to 3) { if k < agt_nlevels(a) then tail = tail + a.levels[k].countdown };
      (agt_idle, Some(tail))
    },
    Some((_, d, _, _)) => ({ a with next = a.last + d }, None()),
  }
  else (a, None())

// With the configuration of path p of queue q closed (AGT_RUN): record the cycle of the
// next reference, or go idle when no level can advance.
private function agt_refresh(q : queue_id, p : mem_path) -> unit = {
  let (a, tail) = agt_refresh_pure(agt_get(q, p));
  match tail { Some(t) => path_tail_until[2 * q + p] = t, None() => () };
  agt_set(q, p, a)
}

function agt_refresh_test(q : queue_id, p : mem_path) -> unit = agt_refresh(q, p)

// The configuration of path p of queue q ended (no configuration record at this cycle).
function agt_finish_config_pure(a : agt) -> (agt, option(nat)) = agt_refresh_pure({ a with state = AGT_RUN })

private function agt_finish_config(q : queue_id, p : mem_path) -> unit = {
  let (a, tail) = agt_finish_config_pure(agt_get(q, p));
  match tail { Some(t) => path_tail_until[2 * q + p] = t, None() => () };
  agt_set(q, p, a)
}

// The AGT reference of path p of queue q due at cycle c, as the counters, delay, step and
// stream step of agt_advance (with the fields received so far, also while the
// configuration is still arriving); None() when none is due (idle, parked, waiting for its
// countdown, or every level received so far exhausted).
function agt_due_pure(a : agt, c : nat, parked : bool) -> option((vector(4, agt_level), nat, int, int)) =
  if a.state == AGT_IDLE | parked then None()
  else match agt_advance(a.levels, agt_nlevels(a)) {
    None() => None(),
    Some((ls, d, st, ss)) => if a.last + d <= c then Some((ls, d, st, ss)) else None(),
  }

private function agt_due(q : queue_id, p : mem_path, c : nat, parked : bool) -> option((vector(4, agt_level), nat, int, int)) =
  agt_due_pure(agt_get(q, p), c, parked)

// The AGT after the reference due at cycle c ran (or was blocked): the counters ls, the offsets advanced by the step and the stream
// step, and c as its last reference. The AGT that a loopable reference opens when it dispatches at cycle c.
function agt_after_reference(a : agt, ls : vector(4, agt_level), s : int, ss : int, c : nat) -> agt =
  { a with levels = ls, offset = a.offset + s, soffset = a.soffset + ss, last = c }

function agt_opened(i : instruction, c : nat) -> agt = { agt_idle with state = AGT_CONFIG, instr = i, t0 = c, last = c }

private function report(q : range(0, 144), c : nat, d : diag) -> unit =
  diag_report(c, q, d)

// Execute stage. Reference k of an instruction dispatched at c0 issues at
// c0 + k and EXECUTES at c0 + k + d_skew (its entry's latency, core/timing.sail): it reads
// its stream operands, SRAM and unit state in that cycle (the operands arrive d_skew cycles after the dispatch), and its diagnostics are reported at that cycle.
// Dispatch checks (queue, unit, barrier, slots, banks) stay at the dispatch cycle. An
// element with d_skew > 0 waits in the execute ring; tick(c) runs the elements due at c,
// in dispatch order, before the queues dispatch.
struct exec_element = { q : queue_id, instr : instruction, c0 : nat, k : nat, off : int, soff : int }

register execute_ring : vector(16, list(exec_element)) = vector_init([||])

private function execute_element(e : exec_element) -> unit = {
  issue_cycle = e.c0;
  issue_unit = issuer_of(e.q);
  issue_queue = e.q;
  issue_element = e.k;
  issue_addr_offset = e.off;
  issue_stream_offset = e.soff;
  issue_thread = 0b0;
  issue_lat = latency_of_instr(e.instr);
  observe_site(entry_of(e.instr), exec_cycle(), e.q, e.k, 0, e.c0);
  ecc_begin();
  match execute(e.instr) {
    Issued()      => {
      if ecc_link_bad != sail_zeros(superlanes) then report(e.q, exec_cycle(), LNK_UNC);
      if ecc_bad != sail_zeros(superlanes) then report(e.q, exec_cycle(), ECC_UNC)
    },
    Diagnostic(d) => {
      report(e.q, exec_cycle(), d);
      if ecc_after_diag & ecc_bad != sail_zeros(superlanes) then report(e.q, exec_cycle(), ECC_UNC)
    },
  }
}

// Test hook: execute i as the current issue, with its latency (the unit tests set the other
// issue registers themselves).
function execute_test(i : instruction) -> ExecutionResult = { issue_lat = latency_of_instr(i); execute(i) }

private function append_element(xs : list(exec_element), e : exec_element) -> list(exec_element) =
  match xs { [||] => [|e|], x :: rest => x :: append_element(rest, e) }

// Issue one reference of the instruction dispatched at cycle c0, k cycles later, with an
// AGT address offset and stream offset.
private function run_element(q : queue_id, i : instruction, c0 : nat, k : nat, off : int, soff : int) -> unit = {
  let e : exec_element = struct { q = q, instr = i, c0 = c0, k = k, off = off, soff = soff };
  let s = latency_of_instr(i).d_skew;
  if s == 0 then execute_element(e)
  else { let x = (c0 + k + s) % 16; execute_ring[x] = append_element(execute_ring[x], e) }
}

// Test hook: issue reference 0 of i on queue q, dispatched at c0 (it executes at c0 + d_skew).
function run_element_test(q : queue_id, i : instruction, c0 : nat) -> unit = run_element(q, i, c0, 0, 0, 0)

// The elements whose execute cycle is c (d_skew <= 15, so the 16-slot ring never aliases).
private function execute_all(xs : list(exec_element)) -> unit =
  match xs { [||] => (), e :: rest => { execute_element(e); execute_all(rest) } }

private function execute_due(c : nat) -> unit = {
  let x = c % 16;
  let xs = execute_ring[x];
  execute_ring[x] = [||];
  execute_all(xs)
}

// The instruction a Repeat re-issues: the queue's last dispatched instruction other than
// NOP, Repeat and AGT configuration (a Read/Write pair is not repeatable).
private function repeatable(i : instruction) -> bool =
  match i { NOP(_) => false, REPEAT(_) => false, _ => agt_config(i) == None() }

// Sync, Notify, Deskew and RuntimeDeskew act at dispatch (the barrier, the queue hold),
// which an iteration would not repeat, so they leave nothing to repeat.
private function clears_repeat(i : instruction) -> bool =
  match i { SYNC() => true, NOTIFY() => true, C2C_DESKEW() => true, C2C_RTDESKEW(_) => true, C2C_HOSTWAIT(_) => true, C2C_DESKEWRECV(_, _, _) => true, _ => false }

// Queue q has a DeskewRecv that has not found its vector yet.
function polling(q : queue_id) -> bool = match queue_poll[q] { Some(_) => true, None() => false }

// The first epoch boundary (a multiple of c2c.hac_period) at or after cycle c.
function next_boundary(c : nat) -> nat = c + emod_int(c2c_hac_period - emod_int(c, c2c_hac_period), c2c_hac_period)

// An instruction that needs the whole queue (every instruction other than a MEM reference, a
// MEM configuration record or, on a MEM queue, Ifetch).
private function dispatch_body(q : queue_id, c : nat, i : instruction) -> unit = {
  if clears_repeat(i) then queue_last[q] = None()
  else if repeatable(i) then { queue_last[q] = Some(i); queue_last_at[q] = c };
  queue_busy_until[q] = c + occupancy(i, c);
  match i {
    SYNC()   => barrier_sync(q, c),
    NOTIFY() => barrier_notify(c),
    C2C_HOSTWAIT(n) => if unsigned(n) < host_doorbells then host_wait(q, c, db_idx(unsigned(n))),
    C2C_DESKEWRECV(chip, link, s) => match deskewrecv_diag(issuer_of(q), chip, link, s) {
      Some(d) => report(q, c, d),
      None() => queue_poll[q] = Some(struct { chip = chip, link = link, s = s, since = c }),
    },
    _        => (),
  };
  let _ = run_element(q, i, c, 0, 0, 0);
  ()
}

// What the hold decision reads of a queue and its two dispatch paths at one cycle, as plain values.
struct path_view = { pbusy : bool, ptail : bool, pstate : agt_state }
struct hold_view = { vparked : bool, vhost : bool, vbusy : bool, vrepeat : bool, vpoll : bool, vp : vector(2, path_view) }

function path_view_of(q : queue_id, p : mem_path, c : nat) -> path_view =
  struct { pbusy = path_busy(q, p, c), ptail = path_tail(q, p, c), pstate = agt_get(q, p).state }

// The views of the two paths as a vector indexed by path (a Sail vector literal lists the highest index first).
function path_views(r : path_view, w : path_view) -> vector(2, path_view) = {
  var v : vector(2, path_view) = vector_init(r);
  v[1] = w;
  v
}

function hold_view_of(q : queue_id, c : nat) -> hold_view =
  struct { vparked = queue_parked(q, c), vhost = host_parked(q, c), vbusy = c < queue_busy_until[q], vrepeat = queue_repeat[q] != None(),
           vpoll = polling(q), vp = path_views(path_view_of(q, 0, c), path_view_of(q, 1, c)) }

// A whole-queue instruction needs both paths idle (design sec. 7.6: "an AGT reference issued in
// cycle c also holds the queue for it"): no AGT loop running or configuration open (state not
// AGT_IDLE: an open configuration runs its loop with the fields received so far, so a reference
// may be due in this very cycle), no path busy or in a tail.
function view_paths_busy(v : hold_view) -> bool =
  v.vp[0].pbusy | v.vp[1].pbusy | v.vp[0].pstate != AGT_IDLE | v.vp[1].pstate != AGT_IDLE | v.vp[0].ptail | v.vp[1].ptail

// The queue is free (design sec. 7.7): not parked, not busy, no Repeat, no poll, both paths idle.
function view_free(v : hold_view) -> bool = not(v.vparked | v.vhost | v.vbusy | v.vrepeat | v.vpoll) & not(view_paths_busy(v))

// The hold decision as a pure function of the view. own_barrier: the instruction is a Sync or Notify; whole_free: an Ifetch (or
// IfetchM) on a MEM queue needs only its own path, so it does not wait for the whole queue; path: the dispatch path of the instruction;
// cfg_record: it is an AGT configuration record.
function hold_decide(v : hold_view, own_barrier : bool, whole_free : bool, path : option(mem_path), cfg_record : bool) -> bool = {
  let held : bool = match path {
    // A path's loop runs while its configuration arrives: it holds the path against
    // new references in both states; only its configuration records get through.
    Some(p) => v.vp[p].pbusy | v.vp[p].ptail | v.vp[p].pstate == AGT_RUN | (v.vp[p].pstate == AGT_CONFIG & not(cfg_record)),
    None()  => not(whole_free) & view_paths_busy(v),
  };
  (not(own_barrier) & v.vparked) | v.vhost | v.vbusy | held | v.vrepeat | v.vpoll
}

// Is instruction i of the head parcel of queue q held at cycle c (design sec. 7.6)? An
// instruction on dispatch path p (Some(p)) needs only its path free; any other instruction
// the whole queue; strict: the parcel breaks the slot rule, so every instruction needs the
// whole queue (both paths, Ifetch included). Held when
//   - the queue is parked by a Sync (a queue's own Sync and Notify are the barrier; the
//     control channel's Sync parks every data queue);
//   - the queue is busy (c < busy-until: Deskew; an instruction issued this cycle) or a
//     Repeat runs;
//   - its path is busy, in the tail of a finished loop, runs a loop, or holds an open
//     configuration that i does not continue; an instruction that needs the whole
//     queue waits for either path to have no loop and no open configuration (view_paths_busy).
private function hold_check(q : queue_id, c : nat, i : instruction, path : option(mem_path), strict : bool) -> bool = {
  let own_barrier : bool = match i { SYNC() => true, NOTIFY() => true, _ => false };
  let ifetch : bool = not(strict) & (match i { IFETCHM(_) => true, IFETCH(_) => true, _ => false });   // IfetchM
  hold_decide(hold_view_of(q, c), own_barrier, ifetch & q <= 87, path, agt_config(i) != None())
}

// An IfetchM issued at cycle c, as a new instruction or as a Repeat iteration: it holds the R path through the
// cycle of its second read (c + 1), and the bank of word a + 1 is taken in c + 1 (the W side of that cycle is checked against it).
// One rule for both, so that `ifetchm a; ifetchm a` and `ifetchm a; repeat 1, d` are one schedule.
private function reserve_local_fetch(q : queue_id, i : instruction, c : nat) -> unit = {
  path_busy_until[2 * q] = fetchm_hold_until(c);
  if q <= 87 then match fetchm_bank_note(i, c, mem_segment[q]) {
    Some(n) => local_fetch_bank[q] = Some(n),
    None() => (),
  }
}

// A Repeat iteration of i at cycle c: the execution, and for an IfetchM the reservation of its second beat.
private function repeat_issue(q : queue_id, i : instruction, c : nat) -> unit = {
  let _ = run_element(q, i, c, 0, 0, 0);
  match i { IFETCHM(_) => reserve_local_fetch(q, i, c), _ => () }
}

// What dispatching i does once nothing holds it: the unit check. The hold conditions are
// those of hold_check, which the head parcel has already passed (an assert, not an outcome).
private function dispatch_check(q : queue_id, c : nat, i : instruction, path : option(mem_path)) -> option(diag) = {
  if hold_check(q, c, i, path, false) then internal_error(__FILE__, __LINE__, "dispatch of a held instruction");
  if not(issued_by(i, issuer_of(q).unit)) then
    Some(match i { ILLEGAL32(_) => DEC_OPCODE, ILLEGAL64(_) => DEC_OPCODE, _ => QUE_UNIT })
  else None()
}

// A Repeat n, d dispatched at cycle c, when the queue's last instruction issued at last_at: whether the first iteration issues in this
// very cycle, and the Repeat that remains. Iterations are at last_at + k (d + 1); the first never before c. A pure function.
function repeat_plan(prev : instruction, n : nat, d : nat, last_at : nat, c : nat) -> (bool, option(repeat_state)) = {
  let gap = d + 1;
  let first = max(last_at + gap, c);
  if first == c then (true, if n <= 1 then None() else Some(struct { instr = prev, left = max(n - 1, 0), gap = gap, next = c + gap }))
  else (false, Some(struct { instr = prev, left = n, gap = gap, next = first }))
}

// The Repeat that remains after an iteration at cycle c.
function repeat_after(r : repeat_state, c : nat) -> option(repeat_state) =
  if r.left <= 1 then None() else Some({ r with left = max(r.left - 1, 0), next = c + r.gap })

private function dispatch_decoded(q : queue_id, c : nat, i : instruction) -> unit =
  match dispatch_check(q, c, i, None()) {
    Some(d) => report(q, c, d),
    None() => match i {
      REPEAT(n, d) => match queue_last[q] {
        None() => report(q, c, RES_NOT_READY),         // nothing to repeat
        Some(prev) => {
          queue_busy_until[q] = c + 1;
          // iterations at p + (d+1), p + 2(d+1), ... from the repeated instruction's
          // issue cycle p; the first not before the Repeat's own cycle c.
          let (now, rest) = repeat_plan(prev, unsigned(n), unsigned(d), queue_last_at[q], c);
          if now then {
            observe_site("repeat", c, q, 0, 0, queue_last_at[q]);
            repeat_issue(q, prev, c);
            queue_last_at[q] = c
          };
          queue_repeat[q] = rest
        },
      },
      _ => dispatch_body(q, c, i),
    },
  }

// ---------------------------------------------------------------------------------------
// Instructions per parcel (design sec. 7.5). A non-MEM queue
// dispatches one instruction per parcel. A MEM queue has slots (two dispatch paths, one per
// SRAM port): one Ifetch; per liquid thread t one R-type reference
// (Read, Gather) and one W-type reference (Write, Scatter); one configuration record per
// queue (the AGT takes one configuration record per cycle); or one instruction that needs the whole queue, alone. When a
// parcel's instructions do not fit (two in one slot, a whole-queue instruction with another, a
// NOP inside the parcel, an instruction that does not decode among several), every instruction
// of the parcel is OTS-E-QUE-001 and none dispatches, so the result does not depend on the
// order of the instructions.

struct cycle_recs = {
  ok     : bool,                          // the instructions dispatch (each in its slot)
  n      : nat,                           // number of instructions
  err    : option(diag),                  // a lone instruction that does not decode
  single : option(instruction),           // a lone instruction that needs the whole queue
  ifetch : option(instruction),
  refs   : vector(4, option(instruction)), // index 2 * path + thread
  cfg    : option(instruction),           // the configuration record (one per cycle)
}

private let no_recs : cycle_recs = struct {
  ok = true, n = 0, err = None(), single = None(), ifetch = None(), refs = vector_init(None()), cfg = None(),
}

// The path a configuration record configures (mem_insts.sail "Dispatch paths"): the path
// whose AGT is configuring; when both are, the one with fewer records so far (R on a tie),
// so a Read/Write pair's records alternate R, W, R, W, ...; with neither (a stray record), R.
function cfg_path_pick(r : bool, w : bool, recs_r : nat, recs_w : nat) -> mem_path =
  if r & w then (if recs_w < recs_r then 1 else 0)
  else if w then 1
  else 0

private function cfg_path(q : queue_id) -> mem_path =
  cfg_path_pick(agts[q].state == AGT_CONFIG, agts_w[q].state == AGT_CONFIG, agt_records[2 * q], agt_records[2 * q + 1])

private function bit_index(t : bits(1)) -> range(0, 1) = if t == 0b1 then 1 else 0

// Sort the decoded records of MEM queue q into their slots; false when two share one.
private function place(r : cycle_recs, i : instruction) -> (cycle_recs, bool) = {
  var x = r;
  var ok : bool = true;
  match (agt_config(i), ref_path(i), i) {
    (Some(_), _, _) => if x.cfg != None() then ok = false else x.cfg = Some(i),
    (None(), Some(p), _) => {
      let k = 2 * p + bit_index(ref_thread(i));
      if x.refs[k] != None() then ok = false else x.refs[k] = Some(i)
    },
    (None(), None(), IFETCH(_)) => if x.ifetch != None() then ok = false else x.ifetch = Some(i),
    (None(), None(), IFETCHM(_)) => if x.ifetch != None() then ok = false else x.ifetch = Some(i),
    _ => ok = false,
  };
  (x, ok)
}

// Place every record of a parcel into its slot: the slots filled, and whether every record decoded and found its own slot.
private function place_all(r : cycle_recs, xs : list(result(instruction, diag)), ok : bool) -> (cycle_recs, bool) =
  match xs {
    [||] => (r, ok),
    y :: ys => match y {
      Err(_) => place_all(r, ys, false),
      Ok(i)  => { let (z, fit) = place(r, i); place_all(z, ys, ok & fit) },
    },
  }

private function items_len(xs : list(result(instruction, diag))) -> nat =
  match xs { [||] => 0, _ :: rest => 1 + items_len(rest) }

// The instructions of a parcel on queue q, sorted into slots (one that needs the whole queue,
// or that does not decode, is left to dispatch()). A NOP inside a parcel is never alone and
// fits no slot (design sec. 7.3).
private function classify_items(q : queue_id, items : list(result(instruction, diag))) -> cycle_recs = {
  let n = items_len(items);
  let r : cycle_recs = { no_recs with n = n };
  match items {
    [||] => r,
    x :: rest => match rest {
      [||] => match x {
        Err(d) => { r with ok = false, err = Some(d) },
        Ok(i)  => if q > 87 then { r with single = Some(i) } else {
          let (y, ok) = place(r, i);
          if ok then y else { r with single = Some(i) }
        },
      },
      _ =>
        if q > 87 then { r with ok = false }
        else {
          let (r1, ok) = place_all(r, items, true);
          // IfetchM reads SRAM through the R port, for every tile (both liquid
          // threads), so it does not fit with an R-type reference (Read, Gather) of any thread.
          let fetchm : bool = match r1.ifetch { Some(IFETCHM(_)) => true, _ => false };
          { r1 with ok = ok & not(fetchm & (r1.refs[0] != None() | r1.refs[1] != None())) }
        },
    },
  }
}

// True when the cycle carries a configuration record: open configurations continue.
private function config_for(r : cycle_recs) -> bool = r.ok & r.cfg != None()

// Two references of one liquid thread on the two paths in one cycle must use different
// banks of the pseudo-dual-port SRAM (ISCA 2020 sec. III-B): at most one R-type and one W-type
// reference concurrently, in different banks.
// An indirect reference's banks come from its map stream; such a pair conflicts (opentsp
// decision: the dispatch cannot prove the banks differ). References of different threads
// use different tiles and never conflict.
struct ref_tag = { thread : bits(1), bank : bank_use }

private function tags_clash(a : ref_tag, b : ref_tag) -> bool =
  a.thread == b.thread & banks_meet(a.bank, b.bank)

private function tag_of(i : instruction, off : int) -> ref_tag = struct { thread = ref_thread(i), bank = ref_bank(i, off) }

// The bank-conflict report (microarchitecture.md sec. 15.5). A reference of a conflicting pair as the report needs it: the
// instruction, its AGT address offset and its AGT stream offset.
type clash_ref = (instruction, int, int)

private function clash_addr(r : clash_ref) -> int =
  match r { (MEM_DIRECT(_, SramAddr(a), _, _), off, _) => unsigned(a) + off, _ => 0 }

// The data stream (with its AGT stream offset) of an R-type reference: a Read's destination or a Gather's. None for IfetchM.
private function clash_stream(r : clash_ref) -> option(streamref) = {
  let (i, _, soff) = r;
  let s0 : option(streamref) = match i {
    MEM_DIRECT(MEM_READ, _, d, _) => Some(d),
    MEM_GATHER(d, _, _, _) => Some(d),
    _ => None(),
  };
  match s0 {
    Some(Streamref(s)) => {
      let e = unsigned(s[4 .. 0]) + soff;
      if 0 <= e & e <= 31 then Some(Streamref([s[5]] @ get_slice_int(5, e, 0))) else None()
    },
    None() => None(),
  }
}

// A bank conflict (OTS-E-MEM-001) of the R-type reference rd and the W-type reference wr dispatched in cycle c: S[2] of
// superlane 0, S[4..16] the write address and S[17..29] the read address (an indirect reference has no address at dispatch:
// zero), at the write gap of the slice in the commit cycle (dispatch + d_func), on the row of the read's data stream.
private function bank_report(q : queue_id, c : nat, a : clash_ref, b : clash_ref) -> unit = {
  let (ia, _, _) = a;
  let (rd, wr) : (clash_ref, clash_ref) = match ref_path(ia) { Some(0) => (a, b), _ => (b, a) };
  match clash_stream(rd) {
    None() => (),
    Some(ds) => {
      let r = stream_row(ds);
      let iu = issuer_of(q);
      match write_gap(iu.unit, iu.slice, r) {
        None() => (),
        Some(g) => {
          var sp : vector(30, bits(superlanes)) = vector_init(sail_zeros(superlanes));
          var s2 : bits(superlanes) = sp[2];
          s2[0] = bitone;
          sp[2] = s2;
          // The low 13 bits of each address (the report has 13 stream positions per address).
          let wa : bits(13) = get_slice_int(13, clash_addr(wr), 0);
          let ra : bits(13) = get_slice_int(13, clash_addr(rd), 0);
          foreach (i from 0 to 12) {
            if wa[i] == bitone then { var m = sp[4 + i]; m[0] = bitone; sp[4 + i] = m };
            if ra[i] == bitone then { var m = sp[17 + i]; m[0] = bitone; sp[17 + i] = m }
          };
          var xs : bits(superlanes) = sail_zeros(superlanes);
          xs[0] = bitone;
          if srf_report(r, g, c + latency_of(T_MEM).d_func, 1, xs, sp) then lint(q, c, LINT_RAS_EVICT)
        },
      }
    },
  }
}

// The reference of path p's AGT due this cycle (st, from agt_due) as the report needs it.
private function loop_clash_ref(q : queue_id, p : mem_path, s : int, ss : int) -> clash_ref = {
  let a = agt_get(q, p);
  (a.instr, a.offset + s, a.soffset + ss)
}

// A configuration record on path p.
private function dispatch_cfg(q : queue_id, c : nat, p : mem_path, i : instruction) -> unit =
  match dispatch_check(q, c, i, Some(p)) {
    Some(d) => report(q, c, d),
    None() => match agt_config(i) {
      // A configuration record continues the configuration of the loopable instruction
      // before it on its path. With no configuration open (a stray record, or after the
      // loop was aborted) it is OTS-E-OPR-001. A fifth level is OTS-E-OPR-001 and aborts the loop.
      Some((f, v)) =>
        if agt_get(q, p).state != AGT_CONFIG then report(q, c, OPR_RANGE)
        else {
          agt_records[2 * q + p] = agt_records[2 * q + p] + 1;
          if not(agt_configure(q, p, f, v)) then { report(q, c, OPR_RANGE); agt_set(q, p, agt_idle) }
        },
      None() => internal_error(__FILE__, __LINE__, "configuration record"),
    },
  }

type agt_step = option((vector(4, agt_level), nat, int, int))

private function due_ref(q : queue_id, p : mem_path, st : agt_step) -> option(clash_ref) =
  match st { Some((_, _, s, ss)) => Some(loop_clash_ref(q, p, s, ss)), None() => None() }

// The AGT reference of path p due at cycle c (st, from agt_due): it runs unless blocked by
// the bank rule (then it is OTS-E-MEM-001 and has no effect); either way it holds its path
// and the loop advances (agt_refresh: next reference, or idle).
private function run_loop(q : queue_id, c : nat, p : mem_path, st : agt_step, blocked : bool, partner : option(clash_ref)) -> unit =
  match st {
    None() => (),
    Some((ls, _, s, ss)) => {
      let a = agt_get(q, p);
      if blocked then {
        report(q, c, MEM_BANK);
        match partner { Some(o) => bank_report(q, c, loop_clash_ref(q, p, s, ss), o), None() => () }
      }
      else { let _ = run_element(q, a.instr, a.t0, max(c - a.t0, 0), a.offset + s, a.soffset + ss); () };
      hold_path(q, p, c);
      agt_set(q, p, agt_after_reference(a, ls, s, ss, c));
      agt_refresh(q, p)
    },
  }

// The bank rule of one MEM queue and cycle, a pure function of the (thread, bank) tags: rt are the tags of the new references
// (index 2 * path + thread), go which of them passed their checks, loop the tags of the two paths' AGT references due,
// fm_active whether an IfetchM uses a bank this cycle (a new fetch, or the second beat of a pending one) and fm_bank that bank.
// Results: vbank, the new references that lose; vfm_clash, a new IfetchM is canceled; vfm_loop, the W path's AGT reference
// loses to the IfetchM; vlblock, the AGT references that lose; vlsrc, the partner of each blocked AGT reference as the
// bank-conflict report needs it (-1 none, 4 the other path's AGT reference, 0..3 the new reference of that index).
struct bank_verdict = { vbank : vector(4, bool), vfm_clash : bool, vfm_loop : bool, vlblock : vector(2, bool), vlsrc : vector(2, int) }

function bank_rule(rt : vector(4, option(ref_tag)), go : vector(4, bool), loop : vector(2, option(ref_tag)), fm_active : bool, fm_bank : bank_use) -> bank_verdict = {
  var bank : vector(4, bool) = vector_init(false);
  foreach (t from 0 to 1) {
    foreach (p from 0 to 1) {
      let k = 2 * p + t;
      let o = 2 * (1 - p) + t;
      match rt[k] {
        Some(me) => if go[k] then {
          let new_clash : bool = match rt[o] { Some(j) => go[o] & tags_clash(me, j), None() => false };
          let loop_clash : bool = match loop[1 - p] { Some(tg) => tags_clash(me, tg), None() => false };
          if new_clash | loop_clash then bank[k] = true
        },
        None() => (),
      }
    }
  };
  // the IfetchM bank rule against the W side of the cycle.
  var fm_clash : bool = false;
  if fm_active then {
    foreach (t from 0 to 1) {
      match rt[2 + t] {
        Some(j) => if go[2 + t] then {
          let hit : bool = banks_meet(j.bank, fm_bank);
          if hit then { bank[2 + t] = true; fm_clash = true }
        },
        None() => (),
      }
    }
  };
  var fm_loop : bool = false;
  if fm_active then match loop[1] {
    Some(tg) => if banks_meet(tg.bank, fm_bank) then { fm_loop = true; fm_clash = true },
    None() => (),
  };
  // An AGT reference is blocked by a clashing reference on the other path: the other
  // path's AGT reference, or a new reference that passed its checks.
  var lblock : vector(2, bool) = vector_init(false);
  var lsrc : vector(2, int) = vector_init(-1);
  foreach (p from 0 to 1) {
    match loop[p] {
      Some(me) => {
        var hit : bool = match loop[1 - p] { Some(o) => tags_clash(me, o), None() => false };
        if hit then lsrc[p] = 4;
        foreach (t from 0 to 1) {
          match rt[2 * (1 - p) + t] { Some(j) => if go[2 * (1 - p) + t] & tags_clash(me, j) then { hit = true; lsrc[p] = 2 * (1 - p) + t }, None() => () }
        };
        lblock[p] = hit | (p == 1 & fm_loop)
      },
      None() => (),
    }
  };
  struct { vbank = bank, vfm_clash = fm_clash, vfm_loop = fm_loop, vlblock = lblock, vlsrc = lsrc }
}

// The (thread, bank) of path p's AGT reference due this cycle.
private function loop_tag(q : queue_id, p : mem_path, st : agt_step) -> option(ref_tag) =
  match st { Some((_, _, s, _)) => { let a = agt_get(q, p); Some(tag_of(a.instr, a.offset + s)) }, None() => None() }

// The AGT references due at cycle c on both paths.
private function due_refs(q : queue_id, c : nat, parked : bool) -> vector(2, agt_step) = {
  var due : vector(2, agt_step) = vector_init(None());
  foreach (p from 0 to 1) { due[p] = agt_due(q, p, c, parked) };
  due
}

// The records of MEM queue q at cycle c that fit their slots, with the AGT references due
// this cycle. The configuration record dispatches before the AGT references are worked
// out, so a reference of this cycle sees it.
private function dispatch_slots(q : queue_id, c : nat, r : cycle_recs, parked : bool) -> unit = {
  var issued : nat = 0;
  var last : option(instruction) = None();
  // A new fetch uses its first bank now; a pending fetch uses
  // its second bank now. Only a new fetch can be canceled by a bank conflict.
  var fm_ok : bool = false;
  var fm_bank : bank_use = BankNone();
  var fm_prior : bool = false;
  match local_fetch_bank[q] {
    Some(bn) => if bn.bn_at == c then { fm_prior = true; fm_bank = addr_bank(bn.bn_addr) },
    None() => (),
  };
  match r.ifetch {
    None() => (),
    Some(i) => {
      let fpath : option(mem_path) = match i { IFETCHM(_) => Some(0), _ => None() };
      match dispatch_check(q, c, i, fpath) {
        Some(d) => report(q, c, d),
        None() => match i {
          IFETCHM(SramAddr(a)) => {
            fm_ok = true;
            fm_bank = addr_bank(unsigned(a))
          },
          _ => { let _ = run_element(q, i, c, 0, 0, 0); issued = issued + 1; last = Some(i) },
        },
      }
    },
  };
  match r.cfg { Some(i) => dispatch_cfg(q, c, cfg_path(q), i), None() => () };
  let due = due_refs(q, c, parked);
  var loop : vector(2, option(ref_tag)) = vector_init(None());
  // the AGT references as the bank-conflict report needs them, taken before any of them runs (running advances the loop)
  var lref : vector(2, option(clash_ref)) = vector_init(None());
  foreach (p from 0 to 1) { loop[p] = loop_tag(q, p, due[p]); lref[p] = due_ref(q, p, due[p]) };
  // References: each checked on its own path; then the bank rule, against the other path's
  // reference of the same thread in this cycle (a new one, or an AGT reference).
  var go : vector(4, bool) = vector_init(false);
  foreach (k from 0 to 3) {
    match r.refs[k] {
      None() => (),
      Some(i) => {
        let p : mem_path = if k >= 2 then 1 else 0;
        match dispatch_check(q, c, i, Some(p)) {
          Some(d) => report(q, c, d),
          None()  => go[k] = true,
        }
      },
    }
  };
  var rt : vector(4, option(ref_tag)) = vector_init(None());
  foreach (k from 0 to 3) { match r.refs[k] { Some(i) => rt[k] = Some(tag_of(i, 0)), None() => () } };
  let bv = bank_rule(rt, go, loop, fm_ok | fm_prior, fm_bank);
  let bank = bv.vbank;
  let fm_clash = bv.vfm_clash;
  if fm_ok then {
    if fm_clash then report(q, c, MEM_BANK)
    else match r.ifetch {
      Some(i) => {
        let _ = run_element(q, i, c, 0, 0, 0);
        reserve_local_fetch(q, i, c);
        issued = issued + 1; last = Some(i)
      },
      None() => (),
    }
  };
  let lblock = bv.vlblock;
  var lpart : vector(2, option(clash_ref)) = vector_init(None());
  foreach (p from 0 to 1) {
    let src = bv.vlsrc[p];
    if src == 4 then lpart[p] = lref[1 - p]
    else if src >= 0 then match r.refs[emod_int(src, 4)] { Some(j) => lpart[p] = Some((j, 0, 0)), None() => () }
  };
  foreach (p from 0 to 1) { run_loop(q, c, p, due[p], lblock[p], lpart[p]) };
  foreach (k from 0 to 3) {
    match r.refs[k] {
      Some(i) => if go[k] then {
        if bank[k] then {
          report(q, c, MEM_BANK);
          // its partner: the other path's new reference of the thread, else that path's AGT reference
          let po : mem_path = if k >= 2 then 0 else 1;
          let o = 2 * po + k % 2;
          var partner : option(clash_ref) = None();
          match loop[po] { Some(tg) => if tags_clash(tag_of(i, 0), tg) then partner = lref[po], None() => () };
          match r.refs[o] { Some(j) => if go[o] & tags_clash(tag_of(i, 0), tag_of(j, 0)) then partner = Some((j, 0, 0)), None() => () };
          match partner { Some(x) => bank_report(q, c, (i, 0, 0), x), None() => () }
        } else {
          let p : mem_path = if k >= 2 then 1 else 0;
          hold_path(q, p, c);
          issued = issued + 1;
          last = Some(i);
          // Reference 0 opens its path's AGT configuration when it dispatches, whatever its
          // execution (at dispatch + d_skew) reports; two references dispatched on the path
          // in this cycle open no loop (opentsp decisions).
          let alone : bool = r.refs[k - (k % 2) + (1 - k % 2)] == None();
          run_element(q, i, c, 0, 0, 0);
          if is_loopable(i) & alone then {
            agt_set(q, p, agt_opened(i, c));
            agt_records[2 * q + p] = 0
          }
        }
      },
      None() => (),
    }
  };
  // Repeat re-issues the queue's last instruction only when a cycle issued exactly one (a
  // Read/Write pair, or a reference with an Ifetch, is not repeatable; M20).
  if issued > 1 then queue_last[q] = None() else if issued == 1 then { queue_last[q] = last; queue_last_at[q] = c }
}

private function dispatch(q : queue_id, c : nat, r : cycle_recs, parked : bool) -> unit = {
  let n = r.n;
  // The queue of an absent MEM slice has no unit. Every instruction of a parcel it
  // dispatches is OTS-E-QUE-002 (one each, whatever the instruction, including one that does
  // not decode and a parcel that would break the slot rule), and nothing else happens.
  if n != 0 & not(queue_present(q)) then foreach (k from 1 to n) { report(q, c, QUE_UNIT) }
  else {
  let slots : bool = n != 0 & r.ok & r.single == None();
  // Without slot instructions, the due AGT references issue on their own (only they can clash).
  if not(slots) then {
    let due = due_refs(q, c, parked);
    let both : bool = match (loop_tag(q, 0, due[0]), loop_tag(q, 1, due[1])) { (Some(a), Some(b)) => tags_clash(a, b), _ => false };
    let l0 = due_ref(q, 0, due[0]);
    let l1 = due_ref(q, 1, due[1]);
    run_loop(q, c, 0, due[0], both, l1);
    run_loop(q, c, 1, due[1], both, l0)
  };
  if n == 0 then ()
  else if not(r.ok) then {
    match r.err {
      Some(d) => report(q, c, d),                                // one instruction that does not decode
      None()  => foreach (k from 1 to n) { report(q, c, QUE_BUSY) },   // a parcel that breaks the slot rule
    }
  }
  else match r.single {
    Some(i) => dispatch_decoded(q, c, i),
    None()  => dispatch_slots(q, c, r, parked),
  }
  }
}

// Writer order for OTS-E-SRF-002: the lowest (queue, plane) writer keeps the value.
private function writer_before(a : commit, b : commit) -> bool = a.q < b.q | (a.q == b.q & a.b < b.b)

private function same_target(a : commit, b : commit) -> bool = a.r == b.r & a.id == b.id & a.g == b.g

// The SXM OR merge (HC34 p.31, ISCA 2020 Fig. 4): the select (SEL) and
// transposer (TSP) outputs of one hemisphere pass through an OR before the turn-out. Two
// writes merge when one comes from that hemisphere's SEL queue and the other from one of
// its TSP queues; any other pair of writes to one target collides (OTS-E-SRF-002).
// SEL = 1, TSP = 2, anything else 0; and the hemisphere (0 west, 1 east).
private function sxm_or_kind(q : range(0, 144)) -> (range(0, 2), range(0, 1)) =
  if q > 143 then (0, 0) else {
    let i = issuer_of(q);
    match i.unit {
      U_SXM_W_SEL => (1, i.sxm), U_SXM_E_SEL => (1, i.sxm),
      U_SXM_W_TSP => (2, i.sxm), U_SXM_E_TSP => (2, i.sxm),
      _           => (0, 0),
    }
  }

private function or_pair(a : commit, b : commit) -> bool = {
  let (ka, ha) = sxm_or_kind(a.q);
  let (kb, hb) = sxm_or_kind(b.q);
  ka != 0 & kb != 0 & ka != kb & ha == hb
}

// Commit e (at position i) against the cycle's list xs: (collides, kept, value). The writes
// to one row/stream/gap in one cycle form a group (the same gap and cycle is the same
// diagonal). The writes of one (queue, byte plane) are one writer: on a MEM queue they are
// the two liquid threads' results, each zero on the other thread's tiles, so their OR is
// the per-tile merge; the first of them in the list stands for the writer. A group
// of exactly two writers that are an OR pair merges: no collision, and the lower writer
// stores the OR of both (the result does not depend on the order of the records). Any
// other group of two or more writers collides: each reports, the lowest (queue, plane)
// writer keeps its own value.
// The writes of a cycle that meet commit e at position i, as sums over the cycle's list: the number of other writers, whether they
// all pair with e as an OR pair, whether e is the first writer, whether e stands for its own (queue, plane) writer, and the ORs of the
// writer's own values and of the other writers' values.
struct win_acc = { others : nat, seen : bits(580), paired : bool, first : bool, stands : bool, own : vec, merged : vec, own_ov : ecc_ov, merged_ov : ecc_ov }

function commit_scan(e : commit, i : nat, ys : list(commit), j : nat, a : win_acc) -> win_acc =
  match ys {
    [||] => a,
    o :: rest => {
      var x = a;
      if same_target(e, o) then {
        if o.q == e.q & o.b == e.b then {
          if j != i then { x.own = x.own | o.value; x.own_ov = x.own_ov | o.ov; if j < i then x.stands = false }
        } else {
          // Count distinct (queue, plane) writers; all of their records still merge below.
          let writer : range(0, 579) = 4 * o.q + o.b;
          if x.seen[writer] == bitzero then { x.others = x.others + 1; x.seen[writer] = bitone };
          if not(or_pair(e, o)) then x.paired = false;
          if writer_before(o, e) then x.first = false;
          x.merged = x.merged | o.value;
          x.merged_ov = x.merged_ov | o.ov
        }
      };
      commit_scan(e, i, rest, j + 1, x)
    },
  }

function commit_wins(e : commit, i : nat, xs : list(commit)) -> (bool, bool, vec, ecc_ov) = {
  let a0 : win_acc = struct { others = 0, seen = sail_zeros(580), paired = true, first = true, stands = true, own = e.value, merged = zero_vec, own_ov = e.ov, merged_ov = ecc_ov_zero };
  let a = commit_scan(e, i, xs, 0, a0);
  if not(a.stands) then (false, false, a.own, a.own_ov)
  else if a.others == 0 then (false, true, a.own, a.own_ov)
  else if a.others == 1 & a.paired then (false, a.first, a.own | a.merged, a.own_ov | a.merged_ov)
  else (true, a.first, a.own, a.own_ov)
}

// Apply the commits due at cycle c, then clear their ring slot. Two writes to one row,
// stream and gap in one cycle are OTS-E-SRF-002, reported on each writer's queue (once per
// written stream); the lowest (queue, plane) writer's value is kept. A SEL and
// a TSP write of one SXM hemisphere are not a collision: the register takes their OR.
private function commit_apply(c : nat, all : list(commit), xs : list(commit), i : nat, hits : vector(145, nat)) -> vector(145, nat) =
  match xs {
    [||] => hits,
    e :: rest => {
      let (collide, win, value, vov) = commit_wins(e, i, all);
      var h = hits;
      if collide then h[e.q] = h[e.q] + 1;
      if win then { let _ = srf_write_ov(e.r, e.id, e.g, c, value, vov); () };
      commit_apply(c, all, rest, i + 1, h)
    },
  }

function commit_due(c : nat) -> unit = {
  let k = c % 128;
  let all = pending[k];
  let hits = commit_apply(c, all, all, 0, vector_init(0));
  // Reports in queue order, so they do not depend on the order of the records.
  foreach (q from 0 to 144) { foreach (j from 1 to hits[q]) { report(q, c, SRF_COLLIDE) } };
  pending[k] = [||]
}

// ---------------------------------------------------------------------------------------
// End of cycle: apply the staged MXM plane parts and SXM maps. Queues that
// staged the same part of one plane, or the same SXM map, in this cycle raise
// OTS-E-RES-003 (once each, in queue order) and none of their staged changes take effect;
// an ACC among them also loses its scheduled stream writes (an MXM queue's only stream
// writes committing at that cycle).

private function q_in(q : range(0, 144), xs : list(range(0, 144))) -> bool =
  match xs { [||] => false, y :: rest => y == q | q_in(q, rest) }

// The other queues that staged an ABC product of the plane of e in this cycle, and whether every such ABC and e are FP8 ABCs.
private function abc_fp8(e : mxm_staged) -> bool = match e.value.abc { n :: _ => n.ptype == W_FP8, _ => false }

private function abc_others(e : mxm_staged, xs : list(mxm_staged)) -> int =
  match xs { [||] => 0, o :: rest => (if o.q != e.q & o.plane == e.plane & o.part == P_ABC then 1 else 0) + abc_others(e, rest) }

private function abc_others_fp8(e : mxm_staged, xs : list(mxm_staged)) -> bool =
  match xs { [||] => true, o :: rest => (if o.q != e.q & o.plane == e.plane & o.part == P_ABC then abc_fp8(o) else true) & abc_others_fp8(e, rest) }

// Two queues staging the same part of one plane clash, for the accumulators the same accumulator. The one exception is the FP8
// rate: with mxm.fp8_rate 2 a plane accepts two FP8 ABCs in one cycle; a third, or an ABC of another type, clashes with them.
private function mxm_clash(e : mxm_staged, xs : list(mxm_staged)) -> bool =
  if e.part == P_ABC then
    abc_others(e, xs) > 0 & not(mxm_fp8_rate == 2 & abc_others(e, xs) == 1 & abc_fp8(e) & abc_others_fp8(e, xs))
  else match xs { [||] => false, o :: rest => (o.q != e.q & o.plane == e.plane & o.part == e.part & o.acc_idx == e.acc_idx) | mxm_clash(e, rest) }

private function sxm_clash(e : sxm_staged, xs : list(sxm_staged)) -> bool =
  match xs { [||] => false, o :: rest => (o.q != e.q & o.sxm == e.sxm & o.map == e.map) | sxm_clash(e, rest) }

private function drop_commits(xs : list(commit), bad : list(range(0, 144))) -> list(commit) =
  match xs { [||] => [||], e :: rest => if q_in(e.q, bad) & is_mxm_queue(e.q) then drop_commits(rest, bad) else e :: drop_commits(rest, bad) }

// The queues with a staged part that another queue staged too, added to bad (a queue once).
private function mxm_bad(all : list(mxm_staged), xs : list(mxm_staged), bad : list(range(0, 144))) -> list(range(0, 144)) =
  match xs {
    [||] => bad,
    e :: rest => mxm_bad(all, rest, if mxm_clash(e, all) & not(q_in(e.q, bad)) then e.q :: bad else bad),
  }

private function sxm_bad(all : list(sxm_staged), xs : list(sxm_staged), bad : list(range(0, 144))) -> list(range(0, 144)) =
  match xs {
    [||] => bad,
    e :: rest => sxm_bad(all, rest, if sxm_clash(e, all) & not(q_in(e.q, bad)) then e.q :: bad else bad),
  }

private function mxm_apply_ok(xs : list(mxm_staged), bad : list(range(0, 144))) -> unit =
  match xs { [||] => (), e :: rest => { if not(q_in(e.q, bad)) then mxm_apply(e); mxm_apply_ok(rest, bad) } }

private function sxm_apply_ok(xs : list(sxm_staged), bad : list(range(0, 144))) -> unit =
  match xs {
    [||] => (),
    e :: rest => {
      if not(q_in(e.q, bad)) then { sxm_maps[e.sxm][e.map] = e.value; sxm_map_bij[e.sxm][e.map] = e.bij; sxm_map_ready[e.sxm][e.map] = e.ready };
      sxm_apply_ok(rest, bad)
    },
  }

function unit_state_flush(c : nat) -> unit = {
  let ms = mxm_stage_list;
  let ss = sxm_stage_list;
  let bad = sxm_bad(ss, ss, mxm_bad(ms, ms, [||]));
  foreach (q from 0 to 143) { if q_in(q, bad) then report(q, c, RES_STATE) };
  mxm_apply_ok(ms, bad);
  sxm_apply_ok(ss, bad);
  match bad {
    [||] => (),
    _ => {
      // An ACC staged in its execute cycle c (dispatch + d_skew) commits at dispatch + d_func.
      let la = entry_latency("ACC", T_MXM_ACC);
      let k = emod_int(c - la.d_skew + la.d_func, 128);
      pending[k] = drop_commits(pending[k], bad)
    },
  };
  mxm_stage_list = [||];
  sxm_stage_list = [||]
}

// Test hook.
function unit_state_flush_test(c : nat) -> unit = unit_state_flush(c)

// The control channel carries Sync, Notify and Config only (NOP is the timing prefix of the
// next parcel; Repeat, Ifetch, SetVL and every slice instruction are OTS-E-QUE-002). A
// control Sync parks every data queue. The control channel has no dispatch paths and no
// Repeat, so nothing holds its head parcel: it dispatches when it is complete and due. It
// has no Ifetch, so its program is its initial image.
private function control(c : nat) -> unit =
  if iq_len[144] > 0 then {
    let p = parcel_scan(144);
    if p.complete & parcel_earliest(144, p) <= c then {
      observe_site("dispatch", c, 144, iq_head[144], p.units, parcel_earliest(144, p));
      iq_text_check(144, c, p.units);
      iq_pop(144, p.units);
      iq_last[144] = c;
      iq_active[144] = c;
      match p.items {
        x :: rest => match rest {
          [||] => match x {
            Err(d) => report(144, c, d),
            Ok(i)  => match i {
              SYNC()    => barrier_sync(144, c),
              NOTIFY()  => barrier_notify(c),
              CONFIG(_) => (),
              _         => report(144, c, QUE_UNIT),
            },
          },
          _ => foreach (k from 1 to p.n) { report(144, c, QUE_BUSY) },
        },
        [||] => (),
      }
    }
  }

// ---------------------------------------------------------------------------------------
// The head parcel and its queue (design sec. 7.4 .. 7.7).

// Queue q is free at cycle t (design sec. 7.7): not parked, not busy, no Repeat, and both
// paths idle (no loop or open configuration, not busy, not in a tail). The control channel is
// always free.
private function free_at(q : range(0, 144), t : nat) -> bool =
  if q == 144 then true
  else {
    let k : queue_id = if q <= 143 then q else 0;
    view_free(hold_view_of(k, t))
  }

// Is any instruction of a parcel that breaks the slot rule held at cycle c (each needs the whole queue)?
private function strict_held(q : queue_id, c : nat, xs : list(result(instruction, diag)), held : bool) -> bool =
  match xs {
    [||] => held,
    y :: ys => strict_held(q, c, ys, held | (match y {
      Ok(i)  => hold_check(q, c, i, None(), true),
      Err(_) => hold_check(q, c, ILLEGAL32(0x00000000), None(), true),
    })),
  }

// The parcel's dispatch needs: is any instruction of the parcel (classified as r) held at cycle c?
private function parcel_held(q : queue_id, c : nat, r : cycle_recs, items : list(result(instruction, diag))) -> bool = {
  var held : bool = false;
  if r.n == 0 then ()
  else if not(r.ok) then {
    // A parcel that breaks the slot rule (or an instruction that does not decode): every
    // instruction needs the whole queue.
    held = strict_held(q, c, items, false)
  }
  else match r.single {
    Some(i) => held = hold_check(q, c, i, None(), false),
    None() => {
      match r.ifetch {
        Some(i) => if hold_check(q, c, i, (match i { IFETCHM(_) => Some(0), _ => None() }), false) then held = true,
        None() => (),
      };
      match r.cfg { Some(i) => if hold_check(q, c, i, Some(cfg_path(q)), false) then held = true, None() => () };
      foreach (k from 0 to 3) {
        match r.refs[k] {
          Some(i) => if hold_check(q, c, i, Some(if k >= 2 then 1 else 0), false) then held = true,
          None() => (),
        }
      }
    },
  };
  held
}

private function earliest_int(a : int, b : int) -> int =
  if a < 0 then b else if b < 0 then a else if a < b then a else b

// The earliest cycle t >= base at which the head parcel is not held, assuming nothing else
// changes (-1: never, it waits for another event). The hold conditions that depend on time
// only end at the busy-until, tail and release cycles, so these are the only candidates.
private function earliest_free(q : queue_id, base : nat, r : cycle_recs, items : list(result(instruction, diag)), cands : list(nat), best : int) -> int =
  match cands {
    [||] => best,
    t :: rest => earliest_free(q, base, r, items, rest, if t >= base & not(parcel_held(q, t, r, items)) then earliest_int(best, t) else best),
  }

private function parcel_dispatchable(q : queue_id, base : nat, r : cycle_recs, items : list(result(instruction, diag))) -> int = {
  var cands : list(nat) = [| base, queue_busy_until[q], path_busy_until[2 * q], path_busy_until[2 * q + 1],
                             path_tail_until[2 * q], path_tail_until[2 * q + 1] |];
  match release_at[q] { Some(x) => cands = x :: cands, None() => () };
  match release_at[144] { Some(x) => cands = x :: cands, None() => () };
  match hw_release[q] { Some(x) => cands = x :: cands, None() => () };
  earliest_free(q, base, r, items, cands, -1)
}

// After a tick at cycle c: the earliest cycle > c at which the complete head parcel of data
// queue q can dispatch (-1: none).
function parcel_event(q : queue_id, c : nat) -> int =
  if iq_len[q] == 0 then -1
  else {
    let p = parcel_scan(q);
    if not(p.complete) then -1
    else {
      let base = max(c + 1, parcel_earliest(q, p));
      parcel_dispatchable(q, base, classify_items(q, p.items), p.items)
    }
  }

function control_event(c : nat) -> int =
  if iq_len[144] == 0 then -1
  else {
    let p = parcel_scan(144);
    if p.complete then max(c + 1, parcel_earliest(144, p)) else -1
  }

// Enqueue (design sec. 7.4): the text of the fetches that are due enters the IQ at the start
// of cycle c, before anything dispatches. A unit lost to a full IQ is the diagnostic
// OTS-E-IQ-002, once per queue and cycle; the lost unit is gone and the rest of its payload
// keeps arriving. Underflow (sec. 7.7):
// when this completes the head parcel after its earliest cycle, and the queue was free in the
// cycle before, the IQ ran dry: logical time was lost (lint OTS-W-IQ-001 at the dispatch).
private function enqueue(q : range(0, 144), c : nat) -> unit =
  match fill_fifo[q] {
    seg :: _ => if seg.ready <= c then {
      let before : bool = iq_len[q] > 0 & parcel_scan(q).complete;
      let (pushed, lost) = iq_deliver(q, c);
      if lost then report(q, c, IQ_OVERFLOW);
      if pushed & not(before) & c > 0 then {
        let p = parcel_scan(q);
        if p.complete & c > parcel_earliest(q, p) & free_at(q, c - 1) then iq_dry[q] = true
      }
    },
    [||] => (),
  }

// One cycle. Order: fetched text enters the IQs, barrier retirement, commits due at c, the
// elements that execute at c (dispatched at c - d_skew), the control channel, then data
// queues 0..143; per queue: the next Repeat iteration; the head parcel (complete, due and not
// held) with the end of an AGT configuration that it does not continue; the AGT references
// due. A configuration record of an open AGT configuration is applied before the AGT step,
// so the reference of this cycle sees it. A parked queue's Repeat iterations and AGT
// references wait; the first one issues in the first unparked cycle, and later ones keep
// their spacing from it. The driver does not rely on being called every cycle.
register last_tick : nat = 0

// ---------------------------------------------------------------------------------------
// The patrol scrubber uses one SRAM operation per eligible idle patrol slot.
// A correction read is followed by writeback in a later idle slot. Any intervening
// slice mutation cancels it; software traffic takes priority and never stalls for patrol.
register scrub_next : nat = 0
register scrub_ptr : vector(88, nat) = vector_init(0)
struct scrub_write = { addr : nat, data : vec, ov : ecc_ov, revision : nat }
register scrub_pending : vector(88, option(scrub_write)) = vector_init(None())

// Is the patrol slot of a MEM queue at scrub cycle m taken by software traffic? The Read of its dispatch cycle d = m - skew:
// a path held through d (busy0, busy1 are the paths' busy-until cycles), or the queue's last issue at d.
function scrub_slot_taken(m : nat, skew : nat, busy0 : nat, busy1 : nat, last : option(nat)) -> bool =
  if m < skew then false
  else {
    let d : nat = m - skew;
    d < busy0 | d < busy1 | (match last { Some(t) => t == d, None() => false })
  }

private function scrub_issued(q : range(0, 87), m : nat) -> bool = {
  let skew = entry_latency("Read", T_MEM).d_skew;
  let last : option(nat) = match queue_last[q] { Some(_) => Some(queue_last_at[q]), None() => None() };
  scrub_slot_taken(m, skew, path_busy_until[2 * q], path_busy_until[2 * q + 1], last)
}

// What one eligible patrol slot does for one slice: nothing (the slice has no scrubber, or software holds the slot; a saved repair
// stays), a patrol read, the writeback of the saved repair, or the drop of a saved repair that a write to the slice has made stale.
enum scrub_action = { SCRUB_SKIP, SCRUB_PATROL, SCRUB_WRITEBACK, SCRUB_DROP }

function scrub_arbitrate(present : bool, taken : bool, saved : bool, fresh : bool) -> scrub_action =
  if not(present) | taken then SCRUB_SKIP
  else if not(saved) then SCRUB_PATROL
  else if fresh then SCRUB_WRITEBACK
  else SCRUB_DROP

private function scrub_slice(s : mem_slice, m : nat) -> unit = {
  let a : nat = scrub_ptr[s];
  var w : vec = read_sram_word(s, a);
  var ov : ecc_ov = read_sram_ecc(s, a);
  var fixed : bool = false;
  if ov != ecc_ov_zero then {
    foreach (g from 0 to (superlanes - 1)) {
      let e = ecc_ov_get(ov, g);
      let (k, d2, pos) = ecc_correct(slice(w, 128 * g, 128), e);
      match k {
        ECC_OK => (),
        ECC_CORRECTED => {
          w[128 * g + 127 .. 128 * g] = d2;
          ov = ecc_ov_set(ov, g, e & 0b1000000000);
          fixed = true;
          ras_scrub_event(m, s, false, a, g, pos)
        },
        ECC_UNCORRECTABLE => ras_scrub_event(m, s, true, a, g, pos),
      }
    };
    if fixed then scrub_pending[s] = Some(struct { addr = a, data = w, ov = ov, revision = sram_revision[s] })
  };
  scrub_ptr[s] = emod_int(a + 1, words_per_slice)
}

// Claim and consume a saved repair once. Any intervening write to the slice
// changes its revision, including a check-bit-only fault.
function scrub_take(s : mem_slice) -> option(scrub_write) = {
  let old = scrub_pending[s];
  scrub_pending[s] = None();
  match old {
    Some(w) => if w.revision == sram_revision[s] then Some(w) else None(),
    None() => None(),
  }
}

private function scrub_cycle(m : nat) -> unit =
  foreach (s from 0 to 87) {
    let pend = scrub_pending[s];
    let held : bool = match pend { Some(_) => true, None() => false };
    let fresh : bool = match pend { Some(w) => w.revision == sram_revision[s], None() => false };
    match scrub_arbitrate(mem_slice_present(s), scrub_issued(s, m), held, fresh) {
      SCRUB_SKIP => (),
      SCRUB_PATROL => scrub_slice(s, m),
      SCRUB_WRITEBACK => match scrub_take(s) {
        Some(w) => {
          write_sram_word(s, w.addr, w.data);
          write_sram_ecc(s, w.addr, w.ov)
        },
        None() => (),
      },
      SCRUB_DROP => { let _ = scrub_take(s); () },
    }
  }

// Run the scrub cycles m <= upto that are still due.
// The number of scrub cycles m <= upto that are due when the next one is `next` and they are `period` apart.
function scrub_due_count(next : nat, period : nat, upto : int) -> int =
  if period > 0 & next <= upto then ediv_int(upto - next, period) + 1 else 0

function scrub_run(upto : int) -> unit =
  foreach (k from 1 to scrub_due_count(scrub_next, scrub_period, upto)) {
    scrub_cycle(scrub_next);
    scrub_next = scrub_next + scrub_period
  }

// Why a due head parcel waits (the observation site name), in priority order.
function wait_reason(host : bool, parked : bool, poll : bool, repeating : bool) -> string =
  if host then "wait-host" else if parked then "wait-barrier"
  else if poll then "wait-poll" else if repeating then "wait-repeat"
  else "wait-resource"

// An open AGT configuration ends at a cycle in which the queue dispatches no configuration record: the parcel does not dispatch (go),
// or the cycle's records carry none.
function config_ends(st : agt_state, go : bool, cfg_in_cycle : bool) -> bool = st == AGT_CONFIG & not(go & cfg_in_cycle)

// A DeskewRecv looks for its vector at each epoch boundary after its dispatch, in a cycle in which the queue is not parked by a
// barrier (the arrival of a vector on the link is checked by the caller).
function poll_epoch(since : nat, c : nat, period : nat, parked : bool) -> bool = c > since & emod_int(c, period) == 0 & not(parked)

// One data queue in cycle c: the DeskewRecv poll, the next Repeat iteration, the head parcel (dispatched when complete, due and
// not held) with the end of the configuration that it does not continue, then dispatch(): the AGT references due.
private function tick_queue(q : range(0, 143), c : nat) -> unit = {
  var p : parcel = struct { complete = false, units = 0, nops = 0, n = 0, items = [||] };
  var due : bool = false;
  if iq_len[q] > 0 then {
    p = parcel_scan(q);
    due = p.complete & c >= parcel_earliest(q, p)
  };
  let recs : cycle_recs = if due then classify_items(q, p.items) else no_recs;
  let parked = queue_parked(q, c);
  // A DeskewRecv looks for its vector at each epoch boundary after its dispatch, in a cycle in which the queue is not
  // parked by a barrier; a vector that has arrived makes it a Receive dispatched in this cycle.
  match queue_poll[q] {
    Some(p) => if (if poll_epoch(p.since, c, c2c_hac_period, parked) then deskewrecv_ready(c, p.link) else false) then {
      queue_poll[q] = None();
      let rcv = C2C_RECEIVE(p.chip, p.link, p.s);
      queue_busy_until[q] = c + occupancy(rcv, c);
      let _ = run_element(q, rcv, c, 0, 0, 0)
    },
    None() => (),
  };
  match queue_repeat[q] {
    Some(r) => if r.next <= c & not(parked) then {
      observe_site("repeat", c, q, 0, 0, queue_last_at[q]);
      repeat_issue(q, r.instr, c);
      queue_last_at[q] = c;
      queue_busy_until[q] = c + 1;
      queue_repeat[q] = repeat_after(r, c)
    },
    None() => (),
  };
  // A configuration ends at the first cycle in which the queue dispatches no configuration
  // instruction (a parcel that is not due, not complete or held dispatches nothing).
  var go : bool = due & not(parcel_held(q, c, recs, p.items));
  var ended : bool = false;
  foreach (path from 0 to 1) {
    if config_ends(agt_get(q, path).state, go, config_for(recs)) then { agt_finish_config(q, path); ended = true }
  };
  if ended & due & not(go) then go = not(parcel_held(q, c, recs, p.items));   // the ended configuration no longer holds the head
  if due & not(go) then
    observe_site(wait_reason(host_parked(q, c), parked, polling(q), queue_repeat[q] != None()), c, q, iq_head[q], p.units, parcel_earliest(q, p));
  var r2 : cycle_recs = no_recs;
  if go then {
    r2 = recs;
    observe_site("dispatch", c, q, iq_head[q], p.units, parcel_earliest(q, p));
    iq_text_check(q, c, p.units);
    iq_pop(q, p.units);
    iq_last[q] = c;
    iq_active[q] = c;
    if iq_dry[q] then { iq_dry[q] = false; lint(q, c, LINT_IQ_UNDERFLOW) }
  };
  // AGT references are due after the cycle's configuration record (if any) is applied,
  // so dispatch() works them out (each holds only its own path).
  dispatch(q, c, r2, parked)
}

function tick(c : nat) -> unit = {
  last_tick = c;
  scrub_run(c - 1);
  local_fetch_complete(c);
  foreach (q from 0 to 144) { enqueue(q, c) };
  barrier_retire(c);
  commit_due(c);
  stream_fetch_complete(c);
  execute_due(c);
  control(c);
  foreach (q from 0 to 143) { tick_queue(q, c) };
  unit_state_flush(c);
  scrub_run(c)
}

// True when nothing is left to happen: no commit is pending, no fetched text is on its way,
// and no queue has Repeat iterations, AGT references or a head parcel left to dispatch,
// except queues parked by a Sync no Notify has released (they never resume).
private function stuck(k : range(0, 144)) -> bool =
  match (parked_since[k], release_at[k]) { (Some(_), None()) => true, _ => false }

function model_idle() -> bool = {
  var idle : bool = true;
  foreach (q from 0 to 87) { if local_fetches[q] != None() then idle = false };
  foreach (k from 0 to 127) { match pending[k] { [||] => (), _ => idle = false } };
  foreach (k from 0 to 15) { match execute_ring[k] { [||] => (), _ => idle = false } };
  foreach (q from 0 to 143) {
    if stream_fetches[q] != None() then idle = false;
    let held = stuck(q) | stuck(144);
    if (agts[q].state != AGT_IDLE | agts_w[q].state != AGT_IDLE | queue_repeat[q] != None()) & not(held) then idle = false;
    if parcel_event(q, last_tick) >= 0 then idle = false
  };
  if control_event(last_tick) >= 0 then idle = false;
  foreach (q from 0 to 144) { match fill_fifo[q] { [||] => (), _ => idle = false } };
  idle
}

// Reset all architectural state (sail-riscv postlude/model.sail init_model).
function reset() -> unit = {
  srf_reset();
  mxm_reset();
  sxm_reset();
  pending = vector_init([||]);
  execute_ring = vector_init([||]);
  queue_busy_until = vector_init(0);
  path_busy_until = vector_init(0);
  local_fetch_bank = vector_init(None());
  local_fetches = vector_init(None());
  local_fetch_busy = vector_init(0);
  stream_fetches = vector_init(None());
  path_tail_until = vector_init(0);
  queue_last_at = vector_init(0);
  agts = vector_init(agt_idle);
  agts_w = vector_init(agt_idle);
  agt_records = vector_init(0);
  mem_reset();
  masks_reset();
  parked_since = vector_init(None());
  release_at = vector_init(None());
  last_notify = None();
  queue_last = vector_init(None());
  queue_repeat = vector_init(None());
  queue_poll = vector_init(None());
  last_tick = 0;
  scrub_next = 0;
  scrub_ptr = vector_init(0);
  scrub_pending = vector_init(None());
  sram_revision = vector_init(0);
  c2c_reset();                                    // no link has accepted a vector
  host_reset();                                   // CSR space, doorbells, parked queues
  iq_reset()                                      // the queues, with their initial images
}
