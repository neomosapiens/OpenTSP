// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Accounting obligations on two instruction-queue transitions of slices/cu/cu_iq.sail . What they are: the real
// `iq_pop` and `iq_push_tagged` run on a queue state, with five counters of units (below) that the obligation itself updates by what the
// transition is meant to do, and the invariant of test/formal/frontend_induction.smt2 (the five counters add up to the units offered) is
// required to hold again. Only `live` (iq_len) is model state: the other four counters are free inputs that cancel, so each obligation is
// the length part of an existing one (p_iq_pop_accounting: the length of p_iq_pop_conservation; p_iq_push_accounting: the length of
// p_iq_append). What they are not: a refinement of the delivery path. The host text FIFO, the padding rule, the first-beat latch, the
// capture of a payload and the service loop of `iq_deliver` are not run (they iterate, and the obligations cannot unroll a loop of
// unbounded length); that refinement is open.
//   held     units retained in the first-beat latch      staged   units queued in the host text FIFO
//   live     iq_len[q]                                    retired  units that left through a dispatch
//   lost     units rejected, padding, overflow            offered  the sum of all five (a ghost counter)
// Every unit offered is in exactly one owner. `accounted` is the invariant of the induction file; each obligation takes any state
// satisfying it, runs one real transition and requires the invariant to hold again with the counters the transition implies.

// The proof environment: the storage callbacks record their last call in registers (as in the model's own obligations).
register proof_slot : int
register proof_data : bits(32)
register proof_tag_slot : int
register proof_tag : int
function iq_unit_write(_, slot, u) = { proof_slot = slot; proof_data = u }
function iq_tag_write(_, slot, tag) = { proof_tag_slot = slot; proof_tag = tag }

function accounted(h : int, s : int, l : int, r : int, d : int, n : int) -> bool =
  h + s + l + r + d == n & 0 <= l & l <= iq_cap & h >= 0 & s >= 0 & r >= 0 & d >= 0

// dispatch-or-hold: a complete head parcel of `pop` units leaves once (iq_pop); the units move from live to retired.
$property
function p_iq_pop_accounting(h : bits(10), len : bits(11), pop : bits(11), held : bits(12), staged : bits(12), retired : bits(12), lost : bits(12)) -> bool = {
  let head = unsigned(h);
  let live = unsigned(len);
  let n = unsigned(pop);
  if head >= iq_cap | live > iq_cap | n > live then true else {
    iq_head[0] = head;
    iq_len[0] = live;
    let offered = unsigned(held) + unsigned(staged) + live + unsigned(retired) + unsigned(lost);
    iq_pop(0, n);
    accounted(unsigned(held), unsigned(staged), iq_len[0], unsigned(retired) + n, unsigned(lost), offered)
  }
}

// deliver-padding-overflow, one text unit: the unit leaves the host text FIFO (staged) and enters the queue (iq_push_tagged), or is lost
// when the queue is full.
$property
function p_iq_push_accounting(h : bits(10), len : bits(11), held : bits(12), staged : bits(12), retired : bits(12), lost : bits(12), payload : bits(32), source : bits(40)) -> bool = {
  let head = unsigned(h);
  let live = unsigned(len);
  let st = unsigned(staged);
  if head >= iq_cap | live > iq_cap | st < 1 then true else {
    iq_head[0] = head;
    iq_len[0] = live;
    let offered = unsigned(held) + st + live + unsigned(retired) + unsigned(lost);
    let ok = iq_push_tagged(0, payload, unsigned(source));
    if ok then accounted(unsigned(held), st - 1, iq_len[0], unsigned(retired), unsigned(lost), offered)
    else accounted(unsigned(held), st - 1, iq_len[0], unsigned(retired), unsigned(lost) + 1, offered)
  }
}

// Reachability twins of the two obligations above (a property that returns true on its guard proves nothing for the inputs the
// guard accepts, and the generic twin of prove_extra.sh is satisfiable through the guard): each is satisfiable exactly when some
// state outside the guard completes the transition with the invariant holding afterwards.
$counterexample
function t_iq_pop_accounting(h : bits(10), len : bits(11), pop : bits(11), held : bits(12), staged : bits(12), retired : bits(12), lost : bits(12)) -> bool = {
  let head = unsigned(h);
  let live = unsigned(len);
  let n = unsigned(pop);
  if head >= iq_cap | live > iq_cap | n > live then true else {
    iq_head[0] = head;
    iq_len[0] = live;
    let offered = unsigned(held) + unsigned(staged) + live + unsigned(retired) + unsigned(lost);
    iq_pop(0, n);
    not(accounted(unsigned(held), unsigned(staged), iq_len[0], unsigned(retired) + n, unsigned(lost), offered))
  }
}

$counterexample
function t_iq_push_accounting(h : bits(10), len : bits(11), held : bits(12), staged : bits(12), retired : bits(12), lost : bits(12), payload : bits(32), source : bits(40)) -> bool = {
  let head = unsigned(h);
  let live = unsigned(len);
  let st = unsigned(staged);
  if head >= iq_cap | live > iq_cap | st < 1 then true else {
    iq_head[0] = head;
    iq_len[0] = live;
    let offered = unsigned(held) + st + live + unsigned(retired) + unsigned(lost);
    let ok = iq_push_tagged(0, payload, unsigned(source));
    if ok then not(accounted(unsigned(held), st - 1, iq_len[0], unsigned(retired), unsigned(lost), offered))
    else not(accounted(unsigned(held), st - 1, iq_len[0], unsigned(retired), unsigned(lost) + 1, offered))
  }
}

// Negative control: a queue that cannot take a unit would have to count it as live: the claim "a delivery always becomes live" fails.
$counterexample
function c_deliver_always_live(len : bits(11), payload : bits(32)) -> bool = {
  let live = unsigned(len);
  if live > iq_cap then false else {
    iq_len[0] = live;
    let ok = iq_push_tagged(0, payload, 0);
    ok
  }
}
