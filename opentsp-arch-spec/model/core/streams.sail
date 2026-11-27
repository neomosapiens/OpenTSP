// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// The stream register file: two rows (E, W) x 32 streams x 45 registers.
//
// Flow rule (microarchitecture.md sec. 2): a value written at gap g with commit cycle c is in
// register g + 1 at cycle c + 1 and moves one register per cycle in its row's flow
// direction; it falls off after register 44. A reader at gap G in cycle t sees
// register G. Hence the value is visible at G exactly when G > g and t - c == G - g.
//
// Representation: d = g - c (the "diagonal") is constant while a value flows. Each
// (row, stream) keeps, per diagonal slot d mod 64, the writes on that diagonal, at most
// one per gap. A diagonal is live for at most 44 cycles (a read happens in its execute
// cycle), well below 64 even with a d_skew_max margin, so slots never alias (an arithmetic
// lemma: props.sail p_srf_live_window). Reading picks the writer with the largest gap below
// the reader (the nearest upstream producer replaces older values from its gap on).

private type slot_index = range(0, 63)

struct srf_write = {
  valid : bool,
  diag  : int,      // gap - commit cycle
  wgap  : range(0, 43),
  value : vec,
  ov    : ecc_ov,   // the ECC overlay of the stored word (core/ecc.sail): zero when the check bits are valid
}

private let empty_write : srf_write = struct { valid = false, diag = 0, wgap = 0, value = sail_zeros(8 * vlanes), ov = ecc_ov_zero }

// At most one write per gap can share a diagonal; 44 entries per slot suffice.
type srf_slot = vector(44, srf_write)
type srf_stream = vector(64, srf_slot)

register srf_e : vector(32, srf_stream)
register srf_w : vector(32, srf_stream)

private function slot_of(d : int) -> slot_index = emod_int(d, 64)

// The slot operations, pure functions of one slot (the writes of one diagonal class of one stream): the stream
// register file operations below only choose the slot and store the result.

// Does a write at gap g on diagonal d meet a write already there (the same gap and diagonal is the same cycle)?
function srf_slot_collides(slot : srf_slot, g : gap, d : int) -> bool = slot[g].valid & slot[g].diag == d

// The slot after a write at gap g on diagonal d (it replaces whatever the gap held).
function srf_slot_write(slot : srf_slot, g : gap, d : int, v : vec, ov : ecc_ov) -> srf_slot = {
  var o = slot;
  o[g] = struct { valid = true, diag = d, wgap = g, value = v, ov = ov };
  o
}

// The nearest upstream write for a reader at gap g on diagonal d: the largest gap below g that holds a write of that
// diagonal, or -1 when there is none.
function srf_slot_best(slot : srf_slot, g : gap, d : int) -> int = {
  var best : int = -1;
  foreach (w from 0 to 43) {
    let e = slot[w];
    if e.valid & e.diag == d & w < g & w > best then best = w
  };
  best
}

// What a reader at gap g on diagonal d finds in the slot: whether there is a value, the value, and its ECC overlay.
function srf_slot_read_ov(slot : srf_slot, g : gap, d : int) -> (bool, vec, ecc_ov) = {
  let best = srf_slot_best(slot, g, d);
  if best >= 0 then { let e = slot[emod_int(best, 44)]; (true, e.value, e.ov) }
  else (false, sail_zeros(8 * vlanes), ecc_ov_zero)
}

function srf_slot_read(slot : srf_slot, g : gap, d : int) -> (bool, vec) = {
  let (found, v, _) = srf_slot_read_ov(slot, g, d);
  (found, v)
}

// A chained read: only the write at the reader's own gap g on diagonal d.
function srf_slot_chain_ov(slot : srf_slot, g : gap, d : int) -> (bool, vec, ecc_ov) = {
  let e = slot[g];
  if e.valid & e.diag == d then (true, e.value, e.ov) else (false, sail_zeros(8 * vlanes), ecc_ov_zero)
}

// The slot after bit n (0..127 data, 128..136 check bits, 137 spare) of superlane sl of the value a reader at gap gp on
// diagonal d sees is flipped; unchanged when the reader sees no value or sl is not a superlane.
function srf_slot_flip(slot : srf_slot, gp : gap, d : int, sl : range(0, 31), n : range(0, 137)) -> srf_slot = {
  let best = srf_slot_best(slot, gp, d);
  var o = slot;
  if best >= 0 & sl < superlanes then {
    let w : range(0, 43) = emod_int(best, 44);
    var e = o[w];
    if n < 128 then e.value = e.value ^ sail_shiftleft(get_slice_int(8 * vlanes, 1, 0), 128 * sl + n);
    e.ov = ecc_ov_set(e.ov, sl, ecc_ov_get(e.ov, sl) ^ ecc_flip_delta(n));
    o[w] = e
  };
  o
}

// Error reports in the spare bits (microarchitecture.md sec. 15.5). A report is a write of the spare bit (bit 137) of
// streams 0..28 of a row at one register of the row (the write gap of the reporting slice) in one cycle. It is kept
// beside the 138-bit register words, on the same diagonal: a reader at a gap above the report that finds a value on
// the diagonal sees the report's spare bits with the value, unless a data write at a gap between them replaced the
// register. sp[s] has bit x set for superlane x of stream s; pr[x] is the priority of the error that wrote the
// auxiliary streams 4..28 of superlane x (Gather 3, Scatter 2, bank conflict 1).
struct srf_rep = { used : bool, rw : bool, diag : int, wgap : range(0, 43), sp : vector(30, bits(superlanes)), pr : vector(superlanes, int) }

private let empty_rep : srf_rep = struct { used = false, rw = false, diag = 0, wgap = 0, sp = vector_init(sail_zeros(superlanes)), pr = vector_init(0) }

register srf_reps : vector(256, srf_rep)
register srf_rep_n : nat = 0

function srf_reset() -> unit = {
  srf_reps = vector_init(empty_rep);
  srf_rep_n = 0;
  let empty_slot : srf_slot = vector_init(empty_write);
  let empty_stream : srf_stream = vector_init(empty_slot);
  srf_e = vector_init(empty_stream);
  srf_w = vector_init(empty_stream)
}

private function get_stream(r : row, s : range(0, 31)) -> srf_stream =
  match r { RowE => srf_e[s], RowW => srf_w[s] }

private function set_stream(r : row, s : range(0, 31), v : srf_stream) -> unit =
  match r { RowE => srf_e[s] = v, RowW => srf_w[s] = v }

// Record a write. Returns true on a same-cycle collision (OTS-E-SRF-002): another write
// on the same row, stream and gap with the same diagonal (hence the same cycle).
function srf_write_ov(r : row, s : range(0, 31), g : gap, c : cycle, v : vec, ov : ecc_ov) -> bool = {
  let d = g - c;
  var stream = get_stream(r, s);
  let slot = stream[slot_of(d)];
  let collide = srf_slot_collides(slot, g, d);
  stream[slot_of(d)] = srf_slot_write(slot, g, d, v, ov);
  set_stream(r, s, stream);
  srf_write_callback(c, r == RowW, s, g, v);
  collide
}

// A write whose check bits are valid (generated by the producer).
function srf_write(r : row, s : range(0, 31), g : gap, c : cycle, v : vec) -> bool = srf_write_ov(r, s, g, c, v, ecc_ov_zero)

enum srf_lookup = {SRF_FOUND, SRF_NONE, SRF_UPSTREAM}

private function rep_sp_set(sp : vector(30, bits(superlanes)), s : int, x : int) -> vector(30, bits(superlanes)) = {
  var o = sp;
  if 0 <= s & s < 30 & 0 <= x & x < superlanes then { var m = o[s]; m[x] = bitone; o[s] = m };
  o
}

// Merge a report of priority prio for the superlanes in xs into the register's report e: the error types S[0..2] OR together, the
// auxiliary streams 4..28 follow the higher priority (a lower one adds nothing, an equal one ORs), and the parity S[3] is the even
// parity of the merged S[0..2]. A pure function of the report and the new one.
function srf_rep_merge(e0 : srf_rep, prio : int, xs : bits(superlanes), sp : vector(30, bits(superlanes))) -> srf_rep = {
  var e = e0;
  foreach (x from 0 to (superlanes - 1)) {
    if xs[x] == bitone then {
      let keep : bool = e.pr[x] > prio;
      let take : bool = e.pr[x] < prio;
      foreach (s from 0 to 29) {
        var m = e.sp[s];
        if s < 3 then { if sp[s][x] == bitone then m[x] = bitone }
        else if s >= 4 then {
          if take then m[x] = sp[s][x]
          else if not(keep) then { if sp[s][x] == bitone then m[x] = bitone }
        };
        e.sp[s] = m
      };
      var p3 = e.sp[3];
      let n3 : int = (if e.sp[0][x] == bitone then 1 else 0) + (if e.sp[1][x] == bitone then 1 else 0) + (if e.sp[2][x] == bitone then 1 else 0);
      p3[x] = if emod_int(n3, 2) == 1 then bitone else bitzero;
      e.sp[3] = p3;
      if prio > e.pr[x] then e.pr[x] = prio
    }
  };
  e
}

// Record a report written in cycle c at gap g of row r for the superlanes in xs: `kind` has S[0..2] (the error types) and `aux`
// the auxiliary streams 4..28, both per superlane (the vectors are indexed by stream). The parity S[3] is computed here
// from the merged S[0..2] (the even parity of the other error bits).
function srf_report(r : row, g : gap, c : cycle, prio : int, xs : bits(superlanes), sp : vector(30, bits(superlanes))) -> bool = {
  let d = g - c;
  let rw = r == RowW;
  var at : int = -1;
  foreach (i from 0 to 255) {
    let e = srf_reps[i];
    if e.used & e.rw == rw & e.diag == d & e.wgap == g then at = i
  };
  let e0 : srf_rep = if at >= 0 then srf_reps[emod_int(at, 256)] else struct { used = true, rw = rw, diag = d, wgap = g, sp = vector_init(sail_zeros(superlanes)), pr = vector_init(0) };
  let e = srf_rep_merge(e0, prio, xs, sp);
  // The array keeps the 256 latest new reports (microarchitecture.md sec. 15.5, "Report retention", ): a new register's report takes the
  // slot of the oldest one. The result is true when that report was still in its row at cycle c (register d + c <= 44): a report that a reader
  // could still have found has been evicted (the lint OTS-W-RAS-001, raised by the caller; no architectural effect).
  if at >= 0 then { srf_reps[emod_int(at, 256)] = e; false }
  else {
    let slot = emod_int(srf_rep_n, 256);
    let old = srf_reps[slot];
    srf_reps[slot] = e;
    srf_rep_n = srf_rep_n + 1;
    old.used & old.diag + c <= 44
  }
}

// True when stream s on row r has at least one LIVE write at cycle t and all live writes
// are at gap >= g (none upstream of a reader at g). A write on diagonal d is live while it
// is still in the row: its register d + t is at most 44 (it falls off after register 44).
// Writes that have left the row do not count either way.
function srf_stream_only_downstream(stream : srf_stream, g : gap, t : int) -> bool = {
  var any : bool = false;
  var upstream : bool = false;
  foreach (k from 0 to 63) {
    foreach (w from 0 to 43) {
      let e = stream[k][w];
      if e.valid & e.diag + t <= 44 then { any = true; if w < g then upstream = true }
    }
  };
  any & not(upstream)
}

private function srf_only_downstream(r : row, s : range(0, 31), g : gap, t : int) -> bool = srf_stream_only_downstream(get_stream(r, s), g, t)

// The spare bits a reader at gap g in cycle t of row r sees on stream s: the reports on its diagonal written at a gap above... below
// the reader, unless a data write on the diagonal at a gap between the report's and the reader's replaced the register.
function srf_spare(r : row, s : range(0, 31), g : gap, t : int) -> bits(superlanes) = {
  var m : bits(superlanes) = sail_zeros(superlanes);
  if srf_rep_n > 0 & s < 30 then {
    let d = g - t;
    let rw = r == RowW;
    let slot = get_stream(r, s)[slot_of(d)];
    foreach (i from 0 to 255) {
      let e = srf_reps[i];
      if e.used & e.rw == rw & e.diag == d & e.wgap < g then {
        var replaced : bool = false;
        foreach (w from 0 to 43) { if w > e.wgap & w < g & slot[w].valid & slot[w].diag == d then replaced = true };
        if not(replaced) then m = m | e.sp[s]
      }
    }
  };
  m
}

// S[0..3] (bit i = stream i) of superlane x as a reader at gap g of row r sees them in cycle t.
function srf_s03(rw : bool, g : int, t : int, x : int) -> bits(4) = {
  var o : bits(4) = 0b0000;
  if 0 <= g & g < 44 & 0 <= x & x < superlanes then {
    foreach (s from 0 to 3) {
      let m = srf_spare(if rw then RowW else RowE, s, g, t);
      if m[x] == bitone then o[s] = bitone
    }
  };
  o
}

private function srf_ov_spare(ov : ecc_ov, m : bits(superlanes)) -> ecc_ov = {
  var o = ov;
  if m != sail_zeros(superlanes) then {
    foreach (x from 0 to (superlanes - 1)) {
      if m[x] == bitone then o = ecc_ov_set(o, x, ecc_ov_get(o, x) | 0b1000000000)
    }
  };
  o
}

// The value seen by a reader at gap g in cycle t, or why there is none. SRF_UPSTREAM (a
// direction error, OTSP-CON-UPSTREAM) when the stream has live writes on this row but every
// one of them is at or downstream of the reader, so no value of it can ever reach the
// reader; SRF_NONE (a timing error) otherwise. This is the flow rule: every read, a VXM
// read included, takes the nearest upstream write.
function srf_read_ov(r : row, s : range(0, 31), g : gap, t : int) -> (srf_lookup, vec, ecc_ov) = {
  let d = g - t;
  let slot = get_stream(r, s)[slot_of(d)];
  let (found, value, ov) = srf_slot_read_ov(slot, g, d);
  if found then (SRF_FOUND, value, srf_ov_spare(ov, srf_spare(r, s, g, t)))
  else if srf_only_downstream(r, s, g, t) then (SRF_UPSTREAM, sail_zeros(8 * vlanes), ecc_ov_zero)
  else (SRF_NONE, sail_zeros(8 * vlanes), ecc_ov_zero)
}

function srf_read(r : row, s : range(0, 31), g : gap, t : int) -> (srf_lookup, vec) = {
  let (k, v, _) = srf_read_ov(r, s, g, t);
  (k, v)
}

// A CHAINED read (VXM ALU chaining, microarchitecture.md sec. 5): the value written AT the reader's own gap g in cycle t - 1, and nothing
// else. No such write is SRF_NONE (a timing error). Only a VXM operand whose chain bit is
// set reads this way; the flow rule (srf_read) never sees a write at the reader's own gap.
function srf_read_chain_ov(r : row, s : range(0, 31), g : gap, t : int) -> (srf_lookup, vec, ecc_ov) = {
  let (found, value, ov) = srf_slot_chain_ov(get_stream(r, s)[slot_of(g - t + 1)], g, g - t + 1);
  if found then (SRF_FOUND, value, ov) else (SRF_NONE, sail_zeros(8 * vlanes), ecc_ov_zero)
}

function srf_read_chain(r : row, s : range(0, 31), g : gap, t : int) -> (srf_lookup, vec) = {
  let (k, v, _) = srf_read_chain_ov(r, s, g, t);
  (k, v)
}

// ECC fault injection (core/ecc.sail): flip bit n (0..127 data, 128..136 check bits, 137 spare)
// of superlane sl of the value that is in register `gap` of row r, stream s at cycle c: the
// nearest upstream write of the diagonal gap - c (the one a reader at `gap` sees). A register
// that holds no value has nothing to flip. The flip stays on the diagonal: readers downstream
// that see the same value see it flipped.
function srf_flip(c : cycle, r : row, s : range(0, 31), gp : gap, sl : range(0, 31), n : range(0, 137)) -> unit = {
  let d = gp - c;
  var stream = get_stream(r, s);
  stream[slot_of(d)] = srf_slot_flip(stream[slot_of(d)], gp, d, sl, n);
  set_stream(r, s, stream)
}
