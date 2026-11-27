// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Stream operands. A read happens at the issuer's read gap on the operand's row, in the
// execute cycle issue_cycle + d_skew (+ element). A result is not written immediately: it is
// SCHEDULED for commit at issue_cycle + d_func (+ element) and becomes part of the SRF
// when the cycle driver reaches that cycle (postlude/tick.sail). That is what makes a
// read in cycle t see only commits from cycles before t, whatever the queue order.

// Byte planes of one multi-byte operand: plane b holds byte b of every lane's element.
type planes = vector(4, vec)

let zero_vec : vec = sail_zeros(8 * vlanes)
let no_planes : planes = [zero_vec, zero_vec, zero_vec, zero_vec]

// A single-plane operand (i8 elements, or one stream of raw bytes).
function one_plane(v : vec) -> planes = [no_planes with 0 = v]

// Stream b of a multi-stream operand starting at s (callers check aligned() first).
private function plane_stream(s : streamref, b : int) -> range(0, 31) = {
  let i = stream_id(s) + b;
  assert(0 <= i & i <= 31, "plane_stream: operand not checked by aligned()");
  i
}

// Width check shared by reads and writes (OTSP-CON-ALIGN; destinations too).
function aligned(s : streamref, w : {1, 2, 4}) -> bool = {
  let id = stream_id(s);
  id % w == 0 & id + w <= 32
}

// Read w consecutive streams starting at s, in cycle t, at the issuer's read gap. chain: a
// chained VXM operand (srf_read_chain: the VXM result committed at the issuer's own gap in
// cycle t - 1); otherwise the flow rule (srf_read). chk: check the ECC of each stream value
// (core/ecc.sail): a single-bit error is corrected in the reader's copy, a multi-bit error
// marks its superlane in ecc_bad. An unchecked read (the MEM Write data and the Gather and
// Scatter map) gives the check bits as stored, for the caller to store or ignore.
function read_planes_full(s : streamref, w : {1, 2, 4}, t : int, chain : bool, chk : bool) -> result((planes, vector(4, ecc_ov)), diag) = {
  let r = stream_row(s);
  match read_gap(issue_unit.unit, issue_unit.slice, r) {
    None() => Err(DIR_UPSTREAM),
    Some(g) =>
      if not(aligned(s, w)) then Err(OPR_ALIGN) else {
        var out : planes = no_planes;
        var ovs : vector(4, ecc_ov) = vector_init(ecc_ov_zero);
        var err : option(diag) = None();
        foreach (b from 0 to (w - 1)) {
          let found = if chain then srf_read_chain_ov(r, plane_stream(s, b), g, t) else srf_read_ov(r, plane_stream(s, b), g, t);
          match found {
            (SRF_FOUND, v, ov) => {
              if chk then {
                let (v2, bad) = ecc_check_value(t, issue_queue, 0, r == RowW, plane_stream(s, b), g, v, ov);
                out[b] = v2;
                ecc_bad = ecc_bad | bad
              } else { out[b] = v; ovs[b] = ov }
            },
            (SRF_UPSTREAM, _, _) => err = Some(DIR_UPSTREAM),
            (SRF_NONE, _, _)     => if err == None() then err = Some(TIM_NO_VALUE),
          }
        };
        match err { Some(d) => Err(d), None() => Ok((out, ovs)) }
      }
  }
}

function read_planes_core(s : streamref, w : {1, 2, 4}, t : int, chain : bool) -> result(planes, diag) =
  match read_planes_full(s, w, t, chain, true) { Ok((p, _)) => Ok(p), Err(d) => Err(d) }

function read_planes(s : streamref, w : {1, 2, 4}, t : int) -> result(planes, diag) = read_planes_core(s, w, t, false)

// The unchecked read of one stream: the value and its check bits as they flow.
function read_plane_raw(s : streamref, t : int) -> result((vec, ecc_ov), diag) =
  match read_planes_full(s, 1, t, false, false) { Ok((p, ovs)) => Ok((p[0], ovs[0])), Err(d) => Err(d) }

// One scheduled stream write.
struct commit = {
  r     : row,
  id    : range(0, 31),
  g     : gap,
  value : vec,
  ov    : ecc_ov,          // the ECC overlay of the value (core/ecc.sail)
  q     : range(0, 144),   // writer queue
  b     : range(0, 3),     // byte plane of a multi-stream result
}

// Pending commits, indexed by commit cycle modulo the ring size. tick applies the slot of
// cycle c (commit_due) before it dispatches the records of c, so a commit must be scheduled
// for a cycle after its dispatch and fewer than 128 cycles ahead. Nothing here checks it at
// schedule time; the configuration does: config/schema.json bounds d_func to 1..63 for
// every stream-writing class (tools/schema_invariants.py D_FUNC_MIN_CLASSES, ). An
// AGT element k runs in cycle t0 + k and commits at t0 + k + d_func (mem_insts.sail
// commit_cycle), so every commit is 1..63 cycles after the cycle that schedules it.
register pending : vector(128, list(commit)) = vector_init([||])

// Schedule w planes for commit at cycle c, at the issuer's write gap on s's row. The producer
// generates the check bits (core/ecc.sail): valid ones, except in the superlanes poisoned by an
// uncorrectable operand of this instruction. fwd: the check bits of plane 0 forwarded as they
// are (the MEM Read forwards the stored ones, the published rule), for the active tiles.
function write_planes_fwd(s : streamref, p : planes, w : {1, 2, 4}, c : nat, fwd : option(ecc_ov)) -> result(unit, diag) = {
  let r = stream_row(s);
  match write_gap(issue_unit.unit, issue_unit.slice, r) {
    None() => Err(DIR_UPSTREAM),
    Some(g) =>
      if not(aligned(s, w)) then Err(OPR_ALIGN) else {
        let k = c % 128;
        // Inactive superlanes (SetVL; on a MEM queue also the other liquid thread's tiles)
        // produce zero (core/sram.sail).
        let q = issue_queue;
        let vl : range(1, superlanes) = if q <= 143 then queue_vl[q] else superlanes;
        let split : range(0, superlanes) = if q <= 87 then mem_thread_split[q] else superlanes;
        foreach (b from 0 to (w - 1)) {
          var ov : ecc_ov = ecc_result_ov();
          match fwd {
            Some(f) => if b == 0 then {
              ov = ecc_ov_zero;
              foreach (g2 from 0 to (superlanes - 1)) {
                if tile_active(vl, split, q <= 87, issue_thread, g2) then ov = ecc_ov_set(ov, g2, ecc_ov_get(f, g2))
              }
            },
            None() => (),
          };
          pending[k] = struct { r = r, id = plane_stream(s, b), g = g, value = merge_tiles(zero_vec, p[b], vl, split, q <= 87, issue_thread), ov = ov, q = q, b = b } :: pending[k]
        };
        Ok(())
      }
  }
}

function write_planes(s : streamref, p : planes, w : {1, 2, 4}, c : nat) -> result(unit, diag) = write_planes_fwd(s, p, w, c, None())

// A 16-stream operand group (MXM LW, SXM Transpose): streams base..base+15, base 0 or 16.
type group16 = vector(16, vec)

function read_planes16(s : streamref, t : int) -> result(group16, diag) = {
  let r = stream_row(s);
  match read_gap(issue_unit.unit, issue_unit.slice, r) {
    None() => Err(DIR_UPSTREAM),
    Some(g) =>
      if stream_id(s) % 16 != 0 then Err(OPR_ALIGN) else {
        var out : group16 = vector_init(zero_vec);
        var err : option(diag) = None();
        ecc_g16 = vector_init(false);
        foreach (b from 0 to 15) {
          match srf_read_ov(r, plane_stream(s, b), g, t) {
            (SRF_FOUND, v, ov) => {
              let (v2, bad) = ecc_check_value(t, issue_queue, 0, r == RowW, plane_stream(s, b), g, v, ov);
              out[b] = v2;
              ecc_g16[b] = bad != sail_zeros(superlanes);
              ecc_bad = ecc_bad | bad
            },
            (SRF_UPSTREAM, _, _) => err = Some(DIR_UPSTREAM),
            (SRF_NONE, _, _)     => if err == None() then err = Some(TIM_NO_VALUE),
          }
        };
        match err { Some(d) => Err(d), None() => Ok(out) }
      }
  }
}

// Byte b of lane l of a per-lane 32-bit result goes to plane b (little-endian elements).
// A lane beyond the end of xs (a product has mxm.rows lanes, fewer than the vector when mxm.rows < 16 * superlanes) is zero.
val words_to_planes : forall 'n, 'n >= 0. vector('n, bits(32)) -> planes
function words_to_planes(xs) = {
  var out : planes = no_planes;
  foreach (b from 0 to 3) {
    var v : vec = zero_vec;
    foreach (l from 0 to (vlanes - 1)) { if l < length(xs) then v[8 * l + 7 .. 8 * l] = xs[l][8 * b + 7 .. 8 * b] };
    out[b] = v
  };
  out
}
