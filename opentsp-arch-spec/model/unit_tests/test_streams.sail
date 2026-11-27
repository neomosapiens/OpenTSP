// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the stream register file (microarchitecture.md sec. 2).

private function v(b : bits(8)) -> vec = replicate_bits(b, vlanes)

// A value written at gap 18, cycle 15 is visible at gap 21 exactly at cycle 18.
$[test]
function test_srf_visible_once() -> unit = {
  srf_reset();
  let _ = srf_write(RowE, 3, 18, 15, v(0x2a));
  let (k1, x) = srf_read(RowE, 3, 21, 18);
  assert(k1 == SRF_FOUND & x == v(0x2a));
  let (k2, _) = srf_read(RowE, 3, 21, 17);
  assert(k2 == SRF_NONE);
  let (k3, _) = srf_read(RowE, 3, 21, 19);
  assert(k3 == SRF_NONE);
}

// The nearest upstream writer on the diagonal wins; readers between the two see the older value.
$[test]
function test_srf_overwrite() -> unit = {
  srf_reset();
  let _ = srf_write(RowE, 2, 8, 15, v(0x01));    // diagonal -7
  let _ = srf_write(RowE, 2, 13, 20, v(0x02));   // diagonal -7, downstream writer
  let (k1, a) = srf_read(RowE, 2, 11, 18);        // between: sees the first
  assert(k1 == SRF_FOUND & a == v(0x01));
  let (k2, b) = srf_read(RowE, 2, 18, 25);        // after: sees the second
  assert(k2 == SRF_FOUND & b == v(0x02));
}

// Two writes on one row, stream and gap in one cycle collide.
$[test]
function test_srf_collision() -> unit = {
  srf_reset();
  assert(not(srf_write(RowW, 0, 18, 10, v(0x01))));
  assert(srf_write(RowW, 0, 18, 10, v(0x02)));
}

// A reader upstream of every writer gets a direction diagnosis, not "no value".
$[test]
function test_srf_upstream() -> unit = {
  srf_reset();
  let _ = srf_write(RowE, 8, 21, 25, v(0x05));   // VXM writes E at gap 21
  let (k, _) = srf_read(RowE, 8, 18, 22);         // west MEM quad 0 reads E at gap 18
  assert(k == SRF_UPSTREAM);
}

// Rows are independent.
$[test]
function test_srf_rows_independent() -> unit = {
  srf_reset();
  let _ = srf_write(RowE, 1, 18, 10, v(0x11));
  let (k, _) = srf_read(RowW, 1, 21, 13);
  assert(k == SRF_NONE);
}

// DIR-001 versus TIM-001 looks only at live writes (still in the row at the read
// cycle). A write that fell off long ago does not make a reader "upstream", and an old
// upstream write does not hide a direction error.
$[test]
function test_srf_upstream_live_only() -> unit = {
  srf_reset();
  let _ = srf_write(RowE, 8, 21, 25, v(0x05));
  let (k1, _) = srf_read(RowE, 8, 18, 200);       // the write left the row long ago
  assert(k1 == SRF_NONE);
  srf_reset();
  let _ = srf_write(RowE, 9, 5, 10, v(0x01));      // stale upstream write
  let _ = srf_write(RowE, 9, 21, 100, v(0x02));    // live downstream write
  let (k2, _) = srf_read(RowE, 9, 18, 102);
  assert(k2 == SRF_UPSTREAM)
}

// R13a (microarchitecture.md sec. 11): the architectural state after reset().
$[test]
function test_reset_state() -> unit = {
  reset();
  // Stream register file empty: any read finds no value.
  let (k, _) = srf_read(RowE, 0, 21, 5);
  assert(k == SRF_NONE);
  foreach (q from 0 to 143) {
    assert(queue_vl[q] == superlanes);              // SetVL: every superlane
    assert(queue_busy_until[q] == 0);               // no queue busy
    assert(agts[q].state == AGT_IDLE);              // no AGT configuration or loop
    assert(queue_last[q] == None());                // nothing to Repeat
    assert(queue_repeat[q] == None())
  };
  foreach (s from 0 to 87) {
    assert(mem_thread_split[s] == superlanes);      // every tile thread 0
    assert(mem_segment[s] == words_per_slice)       // segment: the whole slice
  };
  foreach (k from 0 to 144) {
    assert(parked_since[k] == None() & release_at[k] == None())   // no barrier
  };
  assert(last_notify == None());
  foreach (k from 0 to 127) { assert(match pending[k] { [||] => true, _ => false }) };
  // SXM maps are the identity; MXM planes empty, both accumulators zero.
  foreach (x from 0 to 1) { foreach (m from 0 to 15) { foreach (l from 0 to (vlanes - 1)) {
    assert(unsigned(sxm_maps[x][m][l]) == l)
  } } };
  foreach (x from 0 to 1) { foreach (m from 0 to 15) {
    assert(sxm_map_bij[x][m] & sxm_map_ready[x][m] == 0)       // reset maps: bijective and ready
  } };
  foreach (p from 0 to (2 * mxm_planes - 1)) {
    assert(mxm[p].buf_rows == 0 & mxm[p].rows == 0 & not(mxm[p].iw_open));
    assert(match mxm[p].abc { [||] => true, _ => false });
    foreach (a from 0 to (mxm_acc_depth - 1)) { foreach (l from 0 to (mxm_rows - 1)) { assert(mxm[p].acc[a][l] == 0x00000000) } }
  };
  assert(model_idle())
}

// The SEL and TSP outputs (HC34 p.31, ISCA 2020 Fig. 4) of one SXM hemisphere meet
// in an OR before the turn-out. A SEL write and a TSP write to one stream in one commit
// cycle store the bitwise OR (in either record order); a shifter with a select, or two
// transposers, still collide (the lowest writer is kept).
private function or_case(qa : range(0, 144), va : vec, qb : range(0, 144), vb : vec) -> vec = {
  reset();
  let g : gap = match write_gap(U_SXM_W_SEL, 0, RowE) { Some(x) => x, None() => 0 };
  let c : nat = 50;
  let k = c % 128;
  pending[k] = [| struct { r = RowE, id = 7, g = g, value = va, ov = ecc_ov_zero, q = qa, b = 0 },
                  struct { r = RowE, id = 7, g = g, value = vb, ov = ecc_ov_zero, q = qb, b = 0 } |];
  tick(c);
  let g1 : gap = if g < 43 then g + 1 else g;     // the turn-out write is in register 5 next cycle
  let (f, x) = srf_read(RowE, 7, g1, c + 1);
  assert(f == SRF_FOUND);
  x
}

$[test]
function test_sxm_or_merge() -> unit = {
  assert(or_case(125, v(0x0c), 120, v(0x30)) == v(0x3c));   // SEL + TSP: OR
  assert(or_case(120, v(0x30), 125, v(0x0c)) == v(0x3c));   // record order does not matter
  assert(or_case(121, v(0x01), 125, v(0x80)) == v(0x81));   // the other transposer
  assert(or_case(122, v(0x0c), 125, v(0x30)) == v(0x0c));   // SHF + SEL: collision, 122 kept
  assert(or_case(124, v(0x0c), 125, v(0x30)) == v(0x0c));   // PRM + SEL: collision
  assert(or_case(121, v(0x30), 120, v(0x0c)) == v(0x0c));   // two TSPs: collision, 120 kept
  reset()
}

$[test]
function test_sxm_or_pair_counts_distinct_writers() -> unit = {
  let a : commit = struct { r = RowE, id = 7, g = 4, value = v(0x01), ov = ecc_ov_zero, q = 125, b = 0 };
  let b : commit = struct { r = RowE, id = 7, g = 4, value = v(0x02), ov = ecc_ov_zero, q = 120, b = 0 };
  let c : commit = struct { r = RowE, id = 7, g = 4, value = v(0x04), ov = ecc_ov_zero, q = 120, b = 0 };
  let xs = [|a, b, c|];
  let (bad_a, _, _, _) = commit_wins(a, 0, xs);
  let (bad_b, kept_b, value_b, _) = commit_wins(b, 1, xs);
  let (bad_c, kept_c, _, _) = commit_wins(c, 2, xs);
  assert(not(bad_a) & not(bad_b) & not(bad_c));
  assert(kept_b & not(kept_c) & value_b == v(0x07))
}

// The two liquid threads of one MEM slice are one writer. Each thread's result is
// zero on the other thread's tiles, so their writes to one stream in one commit cycle merge
// (the OR is the per-tile merge) in either order; with a third writer the slice still
// collides and keeps the merged value as the lowest writer.
$[test]
function test_mem_thread_merge() -> unit = {
  assert(or_case(4, v(0x0c), 4, v(0x30)) == v(0x3c));
  assert(or_case(4, v(0x30), 4, v(0x0c)) == v(0x3c));
  reset();
  let g : gap = match write_gap(U_SXM_W_SEL, 0, RowE) { Some(x) => x, None() => 0 };
  let c : nat = 50;
  pending[c % 128] = [| struct { r = RowE, id = 7, g = g, value = v(0x30), ov = ecc_ov_zero, q = 4, b = 0 },
                        struct { r = RowE, id = 7, g = g, value = v(0x40), ov = ecc_ov_zero, q = 9, b = 0 },
                        struct { r = RowE, id = 7, g = g, value = v(0x0c), ov = ecc_ov_zero, q = 4, b = 0 } |];
  tick(c);
  let g1 : gap = if g < 43 then g + 1 else g;
  let (f, x) = srf_read(RowE, 7, g1, c + 1);
  assert(f == SRF_FOUND & x == v(0x3c));
  reset()
}
