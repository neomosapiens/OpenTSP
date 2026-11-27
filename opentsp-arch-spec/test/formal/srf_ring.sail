// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Test of the stream register file ring over the model's own srf_write and srf_read , exhaustive over the writer gap, the
// diagonal difference, the reader gap and the two orders of the writes, and not over the rest: the first write commits at cycle 100 and the two
// values are 0xA5 and 0x5A replicated over the lanes.
// The SMT backend cannot translate srf_write (its trace callback has no generator), and the SMT obligation p_srf_live_window restates
// the slot arithmetic instead of calling the model, so the mutant of the slot function survived the proof gate. This test calls the
// real functions: two writes on one row, stream and gap whose diagonals differ by delta (-63..63, not 0) must not disturb each
// other while both are in the row.
//
// Each case uses a (row, stream) of its own (stream 0..31 of row E or W). A write at gap g with commit cycle c is on diagonal d = g - c and reaches a reader at gap G (> g) at cycle c + (G - g); it is still
// in the row (live) while d + t <= 44. For every writer gap g (0..42), every delta and every reader gap G (g+1..43) the two writes
// A and B (B's diagonal is A's plus delta) are made in both orders; A is read at its own cycle if B is written by then and live, and
// B the same way. The read must find the value written, not the other one.

private function srf_ring_value(b : bits(8)) -> vec = replicate_bits(b, vlanes)

private function srf_ring_cycle(c : int) -> nat = unsigned(get_slice_int(32, c, 0))

private function srf_ring_check(row_e : bool, st : range(0, 31), g : range(0, 43), delta : int, a_first : bool) -> unit = {
  let r : row = if row_e then RowE else RowW;
  let ca : int = 100;
  let can : nat = srf_ring_cycle(ca);
  let da : int = g - ca;
  let db : int = da + delta;
  let cb : int = g - db;
  let cbn : nat = srf_ring_cycle(cb);
  let va = srf_ring_value(0xA5);
  let vb = srf_ring_value(0x5A);
  if a_first then {
    let _ = srf_write(r, st, g, can, va);
    let _ = srf_write(r, st, g, cbn, vb);
    ()
  } else {
    let _ = srf_write(r, st, g, cbn, vb);
    let _ = srf_write(r, st, g, can, va);
    ()
  };
  foreach (rd from (g + 1) to 43) {
    let ta : int = ca + (rd - g);                 // the cycle at which a reader at gap rd sees A
    let tb : int = cb + (rd - g);
    // B is in the row and already written when A is read: db + ta <= 44 (live) and cb <= ta.
    if cb <= ta & db + ta <= 44 then {
      let (k, x) = srf_read(r, st, rd, ta);
      assert(k == SRF_FOUND & x == va, "srf_ring: a write on a live diagonal was lost to another live diagonal (A)")
    };
    if ca <= tb & da + tb <= 44 then {
      let (k, x) = srf_read(r, st, rd, tb);
      assert(k == SRF_FOUND & x == vb, "srf_ring: a write on a live diagonal was lost to another live diagonal (B)")
    }
  }
}

// Each case uses a (row, stream) of its own, so the register file is reset once per 64 cases, not per case.
$[test]
function test_srf_ring_live_diagonals() -> unit = {
  srf_reset();
  var k : int = 0;
  foreach (g from 0 to 42) {
    foreach (delta from -63 to 63) {
      if delta != 0 then {
        foreach (order from 0 to 1) {
          if k >= 64 then { srf_reset(); k = 0 };
          let k5 : range(0, 31) = emod_int(k, 32);
          srf_ring_check(k < 32, k5, g, delta, order == 0);
          k = k + 1
        }
      }
    }
  }
}

// The C backend's start-up code calls main; there is nothing to run.
val main : unit -> unit
function main() = ()
