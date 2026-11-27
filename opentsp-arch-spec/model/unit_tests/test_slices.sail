// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the MXM, SXM and C2C slices, run on the model's own encdec and execute.

private function issue_on(q : queue_id, c : nat) -> unit = {
  issue_cycle = c;
  issue_unit = issuer_of(q);
  issue_queue = q;
  issue_element = 0
}

private function ill32(w : bits(32)) -> bool = match encdec32(w) { ILLEGAL32(_) => true, _ => false }
private function ill64(w : bits(64)) -> bool = match encdec64(w) { ILLEGAL64(_) => true, _ => false }

// C2C Send/Receive: 32-bit layout slice 5, chip_id 21..16, stream 15..10, link 9..6, link bits 6..4 at 5..3, chip bits 8..6 at
// 2..0 (test_c2c.sail covers the link field).
$[test]
function test_c2c_layout() -> unit = {
  let w = encdec32(C2C_SEND(0b000101010, 0b0000000, Streamref(0b100011)));
  assert(w == 0x5 @ 0b000000 @ 0b101010 @ 0b100011 @ 0b0000000000);
  assert(encdec32(w) == C2C_SEND(0b000101010, 0b0000000, Streamref(0b100011)));
  let r = encdec32(C2C_RECEIVE(0b000000001, 0b0000000, Streamref(0b000010)));
  assert(r[27 .. 22] == 0b000001);
  assert(encdec32(w | 0x0000_0001) == C2C_SEND(0b001101010, 0b0000000, Streamref(0b100011)));   // chip bit 6
  assert(encdec32(w | 0x0000_0008) == C2C_SEND(0b000101010, 0b0010000, Streamref(0b100011)));   // link bit 4
  assert(ill32(0x5 @ 0b001010 @ 0x0000 @ 0b000000))   // opcode 10 (2..4 are Deskew, Transmit, RuntimeDeskew; 5..8 the host instructions; 9 DeskewRecv)
}

// The host instructions (opcodes 5..8): PciReceive takes S24..S31 only, HostNotify n = 1..15, HostWait any doorbell.
$[test]
function test_host_layout() -> unit = {
  let pr = encdec32(C2C_PCIRECV(Streamref(0b111001)));
  assert(pr == 0x5 @ 0b000101 @ 0b000000 @ 0b111001 @ 0b0000 @ 0b000000);
  assert(encdec32(pr) == C2C_PCIRECV(Streamref(0b111001)));
  assert(ill32(0x5 @ 0b000101 @ 0b000000 @ 0b100010 @ 0b0000 @ 0b000000));   // S2 is not an incoming host stream
  assert(ill32(0x5 @ 0b000101 @ 0b000000 @ 0b010111 @ 0b0000 @ 0b000000));   // S23
  let hw = encdec32(C2C_HOSTWAIT(0b000011));
  assert(hw == 0x5 @ 0b000111 @ 0x0000 @ 0b000011);
  assert(encdec32(hw) == C2C_HOSTWAIT(0b000011));
  let hn = encdec32(C2C_HOSTNOTIFY(0x9));
  assert(encdec32(hn) == C2C_HOSTNOTIFY(0x9));
  assert(ill32(0x5 @ 0b001000 @ 0x0000 @ 0b00 @ 0x0))   // HostNotify 0
}

// SXM Rotate takes n = 3 or 4 only; Transpose needs 16-aligned groups.
$[test]
function test_sxm_rotate_transpose_encoding() -> unit = {
  let rot = encdec64(SXM_ROTATE(0b0000000011, Streamref(0b100000), Streamref(0b000000)));
  assert(rot[59 .. 54] == 0b000011 & rot[39 .. 30] == 0b0000000011);
  assert(ill64([rot with 39 .. 30 = 0b0000000101]));
  let tr = encdec64(SXM_TRANSPOSE(Streamref(0b110000), Streamref(0b010000)));
  assert(encdec64(tr) == SXM_TRANSPOSE(Streamref(0b110000), Streamref(0b010000)));
  assert(ill64([tr with 41 .. 40 = 0b01]));   // src 0b110001: not 16-aligned
  assert(issued_by(SXM_ROTATE(0b0000000100, Streamref(0b100000), Streamref(0b000000)), U_SXM_W_TSP));
  assert(not(issued_by(SXM_TRANSPOSE(Streamref(0b110000), Streamref(0b010000)), U_SXM_W_PRM)))
}

// Shift and Permute read lanes of later superlanes, so they have their own timing
// class whose result commits no earlier than dispatch + superlanes; the SXM units that stay
// inside a superlane keep timing.sxm.
$[test]
function test_sxm_cross_timing() -> unit = {
  assert(timing_of(SXM_SHIFT(SHIFT_DOWN, 0b000010000, Streamref(0b100000), Streamref(0b000101))) == T_SXM_CROSS);
  assert(timing_of(SXM_MAP(MAP_PERMUTE, 0x0, Streamref(0b100000), Streamref(0b000101))) == T_SXM_CROSS);
  assert(timing_of(SXM_MAP(MAP_DISTRIBUTE, 0x0, Streamref(0b100000), Streamref(0b100001))) == T_SXM);
  assert(timing_of(SXM_TRANSPOSE(Streamref(0b110000), Streamref(0b010000))) == T_SXM);
  assert(timing_of(SXM_SELECT(Streamref(0b100000), Streamref(0b100001), Streamref(0b100010), Streamref(0b100011), Streamref(0b000101))) == T_SXM);
  assert(latency_of(T_SXM_CROSS).d_func >= superlanes)
}

// Rotate needs n*n destination streams on the row.
$[test]
function test_sxm_rotate_range() -> unit = {
  reset();
  issue_on(120, 50);
  assert(execute_test(SXM_ROTATE(0b0000000100, Streamref(0b100000), Streamref(0b010001))) == Diagnostic(OPR_ALIGN))
}

// Each MXM stage needs the previous stage COMMITTED (dispatch + d_func).
// Changes are staged and take effect at the end of the cycle.
$[test]
function test_mxm_stage_readiness() -> unit = {
  reset();
  var p = mxm[0];
  p.buf_rows = 16;
  p.ready_lw = 120;            // the last LW commits at cycle 120
  mxm[0] = p;
  issue_on(105, 119);          // west plane 0, IW one cycle early
  assert(execute_test(MXM_IW(W_I8)) == Diagnostic(RES_NOT_READY));
  issue_on(105, 120);
  assert(execute_test(MXM_IW(W_I8)) == Issued());
  assert(mxm[0].rows == 0);    // staged, not yet applied
  unit_state_flush_test(120);
  assert(mxm[0].rows == 16 & mxm[0].buf_rows == 0);
  assert(mxm[0].ready_iw == 120 + latency_of(T_MXM_IW).d_func);
  issue_on(106, 119 + latency_of(T_MXM_IW).d_func);
  assert(execute_test(MXM_ABC(W_I8, Streamref(0b111000))) == Diagnostic(RES_NOT_READY));
  // ACC before any ABC has committed (D5).
  var q = mxm[0];
  q.abc = [|struct { dispatch = 280, commit = 300, ptype = W_I8, product = vector_init(0x00000000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 }|];
  mxm[0] = q;
  issue_on(106, 299);
  assert(execute_test(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Diagnostic(RES_NOT_READY))
}

// Two queues staging the same plane part in one cycle are RES-003; neither applies.
$[test]
function test_mxm_stage_conflict() -> unit = {
  reset();
  var p = mxm[0];
  p.buf_rows = 16;
  mxm[0] = p;
  issue_on(104, 10);
  assert(execute_test(MXM_IW(W_I8)) == Issued());
  issue_on(105, 10);
  assert(execute_test(MXM_IW(W_I8)) == Issued());
  unit_state_flush_test(10);
  assert(mxm[0].rows == 0 & mxm[0].buf_rows == 16)
}

// A staged SXM map carries its bijective flag and its ready cycle; the flush applies the value,
// the flag and the ready cycle together, and a RES-003 clash drops all three.
$[test]
function test_sxm_map_flag_and_ready() -> unit = {
  reset();
  var mv = sxm_maps[1][3];
  mv[1] = mv[2];
  sxm_stage_list = [| struct { q = 138, sxm = 1, map = 3, value = mv, bij = false, ready = 50 } |];
  assert(sxm_map_bij[1][3] & sxm_map_ready[1][3] == 0);       // staged, not yet applied
  unit_state_flush_test(30);
  assert(not(sxm_map_bij[1][3]) & sxm_map_ready[1][3] == 50 & unsigned(sxm_maps[1][3][1]) == 2);
  assert(sxm_map_bij[0][3] & sxm_map_ready[0][3] == 0 & sxm_map_bij[1][4] & sxm_map_ready[1][4] == 0);
  sxm_stage_list = [| struct { q = 138, sxm = 1, map = 4, value = mv, bij = false, ready = 77 },
                      struct { q = 131, sxm = 1, map = 4, value = mv, bij = false, ready = 78 } |];
  unit_state_flush_test(31);
  assert(sxm_map_bij[1][4] & sxm_map_ready[1][4] == 0 & unsigned(sxm_maps[1][4][1]) == 1)
}

// Lanes 0..2 set to a, b, c (the rest zero).
private function lanes8(a : bits(8), b : bits(8), c : bits(8)) -> vec = {
  var acc : vec = zero_vec;
  acc[7 .. 0] = a;
  acc[15 .. 8] = b;
  acc[23 .. 16] = c;
  acc
}

// The f16 dot product is exact with one rounding: 2^30 + 1 - 2^30 = 1 (per-addition
// f32 rounding would give 0). Inf x 0 is the canonical NaN; -0 x 1 with +0 elsewhere is +0.
$[test]
function test_mxm_f16_single_rounding() -> unit = {
  // w = [2^15, 1, -2^15], x = [2^15, 1, 2^15] (f16 0x7800, 0x3C00, 0xF800).
  let wlo = lanes8(0x00, 0x00, 0x00);
  let whi = lanes8(0x78, 0x3C, 0xF8);
  let xlo = lanes8(0x00, 0x00, 0x00);
  let xhi = lanes8(0x78, 0x3C, 0x78);
  assert(f16_dot(wlo, whi, xlo, xhi) == 0x3F800000);
  // Inf (0x7C00) x 0.
  assert(f16_dot(zero_vec, lanes8(0x7C, 0x00, 0x00), zero_vec, zero_vec) == canon_nan(FMT32));
  // -0 x 1: +0.
  assert(f16_dot(zero_vec, lanes8(0x80, 0x00, 0x00), zero_vec, lanes8(0x3C, 0x00, 0x00)) == 0x00000000)
}

// F16 weights use the even plane of a pair; the odd partner is busy meanwhile.
$[test]
function test_mxm_f16_plane_pair() -> unit = if mxm_planes == 2 then {
  reset();
  issue_on(107, 10);                         // west plane 1
  assert(execute_test(MXM_IW(W_F16)) == Diagnostic(RES_NOT_READY));
  var p = mxm[0];
  p.wtype = W_F16;
  p.rows = 16;
  mxm[0] = p;
  var q = mxm[1];
  q.buf_rows = 16;
  mxm[1] = q;
  issue_on(107, 10);
  assert(execute_test(MXM_IW(W_I8)) == Diagnostic(RES_NOT_READY));   // plane 0 holds f16
  issue_on(109, 10);
  assert(execute_test(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Diagnostic(RES_NOT_READY))
}

// IW with no buffered rows, and ACC with a mismatched result type, are rejected.
$[test]
function test_mxm_operand_checks() -> unit = {
  reset();
  issue_on(104, 10);
  assert(execute_test(MXM_IW(W_I8)) == Diagnostic(RES_NOT_READY));
  var p = mxm[0];
  p.abc = [|struct { dispatch = 0, commit = 5, ptype = W_I8, product = vector_init(0x00000000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 }|];
  mxm[0] = p;
  issue_on(106, 10);
  assert(execute_test(MXM_ACC(A_F32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Diagnostic(OPR_RANGE));
  // ACC destinations are 4 aligned streams (destinations are checked too).
  assert(execute_test(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b000010), 0b00000)) == Diagnostic(OPR_ALIGN))
}

// The MXM weight types are i8, f16, u8 and fp8 (type field 3); the FP8 ABC carries its format in mode bit 13.
$[test]
function test_mxm_type_legality() -> unit = {
  let lw = encdec32(MXM_LW(W_F16, Streamref(0b110000)));
  assert(lw[21 .. 20] == 0b01);
  assert(not(ill32([lw with 21 .. 20 = 0b10])));   // u8 (HC34 p.26)
  assert(not(ill32([lw with 21 .. 20 = 0b11])));   // fp8
  assert(encdec32([lw with 21 .. 20 = 0b11]) == MXM_LW(W_FP8, Streamref(0b110000)));
  let abc = encdec32(MXM_ABC(W_I8, Streamref(0b100100)));
  assert(ill32([abc with 13 .. 13 = 0b1]));        // mode bit 13 is zero for i8, f16 and u8
  assert(encdec32([abc with 21 .. 20 = 0b11]) == MXM_ABC8(0b0, Streamref(0b100100)));    // abc.e4m3
  assert(encdec32([abc with 21 .. 20 = 0b11, 13 .. 13 = 0b1]) == MXM_ABC8(0b1, Streamref(0b100100)));   // abc.e5m2
  assert(encdec32(MXM_ABC8(0b1, Streamref(0b100100))) == [abc with 21 .. 20 = 0b11, 13 .. 13 = 0b1]);
  assert(ill32([lw with 17 .. 14 = 0x1]));   // LW source must be 16-aligned
  let acc = encdec32(MXM_ACC(A_F32, ACC_ACCUMULATE, Streamref(0b001100), 0b00000));
  assert(acc[13] == bitone);
  assert(ill32([acc with 21 .. 20 = 0b01]))
}

// Every data queue issues to its documented unit (exhaustive over the 144 queues).
$[test]
function test_queue_units() -> unit = {
    foreach (n from 0 to 143) {
    let i = issuer_of(n);
    let ok : bool =
      if n <= 43 then i.unit == U_MEM_W & i.slice == n
      else if n <= 87 then i.unit == U_MEM_E & i.slice == n
      else if n <= 103 then i.unit == U_VXM
      else if n <= 109 then i.unit == U_MXM_W & i.plane == (if n <= 106 then 0 else 1)
      else if n <= 115 then i.unit == U_MXM_E & i.plane == mxm_planes + (if n <= 112 then 0 else 1)
      else i.sxm == (if n >= 130 then 1 else 0);
    assert(ok, "queue map")
  };
  // The published split: per hemisphere 6 MXM queues and 14 SXM-side queues, the C2C
  // TX and RX among them. Inside the 14: 4 distributors, 2 transposers (HC34 p.31), 2 shifters,
  // 1 permuter, 1 select, 2 C2C TX and 2 C2C RX.
  foreach (h from 0 to 1) {
    var dst : int = 0; var tsp : int = 0; var shf : int = 0; var prm : int = 0; var sel : int = 0;
    var tx : int = 0; var rx : int = 0;
    foreach (n from (116 + 14 * h) to (129 + 14 * h)) {
      let i = issuer_of(n);
      assert(i.sxm == h, "SXM-side hemisphere");
      match i.unit {
        U_SXM_W_DST => dst = dst + 1, U_SXM_W_TSP => tsp = tsp + 1, U_SXM_W_SHF => shf = shf + 1,
        U_SXM_W_PRM => prm = prm + 1, U_SXM_W_SEL => sel = sel + 1,
        U_SXM_E_DST => dst = dst + 1, U_SXM_E_TSP => tsp = tsp + 1, U_SXM_E_SHF => shf = shf + 1,
        U_SXM_E_PRM => prm = prm + 1, U_SXM_E_SEL => sel = sel + 1,
        U_C2C_W_TX => tx = tx + 1, U_C2C_E_TX => tx = tx + 1,
        U_C2C_W_RX => rx = rx + 1, U_C2C_E_RX => rx = rx + 1,
        _ => assert(false, "SXM-side unit"),
      }
    };
    assert(dst == 4 & tsp == 2 & shf == 2 & prm == 1 & sel == 1 & tx == 2 & rx == 2)
  };
  assert(issuer_of(126).unit == U_C2C_W_TX & issuer_of(128).unit == U_C2C_W_RX);
  assert(issuer_of(140).unit == U_C2C_E_TX & issuer_of(143).unit == U_C2C_E_RX)
}

// ---- Front end ----

// Text decode (design sec. 7.2): the bundle bit is cleared before decode; the slice tag in
// bits 30..28 gives the width; tags 6 and 7 and every word that does not decode are
// OTS-E-DEC-003 (one unit); a 64-bit instruction is its high unit first.
$[test]
function test_decode_text() -> unit = {
  let rd = encdec64(MEM_DIRECT(MEM_READ, sram_addr_of(1), Streamref(0b000000), 0b0));
  let hi : bits(32) = rd[63 .. 32];
  let lo : bits(32) = rd[31 .. 0];
  assert(text_width(hi) == 2 & text_width(0x00400000) == 1);
  assert(decode_text(hi, lo) == Ok(MEM_DIRECT(MEM_READ, sram_addr_of(1), Streamref(0b000000), 0b0)));
  // The bundle bit is not part of the instruction.
  assert(decode_text(0x9 @ hi[27 .. 0], lo) == decode_text(hi, lo));
  assert(decode_text(0x80400000, 0x00000000) == Ok(SYNC()));
  assert(decode_text(0x00400000, 0x00000000) == Ok(SYNC()));
  // Tags 6 and 7 (with or without the bundle bit) do not decode.
  assert(decode_text(0x60400000, 0x00000000) == Err(DEC_OPCODE));
  assert(decode_text(0x70400000, 0x00000000) == Err(DEC_OPCODE));
  assert(decode_text(0xE0400000, 0x00000000) == Err(DEC_OPCODE));
  assert(text_width(0x60400000) == 1 & text_width(0x70400000) == 1);
  // A reserved bit makes a word undecodable; NOP 0 does not decode.
  assert(decode_text(0x00400001, 0x00000000) == Err(DEC_OPCODE));
  assert(decode_text(0x00000000, 0x00000000) == Err(DEC_OPCODE));
  assert(decode_text(hi, lo | 0x00000001) == Err(DEC_OPCODE));
  let lw = encdec32(MXM_LW(W_I8, Streamref(0b110000)));
  assert(decode_text(lw, 0x00000000) == Ok(MXM_LW(W_I8, Streamref(0b110000))))
}

// Every Sync opens its own window, closed by the first Notify at or after it.
$[test]
function test_barrier_multiple_windows() -> unit = {
  reset();
  barrier_sync_test(144, 5);
  barrier_notify_test(10);
  let lat = barrier_release_latency;
  assert(not(barrier_parked(4)));
  assert(barrier_parked(20) == (20 < 10 + lat));
  assert(barrier_parked(9 + lat) & not(barrier_parked(10 + lat)));
  barrier_sync_test(144, 100);
  assert(barrier_parked(1000));                  // no Notify yet: open-ended
  barrier_notify_test(110);
  assert(barrier_parked(100) & barrier_parked(109 + lat));
  assert(not(barrier_parked(110 + lat)))
}

// ---- source alignment: MEM ----

// AGT configuration encodings (HC34 p.25: Countdown, Step, Iterations).
$[test]
function test_agt_encodings() -> unit = {
  let st = encdec64(MEM_STEP(0x0018));
  assert(st[63 .. 60] == 0x1 & st[59 .. 54] == 0b000110 & st[31 .. 16] == 0x0018);
  assert(encdec64(st) == MEM_STEP(0x0018));
  assert(encdec64(MEM_STEP(0xFFFF)) != 0x0000_0000_0000_0000);       // step -1
  assert(ill64(0x1 @ 0b000110 @ 0x00000 @ 0b00 @ get_slice_int(16, step_limit + 1, 0) @ 0x0000));   // one past the limit
  assert(not(ill64(0x1 @ 0b000110 @ 0x00000 @ 0b00 @ get_slice_int(16, step_limit, 0) @ 0x0000)));
  assert(not(ill64(0x1 @ 0b000110 @ 0x00000 @ 0b00 @ get_slice_int(16, 0 - step_limit, 0) @ 0x0000)));
  assert(step_limit == (if addr_bits == 16 then 32767 else 2 ^ addr_bits - 1));
  assert(encdec64(encdec64(MEM_ITERATIONS(0x006F))) == MEM_ITERATIONS(0x006F));
  assert(is_loopable(MEM_DIRECT(MEM_READ, sram_addr_of(0), Streamref(0b000000), 0b0)));
  assert(not(is_loopable(MEM_SETSEGMENT(get_slice_int(17, 16, 0)))));
  // ReadWrite is gone: opcode 2 does not decode (a Read and a Write record share a cycle instead).
  assert(ill64(0x1 @ 0b000010 @ 0b000000000 @ 0b0_0000_0000_0000 @ 0b000000 @ 0b000 @ 0b1_0000_0000_0000 @ 0b000001 @ 0x0))
}

// Loop counters: two levels (inner 3 references step 2, outer 2 references step 10),
// the loop advances by the countdown rule: a reference takes a cycle and
// every Wait(countdown) is unconditional, so the period is 1 + countdown inside a level and
// 1 + inner + outer countdown at the level boundary.
$[test]
function test_agt_advance_two_levels() -> unit = {
  reset();
  var a = agts[0];
  a = { a with state = AGT_CONFIG };
  agts[0] = a;
  assert(agt_configure_test(0, AGT_COUNTDOWN, 0x0001));
  assert(agt_configure_test(0, AGT_STEP, 0x0002));
  assert(agt_configure_test(0, AGT_ITERATIONS, 0x0003));      // inner: 3 references
  assert(agt_configure_test(0, AGT_COUNTDOWN, 0x0004));       // starts level 1
  assert(agt_configure_test(0, AGT_STEP, 0x000A));
  assert(agt_configure_test(0, AGT_ITERATIONS, 0x0002));      // outer: 2 references
  // References: offsets 0, 2, 4 | 14, 16, 18 (outer step added to the last inner address);
  // delays: 2, 2, then 1 + 1 + 4 = 6 at the level boundary, then 2, 2.
  let offs : vector(6, int) = [18, 16, 14, 4, 2, 0];
  let dels : vector(6, int) = [2, 2, 6, 2, 2, 0];   // (Order dec: index 0 is last)
  var ls = agts[0].levels;
  var off : int = 0;
  foreach (k from 1 to 5) {
    match agt_advance_test(ls, 2) {
      None() => assert(false, "loop ended early"),
      Some((l2, d, st, _)) => { off = off + st; assert(off == offs[k], "offset"); assert(d == dels[k], "delay"); ls = l2 },
    }
  };
  assert(agt_advance_test(ls, 2) == None())
}

// Defaults when a field is not received: countdown 0, step 0, numIter 2 (the defaults are
// 0, 0 and 2). Iterations n makes n references;
// n < 1 counts as 1 (decision). Only the levels received so far take part.
$[test]
function test_agt_defaults() -> unit = {
  reset();
  agts[0] = { agts[0] with state = AGT_CONFIG };
  assert(agt_configure_test(0, AGT_STEP, 0x0002));
  let l0 = agts[0].levels[0];
  assert(l0.countdown == 0 & l0.iters == 2 & l0.step == 2);
  match agt_advance_test(agts[0].levels, 1) {
    Some((ls, d, st, _)) => { assert(d == 1 & st == 2); assert(agt_advance_test(ls, 1) == None()) },
    None() => assert(false, "default numIter is 2"),
  };
  assert(agt_advance_test(agts[0].levels, 0) == None());       // no level received
  reset();
  agts[0] = { agts[0] with state = AGT_CONFIG };
  assert(agt_configure_test(0, AGT_ITERATIONS, 0x0000));
  assert(agts[0].levels[0].iters == 1);
  assert(agt_configure_test(0, AGT_COUNTDOWN, 0x0000));         // level 1; explicit 0 counts as 1
  assert(agts[0].levels[1].countdown == 1)
}

// A fifth level is rejected.
$[test]
function test_agt_four_levels_max() -> unit = {
  reset();
  agts[0] = { agts[0] with state = AGT_CONFIG };
  foreach (k from 0 to 3) { assert(agt_configure_test(0, AGT_COUNTDOWN, 0x0001)) };
  assert(not(agt_configure_test(0, AGT_COUNTDOWN, 0x0001)))
}

// Gather/Scatter take one map stream (ISCA 2020 Table I).
$[test]
function test_gather_scatter_one_map() -> unit = {
  let g = encdec64(MEM_GATHER(Streamref(0b000011), Streamref(0b100101), 0x0, 0b0));
  assert(g[47 .. 42] == 0b000011 & g[41 .. 36] == 0b100101 & g[35 .. 0] == sail_zeros(36));
  assert(ill64(g | 0x0000_0000_0000_0001));
  assert(encdec64(g) == MEM_GATHER(Streamref(0b000011), Streamref(0b100101), 0x0, 0b0));
  // Lane index k in 51..48, thread tag in 53; bit 52 is reserved.
  let g2 = encdec64(MEM_SCATTER(Streamref(0b000011), Streamref(0b100101), 0xA, 0b1));
  assert(g2[53] == bitone & g2[51 .. 48] == 0xA);
  assert(ill64(g2 | 0x0010_0000_0000_0000));
  let rd = encdec64(MEM_DIRECT(MEM_WRITE, sram_addr_of(1), Streamref(0b000000), 0b1));
  assert(rd[53] == bitone & encdec64(rd) == MEM_DIRECT(MEM_WRITE, sram_addr_of(1), Streamref(0b000000), 0b1))
}

// StreamStep s (opcode 11, s = 0..31) configures the level between Step and Iterations.
$[test]
function test_agt_streamstep() -> unit = {
  let w = encdec64(MEM_STREAMSTEP(0x0002));
  assert(w[59 .. 54] == 0b001011 & w[31 .. 16] == 0x0002);
  assert(ill64([w with 31 .. 16 = 0x0020]));
  assert(agt_config(MEM_STREAMSTEP(0x0002)) == Some((AGT_STREAMSTEP, 0x0002)));
  reset();
  agts[0] = { agts[0] with state = AGT_CONFIG };
  assert(agt_configure_test(0, AGT_STEP, 0x0001));
  assert(agt_configure_test(0, AGT_STREAMSTEP, 0x0002));
  assert(agt_configure_test(0, AGT_ITERATIONS, 0x0002));
  assert(agts[0].level == 0);
  match agt_advance_test(agts[0].levels, 1) {
    Some((_, _, st, ss)) => assert(st == 1 & ss == 2),
    None() => assert(false, "loop"),
  }
}

// SetSegment bounds-checks direct references.
$[test]
function test_setsegment() -> unit = {
  reset();
  issue_on(0, 10);
  assert(execute_test(MEM_SETSEGMENT(get_slice_int(17, 16, 0))) == Issued());
  assert(mem_segment[0] == 16);
  issue_on(0, 11);
  assert(execute_test(MEM_DIRECT(MEM_READ, sram_addr_of(16), Streamref(0b000000), 0b0)) == Diagnostic(OPR_RANGE));
  assert(not(ill64(encdec64(MEM_SETSEGMENT(get_slice_int(17, 2 ^ addr_bits, 0))))));
  assert(ill64(encdec64(MEM_SETSEGMENT(get_slice_int(17, 2 ^ addr_bits, 0))) | 0x0000_0000_0001_0000))
}

// ---- source alignment: data types (HC34 p.26) ----

$[test]
function test_vxm_type_codes() -> unit = {
  // u8 add: type bits 53..52 = 00, extension 23..22 = 01.
  let w = encdec64(VXM_BIN(VADD, U8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(w[53 .. 52] == 0b00 & w[23 .. 22] == 0b01);
  assert(encdec64(w) == VXM_BIN(VADD, U8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  // Codes 13..15 (extension 11 with type 01..11) stay reserved; extension 11 with type 00 is code 12, e5m2.
  assert(ill64([w with 53 .. 52 = 0b01, 23 .. 22 = 0b11]));
  assert(not(ill64([w with 23 .. 22 = 0b11])));
  // bool add does not decode (extension 10 type 00 is b8); bool and does.
  assert(ill64([w with 23 .. 22 = 0b10]));
  assert(not(ill64(encdec64(VXM_BIN(VAND, B16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0)))))
}

$[test]
function test_unsigned_and_bool_kernels() -> unit = {
  assert(k_binary(1, U8, 0x000000F0, 0x00000020) == 0x000000FF);    // add_sat u8 saturates at 255
  assert(k_binary(2, U8, 0x000000F0, 0x00000020) == 0x00000010);    // add_mod u8 wraps
  assert(k_binary(1, I8, 0x00000070, 0x00000020) == 0x0000007F);    // add_sat i8 at 127
  assert(k_binary(9, U16, 0x0000FFFF, 0x00000001) == 0x0000FFFF);   // max u16: 65535 > 1
  assert(k_binary(9, I16, 0x0000FFFF, 0x00000001) == 0x00000001);   // max i16: -1 < 1
  assert(k_binary(11, B8, 0x00000005, 0x00000080) == 0x000000FF);   // true and true
  assert(k_cmp(0, B8, 0x00000005, 0x00000001) == 0x000000FF);       // bool eq on truth values
  assert(k_cmp(2, U32, 0xFFFFFFFF, 0x00000001) == 0x00000000)       // u32 lt: 4294967295 < 1 false
}

// The unary kernels read an unsigned element as unsigned and
// a boolean as a truth value. (No instruction encodes Neg, Abs or ReLU on an unsigned or boolean
// type, `unop_legal`, so these are kernel-level tests; the kernel is total and defined.)
$[test]
function test_unary_kernels_unsigned_bool() -> unit = {
  // Abs and ReLU of an unsigned value are the value itself.
  assert(k_unary(16, U8, 0x000000FF) == 0x000000FF);
  assert(k_unary(16, U8, 0x00000080) == 0x00000080);
  assert(k_unary(20, U8, 0x00000080) == 0x00000080);
  assert(k_unary(20, U8, 0x000000FF) == 0x000000FF);
  assert(k_unary(16, U16, 0x0000FFFE) == 0x0000FFFE);
  assert(k_unary(20, U16, 0x00008000) == 0x00008000);
  assert(k_unary(16, U32, 0xFFFFFFFF) == 0xFFFFFFFF);
  assert(k_unary(20, U32, 0x80000000) == 0x80000000);
  // Neg of an unsigned value is the modular negation (2^n - x).
  assert(k_unary(15, U8, 0x00000001) == 0x000000FF);
  assert(k_unary(15, U8, 0x00000000) == 0x00000000);
  assert(k_unary(15, U8, 0x00000080) == 0x00000080);
  assert(k_unary(15, U16, 0x00000001) == 0x0000FFFF);
  assert(k_unary(15, U32, 0x00000002) == 0xFFFFFFFE);
  // Not is a bitwise complement inside the element width.
  assert(k_unary(14, U8, 0x000000F0) == 0x0000000F);
  assert(k_unary(14, U16, 0x0000FF00) == 0x000000FF);
  // Signed types are unchanged.
  assert(k_unary(16, I8, 0x000000FF) == 0x00000001);
  assert(k_unary(16, I8, 0x00000080) == 0x00000080);
  assert(k_unary(20, I8, 0x00000080) == 0x00000000);
  assert(k_unary(20, I16, 0x00008001) == 0x00000000);
  assert(k_unary(15, I8, 0x00000001) == 0x000000FF);
  // Booleans: only Not is defined (logical, canonical); every other unary kernel gives 0.
  assert(k_unary(14, B8, 0x00000000) == 0x000000FF);
  assert(k_unary(14, B8, 0x00000005) == 0x00000000);
  assert(k_unary(15, B8, 0x000000FF) == 0x00000000);
  assert(k_unary(16, B16, 0x0000FFFF) == 0x00000000);
  assert(k_unary(20, B32, 0xFFFFFFFF) == 0x00000000)
}

// A float kernel result has no bits above the format width, whatever the lane
// input carried there; TanH(+-0) is the canonical signed zero of the format.
$[test]
function test_float_kernels_no_stray_bits() -> unit = {
  assert(fp_tr_value(21, FMT16, 0x00000000) == Some(0x00000000));
  assert(fp_tr_value(21, FMT16, 0x00008000) == Some(0x00008000));
  assert(fp_tr_value(21, FMT16, 0x12340000) == Some(0x00000000));   // stray high bits dropped
  assert(fp_tr_value(21, FMT16, 0xABCD8000) == Some(0x00008000));
  assert(fp_tr_value(21, FMT32, 0x00000000) == Some(0x00000000));
  assert(fp_tr_value(21, FMT32, 0x80000000) == Some(0x80000000));
  assert(k_unary(20, F16, 0x00013C00) == 0x00003C00);                // ReLU of a positive f16 keeps 16 bits
  assert(k_unary(20, F32, 0x3F800000) == 0x3F800000)
}

$[test]
function test_generic_convert() -> unit = {
  assert(k_convert_any(I32, U8, 0xFFFFFFFF) == 0x00000000);          // -1 -> u8 saturates to 0
  assert(k_convert_any(U16, I8, 0x00000200) == 0x0000007F);          // 512 -> i8 saturates
  assert(k_convert_any(F32, U8, 0x43800000) == 0x000000FF);          // 256.0 -> u8 255
  assert(k_convert_any(F32, B32, 0x80000000) == 0x00000000);         // -0.0 -> false
  assert(k_convert_any(B8, F16, 0x00000001) == 0x00003C00);          // true -> 1.0
  assert(k_convert_any(U8, F32, 0x000000C8) == 0x43480000);          // 200 -> 200.0
  let w = encdec64(VXM_CVT(U8, F32, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
  assert(w[59 .. 54] == 0b100010 & w[27 .. 24] == 0b0011 & w[23 .. 22] == 0b01)
}

$[test]
function test_mxm_u8() -> unit = {
  let lw = encdec32(MXM_LW(W_U8, Streamref(0b110000)));
  assert(lw[21 .. 20] == 0b10);
  assert(encdec32(lw) == MXM_LW(W_U8, Streamref(0b110000)))
}

// Undefined type codes (13, 14, 15: extension 11 with type 01..11; the codes 11 and 12 are the FP8 types) do not decode.
$[test]
function test_undefined_type_codes() -> unit = {
  let w = encdec64(VXM_CVT(U8, F32, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
  assert(ill64([w with 27 .. 24 = 0b1101]));
  assert(ill64([w with 27 .. 24 = 0b1111]));
  assert(not(ill64([w with 27 .. 24 = 0b1011])) & not(ill64([w with 27 .. 24 = 0b1100])));   // e4m3, e5m2
  let b = encdec64(VXM_BIN(VADD, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(not(ill64([b with 53 .. 52 = 0b11, 23 .. 22 = 0b10])));   // code 11: add.e4m3
  assert(not(ill64([b with 53 .. 52 = 0b00, 23 .. 22 = 0b11])));   // code 12: add.e5m2
  assert(ill64([b with 53 .. 52 = 0b01, 23 .. 22 = 0b11]));        // code 13
  assert(ill64([b with 53 .. 52 = 0b10, 23 .. 22 = 0b11]));        // code 14
  assert(ill64([b with 53 .. 52 = 0b11, 23 .. 22 = 0b11]));        // code 15
  let a = encdec64(VXM_BIN(VAND, B8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(ill64([a with 53 .. 52 = 0b11]))                          // and.e4m3 is not defined
}

// ---- source alignment: ICU on every queue, Repeat, C2C Deskew ----

$[test]
function test_icu_on_data_queues() -> unit = {
  // NOP, Sync, Notify, Repeat, Ifetch, Config are common to all functional slices (HC34 p.27).
  assert(issued_by(SYNC(), U_VXM) & issued_by(NOTIFY(), U_MEM_E) & issued_by(NOP(0x0001), U_MXM_W));
  assert(issued_by(REPEAT(0x036, 0x1), U_VXM));
  let r = encdec32(REPEAT(0x036, 0x1));
  assert(r[27 .. 22] == 0b000011 & r[15 .. 4] == 0x036 & r[3 .. 0] == 0x1);
  assert(ill32(0x0 @ 0b000011 @ 0b000000 @ 0x000 @ 0x1));          // Repeat 0 times does not decode
  assert(encdec32(encdec32(IFETCH(Streamref(0b100011)))) == IFETCH(Streamref(0b100011)));
  assert(encdec32(encdec32(CONFIG(0x0006))) == CONFIG(0x0006))
}

// Ifetch has ONE stream operand s (ISCA 2020 sec. III-A3: "a single stream operand
// ... filling the IQ with 640-bytes (a pair of 320-byte vectors)"): the pair is the two
// consecutive vectors of s, read at the execute cycle and one cycle later. Any stream is legal.
$[test]
function test_ifetch_one_stream() -> unit = {
  reset();
  issue_on(88, 30);                                  // a VXM queue: read gap vxm_gap on either row
  let _ = srf_write(RowE, 4, 20, 29, zero_vec);      // stream E:4 reaches gap 21 (the VXM) at cycle 30
  // Only the first vector: nothing on E:4 at cycle 31.
  assert(execute_test(IFETCH(Streamref(0b000100))) == Issued());
  tick(31);
  assert(fill_fifo[88] == [||]);
  reset();
  issue_on(88, 30);
  let _ = srf_write(RowE, 4, 20, 29, zero_vec);
  let _ = srf_write(RowE, 4, 20, 30, zero_vec);      // the next vector of E:4 reaches gap 21 at cycle 31
  assert(execute_test(IFETCH(Streamref(0b000100))) == Issued());
  // The second vector of another stream does not make a pair: the pair is on one stream.
  reset();
  issue_on(88, 30);
  let _ = srf_write(RowE, 4, 20, 29, zero_vec);
  let _ = srf_write(RowE, 5, 20, 30, zero_vec);
  assert(execute_test(IFETCH(Streamref(0b000100))) == Issued());
  tick(31);
  assert(fill_fifo[88] == [||]);
  // An odd stream is legal.
  reset();
  issue_on(88, 30);
  let _ = srf_write(RowE, 5, 20, 29, zero_vec);
  let _ = srf_write(RowE, 5, 20, 30, zero_vec);
  assert(execute_test(IFETCH(Streamref(0b000101))) == Issued())
}

// A C2C RX queue fetches its instruction text where the TX queues of its hemisphere
// read: the row arriving at the chip edge, at the TX gap (west W, east E). It reads no other
// stream, so Ifetch is its only reader.
$[test]
function test_ifetch_rx_queue() -> unit = {
  let tx : gap = match read_gap(U_C2C_W_TX, 0, RowW) { Some(g) => g, None() => 1 };
  let g0 : gap = if tx >= 1 then tx - 1 else 0;
  reset();
  issue_on(128, 30);                                 // west C2C RX
  let _ = srf_write(RowW, 4, g0, 29, zero_vec);
  let _ = srf_write(RowW, 4, g0, 30, zero_vec);
  assert(execute_test(IFETCH(Streamref(0b100100))) == Issued());
  assert(execute_test(IFETCH(Streamref(0b000100))) == Diagnostic(DIR_UPSTREAM));
  reset();
  issue_on(142, 30);                                 // east C2C RX
  let _ = srf_write(RowE, 4, g0, 29, zero_vec);
  let _ = srf_write(RowE, 4, g0, 30, zero_vec);
  assert(execute_test(IFETCH(Streamref(0b000100))) == Issued());
  assert(read_gap(U_C2C_E_RX, 0, RowE) == read_gap(U_C2C_E_TX, 0, RowE));
  reset()
}

// A data-queue Sync parks that queue only; a Notify on any queue releases it.
$[test]
function test_queue_sync() -> unit = {
  reset();
  barrier_sync_test(88, 10);
  barrier_notify_test(20);
  assert(queue_parked(88, 30));
  assert(not(queue_parked(89, 30)));
  assert(not(queue_parked(88, 20 + barrier_release_latency)));
  assert(not(barrier_parked(30)))
}

$[test]
function test_c2c_deskew() -> unit = {
  let w = encdec32(C2C_DESKEW());
  assert(w == 0x5 @ 0b000010 @ 0x0000 @ 0b000000)
}

// Power management and multi-chip synchronization instructions decode (ISCA 2022).
$[test]
function test_power_and_sync_instructions() -> unit = {
  assert(encdec32(encdec32(SETVL(0b001010))) == SETVL(0b001010));
  assert(ill32(0x0 @ 0b000110 @ 0b000000 @ 0b0000000000 @ 0b000000));    // SetVL 0
  assert(not(ill32(0x0 @ 0b000110 @ 0b000000 @ 0b0000000000 @ 0b010101)));    // SetVL 21: legal on every profile (the most superlanes is 32)
  assert(encdec32(encdec32(SETVL(0b100000))) == SETVL(0b100000));         // SetVL 32
  assert(ill32(0x0 @ 0b000110 @ 0b000000 @ 0b0000000000 @ 0b100001));    // SetVL 33
  assert(ill32(0x0 @ 0b000110 @ 0b000000 @ 0b0000000000 @ 0b111111));    // SetVL 63
  assert(encdec64(encdec64(MEM_POWERCONFIG(0x0003))) == MEM_POWERCONFIG(0x0003));
  assert(encdec64(encdec64(MEM_DEEPSLEEP())) == MEM_DEEPSLEEP());
  assert(encdec32(encdec32(C2C_TRANSMIT(0b000000111, 0b0000010))) == C2C_TRANSMIT(0b000000111, 0b0000010));
  assert(encdec32(encdec32(C2C_RTDESKEW(0x00FC))) == C2C_RTDESKEW(0x00FC))
}

// ---------------------------------------------------------------------------------------
// the local-memory form of Ifetch and the optional IQ occupancy check.

// Ifetch from local memory (ISCA 2020 Table I: "Fetch instructions from streams or local
// memory"): IfetchM a reads words a and a+1 of the issuing MEM slice (640 bytes in the
// reference profile). Decision MEM queues only; a+1 inside the segment.
$[test]
function test_ifetch_local_memory() -> unit = {
  reset();
  let w = encdec32(IFETCHM(sram_addr_of(16)));
  assert(w == 0x0 @ 0b000111 @ 0b000000 @ 0x0010);
  assert(encdec32(w) == IFETCHM(sram_addr_of(16)));
  // The address field has 16 bits; a value that needs more than addr_bits bits does not decode.
  if addr_bits < 16 then assert(ill32(0x0 @ 0b000111 @ 0b000000 @ get_slice_int(16, 2 ^ addr_bits, 0)));
  assert(not(ill32(0x0 @ 0b000111 @ 0b000000 @ get_slice_int(16, 2 ^ addr_bits - 1, 0))));
  assert(issued_by(IFETCHM(sram_addr_of(0)), U_MEM_W));
  assert(issued_by(IFETCHM(sram_addr_of(0)), U_MEM_E));
  assert(not(issued_by(IFETCHM(sram_addr_of(0)), U_VXM)));
  issue_on(4, 30);
  assert(execute_test(IFETCHM(sram_addr_of(16))) == Issued());
  // The second word of the pair must be inside the slice (and the segment).
  assert(execute_test(IFETCHM(sram_addr_of(words_per_slice - 1))) == Diagnostic(OPR_RANGE));
  assert(execute_test(IFETCHM(sram_addr_of(words_per_slice - 2))) == Diagnostic(QUE_BUSY));
  local_fetch_complete(31);
  issue_on(4, 32);
  assert(execute_test(IFETCHM(sram_addr_of(words_per_slice - 2))) == Issued());
  reset();
  issue_on(4, 34);
  mem_segment[4] = 16;
  assert(execute_test(IFETCHM(sram_addr_of(15))) == Diagnostic(OPR_RANGE));
  assert(execute_test(IFETCHM(sram_addr_of(14))) == Issued());
  reset()
}

// The transcendentals' Ziv loop converges on every i8 and f16 input and on a stride
// of f32 inputs (every 2^18th bit pattern), so no result comes from a fallback.
$[test]
function test_transcendental_converges() -> unit = {
  foreach (op from 21 to 24) {
    foreach (n from -128 to 127) { assert(is_some(i8_tr_value(op, n))) };
    foreach (w from 0 to 65535) { assert(is_some(fp_tr_value(op, FMT16, get_slice_int(32, w, 0)))) };
    foreach (k from 0 to 16383) { assert(is_some(fp_tr_value(op, FMT32, get_slice_int(32, 262144 * k + 12345, 0)))) }
  }
}
