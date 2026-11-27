// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the VXM, MXM and SXM.

function slice_test_set_issue_context(q : queue_id, c : int) -> unit = {
  assert(c >= 0);
  issue_cycle = c;
  issue_unit = issuer_of(q);
  issue_queue = q;
  issue_element = 0
}

function slice_test_illegal_word32(w : bits(32)) -> bool = match encdec32(w) { ILLEGAL32(_) => true, _ => false }
function slice_test_illegal_word64(w : bits(64)) -> bool = match encdec64(w) { ILLEGAL64(_) => true, _ => false }

// A plane with `n` installed rows of type t, ready since cycle 0.
function slice_test_install_mxm_rows(pl : range(0, 3), t : mxm_wtype, n : range(0, mxm_rows)) -> unit =
  if pl < 2 * mxm_planes then {
    var p = mxm[pl];
    p.wtype = t;
    p.rows = n;
    p.ready_iw = 0;
    mxm[pl] = p
  }

// FP16 initiation follows the configured occupancy on every queue of the paired plane.
$[test]
function test_mxm_f16_abc_occupancy() -> unit = {
  reset();
  if mxm_planes == 2 then {                                 // an f16 plane needs its pair
  slice_test_install_mxm_rows(0, W_F16, 4);
  var p = mxm[0];
  p.abc = [| struct { dispatch = 99, commit = 119, ptype = W_F16, product = vector_init(0x00000000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 } |];
  mxm[0] = p;
  slice_test_set_issue_context(106, 100);                                    // plane 0, the cycle after the f16 ABC
  assert(execute_test(MXM_ABC(W_F16, Streamref(0b100010))) == Diagnostic(if timing_mxm_occupancy_f16 > 1 then RES_NOT_READY else TIM_NO_VALUE));
  slice_test_set_issue_context(104, 100);                                    // another queue of the same plane
  assert(execute_test(MXM_ABC(W_F16, Streamref(0b100010))) == Diagnostic(if timing_mxm_occupancy_f16 > 1 then RES_NOT_READY else TIM_NO_VALUE));
  slice_test_set_issue_context(106, 99 + timing_mxm_occupancy_f16);             // the initiation boundary is free
  assert(execute_test(MXM_ABC(W_F16, Streamref(0b100010))) == Diagnostic(TIM_NO_VALUE))
  };
  // An i8 ABC does not hold the plane: a second one in the next cycle reaches its read.
  reset();
  slice_test_install_mxm_rows(0, W_I8, 4);
  var p8 = mxm[0];
  p8.abc = [| struct { dispatch = 99, commit = 119, ptype = W_I8, product = vector_init(0x00000000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 } |];
  mxm[0] = p8;
  slice_test_set_issue_context(106, 100);
  assert(execute_test(MXM_ABC(W_I8, Streamref(0b100010))) == Diagnostic(TIM_NO_VALUE))
}

// Lanes 0, 1 = a, b; every other lane 0.
function slice_test_first_two_bytes(a : bits(8), b : bits(8)) -> vec = {
  var out : vec = zero_vec;
  out[7 .. 0] = a;
  out[15 .. 8] = b;
  out
}

// Schedule activation stream s (west row) to be visible at the MXM read gap of an ABC
// dispatched at cycle c (read at c + d_skew), by writing it at the gap below in the right cycle.
function slice_test_write_mxm_operand(s : range(0, 31), x : vec, c : nat) -> unit = {
  let t = c + entry_latency("ABC", T_MXM_ABC).d_skew;
  match read_gap(issuer_of(104).unit, 0, RowW) {
    Some(g) => if g > 0 & t > 0 then { let _ = srf_write(RowW, s, g - 1, t - 1, x); () },
    None() => assert(false, "no MXM read gap"),
  }
}

// Integer ABC operand signs (s x s, u x s, s x u, u x u). The installed weight type gives the weight sign, the ABC type the activation sign.
// Weights row 0 = [0xFF, 0x02], activations [0xFF, 0x03]:
//   s x s: (-1)(-1) + 2*3 = 7      s x u: (-1)(255) + 6 = -249
//   u x s: 255(-1) + 6 = -249      u x u: 255*255 + 6 = 65031
$[test]
function test_mxm_mixed_sign() -> unit = {
  foreach (k from 0 to 3) {
    let (wt, at, expect) : (mxm_wtype, mxm_wtype, bits(32)) = match k {
      0 => (W_I8, W_I8, 0x00000007), 1 => (W_I8, W_U8, 0xFFFFFF07),
      2 => (W_U8, W_I8, 0xFFFFFF07), _ => (W_U8, W_U8, 0x0000FE07) };
    reset();
    slice_test_install_mxm_rows(0, wt, 1);
    var p = mxm[0];
    p.weights[0] = slice_test_first_two_bytes(0xFF, 0x02);
    mxm[0] = p;
    slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0xFF, 0x03), 50);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(at, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => { assert(e.product[0] == expect); assert(e.ptype == at) },
      [||] => assert(false, "no ABC entry"),
    }
  };
  // f16 weights with an i8 ABC (and the reverse) stay OTS-E-RES-001.
  reset();
  slice_test_install_mxm_rows(0, W_F16, 1);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC(W_I8, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  reset();
  slice_test_install_mxm_rows(0, W_I8, 1);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC(W_F16, Streamref(0b100100))) == Diagnostic(RES_NOT_READY))
}

// The sum check of an ABC (microarchitecture.md sec. 18). Weights row 0 = [1, 2], row 1 = [3, 4], row 2 = [5, 6],
// activations [10, 20]: the products are 50, 110, 170. One flipped bit in one lane of the exact sums is found, located and
// corrected (the product is the fault-free one); two lanes are not (OTS-E-ECC-003, no result); a fault of another plane
// stays pending; a lane beyond the installed rows carries no product.
function slice_test_check_mxm_abc(faults : list((range(0, 3), nat, range(0, 95)))) -> ExecutionResult = {
  reset();
  slice_test_install_mxm_rows(0, W_I8, 3);
  var p = mxm[0];
  p.weights[0] = slice_test_first_two_bytes(0x01, 0x02);
  p.weights[1] = slice_test_first_two_bytes(0x03, 0x04);
  p.weights[2] = slice_test_first_two_bytes(0x05, 0x06);
  mxm[0] = p;
  slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0x0A, 0x14), 50);
  slice_test_set_issue_context(106, 50);
  var rest = faults;
  while nonempty_list(rest) do {
    match rest {
      [||] => (),
      (q, lane, bit) :: tl => { model_inject_mxm(q, lane, bit); rest = tl },
    }
  };
  let r = execute_test(MXM_ABC(W_I8, Streamref(0b100100)));
  unit_state_flush_test(50);
  r
}

$[test]
function test_mxm_check() -> unit = {
  // no fault: the ordinary product
  assert(slice_test_check_mxm_abc([||]) == Issued());
  match mxm[0].abc {
    e :: _ => { assert(e.product[0] == 0x00000032 & e.product[1] == 0x0000006E & e.product[2] == 0x000000AA) },
    [||] => assert(false, "no ABC entry"),
  };
  assert(model_mxm_pending() == -1);
  // one bit of one lane (several lanes and bits): corrected, the product is the same
  foreach (k from 0 to 5) {
    let (lane, bit) : (nat, range(0, 95)) = match k {
      0 => (0, 0), 1 => (1, 5), 2 => (2, 40), 3 => (2, 95), 4 => (0, 63), _ => (1, 94) };
    assert(slice_test_check_mxm_abc([| (0, lane, bit) |]) == Issued());
    match mxm[0].abc {
      e :: _ => { assert(e.product[0] == 0x00000032 & e.product[1] == 0x0000006E & e.product[2] == 0x000000AA) },
      [||] => assert(false, "no ABC entry"),
    };
    assert(model_mxm_pending() == -1)
  };
  // two bits of one lane: the lane is wrong by a sum of two powers: still one lane, so corrected
  assert(slice_test_check_mxm_abc([| (0, 2, 3), (0, 2, 9) |]) == Issued());
  // two lanes: uncorrectable, the ABC has no effect
  assert(slice_test_check_mxm_abc([| (0, 0, 3), (0, 2, 9) |]) == Diagnostic(ECC_MXM));
  assert(mxm[0].abc == [||]);
  assert(slice_test_check_mxm_abc([| (0, 0, 4), (0, 1, 4) |]) == Diagnostic(ECC_MXM));
  // a lane beyond the installed rows
  assert(slice_test_check_mxm_abc([| (0, 7, 3) |]) == Issued());
  assert(model_mxm_pending() == -1);
  // a fault of another plane is not consumed
  assert(slice_test_check_mxm_abc([| (1, 0, 3) |]) == Issued());
  assert(model_mxm_pending() == 1);
  assert(diag_code(ECC_MXM) == "OTS-E-ECC-003")
}

// Lanes 0, 1, 2 = a, b, c; every other lane 0.
function slice_test_first_three_bytes(a : bits(8), b : bits(8), c : bits(8)) -> vec = {
  var out : vec = zero_vec;
  out[7 .. 0] = a;
  out[15 .. 8] = b;
  out[23 .. 16] = c;
  out
}

// FP8 ABC: exact products and sum, one rounding to f32; the format is the ABC's, for weights and activations alike.
//   [1.0, 2.0] . [0.5, 3.0] = 6.5 = 0x40D00000 in e4m3 (0x38, 0x40 . 0x30, 0x44) and in e5m2 (0x3C, 0x40 . 0x38, 0x42)
//   exactness: [448, 448, 2^-9] . [448, -448, 1] = 2^-9 (a float32 accumulation would give 0)
// An fp8 ABC needs fp8 weights (an i8 or f16 weight plane gives OTS-E-RES-001) and an fp8 ABC and ACC f32 only.
$[test]
function test_mxm_fp8_abc() -> unit = {
  foreach (k from 0 to 3) {
    reset();
    slice_test_install_mxm_rows(0, W_FP8, 1);
    var p = mxm[0];
    if k == 0 then p.weights[0] = slice_test_first_two_bytes(0x38, 0x40)
    else if k == 1 then p.weights[0] = slice_test_first_two_bytes(0x3C, 0x40)
    else if k == 2 then p.weights[0] = slice_test_first_three_bytes(0x7E, 0x7E, 0x01)
    else p.weights[0] = slice_test_first_three_bytes(0x7B, 0x7B, 0x01);
    mxm[0] = p;
    if k == 0 then slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0x30, 0x44), 50)
    else if k == 1 then slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0x38, 0x42), 50)
    else if k == 2 then slice_test_write_mxm_operand(4, slice_test_first_three_bytes(0x7E, 0xFE, 0x38), 50)
    else slice_test_write_mxm_operand(4, slice_test_first_three_bytes(0x7B, 0xFB, 0x3C), 50);
    slice_test_set_issue_context(106, 50);
    let fmt : bits(1) = if k == 0 | k == 2 then 0b0 else 0b1;
    assert(execute_test(MXM_ABC8(fmt, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => {
        assert(e.ptype == W_FP8);
        assert(e.product[0] == (if k <= 1 then 0x40D00000 else if k == 2 then 0x3B000000 else 0x37800000));
        assert(e.product[1] == 0x00000000)       // a row that is not installed
      },
      [||] => assert(false, "no ABC entry"),
    }
  };
  // fp8 weights need an fp8 ABC and the reverse
  reset();
  slice_test_install_mxm_rows(0, W_FP8, 1);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC(W_I8, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  assert(execute_test(MXM_ABC(W_F16, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  reset();
  slice_test_install_mxm_rows(0, W_I8, 1);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC8(0b0, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  // specials: e5m2 infinity times zero is NaN, an infinity gives an infinity, e4m3 NaN is NaN (the canonical f32 NaN)
  foreach (k from 0 to 2) {
    reset();
    slice_test_install_mxm_rows(0, W_FP8, 1);
    var p = mxm[0];
    p.weights[0] = if k == 0 then slice_test_first_two_bytes(0x7C, 0x38) else if k == 1 then slice_test_first_two_bytes(0x7C, 0x38) else slice_test_first_two_bytes(0x7F, 0x38);
    mxm[0] = p;
    slice_test_write_mxm_operand(4, if k == 0 then slice_test_first_two_bytes(0x00, 0x38) else if k == 1 then slice_test_first_two_bytes(0xBC, 0x38) else slice_test_first_two_bytes(0x38, 0x38), 50);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC8(if k == 2 then 0b0 else 0b1, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => assert(e.product[0] == (if k == 1 then 0xFF800000 else 0x7FC00000)),
      [||] => assert(false, "no ABC entry"),
    }
  };
  // ACC f32 follows an fp8 ABC; ACC i32 does not
  reset();
  var q = mxm[0];
  q.wtype = W_FP8;
  q.rows = 1;
  q.abc = [| struct { dispatch = 20, commit = 40, ptype = W_FP8, product = vector_init(0x3F800000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 } |];
  mxm[0] = q;
  slice_test_set_issue_context(106, 60);
  assert(execute_test(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Diagnostic(OPR_RANGE))
}

// ---------------------------------------------------------------------------------------
// ALU chaining inside the VXM (ISCA20 sec. III-C: "The TSP supports chaining together
// two or more vector ALUs within each lane, allowing multiple ALU operations to be performed
// without committing the intermediate results to main memory"). A VXM instruction reads a
// VXM result committed in the previous cycle at its own station (gap layout.vxm_gap), for an
// operand whose chain bit is set.

// Feed f32 operand planes to streams s..s+3 on row E, visible at the VXM in cycle rc.
function slice_test_write_vxm_operand(s : range(0, 28), p : planes, rc : int) -> unit =
  match read_gap(U_VXM, 0, RowE) {
    Some(g) => if g > 0 & rc > 0 then foreach (b from 0 to 3) { let _ = srf_write(RowE, s + b, g - 1, rc - 1, p[b]); () },
    None() => assert(false, "no VXM read gap"),
  }

function slice_test_first_four_f32(a : bits(32), b : bits(32), c : bits(32), d : bits(32)) -> planes = {
  var xs : vector(vlanes, bits(32)) = vector_init(0x00000000);
  xs[0] = a; xs[1] = b; xs[2] = c; xs[3] = d;
  words_to_planes(xs)
}

function slice_test_broadcast_f32(k : bits(32)) -> planes = { let xs : vector(vlanes, bits(32)) = vector_init(k); words_to_planes(xs) }

function slice_test_tick_range(a : int, b : int) -> unit = foreach (c from a to b) { if c >= 0 then tick(c) }

// f32 lane l (0..3) of streams s..s+3 on row E as seen just past the VXM in cycle t.
function slice_test_read_f32_lane(s : range(0, 28), l : range(0, 3), t : int) -> bits(32) = {
  var x : bits(32) = 0x00000000;
  match read_gap(U_VXM, 0, RowE) {
    Some(g) => if g < 43 then foreach (b from 0 to 3) {
      let (k, p) = srf_read(RowE, s + b, g + 1, t);
      assert(k == SRF_FOUND, "chain result missing");
      x[8 * b + 7 .. 8 * b] = slice(p, 8 * l, 8)
    },
    None() => assert(false, "no VXM gap"),
  };
  x
}

// A GELU-style chain of 9 ALUs on 9 VXM queues, f32, tanh form (each reads at dispatch + d_skew,
// the producer's commit + 1):
//   gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
// Only x and the constants come from upstream; every intermediate stays in the VXM (E:20).
$[test]
function test_vxm_alu_chain_gelu() -> unit = {
  reset();
  let x = slice_test_first_four_f32(0x3f800000, 0xbfc00000, 0x40000000, 0x3e800000);   // 1.0, -1.5, 2.0, 0.25
  let ie = Streamref(0b010100);                       // E:20, the intermediate (destination)
  let ic = Vsrc(ie, 0b1);                             // E:20.chain, the chained operand
  let ex = Vsrc(Streamref(0b000000), 0b0);            // E:0..3 x
  let ek = Vsrc(Streamref(0b000100), 0b0);            // E:4..7 a constant
  let t0 : int = 40;
  let first = VXM_BIN(VMUL, F32, ie, ex, ex, 0b0);
  let first_lat = latency_of_instr(first);
  var prev_read : int = t0 + first_lat.d_skew;
  var prev_commit : int = t0 + first_lat.d_func;
  slice_test_write_vxm_operand(0, x, prev_read);
  slice_test_set_issue_context(88, t0);
  assert(execute_test(first) == Issued());            // x^2
  foreach (k from 1 to 8) {
    let instr : instruction = match k {
      1 => VXM_BIN(VMUL, F32, ie, ic, ex, 0b0),       // x^3
      2 => VXM_BIN(VMUL, F32, ie, ic, ek, 0b0),       // * 0.044715
      3 => VXM_BIN(VADD, F32, ie, ic, ex, 0b0),       // + x
      4 => VXM_BIN(VMUL, F32, ie, ic, ek, 0b0),       // * sqrt(2/pi)
      5 => VXM_UN(VTANH, F32, ie, ic, 0b0),
      6 => VXM_BIN(VADD, F32, ie, ic, ek, 0b0),       // + 1
      7 => VXM_BIN(VMUL, F32, ie, ic, ex, 0b0),       // * x
      _ => VXM_BIN(VMUL, F32, Streamref(0b011000), ic, ek, 0b0), // * 0.5 -> E:24
    };
    let lat = latency_of_instr(instr);
    let rc : int = prev_commit + 1;
    let t : int = rc - lat.d_skew;
    slice_test_tick_range(prev_read, rc - 1);
    match k {
      1 => slice_test_write_vxm_operand(0, x, rc),
      2 => slice_test_write_vxm_operand(4, slice_test_broadcast_f32(0x3d372713), rc),
      3 => slice_test_write_vxm_operand(0, x, rc),
      4 => slice_test_write_vxm_operand(4, slice_test_broadcast_f32(0x3f4c422a), rc),
      5 => (),
      6 => slice_test_write_vxm_operand(4, slice_test_broadcast_f32(0x3f800000), rc),
      7 => slice_test_write_vxm_operand(0, x, rc),
      _ => slice_test_write_vxm_operand(4, slice_test_broadcast_f32(0x3f000000), rc),
    };
    slice_test_set_issue_context(88 + k, t);
    assert(execute_test(instr) == Issued());
    prev_read = rc;
    prev_commit = t + lat.d_func
  };
  slice_test_tick_range(prev_read, prev_commit);
  let seen : int = prev_commit + 1;
  assert(slice_test_read_f32_lane(24, 0, seen) == 0x3f57585c);         // gelu(1.0)
  assert(slice_test_read_f32_lane(24, 1, seen) == 0xbdcdad6c);         // gelu(-1.5)
  assert(slice_test_read_f32_lane(24, 2, seen) == 0x3ffa3042);         // gelu(2.0)
  assert(slice_test_read_f32_lane(24, 3, seen) == 0x3e19447f)          // gelu(0.25)
}

// The chained operand exists only in the cycle after the producer's commit: one cycle early
// or late is OTS-E-TIM-001 (a chained read sees nothing but the station's own writes).
$[test]
function test_vxm_alu_chain_timing() -> unit = {
  let producer_lat = entry_latency_typed("Neg", "i8", T_VXM);
  let lat = entry_latency_typed("Add", "i8", T_VXM);
  let step : int = producer_lat.d_func - lat.d_skew + 1;
  let ie = Streamref(0b001000);                       // E:8 (i8)
  foreach (dt from 0 to 2) {
    reset();
    var xs : vector(vlanes, bits(32)) = vector_init(0x00000000);
    xs[0] = 0x00000005;
    let p = words_to_planes(xs);
    match read_gap(U_VXM, 0, RowE) {
      Some(g) => if g > 0 then { let _ = srf_write(RowE, 0, g - 1, 40 + producer_lat.d_skew - 1, p[0]); () },
      None() => (),
    };
    slice_test_set_issue_context(88, 40);
    assert(execute_test(VXM_UN(VNEG, I8, ie, Vsrc(Streamref(0b000000), 0b0), 0b0)) == Issued());
    let t : int = 40 + step + dt - 1;
    slice_test_tick_range(40, t + lat.d_skew - 1);
    slice_test_set_issue_context(89, t);
    let r = execute_test(VXM_BIN(VADD, I8, Streamref(0b001100), Vsrc(ie, 0b1), Vsrc(ie, 0b1), 0b0));
    if dt == 1 then {
      assert(r == Issued());
      slice_test_tick_range(t + lat.d_skew, t + lat.d_func);
      match read_gap(U_VXM, 0, RowE) {
        Some(g) => if g < 43 then {
          let (k, v8) = srf_read(RowE, 12, g + 1, t + lat.d_func + 1);
          assert(k == SRF_FOUND & slice(v8, 0, 8) == 0xF6)     // -5 + -5
        },
        None() => (),
      }
    } else assert(r == Diagnostic(TIM_NO_VALUE))
  }
}

// A VXM operand WITHOUT the chain bit follows the flow rule. A VXM result written
// at the station in cycle t - 1 does not replace the upstream value of the same stream
// (ISCA20 Listing 1: z = g.add(x_strm, y_strm, stream='S_0') with x on S_0), and with no
// value at the reader in that cycle the read is OTS-E-TIM-001 (the station write is not seen).
$[test]
function test_vxm_unchained_read_ignores_station() -> unit = {
  let producer_lat = entry_latency_typed("Neg", "i8", T_VXM);
  let lat = entry_latency_typed("Add", "i8", T_VXM);
  let step : int = producer_lat.d_func - lat.d_skew + 1;
  let e0 = Streamref(0b000000);
  reset();
  // producer: neg.i8 E:0, E:0 at 40 with x = 5 fed on E:0
  slice_test_write_vxm_operand(0, one_plane(sail_zero_extend(0x05, 8 * vlanes)), 40 + producer_lat.d_skew);
  slice_test_set_issue_context(88, 40);
  assert(execute_test(VXM_UN(VNEG, I8, e0, Vsrc(e0, 0b0), 0b0)) == Issued());
  let t : int = 40 + step;
  slice_test_tick_range(40, t + lat.d_skew - 1);
  // consumer one cycle after the commit, E:0 fed with 7 from upstream: the unchained read is 7
  slice_test_write_vxm_operand(0, one_plane(sail_zero_extend(0x07, 8 * vlanes)), t + lat.d_skew);
  slice_test_set_issue_context(89, t);
  assert(execute_test(VXM_BIN(VADD, I8, Streamref(0b001100), Vsrc(e0, 0b0), Vsrc(e0, 0b0), 0b0)) == Issued());
  slice_test_tick_range(t + lat.d_skew, t + lat.d_func);
  match read_gap(U_VXM, 0, RowE) {
    Some(g) => if g < 43 then {
      let (k, v8) = srf_read(RowE, 12, g + 1, t + lat.d_func + 1);
      assert(k == SRF_FOUND & slice(v8, 0, 8) == 0x0E)      // 7 + 7, not -5 + -5
    },
    None() => (),
  };
  // the same consumer with no value fed for its cycle: TIM-001 for the plain read, the chained read finds -5
  reset();
  slice_test_write_vxm_operand(0, one_plane(sail_zero_extend(0x05, 8 * vlanes)), 40 + producer_lat.d_skew);
  slice_test_set_issue_context(88, 40);
  assert(execute_test(VXM_UN(VNEG, I8, e0, Vsrc(e0, 0b0), 0b0)) == Issued());
  slice_test_tick_range(40, t + lat.d_skew - 1);
  slice_test_set_issue_context(89, t);
  assert(execute_test(VXM_BIN(VADD, I8, Streamref(0b001100), Vsrc(e0, 0b0), Vsrc(e0, 0b0), 0b0)) == Diagnostic(TIM_NO_VALUE));
  reset();
  slice_test_write_vxm_operand(0, one_plane(sail_zero_extend(0x05, 8 * vlanes)), 40 + producer_lat.d_skew);
  slice_test_set_issue_context(88, 40);
  assert(execute_test(VXM_UN(VNEG, I8, e0, Vsrc(e0, 0b0), 0b0)) == Issued());
  slice_test_tick_range(40, t + lat.d_skew - 1);
  slice_test_set_issue_context(89, t);
  assert(execute_test(VXM_BIN(VADD, I8, Streamref(0b001100), Vsrc(e0, 0b1), Vsrc(e0, 0b1), 0b0)) == Issued())
}

// The chain bits are bits 21, 20, 19 of the VXM word (first, second, third source); a chain
// bit for an operand the instruction does not have does not decode; text ".chain".
$[test]
function test_vxm_chain_encoding() -> unit = {
  let i = VXM_SELECT(I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b1), Vsrc(Streamref(0b000100), 0b0), Vsrc(Streamref(0b100000), 0b1), 0b0);
  let w = encdec64(i);
  assert(w[21 .. 19] == 0b101 & w[18 .. 0] == sail_zeros(19));
  assert(encdec64(w) == i);
  assert(assembly(i) == "select.i8 E:8, E:0.chain, E:4, W:0.chain");
  let n = encdec64(VXM_UN(VNEG, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b1), 0b0));
  assert(n[21 .. 19] == 0b100);
  assert(match encdec64([n with 20 = bitone]) { ILLEGAL64(_) => true, _ => false });
  assert(match encdec64([n with 16 = bitone]) { ILLEGAL64(_) => true, _ => false });
  // bit 17 is the type-extension bit: with the extension it makes the type code 16 (neg.bf16), without it the word is illegal
  if ext_bf16 then assert(match encdec64([n with 17 = bitone]) { ILLEGAL64(_) => false, _ => true })
  else assert(match encdec64([n with 17 = bitone]) { ILLEGAL64(_) => true, _ => false })
}

// Bit 18 makes every operand of a same-width VXM operation a
// 4-stream group (SG4, ISCA20 sec. III-C) and the ALU performs 4 / width independent
// operations, one per byte (i8) or halfword (i16, f16) of the 32-bit operand: "four INT8
// operations per cycle" per ALU (ScaledML 2020). Text: ".sg4" after the type.
$[test]
function test_vxm_sg4() -> unit = {
  let i = VXM_BIN(VMAX, I8, Streamref(0b100000), Vsrc(Streamref(0b100000), 0b0), Vsrc(Streamref(0b100100), 0b0), 0b1);
  let w = encdec64(i);
  assert(w[18] == bitone & w[17 .. 0] == sail_zeros(18));
  assert(encdec64(w) == i);
  assert(assembly(i) == "max.i8.sg4 W:0, W:0, W:4");
  assert(assembly(VXM_CMP(CMP_LT, I16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b1), 0b1))
         == "cmp.lt.i16.sg4 E:8, E:0, E:4.chain");
  // Convert and Reduce change or fold widths: bit 18 stays reserved there.
  let cv = encdec64(VXM_CVT(I8, F32, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
  assert(match encdec64([cv with 18 = bitone]) { ILLEGAL64(_) => true, _ => false });
  let rd = encdec64(VXM_REDUCE(RED_ADD, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
  assert(match encdec64([rd with 18 = bitone]) { ILLEGAL64(_) => true, _ => false });
  // Execution: lane 0 of a = 0x7f0180ff, b = 0x80027f00 (bytes, low first: a = -1, -128, 1, 127;
  // b = 0, 127, 2, -128).
  let a = slice_test_first_four_f32(0x7f0180ff, 0x00000000, 0x00000000, 0x00000000);
  let b = slice_test_first_four_f32(0x80027f00, 0x00000000, 0x00000000, 0x00000000);
  let ea = Vsrc(Streamref(0b000000), 0b0);
  let eb = Vsrc(Streamref(0b000100), 0b0);
  let ed = Streamref(0b001000);
  let t0 : int = 40;
  foreach (k from 0 to 3) {
    reset();
    let i : instruction = match k {
      0 => VXM_BIN(VMAX, I8, ed, ea, eb, 0b1),
      1 => VXM_BIN(VMAX, I16, ed, ea, eb, 0b1),
      2 => VXM_BIN(VMAX, I32, ed, ea, eb, 0b1),
      _ => VXM_CMP(CMP_LT, I8, ed, ea, eb, 0b1),
    };
    let lat = latency_of_instr(i);      // a profile may time an element type of an entry on its own row
    let seen : int = t0 + lat.d_func + 1;
    slice_test_write_vxm_operand(0, a, t0 + lat.d_skew);
    slice_test_write_vxm_operand(4, b, t0 + lat.d_skew);
    slice_test_set_issue_context(88, t0);
    assert(execute_test(i) == Issued());
    slice_test_tick_range(t0, t0 + lat.d_func);
    let want : bits(32) = match k { 0 => 0x7f027f00, 1 => 0x7f017f00, 2 => 0x7f0180ff, _ => 0x00ffffff };
    assert(slice_test_read_f32_lane(8, 0, seen) == want)
  };
  // A group operand must be SG4-aligned (OTS-E-OPR-002).
  reset();
  slice_test_set_issue_context(88, t0);
  assert(execute_test(VXM_BIN(VADD, I8, ed, Vsrc(Streamref(0b000001), 0b0), eb, 0b1)) == Diagnostic(OPR_ALIGN))
}

// An instruction dispatched at t reads its operands at t + d_skew.
$[test]
function test_skew_reads_after_dispatch() -> unit = {
  let lat = latency_of_instr(VXM_UN(VNEG, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), 0b0));
  assert(lat.d_skew > 0, "the profile's VXM d_skew is positive");
  let x = one_plane(sail_zero_extend(0x05, 8 * vlanes));
  let i = VXM_UN(VNEG, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), 0b0);
  // The operand is visible at the VXM in cycle 40 + d_skew: the read finds it.
  reset();
  slice_test_write_vxm_operand(0, x, 40 + lat.d_skew);
  slice_test_set_issue_context(88, 40);
  assert(execute_test(i) == Issued());
  // Visible only in cycle 40 - d_skew (the opposite sign): nothing to read.
  reset();
  slice_test_write_vxm_operand(0, x, 40 - lat.d_skew);
  slice_test_set_issue_context(88, 40);
  assert(execute_test(i) == Diagnostic(TIM_NO_VALUE))
}

// The cycle driver executes an element dispatched at t in cycle t + d_skew, not before: a value
// that reaches the VXM in that cycle, committed after the dispatch, is the one read.
$[test]
function test_skew_execute_stage() -> unit = {
  let i = VXM_UN(VNEG, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), 0b0);
  let lat = latency_of_instr(i);
  reset();
  run_element_test(88, i, 40);
  assert(not(model_idle()), "the element waits in the execute ring");
  assert(model_next_event(40) == 40 + lat.d_skew);
  slice_test_write_vxm_operand(0, one_plane(sail_zero_extend(0x05, 8 * vlanes)), 40 + lat.d_skew);
  slice_test_tick_range(41, 40 + lat.d_func);
  match read_gap(U_VXM, 0, RowE) {
    Some(g) => if g < 43 then {
      let (k, v8) = srf_read(RowE, 8, g + 1, 40 + lat.d_func + 1);
      assert(k == SRF_FOUND & slice(v8, 0, 8) == 0xFB)      // -5
    },
    None() => (),
  };
  assert(model_idle())
}

// The timing table has one row per catalogue entry. entry_of names the
// entry, the catalogue table agrees with timing_of, and an override replaces the class's row.
$[test]
function test_timing_entry_table() -> unit = {
  assert(timing_config_error() == "");
  assert(entry_of(SXM_SHIFT(SHIFT_DOWN, 0b000010000, Streamref(0b100000), Streamref(0b000101))) == "Shift");
  assert(entry_of(SXM_MAP(MAP_DISTRIBUTE, 0x0, Streamref(0b100000), Streamref(0b100001))) == "Distribute");
  assert(entry_of(VXM_BIN(VSUB_SAT, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0)) == "Sub_sat");
  assert(entry_of(VXM_CONV(CONV_F16_I8, F16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0))) == "Conv_F16_I8");
  assert(entry_of(MEM_DIRECT(MEM_WRITE, sram_addr_of(1), Streamref(0b000000), 0b0)) == "Write");
  assert(entry_of(MXM_IWS(W_I8, Streamref(0b000000))) == "IWS");
  assert(entry_of(C2C_RTDESKEW(0x0001)) == "RuntimeDeskew");
  assert(entry_latency("Shift", T_SXM_CROSS) == latency_of(T_SXM_CROSS));
  assert(entry_row("Add") == latency_of(T_VXM));
  assert(timing_class_name(timing_of(SXM_MAP(MAP_PERMUTE, 0x0, Streamref(0b100000), Streamref(0b000101)))) == "SXM_CROSS")
}

// ---------------------------------------------------------------------------------------
// The MXM structure keys (mxm.planes_per_hemisphere, mxm.acc_depth, mxm.f16_rows, mxm.fp8_rate). Profile-generic: every expected
// value comes from the configuration.

// The queues of an absent plane do not exist, and the east plane follows the planes of the west hemisphere.
$[test]
function test_mxm_planes_per_hemisphere() -> unit = {
  foreach (q from 104 to 115) {
    let k : int = if q <= 109 then (if q <= 106 then 0 else 1) else (if q <= 112 then 0 else 1);
    assert(queue_present(q) == (k < mxm_planes));
    if k < mxm_planes then {
      let want : int = if q <= 109 then k else mxm_planes + k;
      assert(issuer_of(q).plane == want, "plane of a queue")
    }
  };
  // an instruction of an absent plane's queue is OTS-E-QUE-002 at dispatch (tick); no plane exists beyond 2 * mxm_planes
  assert(2 * mxm_planes <= 4)
}

// The ACC index is bit 12 (its low bit) and bits 11..8: the last legal index round-trips, an index at the depth does not decode.
$[test]
function test_mxm_acc_depth() -> unit = {
  let base = encdec32(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), 0b00000));
  let a : bits(5) = get_slice_int(5, mxm_acc_depth - 1, 0);
  let w = encdec32(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), a));
  assert(w[12 .. 12] == [a[0]] & w[11 .. 8] == a[4 .. 1], "the index is bit 12 and bits 11..8");
  assert(encdec32(w) == MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), a));
  if mxm_acc_depth < 32 then {
    let b : bits(5) = get_slice_int(5, mxm_acc_depth, 0);
    assert(slice_test_illegal_word32([base with 12 .. 12 = [b[0]], 11 .. 8 = b[4 .. 1]]), "an index at the depth does not decode")
  };
  // the accumulators are independent parts: the deepest one is staged and applied alone
  reset();
  var p = mxm[0];
  p.rows = 1;
  p.wtype = W_I8;
  p.abc = [| struct { dispatch = 20, commit = 40, ptype = W_I8, product = vector_init(0x00000007), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 } |];
  mxm[0] = p;
  slice_test_set_issue_context(106, 60);
  assert(execute_test(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), a)) == Issued());
  unit_state_flush_test(60);
  assert(mxm[0].acc[mxm_acc_depth - 1][0] == 0x00000007 & mxm[0].acc[0][0] == 0x00000000)
}

// Two FP8 ABCs of one plane in one cycle: both are applied with mxm.fp8_rate 2 (and an even accumulator reads the lower
// queue's product, an odd one the higher queue's), and clash (OTS-E-RES-003, neither applies) with rate 1.
$[test]
function test_mxm_fp8_rate() -> unit = {
  reset();
  let e1 : abc_entry = struct { dispatch = 30, commit = 50, ptype = W_FP8, product = vector_init(0x00000001), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 };
  let e2 : abc_entry = { e1 with q = 106, product = vector_init(0x00000002) };
  var s1 = mxm[0];
  s1.abc = [| e1 |];
  var s2 = mxm[0];
  s2.abc = [| e2 |];
  mxm_stage_list = [| struct { q = 106, plane = 0, part = P_ABC, acc_idx = 0, value = s2 },
                      struct { q = 105, plane = 0, part = P_ABC, acc_idx = 0, value = s1 } |];
  unit_state_flush_test(30);
  match mxm[0].abc {
    a :: b :: [||] => {
      assert(mxm_fp8_rate == 2);
      assert(a.q == 105 & b.q == 106, "the second applied entry is in front")
    },
    [||] => assert(mxm_fp8_rate == 1, "a clash applies neither"),
    _ => assert(false, "one or three entries"),
  };
  if mxm_fp8_rate == 2 then {
    var p = mxm[0];
    p.rows = 1;
    p.wtype = W_FP8;
    mxm[0] = p;
    slice_test_set_issue_context(106, 60);
    assert(execute_test(MXM_ACC(A_F32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Issued());
    slice_test_set_issue_context(104, 61);
    assert(execute_test(MXM_ACC(A_F32, ACC_REPLACE, Streamref(0b010000), 0b00001)) == Issued());
    unit_state_flush_test(61);
    assert(mxm[0].acc[0][0] == 0x00000001 & mxm[0].acc[1][0] == 0x00000002)
  }
}

// An ABC accepted in the cycle an FP8 pair commits does not cost the pair a product: the ACC of the next cycle, before the new ABC
// commits, still takes the lower queue's product on an even accumulator and the higher queue's on an odd one.
$[test]
function test_mxm_fp8_pair_survives_next_abc() -> unit = {
  reset();
  if mxm_fp8_rate == 2 then {
    slice_test_install_mxm_rows(0, W_FP8, 1);
    let e1 : abc_entry = struct { dispatch = 30, commit = 50, ptype = W_FP8, product = vector_init(0x00000001), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 };
    let e2 : abc_entry = { e1 with q = 106, product = vector_init(0x00000002) };
    var p = mxm[0];
    p.abc = [| e2, e1 |];
    mxm[0] = p;
    slice_test_write_mxm_operand(4, zero_vec, 50);
    slice_test_set_issue_context(104, 50);
    assert(execute_test(MXM_ABC8(0b0, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      a :: b :: c :: [||] => assert(a.q == 104 & b.q == 106 & c.q == 105, "the new entry in front of both products of the pair"),
      _ => assert(false, "the pair and the new entry"),
    };
    slice_test_set_issue_context(106, 51);
    assert(execute_test(MXM_ACC(A_F32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Issued());
    slice_test_set_issue_context(105, 51);
    assert(execute_test(MXM_ACC(A_F32, ACC_REPLACE, Streamref(0b010000), 0b00001)) == Issued());
    unit_state_flush_test(51);
    assert(mxm[0].acc[0][0] == 0x00000001 & mxm[0].acc[1][0] == 0x00000002, "even: lower queue, odd: higher queue")
  }
}

// The f16 capacity: an f16 buffer takes 2 * mxm.f16_rows stream values, an i8 buffer mxm.rows (the weight storage holds the larger).
$[test]
function test_mxm_f16_capacity() -> unit = {
  assert(mxm_rows <= 2 * mxm_f16_rows & mxm_f16_rows <= mxm_rows);
  assert(mxm_wvals == 2 * mxm_f16_rows & mxm_wvals >= mxm_rows);
  assert(timing_config_error() == "")
}

// A timing row may be keyed by an element type ("Max.i8"): timing_type_of names the type of a VXM
// instruction, entry_takes_type says which pairs of entry and type exist, and a profile without such rows
// gives every type the entry's row.
$[test]
function test_timing_type_rows() -> unit = {
  let d = Streamref(0b001000);
  let a = Vsrc(Streamref(0b000000), 0b0);
  let b = Vsrc(Streamref(0b000100), 0b0);
  assert(timing_type_of(VXM_BIN(VMAX, I8, d, a, b, 0b0)) == "i8");
  assert(timing_type_of(VXM_BIN(VMAX, F32, d, a, b, 0b0)) == "f32");
  assert(timing_type_of(VXM_UN(VEXP, F16, d, a, 0b0)) == "f16");
  assert(timing_type_of(VXM_CONV(CONV_F16_I8, F16, d, a)) == "f16");
  assert(timing_type_of(VXM_CVT(U8, F32, d, a)) == "u8");
  assert(timing_type_of(VXM_REDUCE(RED_MAX, E4M3, d, a)) == "e4m3");
  assert(timing_type_of(MEM_DIRECT(MEM_WRITE, sram_addr_of(1), Streamref(0b000000), 0b0)) == "");
  assert(timing_type_of(C2C_RTDESKEW(0x0001)) == "");
  assert(isa_entry_takes_type("Max", "i8") & isa_entry_takes_type("Max", "e5m2"));
  assert(not(isa_entry_takes_type("Max", "b8")) & not(isa_entry_takes_type("Add_sat", "f32")));
  assert(isa_entry_takes_type("Exp", "f16") & not(isa_entry_takes_type("Exp", "i32")));
  assert(isa_entry_takes_type("Conv_F16_I8", "f16") & not(isa_entry_takes_type("Conv_F16_I8", "i8")));
  assert(isa_entry_takes_type("Cmp", "b32") & isa_entry_takes_type("Convert", "b8") & not(isa_entry_takes_type("Reduce", "b8")));
  assert(not(isa_entry_takes_type("Read", "i8")) & not(isa_entry_takes_type("Max", "q9")) & not(isa_entry_takes_type("Nothing", "i8")));
  // the row of an instruction is its typed row, else its entry's row; no profile has a row for Max at e5m2
  assert(latency_of_instr(VXM_BIN(VMAX, I8, d, a, b, 0b0)) == entry_latency_typed("Max", "i8", T_VXM));
  assert(latency_of_instr(VXM_BIN(VMAX, E5M2, d, a, b, 0b0)) == entry_row("Max"));
  assert(entry_latency_typed("Read", "", T_MEM) == entry_row("Read"));
  // Permute takes its catalogue class when the profile has no bare-entry override.
  assert(entry_latency_typed("Permute", "", T_SXM_CROSS) == entry_row("Permute"));
  // Probe the helper's bare-key precedence with a deliberately different fallback class.
  // This is not Permute's instruction class. On ref/mini its override is 20, CU is 0:
  // m_t1's erroneous "Permute." lookup must still be killed without assuming every profile has an override.
  assert(entry_latency_typed("Permute", "", T_CU) == entry_latency("Permute", T_CU))
}

