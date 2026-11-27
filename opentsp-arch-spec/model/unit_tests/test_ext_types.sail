// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Tests of the optional extensions wide_acc (the wide accumulator x128) and block_fp8 (the block-scaled type bfp8).
// A test of an extension that a profile does not declare checks that the decoder rejects it; the tests of the behaviour
// run on the profiles that declare it.

function xt_word(opcode : bits(6), ty : bits(2), s : bits(6), mode : bits(1), tx : bits(1)) -> bits(32) =
  0x3 @ opcode @ ty @ s @ mode @ tx @ 0b000000000000

private function xt_put(base : vec, k : int, b : bits(8)) -> vec =
  if 0 <= k & k < vlanes then { var w = base; w[8 * k + 7 .. 8 * k] = b; w } else base

// The decoder rejects the encodings of an extension the profile does not declare, and decodes them when it does.
$[test]
function test_ext_decode_gate() -> unit = {
  let acc_x = xt_word(0b000011, 0b00, 0b001000, 0b0, 0b0);           // ACC with the accumulator type code 0
  if ext_wide_acc then {
    assert(encdec32(acc_x) == MXM_ACC(A_X128, ACC_REPLACE, Streamref(0b001000), 0b00000));
    assert(encdec32(MXM_ACC(A_X128, ACC_ACCUMULATE, Streamref(0b001100), 0b00001)) == xt_word(0b000011, 0b00, 0b001100, 0b1, 0b1))
  } else assert(slice_test_illegal_word32(acc_x));
  assert(slice_test_illegal_word32(xt_word(0b000011, 0b01, 0b001000, 0b0, 0b0)));     // the type code 1 stays unassigned
  let lw_b = xt_word(0b000000, 0b01, 0b110000, 0b0, 0b1);
  let iw_b = xt_word(0b000001, 0b01, 0b000000, 0b0, 0b1);
  let iws_b = xt_word(0b000001, 0b01, 0b110000, 0b1, 0b1);
  let abc_b = xt_word(0b000010, 0b01, 0b100100, 0b0, 0b1);
  if ext_block_fp8 then {
    assert(encdec32(lw_b) == MXM_LW(W_BFP8, Streamref(0b110000)));
    assert(encdec32(iw_b) == MXM_IW(W_BFP8));
    assert(encdec32(iws_b) == MXM_IWS(W_BFP8, Streamref(0b110000)));
    assert(encdec32(abc_b) == MXM_ABC(W_BFP8, Streamref(0b100100)));
    assert(encdec32(MXM_ABC(W_BFP8, Streamref(0b100100))) == abc_b)
  } else {
    assert(slice_test_illegal_word32(lw_b)); assert(slice_test_illegal_word32(iw_b)); assert(slice_test_illegal_word32(iws_b)); assert(slice_test_illegal_word32(abc_b))
  };
  // the extension bit is zero for every other type; with the other type codes it stays unassigned (type code 0 with the bit
  // set is bf16, which a profile that declares the extension bf16 decodes)
  assert(slice_test_illegal_word32(xt_word(0b000010, 0b00, 0b100100, 0b0, 0b1)) == not(ext_bf16));
  assert(slice_test_illegal_word32(xt_word(0b000010, 0b10, 0b100100, 0b0, 0b1)));
  assert(slice_test_illegal_word32(xt_word(0b000010, 0b11, 0b100100, 0b0, 0b1)));
  assert(slice_test_illegal_word32(xt_word(0b000010, 0b11, 0b100100, 0b1, 0b1)))
}

// The configuration check ties the declaration to the width.
$[test]
function test_ext_config_relation() -> unit = {
  assert(timing_config_error() == "");
  assert(ext_wide_acc == (mxm_acc_bits == 128))
}

// The wide accumulator lane: exact integer addition that wraps at 128 bits, sticky special values, one rounding.
$[test]
function test_wlane_arithmetic() -> unit = {
  let one : wlane = struct { wv = pow2i(48), wsp = 0, wnz = false };
  let tiny : wlane = struct { wv = pow2i(23), wsp = 0, wnz = false };       // 2^-25
  // three additions of 2^-25 to 1.0 stay exact; the read-out rounds once: 1 + 3 * 2^-25 is above 1 + 2^-24, so it rounds up
  let s3 = wlane_add(wlane_add(wlane_add(one, tiny), tiny), tiny);
  assert(s3.wv == pow2i(48) + 3 * pow2i(23));
  assert(wlane_read(s3) == 0x3F800001);
  // each step rounded to f32 loses every 2^-25: the same sum in f32 is 1.0
  var f : bits(32) = 0x3F800000;
  foreach (i from 0 to 2) { f = fp_add(FMT32, f, 0x33000000) };
  assert(f == 0x3F800000);
  // wrap at 128 bits
  let top : wlane = struct { wv = pow2i(127) - 1, wsp = 0, wnz = false };
  assert(wlane_add(top, struct { wv = 1, wsp = 0, wnz = false }).wv == 0 - pow2i(127));
  // special values
  let pinf : wlane = struct { wv = 0, wsp = 1, wnz = false };
  let ninf : wlane = struct { wv = 0, wsp = 2, wnz = false };
  let nan : wlane = struct { wv = 0, wsp = 3, wnz = false };
  assert(wlane_read(wlane_add(pinf, one)) == 0x7F800000);
  assert(wlane_read(wlane_add(one, ninf)) == 0xFF800000);
  assert(wlane_read(wlane_add(pinf, ninf)) == 0x7FC00000);
  assert(wlane_read(wlane_add(nan, one)) == 0x7FC00000);
  assert(wlane_read(wlane_add(wlane_add(pinf, ninf), pinf)) == 0x7FC00000);     // NaN is sticky
  // zero: -0 only when every term is -0
  let nz : wlane = struct { wv = 0, wsp = 0, wnz = true };
  assert(wlane_read(nz) == 0x80000000);
  assert(wlane_read(wlane_add(nz, nz)) == 0x80000000);
  assert(wlane_read(wlane_add(nz, wlane_zero)) == 0x00000000);
  assert(wlane_read(wlane_add(one, struct { wv = 0 - pow2i(48), wsp = 0, wnz = false })) == 0x00000000);
  // read-out: round to nearest even, ties to even
  assert(wlane_read(struct { wv = pow2i(48) + pow2i(24), wsp = 0, wnz = false }) == 0x3F800000);        // a tie rounds to the even 1.0
  assert(wlane_read(struct { wv = pow2i(48) + 3 * pow2i(24), wsp = 0, wnz = false }) == 0x3F800002);    // a tie rounds to the even 1 + 2^-22
  assert(wlane_read(struct { wv = 0 - (pow2i(48) + pow2i(24) + 1), wsp = 0, wnz = false }) == 0xBF800001);
  // the largest value of a lane, 2^127 - 1 units, is 2^79 up to rounding: a 128-bit lane cannot overflow f32
  assert(wlane_read(struct { wv = pow2i(127) - 1, wsp = 0, wnz = false }) == 0x67000000)
}

// ACC x128: the ABC's exact products are added to the wide accumulator and the stream gets the read-out. Three accumulations of
// 2^-25 on 1.0 give 1 + 2^-23 in x128 and 1.0 in f32.
$[test]
function test_acc_x128() -> unit = {
  if ext_wide_acc then {
    reset();
    slice_test_install_mxm_rows(0, W_FP8, 1);
    var p = mxm[0];
    var w : vector(mxm_rows, wlane) = vector_init(wlane_zero);
    w[0] = struct { wv = pow2i(23), wsp = 0, wnz = false };
    var pr : vector(mxm_rows, bits(32)) = vector_init(0x00000000);
    pr[0] = 0x33000000;
    p.abc = [| struct { dispatch = 20, commit = 40, ptype = W_FP8, product = pr, wide = w, poison = sail_zeros(superlanes), q = 105 } |];
    p.acc128[0][0] = struct { wv = pow2i(48), wsp = 0, wnz = false };
    p.acc[0][0] = 0x3F800000;
    mxm[0] = p;
    foreach (i from 0 to 2) {
      slice_test_set_issue_context(106, 60 + 2 * i);
      assert(execute_test(MXM_ACC(A_X128, ACC_ACCUMULATE, Streamref(0b001000), 0b00000)) == Issued());
      unit_state_flush_test(60 + 2 * i);
      slice_test_set_issue_context(106, 61 + 2 * i);
      assert(execute_test(MXM_ACC(A_F32, ACC_ACCUMULATE, Streamref(0b010000), 0b00000)) == Issued());
      unit_state_flush_test(61 + 2 * i)
    };
    assert(mxm[0].acc128[0][0].wv == pow2i(48) + 3 * pow2i(23));
    assert(wlane_read(mxm[0].acc128[0][0]) == 0x3F800001);
    assert(mxm[0].acc[0][0] == 0x3F800000);
    // replace overwrites; the other accumulator is untouched
    slice_test_set_issue_context(106, 70);
    assert(execute_test(MXM_ACC(A_X128, ACC_REPLACE, Streamref(0b001000), 0b00001)) == Issued());
    unit_state_flush_test(70);
    assert(mxm[0].acc128[1][0].wv == pow2i(23));
    assert(mxm[0].acc128[0][0].wv == pow2i(48) + 3 * pow2i(23));
    // an integer product has no wide form
    reset();
    var q = mxm[0];
    q.wtype = W_I8;
    q.rows = 1;
    q.abc = [| struct { dispatch = 20, commit = 40, ptype = W_I8, product = vector_init(0x00000000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 } |];
    mxm[0] = q;
    slice_test_set_issue_context(106, 60);
    assert(execute_test(MXM_ACC(A_X128, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Diagnostic(OPR_RANGE))
  }
}

// A f16 ABC keeps the exact sum for the wide accumulator: [2^-24, 1.0] . [2^-24, 1.0] has the exact sum 1 + 2^-48, which the
// f32 product rounds to 1.0 and the wide lane keeps.
$[test]
function test_abc_keeps_exact_sum() -> unit = {
  if ext_wide_acc then {
    reset();
    slice_test_install_mxm_rows(0, W_F16, 1);
    var p = mxm[0];
    var wlo : vec = zero_vec;
    var whi : vec = zero_vec;
    wlo[7 .. 0] = 0x01; whi[7 .. 0] = 0x00;                // lane 0: the f16 subnormal 2^-24 (0x0001)
    wlo[15 .. 8] = 0x00; whi[15 .. 8] = 0x3C;              // lane 1: 1.0 (0x3C00)
    p.weights[0] = wlo; p.weights[1] = whi;
    mxm[0] = p;
    slice_test_write_mxm_operand(4, wlo, 50);
    slice_test_write_mxm_operand(5, whi, 50);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_F16, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => {
        assert(e.product[0] == 0x3F800000);
        assert(e.wide[0].wv == pow2i(48) + 1);
        assert(e.wide[0].wsp == 0);
        assert(e.wide[1].wv == 0)                           // a row that is not installed
      },
      [||] => assert(false, "no ABC entry"),
    }
  }
}

// bfp8: the weight row is the stream pair (data, scale), the activation too; a block is 16 lanes with the scale of its first
// lane (low four bits, two's complement exponent). Lane 0: 1.0 * 2^3 times 2.0 * 2^-2 = 4; lane 1: 0.5 * 2^3 times 0.5 * 2^-2 =
// 0.5 (the scale of lane 1 is not read); the upper nibble of a scale byte is not read; the sum is 4.5.
$[test]
function test_bfp8_abc() -> unit = {
  if ext_block_fp8 then {
    reset();
    slice_test_install_mxm_rows(0, W_BFP8, 1);
    var p = mxm[0];
    p.weights[0] = slice_test_first_two_bytes(0x38, 0x30);                  // e4m3 1.0, 0.5
    p.weights[1] = slice_test_first_two_bytes(0xA3, 0x05);                  // scales: lane 0 = 3 (upper nibble 0xA ignored), lane 1 ignored
    mxm[0] = p;
    slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0x40, 0x30), 50);                 // 2.0, 0.5
    slice_test_write_mxm_operand(5, slice_test_first_two_bytes(0x0E, 0x07), 50);                 // scales: lane 0 = -2
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_BFP8, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => {
        assert(e.ptype == W_BFP8);
        assert(e.product[0] == 0x40900000)                // 4.5
      },
      [||] => assert(false, "no ABC entry"),
    };
    // the largest e4m3 value 448 with the largest scale 7 on both operands: 448 * 2^7 times 448 * 2^7 = 448^2 * 2^14 = 3288334336
    reset();
    slice_test_install_mxm_rows(0, W_BFP8, 1);
    var q = mxm[0];
    q.weights[0] = slice_test_first_two_bytes(0x7E, 0x00);
    q.weights[1] = slice_test_first_two_bytes(0x07, 0x00);
    mxm[0] = q;
    slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0x7E, 0x00), 50);
    slice_test_write_mxm_operand(5, slice_test_first_two_bytes(0x07, 0x00), 50);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_BFP8, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => assert(e.product[0] == 0x4F440000),       // 448 * 448 * 2^14 = 200704 * 16384 = 3288334336
      [||] => assert(false, "no ABC entry"),
    };
    // e4m3 NaN gives NaN
    reset();
    slice_test_install_mxm_rows(0, W_BFP8, 1);
    var r = mxm[0];
    r.weights[0] = slice_test_first_two_bytes(0x7F, 0x00);
    mxm[0] = r;
    slice_test_write_mxm_operand(4, slice_test_first_two_bytes(0x38, 0x00), 50);
    slice_test_write_mxm_operand(5, zero_vec, 50);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_BFP8, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => assert(e.product[0] == 0x7FC00000),
      [||] => assert(false, "no ABC entry"),
    };
    // the types do not mix: bfp8 weights need a bfp8 ABC and the reverse
    reset();
    slice_test_install_mxm_rows(0, W_BFP8, 1);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_F16, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
    assert(execute_test(MXM_ABC(W_I8, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
    assert(execute_test(MXM_ABC8(0b0, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
    reset();
    slice_test_install_mxm_rows(0, W_F16, 1);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_BFP8, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
    // a pair type takes the plane pair like f16: the odd plane is blocked
    reset();
    slice_test_install_mxm_rows(1, W_BFP8, 1);
    slice_test_set_issue_context(107, 50);
    assert(execute_test(MXM_ABC(W_BFP8, Streamref(0b100100))) == Diagnostic(RES_NOT_READY))
  }
}

// A second block (lane 16 and up, a profile with two or more superlanes): the scale of its first lane applies to it.
$[test]
function test_bfp8_second_block() -> unit = {
  if ext_block_fp8 & vlanes >= 32 then {
    reset();
    slice_test_install_mxm_rows(0, W_BFP8, 1);
    var p = mxm[0];
    let wd = xt_put(zero_vec, 16, 0x38);                   // lane 16: 1.0 with scale -8
    let ws = xt_put(zero_vec, 16, 0x08);
    let xd = xt_put(zero_vec, 16, 0x38);                   // lane 16: 1.0 with scale 7
    let xs = xt_put(zero_vec, 16, 0x07);
    p.weights[0] = wd; p.weights[1] = ws;
    mxm[0] = p;
    slice_test_write_mxm_operand(4, xd, 50);
    slice_test_write_mxm_operand(5, xs, 50);
    slice_test_set_issue_context(106, 50);
    assert(execute_test(MXM_ABC(W_BFP8, Streamref(0b100100))) == Issued());
    unit_state_flush_test(50);
    match mxm[0].abc {
      e :: _ => assert(e.product[0] == 0x3F000000),       // 2^-1
      [||] => assert(false, "no ABC entry"),
    }
  }
}
