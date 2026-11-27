// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests of the profile extension `bf16` (microarchitecture.md sec. 5, "Extension types"): the fifth type bit of the
// VXM word, the third weight-type bit of the MXM word, the decode gate (`extensions`), the bfloat16 kernels and the
// bfloat16 matrix product. The decode assertions hold on every profile: a word that uses the extension decodes exactly
// when the profile declares it (test/ext_bf16 declares it; every other profile does not). The execution assertions
// call `execute` directly and so do not depend on the gate.

// One 16-bit value (low and high byte) in lane 0, every other lane zero: low and high byte planes.
function bf_lane0(lo : bits(8), hi : bits(8)) -> (vec, vec) = {
  var l : vec = zero_vec;
  var h : vec = zero_vec;
  l[7 .. 0] = lo;
  h[7 .. 0] = hi;
  (l, h)
}

// The bf16 landmarks of the format: the top half of a binary32.
$[test]
function test_bf16_format() -> unit = {
  assert(eltype_width(BF16) == 2 & encdec_eltype(BF16) == 0b10000 & eltype_name(BF16) == "bf16");
  assert(is_float(BF16) & not(is_int(BF16)) & not(is_bool(BF16)));
  assert(fdecode(FMT_BF16, 0x00003F80).cls == FFinite & fdecode(FMT_BF16, 0x00003F80).mag == 128 & fdecode(FMT_BF16, 0x00003F80).e == -7);   // 1.0
  assert(fdecode(FMT_BF16, 0x00007F80).cls == FInf & fdecode(FMT_BF16, 0x0000FF80).neg);
  assert(fdecode(FMT_BF16, 0x00007FC1).cls == FNaN);
  assert(fdecode(FMT_BF16, 0x00000001).mag == 1 & fdecode(FMT_BF16, 0x00000001).e == -133);   // the smallest subnormal
  assert(canon_nan(FMT_BF16) == 0x00007FC0 & inf_bits(FMT_BF16, false) == 0x00007F80 & inf_bits(FMT_BF16, true) == 0x0000FF80);
  assert(zero_bits(FMT_BF16, true) == 0x00008000 & zero_bits(FMT_BF16, false) == 0x00000000);
  // conversions from binary32: nearest even, overflow to infinity, canonical NaN
  assert(k_convert_any(F32, BF16, 0x3F800000) == 0x00003F80);    // 1.0
  assert(k_convert_any(F32, BF16, 0x3F808000) == 0x00003F80);    // the tie above 1.0 rounds to even (down)
  assert(k_convert_any(F32, BF16, 0x3F818000) == 0x00003F82);    // the tie above 0x3F81 rounds to even (up)
  assert(k_convert_any(F32, BF16, 0x3F808001) == 0x00003F81);    // just above the tie rounds up
  assert(k_convert_any(F32, BF16, 0x7F7FFFFF) == 0x00007F80);    // the largest binary32 rounds to infinity
  assert(k_convert_any(F32, BF16, 0x7FC00000) == 0x00007FC0);    // NaN is the canonical NaN
  assert(k_convert_any(BF16, F32, 0x00003F80) == 0x3F800000);    // the widening is exact
  assert(k_convert_any(BF16, F32, 0x00000001) == 0x00010000 & k_convert_any(F32, BF16, 0x00010000) == 0x00000001);
  assert(k_convert_any(BF16, F16, 0x00003F80) == 0x00003C00);    // 1.0 in f16
  assert(k_convert_any(I8, BF16, 0x00000003) == 0x00004040 & k_convert_any(BF16, I8, 0x00004040) == 0x00000003);
  // arithmetic: one rounding of the exact result to bf16
  assert(k_binary(0, BF16, 0x00003F80, 0x00003F80) == 0x00004000);   // 1 + 1
  assert(k_binary(6, BF16, 0x00003FC0, 0x00004000) == 0x00004040);   // 1.5 * 2 = 3
  assert(k_binary(0, BF16, 0x00003F80, 0x00003B80) == 0x00003F80);   // 1 + 2^-8 ties to even: 1
  assert(k_binary(0, BF16, 0x00007F7F, 0x00007F7F) == 0x00007F80);   // the largest finite twice overflows
  assert(k_binary(0, BF16, 0x00007FC0, 0x00003F80) == 0x00007FC0);   // NaN propagates as the canonical NaN
  assert(k_unary(15, BF16, 0x00003F80) == 0x0000BF80 & k_unary(16, BF16, 0x0000BF80) == 0x00003F80);
  assert(k_cmp(2, BF16, 0x0000BF80, 0x00003F80) == 0x0000FFFF & k_cmp(1, BF16, 0x00007FC0, 0x00007FC0) == 0x0000FFFF);
  assert(binop_legal(VADD, BF16) & binop_legal(VMUL, BF16) & binop_legal(VMAX, BF16) & not(binop_legal(VAND, BF16)) & not(binop_legal(VADD_SAT, BF16)));
  assert(unop_legal(VNEG, BF16) & unop_legal(VRELU, BF16) & not(unop_legal(VNOT, BF16)) & not(unop_legal(VEXP, BF16)))
}

// The VXM word: type code 16 is bit 17 set with the 2 + 2 type bits zero; it decodes only with the extension; 17..31 never.
$[test]
function test_bf16_vxm_decode() -> unit = {
  let add = encdec64(VXM_BIN(VADD, F16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  let bf = [add with 17 .. 17 = 0b1, 53 .. 52 = 0b00];
  assert(add[17 .. 17] == 0b0);
  assert(slice_test_illegal_word64(bf) == not(ext_bf16));
  if ext_bf16 then {
    assert(encdec64(bf) == VXM_BIN(VADD, BF16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
    assert(encdec64(VXM_BIN(VADD, BF16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0)) == bf)
  };
  // codes 17..31 are reserved on every profile
  assert(slice_test_illegal_word64([bf with 53 .. 52 = 0b01]) & slice_test_illegal_word64([bf with 53 .. 52 = 0b11]));
  assert(slice_test_illegal_word64([bf with 23 .. 22 = 0b01]) & slice_test_illegal_word64([bf with 23 .. 22 = 0b11]));
  // the operations bf16 does not have (a boolean, integer or saturating operation) stay illegal even with the extension
  let and_w = encdec64(VXM_BIN(VAND, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(slice_test_illegal_word64([and_w with 17 .. 17 = 0b1]));
  let sat_w = encdec64(VXM_BIN(VADD_SAT, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0), Vsrc(Streamref(0b000100), 0b0), 0b0));
  assert(slice_test_illegal_word64([sat_w with 17 .. 17 = 0b1]));
  // Convert: the source type has the bit 17, the destination type the bit 16 above its 4-bit code
  let cvt = encdec64(VXM_CVT(F32, I8, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
  let to_bf = [cvt with 27 .. 24 = 0b0000, 16 .. 16 = 0b1];
  assert(slice_test_illegal_word64(to_bf) == not(ext_bf16));
  let from_bf = [cvt with 53 .. 52 = 0b00, 23 .. 22 = 0b00, 17 .. 17 = 0b1, 27 .. 24 = 0b0011];
  assert(slice_test_illegal_word64(from_bf) == not(ext_bf16));
  if ext_bf16 then {
    assert(encdec64(to_bf) == VXM_CVT(F32, BF16, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)));
    assert(encdec64(from_bf) == VXM_CVT(BF16, F32, Streamref(0b001000), Vsrc(Streamref(0b000000), 0b0)))
  };
  assert(slice_test_illegal_word64([cvt with 27 .. 24 = 0b0001, 16 .. 16 = 0b1]))     // destination code 17
}

// The MXM word: the weight type is bit 12 above bits 21..20; bf16 is 100, the codes 101..111 are reserved, and bit 12 of
// ACC is its accumulator select, not a type bit.
$[test]
function test_bf16_mxm_decode() -> unit = {
  let lw = encdec32(MXM_LW(W_F16, Streamref(0b110000)));
  let bf_lw = [lw with 21 .. 20 = 0b00, 12 .. 12 = 0b1];
  assert(slice_test_illegal_word32(bf_lw) == not(ext_bf16));
  if ext_bf16 then {
    assert(encdec32(bf_lw) == MXM_LW(W_BF16, Streamref(0b110000)));
    assert(encdec32(MXM_LW(W_BF16, Streamref(0b110000))) == bf_lw)
  };
  // bit 12 with low type bits 01 is the block format bfp8: legal exactly when the profile declares block_fp8 (test_ext_types)
  assert(slice_test_illegal_word32([lw with 12 .. 12 = 0b1]) == not(ext_block_fp8));
  assert(slice_test_illegal_word32([lw with 21 .. 20 = 0b10, 12 .. 12 = 0b1]) & slice_test_illegal_word32([lw with 21 .. 20 = 0b11, 12 .. 12 = 0b1]));
  let iw = encdec32(MXM_IW(W_I8));
  assert(slice_test_illegal_word32([iw with 12 .. 12 = 0b1]) == not(ext_bf16));
  let iws = encdec32(MXM_IWS(W_I8, Streamref(0b110000)));
  assert(slice_test_illegal_word32([iws with 12 .. 12 = 0b1]) == not(ext_bf16));
  let abc = encdec32(MXM_ABC(W_I8, Streamref(0b100100)));
  assert(slice_test_illegal_word32([abc with 12 .. 12 = 0b1]) == not(ext_bf16));
  if ext_bf16 then assert(encdec32([abc with 12 .. 12 = 0b1]) == MXM_ABC(W_BF16, Streamref(0b100100)));
  let abc8 = encdec32(MXM_ABC8(0b0, Streamref(0b100100)));
  assert(slice_test_illegal_word32([abc8 with 12 .. 12 = 0b1]));                // fp8 has no type-extension bit
  let acc = encdec32(MXM_ACC(A_F32, ACC_ACCUMULATE, Streamref(0b001100), 0b00000));
  assert(not(slice_test_illegal_word32([acc with 12 .. 12 = 0b1])));
  assert(entry_of(MXM_ABC(W_BF16, Streamref(0b100100))) == "ABC" & timing_of(MXM_ABC(W_BF16, Streamref(0b100100))) == T_MXM_ABC)
}

// The names of the gate: only bf16 is an extension of this version, once.
$[test]
function test_bf16_extension_names() -> unit = {
  assert(has_extension("bf16", [| "bf16" |]) & not(has_extension("bf16", [||])) & not(has_extension("fp4", [| "bf16" |])));
  assert(ext_bf16 == has_extension("bf16", extensions));
  assert(timing_config_error() == "")
}

// The exact dot product of a bf16 row and a bf16 activation, rounded once to binary32.
//   [1.5] . [2.0] = 3.0
//   exactness: [2^127, 1, -2^127] . [1, 1, 1] = 1 (a binary32 accumulation gives 0)
//   the smallest subnormal times 1.0 is the binary32 subnormal 2^-133; 2^127 * 2^127 overflows to infinity
$[test]
function test_bf16_dot() -> unit = {
  let (wl, wh) = bf_lane0(0xC0, 0x3F);
  let (xl, xh) = bf_lane0(0x00, 0x40);
  assert(bf16_dot(wl, wh, xl, xh) == 0x40400000);
  assert(bf16_dot(slice_test_first_three_bytes(0x00, 0x80, 0x00), slice_test_first_three_bytes(0x7F, 0x3F, 0xFF), slice_test_first_three_bytes(0x80, 0x80, 0x80), slice_test_first_three_bytes(0x3F, 0x3F, 0x3F)) == 0x3F800000);
  // the products are exact to 2^-266: 1 + 2^-12 * 2^-12 is a tie between 1 and 1 + 2^-23 (it rounds to 1); the product of the smallest
  // subnormal with itself, 2^-266, breaks the tie upward
  let tl = slice_test_first_three_bytes(0x80, 0x80, 0x00);
  let th = slice_test_first_three_bytes(0x3F, 0x39, 0x00);
  assert(bf16_dot(tl, th, tl, th) == 0x3F800000);
  let ul = slice_test_first_three_bytes(0x80, 0x80, 0x01);
  let uh = slice_test_first_three_bytes(0x3F, 0x39, 0x00);
  assert(bf16_dot(ul, uh, ul, uh) == 0x3F800001);
  // a product below 2^-259 is exact too: the pair +(16 * 2^-133)^2 and -(2^-133 * 2^-125) is 2^-258 each, so the tie above still rounds to 1
  let pl = [slice_test_first_three_bytes(0x80, 0x80, 0x10) with 31 .. 24 = 0x01];
  let ph = [slice_test_first_three_bytes(0x3F, 0x39, 0x00) with 31 .. 24 = 0x00];
  let ql = [slice_test_first_three_bytes(0x80, 0x80, 0x10) with 31 .. 24 = 0x00];
  let qh = [slice_test_first_three_bytes(0x3F, 0x39, 0x00) with 31 .. 24 = 0x81];
  assert(bf16_dot(pl, ph, ql, qh) == 0x3F800000);
  let (sl, sh) = bf_lane0(0x01, 0x00);
  let (ol, oh) = bf_lane0(0x80, 0x3F);
  assert(bf16_dot(sl, sh, ol, oh) == 0x00010000);
  let (bl, bh) = bf_lane0(0x00, 0x7F);
  assert(bf16_dot(bl, bh, bl, bh) == 0x7F800000);
  // specials: infinity times zero is NaN, an infinity gives an infinity, a NaN operand is the canonical NaN
  let (il, ih) = bf_lane0(0x80, 0x7F);
  let (zl, zh) = bf_lane0(0x00, 0x00);
  assert(bf16_dot(il, ih, zl, zh) == 0x7FC00000 & bf16_dot(il, ih, ol, oh) == 0x7F800000);
  let (nl, nh) = bf_lane0(0xC1, 0x7F);
  assert(bf16_dot(nl, nh, ol, oh) == 0x7FC00000);
  // the f16 kernel is the same function at its own exponent range
  let (hl, hh) = bf_lane0(0x00, 0x3C);
  assert(f16_dot(hl, hh, hl, hh) == 0x3F800000)
}

// bf16 ABC: the weights take a plane pair of two byte planes per row; a bf16 ABC needs bf16 weights and the reverse.
$[test]
function test_bf16_abc() -> unit = if mxm_planes == 2 then {      // a bf16 plane needs its pair
  reset();
  slice_test_install_mxm_rows(0, W_BF16, 1);
  var p = mxm[0];
  let (wl, wh) = bf_lane0(0xC0, 0x3F);       // [1.5]
  p.weights[0] = wl;
  p.weights[1] = wh;
  mxm[0] = p;
  let (xl, xh) = bf_lane0(0x00, 0x40);       // [2.0]
  slice_test_write_mxm_operand(4, xl, 50);
  slice_test_write_mxm_operand(5, xh, 50);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC(W_BF16, Streamref(0b100100))) == Issued());
  unit_state_flush_test(50);
  match mxm[0].abc {
    e :: _ => { assert(e.ptype == W_BF16); assert(e.product[0] == 0x40400000); assert(e.product[1] == 0x00000000) },
    [||] => assert(false, "no ABC entry"),
  };
  // type pairing
  reset();
  slice_test_install_mxm_rows(0, W_BF16, 1);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC(W_F16, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  assert(execute_test(MXM_ABC(W_I8, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  assert(execute_test(MXM_ABC8(0b0, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  reset();
  slice_test_install_mxm_rows(0, W_F16, 1);
  slice_test_set_issue_context(106, 50);
  assert(execute_test(MXM_ABC(W_BF16, Streamref(0b100100))) == Diagnostic(RES_NOT_READY));
  // ACC f32 follows a bf16 ABC; ACC i32 does not
  reset();
  var q = mxm[0];
  q.wtype = W_BF16;
  q.rows = 1;
  q.abc = [| struct { dispatch = 20, commit = 40, ptype = W_BF16, product = vector_init(0x3F800000), wide = vector_init(wlane_zero), poison = sail_zeros(superlanes), q = 105 } |];
  mxm[0] = q;
  slice_test_set_issue_context(106, 60);
  assert(execute_test(MXM_ACC(A_I32, ACC_REPLACE, Streamref(0b001000), 0b00000)) == Diagnostic(OPR_RANGE))
}

// bf16 weights use the even plane of a pair; the odd partner is busy meanwhile, as for f16 (the same capacity).
$[test]
function test_bf16_plane_pair() -> unit = if mxm_planes == 2 then {      // a bf16 plane needs its pair
  reset();
  slice_test_set_issue_context(107, 10);                         // west plane 1
  assert(execute_test(MXM_IW(W_BF16)) == Diagnostic(RES_NOT_READY));
  var p = mxm[0];
  p.wtype = W_BF16;
  p.rows = 16;
  mxm[0] = p;
  var q = mxm[1];
  q.buf_rows = 16;
  mxm[1] = q;
  slice_test_set_issue_context(107, 10);
  assert(execute_test(MXM_IW(W_I8)) == Diagnostic(RES_NOT_READY))    // plane 0 holds bf16
}
