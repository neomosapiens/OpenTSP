// SPDX-License-Identifier: Apache-2.0
// Independent literal-format bit mappings for every canonical finite output.
// The call is the actual production fround using the native nested ilog2.

$property
function p_pack_normal_f32(off : bits(23), ex : bits(9), neg : bool) -> bool = {
 let ee = signed(ex);
 let fr = unsigned(off);
 if ee < -126 | ee > 127 then true else {
  let actual = fround(FMT32, neg, 8388608 + fr, ee - 23);
  let expected = (if neg then 2147483648 else 0) + (ee + 127) * 8388608 + fr;
  actual == get_slice_int(32, expected, 0)
 }
}

$property
function p_pack_subnormal_f32(fr : bits(23), neg : bool) -> bool = {
 let actual = fround(FMT32, neg, unsigned(fr), -149);
 actual == get_slice_int(32, (if neg then 2147483648 else 0) + unsigned(fr), 0)
}

$property
function p_pack_normal_f16(off : bits(10), ex : bits(6), neg : bool) -> bool = {
 let ee = signed(ex);
 let fr = unsigned(off);
 if ee < -14 | ee > 15 then true else {
  let actual = fround(FMT16, neg, 1024 + fr, ee - 10);
  let expected = (if neg then 32768 else 0) + (ee + 15) * 1024 + fr;
  actual == get_slice_int(32, expected, 0)
 }
}

$property
function p_pack_subnormal_f16(fr : bits(10), neg : bool) -> bool = {
 let actual = fround(FMT16, neg, unsigned(fr), -24);
 actual == get_slice_int(32, (if neg then 32768 else 0) + unsigned(fr), 0)
}

$property
function p_pack_normal_bf16(off : bits(7), ex : bits(9), neg : bool) -> bool = {
 let ee = signed(ex);
 let fr = unsigned(off);
 if ee < -126 | ee > 127 then true else {
  let actual = fround(FMT_BF16, neg, 128 + fr, ee - 7);
  let expected = (if neg then 32768 else 0) + (ee + 127) * 128 + fr;
  actual == get_slice_int(32, expected, 0)
 }
}

$property
function p_pack_subnormal_bf16(fr : bits(7), neg : bool) -> bool = {
 let actual = fround(FMT_BF16, neg, unsigned(fr), -133);
 actual == get_slice_int(32, (if neg then 32768 else 0) + unsigned(fr), 0)
}

$property
function p_pack_normal_e4m3(off : bits(3), ex : bits(5), neg : bool) -> bool = {
 let ee = signed(ex);
 let fr = unsigned(off);
 if ee < -6 | ee > 8 | (ee == 8 & fr == 7) then true else {
  let actual = fround(FMT_E4M3, neg, 8 + fr, ee - 3);
  let expected = (if neg then 128 else 0) + (ee + 7) * 8 + fr;
  actual == get_slice_int(32, expected, 0)
 }
}

$property
function p_pack_subnormal_e4m3(fr : bits(3), neg : bool) -> bool = {
 let actual = fround(FMT_E4M3, neg, unsigned(fr), -9);
 actual == get_slice_int(32, (if neg then 128 else 0) + unsigned(fr), 0)
}

$property
function p_pack_normal_e5m2(off : bits(2), ex : bits(6), neg : bool) -> bool = {
 let ee = signed(ex);
 let fr = unsigned(off);
 if ee < -14 | ee > 15 then true else {
  let actual = fround(FMT_E5M2, neg, 4 + fr, ee - 2);
  let expected = (if neg then 128 else 0) + (ee + 15) * 4 + fr;
  actual == get_slice_int(32, expected, 0)
 }
}

$property
function p_pack_subnormal_e5m2(fr : bits(2), neg : bool) -> bool = {
 let actual = fround(FMT_E5M2, neg, unsigned(fr), -16);
 actual == get_slice_int(32, (if neg then 128 else 0) + unsigned(fr), 0)
}
