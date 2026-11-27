// SPDX-License-Identifier: Apache-2.0
// Production decoder bounds needed by unbounded arithmetic composition.
// Every 32-bit carrier, including nonzero high bits for narrower formats.
function finite_domain(f : fmt, x : bits(32), low_e : int, high_e : int, limit : int) -> bool = {
  let d = fdecode(f, x);
  d.cls != FFinite | (d.mag >= 1 & d.mag < limit & d.e >= low_e & d.e <= high_e)
}

$property
function p_f32_finite_domain(x : bits(32)) -> bool = finite_domain(FMT32, x, -149, 104, 16777216)
$property
function p_f16_finite_domain(x : bits(32)) -> bool = finite_domain(FMT16, x, -24, 5, 2048)
$property
function p_bf16_finite_domain(x : bits(32)) -> bool = finite_domain(FMT_BF16, x, -133, 120, 256)
$property
function p_e4m3_finite_domain(x : bits(32)) -> bool = finite_domain(FMT_E4M3, x, -9, 5, 16)
$property
function p_e5m2_finite_domain(x : bits(32)) -> bool = finite_domain(FMT_E5M2, x, -16, 13, 8)

// Independent binary32 field definition with literal IEEE widths/bias, rather
// than using FMT32 to manufacture the expected result.
$property
function p_f32_decode_external(x : bits(32)) -> bool = {
  let d = fdecode(FMT32, x);
  let ex = unsigned(x[30 .. 23]);
  let fr = unsigned(x[22 .. 0]);
  let good = if ex == 255 then (if fr == 0 then d.cls == FInf else d.cls == FNaN)
    else if ex == 0 & fr == 0 then d.cls == FZero
    else d.cls == FFinite & d.mag == fr + (if ex == 0 then 0 else 8388608) & d.e == max(ex, 1) - 150;
  d.neg == (x[31] == bitone) & good &
    (d.cls == FFinite | (d.mag == 0 & d.e == 0))
}

// This suite uses the production NESTED ilog2, without OPENTSP_PROOF_BUILD.
$property
function p_ilog2_24(x : bits(24)) -> bool = {
 let n = unsigned(x);
 let k = ilog2(n);
 n == 0 | (k >= 0 & k <= 23 & pow2i(k) <= n & n < pow2i(k+1))
}
function exp_input_bound(f : fmt, x : bits(32), doubled : bool) -> bool = {
 let a = fdecode(f, x);
 let e = a.e + (if doubled then 1 else 0);
 a.cls != FFinite | a.e + ilog2(a.mag) > (if doubled then 5 else 8) |
   (a.mag >= 1 & e <= 8 & a.mag <= pow2i(9-e))
}

$property
function p_f32_exp_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT32, x, false)

$property
function p_f32_tanh_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT32, x, true)

$property
function p_f16_exp_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT16, x, false)

$property
function p_f16_tanh_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT16, x, true)

$property
function p_bf16_exp_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT_BF16, x, false)

$property
function p_bf16_tanh_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT_BF16, x, true)

$property
function p_e4m3_exp_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT_E4M3, x, false)

$property
function p_e4m3_tanh_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT_E4M3, x, true)

$property
function p_e5m2_exp_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT_E5M2, x, false)

$property
function p_e5m2_tanh_input_bound(x : bits(32)) -> bool = exp_input_bound(FMT_E5M2, x, true)
