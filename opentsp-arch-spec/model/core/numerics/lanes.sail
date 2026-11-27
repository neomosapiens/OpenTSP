// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Per-lane VXM kernels for i8/f16/i32/f32 elements.

// VXM lane kernels (opentsp-arch-spec/docs/microarchitecture.md sec. 5).
// 
// Element layout: an element of width w bytes at lane l is stored as byte b
// (little-endian) in stream base+b, lane l. A VXM instruction therefore reads
// w streams per operand and writes w streams per result; every lane computes
// independently. Kernels below work on one lane's element value (bits(32),
// zero-extended for narrower types).

function wrap_s(v : int, nbits : int) -> int = {
  let m = pow2i(nbits);
  let u = tmod_int(v, m);
  let u2 = if u < 0 then u + m else u;
  if u2 >= pow2i(nbits - 1) then u2 - m else u2
}

function sat_s(v : int, nbits : int) -> int = {
  let hi = pow2i(nbits - 1) - 1;
  let lo = 0 - pow2i(nbits - 1);
  if v > hi then hi else if v < lo then lo else v
}

function bits_of_s(v : int, nbits : int) -> bits(32) =
  get_slice_int(32, if v < 0 then v + pow2i(nbits) else v, 0)

function s_of_bits(x : bits(32), nbits : int) -> int = {
  let u = tmod_int(unsigned(x), pow2i(nbits));
  if u >= pow2i(nbits - 1) then u - pow2i(nbits) else u
}

function nbits_of(t : eltype) -> int = 8 * eltype_width(t)

function fmt_of(t : eltype) -> fmt = match t { F16 => FMT16, BF16 => FMT_BF16, E4M3 => FMT_E4M3, E5M2 => FMT_E5M2, _ => FMT32 }

function is_float(t : eltype) -> bool = match t { F16 => true, BF16 => true, F32 => true, E4M3 => true, E5M2 => true, _ => false }

// An integer element's value: signed or unsigned per its type.
function int_of(t : eltype, x : bits(32)) -> int =
  if is_unsigned(t) then tmod_int(unsigned(x), pow2i(nbits_of(t))) else s_of_bits(x, nbits_of(t))

// Wrap / saturate to the type's range, signed or unsigned.
function wrap_t(t : eltype, v : int) -> int =
  if is_unsigned(t) then emod_int(v, pow2i(nbits_of(t))) else wrap_s(v, nbits_of(t))
function sat_t(t : eltype, v : int) -> int =
  if is_unsigned(t) then (if v < 0 then 0 else if v >= pow2i(nbits_of(t)) then pow2i(nbits_of(t)) - 1 else v)
  else sat_s(v, nbits_of(t))
function bits_t(t : eltype, v : int) -> bits(32) =
  if is_unsigned(t) then get_slice_int(32, v, 0) else bits_of_s(v, nbits_of(t))

function all_ones(t : eltype) -> bits(32) =
  get_slice_int(32, pow2i(nbits_of(t)) - 1, 0)

// Boolean elements (HC34 p.26 BOOL8/16/32): zero is false, any other value is true;
// results are canonical (all ones or zero).
private function bool_bits(t : eltype, b : bool) -> bits(32) = if b then all_ones(t) else 0x00000000

// Binary arithmetic/logical ops, opcodes 0..13.
function k_binary(op : int, t : eltype, x : bits(32), y : bits(32)) -> bits(32) = {
  let n = nbits_of(t);
  if is_bool(t) then {
    let a = (x & all_ones(t)) != 0x00000000;
    let b = (y & all_ones(t)) != 0x00000000;
    if op == 11 then bool_bits(t, a & b)
    else if op == 12 then bool_bits(t, a | b)
    else if op == 13 then bool_bits(t, a != b)
    else 0x00000000
  } else if is_float(t) then {
    let f = fmt_of(t);
    if op == 0 then fp_add(f, x, y)
    else if op == 3 then fp_sub(f, x, y)
    else if op == 6 then fp_mul(f, x, y)
    else if op == 9 then fp_max(f, x, y)
    else if op == 10 then fp_min(f, x, y)
    else 0x00000000
  } else {
    let a = int_of(t, x);
    let b = int_of(t, y);
    let mask = all_ones(t);
    if op == 0 | op == 2 then bits_t(t, wrap_t(t, a + b))
    else if op == 1 then bits_t(t, sat_t(t, a + b))
    else if op == 3 | op == 5 then bits_t(t, wrap_t(t, a - b))
    else if op == 4 then bits_t(t, sat_t(t, a - b))
    else if op == 6 | op == 8 then bits_t(t, wrap_t(t, a * b))
    else if op == 7 then bits_t(t, sat_t(t, a * b))
    else if op == 9 then bits_t(t, if a > b then a else b)
    else if op == 10 then bits_t(t, if a < b then a else b)
    else if op == 11 then (x & y) & mask
    else if op == 12 then (x | y) & mask
    else if op == 13 then xor_vec(x, y) & mask
    else 0x00000000
  }
}

// Unary ops: Not 14, Neg 15, Abs 16, ReLU 20. Transcendentals: see transcendental.sail.
function k_unary(op : int, t : eltype, x : bits(32)) -> bits(32) = {
  let n = nbits_of(t);
  if is_float(t) then {
    let f = fmt_of(t);
    if op == 15 then fp_neg(f, x)
    else if op == 16 then fp_abs(f, x)
    else if op == 20 then {
      let a = fdecode(f, x);
      if a.cls == FNaN then canon_nan(f)
      else if a.neg | a.cls == FZero then zero_bits(f, false)
      else x & all_ones(t)            // no bits above the format width
    } else 0x00000000
  } else if is_bool(t) then {
    // Only Not is defined on a boolean (logical); every other unary kernel gives 0.
    if op == 14 then bool_bits(t, (x & all_ones(t)) == 0x00000000) else 0x00000000
  } else {
    // Signed or unsigned per the element type, like k_binary.
    let a = int_of(t, x);
    if op == 14 then not_vec(x) & all_ones(t)
    else if op == 15 then bits_t(t, wrap_t(t, 0 - a))
    else if op == 16 then bits_t(t, wrap_t(t, if a < 0 then 0 - a else a))
    else if op == 20 then bits_t(t, if a < 0 then 0 else a)
    else 0x00000000
  }
}

// Cmp modes eq ne lt le gt ge = 0..5. Result: all-ones / zero of the element width.
function k_cmp(mode : int, t : eltype, x : bits(32), y : bits(32)) -> bits(32) = {
  let truth : bool =
    if is_bool(t) then {
      let a = (x & all_ones(t)) != 0x00000000;
      let b = (y & all_ones(t)) != 0x00000000;
      if mode == 0 then a == b else a != b
    } else if is_float(t) then {
      let f = fmt_of(t);
      if is_nan(f, x) | is_nan(f, y) then mode == 1
      else {
        let c = fcmp(f, x, y);
        if mode == 0 then c == 0 else if mode == 1 then c != 0
        else if mode == 2 then c < 0 else if mode == 3 then c <= 0
        else if mode == 4 then c > 0 else c >= 0
      }
    } else {
      let a = int_of(t, x);
      let b = int_of(t, y);
      if mode == 0 then a == b else if mode == 1 then a != b
      else if mode == 2 then a < b else if mode == 3 then a <= b
      else if mode == 4 then a > b else a >= b
    };
  if truth then all_ones(t) else 0x00000000
}

function k_nonzero(t : eltype, x : bits(32)) -> bool =
  (x & all_ones(t)) != 0x00000000

// Conversions, opcodes 25..32 (per lane; the element count is preserved).
function conv_src(op : int) -> eltype =
  if op == 25 | op == 27 | op == 29 then I8
  else if op == 26 | op == 32 then F32
  else if op == 28 | op == 31 then F16
  else I32

function conv_dst(op : int) -> eltype =
  if op == 26 | op == 28 | op == 30 then I8
  else if op == 25 | op == 31 then F32
  else if op == 27 | op == 32 then F16
  else I32

function k_convert(op : int, x : bits(32)) -> bits(32) =
  if op == 25 then fp_of_int(FMT32, s_of_bits(x, 8))
  else if op == 26 then bits_of_s(int_of_fp(FMT32, x, 8), 8)
  else if op == 27 then fp_of_int(FMT16, s_of_bits(x, 8))
  else if op == 28 then bits_of_s(int_of_fp(FMT16, x, 8), 8)
  else if op == 29 then bits_of_s(s_of_bits(x, 8), 32)
  else if op == 30 then bits_of_s(sat_s(s_of_bits(x, 32), 8), 8)
  else if op == 31 then fp_convert(FMT16, FMT32, x)
  else if op == 32 then fp_convert(FMT32, FMT16, x)
  else 0x00000000

// Generic conversion between any two element types (VXM Convert, opentsp decision for
// the HC34 p.26 type set): integer -> integer saturates; float -> integer rounds to
// nearest even and saturates (NaN -> 0); integer -> float and float -> float round to
// nearest even; to bool: value != 0 (NaN counts as non-zero); from bool: 1 or 0.
function k_convert_any(src : eltype, dst : eltype, x : bits(32)) -> bits(32) = {
  let truth : bool = if is_float(src) then not(fdecode(fmt_of(src), x).cls == FZero) else (x & all_ones(src)) != 0x00000000;
  if is_bool(dst) then (if truth then all_ones(dst) else 0x00000000)
  else if is_float(dst) then {
    if is_float(src) then fp_convert(fmt_of(src), fmt_of(dst), x)
    else fp_of_int(fmt_of(dst), if is_bool(src) then (if truth then 1 else 0) else int_of(src, x))
  } else {
    let v : int =
      if is_bool(src) then (if truth then 1 else 0)
      else if is_float(src) then {
        let f = fmt_of(src);
        if is_unsigned(dst) then max(int_of_fp(f, x, nbits_of(dst) + 1), 0) else int_of_fp(f, x, nbits_of(dst))
      }
      else int_of(src, x);
    bits_t(dst, sat_t(dst, v))
  }
}
