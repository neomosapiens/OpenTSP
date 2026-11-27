// Entry points of the ISS numerics differential check (tools/sail_dump.sh).
// This file is NOT part of the model: the script builds it together with core/numerics into a
// private C program (tools/sail_dump_driver.c). It only wraps the model's own functions.

function dmp_ty(t : int) -> eltype = match t {
  0 => I8, 1 => F16, 2 => I32, 3 => F32, 4 => U8, 5 => I16, 6 => U16, 7 => U32,
  8 => B8, 9 => B16, _ => B32,
}

// fp_tr_value(op, fmt, x): bit 32 set means "the Ziv loop did not converge".
val dmp_tr : (range(21, 24), range(16, 32), bits(32)) -> bits(33)
function dmp_tr(op, w, x) =
  match fp_tr_value(op, if w == 16 then FMT16 else FMT32, x) {
    Some(r) => 0b0 @ r,
    None()  => 0b1 @ 0x00000000,
  }

// i8_transcendental(op, x) without the internal error: the saturated i8 result, zero-extended.
val dmp_i8 : (range(21, 24), bits(32)) -> bits(33)
function dmp_i8(op, x) =
  match i8_tr_value(op, s_of_bits(x, 8)) {
    Some(r) => 0b0 @ bits_of_s(sat_s(r, 8), 8),
    None()  => 0b1 @ 0x00000000,
  }

val dmp_bin : (range(0, 31), range(0, 10), bits(32), bits(32)) -> bits(32)
function dmp_bin(op, t, x, y) = k_binary(op, dmp_ty(t), x, y)

// The unary dispatch of vxm_insts.sail `apply`: transcendentals 21..24, else k_unary.
val dmp_un : (range(0, 31), range(0, 10), bits(32)) -> bits(32)
function dmp_un(op, t, x) = {
  let ty = dmp_ty(t);
  if 21 <= op & op <= 24 then
    (if is_float(ty) then fp_transcendental(op, fmt_of(ty), x) else i8_transcendental(op, x))
  else k_unary(op, ty, x)
}

val dmp_cmp : (range(0, 5), range(0, 10), bits(32), bits(32)) -> bits(32)
function dmp_cmp(mode, t, x, y) = k_cmp(mode, dmp_ty(t), x, y)

val dmp_cvt : (range(0, 10), range(0, 10), bits(32)) -> bits(32)
function dmp_cvt(st, dt, x) = k_convert_any(dmp_ty(st), dmp_ty(dt), x)

val dmp_conv : (range(25, 32), bits(32)) -> bits(32)
function dmp_conv(op, x) = k_convert(op, x)

// k_nonzero as 0 / 1
val dmp_anybit : (range(0, 10), bits(32)) -> bits(32)
function dmp_anybit(t, x) = if k_nonzero(dmp_ty(t), x) then 0x00000001 else 0x00000000
