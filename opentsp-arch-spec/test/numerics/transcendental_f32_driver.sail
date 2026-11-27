// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Entry points for the exhaustive f32 transcendental check (tools/check_transcendental_f32.sh).
// This file is NOT part of the model (it is not listed in model/tsp.sail_project); the check
// script builds it together with core/numerics into a private C program. It only wraps the
// model's own functions; no model function is re-implemented except the Ziv loop in
// tf32_level, which reports the working precision at which the model's loop stops and is
// used for statistics only (the compared result always comes from fp_tr_value).

// Result of the model's fp_tr_value(op, FMT32, x): bit 32 set means "the Ziv loop did not
// converge" (a model error), otherwise bits 31..0 are the f32 result.
val tf32_eval : (range(21, 24), bits(32)) -> bits(33)

function tf32_eval(op, x) =
  match fp_tr_value(op, FMT32, x) {
    Some(r) => 0b0 @ r,
    None()  => 0b1 @ 0x00000000,
  }

// Working precision p (64, 128, ..., 4096) at which tr_eval stops for a finite non-zero
// input that reaches tr_eval; 0 if it does not stop. Statistics only.
val tf32_level : (range(21, 24), bits(32)) -> bits(32)

function tf32_level(op, x) = {
  let a = fdecode(FMT32, x);
  var p : int = 64;
  var res : option(bits(32)) = None();
  while is_none(res) & p <= 4096 do {
    let (lo, hi, sh) = tr_interval(op, a, p);
    res = round_iv(FMT32, lo, hi, sh);
    if is_none(res) then p = p * 2
  };
  if is_none(res) then 0x00000000 else get_slice_int(32, p, 0)
}

// Accuracy of the model's enclosure at working precision p: floor(log2(max(|lo|,|hi|))) -
// floor(log2(hi - lo)), that is the enclosure's relative width is about 2^-acc (+-1). Used
// to document the evaluator's real limit (bits lost against p). Statistics only.
val tf32_acc : (range(21, 24), bits(32), range(64, 4096)) -> bits(32)

function tf32_acc(op, x, p) = {
  let a = fdecode(FMT32, x);
  let (lo, hi, _) = tr_interval(op, a, p);
  let w = hi - lo;
  let m = if (0 - lo) > hi then 0 - lo else hi;
  if w <= 0 | m <= 0 then 0x00000000
  else {
    let d = ilog2(m) - ilog2(w);
    get_slice_int(32, if d < 0 then 0 else d, 0)
  }
}
