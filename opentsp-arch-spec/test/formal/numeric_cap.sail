// SPDX-License-Identifier: Apache-2.0
// Native finite-domain certificate entry point. This calls the production value
// functions, including specials and the actual seven-round Ziv loop. It does not
// reimplement the loop or claim interval/rounding correctness.
val numeric_cap_eval : (range(0, 5), range(21, 24), bits(32)) -> bits(33)

function numeric_cap_eval(format, op, x) = {
  if format == 5 then
    match i8_tr_value(op, s_of_bits(x, 8)) {
      Some(r) => 0b0 @ get_slice_int(32, r, 0),
      None() => 0b1 @ 0x00000000,
    }
  else {
    let f = if format == 0 then FMT32 else if format == 1 then FMT16
      else if format == 2 then FMT_BF16 else if format == 3 then FMT_E4M3 else FMT_E5M2;
    match fp_tr_value(op, f, x) {
      Some(r) => 0b0 @ r,
      None() => 0b1 @ 0x00000000,
    }
  }
}
