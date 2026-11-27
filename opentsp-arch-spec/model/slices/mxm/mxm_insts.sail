// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// MXM instructions LW, IW, ABC, ACC (ISCA 2020 sec. III-D; microarchitecture.md sec. 6).
// Inputs (LW, ABC) arrive at the plane's read gap on its hemisphere's inward row; ACC
// results leave at its write gap on the other row (core/layout.sail).
//
// 32-bit layout: slice 31..28 = 3, opcode 27..22, type 21..20, stream 19..14, mode 13,
// type extension 12, reserved 11..0. Fields an instruction does not use must be zero. The type extension bit selects the types
// beyond the first four (bf16, extension bf16; bfp8, extension block_fp8) and is zero for every other type.

private function mxm_unit(u : unit_kind) -> bool = u == U_MXM_W | u == U_MXM_E

// The types whose weights and activations are a stream pair (two stream values per row or per vector): f16, and the
// extension types bf16 and bfp8.
private function is_pair(t : mxm_wtype) -> bool = t == W_F16 | t == W_BF16 | t == W_BFP8

// The issuing plane: 0 .. 2 * mxm.planes_per_hemisphere - 1 (the queue of an absent plane never executes).
private function plane() -> range(0, 2 * mxm_planes - 1) = {
  let pl = issue_unit.plane;
  if pl < 2 * mxm_planes then pl else 0
}

// The stream values a weight buffer or an installed matrix of type t holds: mxm.rows i8, u8 or fp8 rows, or two per f16 row
// (mxm.f16_rows rows).
private function mxm_cap(t : mxm_wtype) -> range(0, mxm_wvals) = if is_pair(t) then 2 * mxm_f16_rows else mxm_rows

// The rows an installed matrix of type t has when it holds n stream values (f16 rows take two).
private function rows_of(t : mxm_wtype, n : int) -> range(0, mxm_rows) = {
  let r : int = if n < 0 then 0 else (if is_pair(t) then n / 2 else n);
  if 0 <= r & r <= mxm_rows then r else mxm_rows
}

// Operands are read when the instruction executes (dispatch + d_skew) and results commit at
// dispatch + d_func. Readiness (ready_lw, ready_iw, the ABC entries) is kept in dispatch
// cycles: a stage is ready from its producer's dispatch + d_func, and d_skew < d_func
// (postlude/timing_check.sail) puts that after the producer has executed.

// The weight type is 3 bits: bits 21..20 and the type-extension bit 12 above them (0 for the four types of the base set; bit 12
// set with 0b00 below it is bf16, with 0b01 bfp8: each defined only in a profile that declares its extension, `bf16` or
// `block_fp8`).
mapping wtype_split : mxm_wtype <-> (bits(1), bits(2)) = {
  W_I8 <-> (0b0, 0b00), W_F16 <-> (0b0, 0b01), W_U8 <-> (0b0, 0b10), W_FP8 <-> (0b0, 0b11),
  W_BF16 <-> (0b1, 0b00), W_BFP8 <-> (0b1, 0b01),
}
// The (bit 12, bits 21..20) pairs that name a weight type: bit 12 set takes only the bf16 and bfp8 values.
private function wtype_valid(h : bits(1), t : bits(2)) -> bool = h == 0b0 | t == 0b00 | t == 0b01
// A profile that does not declare an extension rejects its encodings (the decoder finds no instruction).
private function wtype_ok(t : mxm_wtype) -> bool = (t != W_BF16 | ext_bf16) & (t != W_BFP8 | ext_block_fp8)
private function acctype_declared(t : mxm_acctype) -> bool = t != A_X128 | ext_wide_acc
mapping encdec_acctype : mxm_acctype <-> bits(2) = { A_X128 <-> 0b00, A_I32 <-> 0b10, A_F32 <-> 0b11 }
mapping encdec_acc_mode : acc_mode <-> bits(1) = { ACC_REPLACE <-> 0b0, ACC_ACCUMULATE <-> 0b1 }
mapping wtype_name : mxm_wtype <-> string = { W_I8 <-> "i8", W_F16 <-> "f16", W_U8 <-> "u8", W_FP8 <-> "fp8", W_BF16 <-> "bf16", W_BFP8 <-> "bfp8" }
mapping acctype_name : mxm_acctype <-> string = { A_I32 <-> "i32", A_F32 <-> "f32", A_X128 <-> "x128" }

// ---------------------------------------------------------------------------------------
// f16 weights take a plane pair (HC34 p.28: one 160 x 320 f16 matrix per hemisphere): f16
// LW/IW/ABC run on the even plane (0 or 2) only, and while the even plane holds f16
// weights or buffered f16 rows its odd partner (1 or 3) is busy. Either violation is
// OTS-E-RES-001. A pair exists only in a hemisphere with two planes (mxm.planes_per_hemisphere 2): with one plane per
// hemisphere no plane can run f16 (OTS-E-RES-001).

// bf16 and bfp8 (profile extensions) take a plane pair in the same way, with the same capacity as f16.

private function holds_f16(p : mxm_plane) -> bool =
  (p.rows > 0 & is_pair(p.wtype)) | (p.buf_rows > 0 & is_pair(p.btype))

private function wants_pair(t : option(mxm_wtype)) -> bool = match t { Some(x) => is_pair(x), None() => false }

private function plane_blocked(t : option(mxm_wtype)) -> bool = {
  let pl = plane();
  let paired : bool = mxm_planes == 2;
  let odd = paired & (pl == 1 | pl == 3);
  if wants_pair(t) & not(paired) then true
  else odd & (wants_pair(t) | holds_f16(mxm_plane_at(if pl == 1 then 0 else 2)))
}

// ---------------------------------------------------------------------------------------
// LW t, src: load 16 stream values (src..src+15, src base 0 or 16) into the weight buffer.
// i8: each stream value is one 320-column row. f16: stream pairs (2i, 2i+1) are row i's
// low and high bytes. The buffer holds mxm.rows stream values of i8, u8 or fp8 (that many rows) and 2 * mxm.f16_rows of f16
// (mxm.f16_rows rows); a load that would pass that is OTS-E-RES-002.

union clause instruction = MXM_LW : (mxm_wtype, streamref)

mapping clause encdec32 = MXM_LW(wtype_split(h, t), Streamref(s))
  <-> 0x3 @ 0b000000 @ t @ s @ 0b0 @ h @ 0b000000000000
  when (if wtype_valid(h, t) then (s[3 .. 0] == 0x0 & wtype_ok(wtype_split(h, t))) else false)

val nonempty_list : forall ('a : Type). list('a) -> bool
function nonempty_list(xs) = match xs { [||] => false, _ => true }

function clause execute MXM_LW(t, s) = {
  let p = mxm[plane()];
  if plane_blocked(Some(t)) then Diagnostic(RES_NOT_READY)
  else match read_planes16(s, exec_cycle()) {
    Err(e)   => Diagnostic(e),
    Ok(rows) =>
      if p.buf_rows > 0 & p.btype != t then Diagnostic(RES_NOT_READY)
      else {
        let n = p.buf_rows;
        if n + 16 > mxm_cap(t) then Diagnostic(RES_FULL) else {
        var q = p;
        foreach (i from 0 to 15) { q.buf[n + i] = rows[i]; q.buf_bad[n + i] = ecc_g16[i] };
        q.btype = t;
        q.buf_rows = n + 16;
        q.ready_lw = commit_at();
        mxm_stage(plane(), P_BUF, q);
        Issued()
        }
      },
  }
}

mapping clause assembly = MXM_LW(t, s) <-> "lw." ^ wtype_name(t) ^ spc() ^ stream_name(s)

// ---------------------------------------------------------------------------------------
// IW t (mode 0): install the LW buffer as the plane's weights and empty the buffer.
// IW t, src (mode 1): install 16 stream values src..src+15 directly into the array
// (ISCA 2020 Table I: "from streams or LW buffer"). Successive stream installs append
// rows; the first one after reset, a buffer install or an ABC dispatched after the last
// stream install starts a new matrix (opentsp decision).

union clause instruction = MXM_IW : mxm_wtype
union clause instruction = MXM_IWS : (mxm_wtype, streamref)

mapping clause encdec32 = MXM_IW(wtype_split(h, t))
  <-> 0x3 @ 0b000001 @ t @ 0b000000 @ 0b0 @ h @ 0b000000000000
  when (if wtype_valid(h, t) then (wtype_ok(wtype_split(h, t))) else false)
mapping clause encdec32 = MXM_IWS(wtype_split(h, t), Streamref(s))
  <-> 0x3 @ 0b000001 @ t @ s @ 0b1 @ h @ 0b000000000000
  when (if wtype_valid(h, t) then (s[3 .. 0] == 0x0 & wtype_ok(wtype_split(h, t))) else false)

function clause execute MXM_IW(t) = {
  let p = mxm[plane()];
  if plane_blocked(Some(t)) | p.buf_rows == 0 | issue_cycle < p.ready_lw | p.btype != t then Diagnostic(RES_NOT_READY)
  else {
    var q = p;
    q.wtype = t;
    q.weights = p.buf;
    q.w_bad = p.buf_bad;
    q.buf_bad = vector_init(false);
    q.rows = rows_of(t, p.buf_rows);
    q.iw_open = false;
    q.ready_iw = commit_at();
    q.buf_rows = 0;
    mxm_stage(plane(), P_W, q);
    mxm_stage(plane(), P_BUF, q);
    Issued()
  }
}

// True when an ABC was dispatched after cycle c.
private function abc_after(xs : list(abc_entry), c : nat) -> bool =
  match xs { [||] => false, e :: _ => e.dispatch > c }

// The stream values of an f16 matrix of the given rows: two per row (never more than 2 * mxm.f16_rows, which the installs keep).
private function f16_stream_values(rows : int) -> range(0, mxm_wvals) =
  if rows >= 0 & 2 * rows <= mxm_wvals then 2 * rows else mxm_wvals

function clause execute MXM_IWS(t, s) = {
  let p = mxm[plane()];
  if plane_blocked(Some(t)) then Diagnostic(RES_NOT_READY)
  else match read_planes16(s, exec_cycle()) {
    Err(e)   => Diagnostic(e),
    Ok(rows) => {
      let open = p.iw_open & not(abc_after(p.abc, p.iws_cycle));
      // Stream values installed so far in the open matrix (f16 rows take two each).
      let n : range(0, mxm_wvals) = if open then (if is_pair(p.wtype) then f16_stream_values(p.rows) else p.rows) else 0;
      if open & p.wtype != t then Diagnostic(RES_NOT_READY)
      else if n + 16 > mxm_cap(t) then Diagnostic(RES_FULL)
      else {
        var q = p;
        foreach (i from 0 to 15) { q.weights[n + i] = rows[i]; q.w_bad[n + i] = ecc_g16[i] };
        q.wtype = t;
        q.rows = rows_of(t, n + 16);
        q.iw_open = true;
        q.iws_cycle = issue_cycle;
        q.ready_iw = commit_at();
        mxm_stage(plane(), P_W, q);
        Issued()
      }
    },
  }
}

mapping clause assembly = MXM_IWS(t, s) <-> "iw." ^ wtype_name(t) ^ spc() ^ stream_name(s)

mapping clause assembly = MXM_IW(t) <-> "iw." ^ wtype_name(t)

// ---------------------------------------------------------------------------------------
// ABC t, src: product[r] = sum over k of W[r][k] * x[k] for every installed row r. A product
// has mxm.rows lanes: lane r is zero for a row that is not installed, and no row passes
// mxm.rows (the buffer and the matrix hold that many stream values).
// i8/u8: exact products, 32-bit sum with wrap. f16: the exact sum of the exact products,
// rounded ONCE (round to nearest even) to f32 (ISCA 2020 sec. III-D: a single
// rounding step). Special values: any NaN operand, Inf x 0, or +Inf and -Inf products give
// the canonical NaN; otherwise an infinite product gives that infinity. An exact zero sum
// is -0 only when every product is -0, else +0. Overflow rounds to infinity.
// bfp8 (extension block_fp8): the same for a stream pair of e4m3 data and block scales (bfp8_exact). With the extension wide_acc
// a floating-point ABC also keeps the exact sums (abc_entry.wide) for ACC x128.

union clause instruction = MXM_ABC : (mxm_wtype, streamref)

// The FP8 form: type code 3 with the format in mode bit 13 (0 = e4m3, 1 = e5m2), the format of the weights
// and of the activations alike (the installed weights are the type fp8, bytes without a format).
union clause instruction = MXM_ABC8 : (bits(1), streamref)

mapping clause encdec32 = MXM_ABC(wtype_split(h, t), Streamref(s))
  <-> 0x3 @ 0b000010 @ t @ s @ 0b0 @ h @ 0b000000000000
  when (if wtype_valid(h, t) then (wtype_split(h, t) != W_FP8 & wtype_ok(wtype_split(h, t))) else false)
mapping clause encdec32 = MXM_ABC8(f, Streamref(s))
  <-> 0x3 @ 0b000010 @ 0b11 @ s @ f @ 0b0 @ 0b000000000000

private function lane8(v : vec, k : range(0, vlanes - 1)) -> bits(8) = if k < vlanes then slice(v, 8 * k, 8) else 0x00

// The exact dot product of one weight row and an activation vector before rounding: the special-value flags and the sum of
// the finite products in units of 2^-48 (the smallest unit of an f16 product; the fp8 and bfp8 products are multiples of it).
// `dnz`: every product is -0 (or the row has no product that is not -0), which makes a zero sum -0.
struct dsum = { dnan : bool, dpinf : bool, dninf : bool, dnz : bool, dsum : int }

// Round the exact dot product ONCE to f32 (round to nearest even). Special values: any NaN operand, Inf x 0, or +Inf and -Inf
// products give the canonical NaN; otherwise an infinite product gives that infinity. An exact zero sum is -0 only when every
// product is -0, else +0. Overflow rounds to infinity.
// `u` is the unit of the sum: 2^-u (48 for f16, fp8 and bfp8, 266 for bf16).
function dsum_round_u(d : dsum, u : int) -> bits(32) =
  if d.dnan | (d.dpinf & d.dninf) then canon_nan(FMT32)
  else if d.dpinf then inf_bits(FMT32, false)
  else if d.dninf then inf_bits(FMT32, true)
  else if d.dsum == 0 then zero_bits(FMT32, d.dnz)
  else if d.dsum < 0 then fround(FMT32, true, 0 - d.dsum, -u)
  else fround(FMT32, false, d.dsum, -u)

function dsum_round(d : dsum) -> bits(32) = dsum_round_u(d, 48)

// The same value as a lane of the wide accumulator (no rounding).
function dsum_wide(d : dsum) -> wlane =
  if d.dnan | (d.dpinf & d.dninf) then struct { wv = 0, wsp = 3, wnz = false }
  else if d.dpinf then struct { wv = 0, wsp = 1, wnz = false }
  else if d.dninf then struct { wv = 0, wsp = 2, wnz = false }
  else struct { wv = d.dsum, wsp = 0, wnz = d.dsum == 0 & d.dnz }

// The exact dot product of an FP8 weight row and activation vector in format f (e4m3 or e5m2). Products are exact in units
// of 2^-48; the sum of 320 of them is an integer. NaN rules as f16_exact; e4m3 has no infinity, an e5m2 infinity gives an
// infinity.
function fp8_exact(f : fmt, wr : vec, xv : vec) -> dsum = {
  var nan : bool = false;
  var pinf : bool = false;
  var ninf : bool = false;
  var all_negzero : bool = true;
  var sum : int = 0;
  foreach (k from 0 to (vlanes - 1)) {
    let a = fdecode(f, 0x000000 @ lane8(wr, k));
    let b = fdecode(f, 0x000000 @ lane8(xv, k));
    let neg = a.neg != b.neg;
    if a.cls == FNaN | b.cls == FNaN then nan = true
    else if (a.cls == FInf & b.cls == FZero) | (a.cls == FZero & b.cls == FInf) then nan = true
    else if a.cls == FInf | b.cls == FInf then { if neg then ninf = true else pinf = true; all_negzero = false }
    else if a.cls == FZero | b.cls == FZero then { if not(neg) then all_negzero = false }
    else {
      all_negzero = false;
      let m = a.mag * b.mag * pow2i(a.e + b.e + 48);
      sum = if neg then sum - m else sum + m
    }
  };
  struct { dnan = nan, dpinf = pinf, dninf = ninf, dnz = all_negzero, dsum = sum }
}

// The FP8 dot product rounded once to f32 (round to nearest even), as f16_dot.
function fp8_dot(f : fmt, wr : vec, xv : vec) -> bits(32) = dsum_round(fp8_exact(f, wr, xv))

// The exact dot product of row w (lo, hi byte planes) and activation x in the 16-bit float format f. The products are exact in
// units of 2^-u (u = twice the exponent of the smallest subnormal: 48 for f16, 266 for bf16).
function pair_exact(f : fmt, u : int, wlo : vec, whi : vec, xlo : vec, xhi : vec) -> dsum = {
  var nan : bool = false;
  var pinf : bool = false;
  var ninf : bool = false;
  var all_negzero : bool = true;
  var sum : int = 0;                     // in units of 2^-u
  foreach (k from 0 to (vlanes - 1)) {
    let a = fdecode(f, 0x0000 @ lane8(whi, k) @ lane8(wlo, k));
    let b = fdecode(f, 0x0000 @ lane8(xhi, k) @ lane8(xlo, k));
    let neg = a.neg != b.neg;
    if a.cls == FNaN | b.cls == FNaN then nan = true
    else if (a.cls == FInf & b.cls == FZero) | (a.cls == FZero & b.cls == FInf) then nan = true
    else if a.cls == FInf | b.cls == FInf then { if neg then ninf = true else pinf = true; all_negzero = false }
    else if a.cls == FZero | b.cls == FZero then { if not(neg) then all_negzero = false }
    else {
      all_negzero = false;
      let m = a.mag * b.mag * pow2i(a.e + b.e + u);
      sum = if neg then sum - m else sum + m
    }
  };
  struct { dnan = nan, dpinf = pinf, dninf = ninf, dnz = all_negzero, dsum = sum }
}

function f16_exact(wlo : vec, whi : vec, xlo : vec, xhi : vec) -> dsum = pair_exact(FMT16, 48, wlo, whi, xlo, xhi)

// The exact f16 dot product of row w (lo, hi byte planes) and activation x, rounded once.
function f16_dot(wlo : vec, whi : vec, xlo : vec, xhi : vec) -> bits(32) = dsum_round(f16_exact(wlo, whi, xlo, xhi))

// The bf16 dot product (extension bf16), rounded once. Its exact sum is in units of 2^-266, which the 128-bit wide accumulator
// (units of 2^-48) cannot hold: a bf16 ABC keeps no wide products and its result does not feed ACC x128.
function bf16_dot(wlo : vec, whi : vec, xlo : vec, xhi : vec) -> bits(32) = dsum_round_u(pair_exact(FMT_BF16, 266, wlo, whi, xlo, xhi), 266)

// Block-scaled fp8 (extension block_fp8, type bfp8): a vector is a stream pair, the e4m3 elements in the first stream value
// and the block scales in the second. A block is the 16 lanes of one superlane; its scale is the low four bits of the second
// stream value's byte at the block's first lane, a two's-complement exponent -8..7 (the upper four bits and the other lanes
// of the block are not read). The value of lane k is the e4m3 element times 2 to the power of the scale of its block.
private function blk_exp(v : vec, k : range(0, vlanes - 1)) -> int = {
  let b : int = 16 * (k / 16);
  if b >= 0 & 8 * b + 4 <= 8 * vlanes then {
    let n : int = unsigned(slice(v, 8 * b, 4));
    if n >= 8 then n - 16 else n
  } else 0
}

// The exact dot product of a bfp8 weight row (data wd, scales ws) and a bfp8 activation (data xd, scales xs), the sum of
// exact products in units of 2^-48. The e4m3 format has no infinity.
function bfp8_exact(wd : vec, ws : vec, xd : vec, xs : vec) -> dsum = {
  var nan : bool = false;
  var all_negzero : bool = true;
  var sum : int = 0;
  foreach (k from 0 to (vlanes - 1)) {
    let a = fdecode(FMT_E4M3, 0x000000 @ lane8(wd, k));
    let b = fdecode(FMT_E4M3, 0x000000 @ lane8(xd, k));
    let neg = a.neg != b.neg;
    if a.cls == FNaN | b.cls == FNaN then nan = true
    else if a.cls == FZero | b.cls == FZero then { if not(neg) then all_negzero = false }
    else {
      all_negzero = false;
      let m = a.mag * b.mag * pow2i(a.e + b.e + blk_exp(ws, k) + blk_exp(xs, k) + 48);
      sum = if neg then sum - m else sum + m
    }
  };
  struct { dnan = nan, dpinf = false, dninf = false, dnz = all_negzero, dsum = sum }
}

// The wide accumulator (extension wide_acc). A lane is added in exact integer arithmetic and wraps at 128 bits (two's
// complement); a special value is sticky: NaN, or +Inf plus -Inf, gives NaN; an infinity stays, and plus a finite value
// stays. The read-out rounds the sum ONCE to f32 (round to nearest even); a zero sum is -0 only when every term was -0.
function wlane_add(a : wlane, b : wlane) -> wlane =
  if a.wsp == 3 | b.wsp == 3 | (a.wsp == 1 & b.wsp == 2) | (a.wsp == 2 & b.wsp == 1) then struct { wv = 0, wsp = 3, wnz = false }
  else if a.wsp != 0 then a
  else if b.wsp != 0 then b
  else struct { wv = wrap_s(a.wv + b.wv, 128), wsp = 0, wnz = a.wnz & b.wnz }

function wlane_read(w : wlane) -> bits(32) =
  if w.wsp == 3 then canon_nan(FMT32)
  else if w.wsp == 1 then inf_bits(FMT32, false)
  else if w.wsp == 2 then inf_bits(FMT32, true)
  else if w.wv == 0 then zero_bits(FMT32, w.wnz)
  else if w.wv < 0 then fround(FMT32, true, 0 - w.wv, -48)
  else fround(FMT32, false, w.wv, -48)

// Drop ABC results no ACC can use any more: keep the entries up to and including the
// newest one committed by cycle c (later ACCs dispatch at or after c). The newest committed entry may be one of the two
// products of an FP8 pair (mxm.fp8_rate 2: two entries of one dispatch cycle); its partner is kept too, as an ACC of
// cycle c still reads the pair (abc_at).
private function abc_prune(xs : list(abc_entry), c : nat) -> list(abc_entry) =
  match xs {
    [||] => [||],
    e :: rest => if e.commit <= c then (match rest {
        f :: _ => if f.dispatch == e.dispatch then [|e, f|] else [|e|],
        _ => [|e|],
      }) else e :: abc_prune(rest, c),
  }

// The ABC result an ACC of accumulator a reads at cycle c: the newest one committed by c. Two FP8 ABCs that one plane accepted in
// one cycle (mxm.fp8_rate 2) are two results of the same dispatch cycle: an even accumulator takes the one of the lower queue, an
// odd accumulator the one of the higher queue.
private function abc_pair_pick(e : abc_entry, f : abc_entry, a : int) -> abc_entry = {
  let lo = if e.q < f.q then e else f;
  let hi = if e.q < f.q then f else e;
  if emod_int(a, 2) == 0 then lo else hi
}

private function abc_at(xs : list(abc_entry), c : nat, a : int) -> option(abc_entry) =
  match xs {
    [||] => None(),
    e :: rest => if e.commit <= c then (match rest {
        f :: _ => if f.dispatch == e.dispatch & f.commit <= c then Some(abc_pair_pick(e, f, a)) else Some(e),
        _ => Some(e),
      }) else abc_at(rest, c, a),
  }

// FP16 initiation interval is configurable; ref/mini use one cycle (RC-A1).
// Pair exclusion and half the output rows already give quarter-rate chip capacity.
// A longer interval is an explicit research choice, not a second published reduction.
// Keep occupancy <= d_func so every still-busy ABC remains in the uncommitted history.
private function macc_busy_in(xs : list(abc_entry)) -> bool =
  match xs {
    [||] => false,
    e :: rest => (is_pair(e.ptype) & e.dispatch < issue_cycle & issue_cycle < e.dispatch + timing_mxm_occupancy_f16) | macc_busy_in(rest),
  }

private function macc_busy(p : mxm_plane) -> bool = macc_busy_in(p.abc)

// Integer ABC operand signs (s x s, u x s, s x u, u x u). The installed weight type (i8 or u8) gives the
// weight sign and the ABC type gives the activation sign, so all four products are
// expressible: abc.u8 on i8 weights is signed x unsigned, abc.i8 on u8 weights unsigned x
// signed. The published sources do not say how the mode is selected; reading it from the
// two existing type fields is an opentsp decision. f16 weights need an f16 ABC and the
// reverse (OTS-E-RES-001).
private function abc_types_ok(wt : mxm_wtype, t : mxm_wtype) -> bool =
  if is_pair(wt) | is_pair(t) | wt == W_FP8 | t == W_FP8 then wt == t else true

// One row's integer dot product: exact products of the 8-bit lanes with the given signs,
// 32-bit sum with wrap (HC34 p.26: INT8, UINT8).
private function int8_dot(wr : vec, ws : bool, xv : vec, xs : bool) -> bits(32) = {
  var sum : int = 0;
  foreach (k from 0 to (vlanes - 1)) {
    let a : int = if ws then signed(lane8(wr, k)) else unsigned(lane8(wr, k));
    let b : int = if xs then signed(lane8(xv, k)) else unsigned(lane8(xv, k));
    sum = sum + a * b
  };
  bits_of_s(wrap_s(sum, 32), 32)
}

// ---------------------------------------------------------------------------------------
// The sum check of the matrix unit (microarchitecture.md sec. 18). It runs only for an ABC on which the harness
// injected a fault (`F mxm`): the model's arithmetic is exact, so without a fault the actual sums equal the expected
// ones and the check cannot find anything (the trace is identical).
//
// The sums are exact integers: an integer product sum as it is, a floating-point one in units of 2^-48 (f16) or 2^-32
// (fp8), the finite products only (a NaN or infinite element has no exact value and adds nothing to either side).

// The exact value of element k of a stored row or an activation, in units of 2^-24 (f16) or 2^-16 (fp8) for the floating
// formats; None for a NaN or an infinity.
private function fval_int(v : fval, s : int) -> option(int) =
  match v.cls {
    FNaN => None(),
    FInf => None(),
    FZero => Some(0),
    FFinite => Some(if v.neg then 0 - v.mag * pow2i(v.e + s) else v.mag * pow2i(v.e + s)),
  }

private function chk_elem(t : mxm_wtype, f : bits(1), lo : vec, hi : vec, k : range(0, vlanes - 1), sgn : bool) -> option(int) =
  match t {
    W_I8 => Some(if sgn then signed(lane8(lo, k)) else unsigned(lane8(lo, k))),
    W_U8 => Some(if sgn then signed(lane8(lo, k)) else unsigned(lane8(lo, k))),
    W_FP8 => fval_int(fdecode(if f == 0b0 then FMT_E4M3 else FMT_E5M2, 0x000000 @ lane8(lo, k)), 16),
    W_F16 => fval_int(fdecode(FMT16, 0x0000 @ lane8(hi, k) @ lane8(lo, k)), 24),
    W_BFP8 => fval_int(fdecode(FMT_E4M3, 0x000000 @ lane8(lo, k)), 17 + blk_exp(hi, k)),
    W_BF16 => fval_int(fdecode(FMT_BF16, 0x0000 @ lane8(hi, k) @ lane8(lo, k)), 133),
  }

// Flip bit b of the 96-bit two's-complement form of p.
private function flip96(p : int, b : int) -> int = {
  let u = emod_int(p, pow2i(96));
  let one = emod_int(ediv_int(u, pow2i(b)), 2) == 1;
  let u2 = if one then u - pow2i(b) else u + pow2i(b);
  if u2 >= pow2i(95) then u2 - pow2i(96) else u2
}

private function bit_of(r : int, i : int) -> bool = emod_int(ediv_int(r, pow2i(i)), 2) == 1

// The weight row r of the installed weights: its low and high byte planes (f16 rows take two).
// An f16 row is the byte planes 2 * r and 2 * r + 1; an f16 plane has at most mxm.f16_rows rows (the installs keep
// 2 * rows <= 2 * mxm.f16_rows), so the guard is true for every installed row and gives the checker the bound of the indices.
private function f16_row(ws : vector(mxm_wvals, wrow), r : int) -> (vec, vec) =
  if 0 <= r & 2 * r + 1 < mxm_wvals then (ws[2 * r], ws[2 * r + 1]) else (zero_vec, zero_vec)

private function f16_row_bad(bs : vector(mxm_wvals, bool), r : int) -> bool =
  if 0 <= r & 2 * r + 1 < mxm_wvals then bs[2 * r] | bs[2 * r + 1] else false

private function chk_wrow(p : mxm_plane, t : mxm_wtype, r : range(0, mxm_rows - 1)) -> (vec, vec) = {
  match t {
    W_F16 => f16_row(p.weights, r),
    W_BFP8 => f16_row(p.weights, r),
    W_BF16 => f16_row(p.weights, r),
    _ => (p.weights[r], zero_vec),
  }
}

// Exact per-row redundant check. Primary row-major products and independently recomputed column-major nominal
// products use the same captured operands; faults affect primary sums only. Compare final row values, not fault counts.
// Result: 0 no changed row, 1 exactly one changed row (located), 2 every multirow error. The abstract timing is unchanged;
// physical checker protection, duplicate compute and storage costs require a separate circuit implementation.
private function mxm_check(p : mxm_plane, t : mxm_wtype, f : bits(1), x0 : vec, x1 : vec, fl : list((nat, range(0, 95)))) -> (range(0, 2), nat) = {
  let nr = p.rows;
  let ws = p.wtype == W_I8;
  let xs = t == W_I8;
  var xv : vector(vlanes, option(int)) = vector_init(None());
  foreach (k from 0 to (vlanes - 1)) { xv[k] = chk_elem(t, f, x0, x1, k, xs) };
  var ps : vector(mxm_rows, int) = vector_init(0);
  var expected : vector(mxm_rows, int) = vector_init(0);
  foreach (r from 0 to (nr - 1)) {
    let (lo, hi) = chk_wrow(p, t, r);
    foreach (k from 0 to (vlanes - 1)) {
      match (chk_elem(t, f, lo, hi, k, ws), xv[k]) {
        (Some(a), Some(b)) => ps[r] = ps[r] + a * b,
        _ => (),
      }
    }
  };
  foreach (k from 0 to (vlanes - 1)) {
    match xv[k] {
      None() => (),
      Some(xk) => {
        // Independent nominal check: column-major recomputation from the captured operands.
        foreach (r from 0 to (nr - 1)) {
          let (lo, hi) = chk_wrow(p, t, r);
          match chk_elem(t, f, lo, hi, k, ws) {
            None() => (),
            Some(w) => expected[r] = expected[r] + w * xk,
          }
        };
      },
    }
  };
  // the injected faults flip bits of the exact sums; a lane outside the installed rows carries no product
  var faulty : list((nat, range(0, 95))) = fl;
  foreach (_step from 0 to (list_len(fl) - 1)) {
    match faulty {
      [||] => (),
      (lane, bit) :: rest => {
        if lane < nr then ps[lane] = flip96(ps[lane], bit);
        faulty = rest
      },
    }
  };
  // Classify final changed rows, not fault entries or an alias-prone aggregate syndrome.
  var wrong : nat = 0;
  var loc : nat = 0;
  foreach (r from 0 to (nr - 1)) {
    if ps[r] != expected[r] then { wrong = wrong + 1; loc = r }
  };
  if wrong == 0 then (0, 0)
  else if wrong == 1 then (1, loc)
  else (2, 0)
}

// ABC of type t; f is the FP8 format (mode bit 13), 0 for every other type.
private function abc_exec(t : mxm_wtype, f : bits(1), s : streamref) -> ExecutionResult = {
  let p = mxm[plane()];
  let w : {1, 2} = if is_pair(t) then 2 else 1;
  if plane_blocked(Some(t)) | p.rows == 0 | issue_cycle < p.ready_iw | not(abc_types_ok(p.wtype, t)) | macc_busy(p) then Diagnostic(RES_NOT_READY)
  else match read_planes(s, w, exec_cycle()) {
    Err(e) => Diagnostic(e),
    Ok(x)  => {
      // The faults of the harness for this plane are checked before the products are formed
      var mine : list((nat, range(0, 95))) = [||];
      var others : list(mxm_fault) = [||];
      var pend = mxm_fault_list;
      foreach (_step from 0 to (list_len(mxm_fault_list) - 1)) {
        match pend {
          [||] => (),
          mfe :: rest => {
            if mfe.mf_pl == plane() then mine = (mfe.mf_lane, mfe.mf_bit) :: mine else others = mfe :: others;
            pend = rest
          },
        }
      };
      mxm_fault_list = others;
      var verdict : (range(0, 2), nat) = (0, 0);
      match mine {
        [||] => (),
        _ => { let x1 : vec = if is_pair(t) then x[1] else zero_vec; verdict = mxm_check(p, t, f, x[0], x1, mine) },
      };
      let (vstat, vlane) = verdict;
      let ec : nat = if exec_cycle() >= 0 then exec_cycle() else 0;
      if vstat == 2 then { ras_mxm_event(ec, issue_queue, true, plane(), 65535); Diagnostic(ECC_MXM) } else {
      if vstat == 1 then { mxm_corr_callback(ec, issue_queue, plane(), vlane); ras_mxm_event(ec, issue_queue, false, plane(), vlane) };
      var out : vector(mxm_rows, bits(32)) = vector_init(0x00000000);
      var wide : vector(mxm_rows, wlane) = vector_init(wlane_zero);
      let ws = p.wtype == W_I8;          // weight sign: the installed type
      let xs = t == W_I8;                // activation sign: the ABC type
      foreach (r from 0 to (p.rows - 1)) {
        match t {
          W_I8 => out[r] = int8_dot(p.weights[r], ws, x[0], xs),
          W_U8 => out[r] = int8_dot(p.weights[r], ws, x[0], xs),
          W_FP8 => {
            let d = fp8_exact(if f == 0b0 then FMT_E4M3 else FMT_E5M2, p.weights[r], x[0]);
            out[r] = dsum_round(d);
            if ext_wide_acc then wide[r] = dsum_wide(d)
          },
          W_F16 => {
            // An f16 plane has at most mxm.rows / 2 rows (two byte planes each); see f16_row.
            let (wlo, whi) = f16_row(p.weights, r);
            let d = f16_exact(wlo, whi, x[0], x[1]);
            out[r] = dsum_round(d);
            if ext_wide_acc then wide[r] = dsum_wide(d)
          },
          W_BFP8 => {
            // The row is the stream pair (data, scale), like an f16 row.
            let (wd, wsc) = f16_row(p.weights, r);
            let d = bfp8_exact(wd, wsc, x[0], x[1]);
            out[r] = dsum_round(d);
            if ext_wide_acc then wide[r] = dsum_wide(d)
          },
          W_BF16 => {
            let (wlo, whi) = f16_row(p.weights, r);
            out[r] = bf16_dot(wlo, whi, x[0], x[1])
          },
        }
      };
      // Poison (microarchitecture.md sec. 15): every product lane depends on every lane of the activation, so an uncorrectable
      // superlane of it (ecc_bad, which also gives the instruction's OTS-E-ECC-001) poisons every superlane of the product;
      // lane r depends on row r of the installed weights only, so a row with an uncorrectable superlane poisons the
      // superlane r / 16 (an f16 row is the two stream values 2 r and 2 r + 1).
      var poison : bits(superlanes) = if ecc_bad != sail_zeros(superlanes) then sail_ones(superlanes) else sail_zeros(superlanes);
      if poison == sail_zeros(superlanes) then {
        foreach (r from 0 to (p.rows - 1)) {
          let rb : bool = match t {
            W_F16 => f16_row_bad(p.w_bad, r),
            W_BFP8 => f16_row_bad(p.w_bad, r),
            W_BF16 => f16_row_bad(p.w_bad, r),
            _ => p.w_bad[r],
          };
          if rb & r / 16 < superlanes then poison[r / 16] = bitone
        }
      };
      let e : abc_entry = struct { dispatch = issue_cycle, commit = commit_at(), ptype = t, product = out, wide = wide, poison = poison, q = issue_queue };
      var q = p;
      q.abc = e :: abc_prune(p.abc, issue_cycle);
      mxm_stage(plane(), P_ABC, q);
      Issued() }
    },
  }
}

function clause execute MXM_ABC(t, s) = abc_exec(t, 0b0, s)
function clause execute MXM_ABC8(f, s) = abc_exec(W_FP8, f, s)

mapping clause assembly = MXM_ABC(t, s) <-> "abc." ^ wtype_name(t) ^ spc() ^ stream_name(s)

mapping fp8_format_name : bits(1) <-> string = { 0b0 <-> "e4m3", 0b1 <-> "e5m2" }
mapping clause assembly = MXM_ABC8(f, s) <-> "abc." ^ fp8_format_name(f) ^ spc() ^ stream_name(s)

// ---------------------------------------------------------------------------------------
// ACC t, dst, mode, a: accumulator a (0 .. mxm.acc_depth - 1; each MACC has two 32-bit accumulators in the published chip,
// 2 in the reference profile) := product (replace) or acc[a] + product (accumulate: i32 wrap / one f32
// add); write it as 4 byte-plane streams dst..dst+3. The product is that of the newest
// ABC committed by the ACC's dispatch cycle (none: OTS-E-RES-001). i32 needs i8 or u8
// weights, f32 needs a floating-point product (f16, bf16, fp8 or bfp8). The index a is bit 12 (its low bit) and bits 11..8 (the
// four bits above it, reserved zero in the first profiles, so every word with a = 0 or 1 keeps its meaning); an index at or above
// mxm.acc_depth does not decode.
// x128 (extension wide_acc): the exact product of an f16, fp8 or bfp8 ABC is added to (or replaces) a 128-bit fixed-point
// lane of the wide accumulator; the streams get the lane rounded once to f32. The 32-bit accumulators are a separate state.

union clause instruction = MXM_ACC : (mxm_acctype, acc_mode, streamref, bits(5))

mapping clause encdec32 = MXM_ACC(t, m, Streamref(d), (ahi : bits(4)) @ (alo : bits(1)))
  <-> 0x3 @ 0b000011 @ encdec_acctype(t) @ d @ encdec_acc_mode(m) @ alo @ ahi @ 0x00
  when unsigned(ahi @ alo) < mxm_acc_depth & acctype_declared(t)

function clause execute MXM_ACC(t, m, d, ab) = {
  let p = mxm[plane()];
  let a : range(0, 31) = unsigned(ab);
  if plane_blocked(None()) then Diagnostic(RES_NOT_READY)
  else if a >= mxm_acc_depth then Diagnostic(OPR_RANGE)
  else match abc_at(p.abc, issue_cycle, a) {
    None() => Diagnostic(RES_NOT_READY),
    Some(e) => {
      let matches : bool = match (e.ptype, t) {
        (W_I8, A_I32) => true, (W_U8, A_I32) => true,
        (W_F16, A_F32) => true, (W_BF16, A_F32) => true, (W_FP8, A_F32) => true, (W_BFP8, A_F32) => true,
        (W_F16, A_X128) => true, (W_FP8, A_X128) => true, (W_BFP8, A_X128) => true,
        _ => false,
      };
      if not(matches) then Diagnostic(OPR_RANGE)
      else {
        var out : vector(mxm_rows, bits(32)) = vector_init(0x00000000);
        var wnew : vector(mxm_rows, wlane) = p.acc128[a];
        foreach (l from 0 to (mxm_rows - 1)) {                    // a product has mxm.rows lanes
          match t {
            A_X128 => {
              // The wide accumulator keeps the exact sum; the stream gets its read-out, rounded once to f32.
              wnew[l] = match m { ACC_REPLACE => e.wide[l], ACC_ACCUMULATE => wlane_add(p.acc128[a][l], e.wide[l]) };
              out[l] = wlane_read(wnew[l])
            },
            _ => out[l] = match (m, t) {
              (ACC_REPLACE, _)        => e.product[l],
              (ACC_ACCUMULATE, A_I32) => bits_of_s(wrap_s(signed(e.product[l]) + signed(p.acc[a][l]), 32), 32),
              (ACC_ACCUMULATE, _)     => fp_add(FMT32, p.acc[a][l], e.product[l]),
            },
          }
        };
        // The poison of the result: the product's, and the accumulator's when accumulating (the result is inherited,
        // not found by this instruction: it is not reported again).
        let pm : bits(superlanes) = match m { ACC_REPLACE => e.poison, ACC_ACCUMULATE => e.poison | p.acc_poison[a] };
        ecc_inherit = pm;
        // The accumulator changes only if the result can be written: a diagnostic has no effect.
        match write_planes(d, words_to_planes(out), 4, commit_at()) {
          Ok(()) => { var q = p; if t == A_X128 then q.acc128[a] = wnew else q.acc[a] = out; q.acc_poison[a] = pm; mxm_stage_acc(plane(), P_ACC, a, q); Issued() },
          Err(err) => Diagnostic(err),
        }
      }
    },
  }
}

mapping acc_mode_name : acc_mode <-> string = { ACC_REPLACE <-> "replace", ACC_ACCUMULATE <-> "accumulate" }

mapping clause assembly = MXM_ACC(t, m, d, a)
  <-> "acc." ^ acctype_name(t) ^ spc() ^ stream_name(d) ^ sep() ^ acc_mode_name(m) ^ sep() ^ dec_bits_5(a)

// ---------------------------------------------------------------------------------------

function clause timing_of MXM_LW(_) = T_MXM_LW
function clause timing_of MXM_IW(_) = T_MXM_IW
function clause timing_of MXM_IWS(_) = T_MXM_IW
function clause timing_of MXM_ABC(_) = T_MXM_ABC
function clause timing_of MXM_ABC8(_) = T_MXM_ABC
function clause timing_of MXM_ACC(_) = T_MXM_ACC

function clause entry_of MXM_LW(_) = "LW"
function clause entry_of MXM_IW(_) = "IW"
function clause entry_of MXM_IWS(_) = "IWS"
function clause entry_of MXM_ABC(_) = "ABC"
function clause entry_of MXM_ABC8(_) = "ABC8"
function clause entry_of MXM_ACC(_) = "ACC"

function clause issued_by(MXM_LW(_), u) = mxm_unit(u)
function clause issued_by(MXM_IW(_), u) = mxm_unit(u)
function clause issued_by(MXM_IWS(_), u) = mxm_unit(u)
function clause issued_by(MXM_ABC(_), u) = mxm_unit(u)
function clause issued_by(MXM_ABC8(_), u) = mxm_unit(u)
function clause issued_by(MXM_ACC(_), u) = mxm_unit(u)
