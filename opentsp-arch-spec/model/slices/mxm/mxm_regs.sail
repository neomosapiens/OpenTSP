// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// MXM plane state (ISCA 2020 sec. III-D: four independent 320 x 320 planes; HC34 p.28:
// planes 0, 1 west, 2, 3 east; mxm.planes_per_hemisphere planes per hemisphere, plane k of the east hemisphere is plane
// mxm.planes_per_hemisphere + k). Each plane has a weight buffer filled by LW, installed
// weights (IW), the products of its ABCs, and mxm.acc_depth accumulators.
//
// Internal-state timing: a stage's state is usable only once that stage has
// COMMITTED, at its dispatch + d_func. ready_* record those cycles; a later instruction
// that needs the state earlier raises RES_NOT_READY.
//
// Start-of-cycle state: instructions read the plane as it was at the start
// of their execute cycle. Their changes are staged (mxm_stage) and applied when that cycle
// ends (unit_state_flush, called by the cycle driver). The independently written parts
// are the buffer, the weights, the ABC products and each accumulator. Distinct queues
// staging the same part of one plane conflict: each raises OTS-E-RES-003 and none of those
// changes takes effect (their ACC stream results are withdrawn too). The permitted pair
// of FP8 ABCs with mxm.fp8_rate 2 is the explicit exception. Multiple writes from the same
// queue retain the earliest staged value of each part, without a cross-queue collision.
// SXM map changes use the corresponding different-queue conflict and same-queue ordering.

// One weight row per stream value: i8 uses one byte plane per row, f16 two (lo, hi).
type wrow = vec

// One lane of the wide accumulator (extension wide_acc) and of the exact product that feeds it: the exact value as a
// two's-complement integer in units of 2^-48 (`wv`, 128 bits wide in an accumulator), or a special value (`wsp`: 0 finite,
// 1 +Inf, 2 -Inf, 3 NaN) that no integer holds, and `wnz`: the value is zero and every term that made it was -0.
struct wlane = { wv : int, wsp : range(0, 3), wnz : bool }

let wlane_zero : wlane = struct { wv = 0, wsp = 0, wnz = false }

// One ABC result: dispatched at `dispatch` by queue `q`, committed at `commit`.
// `wide`: the exact products of a floating-point ABC (only when the profile declares wide_acc; wlane_zero otherwise).
// `poison`: the superlanes of the product that an uncorrectable operand reached (microarchitecture.md sec. 15: an uncorrectable
// activation poisons every superlane of the product; an uncorrectable weight row r poisons superlane r / 16).
struct abc_entry = { dispatch : nat, commit : nat, ptype : mxm_wtype, product : vector(mxm_rows, bits(32)), wide : vector(mxm_rows, wlane), poison : bits(superlanes), q : range(0, 144) }

struct mxm_plane = {
  btype      : mxm_wtype,              // type of the buffered rows
  buf_rows   : range(0, mxm_wvals),    // stream values held in the weight buffer
  buf        : vector(mxm_wvals, wrow),
  buf_bad    : vector(mxm_wvals, bool), // the stream value of that buffer row had an uncorrectable superlane (poison)
  ready_lw   : nat,                    // cycle the last LW committed
  wtype      : mxm_wtype,              // type of the installed weights
  rows       : range(0, mxm_rows),     // installed rows (i8: streams; f16: streams / 2)
  weights    : vector(mxm_wvals, wrow),
  w_bad      : vector(mxm_wvals, bool), // the same for the installed stream values
  iw_open    : bool,                   // a stream install (IW from streams) is in progress
  iws_cycle  : nat,                    // dispatch cycle of the last IW from streams
  ready_iw   : nat,                    // cycle the last IW committed
  abc        : list(abc_entry),        // ABC results, newest first (ACC picks by commit cycle)
  acc        : vector(mxm_acc_depth, vector(mxm_rows, bits(32))),  // the accumulators
  acc_poison : vector(mxm_acc_depth, bits(superlanes)),            // the superlanes of each accumulator that carry poison
  acc128     : vector(mxm_acc_depth, vector(mxm_rows, wlane)),     // the wide accumulators (extension wide_acc); the poison above covers both forms
}

function empty_plane() -> mxm_plane = struct {
  btype = W_I8, buf_rows = 0, buf = vector_init(zero_vec), buf_bad = vector_init(false), ready_lw = 0,
  wtype = W_I8, rows = 0, weights = vector_init(zero_vec), w_bad = vector_init(false), iw_open = false, iws_cycle = 0, ready_iw = 0,
  abc = [||], acc = vector_init(vector_init(0x00000000)), acc_poison = vector_init(sail_zeros(superlanes)),
  acc128 = vector_init(vector_init(wlane_zero)),
}

register mxm : vector(2 * mxm_planes, mxm_plane) = vector_init(empty_plane())

// The independently written parts of a plane: the buffer, the weights, the ABC products and each accumulator (`acc_idx`).
enum mxm_part = {P_BUF, P_W, P_ABC, P_ACC}

struct mxm_staged = { q : range(0, 144), plane : range(0, 3), part : mxm_part, acc_idx : range(0, 31), value : mxm_plane }

register mxm_stage_list : list(mxm_staged) = [||]

// Matrix-unit check (microarchitecture.md sec. 18): the faults the harness injected into the ABC that executes on a plane in the
// current cycle (`F mxm`): plane, lane, bit (0..95) of the exact sum of that lane. The ABC consumes them; a fault left after
// the tick found no ABC and is an input error of the harness.
struct mxm_fault = { mf_pl : range(0, 3), mf_lane : nat, mf_bit : range(0, 95) }
register mxm_fault_list : list(mxm_fault) = [||]

function mxm_inject(pl : range(0, 3), lane : nat, bit : range(0, 95)) -> unit = mxm_fault_list = struct { mf_pl = pl, mf_lane = lane, mf_bit = bit } :: mxm_fault_list

// Plane i (an empty plane when the profile has no such plane).
function mxm_plane_at(i : int) -> mxm_plane = if 0 <= i & i < 2 * mxm_planes then mxm[i] else empty_plane()

function mxm_reset() -> unit = { mxm = vector_init(empty_plane()); mxm_stage_list = [||]; mxm_fault_list = [||] }

// Stage part `part` of plane `pl` from `v` (applied at the end of the cycle); accumulator `a` for the part P_ACC.
function mxm_stage_acc(pl : range(0, 3), part : mxm_part, a : range(0, 31), v : mxm_plane) -> unit =
  mxm_stage_list = struct { q = issue_queue, plane = pl, part = part, acc_idx = a, value = v } :: mxm_stage_list

function mxm_stage(pl : range(0, 3), part : mxm_part, v : mxm_plane) -> unit = mxm_stage_acc(pl, part, 0, v)

// Two FP8 ABCs of one plane in one cycle (mxm.fp8_rate 2) are both applied: the plane keeps both products, the one applied
// second (a lower queue) in front. A second entry of the same dispatch cycle is put in front of the list as it stands.
function mxm_apply(e : mxm_staged) -> unit = {
  let pl : range(0, 3) = e.plane;
  let ai : range(0, 31) = e.acc_idx;
  if pl < 2 * mxm_planes then {
    var p = mxm[pl];
    let v = e.value;
    match e.part {
      P_BUF  => { p.btype = v.btype; p.buf_rows = v.buf_rows; p.buf = v.buf; p.buf_bad = v.buf_bad; p.ready_lw = v.ready_lw },
      P_W    => { p.wtype = v.wtype; p.rows = v.rows; p.weights = v.weights; p.w_bad = v.w_bad; p.iw_open = v.iw_open;
                  p.iws_cycle = v.iws_cycle; p.ready_iw = v.ready_iw },
      P_ABC  => match (v.abc, p.abc) {
                  (n :: _, h :: _) => if n.dispatch == h.dispatch & n.q != h.q then p.abc = n :: p.abc else p.abc = v.abc,
                  _ => p.abc = v.abc,
                },
      P_ACC  => if ai < mxm_acc_depth then { p.acc[ai] = v.acc[ai]; p.acc128[ai] = v.acc128[ai]; p.acc_poison[ai] = v.acc_poison[ai] },
    };
    mxm[pl] = p
  }
}
