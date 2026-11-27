// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Instruction latencies (microarchitecture.md sec. 1; ISCA 2020 sec. III d_func / d_skew). These are
// opentsp decisions except where isa-db/provenance.yaml cites a published value,
// so they are configuration values.
//
// An instruction dispatched at t reads its operands at t + d_skew and commits its stream
// results at t + d_func. The latency is a table with one row per catalogue entry
// (entry_latency): the entry's row in timing.overrides, else its timing class's row.

// Bounds the model relies on: the pending-commit ring holds 128 cycles,
// the SRF ring 64 diagonals and the execute ring 16 cycles. Each element of a compound
// instruction schedules its own commit d_func cycles ahead; d_func <= 63 (so far below 128)
// keeps the commit ring from aliasing, and d_skew <= 15 keeps the execute ring
// (postlude/tick.sail) from aliasing. An instruction reads its operands in the cycle it
// executes, so a read needs only diagonals from the last 44 cycles (< 64; props.sail p_srf_live_window).
type d_func_t = range(0, 63)
type d_skew_t = range(0, 15)

struct latency = { d_func : d_func_t, d_skew : d_skew_t }

private let lat_cu      : latency = struct { d_func = config timing.cu.d_func,      d_skew = config timing.cu.d_skew }
private let lat_mem     : latency = struct { d_func = config timing.mem.d_func,     d_skew = config timing.mem.d_skew }
private let lat_vxm     : latency = struct { d_func = config timing.vxm.d_func,     d_skew = config timing.vxm.d_skew }
private let lat_mxm_lw  : latency = struct { d_func = config timing.mxm_lw.d_func,  d_skew = config timing.mxm_lw.d_skew }
private let lat_mxm_iw  : latency = struct { d_func = config timing.mxm_iw.d_func,  d_skew = config timing.mxm_iw.d_skew }
private let lat_mxm_abc : latency = struct { d_func = config timing.mxm_abc.d_func, d_skew = config timing.mxm_abc.d_skew }
private let lat_mxm_acc : latency = struct { d_func = config timing.mxm_acc.d_func, d_skew = config timing.mxm_acc.d_skew }
private let lat_sxm     : latency = struct { d_func = config timing.sxm.d_func,     d_skew = config timing.sxm.d_skew }
// Shift and Permute: an output lane may take its value from a later superlane, which
// the hardware stagger delivers up to superlanes - 1 cycles after superlane 0, so the class's
// d_func is at least geometry.superlanes (config/schema.json relation).
private let lat_sxm_cross : latency = struct { d_func = config timing.sxm_cross.d_func, d_skew = config timing.sxm_cross.d_skew }
private let lat_c2c     : latency = struct { d_func = config timing.c2c.d_func,     d_skew = config timing.c2c.d_skew }

// Three latencies are not a row of the table: the delay before a loaded SXM map is usable
// (LoadMap and LoadMapWide), the cycles an f16 ABC holds its plane's MACCs, and the
// shortest hold of a RuntimeDeskew. Each is a key of the profile (config/schema.json; bounds in
// tools/schema_invariants.py) and its value is recorded in isa-db/provenance.yaml.
//   timing.sxm_map_ready          a map is ready for a Permute this many cycles after the execute
//                                 cycle of its LoadMap (the stagger of ISCA20 sec. II-B: the last
//                                 superlane lags the first by superlanes - 1 cycles; ref 20, mini 1)
//   timing.mxm_abc.occupancy_f16  an f16 ABC holds its paired matrix for this many cycles;
//                                 shipped profiles use 1, research variants may choose more
//   timing.c2c.rtdeskew_floor     RuntimeDeskew t holds its queue max(t, floor) cycles (ISCA22 sec.
//                                 3.3 with the HAC-SAC drift dt = 0)
let timing_sxm_map_ready : nat = config timing.sxm_map_ready
let timing_mxm_occupancy_f16 : nat = config timing.mxm_abc.occupancy_f16
let timing_rtdeskew_floor : nat = config timing.c2c.rtdeskew_floor

enum timing_class = {T_CU, T_MEM, T_VXM, T_MXM_LW, T_MXM_IW, T_MXM_ABC, T_MXM_ACC, T_SXM, T_SXM_CROSS, T_C2C}

function latency_of(t : timing_class) -> latency =
  match t {
    T_CU      => lat_cu,
    T_MEM     => lat_mem,
    T_VXM     => lat_vxm,
    T_MXM_LW  => lat_mxm_lw,
    T_MXM_IW  => lat_mxm_iw,
    T_MXM_ABC => lat_mxm_abc,
    T_MXM_ACC => lat_mxm_acc,
    T_SXM     => lat_sxm,
    T_SXM_CROSS => lat_sxm_cross,
    T_C2C     => lat_c2c,
  }

// The catalogue's name of a timing class (isa-db/instructions/*.yaml "timing").
function timing_class_name(t : timing_class) -> string =
  match t {
    T_CU => "CU", T_MEM => "MEM", T_VXM => "VXM", T_MXM_LW => "MXM_LW", T_MXM_IW => "MXM_IW",
    T_MXM_ABC => "MXM_ABC", T_MXM_ACC => "MXM_ACC", T_SXM => "SXM", T_SXM_CROSS => "SXM_CROSS", T_C2C => "C2C",
  }

// Per-entry rows: timing.overrides lists catalogue entries (by
// name, isa-db/instructions/*.yaml "name") whose latency differs from their class's.
struct timing_override = { entry : string, d_func : d_func_t, d_skew : d_skew_t }

let timing_overrides : list(timing_override) = config timing.overrides

private function find_override(name : string, xs : list(timing_override)) -> option(latency) =
  match xs {
    [||] => None(),
    o :: rest => if o.entry == name then Some(struct { d_func = o.d_func, d_skew = o.d_skew }) else find_override(name, rest),
  }

// The latency of catalogue entry `name`, whose timing class is t.
function entry_latency(name : string, t : timing_class) -> latency =
  match find_override(name, timing_overrides) { Some(l) => l, None() => latency_of(t) }

// Per-type rows. A row of timing.overrides has no optional field (every key of a configuration struct is
// required), so the element type is part of the row's key: a row whose entry is "<Entry>.<type>" (for
// example "Max.i8"; the type is named as the assembly writes it, core/types.sail eltype_name) is the row
// of that entry at that element type. An instruction of entry E and element type T takes the row "E.T",
// else the row "E", else its class's row. An entry without an element type (timing_type_of is "") takes the
// last two.
function entry_latency_typed(name : string, tname : string, t : timing_class) -> latency =
  if tname == "" then entry_latency(name, t)
  else match find_override(name ^ "." ^ tname, timing_overrides) {
    Some(l) => l,
    None() => entry_latency(name, t),
  }
