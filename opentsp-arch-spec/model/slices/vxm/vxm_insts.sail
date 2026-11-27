// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// VXM instructions (ISCA 2020 sec. III-C; microarchitecture.md sec. 5). Every lane
// computes independently. Operands are read at gap layout.vxm_gap in cycle
// dispatch + d_skew; results commit at dispatch + d_func (the entry's latency, core/timing.sail).
//
// 64-bit layout: slice 63..60 = 2, opcode 59..54, type 53..52, dst 51..46, lhs 45..40,
// rhs 39..34, mask 33..28, mode 27..24, type extension 23..22, chain bits 21..19 (first,
// second, third source operand), group bit 18 (Bin, Un, Mask, Select,
// Cmp; reserved elsewhere), reserved 17..0. The type
// code is extension @ type (core/types.sail). Operands an instruction does not
// use, and mode bits it does not define, must be zero. An illegal type or mode does
// not decode: legality is part of the encoding.

private function vxm_read_cycle() -> int = exec_cycle()
private function vxm_commit_cycle() -> nat = commit_at()

// Read a VXM source operand: a chained one (chain bit 1) at the station in cycle t - 1's
// commits (srf_read_chain), any other by the flow rule.
private function vxm_read(a : vsrc, w : {1, 2, 4}) -> result(planes, diag) = {
  let Vsrc(s, c) = a;
  read_planes_core(s, w, vxm_read_cycle(), c == 0b1)
}

// The text of the group bit after the type: ".sg4" when set.
mapping sg4_suffix : bits(1) <-> string = { 0b0 <-> "", 0b1 <-> ".sg4" }

// The text of a source operand: the stream, then ".chain" when the chain bit is set.
mapping chain_suffix : bits(1) <-> string = { 0b0 <-> "", 0b1 <-> ".chain" }
mapping vsrc_name : vsrc <-> string = { Vsrc(s, c) <-> stream_name(s) ^ chain_suffix(c) }

// Element l of a multi-byte operand: byte b of lane l lives in plane b.
private function elem(p : planes, w : {1, 2, 4}, l : range(0, vlanes - 1)) -> bits(32) = {
  var x : bits(32) = 0x00000000;
  if l < vlanes then foreach (b from 0 to (w - 1)) {
    x[8 * b + 7 .. 8 * b] = slice(p[b], 8 * l, 8)
  };
  x
}


// Element l of the w-byte sub-operand starting at plane o of a 4-plane group operand.
private function elem_at(p : planes, w : {1, 2, 4}, o : range(0, 3), l : range(0, vlanes - 1)) -> bits(32) = {
  var x : bits(32) = 0x00000000;
  if l < vlanes then foreach (b from 0 to (w - 1)) {
    let k = o + b;
    if k <= 3 then x[8 * b + 7 .. 8 * b] = slice(p[k], 8 * l, 8)
  };
  x
}

// The per-lane operation of a VXM instruction (microarchitecture.md sec. 5; kernels in core/numerics).
union vxm_lane_op = {
  L_BIN    : (int, eltype),          // binary opcode 0..13
  L_UN     : (int, eltype),          // unary opcode 14..24
  L_MASK   : eltype,                 // dst = mask != 0 ? lhs : 0
  L_SELECT : eltype,                 // dst = mask != 0 ? lhs : rhs
  L_CMP    : (int, eltype),          // all-ones / zero
  L_CONV   : int,                    // conversion opcode 25..32
  L_CVT    : (eltype, eltype),       // generic conversion (Convert 34)
}

private function apply(op : vxm_lane_op, a : bits(32), b : bits(32), m : bits(32)) -> bits(32) =
  match op {
    L_BIN(o, t)  => k_binary(o, t, a, b),
    L_UN(o, t)   =>
      if 21 <= o & o <= 24 then
        (if is_float(t) then fp_transcendental(o, fmt_of(t), a) else i8_transcendental(o, a))
      else k_unary(o, t, a),
    L_MASK(t)    => if k_nonzero(t, b) then a else 0x00000000,
    L_SELECT(t)  => if k_nonzero(t, m) then a else b,
    L_CMP(o, t)  => k_cmp(o, t, a, b),
    L_CONV(o)    => k_convert(o, a),
    L_CVT(st, dt) => k_convert_any(st, dt, a),
  }

// Read up to three operands of width ws, apply op per lane, schedule a result of width wd.
// Group bit g = 1 (ISCA20 sec. III-C: "Each of the ALU's 32-bit input
// operands are organized along naturally aligned quad-stream group (SG4)"): every operand is
// the 4-stream group at its stream (which must be SG4-aligned, else OTS-E-OPR-002) and the ALU
// performs 4 / ws independent operations per lane, one on each ws-byte part of the group; the
// result is the 4-stream group at dst. Only operations whose result width equals ws take it.
private function vxm_lanes(op : vxm_lane_op, dst : streamref, a : vsrc, b : option(vsrc),
                           m : option(vsrc), ws : {1, 2, 4}, wd : {1, 2, 4}, g : bits(1)) -> ExecutionResult = {
  let grp : bool = g == 0b1;
  let rw : {1, 2, 4} = if grp then 4 else ws;
  let pa = vxm_read(a, rw);
  let pb : result(planes, diag) = match b { Some(s) => vxm_read(s, rw), None() => Ok(no_planes) };
  let pm : result(planes, diag) = match m { Some(s) => vxm_read(s, rw), None() => Ok(no_planes) };
  match (pa, pb, pm) {
    (Err(e), _, _) => Diagnostic(e),
    (_, Err(e), _) => Diagnostic(e),
    (_, _, Err(e)) => Diagnostic(e),
    (Ok(xa), Ok(xb), Ok(xm)) => {
      var out : vector(vlanes, bits(32)) = vector_init(0x00000000);
      let parts : range(1, 4) = if not(grp) then 1 else if ws == 1 then 4 else if ws == 2 then 2 else 1;
      foreach (l from 0 to (vlanes - 1)) {
        var x : bits(32) = 0x00000000;
        foreach (i from 0 to (parts - 1)) {
          let o : int = i * ws;
          if 0 <= o & o <= 3 then {
            let r = apply(op, elem_at(xa, ws, o, l), elem_at(xb, ws, o, l), elem_at(xm, ws, o, l));
            foreach (j from 0 to (ws - 1)) {
              let k = o + j;
              if k <= 3 then x[8 * k + 7 .. 8 * k] = r[8 * j + 7 .. 8 * j]
            }
          }
        };
        out[l] = if grp then x else apply(op, elem(xa, ws, l), elem(xb, ws, l), elem(xm, ws, l))
      };
      match write_planes(dst, words_to_planes(out), if grp then 4 else wd, vxm_commit_cycle()) {
        Ok(()) => Issued(),
        Err(e) => Diagnostic(e),
      }
    },
  }
}

private function vxm_unit(u : unit_kind) -> bool = u == U_VXM

// A defined type code (the type-extension bit, the extension and type fields of an encoding).
// Codes 11 (e4m3) and 12 (e5m2) are the FP8 types; 13..15 stay reserved. Code 16 (bf16, the type-extension bit set)
// is defined only in a profile that declares the extension `bf16`; 17..31 stay reserved.
function ty_valid(tz : bits(1), tx : bits(2), ty : bits(2)) -> bool =
  if tz == 0b0 then (tx == 0b00 | tx == 0b01 | tx == 0b10 | (tx == 0b11 & ty == 0b00))
  else tx == 0b00 & ty == 0b00 & ext_bf16

// The 5-bit type code split into its three fields: (type-extension bit 17, extension bits 23..22, type bits 53..52).
mapping ty_split : eltype <-> (bits(1), bits(2), bits(2)) = {
  I8  <-> (0b0, 0b00, 0b00), F16 <-> (0b0, 0b00, 0b01), I32 <-> (0b0, 0b00, 0b10), F32 <-> (0b0, 0b00, 0b11),
  U8  <-> (0b0, 0b01, 0b00), I16 <-> (0b0, 0b01, 0b01), U16 <-> (0b0, 0b01, 0b10), U32 <-> (0b0, 0b01, 0b11),
  B8  <-> (0b0, 0b10, 0b00), B16 <-> (0b0, 0b10, 0b01), B32 <-> (0b0, 0b10, 0b10),
  E4M3 <-> (0b0, 0b10, 0b11), E5M2 <-> (0b0, 0b11, 0b00),
  BF16 <-> (0b1, 0b00, 0b00),
}

// The destination type of a Convert: the type-extension bit 16 and the 4-bit code of bits 27..24.
mapping dst_split : eltype <-> (bits(1), bits(4)) = {
  I8  <-> (0b0, 0b0000), F16 <-> (0b0, 0b0001), I32 <-> (0b0, 0b0010), F32 <-> (0b0, 0b0011),
  U8  <-> (0b0, 0b0100), I16 <-> (0b0, 0b0101), U16 <-> (0b0, 0b0110), U32 <-> (0b0, 0b0111),
  B8  <-> (0b0, 0b1000), B16 <-> (0b0, 0b1001), B32 <-> (0b0, 0b1010),
  E4M3 <-> (0b0, 0b1011), E5M2 <-> (0b0, 0b1100),
  BF16 <-> (0b1, 0b0000),
}

// ---------------------------------------------------------------------------------------
// Binary operations, opcodes 0..13.

mapping encdec_vxm_binop : vxm_binop <-> bits(6) = {
  VADD     <-> 0b000000, VADD_SAT <-> 0b000001, VADD_MOD <-> 0b000010,
  VSUB     <-> 0b000011, VSUB_SAT <-> 0b000100, VSUB_MOD <-> 0b000101,
  VMUL     <-> 0b000110, VMUL_SAT <-> 0b000111, VMUL_MOD <-> 0b001000,
  VMAX     <-> 0b001001, VMIN     <-> 0b001010,
  VAND     <-> 0b001011, VOR      <-> 0b001100, VXOR     <-> 0b001101,
}

// Saturating/modular variants are integer-only; bitwise operations take integers and
// booleans; arithmetic takes integers and floats, not booleans (microarchitecture.md sec. 5;
// HC34 p.26 types).
function is_int(t : eltype) -> bool = not(is_float(t)) & not(is_bool(t))

function binop_legal(op : vxm_binop, t : eltype) -> bool =
  match op {
    VADD => not(is_bool(t)), VSUB => not(is_bool(t)), VMUL => not(is_bool(t)),
    VMAX => not(is_bool(t)), VMIN => not(is_bool(t)),
    VAND => not(is_float(t)), VOR => not(is_float(t)), VXOR => not(is_float(t)),
    _    => is_int(t),
  }

union clause instruction = VXM_BIN : (vxm_binop, eltype, streamref, vsrc, vsrc, bits(1))

mapping clause encdec64 = VXM_BIN(op, ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca), Vsrc(Streamref(b), cb), g)
  <-> 0x2 @ encdec_vxm_binop(op) @ ty @ d @ a @ b @ 0b000000 @ 0x0 @ tx @ ca @ cb @ 0b0 @ g @ tz @ 0b00000000000000000
  when (if ty_valid(tz, tx, ty) then ((binop_legal(op, ty_split(tz, tx, ty)))) else false)

function clause execute VXM_BIN(op, t, d, a, b, g) = {
  let w = eltype_width(t);
  vxm_lanes(L_BIN(unsigned(encdec_vxm_binop(op)), t), d, a, Some(b), None(), w, w, g)
}

mapping vxm_binop_mnemonic : vxm_binop <-> string = {
  VADD <-> "add", VADD_SAT <-> "add_sat", VADD_MOD <-> "add_mod",
  VSUB <-> "sub", VSUB_SAT <-> "sub_sat", VSUB_MOD <-> "sub_mod",
  VMUL <-> "mul", VMUL_SAT <-> "mul_sat", VMUL_MOD <-> "mul_mod",
  VMAX <-> "max", VMIN <-> "min", VAND <-> "and", VOR <-> "or", VXOR <-> "xor",
}

mapping clause assembly = VXM_BIN(op, t, d, a, b, g)
  <-> vxm_binop_mnemonic(op) ^ "." ^ eltype_name(t) ^ sg4_suffix(g) ^ spc() ^ stream_name(d) ^ sep()
    ^ vsrc_name(a) ^ sep() ^ vsrc_name(b)

// ---------------------------------------------------------------------------------------
// Unary operations: Not 14, Neg 15, Abs 16, ReLU 20, TanH 21, Exp 22, RSqrt 23, Log 24.

mapping encdec_vxm_unop : vxm_unop <-> bits(6) = {
  VNOT <-> 0b001110, VNEG <-> 0b001111, VABS <-> 0b010000, VRELU <-> 0b010100,
  VTANH <-> 0b010101, VEXP <-> 0b010110, VRSQRT <-> 0b010111, VLOG <-> 0b011000,
}

function unop_legal(op : vxm_unop, t : eltype) -> bool =
  match op {
    VNOT  => not(is_float(t)),
    VNEG  => (is_int(t) & not(is_unsigned(t))) | is_float(t),
    VABS  => (is_int(t) & not(is_unsigned(t))) | is_float(t),
    VRELU => (is_int(t) & not(is_unsigned(t))) | is_float(t),
    _     => t == I8 | t == F16 | t == F32,     // transcendentals: i8, f16, f32 (not the FP8 types)
  }

union clause instruction = VXM_UN : (vxm_unop, eltype, streamref, vsrc, bits(1))

mapping clause encdec64 = VXM_UN(op, ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca), g)
  <-> 0x2 @ encdec_vxm_unop(op) @ ty @ d @ a @ 0b000000 @ 0b000000 @ 0x0 @ tx @ ca @ 0b00 @ g @ tz @ 0b00000000000000000
  when (if ty_valid(tz, tx, ty) then ((unop_legal(op, ty_split(tz, tx, ty)))) else false)

function clause execute VXM_UN(op, t, d, a, g) = {
  let w = eltype_width(t);
  vxm_lanes(L_UN(unsigned(encdec_vxm_unop(op)), t), d, a, None(), None(), w, w, g)
}

mapping vxm_unop_mnemonic : vxm_unop <-> string = {
  VNOT <-> "not", VNEG <-> "neg", VABS <-> "abs", VRELU <-> "relu",
  VTANH <-> "tanh", VEXP <-> "exp", VRSQRT <-> "rsqrt", VLOG <-> "log",
}

mapping clause assembly = VXM_UN(op, t, d, a, g)
  <-> vxm_unop_mnemonic(op) ^ "." ^ eltype_name(t) ^ sg4_suffix(g) ^ spc() ^ stream_name(d) ^ sep() ^ vsrc_name(a)

// ---------------------------------------------------------------------------------------
// Mask 17 (dst = rhs != 0 ? lhs : 0), Select 18 (dst = mask != 0 ? lhs : rhs).

union clause instruction = VXM_MASK : (eltype, streamref, vsrc, vsrc, bits(1))
union clause instruction = VXM_SELECT : (eltype, streamref, vsrc, vsrc, vsrc, bits(1))

mapping clause encdec64 = VXM_MASK(ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca), Vsrc(Streamref(m), cb), g)
  <-> 0x2 @ 0b010001 @ ty @ d @ a @ m @ 0b000000 @ 0x0 @ tx @ ca @ cb @ 0b0 @ g @ tz @ 0b00000000000000000
  when ty_valid(tz, tx, ty)
mapping clause encdec64 = VXM_SELECT(ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca), Vsrc(Streamref(b), cb), Vsrc(Streamref(m), cm), g)
  <-> 0x2 @ 0b010010 @ ty @ d @ a @ b @ m @ 0x0 @ tx @ ca @ cb @ cm @ g @ tz @ 0b00000000000000000
  when ty_valid(tz, tx, ty)

function clause execute VXM_MASK(t, d, a, m, g) = {
  let w = eltype_width(t);
  vxm_lanes(L_MASK(t), d, a, Some(m), None(), w, w, g)
}
function clause execute VXM_SELECT(t, d, a, b, m, g) = {
  let w = eltype_width(t);
  vxm_lanes(L_SELECT(t), d, a, Some(b), Some(m), w, w, g)
}

mapping clause assembly = VXM_MASK(t, d, a, m, g)
  <-> "mask." ^ eltype_name(t) ^ sg4_suffix(g) ^ spc() ^ stream_name(d) ^ sep() ^ vsrc_name(a) ^ sep() ^ vsrc_name(m)
mapping clause assembly = VXM_SELECT(t, d, a, b, m, g)
  <-> "select." ^ eltype_name(t) ^ sg4_suffix(g) ^ spc() ^ stream_name(d) ^ sep() ^ vsrc_name(a) ^ sep()
    ^ vsrc_name(b) ^ sep() ^ vsrc_name(m)

// ---------------------------------------------------------------------------------------
// Cmp 19, mode eq..ge = 0..5 (modes 6..15 do not decode).

// Booleans compare for equality only.
function cmp_legal(mode : cmp_mode, t : eltype) -> bool =
  not(is_bool(t)) | mode == CMP_EQ | mode == CMP_NE

mapping encdec_cmp_mode : cmp_mode <-> bits(4) = {
  CMP_EQ <-> 0x0, CMP_NE <-> 0x1, CMP_LT <-> 0x2, CMP_LE <-> 0x3, CMP_GT <-> 0x4, CMP_GE <-> 0x5,
}

union clause instruction = VXM_CMP : (cmp_mode, eltype, streamref, vsrc, vsrc, bits(1))

mapping clause encdec64 = VXM_CMP(mode, ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca), Vsrc(Streamref(b), cb), g)
  <-> 0x2 @ 0b010011 @ ty @ d @ a @ b @ 0b000000 @ encdec_cmp_mode(mode) @ tx @ ca @ cb @ 0b0 @ g @ tz @ 0b00000000000000000
  when (if ty_valid(tz, tx, ty) then ((cmp_legal(mode, ty_split(tz, tx, ty)))) else false)

function clause execute VXM_CMP(mode, t, d, a, b, g) = {
  let w = eltype_width(t);
  vxm_lanes(L_CMP(unsigned(encdec_cmp_mode(mode)), t), d, a, Some(b), None(), w, w, g)
}

mapping cmp_mnemonic : cmp_mode <-> string = {
  CMP_EQ <-> "eq", CMP_NE <-> "ne", CMP_LT <-> "lt", CMP_LE <-> "le", CMP_GT <-> "gt", CMP_GE <-> "ge",
}

mapping clause assembly = VXM_CMP(mode, t, d, a, b, g)
  <-> "cmp." ^ cmp_mnemonic(mode) ^ "." ^ eltype_name(t) ^ sg4_suffix(g) ^ spc() ^ stream_name(d) ^ sep()
    ^ vsrc_name(a) ^ sep() ^ vsrc_name(b)

// ---------------------------------------------------------------------------------------
// Conversions 25..32: the type field must equal the source type.

mapping encdec_vxm_conv : vxm_conv <-> bits(6) = {
  CONV_I8_F32  <-> 0b011001, CONV_F32_I8  <-> 0b011010, CONV_I8_F16  <-> 0b011011,
  CONV_F16_I8  <-> 0b011100, CONV_I8_I32  <-> 0b011101, CONV_I32_I8  <-> 0b011110,
  CONV_F16_F32 <-> 0b011111, CONV_F32_F16 <-> 0b100000,
}

function conv_types(c : vxm_conv) -> (eltype, eltype) =
  match c {
    CONV_I8_F32  => (I8, F32),  CONV_F32_I8  => (F32, I8),
    CONV_I8_F16  => (I8, F16),  CONV_F16_I8  => (F16, I8),
    CONV_I8_I32  => (I8, I32),  CONV_I32_I8  => (I32, I8),
    CONV_F16_F32 => (F16, F32), CONV_F32_F16 => (F32, F16),
  }

private function conv_src_type(c : vxm_conv) -> eltype = { let (s, _) = conv_types(c); s }

// The source type is redundant with the opcode but is still encoded;
// the instruction carries it so that the mapping stays a bijection, and the guard
// requires it to match.
union clause instruction = VXM_CONV : (vxm_conv, eltype, streamref, vsrc)

mapping clause encdec64 = VXM_CONV(c, ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca))
  <-> 0x2 @ encdec_vxm_conv(c) @ ty @ d @ a @ 0b000000 @ 0b000000 @ 0x0 @ tx @ ca @ 0b00 @ 0b0 @ tz @ 0b00000000000000000
  when (if ty_valid(tz, tx, ty) then ((ty_split(tz, tx, ty) == conv_src_type(c))) else false)

function clause execute VXM_CONV(c, _, d, a) = {
  let (s, t) = conv_types(c);
  vxm_lanes(L_CONV(unsigned(encdec_vxm_conv(c))), d, a, None(), None(), eltype_width(s), eltype_width(t), 0b0)
}

mapping conv_mnemonic : vxm_conv <-> string = {
  CONV_I8_F32 <-> "conv_i8_f32", CONV_F32_I8 <-> "conv_f32_i8", CONV_I8_F16 <-> "conv_i8_f16",
  CONV_F16_I8 <-> "conv_f16_i8", CONV_I8_I32 <-> "conv_i8_i32", CONV_I32_I8 <-> "conv_i32_i8",
  CONV_F16_F32 <-> "conv_f16_f32", CONV_F32_F16 <-> "conv_f32_f16",
}

mapping clause assembly = VXM_CONV(c, t, d, a)
  <-> conv_mnemonic(c) ^ "." ^ eltype_name(t) ^ spc() ^ stream_name(d) ^ sep() ^ vsrc_name(a)

// ---------------------------------------------------------------------------------------
// Convert 34: generic conversion. The type field gives the source type, mode bits
// 27..24 the destination type's code (core/types.sail). Rules in k_convert_any.

union clause instruction = VXM_CVT : (eltype, eltype, streamref, vsrc)

mapping clause encdec64 = VXM_CVT(ty_split(tz, tx, ty), dst_split(dz, dc), Streamref(d), Vsrc(Streamref(a), ca))
  <-> 0x2 @ 0b100010 @ ty @ d @ a @ 0b000000 @ 0b000000 @ dc @ tx @ ca @ 0b00 @ 0b0 @ tz @ dz @ 0b0000000000000000
  when ty_valid(tz, tx, ty) & ty_valid(dz, dc[3 .. 2], dc[1 .. 0])

function clause execute VXM_CVT(st, dt, d, a) =
  vxm_lanes(L_CVT(st, dt), d, a, None(), None(), eltype_width(st), eltype_width(dt), 0b0)

mapping clause assembly = VXM_CVT(st, dt, d, a)
  <-> "convert." ^ eltype_name(st) ^ "." ^ eltype_name(dt) ^ spc() ^ stream_name(d) ^ sep() ^ vsrc_name(a)

function clause timing_of VXM_CVT(_) = T_VXM
function clause issued_by(VXM_CVT(_), u) = vxm_unit(u)

// ---------------------------------------------------------------------------------------
// Reduce 33, op add/max/min = 0..2: per 16-lane superlane group, lane 16g gets the result
// over lanes 16g..16g+15 in ascending order; the other lanes are zero.

mapping encdec_reduce_op : reduce_op <-> bits(4) = { RED_ADD <-> 0x0, RED_MAX <-> 0x1, RED_MIN <-> 0x2 }

union clause instruction = VXM_REDUCE : (reduce_op, eltype, streamref, vsrc)

mapping clause encdec64 = VXM_REDUCE(op, ty_split(tz, tx, ty), Streamref(d), Vsrc(Streamref(a), ca))
  <-> 0x2 @ 0b100001 @ ty @ d @ a @ 0b000000 @ 0b000000 @ encdec_reduce_op(op) @ tx @ ca @ 0b00 @ 0b0 @ tz @ 0b00000000000000000
  when (if ty_valid(tz, tx, ty) then ((not(is_bool(ty_split(tz, tx, ty))))) else false)

function clause execute VXM_REDUCE(op, t, d, a) = {
  let w = eltype_width(t);
  let kop : int = match op { RED_ADD => 0, RED_MAX => 9, RED_MIN => 10 };
  match vxm_read(a, w) {
    Err(e) => Diagnostic(e),
    Ok(p)  => {
      var out : vector(vlanes, bits(32)) = vector_init(0x00000000);
      foreach (g from 0 to (superlanes - 1)) {
        var acc : bits(32) = elem(p, w, 16 * g);
        foreach (j from 1 to 15) { acc = k_binary(kop, t, acc, elem(p, w, 16 * g + j)) };
        out[16 * g] = acc
      };
      match write_planes(d, words_to_planes(out), w, vxm_commit_cycle()) {
        Ok(()) => Issued(),
        Err(e) => Diagnostic(e),
      }
    },
  }
}

mapping reduce_mnemonic : reduce_op <-> string = { RED_ADD <-> "add", RED_MAX <-> "max", RED_MIN <-> "min" }

mapping clause assembly = VXM_REDUCE(op, t, d, a)
  <-> "reduce." ^ reduce_mnemonic(op) ^ "." ^ eltype_name(t) ^ spc() ^ stream_name(d) ^ sep() ^ vsrc_name(a)

// ---------------------------------------------------------------------------------------

function clause timing_of VXM_BIN(_) = T_VXM
function clause timing_of VXM_UN(_) = T_VXM
function clause timing_of VXM_MASK(_) = T_VXM
function clause timing_of VXM_SELECT(_) = T_VXM
function clause timing_of VXM_CMP(_) = T_VXM
function clause timing_of VXM_CONV(_) = T_VXM
function clause timing_of VXM_REDUCE(_) = T_VXM

private mapping binop_entry : vxm_binop <-> string = {
  VADD <-> "Add", VADD_SAT <-> "Add_sat", VADD_MOD <-> "Add_mod", VSUB <-> "Sub", VSUB_SAT <-> "Sub_sat",
  VSUB_MOD <-> "Sub_mod", VMUL <-> "Mul", VMUL_SAT <-> "Mul_sat", VMUL_MOD <-> "Mul_mod", VMAX <-> "Max",
  VMIN <-> "Min", VAND <-> "And", VOR <-> "Or", VXOR <-> "Xor",
}
private mapping unop_entry : vxm_unop <-> string = {
  VNOT <-> "Not", VNEG <-> "Neg", VABS <-> "Abs", VRELU <-> "ReLU", VTANH <-> "TanH", VEXP <-> "Exp",
  VRSQRT <-> "RSqrt", VLOG <-> "Log",
}
private mapping conv_entry : vxm_conv <-> string = {
  CONV_I8_F32 <-> "Conv_I8_F32", CONV_F32_I8 <-> "Conv_F32_I8", CONV_I8_F16 <-> "Conv_I8_F16",
  CONV_F16_I8 <-> "Conv_F16_I8", CONV_I8_I32 <-> "Conv_I8_I32", CONV_I32_I8 <-> "Conv_I32_I8",
  CONV_F16_F32 <-> "Conv_F16_F32", CONV_F32_F16 <-> "Conv_F32_F16",
}
function clause entry_of VXM_BIN(op, _, _, _, _, _) = binop_entry(op)
function clause entry_of VXM_UN(op, _, _, _, _) = unop_entry(op)
function clause entry_of VXM_MASK(_) = "Mask"
function clause entry_of VXM_SELECT(_) = "Select"
function clause entry_of VXM_CMP(_) = "Cmp"
function clause entry_of VXM_CONV(op, _, _, _) = conv_entry(op)
function clause entry_of VXM_CVT(_) = "Convert"
function clause entry_of VXM_REDUCE(_) = "Reduce"

// The element type of a VXM instruction is its `type` operand (the source type of Convert).
function clause timing_type_of VXM_BIN(_, t, _, _, _, _) = eltype_name(t)
function clause timing_type_of VXM_UN(_, t, _, _, _) = eltype_name(t)
function clause timing_type_of VXM_MASK(t, _, _, _, _) = eltype_name(t)
function clause timing_type_of VXM_SELECT(t, _, _, _, _, _) = eltype_name(t)
function clause timing_type_of VXM_CMP(_, t, _, _, _, _) = eltype_name(t)
function clause timing_type_of VXM_CONV(_, t, _, _) = eltype_name(t)
function clause timing_type_of VXM_CVT(t, _, _, _) = eltype_name(t)
function clause timing_type_of VXM_REDUCE(_, t, _, _) = eltype_name(t)

// Whether the entry `name` takes element type t: the legality its encodings above require (the catalogue's
// `type` codes of the entry). A timing row keyed by an element type is only meaningful for such a pair.
function entry_takes_type(name : string, t : eltype) -> bool =
  if binop_entry_backwards_matches(name) then binop_legal(binop_entry_backwards(name), t)
  else if unop_entry_backwards_matches(name) then unop_legal(unop_entry_backwards(name), t)
  else if conv_entry_backwards_matches(name) then conv_src_type(conv_entry_backwards(name)) == t
  else if name == "Mask" | name == "Select" | name == "Cmp" | name == "Convert" then true
  else if name == "Reduce" then not(is_bool(t))
  else false

function clause issued_by(VXM_BIN(_), u) = vxm_unit(u)
function clause issued_by(VXM_UN(_), u) = vxm_unit(u)
function clause issued_by(VXM_MASK(_), u) = vxm_unit(u)
function clause issued_by(VXM_SELECT(_), u) = vxm_unit(u)
function clause issued_by(VXM_CMP(_), u) = vxm_unit(u)
function clause issued_by(VXM_CONV(_), u) = vxm_unit(u)
function clause issued_by(VXM_REDUCE(_), u) = vxm_unit(u)
