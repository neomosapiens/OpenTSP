// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// VXM instruction operand types (declared before the instruction union).

// Binary ALU operations, opcodes 0..13 (ISCA 2020 sec. III-C: _sat / _mod variants).
enum vxm_binop = {VADD, VADD_SAT, VADD_MOD, VSUB, VSUB_SAT, VSUB_MOD, VMUL, VMUL_SAT, VMUL_MOD,
                  VMAX, VMIN, VAND, VOR, VXOR}

// Unary operations: Not 14, Neg 15, Abs 16, ReLU 20, TanH 21, Exp 22, RSqrt 23, Log 24.
enum vxm_unop = {VNOT, VNEG, VABS, VRELU, VTANH, VEXP, VRSQRT, VLOG}

// Conversions, opcodes 25..32. Each fixes its source and destination type.
enum vxm_conv = {CONV_I8_F32, CONV_F32_I8, CONV_I8_F16, CONV_F16_I8, CONV_I8_I32, CONV_I32_I8,
                 CONV_F16_F32, CONV_F32_F16}

enum cmp_mode = {CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE}
enum reduce_op = {RED_ADD, RED_MAX, RED_MIN}

// A VXM source operand: the stream and its chain bit.
// Chain bit 1: the operand is the VXM result committed at the VXM station in the cycle before
// the read (ALU chaining, ISCA20 sec. III-C); 0: the stream value that flows to the station.
union vsrc = { Vsrc : (streamref, bits(1)) }
