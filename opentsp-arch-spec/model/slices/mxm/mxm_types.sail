// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// MXM operand types (declared before the instruction union).

enum mxm_wtype = {W_I8, W_F16, W_U8, W_FP8, W_BF16, W_BFP8}  // weight / activation element type (HC34 p.26: INT8, UINT8, FP16; FP8, bf16 (extension) and bfp8 (extension): implementation choices, microarchitecture.md sec. 6)
enum mxm_acctype = {A_I32, A_F32, A_X128}    // accumulator result type (x128: the wide accumulator, extension wide_acc)
enum acc_mode = {ACC_REPLACE, ACC_ACCUMULATE}
