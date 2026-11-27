// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Where each functional unit sits on each stream row (ISCA 2020 Fig. 4; provenance in
// isa-db/provenance.yaml). A unit at gap g reads stream register g and writes g + 1.
// Gaps are configuration values so that a profile can change the floorplan.

enum unit_kind = {
  U_VXM,
  U_MEM_W, U_MEM_E,
  U_MXM_W, U_MXM_E,
  U_SXM_W_DST, U_SXM_W_TSP, U_SXM_W_SHF, U_SXM_W_PRM, U_SXM_W_SEL,
  U_SXM_E_DST, U_SXM_E_TSP, U_SXM_E_SHF, U_SXM_E_PRM, U_SXM_E_SEL,
  U_C2C_W_TX, U_C2C_W_RX, U_C2C_E_TX, U_C2C_E_RX,
}

private let vxm_gap : gap = config layout.vxm_gap
private let mem_up_gap_base : gap = config layout.mem_up_gap_base
private let mem_down_gap_base : gap = config layout.mem_down_gap_base
private let mxm_in_gap : gap = config layout.mxm_in_gap
private let mxm_acc_gap : gap = config layout.mxm_acc_gap
private let sxm_dst_gap : gap = config layout.sxm_dst_gap
private let sxm_turn_in_gap : gap = config layout.sxm_turn_in_gap
private let sxm_turn_out_gap : gap = config layout.sxm_turn_out_gap
private let c2c_tx_gap : gap = config layout.c2c_tx_gap
private let c2c_rx_gap : gap = config layout.c2c_rx_gap

// The row a west-side unit sees data arriving on from the centre, and the outward row.
function inward_row(east_side : bool) -> row = if east_side then RowW else RowE
function outward_row(east_side : bool) -> row = if east_side then RowE else RowW

// MEM quad q = (slice mod 44) / 4; quad 0 is nearest the VXM.
function mem_quad(s : mem_slice) -> range(0, 10) = quot_positive_round_zero(rem_positive_round_zero(s, 44), 4)

// MEM slice s exists when (s mod 44) < geometry.mem_slices_per_hemisphere. An absent
// slice has no station: it reads and writes no stream row (read_gap, write_gap), and its
// queue reports OTS-E-QUE-002 for every instruction (postlude/tick.sail dispatch).
function mem_slice_present(s : mem_slice) -> bool = rem_positive_round_zero(s, 44) < mem_slices_per_hemisphere

private function gap_or_none(g : int) -> option(gap) =
  if 0 <= g & g <= 43 then Some(g) else None()

// The gap at which unit u (instance inst: the MEM slice for MEM units, otherwise 0)
// READS row r, if any.
function read_gap(u : unit_kind, inst : mem_slice, r : row) -> option(gap) =
  if (u == U_MEM_W | u == U_MEM_E) & not(mem_slice_present(inst)) then None() else
  match (u, r) {
    (U_VXM, _)            => Some(vxm_gap),
    (U_MEM_W, RowE)       => gap_or_none(mem_up_gap_base - mem_quad(inst)),
    (U_MEM_W, RowW)       => gap_or_none(mem_down_gap_base + mem_quad(inst)),
    (U_MEM_E, RowW)       => gap_or_none(mem_up_gap_base - mem_quad(inst)),
    (U_MEM_E, RowE)       => gap_or_none(mem_down_gap_base + mem_quad(inst)),
    (U_MXM_W, RowW)       => Some(mxm_in_gap),
    (U_MXM_E, RowE)       => Some(mxm_in_gap),
    (U_SXM_W_DST, RowW)   => Some(sxm_dst_gap),
    (U_SXM_E_DST, RowE)   => Some(sxm_dst_gap),
    (U_SXM_W_TSP, RowW)   => Some(sxm_turn_in_gap),
    (U_SXM_W_PRM, RowW)   => Some(sxm_turn_in_gap),
    (U_SXM_W_SHF, RowW)   => Some(sxm_turn_in_gap),
    (U_SXM_W_SEL, RowW)   => Some(sxm_turn_in_gap),
    (U_SXM_E_TSP, RowE)   => Some(sxm_turn_in_gap),
    (U_SXM_E_PRM, RowE)   => Some(sxm_turn_in_gap),
    (U_SXM_E_SHF, RowE)   => Some(sxm_turn_in_gap),
    (U_SXM_E_SEL, RowE)   => Some(sxm_turn_in_gap),
    (U_C2C_W_TX, RowW)    => Some(c2c_tx_gap),
    (U_C2C_E_TX, RowE)    => Some(c2c_tx_gap),
    // An RX unit reads no operand stream; it fetches its instruction text (Ifetch) where
    // the TX units of its hemisphere read, on the row arriving at the chip edge.
    (U_C2C_W_RX, RowW)    => Some(c2c_tx_gap),
    (U_C2C_E_RX, RowE)    => Some(c2c_tx_gap),
    _                     => None(),
  }

// The gap at which unit u WRITES row r, if any.
function write_gap(u : unit_kind, inst : mem_slice, r : row) -> option(gap) =
  match (u, r) {
    (U_VXM, _)            => Some(vxm_gap),
    (U_MEM_W, _)          => read_gap(u, inst, r),
    (U_MEM_E, _)          => read_gap(u, inst, r),
    (U_MXM_W, RowE)       => Some(mxm_acc_gap),
    (U_MXM_E, RowW)       => Some(mxm_acc_gap),
    (U_SXM_W_DST, RowW)   => Some(sxm_dst_gap),
    (U_SXM_E_DST, RowE)   => Some(sxm_dst_gap),
    (U_SXM_W_TSP, RowE)   => Some(sxm_turn_out_gap),
    (U_SXM_W_PRM, RowE)   => Some(sxm_turn_out_gap),
    (U_SXM_W_SHF, RowE)   => Some(sxm_turn_out_gap),
    (U_SXM_W_SEL, RowE)   => Some(sxm_turn_out_gap),
    (U_SXM_E_TSP, RowW)   => Some(sxm_turn_out_gap),
    (U_SXM_E_PRM, RowW)   => Some(sxm_turn_out_gap),
    (U_SXM_E_SHF, RowW)   => Some(sxm_turn_out_gap),
    (U_SXM_E_SEL, RowW)   => Some(sxm_turn_out_gap),
    (U_C2C_W_RX, RowE)    => Some(c2c_rx_gap),
    (U_C2C_E_RX, RowW)    => Some(c2c_rx_gap),
    _                     => None(),
  }
