// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Unit tests for the C2C slice and the harness queries (slices/c2c/c2c_insts.sail,
// postlude/harness_api.sail). The unit-test platform (test/unit_tests/stubs.c) is one
// chip (id 0) with both links looped back and no vector ever arriving.

private function c2c_issue(q : queue_id, c : nat) -> unit = {
  issue_cycle = c;
  issue_unit = issuer_of(q);
  issue_queue = q;
  issue_element = 0
}

// Deskew holds its queue until the next HAC epoch boundary; RuntimeDeskew t holds it t
// cycles (no drift on one chip). Both issue on any data queue (ISCA 2022 sec. 3.2, 3.3).
$[test]
function test_c2c_deskew_occupancy() -> unit = {
  let p = c2c_hac_period;
  assert(occupancy(C2C_DESKEW(), 0) == p);
  assert(occupancy(C2C_DESKEW(), 1) == p - 1);
  assert(occupancy(C2C_DESKEW(), p - 1) == 1);
  assert(occupancy(C2C_DESKEW(), 3 * p + 5) == p - 5);
  assert(occupancy(C2C_RTDESKEW(0x0010), 7) == 16);
  assert(occupancy(C2C_RTDESKEW(0x0000), 7) == 1);
  assert(issued_by(C2C_DESKEW(), U_VXM) & issued_by(C2C_DESKEW(), U_MEM_W) & issued_by(C2C_DESKEW(), U_C2C_E_RX));
  assert(issued_by(C2C_RTDESKEW(0x0001), U_MXM_E) & issued_by(C2C_RTDESKEW(0x0001), U_SXM_W_SEL));
  reset();
  c2c_issue(88, 10);
  assert(execute_test(C2C_DESKEW()) == Issued());
  assert(execute_test(C2C_RTDESKEW(0x0004)) == Issued())
}

// Send, Receive and Transmit stay on the TX / RX queues and name one of the chip's
// c2c.links links (ISCA 2020 sec. II: sixteen x4 links); chip_id must name the chip at
// the far end of that link (on the unit-test platform every link loops back: chip 0);
// Receive with nothing arrived is a timing error.
$[test]
function test_c2c_link_instructions() -> unit = {
  assert(c2c_links >= 16 & c2c_links <= 128);   // 16 on ref and mini, 24 on test/c2c_wide
  assert(issued_by(C2C_SEND(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_W_TX));
  assert(not(issued_by(C2C_SEND(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_W_RX)));
  assert(not(issued_by(C2C_SEND(0b000000000, 0b0000000, Streamref(0b000000)), U_VXM)));
  assert(issued_by(C2C_RECEIVE(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_E_RX));
  assert(not(issued_by(C2C_RECEIVE(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_E_TX)));
  assert(issued_by(C2C_TRANSMIT(0b000000000, 0b0001111), U_C2C_E_TX) & not(issued_by(C2C_TRANSMIT(0b000000000, 0b0000000), U_C2C_E_RX)));
  assert(timing_of(C2C_SEND(0b000000000, 0b0000000, Streamref(0b000000))) == T_C2C);
  reset();
  c2c_issue(128, 10);
  assert(execute_test(C2C_RECEIVE(0b000000000, 0b0000000, Streamref(0b000001))) == Diagnostic(TIM_NO_VALUE));
  assert(execute_test(C2C_RECEIVE(0b000000000, 0b0001001, Streamref(0b000001))) == Diagnostic(TIM_NO_VALUE));
  assert(execute_test(C2C_RECEIVE(0b000000011, 0b0000000, Streamref(0b000001))) == Diagnostic(OPR_RANGE));
  c2c_issue(126, 10);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Issued());
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0001111)) == Issued());
  assert(execute_test(C2C_TRANSMIT(0b000000101, 0b0000011)) == Diagnostic(OPR_RANGE));
  // Send reads its stream at the TX gap: nothing is there.
  assert(execute_test(C2C_SEND(0b000000000, 0b0000111, Streamref(0b100000))) == Diagnostic(TIM_NO_VALUE));
  assert(c2c_notification == sail_zero_extend(0x01, 8 * vlanes))
}

// 32-bit layout with the link field: slice 31..28 = 5, opcode 27..22, chip_id 21..16,
// stream 15..10, link 9..6, bits 6..4 of the link 5..3, bits 8..6 of chip_id 2..0 (Transmit: 15..10 reserved).
$[test]
function test_c2c_link_field_layout() -> unit = {
  let w = encdec32(C2C_SEND(0b000101010, 0b0001011, Streamref(0b100011)));
  assert(w == 0x5 @ 0b000000 @ 0b101010 @ 0b100011 @ 0xB @ 0b000000);
  assert(encdec32(w) == C2C_SEND(0b000101010, 0b0001011, Streamref(0b100011)));
  assert(encdec32(0x5 @ 0b000001 @ 0b000001 @ 0b000010 @ 0x3 @ 0b000000) == C2C_RECEIVE(0b000000001, 0b0000011, Streamref(0b000010)));
  assert(encdec32(0x5 @ 0b000011 @ 0b000111 @ 0b000000 @ 0xE @ 0b000000) == C2C_TRANSMIT(0b000000111, 0b0001110));
  // the widened ids: chip 0x1A5 (9 bits), link 0x5D (7 bits)
  let x = encdec32(C2C_SEND(0b110100101, 0b1011101, Streamref(0b100011)));
  assert(x == 0x5 @ 0b000000 @ 0b100101 @ 0b100011 @ 0b1101 @ 0b101 @ 0b110);
  assert(encdec32(x) == C2C_SEND(0b110100101, 0b1011101, Streamref(0b100011)));
  assert(encdec32(0x5 @ 0b000011 @ 0b100101 @ 0b000000 @ 0b1101 @ 0b101 @ 0b110) == C2C_TRANSMIT(0b110100101, 0b1011101));
  // a link above c2c.links - 1, or a chip that is not the peer, names nothing
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b1011101)) == Diagnostic(OPR_RANGE));
  assert(execute_test(C2C_TRANSMIT(0b110100101, 0b0000000)) == Diagnostic(OPR_RANGE));
  assert(match encdec32(0x5 @ 0b000011 @ 0b000111 @ 0b000001 @ 0xE @ 0b000000) { ILLEGAL32(_) => true, _ => false })
}

// The HAC epoch is 252 cycles: an 8-bit counter with 4 values reserved (ISCA 2022 sec. 3.2
// footnote 4); the link latency bounds come from ISCA 2022 Table 2.
$[test]
function test_c2c_published_constants() -> unit = {
  assert(c2c_hac_period == 252);
  assert(occupancy(C2C_DESKEW(), 251) == 1);
  assert(occupancy(C2C_DESKEW(), 252) == 252);
  assert(c2c_link_latency_min == 209 & c2c_link_latency == 216 & c2c_link_latency_max == 228)
}

// The harness queries: an idle model has no next event; a pending commit, an open AGT
// configuration, an AGT reference and a Repeat iteration are events; work parked by a
// Sync without a Notify waits for a record, and a known release delays it.
$[test]
function test_harness_next_event() -> unit = {
  reset();
  assert(model_quiescent());
  assert(model_next_event(5) == -1);
  // A commit scheduled for cycle 30 + d_func.
  c2c_issue(0, 30);
  assert(execute_test(MEM_DIRECT(MEM_READ, sram_addr_of(0), Streamref(0b000000), 0b0)) == Issued());
  assert(not(model_quiescent()));
  assert(model_next_event(30) == 30 + latency_of(T_MEM).d_func);
  assert(model_next_event(100) == 30 + latency_of(T_MEM).d_func + 128);   // the same slot, 128 cycles on
  reset();
  // AGT reference due at 50 on queue 3; Repeat iteration due at 45 on queue 4.
  agts[3] = { agts[3] with state = AGT_RUN, next = 50 };
  assert(model_next_event(10) == 50);
  queue_repeat[4] = Some(struct { instr = NOP(0x0001), left = 2, gap = 5, next = 45 });
  assert(model_next_event(10) == 45);
  // A due-in-the-past event (parked earlier) is due on the next cycle.
  assert(model_next_event(60) == 61);
  // Queue 4 parked from 20 with release at 70: its iteration waits until 70.
  barrier_sync_test(4, 20);
  barrier_notify_test(35);
  assert(model_next_event(40) == 50);
  agts[3] = { agts[3] with state = AGT_IDLE };
  assert(model_next_event(40) == 20 + 15 + barrier_release_latency);
  // A control-channel Sync with no Notify holds every queue: nothing to wait for in time.
  reset();
  queue_repeat[4] = Some(struct { instr = NOP(0x0001), left = 2, gap = 5, next = 45 });
  barrier_sync_test(144, 20);
  assert(model_next_event(30) == -1);
  assert(model_quiescent());
  // An open AGT configuration ends at the first cycle without a configuration record.
  reset();
  agts[7] = { agts[7] with state = AGT_CONFIG };
  assert(model_next_event(12) == 13)
}

// A link accepts one vector per c2c.cycles_per_vector cycles. A Send or Transmit
// on a link that accepted a vector fewer cycles before is OTS-E-LNK-001 and sends nothing
// (the link keeps its last accepted cycle); other links are independent; a Transmit that is
// rejected for another reason does not occupy the link; reset clears the links.
$[test]
function test_c2c_link_busy() -> unit = {
  let n = c2c_cycles_per_vector;
  reset();
  c2c_issue(126, 100);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Issued());
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Diagnostic(LNK_BUSY));
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000001)) == Issued());
  assert(execute_test(C2C_SEND(0b000000000, 0b0000000, Streamref(0b000000))) == Diagnostic(LNK_BUSY));
  c2c_issue(126, 100 + n - 1);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Diagnostic(LNK_BUSY));
  c2c_issue(126, 100 + n);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Issued());
  // the busy window is measured from the last accepted vector, not from a refused one
  c2c_issue(126, 100 + 2 * n - 1);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Diagnostic(LNK_BUSY));
  c2c_issue(126, 100 + 2 * n);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Issued());
  // OPR-001 is checked first and occupies nothing; a Send that finds no data occupies nothing
  c2c_issue(126, 1000);
  assert(execute_test(C2C_TRANSMIT(0b000000011, 0b0000100)) == Diagnostic(OPR_RANGE));
  assert(execute_test(C2C_SEND(0b000000000, 0b0000100, Streamref(0b100000))) == Diagnostic(TIM_NO_VALUE));
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000100)) == Issued());
  assert(diag_code(LNK_BUSY) == "OTS-E-LNK-001" & diag_code(LNK_OVERRUN) == "OTS-E-LNK-002");
  reset();
  c2c_issue(126, 5);
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Issued());
  reset();
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b0000000)) == Issued());
  assert(c2c_rx_depth >= 1)
}

// Packet types of a link: only data and sync packets enter the receive FIFO; a packet occupies a
// link for the share of c2c.cycles_per_vector that its flits are of a vector's (rounded up), so a
// data or sync packet occupies exactly c2c.cycles_per_vector and a csr or ctrl packet no more than that.
$[test]
function test_c2c_packet_types() -> unit = {
  let n = c2c_cycles_per_vector;
  let vf = vector_flits;
  assert(c2c_ptype_queued(PKT_DATA) & c2c_ptype_queued(PKT_SYNC));
  assert(not(c2c_ptype_queued(PKT_CSR)) & not(c2c_ptype_queued(PKT_CTRL)));
  assert(c2c_packet_cycles(PKT_DATA) == n & c2c_packet_cycles(PKT_SYNC) == n);
  assert(c2c_packet_cycles(PKT_CSR) * vf >= n * 3 & (c2c_packet_cycles(PKT_CSR) - 1) * vf < n * 3);
  assert(c2c_packet_cycles(PKT_CTRL) * vf >= n * 2 & (c2c_packet_cycles(PKT_CTRL) - 1) * vf < n * 2);
  assert(c2c_packet_cycles(PKT_CSR) <= n & c2c_packet_cycles(PKT_CTRL) <= c2c_packet_cycles(PKT_CSR) & c2c_packet_cycles(PKT_CTRL) >= 1);
  assert(c2c_ptype_code(PKT_DATA) == 0 & c2c_ptype_code(PKT_CSR) == 1 & c2c_ptype_code(PKT_SYNC) == 2 & c2c_ptype_code(PKT_CTRL) == 3);
  assert(model_ptype_queued(0) & not(model_ptype_queued(1)) & model_ptype_queued(2) & not(model_ptype_queued(3)));
  assert(model_packet_cycles(0) == n & model_packet_cycles(1) == c2c_packet_cycles(PKT_CSR) & model_packet_cycles(3) == c2c_packet_cycles(PKT_CTRL))
}

// Link FEC (microarchitecture.md sec. 17): a transfer is a header flit, the payload flits and a tail flit; a payload flit
// covers 8 lanes, so superlane (f - 1) / 2; the header and the tail flit cover every superlane. The counters of the
// receiving end are CSRs 0x200 + l (corrected) and 0x210 + l (uncorrectable), cleared by reset; a vector with bad
// superlanes poisons the result of its Receive through ecc_link_bad and is reported as OTS-E-LNK-004.
$[test]
function test_link_fec() -> unit = {
  assert(link_flits == vector_flits & link_flits == 2 * superlanes + 1);
  let last : int = link_flits - 1;
  let all : int = link_pow2(superlanes) - 1;
  assert(link_flit_mask(0) == all);
  assert(link_flit_mask(last) == all);
  assert(link_flit_mask(1) == 1);
  assert(link_flit_mask(2) == (if superlanes >= 2 then 3 else 1));
  if superlanes >= 2 then {
    assert(link_flit_mask(3) == 2);
    assert(link_flit_mask(last - 1) == link_pow2(superlanes - 1));
    assert(link_flit_mask(2 * superlanes) == all)
  };
  reset();
  assert(csr_read(512 + 3) == 0x0000000000000000 & csr_read(528 + 3) == 0x0000000000000000);
  model_link_fec(10, 3, 1, false);
  model_link_fec(10, 3, 2, false);
  model_link_fec(11, 3, 5, true);
  model_link_fec(11, 4, 5, true);
  // the same events in the error registers (core/csr.sail ras_link_event): kinds 7 and 8, bits 6 and 7, two of each
  assert(csr_read(16) == (if ras_log_depth > 3 then 0x00000000000000C0 else 0x00000000000002C0));
  assert(csr_read(18) == 0x0000000000000002 & csr_read(19) == 0x0000000000000002);
  assert(csr_read(512 + 3) == 0x0000000000000002);
  assert(csr_read(528 + 3) == 0x0000000000000001);
  assert(csr_read(512 + 4) == 0x0000000000000000 & csr_read(528 + 4) == 0x0000000000000001);
  reset();
  assert(csr_read(512 + 3) == 0x0000000000000000 & csr_read(528 + 3) == 0x0000000000000000);
  // the poison of a vector with a bad superlane
  ecc_begin();
  assert(ecc_ov_get(ecc_result_ov(), 0) == 0b0000000000);
  ecc_link_bad = get_slice_int(superlanes, 1, 0);
  assert(ecc_ov_get(ecc_result_ov(), 0) == ecc_poison_sl);
  ecc_begin();
  assert(ecc_link_bad == sail_zeros(superlanes) & ecc_result_ov() == ecc_ov_zero);
  assert(diag_code(LNK_UNC) == "OTS-E-LNK-004" & diag_code(LNK_PACKET) == "OTS-E-LNK-003")
}

// DeskewRecv (, C2C opcode 9): the layout is Receive's with opcode 9, it issues on the RX queues only, it occupies its
// queue one cycle at its dispatch (the hold is the poll, postlude/tick.sail), and the operand diagnostics that do not
// depend on an arrival are its own (OTS-E-OPR-001). On the unit-test platform no vector ever arrives, so a poll never ends
// and the harness has no event for it (model_poll_event), and the model is quiescent with the queue held for good.
$[test]
function test_c2c_deskewrecv() -> unit = {
  let w = encdec32(C2C_DESKEWRECV(0b000000001, 0b0000011, Streamref(0b000010)));
  assert(w == 0x5 @ 0b001001 @ 0b000001 @ 0b000010 @ 0x3 @ 0b000000);
  assert(encdec32(w) == C2C_DESKEWRECV(0b000000001, 0b0000011, Streamref(0b000010)));
  // the widened ids: chip 0x1A5 (9 bits), link 0x5D (7 bits)
  let x = encdec32(C2C_SEND(0b110100101, 0b1011101, Streamref(0b100011)));
  assert(x == 0x5 @ 0b000000 @ 0b100101 @ 0b100011 @ 0b1101 @ 0b101 @ 0b110);
  assert(encdec32(x) == C2C_SEND(0b110100101, 0b1011101, Streamref(0b100011)));
  assert(encdec32(0x5 @ 0b000011 @ 0b100101 @ 0b000000 @ 0b1101 @ 0b101 @ 0b110) == C2C_TRANSMIT(0b110100101, 0b1011101));
  // a link above c2c.links - 1, or a chip that is not the peer, names nothing
  assert(execute_test(C2C_TRANSMIT(0b000000000, 0b1011101)) == Diagnostic(OPR_RANGE));
  assert(execute_test(C2C_TRANSMIT(0b110100101, 0b0000000)) == Diagnostic(OPR_RANGE));
  assert(issued_by(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_W_RX) & issued_by(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_E_RX));
  assert(not(issued_by(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000)), U_C2C_W_TX)) & not(issued_by(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000)), U_VXM)));
  assert(timing_of(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000))) == T_C2C);
  assert(entry_of(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000))) == "DeskewRecv");
  assert(occupancy(C2C_DESKEWRECV(0b000000000, 0b0000000, Streamref(0b000000)), 300) == 1);
  // the operand diagnostics: the link and its far end; the unit-test platform is one chip, every link loops back to chip 0
  assert(deskewrecv_diag(issuer_of(128), 0b000000000, 0b0000000, Streamref(0b000001)) == None());
  assert(deskewrecv_diag(issuer_of(128), 0b000000011, 0b0000000, Streamref(0b000001)) == Some(OPR_RANGE));
  assert(deskewrecv_diag(issuer_of(128), 0b000000000, 0b0001111, Streamref(0b000001)) == None());
  // the epoch boundaries the poll looks at
  assert(next_boundary(0) == 0 & next_boundary(1) == c2c_hac_period & next_boundary(c2c_hac_period) == c2c_hac_period);
  assert(next_boundary(c2c_hac_period + 1) == 2 * c2c_hac_period);
  // a queue with a poll: held, no event of the harness (nothing has arrived), nothing left to do
  reset();
  assert(not(polling(128)) & model_poll_event() == -1);
  queue_poll[128] = Some(struct { chip = 0b000000000, link = 0b0000000, s = Streamref(0b000001), since = 10 });
  assert(polling(128) & not(polling(129)));
  assert(model_poll_event() == -1);
  assert(model_quiescent());
  reset();
  assert(not(polling(128)))
}
