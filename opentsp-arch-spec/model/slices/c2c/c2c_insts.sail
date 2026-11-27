// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// C2C: chip-to-chip links (ISCA 2020 sec. II and Table I: Send, Receive, Deskew; ISCA 2022
// sec. 2.2, 2.3 and 3: links, hardware-aligned counter (HAC), Transmit, RuntimeDeskew).
//
// Links. A chip has c2c.links links (ref: 16; ISCA 2020 sec. II: "a total of sixteen (16)
// x4 links"; ISCA 2022 sec. 2.2 partitions the pin bandwidth into 7 local and 4 global
// links, recorded as a conflict in isa-db/provenance.yaml). Send, Receive and Transmit name
// the link (field `link`); they issue on the eight C2C queues of the SXM blocks (west TX
// 126..127, RX 128..129; east TX 140..141, RX 142..143: HC34 p.31 draws TX and RX inside the
// SXM; core/queues.sail), and any TX (RX) queue may use any link. The model sees only its own end of a
// link (core/callbacks.sail: c2c_peer, c2c_send, c2c_recv_ready, c2c_recv); the harness
// wires the links and delays every vector by the link's latency, a per-link value in
// [c2c.link_latency_min, c2c.link_latency_max] = [209, 228] (ISCA 2022 Table 2), nominal
// c2c.link_latency = 216 (the Table 2 mean). A link never applies back pressure (ISCA 2022
// sec. 2.3): software schedules both ends, and Deskew makes the schedule independent of
// the actual latency (below). Two more limits apply to a real link: a real link has: bandwidth,
// c2c.cycles_per_vector (a link accepts one vector per that many cycles; Send and Transmit
// on a busy link are OTS-E-LNK-001), and the receive buffer, c2c.rx_depth (the harness
// drops a vector that arrives at a full FIFO and reports OTS-E-LNK-002 for the receiving
// chip). The chip keeps the time of each link's last accepted vector; the FIFO is the
// harness's.
//
//   Send chip, link, s     (TX) reads stream s at the TX gap at dispatch + d_skew and puts
//                          the vector on the link in that cycle. A link that accepted a vector
//                          fewer than c2c.cycles_per_vector cycles before is busy: OTS-E-LNK-001,
//                          nothing is read or sent (checked after OPR-001, before the stream
//                          is read). Two sends on one link in one cycle: the second, in
//                          execution order, is busy. An instruction that raises any other
//                          diagnostic sends nothing and does not occupy the link.
//   Receive chip, link, s  (RX) at dispatch + d_skew takes the oldest vector that has arrived
//                          on the link and writes it to stream s at the RX gap, committing
//                          at dispatch + d_func. No vector arrived yet: OTS-E-TIM-001 (nothing written).
//                          ISCA 2020 Table I: "Receive a 320-byte vector, emplacing it in
//                          main memory"; the model writes a stream and a MEM Write stores
//                          it (the same data path as every slice). A vector whose transfer had an
//                          uncorrectable FEC error is still written, with the superlanes of the bad
//                          flits poisoned, and the Receive reports OTS-E-LNK-004 after its other checks
//                          (link FEC: microarchitecture.md sec. 17).
//   Transmit chip, link    (TX) sends a notification vector (lane 0 = 1, other lanes 0) on
//                          the link, like Send, without reading a stream; the peer consumes
//                          it with Receive (ISCA 2022 sec. 3.2, Fig. 7(b)). Send's packet has
//                          type data, Transmit's type sync (packet types: below, c2c_ptype).
//   Deskew           (any data queue) pauses issue on its queue until the next HAC epoch
//                          boundary (ISCA 2022 sec. 3.2: "it will pause issuing subsequent
//                          instructions on that functional unit until the next time the
//                          HAC overflows"). Model: the HAC is the cycle number modulo the
//                          period P = c2c.hac_period = 252 (ISCA 2022 footnote 4: "the
//                          period is the epoch length or 252 clock cycles. The HAC is an
//                          8-bit counter, but 4 values are reserved for special control
//                          codes."); every modelled chip's HAC is aligned. Deskew at c
//                          holds its queue for P - (c mod P) cycles, so the next record may
//                          issue at the next multiple of P. Because every link latency is
//                          below P, a vector sent at an epoch boundary b has arrived by
//                          b + P whatever the link's latency: a Receive issued after a
//                          Deskew that follows the arrival window is deterministic (Fig.
//                          7(b)).
//   RuntimeDeskew t  (any data queue) delays its queue t cycles plus or minus dt, the
//                          HAC - SAC drift (ISCA 2022 sec. 3.3). Modelled chips have no
//                          drift (dt = 0): it holds its queue max(t, timing.c2c.rtdeskew_floor) cycles (floor 1).
//   DeskewRecv chip, link, s  (RX) the polling loop at program start (ISCA 2022 sec. 3.2, Fig. 7(b)): "the child device is
//                          locally placed into a polling synchronization loop. In the polling loop, at each epoch boundary
//                          (shown as HAC=0), the child device will wait to receive a vector from the parent device. If that
//                          vector has not been transmitted yet, the loop will continue." The child exits the loop floor(L /
//                          period) + 1 clock epochs after the parent's Transmit. The ISA has no branch, so the loop is one
//                          instruction (an implementation choice, isa-db/provenance.yaml C2C.DeskewRecv): it holds its
//                          queue until the next epoch boundary after its dispatch (as Deskew); then, if a vector has
//                          arrived on the link by that cycle, it is a Receive with the same operands dispatched at that
//                          boundary, else it holds the queue until the following boundary and tests again. Nothing is
//                          reported while it waits, and a vector that never comes leaves the queue held for the rest of
//                          the run, as a Sync that no Notify releases. The diagnostics of Receive that do not depend on the
//                          arrival (OTS-E-OPR-001, OTS-E-DIR-001) are reported at the dispatch and the instruction then
//                          holds nothing. The test is at the boundary only: a vector that arrives between two boundaries
//                          is taken at the next one. The harness wakes the chip at those boundaries (c2c_next_arrival,
//                          model_poll_event).
//
// chip_id names the destination (Send, Transmit) or source (Receive) chip. When it is not
// the chip at the far end of the named link (c2c_peer; -1 for an unconnected link) or
// link >= c2c.links, the instruction is OTS-E-OPR-001 and has no effect.
//
// 32-bit layout: slice 31..28 = 5, opcode 27..22, chip_id 21..16 (bits 5..0 of the chip) and 2..0 (bits 8..6), stream
// 15..10 (Transmit: 0), link 9..6 (bits 3..0) and 5..3 (bits 6..4). Deskew has no operands; RuntimeDeskew has t in 15..0;
// DeskewRecv (opcode 9) has the fields of Receive.

let c2c_links : {'n, 1 <= 'n <= 128. int('n)} = {
  let n : int = config c2c.links;
  assert(1 <= n & n <= 128, "c2c.links must be 1..128 (the link field is 7 bits)");
  let m : {'n, 1 <= 'n <= 128. int('n)} = if 1 <= n & n <= 128 then n else 128;
  m
}
// Published latency bounds and nominal value (ISCA 2022 Table 2). The harness applies
// them (c_emulator/tsp_sim.c --link-latency); the model only names them.
let c2c_link_latency : nat = config c2c.link_latency
let c2c_link_latency_min : nat = config c2c.link_latency_min
let c2c_link_latency_max : nat = config c2c.link_latency_max
// A link accepts one vector per c2c.cycles_per_vector cycles (24 in both profiles, derived:
// a 320-byte vector is 2560 bits; four lanes at the 25 Gb/s operating rate
// of ISCA22 sec. 2.3 footnote 2 are 100 Gb/s; at a clock of about 900 MHz that is 23.04 cycles,
// rounded up to 24. The 30 Gb/s of ISCA20 sec. II is the capability: it gives 19.2, which a profile of a chip at full rate rounds up to 20).
// Each link's receive FIFO holds c2c.rx_depth vectors (harness; design
// choice 16).
let c2c_cycles_per_vector : {'n, 'n > 0. int('n)} = {
  let n : int = config c2c.cycles_per_vector;
  assert(n > 0, "c2c.cycles_per_vector must be positive");
  let m : {'n, 'n > 0. int('n)} = if n > 0 then n else 1;
  m
}
let c2c_rx_depth : nat = config c2c.rx_depth
let c2c_hac_period : {'p, 'p > 0. int('p)} = {
  let p : int = config c2c.hac_period;
  assert(p > 0, "c2c.hac_period must be positive");
  let q : {'p, 'p > 0. int('p)} = if p > 0 then p else 1;
  q
}

enum c2c_op = {C2C_OP_SEND, C2C_OP_RECEIVE, C2C_OP_DESKEW, C2C_OP_TRANSMIT, C2C_OP_RTDESKEW, C2C_OP_PCIRECV, C2C_OP_PCISEND, C2C_OP_HOSTWAIT, C2C_OP_HOSTNOTIFY, C2C_OP_DESKEWRECV}

private mapping encdec_c2c_op : c2c_op <-> bits(6) = {
  C2C_OP_SEND    <-> 0b000000,
  C2C_OP_RECEIVE <-> 0b000001,
  C2C_OP_DESKEW  <-> 0b000010,
  C2C_OP_TRANSMIT <-> 0b000011,
  C2C_OP_RTDESKEW <-> 0b000100,
  C2C_OP_PCIRECV <-> 0b000101,
  C2C_OP_PCISEND <-> 0b000110,
  C2C_OP_HOSTWAIT <-> 0b000111,
  C2C_OP_HOSTNOTIFY <-> 0b001000,
  C2C_OP_DESKEWRECV <-> 0b001001,
}

union clause instruction = C2C_SEND : (bits(9), bits(7), streamref)
union clause instruction = C2C_RECEIVE : (bits(9), bits(7), streamref)
union clause instruction = C2C_DESKEW : unit
union clause instruction = C2C_TRANSMIT : (bits(9), bits(7))
union clause instruction = C2C_RTDESKEW : bits(16)
union clause instruction = C2C_DESKEWRECV : (bits(9), bits(7), streamref)

// The host I/O instructions (; the PCIe host interface sits in the C2C slice, so they
// issue on its queues and no queue id changes):
//   PciReceive s    (RX) takes the oldest vector the host's DMA engine has pushed (the ingress
//                         FIFO, core/csr.sail) and writes it to stream s: the published
//                         incoming streams are S24..S31 (the upper eight), either row. Nothing
//                         arrived: OTS-E-TIM-001, as Receive.
//   PciSend s       (TX) reads stream s and appends it to the host's egress list; at most one
//                         per host.cycles_per_vector cycles (OTS-E-HIO-001 otherwise, no effect).
//   HostWait i      (any data queue) parks its queue until the host rings doorbell i
//                         (CSR 0x300 + i, i < host.doorbells, else OTS-E-OPR-001), which it
//                         takes: the next parcel dispatches the cycle after the ring.
//   HostNotify n    (any data queue) raises MSI-X vector n (1..15) at its execute cycle.
// 32-bit layout: slice 31..28 = 5, opcode 27..22 = 5, 6, 7, 8; PciReceive and PciSend: stream
// 15..10; HostWait: i 5..0; HostNotify: n 3..0; every other bit 0.
union clause instruction = C2C_PCIRECV : streamref
union clause instruction = C2C_PCISEND : streamref
union clause instruction = C2C_HOSTWAIT : bits(6)
union clause instruction = C2C_HOSTNOTIFY : bits(4)

mapping clause encdec32 = C2C_SEND(chip_hi : bits(3) @ chip_lo : bits(6), link_hi : bits(3) @ link_lo : bits(4), Streamref(s))
  <-> 0x5 @ encdec_c2c_op(C2C_OP_SEND) @ chip_lo @ s @ link_lo @ link_hi @ chip_hi
mapping clause encdec32 = C2C_RECEIVE(chip_hi : bits(3) @ chip_lo : bits(6), link_hi : bits(3) @ link_lo : bits(4), Streamref(s))
  <-> 0x5 @ encdec_c2c_op(C2C_OP_RECEIVE) @ chip_lo @ s @ link_lo @ link_hi @ chip_hi
mapping clause encdec32 = C2C_DESKEW() <-> 0x5 @ encdec_c2c_op(C2C_OP_DESKEW) @ 0x0000 @ 0b000000
mapping clause encdec32 = C2C_TRANSMIT(chip_hi : bits(3) @ chip_lo : bits(6), link_hi : bits(3) @ link_lo : bits(4))
  <-> 0x5 @ encdec_c2c_op(C2C_OP_TRANSMIT) @ chip_lo @ 0b000000 @ link_lo @ link_hi @ chip_hi
mapping clause encdec32 = C2C_RTDESKEW(t) <-> 0x5 @ encdec_c2c_op(C2C_OP_RTDESKEW) @ 0b000000 @ t
mapping clause encdec32 = C2C_DESKEWRECV(chip_hi : bits(3) @ chip_lo : bits(6), link_hi : bits(3) @ link_lo : bits(4), Streamref(s))
  <-> 0x5 @ encdec_c2c_op(C2C_OP_DESKEWRECV) @ chip_lo @ s @ link_lo @ link_hi @ chip_hi

mapping clause encdec32 = C2C_PCIRECV(Streamref(s))
  <-> 0x5 @ encdec_c2c_op(C2C_OP_PCIRECV) @ 0b000000 @ s @ 0b0000 @ 0b000000
  when unsigned(s[4 .. 0]) >= 24
mapping clause encdec32 = C2C_PCISEND(Streamref(s))
  <-> 0x5 @ encdec_c2c_op(C2C_OP_PCISEND) @ 0b000000 @ s @ 0b0000 @ 0b000000
mapping clause encdec32 = C2C_HOSTWAIT(i) <-> 0x5 @ encdec_c2c_op(C2C_OP_HOSTWAIT) @ 0x0000 @ i
mapping clause encdec32 = C2C_HOSTNOTIFY(n) <-> 0x5 @ encdec_c2c_op(C2C_OP_HOSTNOTIFY) @ 0x0000 @ 0b00 @ n
  when n != 0x0

// The link index, when it names one of the chip's links.
private function c2c_link(link : bits(7)) -> option(range(0, 127)) =
  if unsigned(link) < c2c_links then Some(unsigned(link)) else None()

// chip_id names the chip at the far end of the link.
private function c2c_peer_ok(chip : bits(9), l : range(0, 127)) -> bool = c2c_peer(l) == unsigned(chip)

// The cycle at which each link last accepted a vector (None: never).
register c2c_accepted : vector(128, option(nat)) = vector_init(None())

function c2c_reset() -> unit = c2c_accepted = vector_init(None())

// A link is busy in cycle c when it accepted a vector fewer than c2c.cycles_per_vector cycles
// before. `tick` executes every element in its own execute cycle (d_skew 0 in the dispatch
// step, any other from the execute ring at the start of that cycle, before the queues
// dispatch) and the ticks run in increasing cycles, so the executions of one run reach this
// check in non-decreasing execute-cycle order whatever the d_skew of each C2C instruction
// is. The last accepted cycle is then all the state the test needs, and
// `accepted + cycles_per_vector > c` is the whole test. Two executions in one cycle run ring
// first, then dispatch order: the first is accepted and the second is busy.
private function c2c_busy(l : range(0, 127), c : nat) -> bool = match c2c_accepted[l] {
  Some(a) => a + c2c_cycles_per_vector > c,
  None() => false,
}

// The vector goes on the link in this cycle: the link has accepted it.
private function c2c_put(l : range(0, 127), t : c2c_ptype, v : vec, poison : range(0, 4294967295)) -> unit = {
  c2c_accepted[l] = Some(exec_cycle());
  c2c_send(exec_cycle(), l, c2c_ptype_code(t), v, poison)
}

// The superlanes found uncorrectable in the operand of this Send, as the number that travels with the vector.
private function c2c_poison_int(m : bits(superlanes)) -> range(0, 4294967295) = {
  let n : int = unsigned(m);
  if n >= 0 & n <= 4294967295 then n else 0
}

// Packet types on a link (core/callbacks.sail c2c_ptype). A link packet is a header flit and
// 8-byte payload flits (HC34 p.45: "320-byte vector" in flits of 8 bytes): a data or sync packet
// carries one vector (1 + vlanes / 8 flits, 41 for the reference vector), a csr packet three flits
// (header; operation, chip, address and tag; value), a ctrl packet two (header; payload). The
// fields of csr and ctrl packets are not published (implementation choices,
// isa-db/provenance.yaml host.packet_types, host.packet_flits, host.link_share).
//   - Only data and sync packets enter the receive FIFO (Receive takes them; OTS-E-LNK-002 counts
//     them). A csr or ctrl packet is consumed by the receiving chip's control unit when it
//     arrives and never reaches a Receive.
//   - A packet occupies a link for ceil(c2c.cycles_per_vector * flits / (1 + vlanes / 8)) cycles:
//     a vector for exactly c2c.cycles_per_vector, so the bandwidth of a flit is the same for every type.
//   - The packets of a link share it in the order in which they are made: a csr or ctrl packet
//     leaves the control unit in the first cycle in which the link is free, that is, no packet
//     earlier on the link still occupies it and no data or sync packet that the link accepted
//     before is still on it. The control unit knows what happened before that cycle and nothing
//     of what comes after it (the harness implements this: c_emulator/tsp_sim.c). A Send or
//     Transmit released into the cycles of a csr or ctrl packet in flight is a program error,
//     OTS-E-LNK-003 (no effect; checked after OTS-E-LNK-001).
// The flits of a vector: a header flit and vlanes / 8 payload flits, rounded up.
let vector_flits : nat = {
  let n : int = 1 + ediv_int(vlanes + 7, 8);
  let m : nat = if n >= 0 then n else 0;
  m
}

// HC34 p.45: five payload bytes in the header, eight per body flit,
// three in the tail. Control corruption in header/tail conservatively poisons all lanes.
// The same packet length determines transfer time and fault coordinates.
let link_flits : nat = vector_flits

// 2^g.
function link_pow2(g : int) -> int = {
  var m : int = 1;
  foreach (_step from 0 to (g - 1)) { m = m * 2 };
  m
}

function link_flit_mask(f : int) -> range(0, 4294967295) = {
  let m : int = if f == 0 | f + 1 >= link_flits then link_pow2(superlanes) - 1 else {
    let first = ediv_int(5 + (f - 1) * 8, 16);
    let last = ediv_int(12 + (f - 1) * 8, 16);
    if first == last then link_pow2(first) else link_pow2(first) + link_pow2(last)
  };
  if 0 <= m & m <= 4294967295 then m else 0
}

function c2c_packet_flits(t : c2c_ptype) -> nat = match t {
  PKT_DATA => vector_flits,
  PKT_SYNC => vector_flits,
  PKT_CSR  => 3,
  PKT_CTRL => 2,
}

function c2c_packet_cycles(t : c2c_ptype) -> nat = {
  let n : int = c2c_cycles_per_vector;
  let f : int = c2c_packet_flits(t);
  let vf : int = vector_flits;
  let r : int = ediv_int(n * f + vf - 1, vf);
  let m : nat = if r >= 0 then r else 0;
  m
}

function c2c_ptype_queued(t : c2c_ptype) -> bool = match t {
  PKT_DATA => true,
  PKT_SYNC => true,
  PKT_CSR  => false,
  PKT_CTRL => false,
}

// The notification vector Transmit sends: lane 0 = 1, every other lane 0.
let c2c_notification : vec = sail_zero_extend(0x01, 8 * vlanes)

function clause execute C2C_SEND(chip, link, s) = match c2c_link(link) {
  None() => Diagnostic(OPR_RANGE),
  Some(l) =>
    if not(c2c_peer_ok(chip, l)) then Diagnostic(OPR_RANGE)
    else if c2c_busy(l, exec_cycle()) then Diagnostic(LNK_BUSY)
    else if c2c_packet_busy(exec_cycle(), l) then Diagnostic(LNK_PACKET)
    else match read_planes(s, 1, exec_cycle()) {
      Err(d) => Diagnostic(d),
      Ok(p)  => { c2c_put(l, PKT_DATA, p[0], c2c_poison_int(ecc_bad)); Issued() },
    },
}

function clause execute C2C_TRANSMIT(chip, link) = match c2c_link(link) {
  None() => Diagnostic(OPR_RANGE),
  Some(l) =>
    if not(c2c_peer_ok(chip, l)) then Diagnostic(OPR_RANGE)
    else if c2c_busy(l, exec_cycle()) then Diagnostic(LNK_BUSY)
    else if c2c_packet_busy(exec_cycle(), l) then Diagnostic(LNK_PACKET)
    else { c2c_put(l, PKT_SYNC, c2c_notification, 0); Issued() },
}

function clause execute C2C_RECEIVE(chip, link, s) = match c2c_link(link) {
  None() => Diagnostic(OPR_RANGE),
  Some(l) =>
    if not(c2c_peer_ok(chip, l)) then Diagnostic(OPR_RANGE)
    else if not(c2c_recv_ready(exec_cycle(), l)) then Diagnostic(TIM_NO_VALUE)
    else if not(aligned(s, 1)) then Diagnostic(OPR_ALIGN)
    // check the destination before taking the vector, so a rejected
    // Receive leaves the link FIFO unchanged (errors have no effect).
    else if write_gap(issue_unit.unit, issue_unit.slice, stream_row(s)) == None() then Diagnostic(DIR_UPSTREAM)
    else {
      let v = c2c_recv(exec_cycle(), l);
      // a vector that was uncorrectable when it was sent is poisoned in the same superlanes when it arrives (inherited, not reported)
      ecc_inherit = get_slice_int(superlanes, c2c_recv_poison(exec_cycle(), l), 0);
      ecc_link_bad = get_slice_int(superlanes, c2c_recv_bad(l), 0);
      match write_planes(s, one_plane(v), 1, commit_at()) {
        Ok(()) => Issued(),
        Err(d) => Diagnostic(d),
      }
    },
}

function clause execute C2C_PCIRECV(s) =
  if not(host_recv_ready(exec_cycle())) then Diagnostic(TIM_NO_VALUE)
  else if write_gap(issue_unit.unit, issue_unit.slice, stream_row(s)) == None() then Diagnostic(DIR_UPSTREAM)
  else {
    let v = host_recv(exec_cycle());
    match write_planes(s, one_plane(v), 1, commit_at()) {
      Ok(()) => Issued(),
      Err(d) => Diagnostic(d),
    }
  }

function clause execute C2C_PCISEND(s) =
  if pci_busy(exec_cycle()) then Diagnostic(HIO_BUSY)
  else match read_planes(s, 1, exec_cycle()) {
    Err(d) => Diagnostic(d),
    Ok(p)  => { pci_accepted = Some(exec_cycle()); host_send(exec_cycle(), p[0]); Issued() },
  }

// HostWait parks its queue at dispatch (postlude/tick.sail dispatch_body, host_wait); here only
// the doorbell number is checked.
function clause execute C2C_HOSTWAIT(i) = if unsigned(i) >= host_doorbells then Diagnostic(OPR_RANGE) else Issued()

function clause execute C2C_HOSTNOTIFY(n) = { host_msix(exec_cycle(), unsigned(n)); Issued() }

// Deskew and RuntimeDeskew only hold their queue (occupancy below).
function clause execute C2C_DESKEW() = Issued()
function clause execute C2C_RTDESKEW(_) = Issued()

// DeskewRecv is dispatched by the cycle driver (postlude/tick.sail dispatch_body, poll_start): it holds its queue and polls
// at each epoch boundary; the Receive it becomes at the first boundary at which a vector has arrived is what executes.
// Here only the diagnostics of Receive that do not depend on the arrival (the instruction then holds nothing and has no
// effect): the link and the chip_id of its far end (OTS-E-OPR-001), and a stream that the issuing unit cannot write
// (OTS-E-DIR-001); `iu` is the issuing queue's unit and instance.
function deskewrecv_diag(iu : issuer, chip : bits(9), link : bits(7), s : streamref) -> option(diag) = match c2c_link(link) {
  None() => Some(OPR_RANGE),
  Some(l) =>
    if not(c2c_peer_ok(chip, l)) then Some(OPR_RANGE)
    else if write_gap(iu.unit, iu.slice, stream_row(s)) == None() then Some(DIR_UPSTREAM)
    else None(),
}
function clause execute C2C_DESKEWRECV(_, _, _) = Issued()

// The link index as the poll names it; true when a vector has arrived on it by cycle c (what Receive tests).
function deskewrecv_ready(c : nat, link : bits(7)) -> bool = match c2c_link(link) {
  Some(l) => c2c_recv_ready(c, l),
  None() => false,
}

// The arrival cycle of the next vector that the harness knows on the link (-1: none).
function deskewrecv_next_arrival(link : bits(7)) -> int = match c2c_link(link) {
  Some(l) => c2c_next_arrival(l),
  None() => -1,
}

function clause occupancy(C2C_DESKEW(), c) = c2c_hac_period - emod_int(c, c2c_hac_period)
function clause occupancy(C2C_RTDESKEW(t), _) = max(unsigned(t), timing_rtdeskew_floor)

function clause timing_of C2C_SEND(_, _, _) = T_C2C
function clause timing_of C2C_RECEIVE(_, _, _) = T_C2C
function clause timing_of C2C_DESKEW() = T_C2C
function clause timing_of C2C_TRANSMIT(_, _) = T_C2C
function clause timing_of C2C_RTDESKEW(_) = T_C2C
function clause timing_of C2C_DESKEWRECV(_, _, _) = T_C2C
function clause timing_of C2C_PCIRECV(_) = T_C2C
function clause timing_of C2C_PCISEND(_) = T_C2C
function clause timing_of C2C_HOSTWAIT(_) = T_CU
function clause timing_of C2C_HOSTNOTIFY(_) = T_CU

function clause entry_of C2C_SEND(_, _, _) = "Send"
function clause entry_of C2C_RECEIVE(_, _, _) = "Receive"
function clause entry_of C2C_DESKEW() = "Deskew"
function clause entry_of C2C_TRANSMIT(_, _) = "Transmit"
function clause entry_of C2C_RTDESKEW(_) = "RuntimeDeskew"
function clause entry_of C2C_DESKEWRECV(_, _, _) = "DeskewRecv"
function clause entry_of C2C_PCIRECV(_) = "PciReceive"
function clause entry_of C2C_PCISEND(_) = "PciSend"
function clause entry_of C2C_HOSTWAIT(_) = "HostWait"
function clause entry_of C2C_HOSTNOTIFY(_) = "HostNotify"

private function c2c_tx(u : unit_kind) -> bool = u == U_C2C_W_TX | u == U_C2C_E_TX
private function c2c_rx(u : unit_kind) -> bool = u == U_C2C_W_RX | u == U_C2C_E_RX

function clause issued_by(C2C_SEND(_, _, _), u) = c2c_tx(u)
function clause issued_by(C2C_RECEIVE(_, _, _), u) = c2c_rx(u)
function clause issued_by(C2C_TRANSMIT(_, _), u) = c2c_tx(u)
function clause issued_by(C2C_DESKEWRECV(_, _, _), u) = c2c_rx(u)
// Deskew and RuntimeDeskew act on the queue that issues them, on any functional unit
// (ISCA 2022 sec. 3.2: "on that functional unit"; sec. 3.3).
function clause issued_by(C2C_DESKEW(), _) = true
function clause issued_by(C2C_RTDESKEW(_), _) = true
function clause issued_by(C2C_PCIRECV(_), u) = c2c_rx(u)
function clause issued_by(C2C_PCISEND(_), u) = c2c_tx(u)
function clause issued_by(C2C_HOSTWAIT(_), _) = true
function clause issued_by(C2C_HOSTNOTIFY(_), _) = true

mapping clause assembly = C2C_SEND(chip, link, s)
  <-> "send" ^ spc() ^ hex_bits_9(chip) ^ sep() ^ hex_bits_7(link) ^ sep() ^ stream_name(s)
mapping clause assembly = C2C_RECEIVE(chip, link, s)
  <-> "receive" ^ spc() ^ hex_bits_9(chip) ^ sep() ^ hex_bits_7(link) ^ sep() ^ stream_name(s)
mapping clause assembly = C2C_DESKEW() <-> "deskew"
mapping clause assembly = C2C_TRANSMIT(chip, link) <-> "transmit" ^ spc() ^ hex_bits_9(chip) ^ sep() ^ hex_bits_7(link)
mapping clause assembly = C2C_RTDESKEW(t) <-> "runtime_deskew" ^ spc() ^ hex_bits_16(t)
mapping clause assembly = C2C_DESKEWRECV(chip, link, s)
  <-> "deskew_recv" ^ spc() ^ hex_bits_9(chip) ^ sep() ^ hex_bits_7(link) ^ sep() ^ stream_name(s)
mapping clause assembly = C2C_PCIRECV(s) <-> "pci_receive" ^ spc() ^ stream_name(s)
mapping clause assembly = C2C_PCISEND(s) <-> "pci_send" ^ spc() ^ stream_name(s)
mapping clause assembly = C2C_HOSTWAIT(i) <-> "host_wait" ^ spc() ^ hex_bits_6(i)
mapping clause assembly = C2C_HOSTNOTIFY(n) <-> "host_notify" ^ spc() ^ hex_bits_4(n)
