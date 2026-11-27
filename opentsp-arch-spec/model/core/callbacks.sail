// =====================================================================================
// SPDX-License-Identifier: Apache-2.0
// =====================================================================================

// Callbacks that tell the harness about architectural state changes (sail-riscv
// core/callbacks.sail). The harness implements them (c_emulator/); the unit-test runner
// links no-op stubs (test/unit_tests/stubs.c).

// The trace carries values: srf_write_callback gets the vector committed to the register
// (after any thread or SXM merge; a colliding write reports the vector that is kept), and
// sram_write_callback the whole word as stored (a masked Write or a Scatter lane: the word after
// the write). The harness prints them as vlanes bytes, lane 0 first.
val srf_write_callback = pure {c: "srf_write_callback", cpp: "srf_write_callback"} :
  (/* cycle */ nat, /* row is W */ bool, /* stream */ range(0, 31), /* gap */ range(0, 43), /* value */ vec) -> unit

val sram_write_callback = pure {c: "sram_write_callback", cpp: "sram_write_callback"} :
  (/* cycle */ nat, /* slice */ range(0, 87), /* addr */ bits(16), /* the word written */ vec) -> unit

val diag_callback = pure {c: "diag_callback", cpp: "diag_callback"} :
  (/* cycle */ nat, /* queue (65535 = CU) */ range(0, 65535), /* code */ string) -> unit

// Simulator lints (core/lints.sail): non-normative notes for
// a simulator's user, reported apart from the architectural diagnostics. Same arguments as
// diag_callback, with a lint code `OTS-W-<area>-<nnn>`. A harness that does not want them
// implements this as a no-op; they are never counted as diagnostics and are not part of the
// conformance vectors.
val lint_callback = pure {c: "lint_callback", cpp: "lint_callback"} :
  (/* cycle */ nat, /* queue (65535 = CU) */ range(0, 65535), /* code */ string) -> unit

// Chip-to-chip links (slices/c2c/c2c_insts.sail). The model knows only its own end of a
// link: link 0 .. c2c.links - 1. The harness owns the wiring, the per-link latency and the
// link FIFOs (c_emulator/tsp_sim.c: with one chip every link loops back to itself; with K
// chips, the node topology of ISCA 2022 sec. 2.2). A link carries whole vectors (the
// flow-control unit, ISCA 2022 sec. 2.3) and never applies back pressure. The
// model itself limits a link's bandwidth (OTS-E-LNK-001, c2c.cycles_per_vector); the harness
// bounds each link's FIFO at c2c.rx_depth vectors and reports a dropped arrival as
// OTS-E-LNK-002 of the receiving chip (queue 65535) before the arrival cycle's tick.

// Every transfer on a link has a type: the header flit of a link packet carries "Type (data, CSR,
// Sync, Ctrl)" (HC34 p.45). Send puts a `data` packet on the link, Transmit a `sync` packet; a CSR
// access of the host to a remote chip travels as `csr` packets and the control unit's own
// messages as `ctrl` packets (slices/c2c/c2c_insts.sail says which of them enter the receive FIFO,
// how long each occupies a link, and how csr and ctrl packets share a link with scheduled
// data). The numbers are the order of the published list; the header's bit encoding is not published.
enum c2c_ptype = {PKT_DATA, PKT_CSR, PKT_SYNC, PKT_CTRL}

function c2c_ptype_code(t : c2c_ptype) -> range(0, 3) = match t {
  PKT_DATA => 0,
  PKT_CSR  => 1,
  PKT_SYNC => 2,
  PKT_CTRL => 3,
}

// True when a csr or ctrl packet that left this chip's link at or before cycle c still occupies
// it in cycle c (the control unit's packet is in flight: slices/c2c/c2c_insts.sail, Send and
// Transmit, OTS-E-LNK-003).
val c2c_packet_busy = impure {c: "c2c_packet_busy", cpp: "c2c_packet_busy"} : (/* cycle */ nat, /* link */ range(0, 127)) -> bool

// Chip id at the far end of the link (-1: not connected).
val c2c_peer = impure {c: "c2c_peer", cpp: "c2c_peer"} : (/* link */ range(0, 127)) -> range(-1, 511)

// Put a packet of the given type (c2c_ptype_code) on the link at cycle sent; it reaches the peer's
// end of the link at sent plus the link's latency (harness: in [c2c.link_latency_min,
// c2c.link_latency_max]). The model puts data and sync packets on a link (Send, Transmit); the
// harness's control unit puts csr and ctrl packets on it.
//
// The last argument is the poison of the vector (microarchitecture.md sec. 15): bit g set when superlane g was uncorrectable in the
// sender's stream operand. The vector travels with it, so a vector that was bad when it was sent is bad when it arrives
// (an uncorrectable word crossing a link is not turned into a valid one). 0 for every packet that is not a Send.
val c2c_send = impure {c: "c2c_send", cpp: "c2c_send"} : (/* sent */ nat, /* link */ range(0, 127), /* type */ range(0, 3), vec, /* poison */ range(0, 4294967295)) -> unit

// True when the link's inbound FIFO holds a vector that arrived at or before cycle c.
val c2c_recv_ready = impure {c: "c2c_recv_ready", cpp: "c2c_recv_ready"} : (/* cycle */ nat, /* link */ range(0, 127)) -> bool

// Remove and return the oldest such vector (called only when c2c_recv_ready holds).
val c2c_recv = impure {c: "c2c_recv", cpp: "c2c_recv"} : (/* cycle */ nat, /* link */ range(0, 127)) -> vec

// The poison of the vector the last c2c_recv removed from the link (the argument of the c2c_send that put it there).
val c2c_recv_poison = impure {c: "c2c_recv_poison", cpp: "c2c_recv_poison"} : (/* cycle */ nat, /* link */ range(0, 127)) -> range(0, 4294967295)

// Link FEC (microarchitecture.md sec. 17). After c2c_recv: the superlanes of the vector just taken that the link's FEC
// detected as uncorrectable (bit g set for superlane g; 0 for a vector that arrived intact or whose errors were all
// corrected in flight). The harness knows the flits of the transfer that carried a fault (the F link lines); the model
// maps a flit to its superlanes (core/csr.sail link_flit_mask).
val c2c_recv_bad = impure {c: "c2c_recv_bad", cpp: "c2c_recv_bad"} : (/* link */ range(0, 127)) -> range(0, 4294967295)

// The arrival cycle of the oldest vector of the link's inbound FIFO that has not been taken or dropped, among those the
// harness knows (-1: none). The harness uses it, through model_poll_event, to wake the chip at the epoch boundary at which a
// DeskewRecv looks for that vector (slices/c2c/c2c_insts.sail); no instruction reads it.
val c2c_next_arrival = impure {c: "c2c_next_arrival", cpp: "c2c_next_arrival"} : (/* link */ range(0, 127)) -> int

// The host interface (core/csr.sail, slices/c2c/c2c_insts.sail PciReceive, PciSend). The model
// knows the chip's CSR space and the instructions; the harness owns the host: its script of
// timed actions (c_emulator/tsp_sim.c), the ingress FIFO the host's DMA engine fills
// (host.cycles_per_vector, host.ingress_depth: a vector that finds the FIFO full waits in the DMA
// engine until a PciReceive has freed a slot; nothing is dropped) and the egress list of the
// vectors that PciSend produced. Every chip has its own host interface and its own script.

// The chip's number (CSR 0x000).
val host_chip_id = impure {c: "host_chip_id", cpp: "host_chip_id"} : unit -> range(0, 511)

// True when the ingress FIFO holds a vector that arrived at or before cycle c.
val host_recv_ready = impure {c: "host_recv_ready", cpp: "host_recv_ready"} : (/* cycle */ nat) -> bool

// Remove and return the oldest such vector (called only when host_recv_ready holds).
val host_recv = impure {c: "host_recv", cpp: "host_recv"} : (/* cycle */ nat) -> vec

// PciSend: append a vector to the egress list.
val host_send = impure {c: "host_send", cpp: "host_send"} : (/* cycle */ nat, vec) -> unit

// HostNotify: MSI-X vector n reached the host (not masked by IRQ_MASK).
val host_irq_callback = impure {c: "host_irq_callback", cpp: "host_irq_callback"} : (/* cycle */ nat, /* vector */ range(0, 15)) -> unit

// An ECC event (core/csr.sail ras_ecc_event, which also records it in the error CSRs): one per superlane found
// corrected or uncorrectable by a reader, one per instruction-text word. kind 0: a stream operand (the row is W when
// `row_w`, id the stream, addr the reader's gap), 1: an SRAM word (id the slice, addr the address), 2: a word of
// instruction text (id 0, addr the text word counted from 0 over the queue's text in production order).
val ecc_event_callback = pure {c: "ecc_event_callback", cpp: "ecc_event_callback"} :
  (/* cycle */ nat, /* queue */ range(0, 144), /* kind */ range(0, 2), /* uncorrectable */ bool,
   /* stream: row is W */ bool, /* stream id, or SRAM slice */ range(0, 87), /* gap, SRAM address or text word */ range(0, 16777215),
   /* superlane */ range(0, 31), /* syndrome position */ range(0, 255)) -> unit

// A stream operand whose spare bit is set: the error report of a MEM slice (microarchitecture.md sec. 15.5), logged by its first
// consumer. The row is W when `row_w`, id the stream, gap the reader's gap; s03 is S[0..3] of the superlane (bit i = stream i).
val spare_event_callback = pure {c: "spare_event_callback", cpp: "spare_event_callback"} :
  (/* cycle */ nat, /* queue */ range(0, 144), /* stream: row is W */ bool, /* stream id */ range(0, 87), /* reader's gap */ range(0, 16777215),
   /* superlane */ range(0, 31), /* S[0..3] */ range(0, 15)) -> unit

// The patrol scrubber corrected (or found uncorrectable) the word of superlane `superlane` at address `addr` of MEM slice `slice` in `cycle`
// (microarchitecture.md sec. 15.4).
val scrub_event_callback = pure {c: "scrub_event_callback", cpp: "scrub_event_callback"} :
  (/* cycle */ nat, /* slice */ range(0, 87), /* address */ nat, /* superlane */ range(0, 31), /* uncorrectable */ bool) -> unit

// Optional software observation. Consumers identify image/fetch locations by the
// production base and unit offset. These events have no architectural effects.
val observe_site = impure {c: "observe_site", cpp: "observe_site"} :
  (string, nat, range(0,144), nat, nat, int) -> unit
