//! The model's chip-to-chip callbacks (`core/callbacks.sail`): `c2c_peer`, `c2c_send`,
//! `c2c_recv_ready`, `c2c_recv`. The harness owns the wiring and the FIFOs
//! (`harness::links`); a [`Machine`](crate::Machine) only calls through this trait.
use crate::ctl::{CtlIn, CtlPkt};
use crate::geom::MAX_LANES;

/// The flits of a vector transfer: five payload bytes in the header, eight per body, three in the tail
/// (`link_flits` of `slices/c2c/c2c_insts.sail`; `microarchitecture.md` §17).
// model: slices/c2c/c2c_insts.sail link_flits
pub fn link_flits(vlanes: u64) -> u64 {
    1 + vlanes.div_ceil(8)
}

/// The superlanes (bit g) that flit `f` of a transfer carries: the header and the tail cover every superlane, a payload
/// flit its eight bytes beginning at byte `5 + 8 * (f - 1)`, possibly spanning two superlanes.
// model: slices/c2c/c2c_insts.sail link_flit_mask
pub fn link_flit_mask(f: u64, vlanes: u64, superlanes: u64) -> u32 {
    if f == 0 || f + 1 >= link_flits(vlanes) {
        ((1u64 << superlanes) - 1) as u32
    } else {
        (1 << ((5 + (f - 1) * 8) / 16)) | (1 << ((12 + (f - 1) * 8) / 16))
    }
}

/// One `F link` line: flit `flit` of the vector `chip` sends on `link` in `cycle` arrives with an error the link FEC
/// corrects (`unc` false) or only detects (`unc` true).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkFault {
    pub chip: u32,
    pub cycle: u64,
    pub link: u8,
    pub flit: u8,
    pub unc: bool,
}

/// What the harness found on the links at a cycle (`process_arrivals`), in the harness's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkEvent {
    /// A vector arrived at a full FIFO and was dropped (OTS-E-LNK-002).
    Overrun { cycle: u64 },
    /// The link FEC corrected (`unc` false) or detected an uncorrectable flit of a vector that arrived on `link`
    /// (`T ecc ... link`).
    Fec { cycle: u64, link: u8, unc: bool, flit: u8 },
}

pub trait LinkPort: Send {
    /// The chip at the far end of `link`, if the link is connected (`c2c_peer`).
    // model: core/callbacks.sail c2c_peer
    fn peer(&self, link: u8) -> Option<u16>;
    /// A vector sent at `cycle` on `link` (`c2c_send`); `v` holds `vlanes` lanes, the rest zero.
    // model: core/callbacks.sail c2c_send
    fn send(&mut self, cycle: u64, link: u8, v: &[u8; MAX_LANES]);
    /// A packet of another type than data (Transmit's sync) put on `link` at `cycle`; `ptype` is the model's
    /// `c2c_ptype_code`.
    // model: core/callbacks.sail c2c_send
    fn send_packet(&mut self, cycle: u64, link: u8, _ptype: u8, v: &[u8; MAX_LANES]) {
        self.send(cycle, link, v)
    }
    /// A data vector sent at `cycle` on `link` with the superlanes in `poison` uncorrectable (`c2c_send`'s poison
    /// argument); a port that does not carry poison drops it.
    // model: core/callbacks.sail c2c_send
    fn send_poisoned(&mut self, cycle: u64, link: u8, v: &[u8; MAX_LANES], _poison: u32) {
        self.send(cycle, link, v)
    }
    /// The poison of the vector the last [`LinkPort::recv`] took (`c2c_recv_poison`); 0 when there was none.
    // model: core/callbacks.sail c2c_recv_poison
    fn recv_poison(&self) -> u32 {
        0
    }
    /// A csr or ctrl packet that left `link` at or before `cycle` still occupies it in `cycle` (`c2c_packet_busy`).
    // model: core/callbacks.sail c2c_packet_busy
    fn packet_busy(&self, _cycle: u64, _link: u8) -> bool {
        false
    }
    /// The control unit: a packet that waits for its link to be free to leave on `link` (`cpend_add`).
    fn ctl_add(&mut self, _pkt: CtlPkt, _ptype: u8, _link: usize, _ready: u64) {}
    /// The control packets whose departure is certain leave (`ctl_depart`).
    fn ctl_depart(
        &mut self,
        _frontier: Option<u128>,
        _real_next: Option<u128>,
        _limit: u64,
    ) -> Result<(), crate::machine::RunError> {
        Ok(())
    }
    /// The earliest cycle at which a waiting control packet could leave (`ctl_wake`).
    fn ctl_wake(&self) -> Option<u128> {
        None
    }
    /// The arrival cycle of the next control packet taken in (`ctl_next_event`).
    fn ctl_next_event(&self) -> Option<u128> {
        None
    }
    /// A control packet waits to leave.
    fn ctl_pending(&self) -> bool {
        false
    }
    /// The control packets that arrived at `t`, in (arrival, packet number) order.
    fn ctl_take(&mut self, _t: u64) -> Vec<CtlIn> {
        Vec::new()
    }
    /// Has a vector arrived on `link` by `cycle` (`c2c_recv_ready`)? Decides the link's arrivals
    /// up to `cycle`, so it may change the port.
    // model: core/callbacks.sail c2c_recv_ready
    fn recv_ready(&mut self, cycle: u64, link: u8) -> bool;
    /// The arrival cycle of the oldest vector of `link` that is neither taken nor dropped (`None`: none known), which the chip's
    /// poll event is made from (`DeskewRecv`; `c2c_next_arrival`).
    // model: core/callbacks.sail c2c_next_arrival
    fn next_arrival(&self, _link: u8) -> Option<u128> {
        None
    }
    /// Take the head vector of `link` at `cycle` (`c2c_recv`); zero when there is none.
    // model: core/callbacks.sail c2c_recv
    fn recv(&mut self, cycle: u64, link: u8) -> [u8; MAX_LANES];
    /// The superlanes (bit g) of the vector the last [`recv`](Self::recv) took that the link FEC found uncorrectable
    /// (`c2c_recv_bad`).
    // model: core/callbacks.sail c2c_recv_bad
    fn recv_bad(&self, _link: u8) -> u32 {
        0
    }
    /// The earliest cycle at which the harness must act on a link: a dropped arrival to report,
    /// or an arrival that could overrun its FIFO (`link_next_event`).
    fn next_event(&self) -> Option<u128> {
        None
    }
    /// Decide every arrival up to `upto` and return the cycles of the dropped ones due (one
    /// OTS-E-LNK-002 each, ascending cycle and link; `process_arrivals`).
    fn arrivals(&mut self, _upto: u64) -> Vec<u64> {
        Vec::new()
    }
    /// Decide every arrival up to `upto` and return what the harness reports for them, in order: the link FEC events
    /// and the overruns (`process_arrivals`).
    fn link_events(&mut self, upto: u64) -> Vec<LinkEvent> {
        self.arrivals(upto).into_iter().map(|cycle| LinkEvent::Overrun { cycle }).collect()
    }
}

/// A chip with no link connected.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoLinks;

impl LinkPort for NoLinks {
    fn peer(&self, _link: u8) -> Option<u16> {
        None
    }
    fn send(&mut self, _cycle: u64, _link: u8, _v: &[u8; MAX_LANES]) {}
    fn recv_ready(&mut self, _cycle: u64, _link: u8) -> bool {
        false
    }
    fn recv(&mut self, _cycle: u64, _link: u8) -> [u8; MAX_LANES] {
        [0; MAX_LANES]
    }
}
