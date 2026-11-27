//! The control unit's packets (`microarchitecture.md` §10 "Packet types", §14 remote CSR access; `c_emulator/tsp_sim.c`
//! "The control unit"): every transfer on a link has a type (data, csr, sync, ctrl). Send's packet is data, Transmit's
//! sync; only those enter a receive FIFO. A csr packet (`P csr`: one hop to the chip at the far end of a link, the reply on
//! the same link) and a ctrl packet (`P ctrl`) are taken in by the receiving chip's control unit. The packets of a link
//! share it in the order in which they are made: a csr or ctrl packet leaves at the first cycle at which the link is free
//! (`provenance.yaml` `host.link_share`; the rule is causal), and a Send or Transmit released into a packet in flight is
//! OTS-E-LNK-003.
use crate::geom::MAX_LINKS;

/// The packet types, as the model's `c2c_ptype_code`.
pub const PT_DATA: u8 = 0;
pub const PT_CSR: u8 = 1;
pub const PT_SYNC: u8 = 2;
pub const PT_CTRL: u8 = 3;

/// Whether a packet of the type enters the receive FIFO (`c2c_ptype_queued`).
// model: slices/c2c/c2c_insts.sail c2c_ptype_queued
pub fn queued(ptype: u8) -> bool {
    ptype == PT_DATA || ptype == PT_SYNC
}

/// The operation of a control packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Rd,
    Wr,
    Reply,
    Ctrl,
}

/// The content of a csr or ctrl packet (`ctlpkt_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtlPkt {
    pub op: Op,
    pub origin: u16,
    pub target: u16,
    pub addr: u64,
    pub val: u64,
    pub id: u64,
}

impl CtlPkt {
    /// The order of packets of one link that became ready in the same cycle (`seq_of`).
    pub fn seq(&self) -> u64 {
        self.id * 2 + (self.op == Op::Reply) as u64
    }
}

/// A control packet that has arrived at this chip and waits for its cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtlIn {
    pub pkt: CtlPkt,
    pub ptype: u8,
    /// This chip's link it arrived on.
    pub link: u8,
    pub arrival: u128,
}

/// A packet waiting for its link to be free, to leave this chip (`cpend_t`).
#[derive(Clone, Copy, Debug)]
pub struct Pending {
    pub pkt: CtlPkt,
    pub ptype: u8,
    pub link: usize,
    pub ready: u128,
    pub seq: u64,
}

/// The schedule of the chip's links and its waiting control packets.
#[derive(Clone, Debug)]
pub struct Window {
    /// The cycles at which each link accepted a vector of the program (data or sync), ascending.
    pub accs: Vec<Vec<u64>>,
    /// The last control packet that left a link occupies `[start, end)` (`end` 0: none).
    pub start: [u128; MAX_LINKS],
    pub end: [u128; MAX_LINKS],
    /// The links the chip has (`c2c.links`): the loops over the links stop here.
    pub nlinks: usize,
    pub pending: Vec<Pending>,
    pub cpv: u64,
    /// `packet_cycles` of csr and ctrl packets (data and sync take `cpv`).
    pub csr_cycles: u64,
    pub ctrl_cycles: u64,
}

impl Window {
    pub fn new(cpv: u64, vlanes: u64) -> Window {
        let vf = 1 + vlanes.div_ceil(8);
        let cyc = |f: u64| (cpv * f).div_ceil(vf);
        Window {
            accs: vec![Vec::new(); MAX_LINKS],
            start: [0; MAX_LINKS],
            end: [0; MAX_LINKS],
            nlinks: MAX_LINKS,
            pending: Vec::new(),
            cpv,
            csr_cycles: cyc(3),
            ctrl_cycles: cyc(2),
        }
    }

    // model: slices/c2c/c2c_insts.sail c2c_packet_cycles
    pub fn cycles(&self, ptype: u8) -> u64 {
        match ptype {
            PT_CSR => self.csr_cycles,
            PT_CTRL => self.ctrl_cycles,
            _ => self.cpv,
        }
    }

    pub fn accepted(&mut self, link: usize, c: u64) {
        self.accs[link].push(c);
    }

    pub fn add(&mut self, pkt: CtlPkt, ptype: u8, link: usize, ready: u64) {
        self.pending.push(Pending { pkt, ptype, link, ready: u128::from(ready), seq: pkt.seq() });
    }

    /// The first cycle >= ready at which link `l` is free: after the control packet before it (they share the link) and
    /// after the vector the link accepted last before that cycle (`accs` holds every accepted vector that matters).
    // model: c_emulator/tsp_sim.c ctl_candidate
    fn candidate(&self, l: usize, ready: u128) -> u128 {
        let mut c = ready.max(self.end[l]);
        let a = &self.accs[l];
        loop {
            let lo = a.partition_point(|x| u128::from(*x) < c); // the vectors accepted before c: a[..lo]
            if lo > 0 && u128::from(a[lo - 1]) + u128::from(self.cpv) > c {
                c = u128::from(a[lo - 1]) + u128::from(self.cpv);
            } else {
                return c;
            }
        }
    }

    /// A csr or ctrl packet that left link `l` at or before cycle `c` still occupies it in `c` (`c2c_packet_busy`).
    // model: c_emulator/tsp_sim.c c2c_packet_busy
    pub fn packet_busy(&self, c: u64, l: usize) -> bool {
        self.end[l] != 0 && self.start[l] <= u128::from(c) && u128::from(c) < self.end[l]
    }

    /// The index of the pending packet of link `l` that leaves first.
    // model: c_emulator/tsp_sim.c ctl_head
    fn head(&self, l: usize) -> Option<usize> {
        let mut h: Option<usize> = None;
        for (i, x) in self.pending.iter().enumerate() {
            if x.link != l {
                continue;
            }
            if h.is_none_or(|j| (x.ready, x.seq) < (self.pending[j].ready, self.pending[j].seq)) {
                h = Some(i);
            }
        }
        h
    }

    /// The earliest cycle at which a waiting packet could leave (`ctl_wake`).
    pub fn wake(&self) -> Option<u128> {
        (0..self.nlinks).filter_map(|l| self.head(l).map(|i| self.candidate(l, self.pending[i].ready))).min()
    }

    /// The packets whose departure is certain, as `(packet, departure cycle)` in sending order (`ctl_depart`): the cycle `c`
    /// is at most `real_next` (every vector accepted before `c` is known, since the chip has ticked every cycle below it)
    /// and the ready cycle is below `frontier` (no packet that has not reached the chip arrives before it).
    // model: c_emulator/tsp_sim.c ctl_depart
    pub fn depart(
        &mut self,
        frontier: Option<u128>,
        real_next: Option<u128>,
        limit: u64,
    ) -> Result<Vec<(Pending, u128)>, crate::machine::RunError> {
        let h = match (frontier, real_next) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let mut out = Vec::new();
        for l in 0..self.nlinks {
            while let Some(i) = self.head(l) {
                let p = self.pending[i];
                let c = self.candidate(l, p.ready);
                if frontier.is_some_and(|f| p.ready >= f) || h.is_some_and(|h| c > h) {
                    break;
                }
                if c > u128::from(limit) {
                    return Err(crate::machine::RunError::MaxCycles { cycle: c, limit });
                }
                self.start[l] = c;
                self.end[l] = c + u128::from(self.cycles(p.ptype));
                self.pending.swap_remove(i);
                out.push((p, c));
            }
        }
        Ok(out)
    }
}
