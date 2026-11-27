//! The wiring of the chips' links (`tsp_sim.c`, "C2C links"; `microarchitecture.md` §10): which
//! chip each link reaches, and each directed link's latency under `--link-latency`,
//! `--link-latency-seed` (splitmix64) and `--link-jitter-seed` (`LinkPlan`), and each chip's end of
//! the links: the inbound FIFOs, the vectors sent to other chips, the jitter draws (`ChipLinks`,
//! `PlanLinks`, the machine's `LinkPort`).
use super::cli::LinkMode;
use super::topology::Topology;
use crate::ctl::{self, CtlIn, CtlPkt, Window};
use crate::geom::{Geom, MAX_CHIPS, MAX_LANES, MAX_LINKS};
use crate::links::{link_flit_mask, LinkEvent, LinkFault, LinkPort};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The far end of chip `c`'s link `l` in a system of `chips` chips with `links` links per chip:
/// `(chip, its link)`, or `None` for a link that goes nowhere.
///
/// One chip: every link loops back to itself. K = 2..8: one node, local link l (0..6) of chip i
/// reaches chip (i + l + 1) mod K on its local link K - 2 - l. K above 8: nodes of 8 chips (the last
/// holds the rest) with local links as above inside the node, and global port p = 4 (i mod 8) + g of
/// node n (global link 7 + g) reaches node (n + p + 1) mod N on port N - 2 - p, when that chip exists.
// model: c_emulator/tsp_sim.c link_peer
pub fn link_peer(chips: usize, links: usize, c: usize, l: usize) -> Option<(usize, usize)> {
    if l >= links {
        return None;
    }
    if chips == 1 {
        return Some((c, l));
    }
    let nn = if chips <= 8 { 1 } else { chips.div_ceil(8) };
    let (node, k) = if chips <= 8 { (0, c) } else { (c / 8, c % 8) };
    let nk = if chips <= 8 { chips } else { (chips - 8 * node).min(8) }; // chips in this node
    if l <= 6 {
        if l + 1 >= nk {
            return None;
        }
        return Some((node * 8 + (k + l + 1) % nk, nk - 2 - l));
    }
    if l <= 10 && nn > 1 {
        let p = 4 * k + (l - 7);
        if p + 1 >= nn {
            return None;
        }
        let (m, q) = ((node + p + 1) % nn, nn - 2 - p);
        let peer = m * 8 + q / 4;
        if peer >= chips {
            return None; // the far chip of a node that is not full does not exist
        }
        return Some((peer, 7 + q % 4));
    }
    None
}

/// splitmix64: a fixed, portable generator for `--link-latency-seed`.
// model: c_emulator/tsp_sim.c splitmix
pub fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The start state of the per-(seed, chip, link) jitter generator.
pub fn jitter_state(seed: u64, chip: usize, link: usize) -> u64 {
    // links 0..15 keep the value of the earlier harness; the links above 15 follow all of them
    let idx = if link < 16 { chip * 16 + link + 1 } else { MAX_CHIPS * 16 + chip * (MAX_LINKS - 16) + (link - 16) + 1 };
    seed ^ (idx as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// The static link plan of a run.
#[derive(Clone, Debug)]
pub struct LinkPlan {
    pub chips: usize,
    pub links: usize,
    pub mode: LinkMode,
    /// The latency of chip c's outbound link l (nominal for jitter: the minimum).
    latency: Vec<[u64; MAX_LINKS]>,
    /// The jitter generator state per chip and link (`--link-jitter-seed` only).
    jitter: Vec<[u64; MAX_LINKS]>,
    /// The smallest latency of a connected link: the lock-step window of `--chips`.
    min_latency: Option<u64>,
    /// The adjacency of `--topology`; none: the wiring of `--chips K` (`link_peer`).
    topo: Option<Arc<Topology>>,
}

impl LinkPlan {
    /// `links_setup`: the range checks and the latencies. The error is the stderr text.
    pub fn new(mode: LinkMode, g: &Geom, chips: usize) -> Result<LinkPlan, String> {
        LinkPlan::build(mode, g, chips, None)
    }

    /// The plan of a topology file (`--topology`): its chips, its adjacency, its latencies.
    pub fn with_topology(mode: LinkMode, g: &Geom, topo: Arc<Topology>) -> Result<LinkPlan, String> {
        LinkPlan::build(mode, g, topo.chips, Some(topo))
    }

    fn build(mode: LinkMode, g: &Geom, chips: usize, topo: Option<Arc<Topology>>) -> Result<LinkPlan, String> {
        assert!((1..=MAX_CHIPS).contains(&chips));
        let (lo, hi, nom) = (g.link_latency_min, g.link_latency_max, g.link_latency);
        if lo > nom || nom > hi {
            return Err("need c2c.link_latency_min <= c2c.link_latency <= c2c.link_latency_max".into());
        }
        if let LinkMode::Fixed(n) = mode {
            if n < lo || n > hi {
                return Err(format!("--link-latency must be in [{lo}, {hi}]"));
            }
        }
        if let LinkMode::Map(m) = &mode {
            if let Some((c, l, n)) =
                m.iter().find(|(c, l, n)| *n < lo || *n > hi || *c as usize >= chips || u64::from(*l) >= g.links)
            {
                return Err(if *n < lo || *n > hi {
                    format!("--link-latency-map {c}:{l}={n}: the latency must be in [{lo}, {hi}]")
                } else {
                    format!("--link-latency-map {c}:{l}={n}: no such chip or link")
                });
            }
        }
        // the topology file's jitter seed applies when no link option was given
        let mode = match (mode, topo.as_ref().and_then(|t| t.jitter_seed)) {
            (LinkMode::Nominal, Some(s)) => LinkMode::Jitter(s),
            (m, _) => m,
        };
        let links = g.links as usize;
        let mut x = match &mode {
            LinkMode::Seed(s) => *s,
            _ => 0,
        };
        let mut latency = vec![[0u64; MAX_LINKS]; chips];
        let mut jitter = vec![[0u64; MAX_LINKS]; chips];
        let mut min_latency: Option<u64> = None;
        // Seeded latencies are drawn chip by chip over links 0..15, then over the links above 15, so a profile with at most
        // 16 links draws what the earlier harness drew.
        for pass in 0..2 {
            let range = if pass == 0 { 0..16 } else { 16..MAX_LINKS };
            for c in 0..chips {
                for l in range.clone() {
                    let given = topo.as_ref().map_or(0, |t| t.latency(c, l));
                    let given =
                        if given != 0 { given } else { topo.as_ref().and_then(|t| t.latency_default).unwrap_or(0) };
                    let lat = match &mode {
                        LinkMode::Fixed(n) => *n,
                        LinkMode::Seed(_) => lo + splitmix(&mut x) % (hi - lo + 1),
                        LinkMode::Jitter(_) => lo,
                        LinkMode::Nominal if given != 0 => given,
                        LinkMode::Nominal => nom,
                        LinkMode::Map(m) => {
                            m.iter().find(|(mc, ml, _)| *mc as usize == c && *ml as usize == l).map_or(nom, |e| e.2)
                        }
                    };
                    if let LinkMode::Jitter(s) = &mode {
                        jitter[c][l] = jitter_state(*s, c, l);
                    }
                    latency[c][l] = lat;
                    let wired = match &topo {
                        Some(t) => t.peer(c, l).is_some() && l < links,
                        None => link_peer(chips, links, c, l).is_some(),
                    };
                    if wired && min_latency.is_none_or(|m| lat < m) {
                        min_latency = Some(lat);
                    }
                }
            }
        }
        Ok(LinkPlan { chips, links, mode, latency, jitter, min_latency, topo })
    }

    pub fn peer(&self, chip: usize, link: usize) -> Option<(usize, usize)> {
        match &self.topo {
            Some(t) if link < self.links => t.peer(chip, link),
            Some(_) => None,
            None => link_peer(self.chips, self.links, chip, link),
        }
    }

    pub fn latency(&self, chip: usize, link: usize) -> u64 {
        self.latency[chip][link]
    }

    pub fn jitter_state(&self, chip: usize, link: usize) -> u64 {
        self.jitter[chip][link]
    }

    /// The lock-step window: the smallest connected link latency, 1 when no link is connected.
    pub fn window(&self) -> u64 {
        self.min_latency.unwrap_or(1)
    }
}

/// Where a vector of a FIFO stands: not yet arrived or decided, in the FIFO, taken by a
/// Receive, dropped (OTS-E-LNK-002).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VState {
    Pending,
    Queued,
    Taken,
    Dropped,
}

/// A vector in flight or waiting: `vecmsg_t`. `chip` and `link` name the receiving end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecMsg {
    pub arrival: u128,
    pub chip: usize,
    pub link: usize,
    pub state: VState,
    /// The execute cycle of the Receive that took it (`Taken`).
    pub taken_at: u128,
    /// The packet type (`ctl::PT_*`); a csr or ctrl packet carries `ctl` instead of a vector.
    pub ptype: u8,
    pub ctl: Option<CtlPkt>,
    /// The flits (bit f, up to 65 of them: a header flit and one per 8 lanes) of the transfer that the link FEC corrected in flight, and that it detected and could not
    /// correct (`F link`).
    pub fec_c: u128,
    pub fec_u: u128,
    pub b: [u8; MAX_LANES],
    /// The superlanes that were uncorrectable when the vector was sent (a bit each; 0 for a clean vector).
    pub poison: u32,
}

/// A report to make once its cycle is reached: the link FEC events of an arrived vector (`kind` 0) or an overrun
/// (`kind` 1, OTS-E-LNK-002) (`report_t`).
#[derive(Clone, Copy, Debug)]
struct Report {
    arrival: u128,
    link: usize,
    kind: u8,
    fec_c: u128,
    fec_u: u128,
}

/// A link's inbound FIFO in arrival order (`fifo_t`): the entries from `head` on still matter.
#[derive(Clone, Debug, Default)]
pub struct Fifo {
    v: Vec<VecMsg>,
    head: usize,
}

impl Fifo {
    /// Keep arrival order: a later-sent vector on a link never overtakes (equal latency per link),
    /// so this is an append; the insertion keeps it correct for any order of delivery.
    // model: c_emulator/tsp_sim.c fifo_push
    pub fn push(&mut self, m: VecMsg) {
        let mut i = self.v.len();
        while i > self.head && self.v[i - 1].arrival > m.arrival {
            i -= 1;
        }
        self.v.insert(i, m);
    }

    /// The vectors that have arrived or will arrive and are neither taken nor dropped.
    pub fn len(&self) -> usize {
        self.v[self.head..].iter().filter(|m| matches!(m.state, VState::Pending | VState::Queued)).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decide the pending arrivals up to `upto`: the occupancy a vector arriving at a sees is the
    /// earlier vectors in the FIFO or taken by a Receive that executes at a or later.
    /// Returns the arrival cycles of the dropped ones.
    // model: c_emulator/tsp_sim.c link_resolve
    fn resolve(&mut self, upto: u64, depth: u64, dropped: &mut Vec<u128>) {
        for i in self.head..self.v.len() {
            let a = self.v[i].arrival;
            if a > u128::from(upto) {
                break;
            }
            if self.v[i].state != VState::Pending {
                continue;
            }
            let occ = self.v[self.head..i]
                .iter()
                .filter(|o| o.state == VState::Queued || (o.state == VState::Taken && o.taken_at >= a))
                .count() as u64;
            if occ >= depth {
                self.v[i].state = VState::Dropped;
                dropped.push(a);
            } else {
                self.v[i].state = VState::Queued;
            }
        }
    }

    /// Drop the prefix that can no longer matter: dropped vectors, and vectors taken more than the
    /// largest execution skew (63) before the latest cycle looked at.
    // model: c_emulator/tsp_sim.c fifo_trim
    fn trim(&mut self, latest: u64) {
        while self.head < self.v.len() {
            let m = &self.v[self.head];
            if m.state == VState::Dropped || (m.state == VState::Taken && m.taken_at + 64 < u128::from(latest)) {
                self.head += 1;
            } else {
                break;
            }
        }
        if self.head > 1024 && self.head * 2 > self.v.len() {
            self.v.drain(..self.head);
            self.head = 0;
        }
    }

    /// The first arrival that would find the FIFO full if nothing more were taken.
    // model: c_emulator/tsp_sim.c link_next_event (one link)
    fn next_overrun(&self, depth: u64) -> Option<u128> {
        let mut active = 0u64;
        let mut taken: Vec<u128> = Vec::new();
        for m in &self.v[self.head..] {
            match m.state {
                VState::Pending => {
                    let occ = active + taken.iter().filter(|t| **t >= m.arrival).count() as u64;
                    if occ >= depth {
                        return Some(m.arrival);
                    }
                    active += 1;
                }
                VState::Queued => active += 1,
                VState::Taken if taken.len() < 64 => taken.push(m.taken_at),
                _ => {}
            }
        }
        None
    }

    /// The arrival of the first vector that is pending or queued.
    fn next_arrival(&self) -> Option<u128> {
        self.v[self.head..].iter().find(|m| matches!(m.state, VState::Pending | VState::Queued)).map(|m| m.arrival)
    }

    /// The head of the FIFO: the first queued vector, unless a pending one comes before it.
    // model: c_emulator/tsp_sim.c fifo_front
    fn front(&mut self) -> Option<&mut VecMsg> {
        for m in self.v[self.head..].iter_mut() {
            match m.state {
                VState::Queued => return Some(m),
                VState::Pending => return None,
                _ => {}
            }
        }
        None
    }
}

/// One chip's end of the links: its inbound FIFOs, the vectors it sent to other chips since the
/// last window, and the jitter state of its outbound links (`inbound`, `outbound`, `jit_state`,
/// `jit_last` of `tsp_sim.c`).
pub struct ChipLinks {
    plan: Arc<LinkPlan>,
    chip: usize,
    inbound: Vec<Fifo>,
    outbound: Vec<VecMsg>,
    jit_state: [u64; MAX_LINKS],
    /// The previous arrival on each outbound link + 1 over every packet type of the link (0: none yet).
    jit_last: [u128; MAX_LINKS],
    /// The generator of the packets of the control unit: its own, so that the draws of the vectors do not depend on when a
    /// control packet leaves (the in-order clamp `jit_last` is shared).
    jit_state_c: [u64; MAX_LINKS],
    /// The control unit: the schedule of the links and the waiting packets, and the packets that arrived.
    win: Window,
    cin: Vec<CtlIn>,
    /// Set by the first control packet and never cleared: while it is clear the control unit has nothing to do, so
    /// [`PlanLinks`] answers its questions without the lock.
    ctl_touched: Arc<AtomicBool>,
    geom_lat: (u64, u64),
    /// `c2c.rx_depth`.
    rx_depth: u64,
    /// The reports not yet made: dropped arrivals and the link FEC events of arrived vectors.
    reports: Vec<Report>,
    /// The `F link` lines of this chip, in input order, and whether a vector was sent that they marked.
    faults: Vec<(LinkFault, bool)>,
    /// `c2c_recv_bad` of the vector the last Receive took, and the shape of a vector for the flit map.
    last_bad: u32,
    vlanes: u64,
    superlanes: u64,
    /// Set by the first vector that enters an inbound FIFO and never cleared: while it is clear, the chip is
    /// [`idle`](Self::idle) for good (`reports` only fill from the FIFOs), so [`PlanLinks`] can answer
    /// `next_event` and `arrivals` without the lock.
    touched: Arc<AtomicBool>,
    /// The poison of the vector the last `recv` took (`recv_poison` of `tsp_sim.c`).
    recv_poison: u32,
    observed_receives: Option<Vec<(u64, u8)>>,
}

impl ChipLinks {
    pub fn new(plan: Arc<LinkPlan>, chip: usize, g: &Geom) -> ChipLinks {
        let mut jit_state = [0u64; MAX_LINKS];
        for (l, s) in jit_state.iter_mut().enumerate() {
            *s = plan.jitter_state(chip, l);
        }
        ChipLinks {
            plan,
            chip,
            inbound: vec![Fifo::default(); MAX_LINKS],
            outbound: Vec::new(),
            jit_state,
            jit_last: [0; MAX_LINKS],
            jit_state_c: jit_state.map(|x| !x),
            win: Window { nlinks: (g.links as usize).min(MAX_LINKS), ..Window::new(g.cycles_per_vector, g.vlanes) },
            cin: Vec::new(),
            ctl_touched: Arc::new(AtomicBool::new(false)),
            geom_lat: (g.link_latency_min, g.link_latency_max),
            rx_depth: g.rx_depth,
            reports: Vec::new(),
            faults: Vec::new(),
            last_bad: 0,
            vlanes: g.vlanes,
            superlanes: g.superlanes,
            touched: Arc::new(AtomicBool::new(false)),
            recv_poison: 0,
            observed_receives: None,
        }
    }

    /// Opt in to recording successful vector consumption, independently of execution.
    pub fn observe_receives(&mut self) {
        self.observed_receives = Some(Vec::new());
    }

    /// Take the recorded cycle and link pairs, leaving the observer enabled.
    pub fn drain_receives(&mut self) -> Vec<(u64, u8)> {
        self.observed_receives.as_mut().map(std::mem::take).unwrap_or_default()
    }

    /// Give the chip its `F link` lines (those of its own chip, in input order).
    pub fn set_faults(&mut self, faults: impl IntoIterator<Item = LinkFault>) {
        self.faults = faults.into_iter().map(|f| (f, false)).collect();
    }

    /// The first `F link` line that marked no vector: `(link, cycle)` (`finish_check`).
    pub fn unused_fault(&self) -> Option<(u8, u64)> {
        self.faults.iter().find(|(_, used)| !used).map(|(f, _)| (f.link, f.cycle))
    }

    /// Take the vectors sent to other chips since the last call, in sending order.
    pub fn take_outbound(&mut self) -> Vec<VecMsg> {
        std::mem::take(&mut self.outbound)
    }

    /// Put a vector from another chip into the FIFO of its link.
    pub fn deliver(&mut self, m: VecMsg) {
        let l = m.link;
        if !ctl::queued(m.ptype) {
            // a csr or ctrl packet is taken in by the control unit, in (arrival, packet number) order
            let (pkt, arrival) = (m.ctl.expect("a csr or ctrl packet carries its content"), m.arrival);
            self.ctl_touched.store(true, Ordering::Release);
            let x = CtlIn { pkt, ptype: m.ptype, link: l as u8, arrival };
            let mut i = self.cin.len();
            while i > 0 && (self.cin[i - 1].arrival, self.cin[i - 1].pkt.seq()) > (arrival, pkt.seq()) {
                i -= 1;
            }
            self.cin.insert(i, x);
            return;
        }
        self.touched.store(true, Ordering::Release);
        if m.fec_c | m.fec_u != 0 {
            // link FEC events: reported at the arrival cycle (`process_events`)
            self.reports.push(Report { arrival: m.arrival, link: l, kind: 0, fec_c: m.fec_c, fec_u: m.fec_u });
        }
        self.inbound[l].push(m);
    }

    /// The arrival cycle of the oldest vector of the link's FIFO that is neither taken nor dropped.
    // model: c_emulator/tsp_sim.c c2c_next_arrival
    pub fn next_arrival(&self, link: usize) -> Option<u128> {
        self.inbound[link].next_arrival()
    }

    pub fn inbound_len(&self, link: usize) -> usize {
        self.inbound[link].len()
    }

    // model: c_emulator/tsp_sim.c link_resolve
    fn resolve(&mut self, l: usize, upto: u64) {
        let mut dropped = Vec::new();
        self.inbound[l].resolve(upto, self.rx_depth, &mut dropped);
        self.reports.extend(dropped.into_iter().map(|a| Report { arrival: a, link: l, kind: 1, fec_c: 0, fec_u: 0 }));
    }

    /// The inbound FIFOs of the profile's links (`c2c.links`): the only ones a vector can reach (`link_peer` has no peer above them),
    /// so the scans of every tick walk these and not the `MAX_LINKS` that are allocated.
    fn live(&self) -> &[Fifo] {
        &self.inbound[..self.win.nlinks]
    }

    /// No report waits and no inbound FIFO holds a vector it still looks at: arrivals and events have nothing to do.
    fn idle(&self) -> bool {
        self.reports.is_empty() && self.live().iter().all(|f| f.head == f.v.len())
    }

    /// Decide every arrival up to `upto`; the cycles of the overruns due, ascending (cycle, link).
    pub fn process_arrivals(&mut self, upto: u64) -> Vec<u64> {
        self.process_events(upto)
            .into_iter()
            .filter_map(|e| match e {
                LinkEvent::Overrun { cycle } => Some(cycle),
                LinkEvent::Fec { .. } => None,
            })
            .collect()
    }

    /// Decide every arrival up to `upto`; the reports due, ascending (cycle, link, kind): for a vector the link FEC
    /// events by flit, the FEC acting before the receive FIFO sees the vector, then the overrun.
    // model: c_emulator/tsp_sim.c process_arrivals
    pub fn process_events(&mut self, upto: u64) -> Vec<LinkEvent> {
        if self.idle() {
            return Vec::new();
        }
        for l in 0..self.win.nlinks {
            self.resolve(l, upto);
            self.inbound[l].trim(upto);
        }
        let mut due: Vec<Report> = Vec::new();
        self.reports.retain(|r| {
            if r.arrival <= u128::from(upto) {
                due.push(*r);
                false
            } else {
                true
            }
        });
        due.sort_by_key(|r| (r.arrival, r.link, r.kind));
        let mut out = Vec::new();
        for r in due {
            if r.kind == 1 {
                out.push(LinkEvent::Overrun {
                    cycle: u64::try_from(r.arrival).expect("arrival is due in checked u64 cycle"),
                });
                continue;
            }
            for f in 0..128u8 {
                if (r.fec_c | r.fec_u) >> f & 1 == 1 {
                    out.push(LinkEvent::Fec {
                        cycle: u64::try_from(r.arrival).expect("arrival is due in checked u64 cycle"),
                        link: r.link as u8,
                        unc: r.fec_u >> f & 1 == 1,
                        flit: f,
                    });
                }
            }
        }
        out
    }

    /// The earliest cycle at which the harness must act on a link: a report waiting, or an
    /// arrival that would find the FIFO full if nothing more were taken.
    // model: c_emulator/tsp_sim.c link_next_event
    pub fn next_event(&self) -> Option<u128> {
        if self.idle() {
            return None;
        }
        let r = self.reports.iter().map(|r| r.arrival).min();
        let a = self.live().iter().filter_map(|f| f.next_overrun(self.rx_depth)).min();
        match (r, a) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        }
    }

    // model: c_emulator/tsp_sim.c c2c_peer
    fn peer(&self, link: u8) -> Option<u16> {
        self.plan.peer(self.chip, link as usize).map(|(c, _)| c as u16)
    }

    /// The arrival cycle of a packet that leaves this chip's link `l` at cycle `sent` (`link_arrival`; `control`: a packet of
    /// the control unit).
    // model: c_emulator/tsp_sim.c link_arrival
    fn arrival(&mut self, l: usize, sent: u128, control: bool) -> u128 {
        if !matches!(self.plan.mode, LinkMode::Jitter(_)) {
            return sent + u128::from(self.plan.latency(self.chip, l));
        }
        // per-transfer latency, delivered in order, one packet per cycle: one clamp per link over every packet type
        let (lo, hi) = self.geom_lat;
        let st = if control { &mut self.jit_state_c[l] } else { &mut self.jit_state[l] };
        let last = &mut self.jit_last[l];
        let mut a = sent + u128::from(lo) + u128::from(splitmix(st) % (hi - lo + 1));
        if a < *last {
            a = *last;
        }
        *last = a + 1;
        a
    }

    // model: c_emulator/tsp_sim.c c2c_send
    fn send(&mut self, sent: u64, link: u8, ptype: u8, v: &[u8; MAX_LANES], poison: u32) {
        let l = link as usize;
        let Some((chip, pl)) = self.plan.peer(self.chip, l) else { return }; // the model checks chip_id first
        let arrival = self.arrival(l, u128::from(sent), false);
        let mut m = VecMsg {
            arrival,
            chip,
            link: pl,
            state: VState::Pending,
            taken_at: 0,
            ptype,
            ctl: None,
            fec_c: 0,
            fec_u: 0,
            b: *v,
            poison,
        };
        for (f, used) in self.faults.iter_mut() {
            // the `F link` lines of this vector
            if f.cycle == sent && f.link == link {
                *used = true;
                if f.unc {
                    m.fec_u |= 1 << f.flit;
                } else {
                    m.fec_c |= 1 << f.flit;
                }
            }
        }
        self.win.accepted(l, sent);
        self.out_or_deliver(m);
    }

    // model: c_emulator/tsp_sim.c fifo_or_out
    fn out_or_deliver(&mut self, m: VecMsg) {
        if m.chip == self.chip {
            self.deliver(m);
        } else {
            self.outbound.push(m);
        }
    }

    /// The packets of the control unit whose departure is certain leave (`ctl_depart`).
    pub fn ctl_depart(
        &mut self,
        frontier: Option<u128>,
        real_next: Option<u128>,
        limit: u64,
    ) -> Result<(), crate::machine::RunError> {
        for (p, c) in self.win.depart(frontier, real_next, limit)? {
            let Some((chip, pl)) = self.plan.peer(self.chip, p.link) else { continue };
            let arrival = self.arrival(p.link, c, true);
            let m = VecMsg {
                arrival,
                chip,
                link: pl,
                state: VState::Pending,
                taken_at: 0,
                ptype: p.ptype,
                ctl: Some(p.pkt),
                fec_c: 0,
                fec_u: 0,
                b: [0; MAX_LANES],
                poison: 0,
            };
            self.out_or_deliver(m);
        }
        Ok(())
    }

    pub fn ctl_add(&mut self, pkt: CtlPkt, ptype: u8, link: usize, ready: u64) {
        self.ctl_touched.store(true, Ordering::Release);
        self.win.add(pkt, ptype, link, ready);
    }

    pub fn ctl_wake(&self) -> Option<u128> {
        self.win.wake()
    }

    // model: c_emulator/tsp_sim.c c2c_packet_busy
    pub fn packet_busy(&self, cycle: u64, link: usize) -> bool {
        self.win.packet_busy(cycle, link)
    }

    pub fn ctl_pending(&self) -> bool {
        !self.win.pending.is_empty()
    }

    pub fn ctl_next_event(&self) -> Option<u128> {
        self.cin.first().map(|x| x.arrival)
    }

    pub fn ctl_take(&mut self, t: u64) -> Vec<CtlIn> {
        let n = self.cin.iter().take_while(|x| x.arrival == u128::from(t)).count();
        self.cin.drain(..n).collect()
    }

    // model: c_emulator/tsp_sim.c c2c_recv_ready
    fn recv_ready(&mut self, cycle: u64, link: u8) -> bool {
        self.resolve(link as usize, cycle);
        self.inbound[link as usize].front().is_some_and(|m| m.arrival <= u128::from(cycle))
    }

    // model: c_emulator/tsp_sim.c c2c_recv
    fn recv(&mut self, cycle: u64, link: u8) -> [u8; MAX_LANES] {
        let (vl, sl) = (self.vlanes, self.superlanes);
        let mut bad = 0;
        let v = match self.inbound[link as usize].front() {
            Some(m) => {
                m.state = VState::Taken;
                m.taken_at = u128::from(cycle);
                if let Some(receives) = &mut self.observed_receives {
                    receives.push((cycle, link));
                }
                self.recv_poison = m.poison;
                for f in 0..128u64 {
                    if m.fec_u >> f & 1 == 1 {
                        bad |= link_flit_mask(f, vl, sl);
                    }
                }
                m.b
            }
            None => {
                self.recv_poison = 0;
                [0; MAX_LANES]
            }
        };
        self.last_bad = bad;
        v
    }
}

/// One chip's [`LinkPort`]: a shared handle on its [`ChipLinks`], so the harness can move the
/// vectors between chips while the machine holds the port.
#[derive(Clone)]
pub struct PlanLinks {
    shared: Arc<Mutex<ChipLinks>>,
    /// `ChipLinks::touched`, read without the lock.
    touched: Arc<AtomicBool>,
    /// `ChipLinks::ctl_touched`, read without the lock.
    ctl_touched: Arc<AtomicBool>,
}

impl PlanLinks {
    pub fn new(plan: Arc<LinkPlan>, chip: usize, g: &Geom) -> PlanLinks {
        let links = ChipLinks::new(plan, chip, g);
        let touched = links.touched.clone();
        let ctl_touched = links.ctl_touched.clone();
        PlanLinks { shared: Arc::new(Mutex::new(links)), touched, ctl_touched }
    }

    /// A single chip with every link looped back at the nominal latency: the system the golden
    /// vectors were made with (`tsp_sim <config>` with no link option).
    pub fn loopback(g: &Geom) -> PlanLinks {
        let plan = LinkPlan::new(LinkMode::Nominal, g, 1).expect("the nominal latency is in range for a valid profile");
        PlanLinks::new(Arc::new(plan), 0, g)
    }

    /// The shared state, for the harness.
    pub fn state(&self) -> Arc<Mutex<ChipLinks>> {
        self.shared.clone()
    }
}

impl LinkPort for PlanLinks {
    fn peer(&self, link: u8) -> Option<u16> {
        self.shared.lock().unwrap().peer(link)
    }
    fn send(&mut self, cycle: u64, link: u8, v: &[u8; MAX_LANES]) {
        self.shared.lock().unwrap().send(cycle, link, ctl::PT_DATA, v, 0)
    }
    fn send_packet(&mut self, cycle: u64, link: u8, ptype: u8, v: &[u8; MAX_LANES]) {
        self.shared.lock().unwrap().send(cycle, link, ptype, v, 0)
    }
    fn send_poisoned(&mut self, cycle: u64, link: u8, v: &[u8; MAX_LANES], poison: u32) {
        self.shared.lock().unwrap().send(cycle, link, ctl::PT_DATA, v, poison)
    }
    fn recv_poison(&self) -> u32 {
        self.shared.lock().unwrap().recv_poison
    }
    fn ctl_add(&mut self, pkt: CtlPkt, ptype: u8, link: usize, ready: u64) {
        self.shared.lock().unwrap().ctl_add(pkt, ptype, link, ready)
    }
    fn ctl_depart(
        &mut self,
        frontier: Option<u128>,
        real_next: Option<u128>,
        limit: u64,
    ) -> Result<(), crate::machine::RunError> {
        if self.ctl_touched.load(Ordering::Acquire) {
            self.shared.lock().unwrap().ctl_depart(frontier, real_next, limit)?;
        }
        Ok(())
    }
    fn packet_busy(&self, cycle: u64, link: u8) -> bool {
        self.ctl_touched.load(Ordering::Acquire) && self.shared.lock().unwrap().packet_busy(cycle, link as usize)
    }
    fn ctl_wake(&self) -> Option<u128> {
        if !self.ctl_touched.load(Ordering::Acquire) {
            return None;
        }
        self.shared.lock().unwrap().ctl_wake()
    }
    fn ctl_next_event(&self) -> Option<u128> {
        if !self.ctl_touched.load(Ordering::Acquire) {
            return None;
        }
        self.shared.lock().unwrap().ctl_next_event()
    }
    fn ctl_pending(&self) -> bool {
        self.ctl_touched.load(Ordering::Acquire) && self.shared.lock().unwrap().ctl_pending()
    }
    fn ctl_take(&mut self, t: u64) -> Vec<CtlIn> {
        if !self.ctl_touched.load(Ordering::Acquire) {
            return Vec::new();
        }
        self.shared.lock().unwrap().ctl_take(t)
    }
    fn recv_ready(&mut self, cycle: u64, link: u8) -> bool {
        self.shared.lock().unwrap().recv_ready(cycle, link)
    }
    fn recv(&mut self, cycle: u64, link: u8) -> [u8; MAX_LANES] {
        self.shared.lock().unwrap().recv(cycle, link)
    }
    fn recv_bad(&self, _link: u8) -> u32 {
        self.shared.lock().unwrap().last_bad
    }
    fn next_arrival(&self, link: u8) -> Option<u128> {
        if !self.touched.load(Ordering::Acquire) {
            return None;
        }
        self.shared.lock().unwrap().next_arrival(link as usize)
    }
    fn next_event(&self) -> Option<u128> {
        // asked every tick: no vector has entered a FIFO, so there is nothing to act on (`ChipLinks::idle`)
        if !self.touched.load(Ordering::Acquire) {
            return None;
        }
        self.shared.lock().unwrap().next_event()
    }
    fn arrivals(&mut self, upto: u64) -> Vec<u64> {
        if !self.touched.load(Ordering::Acquire) {
            return Vec::new();
        }
        self.shared.lock().unwrap().process_arrivals(upto)
    }
    fn link_events(&mut self, upto: u64) -> Vec<LinkEvent> {
        if !self.touched.load(Ordering::Acquire) {
            return Vec::new();
        }
        self.shared.lock().unwrap().process_events(upto)
    }
}
