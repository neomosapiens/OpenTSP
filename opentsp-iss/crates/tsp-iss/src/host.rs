//! The host interface of the chip (`core/csr.sail`, `slices/c2c/c2c_insts.sail` PciReceive, PciSend, HostWait,
//! HostNotify, `postlude/harness_api.sail` `model_csr_*`, `model_inject_unit`, `model_set_start`, and the host's side
//! in `c_emulator/tsp_sim.c` `host_*`; `microarchitecture.md` §14, `platform.md` §9).
//!
//! Every chip has its own host interface: one CSR space of 4,096 registers of 64 bits (`CHIP_ID` is the chip's index),
//! its own ingress FIFO and DMA engine, doorbells and interrupt delivery; no host instruction is routed over the links.
//! No instruction reads or writes the CSR space: the host reaches it through the harness (`H` and `R` lines, which act
//! on the chip of the preceding `C <chip>` line), so a program's results never depend on it. The harness owns the
//! host: its script of timed actions (`H` lines), the ingress FIFO the host's DMA engine fills, and the list of the
//! vectors PciSend gave the host. HostWait parks a queue (the registers `hw_since`, `hw_db`, `hw_release` live in the
//! [`Barrier`](crate::driver::barrier::Barrier), beside the other queue parks).
use crate::ctl::{CtlPkt, Op};
use crate::geom::MAX_LANES;
use crate::observe::{HostDma, Observer};
use crate::slices::Issue;
use crate::Machine;
use std::collections::VecDeque;

/// What the host does at one cycle of its script.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostKind {
    /// `H dma`: the DMA engine pushes a vector (`vlanes` bytes; the rest zero) into the ingress FIFO.
    Dma(Box<[u8; MAX_LANES]>),
    /// `H csr .. w` and `H db`: a CSR write.
    CsrWrite { addr: u64, val: u64 },
    /// `H csr .. r`: a CSR read (a `T csr` line).
    CsrRead { addr: u64 },
    /// `H inject`: units of instruction text for a queue's IQ (0..143, or 144 for the control channel).
    Inject { queue: usize, units: Vec<u32> },
    /// `F sram`: bit `bit` (0..127 data, 128..136 check bits, 137 spare) of superlane `sl` of a stored SRAM word flips.
    FlipSram { slice: u8, addr: u32, sl: u8, bit: u8 },
    /// `F srf`: the same for the value in a stream register (row W when `w`).
    FlipSrf { w: bool, stream: u8, gap: u8, sl: u8, bit: u8 },
    /// `F mxm`: bit `bit` (0..95) of the exact sum of lane `lane` of the ABC that executes on `plane` flips (the sum check).
    FlipMxm { plane: u8, lane: u16, bit: u8 },
    /// `F iq`: bit `bit` (0..127 data, 128..136 check bits, 137 spare) of text word `word` of a queue's instruction queue.
    FlipIq { queue: usize, word: u64, bit: u8 },
}

/// One timed action of the host. `cycle` is the cycle the script names; for a `Dma` the arrival after the pacing is
/// worked out by [`Machine::set_host_script`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostEvent {
    pub cycle: u64,
    pub kind: HostKind,
}

/// What a `P` line sends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PktKind {
    /// `P ctrl`.
    Ctrl,
    /// `P csr .. r <addr>`: a read of the CSR of the chip at the far end of the link.
    CsrRead { addr: u64 },
    /// `P csr .. w <addr> <value>`.
    CsrWrite { addr: u64, val: u64 },
}

/// `P ctrl <cycle> <link>` and `P csr <cycle> <link> r|w <addr> [<value>]`: the chip's control unit sends a ctrl or a
/// csr packet on `link` (a remote CSR access is one hop: the far end of the link serves it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PktEvent {
    pub cycle: u64,
    pub id: u64,
    pub link: u8,
    pub kind: PktKind,
}

/// Management-side observation for one execution. Guest W1C and IRQ masks
/// cannot erase evidence before the host commits that execution's result.
#[derive(Clone, Debug, Default)]
pub struct ServiceRas {
    pub events: u64,
    pub causes: u16,
    pub fatal_count: u64,
    pub overflow: bool,
    pub first: Option<(u64, u64)>,
    pub first_fatal: Option<(u64, u64)>,
    /// Pending out-of-band report seen by the management reader of raw SRAM.
    pub stored_report: Option<(u8, u32, u8)>,
    /// Present only while every unsafe event is localized to this SRAM word.
    pub sram_word: Option<(u8, u32)>,
}

/// The error CSRs of the chip: `ERR_STATUS`, `ERR_MASK`, the two counters, the error log and the first-error copy.
// model: core/csr.sail ras_err_status .. ras_first1
#[derive(Clone, Debug)]
pub struct Ras {
    pub status: u64,
    pub mask: u64,
    pub sbe: u64,
    pub mbe: u64,
    /// The log entries (`w0`, `w1`), oldest first.
    pub log: Vec<(u64, u64)>,
    pub first: Option<(u64, u64)>,
    /// The diagnostic log (`diag_log`, `diag_n`, `diag_overflow`): the words of the records of class `data`, oldest
    /// first, and the overflow flag.
    pub diag: Vec<u64>,
    pub diag_overflow: bool,
}

impl Ras {
    pub fn new() -> Ras {
        Ras {
            status: 0,
            mask: 0xAA,
            sbe: 0,
            mbe: 0,
            log: Vec::new(),
            first: None,
            diag: Vec::new(),
            diag_overflow: false,
        }
    }

    // model: core/csr.sail ras_severity
    fn severity(&self) -> u64 {
        if self.status & 0xAA != 0 {
            2
        } else if self.status & 0x155 != 0 {
            1
        } else {
            0
        }
    }
}

impl Default for Ras {
    fn default() -> Ras {
        Ras::new()
    }
}

fn sat_inc(x: u64) -> u64 {
    if x == 0xFFFF_FFFF {
        x
    } else {
        x + 1
    }
}

/// The host interface registers and the host's script.
// model: core/csr.sail (the registers), c_emulator/tsp_sim.c (the host)
#[derive(Clone, Debug)]
pub struct Host {
    doorbell: [bool; 64],
    irq_cause: u64,
    irq_mask: u64,
    irq_count: u64,
    /// Error interrupts (MSI-X vector 0) delivered to the host; `--on-irq` looks at it.
    pub(crate) err_irq: u64,
    /// `--on-irq stop`: the run ends at the end of the cycle of the first error interrupt.
    pub(crate) stop_on_irq: bool,
    /// The run ended at an error interrupt (`stop_on_irq`).
    pub(crate) stopped: bool,
    /// `LINK_CORR` and `LINK_UNCORR` (0x200 + l, 0x210 + l for l < 16, 0x500 + l - 16, 0x600 + l - 16 above): the flits the link FEC corrected and the uncorrectable ones
    /// it detected, per link (`csr_link_corr`, `csr_link_unc`).
    link_corr: [u64; 128],
    link_unc: [u64; 128],
    /// The error CSRs (`core/csr.sail`, the `ras_*` registers).
    pub(crate) ras: Ras,
    pub(crate) service_ras: ServiceRas,
    /// The cycle the PCIe interface last accepted a vector for the host (PciSend).
    pci_accepted: Option<u64>,
    /// The ingress FIFO.
    ingress: VecDeque<[u8; MAX_LANES]>,
    /// The vectors PciSend gave the host: cycle and `MAX_LANES` bytes (lanes at or beyond `vlanes` are zero).
    pub egress: Vec<(u64, [u8; MAX_LANES])>,
    /// The CSR and inject actions of the script, by cycle (stable), and the next to run.
    script: Vec<HostEvent>,
    pos: usize,
    /// The DMA requests, by request cycle (stable), and the next one; the previous arrival; the earliest entry after a
    /// freed slot; and whether the head request found the FIFO full and waits for a PciReceive.
    dma: Vec<(u64, Box<[u8; MAX_LANES]>)>,
    dma_pos: usize,
    dma_last: Option<u64>,
    dma_floor: u128,
    dma_blocked: bool,
    /// `H start`: the cycle the queues start dispatching.
    pub start: u64,
    /// The chip's `P` lines by cycle (stable) and the next to run.
    pk: Vec<PktEvent>,
    pk_pos: usize,
}

impl Host {
    pub fn new() -> Host {
        Host {
            doorbell: [false; 64],
            irq_cause: 0,
            irq_mask: 0xFFFF,
            irq_count: 0,
            err_irq: 0,
            stop_on_irq: false,
            stopped: false,
            link_corr: [0; 128],
            link_unc: [0; 128],
            ras: Ras::new(),
            service_ras: ServiceRas::default(),
            pci_accepted: None,
            ingress: VecDeque::new(),
            egress: Vec::new(),
            script: Vec::new(),
            pos: 0,
            dma: Vec::new(),
            dma_pos: 0,
            dma_last: None,
            dma_floor: 0,
            dma_blocked: false,
            start: 0,
            pk: Vec::new(),
            pk_pos: 0,
        }
    }

    /// The registers go back to reset (the script and the egress list are the harness's, not the chip's).
    // model: core/csr.sail host_reset
    pub fn reset(&mut self) {
        self.doorbell = [false; 64];
        self.irq_cause = 0;
        self.irq_mask = 0xFFFF;
        self.irq_count = 0;
        self.err_irq = 0;
        self.stopped = false;
        self.link_corr = [0; 128];
        self.link_unc = [0; 128];
        self.ras = Ras::new();
        self.pci_accepted = None;
    }

    /// The cycle at which the head DMA request enters the FIFO if it finds room (`None`: none, or it waits for a
    /// PciReceive).
    // model: c_emulator/tsp_sim.c dma_eligible
    fn dma_eligible(&self, cpv: u64) -> Option<u128> {
        if self.dma_blocked {
            return None;
        }
        let (req, _) = self.dma.get(self.dma_pos)?;
        let mut e = u128::from(*req);
        if let Some(l) = self.dma_last {
            e = e.max(u128::from(l) + u128::from(cpv));
        }
        Some(e.max(self.dma_floor))
    }

    /// The cycle of the next action of the script or of the DMA engine (`host_next_event`).
    pub fn next_event(&self, cpv: u64) -> Option<u128> {
        let a = self.script.get(self.pos).map(|e| u128::from(e.cycle));
        match (a, self.dma_eligible(cpv)) {
            (Some(a), Some(d)) => Some(a.min(d)),
            (a, d) => a.or(d),
        }
    }

    /// The cycle of the next `P` line (`pk_next_event`).
    pub fn pk_next_event(&self) -> Option<u64> {
        self.pk.get(self.pk_pos).map(|p| p.cycle)
    }

    /// The ingress FIFO holds a vector (`host_recv_ready`).
    pub fn recv_ready(&self) -> bool {
        !self.ingress.is_empty()
    }

    /// The numbers of the registers a CSR read of `addr` finds (`csr_read`, the part that needs no queue state).
    // model: core/csr.sail csr_read
    pub fn csr_read(&self, addr: u64, chip_id: u64, doorbells: u64) -> u64 {
        match addr {
            0 => chip_id,
            48 => self.irq_cause,
            49 => self.irq_mask,
            50 => self.irq_count,
            a if (768..768 + doorbells).contains(&a) => self.doorbell[(a - 768) as usize] as u64,
            512..=527 => self.link_corr[(addr - 512) as usize],
            528..=543 => self.link_unc[(addr - 528) as usize],
            1280..=1391 => self.link_corr[(addr - 1264) as usize],
            1536..=1647 => self.link_unc[(addr - 1520) as usize],
            _ => 0,
        }
    }

    /// The link FEC corrected (`unc` false) or detected an uncorrectable flit on `link` (`model_link_fec`).
    // model: postlude/harness_api.sail model_link_fec
    pub fn link_fec(&mut self, link: u8, unc: bool) {
        let c = if unc { &mut self.link_unc } else { &mut self.link_corr };
        c[link as usize & 127] += 1;
    }

    /// PciSend acceptance: one vector per `host.cycles_per_vector` cycles (`pci_busy`).
    // model: core/csr.sail pci_busy
    pub fn pci_busy(&self, c: u64, cycles_per_vector: u64) -> bool {
        self.pci_accepted.is_some_and(|a| a + cycles_per_vector > c)
    }

    /// The DMA requests that have not entered the ingress FIFO yet (not yet eligible, or waiting for a free slot).
    pub fn dma_pending(&self) -> usize {
        self.dma.len() - self.dma_pos
    }

    /// The head DMA request found the ingress FIFO full and waits for a PciReceive to free a slot (`T dma wait`).
    pub fn dma_blocked(&self) -> bool {
        self.dma_blocked
    }

    /// The vectors still waiting in the ingress FIFO.
    pub fn ingress_len(&self) -> usize {
        self.ingress.len()
    }
}

impl Default for Host {
    fn default() -> Self {
        Host::new()
    }
}

impl<const L: usize> Machine<L> {
    /// Give the chip its host script (the `H` lines that act on it) and its start cycle (`H start`: the queues start
    /// dispatching at that cycle, as if every queue had last dispatched in the cycle before). The CSR and inject actions
    /// are kept by cycle; the DMA requests are kept apart, by request cycle: a request is eligible at the later of its
    /// cycle and the previous arrival plus `host.cycles_per_vector`, and enters the FIFO then if it finds room.
    // model: c_emulator/tsp_sim.c host_setup; postlude/harness_api.sail model_set_start
    pub fn set_host_script(&mut self, mut events: Vec<HostEvent>, start: u64) {
        events.sort_by_key(|e| e.cycle); // stable
        let mut script = Vec::with_capacity(events.len());
        let mut dma = Vec::new();
        for e in events {
            match e.kind {
                HostKind::Dma(b) => dma.push((e.cycle, b)),
                _ => script.push(e),
            }
        }
        if script
            .iter()
            .any(|e| matches!(e.kind, HostKind::FlipSram { .. } | HostKind::FlipSrf { .. } | HostKind::FlipIq { .. }))
        {
            self.iq.retag_images(); // the program has `F` lines: the text words are tracked
        }
        self.host.script = script;
        self.host.pos = 0;
        self.host.dma = dma;
        self.host.dma_pos = 0;
        self.host.dma_last = None;
        self.host.dma_floor = 0;
        self.host.dma_blocked = false;
        self.host.start = start;
        self.apply_start();
    }

    /// Add timed host actions to the script of a chip that has begun to run: a host that acts while the chips run (the
    /// binding's `System`) names its DMA requests, CSR writes and injections one run at a time. Every event must lie at
    /// or after the first cycle the chip has not ticked (the caller's rule: an earlier event is never reached). The
    /// events join the ones not yet run in cycle order, after those of the same cycle already there (the stable order
    /// of [`Machine::set_host_script`]); the DMA engine's pacing and the start cycle are untouched. Not an addition to
    /// the model: the same events given up front to `set_host_script` run identically.
    // model: c_emulator/tsp_sim.c host_setup (the `H` lines, given while the chips run)
    pub fn host_extend(&mut self, events: Vec<HostEvent>) {
        for e in events {
            let h = &mut self.host;
            match e.kind {
                HostKind::Dma(b) => {
                    let at = h.dma_pos + h.dma[h.dma_pos..].partition_point(|(c, _)| *c <= e.cycle);
                    h.dma.insert(at, (e.cycle, b));
                }
                _ => {
                    let at = h.pos + h.script[h.pos..].partition_point(|x| x.cycle <= e.cycle);
                    h.script.insert(at, e);
                }
            }
        }
    }

    /// The queues parked by a HostWait that no ring has released yet: (queue, doorbell, cycle the HostWait dispatched), by
    /// queue. A host-side view of the state the model keeps in `hw_since`, `hw_db` and `hw_release`.
    // model: core/csr.sail host_wait; postlude/harness_api.sail hw_for_good
    pub fn host_waiting(&self) -> Vec<(usize, u8, u64)> {
        let b = &self.d.barrier;
        (0..crate::geom::QUEUES)
            .filter_map(|q| match (b.hw_since[q], b.hw_release[q]) {
                (Some(since), None) => Some((q, b.hw_db[q], since)),
                _ => None,
            })
            .collect()
    }

    /// Give the chip its control unit's `P` lines (`pk_setup`).
    // model: c_emulator/tsp_sim.c pk_setup
    pub fn set_ctl(&mut self, mut pk: Vec<PktEvent>) {
        pk.sort_by_key(|p| p.cycle); // stable
        self.host.pk = pk;
        self.host.pk_pos = 0;
    }

    fn apply_start(&mut self) {
        let start = self.host.start;
        if start != 0 {
            self.iq.set_start(start);
        }
        self.t_next_set(Some(start));
    }

    /// The host interface of this chip: registers, script, egress list.
    pub fn host(&self) -> &Host {
        &self.host
    }

    /// Did an error interrupt (MSI-X vector 0) reach the host? (`--on-irq`)
    // model: c_emulator/tsp_sim.c err_irq_seen
    pub fn error_irq_seen(&self) -> bool {
        self.host.err_irq > 0
    }

    /// `--on-irq stop`: end the run at the end of the cycle of the first error interrupt.
    pub fn set_stop_on_irq(&mut self, on: bool) {
        self.host.stop_on_irq = on;
    }

    /// Did the run end at an error interrupt (`--on-irq stop`)?
    pub fn stopped_at_irq(&self) -> bool {
        self.host.stopped
    }

    /// The cycle of the next action of this chip's host (`host_next_event`).
    pub(crate) fn host_next_event(&self) -> Option<u128> {
        self.host.next_event(self.geom.host_cycles_per_vector)
    }

    /// A read of the CSR `addr` (`model_csr_read`).
    // model: core/csr.sail csr_read
    pub fn service_ras(&self) -> &ServiceRas {
        &self.host.service_ras
    }

    pub fn csr_read(&self, addr: u64) -> u64 {
        let r = &self.host.ras;
        match addr {
            16 => return r.status,
            17 => return r.severity(),
            18 => return r.sbe,
            19 => return r.mbe,
            20 => return r.log.len() as u64,
            22 => return r.mask,
            a if (32..32 + 2 * self.geom.ras_log_depth).contains(&a) => {
                return match r.log.get(((a - 32) / 2) as usize) {
                    Some(e) => {
                        if a % 2 == 0 {
                            e.0
                        } else {
                            e.1
                        }
                    }
                    None => 0,
                };
            }
            1024 => return r.diag.len() as u64,
            1026 => return r.diag_overflow as u64,
            a if (1040..1040 + self.geom.ras_diag_depth).contains(&a) => {
                return r.diag.get((a - 1040) as usize).copied().unwrap_or(0);
            }
            240 => return r.first.map_or(0, |e| e.0),
            241 => return r.first.map_or(0, |e| e.1),
            a if (256..256 + 144).contains(&a) => return self.geom.fetch_latency,
            _ => {}
        }
        self.host.csr_read(addr, self.chip_id() as u64, self.geom.host_doorbells)
    }

    /// A write of the CSR `addr` at cycle `c` (`model_csr_write`): a write of a doorbell with bit 0 set rings it; a
    /// write of `IRQ_MASK` delivers every vector it enables whose cause bit is set.
    // model: core/csr.sail csr_write
    pub fn csr_write<O: Observer + ?Sized>(&mut self, addr: u64, v: u64, c: u64, obs: &mut O) {
        let doorbells = self.geom.host_doorbells;
        match addr {
            16 => {
                self.host.ras.status &= !v;
                if self.host.ras.status == 0 {
                    self.host.ras.first = None;
                }
            }
            18 => self.host.ras.sbe = 0,
            19 => self.host.ras.mbe = 0,
            21 => {
                if !self.host.ras.log.is_empty() {
                    self.host.ras.log.remove(0);
                }
            }
            22 => self.host.ras.mask = v,
            1025 => {
                if !self.host.ras.diag.is_empty() {
                    self.host.ras.diag.remove(0);
                }
            }
            1026 => {
                if v & 1 == 1 {
                    self.host.ras.diag_overflow = false;
                }
            }
            48 => self.host.irq_cause &= !v,
            49 => {
                let enabled = v & !self.host.irq_mask;
                self.host.irq_mask = v;
                for n in 0..16 {
                    if enabled >> n & 1 == 1 && self.host.irq_cause >> n & 1 == 1 {
                        self.host_deliver(c, n, obs);
                    }
                }
            }
            a if (768..768 + doorbells).contains(&a) => {
                let i = (a - 768) as usize;
                if v & 1 == 1 {
                    self.host_ring(i, c);
                } else {
                    self.host.doorbell[i] = false;
                }
            }
            _ => {}
        }
    }

    /// The host rings doorbell `i` at cycle `c`: the lowest queue parked on it resumes at `c + 1`, else the doorbell
    /// stays set until a HostWait takes it.
    // model: core/csr.sail host_ring
    fn host_ring(&mut self, i: usize, c: u64) {
        if !self.d.barrier.hw_ring(i, c) {
            self.host.doorbell[i] = true;
        }
    }

    /// HostWait `i` dispatched on queue `q` at cycle `c`: takes a ringing doorbell, or parks the queue.
    // model: core/csr.sail host_wait
    pub(crate) fn host_wait(&mut self, q: usize, c: u64, i: usize) {
        if self.host.doorbell[i] {
            self.host.doorbell[i] = false;
        } else {
            self.d.barrier.hw_park(q, c, i);
        }
    }

    /// An interrupt reaches the host in cycle `c`: MSI-X vector `n`.
    // model: core/csr.sail host_deliver
    fn host_deliver<O: Observer + ?Sized>(&mut self, c: u64, n: usize, obs: &mut O) {
        self.host.irq_count += 1;
        if n == 0 {
            self.host.err_irq += 1; // vector 0 is the error interrupt (HostNotify has n = 1..15)
        }
        obs.host_irq(c, n as u8);
    }

    /// HostNotify `n`: MSI-X vector `n` at cycle `c`. The cause bit is set; an enabled vector is delivered.
    // model: core/csr.sail host_msix
    pub(crate) fn host_msix<O: Observer + ?Sized>(&mut self, c: u64, n: usize, obs: &mut O) {
        self.ecc_flush(obs);
        self.host_msix_now(c, n, obs);
    }

    fn host_msix_now<O: Observer + ?Sized>(&mut self, c: u64, n: usize, obs: &mut O) {
        self.host.irq_cause |= 1 << n;
        if self.host.irq_mask >> n & 1 == 1 {
            self.host_deliver(c, n, obs);
        }
    }

    /// Set `SBE_COUNT` (a test of the saturation sets the register directly, like the Sail unit test).
    #[doc(hidden)]
    pub fn ras_set_sbe_count(&mut self, n: u64) {
        self.host.ras.sbe = n;
    }

    /// The model's ECC event as the error registers see it: `kind` 0 a stream operand (`id` the stream, `addr` the
    /// reader's gap, `isw` the row), 1 an SRAM word (`id` the slice), 2 an instruction text word (`addr` the word).
    // model: core/csr.sail ras_ecc_event
    #[allow(clippy::too_many_arguments)]
    pub fn ras_ecc_event<O: Observer + ?Sized>(
        &mut self,
        c: u64,
        q: usize,
        kind: u8,
        unc: bool,
        isw: bool,
        id: u64,
        addr: u64,
        sl: u8,
        pos: u32,
        obs: &mut O,
    ) {
        let loc = match kind {
            0 => addr << 6 | id << 1 | isw as u64,
            1 => addr << 7 | id,
            _ => addr << 8 | q as u64,
        };
        let k = if kind == 2 {
            if unc {
                4
            } else {
                3
            }
        } else if unc {
            2
        } else {
            1
        };
        self.ras_record(c, q, k, loc, sl, pos, (kind == 1).then_some((id as u8, addr as u32)), obs);
    }

    /// A word the patrol scrubber found at `addr` of MEM slice `s`: corrected (kind 1) or left as it is (kind 2); the queue of the entry is
    /// the slice's MEM queue.
    // model: core/csr.sail ras_scrub_event
    #[allow(clippy::too_many_arguments)]
    pub fn ras_scrub_event<O: Observer + ?Sized>(
        &mut self,
        c: u64,
        s: usize,
        unc: bool,
        a: u32,
        sl: u8,
        pos: u32,
        obs: &mut O,
    ) {
        self.ras_record(c, s, if unc { 2 } else { 1 }, (a as u64) << 7 | s as u64, sl, pos, Some((s as u8, a)), obs);
    }

    /// A spare-bit error report seen by its consumer: a log entry of kind 9 (`ERR_STATUS` bit 8) with the stream, the
    /// superlane and S[0..3] as the position.
    // model: core/csr.sail ras_spare_event
    #[allow(clippy::too_many_arguments)]
    pub fn ras_spare_event<O: Observer + ?Sized>(
        &mut self,
        c: u64,
        q: usize,
        isw: bool,
        id: u64,
        gap: u64,
        sl: u8,
        s03: u32,
        obs: &mut O,
    ) {
        let loc = gap << 6 | id << 1 | isw as u64;
        self.ras_record(c, q, 9, loc, sl, s03, None, obs);
    }

    /// A matrix-unit check event (`ras_mxm_event`): a corrected lane (kind 5) or an uncorrectable ABC (kind 6) of the ABC
    /// that executed on `plane` in cycle `c` from `queue`; log word 1 is `plane << 16 | lane`, the lane field 0xFFFF when the
    /// check located none (an uncorrectable ABC).
    // model: core/csr.sail ras_mxm_event
    pub fn ras_mxm_event<O: Observer + ?Sized>(
        &mut self,
        c: u64,
        queue: usize,
        plane: u8,
        lane: Option<usize>,
        obs: &mut O,
    ) {
        let (k, l) = match lane {
            Some(l) => (5, l as u64),
            None => (6, 0xFFFF),
        };
        self.ras_record(c, queue, k, (plane as u64) << 16 | l, 0, 0, None, obs);
    }

    /// A link FEC event (`ras_link_event`): a flit of a vector that arrived on `link` in cycle `c` corrected (kind 7) or
    /// detected uncorrectable (kind 8); no queue; log word 1 is `link << 16 | flit`.
    // model: core/csr.sail ras_link_event
    pub fn ras_link_event<O: Observer + ?Sized>(&mut self, c: u64, link: u8, flit: u8, unc: bool, obs: &mut O) {
        let k = if unc { 8 } else { 7 };
        self.ras_record(c, usize::MAX, k, (link as u64) << 16 | flit as u64, 0, 0, None, obs);
    }

    /// One error event: counted, logged (or dropped on overflow: `ERR_STATUS` bit 9) and, when it sets an enabled
    /// `ERR_STATUS` bit, the error interrupt (MSI-X vector 0) is delivered in cycle `c`. `kind` 1..8, `loc` the 43-bit
    /// location.
    // model: core/csr.sail ras_record
    #[allow(clippy::too_many_arguments)]
    fn ras_record<O: Observer + ?Sized>(
        &mut self,
        c: u64,
        q: usize,
        kind: u64,
        loc: u64,
        sl: u8,
        pos: u32,
        sram: Option<(u8, u32)>,
        obs: &mut O,
    ) {
        let depth = self.geom.ras_log_depth as usize;
        let r = &mut self.host.ras;
        if kind == 2 || kind == 4 || kind == 6 || kind == 8 {
            r.mbe = sat_inc(r.mbe);
        } else if kind != 9 {
            r.sbe = sat_inc(r.sbe);
        }
        let old = r.status;
        let w0 = (q as u64 & 0xFFF) << 52 | kind << 48 | (c & 0xFFFF_FFFF_FFFF);
        let w1 = (pos as u64 & 0x1FF) << 48 | (sl as u64 & 0x1F) << 43 | loc;
        let monitor = &mut self.host.service_ras;
        monitor.events = monitor.events.saturating_add(1);
        monitor.causes |= 1 << (kind - 1);
        monitor.overflow |= r.log.len() >= depth;
        monitor.first.get_or_insert((w0, w1));
        if matches!(kind, 2 | 4 | 6 | 8 | 9) {
            if monitor.fatal_count == 0 {
                monitor.sram_word = sram;
                monitor.first_fatal = Some((w0, w1));
            } else if monitor.sram_word != sram {
                monitor.sram_word = None;
            }
            monitor.fatal_count = monitor.fatal_count.saturating_add(1);
        }
        r.status |= 1 << (kind - 1);
        if r.first.is_none() {
            r.first = Some((w0, w1));
        }
        if r.log.len() < depth {
            r.log.push((w0, w1));
        } else {
            r.status |= 1 << 9;
        }
        if (r.status & !old) & r.mask != 0 {
            self.host_msix_now(c, 0, obs);
        }
    }

    /// PciReceive at cycle `c`: the oldest vector of the ingress FIFO. A DMA request that waits for a free slot enters
    /// in the next cycle.
    // model: slices/c2c/c2c_insts.sail execute C2C_PCIRECV; c_emulator/tsp_sim.c host_recv
    pub(crate) fn pci_recv_pop(&mut self, c: u64) -> [u8; MAX_LANES] {
        match self.host.ingress.pop_front() {
            Some(v) => {
                if self.host.dma_blocked {
                    self.host.dma_blocked = false;
                    self.host.dma_floor = u128::from(c) + 1;
                }
                v
            }
            None => [0; MAX_LANES],
        }
    }

    /// PciSend: the vector goes to the host, and the interface accepts no other for `host.cycles_per_vector` cycles.
    // model: slices/c2c/c2c_insts.sail execute C2C_PCISEND
    pub(crate) fn pci_send_put<O: Observer + ?Sized>(&mut self, ctx: &Issue, v: &[u8; MAX_LANES], obs: &mut O) {
        let c = ctx.exec_cycle();
        self.ecc_flush(obs); // the operand's ECC events precede the vector's `T dma`
        self.host.pci_accepted = Some(c);
        self.host.egress.push((c, *v));
        obs.host_dma(c, HostDma::Out);
    }

    /// The host's actions of cycle `t` (`host_process`): the DMA engine's, then the script's. Returns true when the
    /// model was changed: the cycle is then ticked, so that the model's own events are recomputed from it.
    // model: c_emulator/tsp_sim.c host_process
    pub(crate) fn host_process<O: Observer + ?Sized>(
        &mut self,
        t: u64,
        obs: &mut O,
    ) -> Result<bool, crate::machine::RunError> {
        let mut mutated = false;
        let cpv = self.geom.host_cycles_per_vector;
        while self.host.dma_eligible(cpv) == Some(u128::from(t)) {
            if self.host.ingress.len() as u64 >= self.geom.host_ingress_depth {
                self.host.dma_blocked = true;
                obs.host_dma(t, HostDma::Wait);
            } else {
                let b = std::mem::replace(&mut self.host.dma[self.host.dma_pos].1, Box::new([0; MAX_LANES]));
                self.host.dma_pos += 1;
                self.host.ingress.push_back(*b);
                self.host.dma_last = Some(t);
                obs.host_dma(t, HostDma::In);
            }
        }
        while self.host.script.get(self.host.pos).is_some_and(|e| e.cycle == t) {
            let ev = std::mem::replace(
                &mut self.host.script[self.host.pos],
                HostEvent { cycle: 0, kind: HostKind::Inject { queue: 0, units: Vec::new() } },
            );
            self.host.pos += 1;
            match ev.kind {
                HostKind::CsrWrite { addr, val } => {
                    self.csr_write(addr, val, t, obs);
                    mutated = true;
                }
                HostKind::CsrRead { addr } => {
                    let v = self.csr_read(addr);
                    obs.host_csr(t, addr, v);
                }
                HostKind::Inject { queue, units } => {
                    for u in units {
                        self.iq.inject(queue, t, u);
                    }
                    mutated = true;
                }
                HostKind::FlipSram { slice, addr, sl, bit } => {
                    self.ecc_flip_sram(slice, addr, sl as usize, bit as u32);
                    mutated = true;
                }
                HostKind::FlipSrf { w, stream, gap, sl, bit } => {
                    self.ecc_flip_srf(t, w, stream, gap, sl as usize, bit as u32);
                    mutated = true;
                }
                HostKind::FlipMxm { plane, lane, bit } => {
                    self.mxm.inject(crate::slices::mxm::MxmFault { plane, lane, bit, cycle: t });
                    mutated = true;
                }
                HostKind::FlipIq { queue, word, bit } => {
                    if !self.ecc_flip_iq(queue, word, bit as u32) {
                        return Err(crate::machine::RunError::Input(format!(
                            "F iq at cycle {t}: text word {word:x} of queue {queue:x} is not in the instruction queue"
                        )));
                    }
                    mutated = true;
                }
                HostKind::Dma(_) => unreachable!("DMA requests are kept apart"),
            }
        }
        Ok(mutated)
    }

    /// The control unit's packets of cycle `t` (`ctl_process`): the packets that arrived (consumed here) and the chip's
    /// own `P` lines. Returns true when a CSR was written (the cycle is then ticked).
    // model: c_emulator/tsp_sim.c ctl_process
    pub(crate) fn ctl_process<O: Observer + ?Sized>(&mut self, t: u64, obs: &mut O) -> bool {
        let mut mutated = false;
        let me = self.chip_id();
        for x in self.links.ctl_take(t) {
            obs.host_pkt(t, x.ptype == crate::ctl::PT_CTRL, x.link);
            if x.ptype == crate::ctl::PT_CTRL {
                continue;
            }
            let mut p = x.pkt;
            if p.op == Op::Reply {
                obs.host_csr_remote(t, p.target, p.addr, p.val);
            } else if p.op == Op::Wr {
                self.csr_write(p.addr, p.val, t, obs);
                mutated = true;
            } else {
                p.op = Op::Reply;
                p.val = self.csr_read(p.addr);
                // the answer goes back on the link the request came in
                self.links.ctl_add(p, crate::ctl::PT_CSR, x.link as usize, t);
            }
        }
        while self.host.pk.get(self.host.pk_pos).is_some_and(|k| k.cycle == t) {
            let k = self.host.pk[self.host.pk_pos];
            self.host.pk_pos += 1;
            match k.kind {
                PktKind::Ctrl => {
                    let p = CtlPkt { op: Op::Ctrl, origin: me, target: me, addr: 0, val: 0, id: k.id };
                    self.links.ctl_add(p, crate::ctl::PT_CTRL, k.link as usize, t);
                }
                PktKind::CsrRead { addr } | PktKind::CsrWrite { addr, .. } => {
                    let (op, val) = match k.kind {
                        PktKind::CsrWrite { val, .. } => (Op::Wr, val),
                        _ => (Op::Rd, 0),
                    };
                    let peer = self.links.peer(k.link).expect("a P line names a connected link");
                    let p = CtlPkt { op, origin: me, target: peer, addr, val, id: k.id };
                    self.links.ctl_add(p, crate::ctl::PT_CSR, k.link as usize, t);
                }
            }
        }
        mutated
    }
}
