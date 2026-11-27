//! N chips in lock step with the host interface of every chip, with no Python in it (design
//! the C2C execution model). `lib.rs` is the glue.
//!
//! The wiring, the link latencies and the lock-step windows are the ones of `tsp_sim --chips N`
//! (`tsp_iss::harness::links`, `harness::chips::drive`): `g` is the smallest next event over all chips and `L` the smallest
//! latency of a connected link (1 when none is); every chip runs the window `[g, g + L)`; the vectors sent in the window
//! are routed into the receivers' FIFOs, in chip order and sending order, at its end. A vector sent at `t >= g` arrives
//! at `t + L >= g + L`, so the result equals cycle-by-cycle execution. What differs from the harness is who drives:
//! the host (the caller) acts between runs. Every chip has its own host interface (CSR space, ingress FIFO and DMA
//! engine, doorbells, interrupts; `microarchitecture.md` §14): [`System::dma_write`] and [`System::csr_write`] are
//! the host's actions on one chip, [`System::egress`] the vectors PciSend gave it.
//!
//! A run stops at the end of a window, never inside one, so that every chip stands at the same cycle ([`System::cycle`])
//! and a stopped system resumes as if it had not stopped: when a chip delivered an interrupt (HostNotify), when a
//! diagnostic was reported, when the next event lies beyond the cycle limit, or when nothing is left to do.
#[path = "linked_overlap.rs"]
mod linked_overlap;

use crate::session::{Diagnostic, Error};
use ots_format::read_object;
use ots_profile::Profile;
use std::sync::{Arc, Mutex};
use tsp_iss::harness::cli::LinkMode;
use tsp_iss::harness::links::{ChipLinks, LinkPlan, PlanLinks};
use tsp_iss::host::{HostEvent, HostKind};
use tsp_iss::observe::{HostDma, Observer, Row};
use tsp_iss::{Geom, Machine};

type Result<T> = std::result::Result<T, Error>;

/// How the links of the chips are timed (the link options of `tsp_sim`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Links {
    /// `c2c.link_latency` on every link.
    Nominal,
    /// `--link-latency N`.
    Fixed(u64),
    /// `--link-latency-seed S`.
    Seed(u64),
    /// `--link-jitter-seed S`.
    Jitter(u64),
}

/// A stream write: cycle, row letter, stream, gap, value.
pub type SrfEvent = (u64, char, u8, u8, Vec<u8>);
/// An SRAM write: cycle, slice, address, word.
pub type SramEvent = (u64, u8, u32, Vec<u8>);

/// What one chip's run reported.
#[derive(Default)]
pub struct Rec {
    /// `trace`: keep the `T srf` and `T sram` events.
    pub trace: bool,
    pub diags: Vec<Diagnostic>,
    pub irqs: Vec<(u64, u8)>,
    pub dma: Vec<(u64, &'static str)>,
    pub csr: Vec<(u64, u64, u64)>,
    pub srf: Vec<SrfEvent>,
    pub sram: Vec<SramEvent>,
    event_clock: Option<Vec<u64>>,
}

impl Observer for Rec {
    fn horizon_event(&mut self, at: u64) {
        if let Some(times) = self.event_clock.as_mut() {
            times.push(at);
        }
    }
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        if self.trace {
            self.srf.push((c, row.letter(), stream, gap, v.to_vec()));
        }
    }
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        if self.trace {
            self.sram.push((c, slice, addr, v.to_vec()));
        }
    }
    fn diag(&mut self, c: u64, queue: u16, code: tsp_iss::Diag) {
        self.diags.push((c, queue, code.as_str().to_string()));
    }
    fn host_dma(&mut self, c: u64, what: HostDma) {
        self.dma.push((c, what.word()));
    }
    fn host_irq(&mut self, c: u64, vector: u8) {
        self.irqs.push((c, vector));
    }
    fn host_csr(&mut self, c: u64, addr: u64, value: u64) {
        self.csr.push((c, addr, value));
    }
}

struct Node {
    machine: Machine,
    links: Arc<Mutex<ChipLinks>>,
    rec: Rec,
    loaded: bool,
    /// Interrupts and diagnostics already handed to the caller by a run.
    irq_seen: usize,
    diag_seen: usize,
    finished: bool,
    /// The last cycle this chip ticked.
    last_tick: Option<u64>,
    active_ticks: u64,
    /// The object loaded (its bytes), for a restart.
    object: Option<Vec<u8>>,
}

/// Why a run returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Every chip is quiescent and nothing is left to happen.
    Done,
    /// A chip delivered an interrupt (HostNotify) in the last window.
    Notify,
    /// A chip reported a diagnostic in the last window.
    Diagnostic,
    /// The next event lies beyond the cycle limit.
    Limit,
}

impl Reason {
    pub fn name(self) -> &'static str {
        match self {
            Reason::Done => "done",
            Reason::Notify => "notify",
            Reason::Diagnostic => "diagnostic",
            Reason::Limit => "limit",
        }
    }
}

pub struct RunOutcome {
    pub reason: Reason,
    /// The cycle every chip stands at: all cycles below it are ticked.
    pub cycle: u128,
    /// The interrupts delivered by this run: (chip, cycle, vector).
    pub notifies: Vec<(usize, u64, u8)>,
    /// The diagnostics reported by this run: (chip, cycle, queue, code).
    pub diagnostics: Vec<(usize, Diagnostic)>,
}

pub struct System {
    profile: Profile,
    geom: Geom,
    plan: Arc<LinkPlan>,
    trace: bool,
    window: u64,
    nodes: Vec<Node>,
    /// Every cycle below `now` is ticked on every chip.
    now: u128,
    ran: bool,
    nominal_links: bool,
    linked_schedule: Option<LinkedSchedule>,
}

impl System {
    pub fn new(profile: &str, chips: usize, links: Links, window: Option<u64>, trace: bool) -> Result<System> {
        let profile = Profile::load(profile).map_err(|e| Error::Value(e.to_string()))?;
        let geom = Geom::new(&profile);
        if chips == 0 || chips as u64 > ots_format::MAX_WIRED_CHIPS {
            return Err(Error::Value(format!(
                "{chips} chips: the built-in wiring supports 1..{} chips",
                ots_format::MAX_WIRED_CHIPS
            )));
        }
        geom.check_link_latency().map_err(|m| Error::Value(format!("profile {}: {m}", profile.name())))?;
        geom.check_lanes().map_err(Error::Value)?;
        let mode = match links {
            Links::Nominal => LinkMode::Nominal,
            Links::Fixed(n) => LinkMode::Fixed(n),
            Links::Seed(s) => LinkMode::Seed(s),
            Links::Jitter(s) => LinkMode::Jitter(s),
        };
        let plan = Arc::new(LinkPlan::new(mode, &geom, chips).map_err(Error::Value)?);
        if chips > 1 && plan.window() == 0 {
            return Err(Error::Value("a system of several chips needs link latencies >= 1".into()));
        }
        // `--window N` of the harness: a shorter window, for the host that must act close to the cycle of an event; the results
        // do not depend on it (only the speed does)
        let window = match window {
            None => plan.window(),
            Some(0) => return Err(Error::Value("window: at least 1 cycle".into())),
            Some(n) => n.min(plan.window()),
        };
        let mut nodes = Vec::with_capacity(chips);
        for c in 0..chips {
            let port = PlanLinks::new(plan.clone(), c, &geom);
            let state = port.state();
            let mut machine = Machine::new(&profile, c as u16, Box::new(port)).map_err(Error::Value)?;
            machine.set_activity_counting(false);
            nodes.push(Node {
                machine,
                links: state,
                rec: Rec { trace, ..Rec::default() },
                loaded: false,
                irq_seen: 0,
                diag_seen: 0,
                finished: false,
                last_tick: None,
                active_ticks: 0,
                object: None,
            });
        }
        Ok(System {
            profile,
            geom,
            plan,
            trace,
            window,
            nodes,
            now: 0,
            ran: false,
            nominal_links: links == Links::Nominal,
            linked_schedule: None,
        })
    }

    /// Admit only an empty nominal four-chip system. The caller supplies a sealed data-only panel.
    pub(crate) fn prepare_linked_panel(&mut self) {
        if self.profile.name() == "ref"
            && self.nodes.len() == 4
            && self.nominal_links
            && self.window == 216
            && !self.trace
            && !self.ran
            && self.nodes.iter().all(|node| !node.loaded)
        {
            self.linked_schedule = Some(LinkedSchedule::new());
        }
    }

    pub fn chips(&self) -> usize {
        self.nodes.len()
    }

    pub fn profile_name(&self) -> &str {
        self.profile.name()
    }

    pub fn vlanes(&self) -> usize {
        self.geom.vlanes as usize
    }

    /// The window of the lock step (the smallest connected link latency, or the shorter one asked for).
    pub fn window(&self) -> u64 {
        self.window
    }

    /// The cycle every chip stands at.
    pub fn cycle(&self) -> u128 {
        self.now
    }

    fn node(&self, chip: usize) -> Result<&Node> {
        let n = self.nodes.len();
        self.nodes
            .get(chip)
            .ok_or_else(|| Error::Value(format!("chip {chip}: the system has {n} chips (0..{})", n - 1)))
    }

    fn node_mut(&mut self, chip: usize) -> Result<&mut Node> {
        let n = self.nodes.len();
        self.nodes
            .get_mut(chip)
            .ok_or_else(|| Error::Value(format!("chip {chip}: the system has {n} chips (0..{})", n - 1)))
    }

    /// The object of chip `chip`: its queues' text, before the first run. An object of another profile is refused.
    pub fn load_object(&mut self, chip: usize, elf: &[u8]) -> Result<()> {
        self.linked_schedule = None;
        let profile = self.profile.name().to_string();
        if self.ran {
            return Err(Error::Value("load_object: the system has run; load every object before the first run".into()));
        }
        let (_, images, _) = read_object(elf, Some(&profile), None)
            .map_err(|e| Error::Value(format!("chip {chip}: {}", e.render("").trim_start_matches(": "))))?;
        let node = self.node_mut(chip)?;
        if node.loaded {
            return Err(Error::Value(format!("chip {chip}: an object is loaded already")));
        }
        for (q, units) in images.iter() {
            node.machine.load_image(q, units).map_err(|e| Error::Value(format!("chip {chip}: {e}")))?;
        }
        node.loaded = true;
        node.object = Some(elf.to_vec());
        Ok(())
    }

    /// The restart of `platform.md` §1.1 on every chip at once: the state of a system just built and loaded (cycle 0, empty instruction queues, MXM and SXM state,
    /// stream registers, links, host interface with its ingress FIFO, egress, doorbells and interrupts, no diagnostics) with the SRAM of every chip kept, and
    /// the objects loaded again. A restarted system and a new system whose SRAM was written back word for word run the same: the weights and the pools
    /// stay where they are, and the host writes only what changes from run to run. Activity counts remain cumulative; only their
    /// cycle bookkeeping restarts, as in `Machine::reset`.
    pub fn restart(&mut self) -> Result<()> {
        for c in 0..self.nodes.len() {
            let port = PlanLinks::new(self.plan.clone(), c, &self.geom);
            let state = port.state();
            let node = &mut self.nodes[c];
            node.machine.restart_with_links(Box::new(port));
            node.links = state;
            node.rec = Rec { trace: self.trace, ..Rec::default() };
            node.loaded = false;
            node.irq_seen = 0;
            node.diag_seen = 0;
            node.finished = false;
            node.last_tick = None;
            node.active_ticks = 0;
        }
        self.now = 0;
        self.ran = false;
        self.linked_schedule = None;
        for c in 0..self.nodes.len() {
            if let Some(object) = self.nodes[c].object.take() {
                self.load_object(c, &object)?;
            }
        }
        Ok(())
    }

    /// Words `addr..addr+n` of `slice` of chip `chip` as `n * vlanes` bytes.
    pub fn read_words(&self, chip: usize, slice: u8, addr: u32, n: u32) -> Result<Vec<u8>> {
        let vl = self.vlanes();
        let node = self.node(chip)?;
        let end = u64::from(addr) + u64::from(n);
        if end > self.geom.words_per_slice {
            return Err(Error::Chip(format!(
                "chip {chip}: read_words: slice {slice} words {addr:#x}..{end:#x} go beyond the slice end {:#x}",
                self.geom.words_per_slice
            )));
        }
        let mut out = Vec::with_capacity(n as usize * vl);
        for k in 0..n {
            let w = node.machine.sram_word(slice, addr + k).map_err(|e| Error::Value(format!("chip {chip}: {e}")))?;
            out.extend_from_slice(&w[..vl]);
        }
        Ok(out)
    }

    /// Whole words from `addr` on, through the host's SRAM aperture, at the cycle the system stands at.
    pub fn write_words(&mut self, chip: usize, slice: u8, addr: u32, data: &[u8]) -> Result<()> {
        let vl = self.vlanes();
        if !data.len().is_multiple_of(vl) {
            return Err(Error::Value(format!("{} bytes are not a whole number of {vl}-byte words", data.len())));
        }
        let words = (data.len() / vl) as u64;
        let end = u64::from(addr) + words;
        if end > self.geom.words_per_slice {
            return Err(Error::Chip(format!(
                "chip {chip}: write_words: slice {slice} words {addr:#x}..{end:#x} go beyond the slice end {:#x}",
                self.geom.words_per_slice
            )));
        }
        let node = self.node_mut(chip)?;
        for (k, w) in data.chunks(vl).enumerate() {
            node.machine.preload(slice, addr + k as u32, w).map_err(|e| Error::Value(format!("chip {chip}: {e}")))?;
        }
        Ok(())
    }

    /// The first cycle at which the host may act now: the cycle the system stands at.
    fn host_cycle(&self, cycle: Option<u64>) -> Result<u64> {
        match cycle {
            None => u64::try_from(self.now)
                .map_err(|_| Error::Value(format!("host cycle {} is beyond the u64 timestamp domain", self.now))),
            Some(c) if u128::from(c) >= self.now => Ok(c),
            Some(c) => Err(Error::Value(format!("cycle {c} is in the past: the system stands at cycle {}", self.now))),
        }
    }

    /// The host's DMA engine takes a vector (`vlanes` bytes) for chip `chip`'s ingress FIFO, requested at `cycle` (default:
    /// the cycle the system stands at). It enters the FIFO at the later of that cycle and the previous arrival plus
    /// `host.cycles_per_vector`, and waits (nothing is dropped) while the FIFO is full. Returns the cycle of the request.
    pub fn dma_write(&mut self, chip: usize, vector: &[u8], cycle: Option<u64>) -> Result<u64> {
        let vl = self.vlanes();
        if vector.len() != vl {
            return Err(Error::Value(format!("a DMA vector is {vl} bytes (one per lane), not {}", vector.len())));
        }
        let at = self.host_cycle(cycle)?;
        let mut b = Box::new([0u8; tsp_iss::geom::MAX_LANES]);
        b[..vl].copy_from_slice(vector);
        self.node_mut(chip)?.machine.host_extend(vec![HostEvent { cycle: at, kind: HostKind::Dma(b) }]);
        Ok(at)
    }

    /// Submit immutable queue images through the same timed host-injection path as the reference harness.
    pub fn inject_text(&mut self, chip: usize, cycle: u64, images: Vec<(u16, Vec<u32>)>) -> Result<()> {
        let at = self.host_cycle(Some(cycle))?;
        let cap = self.geom.iq_units;
        let mut events = Vec::with_capacity(images.len());
        if let Some(schedule) = self.linked_schedule.as_mut() {
            if chip >= 4 {
                return Err(Error::Value("linked panel chip out of range".into()));
            }
            for (q, units) in &images {
                match q {
                    126 if !self.ran => schedule.tx[chip].push((cycle, units.clone())),
                    142 if !self.ran => schedule.rx[chip].extend_from_slice(units),
                    126 | 142 | 0xffff => schedule.disqualified = true,
                    _ if ots_isa::queue_name(*q).is_some_and(|name| name.starts_with("C2C.")) => {
                        schedule.disqualified = true;
                    }
                    _ => (),
                }
            }
        }
        for (q, units) in images {
            let queue = match q {
                0..=143 => usize::from(q),
                0xffff => 144,
                _ => return Err(Error::Value(format!("invalid instruction queue {q}"))),
            };
            if units.is_empty() || units.len() > cap {
                return Err(Error::Value(format!("queue {q}: expected 1..{cap} instruction units")));
            }
            events.push(HostEvent { cycle: at, kind: HostKind::Inject { queue, units } });
        }
        self.node_mut(chip)?.machine.host_extend(events);
        Ok(())
    }

    /// A write of CSR `addr` by the host of chip `chip` at `cycle` (default: now): a doorbell (768 + i) with bit 0 set rings
    /// it, `IRQ_CAUSE` (48) clears the bits written as ones, `IRQ_MASK` (49) enables vectors.
    pub fn csr_write(&mut self, chip: usize, addr: u64, val: u64, cycle: Option<u64>) -> Result<u64> {
        if let Some(schedule) = self.linked_schedule.as_mut() {
            schedule.disqualified = true;
        }
        let at = self.host_cycle(cycle)?;
        self.node_mut(chip)?.machine.host_extend(vec![HostEvent { cycle: at, kind: HostKind::CsrWrite { addr, val } }]);
        Ok(at)
    }

    /// A read of CSR `addr` of chip `chip` as the system stands (`CHIP_ID` 0, `IRQ_CAUSE` 48, `IRQ_MASK` 49, `IRQ_COUNT` 50,
    /// the doorbells from 768).
    pub fn csr_read(&self, chip: usize, addr: u64) -> Result<u64> {
        Ok(self.node(chip)?.machine.csr_read(addr))
    }

    /// The vectors PciSend gave the host of chip `chip` from the `start`-th on: (cycle, `vlanes` bytes).
    pub fn egress(&self, chip: usize, start: usize) -> Result<Vec<(u64, Vec<u8>)>> {
        let vl = self.vlanes();
        let node = self.node(chip)?;
        Ok(node.machine.host().egress.iter().skip(start).map(|(c, b)| (*c, b[..vl].to_vec())).collect())
    }

    pub fn observe_link_receives(&mut self) {
        for node in &mut self.nodes {
            node.links.lock().unwrap().observe_receives();
        }
    }

    pub fn drain_link_receives(&mut self, chip: usize) -> Result<Vec<(u64, u8)>> {
        Ok(self.node(chip)?.links.lock().unwrap().drain_receives())
    }

    /// The queues of chip `chip` parked by a HostWait that no ring has released: `(queue, doorbell, cycle)`.
    pub fn host_waiting(&self, chip: usize) -> Result<Vec<(usize, u8, u64)>> {
        Ok(self.node(chip)?.machine.host_waiting())
    }

    /// The ingress FIFO of chip `chip`: vectors in it, DMA requests that did not enter yet, whether the head request waits for a free slot.
    pub fn ingress(&self, chip: usize) -> Result<(usize, usize, bool)> {
        let h = self.node(chip)?.machine.host();
        Ok((h.ingress_len(), h.dma_pending(), h.dma_blocked()))
    }

    /// Count the activity of every chip (`slices::activity`) from now on, or stop. Off when the system is built.
    /// Collected counts survive disabling and restart; enabling resumes accumulation.
    pub fn set_activity_counting(&mut self, on: bool) {
        for n in &mut self.nodes {
            n.machine.set_activity_counting(on);
        }
    }

    /// The cumulative instructions chip `chip` issued while counting was enabled, including before a restart.
    pub fn activity_issued(&self, chip: usize) -> Result<u64> {
        Ok(self.node(chip)?.machine.activity().issued.iter().sum())
    }

    /// The cycles chip `chip` executed: one more than the last cycle it ticked (0 for a chip that never ticked).
    pub fn cycles(&self, chip: usize) -> Result<u128> {
        Ok(self.node(chip)?.last_tick.map_or(0, |t| u128::from(t) + 1))
    }

    /// Cycles in which this chip executed a tick, excluding fast-forward over idle cycles.
    pub fn active_ticks(&self, chip: usize) -> Result<u64> {
        Ok(self.node(chip)?.active_ticks)
    }

    pub fn inject_sram(&mut self, chip: usize, slice: u8, addr: u32, lane: u8, bit: u8) -> Result<()> {
        if let Some(schedule) = self.linked_schedule.as_mut() {
            schedule.disqualified = true;
        }
        self.node_mut(chip)?.machine.service_flip_sram(slice, addr, lane, bit).map_err(Error::Value)
    }

    pub fn service_scrub(&mut self) -> Result<()> {
        let cycle = u64::try_from(self.now)
            .map_err(|_| Error::Value("scrub cycle is beyond the u64 timestamp domain".into()))?;
        for node in &mut self.nodes {
            node.machine.service_scrub(cycle);
        }
        Ok(())
    }

    pub fn service_ras(&self, chip: usize) -> Result<&tsp_iss::host::ServiceRas> {
        Ok(self.node(chip)?.machine.service_ras())
    }

    pub fn diagnostics(&self, chip: usize) -> Result<&[Diagnostic]> {
        Ok(&self.node(chip)?.rec.diags)
    }

    pub fn irqs(&self, chip: usize) -> Result<&[(u64, u8)]> {
        Ok(&self.node(chip)?.rec.irqs)
    }

    pub fn dma_events(&self, chip: usize) -> Result<&[(u64, &'static str)]> {
        Ok(&self.node(chip)?.rec.dma)
    }

    pub fn trace(&self, chip: usize) -> Result<(&[SrfEvent], &[SramEvent])> {
        let r = &self.node(chip)?.rec;
        Ok((&r.srf, &r.sram))
    }

    /// `model_finish` on every chip, once, after a run that ended `done`: a queue with text that cannot dispatch raises
    /// OTS-E-IQ-004. Returns the diagnostics it added (chip, record).
    pub fn finish(&mut self) -> Result<Vec<(usize, Diagnostic)>> {
        if self.nodes.iter().any(|n| !n.machine.quiescent() || n.machine.report_time().is_some()) {
            return Err(Error::Chip("finish: the system is not done (a run to `done` comes first)".into()));
        }
        let mut added = Vec::new();
        for (c, n) in self.nodes.iter_mut().enumerate() {
            if n.finished {
                continue;
            }
            let before = n.rec.diags.len();
            n.machine.finish(&mut n.rec);
            n.finished = true;
            added.extend(n.rec.diags[before..].iter().cloned().map(|d| (c, d)));
        }
        Ok(added)
    }

    /// Return None before advancing if the closed set of sends cannot be established.
    pub(crate) fn run_linked_panel(&mut self, limit: u64) -> Result<Option<RunOutcome>> {
        if std::env::var("OPENTSP_ISS_CHIP_WORKERS").as_deref() != Ok("4") {
            return Ok(None);
        }
        let Some(schedule) = self.linked_schedule.as_mut() else { return Ok(None) };
        if schedule.disqualified || (schedule.sends.is_none() && schedule.compile(&self.profile, &self.plan).is_err()) {
            return Ok(None);
        }
        let sends = schedule.sends.as_ref().expect("compiled sends");
        let end = u128::from(limit) + 1;
        let Some(last) = sends.iter().filter_map(|source| source.last()).map(|p| p.ready).max() else {
            return Ok(None);
        };
        if self.now >= end || end - self.now > 1_000_000 {
            return Ok(None);
        }
        let terminal = end > last;
        if terminal {
            for node in &mut self.nodes {
                node.rec.event_clock = Some(Vec::new());
            }
        }
        let execution = linked_overlap::try_run(&mut self.nodes, self.now, end, self.window, sends, terminal)
            .unwrap_or_else(|| run_linked_schedule(&mut self.nodes, self.now, end, sends, terminal));
        let clocks =
            self.nodes.iter_mut().map(|node| node.rec.event_clock.take().unwrap_or_default()).collect::<Vec<_>>();
        let reason = execution?;
        self.ran = true;
        self.now = match reason {
            Reason::Done => original_done_clock(self.now, end, self.window, &clocks),
            _ => end,
        };
        Ok(Some(self.outcome(reason)))
    }

    /// Run in lock-step windows from the cycle the system stands at (see the module comment for when it returns).
    /// `limit`: no cycle beyond it is ticked. `stop_on_notify`, `stop_on_diag`: return at the end of the window in which a
    /// chip delivered an interrupt, or reported a diagnostic.
    pub fn run(&mut self, limit: Option<u64>, stop_on_notify: bool, stop_on_diag: bool) -> Result<RunOutcome> {
        let jobs = std::env::var("OPENTSP_ISS_CHIP_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .min(self.nodes.len());
        self.linked_schedule = None;
        if jobs > 1 {
            self.ran = true;
            let reason = run_parallel_windows(
                &mut self.nodes,
                &mut self.now,
                self.window,
                limit,
                stop_on_notify,
                stop_on_diag,
                jobs,
            )?;
            return Ok(self.outcome(reason));
        }
        let end = limit.map(|l| u128::from(l) + 1);
        self.ran = true;
        loop {
            let g = self.nodes.iter().filter_map(|n| n.machine.report_time()).min();
            let Some(g) = g else {
                for (c, n) in self.nodes.iter().enumerate() {
                    if !n.machine.quiescent() {
                        return Err(Error::Chip(format!("chip {c}: model not quiescent with no event left")));
                    }
                }
                return Ok(self.outcome(Reason::Done));
            };
            if limit.is_some_and(|l| g > u128::from(l)) {
                // nothing happens up to the limit: the system stands at the cycle after it
                self.now = self.now.max(end.unwrap());
                return Ok(self.outcome(Reason::Limit));
            }
            let mut wend = g + u128::from(self.window);
            if let Some(end) = end {
                wend = wend.min(end);
            }
            if wend <= g {
                return Err(Error::Chip("system cycle range exhausted".into()));
            }
            for (c, n) in self.nodes.iter_mut().enumerate() {
                let sum = n
                    .machine
                    .run_until_wide(wend, u64::MAX, &mut n.rec)
                    .map_err(|e| Error::Chip(format!("chip {c}: {e}")))?;
                n.last_tick = sum.last_cycle.or(n.last_tick);
                n.active_ticks += sum.ticks;
            }
            // route what the chips sent in the window, in chip order and sending order, then deliver
            let mut route: Vec<Vec<_>> = vec![Vec::new(); self.nodes.len()];
            for n in &self.nodes {
                for m in n.links.lock().unwrap().take_outbound() {
                    let to = m.chip;
                    route[to].push(m);
                }
            }
            for (n, msgs) in self.nodes.iter().zip(route) {
                let mut links = n.links.lock().unwrap();
                for m in msgs {
                    links.deliver(m);
                }
            }
            self.now = wend;
            let any_notify = self.nodes.iter().any(|n| n.rec.irqs.len() > n.irq_seen);
            let any_diag = self.nodes.iter().any(|n| n.rec.diags.len() > n.diag_seen);
            if (stop_on_notify && any_notify) || (stop_on_diag && any_diag) {
                let why = if stop_on_diag && any_diag { Reason::Diagnostic } else { Reason::Notify };
                return Ok(self.outcome(why));
            }
        }
    }

    /// Hand the interrupts and diagnostics since the last outcome to the caller.
    fn outcome(&mut self, reason: Reason) -> RunOutcome {
        let (mut notifies, mut diagnostics) = (Vec::new(), Vec::new());
        for (c, n) in self.nodes.iter_mut().enumerate() {
            notifies.extend(n.rec.irqs[n.irq_seen..].iter().map(|(cy, v)| (c, *cy, *v)));
            diagnostics.extend(n.rec.diags[n.diag_seen..].iter().cloned().map(|d| (c, d)));
            n.irq_seen = n.rec.irqs.len();
            n.diag_seen = n.rec.diags.len();
        }
        RunOutcome { reason, cycle: self.now, notifies, diagnostics }
    }
}

/// Execute independent chip work inside the existing link horizon.
fn run_parallel_windows(
    nodes: &mut [Node],
    now: &mut u128,
    window: u64,
    limit: Option<u64>,
    stop_on_notify: bool,
    stop_on_diag: bool,
    workers: usize,
) -> Result<Reason> {
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::Barrier;

    let cells: Vec<_> = nodes.iter_mut().map(Mutex::new).collect();
    let horizon = Mutex::new(Some(0u128));
    let begin = Barrier::new(workers + 1);
    let complete = Barrier::new(workers + 1);
    let errors = Mutex::new(Vec::<(usize, Error)>::new());
    let panics = Mutex::new(Vec::<(usize, Box<dyn std::any::Any + Send>)>::new());
    let end = limit.map(|l| u128::from(l) + 1);
    let update = |i: usize, wend: u128| -> Result<()> {
        let mut cell = cells[i].lock().unwrap();
        let node = &mut **cell;
        let sum = node
            .machine
            .run_until_wide(wend, u64::MAX, &mut node.rec)
            .map_err(|e| Error::Chip(format!("chip {i}: {e}")))?;
        node.last_tick = sum.last_cycle.or(node.last_tick);
        node.active_ticks += sum.ticks;
        Ok(())
    };

    std::thread::scope(|scope| {
        for worker in 0..workers {
            let update = &update;
            let begin = &begin;
            let complete = &complete;
            let horizon = &horizon;
            let errors = &errors;
            let panics = &panics;
            let count = cells.len();
            scope.spawn(move || loop {
                begin.wait();
                let Some(wend) = *horizon.lock().unwrap() else { break };
                for i in (worker..count).step_by(workers) {
                    match catch_unwind(AssertUnwindSafe(|| update(i, wend))) {
                        Ok(Ok(())) => (),
                        Ok(Err(error)) => errors.lock().unwrap().push((i, error)),
                        Err(payload) => panics.lock().unwrap().push((i, payload)),
                    }
                }
                complete.wait();
            });
        }
        let drive = catch_unwind(AssertUnwindSafe(|| -> Result<Reason> {
            let mut route = (0..cells.len()).map(|_| Vec::new()).collect::<Vec<_>>();
            let mut due = vec![None; cells.len()];
            loop {
                for (i, cell) in cells.iter().enumerate() {
                    due[i] = cell.lock().unwrap().machine.report_time();
                }
                let Some(g) = due.iter().flatten().copied().min() else {
                    for (i, cell) in cells.iter().enumerate() {
                        if !cell.lock().unwrap().machine.quiescent() {
                            return Err(Error::Chip(format!("chip {i}: model not quiescent with no event left")));
                        }
                    }
                    return Ok(Reason::Done);
                };
                if limit.is_some_and(|l| g > u128::from(l)) {
                    *now = (*now).max(end.unwrap());
                    return Ok(Reason::Limit);
                }
                let wend = end.map_or(g + u128::from(window), |e| e.min(g + u128::from(window)));
                if wend <= g {
                    return Err(Error::Chip("system cycle range exhausted".into()));
                }
                if due.iter().flatten().filter(|&&t| t < wend).count() >= 2 {
                    *horizon.lock().unwrap() = Some(wend);
                    begin.wait();
                    complete.wait();
                    let mut failed = errors.lock().unwrap();
                    let mut unwind = panics.lock().unwrap();
                    failed.sort_by_key(|(chip, _)| *chip);
                    unwind.sort_by_key(|(chip, _)| *chip);
                    if !unwind.is_empty() && (failed.is_empty() || unwind[0].0 < failed[0].0) {
                        resume_unwind(unwind.remove(0).1);
                    }
                    if !failed.is_empty() {
                        return Err(failed.remove(0).1);
                    }
                } else {
                    for i in 0..cells.len() {
                        update(i, wend)?;
                    }
                }
                for cell in &cells {
                    let n = cell.lock().unwrap();
                    for msg in n.links.lock().unwrap().take_outbound() {
                        route[msg.chip].push(msg);
                    }
                }
                for (cell, messages) in cells.iter().zip(&mut route) {
                    if !messages.is_empty() {
                        let n = cell.lock().unwrap();
                        let mut links = n.links.lock().unwrap();
                        for msg in messages.drain(..) {
                            links.deliver(msg);
                        }
                    }
                }
                *now = wend;
                let stop_diag = stop_on_diag
                    && cells.iter().any(|cell| {
                        let n = cell.lock().unwrap();
                        n.rec.diags.len() > n.diag_seen
                    });
                let stop_irq = stop_on_notify
                    && cells.iter().any(|cell| {
                        let n = cell.lock().unwrap();
                        n.rec.irqs.len() > n.irq_seen
                    });
                if stop_diag || stop_irq {
                    return Ok(if stop_diag { Reason::Diagnostic } else { Reason::Notify });
                }
            }
        }));
        *horizon.lock().unwrap() = None;
        begin.wait();
        drive.unwrap_or_else(|p| resume_unwind(p))
    })
}

// Private study of exclusive clock horizons from the actual submitted TX queue.
struct LinkedSchedule {
    tx: [Vec<(u64, Vec<u32>)>; 4],
    rx: [Vec<u32>; 4],
    sends: Option<[Vec<PredictedSend>; 4]>,
    disqualified: bool,
}

struct PredictedSend {
    sent: u128,
    ready: u128,
    target: usize,
    link: usize,
}

impl LinkedSchedule {
    fn new() -> Self {
        Self { tx: Default::default(), rx: Default::default(), sends: None, disqualified: false }
    }

    fn compile(&mut self, profile: &Profile, plan: &LinkPlan) -> std::result::Result<(), String> {
        use ots_format::text::{raise_images, Images};
        use ots_isa::{Insn, Isa};

        let isa = Isa::new(profile);
        for (chip, words) in self.rx.iter().enumerate() {
            let mut image = Images::new();
            image.extend(142, words);
            for parcel in raise_images(&image) {
                let insn =
                    isa.decode_word(parcel.word, parcel.units).map_err(|e| format!("chip {chip}: {e:?} in RX text"))?;
                if !matches!(insn, Insn::PciReceive { .. } | Insn::Receive { .. }) {
                    return Err(format!("chip {chip}: unsupported RX operation {}", insn.entry().mnemonic));
                }
            }
        }
        let mut sends: [Vec<PredictedSend>; 4] = Default::default();
        for (source, segments) in self.tx.iter().enumerate() {
            if segments.first().map(|entry| entry.0) != Some(0) || segments.windows(2).any(|w| w[0].0 >= w[1].0) {
                return Err(format!("source {source}: TX append order"));
            }
            let mut messages = Images::new();
            for (_, words) in segments {
                messages.extend(126, words);
            }
            let all = raise_images(&messages);
            // A Send at its static timestamp is the earliest possible delivery.
            // The dynamic comparison below also rejects any different actual timestamp.
            // PCI output and Repeat are admitted only on terminal chip 3.
            for parcel in all {
                let insn = isa
                    .decode_word(parcel.word, parcel.units)
                    .map_err(|e| format!("source {source}: {e:?} in TX text"))?;
                match insn {
                    Insn::Send { chip_id, link, .. } if source < 3 => {
                        let src_link = link as usize;
                        let Some((target, link)) = plan.peer(source, src_link) else {
                            return Err(format!("source {source}: disconnected TX in text"));
                        };
                        if target != source + 1 || target != chip_id as usize {
                            return Err(format!("source {source}: unexpected direction"));
                        }
                        let sent = u128::try_from(parcel.cycle).map_err(|_| "negative Send cycle")?;
                        let ready = sent + u128::from(plan.latency(source, src_link));
                        sends[source].push(PredictedSend { sent, ready, target, link });
                    }
                    _ if source == 3 && matches!(insn.entry().mnemonic, "pci_send" | "repeat") => (),
                    _ => return Err(format!("source {source}: unsupported TX operation {}", insn.entry().mnemonic)),
                }
            }
            if source < 3 && (sends[source].is_empty() || sends[source].windows(2).any(|w| w[0].sent >= w[1].sent)) {
                return Err(format!(
                    "source {source}: no sends or non-monotone TX, {} segments, {} sends",
                    segments.len(),
                    sends[source].len()
                ));
            }
        }
        self.sends = Some(sends);
        Ok(())
    }
}

fn run_linked_schedule(
    nodes: &mut [Node],
    now: u128,
    end: u128,
    sends: &[Vec<PredictedSend>; 4],
    may_finish: bool,
) -> Result<Reason> {
    use tsp_iss::ctl::PT_DATA;
    let mut progress = [now; 4];
    let mut indices: [usize; 4] = std::array::from_fn(|i| sends[i].partition_point(|p| p.sent < now));
    while progress.iter().any(|&cycle| cycle < end) {
        let mut horizon = [end; 4];
        for (source, expected) in sends.iter().enumerate() {
            if let Some(next) = expected.get(indices[source]) {
                horizon[next.target] = horizon[next.target].min(next.ready);
            }
        }
        if horizon == progress || horizon.iter().zip(progress).any(|(&to, from)| to < from) {
            return Err(Error::Chip("linked panel cannot advance while respecting unissued sends".into()));
        }
        let active = std::array::from_fn::<_, 4, _>(|i| {
            horizon[i] > progress[i] && nodes[i].machine.report_time().is_some_and(|next| next < horizon[i])
        });
        let parallel = active.iter().filter(|&&due| due).count() > 1;
        std::thread::scope(|scope| -> Result<()> {
            let mut jobs = Vec::new();
            for (i, node) in nodes.iter_mut().enumerate() {
                if horizon[i] == progress[i] {
                    continue;
                }
                let to = horizon[i];
                if !parallel || !active[i] {
                    advance_linked_schedule_node(node, i, to)?;
                } else {
                    jobs.push(scope.spawn(move || advance_linked_schedule_node(node, i, to)));
                }
            }
            for job in jobs {
                job.join().unwrap()?;
            }
            Ok(())
        })?;
        for (source, node) in nodes.iter().enumerate() {
            let traffic = node.links.lock().unwrap().take_outbound();
            let expected = &sends[source];
            let last = expected.partition_point(|p| p.sent < horizon[source]);
            let first = indices[source];
            if traffic.len() != last - first {
                return Err(Error::Chip(format!(
                    "linked panel packet count, source {source} [{first},{last}): actual {}",
                    traffic.len()
                )));
            }
            for (packet, prediction) in traffic.into_iter().zip(&expected[first..last]) {
                if packet.chip != prediction.target
                    || packet.link != prediction.link
                    || packet.arrival != prediction.ready
                    || packet.ptype != PT_DATA
                    || packet.ctl.is_some()
                    || packet.arrival < horizon[packet.chip]
                {
                    return Err(Error::Chip(format!(
                        "linked panel arrival violation: source {source}, target {}, arrival {}, target advanced to {}",
                        packet.chip, packet.arrival, horizon[packet.chip],
                    )));
                }
                nodes[packet.chip].links.lock().unwrap().deliver(packet);
            }
            indices[source] = last;
        }
        progress = horizon;
    }
    let next = nodes.iter().filter_map(|n| n.machine.report_time()).min();
    let reason = match next {
        Some(t) if t >= end => Reason::Limit,
        None if may_finish && nodes.iter().all(|n| n.machine.quiescent()) => Reason::Done,
        _ => return Err(Error::Chip("linked panel unexpectedly changed the public limit outcome".into())),
    };
    if nodes.iter().any(|node| node.rec.diags.len() != node.diag_seen || node.rec.irqs.len() != node.irq_seen) {
        return Err(Error::Chip("linked panel reported an unexpected interrupt or diagnostic".into()));
    }
    Ok(reason)
}

// In the data-only scope report_time is next_time. From each prior fence,
// the original driver starts each window at the earliest visited event at or
// after the prior exclusive bound. Recover the public clock by reproducing
// those bounds from this run's event clocks, without replaying any effects.
fn original_done_clock(from: u128, limit_end: u128, window: u64, clocks: &[Vec<u64>]) -> u128 {
    let mut bound = from;
    loop {
        let next = clocks
            .iter()
            .filter_map(|times| {
                let i = times.partition_point(|&cycle| u128::from(cycle) < bound);
                times.get(i)
            })
            .min();
        let Some(next) = next else {
            return bound;
        };
        bound = (u128::from(*next) + u128::from(window)).min(limit_end);
    }
}

#[cfg_attr(target_arch = "x86_64", inline(never))]
fn advance_linked_schedule_node(node: &mut Node, chip: usize, to: u128) -> Result<()> {
    let sum = node
        .machine
        .run_until_wide(to, u64::MAX, &mut node.rec)
        .map_err(|e| Error::Chip(format!("chip {chip}: {e}")))?;
    node.last_tick = sum.last_cycle.or(node.last_tick);
    node.active_ticks += sum.ticks;
    Ok(())
}
