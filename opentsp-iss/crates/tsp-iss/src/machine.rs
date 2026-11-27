//! One chip: the state of the Sail model and the harness's loop over its cycles.
//!
//! It holds the initial state (IQ images, SRAM preload) and the run loop: the machine ticks the
//! cycles `model_next_event` returns (`driver`), dispatches the text of the IQs, executes the units'
//! instructions and reports the diagnostics of what it dispatches.
//!
//! `Machine` has no global state, so any number of machines (chips, parallel vector runs) can live
//! in one process.
use crate::driver::Driver;
use crate::frontend::Iq;
use crate::geom::{queue_index, Geom, MAX_LANES, QUEUES, SLICES};
use crate::links::LinkPort;
use crate::observe::{Observer, Phase, Row};
use crate::sram::{Masks, Sram};
use crate::srf::{Srf, SrfLookup};
use ots_isa::{Isa, PADDING};
use ots_profile::{Profile, TimingTable};
use std::fmt;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use tsp_numerics::fault::ModelFault;
use tsp_numerics::hostfp;

/// An input the machine refuses (`tsp_sim`'s input errors, as a library error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadError {
    /// Not 0..143 and not 0xffff.
    Queue(u16),
    /// The slice is not 0..87.
    SliceRange(u64),
    /// The slice is not present (`slice mod 44 >= geometry.mem_slices_per_hemisphere`).
    SliceAbsent(u64),
    /// The address is not below `sram.words_per_slice`.
    Address(u64),
    /// More bytes than the profile has lanes (`geometry.superlanes` x 16).
    TooManyBytes { bytes: usize, lanes: usize },
    /// The queue's image would exceed `icu.iq_depth` bytes after the padding is dropped.
    ImageTooLong { queue: u16, units: usize, limit: usize },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Queue(q) => write!(f, "queue {q:#x} out of range (0..8f or ffff)"),
            LoadError::SliceRange(s) => write!(f, "slice {s:#x} out of range (0..57 hex)"),
            LoadError::SliceAbsent(s) => {
                write!(f, "slice {s:#x} does not exist (slice mod 44 >= geometry.mem_slices_per_hemisphere)")
            }
            LoadError::Address(a) => write!(f, "address {a:#x} out of range (>= words_per_slice)"),
            LoadError::TooManyBytes { bytes, lanes } => {
                write!(f, "{bytes} bytes: an S line has more bytes than the {lanes} lanes of this profile")
            }
            LoadError::ImageTooLong { queue, units, limit } => {
                write!(
                    f,
                    "initial IQ image of queue {queue:#x} has {units} units, more than icu.iq_depth allows ({limit})"
                )
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Why a run did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// The run needs `cycle`, beyond the limit (`tsp_sim` exit 3).
    MaxCycles { cycle: u128, limit: u64 },
    /// An internal fault: the model's invariant failed (`tsp_sim` exit 4).
    Internal(String),
    /// The input names something that is not there at run time: an `F iq` line for a text word that is not in the
    /// instruction queue (`tsp_sim` exit 2).
    Input(String),
}

impl RunError {
    /// A model fault at `cycle`, worded as `tsp_sim` words it. // model: c_emulator/tsp_sim.c check_model
    pub fn fault(cycle: u64, msg: &str) -> RunError {
        RunError::Internal(format!("model fault at cycle {cycle}: {msg}"))
    }
}

/// Run `f` for the tick of `cycle`: a [`ModelFault`] raised below it (the numerics) comes back as
/// `RunError::Internal`; any other panic goes on unchanged.
pub fn fault_boundary<R>(cycle: u64, f: impl FnOnce() -> R) -> Result<R, RunError> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| match payload.downcast::<ModelFault>() {
        Ok(fault) => RunError::fault(cycle, &fault.0),
        Err(other) => resume_unwind(other),
    })
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::MaxCycles { cycle, limit } => {
                write!(f, "the run needs cycle {cycle}, beyond --max-cycles {limit}; stopped (output incomplete)")
            }
            RunError::Internal(m) | RunError::Input(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RunError {}

/// What a finished run did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunSummary {
    /// Cycles ticked (not the cycles simulated: idle cycles are skipped).
    pub ticks: u64,
    /// The last cycle ticked, if any.
    pub last_cycle: Option<u64>,
}

/// The state of one queue, for a debugger and for tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueueState {
    /// Units waiting in the IQ.
    pub iq_units: usize,
    /// The most units the IQ has held at once, and the fewest it held after a dispatch (the units
    /// waiting now when nothing has dispatched).
    pub iq_max: usize,
    pub iq_min: usize,
}

pub struct Machine<const L: usize = { MAX_LANES }> {
    pub(crate) geom: Geom,
    chip_id: u16,
    pub(crate) links: Box<dyn LinkPort>,
    /// The instruction set of the profile (the decoder).
    pub(crate) isa: Isa,
    /// The latency of every catalogue entry.
    pub(crate) timing: TimingTable,
    /// The initial IQ image of each queue (padding dropped); `reset` of the model loads it.
    pub(crate) images: Vec<Vec<u32>>,
    pub(crate) sram: Sram,
    /// The stream register file (`srf_e`, `srf_w`).
    pub(crate) srf: Srf,
    pub(crate) masks: Masks,
    /// The MXM planes and the changes staged in the current cycle (`slices/mxm`).
    pub(crate) mxm: crate::slices::mxm::Mxm,
    /// Host-only predecoded weights for installed BF16 matrices.
    pub(crate) mxm_bf16_weights: Vec<Option<crate::slices::mxm::Bf16Cache>>,
    /// The SXM maps and the map loads staged in the current cycle (`slices/sxm`).
    pub(crate) sxm: crate::slices::sxm::Sxm,
    /// Segment size per slice for bounds checks (SetSegment). Reset: the whole slice.
    pub(crate) mem_segment: [u64; 88],
    /// The cycle at which each link last accepted a vector (`None`: never).
    // model: slices/c2c/c2c_insts.sail c2c_accepted
    pub(crate) c2c_accepted: [Option<u64>; crate::geom::MAX_LINKS],
    pub(crate) iq: Iq,
    /// The registers of the cycle driver.
    pub(crate) d: Driver<L>,
    /// The next cycle to tick; `None`: nothing left to do.
    pub(crate) t_next: Option<u128>,
    pub(crate) last_visited: Option<u64>,
    pub(crate) edit_failure: Option<String>,
    /// Tick every cycle up to the model's next event, not only the event (`--tick-every-cycle`).
    pub(crate) every_cycle: bool,
    /// With `every_cycle`: the cycle after the last one ticked.
    pub(crate) cursor: u128,
    /// The schedule record and replay of the run (`driver::replay`).
    pub(crate) rp: crate::driver::replay::Rp,
    /// The read and write gap of every queue's unit (`operands::GapTable`), a function of the profile.
    pub(crate) gaps: crate::operands::GapTable,
    /// The host interface: CSR registers, the host's script, the ingress FIFO and the egress list (`host`).
    pub(crate) host: crate::host::Host,
    /// SECDED ECC state: the overlays of faulted words and the current instruction's poison (`ecc`).
    pub(crate) ecc: crate::ecc::Ecc,
    /// What each unit class did (`slices::activity`); never read by the model.
    pub(crate) activity: crate::slices::activity::Activity,
}

/// A machine at the widest vector, whatever the profile: `Machine::new`.
impl Machine {
    /// A chip in reset with empty IQs and SRAM. Rejects profiles that fail
    /// [`Profile::validate`] or exceed the harness vector width.
    pub fn new(profile: &Profile, chip_id: u16, links: Box<dyn LinkPort>) -> Result<Machine, String> {
        Machine::with_lanes(profile, chip_id, links)
    }
}

impl<const L: usize> Machine<L> {
    /// [`Machine::new`] for a machine whose per-cycle records hold `L` lanes (`L` of at least the profile's `vlanes`): a profile of
    /// at most [`crate::geom::NARROW_LANES`] lanes runs at `L = NARROW_LANES` and pays nothing for the lanes of the widest profile.
    pub fn with_lanes(profile: &Profile, chip_id: u16, links: Box<dyn LinkPort>) -> Result<Machine<L>, String> {
        Geom::check_vector_width(profile.vlanes())?;
        if profile.vlanes() > L as u64 {
            return Err(format!("geometry.superlanes: {} lanes, this machine holds at most {L}", profile.vlanes()));
        }
        profile.validate().map_err(|e| e.to_string())?;
        let geom = Geom::new(profile);
        geom.check_link_latency()?;
        Ok(Self::fresh(geom, Isa::new(profile), profile.timing_table(), chip_id, links))
    }

    fn fresh(geom: Geom, isa: Isa, timing: TimingTable, chip_id: u16, links: Box<dyn LinkPort>) -> Self {
        let sram = Sram::new(&geom);
        Self {
            chip_id,
            links,
            isa,
            timing,
            images: vec![Vec::new(); QUEUES],
            sram,
            srf: Srf::new(geom.vlanes as usize),
            masks: Masks::new(geom.superlanes),
            mxm: crate::slices::mxm::Mxm::with(
                geom.mxm_planes as usize,
                geom.mxm_acc_depth as usize,
                geom.mxm_fp8_rate,
            ),
            mxm_bf16_weights: (0..2 * geom.mxm_planes as usize).map(|_| None).collect(),
            sxm: crate::slices::sxm::Sxm::new(),
            mem_segment: [geom.words_per_slice; 88],
            c2c_accepted: [None; crate::geom::MAX_LINKS],
            iq: Iq::new(&geom),
            d: Driver::with_lanes(geom.barrier_release_latency),
            t_next: Some(0),
            last_visited: None,
            edit_failure: None,
            every_cycle: false,
            cursor: 0,
            rp: crate::driver::replay::Rp::new(),
            gaps: crate::operands::GapTable::new(&geom),
            host: crate::host::Host::new(),
            ecc: crate::ecc::Ecc::new(),
            activity: crate::slices::activity::Activity::with_structure(
                geom.vlanes,
                geom.superlanes,
                crate::slices::activity::Structure {
                    vlanes: geom.vlanes,
                    mem_slices: 2 * geom.mem_slices,
                    mxm_planes: 2 * geom.mxm_planes,
                    mxm_rows: geom.mxm_rows.unwrap_or(0),
                    c2c_links: geom.links,
                },
            ),
            geom,
        }
    }

    /// Restart a chip with fresh link state. Preserve SRAM and cumulative activity;
    /// clear queue images, host state, ECC overlays and execution bookkeeping.
    /// The system coordinator reloads each chip's object after restarting the links.
    pub fn restart_with_links(&mut self, links: Box<dyn LinkPort>) {
        let mut fresh = Self::fresh(self.geom.clone(), self.isa.clone(), self.timing.clone(), self.chip_id, links);
        std::mem::swap(&mut fresh.activity, &mut self.activity);
        fresh.activity.restart();
        #[cfg(feature = "replay")]
        self.sram.journal_end();
        std::mem::swap(&mut fresh.sram, &mut self.sram);
        *self = fresh;
    }

    /// Enable or disable collection without changing architectural state or previous counts.
    pub fn set_activity_counting(&mut self, on: bool) {
        self.activity.set_enabled(on);
    }

    pub fn geom(&self) -> &Geom {
        &self.geom
    }

    pub fn chip_id(&self) -> u16 {
        self.chip_id
    }

    /// The link wiring of this chip.
    pub fn links(&self) -> &dyn LinkPort {
        self.links.as_ref()
    }

    /// **ISS extension:** the SHA-256 of every register of the machine except the SRAM contents (the IQs and the
    /// initial images, the SRF, the MXM, SXM, masks, segments, link acceptance times, the driver's registers and the
    /// clock). The fields of the units are taken over their `Debug` text, so a field added to a unit is covered; the
    /// fields of `Machine` itself are destructured without `..`, so a new one does not compile until it is hashed or
    /// named as left out here: the SRAM (its own digest), and the geometry, the instruction set, the timing table and
    /// the link port, which are configuration of the profile and the wiring, not state, and the activity counters, which
    /// the model never reads.
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        use std::fmt::Write as _;
        struct Sink(Sha256);
        impl std::fmt::Write for Sink {
            fn write_str(&mut self, s: &str) -> std::fmt::Result {
                self.0.update(s.as_bytes());
                Ok(())
            }
        }
        let Machine {
            geom: _,
            chip_id,
            links: _,
            isa: _,
            timing: _,
            images,
            sram: _,
            srf,
            masks,
            mxm,
            mxm_bf16_weights: _,
            sxm,
            mem_segment,
            c2c_accepted,
            iq,
            d,
            t_next,
            last_visited: _,
            edit_failure: _,
            every_cycle,
            cursor,
            rp: _,
            gaps: _,
            host,
            ecc,
            activity: _,
        } = self;
        let mut h = Sink(Sha256::new());
        let _ = write!(
            h,
            "chip_id={chip_id:?}|images={images:?}|srf={srf:?}|masks={masks:?}|mxm={mxm:?}|sxm={sxm:?}\
             |mem_segment={mem_segment:?}|c2c_accepted={c2c_accepted:?}|iq={iq:?}|d={d:?}\
             |t_next={t_next:?}|every_cycle={every_cycle:?}|cursor={cursor:?}|host={host:?}|ecc={ecc:?}"
        );
        h.0.finalize().into()
    }

    /// Append text units to a queue's initial IQ image. Padding units (`ffffffff`) are dropped;
    /// the image may hold at most `icu.iq_depth / 4` units (`Q` lines of one queue append).
    pub fn load_image(&mut self, queue: u16, units: &[u32]) -> Result<(), LoadError> {
        let i = queue_index(queue).ok_or(LoadError::Queue(queue))?;
        let limit = self.geom.iq_units;
        let img = &mut self.images[i];
        for &u in units {
            if u == PADDING {
                continue;
            }
            if img.len() >= limit {
                self.iq.parse(&self.isa, i);
                return Err(LoadError::ImageTooLong { queue, units: img.len() + 1, limit });
            }
            img.push(u);
            // the model's `reset` pushes the image into the IQ; the machine is in reset until its
            // first tick, so the IQ takes it now
            self.iq.push(i, u);
            self.d.touch(i);
        }
        self.iq.parse(&self.isa, i);
        Ok(())
    }

    /// Set an SRAM word before cycle 0: more than `vlanes` bytes are an error, missing lanes are zero.
    pub fn preload(&mut self, slice: u8, addr: u32, bytes: &[u8]) -> Result<(), LoadError> {
        self.check_word(slice as u64, addr as u64)?;
        if bytes.len() as u64 > self.geom.vlanes {
            return Err(LoadError::TooManyBytes { bytes: bytes.len(), lanes: self.geom.vlanes as usize });
        }
        self.sram.write(slice as usize, addr as u64, bytes);
        Ok(())
    }

    /// Qualification port equivalent to a harness F sram at this boundary.
    pub fn service_flip_sram(&mut self, slice: u8, addr: u32, lane: u8, bit: u8) -> Result<(), String> {
        self.check_word(u64::from(slice), u64::from(addr)).map_err(|e| e.to_string())?;
        if u64::from(lane) >= self.geom.superlanes || bit > 137 {
            return Err("fault outside the protected word".into());
        }
        self.ecc_flip_sram(slice, addr, usize::from(lane), u32::from(bit));
        Ok(())
    }

    fn check_word(&self, slice: u64, addr: u64) -> Result<(), LoadError> {
        if slice >= SLICES as u64 {
            return Err(LoadError::SliceRange(slice));
        }
        if !self.geom.slice_present(slice) {
            return Err(LoadError::SliceAbsent(slice));
        }
        if addr >= self.geom.words_per_slice {
            return Err(LoadError::Address(addr));
        }
        Ok(())
    }

    /// The SRAM word at `slice`, `addr` (a `W` line): `MAX_LANES` bytes, lanes at or beyond `vlanes` zero.
    pub fn sram_word(&self, slice: u8, addr: u32) -> Result<[u8; MAX_LANES], LoadError> {
        self.check_word(slice as u64, addr as u64)?;
        Ok(self.sram.read(slice as usize, addr as u64))
    }

    /// The state of queue `q` (0..143 or 0xffff); `None` for any other id.
    pub fn queue_state(&self, q: u16) -> Option<QueueState> {
        queue_index(q).map(|i| QueueState {
            iq_units: self.iq.len[i],
            iq_max: self.iq.hi[i],
            iq_min: self.iq.lo[i].min(self.iq.len[i]),
        })
    }

    /// The vector that stream `stream` of `row` holds at register `gap` in `cycle`, if one was
    /// written (the flow rule of `srf_read`: lanes at or beyond `vlanes` are zero).
    pub fn srf_value(&self, row: Row, stream: u8, gap: u8, cycle: u64) -> Option<[u8; MAX_LANES]> {
        match self.srf.read(row.into(), stream, gap, cycle as i128) {
            (SrfLookup::Found, v) => Some(v),
            _ => None,
        }
    }

    /// `model_next_event`: the next cycle after `c` at which anything happens, or `None`.
    pub fn next_event(&self, c: u64) -> Option<u128> {
        self.model_next_event(c)
    }

    /// `model_quiescent`: nothing is left to do.
    pub fn quiescent(&self) -> bool {
        if let Some(q) = self.rp_quiescent() {
            return q;
        }
        self.model_idle()
    }

    /// Reset the architectural state of the chip and load the initial IQ images.
    // model: postlude/tick.sail reset
    pub fn reset(&mut self) {
        self.last_visited = None;
        self.edit_failure = None;
        self.d = Driver::with_lanes(self.geom.barrier_release_latency);
        self.activity.restart(); // the cycles count from 0 again; the counts go on
        self.masks.reset(self.geom.superlanes);
        self.mem_segment = [self.geom.words_per_slice; 88]; // model: slices/mem/mem_insts.sail mem_reset
        self.c2c_accepted = [None; crate::geom::MAX_LINKS]; // model: slices/c2c/c2c_insts.sail c2c_reset
        self.host.reset(); // model: core/csr.sail host_reset
        self.ecc.reset();
        self.srf.reset();
        self.mxm = crate::slices::mxm::Mxm::with(
            self.geom.mxm_planes as usize,
            self.geom.mxm_acc_depth as usize,
            self.geom.mxm_fp8_rate,
        ); // model: slices/mxm/mxm_regs.sail mxm_reset
        self.mxm_bf16_weights.iter_mut().for_each(|entry| *entry = None);
        self.sxm = crate::slices::sxm::Sxm::new(); // model: slices/sxm/sxm_regs.sail sxm_reset
        self.iq.clear();
        for q in 0..crate::geom::QUEUES {
            for (k, &u) in self.images[q].iter().enumerate() {
                self.iq.push_image(q, u, k);
            }
            self.iq.image_done(q, self.images[q].len());
            self.iq.parse(&self.isa, q);
        }
        self.t_next = Some(u128::from(self.host.start));
        if self.host.start != 0 {
            self.iq.set_start(self.host.start);
        }
        self.cursor = 0;
        self.rp.rewind();
        if self.rp.compiled {
            self.t_next = self.rp_next(None);
        }
    }

    pub(crate) fn t_next_set(&mut self, t: Option<u64>) {
        self.t_next = t.map(u128::from);
    }

    /// Tick every cycle from the last one ticked to the model's next event, instead of only that
    /// event (a check that the cycles between events change nothing).
    pub fn set_tick_every_cycle(&mut self, on: bool) {
        self.every_cycle = on;
    }

    /// The next cycle to tick, or `None` when the run is over.
    pub fn next_tick(&self) -> Option<u128> {
        self.t_next
    }

    /// The next cycle at which the chip acts: its next tick or a link event (an arrival that could
    /// overrun, a drop to report), `None` when nothing is left.
    // model: c_emulator/tsp_sim.c next_time
    pub fn next_time(&self) -> Option<u128> {
        let min = |a: Option<u128>, b: Option<u128>| match (a, b) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let link = min(
            min(self.links.next_event(), self.host_next_event()),
            min(min(self.links.ctl_next_event(), self.host.pk_next_event().map(u128::from)), self.poll_event()),
        );
        let next = match (link, self.t_next) {
            (Some(a), Some(t)) => Some(a.min(t)),
            (a, t) => a.or(t),
        };
        // In stop mode a patrol slot can raise an error interrupt. Visit it
        // before later host/link activity, while real work remains; patrol
        // alone must not keep a quiescent machine running indefinitely.
        if self.host.stop_on_irq && self.geom.scrub_period != 0 {
            next.map(|t| t.min(self.d.scrub_next))
        } else {
            next
        }
    }

    /// What the chip reports to the lock-step coordinator: its next time, or the earliest cycle at which a control packet
    /// that waits for a gap in the schedule could leave, whichever is earlier (`report_time`).
    // model: c_emulator/tsp_sim.c report_time
    pub fn report_time(&self) -> Option<u128> {
        match (self.next_time(), self.links.ctl_wake()) {
            (Some(a), Some(w)) => Some(a.min(w)),
            (a, w) => a.or(w),
        }
    }

    /// Tick every cycle that has something to do, below `wend` (`tsp_sim.c` `run_until`), stopping
    /// with [`RunError::MaxCycles`] before a cycle beyond `max_cycles`. The dropped arrivals of a
    /// cycle are reported (OTS-E-LNK-002, queue 65535) before its tick, also in a cycle with no
    /// tick. With [`Machine::set_tick_every_cycle`] the idle cycles between events are
    /// ticked too. Returns the cycles ticked.
    pub fn run_until<O: Observer + ?Sized>(
        &mut self,
        wend: u64,
        max_cycles: u64,
        obs: &mut O,
    ) -> Result<RunSummary, RunError> {
        self.run_horizon(Some(u128::from(wend)), max_cycles, obs)
    }

    /// Exclusive wide horizon; every executed callback cycle is checked against the u64 budget.
    pub fn run_until_wide<O: Observer + ?Sized>(
        &mut self,
        wend: u128,
        max_cycles: u64,
        obs: &mut O,
    ) -> Result<RunSummary, RunError> {
        self.run_horizon(Some(wend), max_cycles, obs)
    }

    pub(crate) fn run_horizon<O: Observer + ?Sized>(
        &mut self,
        wend: Option<u128>,
        max_cycles: u64,
        obs: &mut O,
    ) -> Result<RunSummary, RunError> {
        if let Some(message) = &self.edit_failure {
            return Err(RunError::Input(message.clone()));
        }
        // f32 add, sub and mul run on this thread's FPU: refuse an environment they would be wrong in
        hostfp::check_host_fpu().map_err(RunError::Internal)?;
        self.rp_start(obs.tracks_waits());
        if crate::driver::replay::ENABLED && (self.rp.compiled | self.rp.rec_on) {
            self.run_ticks_replay(wend, max_cycles, obs)
        } else {
            self.run_ticks::<O, false>(wend, max_cycles, obs)
        }
    }

    /// The loop of a run that records or replays a schedule, out of line.
    #[cold]
    #[inline(never)]
    fn run_ticks_replay<O: Observer + ?Sized>(
        &mut self,
        wend: Option<u128>,
        max_cycles: u64,
        obs: &mut O,
    ) -> Result<RunSummary, RunError> {
        self.run_ticks::<O, true>(wend, max_cycles, obs)
    }

    /// The loop of [`Machine::run_until`]; `REPLAY` is whether the run records or replays a schedule (`driver::replay`),
    /// chosen once, so that the loop of a normal run has no test for it.
    fn run_ticks<O: Observer + ?Sized, const REPLAY: bool>(
        &mut self,
        wend: Option<u128>,
        max_cycles: u64,
        obs: &mut O,
    ) -> Result<RunSummary, RunError> {
        let mut sum = RunSummary::default();
        let prof = obs.profiling();
        self.d.early_close =
            self.d.fast_on && self.d.index_on && !self.d.index_check && !self.every_cycle && !self.iq.observe_sites;
        loop {
            if self.links.ctl_pending() {
                // the control packets whose departure is now certain
                let rn = self.next_time();
                self.links.ctl_depart(wend, rn, max_cycles)?;
            }
            let Some(event) = self.next_time() else { break };
            if event > u128::from(max_cycles) {
                if wend.is_none_or(|end| event < end) {
                    return Err(RunError::MaxCycles { cycle: event, limit: max_cycles });
                }
                break;
            }
            let idle = self.every_cycle && self.t_next.is_some_and(|n| n > self.cursor);
            let t = if idle { event.min(self.cursor) } else { event };
            if wend.is_some_and(|end| t >= end) {
                break;
            }
            let wide_t = t;
            let patrol_due = self.host.stop_on_irq && self.geom.scrub_period != 0 && self.d.scrub_next == wide_t;
            let t = u64::try_from(t).expect("time is within the checked u64 budget");
            obs.horizon_event(t);
            self.scrub_run(t.checked_sub(1), obs); // the patrol scrub cycles before t, before anything acts at the start of t
            for e in self.links.link_events(t) {
                match e {
                    crate::links::LinkEvent::Overrun { cycle } => obs.diag(cycle, 0xffff, ots_isa::Code::Lnk002),
                    crate::links::LinkEvent::Fec { cycle, link, unc, flit } => {
                        obs.link_fec(cycle, link, unc, flit);
                        self.host.link_fec(link, unc);
                        self.ras_link_event(cycle, link, flit, unc, obs);
                        if unc {
                            self.ecc.live = true; // the Receive that takes the vector poisons its superlanes
                        }
                    }
                }
            }
            let mut mutated = self.host_process(t, obs)?; // the host's actions of cycle t
            if self.ctl_process(t, obs) {
                mutated = true; // the control unit's: arrived csr and ctrl packets
            }
            if self.poll_event() == Some(wide_t) {
                mutated = true; // an epoch boundary at which a DeskewRecv looks for its vector
            }
            if self.links.ctl_pending() {
                self.links.ctl_depart(wend, Some(u128::from(t)), max_cycles)?; // the packets made in cycle t that leave in it, before the tick
            }
            if (mutated || patrol_due || (self.host.stop_on_irq && self.host.err_irq > 0))
                && self.t_next.is_none_or(|n| n > wide_t)
            {
                // Complete the IRQ cycle even when an external arrival, rather
                // than an instruction, raised the interrupt before the tick.
                self.t_next = Some(wide_t);
            }
            if self.t_next != Some(wide_t) && !(idle && wide_t == self.cursor) {
                continue; // an arrival with nothing to tick
            }
            let m = crate::driver::phase::begin();
            if REPLAY {
                self.try_tick(t, obs)?;
            } else {
                self.try_tick_normal(t, obs)?;
            }
            if prof {
                obs.phase(Phase::Event);
            }
            // `F mxm`: a fault no ABC consumed in its cycle is an input error, found after the tick of that cycle
            // model: postlude/harness_api.sail model_mxm_pending; c_emulator/tsp_sim.c main loop after the tick
            if let Some((plane, _)) = self.mxm.pending() {
                return Err(RunError::Input(format!(
                    "F mxm: no ABC executes on plane {plane} in cycle {t:x} (chip {})",
                    self.chip_id()
                )));
            }
            sum.ticks += 1;
            sum.last_cycle = Some(t);
            self.cursor = wide_t + 1;
            crate::driver::phase::end(14, m);
            if self.host.stop_on_irq && self.host.err_irq > 0 {
                self.host.stopped = true; // --on-irq stop: the end of the cycle of the first error interrupt
                return Ok(sum);
            }
            let m = crate::driver::phase::begin();
            self.t_next = match if REPLAY && self.rp.compiled { self.rp_next(Some(t)) } else { self.next_event(t) } {
                Some(ev) if ev <= wide_t => {
                    return Err(RunError::fault(t, &format!("bad next event {ev} from the model")))
                }
                ev => ev,
            };
            crate::driver::phase::end(13, m);
        }
        Ok(sum)
    }

    /// The end-of-run report (`model_finish`), called once after [`Machine::quiescent`] holds: each queue whose IQ still
    /// holds text that cannot dispatch and that is not parked at a Sync no Notify releases raises OTS-E-IQ-004, in
    /// ascending queue order with the control channel last, at the cycle of the queue's last IQ activity.
    // model: postlude/harness_api.sail model_finish
    pub fn finish<O: Observer + ?Sized>(&mut self, obs: &mut O) {
        if self.rp_quiescent().is_some() {
            // a compiled replay keeps no IQ contents: it reports the end-of-run records the recording found
            let ends = self.rp.exp.as_ref().map(|s| s.end_diags.clone()).unwrap_or_default();
            for (q, c) in ends {
                self.diag(obs, q as usize, c, ots_isa::Code::Iq004);
            }
            return;
        }
        let b = &self.d.barrier;
        let polling = |k: usize| k < crate::geom::DATA_QUEUES && self.d.queue_poll[k].is_some();
        let parked_for_good = |k: usize| {
            matches!((b.parked_since[k], b.release_at[k]), (Some(_), None)) || b.hw_for_good(k) || polling(k)
        };
        let control_stuck = parked_for_good(144);
        let mut found: Vec<(u16, u64)> = Vec::new();
        for q in 0..crate::geom::DATA_QUEUES {
            if self.iq.len[q] > 0 && !(parked_for_good(q) || control_stuck) && self.iq.stranded(&self.isa, q) {
                found.push((q as u16, self.iq.active[q]));
            }
        }
        if self.iq.len[144] > 0 && !(control_stuck) && self.iq.stranded(&self.isa, 144) {
            found.push((144, self.iq.active[144]));
        }
        for &(q, c) in &found {
            self.diag(obs, q as usize, c, ots_isa::Code::Iq004);
        }
        self.report_parked(obs);
        if self.rp.rec_on {
            self.rp.rec.end_diags = found;
        }
    }

    /// Run to the end: tick `model_next_event` cycles from cycle 0 until nothing is left
    /// then check that the model is quiescent.
    pub fn run<O: Observer + ?Sized>(&mut self, max_cycles: u64, obs: &mut O) -> Result<RunSummary, RunError> {
        let s = self.run_horizon(None, max_cycles, obs)?;
        if self.host.stopped {
            return Ok(s); // --on-irq stop: no end-of-run checks, the chip did not run to its end
        }
        #[cfg(feature = "opprof")]
        crate::slices::opprof::dump();
        crate::driver::phase::report(crate::driver::phase::completed_ticks());
        if !self.quiescent() {
            return Err(RunError::Internal("model not quiescent with no event left".into()));
        }
        self.finish(obs);
        self.rp_finish()?;
        #[cfg(feature = "replay")]
        if self.rp.text_watch.take().is_some() {
            self.sram.journal_end();
        }
        Ok(s)
    }
}
