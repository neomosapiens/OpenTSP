//! The platform interface of one chip: what a host does around a run (`platform.md` §1–§7).
//!
//! [`Platform`] wraps a [`Machine`] behind the host-visible register set. The host resets the chip,
//! appends the program to the IQ load windows ([`Platform::write_iq_window`]), writes the data
//! through the SRAM aperture ([`Platform::write_sram_word`]), sets the C2C peers, writes the start
//! bit ([`Platform::start`]) and reads the results and the diagnostics when the chip is done. The
//! register offsets and widths are a convention of the document, so the registers are methods.
//!
//! * **Start, done, stop.** The start bit defines cycle 0. [`Platform::advance`] lets the running
//!   chip execute cycles; the chip is done when the model is quiescent (`STATUS.quiescent`, and the
//!   interrupt of [`Platform::irq_pending`]). [`Platform::stop`] freezes it in the current cycle.
//!   A chip that is done or stopped starts again only after [`Platform::reset`], which clears the
//!   SRAM, the IQs, the cycle counter and every status and diagnostic register; the C2C peers and
//!   the interrupt enables are configuration and survive it.
//! * **Cycle counter.** `CYCLE` counts superlane-0 cycles from the start bit: the number of cycles
//!   executed so far. When the chip is done it holds one more than the last cycle that did anything.
//! * **IQ load window.** One window per queue (0..143 and the control channel `0xffff`), selected by
//!   `IQ_SELECT`. Padding units are dropped; an append that would exceed `icu.iq_depth` is refused
//!   whole and sets `STATUS.load_error`, which stays set until reset. Writable before start only.
//! * **SRAM aperture.** Addressed by (slice, word, superlane) in `SRAM_ADDR`; `SRAM_DATA` moves the
//!   16 lane bytes of that tile. Usable while the chip is not running.
//! * **Diagnostics.** The chip reports every diagnostic of the model to [`Platform::diagnostics`] (the
//!   simulator's full list). The error FIFO and `ERR_FIRST` take only the records that the log filter of
//!   `microarchitecture.md` §16.3 admits, a superset of the codes of class `data` of §8: OTS-E-SXM-001, OTS-E-DEC-003, OTS-E-IQ-002,
//!   OTS-E-IQ-004 (an end-of-run record: it follows every run-time record, whatever its stamp), OTS-E-OPR-001 of
//!   a MEM or an `sxselect` queue (Gather, Scatter, `sxselect`; the filter keeps every MEM queue, a documented
//!   superset), and OTS-E-TIM-001 of a C2C receive queue. The FIFO holds
//!   `ras.diag_depth` records ([`ERR_FIFO_DEPTH`], 16, when the profile has no such key); a record that finds it full is dropped, sets
//!   `STATUS.fifo_overflow` (sticky until reset) and is not counted. `ERR_FIRST` keeps the first
//!   record raised since reset, whatever the FIFO does.
//!
//! ```
//! use tsp_iss::platform::Platform;
//!
//! let profile = ots_profile::Profile::load("mini").unwrap();
//! let mut p = Platform::new(&profile, 0).unwrap();
//! // the program: a MEM Read of slice 4, address 1 onto E:0, and a MEM Write of slice 0, address 3
//! p.select_iq(4).unwrap();
//! p.write_iq_window(&[0x14, 0x1000_0001, 0]).unwrap();
//! p.select_iq(0).unwrap();
//! p.write_iq_window(&[0x1a, 0x1040_0003, 0]).unwrap();
//! // the data
//! p.write_sram_word(4, 1, &[9; 16]).unwrap();
//! p.start().unwrap();
//! let cycles = p.run_to_done(1000).unwrap();
//! assert!(p.status().quiescent && !p.status().error);
//! assert_eq!(p.cycle(), cycles);
//! assert_eq!(p.read_sram_word(0, 3).unwrap()[..16], [9; 16]);
//! ```
use crate::driver::replay::{Began, RecordOn, ReplayCache, ReplayCounters};
use crate::geom::MAX_LINKS;
use crate::links::LinkPort;
use crate::machine::{LoadError, Machine, RunError};
use crate::observe::Observer;
use ots_isa::Code;
use ots_profile::{Profile, SPEC_VERSION};
use std::collections::VecDeque;
use std::fmt;

#[cfg(feature = "replay")]
mod param;
#[cfg(feature = "replay")]
pub use param::{CheckMode, ParamSpec, ParamStatus, ParamWhy, ShapeStatus};

/// The value of the `ID` register: "OTSP".
pub const PLATFORM_ID: u32 = 0x4f54_5350;

/// The depth the error FIFO has when the profile says nothing else. The FIFO mirrors the chip's diagnostic log
/// (`DIAG_COUNT`, `DIAG[i]` of the CSR space), so its depth is the profile key `ras.diag_depth` (1..64: 16 in `ref` and
/// `mini`, fewer in some test profiles).
pub const ERR_FIFO_DEPTH: usize = 16;

/// The CSR addresses of the chip's diagnostic log (`core/csr.sail`).
pub const DIAG_COUNT: u64 = 0x400;
pub const DIAG_POP: u64 = 0x401;
pub const DIAG_STATUS: u64 = 0x402;
pub const DIAG_LOG: u64 = 0x410;

/// The lane bytes of one tile: what one `SRAM_DATA` access moves.
pub const TILE_BYTES: usize = 16;

/// One diagnostic: the superlane-0 cycle, the queue (0xffff is the control channel) and the code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub cycle: u64,
    pub queue: u16,
    pub code: Code,
}

/// The `SRAM_ADDR` register: one tile of one word of one slice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SramAddr {
    /// The slice, numbered as `topology.md` §6.
    pub slice: u8,
    /// `0 .. sram.words_per_slice - 1`.
    pub word: u32,
    /// `0 .. superlanes - 1`.
    pub superlane: u8,
}

/// The `STATUS` register.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Status {
    /// Started and neither done nor stopped.
    pub running: bool,
    /// The chip ran to quiescence since the last reset.
    pub quiescent: bool,
    /// The error FIFO is not empty or `ERR_FIRST` is valid.
    pub error: bool,
    /// An IQ window append was refused since the last reset.
    pub load_error: bool,
    /// A record was dropped by a full error FIFO since the last reset.
    pub fifo_overflow: bool,
}

impl Status {
    pub const RUNNING: u32 = 1;
    pub const QUIESCENT: u32 = 1 << 1;
    pub const ERROR: u32 = 1 << 2;
    pub const LOAD_ERROR: u32 = 1 << 3;
    pub const FIFO_OVERFLOW: u32 = 1 << 4;

    /// The register value: one bit per flag, in the order of the fields.
    pub fn bits(&self) -> u32 {
        [self.running, self.quiescent, self.error, self.load_error, self.fifo_overflow]
            .iter()
            .enumerate()
            .fold(0, |v, (i, f)| v | (u32::from(*f) << i))
    }
}

/// The `IRQ_ENABLE` register: the interrupt on done and on error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IrqEnable {
    pub quiescent: bool,
    pub error: bool,
}

/// Where the chip is between reset and done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Reset, program and data being loaded; the start bit is not written.
    Reset,
    Running,
    /// Quiescent after a run.
    Done,
    /// Frozen by the stop bit.
    Stopped,
}

/// Why a host access was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlatformError {
    /// The chip cannot be built from the profile (a geometry wider than the harness holds).
    Config(String),
    /// The access is not allowed in this phase (IQ windows and peers: before start only; the
    /// aperture: not while running; start: after reset only; advance: while running only).
    WrongPhase(Phase),
    /// A queue, slice or word the chip does not have, or an image beyond `icu.iq_depth`.
    Load(LoadError),
    /// A superlane not below `superlanes`.
    Superlane(u8),
    /// A link not below `c2c.links`.
    Link(u8),
    /// The chip is not done after the cycle limit (it is still running; the host may stop it).
    NotDone { cycle: u64 },
    /// The model broke an invariant.
    Fault(RunError),
}

impl fmt::Display for PlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlatformError::Config(m) => f.write_str(m),
            PlatformError::WrongPhase(p) => write!(f, "not allowed while the chip is in phase {p:?}"),
            PlatformError::Load(e) => e.fmt(f),
            PlatformError::Superlane(s) => write!(f, "superlane {s} out of range"),
            PlatformError::Link(l) => write!(f, "link {l} out of range"),
            PlatformError::NotDone { cycle } => write!(f, "not done at cycle {cycle}"),
            PlatformError::Fault(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for PlatformError {}

impl From<LoadError> for PlatformError {
    fn from(e: LoadError) -> Self {
        PlatformError::Load(e)
    }
}

/// The link side of a chip with no fabric: the peers of `C2C_PEER`, and vectors that go nowhere.
struct Peers([Option<u16>; MAX_LINKS]);

impl LinkPort for Peers {
    fn peer(&self, link: u8) -> Option<u16> {
        self.0.get(link as usize).copied().flatten()
    }
    fn send(&mut self, _cycle: u64, _link: u8, _v: &[u8; crate::geom::MAX_LANES]) {}
    fn recv_ready(&mut self, _cycle: u64, _link: u8) -> bool {
        false
    }
    fn recv(&mut self, _cycle: u64, _link: u8) -> [u8; crate::geom::MAX_LANES] {
        [0; crate::geom::MAX_LANES]
    }
}

/// Does the hardware log this report? The log filter of `microarchitecture.md` §16.3, a superset of the codes of class
/// `data` in the `Class` column of §8 (which `platform.md` §7.1 names for hardware conformance): the rules that read
/// SRAM words, stream values or the instruction text. IQ-002 (an IQ overflow) and IQ-004 (text left at completion) are in it. IQ-004 is an end-of-run record: the
/// completion check raises it after every run-time record and stamps it with the cycle of the queue's last activity,
/// so it enters the FIFO last and the FIFO is then not monotonic in cycle. OPR-001 is kept for every MEM queue and
/// every `*_SEL` queue: the observer carries only (cycle, queue, code), so the per-instruction exception of the table
/// (Gather and Scatter tile addresses, an `sxselect` select) cannot be told from a Read or Write segment error, and
/// the filter is a documented superset.
fn data_dependent(queue: u16, code: Code) -> bool {
    crate::driver::diag_is_data(if queue == 0xffff { 144 } else { queue as usize }, code)
}

/// The diagnostics structures and the observer that feeds them.
#[derive(Default, Debug)]
struct Diagnostics {
    /// The depth of the FIFO: `ras.diag_depth`, the depth of the chip's diagnostic log it mirrors.
    depth: usize,
    all: Vec<Record>,
    fifo: VecDeque<Record>,
    first: Option<Record>,
    overflow: bool,
}

impl Observer for Diagnostics {
    fn diag(&mut self, cycle: u64, queue: u16, code: Code) {
        let r = Record { cycle, queue, code };
        self.all.push(r);
        if !data_dependent(queue, code) {
            return;
        }
        self.first.get_or_insert(r);
        if self.fifo.len() == self.depth {
            self.overflow = true;
        } else {
            self.fifo.push_back(r);
        }
    }
}

/// One chip behind its platform registers.
pub struct Platform {
    profile: Profile,
    chip_id: u16,
    machine: Machine,
    phase: Phase,
    cycle: u64,
    last_tick: Option<u64>,
    diag: Diagnostics,
    peers: [Option<u16>; MAX_LINKS],
    iq_select: u16,
    sram_addr: SramAddr,
    load_error: bool,
    irq_enable: IrqEnable,
    replay_sessions: bool,
    replay: ReplayState,
    #[cfg(feature = "replay")]
    param: param::ParamState,
}

/// A host call that a replayed run must be able to repeat on the normal driver.
#[derive(Clone, Copy, Debug)]
enum Call {
    Advance(u64),
    Flip(u8, u32, u8, u8),
    Scrub,
    /// The host popped a record from the error FIFO.
    Pop,
}

/// What a run served by a stored schedule keeps so that a divergence (or a shadow check) can run it again on the normal
/// driver: the program and the SRAM at the start, and the calls the host made since.
struct ReplayRun {
    key: [u8; 32],
    images: Vec<Vec<u32>>,
    /// The non-zero SRAM words at the start: (slice, address) and, in the same order, `vlanes` bytes each.
    words: Vec<(u8, u32)>,
    data: Vec<u8>,
    log: Vec<Call>,
    shadow: bool,
    /// A run served by the schedule of a bucket program shape (`start_plan`): the shape. It keeps no SRAM snapshot: the
    /// SRAM's journal (`Sram::journal_begin`) puts the SRAM back to the state at the start.
    param: Option<String>,
}

/// The replay sessions of a platform (`Platform::set_replay_sessions`).
struct ReplayState {
    cache: std::sync::Arc<ReplayCache>,
    run: Option<ReplayRun>,
    /// A service call (a fault flip or a scrub) was made since the reset and before the start: it changed the SRAM and
    /// the check bits, state a re-run from the SRAM at the start cannot rebuild, so the run is not served from a schedule
    /// and not recorded.
    tainted: bool,
    /// `TSP_ISS_REPLAY_SHADOW=N`: every Nth served run is repeated on the normal driver and compared (0: never).
    shadow_every: u64,
    served: u64,
    /// Whether the run takes the SRAM snapshot that a divergence needs (`TSP_ISS_REPLAY_SNAPSHOT=0` turns it off).
    snapshot: bool,
}

impl ReplayState {
    fn new() -> ReplayState {
        let num = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        ReplayState {
            cache: ReplayCache::global().clone(),
            run: None,
            tainted: false,
            shadow_every: num("TSP_ISS_REPLAY_SHADOW").unwrap_or(0),
            served: 0,
            snapshot: std::env::var_os("TSP_ISS_REPLAY_SNAPSHOT").is_none_or(|v| v != "0"),
        }
    }
}

impl Platform {
    /// A chip of `profile` with chip id `chip_id`, in reset.
    pub fn new(profile: &Profile, chip_id: u16) -> Result<Platform, PlatformError> {
        let peers = [None; MAX_LINKS];
        let machine = Platform::build(profile, chip_id, peers, false)?;
        Ok(Platform {
            profile: profile.clone(),
            chip_id,
            machine,
            phase: Phase::Reset,
            cycle: 0,
            last_tick: None,
            diag: Diagnostics {
                depth: profile.uint("ras.diag_depth").map_or(ERR_FIFO_DEPTH, |d| d as usize),
                ..Diagnostics::default()
            },
            peers,
            iq_select: 0,
            sram_addr: SramAddr::default(),
            load_error: false,
            irq_enable: IrqEnable::default(),
            replay_sessions: std::env::var_os("TSP_ISS_REPLAY_SESSION").is_some_and(|v| !v.is_empty() && v != "0"),
            replay: ReplayState::new(),
            #[cfg(feature = "replay")]
            param: param::ParamState::new(),
        })
    }

    /// **Schedule playback, an ISS speed device with no architectural meaning (`platform.md` §1.2):** serve the second and later runs of one program in this process from the schedule that
    /// an earlier run recorded (`driver::replay`, the compiled replay): the first run of a program records, every
    /// later run of the same program (same profile and IQ images, any data) replays. Same results, diagnostics and
    /// cycle counts as without it. Off unless `TSP_ISS_REPLAY_SESSION=1` is set when the platform is created.
    ///
    /// The cache is bounded (`TSP_ISS_REPLAY_BUDGET`, `TSP_ISS_REPLAY_ENTRIES`, least recently used out), a program is
    /// recorded on its **second** run (the first costs nothing), and a replay whose instruction text differs from the
    /// recording is **not an error**: the run is repeated on the normal driver from the state at its start and gives
    /// the result it would have given without sessions ([`Platform::replay_counters`] counts it). `TSP_ISS_REPLAY_SHADOW=N`
    /// repeats every Nth served run on the normal driver and compares the results.
    pub fn set_replay_sessions(&mut self, on: bool) {
        self.replay_sessions = on;
        if !on {
            self.replay.run = None;
        }
    }

    /// Use `cache` instead of the process's schedule cache (tests, or a host that wants its own bound).
    pub fn set_replay_cache(&mut self, cache: std::sync::Arc<ReplayCache>) {
        self.replay.cache = cache;
    }

    /// Repeat every `n`th served run on the normal driver and compare (0: never), as `TSP_ISS_REPLAY_SHADOW`.
    pub fn set_replay_shadow(&mut self, n: u64) {
        self.replay.shadow_every = n;
    }

    /// Take (or not) the SRAM snapshot of a served run that the divergence fallback needs. Without it a divergence
    /// ends the run with the model fault, as before the cache was bounded. On by default.
    pub fn set_replay_snapshot(&mut self, on: bool) {
        self.replay.snapshot = on;
    }

    /// The counters of the schedule cache this platform uses: hits, misses, divergences, evictions, retained bytes.
    pub fn replay_counters(&self) -> ReplayCounters {
        self.replay.cache.counters()
    }

    /// What the replay did for the run of this platform's chip (a debugger and test hook).
    pub fn replay_mode(&self) -> crate::driver::replay::Mode {
        self.machine.replay_mode()
    }

    /// A chip in reset. A serving platform does not count activity unless asked (`set_activity_counting`): nobody reads the
    /// counters, and `counting` carries the setting across a reset or a restart.
    fn build(
        profile: &Profile,
        chip_id: u16,
        peers: [Option<u16>; MAX_LINKS],
        counting: bool,
    ) -> Result<Machine, PlatformError> {
        let mut machine = Machine::new(profile, chip_id, Box::new(Peers(peers))).map_err(PlatformError::Config)?;
        machine.activity.set_enabled(counting);
        Ok(machine)
    }

    /// Count the chip's activity (`slices::activity`) from now on, or stop. Off when the platform is built.
    pub fn set_activity_counting(&mut self, on: bool) {
        self.machine.activity.set_enabled(on);
    }

    /// The activity counters of the chip: all zero unless [`Platform::set_activity_counting`] turned them on.
    pub fn activity(&self) -> &crate::slices::activity::Activity {
        &self.machine.activity
    }

    // ---- identification ----

    /// `ID`: [`PLATFORM_ID`].
    pub fn id(&self) -> u32 {
        PLATFORM_ID
    }

    /// `SPEC_VERSION`: major in the upper 16 bits, minor in the lower (0.1 is `0x0000_0001`).
    pub fn spec_version(&self) -> u32 {
        let (major, minor) = SPEC_VERSION.split_once('.').expect("the specification version is major.minor");
        let part = |s: &str| s.parse::<u32>().expect("the specification version is numeric");
        (part(major) << 16) | part(minor)
    }

    /// `PROFILE_DIGEST`: the first 12 bytes of the SHA-256 of the configuration file (the digest an
    /// object carries), read as three 32-bit registers.
    pub fn profile_digest(&self) -> [u8; 12] {
        self.profile.digest12()
    }

    /// `ICU_FETCH_LATENCY`: `icu.fetch_latency`.
    pub fn icu_fetch_latency(&self) -> u64 {
        self.machine.geom.fetch_latency
    }

    /// `ICU_FETCH_BYTES`: `icu.fetch_bytes_per_cycle`.
    pub fn icu_fetch_bytes(&self) -> u64 {
        self.profile.uint("icu.fetch_bytes_per_cycle").unwrap_or(0)
    }

    /// `ICU_IQ_DEPTH`: `icu.iq_depth`, in bytes.
    pub fn icu_iq_depth(&self) -> u64 {
        self.machine.geom.iq_depth
    }

    /// `C2C_PEER[link]`: the chip id at the far end of `link`, `None` when no peer is set.
    pub fn c2c_peer(&self, link: u8) -> Result<Option<u16>, PlatformError> {
        self.check_link(link)?;
        Ok(self.peers[link as usize])
    }

    /// Write `C2C_PEER[link]` (before start only).
    pub fn set_c2c_peer(&mut self, link: u8, peer: Option<u16>) -> Result<(), PlatformError> {
        self.check_link(link)?;
        self.need_phase(Phase::Reset)?;
        self.peers[link as usize] = peer;
        self.machine.links = Box::new(Peers(self.peers));
        Ok(())
    }

    fn check_link(&self, link: u8) -> Result<(), PlatformError> {
        if u64::from(link) < self.machine.geom.links.min(MAX_LINKS as u64) {
            Ok(())
        } else {
            Err(PlatformError::Link(link))
        }
    }

    fn need_phase(&self, want: Phase) -> Result<(), PlatformError> {
        if self.phase == want {
            Ok(())
        } else {
            Err(PlatformError::WrongPhase(self.phase))
        }
    }

    // ---- control ----

    /// `CTRL.reset`: the chip as `model_reset` leaves it, with empty IQs, a cleared SRAM, cycle 0 and
    /// no status or diagnostics.
    pub fn reset(&mut self) -> Result<(), PlatformError> {
        self.machine = Platform::build(&self.profile, self.chip_id, self.peers, self.machine.activity.enabled())?;
        self.clear_state();
        Ok(())
    }

    /// The optional **restart** of `platform.md` §1.1 (`CTRL.restart`): the state after [`Platform::reset`] with the
    /// SRAM contents kept: empty IQs, MXM and SXM state, SRF, tile masks, cycle 0, no status, diagnostics, error FIFO or
    /// `ERR_FIRST`, `STATUS.load_error` clear. The C2C peers and the interrupt enables survive as they survive a reset.
    ///
    /// It conforms to the equivalence of §1.1: a restart followed by loading a program is the same run as a cold
    /// reset, every word the SRAM held written back through the aperture (the write-back restart of §1.2), then the
    /// program. The host loop that serves many runs from one resident set of weights and caches uses the restart to
    /// skip the write-back. [`Platform::sram_words`] and [`Platform::sram_digest`] are the observers of the equivalence.
    pub fn restart(&mut self) -> Result<(), PlatformError> {
        #[cfg(feature = "replay")]
        self.machine.sram.journal_end(); // the journal of a served run belongs to that run, not to the SRAM that survives it
        let mut fresh = Platform::build(&self.profile, self.chip_id, self.peers, self.machine.activity.enabled())?;
        std::mem::swap(&mut fresh.sram, &mut self.machine.sram);
        self.machine = fresh;
        self.clear_state();
        Ok(())
    }

    fn clear_state(&mut self) {
        self.replay.run = None;
        #[cfg(feature = "replay")]
        {
            self.param.run = None;
            self.param_finish_replacement();
        }
        self.replay.tainted = false;
        self.phase = Phase::Reset;
        self.cycle = 0;
        self.last_tick = None;
        self.diag = Diagnostics { depth: self.diag.depth, ..Diagnostics::default() };
        self.iq_select = 0;
        self.sram_addr = SramAddr::default();
        self.load_error = false;
    }

    /// `CTRL.start`: cycle 0 is now. Only a chip in reset starts.
    pub fn start(&mut self) -> Result<(), PlatformError> {
        self.need_phase(Phase::Reset)?;
        self.phase = Phase::Running;
        if self.replay_sessions && !self.replay.tainted {
            let cache = self.replay.cache.clone();
            if self.machine.replay_session_begin_in(&cache, RecordOn::Second) == Began::Replaying {
                self.replay.served += 1;
                let shadow =
                    self.replay.shadow_every > 0 && self.replay.served.is_multiple_of(self.replay.shadow_every);
                self.replay.run = (self.replay.snapshot || shadow).then(|| self.snapshot_run(shadow));
            }
        }
        Ok(())
    }

    /// The state a served run starts from: its program, its SRAM, and the key of its schedule.
    fn snapshot_run(&self, shadow: bool) -> ReplayRun {
        let vl = self.machine.geom.vlanes as usize;
        let mut words = Vec::new();
        let mut data = Vec::new();
        for (s, a, w) in self.machine.sram.nonzero_words() {
            words.push((s as u8, a as u32));
            data.extend_from_slice(&w[..vl]);
        }
        ReplayRun {
            key: self.machine.replay_key().expect("a served run has a schedule"),
            images: self.machine.images.clone(),
            words,
            data,
            log: Vec::new(),
            shadow,
            param: None,
        }
    }

    /// `CTRL.stop`: freeze a running chip in the current cycle. A chip that is not running keeps its
    /// state.
    pub fn stop(&mut self) {
        if self.phase == Phase::Running {
            self.phase = Phase::Stopped;
        }
    }

    /// Where the chip is between reset and done.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Let the running chip execute up to `cycles` more cycles; it is done when nothing is left to
    /// do before that.
    pub fn advance(&mut self, cycles: u64) -> Result<(), PlatformError> {
        self.need_phase(Phase::Running)?;
        if let Some(r) = &mut self.replay.run {
            r.log.push(Call::Advance(cycles));
        }
        #[cfg(feature = "replay")]
        let began = std::time::Instant::now();
        let mut res = self.advance_machine(cycles);
        if self.replay.run.is_some() {
            let diverged = match &res {
                Err(PlatformError::Fault(_)) => self.machine.rp.diverged,
                Ok(()) => self.phase == Phase::Done && self.machine.rp_finish().is_err(),
                Err(_) => false,
            };
            if diverged {
                res = self.replay_fallback();
            } else if res.is_ok() && self.phase == Phase::Done && self.param_served() {
                #[cfg(feature = "replay")]
                if let Err(g) = self.param_text_gate() {
                    res = self.param_fallback(ParamWhy::Text(g.to_string()));
                }
            } else if res.is_ok() && self.phase == Phase::Done && self.replay.run.as_ref().is_some_and(|r| r.shadow) {
                res = self.replay_shadow();
            }
        }
        #[cfg(feature = "replay")]
        self.param_after_advance(began.elapsed(), res.is_ok());
        if res.is_ok() && self.phase == Phase::Done {
            #[cfg(feature = "replay")]
            self.param_finish_replacement();
            self.replay.run = None;
            if self.replay_sessions {
                self.machine.replay_session_end();
            }
        }
        res
    }

    fn advance_machine(&mut self, cycles: u64) -> Result<(), PlatformError> {
        let wend = self.cycle.saturating_add(cycles);
        let sum = self.machine.run_until(wend, u64::MAX, &mut self.diag).map_err(PlatformError::Fault)?;
        self.last_tick = sum.last_cycle.or(self.last_tick);
        if self.machine.next_time().is_some() {
            self.cycle = wend;
            return Ok(());
        }
        if !self.machine.quiescent() {
            return Err(PlatformError::Fault(RunError::Internal("model not quiescent with no event left".into())));
        }
        self.machine.finish(&mut self.diag);
        self.cycle = self.last_tick.map_or(0, |t| t + 1);
        self.phase = Phase::Done;
        Ok(())
    }

    /// The run so far, again, on the normal driver: a new chip with the program and the SRAM of the start of `run`, the
    /// host's calls repeated in order. What it returns is what the run gives without sessions.
    fn normal_twin(&self, run: &ReplayRun) -> Result<(Platform, Result<(), PlatformError>), PlatformError> {
        let mut t = self.twin_shell(run)?;
        let vl = self.machine.geom.vlanes as usize;
        for (i, (slice, addr)) in run.words.iter().enumerate() {
            t.machine.preload(*slice, *addr, &run.data[i * vl..(i + 1) * vl])?;
        }
        let result = Platform::twin_calls(&mut t, run);
        Ok((t, result))
    }

    /// A new chip on the normal driver with the program of `run` loaded and no SRAM yet.
    fn twin_shell(&self, run: &ReplayRun) -> Result<Platform, PlatformError> {
        let mut t = Platform::new(&self.profile, self.chip_id)?;
        t.replay_sessions = false;
        t.peers = self.peers;
        t.machine.links = Box::new(Peers(self.peers));
        for (slot, units) in run.images.iter().enumerate() {
            if !units.is_empty() {
                t.machine.load_image(crate::harness::stats::queue_of(slot), units)?;
            }
        }
        Ok(t)
    }

    /// Start `t` and repeat the host's calls of `run` on it, in order; the result of the last.
    fn twin_calls(t: &mut Platform, run: &ReplayRun) -> Result<(), PlatformError> {
        t.phase = Phase::Running;
        let mut result = Ok(());
        for call in &run.log {
            result = match *call {
                Call::Advance(n) => t.advance_machine(n),
                Call::Flip(slice, addr, lane, bit) => {
                    // the host saw this result when it called; the call that failed then fails the same way now
                    let _ = t.machine.service_flip_sram(slice, addr, lane, bit);
                    Ok(())
                }
                Call::Scrub => {
                    t.machine.service_scrub(t.cycle);
                    Ok(())
                }
                Call::Pop => {
                    t.diag.fifo.pop_front();
                    t.machine.csr_write(DIAG_POP, 1, t.cycle, &mut crate::observe::NoObserver);
                    Ok(())
                }
            };
            if result.is_err() {
                break;
            }
        }
        result
    }

    /// Take the state of `twin`, a run on the normal driver of the same calls (the host's pops of the error FIFO
    /// included), as this chip's.
    fn adopt(&mut self, twin: Platform) {
        self.machine = twin.machine;
        self.diag = twin.diag;
        self.cycle = twin.cycle;
        self.last_tick = twin.last_tick;
        self.phase = twin.phase;
    }

    /// Whether the run in flight is served by the schedule of a bucket program shape (`start_plan`).
    fn param_served(&self) -> bool {
        self.replay.run.as_ref().is_some_and(|r| r.param.is_some())
    }

    /// A replayed run whose text differs from the recording: count it, drop the schedule, and give the result of the
    /// normal driver (no error: the run is the one the host asked for).
    fn replay_fallback(&mut self) -> Result<(), PlatformError> {
        #[cfg(feature = "replay")]
        if self.param_served() {
            return self.param_fallback(ParamWhy::Diverged);
        }
        let run = self.replay.run.take().expect("a served run");
        self.replay.cache.diverged(&run.key);
        let (twin, result) = self.normal_twin(&run)?;
        self.adopt(twin);
        result
    }

    /// The shadow check of a served run that is done: repeat it on the normal driver and compare the result (the
    /// cycle count, the diagnostics, the SRAM). A difference is counted, and the normal driver's result stands.
    fn replay_shadow(&mut self) -> Result<(), PlatformError> {
        let run = self.replay.run.take().expect("a served run");
        let (twin, result) = self.normal_twin(&run)?;
        let same = result.is_ok()
            && twin.cycle == self.cycle
            && twin.last_tick == self.last_tick
            && twin.phase == self.phase
            && twin.diag.all == self.diag.all
            && twin.sram_digest() == self.sram_digest();
        self.replay.cache.shadow(same);
        if !same {
            self.replay.cache.diverged(&run.key);
            self.adopt(twin);
            return result;
        }
        Ok(())
    }

    /// Run a started chip to done within `limit` cycles of the start bit and return the cycle
    /// counter. A chip still running at the limit stays running ([`PlatformError::NotDone`]).
    pub fn run_to_done(&mut self, limit: u64) -> Result<u64, PlatformError> {
        self.advance(limit.saturating_sub(self.cycle))?;
        match self.phase {
            Phase::Done => Ok(self.cycle),
            _ => Err(PlatformError::NotDone { cycle: self.cycle }),
        }
    }

    // ---- IQ load window ----

    /// `IQ_SELECT`: the queue the load window writes (0..143 or 0xffff). Reset value 0.
    pub fn select_iq(&mut self, queue: u16) -> Result<(), PlatformError> {
        self.machine.queue_state(queue).ok_or(LoadError::Queue(queue))?;
        self.iq_select = queue;
        Ok(())
    }

    /// `IQ_SELECT` as last written.
    pub fn selected_iq(&self) -> u16 {
        self.iq_select
    }

    /// `IQ_WINDOW`: append `units` to the selected queue's IQ image, in order. Padding units are
    /// dropped. An append that would take the image beyond `icu.iq_depth` is refused whole and sets
    /// `STATUS.load_error`.
    pub fn write_iq_window(&mut self, units: &[u32]) -> Result<(), PlatformError> {
        self.need_phase(Phase::Reset)?;
        let held = self.machine.queue_state(self.iq_select).map_or(0, |s| s.iq_units);
        let new = units.iter().filter(|u| **u != ots_isa::PADDING).count();
        let limit = self.machine.geom.iq_units;
        if held + new > limit {
            self.load_error = true;
            return Err(LoadError::ImageTooLong { queue: self.iq_select, units: held + new, limit }.into());
        }
        Ok(self.machine.load_image(self.iq_select, units)?)
    }

    // ---- SRAM aperture ----

    /// Write `SRAM_ADDR`.
    pub fn set_sram_addr(&mut self, addr: SramAddr) -> Result<(), PlatformError> {
        self.machine.sram_word(addr.slice, addr.word)?;
        if u64::from(addr.superlane) >= self.machine.geom.superlanes {
            return Err(PlatformError::Superlane(addr.superlane));
        }
        self.sram_addr = addr;
        Ok(())
    }

    /// `SRAM_ADDR` as last written (reset value: slice 0, word 0, superlane 0).
    pub fn sram_addr(&self) -> SramAddr {
        self.sram_addr
    }

    /// `SRAM_DATA` read: the 16 lane bytes of the tile at `SRAM_ADDR`.
    pub fn read_sram_data(&self) -> Result<[u8; TILE_BYTES], PlatformError> {
        if self.phase == Phase::Running {
            return Err(PlatformError::WrongPhase(self.phase));
        }
        let a = self.sram_addr;
        let word = self.machine.sram_word(a.slice, a.word)?;
        let at = usize::from(a.superlane) * TILE_BYTES;
        Ok(word[at..at + TILE_BYTES].try_into().expect("a tile is 16 bytes"))
    }

    /// `SRAM_DATA` write: set the 16 lane bytes of the tile at `SRAM_ADDR`; the other tiles of the
    /// word keep their bytes.
    pub fn write_sram_data(&mut self, tile: &[u8; TILE_BYTES]) -> Result<(), PlatformError> {
        if self.phase == Phase::Running {
            return Err(PlatformError::WrongPhase(self.phase));
        }
        let a = self.sram_addr;
        let mut word = self.machine.sram_word(a.slice, a.word)?;
        let at = usize::from(a.superlane) * TILE_BYTES;
        word[at..at + TILE_BYTES].copy_from_slice(tile);
        let lanes = self.machine.geom.vlanes as usize;
        Ok(self.machine.preload(a.slice, a.word, &word[..lanes])?)
    }

    /// A burst: write `bytes` (lane 0 first, at most `vlanes`) to a word, one tile access per 16
    /// bytes; a last partial tile is padded with zeros and the tiles beyond `bytes` are untouched.
    pub fn write_sram_word(&mut self, slice: u8, word: u32, bytes: &[u8]) -> Result<(), PlatformError> {
        if bytes.len() as u64 > self.machine.geom.vlanes {
            let lanes = self.machine.geom.vlanes as usize;
            return Err(LoadError::TooManyBytes { bytes: bytes.len(), lanes }.into());
        }
        for (superlane, chunk) in bytes.chunks(TILE_BYTES).enumerate() {
            let mut tile = [0; TILE_BYTES];
            tile[..chunk.len()].copy_from_slice(chunk);
            self.set_sram_addr(SramAddr { slice, word, superlane: superlane as u8 })?;
            self.write_sram_data(&tile)?;
        }
        Ok(())
    }

    /// A burst: the `vlanes` lane bytes of a word, one tile access per superlane.
    pub fn read_sram_word(&mut self, slice: u8, word: u32) -> Result<Vec<u8>, PlatformError> {
        let mut out = Vec::with_capacity(self.machine.geom.vlanes as usize);
        for superlane in 0..self.machine.geom.superlanes {
            self.set_sram_addr(SramAddr { slice, word, superlane: superlane as u8 })?;
            out.extend_from_slice(&self.read_sram_data()?);
        }
        Ok(out)
    }

    /// **ISS extension (a debugging observer, `platform.md` §1.3):** the SRAM words that are not all zero, as (slice,
    /// word address, `vlanes` bytes, lane 0 first), in slice then address order. A host-side view of the whole SRAM
    /// for the write-back that [`Platform::restart`]'s equivalence names; not a register and not usable while the chip runs on hardware.
    pub fn sram_words(&self) -> Vec<(u8, u32, Vec<u8>)> {
        let vl = self.machine.geom.vlanes as usize;
        self.machine.sram.nonzero_words().map(|(s, a, w)| (s as u8, a as u32, w[..vl].to_vec())).collect()
    }

    /// **ISS extension:** the SHA-256 of the SRAM contents: for each non-zero word in [`Platform::sram_words`]'s
    /// order, the slice (1 byte), the address (4 bytes, little-endian) and the `vlanes` bytes. An all-zero SRAM
    /// hashes the empty string; the value does not depend on which pages are allocated.
    pub fn sram_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for (s, a, w) in self.machine.sram.nonzero_words() {
            h.update([s as u8]);
            h.update((a as u32).to_le_bytes());
            h.update(w);
        }
        h.finalize().into()
    }

    /// **ISS extension:** the SHA-256 of all chip state except the SRAM contents ([`Platform::sram_digest`] covers
    /// those): the machine ([`Machine::state_digest`]), the phase, the cycle count, the last tick, every diagnostic,
    /// `IQ_SELECT`, `SRAM_ADDR` and the load-error flag. The configuration (profile, chip id, link peers, interrupt
    /// enables) is not included; the fields of `Platform` are destructured without `..`, so a new one does not compile
    /// until it is hashed or named as configuration here. A chip after [`Platform::restart`] has the digest of a
    /// new chip.
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let Platform {
            profile: _,
            chip_id: _,
            machine,
            phase,
            cycle,
            last_tick,
            diag,
            peers: _,
            iq_select,
            sram_addr,
            load_error,
            irq_enable: _,
            replay_sessions: _,
            replay: _,
            #[cfg(feature = "replay")]
                param: _,
        } = self;
        let mut h = Sha256::new();
        h.update(machine.state_digest());
        h.update(
            format!(
                "phase={phase:?}|cycle={cycle}|last_tick={last_tick:?}|diag={diag:?}|iq_select={iq_select}\
                 |sram_addr={sram_addr:?}|load_error={load_error}"
            )
            .as_bytes(),
        );
        h.finalize().into()
    }

    // ---- status ----

    /// `STATUS`.
    pub fn status(&self) -> Status {
        Status {
            running: self.phase == Phase::Running,
            quiescent: self.phase == Phase::Done,
            error: !self.diag.fifo.is_empty() || self.diag.first.is_some(),
            load_error: self.load_error,
            fifo_overflow: self.diag.overflow,
        }
    }

    /// `CYCLE`: the superlane-0 cycles executed since the start bit.
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// Write `IRQ_ENABLE`.
    pub fn set_irq_enable(&mut self, enable: IrqEnable) {
        self.irq_enable = enable;
    }

    /// `IRQ_ENABLE` as last written.
    pub fn irq_enable(&self) -> IrqEnable {
        self.irq_enable
    }

    /// The interrupt line: an enabled condition (done, error) holds.
    pub fn irq_pending(&self) -> bool {
        let s = self.status();
        (self.irq_enable.quiescent && s.quiescent) || (self.irq_enable.error && s.error)
    }

    // ---- diagnostics ----

    /// `ERR_FIRST`: the first record raised since reset, `None` while its valid bit is clear.
    pub fn err_first(&self) -> Option<Record> {
        self.diag.first
    }

    /// The chip's diagnostic log in the CSR space (`DIAG_COUNT`, `DIAG[i]`; `microarchitecture.md` §16.3), which the FIFO mirrors: the words of
    /// the entries, oldest first, and the overflow flag (`DIAG_STATUS` bit 0).
    pub fn chip_diag_log(&self) -> (Vec<u64>, bool) {
        let n = self.machine.csr_read(DIAG_COUNT);
        ((0..n).map(|i| self.machine.csr_read(DIAG_LOG + i)).collect(), self.machine.csr_read(DIAG_STATUS) & 1 == 1)
    }

    /// `ERR_COUNT`: the records the FIFO holds.
    pub fn err_count(&self) -> usize {
        self.diag.fifo.len()
    }

    /// `ERR_FIFO`: pop the oldest record (and the oldest entry of the chip's diagnostic log it mirrors: `DIAG_POP`).
    pub fn err_pop(&mut self) -> Option<Record> {
        let r = self.diag.fifo.pop_front();
        if r.is_some() {
            self.machine.csr_write(DIAG_POP, 1, self.cycle, &mut crate::observe::NoObserver);
            if let Some(run) = &mut self.replay.run {
                run.log.push(Call::Pop);
            }
        }
        r
    }

    /// Every diagnostic the model reported since reset, in the model's reporting order: the
    /// simulator's list, which the golden vectors compare. It is not a register.
    pub fn service_flip_sram(&mut self, slice: u8, addr: u32, lane: u8, bit: u8) -> Result<(), String> {
        #[cfg(feature = "replay")]
        self.param_on_service().map_err(|e| e.to_string())?;
        if let Some(run) = &mut self.replay.run {
            run.log.push(Call::Flip(slice, addr, lane, bit));
        } else if self.phase == Phase::Reset {
            self.replay.tainted = true;
        }
        self.machine.service_flip_sram(slice, addr, lane, bit)
    }

    pub fn service_scrub(&mut self) {
        #[cfg(feature = "replay")]
        let _ = self.param_on_service(); // a failure here is the normal driver's own result, which the next call returns
        if let Some(run) = &mut self.replay.run {
            run.log.push(Call::Scrub);
        } else if self.phase == Phase::Reset {
            self.replay.tainted = true;
        }
        self.machine.service_scrub(self.cycle);
    }

    pub fn service_ras(&self) -> &crate::host::ServiceRas {
        self.machine.service_ras()
    }

    pub fn diagnostics(&self) -> &[Record] {
        &self.diag.all
    }
}
