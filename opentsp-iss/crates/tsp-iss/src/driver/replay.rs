//! Compiled replay of a run's schedule. **It serves the second and later runs of one
//! program in one process; it does nothing for the first run** (which records, at a small cost).
//!
//! The machine is deterministic and statically scheduled: which ticks happen, which instructions dispatch and execute in
//! them, and what the driver reports is a function of the program text, the profile and the machine's initial state,
//! never of the data. The first run records, per tick, the operations the tick body produced (the elements that
//! executed with their resolved operands, the dispatches, the driver's diagnostics and lints, the wait reports); a later
//! run of the same program replays them: per tick the commits due, the recorded operations in order, the end-of-cycle
//! flush. No IQ, parse, hold check, calendar, event search or AGT and Repeat bookkeeping runs. The semantics, the
//! commit ring and the observer calls are the code of a normal run, so the trace is the same bytes.
//!
//! A recording is used only when it provably fits:
//! * the key (`Machine::schedule_key`) is the digest of the profile and of the initial IQ images of a machine that has
//!   not ticked, so a different program or profile refuses the replay and the run takes the normal driver;
//! * the only way data reaches the schedule is the instruction text that `Ifetch` and `IfetchM` read from the SRAM
//!   (a data-dependent write over text), so the exact bytes and identity of every fetched payload are kept per tick and compared at the end
//!   of every replayed tick: a run whose text differs stops with a divergence error, at least `icu.fetch_latency`
//!   cycles before the text can act;
//! * the one other way data reaches the tick set is a diagnostic that depends on data (for example an SXM map): it
//!   drops a write, so the commit of that write is a tick of one run and not of the other. A replay therefore takes
//!   `min(next recorded tick, next pending commit)` as its next tick; a tick that only commits is not in the schedule
//!   and is accepted as such, any other tick that is not in the schedule is a divergence. (A recorded tick whose
//!   commit the other data drops still runs, empty: the trace is the same, `--stats` ticks may count it.);
//! * a run that touched a chip-to-chip link is not recorded (the link state is not part of the key), a run that ticks
//!   every cycle or tracks waits otherwise than the recording did takes the normal driver;
//! * a tick that is not in the schedule, a schedule left unfinished and a machine that is not quiescent at the end
//!   are errors, never output;
//! * `TSP_ISS_REPLAY_CHECK=1` runs the normal driver and compares its operations with the recording per tick (cycle,
//!   operations, executed elements, the fetched-text hash): the strict check, for tests and audits.
use super::execute::ExecElement;
use crate::machine::RunError;
use crate::observe::{Hold, Observer, Phase, Row};
use crate::Machine;
use ots_isa::{Code as Diag, EntryId};
use sha2::{Digest, Sha256};
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A quick digest and exact current-tick fetched bytes (not part of the machine state digest).
/// The digest is only a rejection filter; byte equality decides replay safety.
#[derive(Clone, Default)]
pub struct TextHash {
    pub on: bool,
    pub h: u64,
    bytes: Vec<u8>,
}

impl fmt::Debug for TextHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TextHash")
    }
}

impl TextHash {
    /// One fetched payload of queue `q` executed at cycle `t`.
    #[inline(always)]
    pub fn note(&mut self, q: usize, t: u64, a: &[u8], b: &[u8]) {
        if self.on {
            self.note_on(q, t, a, b);
        }
    }

    #[cold]
    #[inline(never)]
    fn note_on(&mut self, q: usize, t: u64, a: &[u8], b: &[u8]) {
        let mut h = (self.h ^ (q as u64) ^ (t << 8)).wrapping_mul(0x0000_0100_0000_01b3);
        for v in [a, b] {
            let mut chunks = v.chunks_exact(8);
            for c in &mut chunks {
                h = (h ^ u64::from_le_bytes(c.try_into().expect("8 bytes"))).wrapping_mul(0x0000_0100_0000_01b3);
            }
            for &x in chunks.remainder() {
                h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        self.h = h;
        self.bytes.extend_from_slice(&(q as u64).to_le_bytes());
        self.bytes.extend_from_slice(&t.to_le_bytes());
        self.bytes.extend_from_slice(&(a.len() as u64).to_le_bytes());
        self.bytes.extend_from_slice(&(b.len() as u64).to_le_bytes());
        self.bytes.extend_from_slice(a);
        self.bytes.extend_from_slice(b);
    }

    fn tick(&self) -> u32 {
        (self.h ^ (self.h >> 32)) as u32
    }
}

/// What a tick does besides the commits due and the end-of-cycle flush, in order (the compiled replay, level 2):
/// the elements that execute, and what the driver reports. Every other thing the tick body does (the hold checks, the
/// IQ, the calendar, the AGT and Repeat bookkeeping) only decides which of these happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// The next element of `Schedule::execs` executes (its diagnostic, a function of the data, is reported by the execution).
    Exec,
    /// The head parcel of the queue dispatched (`Observer::dispatch`).
    Dispatch(u16),
    /// A diagnostic of the driver (a dispatch check).
    Diag(u16, Diag),
    /// A lint of the driver: 0 IQ underflow, 1 an evicted spare-bit report.
    Lint(u16, u8),
    /// A wait report (`--stats`): the queue, why, since when.
    Wait(u16, Hold, u64),
}

const LINTS: [&str; 2] = [super::LINT_IQ_UNDERFLOW, super::LINT_RAS_EVICT];

/// The observer of a recording run: forwards everything and notes what the driver reports (the compiled replay's
/// `Op`s). It asks for the phase markers (`profiling`) to know whether a report comes from the driver (control,
/// calendar, dispatch) or from a commit, a flush or an execution, which a replay does itself.
struct RecObs<'a, O: Observer + ?Sized> {
    inner: &'a mut O,
    ops: &'a mut Vec<Op>,
    execs: &'a mut Vec<ExecElement>,
    inner_prof: bool,
    log: bool,
    in_exec: bool,
}

impl<O: Observer + ?Sized> RecObs<'_, O> {
    #[inline]
    fn driver(&self) -> bool {
        self.log && !self.in_exec
    }
}

impl<O: Observer + ?Sized> Observer for RecObs<'_, O> {
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        self.inner.srf(c, row, stream, gap, v)
    }
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.inner.sram(c, slice, addr, v)
    }
    fn diag(&mut self, c: u64, queue: u16, code: Diag) {
        if self.driver() {
            self.ops.push(Op::Diag(queue, code));
        }
        self.inner.diag(c, queue, code)
    }
    fn lint(&mut self, c: u64, queue: u16, code: &'static str) {
        if self.driver() {
            let k = LINTS.iter().position(|l| *l == code).expect("a driver lint") as u8;
            self.ops.push(Op::Lint(queue, k));
        }
        self.inner.lint(c, queue, code)
    }
    fn dispatch(&mut self, c: u64, queue: u16) {
        if self.driver() {
            self.ops.push(Op::Dispatch(queue));
        }
        self.inner.dispatch(c, queue)
    }
    fn wait(&mut self, c: u64, queue: u16, why: Hold, since: u64) {
        if self.driver() {
            self.ops.push(Op::Wait(queue, why, since));
        }
        self.inner.wait(c, queue, why, since)
    }
    fn fetch(&mut self, c: u64, queue: u16, bytes: u32) {
        self.inner.fetch(c, queue, bytes)
    }

    fn host_dma(&mut self, c: u64, what: crate::observe::HostDma) {
        self.inner.host_dma(c, what)
    }

    fn host_irq(&mut self, c: u64, vector: u8) {
        self.inner.host_irq(c, vector)
    }

    fn host_csr(&mut self, c: u64, addr: u64, value: u64) {
        self.inner.host_csr(c, addr, value)
    }

    fn host_csr_remote(&mut self, c: u64, chip: u16, addr: u64, value: u64) {
        self.inner.host_csr_remote(c, chip, addr, value)
    }

    fn ecc(&mut self, e: &crate::ecc::EccEvent) {
        self.inner.ecc(e);
    }
    fn mxm_corr(&mut self, c: u64, queue: u16, plane: u8, lane: u16) {
        self.inner.mxm_corr(c, queue, plane, lane);
    }
    fn scrub(&mut self, c: u64, slice: u8, addr: u32, sl: u8, unc: bool) {
        self.inner.scrub(c, slice, addr, sl, unc);
    }
    fn link_fec(&mut self, c: u64, link: u8, unc: bool, flit: u8) {
        self.inner.link_fec(c, link, unc, flit);
    }

    fn host_pkt(&mut self, c: u64, ctrl: bool, link: u8) {
        self.inner.host_pkt(c, ctrl, link)
    }
    fn tracks_waits(&self) -> bool {
        self.inner.tracks_waits()
    }
    fn profiling(&self) -> bool {
        true
    }
    fn phase(&mut self, p: Phase) {
        self.log = matches!(p, Phase::Driver | Phase::Calendar | Phase::Control | Phase::Dispatch);
        if self.inner_prof {
            self.inner.phase(p)
        }
    }
    fn exec_element(&mut self, e: &ExecElement) {
        self.execs.push(*e);
    }
    fn exec_begin(&mut self, entry: usize, queue: usize, c: u64) {
        self.in_exec = true;
        self.ops.push(Op::Exec);
        if self.inner_prof {
            self.inner.exec_begin(entry, queue, c)
        }
    }
    fn exec_end(&mut self) {
        self.in_exec = false;
        if self.inner_prof {
            self.inner.exec_end()
        }
    }
}

/// The schedule of one run.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Schedule {
    pub key: [u8; 32],
    /// Recorded with an observer that tracks waits (the `Wait` operations are then present).
    pub waits: bool,
    /// The cycle of every tick.
    pub cycles: Vec<u64>,
    /// The fetched-text hash at the end of every tick.
    pub text: Vec<u32>,
    /// Exact fetch identities/payloads of each tick, including empty ticks.
    pub fetched: Vec<u8>,
    pub fetched_off: Vec<usize>,
    /// The compiled form: `ops[ops_off[i]..ops_off[i + 1]]` are the operations of tick `i`, and the `Exec`s of the
    /// run take `execs` in order (`execs_off[i]` is the first of tick `i`).
    pub ops: Vec<Op>,
    pub ops_off: Vec<u32>,
    pub execs: Vec<ExecElement>,
    pub execs_off: Vec<u32>,
    /// The IQ levels at the end of the run (length, highest, lowest) for `--stats` after a compiled replay.
    pub iq_final: Vec<[u32; 3]>,
    /// The end-of-run records (`Machine::finish`: IQ-004, text left at completion) as (queue, cycle), in order: a compiled
    /// replay keeps no IQ contents to find them again, so it reports these.
    pub end_diags: Vec<(u16, u64)>,
}

impl Schedule {
    pub fn ticks(&self) -> usize {
        self.cycles.len()
    }

    /// An empty schedule for a recording.
    fn fresh(key: [u8; 32]) -> Schedule {
        Schedule { key, ops_off: vec![0], execs_off: vec![0], fetched_off: vec![0], ..Schedule::default() }
    }

    /// The heap bytes this schedule retains (the vectors' lengths times their element sizes, plus the struct): what the
    /// cache budget counts. A schedule of the P(8,4,128) bucket is about 62 MB (measured: `retained_bytes` 62,194,612).
    pub fn bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Schedule>()
            + self.cycles.len() * size_of::<u64>()
            + self.text.len() * size_of::<u32>()
            + self.fetched.len()
            + self.fetched_off.len() * size_of::<usize>()
            + self.ops.len() * size_of::<Op>()
            + self.ops_off.len() * size_of::<u32>()
            + self.execs.len() * size_of::<ExecElement>()
            + self.execs_off.len() * size_of::<u32>()
            + self.iq_final.len() * size_of::<[u32; 3]>()
            + self.end_diags.len() * size_of::<(u16, u64)>()
    }

    fn fetched_at(&self, i: usize) -> Option<&[u8]> {
        let start = *self.fetched_off.get(i)?;
        let end = *self.fetched_off.get(i + 1)?;
        self.fetched.get(start..end)
    }

    /// Can the compiled replay serve this schedule?
    pub fn compiled(&self) -> bool {
        self.ops_off.len() == self.cycles.len() + 1 && self.execs_off.len() == self.cycles.len() + 1
    }
}

/// Whether this build can record and replay schedules (cargo feature `replay`, off by default). When it cannot, the
/// tick loop does not contain the replay path and every request to record or replay is refused.
pub const ENABLED: bool = cfg!(feature = "replay");

/// How a run uses the recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    #[default]
    Off,
    /// The normal driver, noting its schedule.
    Record,
    /// The recorded schedule replaces the event search.
    Replay,
    /// The normal driver, compared per tick with the recorded schedule.
    Check,
}

/// What `Machine::replay_schedule` refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// The program, profile or starting state is not the recorded one.
    Key,
    /// The run is not one the schedule can serve (links, every-cycle ticking, a run already begun).
    NotApplicable(&'static str),
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Key => f.write_str("the schedule was recorded for another program, profile or starting state"),
            Refused::NotApplicable(w) => write!(f, "the schedule cannot serve this run: {w}"),
        }
    }
}

/// The replay state of a machine.
#[derive(Debug, Default)]
pub struct Rp {
    pub mode: Mode,
    /// Mode is Record or Check: the tick loop notes its schedule.
    pub rec_on: bool,
    pub rec: Schedule,
    pub exp: Option<Arc<Schedule>>,
    /// Physical text ranges whose exact word-write history belongs to a parametrised run.
    #[cfg(feature = "replay")]
    pub(crate) text_watch: Option<Vec<(usize, u64, u64)>>,
    /// The tick that is next in the recording (Replay, Check).
    pub pos: usize,
    /// Operations and executed elements of the tick being recorded start here.
    cur_ops: usize,
    cur_execs: usize,
    /// A chip-to-chip instruction executed: the run is not recordable.
    pub used_links: bool,
    /// A pending commit cycle and its independent presence bit.
    commit_at: AtomicU64,
    commit_valid: AtomicBool,
    started: bool,
    /// The cache that asked for this run's recording (see `replay_session_begin_in`): `replay_session_end` stores it there.
    session: Option<Arc<ReplayCache>>,
    /// The run ended in a replay divergence (a text hash, a tick not in the schedule or an unfinished schedule): the
    /// schedule does not describe this run, which has not run as the normal driver would have run it.
    pub diverged: bool,
    /// Mode is Replay and the schedule is compiled: ticks run `tick_compiled`.
    pub compiled: bool,
    /// A compiled replay ran its last tick: the driver's registers were not kept, the machine is quiescent as recorded.
    pub compiled_done: bool,
}

impl Rp {
    pub fn new() -> Rp {
        Rp::default()
    }

    /// Back to the start of the run, keeping the mode and the schedule.
    pub fn rewind(&mut self) {
        self.pos = 0;
        self.cur_ops = 0;
        self.cur_execs = 0;
        self.started = false;
        self.used_links = false;
        self.rec = Schedule::fresh(self.rec.key);
        self.compiled_done = false;
        self.diverged = false;
    }

    /// A chip-to-chip instruction executes (`slices/c2c`).
    #[inline]
    pub fn touch_links(&mut self) {
        self.used_links = true;
    }
}

/// The default byte budget of the process's schedule cache (`TSP_ISS_REPLAY_BUDGET`).
pub const DEFAULT_BUDGET: usize = 512 << 20;
/// The default number of schedules the cache keeps (`TSP_ISS_REPLAY_ENTRIES`).
pub const DEFAULT_ENTRIES: usize = 4;
/// The keys seen once that wait for a second sight (the oldest are forgotten).
const SEEN_MAX: usize = 1024;

/// When a program is recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordOn {
    /// The first run of a key records (the test device `--replay-runs`).
    First,
    /// The first sight of a key runs the normal driver and costs nothing; the second records; the later ones replay
    /// (serving: a plan that never recurs pays no recording and keeps no memory).
    Second,
}

/// What the cache counts, for `--stats` and `Platform::replay_counters`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplayCounters {
    /// Runs served from a stored schedule.
    pub hits: u64,
    /// Runs that found no stored schedule (the normal driver ran, or recorded).
    pub misses: u64,
    /// Replays whose text or tick differed from the recording: the schedule was dropped and the run took the normal driver.
    pub divergences: u64,
    /// Schedules removed to keep the budget.
    pub evictions: u64,
    /// Runs that recorded a schedule.
    pub recordings: u64,
    /// Schedules too big for the budget, not stored.
    pub oversize: u64,
    /// A stored schedule that the run could not use (an observer or a link the recording did not have).
    pub refused: u64,
    /// Shadow checks (a replayed run repeated on the normal driver) and the ones that differed.
    pub shadow_checks: u64,
    pub shadow_mismatches: u64,
    /// Parametrised replay (`Platform::start_plan`, `platform/param.rs`), counted over the platforms that use this cache:
    /// runs served from the recorded schedule of a bucket program shape with the plan's site values, runs the gate refused
    /// (they took the normal driver), served runs that were repeated on the normal driver (a divergence, a fault or a write over the
    /// text during the run), runs that recorded a shape, first sights of a shape, runs of a shape that measured under the
    /// minimum gain and took the normal driver, and shapes that cannot be parametrised.
    pub param_served: u64,
    pub param_refused: u64,
    pub param_fallback: u64,
    pub param_recordings: u64,
    pub param_first_sight: u64,
    pub param_small: u64,
    pub param_unbuildable: u64,
    /// The parametrised schedules held now (one per shape of a platform that uses this cache) and the bytes they hold; the bytes are
    /// part of `retained_bytes`, and the budget is one for these and the ordinary schedules together.
    pub param_shapes: u64,
    pub param_bytes: u64,
    /// The schedules and bytes stored now, and the limits.
    pub schedules: u64,
    pub retained_bytes: u64,
    pub budget_bytes: u64,
    pub max_schedules: u64,
}

/// What [`ReplayCache::param_note`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamEvent {
    Served,
    Refused,
    Fallback,
    Recorded,
    FirstSight,
    Small,
    Unbuildable,
}

/// A bounded LRU cache of schedules (`Arc<Schedule>`) keyed by [`Machine::schedule_key`]: at most `max_entries`
/// schedules and `budget` bytes; the least recently used schedule goes first. The parametrised schedules of the platforms that use
/// the cache ([`ParamLease`]) count against the same byte budget and are evicted in the same order of use. The process has one
/// ([`ReplayCache::global`]), a test makes its own.
#[derive(Debug)]
pub struct ReplayCache {
    budget: usize,
    max_entries: usize,
    inner: Mutex<CacheInner>,
}

#[derive(Debug, Default)]
struct CacheInner {
    map: HashMap<[u8; 32], Arc<Schedule>>,
    /// Least recently used first.
    lru: VecDeque<[u8; 32]>,
    seen: VecDeque<[u8; 32]>,
    seen_set: HashSet<[u8; 32]>,
    /// The keys whose schedule is bigger than the whole budget: recording them again would cost every run and keep nothing.
    big: HashSet<[u8; 32]>,
    bytes: usize,
    c: ReplayCounters,
    /// The tick of the last use of each stored schedule and of each parametrised one, so that one LRU order covers both.
    clock: u64,
    ticks: HashMap<[u8; 32], u64>,
    params: HashMap<u64, ParamHold>,
    next_param: u64,
}

#[derive(Debug)]
struct ParamHold {
    bytes: usize,
    used: u64,
    evicted: Arc<AtomicBool>,
    slot: Arc<ParamSlot>,
}

/// Where a parametrised schedule lives: the cache and the platform that recorded it share the slot. The platform works on the
/// schedule through [`ParamSlot::with`]; when the cache needs the room for something used more recently it empties the slot, so the
/// memory is freed at that moment, whether or not the platform steps that shape again. The schedule is a
/// `driver::replay_param::ParamSchedule` (that module needs the `replay` feature, this one does not, so the slot is untyped).
#[derive(Default)]
pub struct ParamSlot(Mutex<Option<Box<dyn Any + Send>>>);

impl fmt::Debug for ParamSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ParamSlot")
    }
}

impl ParamSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Box<dyn Any + Send>>> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Run `f` on the schedule, which is a `T`; `None` when the cache has taken it. The cache cannot take it while `f` runs.
    pub fn with<T: Any, R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        self.lock().as_mut().and_then(|b| b.downcast_mut::<T>()).map(f)
    }

    /// Whether the slot still holds its schedule.
    pub fn held(&self) -> bool {
        self.lock().is_some()
    }

    /// Free the schedule now.
    pub fn free(&self) {
        let old = self.lock().take();
        drop(old);
    }
}

/// The room a platform's parametrised schedule holds in the cache's budget. Dropping the lease gives the room back; when the cache
/// needs the room for something used more recently it takes it and frees the schedule in the slot ([`ParamLease::slot`]); the lease
/// says so ([`ParamLease::evicted`]).
#[derive(Debug)]
pub struct ParamLease {
    cache: Arc<ReplayCache>,
    id: u64,
    evicted: Arc<AtomicBool>,
    slot: Arc<ParamSlot>,
}

impl ParamLease {
    /// The cache took the room back (or forgot everything): the schedule is gone.
    pub fn evicted(&self) -> bool {
        self.evicted.load(Ordering::SeqCst)
    }

    /// The slot that holds the schedule of this lease.
    pub fn slot(&self) -> &Arc<ParamSlot> {
        &self.slot
    }

    /// The schedule was used: it is the most recently used one.
    pub fn touch(&self) {
        self.cache.param_touch(self.id);
    }
}

impl Drop for ParamLease {
    fn drop(&mut self) {
        self.cache.param_release(self.id);
    }
}

/// What [`ReplayCache::begin`] found for a key.
pub enum Lookup {
    /// A stored schedule.
    Hit(Arc<Schedule>),
    /// Nothing stored: record this run.
    Record,
    /// Nothing stored and the key is new (policy [`RecordOn::Second`]): run the normal driver, remember the key.
    FirstSight,
    /// The key's schedule was bigger than the whole budget: run the normal driver, record nothing.
    TooBig,
}

impl ReplayCache {
    pub fn new(budget: usize, max_entries: usize) -> ReplayCache {
        ReplayCache { budget, max_entries: max_entries.max(1), inner: Mutex::new(CacheInner::default()) }
    }

    /// The limits of the environment: `TSP_ISS_REPLAY_BUDGET` (bytes, with an optional `k`, `m` or `g` suffix; default
    /// 512 MB) and `TSP_ISS_REPLAY_ENTRIES` (default 4). A value that does not parse is reported on stderr and ignored.
    pub fn from_env() -> ReplayCache {
        let get = |name: &str| -> Option<usize> {
            let v = std::env::var(name).ok()?;
            match parse_size(&v) {
                Some(n) => Some(n),
                None => {
                    eprintln!(
                        "tsp-iss: {name}={v:?} is not a size (bytes, optionally with k, m or g); using the default"
                    );
                    None
                }
            }
        };
        ReplayCache::new(
            get("TSP_ISS_REPLAY_BUDGET").unwrap_or(DEFAULT_BUDGET),
            get("TSP_ISS_REPLAY_ENTRIES").unwrap_or(DEFAULT_ENTRIES),
        )
    }

    /// The cache of the process, limits from the environment at its first use.
    pub fn global() -> &'static Arc<ReplayCache> {
        static CACHE: OnceLock<Arc<ReplayCache>> = OnceLock::new();
        CACHE.get_or_init(|| Arc::new(ReplayCache::from_env()))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Look the key up: a hit refreshes its place in the LRU order, a miss follows the policy. Counts hit or miss.
    pub fn begin(&self, key: &[u8; 32], on: RecordOn) -> Lookup {
        let mut g = self.lock();
        if let Some(s) = g.map.get(key).cloned() {
            g.touch(key);
            g.c.hits += 1;
            return Lookup::Hit(s);
        }
        g.c.misses += 1;
        if g.big.contains(key) {
            return Lookup::TooBig;
        }
        if on == RecordOn::Second && !g.seen_set.contains(key) {
            g.seen_set.insert(*key);
            g.seen.push_back(*key);
            while g.seen.len() > SEEN_MAX {
                if let Some(old) = g.seen.pop_front() {
                    g.seen_set.remove(&old);
                }
            }
            return Lookup::FirstSight;
        }
        Lookup::Record
    }

    /// A hit that the run could not use: it did not count as a served run.
    pub fn unhit(&self) {
        let mut g = self.lock();
        g.c.hits = g.c.hits.saturating_sub(1);
        g.c.refused += 1;
    }

    /// Store a schedule recorded for its key, evicting the least recently used ones to stay within the budget and the
    /// number of schedules; a schedule bigger than the whole budget is not stored.
    pub fn insert(&self, s: Schedule) {
        let bytes = s.bytes();
        let mut g = self.lock();
        g.c.recordings += 1;
        if bytes > self.budget {
            g.c.oversize += 1;
            if g.big.len() >= SEEN_MAX {
                g.big.clear();
            }
            g.big.insert(s.key);
            return;
        }
        if let Some(old) = g.map.remove(&s.key) {
            g.bytes -= old.bytes();
            g.lru.retain(|k| *k != s.key);
            g.ticks.remove(&s.key);
        }
        g.make_room(bytes, self.budget, Some(self.max_entries));
        g.seen_set.remove(&s.key);
        g.seen.retain(|k| *k != s.key);
        g.bytes += bytes;
        g.lru.push_back(s.key);
        let t = g.tick();
        g.ticks.insert(s.key, t);
        g.map.insert(s.key, Arc::new(s));
    }

    /// A replay of the schedule stored for `key` diverged: drop it, so that the next sight of the key records again, and
    /// count the divergence.
    pub fn diverged(&self, key: &[u8; 32]) {
        let mut g = self.lock();
        g.c.divergences += 1;
        if let Some(old) = g.map.remove(key) {
            g.bytes -= old.bytes();
            g.lru.retain(|k| k != key);
            g.ticks.remove(key);
            g.seen_set.insert(*key);
            g.seen.push_back(*key);
        }
    }

    /// Take `bytes` of the budget for the parametrised schedule `ps` (its `ParamSchedule::bytes`), evicting what was used less recently,
    /// stored schedules and other platforms' parametrised ones alike; the lease owns the room and its slot holds the schedule. `None`
    /// (the schedule is dropped) when it is bigger than the whole budget.
    pub fn param_reserve(self: &Arc<Self>, bytes: usize, ps: Box<dyn Any + Send>) -> Option<ParamLease> {
        let mut g = self.lock();
        if bytes > self.budget {
            g.c.oversize += 1;
            return None;
        }
        g.make_room(bytes, self.budget, None);
        let (id, used) = (g.next_param, g.tick());
        g.next_param += 1;
        let evicted = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(ParamSlot(Mutex::new(Some(ps))));
        g.bytes += bytes;
        g.params.insert(id, ParamHold { bytes, used, evicted: evicted.clone(), slot: slot.clone() });
        Some(ParamLease { cache: self.clone(), id, evicted, slot })
    }

    fn param_touch(&self, id: u64) {
        let mut g = self.lock();
        let t = g.tick();
        if let Some(h) = g.params.get_mut(&id) {
            h.used = t;
        }
    }

    fn param_release(&self, id: u64) {
        let mut g = self.lock();
        if let Some(h) = g.params.remove(&id) {
            g.bytes -= h.bytes;
        }
    }

    /// Count one event of parametrised replay (`Platform::start_plan`).
    pub fn param_note(&self, what: ParamEvent) {
        let mut g = self.lock();
        let c = &mut g.c;
        match what {
            ParamEvent::Served => c.param_served += 1,
            ParamEvent::Refused => c.param_refused += 1,
            ParamEvent::Fallback => c.param_fallback += 1,
            ParamEvent::Recorded => c.param_recordings += 1,
            ParamEvent::FirstSight => c.param_first_sight += 1,
            ParamEvent::Small => c.param_small += 1,
            ParamEvent::Unbuildable => c.param_unbuildable += 1,
        }
    }

    /// A shadow check ran; `same` is whether the replayed run and the normal driver agreed.
    pub fn shadow(&self, same: bool) {
        let mut g = self.lock();
        g.c.shadow_checks += 1;
        if !same {
            g.c.shadow_mismatches += 1;
        }
    }

    /// The counters and the current size.
    pub fn counters(&self) -> ReplayCounters {
        let g = self.lock();
        ReplayCounters {
            schedules: g.map.len() as u64,
            retained_bytes: g.bytes as u64,
            param_shapes: g.params.len() as u64,
            param_bytes: g.params.values().map(|h| h.bytes as u64).sum(),
            budget_bytes: self.budget as u64,
            max_schedules: self.max_entries as u64,
            ..g.c
        }
    }

    /// The number of schedules stored.
    pub fn len(&self) -> usize {
        self.lock().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget every schedule and key (the counters stay).
    pub fn clear(&self) {
        let mut g = self.lock();
        g.map.clear();
        g.lru.clear();
        g.seen.clear();
        g.seen_set.clear();
        g.big.clear();
        g.ticks.clear();
        let ids: Vec<u64> = g.params.keys().copied().collect();
        for id in ids {
            if let Some(h) = g.params.remove(&id) {
                h.evicted.store(true, Ordering::SeqCst);
                h.slot.free();
            }
        }
        g.bytes = 0;
    }

    /// The stored schedules, least recently used first (a test and debugger hook: no counter, no LRU touch).
    pub fn schedules(&self) -> Vec<Arc<Schedule>> {
        let g = self.lock();
        g.lru.iter().filter_map(|k| g.map.get(k).cloned()).collect()
    }

    /// Whether a schedule for `key` is stored (no counter, no LRU touch).
    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.lock().map.contains_key(key)
    }
}

impl CacheInner {
    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn touch(&mut self, key: &[u8; 32]) {
        if let Some(i) = self.lru.iter().position(|k| k == key) {
            if let Some(k) = self.lru.remove(i) {
                self.lru.push_back(k);
                let t = self.tick();
                self.ticks.insert(k, t);
            }
        }
    }

    /// Evict, least recently used first across the stored schedules and the parametrised ones, until `need` more bytes fit in `budget`
    /// (and, when `max_entries` is given, one more stored schedule fits). A parametrised schedule is evicted only for bytes.
    fn make_room(&mut self, need: usize, budget: usize, max_entries: Option<usize>) {
        loop {
            let over_bytes = self.bytes + need > budget;
            let over_entries = max_entries.is_some_and(|m| self.map.len() >= m);
            if !over_bytes && !over_entries {
                break;
            }
            let reg = self.lru.front().map(|k| (self.ticks.get(k).copied().unwrap_or(0), *k));
            let par = if over_bytes { self.params.iter().map(|(id, h)| (h.used, *id)).min() } else { None };
            match (reg, par) {
                (Some((rt, _)), Some((pt, id))) if pt < rt => self.evict_param(id),
                (Some((_, k)), _) => {
                    self.lru.pop_front();
                    self.ticks.remove(&k);
                    if let Some(old) = self.map.remove(&k) {
                        self.bytes -= old.bytes();
                        self.c.evictions += 1;
                    }
                }
                (None, Some((_, id))) => self.evict_param(id),
                (None, None) => break,
            }
        }
    }

    fn evict_param(&mut self, id: u64) {
        if let Some(h) = self.params.remove(&id) {
            self.bytes -= h.bytes;
            h.evicted.store(true, Ordering::SeqCst);
            h.slot.free();
            self.c.evictions += 1;
        }
    }
}

/// `1048576`, `512k`, `64m`, `2g` (powers of 1024).
pub fn parse_size(v: &str) -> Option<usize> {
    let v = v.trim();
    let (digits, mul) = match v.chars().last()?.to_ascii_lowercase() {
        'k' => (&v[..v.len() - 1], 1usize << 10),
        'm' => (&v[..v.len() - 1], 1 << 20),
        'g' => (&v[..v.len() - 1], 1 << 30),
        _ => (v, 1),
    };
    digits.trim().parse::<usize>().ok()?.checked_mul(mul)
}

/// Schedules kept in the process for `Machine::replay_session_begin`.
pub fn session_cache_len() -> usize {
    ReplayCache::global().len()
}

/// Forget the schedules of the process.
pub fn session_cache_clear() {
    ReplayCache::global().clear();
}

/// What `Machine::replay_session_begin` decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Began {
    Recording,
    Replaying,
    /// This run cannot be recorded or replayed; it takes the normal driver.
    Normal,
}

fn check_env() -> bool {
    std::env::var_os("TSP_ISS_REPLAY_CHECK").is_some_and(|v| !v.is_empty() && v != "0")
}

impl<const L: usize> Machine<L> {
    /// The key of the run about to start: the profile and the initial IQ images. A machine that has not ticked is in
    /// the state `reset` gives (every other register is a function of the images), which `rp_applicable` requires;
    /// the SRAM is not part of the key. [`Machine::schedule_key_full`] digests every register instead.
    pub fn schedule_key(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"opentsp-schedule-key-2");
        h.update(format!("{:?}|{:?}|{:?}", self.geom, self.timing, self.images).as_bytes());
        h.finalize().into()
    }

    /// The key of the run about to start: the profile and every register of the machine except the SRAM.
    pub fn schedule_key_full(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"opentsp-schedule-key-1");
        h.update(format!("{:?}|{:?}", self.geom, self.timing).as_bytes());
        h.update(self.state_digest());
        h.finalize().into()
    }

    fn rp_applicable(&self) -> Result<(), Refused> {
        if !ENABLED {
            return Err(Refused::NotApplicable("this build has no schedule replay (cargo feature `replay`)"));
        }
        if self.iq.observe_sites {
            return Err(Refused::NotApplicable("site observation needs the live instruction queues"));
        }
        if self.rp.mode != Mode::Off && self.rp.started {
            return Err(Refused::NotApplicable("a run is already under way"));
        }
        if self.next_tick() != Some(0) {
            return Err(Refused::NotApplicable("the machine has ticked"));
        }
        Ok(())
    }

    /// Note the schedule of the run that follows (the normal driver). Take it with [`Machine::take_schedule`].
    pub fn record_schedule(&mut self) -> Result<(), Refused> {
        self.rp_applicable()?;
        let key = self.schedule_key();
        self.rp = Rp { mode: Mode::Record, rec_on: true, ..Rp::default() };
        self.rp.rec = Schedule::fresh(key);
        self.iq.th = TextHash { on: true, ..TextHash::default() };
        Ok(())
    }

    /// The schedule of the run that finished, `None` when the run is not recordable (a link was used) or not
    /// recorded. The machine goes back to the normal driver.
    pub fn take_schedule(&mut self) -> Option<Schedule> {
        let iq = &self.iq;
        let iq_final: Vec<[u32; 3]> =
            (0..crate::geom::QUEUES).map(|q| [iq.len[q] as u32, iq.hi[q] as u32, iq.lo[q] as u32]).collect();
        let rp = std::mem::take(&mut self.rp);
        self.iq.th = TextHash::default();
        if rp.mode != Mode::Record || rp.used_links || rp.rec.cycles.is_empty() {
            return None;
        }
        let mut rec = rp.rec;
        rec.iq_final = iq_final;
        Some(rec)
    }

    /// Replay `s` in the run that follows: the key must be the machine's. With `TSP_ISS_REPLAY_CHECK=1` the
    /// normal driver runs and every tick is compared with `s`.
    pub fn replay_schedule(&mut self, s: Arc<Schedule>) -> Result<(), Refused> {
        self.replay_schedule_with(s, check_env())
    }

    /// [`Machine::replay_schedule`] with the check chosen by the caller instead of the environment.
    pub fn replay_schedule_with(&mut self, s: Arc<Schedule>, check: bool) -> Result<(), Refused> {
        self.rp_applicable()?;
        if s.key != self.schedule_key() {
            return Err(Refused::Key);
        }
        if s.cycles.is_empty() {
            return Err(Refused::NotApplicable("the schedule has no tick"));
        }
        if !check && !s.compiled() {
            return Err(Refused::NotApplicable("the schedule has no operations"));
        }
        let mode = if check { Mode::Check } else { Mode::Replay };
        self.rp = Rp { mode, rec_on: check, compiled: !check, exp: Some(s.clone()), ..Rp::default() };
        self.rp.rec = Schedule::fresh(s.key);
        self.iq.th = TextHash { on: true, ..TextHash::default() };
        if !check {
            self.t_next_set(Some(s.cycles[0]));
        }
        Ok(())
    }

    /// Switch the replay off before the first tick (an observer the schedule does not fit).
    fn rp_abandon(&mut self) {
        #[cfg(feature = "replay")]
        if self.rp.text_watch.is_some() {
            self.sram.journal_end();
        }
        self.rp = Rp::default();
        self.iq.th = TextHash::default();
        self.t_next_set(Some(0));
    }

    /// The in-process cache: replay the schedule recorded for this program in this process, or record one (to be
    /// stored by [`Machine::replay_session_end`]). The process's cache, the first run of a key records (the test
    /// device `--replay-runs`).
    pub fn replay_session_begin(&mut self) -> Began {
        self.replay_session_begin_in(ReplayCache::global(), RecordOn::First)
    }

    /// [`Machine::replay_session_begin`] with the cache and the recording policy chosen by the caller. A hit is
    /// counted when the schedule serves the run, a miss when none is stored. With [`RecordOn::Second`] the first
    /// sight of a key runs the normal driver (`Began::Normal`).
    pub fn replay_session_begin_in(&mut self, cache: &Arc<ReplayCache>, on: RecordOn) -> Began {
        let key = self.schedule_key();
        match cache.begin(&key, on) {
            Lookup::Hit(s) => match self.replay_schedule(s) {
                Ok(()) => Began::Replaying,
                Err(_) => {
                    cache.unhit();
                    Began::Normal
                }
            },
            Lookup::FirstSight | Lookup::TooBig => Began::Normal,
            Lookup::Record => match self.record_schedule() {
                Ok(()) => {
                    self.rp.session = Some(cache.clone());
                    Began::Recording
                }
                Err(_) => Began::Normal,
            },
        }
    }

    /// After a run that finished well: keep the schedule a session run recorded.
    pub fn replay_session_end(&mut self) {
        if let Some(cache) = self.rp.session.clone() {
            if let Some(s) = self.take_schedule() {
                cache.insert(s);
            }
        }
    }

    /// The key of the schedule this run replays (`None`: the run does not replay).
    pub fn replay_key(&self) -> Option<[u8; 32]> {
        self.rp.exp.as_ref().map(|s| s.key)
    }

    /// The mode of the run (for the reports of the CLI).
    pub fn replay_mode(&self) -> Mode {
        self.rp.mode
    }

    /// The first tick of a run: a replay needs the observer to track waits as the recording's did.
    pub(crate) fn rp_start(&mut self, waits: bool) {
        if self.rp.started {
            return;
        }
        self.rp.started = true;
        if self.rp.mode == Mode::Off {
            return;
        }
        if self.every_cycle {
            self.rp_abandon(); // the idle cycles between events are ticks too: no schedule says what they do
            return;
        }
        let want = self.rp.exp.as_ref().map(|e| e.waits);
        if self.rp.rec_on && self.rp.exp.is_none() || want == Some(waits) {
            self.rp.rec.waits = waits;
            return;
        }
        self.rp_abandon(); // recorded with the other setting of `tracks_waits`
    }

    /// The end of a tick: close its record, compare it in a check, verify the text in a replay.
    pub(crate) fn rp_end(&mut self, c: u64) -> Result<(), RunError> {
        let text = self.iq.th.tick();
        self.rp.rec.cycles.push(c);
        self.rp.rec.text.push(text);
        self.rp.rec.fetched.extend_from_slice(&self.iq.th.bytes);
        self.rp.rec.fetched_off.push(self.rp.rec.fetched.len());
        if self.rp.mode == Mode::Check {
            let s = self.rp.exp.as_ref().expect("check");
            let i = self.rp.pos;
            if s.cycles.get(i) != Some(&c)
                || s.text.get(i) != Some(&text)
                || (self.iq.th.on && s.fetched_at(i) != Some(self.iq.th.bytes.as_slice()))
            {
                self.rp.diverged = true;
                return Err(RunError::fault(
                    c,
                    &format!("replay divergence: tick {i} (cycle {c}) differs from the schedule"),
                ));
            }
            self.rp.pos += 1;
        }
        self.iq.th.bytes.clear();
        Ok(())
    }

    /// The next tick of a replay, after the tick at `c`: the next one of the schedule, or an earlier cycle at which
    /// a commit is pending. Which writes commit depends on the data (a diagnostic of an operation, as of a Permute
    /// whose map is not a permutation, drops its write), so the schedule of one run cannot say which cycles have a
    /// commit to apply: the commit ring does. Every other cause of a tick is a function of the program.
    pub(crate) fn rp_next(&self, c: Option<u64>) -> Option<u128> {
        let rec = self.rp.exp.as_ref().and_then(|s| s.cycles.get(self.rp.pos).copied().map(u128::from));
        let pend = self.next_pending_commit(c);
        let executable = pend.and_then(|b| u64::try_from(b).ok());
        self.rp.commit_at.store(executable.unwrap_or(0), Ordering::Relaxed);
        self.rp.commit_valid.store(executable.is_some(), Ordering::Relaxed);
        match (rec, pend) {
            (a, None) => a,
            (None, b) => b,
            (Some(a), Some(b)) => Some(a.min(b)),
        }
    }

    /// The first cycle after `c` whose commit-ring slot holds a commit.
    fn next_pending_commit(&self, c: Option<u64>) -> Option<u128> {
        let d = &self.d;
        let n = super::PENDING_SLOTS as u128;
        let base = c.map_or(0, |c| u128::from(c) + 1);
        let mut occ = d.pending_occ.rotate_right((base % n) as u32);
        while occ != 0 {
            let i = occ.trailing_zeros() as u128;
            occ &= occ - 1;
            if !d.pending[((base + i) % n) as usize].is_empty() {
                return Some(base + i);
            }
        }
        None
    }

    /// The end of a run: a replay or a check must have used the whole schedule.
    pub(crate) fn rp_finish(&mut self) -> Result<(), RunError> {
        if let Some(s) = &self.rp.exp {
            if self.rp.pos != s.cycles.len() {
                self.rp.diverged = true;
                return Err(RunError::Internal(format!(
                    "replay divergence: the run ended after {} of {} ticks of the schedule",
                    self.rp.pos,
                    s.cycles.len()
                )));
            }
        }
        #[cfg(feature = "replay")]
        if let Some(watch) = &self.rp.text_watch {
            if self.sram.journal_pages().is_none()
                || watch.iter().any(|&(slice, addr, words)| self.sram.journal_wrote_range(slice, addr, words))
            {
                self.rp.diverged = true;
                return Err(RunError::Internal(
                    "replay divergence: instruction text was written or its write history was lost".into(),
                ));
            }
        }
        Ok(())
    }
}

impl<const L: usize> Machine<L> {
    /// Record the tick (the normal driver) with the operations of its driver: the tick body runs against a `RecObs`.
    pub(crate) fn tick_recording<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) -> Result<(), RunError> {
        let mut ops = std::mem::take(&mut self.rp.rec.ops);
        let mut execs = std::mem::take(&mut self.rp.rec.execs);
        let inner_prof = obs.profiling();
        let r = {
            let mut ro = RecObs { inner: obs, ops: &mut ops, execs: &mut execs, inner_prof, log: true, in_exec: false };
            self.tick_cycle(c, &mut ro)
        };
        let r = r.and_then(|()| self.rp_end(c));
        self.rp.rec.ops = ops;
        self.rp.rec.execs = execs;
        r?;
        let (oend, eend) = (self.rp.rec.ops.len(), self.rp.rec.execs.len());
        self.rp.rec.ops_off.push(oend as u32);
        self.rp.rec.execs_off.push(eend as u32);
        if self.rp.mode == Mode::Check {
            // `rp_end` has counted the tick: `pos` is the next one
            let s = self.rp.exp.as_ref().expect("check");
            let i = self.rp.pos - 1;
            let same = !s.compiled()
                || (s.ops.get(s.ops_off[i] as usize..s.ops_off[i + 1] as usize)
                    == Some(&self.rp.rec.ops[self.rp.cur_ops..oend])
                    && s.execs.get(s.execs_off[i] as usize..s.execs_off[i + 1] as usize)
                        == Some(&self.rp.rec.execs[self.rp.cur_execs..eend]));
            if !same {
                self.rp.diverged = true;
                return Err(RunError::fault(
                    c,
                    &format!("replay divergence: tick {i} (cycle {c}) runs other operations than the schedule"),
                ));
            }
        }
        self.rp.cur_ops = oend;
        self.rp.cur_execs = eend;
        Ok(())
    }

    /// One tick of the compiled replay: the commits due, the recorded operations in their order, the end-of-cycle flush.
    /// Nothing of the IQs, the hold checks, the calendar or the AGT and Repeat registers runs.
    pub(crate) fn tick_compiled<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) -> Result<(), RunError> {
        let s = self.rp.exp.take().expect("replaying");
        let r = self.tick_compiled_with(&s, c, obs);
        self.rp.exp = Some(s);
        r
    }

    fn tick_compiled_with<O: Observer + ?Sized>(&mut self, s: &Schedule, c: u64, obs: &mut O) -> Result<(), RunError> {
        let i = self.rp.pos;
        let extra = match s.cycles.get(i).copied() {
            Some(t) if t == c => false,
            Some(t)
                if t > c
                    && self.rp.commit_valid.load(Ordering::Relaxed)
                    && self.rp.commit_at.load(Ordering::Relaxed) == c =>
            {
                true
            }
            None if self.rp.commit_valid.load(Ordering::Relaxed) && self.rp.commit_at.load(Ordering::Relaxed) == c => {
                true
            }
            _ => {
                self.rp.diverged = true;
                return Err(RunError::fault(
                    c,
                    &format!("replay divergence: tick {i} of the schedule is not cycle {c}"),
                ));
            }
        };
        self.local_fetch_complete(c, obs);
        self.commit_due(c, obs);
        self.stream_fetch_complete(c, obs);
        if extra {
            self.unit_state_flush(c, obs);
            return Ok(());
        }
        let ops = &s.ops[s.ops_off[i] as usize..s.ops_off[i + 1] as usize];
        let mut ei = s.execs_off[i] as usize;
        for op in ops {
            match *op {
                Op::Exec => {
                    let e = &s.execs[ei];
                    ei += 1;
                    self.execute_element(e, obs);
                    if matches!(e.instr.id(), EntryId::Ifetch | EntryId::IfetchM) {
                        self.iq.fill[e.q].clear(); // the text only feeds the hash; nothing reads the IQ in a compiled replay
                    }
                }
                Op::Dispatch(q) => obs.dispatch(c, q),
                Op::Diag(q, d) => self.diag(obs, if q == 0xffff { 144 } else { q as usize }, c, d),
                Op::Lint(q, k) => obs.lint(c, q, LINTS[k as usize]),
                Op::Wait(q, why, since) => obs.wait(c, q, why, since),
            }
        }
        self.unit_state_flush(c, obs);
        if s.text[i] != self.iq.th.tick() || (self.iq.th.on && s.fetched_at(i) != Some(self.iq.th.bytes.as_slice())) {
            self.rp.diverged = true;
            return Err(RunError::fault(
                c,
                "replay divergence: the instruction text fetched differs from the recorded run",
            ));
        }
        self.iq.th.bytes.clear();
        self.rp.pos += 1;
        if self.rp.pos == s.cycles.len() {
            self.rp_install_final(s);
        }
        Ok(())
    }

    /// The last tick of a compiled replay: the IQ levels `--stats` reports, and the quiescence the recording ended in.
    fn rp_install_final(&mut self, s: &Schedule) {
        for (q, l) in s.iq_final.iter().enumerate() {
            self.iq.len[q] = l[0] as usize;
            self.iq.hi[q] = l[1] as usize;
            self.iq.lo[q] = l[2] as usize;
        }
        self.rp.compiled_done = true;
    }

    /// A compiled replay that finished: the recorded run was quiescent and what a replay changes since then is the commit ring.
    pub(crate) fn rp_quiescent(&self) -> Option<bool> {
        self.rp.compiled_done.then(|| self.d.pending.iter().all(|s| s.is_empty()))
    }
}

#[cfg(feature = "replay")]
#[cfg(test)]
#[path = "../../tests/private/replay_text.rs"]
mod text_tests;
