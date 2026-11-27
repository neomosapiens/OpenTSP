//! Parametrised schedule replay in the platform (`opentsp-iss/README.md`; behind the cargo feature
//! `replay`, off unless the host switches it on with [`Platform::set_param_replay`]).
//!
//! A *shape* is a bucket program as the host serves it: the patch sites (the fields the host rewrites for every step), the whole
//! text of its queues and where the SRAM holds that text. The host registers a shape once ([`Platform::add_param_shape`]) and starts
//! each step with [`Platform::start_plan`] instead of [`Platform::start`], handing over the step's site values (one per site, in the
//! order of the table). The platform then does, per shape:
//!
//! 1. **first sight**: the normal driver runs (it costs nothing and is the baseline of the measurement);
//! 2. **second sight**: the normal driver runs and records its schedule; the schedule is parametrised for the plan's values
//!    (`ParamSchedule::build`); a shape that cannot be (an element that fits no instruction, a bank clash in the recording) is
//!    served by the normal driver from then on and counted;
//! 3. **later steps**: the new values go into the recorded schedule (`instantiate`: range, grid, decode, bank rules) and the gate
//!    (`arm`: no ECC fault live, the images equal outside the sites, every word of the text arena equal to the plan's text) starts
//!    the compiled replay. A step that a gate refuses takes the normal driver (counted as refused). One served step in 64 of each
//!    shape (the first included) runs in the per-tick check, the normal driver compared with the schedule tick by tick
//!    ([`CheckMode`]; `TSP_ISS_REPLAY_CHECK` overrides): the gate is a list of rules, the check is the model's own dispatch, and a
//!    divergence it finds is a fallback to the normal driver like any other;
//! 4. **the measured gain**: the baseline is a warm normal run (a ready shape runs one step in 32 on the normal driver, a shape served
//!    by the normal driver every step; the run of the first sight and the run that records are cold or slower and are not baselines).
//!    A decision is made after every five replayed steps that did not run in the check, on the median of those five; a shape whose
//!    gain is under the minimum ([`Platform::set_param_min_gain`], default 1.2) in two decisions in a row is served by the normal
//!    driver, and it is measured again (recorded and probed) after 256 steps. The measurement is in [`Platform::replay_status`].
//!    The rule leans towards demotion on a noisy host: it compares the minimum of the last four warm normal runs with the median of the
//!    five replays, so a spike in the replays can demote a shape that two quiet decisions would keep (it takes two in a row, and the
//!    shape is measured again after 256 steps).
//!
//! What closes the hole of the proof of concept (a fault or a write over the text *during* a run). The replay does not read the
//! text it runs: the gate checked it before the start. A served run therefore keeps a journal of the SRAM pages it writes
//! (`Sram::journal_begin`) and the host calls it makes; then
//! * a fault flip or a scrub between start and done sends the run to the normal driver at that call: the SRAM is put back to the
//!   start, a new chip repeats the calls so far on the normal driver and takes over (counted as a fallback);
//! * a run that is done is checked again against the text of the plan (`text_gate`): a word over the text, written by the program
//!   itself, sends it to the normal driver the same way;
//! * a divergence of the replay (a tick that is not the schedule's) does the same and the shape records again.
//!
//! The journal retains exact word-write history, including writes restored before completion. Neighboring data words
//! on the same rollback page do not force a fallback. The successful arm owns the history; the platform keeps it until
//! all completion gates pass and then closes it, or restores the preimages before a normal-driver fallback.
use super::*;
use crate::driver::replay::{ParamEvent, ParamLease};
use crate::driver::replay_param::{Gate, ParamSchedule, QueueText, Run, Site};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Warm normal-driver timings kept per shape, and the replayed runs that decide the measured gain.
const NORMAL_SAMPLES: usize = 4;
const PROBE_REPLAYS: usize = 5;
/// A ready shape runs one step in this many on the normal driver, so that the baseline of the measured gain is a recent warm run.
const SAMPLE_EVERY: u64 = 32;
/// Decisions under the minimum gain, in a row, that serve a shape by the normal driver.
const LOW_DECISIONS: u32 = 2;
/// A shape served by the normal driver for its small gain is measured again after this many steps.
const REPROBE_AFTER: u64 = 256;
/// A shape whose replay diverged this often is served by the normal driver from then on.
const MAX_DIVERGENCES: u64 = 3;
/// The default minimum gain (measured normal time over replayed time, plan work included).
pub const DEFAULT_MIN_GAIN: f64 = 1.2;
/// Of the served steps of a shape, one in this many runs in the per-tick check (the first one included) unless the host or
/// `TSP_ISS_REPLAY_CHECK` says otherwise: the gate is a list of rules, the check is the model's own dispatch.
pub const DEFAULT_CHECK_EVERY: u64 = 64;

/// Which served steps run in the per-tick check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckMode {
    /// None.
    Never,
    /// Every served step.
    Always,
    /// One in `n` served steps of each shape, the first included.
    Every(u64),
}

impl CheckMode {
    fn of_env(v: Option<std::ffi::OsString>) -> CheckMode {
        match v {
            None => CheckMode::Every(DEFAULT_CHECK_EVERY),
            Some(v) if v.is_empty() || v == "0" => CheckMode::Never,
            Some(_) => CheckMode::Always,
        }
    }

    fn runs(self, served: u64) -> bool {
        match self {
            CheckMode::Never => false,
            CheckMode::Always => true,
            CheckMode::Every(n) => served.is_multiple_of(n.max(1)),
        }
    }
}

/// What the host tells the platform about a bucket program shape.
#[derive(Clone, Debug)]
pub struct ParamSpec {
    /// The patch sites, in the order of the values of [`Platform::start_plan`].
    pub sites: Vec<Site>,
    /// The whole text of every queue of the program (initial image, then the blocks, padding included), with any plan's values.
    pub text: Vec<QueueText>,
    /// Where the SRAM holds the blocks of that text.
    pub arena: Vec<Run>,
}

/// Why a shape is served by the normal driver, or why a served run was repeated on it.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamWhy {
    /// The measured gain was under the minimum: the gain, the normal and the replayed time of a step in microseconds.
    Small { gain: f64, normal_us: f64, replay_us: f64 },
    /// The recorded schedule cannot be parametrised (or cannot be recorded).
    Unbuildable(String),
    /// The replay diverged [`MAX_DIVERGENCES`] times.
    Unstable,
    /// The schedule is bigger than the whole budget of the replay cache.
    TooBig,
    /// (a fallback) the replay diverged from the schedule.
    Diverged,
    /// (a fallback) the text of the plan was not in the SRAM when the run was done.
    Text(String),
    /// (a fallback) the host injected a fault or scrubbed during the run.
    Fault,
}

enum State {
    First,
    Seen,
    Ready,
    Normal(ParamWhy),
}

struct Shape {
    spec: ParamSpec,
    state: State,
    /// The last warm normal runs (a ready shape samples one in [`SAMPLE_EVERY`] of its steps, a shape served by the normal driver
    /// for its small gain every step): the baseline of the measured gain. The run of the first sight and the run that records
    /// are not in it (they are cold, and recording costs more).
    normal_ns: Vec<u64>,
    replay_ns: Vec<u64>,
    /// Replayed steps timed since the measured gain was last decided, and the decisions under the minimum in a row.
    since_probe: usize,
    low_decisions: u32,
    /// Steps since the last warm normal sample (a ready shape), and since the shape went to the normal driver for its gain.
    since_sample: u64,
    small_steps: u64,
    served: u64,
    /// Served steps that ran in the per-tick check.
    checked: u64,
    refused: u64,
    fallback: u64,
    normal_runs: u64,
    divergences: u64,
    bytes: usize,
    used: u64,
    /// The stored schedule and the room it holds in the cache's budget (the one budget of the process, shared with the other
    /// platforms and the ordinary schedules). The schedule lives in the lease's slot: when the cache evicts the shape, the memory is
    /// freed there, whether or not this platform steps the shape again.
    lease: Option<ParamLease>,
}

impl Shape {
    /// Forget the stored schedule and give its room back.
    fn drop_schedule(&mut self) {
        if let Some(l) = self.lease.take() {
            l.slot().free();
        }
        self.bytes = 0;
    }

    /// The schedule is stored and in memory (the cache may have taken it).
    fn held(&self) -> bool {
        self.lease.as_ref().is_some_and(|l| l.slot().held())
    }
}

#[derive(Clone, Debug)]
enum Kind {
    /// A run on the normal driver; `sample`: its time is a baseline of the measured gain.
    Normal {
        sample: bool,
    },
    Record(Vec<i32>),
    /// `checked`: it also ran in the per-tick check, so its time says nothing of the replay.
    Served {
        checked: bool,
    },
}

pub(super) struct ParamRun {
    shape: String,
    kind: Kind,
    /// Plan work of the start, and the time spent in `advance` so far.
    spent: Duration,
}

pub(super) struct ParamState {
    on: bool,
    min_gain: f64,
    check: CheckMode,
    shapes: HashMap<String, Shape>,
    replacement: Option<(String, ParamSpec)>,
    pub(super) run: Option<ParamRun>,
    tick: u64,
    /// A replacement for the clock of the measured gain (tests): the time in nanoseconds of the step that ended, given whether it was
    /// served by the schedule.
    timing: Option<Box<dyn FnMut(bool) -> u64 + Send>>,
}

impl ParamState {
    pub(super) fn new() -> ParamState {
        let min_gain = std::env::var("TSP_ISS_PARAM_MIN_GAIN")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(DEFAULT_MIN_GAIN);
        ParamState {
            on: false,
            min_gain,
            check: CheckMode::of_env(std::env::var_os("TSP_ISS_REPLAY_CHECK")),
            shapes: HashMap::new(),
            replacement: None,
            run: None,
            tick: 0,
            timing: None,
        }
    }
}

/// What the platform did for one shape (`Platform::replay_status`).
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeStatus {
    pub shape: String,
    /// `first`, `seen`, `ready` (a schedule is stored) or `normal` (served by the normal driver, see `why`).
    pub state: &'static str,
    pub why: Option<ParamWhy>,
    pub served: u64,
    /// Served steps that ran in the per-tick check (see [`CheckMode`]).
    pub checked: u64,
    pub refused: u64,
    pub fallback: u64,
    pub normal_runs: u64,
    /// Measured normal time over replayed time (once a replayed step was timed), and the two timings in microseconds: the
    /// fastest normal run and the median replayed one.
    pub gain: Option<f64>,
    pub normal_us: Option<f64>,
    pub replay_us: Option<f64>,
    pub retained_bytes: u64,
}

/// The parametrised replay of this platform: whether it is on, the threshold, the totals and one row per shape.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamStatus {
    pub enabled: bool,
    pub min_gain: f64,
    pub check: CheckMode,
    pub served: u64,
    pub checked: u64,
    pub refused: u64,
    pub fallback: u64,
    pub normal_runs: u64,
    pub shapes: Vec<ShapeStatus>,
}

fn median(v: &[u64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    match s.len() {
        0 => 0.0,
        n if n % 2 == 1 => s[n / 2] as f64,
        n => (s[n / 2 - 1] + s[n / 2]) as f64 / 2.0,
    }
}

impl Platform {
    /// **ISS speed device, opt-in (`platform.md` §1.2):** serve the steps of a registered bucket program shape from one recorded
    /// schedule (see the module). Off by default; a build without the cargo feature `replay` has no such device.
    /// Changes the selection for subsequent starts. An in-flight run retains its integrity checks, rollback and journal
    /// cleanup through completion, even when replay is disabled between calls to `advance`.
    pub fn set_param_replay(&mut self, on: bool) {
        self.param.on = on;
    }

    /// Whether [`Platform::set_param_replay`] is on.
    pub fn param_replay(&self) -> bool {
        self.param.on
    }

    /// The minimum measured gain (normal time over replayed time) under which a shape is served by the normal driver.
    pub fn set_param_min_gain(&mut self, gain: f64) {
        self.param.min_gain = gain;
    }

    /// Run every served step (`true`) or none (`false`) in the per-tick check (the normal driver compared with the schedule tick by
    /// tick). The default is one served step in [`DEFAULT_CHECK_EVERY`] of each shape; `TSP_ISS_REPLAY_CHECK` (`0` or any other
    /// value) sets the mode at start-up.
    pub fn set_param_check(&mut self, on: bool) {
        self.param.check = if on { CheckMode::Always } else { CheckMode::Never };
    }

    /// Run one served step in `n` of each shape in the per-tick check.
    pub fn set_param_check_every(&mut self, n: u64) {
        self.param.check = CheckMode::Every(n);
    }

    /// The per-tick check mode of the served steps.
    pub fn param_check(&self) -> CheckMode {
        self.param.check
    }

    /// Replace the clock of the measured gain by `clock`, which gets whether the step that ended was served by the schedule and gives
    /// its time in nanoseconds (tests of the gain rule inject timings with it). `None` restores the clock.
    pub fn set_param_timing(&mut self, clock: Option<Box<dyn FnMut(bool) -> u64 + Send>>) {
        self.param.timing = clock;
    }

    /// Register (or replace) the shape `id`. A shape that is replaced is forgotten with its stored schedule.
    /// Replacement of the active shape takes effect after its run completes or is reset/restarted. Until then its original
    /// specification, schedule, integrity checks and accounting remain owned by that run. Repeated replacements keep the last.
    pub fn add_param_shape(&mut self, id: &str, spec: ParamSpec) {
        if self.phase == Phase::Running && self.param.run.as_ref().is_some_and(|r| r.shape == id) {
            self.param.replacement = Some((id.to_string(), spec));
            return;
        }
        if self.param.replacement.as_ref().is_some_and(|(pending, _)| pending == id) {
            self.param.replacement = None;
        }
        self.param.shapes.insert(
            id.to_string(),
            Shape {
                spec,
                state: State::First,
                normal_ns: Vec::new(),
                replay_ns: Vec::new(),
                since_probe: 0,
                low_decisions: 0,
                since_sample: 0,
                small_steps: 0,
                served: 0,
                checked: 0,
                refused: 0,
                fallback: 0,
                normal_runs: 0,
                divergences: 0,
                bytes: 0,
                used: 0,
                lease: None,
            },
        );
    }

    pub(super) fn param_finish_replacement(&mut self) {
        if let Some((id, spec)) = self.param.replacement.take() {
            self.add_param_shape(&id, spec);
        }
    }

    /// `CTRL.start` for a step of the registered shape `shape` with the site values `values` (the program's images and the SRAM are
    /// loaded as for [`Platform::start`]). With the device off, or for a shape that is not registered, it is `start`. The step
    /// gives the results, diagnostics and cycle count of the normal driver whichever way it runs.
    pub fn start_plan(&mut self, shape: &str, values: &[i32]) -> Result<(), PlatformError> {
        if !self.param.on || !self.param.shapes.contains_key(shape) {
            return self.start();
        }
        self.need_phase(Phase::Reset)?;
        let t0 = Instant::now();
        self.phase = Phase::Running;
        let cache = self.replay.cache.clone();
        let tainted = self.replay.tainted;
        self.param.tick += 1;
        let tick = self.param.tick;
        let mode = self.param.check;
        let sh = self.param.shapes.get_mut(shape).expect("registered");
        let check = mode.runs(sh.served);
        sh.used = tick;
        if matches!(sh.state, State::Ready) && !sh.held() {
            // the cache needed the room for something used more recently and freed the schedule: the shape records again
            sh.drop_schedule();
            sh.state = State::Seen;
        }
        let mut kind = Kind::Normal { sample: false };
        if tainted {
            // a fault flip or a scrub between reset and start: the SRAM and the check bits are not the plan's
            sh.refused += 1;
            cache.param_note(ParamEvent::Refused);
        } else {
            match &sh.state {
                State::Normal(why) => {
                    sh.normal_runs += 1;
                    if matches!(why, ParamWhy::Small { .. }) {
                        cache.param_note(ParamEvent::Small);
                        // a warm normal run: it keeps the baseline current, and after a while the gain is measured again
                        kind = Kind::Normal { sample: true };
                        sh.small_steps += 1;
                        if sh.small_steps >= REPROBE_AFTER {
                            sh.small_steps = 0;
                            sh.state = State::Seen;
                        }
                    }
                }
                State::First => {
                    sh.state = State::Seen;
                    cache.param_note(ParamEvent::FirstSight);
                }
                State::Seen => match self.machine.record_schedule() {
                    Ok(()) => kind = Kind::Record(values.to_vec()),
                    Err(r) => {
                        sh.state = State::Normal(ParamWhy::Unbuildable(r.to_string()));
                        cache.param_note(ParamEvent::Unbuildable);
                    }
                },
                State::Ready if sh.normal_ns.is_empty() || sh.since_sample >= SAMPLE_EVERY => {
                    // a warm normal run is the baseline of the gain: the first one after the recording, then one in a while
                    sh.since_sample = 0;
                    sh.normal_runs += 1;
                    kind = Kind::Normal { sample: true };
                }
                State::Ready => {
                    sh.since_sample += 1;
                    if let Some(l) = &sh.lease {
                        l.touch();
                    }
                    let slot = sh.lease.as_ref().map(|l| l.slot().clone());
                    let machine = &mut self.machine;
                    let gate = slot.and_then(|slot| {
                        slot.with(|ps: &mut ParamSchedule| {
                            ps.instantiate(machine, values).map(|_| ()).and_then(|()| ps.arm(machine, check))
                        })
                    });
                    match gate {
                        Some(Ok(())) => kind = Kind::Served { checked: check },
                        Some(Err(_)) => {
                            sh.refused += 1;
                            cache.param_note(ParamEvent::Refused);
                        }
                        None => {
                            // another platform's step took the room between the check above and now
                            sh.drop_schedule();
                            sh.state = State::Seen;
                        }
                    }
                }
            }
        }
        if matches!(kind, Kind::Served { .. }) {
            // ParamSchedule::arm already began the journal, before any replay execution.
            self.replay.run = Some(ReplayRun {
                key: [0; 32],
                images: self.machine.images.clone(),
                words: Vec::new(),
                data: Vec::new(),
                log: Vec::new(),
                shadow: false,
                param: Some(shape.to_string()),
            });
            if let Some(sh) = self.param.shapes.get_mut(shape) {
                sh.served += 1;
                sh.checked += u64::from(check);
            }
            cache.param_note(ParamEvent::Served);
        }
        self.param.run = Some(ParamRun { shape: shape.to_string(), kind, spent: t0.elapsed() });
        Ok(())
    }

    /// The parametrised replay of this platform: the totals and one row per registered shape.
    pub fn replay_status(&self) -> ParamStatus {
        let mut shapes: Vec<ShapeStatus> = self
            .param
            .shapes
            .iter()
            .map(|(id, sh)| {
                let (normal, replay) = (
                    sh.normal_ns.iter().copied().min().map(|n| n as f64 / 1000.0),
                    (!sh.replay_ns.is_empty()).then(|| median(&sh.replay_ns) / 1000.0),
                );
                let (state, why) = match &sh.state {
                    State::First => ("first", None),
                    State::Seen => ("seen", None),
                    State::Ready => ("ready", None),
                    State::Normal(w) => ("normal", Some(w.clone())),
                };
                ShapeStatus {
                    shape: id.clone(),
                    state,
                    why,
                    served: sh.served,
                    checked: sh.checked,
                    refused: sh.refused,
                    fallback: sh.fallback,
                    normal_runs: sh.normal_runs,
                    gain: normal.zip(replay).map(|(n, r)| n / r),
                    normal_us: normal,
                    replay_us: replay,
                    retained_bytes: if sh.held() { sh.bytes as u64 } else { 0 },
                }
            })
            .collect();
        shapes.sort_by(|a, b| a.shape.cmp(&b.shape));
        ParamStatus {
            enabled: self.param.on,
            min_gain: self.param.min_gain,
            check: self.param.check,
            served: shapes.iter().map(|s| s.served).sum(),
            checked: shapes.iter().map(|s| s.checked).sum(),
            refused: shapes.iter().map(|s| s.refused).sum(),
            fallback: shapes.iter().map(|s| s.fallback).sum(),
            normal_runs: shapes.iter().map(|s| s.normal_runs).sum(),
            shapes,
        }
    }

    /// A fault flip or a scrub made by the host while a run is in flight (`service_flip_sram`, `service_scrub`): a run that is
    /// served from a schedule goes to the normal driver now; a run that records is not kept.
    pub(super) fn param_on_service(&mut self) -> Result<(), PlatformError> {
        if self.phase != Phase::Running {
            return Ok(());
        }
        if self.param_served() {
            return self.param_fallback(ParamWhy::Fault);
        }
        if let Some(run) = &mut self.param.run {
            if let Kind::Record(_) = run.kind {
                run.kind = Kind::Normal { sample: false };
            }
        }
        Ok(())
    }

    /// The done run that was served is checked against the text of the plan again: every word of the text arena in the SRAM.
    pub(super) fn param_text_gate(&mut self) -> Result<(), Gate> {
        let Some(run) = &self.param.run else {
            return Err(Gate::Text("the served run lost its shape ownership".into()));
        };
        let Some(lease) = self.param.shapes.get(&run.shape).and_then(|s| s.lease.as_ref()) else {
            return Err(Gate::Text("the served run lost its shape schedule".into()));
        };
        // a schedule that the cache took while the run was in flight cannot vouch for the text: the run goes to the normal driver
        lease
            .slot()
            .with(|ps: &mut ParamSchedule| ps.text_gate(&self.machine))
            .unwrap_or_else(|| Err(Gate::Text("the cache took the schedule while the run was in flight".into())))
    }

    /// The run in flight was served and cannot stand: put the SRAM back to the start of the run, repeat the host's calls so far on
    /// a new chip on the normal driver and take it over. The result is the one the normal driver gives.
    pub(super) fn param_fallback(&mut self, why: ParamWhy) -> Result<(), PlatformError> {
        let run = self.replay.run.take().expect("a served run");
        let shape = run.param.clone().expect("a run served by a shape");
        self.replay.cache.param_note(ParamEvent::Fallback);
        self.machine.sram.journal_undo();
        let mut twin = self.twin_shell(&run)?;
        std::mem::swap(&mut twin.machine.sram, &mut self.machine.sram);
        let result = Platform::twin_calls(&mut twin, &run);
        self.adopt(twin);
        if let Some(sh) = self.param.shapes.get_mut(&shape) {
            sh.fallback += 1;
            if matches!(why, ParamWhy::Diverged) {
                sh.divergences += 1;
                sh.drop_schedule();
                sh.state =
                    if sh.divergences >= MAX_DIVERGENCES { State::Normal(ParamWhy::Unstable) } else { State::Seen };
            }
        }
        if let Some(r) = &mut self.param.run {
            r.kind = Kind::Normal { sample: false };
        }
        result
    }

    /// After every `advance`: when the run is done, close its journal, keep a recorded schedule and note the timings.
    pub(super) fn param_after_advance(&mut self, took: Duration, ok: bool) {
        let Some(run) = &mut self.param.run else { return };
        run.spent += took;
        if !ok || self.phase != Phase::Done {
            return;
        }
        let run = self.param.run.take().expect("a run");
        self.machine.sram.journal_end();
        let mut spent = run.spent.as_nanos() as u64;
        if let Some(clock) = &mut self.param.timing {
            spent = clock(matches!(run.kind, Kind::Served { .. }));
        }
        let min_gain = self.param.min_gain;
        match run.kind {
            Kind::Normal { sample } => {
                if let (true, Some(sh)) = (sample, self.param.shapes.get_mut(&run.shape)) {
                    if sh.normal_ns.len() >= NORMAL_SAMPLES {
                        sh.normal_ns.remove(0);
                    }
                    sh.normal_ns.push(spent);
                }
            }
            Kind::Record(values) => self.param_record(&run.shape, values),
            Kind::Served { checked: true } => {}
            Kind::Served { checked: false } => {
                let Some(sh) = self.param.shapes.get_mut(&run.shape) else { return };
                if sh.replay_ns.len() >= 2 * PROBE_REPLAYS {
                    sh.replay_ns.remove(0);
                }
                sh.replay_ns.push(spent);
                sh.since_probe += 1;
                if sh.since_probe >= PROBE_REPLAYS && sh.replay_ns.len() >= PROBE_REPLAYS && !sh.normal_ns.is_empty() {
                    sh.since_probe = 0;
                    let normal = sh.normal_ns.iter().copied().min().expect("a sample") as f64;
                    // the decision is on the last probes only, so that a spike in one decision is not in the next
                    let replay = median(&sh.replay_ns[sh.replay_ns.len() - PROBE_REPLAYS..]);
                    let gain = normal / replay;
                    if gain >= min_gain {
                        sh.low_decisions = 0;
                    } else {
                        // one slow decision may be a load spike of the five probes: it takes two in a row, and a shape served by
                        // the normal driver for its gain is measured again after [`REPROBE_AFTER`] steps
                        sh.low_decisions += 1;
                        if sh.low_decisions >= LOW_DECISIONS {
                            sh.state = State::Normal(ParamWhy::Small {
                                gain,
                                normal_us: normal / 1000.0,
                                replay_us: replay / 1000.0,
                            });
                            sh.drop_schedule();
                            sh.small_steps = 0;
                            sh.low_decisions = 0;
                        }
                    }
                }
            }
        }
    }

    /// The recording run is done: parametrise its schedule for the plan, or serve the shape by the normal driver from now on.
    fn param_record(&mut self, shape: &str, values: Vec<i32>) {
        let cache = self.replay.cache.clone();
        let sched = self.machine.take_schedule();
        let Some(sh) = self.param.shapes.get_mut(shape) else { return };
        // the recording run is not a baseline of the measured gain: noting the schedule costs more than a plain run
        let unbuildable = |sh: &mut Shape, why: String| {
            sh.state = State::Normal(ParamWhy::Unbuildable(why));
            cache.param_note(ParamEvent::Unbuildable);
        };
        let Some(sched) = sched else {
            return unbuildable(sh, "the run left no schedule".into());
        };
        let built = ParamSchedule::build(&self.machine, sched, sh.spec.sites.clone(), &sh.spec.text, &values);
        let mut ps = match built {
            Ok(ps) => ps,
            Err(e) => return unbuildable(sh, e.to_string()),
        };
        ps.set_arena(sh.spec.arena.clone());
        if ps.text_gate(&self.machine).is_err() {
            // the program wrote over its text while it was recorded, or the host's text is not the plan's: not a schedule to keep
            sh.state = State::Seen;
            sh.refused += 1;
            cache.param_note(ParamEvent::Refused);
            return;
        }
        let bytes = ps.bytes();
        // one budget for the process: the cache evicts what was used less recently, stored schedules and other platforms' shapes alike
        let Some(lease) = cache.param_reserve(bytes, Box::new(ps)) else {
            sh.state = State::Normal(ParamWhy::TooBig);
            return;
        };
        sh.bytes = bytes;
        sh.lease = Some(lease);
        sh.state = State::Ready;
        sh.replay_ns.clear();
        sh.since_probe = 0;
        sh.low_decisions = 0;
        sh.since_sample = 0;
        cache.param_note(ParamEvent::Recorded);
    }
}
