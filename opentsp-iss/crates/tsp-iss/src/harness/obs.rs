//! Observability of a run, all opt-in (`tsp-iss run --run-profile-json` and the `--obs-*` flags): the run profile
//! (where the wall time of a run goes, by phase of the tick, by slice and by opcode) and the structured artifacts of
//! a run (diagnostics, utilization, events, probes). The machine pays nothing when none is asked for: [`Recorder`]
//! is an [`Observer`] wrapper that is only put in front of the run's observer when a flag is given, and the hooks
//! the driver calls for it (`profiling`, `phase`, `exec_begin`, `exec_end`) are compiled out for every other
//! observer. None of it touches the trace on stdout: the artifacts are separate files.
use crate::geom::MAX_LINKS;
use crate::observe::{Hold, Observer, Phase, Row};
use ots_isa::{queue_name, Code, ENTRIES};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::time::Instant;

/// The functional units of the utilization report, by slice name of the catalogue.
pub const UNITS: [&str; 6] = ["CU", "MEM", "VXM", "MXM", "SXM", "C2C"];

/// Most events written to `events.jsonl` (the rest is counted and the file marked truncated), and most per probe window.
pub const MAX_EVENTS: u64 = 2_000_000;
/// Most diagnostics listed in `diagnostics.json` (all are counted).
pub const MAX_DIAGNOSTICS: usize = 1_000_000;
/// Default width of a utilization timeline bin, in cycles.
pub const DEFAULT_BIN: u64 = 1024;

/// A probe of `--obs-probe` (`type@cycle` or `type@begin-end`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// The stream-register writes in flight at the cycle.
    SrfSnapshot(u64),
    /// The cumulative utilization up to the cycle, and what executes in it.
    Utilization(u64),
    /// The events of the cycles `begin..=end`.
    EventWindow(u64, u64),
}

impl Probe {
    /// `srf_snapshot@C`, `utilization@C` or `event_window@A-B` (decimal).
    pub fn parse(spec: &str) -> Result<Probe, String> {
        let bad = || {
            format!("--obs-probe {spec:?}: expected srf_snapshot@CYCLE, utilization@CYCLE or event_window@BEGIN-END")
        };
        let (kind, at) = spec.split_once('@').ok_or_else(bad)?;
        let num = |s: &str| -> Result<u64, String> {
            if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            s.parse().map_err(|_| bad())
        };
        match kind {
            "srf_snapshot" => Ok(Probe::SrfSnapshot(num(at)?)),
            "utilization" => Ok(Probe::Utilization(num(at)?)),
            "event_window" => {
                let (a, b) = at.split_once('-').ok_or_else(bad)?;
                let (a, b) = (num(a)?, num(b)?);
                if a > b {
                    return Err(format!("--obs-probe {spec:?}: the window ends before it begins"));
                }
                Ok(Probe::EventWindow(a, b))
            }
            _ => Err(bad()),
        }
    }
}

/// What to record, from the flags of `run`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsConfig {
    /// Summary counters without enabling unrequested artifact families.
    pub summary: bool,
    /// `--run-profile-json`: the timers of the run profile.
    pub profile: bool,
    /// `--obs-diagnostics`: every diagnostic with its cycle and queue.
    pub diagnostics: bool,
    /// `--obs-utilization`: executions and active cycles per unit and queue, and a timeline.
    pub utilization: bool,
    /// `--obs-events`: the event stream.
    pub events: bool,
    /// `--obs-probe`, `--srf-snapshot`.
    pub probes: Vec<Probe>,
    /// `--obs-bin N`: cycles per utilization timeline bin.
    pub bin: u64,
}

impl Default for ObsConfig {
    fn default() -> ObsConfig {
        ObsConfig {
            summary: false,
            profile: false,
            diagnostics: false,
            utilization: false,
            events: false,
            probes: Vec::new(),
            bin: DEFAULT_BIN,
        }
    }
}

impl ObsConfig {
    /// Whether any recording is on (the observer wrapper is needed).
    pub fn any(&self) -> bool {
        self.summary || self.profile || self.diagnostics || self.utilization || self.events || !self.probes.is_empty()
    }

    /// Whether the driver's phase and execution hooks are needed.
    fn hooks(&self) -> bool {
        self.summary
            || self.profile
            || self.utilization
            || self.events
            || self.probes.iter().any(|p| matches!(p, Probe::Utilization(_) | Probe::EventWindow(..)))
    }
}

/// The run profile's counters of one chip. Time is read once per marker, so the figures carry the cost of the
/// reads: [`clock`] measures it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prof {
    cur: Phase,
    saved: Phase,
    last: Instant,
    entry: usize,
    /// Nanoseconds per [`Phase`] (in [`Phase::ALL`] order).
    pub phase_ns: [u64; Phase::ALL.len()],
    /// Executions and nanoseconds per catalogue entry.
    pub ops: Vec<[u64; 2]>,
}

impl Prof {
    fn new() -> Prof {
        Prof {
            cur: Phase::Event,
            saved: Phase::Event,
            last: Instant::now(),
            entry: 0,
            phase_ns: [0; Phase::ALL.len()],
            ops: vec![[0; 2]; ENTRIES.len()],
        }
    }

    /// Charge the time since the last marker to the current phase.
    #[inline]
    fn charge(&mut self, now: Instant) -> u64 {
        let dt = now.duration_since(self.last).as_nanos() as u64;
        self.phase_ns[self.cur as usize] += dt;
        self.last = now;
        dt
    }
}

/// Executions, and cycles with at least one, of a unit or a queue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Busy {
    pub executions: u64,
    pub active_cycles: u64,
    pub first_cycle: Option<u64>,
    pub last_cycle: Option<u64>,
}

impl Busy {
    fn add(&mut self, c: u64) {
        self.executions += 1;
        if self.last_cycle != Some(c) {
            self.active_cycles += 1;
        }
        self.first_cycle.get_or_insert(c);
        self.last_cycle = Some(c);
    }
}

/// `--obs-utilization`: what executed, where and when.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Util {
    pub units: [Busy; UNITS.len()],
    /// By queue index (the control channel is 144).
    pub queues: Vec<Busy>,
    /// Executions per timeline bin, per unit.
    pub bins: Vec<[u64; UNITS.len()]>,
    /// Bins beyond the dense prefix, keyed by their exact absolute bin index.
    overflow_bins: std::collections::BTreeMap<u64, [u64; UNITS.len()]>,
    /// Executions per catalogue entry.
    pub ops: Vec<u64>,
    /// The unit of each catalogue entry.
    unit_of: Vec<u8>,
}

impl Util {
    fn new() -> Util {
        let unit_of = ENTRIES
            .iter()
            .map(|e| UNITS.iter().position(|u| *u == e.slice).expect("every catalogue slice is a unit") as u8)
            .collect();
        Util {
            units: [Busy::default(); UNITS.len()],
            queues: vec![Busy::default(); 145],
            bins: Vec::new(),
            overflow_bins: Default::default(),
            ops: vec![0; ENTRIES.len()],
            unit_of,
        }
    }

    fn record(&mut self, entry: usize, queue: usize, c: u64, bin: u64) {
        let u = self.unit_of[entry] as usize;
        self.ops[entry] += 1;
        self.units[u].add(c);
        self.queues[queue.min(144)].add(c);
        let b = c / bin;
        if b < MAX_EVENTS {
            let b = usize::try_from(b).expect("bounded dense bin index");
            if self.bins.len() <= b {
                self.bins.resize(b + 1, [0; UNITS.len()]);
            }
            self.bins[b][u] += 1;
        } else {
            self.overflow_bins.entry(b).or_default()[u] += 1;
        }
    }
    /// Whether an execution falls beyond the bounded dense prefix.
    pub fn sparse_timeline(&self) -> bool {
        !self.overflow_bins.is_empty()
    }

    /// Number of logical bins, including all implicit empty bins.
    pub fn timeline_bins(&self) -> u128 {
        self.overflow_bins.last_key_value().map_or(self.bins.len() as u128, |(b, _)| u128::from(*b) + 1)
    }

    /// Number of bins that contain an execution.
    pub fn busy_bins(&self) -> usize {
        self.bins.iter().filter(|b| b.iter().any(|n| *n > 0)).count() + self.overflow_bins.len()
    }
}

/// A stream write seen by the SRF snapshot probes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SrfWrite {
    c: u64,
    row: Row,
    stream: u8,
    gap: u8,
    value: Vec<u8>,
}

/// What a recording run counted, per chip: the state a [`Recorder`] writes into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObsState {
    pub cfg: ObsConfig,
    hooks: bool,
    pub prof: Option<Prof>,
    /// Parcels dispatched.
    pub dispatches: u64,
    /// Elements executed.
    pub executions: u64,
    /// `--obs-diagnostics`: cycle, queue and code of each diagnostic (at most [`MAX_DIAGNOSTICS`]), and the count.
    pub diags: Vec<(u64, u16, &'static str)>,
    pub diag_total: u64,
    /// The positions in `diags` of the diagnostics that came from fetched or injected text (`Observer::diag_fetched`), in
    /// ascending order, and how many such diagnostics each code had in all (the list may be cut).
    pub fetched_at: Vec<usize>,
    pub fetched_by_code: std::collections::BTreeMap<&'static str, u64>,
    /// The same for the diagnostics that are data by the instruction that raised them (`Observer::diag_data`).
    pub data_at: Vec<usize>,
    pub data_by_code: std::collections::BTreeMap<&'static str, u64>,
    /// The queues that ended the run parked with undispatched text: queue, the cycle it was parked, the cause.
    pub parked: Vec<(u16, u64, &'static str)>,
    /// The DeskewRecvs that found their vector: queue, the cycle of dispatch, the cycle of release.
    pub deskew_released: Vec<(u16, u64, u64)>,
    /// The outbound links a Send or Transmit executed on, and the links a Receive or DeskewRecv named (a bit per link).
    pub links_sent: u128,
    pub links_received: u128,
    pub util: Option<Util>,
    /// `--obs-events`: the JSON lines, the number of events, and whether some were left out.
    pub events: Vec<u8>,
    pub event_count: u64,
    pub events_truncated: bool,
    /// Per `event_window` probe: its lines, and whether some were left out.
    pub windows: Vec<(u64, u64, Vec<String>, bool)>,
    /// Per `srf_snapshot` / `utilization` probe: its cycle and result once taken.
    pub snapshots: Vec<(Probe, Option<Value>)>,
    srf_ring: VecDeque<SrfWrite>,
    /// The executions of the cycles the `utilization` probes ask about: cycle, queue, entry.
    at_cycle: Vec<(u64, usize, usize)>,
}

impl ObsState {
    pub fn new(cfg: &ObsConfig) -> ObsState {
        ObsState {
            cfg: cfg.clone(),
            hooks: cfg.hooks(),
            prof: cfg.profile.then(Prof::new),
            dispatches: 0,
            executions: 0,
            diags: Vec::new(),
            diag_total: 0,
            fetched_at: Vec::new(),
            fetched_by_code: Default::default(),
            data_at: Vec::new(),
            data_by_code: Default::default(),
            parked: Vec::new(),
            deskew_released: Vec::new(),
            links_sent: 0,
            links_received: 0,
            util: (cfg.utilization || cfg.probes.iter().any(|p| matches!(p, Probe::Utilization(_)))).then(Util::new),
            events: Vec::new(),
            event_count: 0,
            events_truncated: false,
            windows: cfg
                .probes
                .iter()
                .filter_map(|p| match p {
                    Probe::EventWindow(a, b) => Some((*a, *b, Vec::new(), false)),
                    _ => None,
                })
                .collect(),
            snapshots: cfg
                .probes
                .iter()
                .filter(|p| matches!(p, Probe::SrfSnapshot(_) | Probe::Utilization(_)))
                .map(|p| (*p, None))
                .collect(),
            srf_ring: VecDeque::new(),
            at_cycle: Vec::new(),
        }
    }

    /// The run (or the next window of it) starts now: the time since the last marker is not the run's.
    pub fn resume(&mut self) {
        if let Some(p) = &mut self.prof {
            p.cur = Phase::Event;
            p.last = Instant::now();
        }
    }

    /// The run (or the window) is over: the rest goes to the phase it ended in, and the probes still waiting for a
    /// cycle after the last one are taken.
    pub fn suspend(&mut self) {
        if let Some(p) = &mut self.prof {
            p.charge(Instant::now());
        }
    }

    /// The run is over: take the probes that waited for a later cycle.
    pub fn finish(&mut self) {
        self.take_snapshots(u128::from(u64::MAX) + 1);
    }

    /// Whether an event at cycle `c` goes anywhere.
    fn wanted(&self, c: u64) -> bool {
        (self.cfg.events && !self.events_truncated) || self.windows.iter().any(|w| w.0 <= c && c <= w.1)
    }

    /// File an event of cycle `c` (a JSON line) in the event stream and the windows it falls in.
    fn event(&mut self, c: u64, line: impl FnOnce() -> String) {
        if !self.wanted(c) {
            return;
        }
        let line = line();
        if self.cfg.events {
            if self.event_count < MAX_EVENTS {
                self.events.extend_from_slice(line.as_bytes());
                self.events.push(b'\n');
                self.event_count += 1;
            } else {
                self.events_truncated = true;
            }
        }
        for w in &mut self.windows {
            if w.0 <= c && c <= w.1 {
                if (w.2.len() as u64) < MAX_EVENTS {
                    w.2.push(line.clone());
                } else {
                    w.3 = true;
                }
            }
        }
    }

    /// Take the snapshot probes whose cycle lies before `c`: every event of their cycle is in.
    fn take_snapshots(&mut self, c: u128) {
        for i in 0..self.snapshots.len() {
            let (p, done) = (self.snapshots[i].0, self.snapshots[i].1.is_some());
            let at = match p {
                Probe::SrfSnapshot(a) | Probe::Utilization(a) => a,
                Probe::EventWindow(..) => continue,
            };
            if done || u128::from(at) >= c {
                continue;
            }
            let v = match p {
                Probe::SrfSnapshot(a) => self.srf_snapshot(a),
                _ => self.util_snapshot(at),
            };
            self.snapshots[i].1 = Some(v);
        }
    }

    /// The writes in flight at cycle `at`: a value written at cycle `w`, gap `g`, is in register `g + at - w` (`at > w`)
    /// until it falls off after register 44 (`core/streams.sail`).
    fn srf_snapshot(&self, at: u64) -> Value {
        let writes: Vec<Value> = self
            .srf_ring
            .iter()
            .filter(|w| w.c < at && u128::from(w.gap) + u128::from(at - w.c) <= 44)
            .map(|w| {
                json!({
                    "row": w.row.letter().to_string(),
                    "stream": w.stream,
                    "written_cycle": w.c,
                    "written_gap": w.gap,
                    "register": w.gap as u64 + (at - w.c),
                    "value": w.value.iter().map(|b| format!("{b:02x}")).collect::<String>(),
                })
            })
            .collect();
        json!({"type": "srf_snapshot", "cycle": at, "writes": writes})
    }

    fn util_snapshot(&self, at: u64) -> Value {
        let u = self.util.as_ref().expect("a utilization probe records utilization");
        let executing: Vec<Value> = self
            .at_cycle
            .iter()
            .filter(|(c, _, _)| *c == at)
            .map(|(_, q, e)| json!({"queue": q, "name": queue_name(queue_id_of(*q)), "op": ENTRIES[*e].name}))
            .collect();
        json!({"type": "utilization", "cycle": at, "up_to_cycle": units_json(&u.units), "executing": executing})
    }
}

/// The queue id of a queue index (the control channel is index 144, id 0xffff).
fn queue_id_of(q: usize) -> u16 {
    if q >= 144 {
        0xffff
    } else {
        q as u16
    }
}

fn units_json(units: &[Busy; UNITS.len()]) -> Value {
    let m: serde_json::Map<String, Value> = UNITS
        .iter()
        .zip(units)
        .map(|(n, b)| (n.to_string(), json!({"executions": b.executions, "active_cycles": b.active_cycles})))
        .collect();
    Value::Object(m)
}

/// An observer that records into `st` and forwards every event to `inner`.
pub struct Recorder<'a, O: Observer + ?Sized> {
    pub inner: &'a mut O,
    pub st: &'a mut ObsState,
}

impl<O: Observer + ?Sized> Observer for Recorder<'_, O> {
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        let st = &mut *self.st;
        if !st.snapshots.is_empty() {
            st.take_snapshots(u128::from(c));
            // 45 cycles is the longest a value flows
            while st.srf_ring.front().is_some_and(|w| u128::from(w.c) + 45 < u128::from(c)) {
                st.srf_ring.pop_front();
            }
            if st.snapshots.iter().any(|s| matches!(s.0, Probe::SrfSnapshot(_))) {
                st.srf_ring.push_back(SrfWrite { c, row, stream, gap, value: v.to_vec() });
            }
        }
        st.event(c, || {
            format!("{{\"c\":{c},\"kind\":\"srf\",\"row\":\"{}\",\"stream\":{stream},\"gap\":{gap}}}", row.letter())
        });
        self.inner.srf(c, row, stream, gap, v);
    }

    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.st.event(c, || format!("{{\"c\":{c},\"kind\":\"sram\",\"slice\":{slice},\"addr\":{addr}}}"));
        self.inner.sram(c, slice, addr, v);
    }

    fn diag(&mut self, c: u64, queue: u16, code: Code) {
        let st = &mut *self.st;
        st.diag_total += 1;
        if st.cfg.diagnostics && st.diags.len() < MAX_DIAGNOSTICS {
            st.diags.push((c, queue, code.as_str()));
        }
        st.event(c, || format!("{{\"c\":{c},\"kind\":\"diag\",\"queue\":{queue},\"code\":\"{}\"}}", code.as_str()));
        self.inner.diag(c, queue, code);
    }

    fn diag_fetched(&mut self, c: u64, queue: u16, code: Code) {
        let st = &mut *self.st;
        *st.fetched_by_code.entry(code.as_str()).or_default() += 1;
        // the diagnostic just reported is the last of the list unless the list was cut before it
        if st.cfg.diagnostics && st.diag_total == st.diags.len() as u64 && !st.diags.is_empty() {
            st.fetched_at.push(st.diags.len() - 1);
        }
        self.inner.diag_fetched(c, queue, code);
    }

    fn diag_data(&mut self, c: u64, queue: u16, code: Code) {
        let st = &mut *self.st;
        *st.data_by_code.entry(code.as_str()).or_default() += 1;
        if st.cfg.diagnostics && st.diag_total == st.diags.len() as u64 && !st.diags.is_empty() {
            st.data_at.push(st.diags.len() - 1);
        }
        self.inner.diag_data(c, queue, code);
    }

    fn parked_text(&mut self, queue: u16, since: u64, cause: &'static str) {
        self.st.parked.push((queue, since, cause));
        self.inner.parked_text(queue, since, cause);
    }

    fn deskew_released(&mut self, queue: u16, since: u64, c: u64) {
        if self.st.cfg.diagnostics && self.st.deskew_released.len() < MAX_DIAGNOSTICS {
            self.st.deskew_released.push((queue, since, c));
        }
        self.inner.deskew_released(queue, since, c);
    }

    fn link_op(&mut self, link: u8, send: bool) {
        if usize::from(link) < MAX_LINKS {
            *(if send { &mut self.st.links_sent } else { &mut self.st.links_received }) |= 1 << link;
        }
        self.inner.link_op(link, send);
    }

    fn lint(&mut self, c: u64, queue: u16, code: &'static str) {
        self.inner.lint(c, queue, code);
    }

    fn dispatch(&mut self, c: u64, queue: u16) {
        self.st.dispatches += 1;
        self.st.event(c, || format!("{{\"c\":{c},\"kind\":\"dispatch\",\"queue\":{queue}}}"));
        self.inner.dispatch(c, queue);
    }

    fn wait(&mut self, c: u64, queue: u16, why: Hold, since: u64) {
        self.inner.wait(c, queue, why, since);
    }

    fn fetch(&mut self, c: u64, queue: u16, bytes: u32) {
        self.st.event(c, || format!("{{\"c\":{c},\"kind\":\"fetch\",\"queue\":{queue},\"bytes\":{bytes}}}"));
        self.inner.fetch(c, queue, bytes);
    }

    fn host_dma(&mut self, c: u64, what: crate::observe::HostDma) {
        self.inner.host_dma(c, what);
    }

    fn host_irq(&mut self, c: u64, vector: u8) {
        self.inner.host_irq(c, vector);
    }

    fn host_csr(&mut self, c: u64, addr: u64, value: u64) {
        self.inner.host_csr(c, addr, value);
    }

    fn host_csr_remote(&mut self, c: u64, chip: u16, addr: u64, value: u64) {
        self.inner.host_csr_remote(c, chip, addr, value);
    }

    fn ecc(&mut self, e: &crate::ecc::EccEvent) {
        self.inner.ecc(e);
    }
    fn link_fec(&mut self, c: u64, link: u8, unc: bool, flit: u8) {
        self.inner.link_fec(c, link, unc, flit);
    }
    fn mxm_corr(&mut self, c: u64, queue: u16, plane: u8, lane: u16) {
        self.inner.mxm_corr(c, queue, plane, lane);
    }
    fn scrub(&mut self, c: u64, slice: u8, addr: u32, sl: u8, unc: bool) {
        self.inner.scrub(c, slice, addr, sl, unc);
    }

    fn host_pkt(&mut self, c: u64, ctrl: bool, link: u8) {
        self.inner.host_pkt(c, ctrl, link);
    }

    fn site(&mut self, chip: u16, cycle: u64, q: usize, event: &str, detail: std::fmt::Arguments<'_>) {
        self.inner.site(chip, cycle, q, event, detail);
    }
    fn tracks_waits(&self) -> bool {
        self.inner.tracks_waits()
    }

    fn profiling(&self) -> bool {
        self.st.hooks
    }

    fn phase(&mut self, p: Phase) {
        if let Some(pr) = &mut self.st.prof {
            pr.charge(Instant::now());
            pr.cur = p;
        }
    }

    fn exec_begin(&mut self, entry: usize, queue: usize, c: u64) {
        let st = &mut *self.st;
        st.executions += 1;
        if let Some(pr) = &mut st.prof {
            pr.charge(Instant::now());
            pr.saved = pr.cur;
            pr.cur = Phase::Execute;
            pr.entry = entry;
        }
        if st.snapshots.iter().any(|s| matches!(s.0, Probe::Utilization(_))) {
            st.take_snapshots(u128::from(c));
            if st.snapshots.iter().any(|s| s.0 == Probe::Utilization(c)) {
                st.at_cycle.push((c, queue, entry));
            }
        }
        if let Some(u) = &mut st.util {
            u.record(entry, queue, c, st.cfg.bin.max(1));
        }
        st.event(c, || format!("{{\"c\":{c},\"kind\":\"exec\",\"queue\":{queue},\"op\":\"{}\"}}", ENTRIES[entry].name));
    }

    fn exec_end(&mut self) {
        if let Some(pr) = &mut self.st.prof {
            let dt = pr.charge(Instant::now());
            let o = &mut pr.ops[pr.entry];
            o[0] += 1;
            o[1] += dt;
            pr.cur = pr.saved;
        }
    }
}

/// The clock the profile reads: the smallest step it shows and the cost of one read, in nanoseconds.
pub fn clock() -> (u64, f64) {
    let mut step = u64::MAX;
    let mut prev = Instant::now();
    for _ in 0..20_000 {
        let now = Instant::now();
        let d = now.duration_since(prev).as_nanos() as u64;
        if d > 0 {
            step = step.min(d);
        }
        prev = now;
    }
    const N: u32 = 200_000;
    let t0 = Instant::now();
    let mut keep = t0;
    for _ in 0..N {
        keep = Instant::now();
    }
    let cost = keep.duration_since(t0).as_nanos() as f64 / N as f64;
    (step, cost)
}

/// What the run profile says of one chip, with the wall time of the part of the run it covers.
pub struct ChipProfile<'a> {
    pub st: &'a ObsState,
    /// Building the machine and loading its images.
    pub setup_ns: u64,
    /// The run loop.
    pub run_ns: u64,
    pub ticks: u64,
    pub last_cycle: Option<u64>,
}

/// The run profile JSON of a run: the schema is [`PROFILE_SCHEMA`].
pub fn profile_json(profile_name: &str, wall_ns: u64, load: Value, chips: &[ChipProfile]) -> Value {
    let (step, cost) = clock();
    let per_chip: Vec<Value> = chips
        .iter()
        .map(|c| {
            let p = c.st.prof.as_ref().expect("the run profile is on");
            let phases: serde_json::Map<String, Value> =
                Phase::ALL.iter().map(|ph| (ph.name().to_string(), json!(p.phase_ns[*ph as usize]))).collect();
            let profiled: u64 = p.phase_ns.iter().sum();
            let mut slices: std::collections::BTreeMap<&str, [u64; 2]> = Default::default();
            let mut opcodes = serde_json::Map::new();
            let mut order: Vec<usize> = (0..ENTRIES.len()).filter(|&i| p.ops[i][0] > 0).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(p.ops[i][1]));
            for i in order {
                let (e, o) = (&ENTRIES[i], p.ops[i]);
                let s = slices.entry(e.slice).or_default();
                s[0] += o[0];
                s[1] += o[1];
                opcodes.insert(e.name.to_string(), json!({"slice": e.slice, "count": o[0], "ns": o[1]}));
            }
            let slices: serde_json::Map<String, Value> =
                slices.into_iter().map(|(k, v)| (k.to_string(), json!({"count": v[0], "ns": v[1]}))).collect();
            let cycles = c.last_cycle.map_or(0, |l| u128::from(l) + 1);
            json!({
                "setup_ns": c.setup_ns,
                "run_ns": c.run_ns,
                "ticks": c.ticks,
                "active_cycles": c.ticks,
                "idle_cycles": cycles.saturating_sub(u128::from(c.ticks)),
                "last_cycle": c.last_cycle,
                "dispatches": c.st.dispatches,
                "executions": c.st.executions,
                "phases_ns": phases,
                "profiled_ns": profiled,
                "unprofiled_ns": c.run_ns.saturating_sub(profiled),
                "slices": slices,
                "opcodes": opcodes,
            })
        })
        .collect();
    json!({
        "schema": PROFILE_SCHEMA,
        "profile": profile_name,
        "wall_ns": wall_ns,
        "clock": {"step_ns": step, "read_ns": cost},
        "load": load,
        "chips": per_chip,
    })
}

/// The version of the run profile document: raised when a key changes meaning or goes away.
pub const PROFILE_SCHEMA: u64 = 1;

/// The version of the `--obs-*` artifacts: raised when a key changes meaning or goes away.
pub const OBS_SCHEMA: u64 = 1;

/// `diagnostics.json`: every diagnostic of the chip, in report order.
pub fn diagnostics_json(chip: usize, st: &ObsState) -> Value {
    let list: Vec<Value> = st
        .diags
        .iter()
        .enumerate()
        .map(|(i, (c, q, code))| {
            let mut d = json!({"cycle": c, "queue": q, "queue_name": queue_name(*q), "code": code});
            if st.fetched_at.binary_search(&i).is_ok() {
                d["fetched"] = json!(true);
            }
            if st.data_at.binary_search(&i).is_ok() {
                d["data"] = json!(true);
            }
            d
        })
        .collect();
    let mut by_code: std::collections::BTreeMap<&str, u64> = Default::default();
    for d in &st.diags {
        *by_code.entry(d.2).or_default() += 1;
    }
    let mut doc = json!({
        "chip": chip,
        "total": st.diag_total,
        "listed": st.diags.len(),
        "truncated": st.diag_total > st.diags.len() as u64,
        "by_code": by_code,
        "diagnostics": list,
    });
    // present only when the run had such diagnostics or queues
    if !st.fetched_by_code.is_empty() {
        doc["by_code_fetched"] = json!(st.fetched_by_code);
    }
    if !st.data_by_code.is_empty() {
        doc["by_code_data"] = json!(st.data_by_code);
    }
    if !st.parked.is_empty() {
        let parked: Vec<Value> = st
            .parked
            .iter()
            .map(|(q, since, why)| json!({"queue": q, "queue_name": queue_name(*q), "since": since, "cause": why}))
            .collect();
        doc["parked"] = json!(parked);
    }
    // the timing a link latency can change: where each DeskewRecv was released, and the links the chip sent on
    if !st.deskew_released.is_empty() {
        doc["deskew_released"] = json!(st.deskew_released.iter().map(|(q, s, c)| json!([q, s, c])).collect::<Vec<_>>());
    }
    for (key, bits) in [("links_sent", st.links_sent), ("links_received", st.links_received)] {
        if bits != 0 {
            doc[key] = json!((0..MAX_LINKS).filter(|l| bits >> l & 1 == 1).collect::<Vec<_>>());
        }
    }
    doc
}

/// `utilization.json` of one chip: executions and active cycles per unit and queue, and a timeline of executions per
/// bin. `cycles` is the number of cycles simulated (the last cycle ticked plus one).
pub fn utilization_json(chip: usize, st: &ObsState, cycles: u128) -> Value {
    utilization_document(chip, st, cycles, false)
}

/// The explicit sparse wire format, also used for every chip in a mixed sparse envelope.
pub fn sparse_utilization_json(chip: usize, st: &ObsState, cycles: u128) -> Value {
    utilization_document(chip, st, cycles, true)
}

fn utilization_document(chip: usize, st: &ObsState, cycles: u128, force_sparse: bool) -> Value {
    let u = st.util.as_ref().expect("utilization is on");
    let busy = |b: &Busy| {
        json!({
            "executions": b.executions,
            "active_cycles": b.active_cycles,
            "utilization": if cycles == 0 { 0.0 } else { b.active_cycles as f64 / cycles as f64 },
            "first_cycle": b.first_cycle,
            "last_cycle": b.last_cycle,
        })
    };
    let units: serde_json::Map<String, Value> =
        UNITS.iter().zip(&u.units).map(|(n, b)| (n.to_string(), busy(b))).collect();
    let queues: Vec<Value> = u
        .queues
        .iter()
        .enumerate()
        .filter(|(_, b)| b.executions > 0)
        .map(|(q, b)| {
            let mut v = busy(b);
            v["queue"] = json!(queue_id_of(q));
            v["name"] = json!(queue_name(queue_id_of(q)));
            v
        })
        .collect();
    let bin = st.cfg.bin.max(1);
    let sparse = force_sparse || u.sparse_timeline();
    let row = |index: u64, counts: &[u64; UNITS.len()]| {
        let m: serde_json::Map<String, Value> =
            UNITS.iter().zip(counts).filter(|(_, n)| **n > 0).map(|(n, x)| (n.to_string(), json!(x))).collect();
        let mut value = json!({"start": index * bin, "executions": m});
        if sparse {
            value["index"] = json!(index);
        }
        value
    };
    let mut timeline: Vec<Value> = u
        .bins
        .iter()
        .enumerate()
        .filter(|(_, b)| !sparse || b.iter().any(|n| *n > 0))
        .map(|(i, b)| row(i as u64, b))
        .collect();
    timeline.extend(u.overflow_bins.iter().map(|(i, b)| row(*i, b)));
    let mut doc = json!({
        "chip": chip,
        "cycles": cycles,
        "bin_cycles": bin,
        "units": units,
        "queues": queues,
        "timeline": timeline,
    });
    if sparse {
        let timeline = doc.as_object_mut().expect("utilization object").remove("timeline").expect("timeline rows");
        doc["timeline_nonzero"] = timeline;
        doc["timeline_encoding"] = json!("absolute-nonzero-v1");
        doc["timeline_sparse"] = json!(true);
        doc["timeline_bins"] = json!(u.timeline_bins());
        doc["timeline_omitted_bins_are_zero"] = json!(true);
    }
    doc
}

/// `probes.json` of one chip: the probes in the order of the flags.
pub fn probes_json(chip: usize, st: &ObsState, cfg: &ObsConfig) -> Value {
    let (mut snaps, mut wins) = (st.snapshots.iter(), st.windows.iter());
    let results: Vec<Value> = cfg
        .probes
        .iter()
        .map(|p| match p {
            Probe::EventWindow(a, b) => {
                let w = wins.next().expect("one window per probe");
                let events: Vec<Value> = w.2.iter().map(|l| serde_json::from_str(l).expect("event line")).collect();
                json!({"type": "event_window", "begin": a, "end": b, "truncated": w.3, "events": events})
            }
            _ => {
                let s = snaps.next().expect("one snapshot per probe");
                s.1.clone()
                    .unwrap_or_else(|| json!({"type": "unavailable", "reason": "the run ended before the cycle"}))
            }
        })
        .collect();
    json!({"chip": chip, "probes": results})
}
