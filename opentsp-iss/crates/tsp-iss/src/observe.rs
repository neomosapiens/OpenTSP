//! The model's callbacks (`core/callbacks.sail`) as a trait: what a run reports.
//!
//! Every method has a no-op default, and the machine is generic over the observer, so a run
//! without one pays nothing.
use ots_isa::Code;

/// A stream row: E flows east, W flows west.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    East,
    West,
}

impl Row {
    /// The trace letter, `E` or `W`.
    pub fn letter(self) -> char {
        match self {
            Row::East => 'E',
            Row::West => 'W',
        }
    }
}

/// Why a queue's head parcel waits (`microarchitecture.md` §3, "Dispatch or wait").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hold {
    Parked,
    Busy,
    Path,
    Loop,
    Repeat,
    Configuration,
    IncompleteText,
}

/// What happened to a vector at the PCIe interface (`T dma`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostDma {
    /// The host's DMA engine pushed a vector into the ingress FIFO.
    In,
    /// The ingress FIFO was full: the vector waits in the DMA engine until a PciReceive frees a slot (nothing is dropped).
    Wait,
    /// PciSend gave the host a vector.
    Out,
}

impl HostDma {
    pub fn word(self) -> &'static str {
        match self {
            HostDma::In => "in",
            HostDma::Wait => "wait",
            HostDma::Out => "out",
        }
    }
}

/// The parts of one tick, for the run profile (`Observer::phase`): the time from one marker to the next is charged
/// to the phase the first marker names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Between ticks: the model's next event, the link arrivals, the run loop.
    Event,
    /// Fetched text into the IQs, and each queue's head, hold and wake-cycle work.
    Driver,
    /// The calendar of wake cycles.
    Calendar,
    /// Barrier retirement and the commits due.
    Commit,
    /// Instructions executing (the execute ring, and the elements a dispatch executes at once).
    Execute,
    /// The control channel.
    Control,
    /// The data queues' dispatch (the AGT, Repeat and parcel dispatch, without the execution).
    Dispatch,
    /// End of cycle: the staged MXM and SXM changes.
    Flush,
}

impl Phase {
    pub const ALL: [Phase; 8] = [
        Phase::Event,
        Phase::Driver,
        Phase::Calendar,
        Phase::Commit,
        Phase::Execute,
        Phase::Control,
        Phase::Dispatch,
        Phase::Flush,
    ];

    /// The name in the run profile.
    pub fn name(self) -> &'static str {
        match self {
            Phase::Event => "event",
            Phase::Driver => "driver",
            Phase::Calendar => "calendar",
            Phase::Commit => "commit",
            Phase::Execute => "execute",
            Phase::Control => "control",
            Phase::Dispatch => "dispatch",
            Phase::Flush => "flush",
        }
    }
}

/// What a run reports. `srf` and `sram` receive the committed value (`vlanes` bytes, lane 0 first,
/// after any merge). A library user reads the values like this:
///
/// ```
/// use tsp_iss::{Machine, NoLinks, Observer, Row};
///
/// #[derive(Default)]
/// struct Writes {
///     srf: Vec<(u64, Row, u8, Vec<u8>)>,
///     sram: Vec<(u64, u8, u32, Vec<u8>)>,
/// }
/// impl Observer for Writes {
///     fn srf(&mut self, c: u64, row: Row, stream: u8, _gap: u8, v: &[u8]) {
///         self.srf.push((c, row, stream, v.to_vec()));
///     }
///     fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
///         self.sram.push((c, slice, addr, v.to_vec()));
///     }
/// }
///
/// let profile = ots_profile::Profile::load("mini").unwrap();
/// let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
/// m.load_image(0, &[0x1a, 0x1040_0003, 0]).unwrap(); // a MEM Write of slice 0, address 3, from E:0
/// m.load_image(4, &[0x14, 0x1000_0001, 0]).unwrap(); // a MEM Read of slice 4, address 1, onto E:0
/// m.preload(4, 1, &[9; 16]).unwrap();
/// let mut w = Writes::default();
/// m.run(1000, &mut w).unwrap();
/// assert_eq!(w.srf, [(25, Row::East, 0, vec![9; 16])]);
/// assert_eq!(w.sram, [(26, 0, 3, vec![9; 16])]);
/// ```
pub trait Observer {
    /// One optional Sail site observation, delivered at the causal point in the tick.
    /// The detail is formatted as specified in the porting guide.
    fn site(&mut self, _chip: u16, _cycle: u64, _queue: usize, _event: &str, _detail: std::fmt::Arguments<'_>) {}

    /// A stream write committed at cycle `c` (`T srf`): row, stream, gap (`srf_write_callback`);
    /// `v` is the committed vector. // model: core/callbacks.sail srf_write_callback
    fn srf(&mut self, _c: u64, _row: Row, _stream: u8, _gap: u8, _v: &[u8]) {}
    /// An SRAM word written at cycle `c` (`T sram`; `sram_write_callback`); `v` is the word.
    // model: core/callbacks.sail sram_write_callback
    fn sram(&mut self, _c: u64, _slice: u8, _addr: u32, _v: &[u8]) {}
    /// A diagnostic reported at cycle `c` for `queue` (`T diag`; 0xffff is the control channel).
    // model: core/callbacks.sail diag_callback
    fn diag(&mut self, _c: u64, _queue: u16, _code: Code) {}
    /// The diagnostic just reported by [`Observer::diag`] came from instruction text that was not in the object's own
    /// image: a unit fetched from SRAM or injected by the host. Its class is `data` whatever the code's class in the
    /// table (`microarchitecture.md` section 8), because the units decode to whatever the fetched words are.
    fn diag_fetched(&mut self, _c: u64, _queue: u16, _code: Code) {}
    /// The diagnostic just reported by [`Observer::diag`] is of class `data` because of the instruction that raised it: an
    /// OPR-001 of a Gather or Scatter tile address or of an `sxselect` select (`microarchitecture.md` section 8). The same code
    /// from another instruction is of class `schedule`.
    fn diag_data(&mut self, _c: u64, _queue: u16, _code: Code) {}
    /// At the end of the run, queue `queue` still holds text that never dispatched because the queue is parked for good
    /// (a HostWait the host never releases, a DeskewRecv whose vector never arrives, a Sync no Notify releases);
    /// `since` is the cycle it was parked. The end-of-run report raises no diagnostic for such a queue.
    fn parked_text(&mut self, _queue: u16, _since: u64, _cause: &'static str) {}
    /// A DeskewRecv of `queue`, dispatched at `since`, found its vector at the epoch boundary `c` and became a Receive. The
    /// release cycle is the one place where a program's timing depends on a link latency.
    fn deskew_released(&mut self, _queue: u16, _since: u64, _c: u64) {}
    /// The chip executed an instruction that names its link `link`: a Send or Transmit (`send`) on the outbound link, or a
    /// Receive or DeskewRecv on the inbound one.
    fn link_op(&mut self, _link: u8, _send: bool) {}
    fn horizon_event(&mut self, _at: u64) {}
    /// A simulator lint at cycle `c` for `queue` (`T lint` with `--lint`; `lint_callback`): not a
    /// diagnostic, never counted in `E`, never changes the machine (`core/lints.sail`).
    // model: core/callbacks.sail lint_callback
    fn lint(&mut self, _c: u64, _queue: u16, _code: &'static str) {}
    /// A vector crossed the PCIe interface at cycle `c` (`T dma`; the host's DMA engine, PciSend).
    // model: c_emulator/tsp_sim.c host_process, host_send
    fn host_dma(&mut self, _c: u64, _what: HostDma) {}
    /// MSI-X vector `vector` reached the host at cycle `c` (`T irq`; HostNotify, not masked by `IRQ_MASK`).
    // model: core/callbacks.sail host_irq_callback
    fn host_irq(&mut self, _c: u64, _vector: u8) {}
    /// The host read CSR `addr` and got `value` at cycle `c` (`T csr`).
    // model: c_emulator/tsp_sim.c host_process
    fn host_csr(&mut self, _c: u64, _addr: u64, _value: u64) {}
    /// The reply of a read of chip `chip`'s CSR `addr` (value `value`) reached the host chip at cycle `c`
    /// (`T csr <cycle> <chip> <addr> <value>`).
    // model: c_emulator/tsp_sim.c ctl_process
    fn host_csr_remote(&mut self, _c: u64, _chip: u16, _addr: u64, _value: u64) {}
    /// The control unit took in a csr (`ctrl` false) or ctrl packet that arrived on `link` at cycle `c` (`T pkt`).
    // model: c_emulator/tsp_sim.c ctl_process
    fn host_pkt(&mut self, _c: u64, _ctrl: bool, _link: u8) {}
    /// An ECC correction or uncorrectable superlane (`T ecc`; `ecc_event_callback`): not part of the trace digest.
    // model: core/callbacks.sail ecc_event_callback
    fn ecc(&mut self, _e: &crate::ecc::EccEvent) {}
    /// The link FEC corrected (`unc` false) or found uncorrectable a flit of a vector that arrived on `link` at cycle `c`
    /// (`T ecc <c> link <link> corr|unc flit <flit>`; `model_link_fec`): not part of the trace digest.
    // model: c_emulator/tsp_sim.c process_arrivals
    fn link_fec(&mut self, _c: u64, _link: u8, _unc: bool, _flit: u8) {}
    /// The sum check of an ABC corrected the one wrong lane of its product (`T ecc <c> queue <q> mxm corr plane <p> lane <l>`;
    /// `mxm_corr_callback`): not part of the trace digest.
    // model: core/ecc.sail mxm_corr_callback
    fn mxm_corr(&mut self, _c: u64, _queue: u16, _plane: u8, _lane: u16) {}
    /// The patrol scrubber corrected (`unc` false) or found uncorrectable a word of superlane `sl` at `addr` of MEM slice `slice`
    /// (`T ecc <c> scrub slice <s> addr <a> sl <g>`; `scrub_event_callback`): not part of the trace digest.
    // model: core/callbacks.sail scrub_event_callback
    fn scrub(&mut self, _c: u64, _slice: u8, _addr: u32, _sl: u8, _unc: bool) {}
    /// Hooks for a debugger and statistics (no trace line). `dispatch`: the head parcel of `queue`
    /// dispatched at cycle `c`.
    fn dispatch(&mut self, _c: u64, _queue: u16) {}
    /// The head parcel of `queue` could not dispatch at cycle `c` (a ticked cycle) for reason `why`:
    /// it is due but held, or its text is incomplete. `since` (at most `c`) is the first cycle at
    /// which it was due, the cycles between not being ticked when nothing else happened. Called only
    /// when [`Observer::tracks_waits`].
    fn wait(&mut self, _c: u64, _queue: u16, _why: Hold, _since: u64) {}
    /// An `Ifetch` or `IfetchM` of `queue` executed at cycle `c`: `bytes` of text start to arrive.
    fn fetch(&mut self, _c: u64, _queue: u16, _bytes: u32) {}
    /// Whether the driver should work out why a parcel waits and call [`Observer::wait`] (a run
    /// without statistics skips the work).
    fn tracks_waits(&self) -> bool {
        false
    }
    /// Whether the driver should call [`Observer::phase`], [`Observer::exec_begin`] and [`Observer::exec_end`] (the run
    /// profile and the occupancy). A run without them pays nothing: the observer types that do not want them return
    /// `false` from an inlined default, so the calls are not compiled in.
    fn profiling(&self) -> bool {
        false
    }
    /// A tick phase starts (called only when [`Observer::profiling`]).
    fn phase(&mut self, _p: Phase) {}
    /// The element of catalogue entry `entry` (an `EntryId` index) of `queue` starts to execute in cycle `c` (called
    /// only when [`Observer::profiling`]); [`Observer::exec_end`] follows it.
    fn exec_begin(&mut self, _entry: usize, _queue: usize, _c: u64) {}
    fn exec_end(&mut self) {}
    /// The element about to execute, just before [`Observer::exec_begin`] (called only when [`Observer::profiling`]);
    /// the schedule recording of `driver::replay` keeps it. Only the recording observer implements it.
    #[doc(hidden)]
    fn exec_element(&mut self, _e: &crate::driver::execute::ExecElement) {}
}

/// The observer that ignores everything.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoObserver;
impl Observer for NoObserver {}
