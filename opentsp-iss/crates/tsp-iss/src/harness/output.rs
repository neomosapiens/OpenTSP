//! The output of a run (`tsp_sim.c` `print_results`, the callbacks): with `--trace` the `T` lines as
//! the events happen, then `E <code|-> <cycle> <count>`, then one `W <slice> <addr> <bytes>` per
//! `D` line. All trace numbers are decimal; `W` and the SRAM numbers of `W` are hex.
use crate::geom::MAX_LANES;
use crate::observe::{Observer, Row};
use ots_isa::Code;
use std::io::{self, Write};

/// The least number of bytes of a `W` line: the width of the format since the first profile; a profile with more lanes prints its own width.
pub const DUMP_MIN_LANES: usize = 320;

/// Append each byte as two lowercase hex digits, optionally prefixed by a space.
#[inline]
pub(super) fn append_hex(out: &mut String, bytes: &[u8], spaced: bool) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.reserve(bytes.len() * if spaced { 3 } else { 2 });
    for &byte in bytes {
        if spaced {
            out.push(' ');
        }
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0xf) as usize]));
    }
}

/// How a [`TraceObserver`] decides whether to write `T` lines: from its `trace` field at run time (the default, what
/// [`TraceObserver::new`] makes), or fixed by the type, so that a run without `--trace` carries no per-event test.
pub const TRACE_DYNAMIC: u8 = 0;
/// A [`TraceObserver`] that never writes `T` lines (`--trace` off): the hooks compile to nothing.
pub const TRACE_OFF: u8 = 1;
/// A [`TraceObserver`] that always writes `T` lines (`--trace` on).
pub const TRACE_ON: u8 = 2;

/// An [`Observer`] that writes the `T` lines (when tracing) and counts the diagnostics.
pub struct TraceObserver<'a, const MODE: u8 = TRACE_DYNAMIC> {
    out: &'a mut dyn Write,
    trace: bool,
    /// `--lint`: write the `T lint` lines (not counted in `diags`).
    pub lint: bool,
    /// Diagnostics reported so far.
    pub diags: u64,
    /// The first diagnostic: its code and cycle.
    pub first: Option<(Code, u64)>,
    /// The first write error, if any (reported when the run is over).
    pub error: Option<io::Error>,
}

impl<'a> TraceObserver<'a, TRACE_DYNAMIC> {
    pub fn new(out: &'a mut dyn Write, trace: bool) -> TraceObserver<'a> {
        TraceObserver { out, trace, lint: false, diags: 0, first: None, error: None }
    }

    /// A writer that continues a run whose earlier part (`diags`, `first`, `error`) a previous
    /// observer on another sink saw: the chips of a lock-step run take turns window by window.
    pub fn resume(
        out: &'a mut dyn Write,
        trace: bool,
        diags: u64,
        first: Option<(Code, u64)>,
        error: Option<io::Error>,
    ) -> TraceObserver<'a> {
        TraceObserver { out, trace, lint: false, diags, first, error }
    }
}

impl<'a> TraceObserver<'a, TRACE_OFF> {
    /// An observer for a run without `--trace`: it counts diagnostics and never writes a `T` line.
    pub fn untraced(out: &'a mut dyn Write) -> TraceObserver<'a, TRACE_OFF> {
        TraceObserver { out, trace: false, lint: false, diags: 0, first: None, error: None }
    }
}

impl<'a> TraceObserver<'a, TRACE_ON> {
    /// An observer for a run with `--trace`.
    pub fn traced(out: &'a mut dyn Write) -> TraceObserver<'a, TRACE_ON> {
        TraceObserver { out, trace: true, lint: false, diags: 0, first: None, error: None }
    }
}

impl<const MODE: u8> TraceObserver<'_, MODE> {
    /// Does this observer write `T` lines? A constant for the two fixed modes.
    #[inline(always)]
    fn tracing(&self) -> bool {
        match MODE {
            TRACE_OFF => false,
            TRACE_ON => true,
            _ => self.trace,
        }
    }

    fn put(&mut self, args: std::fmt::Arguments<'_>) {
        if self.error.is_none() {
            if let Err(e) = self.out.write_fmt(args) {
                self.error = Some(e);
            }
        }
    }

    /// The ` <hex>` value suffix of a `T srf` or `T sram` line, lane 0 first; nothing for an empty value.
    fn value(v: &[u8]) -> String {
        if v.is_empty() {
            return String::new();
        }
        let mut s = String::with_capacity(1 + 2 * v.len());
        s.push(' ');
        append_hex(&mut s, v, false);
        s
    }

    /// Flush the sink.
    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }

    /// Write text after the results (the `Y` and `X` lines of the host interface).
    pub fn tail(&mut self, text: &str) {
        self.put(format_args!("{text}"));
    }

    /// Write the `E` line and the `W` lines of the given words, `DUMP_MIN_LANES` bytes each.
    pub fn finish(&mut self, words: &[(u8, u32, [u8; MAX_LANES])]) {
        self.finish_lanes(words, 0)
    }

    /// Write the `E` line and the `W` lines of the given words: `DUMP_MIN_LANES` bytes each, or `vlanes` when that is more (lanes at
    /// or beyond `vlanes` are zero).
    pub fn finish_lanes(&mut self, words: &[(u8, u32, [u8; MAX_LANES])], vlanes: usize) {
        let lanes = DUMP_MIN_LANES.max(vlanes).min(MAX_LANES);
        let n = self.diags;
        match self.first {
            // A diagnostic cycle is an admitted unsigned timestamp. Absence alone prints -1.
            Some((c, cycle)) => self.put(format_args!("E {} {} {n}\n", c.as_str(), cycle)),
            None => self.put(format_args!("E - -1 {n}\n")),
        }
        for (slice, addr, w) in words {
            let mut line = format!("W {slice:x} {addr:x}");
            append_hex(&mut line, &w[..lanes], true);
            line.push('\n');
            self.put(format_args!("{line}"));
        }
    }
}

impl<const MODE: u8> Observer for TraceObserver<'_, MODE> {
    fn site(&mut self, chip: u16, cycle: u64, q: usize, event: &str, detail: std::fmt::Arguments<'_>) {
        self.put(format_args!("O {chip} {cycle} {q} {event} {detail}\n"));
    }

    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        if self.tracing() {
            self.put(format_args!("T srf {c} {}:{stream} gap {gap}{}\n", row.letter(), Self::value(v)));
        }
    }

    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        if self.tracing() {
            self.put(format_args!("T sram {c} slice {slice} addr {addr}{}\n", Self::value(v)));
        }
    }

    fn diag(&mut self, c: u64, queue: u16, code: Code) {
        if self.first.is_none() {
            self.first = Some((code, c));
        }
        self.diags += 1;
        if self.tracing() {
            self.put(format_args!("T diag {c} queue {queue} {}\n", code.as_str()));
        }
    }

    fn ecc(&mut self, e: &crate::ecc::EccEvent) {
        if self.tracing() {
            let q = if e.queue == 144 { 65535 } else { e.queue };
            self.put(format_args!(
                "T ecc {} queue {q} {} {}\n",
                e.cycle,
                if e.spare {
                    "spare"
                } else if e.unc {
                    "unc"
                } else {
                    "corr"
                },
                e.place()
            ));
        }
    }

    fn scrub(&mut self, c: u64, slice: u8, addr: u32, sl: u8, unc: bool) {
        if self.tracing() {
            self.put(format_args!(
                "T ecc {c} scrub slice {slice} addr {addr} sl {sl}{}\n",
                if unc { " unc" } else { "" }
            ));
        }
    }
    fn mxm_corr(&mut self, c: u64, queue: u16, plane: u8, lane: u16) {
        if self.tracing() {
            self.put(format_args!("T ecc {c} queue {queue} mxm corr plane {plane} lane {lane}\n"));
        }
    }

    fn link_fec(&mut self, c: u64, link: u8, unc: bool, flit: u8) {
        if self.tracing() {
            self.put(format_args!("T ecc {c} link {link} {} flit {flit}\n", if unc { "unc" } else { "corr" }));
        }
    }

    fn host_dma(&mut self, c: u64, what: crate::observe::HostDma) {
        if self.tracing() {
            self.put(format_args!("T dma {c} {}\n", what.word()));
        }
    }

    fn host_irq(&mut self, c: u64, vector: u8) {
        if self.tracing() {
            self.put(format_args!("T irq {c} msix {vector}\n"));
        }
    }

    fn host_csr(&mut self, c: u64, addr: u64, value: u64) {
        if self.tracing() {
            self.put(format_args!("T csr {c} {addr:03x} {value:x}\n"));
        }
    }

    fn host_csr_remote(&mut self, c: u64, chip: u16, addr: u64, value: u64) {
        if self.tracing() {
            self.put(format_args!("T csr {c} {chip} {addr:03x} {value:x}\n"));
        }
    }

    fn host_pkt(&mut self, c: u64, ctrl: bool, link: u8) {
        if self.tracing() {
            self.put(format_args!("T pkt {c} {} {link}\n", if ctrl { "ctrl" } else { "csr" }));
        }
    }

    fn lint(&mut self, c: u64, queue: u16, code: &'static str) {
        if self.lint {
            self.put(format_args!("T lint {c} queue {queue} {code}\n"));
        }
    }
}
