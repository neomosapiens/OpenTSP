//! The program on stdin (`tsp_sim.c` `read_input`; `porting-guide.md` §2.3):
//!
//! ```text
//! S <slice> <addr> <b0> .. <bn>   preload one SRAM word (hex bytes, lane 0 first)
//! Q <queue> <u0> <u1> ...         the initial IQ image of a queue: 32-bit text units
//! D <slice> <addr>                dump this SRAM word after the run
//! C <chip>                        the following lines are for chip <chip> (--chips)
//! R <csr>                         print this CSR after the run (an `X` line)
//! H start <cycle>                 the queues start dispatching at <cycle> (default 0)
//!                                 (every chip has its own host interface: the H and R lines after a `C <chip>` line act on
//!                                 that chip, a line without one on chip 0)
//! H dma <cycle> <b0> .. <bn>      the host's DMA engine pushes a vector (the ingress FIFO)
//! H db <cycle> <i>                the host rings doorbell i (CSR 0x300 + i)
//! H csr <cycle> r|w <addr> [<v>]  the host reads or writes a CSR (a read prints `T csr`)
//! P ctrl <cycle> <link>           the control unit of the chip sends a ctrl packet on <link>
//! P csr <cycle> <link> r|w <addr> [<v>]  the control unit sends a csr packet on <link>: the chip at its far end reads or
//!                                 writes its CSR and answers a read (host.remote_csr)
//! H inject <cycle> <queue> <u0> <u1> ...  the host pushes instruction text into a queue's IQ
//! F sram <cycle> <slice> <addr> <superlane> <bit> [t|p]   ECC fault injection: a bit of a stored SRAM word flips
//! F srf <cycle> e|w <stream> <gap> <superlane> <bit> [t|p]  the same for the value in a stream register
//! F link <cycle> <link> <flit> corr|uncorr [t|p]  link FEC fault: a flit of the vector the chip sends on <link> in <cycle>
//!                                 arrives with an error the FEC corrects (corr) or only detects (uncorr)
//! ```
//!
//! Every number is hex (`strtoull` base 16, so `0x` is accepted). Leading spaces and tabs are
//! skipped; blank lines and lines starting with `#` are ignored; a tag is followed by a space or
//! tab; fields are separated by spaces, tabs or CR. Anything else is an input error naming the
//! line.
use super::cnum::strtoull16;
use crate::geom::{Geom, MAX_LANES, SLICES};
use crate::host::{HostKind, PktKind};
use crate::links::{link_flits, LinkFault};
use ots_format::text::Images;
use ots_isa::PADDING;
use std::fmt;

/// An input error: `input line <n>: <reason>`, exit 2.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputError {
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "input line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for InputError {}

/// What the program says about one SRAM word before the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preload {
    pub slice: u8,
    pub addr: u32,
    /// At most `MAX_LANES` bytes; the machine drops those beyond `vlanes`.
    pub bytes: Vec<u8>,
}

/// One word to dump after the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dump {
    pub chip: u32,
    pub slice: u8,
    pub addr: u32,
}

/// What one chip is given.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChipInput {
    /// The initial IQ images, padding dropped.
    pub images: Images,
    pub preloads: Vec<Preload>,
}

/// One `H` line (not `H start`): the host's action `kind` at `cycle`; `chip` is the chip the line acts on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostLine {
    pub chip: u32,
    pub cycle: u64,
    pub kind: HostKind,
    /// The number of the line among the packets of the input (`H` and `P` lines in input order).
    pub id: u64,
    /// An `F` line marked `p` (persistent): `--on-irq replay` applies it again in the replay run.
    pub persist: bool,
}

/// One `P` line: the control unit of `chip` sends a ctrl or csr packet (`kind`) on `link`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PktLine {
    pub chip: u32,
    pub cycle: u64,
    pub link: u8,
    pub kind: PktKind,
    pub id: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    /// The wiring of `--topology`; none: the wiring of `--chips K`.
    pub topo: Option<std::sync::Arc<super::topology::Topology>>,
    pub chips: Vec<ChipInput>,
    /// Every `D` line, in input order, with its chip.
    pub dumps: Vec<Dump>,
    /// Every `R` line: the chip and the CSR to print after the run, in input order.
    pub rcsr: Vec<(u32, u64)>,
    /// Every `H` line except `H start`, in input order.
    pub host: Vec<HostLine>,
    /// `H start` per chip (0: the default).
    pub starts: Vec<u64>,
    /// Every `P` line, in input order.
    pub pkts: Vec<PktLine>,
    /// Every `F link` line, in input order.
    pub link_faults: Vec<LinkFault>,
    /// Per `F link` line (same order): marked `p` (persistent), so `--on-irq replay` applies it again in the replay run.
    pub link_persist: Vec<bool>,
}

/// Does link `link` of chip `chip` reach another end (in the topology file, or in the wiring of `--chips K`)?
fn wired(prog: &Program, chips: usize, g: &Geom, chip: usize, link: usize) -> bool {
    match &prog.topo {
        Some(t) => link < g.links as usize && t.peer(chip, link).is_some(),
        None => super::links::link_peer(chips, g.links as usize, chip, link).is_some(),
    }
}

/// What the input checks against.
#[derive(Clone, Copy, Debug)]
pub struct Limits<'a> {
    pub geom: &'a Geom,
    pub chips: u32,
    /// The wiring of `--topology`; none: the wiring of `--chips K`.
    pub topo: Option<&'a std::sync::Arc<super::topology::Topology>>,
}

/// An `F` line given on the command line (`tsp-iss run --inject`): `text` is the line after the `F` (the leading `F` is
/// accepted), for chip `chip`. The checks are those of the input line, so a bad one reads like a bad line of the input,
/// without the line number.
// model: c_emulator/tsp_sim.c read_fault_line
pub fn add_fault_line(prog: &mut Program, chip: usize, text: &str, g: &Geom) -> Result<(), String> {
    let t = text.trim_start();
    let body = match t.strip_prefix('F') {
        Some(r) if r.starts_with([' ', '\t']) => r,
        _ => t,
    };
    let mut ln = Line { n: 0, s: body.as_bytes(), pos: 0 };
    let chips = prog.chips.len() as u32;
    ln.fault_line(prog, chip, g, chips).map_err(|e| e.msg)
}

struct Line<'a> {
    n: usize,
    s: &'a [u8],
    pos: usize,
}

impl Line<'_> {
    fn err<T>(&self, msg: &str) -> Result<T, InputError> {
        Err(InputError { line: self.n, msg: msg.to_string() })
    }

    /// The next hex field; `None` at the end of the line (`hexfield`).
    fn field(&mut self) -> Result<Option<u64>, InputError> {
        while self.pos < self.s.len() && matches!(self.s[self.pos], b' ' | b'\t' | b'\r') {
            self.pos += 1;
        }
        if self.pos >= self.s.len() {
            return Ok(None);
        }
        let rest = &self.s[self.pos..];
        if rest[0] == b'-' || rest[0] == b'+' {
            return self.err("expected a hex number");
        }
        let n = strtoull16(rest);
        let end = self.pos + n.end;
        let ok_end = n.end != 0 && (end >= self.s.len() || matches!(self.s[end], b' ' | b'\t' | b'\r'));
        if !ok_end {
            return self.err("expected a hex number");
        }
        if n.overflow {
            return self.err("number out of range");
        }
        self.pos = end;
        Ok(Some(n.mag))
    }

    /// The next word (a run of lowercase letters, at most 15); empty at the end of the line (`wordfield`).
    fn word(&mut self) -> String {
        while self.pos < self.s.len() && matches!(self.s[self.pos], b' ' | b'\t' | b'\r') {
            self.pos += 1;
        }
        let mut w = String::new();
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_lowercase() && w.len() < 15 {
            w.push(self.s[self.pos] as char);
            self.pos += 1;
        }
        w
    }

    /// The rest of an `H` line, after the `H` (`read_host_line`).
    fn host_line(&mut self, prog: &mut Program, chip: usize, g: &Geom) -> Result<(), InputError> {
        let w = self.word();
        if w.is_empty() {
            return self.err("unknown H line (expected start, dma, db, csr or inject)");
        }
        let cycle = self.need("cycle")?;
        let kind = match w.as_str() {
            "start" => {
                self.at_end()?;
                prog.starts[chip] = cycle;
                return Ok(());
            }
            "dma" => {
                let mut b = [0u8; MAX_LANES];
                let mut n = 0usize;
                while let Some(x) = self.field()? {
                    if x > 0xff {
                        return self.err("byte out of range");
                    }
                    if n as u64 == g.vlanes {
                        return self
                            .err(&format!("an H dma line has more bytes than the {} lanes of this profile", g.vlanes));
                    }
                    b[n] = x as u8;
                    n += 1;
                }
                HostKind::Dma(Box::new(b))
            }
            "db" => {
                let i = self.need("doorbell")?;
                self.at_end()?;
                if i >= g.host_doorbells {
                    return self.err("doorbell out of range (host.doorbells)");
                }
                HostKind::CsrWrite { addr: 0x300 + i, val: 1 }
            }
            "csr" => {
                let rw = self.word();
                let addr = self.need("csr address")?;
                if addr > 0xfff {
                    return self.err("csr address out of range (0..fff)");
                }
                match rw.as_str() {
                    "w" => {
                        let val = self.need("value")?;
                        self.at_end()?;
                        HostKind::CsrWrite { addr, val }
                    }
                    "r" => {
                        self.at_end()?;
                        HostKind::CsrRead { addr }
                    }
                    _ => return self.err("H csr: expected r or w"),
                }
            }
            "inject" => {
                let q = self.need("queue")?;
                if !(q <= 143 || q == 0xffff) {
                    return self.err("queue out of range (0..8f or ffff)");
                }
                let mut units = Vec::new();
                while let Some(x) = self.field()? {
                    if x > 0xffff_ffff {
                        return self.err("text unit out of range (more than 32 bits)");
                    }
                    units.push(x as u32);
                }
                HostKind::Inject { queue: if q == 0xffff { 144 } else { q as usize }, units }
            }
            _ => return self.err("unknown H line (expected start, dma, db, csr or inject)"),
        };
        let id = (prog.host.len() + prog.pkts.len()) as u64;
        prog.host.push(HostLine { chip: chip as u32, cycle, kind, id, persist: false });
        Ok(())
    }

    /// The rest of an `F` line, after the `F` (`read_fault_line`): an ECC fault injection, kept with the host's script.
    // model: c_emulator/tsp_sim.c read_fault_line
    fn fault_line(&mut self, prog: &mut Program, chip: usize, g: &Geom, chips: u32) -> Result<(), InputError> {
        let w = self.word();
        if w == "link" {
            return self.link_fault_line(prog, chip, g, chips);
        }
        if w == "mxm" {
            return self.mxm_fault_line(prog, chip, g);
        }
        let sram = w == "sram";
        if !sram && w != "srf" && w != "iq" {
            return self.err("unknown F line (expected sram, srf, iq, link or mxm)");
        }
        let cycle = self.need("cycle")?;
        let mut kind = if w == "iq" {
            let (q, wd, bit) = (self.need("queue")?, self.need("word")?, self.need("bit")?);
            if q > 144 {
                return self.err("queue out of range (0..90 hex)");
            }
            if wd > 0xff_ffff {
                return self.err("text word out of range (0..ffffff hex)");
            }
            if bit > 137 {
                return self.err("bit out of range (0..89 hex: data 0..7f, check bits 80..88, spare 89)");
            }
            HostKind::FlipIq { queue: q as usize, word: wd, bit: bit as u8 }
        } else if sram {
            let (s, a) = (self.need("slice")?, self.need("address")?);
            if s >= SLICES as u64 {
                return self.err("slice out of range (0..57 hex)");
            }
            if !g.slice_present(s) {
                return self.err("slice does not exist (slice mod 44 >= geometry.mem_slices_per_hemisphere)");
            }
            if a >= g.words_per_slice {
                return self.err("address out of range (>= words_per_slice)");
            }
            HostKind::FlipSram { slice: s as u8, addr: a as u32, sl: 0, bit: 0 }
        } else {
            let rw = self.word();
            if rw != "e" && rw != "w" {
                return self.err("F srf: expected the row e or w");
            }
            let (st, gap) = (self.need("stream")?, self.need("gap")?);
            if st > 31 {
                return self.err("stream out of range (0..1f)");
            }
            if gap > 43 {
                return self.err("gap out of range (0..2b)");
            }
            HostKind::FlipSrf { w: rw == "w", stream: st as u8, gap: gap as u8, sl: 0, bit: 0 }
        };
        let iq = matches!(kind, HostKind::FlipIq { .. });
        let (sl, bit) = if iq { (0, 0) } else { (self.need("superlane")?, self.need("bit")?) };
        if !iq && sl >= g.superlanes {
            return self.err("superlane out of range (>= geometry.superlanes)");
        }
        if !iq && bit > 137 {
            return self.err("bit out of range (0..89 hex: data 0..7f, check bits 80..88, spare 89)");
        }
        match &mut kind {
            HostKind::FlipSram { sl: s, bit: b, .. } | HostKind::FlipSrf { sl: s, bit: b, .. } => {
                *s = sl as u8;
                *b = bit as u8;
            }
            _ => {}
        }
        let tp = self.word();
        if !tp.is_empty() && tp != "t" && tp != "p" {
            return self.err("F: expected t or p at the end of the line");
        }
        self.at_end()?;
        if prog.host.iter().any(|o| o.chip == chip as u32 && o.cycle == cycle && o.kind == kind) {
            return self.err("F: two lines flip the same bit in one cycle");
        }
        let id = (prog.host.len() + prog.pkts.len()) as u64;
        prog.host.push(HostLine { chip: chip as u32, cycle, kind, id, persist: tp == "p" });
        Ok(())
    }

    /// The rest of an `F mxm` line, after the word `mxm` (`read_mxm_fault`): a bit of the exact sum of a lane of an ABC flips,
    /// kept with the host's script.
    // model: c_emulator/tsp_sim.c read_mxm_fault
    fn mxm_fault_line(&mut self, prog: &mut Program, chip: usize, g: &Geom) -> Result<(), InputError> {
        let cycle = self.need("cycle")?;
        let (plane, lane, bit) = (self.need("plane")?, self.need("lane")?, self.need("bit")?);
        if plane > 3 {
            return self.err("plane out of range (0..3)");
        }
        if plane >= 2 * g.mxm_planes {
            return self.err("plane out of range (>= 2 * mxm.planes_per_hemisphere)");
        }
        if lane >= g.mxm_rows.expect("profile.validate requires mxm.rows") {
            return self.err("lane out of range (>= mxm.rows)");
        }
        if bit > 95 {
            return self.err("bit out of range (0..5f)");
        }
        let tp = self.word();
        if !tp.is_empty() && tp != "t" && tp != "p" {
            return self.err("F: expected t or p at the end of the line");
        }
        self.at_end()?;
        let kind = HostKind::FlipMxm { plane: plane as u8, lane: lane as u16, bit: bit as u8 };
        if prog.host.iter().any(|o| o.chip == chip as u32 && o.cycle == cycle && o.kind == kind) {
            return self.err("F: two lines flip the same bit in one cycle");
        }
        let id = (prog.host.len() + prog.pkts.len()) as u64;
        prog.host.push(HostLine { chip: chip as u32, cycle, kind, id, persist: tp == "p" });
        Ok(())
    }

    /// The rest of an `F link` line, after the word `link` (`read_link_fault`): one flit of the vector the chip sends on a
    /// link in a cycle, kept for the link's send.
    // model: c_emulator/tsp_sim.c read_link_fault
    fn link_fault_line(&mut self, prog: &mut Program, chip: usize, g: &Geom, chips: u32) -> Result<(), InputError> {
        let cycle = self.need("cycle")?;
        let link = self.need("link")?;
        let flit = self.need("flit")?;
        if link >= g.links {
            return self.err("link out of range (>= c2c.links)");
        }
        if !wired(prog, chips as usize, g, chip, link as usize) {
            return self.err("F link: that link is not connected");
        }
        if flit >= link_flits(g.vlanes) {
            return self.err("flit out of range (>= the vector's flits: header, payload flits, tail)");
        }
        let o = self.word();
        let unc = o == "uncorr";
        if !unc && o != "corr" {
            return self.err("F link: expected corr or uncorr");
        }
        let tp = self.word();
        if !tp.is_empty() && tp != "t" && tp != "p" {
            return self.err("F: expected t or p at the end of the line");
        }
        self.at_end()?;
        let f = LinkFault { chip: chip as u32, cycle, link: link as u8, flit: flit as u8, unc };
        if prog.link_faults.iter().any(|o| o.chip == f.chip && o.cycle == cycle && o.link == f.link && o.flit == f.flit)
        {
            return self.err("F link: two lines mark the same flit of one vector");
        }
        prog.link_faults.push(f);
        prog.link_persist.push(tp == "p");
        Ok(())
    }

    /// The rest of a `P` line, after the `P` (`read_pk_line`).
    fn pkt_line(&mut self, prog: &mut Program, chip: usize, g: &Geom) -> Result<(), InputError> {
        let w = self.word();
        let csr = w == "csr";
        if !csr && w != "ctrl" {
            return self.err("unknown packet type (expected ctrl or csr)");
        }
        let cycle = self.need("cycle")?;
        let link = self.need("link")?;
        let mut kind = PktKind::Ctrl;
        if csr {
            if !g.host_remote_csr {
                return self.err("P csr: remote CSR access is off (config host.remote_csr)");
            }
            let rw = self.word();
            if rw != "w" && rw != "r" {
                return self.err("P csr: expected r or w");
            }
            let addr = self.need("csr address")?;
            if addr > 0xfff {
                return self.err("csr address out of range (0..fff)");
            }
            kind = if rw == "w" {
                PktKind::CsrWrite { addr, val: self.need("value")? }
            } else {
                PktKind::CsrRead { addr }
            };
        }
        self.at_end()?;
        if link >= g.links || !wired(prog, prog.chips.len(), g, chip, link as usize) {
            return self.err("P: that link is not connected");
        }
        let id = (prog.host.len() + prog.pkts.len()) as u64;
        prog.pkts.push(PktLine { chip: chip as u32, cycle, link: link as u8, kind, id });
        Ok(())
    }

    fn need(&mut self, what: &str) -> Result<u64, InputError> {
        match self.field()? {
            Some(v) => Ok(v),
            None => self.err(&format!("missing {what}")),
        }
    }

    fn at_end(&mut self) -> Result<(), InputError> {
        match self.field()? {
            Some(_) => self.err("too many fields"),
            None => Ok(()),
        }
    }
}

/// The checks of an `S` or `D` line on its slice and address, in `tsp_sim`'s order.
pub fn check_word(g: &Geom, slice: u64, addr: u64) -> Result<(), &'static str> {
    if slice >= SLICES as u64 {
        return Err("slice out of range (0..57 hex)");
    }
    if !g.slice_present(slice) {
        return Err("slice does not exist (slice mod 44 >= geometry.mem_slices_per_hemisphere)");
    }
    if addr >= g.words_per_slice {
        return Err("address out of range (>= words_per_slice)");
    }
    Ok(())
}

impl ChipInput {
    /// Add text units to a queue as `Q` lines do: the queue is 0..8f or ffff, padding is dropped and the
    /// queue's whole image fits `icu.iq_depth` (the machine would refuse it otherwise).
    pub fn add_image(&mut self, g: &Geom, queue: u16, units: &[u32]) -> Result<(), String> {
        if !(queue <= 143 || queue == 0xffff) {
            return Err(format!("queue {queue:#x} out of range (0..8f or ffff)"));
        }
        let mut len = self.images.get(queue).map_or(0, |u| u.len());
        let units: Vec<u32> = units.iter().copied().filter(|u| *u != PADDING).collect();
        len += units.len();
        if len > g.iq_units {
            return Err(format!("initial IQ image of queue {queue:x} longer than icu.iq_depth bytes"));
        }
        if !units.is_empty() {
            self.images.extend(queue, &units);
        }
        Ok(())
    }

    /// Add an SRAM preload as an `S` line does (at most `vlanes` bytes).
    pub fn add_preload(&mut self, g: &Geom, slice: u64, addr: u64, bytes: &[u8]) -> Result<(), String> {
        check_word(g, slice, addr).map_err(|m| format!("{m}: slice {slice:x} address {addr:x}"))?;
        if bytes.len() as u64 > g.vlanes {
            return Err(format!("an S line has more bytes than the {} lanes of this profile", g.vlanes));
        }
        self.preloads.push(Preload { slice: slice as u8, addr: addr as u32, bytes: bytes.to_vec() });
        Ok(())
    }
}

/// Parse the program. The first error ends the parse.
pub fn parse(input: &[u8], lim: Limits<'_>) -> Result<Program, InputError> {
    let g = lim.geom;
    let mut prog = Program {
        topo: lim.topo.cloned(),
        chips: vec![ChipInput::default(); lim.chips as usize],
        starts: vec![0; lim.chips as usize],
        ..Program::default()
    };
    let mut chip = 0usize;
    let mut lines: Vec<&[u8]> = input.split(|b| *b == b'\n').collect();
    if input.last() == Some(&b'\n') || input.is_empty() {
        lines.pop(); // no line after the last line feed
    }
    for (i, raw) in lines.into_iter().enumerate() {
        // C strings end at a NUL
        let raw = &raw[..raw.iter().position(|b| *b == 0).unwrap_or(raw.len())];
        let mut ln = Line { n: i + 1, s: raw, pos: 0 };
        while ln.pos < raw.len() && matches!(raw[ln.pos], b' ' | b'\t') {
            ln.pos += 1;
        }
        if ln.pos >= raw.len() || raw[ln.pos] == b'\r' || raw[ln.pos] == b'#' {
            continue;
        }
        let tag = raw[ln.pos];
        ln.pos += 1;
        if !matches!(raw.get(ln.pos), Some(b' ' | b'\t')) {
            return ln.err("unknown line (expected S, Q, D, C, R, H, F or P)");
        }
        match tag {
            b'S' | b'D' => {
                let s = ln.need("slice")?;
                let a = ln.need("address")?;
                if let Err(m) = check_word(g, s, a) {
                    return ln.err(m);
                }
                if tag == b'S' {
                    let mut bytes = Vec::new();
                    while let Some(x) = ln.field()? {
                        if x > 0xff {
                            return ln.err("byte out of range");
                        }
                        if bytes.len() as u64 == g.vlanes {
                            return ln
                                .err(&format!("an S line has more bytes than the {} lanes of this profile", g.vlanes));
                        }
                        bytes.push(x as u8);
                    }
                    prog.chips[chip].preloads.push(Preload { slice: s as u8, addr: a as u32, bytes });
                } else {
                    ln.at_end()?;
                    prog.dumps.push(Dump { chip: chip as u32, slice: s as u8, addr: a as u32 });
                }
            }
            b'Q' => {
                let q = ln.need("queue")?;
                if !(q <= 143 || q == 0xffff) {
                    return ln.err("queue out of range (0..8f or ffff)");
                }
                let images = &mut prog.chips[chip].images;
                let mut len = images.get(q as u16).map_or(0, |u| u.len());
                let mut units = Vec::new();
                while let Some(x) = ln.field()? {
                    if x > 0xffff_ffff {
                        return ln.err("text unit out of range (more than 32 bits)");
                    }
                    if x as u32 == PADDING {
                        continue; // padding is never enqueued
                    }
                    if len >= g.iq_units {
                        return ln.err("initial IQ image longer than icu.iq_depth bytes");
                    }
                    len += 1;
                    units.push(x as u32);
                }
                if !units.is_empty() {
                    images.extend(q as u16, &units);
                }
            }
            b'C' => {
                let k = ln.need("chip")?;
                ln.at_end()?;
                if k >= lim.chips as u64 {
                    return ln.err("chip out of range (--chips)");
                }
                chip = k as usize;
            }
            b'R' => {
                let a = ln.need("csr address")?;
                ln.at_end()?;
                if a > 0xfff {
                    return ln.err("csr address out of range (0..fff)");
                }
                prog.rcsr.push((chip as u32, a));
            }
            b'H' => ln.host_line(&mut prog, chip, g)?,
            b'P' => ln.pkt_line(&mut prog, chip, g)?,
            b'F' => ln.fault_line(&mut prog, chip, g, lim.chips)?,
            _ => return ln.err("unknown line (expected S, Q, D, C, R, H, F or P)"),
        }
    }
    Ok(prog)
}
