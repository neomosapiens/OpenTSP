//! SECDED ECC on SRAM words, stream registers and data paths (`core/ecc.sail`; `microarchitecture.md` §15).
//!
//! The model's representation is kept: the data bits as stored (a flipped bit is flipped) plus an overlay of 10 bits per
//! superlane, `e[8:0]` = the stored check bits XOR the code of the stored data, `e[9]` = the spare bit. An overlay of
//! zero is a valid word, so the check bits are implicit and a run in which nothing was ever faulted never builds one:
//! every hook of the machine is behind [`Ecc::live`], which stays false until a fault is injected (a poisoned result
//! needs an uncorrectable operand, which needs a fault), so a fault-free run pays one branch per read and write.
//!
//! Overlays are interned: an overlay (the 10 bits of every superlane) is a small id, 0 for the all-valid overlay, so the
//! stream registers, the commit ring and the SRAM carry a `u32`.
use crate::layout::RowId;
use crate::observe::Observer;
use crate::sram::tile_active;
use crate::Machine;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

/// The most superlanes of any profile (`MAX_LANES / 16`).
pub const MAX_SL: usize = crate::geom::MAX_LANES / 16;

/// The overlay of every superlane of a word: 10 bits each.
pub type Ov = [u16; MAX_SL];

/// The all-valid overlay.
pub const OV_ZERO: Ov = [0; MAX_SL];

/// The overlay id of the all-valid overlay.
pub const OV0: u32 = 0;

/// The overlay of a poisoned superlane: `c[0]` and `c[1]` inverted (a double error: syndrome 3, even parity, so every
/// later reader finds it uncorrectable); the spare bit stays clear (it belongs to the error report; implementation choice).
// model: core/ecc.sail ecc_poison_sl
pub const POISON_SL: u16 = 0b00_0000_0011;

fn is_pow2(p: u32) -> bool {
    matches!(p, 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256)
}

fn log2(p: u32) -> u32 {
    31 - p.max(1).leading_zeros()
}

/// The data bit index of codeword position `p` (p >= 3, not a power of two).
// model: core/ecc.sail ecc_data_index
pub fn data_index(p: u32) -> u32 {
    p - log2(p) - 2
}

/// The codeword position of data bit `j` (0..127).
// model: core/ecc.sail ecc_pos
pub fn pos(j: u32) -> u32 {
    let mut p = j + 3;
    while is_pow2(p) || data_index(p) < j {
        p += 1;
    }
    p
}

fn par10(x: u32) -> bool {
    (x & 0x3ff).count_ones() % 2 == 1
}

/// The overlay change when bit `n` of the 138-bit word flips (0..127 data, 128..136 check bits `c[0..8]`, 137 spare).
// model: core/ecc.sail ecc_flip_delta
pub fn flip_delta(n: u32) -> u16 {
    if n < 128 {
        let pb = pos(n);
        let c8 = !par10(pb);
        (if c8 { 0x100 } else { 0 }) | pb as u16
    } else if n < 137 {
        1 << (n - 128)
    } else {
        0x200
    }
}

/// What the check of one superlane word finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Ok,
    Corrected,
    Uncorrectable,
}

/// The class of overlay `e` and the position the syndrome names (0 for the overall parity bit `c[8]`; for an
/// uncorrectable word the raw syndrome).
// model: core/ecc.sail ecc_classify
pub fn classify(e: u16) -> (Class, u32) {
    let s = (e & 0xff) as u32;
    let nonzero = e & 0x1ff != 0;
    let q = par10((e & 0x1ff) as u32);
    if !nonzero {
        (Class::Ok, 0)
    } else if q {
        if s == 0 || is_pow2(s) || (3..=136).contains(&s) {
            (Class::Corrected, s)
        } else {
            (Class::Uncorrectable, s)
        }
    } else {
        (Class::Uncorrectable, s)
    }
}

/// The data bit a single-error syndrome names (`None` when it names a check bit or the overall parity bit): the bit
/// [`classify`] says to flip back.
// model: core/ecc.sail ecc_correct
pub fn data_flip(e: u16) -> Option<u32> {
    match classify(e) {
        (Class::Corrected, s) if s >= 3 && !is_pow2(s) => Some(data_index(s)),
        _ => None,
    }
}

/// Where an event happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Where {
    /// A stream operand: the row (true: W) and the stream.
    Stream { w: bool, id: u8, gap: u8 },
    /// A stored SRAM word: slice and address.
    Sram { slice: u8, addr: u32 },
    /// A word of instruction text: the word index counted from 0 over the queue's text in production order.
    Text { word: u32 },
}

/// One correction or uncorrectable superlane (`T ecc`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EccEvent {
    pub cycle: u64,
    /// The reading queue (144: the control channel).
    pub queue: usize,
    pub unc: bool,
    /// A spare-bit error report found by the consumer (`T ecc ... spare`): `pos` is S[0..3] of the superlane.
    pub spare: bool,
    pub at: Where,
    pub sl: u8,
    pub pos: u32,
}

impl EccEvent {
    /// The part of the `T ecc` line after `corr|unc`: the place and the position (`c_emulator/tsp_sim.c ecc_event_callback`).
    pub fn place(&self) -> String {
        match self.at {
            Where::Stream { w, id, .. } => {
                format!("{}:{id} sl {} pos {}", if w { 'W' } else { 'E' }, self.sl, self.pos)
            }
            Where::Sram { slice, addr } => format!("sram {slice}:{addr} sl {} pos {}", self.sl, self.pos),
            Where::Text { word } => format!("text word {word} pos {}", self.pos),
        }
    }
}

/// The ECC state of one chip.
#[derive(Debug, Default)]
pub struct Ecc {
    /// Some fault was injected: every hook looks at overlays. Never cleared by a run.
    pub live: bool,
    /// The overlays by id (index 0 is [`OV_ZERO`]) and their ids.
    ovs: Vec<Ov>,
    ids: BTreeMap<Ov, u32>,
    /// The overlay of a stored SRAM word, by (slice, address); a word that is not here has a valid one.
    sram: BTreeMap<(u8, u32), u32>,
    pub(crate) sram_revision: BTreeMap<u8, u128>,
    /// Superlanes found uncorrectable by the operand checks of the current instruction (`ecc_bad`).
    pub bad: Cell<u32>,
    /// `ecc_dep_all` and `ecc_after_diag` of the model.
    pub dep_all: Cell<bool>,
    pub after_diag: Cell<bool>,
    /// The OPR-001 the current instruction is about to report is of class `data`: a Gather or Scatter tile address, an `sxselect`
    /// select (the good tiles and lanes still take effect). Taken by [`Machine::diag`]; cleared at the start of each element.
    pub data_opr: Cell<bool>,
    /// The superlanes whose result is poisoned because an input of the unit was (an accumulator, a received vector):
    /// `ecc_inherit`. Inherited poison is not reported again; the consumer reports it.
    pub inherit: Cell<u32>,
    /// The streams of the last 16-stream read that had an uncorrectable superlane, a bit per stream (`ecc_g16`).
    pub g16: Cell<u16>,
    /// The superlanes the last [`Ecc::check_value`] found uncorrectable.
    pub last_bad: Cell<u32>,
    /// Superlanes of the vector the current Receive took that the link FEC found uncorrectable (`ecc_link_bad`).
    pub link_bad: Cell<u32>,
    /// The events of the current instruction, in the model's order, until the executor passes them to the observer.
    pub events: RefCell<Vec<EccEvent>>,
    /// Corrected copies of operands that the borrowing reads hand out (see [`Ecc::scratch`]).
    scratch: RefCell<Vec<Box<[u8]>>>,
}

impl Ecc {
    pub fn new() -> Ecc {
        Ecc::default()
    }

    /// Back to reset (the model's registers; the SRAM overlays are the harness's and are dropped with the SRAM).
    pub fn reset(&mut self) {
        self.scratch.get_mut().clear();
        self.bad.set(0);
        self.dep_all.set(false);
        self.after_diag.set(false);
        self.inherit.set(0);
        self.g16.set(0);
        self.link_bad.set(0);
        self.events.borrow_mut().clear();
    }

    fn ov(&self, id: u32) -> Ov {
        if id == OV0 {
            OV_ZERO
        } else {
            self.ovs[id as usize - 1]
        }
    }

    /// The id of overlay `o`.
    pub fn intern(&mut self, o: Ov) -> u32 {
        if o == OV_ZERO {
            return OV0;
        }
        if let Some(&id) = self.ids.get(&o) {
            return id;
        }
        self.ovs.push(o);
        let id = self.ovs.len() as u32;
        self.ids.insert(o, id);
        id
    }

    /// The 10 bits of superlane `g` of overlay `id`.
    pub fn get(&self, id: u32, g: usize) -> u16 {
        if id == OV0 {
            0
        } else {
            self.ovs[id as usize - 1][g]
        }
    }

    /// Overlay `id` with superlane `g` replaced by `e` (`ecc_ov_set`).
    pub fn set(&mut self, id: u32, g: usize, e: u16) -> u32 {
        let mut o = self.ov(id);
        o[g] = e;
        self.intern(o)
    }

    /// The bitwise OR of two overlays (an OR-merged stream value).
    pub fn or(&mut self, a: u32, b: u32) -> u32 {
        if a == b || b == OV0 {
            return a;
        }
        if a == OV0 {
            return b;
        }
        let (x, y) = (self.ov(a), self.ov(b));
        let mut o = OV_ZERO;
        for g in 0..MAX_SL {
            o[g] = x[g] | y[g];
        }
        self.intern(o)
    }

    /// The overlay of a stream result: the poison pattern in the superlanes of `mask`, valid elsewhere
    /// (`ecc_result_ov`).
    pub fn poison(&mut self, mask: u32) -> u32 {
        if mask == 0 {
            return OV0;
        }
        let mut o = OV_ZERO;
        for (g, e) in o.iter_mut().enumerate() {
            if mask >> g & 1 == 1 {
                *e = POISON_SL;
            }
        }
        self.intern(o)
    }

    /// Overlay `id` restricted to the superlanes of `active` (the others valid): the check bits a MEM Read forwards for
    /// its active tiles.
    pub fn only(&mut self, id: u32, active: impl Fn(usize) -> bool) -> u32 {
        if id == OV0 {
            return OV0;
        }
        let mut o = self.ov(id);
        for (g, e) in o.iter_mut().enumerate() {
            if !active(g) {
                *e = 0;
            }
        }
        self.intern(o)
    }

    pub(crate) fn protected_words(&self) -> Vec<(u8, u32)> {
        self.sram.keys().copied().collect()
    }

    /// The overlay of the stored SRAM word at `slice`, `addr`.
    pub fn sram_ov(&self, slice: usize, addr: u32) -> u32 {
        if self.sram.is_empty() {
            return OV0;
        }
        self.sram.get(&(slice as u8, addr)).copied().unwrap_or(OV0)
    }

    /// Store the overlay of the SRAM word at `slice`, `addr`.
    pub fn set_sram_ov(&mut self, slice: usize, addr: u32, id: u32) {
        *self.sram_revision.entry(slice as u8).or_default() += 1;
        if id == OV0 {
            self.sram.remove(&(slice as u8, addr));
        } else {
            self.sram.insert((slice as u8, addr), id);
        }
    }

    /// The superlanes the poison of the current instruction covers (`ecc_poison_mask`): the uncorrectable ones, or every
    /// superlane (of `superlanes`) for a unit whose output depends on every input superlane.
    // model: core/ecc.sail ecc_poison_mask
    pub fn poison_mask(&self, superlanes: u64) -> u32 {
        let bad = self.bad.get();
        (if self.dep_all.get() && bad != 0 { ((1u64 << superlanes) - 1) as u32 } else { bad })
            | self.inherit.get()
            | self.link_bad.get()
    }

    /// Check one superlane's 16 bytes against its overlay `e`: correct a single-bit error in `data` (the reader's
    /// copy), record the event, and return whether the superlane is uncorrectable.
    #[allow(clippy::too_many_arguments)]
    pub fn check_sl(&self, data: &mut [u8], e: u16, cycle: u64, queue: usize, at: Where, sl: usize) -> bool {
        let (k, s) = classify(e);
        match k {
            Class::Ok => false,
            Class::Corrected => {
                if s >= 3 && !is_pow2(s) {
                    let j = data_index(s) as usize;
                    data[j / 8] ^= 1 << (j % 8);
                }
                self.events.borrow_mut().push(EccEvent {
                    cycle,
                    queue,
                    unc: false,
                    spare: false,
                    at,
                    sl: sl as u8,
                    pos: s,
                });
                false
            }
            Class::Uncorrectable => {
                self.events.borrow_mut().push(EccEvent {
                    cycle,
                    queue,
                    unc: true,
                    spare: false,
                    at,
                    sl: sl as u8,
                    pos: s,
                });
                true
            }
        }
    }

    /// Check a stream value (`vlanes` bytes) against overlay `id`, superlane by superlane (`ecc_check_value`): the
    /// reader gets the corrected data in `v`, `bad` gets a bit for each uncorrectable superlane.
    ///
    /// `spare` has a bit for each superlane whose spare bit is set by an error report of the register file (the model
    /// merges it into the overlay's spare bit e[9]); `s03(x)` gives S[0..3] of superlane x. A set spare bit is logged
    /// before the superlane's check (`ecc_check_value`).
    #[allow(clippy::too_many_arguments)]
    // model: core/ecc.sail ecc_check_value
    pub fn check_value(
        &self,
        v: &mut [u8],
        id: u32,
        cycle: u64,
        queue: usize,
        at: Where,
        spare: u32,
        s03: &dyn Fn(usize) -> u32,
    ) {
        self.last_bad.set(0);
        if id == OV0 && spare == 0 {
            return;
        }
        let o = self.ov(id);
        let mut bad = 0u32;
        for (g, &e) in o.iter().enumerate().take(v.len() / 16) {
            if spare >> g & 1 == 1 || e & 0x200 != 0 {
                self.events.borrow_mut().push(EccEvent {
                    cycle,
                    queue,
                    unc: false,
                    spare: true,
                    at,
                    sl: g as u8,
                    pos: s03(g),
                });
            }
            if e != 0 && self.check_sl(&mut v[16 * g..16 * g + 16], e, cycle, queue, at, g) {
                bad |= 1 << g;
            }
        }
        self.last_bad.set(bad);
        self.bad.set(self.bad.get() | bad);
    }
}

impl Ecc {
    /// The start of an instruction (`ecc_begin`).
    #[cold]
    pub fn begin(&mut self) {
        self.scratch.get_mut().clear();
        self.bad.set(0);
        self.dep_all.set(false);
        self.after_diag.set(false);
        self.inherit.set(0);
        self.link_bad.set(0);
    }

    /// Keep a corrected copy of an operand for as long as the machine is borrowed: the borrowing reads
    /// (`read_planes_refs`) return slices, and a corrected operand lives nowhere in the register file. The copies are
    /// dropped by [`Ecc::begin`], which needs `&mut` and so cannot run while a slice is out.
    pub fn scratch(&self, v: Vec<u8>) -> &[u8] {
        let b = v.into_boxed_slice();
        let p: *const [u8] = &*b;
        self.scratch.borrow_mut().push(b);
        // SAFETY: the box's heap block does not move when the vector of boxes grows, and the block is dropped only by
        // `begin` and `reset`, which take `&mut self`: no slice returned here is alive then.
        unsafe { &*p }
    }
}

impl<const L: usize> Machine<L> {
    /// The overlay of the result of the current instruction (`ecc_result_ov`): the poison pattern in the poisoned
    /// superlanes, valid check bits elsewhere.
    #[inline(always)]
    pub(crate) fn ecc_result_ov(&mut self) -> u32 {
        if self.ecc.live {
            self.ecc_result_ov_live()
        } else {
            OV0
        }
    }

    #[cold]
    #[inline(never)]
    fn ecc_result_ov_live(&mut self) -> u32 {
        let m = self.ecc.poison_mask(self.geom.superlanes);
        self.ecc.poison(m)
    }

    /// The check bits a MEM Read forwards: those of the stored word at `slice`, `idx`, for the active tiles.
    #[cold]
    #[inline(never)]
    pub(crate) fn ecc_forward_ov(
        &mut self,
        slice: usize,
        idx: u64,
        vl: u64,
        split: u64,
        use_thread: bool,
        t: u8,
    ) -> u32 {
        let id = self.ecc.sram_ov(slice, idx as u32);
        self.ecc.only(id, |g| tile_active(vl, split, use_thread, t, g as u64))
    }

    /// Store the overlay of the SRAM word just written (valid check bits when `id` is 0).
    pub(crate) fn ecc_sram_store(&mut self, slice: usize, addr: u32, id: u32) {
        if self.ecc.live {
            self.ecc.set_sram_ov(slice, addr, id);
        }
    }

    /// The end of an instruction on a chip with faults: its events reach the observer, then its diagnostic, then
    /// OTS-E-ECC-001 when an operand had an uncorrectable superlane (and the instruction's diagnostic allows it).
    // model: postlude/tick.sail execute_element
    #[cold]
    #[inline(never)]
    pub(crate) fn ecc_finish<O: Observer + ?Sized>(&mut self, q: usize, c: u64, x: crate::slices::Exec, obs: &mut O) {
        use crate::slices::Exec;
        self.ecc_flush(obs);
        let bad = self.ecc.bad.get() != 0;
        match x {
            Exec::Diagnostic(d) => {
                self.diag(obs, q, c, d);
                if self.ecc.after_diag.get() && bad {
                    self.diag(obs, q, c, ots_isa::Code::Ecc001);
                }
            }
            Exec::Issued => {
                if self.ecc.link_bad.get() != 0 {
                    self.diag(obs, q, c, ots_isa::Code::Lnk004);
                }
                if bad {
                    self.diag(obs, q, c, ots_isa::Code::Ecc001);
                }
            }
        }
    }

    /// Pass the events of the current instruction to the observer, in order.
    #[cold]
    #[inline(never)]
    pub(crate) fn ecc_flush<O: Observer + ?Sized>(&mut self, obs: &mut O) {
        let ev = std::mem::take(&mut *self.ecc.events.borrow_mut());
        for e in ev {
            obs.ecc(&e);
            match e.at {
                Where::Stream { w, id, gap } if e.spare => {
                    self.ras_spare_event(e.cycle, e.queue, w, id as u64, gap as u64, e.sl, e.pos, obs)
                }
                Where::Stream { w, id, gap } => {
                    self.ras_ecc_event(e.cycle, e.queue, 0, e.unc, w, id as u64, gap as u64, e.sl, e.pos, obs)
                }
                Where::Sram { slice, addr } => {
                    self.ras_ecc_event(e.cycle, e.queue, 1, e.unc, false, slice as u64, addr as u64, e.sl, e.pos, obs)
                }
                Where::Text { word } => {
                    self.ras_ecc_event(e.cycle, e.queue, 2, e.unc, false, 0, word as u64, e.sl, e.pos, obs)
                }
            }
        }
    }

    /// The head parcel of queue `q` (`units` of them) leaves the IQ in cycle `c`: every instruction of it (the timing prefix
    /// included) checks the one or two text words it lies in, in program order: a correctable error is reported and
    /// corrected in the queue, an uncorrectable one is reported each time.
    // model: postlude/iq.sail iq_text_check
    pub(crate) fn text_check<O: Observer + ?Sized>(&mut self, q: usize, c: u64, units: usize, obs: &mut O) {
        if !self.iq.tx.live {
            return;
        }
        self.ecc_flush(obs);
        let mut pos = 0;
        while pos < units {
            let (w, a, b) = self.iq.text_words(q, pos);
            self.text_word_check(q, c, a, obs);
            if b != a {
                self.text_word_check(q, c, b, obs);
            }
            pos += w;
        }
    }

    // model: postlude/iq.sail iq_word_check
    fn text_word_check<O: Observer + ?Sized>(&mut self, q: usize, c: u64, w: Option<u128>, obs: &mut O) {
        let Some(w) = w else { return };
        let e = self.iq.tx.ov(q, w);
        if e == 0 {
            return;
        }
        let (k, s) = classify(e);
        if k == Class::Ok {
            return;
        }
        let unc = k == Class::Uncorrectable;
        let ev = EccEvent {
            cycle: c,
            queue: q,
            unc,
            spare: false,
            at: Where::Text { word: (w & 0xff_ffff) as u32 },
            sl: 0,
            pos: s,
        };
        obs.ecc(&ev);
        self.ras_ecc_event(c, q, 2, unc, false, 0, (w & 0xff_ffff) as u64, 0, s, obs);
        if !unc {
            self.iq.text_fix(q, w, e);
        }
    }

    /// `F iq`: flip bit `n` of text word `w` of queue `q`; false when the word is not in the queue.
    // model: postlude/harness_api.sail model_flip_iq
    pub(crate) fn ecc_flip_iq(&mut self, q: usize, w: u64, n: u32) -> bool {
        if !self.iq.flip_text(q, w, n) {
            return false;
        }
        self.iq.reparse_reset(q);
        self.iq.parse(&self.isa, q);
        self.d.touch(q);
        true
    }

    /// `F sram`: flip bit `n` of superlane `sl` of the stored word at `slice`, `addr`.
    // model: postlude/harness_api.sail model_flip_sram
    pub(crate) fn ecc_flip_sram(&mut self, slice: u8, addr: u32, sl: usize, n: u32) {
        self.ecc.live = true;
        if (addr as u64) < self.geom.words_per_slice && (sl as u64) < self.geom.superlanes {
            if n < 128 {
                let mut w = self.sram.read(slice as usize, addr as u64);
                w[16 * sl + n as usize / 8] ^= 1 << (n % 8);
                self.sram.write(slice as usize, addr as u64, &w[..self.geom.vlanes as usize]);
            }
            let id = self.ecc.sram_ov(slice as usize, addr);
            let e = self.ecc.get(id, sl) ^ flip_delta(n);
            let nid = self.ecc.set(id, sl, e);
            self.ecc.set_sram_ov(slice as usize, addr, nid);
        }
    }

    /// `F srf`: flip bit `n` of superlane `sl` of the value register `gap` of the row's stream `stream` holds at `c`.
    // model: postlude/harness_api.sail model_flip_srf
    pub(crate) fn ecc_flip_srf(&mut self, c: u64, w: bool, stream: u8, gap: u8, sl: usize, n: u32) {
        self.ecc.live = true;
        let row = if w { RowId::W } else { RowId::E };
        let ecc = &mut self.ecc;
        self.srf.flip(c, (row, stream, gap), sl, n, |old| {
            let e = ecc.get(old, sl) ^ flip_delta(n);
            ecc.set(old, sl, e)
        });
    }
}
