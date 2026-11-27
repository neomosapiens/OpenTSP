//! The instruction queues and the head parcel (`slices/cu/cu_iq.sail`, `postlude/iq.sail`).
//!
//! Every queue (the 144 data queues and the control channel, index 144) has an IQ of 32-bit text
//! units of `icu.iq_depth` bytes and a fill FIFO of fetched payloads still on their way. The
//! initial image of each queue is loaded before cycle 0; `Ifetch` and `IfetchM` add to it through
//! [`Iq::fetch`] (the model's `iq_fetch`).
use crate::driver::classify::{classify_into, classify_items, CycleRecs};
use crate::geom::{Geom, QUEUES};
use ots_isa::{Code as Diag, Insn, Isa, PADDING};
use std::collections::VecDeque;
use std::sync::Arc;

/// Fetched payloads in the order their fetches executed; the units of a payload are taken at
/// `icu.fetch_bytes_per_cycle` from cycle `ready` = (execute cycle) + `icu.fetch_latency`.
// model: slices/cu/cu_iq.sail fill_seg
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FillSeg {
    pub ready: u128,
    pub units: VecDeque<u32>,
    /// The production index of the payload's first unit (`-1`: host-injected text, in no text word), the units taken so
    /// far, and the overlay of each of its text words (empty: all valid).
    pub base: i128,
    pub taken: usize,
    pub ovw: Vec<u16>,
}

/// The tag of a ring slot whose unit is in no protected text word.
pub const NO_WORD: u32 = u32::MAX;

/// ECC on instruction text (`microarchitecture.md` §16, `postlude/iq.sail`): a text word is four consecutive units of
/// one source (16 bytes); the check bits are kept as the overlay of the model's representation (10 bits: the stored
/// check bits XOR the code of the stored data, plus the spare bit), by word, for the words that have one. Tags (the
/// production index of each unit of the rings) are kept only when the program has `F` lines (`track`).
// model: slices/cu/cu_iq.sail iq_tag_read, iq_prod, iq_wov, iq_ecc_live
#[derive(Debug, Default)]
pub struct TextEcc {
    pub track: bool,
    /// Some overlay was set since reset: the scan looks at overlays.
    pub live: bool,
    tags: Vec<Box<[i128]>>,
    prod: Vec<u128>,
    wov: Vec<Vec<(u128, u16)>>,
}

impl TextEcc {
    /// The overlay of text word `w` of queue `q` (0: valid).
    pub fn ov(&self, q: usize, w: u128) -> u16 {
        self.wov[q].iter().find(|x| x.0 == w).map_or(0, |x| x.1)
    }

    // model: slices/cu/cu_iq.sail iq_word_ov_set
    pub fn set_ov(&mut self, q: usize, w: u128, e: u16) {
        self.wov[q].retain(|x| x.0 != w);
        if e != 0 {
            self.wov[q].push((w, e));
            self.live = true;
        }
    }
}

/// The units of one queue as the decoder sees them: corrected where the word they lie in has a correctable error
/// (`iq_view`), and whether an instruction lies in an uncorrectable word (`iq_text_unc`).
struct View<'a> {
    ring: &'a [u32],
    tags: &'a [i128],
    wov: &'a [(u128, u16)],
    head: usize,
    cap: usize,
    live: bool,
}

impl View<'_> {
    #[inline]
    fn slot(&self, i: usize) -> usize {
        let x = self.head + i;
        if x >= self.cap {
            x - self.cap
        } else {
            x
        }
    }

    #[inline]
    fn raw(&self, i: usize) -> u32 {
        self.ring[self.slot(i)]
    }

    fn word(&self, i: usize) -> Option<u128> {
        match self.tags.get(self.slot(i)) {
            Some(&t) if t >= 0 => Some(t as u128 / 4),
            _ => None,
        }
    }

    fn ov(&self, w: u128) -> u16 {
        self.wov.iter().find(|x| x.0 == w).map_or(0, |x| x.1)
    }

    // model: postlude/iq.sail iq_view
    #[inline]
    fn unit(&self, i: usize) -> u32 {
        let u = self.raw(i);
        if !self.live {
            return u;
        }
        let Some(w) = self.word(i) else { return u };
        match crate::ecc::data_flip(self.ov(w)) {
            Some(j) if i128::from(j / 32) == self.tags[self.slot(i)] % 4 => u ^ 1 << (j % 32),
            _ => u,
        }
    }

    fn word_unc(&self, i: usize) -> bool {
        self.word(i).is_some_and(|w| {
            let e = self.ov(w);
            e != 0 && crate::ecc::classify(e).0 == crate::ecc::Class::Uncorrectable
        })
    }

    // model: postlude/iq.sail iq_text_unc
    #[inline]
    fn unc(&self, pos: usize, w: usize) -> bool {
        self.live && (self.word_unc(pos) || (w == 2 && self.word_unc(pos + 1)))
    }

    // model: postlude/iq.sail iq_decode_at
    fn decode(&self, isa: &Isa, pos: usize, w: usize) -> Result<Insn, Diag> {
        if self.unc(pos, w) {
            return Err(Diag::Ecc002);
        }
        let (u0, u1) = (self.unit(pos), if w == 2 { self.unit(pos + 1) } else { 0 });
        decode_text(isa, u0, u1)
    }
}

/// The IQs of all queues (`iq_head`, `iq_len`, `iq_last`, `iq_dry`, `fill_fifo` and the rings).
#[derive(Debug)]
pub struct Iq {
    pub(crate) observe_sites: bool,
    cap: usize,
    rate: usize,
    latency: u64,
    /// `icu.fetch_staging_depth`: the payloads a queue's fill FIFO holds at most.
    staging_depth: usize,
    vlanes: usize,
    ring: Vec<Box<[u32]>>,
    head: [usize; QUEUES],
    /// Units held.
    pub len: [usize; QUEUES],
    /// The most units held at once, and the fewest held after a dispatch (`usize::MAX`: nothing
    /// was dispatched); for the statistics.
    pub hi: [usize; QUEUES],
    pub lo: [usize; QUEUES],
    /// Cycle of the queue's last dispatch (`NO_DISPATCH`, the all-ones value: none, the cycle before cycle 0).
    pub last: [i128; QUEUES],
    /// OTS-E-IQ-001 pending for the head parcel.
    pub dry: [bool; QUEUES],
    /// The last cycle in which a unit entered the queue's IQ or the queue dispatched (0: the initial image);
    /// the cycle of an OTS-E-IQ-004.
    pub active: [u64; QUEUES],
    /// The origin of the text: the units of the object's own image per queue, the units taken from the IQ so far, and
    /// whether the parcel taken last began after the image (a unit fetched from SRAM or injected by the host). The IQ is
    /// first in, first out and the image is loaded before cycle 0, so the units from the start of the queue's text
    /// to `image_len` are the image and every later one is not.
    image_len: [usize; QUEUES],
    consumed: [usize; QUEUES],
    taken_fetched: [bool; QUEUES],
    pub fill: Vec<VecDeque<FillSeg>>,
    /// The queues whose fill FIFO is not empty (145 bits); written with `fill` by `fetch`, `deliver` and `clear`.
    fill_set: [u64; 3],
    /// The text of each queue as parcels, decoded when it entered the IQ ([`Iq::parse`]).
    parsed: Vec<Parsed>,
    decode_cache: DecodeCache,
    /// `classify_items` of each queue's head parcel (`CycleRecs::none()` with no complete parcel), made when a parcel
    /// becomes the head: one record per queue, not one per parcel held.
    head_recs: Box<[CycleRecs]>,
    /// The hash of the fetched text, for the schedule replay (`driver::replay`).
    pub th: crate::driver::replay::TextHash,
    /// ECC on instruction text.
    pub tx: TextEcc,
}

/// One complete parcel of a queue's IQ: where it ends and how it dispatches. Its instructions are the first `n` of
/// the queue's item list (`Parsed::items`), in order of the parcels.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ParcelRec {
    units: u32,
    n: u32,
    nops: u64,
}

/// The parse of one queue's IQ, kept up to date as units enter and parcels leave: the complete parcels from the
/// head, then the parcel being assembled. Everything in it is a function of the units the IQ holds.
#[derive(Clone, Debug, Default)]
struct Parsed {
    parcels: VecDeque<ParcelRec>,
    /// The decoded instructions of the parcels in `parcels`, then of the one being assembled; the head parcel's
    /// start at `items_head`.
    items: Vec<Result<Insn, Diag>>,
    items_head: usize,
    /// Units, counted from the IQ head, covered by the complete parcels, and scanned so far.
    start: usize,
    pos: usize,
    /// The timing prefix and the instruction count of the parcel being assembled.
    nops: u64,
    pend_n: usize,
}

impl Parsed {
    fn reset(&mut self) {
        self.parcels.clear();
        self.items.clear();
        self.items_head = 0;
        self.start = 0;
        self.pos = 0;
        self.nops = 0;
        self.pend_n = 0;
    }
}

/// The head parcel of a queue as the IQ holds it (see [`Iq::head`]); what no parcel looks like is `complete: false`
/// with no units, instructions or records.
#[derive(Clone, Copy, Debug)]
pub struct HeadRef<'a> {
    pub complete: bool,
    pub units: usize,
    pub nops: u64,
    pub n: usize,
    pub items: &'a [Result<Insn, Diag>],
    pub recs: &'a CycleRecs,
}

static NO_RECS: CycleRecs = CycleRecs::none();

/// The head parcel of a queue with its instructions sorted into slots, found by scanning the units: the reference
/// that [`Iq::head`] is checked against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    pub parcel: Parcel,
    /// `classify_items` of the parcel, `CycleRecs::none()` when it is not complete.
    pub recs: CycleRecs,
}

impl Iq {
    pub fn new(g: &Geom) -> Iq {
        Iq {
            observe_sites: false,
            cap: g.iq_units,
            rate: g.iq_rate,
            latency: g.fetch_latency,
            staging_depth: g.fetch_staging_depth as usize,
            vlanes: g.vlanes as usize,
            ring: (0..QUEUES).map(|_| vec![0u32; g.iq_units].into_boxed_slice()).collect(),
            head: [0; QUEUES],
            len: [0; QUEUES],
            hi: [0; QUEUES],
            lo: [usize::MAX; QUEUES],
            last: [NO_DISPATCH; QUEUES],
            dry: [false; QUEUES],
            active: [0; QUEUES],
            image_len: [0; QUEUES],
            consumed: [0; QUEUES],
            taken_fetched: [false; QUEUES],
            fill: (0..QUEUES).map(|_| VecDeque::new()).collect(),
            fill_set: [0; 3],
            parsed: (0..QUEUES).map(|_| Parsed::default()).collect(),
            decode_cache: DecodeCache::new(),
            head_recs: vec![CycleRecs::none(); QUEUES].into_boxed_slice(),
            th: Default::default(),
            tx: TextEcc { prod: vec![0; QUEUES], wov: vec![Vec::new(); QUEUES], ..Default::default() },
        }
    }

    /// Keep the text-word tags of the rings (the program has `F` lines): called before the images are loaded.
    pub fn track_text(&mut self) {
        if !self.tx.track {
            self.tx.track = true;
            self.tx.tags = (0..QUEUES).map(|_| vec![-1; self.cap].into_boxed_slice()).collect();
        }
    }

    /// Source suffix of a parcel, in IQ order across ring wrap and bundle boundaries.
    pub(crate) fn site_sources(&self, q: usize, units: usize, image_units: usize) -> String {
        use std::fmt::Write;
        let base = (4 * image_units.div_ceil(4)) as i128;
        let size = (self.vlanes / 2) as i128;
        let mut sources = String::new();
        for i in 0..units {
            let tag = self.tx.tags[q][self.ring_slot(self.head[q] + i)];
            if tag < 0 {
                sources.push_str(" host");
            } else if tag < base {
                write!(sources, " image:{tag}").unwrap();
            } else {
                let offset = tag - base;
                write!(sources, " fetch:{}:{}", offset / size, offset % size).unwrap();
            }
        }
        sources
    }

    fn view(&self, q: usize) -> View<'_> {
        View {
            ring: &self.ring[q],
            tags: if self.tx.track { &self.tx.tags[q] } else { &[] },
            wov: &self.tx.wov[q],
            head: self.head[q],
            cap: self.cap,
            live: self.tx.live,
        }
    }

    /// The program has `F` lines: keep the tags, and tag the images already loaded (before cycle 0 the rings hold the
    /// images from slot 0).
    pub fn retag_images(&mut self) {
        self.track_text();
        for q in 0..QUEUES {
            let n = self.len[q];
            for k in 0..n {
                let slot = self.ring_slot(self.head[q] + k);
                self.tx.tags[q][slot] = k as i128;
            }
            self.tx.prod[q] = 4 * n.div_ceil(4) as u128;
        }
    }

    /// The text word of the unit at distance `i` from the head of queue `q`.
    pub fn unit_word(&self, q: usize, i: usize) -> Option<u128> {
        self.view(q).word(i)
    }

    /// The text width of the instruction at distance `pos` (1 or 2 units) and the words it touches, as the decoder sees
    /// the text.
    // model: postlude/iq.sail iq_text_check
    pub fn text_words(&self, q: usize, pos: usize) -> (usize, Option<u128>, Option<u128>) {
        let v = self.view(q);
        let w = text_width(v.unit(pos));
        (w, v.word(pos), if w == 2 { v.word(pos + 1) } else { None })
    }

    /// A correctable error of text word `w` was corrected: the data bit of the ring is fixed and the overlay dropped.
    // model: postlude/iq.sail iq_word_check
    pub fn text_fix(&mut self, q: usize, w: u128, e: u16) {
        if let Some(j) = crate::ecc::data_flip(e) {
            for i in 0..self.len[q] {
                let slot = self.ring_slot(self.head[q] + i);
                if self.tx.tags[q][slot] == (4 * w + u128::from(j / 32)) as i128 {
                    self.ring[q][slot] ^= 1 << (j % 32);
                }
            }
        }
        self.tx.set_ov(q, w, 0);
    }

    /// The text of queue `q` changed under the parse (an `F iq` flip): parse it again from the head.
    pub fn reparse_reset(&mut self, q: usize) {
        self.parsed[q].reset();
        self.head_recs[q] = CycleRecs::none();
    }

    /// `F iq`: flip bit `n` (0..0x89) of text word `w` of queue `q`; false when the word is not in the queue.
    // model: postlude/harness_api.sail model_flip_iq
    pub fn flip_text(&mut self, q: usize, w: u64, n: u32) -> bool {
        let w = u128::from(w);
        let at = (0..self.len[q]).find(|&i| self.view(q).word(i) == Some(w));
        if at.is_none() {
            return false;
        }
        if n < 128 {
            for i in 0..self.len[q] {
                let slot = self.ring_slot(self.head[q] + i);
                if self.tx.tags[q][slot] == (4 * w + u128::from(n / 32)) as i128 {
                    self.ring[q][slot] ^= 1 << (n % 32);
                }
            }
        }
        let e = self.tx.ov(q, w) ^ crate::ecc::flip_delta(n);
        self.tx.set_ov(q, w, e);
        self.tx.live = true;
        true
    }

    /// One unit of the initial image of queue `q`, `idx` its index in the image (padding dropped).
    // model: slices/cu/cu_iq.sail iq_reset
    pub fn push_image(&mut self, q: usize, u: u32, idx: usize) -> bool {
        self.push_with_source(q, u, idx as i128)
    }

    /// The image of queue `q` is loaded: `n` units; the next text word starts after its last (partial) word.
    pub fn image_done(&mut self, q: usize, n: usize) {
        self.tx.prod[q] = 4 * n.div_ceil(4) as u128;
        self.image_len[q] = n;
    }

    /// Did the parcel last taken from queue `q`'s IQ begin after the object's own image: text that a fetch brought from
    /// SRAM or that the host injected, whose units decode to whatever those words are?
    pub fn taken_fetched(&self, q: usize) -> bool {
        self.taken_fetched[q]
    }

    /// Empty every queue (`iq_reset` before the images are loaded).
    pub fn clear(&mut self) {
        self.head = [0; QUEUES];
        self.len = [0; QUEUES];
        self.hi = [0; QUEUES];
        self.lo = [usize::MAX; QUEUES];
        self.last = [NO_DISPATCH; QUEUES];
        self.dry = [false; QUEUES];
        self.active = [0; QUEUES];
        self.image_len = [0; QUEUES];
        self.consumed = [0; QUEUES];
        self.taken_fetched = [false; QUEUES];
        for f in &mut self.fill {
            f.clear();
        }
        self.fill_set = [0; 3];
        for p in &mut self.parsed {
            p.reset();
        }
        self.head_recs.fill(CycleRecs::none());
        self.th.h = 0;
        self.tx.live = false;
        self.tx.prod.fill(0);
        for w in &mut self.tx.wov {
            w.clear();
        }
    }

    pub(crate) fn rebuild_after_edit(&mut self, isa: &Isa) -> Result<(), String> {
        if self.fill.len() != QUEUES || self.len.iter().any(|&n| n > self.cap) {
            return Err("state edit left an invalid IQ shape".into());
        }
        self.fill_set = [0; 3];
        for q in 0..QUEUES {
            if !self.fill[q].is_empty() {
                self.fill_set[q >> 6] |= 1 << (q & 63);
            }
            self.parsed[q].reset();
            self.head_recs[q] = CycleRecs::none();
            self.parse(isa, q);
        }
        Ok(())
    }

    /// The queues (0..=144) whose fill FIFO holds a payload, as a bit set.
    pub fn fill_queues(&self) -> [u64; 3] {
        self.fill_set
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn rate(&self) -> usize {
        self.rate
    }

    fn ring_slot(&self, x: usize) -> usize {
        if x >= self.cap {
            x - self.cap
        } else {
            x
        }
    }

    /// The i-th unit from the head (`i < len <= cap`).
    // model: slices/cu/cu_iq.sail iq_peek
    pub fn peek(&self, q: usize, i: usize) -> u32 {
        self.ring[q][self.ring_slot(self.head[q] + i)]
    }

    // model: slices/cu/cu_iq.sail iq_pop
    pub fn pop(&mut self, q: usize, n: usize) {
        self.taken_fetched[q] = self.consumed[q] >= self.image_len[q];
        self.consumed[q] += n;
        self.head[q] = self.ring_slot(self.head[q] + n);
        self.len[q] = self.len[q].saturating_sub(n);
        self.lo[q] = self.lo[q].min(self.len[q]);
        let p = &mut self.parsed[q];
        // The driver takes whole parcels; any other cut leaves the parse to start again from the head.
        match p.parcels.front() {
            Some(f) if f.units as usize == n => {
                p.items_head += f.n as usize;
                p.start -= n;
                p.pos -= n;
                p.parcels.pop_front();
                if p.parcels.is_empty() && p.pend_n == 0 {
                    p.items.clear();
                    p.items_head = 0;
                } else if p.items_head >= 4096 {
                    p.items.drain(..p.items_head);
                    p.items_head = 0;
                }
                let r = &mut self.head_recs[q];
                *r = CycleRecs::none();
                if let Some(f) = p.parcels.front() {
                    classify_into(q, &p.items[p.items_head..p.items_head + f.n as usize], r);
                }
            }
            _ => {
                p.reset();
                self.head_recs[q] = CycleRecs::none();
            }
        }
    }

    /// Append one unit; `false` (the unit is lost) when the IQ already holds `iq_depth` bytes.
    // model: slices/cu/cu_iq.sail iq_push
    pub fn push(&mut self, q: usize, u: u32) -> bool {
        self.push_tagged(q, u, NO_WORD)
    }

    /// [`Iq::push`] of a unit with its production index (`NO_WORD`: none).
    // model: slices/cu/cu_iq.sail iq_push_tagged
    pub fn push_tagged(&mut self, q: usize, u: u32, tag: u32) -> bool {
        self.push_with_source(q, u, if tag == NO_WORD { -1 } else { i128::from(tag) })
    }

    fn push_with_source(&mut self, q: usize, u: u32, tag: i128) -> bool {
        if self.len[q] >= self.cap {
            return false;
        }
        let slot = self.ring_slot(self.head[q] + self.len[q]);
        self.ring[q][slot] = u;
        if self.tx.track {
            self.tx.tags[q][slot] = tag;
        }
        self.len[q] += 1;
        self.hi[q] = self.hi[q].max(self.len[q]);
        true
    }

    /// The fill rate in cycle `c`: up to `iq_rate` units from the head of the FIFO, from payloads
    /// that are ready. Padding is dropped (it still takes its place in the rate). Returns
    /// (a unit was enqueued, a unit was lost to a full IQ).
    // model: slices/cu/cu_iq.sail iq_deliver
    pub fn deliver(&mut self, q: usize, c: impl TryInto<u64>) -> (bool, bool) {
        let c = c.try_into().ok().expect("delivering an admitted u64 cycle");
        let mut budget = self.rate;
        let (mut pushed, mut lost) = (false, false);
        while budget > 0 {
            let Some(seg) = self.fill[q].front_mut() else { break };
            if seg.ready > u128::from(c) {
                break;
            }
            match seg.units.pop_front() {
                None => {
                    self.fill[q].pop_front();
                }
                Some(u) => {
                    budget = budget.saturating_sub(1);
                    let (base, taken) = (seg.base, seg.taken);
                    seg.taken += 1;
                    if u != PADDING {
                        let ov = if base >= 0 { seg.ovw.get(taken / 4).copied().unwrap_or(0) } else { 0 };
                        let tag = if base >= 0 { base + taken as i128 } else { -1 };
                        if self.push_with_source(q, u, tag) {
                            pushed = true;
                            self.active[q] = c;
                            if ov != 0 {
                                self.tx.set_ov(q, (base as u128 + taken as u128) / 4, ov);
                            }
                        } else {
                            lost = true;
                        }
                    }
                }
            }
        }
        // Drop an emptied head payload.
        if self.fill[q].front().is_some_and(|s| s.units.is_empty()) {
            self.fill[q].pop_front();
        }
        if self.fill[q].is_empty() {
            self.fill_set[q >> 6] &= !(1 << (q & 63));
        }
        (pushed, lost)
    }

    /// A fetch whose payload is complete at cycle `t` on queue `q` (an Ifetch: its second vector is read at its
    /// execute cycle + 1; an IfetchM: its second SRAM read at execute cycle + 1): payload `a` (lane 0 first) then `b`; each holds `vlanes`
    /// bytes. `false`, with no effect, when the queue's fill FIFO already holds `icu.fetch_staging_depth` payloads
    /// (OTS-E-IQ-003).
    // model: slices/cu/cu_iq.sail iq_fetch
    pub fn fetch(&mut self, q: usize, t: u64, a: &[u8], b: &[u8]) -> bool {
        self.fetch_ov(q, t, a, b, Vec::new())
    }

    /// [`Iq::fetch`] with the ECC overlays of the payload's text words (`ovw`: those of `a` then of `b`, a superlane
    /// each; empty when all are valid).
    // model: slices/cu/cu_iq.sail iq_fetch_ov
    pub(crate) fn can_fetch(&self, q: usize) -> bool {
        self.fill[q].len() < self.staging_depth
    }

    pub fn fetch_ov(&mut self, q: usize, t: u64, a: &[u8], b: &[u8], ovw: Vec<u16>) -> bool {
        if self.fill[q].len() >= self.staging_depth {
            return false;
        }
        self.th.note(q, t, a, b);
        let n = self.vlanes / 4;
        let mut words = Vec::with_capacity(2 * n);
        for v in [a, b] {
            words.extend(v[..4 * n].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
        }
        let base = self.tx.prod[q] as i128;
        self.tx.prod[q] += words.len() as u128;
        let units = VecDeque::from(words);
        self.fill[q].push_back(FillSeg { ready: u128::from(t) + u128::from(self.latency), units, base, taken: 0, ovw });
        self.fill_set[q >> 6] |= 1 << (q & 63);
        true
    }

    /// `H start c`: the queues start dispatching at cycle `c`, as if every queue had last dispatched in cycle `c - 1`.
    // model: postlude/harness_api.sail model_set_start
    pub fn set_start(&mut self, c: u64) {
        self.last = [i128::from(c) - 1; QUEUES]; // c = 0: the cycle before cycle 0, NO_DISPATCH
    }

    /// `H inject c queue units`: one unit of host text for queue `q`, entering the IQ from cycle `c` on through
    /// Enqueue, with the overflow rule of the IQ (OTS-E-IQ-002). The units of one line follow each other in order; a
    /// payload that is the last of the fill FIFO and ready at `c` takes the unit.
    // model: postlude/harness_api.sail model_inject_unit
    pub fn inject(&mut self, q: usize, c: impl TryInto<u64>, u: u32) {
        let c = c.try_into().ok().expect("injecting an admitted u64 cycle");
        match self.fill[q].back_mut() {
            Some(seg) if seg.base < 0 && seg.ready == u128::from(c) => seg.units.push_back(u),
            _ => self.fill[q].push_back(FillSeg {
                ready: u128::from(c),
                units: VecDeque::from([u]),
                base: -1,
                taken: 0,
                ovw: Vec::new(),
            }),
        }
        self.fill_set[q >> 6] |= 1 << (q & 63);
    }

    /// The text left in queue `q`'s IQ can never form a parcel: after leading NOPs (a timing prefix with nothing
    /// to time) it holds the first unit of a two-unit instruction, or instructions up to one whose bundle bit is
    /// set with nothing after it, or a complete parcel that never dispatched. Used once, at the end of the run.
    // model: postlude/iq.sail iq_stranded
    pub fn stranded(&self, isa: &Isa, q: usize) -> bool {
        let len = self.len[q];
        let v = self.view(q);
        let (mut pos, mut n, mut left, mut done) = (0usize, 0usize, false, false);
        while !done {
            if pos >= len {
                break;
            }
            let u0 = v.unit(pos);
            let w = text_width(u0);
            if w == 2 && pos + 1 >= len {
                left = true;
                break;
            }
            let r = v.decode(isa, pos, w);
            let bundle = u0 >> 31 == 1;
            let nop = match r {
                Ok(Insn::NOP { n: k }) if !bundle && n == 0 => k as usize,
                _ => 0,
            };
            pos += w;
            if nop == 0 {
                n += 1;
                left = true;
                if !bundle {
                    done = true;
                }
            }
        }
        left
    }
}

/// The head parcel of a queue: the leading NOPs (a timing prefix: NOPs with bundle bit 0 before
/// the first instruction), then the instructions up to and including the first one with bundle bit
/// 0. `complete`: every unit of the parcel is in the IQ; `units`: the units it takes from the IQ
/// (the prefix included); `nops`: the sum of the prefix; `items`: the instructions, decoded.
// model: postlude/iq.sail parcel
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Parcel {
    pub complete: bool,
    pub units: usize,
    pub nops: u64,
    pub n: usize,
    pub items: Arc<[Result<Insn, Diag>]>,
}

/// Number of units of the instruction that starts with `u` (1 or 2).
// model: postlude/iq.sail text_width
pub fn text_width(u: u32) -> usize {
    match (u >> 28) & 7 {
        1 | 2 | 4 => 2,
        _ => 1,
    }
}

/// The instruction that starts with `u0` (and `u1`, for a two-unit instruction).
// model: postlude/iq.sail decode_text
pub fn decode_text(isa: &Isa, u0: u32, u1: u32) -> Result<Insn, Diag> {
    let units = [u0, u1];
    isa.decode_text(&units[..text_width(u0)]).map(|(i, _)| i)
}

/// One entry of the catalogue that can match a word, with the three masks the decoder tests, copied out of the
/// large `Entry` rows so that the scan touches one small array.
#[derive(Clone, Copy)]
struct Cand {
    mask: u64,
    fixed: u64,
    /// The bits that are neither fixed nor an operand: reserved, must be zero.
    reserved: u64,
    idx: u16,
    /// All candidates through this one have profile-independent operand bounds.
    cacheable: bool,
}

/// The entries of the catalogue by the ten bits below the bundle bit of a word of each width (`KEY_BITS`), in
/// catalogue order, so that the first match of a scan of the bucket is the first match of a scan of the catalogue.
struct Buckets {
    /// `(start, len)` in `cands`, index `1024 * (width == 64) + key`.
    span: Vec<(u32, u32)>,
    cands: Vec<Cand>,
}

const KEY_BITS: u32 = 10;

fn key_of(width: u8, w: u64) -> usize {
    let top = if width == 64 { 62 } else { 30 };
    (w >> (top + 1 - KEY_BITS)) as usize & ((1 << KEY_BITS) - 1)
}

fn buckets() -> &'static Buckets {
    static B: std::sync::OnceLock<Buckets> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let mut span = Vec::new();
        let mut cands = Vec::new();
        for width in [32u8, 64] {
            let top = if width == 64 { 62 } else { 30 };
            let shift = top + 1 - KEY_BITS;
            let all = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
            for k in 0..1usize << KEY_BITS {
                let start = cands.len() as u32;
                let mut cacheable = true;
                for (idx, e) in ots_isa::ENTRIES.iter().enumerate() {
                    let (m, f) = ((e.mask >> shift) as usize & ((1 << KEY_BITS) - 1), (e.fixed >> shift) as usize);
                    if e.width == width && (k ^ f) & m == 0 {
                        cacheable &= e.vars.iter().all(|v| v.max_expr.is_none());
                        cands.push(Cand {
                            cacheable,
                            mask: e.mask,
                            fixed: e.fixed,
                            reserved: all & !e.mask & !e.var_mask,
                            idx: idx as u16,
                        });
                    }
                }
                span.push((start, cands.len() as u32 - start));
            }
        }
        Buckets { span, cands }
    })
}

/// Fixed-field forms of the two common direct SRAM instructions.
fn decode_direct_sram(isa: &Isa, w: u64) -> Option<Insn> {
    // The fixed match bits of Read and Write are equal except for bit 54.
    if w & 0xff9f_0000_03ff_ffff != 0x1000_0000_0000_0000 {
        return None;
    }
    let write = w & (1 << 54) != 0;
    let id = if write { ots_isa::EntryId::Write } else { ots_isa::EntryId::Read };
    let addr = ((w >> 32) & 0xffff) as u32;
    let max_addr = isa.max_of(&id.entry().vars[0])?;
    if i64::from(addr) > max_addr {
        return None;
    }
    let stream = ((w >> 26) & 0x3f) as u32;
    let thread = ((w >> 53) & 1) as u32;
    Some(if write { Insn::Write { addr, src: stream, thread } } else { Insn::Read { addr, dst: stream, thread } })
}

/// [`Isa::decode`] of a word of `width` bits: the first entry, in catalogue order, whose fixed bits match, whose
/// reserved bits are zero and whose operands are all legal. Only the entries of the word's bucket are tried.
fn decode_word_fast(isa: &Isa, width: u8, w: u64) -> Option<(Insn, bool)> {
    if width == 64 {
        if let Some(insn) = decode_direct_sram(isa, w) {
            return Some((insn, false)); // its address bound belongs to this profile
        }
    }
    let b = buckets();
    let (start, len) = b.span[(width == 64) as usize * (1 << KEY_BITS) + key_of(width, w)];
    for c in &b.cands[start as usize..(start + len) as usize] {
        if w & c.mask != c.fixed || w & c.reserved != 0 {
            continue;
        }
        let e = &ots_isa::ENTRIES[c.idx as usize];
        let mut f = [0u32; ots_isa::MAX_VARS];
        for (i, v) in e.vars.iter().enumerate() {
            f[i] = v.extract(w);
        }
        if isa.fields_legal(e, &f) {
            return Some((Insn::from_fields(e.id, &f), c.cacheable));
        }
    }
    None
}

/// For each tag of a word's first unit: 1 narrow, 2 wide, 0 neither (the narrow tags take precedence, as in `Isa::decode_text`).
fn tag_kind() -> &'static [u8; 8] {
    static K: std::sync::OnceLock<[u8; 8]> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        std::array::from_fn(|t| {
            if ots_isa::NARROW_TAGS.contains(&(t as u8)) {
                1
            } else if ots_isa::WIDE_TAGS.contains(&(t as u8)) {
                2
            } else {
                0
            }
        })
    })
}

/// [`decode_text`] without the scan of the whole catalogue (the model's decode rule of `Isa::decode_text`, with
/// [`decode_word_fast`] for the word); the result is the same for every pair of units.
pub fn decode_text_fast(isa: &Isa, u0: u32, u1: u32) -> Result<Insn, Diag> {
    decode_with_policy(isa, u0, u1).map(|(insn, _)| insn)
}

fn decode_with_policy(isa: &Isa, u0: u32, u1: u32) -> Result<(Insn, bool), Diag> {
    let u0 = u0 & 0x7FFF_FFFF;
    let tag = ((u0 >> 28) & 7) as usize;
    let (width, w) = match tag_kind()[tag] {
        1 => (32, u0 as u64),
        2 => (64, (u0 as u64) << 32 | u1 as u64),
        _ => return Err(Diag::Dec003),
    };
    decode_word_fast(isa, width, w).ok_or(Diag::Dec003)
}

/// Recently executed encodings recur across queues and instruction fetches. Store only
/// successful, profile-independent decodes. ECC is checked on every instruction fetch.
#[derive(Debug)]
struct DecodeCache {
    slots: Box<[(u64, Insn)]>,
}

impl DecodeCache {
    const INDEX_BITS: u32 = 10;

    fn new() -> Self {
        // A real key has its top (bundle) bit clear.
        Self { slots: vec![(u64::MAX, Insn::NOP { n: 0 }); 1 << Self::INDEX_BITS].into_boxed_slice() }
    }

    #[inline]
    fn decode(&mut self, isa: &Isa, u0: u32, u1: u32) -> Result<Insn, Diag> {
        let first = u0 & 0x7fff_ffff;
        let key = (u64::from(first) << 32) | u64::from(u1);
        let hash = (first.rotate_left(13) ^ u1).wrapping_mul(0x9e37_79b9);
        let slot = &mut self.slots[(hash >> (32 - Self::INDEX_BITS)) as usize];
        if slot.0 == key {
            return Ok(slot.1);
        }
        let (insn, cacheable) = decode_with_policy(isa, first, u1)?;
        if cacheable {
            *slot = (key, insn);
        }
        Ok(insn)
    }
}

impl Iq {
    /// The head parcel of queue `q`; not complete (and empty) when the IQ does not hold one, found by scanning
    /// the units (the model's `parcel_scan`).
    // model: postlude/iq.sail parcel_scan
    pub fn parcel_scan(&self, isa: &Isa, q: usize) -> Parcel {
        self.scan_head(isa, q)
    }

    /// Decode and classify the units that entered queue `q`'s IQ since the last call (`push`, `deliver`, the
    /// image load), so that the head parcel is known without scanning. Every unit is decoded once.
    // model: postlude/iq.sail parcel_scan; postlude/tick.sail cycle_recs
    pub fn parse(&mut self, isa: &Isa, q: usize) {
        let len = self.len[q];
        let v = View {
            ring: &self.ring[q],
            tags: if self.tx.track { &self.tx.tags[q] } else { &[] },
            wov: &self.tx.wov[q],
            head: self.head[q],
            cap: self.cap,
            live: self.tx.live,
        };
        let st = &mut self.parsed[q];
        while st.pos < len {
            let u0 = v.unit(st.pos);
            let w = text_width(u0);
            if w == 2 && st.pos + 1 >= len {
                break;
            }
            let m = crate::driver::phase::begin();
            let r = if v.unc(st.pos, w) {
                Err(Diag::Ecc002)
            } else {
                let u1 = if w == 2 { v.unit(st.pos + 1) } else { 0 };
                self.decode_cache.decode(isa, u0, u1)
            };
            crate::driver::phase::end(crate::driver::phase::DEC, m);
            let bundle = u0 >> 31 == 1;
            let nop = match r {
                Ok(Insn::NOP { n }) if !bundle && st.pend_n == 0 => n as u64,
                _ => 0,
            };
            st.pos += w;
            if nop > 0 {
                st.nops += nop;
            } else {
                st.items.push(r);
                st.pend_n += 1;
                if !bundle {
                    let from = st.items.len() - st.pend_n;
                    let m = crate::driver::phase::begin();
                    if st.parcels.is_empty() {
                        let r = &mut self.head_recs[q];
                        *r = CycleRecs::none();
                        classify_into(q, &st.items[from..], r);
                    }
                    st.parcels.push_back(ParcelRec {
                        units: (st.pos - st.start) as u32,
                        nops: st.nops,
                        n: st.pend_n as u32,
                    });
                    crate::driver::phase::end(crate::driver::phase::CLS, m);
                    st.start = st.pos;
                    st.nops = 0;
                    st.pend_n = 0;
                }
            }
        }
    }

    /// The head parcel of queue `q` and its slots, from the parse ([`Iq::parse`] has run since the last push).
    #[inline]
    pub fn head(&self, q: usize) -> HeadRef<'_> {
        let p = &self.parsed[q];
        match p.parcels.front() {
            Some(f) => HeadRef {
                complete: true,
                units: f.units as usize,
                nops: f.nops,
                n: f.n as usize,
                items: &p.items[p.items_head..p.items_head + f.n as usize],
                recs: &self.head_recs[q],
            },
            None => HeadRef { complete: false, units: 0, nops: 0, n: 0, items: &[], recs: &NO_RECS },
        }
    }

    /// `(complete, units, nops)` of queue `q`'s head: what a cycle needs of a queue whose parcel is not due yet.
    pub fn head_brief(&self, q: usize) -> (bool, usize, u64) {
        match self.parsed[q].parcels.front() {
            Some(f) => (true, f.units as usize, f.nops),
            None => (false, 0, 0),
        }
    }

    /// The head of queue `q` scanned afresh: what [`Iq::head`] must return for the units the IQ holds. Equality of
    /// the two is the check that the parse follows every push and pop.
    pub fn head_uncached(&self, isa: &Isa, q: usize) -> Head {
        let parcel = self.scan_head(isa, q);
        let recs = if parcel.complete { classify_items(q, &parcel.items) } else { CycleRecs::none() };
        Head { parcel, recs }
    }

    /// [`Iq::head`] of queue `q` is the head a fresh scan finds.
    pub fn head_matches_scan(&self, isa: &Isa, q: usize) -> bool {
        let (h, f) = (self.head(q), self.head_uncached(isa, q));
        h.complete == f.parcel.complete
            && h.units == f.parcel.units
            && h.nops == f.parcel.nops
            && h.n == f.parcel.n
            && h.items == &f.parcel.items[..]
            && *h.recs == f.recs
    }

    fn scan_head(&self, isa: &Isa, q: usize) -> Parcel {
        let len = self.len[q];
        let v = self.view(q);
        let (mut pos, mut nops) = (0usize, 0u64);
        let mut items: Vec<Result<Insn, Diag>> = Vec::new();
        let mut complete = false;
        loop {
            if pos >= len {
                break;
            }
            let u0 = v.unit(pos);
            let w = text_width(u0);
            if w == 2 && pos + 1 >= len {
                break;
            }
            let r = v.decode(isa, pos, w);
            let bundle = u0 >> 31 == 1;
            let nop = match r {
                Ok(Insn::NOP { n }) if !bundle && items.is_empty() => n as u64,
                _ => 0,
            };
            pos += w;
            if nop > 0 {
                nops += nop;
            } else {
                items.push(r);
                if !bundle {
                    complete = true;
                    break;
                }
            }
        }
        if complete {
            Parcel { complete: true, units: pos, nops, n: items.len(), items: items.into() }
        } else {
            Parcel::default()
        }
    }
}

/// The earliest dispatch cycle of a complete head parcel of queue `q`: one cycle after the
/// queue's last dispatch, plus the NOPs before it (design sec. 7.3).
// model: postlude/iq.sail parcel_earliest
pub fn parcel_earliest(iq: &Iq, q: usize, p: &Parcel) -> u128 {
    earliest_after_nops(iq, q, p.nops)
}

/// [`parcel_earliest`] of a head parcel known only by its timing prefix.
pub fn earliest_after_nops(iq: &Iq, q: usize, nops: u64) -> u128 {
    // The explicit pre-zero coordinate gives cycle 0. A future cycle beyond u64::MAX
    // remains exact so the run budget can reject it before execution.
    u128::try_from(iq.last[q] + 1).expect("last dispatch or explicit pre-zero coordinate") + u128::from(nops)
}

/// `Iq::last` of a queue that has not dispatched: the explicit mathematical coordinate before cycle 0.
pub const NO_DISPATCH: i128 = -1;
