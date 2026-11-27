//! The stream register file (`core/streams.sail`): two rows (E, W) x 32 streams x 45 registers.
//!
//! Flow rule (`microarchitecture.md` §2): a value written at gap g with commit cycle c is in
//! register g + 1 at cycle c + 1 and moves one register per cycle in its row's flow direction; it
//! falls off after register 44. A reader at gap G in cycle t sees register G. Hence the value is
//! visible at G exactly when G > g and t - c == G - g.
//!
//! Representation (the model's): d = g - c (the "diagonal") is constant while a value flows. Each
//! (row, stream) keeps, per diagonal slot d mod 64, the writes on that diagonal, at most one per
//! gap. A diagonal is live for at most 44 cycles, well below 64 even with a `d_skew_max` margin,
//! so slots never alias (proof obligation P-SRF-3). Reading picks the writer with the largest gap
//! below the reader (the nearest upstream producer replaces older values from its gap on).
//!
//! ISS storage: each gap's 64 entries are allocated on the first write to that gap,
//! and an entry holds an id into a value arena (`vlanes` bytes per value, recycled 64 cycles after
//! the commit that wrote it: no read can reach an older write), so memory follows use. Reads and writes behave as the Sail vectors do,
//! aliasing of slots included.
use crate::ecc::MAX_SL;
use crate::geom::{MAX_LANES, STREAMS_PER_ROW};
use crate::layout::RowId;
use crate::machine::Machine;
use crate::observe::{Observer, Row};
use std::collections::VecDeque;

/// One stream value: one byte per lane, lane 0 first; lanes at or beyond `vlanes` are zero
/// (`vec` of `core/types.sail`).
pub type Vec320<const L: usize = { MAX_LANES }> = [u8; L];

/// `v` as a vector of the widest width: its bytes first, zeros after (the interface of the links and the host is the widest width).
pub fn widen<const L: usize>(v: &[u8; L]) -> Vec320 {
    let mut out = [0u8; MAX_LANES];
    out[..L].copy_from_slice(v);
    out
}

/// The first `L` bytes of a vector of the widest width (the lanes of a machine of `L` lanes; the rest are zero there).
pub fn narrow<const L: usize>(v: &Vec320) -> Vec320<L> {
    let mut out = [0u8; L];
    out.copy_from_slice(&v[..L]);
    out
}

/// Registers per row's slot count: diagonals modulo 64.
const SLOTS: usize = 64;
/// A slot holds one write per gap 0..43.
const GAPS: usize = 44;
/// "no value": an entry that is not valid.
const NO_VALUE: u32 = u32::MAX;

/// `srf_lookup`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrfLookup {
    Found,
    None,
    Upstream,
}

/// `srf_write` of the model, minus the gap (the entry's index) and the value (an arena id).
#[derive(Clone, Copy, Debug)]
struct Write {
    /// `valid` and the value in one: [`NO_VALUE`] when the entry was never written.
    value: u32,
    /// Actual commit cycle, distinct over the full admitted u64 domain.
    cycle: u64,
}

impl Write {
    fn diagonal(self, gap: usize) -> i128 {
        gap as i128 - i128::from(self.cycle)
    }
}

const EMPTY: Write = Write { value: NO_VALUE, cycle: 0 };

/// The writes of one gap, one entry per diagonal slot: the entries of one gap are adjacent, so the writes of
/// consecutive cycles to one gap (the usual traffic of a stream) share cache lines.
type GapRow = [Write; SLOTS];
/// The writes of one stream, and per slot the set of gaps whose entry holds a value (bit `w` of `occ[slot]`: the
/// entry of gap `w` is not [`NO_VALUE`]), which a read scans instead of the 44 entries.
#[derive(Clone, Debug)]
struct Stream {
    gaps: [Option<Box<GapRow>>; GAPS],
    occ: [u64; SLOTS],
}

impl Stream {
    // Allocate the stream index on first use. Its gap rows remain unallocated.
    #[cold]
    #[inline(never)]
    fn new() -> Box<Stream> {
        Box::new(Stream { gaps: std::array::from_fn(|_| None), occ: [0; SLOTS] })
    }

    #[inline]
    fn entry(&self, gap: usize, slot: usize) -> Write {
        self.gaps[gap].as_deref().map_or(EMPTY, |row| row[slot])
    }
}

/// Cycles after its commit at which a value is recycled. A write at gap `w` is read at gap `G` in cycle `t` only when
/// `t - c == G - w`, so only within `GAPS - 1` = 43 cycles of its commit `c` (the chained read: one), and the machine
/// reads at the cycle of its newest commit, so 44 would be exact (mutants 40 and 43 fail the tests; 44 fails only the
/// two random tests that read behind the newest commit). The margin of 20 cycles covers a reader behind the newest
/// commit, as those tests' are (up to 8 cycles).
const RECYCLE_AFTER: u64 = 64;

/// Values written to the file: `vlanes` bytes each. A value is recycled [`RECYCLE_AFTER`] cycles after the commit that
/// wrote it, in the order of the writes, so the working set is the commits of the last 64 cycles (a few dozen
/// vectors, hot in the cache) instead of every value an entry still holds.
#[derive(Debug)]
struct Arena {
    vlanes: usize,
    bytes: Vec<u8>,
    free: Vec<u32>,
    /// (commit cycle, id) of the values in use, oldest write first.
    used: VecDeque<(u64, u32)>,
}

impl Arena {
    /// Store `v`, written by the commit of cycle `c`. Values committed at least [`RECYCLE_AFTER`] cycles before `c` are
    /// first released (a read cannot reach them any more); a stale entry that still holds the id is never matched
    /// (matching needs the diagonal, which fixes the commit cycle within 43 of the reader's).
    fn alloc(&mut self, c: u64, v: &[u8]) -> u32 {
        while let Some(&(ac, id)) = self.used.front() {
            if u128::from(ac) + u128::from(RECYCLE_AFTER) > u128::from(c) {
                break;
            }
            self.free.push(id);
            self.used.pop_front();
        }
        let vl = self.vlanes;
        let id = match self.free.pop() {
            Some(id) => {
                crate::copy::copy(&mut self.bytes[id as usize * vl..][..vl], &v[..vl]);
                id
            }
            None => {
                let id = (self.bytes.len() / vl) as u32;
                self.bytes.extend_from_slice(&v[..vl]);
                id
            }
        };
        self.used.push_back((c, id));
        id
    }

    fn get_ref(&self, id: u32) -> &[u8] {
        let vl = self.vlanes;
        &self.bytes[id as usize * vl..][..vl]
    }

    fn get_mut(&mut self, id: u32) -> &mut [u8] {
        let vl = self.vlanes;
        &mut self.bytes[id as usize * vl..][..vl]
    }
}

/// One spare-bit error report (`srf_rep` of the model): the spare bits of streams 0..28 written at one register of a row in
/// one cycle; `sp[s]` has bit x set for superlane x of stream s, `pr[x]` is the priority of the error whose auxiliary
/// streams 4..28 hold superlane x (Gather 3, Scatter 2, bank conflict 1).
// model: core/streams.sail srf_rep
#[derive(Clone, Debug)]
struct Rep {
    rw: bool,
    diag: i128,
    wgap: u8,
    sp: [u32; 30],
    pr: [i8; MAX_SL],
}

/// `srf_e` and `srf_w`.
#[derive(Debug)]
pub struct Srf {
    /// The reports (a ring of 256 like the model's `srf_reps`) and the number written.
    reps: Vec<Rep>,
    rep_n: usize,
    rows: [Vec<Option<Box<Stream>>>; 2],
    arena: Arena,
    /// The ECC overlay ids (`crate::ecc`) of the writes whose check bits are not valid, by (row, stream, gap, slot): a
    /// write that is not here has valid check bits. Empty unless a fault was injected, so the write path of a fault-free
    /// run only tests it for emptiness.
    ovs: std::collections::BTreeMap<(u8, u8, u8, u16), u32>,
}

/// The mathematical diagonal g - t. Signed128-bit arithmetic represents every u64 cycle
/// and the negative pre-cycle read coordinates without wrapping expired writes into liveness.
fn diag(g: u8, t: i128) -> i128 {
    g as i128 - t
}

/// `slot_of`: `emod_int(d, 64)`.
// model: core/streams.sail slot_of
fn slot_of(d: i128) -> usize {
    d.rem_euclid(SLOTS as i128) as usize
}

fn row_index(r: RowId) -> usize {
    match r {
        RowId::E => 0,
        RowId::W => 1,
    }
}

impl Srf {
    /// An empty file for vectors of `vlanes` lanes.
    pub fn new(vlanes: usize) -> Srf {
        assert!((1..=MAX_LANES).contains(&vlanes), "vlanes {vlanes}");
        Srf {
            reps: Vec::new(),
            rep_n: 0,
            rows: [vec![None; STREAMS_PER_ROW], vec![None; STREAMS_PER_ROW]],
            arena: Arena { vlanes, bytes: Vec::new(), free: Vec::new(), used: VecDeque::new() },
            ovs: Default::default(),
        }
    }

    // model: core/streams.sail srf_reset
    pub fn reset(&mut self) {
        let vlanes = self.arena.vlanes;
        *self = Srf::new(vlanes);
    }

    /// Streams written so far (a test hook for the lazy allocation).
    pub fn streams_allocated(&self) -> usize {
        self.rows.iter().flatten().filter(|s| s.is_some()).count()
    }

    /// Values held (a test hook for the arena).
    pub fn values_live(&self) -> usize {
        self.arena.bytes.len() / self.arena.vlanes - self.arena.free.len()
    }

    // model: core/streams.sail get_stream
    fn get_stream(&self, r: RowId, s: usize) -> Option<&Stream> {
        self.rows[row_index(r)][s].as_deref()
    }

    /// Record a write; no callback (see [`Machine::srf_write`]). Returns true on a same-cycle
    /// collision (OTS-E-SRF-002): another write on the same row, stream and gap with the same
    /// diagonal (hence the same cycle).
    // model: core/streams.sail srf_write
    pub fn write(&mut self, r: RowId, s: u8, g: u8, c: u64, v: &[u8]) -> bool {
        self.write_ov(r, s, g, c, v, 0)
    }

    /// [`Srf::write`] of a value with ECC overlay `ov`.
    // model: core/streams.sail srf_write_ov
    pub fn write_ov(&mut self, r: RowId, s: u8, g: u8, c: u64, v: &[u8], ov: u32) -> bool {
        let d = diag(g, c as i128);
        let id = self.arena.alloc(c, v);
        let stream = self.rows[row_index(r)][s as usize].get_or_insert_with(Stream::new);
        let slot = slot_of(d);
        let row = stream.gaps[g as usize].get_or_insert_with(|| Box::new([EMPTY; SLOTS]));
        let old = row[slot];
        let collide = old.value != NO_VALUE && old.diagonal(g as usize) == d;
        row[slot] = Write { value: id, cycle: c };
        stream.occ[slot] |= 1u64 << g;
        if ov != 0 {
            self.ovs.insert((row_index(r) as u8, s, g, slot_of(d) as u16), ov);
        } else if !self.ovs.is_empty() {
            self.ovs.remove(&(row_index(r) as u8, s, g, slot_of(d) as u16));
        }
        collide
    }

    /// True when stream `s` on row `r` has at least one LIVE write at cycle `t` and all live writes
    /// are at gap >= `g` (none upstream of a reader at `g`). A write on diagonal d is live while it
    /// is still in the row: its register d + t is at most 44 (it falls off after register 44).
    /// Writes that have left the row do not count either way.
    // model: core/streams.sail srf_only_downstream
    pub fn only_downstream(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> bool {
        let t = t.into();
        let Some(stream) = self.get_stream(r, s as usize) else { return false };
        let (mut any, mut upstream) = (false, false);
        for (w, row) in stream.gaps.iter().enumerate() {
            let Some(row) = row.as_deref() else { continue };
            for e in row.iter() {
                if e.value != NO_VALUE && e.diagonal(w) + t <= 44 {
                    any = true;
                    if (w as u8) < g {
                        upstream = true;
                    }
                }
            }
        }
        any && !upstream
    }

    /// The value seen by a reader at gap `g` in cycle `t`, or why there is none. `Upstream` (a
    /// direction error, OTSP-CON-UPSTREAM) when the stream has live writes on this row but every one
    /// of them is at or downstream of the reader, so no value of it can ever reach the reader;
    /// `None` (a timing error) otherwise. This is the flow rule: every read, a VXM read included,
    /// takes the nearest upstream write.
    // model: core/streams.sail srf_read
    pub fn read(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> (SrfLookup, Vec320) {
        let t = t.into();
        self.read_lanes(r, s, g, t)
    }

    /// [`Srf::read`] into a vector of `L` lanes (`L` of at least `vlanes`).
    pub fn read_lanes<const L: usize>(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> (SrfLookup, Vec320<L>) {
        let t = t.into();
        match self.read_ref(r, s, g, t) {
            (SrfLookup::Found, Some(v)) => {
                let mut out = [0u8; L];
                crate::copy::copy(&mut out[..v.len()], v);
                (SrfLookup::Found, out)
            }
            (l, _) => (l, [0; L]),
        }
    }

    /// [`Srf::read`] without the copy: the `vlanes` bytes of the value in the arena (`Some` exactly when `Found`).
    pub fn read_ref(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> (SrfLookup, Option<&[u8]>) {
        let t = t.into();
        let d = diag(g, t);
        if let Some(stream) = self.get_stream(r, s as usize) {
            let k = slot_of(d);
            // the gaps below the reader that hold a value, nearest first; the entry of the diagonal is the first match
            let mut m = stream.occ[k] & ((1u64 << (g as usize).min(GAPS)) - 1);
            while m != 0 {
                let w = 63 - m.leading_zeros() as usize;
                let e = stream.entry(w, k);
                if e.diagonal(w) == d {
                    return (SrfLookup::Found, Some(self.arena.get_ref(e.value)));
                }
                m &= !(1u64 << w);
            }
        }
        if self.only_downstream(r, s, g, t) {
            (SrfLookup::Upstream, None)
        } else {
            (SrfLookup::None, None)
        }
    }

    /// The entry a reader at gap `g` in cycle `t` takes (the nearest upstream write of the diagonal; with `chain` the
    /// write at the reader's own gap in cycle t - 1): its gap, its arena id and its overlay.
    fn entry_at(&self, r: RowId, s: u8, g: u8, t: i128, chain: bool) -> Option<(usize, Write)> {
        let stream = self.get_stream(r, s as usize)?;
        if chain {
            let d = diag(g, t) + 1;
            let e = stream.entry(g as usize, slot_of(d));
            return (e.value != NO_VALUE && e.diagonal(g as usize) == d).then_some((g as usize, e));
        }
        let d = diag(g, t);
        let k = slot_of(d);
        let mut m = stream.occ[k] & ((1u64 << (g as usize).min(GAPS)) - 1);
        while m != 0 {
            let w = 63 - m.leading_zeros() as usize;
            let e = stream.entry(w, k);
            if e.diagonal(w) == d {
                return Some((w, e));
            }
            m &= !(1u64 << w);
        }
        None
    }

    /// Record a report written in cycle `c` at gap `g` of row `r` for the superlanes in `xs` (`srf_report`): S[0..2] and
    /// the auxiliary streams from `sp` (indexed by stream), the parity S[3] from the merged S[0..2]. In a superlane that
    /// already holds a report of this register, the auxiliary streams of the higher priority win.
    // model: core/streams.sail srf_report
    /// The result is true when a report that was still in its row at cycle `c` (register `diag + c <= 44`) was evicted from the
    /// 256 the array keeps (lint OTS-W-RAS-001, ; the caller raises it).
    pub fn report(&mut self, r: RowId, g: u8, c: impl TryInto<i128>, prio: i8, xs: u32, sp: &[u32; 30]) -> bool {
        let c = c.try_into().ok().expect("bounded future cycle fits signed128");
        let d = diag(g, c);
        let rw = r == RowId::W;
        let at = self.reps.iter().position(|e| e.rw == rw && e.diag == d && e.wgap == g);
        let mut e = match at {
            Some(i) => self.reps[i].clone(),
            None => Rep { rw, diag: d, wgap: g, sp: [0; 30], pr: [0; MAX_SL] },
        };
        for x in 0..MAX_SL {
            if xs >> x & 1 == 0 {
                continue;
            }
            let (keep, take) = (e.pr[x] > prio, e.pr[x] < prio);
            #[allow(clippy::needless_range_loop)]
            for s in 0..30usize {
                let bit = sp[s] & (1 << x);
                if s < 3 {
                    e.sp[s] |= bit;
                } else if s >= 4 {
                    if take {
                        e.sp[s] = e.sp[s] & !(1 << x) | bit;
                    } else if !keep {
                        e.sp[s] |= bit;
                    }
                }
            }
            let n3 = (e.sp[0] >> x & 1) + (e.sp[1] >> x & 1) + (e.sp[2] >> x & 1);
            e.sp[3] = e.sp[3] & !(1 << x) | (n3 & 1) << x;
            if prio > e.pr[x] {
                e.pr[x] = prio;
            }
        }
        match at {
            Some(i) => {
                self.reps[i] = e;
                false
            }
            None => {
                let mut evicted = false;
                if self.reps.len() < 256 {
                    self.reps.push(e);
                } else {
                    let k = self.rep_n % 256;
                    evicted = self.reps[k].diag + c <= 44;
                    self.reps[k] = e;
                }
                self.rep_n += 1;
                evicted
            }
        }
    }

    /// The spare bits a reader at gap `g` in cycle `t` of row `r` sees on stream `s`, a bit per superlane (`srf_spare`): the
    /// reports of its diagonal written below its gap, unless a data write of the diagonal at a gap between the
    /// report's and the reader's replaced the register.
    // model: core/streams.sail srf_spare
    pub fn spare(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> u32 {
        let t = t.into();
        if self.reps.is_empty() || s >= 30 {
            return 0;
        }
        let d = diag(g, t);
        let rw = r == RowId::W;
        let stream = self.get_stream(r, s as usize);
        let mut m = 0u32;
        for e in &self.reps {
            if e.rw == rw && e.diag == d && e.wgap < g {
                let replaced = stream.is_some_and(|st| {
                    let k = slot_of(d);
                    (e.wgap as usize + 1..g as usize).any(|w| {
                        let x = st.entry(w, k);
                        x.value != NO_VALUE && x.diagonal(w) == d
                    })
                });
                if !replaced {
                    m |= e.sp[s as usize];
                }
            }
        }
        m
    }

    /// S[0..3] of superlane `x` as a reader at gap `g` of row `r` sees them in cycle `t` (`srf_s03`).
    // model: core/streams.sail srf_s03
    pub fn s03(&self, r: RowId, g: u8, t: impl Into<i128>, x: usize) -> u32 {
        let t = t.into();
        (0..4u8).map(|s| (self.spare(r, s, g, t) >> x & 1) << s).sum()
    }

    /// The ECC overlay id of the value a read at gap `g` in cycle `t` takes (0 when there is none).
    pub fn ov_at(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>, chain: bool) -> u32 {
        let t = t.into();
        if self.ovs.is_empty() {
            return 0;
        }
        self.entry_at(r, s, g, t, chain).map_or(0, |(w, e)| self.ov_of(r, s, w, e.diagonal(w)))
    }

    fn ov_of(&self, r: RowId, s: u8, w: usize, diag: i128) -> u32 {
        self.ovs.get(&(row_index(r) as u8, s, w as u8, slot_of(diag) as u16)).copied().unwrap_or(0)
    }

    /// `srf_flip` (`F srf`): flip bit `n` (0..127 data, 128..136 check bits, 137 spare) of superlane `sl` of the value
    /// register `gap` of stream `s` of row `r` (`reg` = (r, s, gap)) holds at cycle `c`, the nearest upstream write of the diagonal; `ov_of` gives the new overlay id
    /// from the old one. Nothing happens when the register holds no value.
    // model: core/streams.sail srf_flip
    pub fn flip(&mut self, c: u64, reg: (RowId, u8, u8), sl: usize, n: u32, ov_of: impl FnOnce(u32) -> u32) {
        let (r, s, gap) = reg;
        let Some((w, e)) = self.entry_at(r, s, gap, c as i128, false) else { return };
        if n < 128 {
            let lane = 16 * sl + n as usize / 8;
            let v = self.arena.get_mut(e.value);
            if lane < v.len() {
                v[lane] ^= 1 << (n % 8);
            }
        }
        let old = self.ov_of(r, s, w, e.diagonal(w));
        let new = ov_of(old);
        let key = (row_index(r) as u8, s, w as u8, slot_of(e.diagonal(w)) as u16);
        if new != 0 {
            self.ovs.insert(key, new);
        } else {
            self.ovs.remove(&key);
        }
    }

    /// A CHAINED read (VXM ALU chaining, `microarchitecture.md` §5): the value written AT the reader's own gap `g` in cycle t - 1, and nothing else. No
    /// such write is `None` (a timing error). Only a VXM operand whose chain bit is set reads this
    /// way; the flow rule (`read`) never sees a write at the reader's own gap.
    // model: core/streams.sail srf_read_chain
    pub fn read_chain(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> (SrfLookup, Vec320) {
        let t = t.into();
        self.read_chain_lanes(r, s, g, t)
    }

    /// [`Srf::read_chain`] into a vector of `L` lanes.
    pub fn read_chain_lanes<const L: usize>(
        &self,
        r: RowId,
        s: u8,
        g: u8,
        t: impl Into<i128>,
    ) -> (SrfLookup, Vec320<L>) {
        let t = t.into();
        match self.read_chain_ref(r, s, g, t) {
            Some(v) => {
                let mut out = [0u8; L];
                crate::copy::copy(&mut out[..v.len()], v);
                (SrfLookup::Found, out)
            }
            None => (SrfLookup::None, [0; L]),
        }
    }

    /// [`Srf::read_chain`] without the copy.
    pub fn read_chain_ref(&self, r: RowId, s: u8, g: u8, t: impl Into<i128>) -> Option<&[u8]> {
        let t = t.into();
        let stream = self.get_stream(r, s as usize)?;
        let d = diag(g, t) + 1;
        let e = stream.entry(g as usize, slot_of(d));
        (e.value != NO_VALUE && e.diagonal(g as usize) == d).then(|| self.arena.get_ref(e.value))
    }
}

impl From<Row> for RowId {
    fn from(r: Row) -> RowId {
        match r {
            Row::East => RowId::E,
            Row::West => RowId::W,
        }
    }
}

impl From<RowId> for Row {
    fn from(r: RowId) -> Row {
        match r {
            RowId::E => Row::East,
            RowId::W => Row::West,
        }
    }
}

impl<const L: usize> Machine<L> {
    /// Record a write and tell the observer (`srf_write_callback`: cycle, row, stream, gap, and
    /// the committed value). Returns true on a same-cycle collision.
    // model: core/streams.sail srf_write
    pub fn srf_write<O: Observer + ?Sized>(&mut self, r: RowId, s: u8, g: u8, c: u64, v: &[u8], obs: &mut O) -> bool {
        self.srf_write_ov(r, s, g, c, v, 0, obs)
    }

    /// [`Machine::srf_write`] of a value with ECC overlay `ov`.
    // model: core/streams.sail srf_write_ov
    #[allow(clippy::too_many_arguments)]
    pub fn srf_write_ov<O: Observer + ?Sized>(
        &mut self,
        r: RowId,
        s: u8,
        g: u8,
        c: u64,
        v: &[u8],
        ov: u32,
        obs: &mut O,
    ) -> bool {
        let collide = self.srf.write_ov(r, s, g, c, v, ov);
        let vl = self.geom.vlanes as usize;
        obs.srf(c, r.into(), s, g, &v[..vl]);
        collide
    }
}
