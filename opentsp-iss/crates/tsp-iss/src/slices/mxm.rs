//! MXM instructions LW, IW, IWS, ABC, ACC (`slices/mxm/mxm_insts.sail`, `mxm_regs.sail`,
//! `mxm_types.sail`; ISCA 2020 sec. III-D; `microarchitecture.md` sec. 6).
//!
//! Independent planes: `mxm.planes_per_hemisphere` per hemisphere (0, 1 west; 2, 3 east with two). Each has a weight
//! buffer filled by LW, installed weights (IW, IWS), the products of its ABCs and `mxm.acc_depth` accumulators. State is
//! usable only once the stage that made it has committed (`ready_*`, in dispatch cycles).
//!
//! Start-of-cycle state: an instruction reads the plane as it was at the start of
//! its execute cycle. Its changes are STAGED ([`Machine::mxm_stage`]) and applied when the cycle
//! ends (`unit_state_flush`, `driver::flush`). A plane has independently written parts (the
//! buffer, the weights, the ABC products and each accumulator); two instructions of different
//! queues that change the same part in one cycle both raise OTS-E-RES-003 and neither has any effect (a plane with
//! `mxm.fp8_rate` 2 accepts two FP8 ABCs in one cycle).
//!
//! The model stages a copy of the whole plane for a part; the ISS stages the part's fields only
//! (`Value`). Big parts (the weight rows, the buffer, a product, an accumulator) are `Arc`s, so a
//! staged copy and the unit state share storage until one of them changes.
use super::pool::Pool;
use super::{Exec, Issue};
use crate::geom::MAX_LANES;
use crate::observe::Observer;
use crate::operands::{words_to_planes_lanes, Group16, Planes};
use crate::srf::Vec320;
use crate::Machine;
use ots_isa::{Code as Diag, Insn};
use std::sync::{Arc, Weak};
use tsp_numerics::fp::{self, FMT32, FMT_E4M3, FMT_E5M2};
use tsp_numerics::mxm::{
    bf16_dot_prepared, bfp8_exact, dsum_round, f16_dot_prepared, f16_dot_prepared_bounded, f16_exact,
    f16_row_max_exponent, fp8_exact, int8_dot, Bf16Act, Bf16Weights, DSum, F16Act, Fp8Act, Wide,
};

/// `mxm_wtype`: weight / activation element type (HC34 p.26: INT8, UINT8, FP16; FP8, : stored as bytes like i8).
// model: slices/mxm/mxm_types.sail mxm_wtype
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WType {
    I8,
    F16,
    U8,
    Fp8,
    /// Block-scaled fp8 (extension block_fp8): a stream pair of e4m3 data and block scales.
    Bfp8,
    /// bfloat16, a profile extension: two byte planes per element and a plane pair, as f16.
    Bf16,
}

impl WType {
    /// The types whose rows and activations are a stream pair (f16, bf16 and bfp8).
    // model: slices/mxm/mxm_insts.sail is_pair
    pub fn is_pair(self) -> bool {
        matches!(self, WType::F16 | WType::Bf16 | WType::Bfp8)
    }
}

// model: slices/mxm/mxm_insts.sail wtype_split
fn wtype(code: u32) -> WType {
    match code {
        0 => WType::I8,
        1 => WType::F16,
        2 => WType::U8,
        3 => WType::Fp8,
        4 => WType::Bf16,
        5 => WType::Bfp8,
        _ => unreachable!("MXM type code {code} is not legal"),
    }
}

/// One weight row per stream value: i8 uses one byte plane per row, f16 two (lo, hi).
type Rows = Arc<Vec<Vec320>>;

/// The buffer or weight rows whose stream value had an uncorrectable superlane when it was read, as sorted row
/// numbers (`mxm_plane.buf_bad`, `w_bad`; empty when nothing was ever faulted, which allocates nothing to share).
// model: slices/mxm/mxm_regs.sail mxm_plane buf_bad, w_bad
type Bad = Arc<Vec<u16>>;

/// `bad` cut to its first `keep` rows, then rows `keep + i` added for each bit `i` of `add` (the 16 rows of a load).
fn bad_append(bad: &Bad, keep: usize, add: u16) -> Bad {
    if add == 0 && bad.last().is_none_or(|&r| (r as usize) < keep) {
        return bad.clone();
    }
    let mut v: Vec<u16> = bad.iter().copied().filter(|&r| (r as usize) < keep).collect();
    v.extend((0..16u16).filter(|i| add >> i & 1 == 1).map(|i| keep as u16 + i));
    Arc::new(v)
}

/// A 32-bit word per lane.
pub type Words = [u32; MAX_LANES];

/// One ABC result: dispatched at `dispatch`, committed at `commit`.
// model: slices/mxm/mxm_regs.sail abc_entry
#[derive(Clone, Debug)]
pub struct AbcEntry {
    pub dispatch: u64,
    pub commit: u128,
    pub ptype: WType,
    /// The queue that dispatched the ABC.
    pub q: usize,
    pub product: Arc<Words>,
    /// The exact products of a floating-point ABC, for the wide accumulator: `None` unless the profile declares wide_acc and the
    /// ABC has a wide form, so a profile without the extension pays no allocation and no reference count for them.
    // model: slices/mxm/mxm_regs.sail abc_entry wide
    pub wide: Option<Arc<Vec<Wide>>>,
    /// The superlanes whose lanes are poisoned (`abc_entry.poison`): an uncorrectable activation poisons every
    /// superlane, an uncorrectable weight row `r` the superlane `r / 16`.
    pub poison: u32,
}

/// The last exact integer product of this plane. All architectural history
/// and error reporting remain outside the cache.
#[derive(Clone, Debug)]
struct Int8Recent {
    weights: Weak<Vec<Vec320>>,
    rows: usize,
    weights_signed: bool,
    input_signed: bool,
    activation: Box<[u8]>,
    product: Arc<Words>,
}

impl Int8Recent {
    fn get(&self, p: &Plane, ws: bool, xs: bool, input: &[u8]) -> Option<Arc<Words>> {
        (std::ptr::eq(self.weights.as_ptr(), Arc::as_ptr(&p.weights))
            && self.rows == p.rows
            && self.weights_signed == ws
            && self.input_signed == xs
            && self.activation.as_ref() == input)
            .then(|| self.product.clone())
    }
}

/// BF16 coefficients for one installed immutable matrix and active aperture.
pub(crate) struct Bf16Cache {
    weights: Weak<Vec<Vec320>>,
    rows: usize,
    lanes: usize,
    packed: Bf16Weights,
}

impl Bf16Cache {
    fn new(weights: &Rows, rows: usize, lanes: usize) -> Self {
        Self {
            weights: Arc::downgrade(weights),
            rows,
            lanes,
            packed: Bf16Weights::new(weights.as_slice(), rows, lanes),
        }
    }

    fn matches(&self, weights: &Rows, rows: usize, lanes: usize) -> bool {
        self.rows == rows && self.lanes == lanes && std::ptr::eq(self.weights.as_ptr(), Arc::as_ptr(weights))
    }
}

#[cfg(test)]
#[path = "../../tests/private/bf16_weight_cache.rs"]
mod bf16_weight_cache_tests;

/// One plane's state.
// model: slices/mxm/mxm_regs.sail mxm_plane
#[derive(Clone, Debug)]
pub struct Plane {
    /// Type of the buffered rows.
    pub btype: WType,
    /// Stream values held in the weight buffer (`buf.len()`; at most `2 * mxm.f16_rows`).
    pub buf_rows: usize,
    pub buf: Rows,
    pub buf_bad: Bad,
    /// Cycle the last LW committed.
    pub ready_lw: u128,
    /// Type of the installed weights.
    pub wtype: WType,
    /// Installed rows (i8: streams; f16: streams / 2).
    pub rows: usize,
    pub weights: Rows,
    pub w_bad: Bad,
    /// For f16 weights, the highest exponent of each installed row. This
    /// cache retains the matrix allocation so overwrites cannot alias it.
    pub row_exponents: Option<(Rows, Arc<Vec<u8>>)>,
    int8_recent: Option<Int8Recent>,
    /// A stream install (IW from streams) is in progress.
    pub iw_open: bool,
    /// Dispatch cycle of the last IW from streams.
    pub iws_cycle: u64,
    /// Cycle the last IW committed.
    pub ready_iw: u128,
    /// ABC results, newest first (ACC picks by commit cycle).
    pub abc: Vec<AbcEntry>,
    /// The accumulators (`mxm.acc_depth`).
    pub acc: Vec<Arc<Words>>,
    /// The superlanes of each accumulator that hold poison (`acc_poison`).
    pub acc_poison: Vec<u32>,
    /// The wide accumulators (extension wide_acc), one per accumulator; a lane past the end of the vector is zero.
    // model: slices/mxm/mxm_regs.sail mxm_plane acc128
    pub acc128: Vec<Arc<Vec<Wide>>>,
}

impl Plane {
    // model: slices/mxm/mxm_regs.sail empty_plane
    fn empty(acc_depth: usize) -> Plane {
        let zero: Arc<Words> = Arc::new([0; MAX_LANES]);
        Plane {
            btype: WType::I8,
            buf_rows: 0,
            buf: Arc::new(Vec::new()),
            buf_bad: Arc::new(Vec::new()),
            ready_lw: 0,
            wtype: WType::I8,
            rows: 0,
            weights: Arc::new(Vec::new()),
            w_bad: Arc::new(Vec::new()),
            row_exponents: None,
            int8_recent: None,
            iw_open: false,
            iws_cycle: 0,
            ready_iw: 0,
            abc: Vec::new(),
            acc: vec![zero; acc_depth],
            acc_poison: vec![0; acc_depth],
            acc128: (0..acc_depth).map(|_| Arc::new(Vec::new())).collect(),
        }
    }
}

/// The independently written parts of a plane.
// model: slices/mxm/mxm_regs.sail mxm_part
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Buf,
    W,
    Abc,
    /// Accumulator `a`.
    Acc(u8),
}

/// The new value of one part: the fields of `mxm_plane` the part owns.
#[derive(Clone, Debug)]
pub enum Value {
    Buf {
        btype: WType,
        buf_rows: usize,
        buf: Rows,
        bad: Bad,
        ready_lw: u128,
    },
    /// An LW: the buffer cut to its first `keep` rows, then the 16 rows of the load (`Buf`, without the copy of the
    /// buffer): the result is the same, built when the cycle ends.
    BufAppend {
        btype: WType,
        keep: usize,
        rows: Box<Group16>,
        /// The streams of the load that were uncorrectable (`ecc_g16`).
        bad: u16,
        ready_lw: u128,
    },
    W {
        wtype: WType,
        rows: usize,
        weights: Rows,
        bad: Bad,
        iw_open: bool,
        iws_cycle: u64,
        ready_iw: u128,
    },
    /// An IWS that appends to the matrix visible at the start of the cycle.
    WAppend {
        wtype: WType,
        keep: usize,
        rows: Box<Group16>,
        bad: u16,
        iws_cycle: u64,
        ready_iw: u128,
    },
    Abc(Vec<AbcEntry>),
    Acc(usize, Arc<Words>, u32),
    /// A write of the wide accumulator.
    Acc128(usize, Arc<Vec<Wide>>, u32),
}

impl Value {
    fn part(&self) -> Part {
        match self {
            Value::Buf { .. } | Value::BufAppend { .. } => Part::Buf,
            Value::W { .. } | Value::WAppend { .. } => Part::W,
            Value::Abc(_) => Part::Abc,
            Value::Acc(a, ..) | Value::Acc128(a, ..) => Part::Acc(*a as u8),
        }
    }
}

/// `mxm_staged`: a change staged by `queue` to part `part` of `plane`.
#[derive(Clone, Debug)]
pub struct Staged {
    pub q: usize,
    pub plane: usize,
    pub value: Value,
}

/// The MXM registers: the planes and the changes staged in the current cycle.
#[derive(Clone, Debug)]
pub struct Mxm {
    pub planes: Vec<Plane>,
    /// `mxm.fp8_rate`: the FP8 ABCs a plane accepts in one cycle.
    pub fp8_rate: u64,
    /// The faults the harness injected (`F mxm`) and no ABC has consumed yet, in injection order.
    // model: slices/mxm/mxm_regs.sail mxm_fault_list
    pub faults: Vec<MxmFault>,
    /// In staging order (oldest first); the model's list is newest first.
    pub staged: Vec<Staged>,
}

/// A flip of bit `bit` of the exact sum of lane `lane` of the ABC that executes on `plane` in `cycle` (`F mxm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MxmFault {
    pub plane: u8,
    pub lane: u16,
    pub bit: u8,
    pub cycle: u64,
}

impl Mxm {
    /// `F mxm`: a fault for the next ABC of its plane.
    // model: postlude/harness_api.sail model_inject_mxm
    pub fn inject(&mut self, f: MxmFault) {
        self.faults.push(f);
    }

    /// The plane and cycle of the fault no ABC consumed, as the harness reports it after the cycle (`model_mxm_pending`):
    /// the earliest cycle, and in it the plane of the last line.
    // model: postlude/harness_api.sail model_mxm_pending
    pub fn pending(&self) -> Option<(u8, u64)> {
        let c = self.faults.first()?.cycle;
        self.faults.iter().rev().find(|f| f.cycle == c).map(|f| (f.plane, c))
    }

    // model: slices/mxm/mxm_regs.sail mxm_reset
    /// The registers of the reference profile: 4 planes, 2 accumulators, FP8 at the i8 rate.
    pub fn new() -> Mxm {
        Mxm::with(2, 2, 1)
    }

    /// The registers of a profile with `planes_per_hemisphere` planes per hemisphere and `acc_depth` accumulators.
    pub fn with(planes_per_hemisphere: usize, acc_depth: usize, fp8_rate: u64) -> Mxm {
        Mxm {
            planes: (0..2 * planes_per_hemisphere).map(|_| Plane::empty(acc_depth)).collect(),
            fp8_rate,
            staged: Vec::new(),
            faults: Vec::new(),
        }
    }
}

impl Default for Mxm {
    fn default() -> Mxm {
        Mxm::new()
    }
}

impl Mxm {
    // model: slices/mxm/mxm_regs.sail mxm_apply
    fn apply(&mut self, e: Staged) {
        let p = &mut self.planes[e.plane];
        match e.value {
            Value::Buf { btype, buf_rows, buf, bad, ready_lw } => {
                p.btype = btype;
                p.buf_rows = buf_rows;
                p.buf = buf;
                p.buf_bad = bad;
                p.ready_lw = ready_lw;
            }
            Value::BufAppend { btype, keep, rows, bad, ready_lw } => {
                p.btype = btype;
                p.buf_rows = keep + rows.len();
                p.buf_bad = bad_append(&p.buf_bad, keep, bad);
                p.ready_lw = ready_lw;
                if keep == 0 {
                    // no rows kept: a fresh buffer, never a copy of rows that an installed matrix may share
                    let mut b = Vec::with_capacity(MAX_LANES);
                    b.extend_from_slice(&rows[..]);
                    p.buf = Arc::new(b);
                } else {
                    let b = Arc::make_mut(&mut p.buf);
                    b.truncate(keep);
                    b.extend_from_slice(&rows[..]);
                }
            }
            Value::W { wtype, rows, weights, bad, iw_open, iws_cycle, ready_iw } => {
                p.wtype = wtype;
                p.rows = rows;
                p.weights = weights;
                p.w_bad = bad;
                p.row_exponents = None;
                p.iw_open = iw_open;
                p.iws_cycle = iws_cycle;
                p.ready_iw = ready_iw;
            }
            Value::WAppend { wtype, keep, rows, bad, iws_cycle, ready_iw } => {
                p.wtype = wtype;
                p.rows = installed_rows(wtype, keep + rows.len());
                // A changed installed matrix invalidates the cached exponent scan.
                p.row_exponents = None;
                if keep == 0 {
                    let mut weights = Vec::with_capacity(MAX_LANES);
                    weights.extend_from_slice(&rows[..]);
                    p.weights = Arc::new(weights);
                } else {
                    let weights = Arc::make_mut(&mut p.weights);
                    weights.truncate(keep);
                    weights.extend_from_slice(&rows[..]);
                }
                p.w_bad = bad_append(&p.w_bad, keep, bad);
                p.iw_open = true;
                p.iws_cycle = iws_cycle;
                p.ready_iw = ready_iw;
            }
            Value::Abc(mut abc) => {
                // a second entry of the same dispatch cycle from another queue goes in front of the list as it stands
                match (abc.first(), p.abc.first()) {
                    (Some(n), Some(h)) if n.dispatch == h.dispatch && n.q != h.q => p.abc.insert(0, abc.swap_remove(0)),
                    _ => p.abc = abc,
                }
            }
            Value::Acc(a, v, pm) => {
                p.acc[a] = v;
                p.acc_poison[a] = pm;
            }
            Value::Acc128(a, v, pm) => {
                p.acc128[a] = v;
                p.acc_poison[a] = pm;
            }
        }
    }

    // model: postlude/tick.sail mxm_clash
    fn clash(&self, e: &Staged) -> bool {
        let others =
            || self.staged.iter().filter(|o| o.q != e.q && o.plane == e.plane && o.value.part() == e.value.part());
        if e.value.part() == Part::Abc {
            // mxm.fp8_rate 2: a plane accepts two FP8 ABCs in one cycle; a third, or an ABC of another type, clashes
            let fp8 =
                |s: &Staged| matches!(&s.value, Value::Abc(a) if a.first().is_some_and(|n| n.ptype == WType::Fp8));
            let n = others().count();
            return n > 0 && !(self.fp8_rate == 2 && n == 1 && fp8(e) && others().all(fp8));
        }
        others().next().is_some()
    }

    /// The queues with a clashing MXM part (a set, 0..143).
    // model: postlude/tick.sail unit_state_flush (the MXM half of `bad`)
    pub fn clashing_queues(&self, bad: &mut [bool; 145]) {
        self.mark_clashing_queues(bad);
    }

    /// Mark this unit's collisions and report whether it found any.
    pub(crate) fn mark_clashing_queues(&self, bad: &mut [bool; 145]) -> bool {
        let mut collided = false;
        for e in &self.staged {
            if self.clash(e) {
                bad[e.q] = true;
                collided = true;
            }
        }
        collided
    }

    /// Apply the staged changes of the queues not in `bad`, the model's order (newest first, so
    /// that when one queue staged a part twice the oldest stands), and clear the stage list.
    // model: postlude/tick.sail unit_state_flush (mxm_apply loop, mxm_stage_list = [||])
    pub fn flush(&mut self, bad: &[bool; 145]) {
        let mut staged = std::mem::take(&mut self.staged);
        // An appended load is built from the plane as it stands when the cycle ends; when the same part is staged
        // more than once the later stagings overwrite it, so it is built now, from the plane as the cycle began.
        for i in 0..staged.len() {
            if let Value::BufAppend { btype, keep, rows, bad, ready_lw } = &staged[i].value {
                let (plane, part) = (staged[i].plane, Part::Buf);
                let shared =
                    staged.iter().enumerate().any(|(j, o)| j != i && o.plane == plane && o.value.part() == part);
                if shared {
                    let mut buf = self.planes[plane].buf.clone();
                    let b = Arc::make_mut(&mut buf);
                    b.truncate(*keep);
                    b.extend_from_slice(&rows[..]);
                    let bad = bad_append(&self.planes[plane].buf_bad, *keep, *bad);
                    let v = Value::Buf { btype: *btype, buf_rows: keep + rows.len(), buf, bad, ready_lw: *ready_lw };
                    staged[i].value = v;
                }
            }
            if let Value::WAppend { wtype, keep, rows, bad, iws_cycle, ready_iw } = &staged[i].value {
                let plane = staged[i].plane;
                let shared = staged
                    .iter()
                    .enumerate()
                    .any(|(j, other)| i != j && plane == other.plane && other.value.part() == Part::W);
                if shared {
                    // Each staged write sees the same prefix and poison state,
                    // including when the same queue issues two writes to this part.
                    let p = &self.planes[plane];
                    let mut weights = p.weights[..(*keep).min(p.weights.len())].to_vec();
                    weights.extend_from_slice(&rows[..]);
                    staged[i].value = Value::W {
                        wtype: *wtype,
                        rows: installed_rows(*wtype, *keep + rows.len()),
                        weights: Arc::new(weights),
                        bad: bad_append(&p.w_bad, *keep, *bad),
                        iw_open: true,
                        iws_cycle: *iws_cycle,
                        ready_iw: *ready_iw,
                    };
                }
            }
        }
        for e in staged.into_iter().rev() {
            if !bad[e.q] {
                self.apply(e);
            }
        }
    }
}

/// f16, bf16 and bfp8 take two stream values per element and a plane pair. // model: slices/mxm/mxm_insts.sail is_pair
fn is_pair(t: WType) -> bool {
    t.is_pair()
}

fn installed_rows(t: WType, streams: usize) -> usize {
    if is_pair(t) {
        streams / 2
    } else {
        streams
    }
}

// model: slices/mxm/mxm_insts.sail holds_f16
fn holds_f16(p: &Plane) -> bool {
    (p.rows > 0 && is_pair(p.wtype)) || (p.buf_rows > 0 && is_pair(p.btype))
}

// model: slices/mxm/mxm_insts.sail abc_after
fn abc_after(xs: &[AbcEntry], c: u64) -> bool {
    xs.first().is_some_and(|e| e.dispatch > c)
}

/// Number of live products carried into the next ABC history.
fn abc_retained(xs: &[AbcEntry], c: u64) -> usize {
    let Some(i) = xs.iter().position(|entry| entry.commit <= u128::from(c)) else {
        return xs.len();
    };
    // The first committed product's paired FP8 dispatch remains accessible.
    let paired = xs.get(i + 1).is_some_and(|partner| partner.dispatch == xs[i].dispatch);
    i + 1 + usize::from(paired)
}

/// The ABC result an ACC of accumulator `a` reads at cycle `c`: the newest one committed by `c`. Two FP8 ABCs that one
/// plane accepted in one cycle are two results of the same dispatch cycle: an even accumulator takes the one of the lower
/// queue, an odd accumulator the one of the higher queue.
// model: slices/mxm/mxm_insts.sail abc_at, abc_pair_pick
fn abc_at(xs: &[AbcEntry], c: u64, a: usize) -> Option<&AbcEntry> {
    let i = xs.iter().position(|e| e.commit <= u128::from(c))?;
    let e = &xs[i];
    match xs.get(i + 1) {
        Some(f) if f.dispatch == e.dispatch && f.commit <= u128::from(c) => {
            let (lo, hi) = if e.q < f.q { (e, f) } else { (f, e) };
            Some(if a.is_multiple_of(2) { lo } else { hi })
        }
        _ => Some(e),
    }
}

/// An f16 ABC holds the plane's MACCs for `timing.mxm_abc.occupancy_f16` cycles: an ABC on that plane, from any of
/// its queues, in the cycles after an f16 ABC while it holds the plane is OTS-E-RES-001. The occupancy is at most the
/// ABC's d_func, so an entry that still holds the plane has not committed and is still in the list.
// model: slices/mxm/mxm_insts.sail macc_busy_in, macc_busy
fn macc_busy(p: &Plane, issue_cycle: u64, occupancy: u64) -> bool {
    p.abc.iter().any(|e| is_pair(e.ptype) && e.dispatch < issue_cycle && issue_cycle < e.dispatch + occupancy)
}

/// f16 weights need an f16 ABC and the reverse, and so do FP8 weights an FP8 ABC (OTS-E-RES-001); the integer types mix
/// freely (the installed type gives the weight sign, the ABC type the activation sign).
// model: slices/mxm/mxm_insts.sail abc_types_ok
fn abc_types_ok(wt: WType, t: WType) -> bool {
    if is_pair(wt) || is_pair(t) || wt == WType::Fp8 || t == WType::Fp8 {
        wt == t
    } else {
        true
    }
}

/// Lanes `lo..lo + o.len()` of an ABC product: lane `r` is the dot product of weight row `r` (f16: of the row pair
/// `2r`, `2r + 1`, the low and high bytes) and the activation. `act` is the prepared activation of an f16 ABC.
#[allow(clippy::too_many_arguments)] // the operands of one ABC
fn abc_rows(
    weights: &[Vec320],
    vl: usize,
    ws: bool,
    xs: bool,
    x: &Planes,
    act: Option<&F16Act>,
    exponents: Option<&[u8]>,
    lo: usize,
    o: &mut [u32],
) {
    for (k, v) in o.iter_mut().enumerate() {
        let r = lo + k;
        *v = match act {
            // An f16 plane has at most `mxm.f16_rows` rows (two byte planes each).
            Some(act) => {
                let (wlo, whi) = (&weights[2 * r][..vl], &weights[2 * r + 1][..vl]);
                match exponents {
                    Some(s) => f16_dot_prepared_bounded(wlo, whi, act, s[r]),
                    None => f16_dot_prepared(wlo, whi, act),
                }
            }
            None => int8_dot(&weights[r][..vl], ws, &x[0][..vl], xs),
        };
    }
}

/// The ABC product of `rows` FP8 weight rows in the format `f` (`FMT_E4M3` or `FMT_E5M2`): lane `r` is the exact dot product of
/// row `r` and the activation, rounded once to f32; lanes past `rows` are zero.
// model: slices/mxm/mxm_insts.sail execute MXM_ABC8
pub fn abc_product_fp8(f: fp::Fmt, weights: &[Vec320], rows: usize, vl: usize, x: &Planes) -> Words {
    let mut out: Words = [0; MAX_LANES];
    let act = Fp8Act::new(f, &x[0][..vl]);
    for (r, v) in out.iter_mut().enumerate().take(rows.min(MAX_LANES)) {
        *v = act.dot(&weights[r][..vl]);
    }
    out
}

/// The ABC product of the floating-point types by the exact sums: the f32 product (the sum rounded once) and, when `keep`, the
/// exact sums as wide lanes (the wide accumulator's input; `None` when not kept). Lanes past `rows` are zero. `f`: the FP8 format of an ABC8.
// model: slices/mxm/mxm_insts.sail execute MXM_ABC (the exact forms)
fn abc_product_exact(
    t: WType,
    f: u32,
    weights: &[Vec320],
    rows: usize,
    vl: usize,
    x: &Planes,
    keep: bool,
) -> (Words, Option<Vec<Wide>>) {
    let mut out: Words = [0; MAX_LANES];
    let mut wide: Option<Vec<Wide>> = keep.then(|| vec![Wide::default(); MAX_LANES]);
    for r in 0..rows.min(MAX_LANES) {
        let d: DSum = match t {
            WType::Fp8 => fp8_exact(if f & 1 == 0 { FMT_E4M3 } else { FMT_E5M2 }, &weights[r][..vl], &x[0][..vl]),
            WType::F16 => f16_exact(&weights[2 * r][..vl], &weights[2 * r + 1][..vl], &x[0][..vl], &x[1][..vl]),
            WType::Bfp8 => bfp8_exact(&weights[2 * r][..vl], &weights[2 * r + 1][..vl], &x[0][..vl], &x[1][..vl]),
            _ => unreachable!("an exact product of an integer type"),
        };
        out[r] = dsum_round(&d);
        if let Some(w) = wide.as_mut() {
            w[r] = Wide::of_dsum(&d);
        }
    }
    (out, wide)
}

/// The ABC product of `rows` bf16 weight rows (row pairs `2r`, `2r + 1`, the low and high bytes): lane `r` is the exact dot
/// product of row `r` and the activation, rounded once to f32; lanes past `rows` are zero.
// model: slices/mxm/mxm_insts.sail abc_exec (W_BF16)
pub fn abc_product_bf16(weights: &[Vec320], rows: usize, vl: usize, x: &Planes) -> Words {
    let mut out: Words = [0; MAX_LANES];
    let act = Bf16Act::new(&x[0][..vl], &x[1][..vl]);
    for (r, v) in out.iter_mut().enumerate().take(rows.min(MAX_LANES)) {
        *v = bf16_dot_prepared(&weights[2 * r][..vl], &weights[2 * r + 1][..vl], &act);
    }
    out
}

fn prepare(t: WType, vl: usize, x: &Planes) -> Option<F16Act<'_>> {
    (t == WType::F16).then(|| F16Act::new(&x[0][..vl], &x[1][..vl]))
}

/// The ABC product of `rows` weight rows (f16: row pairs), lanes past `rows` zero: the serial loop. `ws` / `xs`: the
/// weight / activation lanes are signed (int8). [`abc_product_split`] must equal it.
// model: slices/mxm/mxm_insts.sail execute MXM_ABC
pub fn abc_product_serial(
    t: WType,
    weights: &[Vec320],
    rows: usize,
    vl: usize,
    ws: bool,
    xs: bool,
    x: &Planes,
) -> Words {
    let mut out: Words = [0; MAX_LANES];
    let act = prepare(t, vl, x);
    abc_rows(weights, vl, ws, xs, x, act.as_ref(), None, 0, &mut out[..rows.min(MAX_LANES)]);
    out
}

/// The output pointer of a split ABC: the chunks write disjoint ranges of it.
struct OutPtr(*mut u32);
impl OutPtr {
    fn at(&self, lo: usize) -> *mut u32 {
        // SAFETY: the caller keeps `lo` inside the product.
        unsafe { self.0.add(lo) }
    }
}
// SAFETY: a chunk writes only its own rows `[lo, hi)` of the product (see `split`).
unsafe impl Sync for OutPtr {}

#[allow(clippy::too_many_arguments)] // the operands of one ABC
fn split(
    weights: &[Vec320],
    rows: usize,
    vl: usize,
    ws: bool,
    xs: bool,
    x: &Planes,
    act: Option<&F16Act>,
    exponents: Option<&[u8]>,
    pool: &Pool,
    chunk_rows: usize,
) -> Words {
    let chunk_rows = chunk_rows.max(1);
    let mut out: Words = [0; MAX_LANES];
    let rows = rows.min(MAX_LANES);
    let optr = OutPtr(out.as_mut_ptr());
    pool.run(rows.div_ceil(chunk_rows), &|c| {
        let (lo, hi) = (c * chunk_rows, ((c + 1) * chunk_rows).min(rows));
        // SAFETY: chunk `c` runs once and owns rows `lo..hi` of `out`, which no other chunk touches and which outlives
        // `run`.
        let o = unsafe { std::slice::from_raw_parts_mut(optr.at(lo), hi - lo) };
        abc_rows(weights, vl, ws, xs, x, act, exponents, lo, o);
    });
    out
}

/// [`abc_product_serial`] with the rows cut into chunks of `chunk_rows` that `pool` runs on its threads. The output
/// lanes are independent and the reduction order inside a lane is that of the serial loop, so the result is equal bit
/// for bit whatever the chunk size and the pool.
#[allow(clippy::too_many_arguments)] // the operands of one ABC and the split
pub fn abc_product_split(
    t: WType,
    weights: &[Vec320],
    rows: usize,
    vl: usize,
    ws: bool,
    xs: bool,
    x: &Planes,
    pool: &Pool,
    chunk_rows: usize,
) -> Words {
    split(weights, rows, vl, ws, xs, x, prepare(t, vl, x).as_ref(), None, pool, chunk_rows)
}

/// Serial work below which a fork-join does not pay (about twice the median cost of a join of four threads, see
/// `task-45-report.md`), in ns.
const SPLIT_MIN_NS: usize = 3_000;

/// The ABC product, split over the shared pool when the estimated serial time is above [`SPLIT_MIN_NS`].
#[allow(clippy::too_many_arguments)] // the operands of one ABC
fn abc_product(
    t: WType,
    weights: &[Vec320],
    rows: usize,
    vl: usize,
    ws: bool,
    xs: bool,
    x: &Planes,
    exponents: Option<&[u8]>,
) -> Words {
    let pool = Pool::shared();
    let act = prepare(t, vl, x);
    // estimated ns per row (measured on this machine): an f16 row costs about 30 plus 6 per chunk of eight lanes holding a
    // non-zero activation; an int8 row about one per 64 lanes (a whole plane of 320 rows takes about 1.4 us, less than a fork)
    let per_row = act.as_ref().map_or((vl / 64).max(1), |a| 30 + 6 * a.chunk_count());
    if pool.width() > 1 && rows * per_row >= SPLIT_MIN_NS {
        split(weights, rows, vl, ws, xs, x, act.as_ref(), exponents, pool, rows.div_ceil(2 * pool.width()).max(4))
    } else {
        let mut out: Words = [0; MAX_LANES];
        abc_rows(weights, vl, ws, xs, x, act.as_ref(), exponents, 0, &mut out[..rows.min(MAX_LANES)]);
        out
    }
}

impl<const L: usize> Machine<L> {
    /// `mxm.rows`: stream values the weight buffer and an installed matrix hold.
    // model: core/geometry.sail mxm_rows
    fn mxm_rows(&self) -> usize {
        self.geom.mxm_rows.expect("profile.validate requires mxm.rows") as usize
    }

    /// Stage part `v` of plane `pl` (applied at the end of the cycle).
    // model: slices/mxm/mxm_regs.sail mxm_stage
    fn mxm_stage(&mut self, ctx: &Issue, pl: usize, v: Value) {
        self.mxm.staged.push(Staged { q: ctx.queue, plane: pl, value: v });
    }

    /// `mxm.f16_rows`: rows of an f16 matrix.
    // model: core/geometry.sail mxm_f16_rows
    fn mxm_f16_rows(&self) -> usize {
        self.geom.mxm_f16_rows as usize
    }

    /// The stream values a weight buffer or an installed matrix of type `t` holds: `mxm.rows` i8, u8 or fp8 rows, or two
    /// per f16 or bf16 row.
    // model: slices/mxm/mxm_insts.sail mxm_cap
    fn mxm_cap(&self, t: WType) -> usize {
        if is_pair(t) {
            2 * self.mxm_f16_rows()
        } else {
            self.mxm_rows()
        }
    }

    /// The plane of queue `q` (0 .. 2 * `mxm.planes_per_hemisphere` - 1).
    // model: slices/mxm/mxm_insts.sail plane
    fn plane_of(&self, q: usize) -> usize {
        crate::layout::mxm_plane(&self.geom, q)
    }

    /// An odd plane has no f16 of its own: f16 runs on the even plane of the pair, and while that
    /// plane holds f16 weights or buffered f16 rows its odd partner is busy. A pair exists only in a hemisphere with two
    /// planes: with one plane per hemisphere no plane runs f16.
    // model: slices/mxm/mxm_insts.sail plane_blocked
    fn plane_blocked(&self, pl: usize, t: Option<WType>) -> bool {
        let paired = self.geom.mxm_planes == 2;
        if t.is_some_and(is_pair) && !paired {
            return true;
        }
        let odd = paired && (pl == 1 || pl == 3);
        odd && (t.is_some_and(is_pair) || holds_f16(&self.mxm.planes[if pl == 1 { 0 } else { 2 }]))
    }

    /// The stream operand of a 16-stream field (`stream_group16`: the field holds the stream / 16).
    fn group16(src: u32) -> u32 {
        src << 4
    }

    // model: slices/mxm/mxm_insts.sail execute MXM_LW
    fn mxm_lw(&mut self, ctx: &Issue, t: WType, src: u32) -> Exec {
        let pl = self.plane_of(ctx.queue);
        if self.plane_blocked(pl, Some(t)) {
            return Exec::Diagnostic(Diag::Res001);
        }
        let rows = match self.read_planes16_boxed(ctx, Self::group16(src), ctx.exec_cycle() as i128) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(r) => r,
        };
        let p = &self.mxm.planes[pl];
        if p.buf_rows > 0 && p.btype != t {
            return Exec::Diagnostic(Diag::Res001);
        }
        let n = p.buf_rows;
        if n + 16 > self.mxm_cap(t) {
            return Exec::Diagnostic(Diag::Res002);
        }
        let bad = if self.ecc.live { self.ecc.g16.get() } else { 0 };
        let v = Value::BufAppend { btype: t, keep: n, rows, bad, ready_lw: ctx.commit_at() };
        self.mxm_stage(ctx, pl, v);
        Exec::Issued
    }

    // model: slices/mxm/mxm_insts.sail execute MXM_IW
    fn mxm_iw(&mut self, ctx: &Issue, t: WType) -> Exec {
        let pl = self.plane_of(ctx.queue);
        if self.plane_blocked(pl, Some(t)) {
            return Exec::Diagnostic(Diag::Res001);
        }
        let p = &self.mxm.planes[pl];
        if p.buf_rows == 0 || u128::from(ctx.cycle) < p.ready_lw || p.btype != t {
            return Exec::Diagnostic(Diag::Res001);
        }
        let w = Value::W {
            wtype: t,
            rows: if is_pair(t) { p.buf_rows / 2 } else { p.buf_rows },
            weights: p.buf.clone(),
            bad: p.buf_bad.clone(),
            iw_open: false,
            iws_cycle: p.iws_cycle,
            ready_iw: ctx.commit_at(),
        };
        // the buffer is emptied; its rows stay (`buf` is unchanged, `buf_rows` is 0)
        let b = Value::Buf {
            btype: p.btype,
            buf_rows: 0,
            buf: p.buf.clone(),
            bad: Arc::new(Vec::new()),
            ready_lw: p.ready_lw,
        };
        self.mxm_stage(ctx, pl, w);
        self.mxm_stage(ctx, pl, b);
        Exec::Issued
    }

    // model: slices/mxm/mxm_insts.sail execute MXM_IWS
    fn mxm_iws(&mut self, ctx: &Issue, t: WType, src: u32) -> Exec {
        let pl = self.plane_of(ctx.queue);
        if self.plane_blocked(pl, Some(t)) {
            return Exec::Diagnostic(Diag::Res001);
        }
        let rows = match self.read_planes16_boxed(ctx, Self::group16(src), ctx.exec_cycle() as i128) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(r) => r,
        };
        let p = &self.mxm.planes[pl];
        let open = p.iw_open && !abc_after(&p.abc, p.iws_cycle);
        // Stream values installed so far in the open matrix (f16 rows take two each).
        let n = if open {
            if is_pair(p.wtype) {
                2 * p.rows.min(self.mxm_f16_rows())
            } else {
                p.rows
            }
        } else {
            0
        };
        if open && p.wtype != t {
            return Exec::Diagnostic(Diag::Res001);
        }
        if n + 16 > self.mxm_cap(t) {
            return Exec::Diagnostic(Diag::Res002);
        }
        let bad = if self.ecc.live { self.ecc.g16.get() } else { 0 };
        let v = Value::WAppend { wtype: t, keep: n, rows, bad, iws_cycle: ctx.cycle, ready_iw: ctx.commit_at() };
        self.mxm_stage(ctx, pl, v);
        Exec::Issued
    }

    /// The superlanes an ABC poisons: all of them when its activation had an uncorrectable superlane, else the
    /// superlane `r / 16` of each uncorrectable weight row `r` (f16 row `r` is buffer rows `2r` and `2r + 1`).
    // model: slices/mxm/mxm_insts.sail abc_exec
    fn abc_poison(&self, p: &Plane, t: WType) -> u32 {
        let sl = self.geom.superlanes;
        if self.ecc.bad.get() != 0 {
            return ((1u64 << sl) - 1) as u32;
        }
        let mut poison = 0u32;
        for &i in p.w_bad.iter() {
            let r = if is_pair(t) { i as usize / 2 } else { i as usize };
            if r < p.rows && ((r / 16) as u64) < sl {
                poison |= 1 << (r / 16);
            }
        }
        poison
    }

    // model: slices/mxm/mxm_insts.sail abc_exec, execute MXM_ABC, MXM_ABC8
    /// `f`: the FP8 format of an ABC8 (mode bit 13: 0 e4m3, 1 e5m2); 0 for every other type.
    fn mxm_abc<O: Observer + ?Sized>(&mut self, ctx: &Issue, t: WType, f: u32, src: u32, obs: &mut O) -> Exec {
        let pl = self.plane_of(ctx.queue);
        let w: u8 = if is_pair(t) { 2 } else { 1 };
        let p = &self.mxm.planes[pl];
        if self.plane_blocked(pl, Some(t))
            || p.rows == 0
            || u128::from(ctx.cycle) < p.ready_iw
            || !abc_types_ok(p.wtype, t)
            || macc_busy(p, ctx.cycle, self.geom.mxm_occupancy_f16)
        {
            return Exec::Diagnostic(Diag::Res001);
        }
        let x: Planes = match self.read_planes(ctx, src, w, ctx.exec_cycle() as i128) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(x) => crate::operands::widen_planes(&x),
        };
        // the model records the operand's ECC events (`read_planes`) before the sum-check event: deliver them first
        self.ecc_flush(obs);
        let vl = self.geom.vlanes as usize;
        // the faults of the harness for this plane are checked before the products are formed (the sum check). The harness
        // injects a fault in the cycle the ABC executes (an earlier one is an input error at the end of the run), so only
        // the faults of this cycle belong to this ABC.
        let at = ctx.exec_cycle();
        if self.mxm.faults.iter().any(|f| f.plane as usize == pl && f.cycle == at) {
            let mine: Vec<(usize, u32)> = self
                .mxm
                .faults
                .iter()
                .filter(|f| f.plane as usize == pl && f.cycle == at)
                .map(|f| (f.lane as usize, f.bit as u32))
                .collect();
            self.mxm.faults.retain(|f| !(f.plane as usize == pl && f.cycle == at));
            let p = &self.mxm.planes[pl];
            match super::mxm_check::mxm_check(&p.weights, p.rows, vl, p.wtype, t, f, &x, &mine) {
                super::mxm_check::Verdict::Uncorrectable => {
                    self.ras_mxm_event(ctx.exec_cycle(), ctx.queue, pl as u8, None, obs); // model: core/csr.sail ras_mxm_event
                    return Exec::Diagnostic(Diag::Ecc003);
                }
                super::mxm_check::Verdict::Corrected(lane) => {
                    obs.mxm_corr(ctx.exec_cycle(), ctx.queue as u16, pl as u8, lane as u16);
                    self.ras_mxm_event(ctx.exec_cycle(), ctx.queue, pl as u8, Some(lane), obs);
                }
                super::mxm_check::Verdict::Clean => {}
            }
        }
        if t == WType::F16 {
            // Record the exponent bound once per installed matrix.
            let p = &mut self.mxm.planes[pl];
            if !p.row_exponents.as_ref().is_some_and(|(w, _)| Arc::ptr_eq(w, &p.weights)) {
                let exponents = (0..p.rows).map(|r| f16_row_max_exponent(&p.weights[2 * r + 1][..vl])).collect();
                p.row_exponents = Some((p.weights.clone(), Arc::new(exponents)));
            }
        }
        let cacheable = matches!(t, WType::I8 | WType::U8) && !self.ecc.live;
        let p = &mut self.mxm.planes[pl];
        let ws = p.wtype == WType::I8; // weight sign: the installed type
        let xs = t == WType::I8; // activation sign: the ABC type
                                 // Use bounds only for the matrix they describe.
        let exponents = match &p.row_exponents {
            Some((w, exponents)) if t == WType::F16 && Arc::ptr_eq(w, &p.weights) => Some(&exponents[..]),
            _ => None,
        };
        let mut wide: Option<Arc<Vec<Wide>>> = None;
        let out = if cacheable {
            let input = &x[0][..vl];
            if let Some(product) = p.int8_recent.as_ref().and_then(|e| e.get(p, ws, xs, input)) {
                product
            } else {
                let product = Arc::new(abc_product(t, &p.weights, p.rows, vl, ws, xs, &x, None));
                let mut activation = p
                    .int8_recent
                    .take()
                    .map(|entry| entry.activation)
                    .filter(|storage| storage.len() == input.len())
                    .unwrap_or_else(|| input.to_vec().into_boxed_slice());
                activation.copy_from_slice(input);
                p.int8_recent = Some(Int8Recent {
                    weights: Arc::downgrade(&p.weights),
                    rows: p.rows,
                    weights_signed: ws,
                    input_signed: xs,
                    activation,
                    product: product.clone(),
                });
                product
            }
        } else {
            let words = if t == WType::Bfp8 {
                let (out, w) = abc_product_exact(t, f, &p.weights, p.rows, vl, &x, self.geom.ext_wide_acc);
                wide = w.map(Arc::new);
                out
            } else if self.geom.ext_wide_acc && (t == WType::F16 || t == WType::Fp8) {
                // the exact sums of the rows are kept for ACC x128; the f32 product is their rounding
                let (out, w) = abc_product_exact(t, f, &p.weights, p.rows, vl, &x, true);
                wide = w.map(Arc::new);
                out
            } else if t == WType::Fp8 {
                abc_product_fp8(if f & 1 == 0 { FMT_E4M3 } else { FMT_E5M2 }, &p.weights, p.rows, vl, &x)
            } else if t == WType::Bf16 {
                let entry = &mut self.mxm_bf16_weights[pl];
                if !entry.as_ref().is_some_and(|cache| cache.matches(&p.weights, p.rows, vl)) {
                    *entry = Some(Bf16Cache::new(&p.weights, p.rows, vl));
                }
                let cache = entry.as_ref().expect("cache just prepared");
                let act = Bf16Act::new(&x[0][..vl], &x[1][..vl]);
                let mut out = [0u32; MAX_LANES];
                for (r, word) in out.iter_mut().enumerate().take(p.rows) {
                    *word = cache.packed.dot(r, &p.weights[2 * r][..vl], &p.weights[2 * r + 1][..vl], &act);
                }
                out
            } else {
                abc_product(t, &p.weights, p.rows, vl, ws, xs, &x, exponents)
            };
            Arc::new(words)
        };
        let p = &self.mxm.planes[pl];
        let poison = if self.ecc.live { self.abc_poison(p, t) } else { 0 };
        let e = AbcEntry {
            dispatch: ctx.cycle,
            commit: ctx.commit_at(),
            ptype: t,
            q: ctx.queue,
            product: out,
            wide,
            poison,
        };
        let retained = abc_retained(&p.abc, ctx.cycle);
        let mut abc = Vec::with_capacity(1 + retained);
        abc.push(e);
        abc.extend_from_slice(&p.abc[..retained]);
        self.mxm_stage(ctx, pl, Value::Abc(abc));
        Exec::Issued
    }

    // model: slices/mxm/mxm_insts.sail execute MXM_ACC
    fn mxm_acc(&mut self, ctx: &Issue, t: u32, mode: u32, dst: u32, acc: u32) -> Exec {
        let pl = self.plane_of(ctx.queue);
        let a = acc as usize;
        if self.plane_blocked(pl, None) {
            return Exec::Diagnostic(Diag::Res001);
        }
        let p = &self.mxm.planes[pl];
        if a >= p.acc.len() {
            return Exec::Diagnostic(Diag::Opr001); // the decoder does not admit an index at or above mxm.acc_depth
        }
        let Some(e) = abc_at(&p.abc, ctx.cycle, a) else { return Exec::Diagnostic(Diag::Res001) };
        let wide_acc = t == 0; // acctype: 0 = x128 (extension wide_acc), 2 = i32, 3 = f32
        let f32_acc = t == 3;
        let matches = match e.ptype {
            WType::I8 | WType::U8 => t == 2,
            WType::F16 | WType::Fp8 | WType::Bfp8 => wide_acc || f32_acc,
            WType::Bf16 => f32_acc,
        };
        if !matches {
            return Exec::Diagnostic(Diag::Opr001);
        }
        let mut out: Words = [0; MAX_LANES];
        let mut wnew: Vec<Wide> = Vec::new();
        let rows = self.mxm_rows();
        let recent = &e.product[..rows];
        let previous = &p.acc[a][..rows];
        let out_rows = &mut out[..rows];
        if wide_acc {
            if !out_rows.is_empty() {
                wnew = p.acc128[a].to_vec();
                wnew.resize(MAX_LANES, Wide::default());
            }
            for (l, (lane, dst)) in wnew.iter_mut().zip(out_rows).enumerate() {
                let prod = e.wide.as_deref().and_then(|w| w.get(l)).copied().unwrap_or_default();
                *lane = if mode & 1 == 0 { prod } else { lane.plus(prod) };
                *dst = lane.read();
            }
        } else if mode & 1 == 0 {
            out_rows.copy_from_slice(recent);
        } else if f32_acc {
            for ((dst, &acc), &product) in out_rows.iter_mut().zip(previous).zip(recent) {
                *dst = fp::fp_add(FMT32, acc, product);
            }
        } else {
            for ((dst, &acc), &product) in out_rows.iter_mut().zip(previous).zip(recent) {
                *dst = acc.wrapping_add(product);
            }
        }
        // The poison of the result: the product's, and with an accumulating ACC the accumulator's too. It is inherited
        // (not reported again): the reader of the result reports it.
        let pm = if mode & 1 == 0 { e.poison } else { e.poison | p.acc_poison[a] };
        if pm != 0 {
            self.ecc.inherit.set(pm);
        }
        // The accumulator changes only if the result can be written: a diagnostic has no effect.
        let planes = words_to_planes_lanes(&out, self.geom.vlanes as usize);
        match self.write_planes(ctx, dst, &planes, 4, ctx.commit_at()) {
            Ok(()) => {
                if wide_acc {
                    self.mxm_stage(ctx, pl, Value::Acc128(a, Arc::new(wnew), pm));
                } else {
                    self.mxm_stage(ctx, pl, Value::Acc(a, Arc::new(out), pm));
                }
                Exec::Issued
            }
            Err(err) => Exec::Diagnostic(err),
        }
    }

    // model: slices/mxm/mxm_insts.sail execute (MXM_LW, MXM_IW, MXM_IWS, MXM_ABC, MXM_ABC8, MXM_ACC)
    pub fn execute_mxm<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, obs: &mut O) -> Exec {
        match *i {
            Insn::LW { r#type, src } => self.mxm_lw(ctx, wtype(r#type), src),
            Insn::IW { r#type } => self.mxm_iw(ctx, wtype(r#type)),
            Insn::IWS { r#type, src } => self.mxm_iws(ctx, wtype(r#type), src),
            Insn::ABC { r#type, src } => self.mxm_abc(ctx, wtype(r#type), 0, src, obs),
            Insn::ABC8 { fmt, src } => self.mxm_abc(ctx, WType::Fp8, fmt, src, obs),
            Insn::ACC { r#type, dst, mode, acc } => self.mxm_acc(ctx, r#type, mode, dst, acc),
            _ => unreachable!("not an MXM instruction: {:?}", i.id()),
        }
    }
}
