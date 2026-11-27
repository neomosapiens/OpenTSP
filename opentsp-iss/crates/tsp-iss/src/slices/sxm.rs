//! SXM instructions Shift, Permute, Distribute, Rotate, Transpose, LoadMap, LoadMapWide and
//! sxselect (`slices/sxm/sxm_insts.sail`, `sxm_regs.sail`, `sxm_types.sail`; ISCA 2020 sec. III-E;
//! `microarchitecture.md` sec. 7).
//!
//! Each of the two SXMs (0 west, 1 east) holds 16 maps of one 10-bit entry per lane; an unloaded
//! map is the identity. A map load (LoadMap, LoadMapWide) is STAGED and applied when the cycle
//! ends (`unit_state_flush`, `driver::flush`): two queues that load one map in one cycle both
//! raise OTS-E-RES-003 and neither load has any effect. A load also records whether the map is
//! bijective (its entries below `vlanes` are distinct) and the first cycle a Permute may use it:
//! the execute cycle of the load + `geometry.superlanes` (per-tile partial effects).
//!
//! The distributor (DST) works in its row; the transposer (TSP) and permuter/shifter (PRM) read
//! the outward row and write the inward one (`layout`). Lanes at or beyond `vlanes` stay zero.
use super::{Exec, Issue};
use crate::geom::MAX_LANES;
use crate::layout::issuer_of;
use crate::observe::Observer;
use crate::operands::stream_id;
use crate::srf::Vec320;
use crate::Machine;
use ots_isa::{Code as Diag, Insn};

/// `perm_map`: one 10-bit entry per lane.
// model: slices/sxm/sxm_regs.sail perm_map
pub type PermMap = [u16; MAX_LANES];

// model: slices/sxm/sxm_regs.sail identity_map
pub fn identity_map() -> PermMap {
    std::array::from_fn(|l| l as u16)
}

/// `sxm_staged`: a map load staged by `q`.
#[derive(Clone, Debug)]
pub struct Staged {
    pub q: usize,
    pub sxm: usize,
    pub map: usize,
    pub value: PermMap,
    pub bij: bool,
    pub ready: u64,
}

/// The SXM registers (`sxm_maps`, `sxm_map_bij`, `sxm_map_ready`) and the loads staged in the
/// current cycle.
#[derive(Clone, Debug)]
pub struct Sxm {
    pub maps: Box<[[PermMap; 16]; 2]>,
    pub bij: [[bool; 16]; 2],
    pub ready: [[u64; 16]; 2],
    /// In staging order (oldest first); the model's list is newest first.
    pub staged: Vec<Staged>,
}

impl Sxm {
    // model: slices/sxm/sxm_regs.sail sxm_reset
    pub fn new() -> Sxm {
        let id = identity_map();
        Sxm { maps: Box::new([[id; 16]; 2]), bij: [[true; 16]; 2], ready: [[0; 16]; 2], staged: Vec::new() }
    }

    /// Map `m` of SXM `x`.
    pub fn map(&self, x: usize, m: usize) -> &PermMap {
        &self.maps[x][m]
    }

    /// Whether the latest load of the map was bijective.
    pub fn bijective(&self, x: usize, m: usize) -> bool {
        self.bij[x][m]
    }

    /// The first cycle a Permute may use the map.
    pub fn ready(&self, x: usize, m: usize) -> u64 {
        self.ready[x][m]
    }

    // model: postlude/tick.sail sxm_clash
    fn clash(&self, e: &Staged) -> bool {
        self.staged.iter().any(|o| o.q != e.q && o.sxm == e.sxm && o.map == e.map)
    }

    /// The queues with a clashing map load (the SXM half of `bad`, a set, 0..143).
    // model: postlude/tick.sail unit_state_flush (the SXM half of `bad`)
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

    /// Apply the staged loads of the queues not in `bad`, in the model's order (newest first, so
    /// that when one queue loaded a map twice the oldest stands), and clear the stage list.
    // model: postlude/tick.sail unit_state_flush (sxm_maps loop, sxm_stage_list = [||])
    pub fn flush(&mut self, bad: &[bool; 145]) {
        let staged = std::mem::take(&mut self.staged);
        for e in staged.iter().rev() {
            if !bad[e.q] {
                self.maps[e.sxm][e.map] = e.value;
                self.bij[e.sxm][e.map] = e.bij;
                self.ready[e.sxm][e.map] = e.ready;
            }
        }
    }
}

impl Default for Sxm {
    fn default() -> Sxm {
        Sxm::new()
    }
}

/// Shift direction (`shift_dir`).
// model: slices/sxm/sxm_types.sail shift_dir
#[derive(Clone, Copy, PartialEq, Eq)]
enum ShiftDir {
    Up,
    Down,
}

/// Lane `l` of `v`, zero outside `0..vlanes`.
// model: slices/sxm/sxm_insts.sail sxm_lane
fn sxm_lane(v: &[u8], l: i64, vlanes: usize) -> u8 {
    if 0 <= l && (l as usize) < vlanes {
        v[l as usize]
    } else {
        0
    }
}

/// Set lane `l` of `v` (a no-op outside `0..vlanes`; every caller writes inside, so the vector's
/// lanes at and beyond `vlanes` stay zero).
// model: slices/sxm/sxm_insts.sail sxm_set_lane
fn sxm_set_lane<const L: usize>(v: &mut Vec320<L>, l: usize, b: u8, vlanes: usize) {
    if l < vlanes {
        v[l] = b;
    }
}

/// Stream reference d + k on the same row (callers check the stream range first).
// model: slices/sxm/sxm_insts.sail sxm_stream_plus
fn sxm_stream_plus(d: u32, k: u32) -> u32 {
    (d & !31) | ((stream_id(d) as u32 + k) & 31)
}

impl<const L: usize> Machine<L> {
    /// The SXM (0 west, 1 east) of the issuing queue.
    // model: slices/sxm/sxm_insts.sail sx
    fn sx(ctx: &Issue) -> usize {
        issuer_of(ctx.queue).sxm as usize
    }

    // model: slices/sxm/sxm_insts.sail sxm_emit
    fn sxm_emit(&mut self, ctx: &Issue, d: u32, v: Vec320<L>) -> Exec {
        match self.write_plane(ctx, d, &v, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(e) => Exec::Diagnostic(e),
        }
    }

    /// [`Machine::sxm_read1`] without the copy: the `vlanes` bytes in the register file (see `read_plane_ref`).
    fn sxm_ref(&self, ctx: &Issue, s: u32) -> Result<&[u8], Diag> {
        if self.ecc.live {
            // a checked operand lives in the instruction's scratch, corrected for this reader
            let v = self.read_plane(ctx, s, ctx.exec_cycle() as i128)?;
            return Ok(self.ecc.scratch(v[..self.geom.vlanes as usize].to_vec()));
        }
        crate::operands::read_plane_at(&self.srf, &self.geom, ctx, s, ctx.exec_cycle() as i128)
    }

    /// Read the one-stream operand `s` in the execute cycle.
    fn sxm_read1(&self, ctx: &Issue, s: u32) -> Result<Vec320<L>, Diag> {
        self.read_plane(ctx, s, ctx.exec_cycle() as i128)
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_SHIFT
    fn sxm_shift(&mut self, ctx: &Issue, dir: ShiftDir, n: u32, s: u32, d: u32) -> Exec {
        let p = match self.sxm_ref(ctx, s) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(p) => p,
        };
        // an output superlane takes lanes of its neighbours: one bad superlane poisons every output superlane
        self.ecc.dep_all.set(true);
        let vl = self.geom.vlanes as usize;
        let k = n as i64;
        let mut v: Vec320<L> = [0; L];
        for l in 0..vl {
            let j = match dir {
                ShiftDir::Up => l as i64 - k,
                ShiftDir::Down => l as i64 + k,
            };
            sxm_set_lane(&mut v, l, sxm_lane(p, j, vl), vl);
        }
        self.sxm_emit(ctx, d, v)
    }

    // model: slices/sxm/sxm_insts.sail sxm_map_distinct
    pub fn sxm_map_distinct(m: &PermMap, vlanes: usize) -> bool {
        let mut seen = [false; MAX_LANES];
        let mut ok = true;
        for &e in m.iter().take(vlanes) {
            let e = e as usize;
            if e >= vlanes {
                continue;
            }
            if seen[e] {
                ok = false;
            } else {
                seen[e] = true;
            }
        }
        ok
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_MAP (Permute)
    fn sxm_permute(&mut self, ctx: &Issue, mi: usize, s: u32, d: u32) -> Exec {
        let x = Self::sx(ctx);
        let p = match self.sxm_ref(ctx, s) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(p) => p,
        };
        if ctx.exec_cycle() < self.sxm.ready(x, mi) {
            return Exec::Diagnostic(Diag::Res001);
        }
        if !self.sxm.bijective(x, mi) {
            return Exec::Diagnostic(Diag::Sxm001);
        }
        self.ecc.dep_all.set(true); // a lane may come from any superlane
        let vl = self.geom.vlanes as usize;
        let m = self.sxm.map(x, mi);
        let mut v: Vec320<L> = [0; L];
        // lane l takes lane m[l] of the source, or stays zero when the entry is not a lane of the vector
        for (o, &e) in v[..vl].iter_mut().zip(&m[..vl]) {
            let e = e as usize;
            *o = if e < vl { p[e] } else { 0 };
        }
        self.sxm_emit(ctx, d, v)
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_MAP (Distribute)
    fn sxm_distribute(&mut self, ctx: &Issue, mi: usize, s: u32, d: u32) -> Exec {
        let x = Self::sx(ctx);
        let p = match self.sxm_ref(ctx, s) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(p) => p,
        };
        let vl = self.geom.vlanes as usize;
        let m = self.sxm.map(x, mi);
        let mut v: Vec320<L> = [0; L];
        for g in 0..self.geom.superlanes as usize {
            for j in 0..16 {
                let e = m[16 * g + j] as usize;
                if e < 16 {
                    sxm_set_lane(&mut v, 16 * g + j, sxm_lane(p, (16 * g + e) as i64, vl), vl);
                }
            }
        }
        self.sxm_emit(ctx, d, v)
    }

    /// Stage the map `mi` of this SXM as `m[l] = (hi[l] mod 4) * 256 + lo[l]`.
    // model: slices/sxm/sxm_insts.sail sxm_load_map
    fn sxm_load_map(&mut self, ctx: &Issue, mi: usize, lo: &Vec320<L>, hi: &Vec320<L>) {
        let vl = self.geom.vlanes as usize;
        let mut m: PermMap = [0; MAX_LANES];
        for (l, e) in m.iter_mut().enumerate() {
            *e = (sxm_lane(hi, l as i64, vl) as u16 % 4) * 256 + sxm_lane(lo, l as i64, vl) as u16;
        }
        let bij = Self::sxm_map_distinct(&m, vl);
        let ready = ctx.exec_cycle() + self.geom.sxm_map_ready;
        self.sxm.staged.push(Staged { q: ctx.queue, sxm: Self::sx(ctx), map: mi, value: m, bij, ready });
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_LOADMAP
    fn sxm_loadmap(&mut self, ctx: &Issue, mi: usize, s: u32) -> Exec {
        match self.sxm_read1(ctx, s) {
            Err(e) => Exec::Diagnostic(e),
            Ok(p) => {
                self.sxm_load_map(ctx, mi, &p, &[0; L]);
                Exec::Issued
            }
        }
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_LOADMAPWIDE
    fn sxm_loadmapwide(&mut self, ctx: &Issue, mi: usize, s: u32, s2: u32) -> Exec {
        // both operands are read (and the first error stands) before either is used
        let lo = self.sxm_read1(ctx, s);
        let hi = self.sxm_read1(ctx, s2);
        match (lo, hi) {
            (Err(e), _) | (_, Err(e)) => Exec::Diagnostic(e),
            (Ok(lo), Ok(hi)) => {
                self.sxm_load_map(ctx, mi, &lo, &hi);
                Exec::Issued
            }
        }
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_ROTATE
    fn sxm_rotate(&mut self, ctx: &Issue, nb: u32, s: u32, d: u32) -> Exec {
        let n: usize = if nb == 3 { 3 } else { 4 };
        if stream_id(d) as usize + n * n > 32 {
            return Exec::Diagnostic(Diag::Opr002);
        }
        let p = match self.sxm_read1(ctx, s) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(p) => p,
        };
        let vl = self.geom.vlanes as usize;
        let mut res = Exec::Issued;
        for t in 0..n * n {
            let (tr, tc) = (t / n, t % n);
            let mut v: Vec320<L> = [0; L];
            for g in 0..self.geom.superlanes as usize {
                for r in 0..n {
                    for c in 0..n {
                        let from = 16 * g + ((r + tr) % n) * n + (c + tc) % n;
                        sxm_set_lane(&mut v, 16 * g + r * n + c, sxm_lane(&p, from as i64, vl), vl);
                    }
                }
            }
            if let Err(e) = self.write_plane(ctx, sxm_stream_plus(d, t as u32), &v, ctx.commit_at()) {
                res = Exec::Diagnostic(e);
            }
        }
        res
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_TRANSPOSE
    fn sxm_transpose(&mut self, ctx: &Issue, s: u32, d: u32) -> Exec {
        let vl = self.geom.vlanes as usize;
        let mut res = Exec::Issued;
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        let fast_rows =
            if L == 320 && vl == 320 && self.geom.superlanes == 20 { self.sxm_borrow_planes16(ctx, s) } else { None };
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        if let Some(rows) = fast_rows {
            let mut columns = [[0; L]; 16];
            transpose_neon(&rows, &mut columns);
            for (c, column) in columns.iter().enumerate() {
                if let Err(e) = self.write_plane(ctx, sxm_stream_plus(d, c as u32), column, ctx.commit_at()) {
                    res = Exec::Diagnostic(e);
                }
            }
            return res;
        }
        let rows = match self.read_planes16(ctx, s, ctx.exec_cycle() as i128) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(r) => r,
        };
        for c in 0..16usize {
            let mut v: Vec320<L> = [0; L];
            for g in 0..self.geom.superlanes as usize {
                for (r, row) in rows.iter().enumerate() {
                    sxm_set_lane(&mut v, 16 * g + r, sxm_lane(row, (16 * g + c) as i64, vl), vl);
                }
            }
            if let Err(e) = self.write_plane(ctx, sxm_stream_plus(d, c as u32), &v, ctx.commit_at()) {
                res = Exec::Diagnostic(e);
            }
        }
        res
    }

    // model: slices/sxm/sxm_insts.sail execute SXM_SELECT
    fn sxm_select(&mut self, ctx: &Issue, a: u32, b: u32, c: u32, m: u32, d: u32) -> Exec {
        let pa = self.sxm_ref(ctx, a);
        let pb = self.sxm_ref(ctx, b);
        let pc = self.sxm_ref(ctx, c);
        let pm = self.sxm_ref(ctx, m);
        let (pa, pb, pc, pm) = match (pa, pb, pc, pm) {
            (Err(e), ..) | (_, Err(e), ..) | (_, _, Err(e), _) | (_, _, _, Err(e)) => return Exec::Diagnostic(e),
            (Ok(pa), Ok(pb), Ok(pc), Ok(pm)) => (pa, pb, pc, pm),
        };
        // Per lane: a lane whose sel is above 2 is zero, the others are selected and written,
        // and the instruction reports one OPR-001 (an exception to "a diagnostic means no effect").
        let vl = self.geom.vlanes as usize;
        let mut v: Vec320<L> = [0; L];
        let mut bad = false;
        for l in 0..vl {
            let sel = pm[l];
            v[l] = match sel {
                0 => pa[l],
                1 => pb[l],
                2 => pc[l],
                _ => 0,
            };
            bad |= sel > 2;
        }
        let r = self.sxm_emit(ctx, d, v);
        if bad {
            self.ecc.data_opr.set(true);
            Exec::Diagnostic(Diag::Opr001)
        } else {
            r
        }
    }

    // model: slices/sxm/sxm_insts.sail execute (the SXM clauses)
    pub fn execute_sxm<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, _obs: &mut O) -> Exec {
        match *i {
            Insn::Shift { dst, src, dir, amount } => {
                self.sxm_shift(ctx, if dir & 1 == 0 { ShiftDir::Up } else { ShiftDir::Down }, amount, src, dst)
            }
            Insn::Permute { dst, src, map } => self.sxm_permute(ctx, map as usize, src, dst),
            Insn::Distribute { dst, src, map } => self.sxm_distribute(ctx, map as usize, src, dst),
            Insn::Rotate { dst, src, n } => self.sxm_rotate(ctx, n, src, dst),
            // the field of a 16-stream group holds the stream / 16
            Insn::Transpose { dst, src } => self.sxm_transpose(ctx, src << 4, dst << 4),
            Insn::LoadMap { src, map } => self.sxm_loadmap(ctx, map as usize, src),
            Insn::LoadMapWide { src, map, src2 } => self.sxm_loadmapwide(ctx, map as usize, src, src2),
            Insn::SelectS { dst, a, b, sel, c } => self.sxm_select(ctx, a, b, c, sel, dst),
            _ => unreachable!("not an SXM instruction: {:?}", i.id()),
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(never)]
fn transpose_neon<const L: usize>(rows: &[&[u8; 320]; 16], cols: &mut crate::operands::Group16<L>) {
    use std::arch::aarch64::*;
    debug_assert_eq!(L, 320);
    // Each tile consists of 16 active bytes from each input row. The low and
    // high zip instructions interleave groups of 1, 2, 4, then 8 bytes.
    // SAFETY: The caller checks a full 320-lane geometry and L == 320.
    // Every 16-byte load and store below stays within one row. NEON is
    // available on all Apple targets that use the AArch64 ABI.
    unsafe {
        for offset in (0..L).step_by(16) {
            let bytes: [uint8x16_t; 16] = std::array::from_fn(|i| vld1q_u8(rows[i].as_ptr().add(offset)));
            let pairs: [uint16x8_t; 16] = std::array::from_fn(|i| {
                let base = i & !1;
                let a = bytes[base];
                let b = bytes[base + 1];
                vreinterpretq_u16_u8(if i % 2 == 0 { vzip1q_u8(a, b) } else { vzip2q_u8(a, b) })
            });
            let quads: [uint32x4_t; 16] = std::array::from_fn(|i| {
                let base = (i & !3) + ((i % 4) / 2);
                let a = pairs[base];
                let b = pairs[base + 2];
                vreinterpretq_u32_u16(if i % 2 == 0 { vzip1q_u16(a, b) } else { vzip2q_u16(a, b) })
            });
            let octets: [uint64x2_t; 16] = std::array::from_fn(|i| {
                let base = (i & !7) + ((i % 8) / 2);
                let a = quads[base];
                let b = quads[base + 4];
                vreinterpretq_u64_u32(if i % 2 == 0 { vzip1q_u32(a, b) } else { vzip2q_u32(a, b) })
            });
            for (i, col) in cols.iter_mut().enumerate() {
                let a = octets[i / 2];
                let b = octets[i / 2 + 8];
                let tile = if i % 2 == 0 { vzip1q_u64(a, b) } else { vzip2q_u64(a, b) };
                vst1q_u8(col.as_mut_ptr().add(offset), vreinterpretq_u8_u64(tile));
            }
        }
    }
}
