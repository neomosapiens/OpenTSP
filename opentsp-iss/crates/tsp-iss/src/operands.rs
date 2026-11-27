//! Stream operands (`sys/operands.sail`). A read happens at the issuer's read gap on the operand's
//! row, in the execute cycle `issue_cycle + d_skew (+ element)`. A result is not written
//! immediately: it is SCHEDULED for commit at `issue_cycle + d_func (+ element)` and becomes part of
//! the stream register file when the cycle driver reaches that cycle (`driver::commit`). That is
//! what makes a read in cycle t see only commits from cycles before t, whatever the queue order.
//!
//! The model's registers `issue_unit`, `issue_queue` and `issue_thread` are the fields of the
//! [`Issue`] context passed to every function. The instructions that call them are the units'.
use crate::geom::MAX_LANES;
use crate::layout::{issuer_of, read_gap, write_gap, Issuer, RowId, UnitKind};
use crate::slices::Issue;
use crate::sram::merge_word_into;
use crate::srf::{SrfLookup, Vec320};
use crate::Machine;
use ots_isa::Code as Diag;

/// Byte planes of one multi-byte operand: plane b holds byte b of every lane's element.
pub type Planes<const L: usize = { MAX_LANES }> = [[u8; L]; 4];

/// `zero_vec`, `no_planes`.
pub const ZERO_VEC: Vec320 = [0; MAX_LANES];
pub const NO_PLANES: Planes = [ZERO_VEC; 4];

/// A zero vector to borrow (the planes of a multi-byte operand above its width).
pub static ZERO_PLANE: Vec320 = [0; MAX_LANES];

/// The planes of an operand borrowed from the stream register file: `vlanes` bytes each.
pub type PlaneRefs<'a> = [&'a [u8]; 4];

/// `p` at the widest width (the matrix unit keeps its state and works at that width).
pub fn widen_planes<const L: usize>(p: &Planes<L>) -> Planes {
    let mut out = NO_PLANES;
    for (o, i) in out.iter_mut().zip(p) {
        o[..L].copy_from_slice(i);
    }
    out
}

/// A single-plane operand (i8 elements, or one stream of raw bytes).
// model: sys/operands.sail one_plane
pub fn one_plane<const L: usize>(v: [u8; L]) -> Planes<L> {
    let mut p = [[0u8; L]; 4];
    p[0] = v;
    p
}

/// A 16-stream operand group (MXM LW, SXM Transpose): streams base..base+15, base 0 or 16.
pub type Group16<const L: usize = { MAX_LANES }> = [[u8; L]; 16];

/// The bit of a stream reference that selects the row (0 = E, 1 = W): `core/types.sail`.
pub const STREAM_ROW_BIT: u32 = 5;

/// The row of an encoded stream reference.
// model: core/types.sail stream_row
pub fn stream_row(s: u32) -> RowId {
    if (s >> STREAM_ROW_BIT) & 1 == 0 {
        RowId::E
    } else {
        RowId::W
    }
}

/// The stream (0..31) of an encoded stream reference.
// model: core/types.sail stream_id
pub fn stream_id(s: u32) -> u8 {
    (s & 31) as u8
}

/// Stream b of a multi-stream operand starting at s (callers check `aligned` first).
// model: sys/operands.sail plane_stream
fn plane_stream(s: u32, b: usize) -> u8 {
    let i = stream_id(s) as usize + b;
    assert!(i <= 31, "plane_stream: operand not checked by aligned()");
    i as u8
}

/// Width check shared by reads and writes (OTSP-CON-ALIGN: destinations too).
// model: sys/operands.sail aligned
pub fn aligned(s: u32, w: u8) -> bool {
    let id = stream_id(s) as u32;
    let w = w as u32;
    id.is_multiple_of(w) && id + w <= 32
}

/// One scheduled stream write.
// model: sys/operands.sail commit
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commit<const L: usize = { MAX_LANES }> {
    pub r: RowId,
    pub id: u8,
    pub g: u8,
    pub value: [u8; L],
    /// Writer queue (144: the control channel).
    pub q: usize,
    /// Byte plane of a multi-stream result.
    pub b: u8,
    /// The ECC overlay id of the value (`crate::ecc`): 0 when its check bits are valid.
    pub ov: u32,
}

/// The fields of a [`Commit`] but its value.
#[derive(Clone, Copy, Debug)]
pub struct CommitAt {
    pub r: RowId,
    pub id: u8,
    pub g: u8,
    pub q: usize,
    pub b: u8,
    pub ov: u32,
}

/// Append `Commit { value: merge_word(new, ..), ..at }` to `list`, building the vector value in the list's own storage
/// (a value returned by `merge_word` is moved once into the `Commit` and once more by `push`). `mask` is
/// `(vl, split, use_thread, thread, superlanes)`, the last five arguments of `merge_word`.
pub fn push_merged<const L: usize>(
    list: &mut Vec<Commit<L>>,
    at: CommitAt,
    new: Option<&[u8]>,
    mask: (u64, u64, bool, u8, u64),
) {
    list.reserve(1);
    let len = list.len();
    // SAFETY: after `reserve(1)` the allocation holds at least `len + 1` commits. Every field of the commit at index
    // `len` is written below (the scalar fields one by one, `value` zeroed and then filled by `merge_word_into`) before
    // `set_len` makes it part of the list, and no field has a destructor, so nothing uninitialised is read or dropped.
    unsafe {
        let p = list.as_mut_ptr().add(len);
        std::ptr::addr_of_mut!((*p).r).write(at.r);
        std::ptr::addr_of_mut!((*p).id).write(at.id);
        std::ptr::addr_of_mut!((*p).g).write(at.g);
        std::ptr::addr_of_mut!((*p).q).write(at.q);
        std::ptr::addr_of_mut!((*p).b).write(at.b);
        std::ptr::addr_of_mut!((*p).ov).write(at.ov);
        let value = std::ptr::addr_of_mut!((*p).value);
        value.write([0u8; L]);
        merge_word_into(&mut *value, new, mask.0, mask.1, mask.2, mask.3, mask.4);
        list.set_len(len + 1);
    }
}

/// The issuing context's unit (`issue_unit`): the queue's, and the register's reset value (the VXM)
/// for the control channel, which writes no stream.
pub fn issue_unit(q: usize) -> Issuer {
    if q <= 143 {
        issuer_of(q)
    } else {
        Issuer { unit: UnitKind::Vxm, slice: 0, plane: 0, sxm: 0 }
    }
}

/// The gap at which each queue's unit reads and writes each row (`read_gap` and `write_gap` of `issue_unit(q)`), for the
/// queues 0..143 and the control channel, computed once per machine: a function of the profile's layout only.
#[derive(Clone, Debug)]
pub struct GapTable {
    read: [[Option<u8>; 2]; 145],
    write: [[Option<u8>; 2]; 145],
}

impl GapTable {
    pub fn new(g: &crate::geom::Geom) -> GapTable {
        let mut t = GapTable { read: [[None; 2]; 145], write: [[None; 2]; 145] };
        for q in 0..145 {
            let iu = issue_unit(q);
            for r in [RowId::E, RowId::W] {
                t.read[q][r as usize] = read_gap(g, iu.unit, iu.slice as usize, r);
                t.write[q][r as usize] = write_gap(g, iu.unit, iu.slice as usize, r);
            }
        }
        t
    }

    /// `read_gap(issue_unit(q))` of row `r`.
    #[inline]
    pub fn read(&self, q: usize, r: RowId) -> Option<u8> {
        self.read[q][r as usize]
    }

    /// `write_gap(issue_unit(q))` of row `r`.
    #[inline]
    pub fn write(&self, q: usize, r: RowId) -> Option<u8> {
        self.write[q][r as usize]
    }
}

/// [`Machine::read_plane`] without the copy: the `vlanes` bytes in the register file, from the fields alone so
/// that the caller can write another field of the machine while it holds them.
// model: sys/operands.sail read_planes (w = 1)
pub fn read_plane_at<'a>(
    srf: &'a crate::srf::Srf,
    geom: &crate::geom::Geom,
    ctx: &Issue,
    s: u32,
    t: impl Into<i128>,
) -> Result<&'a [u8], Diag> {
    let t = t.into();
    let r = stream_row(s);
    let iu = issue_unit(ctx.queue);
    let Some(g) = read_gap(geom, iu.unit, iu.slice as usize, r) else { return Err(Diag::Dir001) };
    match srf.read_ref(r, stream_id(s), g, t) {
        (SrfLookup::Found, Some(v)) => Ok(v),
        (SrfLookup::Upstream, _) => Err(Diag::Dir001),
        _ => Err(Diag::Tim001),
    }
}

impl Machine {
    /// [`read_plane_at`] as an associated function of the widest machine.
    pub fn read_plane_ref<'a>(
        srf: &'a crate::srf::Srf,
        geom: &crate::geom::Geom,
        ctx: &Issue,
        s: u32,
        t: impl Into<i128>,
    ) -> Result<&'a [u8], Diag> {
        let t = t.into();
        read_plane_at(srf, geom, ctx, s, t)
    }
}

impl<const L: usize> Machine<L> {
    /// Read w consecutive streams starting at s, in cycle t, at the issuer's read gap. `chain`: a
    /// chained VXM operand (`srf_read_chain`: the VXM result committed at the issuer's own gap in
    /// cycle t - 1); otherwise the flow rule (`srf_read`).
    // model: sys/operands.sail read_planes_core
    pub fn read_planes_core(
        &self,
        ctx: &Issue,
        s: u32,
        w: u8,
        t: impl Into<i128>,
        chain: bool,
    ) -> Result<Planes<L>, Diag> {
        let t = t.into();
        let r = stream_row(s);
        let Some(g) = self.gaps.read(ctx.queue, r) else { return Err(Diag::Dir001) };
        if !aligned(s, w) {
            return Err(Diag::Opr002);
        }
        let mut out = [[0u8; L]; 4];
        let mut err: Option<Diag> = None;
        for (b, slot) in out.iter_mut().enumerate().take(w as usize) {
            let id = plane_stream(s, b);
            let found = if chain { self.srf.read_chain_lanes(r, id, g, t) } else { self.srf.read_lanes(r, id, g, t) };
            match found {
                (SrfLookup::Found, v) => {
                    *slot = v;
                    if self.ecc.live {
                        self.ecc_check_plane(ctx, r, id, g, t, chain, slot);
                    }
                }
                (SrfLookup::Upstream, _) => err = Some(Diag::Dir001),
                (SrfLookup::None, _) => {
                    if err.is_none() {
                        err = Some(Diag::Tim001)
                    }
                }
            }
        }
        match err {
            Some(d) => Err(d),
            None => Ok(out),
        }
    }

    /// [`Machine::read_planes_core`] without the copy: plane b of the result is the arena's bytes of stream
    /// `s + b` (`vlanes` of them), the planes at or above `w` are `ZERO_PLANE`'s. Same errors as the copying read.
    // model: sys/operands.sail read_planes_core
    pub fn read_planes_refs(
        &self,
        ctx: &Issue,
        s: u32,
        w: u8,
        t: impl Into<i128>,
        chain: bool,
    ) -> Result<PlaneRefs<'_>, Diag> {
        let t = t.into();
        let r = stream_row(s);
        let Some(g) = self.gaps.read(ctx.queue, r) else { return Err(Diag::Dir001) };
        if !aligned(s, w) {
            return Err(Diag::Opr002);
        }
        let vl = self.geom.vlanes as usize;
        let zero: &[u8] = &ZERO_PLANE[..vl];
        let mut out: PlaneRefs<'_> = [zero; 4];
        let mut err: Option<Diag> = None;
        for (b, slot) in out.iter_mut().enumerate().take(w as usize) {
            let id = plane_stream(s, b);
            let (look, v) = if chain {
                match self.srf.read_chain_ref(r, id, g, t) {
                    Some(v) => (SrfLookup::Found, Some(v)),
                    None => (SrfLookup::None, None),
                }
            } else {
                self.srf.read_ref(r, id, g, t)
            };
            match (look, v) {
                (SrfLookup::Found, Some(v)) => {
                    *slot = v;
                    if self.ecc.live {
                        *slot = self.ecc_checked_ref(ctx, r, id, g, t, chain, v);
                    }
                }
                (SrfLookup::Upstream, _) => err = Some(Diag::Dir001),
                _ => {
                    if err.is_none() {
                        err = Some(Diag::Tim001)
                    }
                }
            }
        }
        match err {
            Some(d) => Err(d),
            None => Ok(out),
        }
    }

    /// One stream as a vector: `read_planes` of width 1, without the four-plane operand.
    // model: sys/operands.sail read_planes (w = 1)
    pub fn read_plane(&self, ctx: &Issue, s: u32, t: impl Into<i128>) -> Result<Vec320<L>, Diag> {
        let t = t.into();
        let r = stream_row(s);
        let Some(g) = self.gaps.read(ctx.queue, r) else { return Err(Diag::Dir001) };
        match self.srf.read_lanes(r, stream_id(s), g, t) {
            (SrfLookup::Found, mut v) => {
                if self.ecc.live {
                    self.ecc_check_plane(ctx, r, stream_id(s), g, t, false, &mut v);
                }
                Ok(v)
            }
            (SrfLookup::Upstream, _) => Err(Diag::Dir001),
            (SrfLookup::None, _) => Err(Diag::Tim001),
        }
    }

    /// The plane `v` just read, checked and corrected, as a slice that lives as long as the instruction.
    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    fn ecc_checked_ref(
        &self,
        ctx: &Issue,
        r: RowId,
        id: u8,
        g: u8,
        t: impl Into<i128>,
        chain: bool,
        v: &[u8],
    ) -> &[u8] {
        let t = t.into();
        let mut c = v.to_vec();
        self.ecc_check_plane(ctx, r, id, g, t, chain, &mut c);
        self.ecc.scratch(c)
    }

    /// The unchecked read of one stream: the value and its ECC overlay id as they flow (`read_plane_raw`).
    // model: sys/operands.sail read_plane_raw
    pub fn read_plane_raw(&self, ctx: &Issue, s: u32, t: impl Into<i128>) -> Result<(Vec320<L>, u32), Diag> {
        let t = t.into();
        let r = stream_row(s);
        let Some(g) = self.gaps.read(ctx.queue, r) else { return Err(Diag::Dir001) };
        match self.srf.read_lanes(r, stream_id(s), g, t) {
            (SrfLookup::Found, v) => Ok((v, self.srf.ov_at(r, stream_id(s), g, t, false))),
            (SrfLookup::Upstream, _) => Err(Diag::Dir001),
            (SrfLookup::None, _) => Err(Diag::Tim001),
        }
    }

    /// Check the plane just read from stream `id` against its overlay and correct it for the reader (`ecc_check_value`).
    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    pub(crate) fn ecc_check_plane(
        &self,
        ctx: &Issue,
        r: RowId,
        id: u8,
        g: u8,
        t: impl Into<i128>,
        chain: bool,
        v: &mut [u8],
    ) {
        let t = t.into();
        let ov = self.srf.ov_at(r, id, g, t, chain);
        let vl = self.geom.vlanes as usize;
        // a chained read takes the register written at the reader's own gap: no report is on it (srf_read_chain_ov)
        let spare = if chain { 0 } else { self.srf.spare(r, id, g, t) };
        self.ecc.check_value(
            &mut v[..vl],
            ov,
            // `t` is the wrapped 64-bit cycle of the read as a signed number: at or above 2^63 it is negative, the cycle is not
            t as u64,
            ctx.queue,
            crate::ecc::Where::Stream { w: r == RowId::W, id, gap: g },
            spare,
            &|x| self.srf.s03(r, g, t, x),
        );
    }

    // model: sys/operands.sail read_planes
    pub fn read_planes(&self, ctx: &Issue, s: u32, w: u8, t: impl Into<i128>) -> Result<Planes<L>, Diag> {
        let t = t.into();
        self.read_planes_core(ctx, s, w, t, false)
    }

    /// Schedule w planes for commit at cycle `c`, at the issuer's write gap on s's row. Inactive
    /// superlanes (SetVL; on a MEM queue also the other liquid thread's tiles) produce zero
    /// (`core/sram.sail`).
    ///
    /// The model prepends to the ring slot's list; the ISS appends, so a slot holds the model's list
    /// reversed (`commit_due` restores the model's order).
    // model: sys/operands.sail write_planes
    pub fn write_planes(&mut self, ctx: &Issue, s: u32, p: &Planes<L>, w: u8, c: u128) -> Result<(), Diag> {
        let r = stream_row(s);
        let Some(g) = self.gaps.write(ctx.queue, r) else { return Err(Diag::Dir001) };
        if !aligned(s, w) {
            return Err(Diag::Opr002);
        }
        let k = (c % crate::driver::PENDING_SLOTS as u128) as usize;
        let q = ctx.queue;
        let vl = if q <= 143 { self.masks.queue_vl[q] as u64 } else { self.geom.superlanes };
        let split = if q <= 87 { self.masks.mem_thread_split[q] as u64 } else { self.geom.superlanes };
        for (b, plane) in p.iter().enumerate().take(w as usize) {
            let at = CommitAt { r, id: plane_stream(s, b), g, q, b: b as u8, ov: self.ecc_result_ov() };
            push_merged(
                &mut self.d.pending[k],
                at,
                Some(plane),
                (vl, split, q <= 87, ctx.thread, self.geom.superlanes),
            );
        }
        self.d.pending_occ |= 1 << k;
        self.activity.srf_write(r == RowId::W, g, w as u64);
        Ok(())
    }

    /// One stream scheduled for commit at cycle `c`: `write_planes` of width 1 for the vector `v`.
    // model: sys/operands.sail write_planes (w = 1)
    pub fn write_plane(&mut self, ctx: &Issue, s: u32, v: &Vec320<L>, c: u128) -> Result<(), Diag> {
        let r = stream_row(s);
        let Some(g) = self.gaps.write(ctx.queue, r) else { return Err(Diag::Dir001) };
        let k = (c % crate::driver::PENDING_SLOTS as u128) as usize;
        let q = ctx.queue;
        let vl = if q <= 143 { self.masks.queue_vl[q] as u64 } else { self.geom.superlanes };
        let split = if q <= 87 { self.masks.mem_thread_split[q] as u64 } else { self.geom.superlanes };
        let at = CommitAt { r, id: stream_id(s), g, q, b: 0, ov: self.ecc_result_ov() };
        push_merged(&mut self.d.pending[k], at, Some(v), (vl, split, q <= 87, ctx.thread, self.geom.superlanes));
        self.d.pending_occ |= 1 << k;
        self.activity.srf_write(r == RowId::W, g, 1);
        Ok(())
    }

    /// `write_plane` of the SRAM word at `slice`, `idx` (an unwritten word is zero): one copy from the page into the
    /// commit.
    // model: sys/operands.sail write_planes (w = 1)
    pub fn write_plane_sram(&mut self, ctx: &Issue, s: u32, slice: usize, idx: u64, c: u128) -> Result<(), Diag> {
        let r = stream_row(s);
        let Some(g) = self.gaps.write(ctx.queue, r) else { return Err(Diag::Dir001) };
        let k = (c % crate::driver::PENDING_SLOTS as u128) as usize;
        let q = ctx.queue;
        let vl = if q <= 143 { self.masks.queue_vl[q] as u64 } else { self.geom.superlanes };
        let split = if q <= 87 { self.masks.mem_thread_split[q] as u64 } else { self.geom.superlanes };
        // the MEM Read forwards the stored check bits of the word with its data, for the active tiles
        let ov = if self.ecc.live { self.ecc_forward_ov(slice, idx, vl, split, q <= 87, ctx.thread) } else { 0 };
        let at = CommitAt { r, id: stream_id(s), g, q, b: 0, ov };
        let new = self.sram.word_ref(slice, idx);
        push_merged(&mut self.d.pending[k], at, new, (vl, split, q <= 87, ctx.thread, self.geom.superlanes));
        self.d.pending_occ |= 1 << k;
        self.activity.srf_write(r == RowId::W, g, 1);
        Ok(())
    }

    /// `read_planes16` into a heap block: the sixteen vectors are copied once, from the SRF arena into the block
    /// (the by-value result of `read_planes16` moves 5 KB, and the caller's `Box::new` moves it again).
    // model: sys/operands.sail read_planes16
    pub fn read_planes16_boxed(&self, ctx: &Issue, s: u32, t: impl Into<i128>) -> Result<Box<Group16>, Diag> {
        let t = t.into();
        let r = stream_row(s);
        let Some(g) = self.gaps.read(ctx.queue, r) else { return Err(Diag::Dir001) };
        if !stream_id(s).is_multiple_of(16) {
            return Err(Diag::Opr002);
        }
        // SAFETY: `Group16` is an array of byte arrays, for which the all-zero bit pattern is a valid value, so
        // the zeroed allocation is initialised.
        let mut out: Box<Group16> = unsafe { Box::<Group16>::new_zeroed().assume_init() };
        let mut err: Option<Diag> = None;
        self.ecc.g16.set(0);
        for (b, slot) in out.iter_mut().enumerate() {
            match self.srf.read_ref(r, plane_stream(s, b), g, t) {
                (SrfLookup::Found, Some(v)) => {
                    crate::copy::copy(&mut slot[..v.len()], v);
                    if self.ecc.live {
                        self.ecc_check_plane(ctx, r, plane_stream(s, b), g, t, false, slot);
                        if self.ecc.last_bad.get() != 0 {
                            self.ecc.g16.set(self.ecc.g16.get() | 1 << b);
                        }
                    }
                }
                (SrfLookup::Found, None) => {}
                (SrfLookup::Upstream, _) => err = Some(Diag::Dir001),
                (SrfLookup::None, _) => {
                    if err.is_none() {
                        err = Some(Diag::Tim001)
                    }
                }
            }
        }
        match err {
            Some(d) => Err(d),
            None => Ok(out),
        }
    }

    /// Borrow the 16 valid 320-lane operands for SXM. Uncertain or faulty reads use the
    /// regular reader, which performs correction and reports failures in the required order.
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    pub(crate) fn sxm_borrow_planes16(&self, ctx: &Issue, s: u32) -> Option<[&[u8; 320]; 16]> {
        if self.ecc.live || !stream_id(s).is_multiple_of(16) {
            return None;
        }
        let row = stream_row(s);
        let gap = self.gaps.read(ctx.queue, row)?;
        let time = ctx.exec_cycle() as i128;
        let mut result = [&[0u8; 320]; 16];
        for (index, slot) in result.iter_mut().enumerate() {
            let (status, value) = self.srf.read_ref(row, plane_stream(s, index), gap, time);
            if status != SrfLookup::Found {
                return None;
            }
            *slot = value?.try_into().ok()?;
        }
        self.ecc.g16.set(0);
        Some(result)
    }

    // model: sys/operands.sail read_planes16
    pub fn read_planes16(&self, ctx: &Issue, s: u32, t: impl Into<i128>) -> Result<Group16, Diag> {
        let t = t.into();
        let r = stream_row(s);
        let Some(g) = self.gaps.read(ctx.queue, r) else { return Err(Diag::Dir001) };
        if !stream_id(s).is_multiple_of(16) {
            return Err(Diag::Opr002);
        }
        let mut out: Group16 = [[0u8; MAX_LANES]; 16];
        let mut err: Option<Diag> = None;
        self.ecc.g16.set(0);
        for (b, slot) in out.iter_mut().enumerate() {
            match self.srf.read(r, plane_stream(s, b), g, t) {
                (SrfLookup::Found, v) => {
                    *slot = v;
                    if self.ecc.live {
                        self.ecc_check_plane(ctx, r, plane_stream(s, b), g, t, false, slot);
                        if self.ecc.last_bad.get() != 0 {
                            self.ecc.g16.set(self.ecc.g16.get() | 1 << b);
                        }
                    }
                }
                (SrfLookup::Upstream, _) => err = Some(Diag::Dir001),
                (SrfLookup::None, _) => {
                    if err.is_none() {
                        err = Some(Diag::Tim001)
                    }
                }
            }
        }
        match err {
            Some(d) => Err(d),
            None => Ok(out),
        }
    }
}

/// Byte b of lane l of a per-lane 32-bit result goes to plane b (little-endian elements); lanes at
/// or beyond `vlanes` stay zero.
// model: sys/operands.sail words_to_planes
pub fn words_to_planes(xs: &[u32; MAX_LANES], vlanes: usize) -> Planes {
    words_to_planes_lanes(xs, vlanes)
}

/// [`words_to_planes`] for the planes of `L` lanes, from at least `vlanes` words.
pub fn words_to_planes_lanes<const L: usize>(xs: &[u32], vlanes: usize) -> Planes<L> {
    let mut out = [[0u8; L]; 4];
    #[cfg(target_arch = "aarch64")]
    {
        let [a, b, c, d] = &mut out;
        let n = vlanes.min(L).min(xs.len());
        tsp_numerics::planes::words_to_planes(xs, 4, n, &mut [a, b, c, d]);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (b, row) in out.iter_mut().enumerate() {
        for (o, x) in row.iter_mut().zip(xs).take(vlanes) {
            *o = (x >> (8 * b)) as u8;
        }
    }
    out
}
