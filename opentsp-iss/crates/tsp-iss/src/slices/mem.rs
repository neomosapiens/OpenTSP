//! MEM slice instructions (`slices/mem/mem_insts.sail`; ISCA 2020 sec. III-B; HC34 p.25).
//! Every MEM queue accesses its own slice (`issuer_of(q).slice`). Results go to the
//! slice's write gap on the destination stream's row; operands come from its read gap on the
//! source's row.
//!
//! The loopable references (Read, Write, Gather, Scatter) are issued by the cycle driver
//! (`driver::dispatch` `dispatch_slots`), reference 0 at the dispatch cycle and the AGT's
//! references at their cycles with `addr_offset` and `stream_offset` set (`driver::agt`); the AGT
//! records (Countdown, Step, StreamStep, Iterations) act at dispatch and execute as no-ops.
//!
//! Tile gating (`core/sram.sail`): a reference carries a liquid-thread tag t. A tile
//! executes it only when it is active: below the queue's SetVL length and in thread t (Config SL
//! on the queue). An inactive tile changes no SRAM, checks no address, and its superlane of the
//! result stream is zero (`write_planes` zeroes it).
use super::{Exec, Issue};
use crate::ecc::{Where, MAX_SL};
use crate::layout::{issuer_of, read_gap, write_gap, RowId};
use crate::observe::Observer;
use crate::operands::{issue_unit, stream_id, stream_row, STREAM_ROW_BIT};
use crate::sram::{all_tiles_active, tile_active};
use crate::srf::Vec320;
use crate::Machine;
use ots_isa::{Code as Diag, EntryId, Insn};

fn is_loopable_entry(i: &Insn) -> bool {
    matches!(i.id(), EntryId::Read | EntryId::Write | EntryId::Gather | EntryId::Scatter)
}

/// Which instructions the AGT can loop: Read, Write, Gather and Scatter.
// model: slices/mem/mem_insts.sail is_loopable
pub fn is_loopable(i: &Insn) -> bool {
    debug_assert_eq!(i.entry().agt_loopable, is_loopable_entry(i));
    is_loopable_entry(i)
}

/// The SRAM bank an access uses, as the dispatch sees it: none (an address outside the slice
/// accesses no bank; its execution is OTS-E-OPR-001), one bank, or any bank (an
/// indirect reference: its addresses come from a stream).
// model: slices/mem/mem_insts.sail bank_use
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankUse {
    None,
    Of(u8),
    Any,
}

/// Two accesses meet in a bank: never when one uses no bank, always when one may use any.
// model: slices/mem/mem_insts.sail banks_meet
pub fn banks_meet(a: BankUse, b: BankUse) -> bool {
    match (a, b) {
        (BankUse::None, _) | (_, BankUse::None) => false,
        (BankUse::Of(x), BankUse::Of(y)) => x == y,
        _ => true,
    }
}

/// The `(thread, bank)` of a reference at the dispatch (`ref_tag`).
// model: postlude/tick.sail ref_tag
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefTag {
    pub thread: u8,
    pub bank: BankUse,
}

/// A reference of a conflicting pair for the report: the instruction, its AGT address offset and its AGT stream offset.
// model: postlude/tick.sail clash_ref
pub type ClashRef = (Insn, i64, i64);

// model: postlude/tick.sail tags_clash
pub fn tags_clash(a: RefTag, b: RefTag) -> bool {
    a.thread == b.thread && banks_meet(a.bank, b.bank)
}

/// Verdict of the bank rule of one MEM queue and cycle (`bank_verdict`).
// model: postlude/tick.sail bank_verdict
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankVerdict {
    pub vbank: [bool; 4],
    pub vfm_clash: bool,
    pub vfm_loop: bool,
    pub vlblock: [bool; 2],
    /// Partner of each blocked AGT reference: -1 none, 4 the other path's AGT, 0..3 a new reference index.
    pub vlsrc: [i32; 2],
}

/// The bank rule of one MEM queue and cycle (`postlude/tick.sail` `bank_rule`): pure function of the
/// (thread, bank) tags of new references, which of them passed their checks, the AGT references due,
/// and whether an IfetchM uses a bank this cycle.
// model: postlude/tick.sail bank_rule
pub fn bank_rule(
    rt: [Option<RefTag>; 4],
    go: [bool; 4],
    loop_tags: [Option<RefTag>; 2],
    fm_active: bool,
    fm_bank: BankUse,
) -> BankVerdict {
    let mut bank = [false; 4];
    for t in 0..2 {
        for p in 0..2 {
            let k = 2 * p + t;
            let o = 2 * (1 - p) + t;
            if let Some(me) = rt[k] {
                if go[k] {
                    let new_clash = match rt[o] {
                        Some(j) => go[o] && tags_clash(me, j),
                        None => false,
                    };
                    let loop_clash = match loop_tags[1 - p] {
                        Some(tg) => tags_clash(me, tg),
                        None => false,
                    };
                    if new_clash || loop_clash {
                        bank[k] = true;
                    }
                }
            }
        }
    }
    let mut fm_clash = false;
    if fm_active {
        for t in 0..2 {
            if let Some(j) = rt[2 + t] {
                if go[2 + t] && banks_meet(j.bank, fm_bank) {
                    bank[2 + t] = true;
                    fm_clash = true;
                }
            }
        }
    }
    let mut fm_loop = false;
    if fm_active {
        if let Some(tg) = loop_tags[1] {
            if banks_meet(tg.bank, fm_bank) {
                fm_loop = true;
                fm_clash = true;
            }
        }
    }
    let mut lblock = [false; 2];
    let mut lsrc = [-1; 2];
    for p in 0..2 {
        if let Some(me) = loop_tags[p] {
            let mut hit = match loop_tags[1 - p] {
                Some(o) => tags_clash(me, o),
                None => false,
            };
            if hit {
                lsrc[p] = 4;
            }
            for t in 0..2 {
                let idx = 2 * (1 - p) + t;
                if let Some(j) = rt[idx] {
                    if go[idx] && tags_clash(me, j) {
                        hit = true;
                        lsrc[p] = idx as i32;
                    }
                }
            }
            lblock[p] = hit || (p == 1 && fm_loop);
        }
    }
    BankVerdict { vbank: bank, vfm_clash: fm_clash, vfm_loop: fm_loop, vlblock: lblock, vlsrc: lsrc }
}

/// Byte `l` of `v`, zero at or beyond `vlanes`.
// model: slices/mem/mem_insts.sail lane_byte
fn lane_byte<const L: usize>(v: &Vec320<L>, l: usize, vlanes: usize) -> u8 {
    if l < vlanes {
        v[l]
    } else {
        0
    }
}

impl<const L: usize> Machine<L> {
    /// The bank of address `e` (an effective address, before the range check).
    // model: slices/mem/mem_insts.sail addr_bank
    pub fn addr_bank(&self, e: i64) -> BankUse {
        if 0 <= e && (e as u64) < self.geom.words_per_slice {
            BankUse::Of(self.sram_bank(e as u32))
        } else {
            BankUse::None
        }
    }

    /// The bank of a direct reference at address offset `off`; any bank for an indirect one.
    // model: slices/mem/mem_insts.sail ref_bank
    pub fn ref_bank(&self, i: &Insn, off: i64) -> BankUse {
        match *i {
            Insn::Read { addr, .. } | Insn::Write { addr, .. } => self.addr_bank(addr as i64 + off),
            _ => BankUse::Any,
        }
    }

    // model: postlude/tick.sail tag_of
    pub fn tag_of(&self, i: &Insn, off: i64) -> RefTag {
        RefTag { thread: crate::driver::classify::ref_thread(i), bank: self.ref_bank(i, off) }
    }

    /// The address of a reference of a bank conflict as the report needs it (`clash_addr`): the direct address plus the AGT
    /// offset; zero for an indirect reference (its addresses come from a stream, not known at dispatch).
    // model: postlude/tick.sail clash_addr
    fn clash_addr(r: &ClashRef) -> i64 {
        match r.0 {
            Insn::Read { addr, .. } | Insn::Write { addr, .. } => addr as i64 + r.1,
            _ => 0,
        }
    }

    /// The data stream (with its AGT stream offset) of an R-type reference: a Read's destination or a Gather's.
    // model: postlude/tick.sail clash_stream
    fn clash_stream(r: &ClashRef) -> Option<u32> {
        let s = match r.0 {
            Insn::Read { dst, .. } | Insn::Gather { dst, .. } => dst,
            _ => return None,
        };
        let e = stream_id(s) as i64 + r.2;
        (0..=31).contains(&e).then_some(((s >> STREAM_ROW_BIT) & 1) << STREAM_ROW_BIT | e as u32)
    }

    /// The bank-conflict report (OTS-E-MEM-001 of the R-type reference `rd` and the W-type one `wr` dispatched in cycle `c`):
    /// S[2] of superlane 0, S[4..16] the write address and S[17..29] the read address, at the write gap of the slice in the
    /// commit cycle (dispatch + d_func), on the row of the read's data stream. A pair with no R-type data stream
    /// (an IfetchM) reports nothing in the spare bits.
    // model: postlude/tick.sail bank_report
    pub fn bank_report<O: Observer + ?Sized>(&mut self, q: usize, c: u64, a: &ClashRef, b: &ClashRef, obs: &mut O) {
        let (rd, wr) = if crate::driver::classify::ref_path(&a.0) == Some(0) { (a, b) } else { (b, a) };
        let Some(ds) = Self::clash_stream(rd) else { return };
        let r = stream_row(ds);
        let iu = issue_unit(q);
        let Some(g) = write_gap(&self.geom, iu.unit, iu.slice as usize, r) else { return };
        let mut sp = [0u32; 30];
        sp[2] = 1;
        let (wa, ra) = (Self::clash_addr(wr) as u32 & 0x1FFF, Self::clash_addr(rd) as u32 & 0x1FFF);
        for i in 0..13 {
            if wa >> i & 1 == 1 {
                sp[4 + i] |= 1;
            }
            if ra >> i & 1 == 1 {
                sp[17 + i] |= 1;
            }
        }
        let commit = c + self.latency_of_instr(&rd.0).d_func as u64;
        if self.srf.report(r, g, commit, 1, 1, &sp) {
            crate::driver::lint(obs, q, c, crate::driver::LINT_RAS_EVICT);
        }
        self.ecc.live = true; // every reader looks at the spare bits from now on
    }

    /// Tile `g` of the issuing slice executes the current MEM instruction (thread `ctx.thread`).
    // model: slices/mem/mem_insts.sail tile_on
    fn tile_on(&self, ctx: &Issue, g: u64) -> bool {
        let q = ctx.queue;
        if q <= 87 {
            tile_active(self.masks.queue_vl[q] as u64, self.masks.mem_thread_split[q] as u64, true, ctx.thread, g)
        } else {
            true
        }
    }

    /// Every tile of the issuing slice executes the current MEM instruction (`tile_on` for all `g`).
    fn all_tiles_on(&self, ctx: &Issue) -> bool {
        let q = ctx.queue;
        q > 87
            || all_tiles_active(
                self.masks.queue_vl[q] as u64,
                self.masks.mem_thread_split[q] as u64,
                true,
                ctx.thread,
                self.geom.superlanes,
            )
    }

    /// The data stream of the current reference: the instruction's stream plus the AGT StreamStep
    /// offset, on the same row; past stream 31 or below 0 it is OTS-E-OPR-001.
    // model: slices/mem/mem_insts.sail eff_stream
    fn eff_stream(&self, ctx: &Issue, s: u32) -> Option<u32> {
        let e = stream_id(s) as i64 + ctx.stream_offset;
        (0..=31).contains(&e).then_some(((s >> STREAM_ROW_BIT) & 1) << STREAM_ROW_BIT | e as u32)
    }

    /// The effective address of the current reference: the instruction's address plus the AGT
    /// offset, bounds-checked against the segment: below 0 or at or above the segment size it is
    /// OTS-E-OPR-001 (no wrap-around).
    // model: slices/mem/mem_insts.sail eff_addr
    fn eff_addr(&self, ctx: &Issue, a: u32) -> Option<u32> {
        let e = a as i64 + ctx.addr_offset;
        let seg = self.mem_segment[issuer_of(ctx.queue).slice as usize] as i64;
        (0 <= e && e < seg && e < 1i64 << self.geom.addr_bits).then_some(e as u32)
    }

    /// The address tile `g` takes from its slice of the map stream: the bottom bytes, little-endian
    /// (lanes 16g and 16g + 1; the low `addr_bits` bits, the bits above are ignored).
    // model: slices/mem/mem_insts.sail tile_addr
    fn tile_addr(&self, m: &Vec320<L>, g: u64) -> u32 {
        let vl = self.geom.vlanes as usize;
        let l = 16 * g as usize;
        let halfword = ((lane_byte(m, l + 1, vl) as u32) << 8) | lane_byte(m, l, vl) as u32;
        halfword & ((1u32 << self.geom.addr_bits) - 1)
    }

    // model: slices/mem/mem_insts.sail execute (MEM_DIRECT, MEM_GATHER, MEM_SCATTER, AGT
    // records, MEM_SETSEGMENT, MEM_POWERCONFIG, MEM_DEEPSLEEP)
    pub fn execute_mem<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, obs: &mut O) -> Exec {
        match *i {
            Insn::Read { addr, dst, thread } => self.execute_direct(ctx, true, addr, dst, thread, obs),
            Insn::Write { addr, src, thread } => self.execute_direct(ctx, false, addr, src, thread, obs),
            Insn::Gather { dst, map, k, thread } => self.execute_gather(ctx, dst, map, k, thread, obs),
            Insn::Scatter { src, map, k, thread } => self.execute_scatter(ctx, src, map, k, thread, obs),
            // Executed on their own, the AGT records have no effect: the cycle driver applies them.
            Insn::SetSegment { n } => {
                let s = issuer_of(ctx.queue).slice as usize;
                self.mem_segment[s] = (n as u64).min(self.geom.words_per_slice);
                Exec::Issued
            }
            _ => {
                debug_assert!(matches!(
                    i.id(),
                    EntryId::Countdown
                        | EntryId::Step
                        | EntryId::StreamStep
                        | EntryId::Iterations
                        | EntryId::PowerConfig
                        | EntryId::DeepSleep
                ));
                Exec::Issued
            }
        }
    }

    // model: slices/mem/mem_insts.sail execute (MEM_DIRECT)
    fn execute_direct<O: Observer + ?Sized>(
        &mut self,
        ctx: &Issue,
        read: bool,
        a0: u32,
        s0: u32,
        t: u32,
        obs: &mut O,
    ) -> Exec {
        let ctx = Issue { thread: t as u8, ..*ctx };
        let slice = issuer_of(ctx.queue).slice as usize;
        let (Some(a), Some(s)) = (self.eff_addr(&ctx, a0), self.eff_stream(&ctx, s0)) else {
            return Exec::Diagnostic(Diag::Opr001);
        };
        if read {
            // Inactive tiles' superlanes are zeroed by write_plane_sram.
            match self.sram_index(a) {
                None => Exec::Diagnostic(Diag::Opr001),
                Some(i) => match self.write_plane_sram(&ctx, s, slice, i, ctx.commit_at()) {
                    Ok(()) => Exec::Issued,
                    Err(d) => Exec::Diagnostic(d),
                },
            }
        } else if self.ecc.live {
            // the Write does not check the ECC: it stores the stream's 138 bits as they are, tile by tile
            let (p, pov) = match self.read_plane_raw(&ctx, s, ctx.exec_cycle() as i128) {
                Err(d) => return Exec::Diagnostic(d),
                Ok(x) => x,
            };
            let wspare = {
                let r = stream_row(s);
                let iu = issue_unit(ctx.queue);
                match read_gap(&self.geom, iu.unit, iu.slice as usize, r) {
                    Some(g) => self.srf.spare(r, stream_id(s), g, ctx.exec_cycle() as i128),
                    None => 0,
                }
            };
            let Some(old) = self.sram_read(slice, a) else { return Exec::Diagnostic(Diag::Opr001) };
            let mut w = old;
            let mut nid = self.ecc.sram_ov(slice, a);
            for g in 0..self.geom.superlanes {
                if self.tile_on(&ctx, g) {
                    let r = 16 * g as usize..16 * (g as usize + 1);
                    w[r.clone()].copy_from_slice(&p[r]);
                    let mut e = self.ecc.get(pov, g as usize);
                    // the report's spare bits flow with the value into the stored word (srf_read_ov)
                    if wspare >> g & 1 == 1 {
                        e |= 0x200;
                    }
                    nid = self.ecc.set(nid, g as usize, e);
                }
            }
            if self.sram_write(ctx.exec_cycle(), slice, a, &w, obs) {
                self.ecc_sram_store(slice, a, nid);
                Exec::Issued
            } else {
                Exec::Diagnostic(Diag::Opr001)
            }
        } else if self.all_tiles_on(&ctx) {
            // every tile takes the stream: the stream's bytes replace the old word whole (an address outside the
            // slice still fails the write)
            let t = ctx.exec_cycle() as i128;
            let p = match crate::operands::read_plane_at(&self.srf, &self.geom, &ctx, s, t) {
                Err(d) => return Exec::Diagnostic(d),
                Ok(p) => p,
            };
            let Some(i) = self.sram_index(a) else { return Exec::Diagnostic(Diag::Opr001) };
            self.sram.write(slice, i, p);
            obs.sram(ctx.exec_cycle(), slice as u8, a, p);
            Exec::Issued
        } else {
            match self.read_plane(&ctx, s, ctx.exec_cycle() as i128) {
                Err(d) => Exec::Diagnostic(d),
                Ok(p) => {
                    let Some(old) = self.sram_read(slice, a) else { return Exec::Diagnostic(Diag::Opr001) };
                    let mut w = old;
                    for g in 0..self.geom.superlanes {
                        if self.tile_on(&ctx, g) {
                            let r = 16 * g as usize..16 * (g as usize + 1);
                            w[r.clone()].copy_from_slice(&p[r]);
                        }
                    }
                    if self.sram_write(ctx.exec_cycle(), slice, a, &w, obs) {
                        Exec::Issued
                    } else {
                        Exec::Diagnostic(Diag::Opr001)
                    }
                }
            }
        }
    }

    /// The error report of a Gather or Scatter for the tiles in `bounds` (`mem_report`): S[0] (Gather) or S[1] (Scatter) of
    /// the tile's superlane, S[4..16] its 13-bit address, S[17..22] the map stream (row bit, then the id), S[23..28] the
    /// data stream; written in the commit cycle at the slice's write gap on the row of the data stream.
    // model: slices/mem/mem_insts.sail mem_report
    #[allow(clippy::too_many_arguments)]
    fn mem_report<O: Observer + ?Sized>(
        &mut self,
        ctx: &Issue,
        prio: i8,
        bounds: u32,
        addrs: &[u32; MAX_SL],
        data: u32,
        map: u32,
        obs: &mut O,
    ) {
        let r = stream_row(data);
        let iu = issue_unit(ctx.queue);
        let Some(g) = write_gap(&self.geom, iu.unit, iu.slice as usize, r) else { return };
        let mut sp = [0u32; 30];
        let (ms, ds) = (stream_id(map) as u32, stream_id(data) as u32);
        #[allow(clippy::needless_range_loop)]
        for x in 0..self.geom.superlanes as usize {
            if bounds >> x & 1 == 0 {
                continue;
            }
            sp[if prio == 3 { 0 } else { 1 }] |= 1 << x;
            for i in 0..13 {
                if addrs[x] >> i & 1 == 1 {
                    sp[4 + i] |= 1 << x;
                }
            }
            if stream_row(map) == RowId::W {
                sp[17] |= 1 << x;
            }
            for i in 0..5 {
                if ms >> i & 1 == 1 {
                    sp[18 + i] |= 1 << x;
                }
                if ds >> i & 1 == 1 {
                    sp[24 + i] |= 1 << x;
                }
            }
            if r == RowId::W {
                sp[23] |= 1 << x;
            }
        }
        if self.srf.report(r, g, ctx.commit_at(), prio, bounds, &sp) {
            crate::driver::lint(obs, ctx.queue, ctx.exec_cycle(), crate::driver::LINT_RAS_EVICT);
        }
        self.ecc.live = true; // every reader looks at the spare bits from now on
    }

    // model: slices/mem/mem_insts.sail execute (MEM_GATHER)
    fn execute_gather<O: Observer + ?Sized>(
        &mut self,
        ctx: &Issue,
        d0: u32,
        m: u32,
        kb: u32,
        t: u32,
        obs: &mut O,
    ) -> Exec {
        let ctx = Issue { thread: t as u8, ..*ctx };
        let slice = issuer_of(ctx.queue).slice as usize;
        let k = kb as usize;
        let vl = self.geom.vlanes as usize;
        let ds = self.eff_stream(&ctx, d0);
        let pm = self.read_plane(&ctx, m, ctx.exec_cycle() as i128);
        let (Some(d), Ok(pm)) = (ds, pm) else {
            return match (ds, pm) {
                (None, _) => Exec::Diagnostic(Diag::Opr001),
                (_, Err(e)) => Exec::Diagnostic(e),
                _ => unreachable!(),
            };
        };
        // the map stream is an operand like any other: a tile whose map superlane is uncorrectable makes no access
        let map_bad = self.ecc.bad.get();
        // Each active tile acts alone: a tile whose effective address is outside the
        // segment or the slice drives zero on its lane, the good tiles' lanes are written, and
        // the instruction reports one OPR-001 (an exception to "a diagnostic means no effect").
        let mut out = [0u8; L];
        let mut bad = false;
        let mut bounds = 0u32;
        let mut taddr = [0u32; MAX_SL];
        for g in 0..self.geom.superlanes {
            taddr[g as usize] = self.tile_addr(&pm, g);
            if self.tile_on(&ctx, g) && map_bad >> g & 1 == 0 {
                match self.eff_addr(&ctx, self.tile_addr(&pm, g)) {
                    None => {
                        bad = true;
                        bounds |= 1 << g;
                    }
                    Some(a) => match self.sram_read(slice, a) {
                        Some(mut w) => {
                            if self.ecc.live {
                                // the tile checks its own superlane of the stored word before it uses the byte
                                let e = self.ecc.get(self.ecc.sram_ov(slice, a), g as usize);
                                let r = 16 * g as usize..16 * (g as usize + 1);
                                let at = Where::Sram { slice: slice as u8, addr: a };
                                let ex = ctx.exec_cycle();
                                if self.ecc.check_sl(&mut w[r], e, ex, ctx.queue, at, g as usize) {
                                    self.ecc.bad.set(self.ecc.bad.get() | 1 << g);
                                }
                            }
                            let l = 16 * g as usize + k;
                            if l < vl {
                                out[l] = lane_byte(&w, l, vl);
                            }
                        }
                        None => {
                            bad = true;
                            bounds |= 1 << g;
                        }
                    },
                }
            }
        }
        let r = match self.write_plane(&ctx, d, &out, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(e) => Exec::Diagnostic(e),
        };
        if bad {
            self.mem_report(&ctx, 3, bounds, &taddr, d0, m, obs);
            self.ecc.after_diag.set(true);
            self.ecc.data_opr.set(true);
            Exec::Diagnostic(Diag::Opr001)
        } else {
            r
        }
    }

    // model: slices/mem/mem_insts.sail execute (MEM_SCATTER)
    fn execute_scatter<O: Observer + ?Sized>(
        &mut self,
        ctx: &Issue,
        s0: u32,
        m: u32,
        kb: u32,
        t: u32,
        obs: &mut O,
    ) -> Exec {
        let ctx = Issue { thread: t as u8, ..*ctx };
        let slice = issuer_of(ctx.queue).slice as usize;
        let k = kb as usize;
        let vl = self.geom.vlanes as usize;
        let Some(s) = self.eff_stream(&ctx, s0) else { return Exec::Diagnostic(Diag::Opr001) };
        let t = ctx.exec_cycle() as i128;
        let ps = match self.read_plane(&ctx, s, t) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(ps) => ps,
        };
        // the data operand's uncorrectable superlanes are set aside while the map is read, so that the map's can be
        // told apart: a tile whose map superlane is uncorrectable writes nothing
        let data_bad = self.ecc.bad.replace(0);
        let pm = match self.read_plane(&ctx, m, t) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(pm) => pm,
        };
        let map_bad = self.ecc.bad.get();
        self.ecc.bad.set(data_bad | map_bad);
        // Each active tile acts alone: the tiles with an effective address inside the
        // segment write their lane, the others write nothing, and the instruction reports one
        // OPR-001 if any tile was bad. Tiles write in ascending order (each writes only its own
        // lane of the word).
        let mut bad = false;
        let mut bounds = 0u32;
        let mut taddr = [0u32; MAX_SL];
        for g in 0..self.geom.superlanes {
            taddr[g as usize] = self.tile_addr(&pm, g);
            if self.tile_on(&ctx, g) && map_bad >> g & 1 == 0 {
                match self.eff_addr(&ctx, self.tile_addr(&pm, g)) {
                    None => {
                        bad = true;
                        bounds |= 1 << g;
                    }
                    Some(a) => match self.sram_read(slice, a) {
                        Some(mut w) => {
                            let mut nov = None;
                            if self.ecc.live {
                                // the tile checks the old superlane, replaces its byte and regenerates the superlane's
                                // check bits; a superlane with an uncorrectable old word or data byte is stored poisoned
                                let wov = self.ecc.sram_ov(slice, a);
                                let e = self.ecc.get(wov, g as usize);
                                let r = 16 * g as usize..16 * (g as usize + 1);
                                let at = Where::Sram { slice: slice as u8, addr: a };
                                if self.ecc.check_sl(&mut w[r], e, ctx.exec_cycle(), ctx.queue, at, g as usize) {
                                    self.ecc.bad.set(self.ecc.bad.get() | 1 << g);
                                }
                                let poisoned = self.ecc.bad.get() >> g & 1 == 1;
                                let ne = if poisoned { crate::ecc::POISON_SL } else { 0 };
                                nov = Some(self.ecc.set(wov, g as usize, ne));
                            }
                            let l = 16 * g as usize + k;
                            if l < vl {
                                w[l] = ps[l];
                            }
                            self.ecc_flush(obs);
                            let _ = self.sram_write(ctx.exec_cycle(), slice, a, &w, obs);
                            if let Some(id) = nov {
                                self.ecc_sram_store(slice, a, id);
                            }
                        }
                        None => {
                            bad = true;
                            bounds |= 1 << g;
                        }
                    },
                }
            }
        }
        if bad {
            self.mem_report(&ctx, 2, bounds, &taddr, s0, m, obs);
            self.ecc.after_diag.set(true);
            self.ecc.data_opr.set(true);
            Exec::Diagnostic(Diag::Opr001)
        } else {
            Exec::Issued
        }
    }
}
