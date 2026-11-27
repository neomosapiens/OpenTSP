//! VXM instructions (`slices/vxm/vxm_insts.sail`; ISCA 2020 sec. III-C; `microarchitecture.md`
//! sec. 5). Every lane computes independently. Operands are read at the VXM gap in cycle
//! dispatch + d_skew; results commit at dispatch + d_func (the entry's latency, `timing.overrides`
//! included: `Machine::latency_of_instr`).
//!
//! The 35 entries are the binary operations (opcodes 0..13), the unary operations (Not 14, Neg
//! 15, Abs 16, ReLU 20, TanH 21, Exp 22, RSqrt 23, Log 24), Mask 17, Select 18, Cmp 19, the eight
//! fixed conversions (25..32), Reduce 33 and Convert 34. Bit 18 of the word (`sg4`) makes every operand a 4-stream group and performs 4 / width operations per lane;
//! the chain bits read a VXM result committed at the station in the cycle
//! before the read. The kernels are the `tsp-numerics` crate's.
use super::{Exec, Issue};
use crate::geom::MAX_LANES;
use crate::observe::Observer;
use crate::operands::{words_to_planes_lanes, Planes, ZERO_PLANE};
use crate::Machine;
use ots_isa::Insn;
use tsp_numerics::lanes::{self, LaneOp, ReduceOp};
use tsp_numerics::EType;

/// A source operand: the stream and its chain bit (`vsrc`).
#[derive(Clone, Copy, Debug)]
struct Vsrc {
    s: u32,
    chain: bool,
}

fn vs(s: u32, c: u32) -> Vsrc {
    Vsrc { s, chain: c & 1 == 1 }
}

fn ety(code: u32) -> EType {
    EType::from_code(code as u8).expect("decoded VXM type code")
}

/// Element `l` of a multi-byte operand: byte `b` of lane `l` lives in plane `b`.
// model: slices/vxm/vxm_insts.sail elem
fn elem<const L: usize>(p: &Planes<L>, w: usize, l: usize, vlanes: usize) -> u32 {
    let mut x = 0u32;
    if l < vlanes {
        for (b, plane) in p.iter().enumerate().take(w) {
            x |= (plane[l] as u32) << (8 * b);
        }
    }
    x
}

/// Element `l` of the `w`-byte sub-operand starting at plane `o` of a 4-plane group operand.
// model: slices/vxm/vxm_insts.sail elem_at
fn elem_at<const L: usize>(p: &Planes<L>, w: usize, o: usize, l: usize, vlanes: usize) -> u32 {
    let mut x = 0u32;
    if l < vlanes {
        for b in 0..w {
            let k = o + b;
            if k <= 3 {
                x |= (p[k][l] as u32) << (8 * b);
            }
        }
    }
    x
}

impl<const L: usize> Machine<L> {
    /// Read a VXM source operand: a chained one (chain bit 1) at the station in cycle t - 1's
    /// commits, any other by the flow rule.
    // model: slices/vxm/vxm_insts.sail vxm_read
    fn vxm_read(&self, ctx: &Issue, a: Vsrc, w: u8) -> Result<Planes<L>, ots_isa::Code> {
        self.read_planes_core(ctx, a.s, w, ctx.exec_cycle() as i128, a.chain)
    }

    /// Read up to three operands of width `ws`, apply `op` per lane, schedule a result of width
    /// `wd`. Group bit `g`: every operand is the 4-stream group at its stream (SG4
    /// aligned, else OTS-E-OPR-002) and the ALU performs 4 / `ws` independent operations per
    /// lane, one on each `ws`-byte part of the group; the result is the 4-stream group at `dst`.
    // model: slices/vxm/vxm_insts.sail vxm_lanes
    #[allow(clippy::too_many_arguments)] // the arguments of the model's vxm_lanes
    fn vxm_lanes(
        &mut self,
        ctx: &Issue,
        op: LaneOp,
        dst: u32,
        a: Vsrc,
        b: Option<Vsrc>,
        m: Option<Vsrc>,
        ws: usize,
        wd: usize,
        g: bool,
    ) -> Exec {
        if !g {
            return self.vxm_lanes_planes(ctx, op, dst, a, b, m, ws, wd);
        }
        let pa = self.vxm_read(ctx, a, 4);
        let pb = b.map(|s| self.vxm_read(ctx, s, 4)).transpose();
        let pm = m.map(|s| self.vxm_read(ctx, s, 4)).transpose();
        let (xa, xb, xm) = match (pa, pb, pm) {
            (Err(e), _, _) => return Exec::Diagnostic(e),
            (_, Err(e), _) => return Exec::Diagnostic(e),
            (_, _, Err(e)) => return Exec::Diagnostic(e),
            (Ok(a), Ok(b), Ok(m)) => (a, b, m),
        };
        let vl = self.geom.vlanes as usize;
        let mut out = [0u32; L];
        let (xb, xm) = (xb.unwrap_or([[0u8; L]; 4]), xm.unwrap_or([[0u8; L]; 4]));
        let parts = match ws {
            1 => 4,
            2 => 2,
            _ => 1,
        };
        #[allow(clippy::needless_range_loop)] // lane l is passed to the kernels and indexes the result
        for l in 0..vl {
            let mut x = 0u32;
            for i in 0..parts {
                let o = i * ws;
                if o <= 3 {
                    let r = lanes::apply(
                        op,
                        elem_at(&xa, ws, o, l, vl),
                        elem_at(&xb, ws, o, l, vl),
                        elem_at(&xm, ws, o, l, vl),
                    );
                    for j in 0..ws {
                        let k = o + j;
                        if k <= 3 {
                            x = (x & !(0xff << (8 * k))) | (((r >> (8 * j)) & 0xff) << (8 * k));
                        }
                    }
                }
            }
            out[l] = x;
        }
        let planes = words_to_planes_lanes(&out, vl);
        match self.write_planes(ctx, dst, &planes, 4, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(e) => Exec::Diagnostic(e),
        }
    }

    /// [`Machine::vxm_lanes`] without the group bit: the operands are borrowed from the register file, the
    /// kernel works on the byte planes and writes the result planes.
    #[allow(clippy::too_many_arguments)] // the arguments of vxm_lanes
    fn vxm_lanes_planes(
        &mut self,
        ctx: &Issue,
        op: LaneOp,
        dst: u32,
        a: Vsrc,
        b: Option<Vsrc>,
        m: Option<Vsrc>,
        ws: usize,
        wd: usize,
    ) -> Exec {
        let vl = self.geom.vlanes as usize;
        let mut planes = [[0u8; L]; 4];
        {
            // an absent operand is read as zero planes
            let t = ctx.exec_cycle() as i128;
            let read = |s: Vsrc| self.read_planes_refs(ctx, s.s, ws as u8, t, s.chain);
            let pa = read(a);
            let pb = b.map(read).transpose();
            let pm = m.map(read).transpose();
            let (xa, xb, xm) = match (pa, pb, pm) {
                (Err(e), _, _) => return Exec::Diagnostic(e),
                (_, Err(e), _) => return Exec::Diagnostic(e),
                (_, _, Err(e)) => return Exec::Diagnostic(e),
                (Ok(a), Ok(b), Ok(m)) => (a, b, m),
            };
            let zero = [&ZERO_PLANE[..vl]; 4];
            let (xb, xm) = (xb.unwrap_or(zero), xm.unwrap_or(zero));
            let [o0, o1, o2, o3] = &mut planes;
            // 8-bit integer binary operations work on the byte planes directly
            if let (1, LaneOp::Bin(o, t @ (EType::I8 | EType::U8))) = (ws, op) {
                lanes::bin8(o, t == EType::I8, xa[0], xb[0], &mut o0[..vl]);
            } else {
                let mut out = [&mut o0[..vl], &mut o1[..vl], &mut o2[..vl], &mut o3[..vl]];
                tsp_numerics::planes::apply_planes(op, ws, wd, &xa, &xb, &xm, &mut out);
            }
        }
        match self.write_planes(ctx, dst, &planes, wd as u8, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(e) => Exec::Diagnostic(e),
        }
    }

    /// Reduce 33: per 16-lane superlane group, lane 16g gets the result over lanes 16g..16g+15 in
    /// ascending order (rounding at each step, ); the other lanes are zero.
    // model: slices/vxm/vxm_insts.sail execute VXM_REDUCE
    fn vxm_reduce(&mut self, ctx: &Issue, op: ReduceOp, t: EType, dst: u32, a: Vsrc) -> Exec {
        let w = t.width() as usize;
        let p = match self.vxm_read(ctx, a, w as u8) {
            Err(e) => return Exec::Diagnostic(e),
            Ok(p) => p,
        };
        let vl = self.geom.vlanes as usize;
        let mut out = [0u32; L];
        for g in 0..self.geom.superlanes as usize {
            let mut grp = [0u32; 16];
            for (j, e) in grp.iter_mut().enumerate() {
                *e = elem(&p, w, 16 * g + j, vl);
            }
            if 16 * g < MAX_LANES {
                out[16 * g] = lanes::reduce_fold(op, t, &grp);
            }
        }
        match self.write_planes(ctx, dst, &words_to_planes_lanes(&out, vl), w as u8, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(e) => Exec::Diagnostic(e),
        }
    }

    // model: slices/vxm/vxm_insts.sail execute (VXM_BIN, VXM_UN, VXM_MASK, VXM_SELECT, VXM_CMP,
    // VXM_CONV, VXM_CVT, VXM_REDUCE)
    pub fn execute_vxm<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, _obs: &mut O) -> Exec {
        use ots_isa::EntryId as E;
        match *i {
            Insn::Add { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Add_sat { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Add_mod { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Sub { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Sub_sat { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Sub_mod { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Mul { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Mul_sat { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Mul_mod { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Max { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Min { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::And { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Or { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain }
            | Insn::Xor { r#type: t, dst, lhs, rhs, sg4, lhs_chain, rhs_chain } => {
                let op = match i.id() {
                    E::Add => 0,
                    E::Add_sat => 1,
                    E::Add_mod => 2,
                    E::Sub => 3,
                    E::Sub_sat => 4,
                    E::Sub_mod => 5,
                    E::Mul => 6,
                    E::Mul_sat => 7,
                    E::Mul_mod => 8,
                    E::Max => 9,
                    E::Min => 10,
                    E::And => 11,
                    E::Or => 12,
                    _ => 13,
                };
                let t = ety(t);
                let w = t.width() as usize;
                self.vxm_lanes(
                    ctx,
                    LaneOp::Bin(op, t),
                    dst,
                    vs(lhs, lhs_chain),
                    Some(vs(rhs, rhs_chain)),
                    None,
                    w,
                    w,
                    sg4 == 1,
                )
            }
            Insn::Not { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::Neg { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::Abs { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::ReLU { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::TanH { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::Exp { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::RSqrt { r#type: t, dst, lhs, sg4, lhs_chain }
            | Insn::Log { r#type: t, dst, lhs, sg4, lhs_chain } => {
                let op = match i.id() {
                    E::Not => 14,
                    E::Neg => 15,
                    E::Abs => 16,
                    E::ReLU => 20,
                    E::TanH => 21,
                    E::Exp => 22,
                    E::RSqrt => 23,
                    _ => 24,
                };
                let t = ety(t);
                let w = t.width() as usize;
                self.vxm_lanes(ctx, LaneOp::Un(op, t), dst, vs(lhs, lhs_chain), None, None, w, w, sg4 == 1)
            }
            // Mask 17: dst = rhs != 0 ? lhs : 0 (the operand `mask` is the second source).
            Insn::Mask { r#type: t, dst, lhs, mask, sg4, lhs_chain, mask_chain } => {
                let t = ety(t);
                let w = t.width() as usize;
                self.vxm_lanes(
                    ctx,
                    LaneOp::Mask(t),
                    dst,
                    vs(lhs, lhs_chain),
                    Some(vs(mask, mask_chain)),
                    None,
                    w,
                    w,
                    sg4 == 1,
                )
            }
            // Select 18: dst = mask != 0 ? lhs : rhs.
            Insn::Select { r#type: t, dst, lhs, rhs, mask, sg4, lhs_chain, rhs_chain, mask_chain } => {
                let t = ety(t);
                let w = t.width() as usize;
                self.vxm_lanes(
                    ctx,
                    LaneOp::Select(t),
                    dst,
                    vs(lhs, lhs_chain),
                    Some(vs(rhs, rhs_chain)),
                    Some(vs(mask, mask_chain)),
                    w,
                    w,
                    sg4 == 1,
                )
            }
            Insn::Cmp { r#type: t, dst, lhs, rhs, mode, sg4, lhs_chain, rhs_chain } => {
                let t = ety(t);
                let w = t.width() as usize;
                self.vxm_lanes(
                    ctx,
                    LaneOp::Cmp(mode, t),
                    dst,
                    vs(lhs, lhs_chain),
                    Some(vs(rhs, rhs_chain)),
                    None,
                    w,
                    w,
                    sg4 == 1,
                )
            }
            Insn::Conv_I8_F32 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_F32_I8 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_I8_F16 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_F16_I8 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_I8_I32 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_I32_I8 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_F16_F32 { dst, lhs, lhs_chain, .. }
            | Insn::Conv_F32_F16 { dst, lhs, lhs_chain, .. } => {
                let op = match i.id() {
                    E::Conv_I8_F32 => 25,
                    E::Conv_F32_I8 => 26,
                    E::Conv_I8_F16 => 27,
                    E::Conv_F16_I8 => 28,
                    E::Conv_I8_I32 => 29,
                    E::Conv_I32_I8 => 30,
                    E::Conv_F16_F32 => 31,
                    _ => 32,
                };
                let (s, t) = (lanes::conv_src(op), lanes::conv_dst(op));
                self.vxm_lanes(
                    ctx,
                    LaneOp::Conv(op),
                    dst,
                    vs(lhs, lhs_chain),
                    None,
                    None,
                    s.width() as usize,
                    t.width() as usize,
                    false,
                )
            }
            // Convert 34: the type field is the source type, `dtype` the destination's code.
            Insn::Convert { r#type: t, dst, lhs, dtype, lhs_chain } => {
                let (st, dt) = (ety(t), ety(dtype));
                self.vxm_lanes(
                    ctx,
                    LaneOp::Cvt(st, dt),
                    dst,
                    vs(lhs, lhs_chain),
                    None,
                    None,
                    st.width() as usize,
                    dt.width() as usize,
                    false,
                )
            }
            Insn::Reduce { r#type: t, dst, lhs, mode, lhs_chain } => {
                let op = match mode {
                    0 => ReduceOp::Add,
                    1 => ReduceOp::Max,
                    _ => ReduceOp::Min,
                };
                self.vxm_reduce(ctx, op, ety(t), dst, vs(lhs, lhs_chain))
            }
            _ => unreachable!("not a VXM instruction: {:?}", i.id()),
        }
    }
}
