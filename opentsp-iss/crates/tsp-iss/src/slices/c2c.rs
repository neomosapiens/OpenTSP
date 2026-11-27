//! C2C instructions Send, Receive, Transmit, Deskew and RuntimeDeskew (`slices/c2c/c2c_insts.sail`;
//! ISCA 2020 sec. II and Table I; ISCA 2022 sec. 2.2, 2.3 and 3).
//!
//! The model sees only its own end of a link (`core/callbacks.sail`: `c2c_peer`, `c2c_send`,
//! `c2c_recv_ready`, `c2c_recv`, here the [`LinkPort`](crate::LinkPort) of the machine); the harness
//! wires the links and delays every vector by the link's latency. Deskew and RuntimeDeskew only
//! hold their queue (`driver::dispatch` `occupancy`); their execute clauses do nothing.
use super::{Exec, Issue};
use crate::layout::write_gap;
use crate::observe::Observer;
use crate::operands::{aligned, issue_unit, one_plane, stream_row};
use crate::srf::{narrow, widen, Vec320};
use crate::Machine;
use ots_isa::{Code as Diag, Insn};

/// `c2c_notification`: the vector Transmit sends, lane 0 = 1 and every other lane 0.
// model: slices/c2c/c2c_insts.sail c2c_notification
fn c2c_notification<const L: usize>() -> Vec320<L> {
    let mut v = [0u8; L];
    v[0] = 1;
    v
}

impl<const L: usize> Machine<L> {
    /// The link index, when it names one of the chip's links.
    // model: slices/c2c/c2c_insts.sail c2c_link
    fn c2c_link(&self, link: u32) -> Option<u8> {
        if (link as u64) < self.geom.links {
            Some(link as u8)
        } else {
            None
        }
    }

    /// `chip` names the chip at the far end of the link.
    // model: slices/c2c/c2c_insts.sail c2c_peer_ok
    fn c2c_peer_ok(&self, chip: u32, l: u8) -> bool {
        self.links.peer(l) == Some(chip as u16)
    }

    /// A link is busy in cycle `c` when it accepted a vector fewer than `c2c.cycles_per_vector`
    /// cycles before; executions reach this in non-decreasing cycle order.
    // model: slices/c2c/c2c_insts.sail c2c_busy
    fn c2c_busy(&self, l: u8, c: u64) -> bool {
        self.c2c_accepted[l as usize].is_some_and(|a| a + self.geom.cycles_per_vector > c)
    }

    /// The vector goes on the link in this cycle: the link has accepted it.
    // model: slices/c2c/c2c_insts.sail c2c_put
    fn c2c_put(&mut self, ctx: &Issue, l: u8, ptype: u8, v: &Vec320<L>, poison: u32) {
        self.c2c_accepted[l as usize] = Some(ctx.exec_cycle());
        if poison != 0 {
            self.links.send_poisoned(ctx.exec_cycle(), l, &widen(v), poison);
        } else {
            self.links.send_packet(ctx.exec_cycle(), l, ptype, &widen(v));
        }
    }

    // model: slices/c2c/c2c_insts.sail execute C2C_SEND
    fn c2c_send_insn(&mut self, ctx: &Issue, chip: u32, link: u32, s: u32) -> Exec {
        let Some(l) = self.c2c_link(link) else { return Exec::Diagnostic(Diag::Opr001) };
        if !self.c2c_peer_ok(chip, l) {
            return Exec::Diagnostic(Diag::Opr001);
        }
        if self.c2c_busy(l, ctx.exec_cycle()) {
            return Exec::Diagnostic(Diag::Lnk001);
        }
        if self.links.packet_busy(ctx.exec_cycle(), l) {
            return Exec::Diagnostic(Diag::Lnk003);
        }
        match self.read_planes(ctx, s, 1, ctx.exec_cycle() as i128) {
            Err(d) => Exec::Diagnostic(d),
            Ok(p) => {
                // a vector with an uncorrectable superlane goes out poisoned in the same superlanes
                let poison = if self.ecc.live { self.ecc.bad.get() } else { 0 };
                self.c2c_put(ctx, l, crate::ctl::PT_DATA, &p[0], poison);
                Exec::Issued
            }
        }
    }

    // model: slices/c2c/c2c_insts.sail execute C2C_TRANSMIT
    fn c2c_transmit_insn(&mut self, ctx: &Issue, chip: u32, link: u32) -> Exec {
        let Some(l) = self.c2c_link(link) else { return Exec::Diagnostic(Diag::Opr001) };
        if !self.c2c_peer_ok(chip, l) {
            return Exec::Diagnostic(Diag::Opr001);
        }
        if self.c2c_busy(l, ctx.exec_cycle()) {
            return Exec::Diagnostic(Diag::Lnk001);
        }
        if self.links.packet_busy(ctx.exec_cycle(), l) {
            return Exec::Diagnostic(Diag::Lnk003);
        }
        self.c2c_put(ctx, l, crate::ctl::PT_SYNC, &c2c_notification(), 0);
        Exec::Issued
    }

    // model: slices/c2c/c2c_insts.sail execute C2C_RECEIVE
    fn c2c_receive_insn(&mut self, ctx: &Issue, chip: u32, link: u32, s: u32) -> Exec {
        let Some(l) = self.c2c_link(link) else { return Exec::Diagnostic(Diag::Opr001) };
        if !self.c2c_peer_ok(chip, l) {
            return Exec::Diagnostic(Diag::Opr001);
        }
        if !self.links.recv_ready(ctx.exec_cycle(), l) {
            return Exec::Diagnostic(Diag::Tim001);
        }
        if !aligned(s, 1) {
            return Exec::Diagnostic(Diag::Opr002);
        }
        // h2: the destination is checked before the vector is taken, so a rejected
        // Receive leaves the link FIFO unchanged (errors have no effect).
        let iu = issue_unit(ctx.queue);
        if write_gap(&self.geom, iu.unit, iu.slice as usize, stream_row(s)).is_none() {
            return Exec::Diagnostic(Diag::Dir001);
        }
        let v = self.links.recv(ctx.exec_cycle(), l);
        // a poisoned vector arrives poisoned in the same superlanes (inherited: the reader reports it)
        let poison = self.links.recv_poison();
        if poison != 0 {
            self.ecc.live = true;
            self.ecc.inherit.set(poison);
        }
        self.ecc.link_bad.set(self.links.recv_bad(l));
        match self.write_planes(ctx, s, &one_plane(narrow::<L>(&v)), 1, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(d) => Exec::Diagnostic(d),
        }
    }

    /// The diagnostics of Receive that do not depend on the arrival, which DeskewRecv reports at its dispatch (and then
    /// holds nothing): a link or a chip_id that is not the far end of the link (OTS-E-OPR-001), and a stream that the issuing
    /// unit cannot write (OTS-E-DIR-001).
    // model: slices/c2c/c2c_insts.sail deskewrecv_diag
    pub(crate) fn deskewrecv_diag(&self, q: usize, chip: u32, link: u32, s: u32) -> Option<Diag> {
        let Some(l) = self.c2c_link(link) else { return Some(Diag::Opr001) };
        if !self.c2c_peer_ok(chip, l) {
            return Some(Diag::Opr001);
        }
        let iu = issue_unit(q);
        if write_gap(&self.geom, iu.unit, iu.slice as usize, stream_row(s)).is_none() {
            return Some(Diag::Dir001);
        }
        None
    }

    /// Has a vector arrived on the link by cycle `c`, as a DeskewRecv's poll asks (`deskewrecv_ready`)?
    // model: slices/c2c/c2c_insts.sail deskewrecv_ready
    pub(crate) fn deskewrecv_ready(&mut self, c: u64, link: u32) -> bool {
        match self.c2c_link(link) {
            Some(l) => self.links.recv_ready(c, l),
            None => false,
        }
    }

    /// The arrival cycle of the next vector that the harness knows on the link (`deskewrecv_next_arrival`).
    // model: slices/c2c/c2c_insts.sail deskewrecv_next_arrival
    pub(crate) fn deskewrecv_next_arrival(&self, link: u32) -> Option<u128> {
        self.c2c_link(link).and_then(|l| self.links.next_arrival(l))
    }

    /// PciReceive: the oldest vector of the host's ingress FIFO onto the stream; an empty FIFO is OTS-E-TIM-001, as
    /// Receive. The destination is checked before the vector is taken.
    // model: slices/c2c/c2c_insts.sail execute C2C_PCIRECV
    fn pci_receive_insn(&mut self, ctx: &Issue, s: u32) -> Exec {
        if !self.host.recv_ready() {
            return Exec::Diagnostic(Diag::Tim001);
        }
        let iu = issue_unit(ctx.queue);
        if write_gap(&self.geom, iu.unit, iu.slice as usize, stream_row(s)).is_none() {
            return Exec::Diagnostic(Diag::Dir001);
        }
        let v = self.pci_recv_pop(ctx.exec_cycle());
        match self.write_planes(ctx, s, &one_plane(narrow::<L>(&v)), 1, ctx.commit_at()) {
            Ok(()) => Exec::Issued,
            Err(d) => Exec::Diagnostic(d),
        }
    }

    /// PciSend: the vector of the stream goes to the host; at most one per `host.cycles_per_vector` cycles
    /// (OTS-E-HIO-001 otherwise, with no effect).
    // model: slices/c2c/c2c_insts.sail execute C2C_PCISEND
    fn pci_send_insn<O: Observer + ?Sized>(&mut self, ctx: &Issue, s: u32, obs: &mut O) -> Exec {
        if self.host.pci_busy(ctx.exec_cycle(), self.geom.host_cycles_per_vector) {
            return Exec::Diagnostic(Diag::Hio001);
        }
        match self.read_planes(ctx, s, 1, ctx.exec_cycle() as i128) {
            Err(d) => Exec::Diagnostic(d),
            Ok(p) => {
                self.pci_send_put(ctx, &widen(&p[0]), obs);
                Exec::Issued
            }
        }
    }

    // model: slices/c2c/c2c_insts.sail execute (the C2C clauses)
    pub fn execute_c2c<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, obs: &mut O) -> Exec {
        self.rp.touch_links();
        match *i {
            Insn::Send { chip_id, link, stream } => {
                obs.link_op(link.min(255) as u8, true);
                self.c2c_send_insn(ctx, chip_id, link, stream)
            }
            Insn::Receive { chip_id, link, stream } => {
                obs.link_op(link.min(255) as u8, false);
                self.c2c_receive_insn(ctx, chip_id, link, stream)
            }
            Insn::Transmit { chip_id, link } => {
                obs.link_op(link.min(255) as u8, true);
                self.c2c_transmit_insn(ctx, chip_id, link)
            }
            // Deskew and RuntimeDeskew only hold their queue (occupancy)
            Insn::Deskew | Insn::RuntimeDeskew { .. } => Exec::Issued,
            // DeskewRecv: the polling is the driver's (`dispatch_body`, `tick`); what executes is the Receive it becomes
            Insn::DeskewRecv { link, .. } => {
                obs.link_op(link.min(255) as u8, false);
                Exec::Issued
            }
            Insn::PciReceive { stream } => self.pci_receive_insn(ctx, stream),
            Insn::PciSend { stream } => self.pci_send_insn(ctx, stream, obs),
            // HostWait parks its queue at dispatch (`dispatch_body`); here only the doorbell number is checked
            Insn::HostWait { i } => {
                if (i as u64) >= self.geom.host_doorbells {
                    Exec::Diagnostic(Diag::Opr001)
                } else {
                    Exec::Issued
                }
            }
            Insn::HostNotify { n } => {
                self.host_msix(ctx.exec_cycle(), n as usize, obs);
                Exec::Issued
            }
            _ => unreachable!("not a C2C instruction: {:?}", i.id()),
        }
    }
}
