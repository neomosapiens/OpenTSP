//! ICU instructions (`slices/cu/cu_insts.sail`; ISCA 2020 sec. III-A and Table I; HC34 p.25, p.27):
//! NOP, Sync, Notify, Repeat, Ifetch, Config, SetVL. They are common to all functional slices, so
//! they issue on every data queue. The control channel (queue 0xFFFF) carries Sync, Notify and
//! Config only (`driver::dispatch` `control`).
//!
//! NOP, Sync, Notify and Repeat do their work at dispatch (the barrier, Repeat, the queue hold):
//! their `execute` only issues. `Config` splits a MEM slice's tiles into liquid threads and
//! `SetVL` sets the vector length of the issuing queue. `Ifetch` reads two consecutive vectors of one stream and
//! `IfetchM` reads the slice's SRAM; both hand the text to the issuing queue's fill FIFO.
use super::{Exec, Issue};
use crate::driver::{LocalFetch, StreamFetch};
use crate::layout::issuer_of;
use crate::observe::Observer;
use crate::operands::ZERO_PLANE;
use crate::Machine;
use ots_isa::{Code as Diag, EntryId, Insn};

impl<const L: usize> Machine<L> {
    // model: slices/cu/cu_insts.sail execute (NOP, SYNC, NOTIFY, REPEAT, CONFIG, SETVL, IFETCH);
    // slices/cu/cu_ifetchm.sail execute (IFETCHM)
    pub fn execute_cu<O: Observer + ?Sized>(&mut self, i: &Insn, ctx: &Issue, _obs: &mut O) -> Exec {
        let q = ctx.queue;
        match *i {
            // Config v: on a MEM queue, v = SL (0..19) splits the slice's tiles into liquid
            // threads: tiles 0..SL are thread 0, the rest thread 1 (Config 6: tiles 0-6 are thread 0 and
            // tiles 7-19 thread 1). Values above
            // 19 put every tile in thread 0. Elsewhere it configures power only.
            Insn::Config { v } => {
                if q <= 87 {
                    let sl = v as u64;
                    let sp = self.geom.superlanes;
                    self.masks.mem_thread_split[q] = if sl < sp { sl + 1 } else { sp } as u8;
                }
                Exec::Issued
            }
            // SetVL n: the vector length of the issuing queue's slice becomes n x 16 lanes; the other
            // superlanes are inactive (ISCA 2020 sec. II-F). n = 1..20.
            Insn::SetVL { n } => {
                if q <= 143 {
                    self.masks.queue_vl[q] = (n as u64).clamp(1, self.geom.superlanes.max(1)) as u8;
                }
                Exec::Issued
            }
            // Ifetch s: the ONE stream operand s (any stream) carries two consecutive vectors, read at the
            // execute cycle t and at t + 1; a missing value in either read is TIM-001 and nothing is fetched. The
            // 2 x vlanes bytes enter the issuing queue's IQ from t + 1 + `icu.fetch_latency`, at
            // `icu.fetch_bytes_per_cycle`. A full staging FIFO fetches nothing (IQ-003).
            Insn::Ifetch { stream } => match self.read_planes(ctx, stream, 1, ctx.exec_cycle() as i128) {
                Err(d) => Exec::Diagnostic(d),
                Ok(_) if !self.iq.can_fetch(q) => Exec::Diagnostic(Diag::Iq003),
                Ok(a) => {
                    self.d.stream_fetches.push(StreamFetch {
                        issue: *ctx,
                        stream,
                        first: crate::srf::widen(&a[0]),
                        bad: self.ecc.bad.get(),
                    });
                    Exec::Issued
                }
            },
            // IfetchM a: the words a and a + 1 of the issuing slice, inside its segment;
            // only MEM queues issue it (QUE-002 elsewhere), an address pair outside is OPR-001.
            Insn::IfetchM { addr } => {
                let e = addr as u64 + 1;
                if q > 87 {
                    Exec::Diagnostic(Diag::Que002)
                } else if e >= self.mem_segment[q] || e >= self.geom.words_per_slice {
                    Exec::Diagnostic(Diag::Opr001)
                } else {
                    let slice = issuer_of(q).slice as usize;
                    let at = ctx.exec_cycle();
                    if u128::from(at) < self.d.local_fetch_busy[q] {
                        return Exec::Diagnostic(Diag::Que001);
                    }
                    if !self.iq.can_fetch(q) {
                        return Exec::Diagnostic(Diag::Iq003);
                    }
                    let first = self.sram.read(slice, addr as u64);
                    let ov = self.ecc.sram_ov(slice, addr);
                    let checks = (0..self.geom.superlanes as usize).map(|g| self.ecc.get(ov, g)).collect();
                    self.d.local_fetches.push(LocalFetch {
                        queue: q,
                        ready: u128::from(at) + 1,
                        addr: e as u32,
                        first,
                        checks,
                    });
                    self.d.local_fetch_busy[q] = u128::from(at) + 2;
                    Exec::Issued
                }
            }
            _ => {
                debug_assert!(matches!(i.id(), EntryId::NOP | EntryId::Sync | EntryId::Notify | EntryId::Repeat));
                Exec::Issued
            }
        }
    }
    fn admitted_fetch<O: Observer + ?Sized>(&self, c: u64, q: usize, obs: &mut O) {
        if self.iq.observe_sites {
            let base = self.iq.fill[q].back().expect("accepted fetch").base;
            obs.site(self.chip_id(), c, q, "fetch", format_args!("{base} {}", self.geom.vlanes / 2));
        }
        obs.fetch(c, q as u16, 2 * self.geom.vlanes as u32);
    }

    pub(crate) fn stream_fetch_complete<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        if self.d.stream_fetches.is_empty() {
            return;
        }
        self.d.stream_fetches.sort_by_key(|f| f.issue.queue);
        let mut pending = std::mem::take(&mut self.d.stream_fetches);
        pending.retain(|f| {
            if f.issue.exec_cycle() >= c {
                return true;
            }
            let q = f.issue.queue;
            self.ecc.begin();
            let outcome = match self.read_planes(&f.issue, f.stream, 1, c as i128) {
                Err(d) => Exec::Diagnostic(d),
                Ok(b) => {
                    let bad = self.ecc.bad.get();
                    let vl = self.geom.vlanes as usize;
                    let mut checks = Vec::new();
                    if self.iq.tx.track && (bad | f.bad) != 0 {
                        for mask in [f.bad, bad] {
                            checks.extend((0..self.geom.superlanes).map(|g| {
                                if mask >> g & 1 == 1 {
                                    crate::ecc::POISON_SL
                                } else {
                                    0
                                }
                            }));
                        }
                    }
                    if self.iq.fetch_ov(q, c, &f.first[..vl], &b[0][..vl], checks) {
                        // The second operand read reports ECC and raises its IRQ before
                        // Sail observes admission. The instruction diagnostic follows admission.
                        if self.ecc.live && self.iq.observe_sites {
                            self.ecc_flush(obs);
                        }
                        self.admitted_fetch(c, q, obs);
                        Exec::Issued
                    } else {
                        Exec::Diagnostic(Diag::Iq003)
                    }
                }
            };
            if self.ecc.live {
                self.ecc_finish(q, c, outcome, obs);
            } else if let Exec::Diagnostic(d) = outcome {
                self.diag(obs, q, c, d);
            }
            if self.rp.compiled {
                self.iq.fill[q].clear();
            }
            false
        });
        self.d.stream_fetches = pending;
    }

    // model: slices/cu/cu_ifetchm.sail local_fetch_complete
    pub(crate) fn local_fetch_complete<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        if self.d.local_fetches.is_empty() {
            return;
        }
        self.d.local_fetches.sort_by_key(|f| f.queue);
        let mut pos = 0;
        while pos < self.d.local_fetches.len() {
            if self.d.local_fetches[pos].ready > u128::from(c) {
                pos += 1;
                continue;
            }
            let mut f = self.d.local_fetches.remove(pos);
            let slice = issuer_of(f.queue).slice as usize;
            let vl = self.geom.vlanes as usize;
            let second = self.sram.word_ref(slice, f.addr as u64).unwrap_or(&ZERO_PLANE[..vl]);
            let ov = self.ecc.sram_ov(slice, f.addr);
            f.checks.extend((0..self.geom.superlanes as usize).map(|g| self.ecc.get(ov, g)));
            if self.iq.fetch_ov(f.queue, c, &f.first[..vl], second, f.checks) {
                self.admitted_fetch(c, f.queue, obs);
            } else {
                self.diag(obs, f.queue, c, Diag::Iq003);
            }
            if self.rp.compiled {
                self.iq.fill[f.queue].clear();
            }
        }
    }
}
