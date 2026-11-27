//! Patrol uses separate read and writeback slots, yielding to foreground SRAM traffic.
//! Any intervening slice mutation invalidates captured repair through write-snooping metadata.
//! The decoder observation is logged at read time; repair may be delayed or canceled.
use crate::ecc::{classify, data_flip, Class, OV0};
use crate::observe::Observer;
use crate::Machine;

#[derive(Debug)]
pub struct ScrubWrite {
    addr: u32,
    data: [u8; crate::geom::MAX_LANES],
    overlay: u32,
    revision: (u128, u128),
}

impl<const L: usize> Machine<L> {
    fn scrub_revision(&self, s: usize) -> (u128, u128) {
        (self.sram.revision[s], self.ecc.sram_revision.get(&(s as u8)).copied().unwrap_or(0))
    }

    /// Run the scrub cycles m <= `upto` that are still due.
    // model: postlude/tick.sail scrub_run
    pub(crate) fn scrub_run<O: Observer + ?Sized>(&mut self, upto: Option<u64>, obs: &mut O) {
        let period = u128::from(self.geom.scrub_period);
        let Some(upto) = upto.map(u128::from) else { return };
        if period == 0 {
            return;
        }
        // `scrub_next` is 128 bits wide: it may pass 2^64 once the run is at the end of the cycle range
        while self.d.scrub_next <= upto {
            let m = self.d.scrub_next as u64;
            self.scrub_cycle(m, obs);
            self.d.scrub_next += period;
        }
    }

    // model: postlude/tick.sail scrub_cycle
    fn scrub_cycle<O: Observer + ?Sized>(&mut self, m: u64, obs: &mut O) {
        for s in 0..88usize {
            if (s as u64 % 44) < self.geom.mem_slices && !self.scrub_issued(s, m) {
                if let Some(w) = self.d.scrub_pending[s].take() {
                    if w.revision == self.scrub_revision(s) {
                        self.sram.write(s, w.addr as u64, &w.data[..self.geom.vlanes as usize]);
                        self.ecc.set_sram_ov(s, w.addr, w.overlay);
                    }
                } else {
                    self.scrub_slice(s, m, obs);
                }
            }
        }
    }

    // model: postlude/tick.sail scrub_issued
    fn scrub_issued(&self, q: usize, m: u64) -> bool {
        let skew = self.latency_of_instr(&ots_isa::Insn::Read { addr: 0, dst: 0, thread: 0 }).d_skew as u64;
        let Some(d) = m.checked_sub(skew) else {
            return false;
        };
        self.d.path_busy(q, 0, d)
            || self.d.path_busy(q, 1, d)
            || (self.d.queue_last[q].is_some() && self.d.queue_last_at[q] == d)
    }

    // model: postlude/tick.sail scrub_slice
    fn scrub_slice<O: Observer + ?Sized>(&mut self, s: usize, m: u64, obs: &mut O) {
        let a = self.d.scrub_ptr[s];
        self.d.scrub_ptr[s] = ((a as u64 + 1) % self.geom.words_per_slice) as u32;
        if !self.ecc.live {
            return;
        }
        if let Some(w) = self.scrub_read(s, a, m, false, obs) {
            self.d.scrub_pending[s] = Some(w);
        }
    }

    /// Management scan after execution has stopped, before raw host reads or
    /// warm restart. Unlisted words have an all-zero overlay by construction.
    pub fn service_scrub(&mut self, cycle: u64) {
        for (slice, addr) in self.ecc.protected_words() {
            let s = usize::from(slice);
            if let Some(w) = self.scrub_read(s, addr, cycle, true, &mut crate::NoObserver) {
                self.sram.write(s, u64::from(addr), &w.data[..self.geom.vlanes as usize]);
                self.ecc.set_sram_ov(s, addr, w.overlay);
            }
        }
    }

    fn scrub_read<O: Observer + ?Sized>(
        &mut self,
        s: usize,
        a: u32,
        m: u64,
        management: bool,
        obs: &mut O,
    ) -> Option<Box<ScrubWrite>> {
        let id = self.ecc.sram_ov(s, a);
        if id == OV0 {
            return None;
        }
        let mut w = self.sram.read(s, a as u64);
        let mut nid = id;
        let mut fixed = false;
        for g in 0..self.geom.superlanes as usize {
            let e = self.ecc.get(id, g);
            if management && e & 0x200 != 0 {
                let monitor = &mut self.host.service_ras;
                monitor.stored_report.get_or_insert((s as u8, a, g as u8));
                monitor.events = monitor.events.saturating_add(1);
                monitor.fatal_count = monitor.fatal_count.saturating_add(1);
                monitor.sram_word = None;
            }
            match classify(e) {
                (Class::Ok, _) => {}
                (Class::Corrected, pos) => {
                    if let Some(j) = data_flip(e) {
                        w[16 * g + j as usize / 8] ^= 1 << (j % 8);
                    }
                    nid = self.ecc.set(nid, g, e & 0x200);
                    fixed = true;
                    obs.scrub(m, s as u8, a, g as u8, false);
                    self.ras_scrub_event(m, s, false, a, g as u8, pos, obs);
                }
                (Class::Uncorrectable, pos) => {
                    obs.scrub(m, s as u8, a, g as u8, true);
                    self.ras_scrub_event(m, s, true, a, g as u8, pos, obs);
                }
            }
        }
        fixed.then(|| Box::new(ScrubWrite { addr: a, data: w, overlay: nid, revision: self.scrub_revision(s) }))
    }
}
