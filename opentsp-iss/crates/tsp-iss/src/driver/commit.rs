//! The commit ring (`postlude/tick.sail` `commit_due`; `sys/operands.sail` `pending`): the stream
//! writes scheduled for each cycle modulo 128, applied by `tick` before anything dispatches, with
//! OTS-E-SRF-002 for colliding writers, the liquid-thread merge of one MEM slice and the SXM
//! select/transposer OR.
//!
//! The model's list of a slot is built by prepending, so its first element is the newest commit.
//! The ISS appends (`write_planes`): a slot is the model's list reversed, and `commit_due` reverses
//! it back before it runs the model's algorithm, whose results (which writer stands for a thread
//! pair, the order of the `T srf` lines) depend on the list order.
use super::{phase, Driver};
use crate::layout::{issuer_of, UnitKind};
use crate::observe::Observer;
pub use crate::operands::Commit;
use crate::Machine;
use ots_isa::Code as Diag;

impl<const L: usize> Driver<L> {
    /// Set the commits of ring slot `k` from a list in the model's order (newest first). A test and
    /// debugger hook.
    pub fn set_pending_list(&mut self, k: usize, mut list: Vec<Commit<L>>) {
        list.reverse();
        if !list.is_empty() {
            self.pending_occ |= 1 << k;
        }
        self.pending[k] = list;
    }
}

/// Writer order for OTS-E-SRF-002: the lowest (queue, plane) writer keeps the value.
// model: postlude/tick.sail writer_before
fn writer_before<const L: usize>(a: &Commit<L>, b: &Commit<L>) -> bool {
    a.q < b.q || (a.q == b.q && a.b < b.b)
}

// model: postlude/tick.sail same_target
fn same_target<const L: usize>(a: &Commit<L>, b: &Commit<L>) -> bool {
    a.r == b.r && a.id == b.id && a.g == b.g
}

/// The SXM OR merge (HC34 p.31, ISCA 2020 Fig. 4): the select (SEL) and transposer
/// (TSP) outputs of one hemisphere pass through an OR before the turn-out. Two writes merge when
/// one comes from that hemisphere's SEL queue and the other from one of its TSP queues; any other
/// pair of writes to one target collides (OTS-E-SRF-002). SEL = 1, TSP = 2, anything else 0; and
/// the hemisphere (0 west, 1 east).
// model: postlude/tick.sail sxm_or_kind
fn sxm_or_kind(q: usize) -> (u8, u8) {
    if q > 143 {
        return (0, 0);
    }
    let i = issuer_of(q);
    match i.unit {
        UnitKind::SxmWSel | UnitKind::SxmESel => (1, i.sxm),
        UnitKind::SxmWTsp | UnitKind::SxmETsp => (2, i.sxm),
        _ => (0, 0),
    }
}

// model: postlude/tick.sail or_pair
fn or_pair<const L: usize>(a: &Commit<L>, b: &Commit<L>) -> bool {
    let (ka, ha) = sxm_or_kind(a.q);
    let (kb, hb) = sxm_or_kind(b.q);
    ka != 0 && kb != 0 && ka != kb && ha == hb
}

fn or_into<const L: usize>(acc: &mut [u8; L], x: &[u8; L]) {
    for (a, b) in acc.iter_mut().zip(x.iter()) {
        *a |= *b;
    }
}

/// Commit `e` (at position `i`) against the cycle's list `xs` (in the model's order): (collides,
/// kept, value). The writes to one row/stream/gap in one cycle form a group (the same gap and cycle
/// is the same diagonal). The writes of one (queue, byte plane) are one writer: on a MEM queue
/// they are the two liquid threads' results, each zero on the other thread's tiles, so their OR is
/// the per-tile merge; the first of them in the list stands for the writer. A group of
/// exactly two writers that are an OR pair merges: no collision, and the lower writer stores the OR
/// of both (the result does not depend on the order of the records). Any other group of two or
/// more writers collides: each reports, the lowest (queue, plane) writer keeps its own value.
// model: postlude/tick.sail commit_wins
fn commit_wins<const L: usize>(
    e: &Commit<L>,
    i: usize,
    xs: &[Commit<L>],
    ecc: &mut crate::ecc::Ecc,
) -> (bool, bool, [u8; L], u32) {
    let mut others = 0u64;
    let mut seen = [false; 145 * 4];
    let mut paired = true;
    let mut first = true;
    let mut stands = true;
    let mut own = e.value;
    let mut merged = [0u8; L];
    let (mut own_ov, mut merged_ov) = (e.ov, 0u32);
    for (j, o) in xs.iter().enumerate() {
        if same_target(e, o) {
            if o.q == e.q && o.b == e.b {
                if j != i {
                    or_into(&mut own, &o.value);
                    own_ov = ecc.or(own_ov, o.ov);
                    if j < i {
                        stands = false;
                    }
                }
            } else {
                let writer = 4 * o.q + usize::from(o.b);
                if !seen[writer] {
                    others += 1;
                    seen[writer] = true;
                }
                if !or_pair(e, o) {
                    paired = false;
                }
                if writer_before(o, e) {
                    first = false;
                }
                or_into(&mut merged, &o.value);
                merged_ov = ecc.or(merged_ov, o.ov);
            }
        }
    }
    if !stands {
        (false, false, own, own_ov)
    } else if others == 0 {
        (false, true, own, own_ov)
    } else if others == 1 && paired {
        let mut v = own;
        or_into(&mut v, &merged);
        (false, first, v, ecc.or(own_ov, merged_ov))
    } else {
        (true, first, own, own_ov)
    }
}

impl<const L: usize> Machine<L> {
    /// Apply the commits due at cycle `c`, then clear their ring slot. Two writes to one row,
    /// stream and gap in one cycle are OTS-E-SRF-002, reported on each writer's queue (once per
    /// written stream); the lowest (queue, plane) writer's value is kept. A SEL and a
    /// TSP write of one SXM hemisphere are not a collision: the register takes their OR.
    // model: postlude/tick.sail commit_due
    pub fn commit_due<O: Observer + ?Sized>(&mut self, c: u64, obs: &mut O) {
        let k = (c % super::PENDING_SLOTS as u64) as usize;
        self.d.pending_occ &= !(1 << k);
        if self.d.pending[k].is_empty() {
            return;
        }
        let mut all = std::mem::take(&mut self.d.pending[k]);
        // The values are applied in the model's list order (newest commit first), which is the ring's order reversed.
        // Only writes to one target interact, so a cycle in which no two writes share one needs no judging.
        let n = all.len();
        let lone = if n <= 8 {
            (0..n).all(|i| (i + 1..n).all(|j| !same_target(&all[i], &all[j])))
        } else {
            // One bit per target (row, stream, gap): a second commit to a target finds its bit set. The bits are cleared
            // again below, so the set is all zero between calls and a cycle costs O(n), not a sort.
            let seen = &mut self.d.commit_seen;
            if seen.is_empty() {
                seen.resize(1 << 11, 0); // 2^17 targets
            }
            let key = |e: &Commit<L>| ((e.r as usize) << 16) | ((e.id as usize) << 8) | e.g as usize;
            let mut lone = true;
            let mut set = 0;
            for e in all.iter() {
                let k = key(e);
                if seen[k >> 6] & (1u64 << (k & 63)) != 0 {
                    lone = false;
                    break;
                }
                seen[k >> 6] |= 1u64 << (k & 63);
                set += 1;
            }
            for e in &all[..set] {
                seen[key(e) >> 6] = 0;
            }
            lone
        };
        if lone {
            for e in all.iter().rev() {
                let m = phase::begin();
                self.srf_write_ov(e.r, e.id, e.g, c, &e.value, e.ov, obs);
                phase::sem_end(phase::SRF, m);
            }
            all.clear();
            self.d.pending[k] = all;
            return;
        }
        all.reverse();
        // Each group of equal targets is judged on its own (its members in list order); a lone write wins with its
        // own value.
        let mut order = std::mem::take(&mut self.d.commit_order);
        order.clear();
        order.extend(0..all.len() as u32);
        order.sort_by_key(|&i| {
            let e = &all[i as usize];
            ((e.r as u32) << 16) | ((e.id as u32) << 8) | e.g as u32
        });
        let mut hits = [0u64; crate::geom::QUEUES];
        let mut verdict: Vec<(bool, bool, Option<usize>)> = vec![(false, true, None); all.len()];
        let mut values: Vec<([u8; L], u32)> = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && same_target(&all[order[start] as usize], &all[order[end] as usize]) {
                end += 1;
            }
            if end - start > 1 {
                let group: Vec<Commit<L>> = order[start..end].iter().map(|&i| all[i as usize].clone()).collect();
                for (pos, &i) in order[start..end].iter().enumerate() {
                    let (collide, win, value, vov) = commit_wins(&group[pos], pos, &group, &mut self.ecc);
                    values.push((value, vov));
                    verdict[i as usize] = (collide, win, Some(values.len() - 1));
                }
            }
            start = end;
        }
        for (e, &(collide, win, value)) in all.iter().zip(&verdict) {
            if collide {
                hits[e.q] += 1;
            }
            if win {
                let (v, ov) = value.map_or((&e.value, e.ov), |v| (&values[v].0, values[v].1));
                self.srf_write_ov(e.r, e.id, e.g, c, v, ov, obs);
            }
        }
        // Reports in queue order, so they do not depend on the order of the records.
        for (q, n) in hits.iter().enumerate() {
            for _ in 0..*n {
                self.diag(obs, q, c, Diag::Srf002);
            }
        }
        self.d.commit_order = order;
        all.clear();
        self.d.pending[k] = all; // the slot keeps its allocation
    }
}
