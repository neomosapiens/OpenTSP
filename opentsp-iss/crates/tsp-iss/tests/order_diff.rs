//! The reporting order within a cycle (`microarchitecture.md` §8, steps 0 to 6) for the pairs of
//! codes the golden vectors never put in one cycle: LNK-002 (step 0), SRF-002 (step 2), the
//! execution diagnostics of earlier dispatches (step 3), the control channel (step 4), the queues
//! (step 5: QUE-001, QUE-002, MEM-001, OPR-001, RES-001, RES-002, SXM-001, LNK-001, TIM-001) and
//! RES-003 (step 6).
//!
//! Each program is built with exact timing so that a set of diagnostics falls in one cycle. It is
//! run on `tsp_sim` and on the ISS with `--trace` (also ticking every cycle): exit code and stdout
//! must be byte-identical, and the diagnostics of that cycle must be exactly the list the case
//! states, which is the order of §8.
mod common;
use common::prog::*;
use common::*;
use ots_isa::Insn;
use std::collections::BTreeSet;
use tsp_iss::harness::links::link_peer;
use tsp_iss::layout::{issuer_of, read_gap, RowId, UnitKind};

fn add(dst: u32, lhs: u32, rhs: u32) -> Insn {
    Insn::Add { r#type: 0, dst, lhs, rhs, sg4: 0, lhs_chain: 0, rhs_chain: 0 }
}

struct Ctx {
    e: Env,
    r: Rng,
    p: Prog,
}

fn ctx(profile: &str) -> Ctx {
    Ctx { e: env(profile), r: Rng::new(7), p: Prog { exact: true, ..Prog::default() } }
}

impl Ctx {
    fn vxm_gap(&self) -> u8 {
        read_gap(&self.e.geom, UnitKind::Vxm, 0, RowId::E).unwrap()
    }

    /// Two Adds on queues 88 and 89 that write E:4 in one cycle `c`: SRF-002 for both, at `c`.
    fn collide(&mut self, c: i64) {
        let (d_func, d_skew) = lat(&self.e, &add(4, 0, 1));
        let t = c - d_func;
        let g = self.vxm_gap();
        self.p.feed(&mut self.r, &self.e, RowId::E, g, 0, 2, t + d_skew);
        assert!(self.p.put(88, t, add(4, 0, 1)) && self.p.put(89, t, add(4, 0, 1)));
    }

    /// An Add on `q` with no operand, executing at `c`: TIM-001.
    fn tim(&mut self, c: i64, q: u16) {
        let (_, d_skew) = lat(&self.e, &add(8, 2, 3));
        assert!(self.p.put(q, c - d_skew, add(8, 2, 3)));
    }

    /// The same LW from two queues of MXM plane 0 executing at `c`: RES-003 for both.
    fn twin_lw(&mut self, c: i64) {
        let lw = Insn::LW { r#type: 0, src: 2 };
        let (_, d_skew) = lat(&self.e, &lw);
        let g = read_gap(&self.e.geom, UnitKind::MxmW, 0, RowId::W).unwrap();
        self.p.feed(&mut self.r, &self.e, RowId::W, g, 0, 16, c);
        assert!(self.p.put(104, c - d_skew, lw) && self.p.put(105, c - d_skew, lw));
    }

    /// A parcel of two Adds on `q` (a non-MEM queue) dispatching at `c`: QUE-001 for each.
    fn que001(&mut self, c: i64, q: u16) {
        assert!(self.p.put_all(q, c, vec![add(8, 2, 3), add(12, 2, 3)]));
    }

    /// A Read and a Write of one bank in one parcel on MEM queue `q` at `c`: MEM-001 for each.
    fn mem001(&mut self, c: i64, q: u16) {
        let g = vec![Insn::Read { addr: 0, dst: 2, thread: 0 }, Insn::Write { addr: 0, src: 3, thread: 0 }];
        assert!(self.p.put_all(q, c, g));
    }

    /// `n` Sends on TX queue 126, link 0, from cycle `t0`, `gap` cycles apart (each fed).
    fn sends(&mut self, t0: i64, gap: i64, n: i64) {
        let l = 0;
        let (peer, _) = link_peer(1, self.e.links, 0, l).unwrap_or((0, l));
        let ins = Insn::Send { chip_id: peer as u32, link: l as u32, stream: 32 + 3 };
        let (_, d_skew) = lat(&self.e, &ins);
        let g = read_gap(&self.e.geom, issuer_of(126).unit, 0, RowId::W).unwrap();
        for k in 0..n {
            let t = t0 + k * gap;
            self.p.feed(&mut self.r, &self.e, RowId::W, g, 3, 1, t + d_skew);
            assert!(self.p.put(126, t, ins));
        }
    }

    /// A LoadMap of `map` from queue `q` (west SXM) executing at `c`, its source stream `src`
    /// carrying `entries` (the low byte of each lane's entry).
    fn load_map(&mut self, c: i64, q: u16, map: u32, src: u32, entries: &[u16]) {
        let ins = Insn::LoadMap { src: 32 + src, map };
        let (_, d_skew) = lat(&self.e, &ins);
        let g = read_gap(&self.e.geom, issuer_of(q as usize).unit, 0, RowId::W).unwrap();
        let data: Vec<u8> = entries.iter().map(|x| *x as u8).collect();
        self.p.feed_data(&mut self.r, &self.e, RowId::W, g, src, &data, c, &mut Default::default());
        assert!(self.p.put(q, c - d_skew, ins));
    }

    /// An Add on the control channel at `c`: QUE-002.
    fn que002_control(&mut self, c: i64) {
        assert!(self.p.put(0xffff, c, add(8, 2, 3)));
    }
}

/// The diagnostics of `cycle` as (queue, code without its prefix), in the order printed.
fn at(out: &str, cycle: u64) -> Vec<(u32, String)> {
    out.lines()
        .filter(|l| l.starts_with("T diag "))
        .filter_map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            (f[2].parse::<u64>().unwrap() == cycle)
                .then(|| (f[4].parse().unwrap(), f[5].strip_prefix("OTS-E-").unwrap().to_string()))
        })
        .collect()
}

/// Run `text` on both implementations, require identical output, return the reference's.
fn run_both(profile: &str, args: &[&str], text_in: &str) -> String {
    let cfg = if matches!(profile, "ref" | "mini") {
        config(profile)
    } else {
        spec().join("config/test").join(format!("{profile}.json"))
    };
    let mut a = vec![cfg.to_str().unwrap(), "--trace"];
    a.extend_from_slice(args);
    let theirs = run_sim(&a, text_in);
    let ours = run_iss(&a, text_in);
    let mut every = a.clone();
    every.push("--tick-every-cycle");
    let ticked = run_bin(iss(), &every, text_in.as_bytes());
    for (what, o) in [("ISS", &ours), ("ISS ticking every cycle", &ticked)] {
        assert!(
            o.status.code() == theirs.status.code() && o.stdout == theirs.stdout,
            "{what} differs on {profile}:\n{text_in}\n{}\ntsp_sim:\n{}",
            text(&o.stdout),
            text(&theirs.stdout)
        );
    }
    assert_eq!(theirs.status.code(), Some(0), "{}", text(&theirs.stderr));
    text(&theirs.stdout)
}

fn expect(out: &str, cycle: u64, want: &[(u32, &str)]) {
    let want: Vec<(u32, String)> = want.iter().map(|(q, c)| (*q, c.to_string())).collect();
    let diags: Vec<&str> = out.lines().filter(|l| l.starts_with("T diag ")).collect();
    assert_eq!(at(out, cycle), want, "cycle {cycle}:\n{}", diags.join("\n"));
}

/// The ordered pairs of codes of different diagnostics of `cycle` include `want` (each (first, last)).
fn has_pairs(out: &str, cycle: u64, want: &[(&str, &str)]) {
    let lines: Vec<&str> =
        out.lines().filter(|l| l.starts_with("T diag ") && l.split(' ').nth(2) == Some(&cycle.to_string())).collect();
    let mut got = BTreeSet::new();
    pairs(&lines.join("\n"), &mut got, &mut BTreeSet::new());
    for (a, b) in want {
        assert!(got.contains(&(format!("OTS-E-{a}"), format!("OTS-E-{b}"))), "{a} then {b} in {got:?}");
    }
}

// step 4 before step 5 before step 6, and the dispatch checks of queues in ascending order:
// QUE-002 on the control channel, MEM-001, QUE-001, TIM-001, RES-003 in one cycle
#[test]
fn control_queue_memory_slot_operand_and_state_reports_come_in_the_order_of_the_steps() {
    let mut c = ctx("ref");
    let t = 100;
    c.que002_control(t);
    c.mem001(t, 0);
    c.que001(t, 90);
    c.tim(t, 89);
    c.twin_lw(t);
    let out = run_both("ref", &[], &c.p.text());
    expect(
        &out,
        t as u64,
        &[
            (89, "TIM-001"),
            (65535, "QUE-002"),
            (0, "MEM-001"),
            (0, "MEM-001"),
            (90, "QUE-001"),
            (90, "QUE-001"),
            (104, "RES-003"),
            (105, "RES-003"),
        ],
    );
    has_pairs(
        &out,
        t as u64,
        &[("MEM-001", "QUE-001"), ("QUE-001", "RES-003"), ("QUE-002", "MEM-001"), ("MEM-001", "RES-003")],
    );
}

// step 0, 2, 3, 6 in one cycle: the third vector of a link whose receive FIFO holds two is dropped
// at its arrival (LNK-002), as two Adds collide (SRF-002), an Add lacks its operands and two
// queues load one MXM weight buffer
#[test]
fn a_dropped_vector_comes_before_collisions_operand_reports_and_state_changes() {
    let sends = |c: &mut Ctx| c.sends(60, 3, 3);
    // the arrival cycle of the third vector, from a run of the sends alone
    let mut probe = ctx("link_tight");
    sends(&mut probe);
    let first = run_both("link_tight", &[], &probe.p.text());
    let lnk = first.lines().find(|l| l.contains("LNK-002")).expect("a vector is dropped");
    let t: i64 = lnk.split(' ').nth(2).unwrap().parse().unwrap();
    let mut c = ctx("link_tight");
    sends(&mut c);
    c.collide(t);
    c.tim(t, 90);
    c.twin_lw(t);
    let out = run_both("link_tight", &[], &c.p.text());
    expect(
        &out,
        t as u64,
        &[(65535, "LNK-002"), (88, "SRF-002"), (89, "SRF-002"), (90, "TIM-001"), (104, "RES-003"), (105, "RES-003")],
    );
    has_pairs(
        &out,
        t as u64,
        &[
            ("LNK-002", "SRF-002"),
            ("LNK-002", "TIM-001"),
            ("LNK-002", "RES-003"),
            ("SRF-002", "RES-003"),
            ("TIM-001", "RES-003"),
        ],
    );
}

// a Send on a link that took a vector one cycle earlier (LNK-001) with a collision, a slot-rule
// break, a bank clash and a missing operand in the same cycle
#[test]
fn a_busy_link_is_reported_with_the_queues_around_it() {
    let mut c = ctx("link_tight");
    let t = 100;
    c.sends(t - 1, 1, 2); // the second Send, at t, finds the link busy (3 cycles per vector)
    c.collide(t);
    c.que001(t, 91);
    c.mem001(t, 0);
    c.tim(t, 90);
    let out = run_both("link_tight", &[], &c.p.text());
    expect(
        &out,
        t as u64,
        &[
            (88, "SRF-002"),
            (89, "SRF-002"),
            (90, "TIM-001"),
            (0, "MEM-001"),
            (0, "MEM-001"),
            (91, "QUE-001"),
            (91, "QUE-001"),
            (126, "LNK-001"),
        ],
    );
    has_pairs(
        &out,
        t as u64,
        &[("SRF-002", "LNK-001"), ("MEM-001", "LNK-001"), ("QUE-001", "LNK-001"), ("TIM-001", "LNK-001")],
    );
}

// a Permute whose map is not bijective (SXM-001) in the cycle two other queues load one map
// (RES-003), with a bank clash, a slot-rule break and a missing operand
#[test]
fn a_bad_map_is_reported_before_the_state_changes_of_the_same_cycle() {
    let mut c = ctx("ref");
    let n = c.e.vlanes as u16;
    let mut entries: Vec<u16> = (0..n).collect();
    entries[1] = 0; // two lanes take lane 0
    let load = lat(&c.e, &Insn::LoadMap { src: 0, map: 0 });
    let t_load = 70;
    c.load_map(t_load + load.1, 116, 1, 1, &entries);
    // a Permute at the map's ready cycle: the execute cycle of the load plus `superlanes`
    let t = t_load + load.1 + c.e.superlanes as i64;
    let perm = Insn::Permute { dst: 5, src: 32 + 6, map: 1 };
    let (_, d_skew) = lat(&c.e, &perm);
    let g = read_gap(&c.e.geom, issuer_of(124).unit, 0, RowId::W).unwrap();
    c.p.feed(&mut c.r, &c.e, RowId::W, g, 6, 1, t);
    assert!(c.p.put(124, t - d_skew, perm));
    let ident: Vec<u16> = (0..n).collect();
    c.load_map(t, 117, 2, 8, &ident);
    c.load_map(t, 118, 2, 9, &ident);
    c.mem001(t, 0);
    c.que001(t, 91);
    c.tim(t, 90);
    let out = run_both("ref", &[], &c.p.text());
    expect(
        &out,
        t as u64,
        &[
            (90, "TIM-001"),
            (0, "MEM-001"),
            (0, "MEM-001"),
            (91, "QUE-001"),
            (91, "QUE-001"),
            (124, "SXM-001"),
            (117, "RES-003"),
            (118, "RES-003"),
        ],
    );
    has_pairs(
        &out,
        t as u64,
        &[("MEM-001", "SXM-001"), ("QUE-001", "SXM-001"), ("SXM-001", "RES-003"), ("TIM-001", "SXM-001")],
    );
}

// a second weight group on a matrix of one group (RES-002) with the queues around it
#[test]
fn a_full_weight_buffer_is_reported_with_the_queues_around_it() {
    let mut c = ctx("mini");
    let t = 100;
    let lw = |id: u32| Insn::LW { r#type: 0, src: (32 + id) / 16 };
    let (_, d_skew) = lat(&c.e, &lw(0));
    let g = read_gap(&c.e.geom, UnitKind::MxmW, 0, RowId::W).unwrap();
    for (k, id) in [0u32, 16].into_iter().enumerate() {
        let tk = t - 1 + k as i64; // the second group at `t`
        c.p.feed(&mut c.r, &c.e, RowId::W, g, id, 16, tk + d_skew);
        assert!(c.p.put(104, tk, lw(id)));
    }
    c.que001(t, 91);
    c.mem001(t, 0);
    c.tim(t, 90);
    let out = run_both("mini", &[], &c.p.text());
    expect(
        &out,
        t as u64,
        &[(90, "TIM-001"), (0, "MEM-001"), (0, "MEM-001"), (91, "QUE-001"), (91, "QUE-001"), (104, "RES-002")],
    );
    has_pairs(&out, t as u64, &[("TIM-001", "RES-002"), ("MEM-001", "RES-002"), ("QUE-001", "RES-002")]);
}
