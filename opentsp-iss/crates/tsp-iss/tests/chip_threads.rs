//! Chip threads: the chips of a window run on `--jobs N` threads, each chip's output is
//! buffered and printed in chip order, so the bytes do not depend on the number of threads.
//!
//! The programs are random Send, Receive and Transmit traffic on 16 chips (two nodes) with every latency
//! option, sometimes with a small `--max-cycles`; each is run with 1, 3 and 16 threads and every
//! byte of stdout, stderr and the exit code must be equal.
mod common;
use common::*;
use ots_isa::{encode, Insn};
use ots_profile::Profile;
use std::fmt::Write as _;
use std::sync::Arc;
use tsp_iss::geom::{Geom, MAX_LANES};
use tsp_iss::harness::chips::{run_chips, ChipRun};
use tsp_iss::harness::cli::LinkMode;
use tsp_iss::harness::links::{link_peer, LinkPlan, PlanLinks};
use tsp_iss::links::LinkPort;
use tsp_iss::Machine;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

const TX: [u16; 4] = [126, 127, 140, 141];
const RX: [u16; 4] = [128, 129, 142, 143];
const CHIPS: usize = 16;

fn units_of(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

struct Case {
    args: Vec<String>,
    text: String,
}

type Timed = Vec<(u16, Vec<(u64, Insn)>)>;

fn case(r: &mut Rng, links: usize) -> Case {
    let mut q: Vec<Timed> = vec![Vec::new(); CHIPS];
    let add = |q: &mut Vec<Timed>, c: usize, qn: u16, t: u64, i: Insn| match q[c].iter_mut().find(|e| e.0 == qn) {
        Some(e) => e.1.push((t, i)),
        None => q[c].push((qn, vec![(t, i)])),
    };
    let (lat, mut args): (u64, Vec<String>) = match r.below(4) {
        0 => (216, vec![]),
        1 => {
            let n = 209 + r.below(20);
            (n, vec!["--link-latency".into(), n.to_string()])
        }
        2 => (209 + r.below(20), vec!["--link-latency-seed".into(), (r.below(50) + 1).to_string()]),
        _ => (209 + r.below(20), vec!["--link-jitter-seed".into(), (r.below(50) + 1).to_string()]),
    };
    args.extend(["--chips".into(), CHIPS.to_string()]);
    if r.chance(15) {
        args.extend(["--max-cycles".into(), ["100", "230", "260", "500"][r.below(4) as usize].into()]);
    }
    for _ in 0..4 + r.below(24) {
        let c = r.below(CHIPS as u64) as usize;
        let l = r.below(links as u64) as usize;
        let (peer, pl) = match link_peer(CHIPS, links, c, l) {
            Some((p, pl)) if r.chance(92) => (p, pl),
            _ => (r.below(CHIPS as u64) as usize, r.below(links as u64) as usize),
        };
        let t = 1 + r.below(20);
        let tx = TX[r.below(4) as usize];
        let ins = match r.below(4) {
            0 => Insn::Send { chip_id: peer as u32, link: l as u32, stream: r.below(64) as u32 },
            _ => Insn::Transmit { chip_id: peer as u32, link: l as u32 },
        };
        add(&mut q, c, tx, t, ins);
        let near = (t as i64 + lat as i64 + r.below(7) as i64 - 3).max(1) as u64;
        let rx = RX[r.below(4) as usize];
        add(&mut q, peer, rx, near, Insn::Receive { chip_id: c as u32, link: pl as u32, stream: r.below(64) as u32 });
    }
    let mut text = String::new();
    for (c, queues) in q.iter_mut().enumerate() {
        writeln!(text, "C {c:x}").unwrap();
        for (qn, items) in queues.iter_mut() {
            items.sort_by_key(|e| e.0);
            let mut units: Vec<u32> = Vec::new();
            let mut now = 0u64;
            for (t, i) in items.iter() {
                if *t > now + 1 {
                    units.extend(units_of(&Insn::NOP { n: (*t - now - 1).min(1000) as u32 }));
                }
                now = (*t).max(now + 1);
                units.extend(units_of(i));
            }
            write!(text, "Q {qn:x}").unwrap();
            for u in units {
                write!(text, " {u:x}").unwrap();
            }
            text.push('\n');
        }
    }
    Case { args, text }
}

fn run(cfg: &str, c: &Case, jobs: &str) -> std::process::Output {
    let mut args: Vec<&str> = vec![cfg, "--trace", "--jobs", jobs];
    args.extend(c.args.iter().map(String::as_str));
    run_iss(&args, &c.text)
}

#[test]
fn one_thread_equals_many_on_16_chips() {
    let cfg = config("ref").to_str().unwrap().to_string();
    let links = 11;
    let mut r = Rng(16);
    let cases: Vec<Case> = (0..40).map(|_| case(&mut r, links)).collect();
    let delivered = std::sync::atomic::AtomicUsize::new(0);
    let bad = parallel(&cases, |c| {
        let one = run(&cfg, c, "1");
        if !matches!(one.status.code(), Some(0 | 3)) {
            return Err(format!("{:?}: exit {:?}\n{}", c.args, one.status.code(), text(&one.stderr)));
        }
        // a Receive that finds its vector writes it to a stream: a `T srf` line in the trace
        delivered.fetch_add(
            text(&one.stdout).lines().filter(|l| l.starts_with("T srf ")).count(),
            std::sync::atomic::Ordering::SeqCst,
        );
        // each thread count three times: the schedule of the threads differs from run to run
        for jobs in ["3", "3", "3", "16", "16", "16"] {
            let many = run(&cfg, c, jobs);
            if one.status.code() != many.status.code() || one.stdout != many.stdout || one.stderr != many.stderr {
                return Err(format!(
                    "{:?}: --jobs 1 and --jobs {jobs} differ (exit {:?} / {:?})\n{}\n{}",
                    c.args,
                    one.status.code(),
                    many.status.code(),
                    text(&one.stderr),
                    text(&many.stderr)
                ));
            }
        }
        // the runs are real: chips answered
        if one.status.code() == Some(0) && !text(&one.stdout).contains("C 15\n") {
            return Err(format!("{:?}: no output of chip 15", c.args));
        }
        Ok(())
    });
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    let n = delivered.into_inner();
    eprintln!("chip_threads: {n} vectors received in {} programs", cases.len());
    assert!(n >= cases.len(), "vectors must really be delivered between chips, or the comparison is vacuous: {n}");
}

/// An empty program on 64 chips (eight nodes) and the options of the thread count.
#[test]
fn jobs_option_and_empty_chips() {
    let cfg = config("ref").to_str().unwrap().to_string();
    let a = run_iss(&[&cfg, "--chips", "64", "--jobs", "1"], "");
    let b = run_iss(&[&cfg, "--chips", "64", "--jobs", "8"], "");
    assert_eq!(a.status.code(), Some(0), "{}", text(&a.stderr));
    assert_eq!(a.stdout, b.stdout);
    assert_eq!(text(&a.stdout).matches("\nE - -1 0\n").count(), 64);
    for bad in ["0", "x", "-1", "1025", ""] {
        let o = run_iss(&[&cfg, "--jobs", bad], "");
        assert_eq!(o.status.code(), Some(2), "--jobs {bad:?}");
        assert!(text(&o.stderr).starts_with("tsp-iss: --jobs must be a number from 1 to 1024"), "{}", text(&o.stderr));
    }
    let o = run_iss(&[&cfg, "--jobs"], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("usage"), "{}", text(&o.stderr));
}

/// `tsp-iss run` takes `--jobs` too, and its output does not depend on it.
#[test]
fn run_takes_jobs() {
    let mut r = Rng(5);
    let c = case(&mut r, 11);
    let d = tempfile::tempdir().unwrap();
    let prog = d.path().join("p.txt");
    // an object or a text program is chip 0's: the other chips are empty
    let chip0: String = c.text.lines().skip(1).take_while(|l| !l.starts_with("C ")).map(|l| format!("{l}\n")).collect();
    std::fs::write(&prog, chip0).unwrap();
    let go = |jobs: &str| {
        let mut args = vec!["run", "--profile", "ref", "--trace", "--jobs", jobs];
        args.extend(c.args.iter().filter(|a| !a.ends_with(".json")).map(String::as_str));
        args.push(prog.to_str().unwrap());
        run_iss(&args, "")
    };
    let (one, many) = (go("1"), go("8"));
    assert!(matches!(one.status.code(), Some(0 | 3)), "{}", text(&one.stderr));
    assert_eq!((one.status.code(), &one.stdout, &one.stderr), (many.status.code(), &many.stdout, &many.stderr));
}

/// A port whose `next_event` panics with the chip's number: a chip that fails inside a window.
struct PanicPort(usize);

impl LinkPort for PanicPort {
    fn peer(&self, _link: u8) -> Option<u16> {
        None
    }
    fn send(&mut self, _cycle: u64, _link: u8, _v: &[u8; MAX_LANES]) {}
    fn recv_ready(&mut self, _cycle: u64, _link: u8) -> bool {
        false
    }
    fn recv(&mut self, _cycle: u64, _link: u8) -> [u8; MAX_LANES] {
        [0; MAX_LANES]
    }
    fn next_event(&self) -> Option<u128> {
        panic!("port of chip {}", self.0)
    }
}

/// Of several chips that panic in one window the panic of the lowest index surfaces whatever the number of
/// threads (the one a single thread meets first), so the message is deterministic.
#[test]
fn the_panic_of_the_lowest_chip_surfaces_on_every_thread_count() {
    let profile = Profile::load("ref").unwrap();
    let geom = Geom::new(&profile);
    let plan = Arc::new(LinkPlan::new(LinkMode::Nominal, &geom, 8).unwrap());
    let message = |jobs: usize| {
        let mut chips: Vec<ChipRun> = (0..8)
            .map(|c| {
                let state = PlanLinks::new(plan.clone(), c, &geom).state();
                // chips 2, 5 and 6 panic; the others have a NOP queued and run normally
                let port: Box<dyn LinkPort> = if matches!(c, 2 | 5 | 6) {
                    Box::new(PanicPort(c))
                } else {
                    Box::new(PlanLinks::new(plan.clone(), c, &geom))
                };
                let mut m = Machine::new(&profile, c as u16, port).unwrap();
                m.load_image(0, &[encode(&Insn::NOP { n: 3 }) as u32]).unwrap();
                ChipRun::new(m, state)
            })
            .collect();
        let r =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_chips(&mut chips, 209, 1000, false, jobs)));
        let p = r.expect_err("a chip panicked: the run must panic");
        p.downcast_ref::<String>().cloned().unwrap_or_default()
    };
    for jobs in [1, 2, 4, 8] {
        for _ in 0..5 {
            assert_eq!(message(jobs), "port of chip 2", "jobs {jobs}");
        }
    }
}
