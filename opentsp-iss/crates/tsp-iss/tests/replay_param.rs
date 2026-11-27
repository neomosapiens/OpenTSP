//! Parametrised schedule replay (`driver::replay_param`): one recorded schedule serves other
//! plans of the same program, the plans differing only in the values of the sites the host patches.
//!
//! The synthetic tests run on the generated mixed programs of `tests/data/replay/` with a site on the address of every MEM Read
//! and Write of the initial images: a plan is a new address per site that keeps its bank. The real bucket (the 64-step decode
//! sequence of the P(8,4,128) program) is `real_bucket.rs`'s part of this file, run by `just replay-plan` (`#[ignore]`: it needs
//! the inputs that recipe makes).
//!
//! Run: `cargo test -p tsp-iss --features replay --test replay_param`.
#![cfg(feature = "replay")]
use ots_isa::Insn;
use ots_profile::Profile;
use std::path::PathBuf;
use std::sync::Arc;
use tsp_iss::driver::replay::{Op, Schedule};
use tsp_iss::driver::replay_param::{Gate, ParamSchedule, QueueText, Site};
use tsp_iss::frontend::{decode_text, text_width};
use tsp_iss::links::NoLinks;
use tsp_iss::machine::{Machine, RunError};
use tsp_iss::observe::{Hold, Observer, Row};

const MAX_CYCLES: u64 = 6000;

#[test]
fn a_text_word_written_then_restored_is_not_safe_for_parametric_replay() {
    use tsp_iss::driver::replay_param::Run;
    for name in names() {
        let p = load(&name);
        let (sites, values, mut text) = sites_of(&p);
        if sites.is_empty() || text.is_empty() {
            continue;
        }
        let Some((recorded, schedule)) = record(&p) else { continue };
        let upw = recorded.geom().vlanes as usize / 4;
        let q = text[0].queue;
        while text[0].units.len() < upw {
            text[0].units.push(ots_isa::PADDING);
        }
        let wanted: Vec<u8> = text[0].units[..upw].iter().flat_map(|u| u.to_le_bytes()).collect();
        let Ok(mut plan) = ParamSchedule::build(&recorded, schedule, sites, &text, &values) else { continue };
        plan.set_arena(vec![Run { queue: q, start: 0, slice: 0, addr: 64, words: 1 }]);
        let mut current = machine(&p);
        current.preload(0, 64, &wanted).unwrap();
        plan.text_gate(&current).unwrap();
        current.edit_state(|state| state.sram.journal_begin()).unwrap();
        current.preload(0, 65, &vec![0xa5; wanted.len()]).unwrap();
        plan.text_gate(&current).expect("neighboring data on the same page is not instruction text");
        let mut changed = wanted.clone();
        changed[0] ^= 1;
        current.preload(0, 64, &changed).unwrap();
        current.preload(0, 64, &wanted).unwrap();
        assert_eq!(&current.sram_word(0, 64).unwrap()[..wanted.len()], wanted.as_slice());
        assert!(
            matches!(plan.text_gate(&current), Err(Gate::Text(_))),
            "end-state equality hid an in-run write over fetched text"
        );
        for checked in [false, true] {
            let mut direct = machine(&p);
            direct.preload(0, 64, &wanted).unwrap();
            plan.arm(&mut direct, checked).unwrap();
            direct.preload(0, 64, &changed).unwrap();
            direct.preload(0, 64, &wanted).unwrap();
            let result = direct.run(MAX_CYCLES, &mut Log::default());
            assert!(
                matches!(result, Err(RunError::Internal(ref why)) if why.contains("replay") && why.contains("text")),
                "a direct completed run must fail closed when fetched text was written: {result:?}"
            );
        }
        for checked in [false, true] {
            for partial in [false, true] {
                let mut direct = machine(&p);
                direct.preload(0, 64, &wanted).unwrap();
                plan.arm(&mut direct, checked).unwrap();
                if partial {
                    direct.run_until(3, MAX_CYCLES, &mut Log::default()).unwrap();
                }
                direct.run(MAX_CYCLES, &mut Log::default()).unwrap();
                direct.run(MAX_CYCLES, &mut Log::default()).expect("completed watch must not become missing history");
            }
        }
        return;
    }
    panic!("no parametrised corpus program exercised the text-write journal");
}

#[derive(Default)]
struct Log {
    ev: Vec<String>,
}

impl Observer for Log {
    fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
        self.ev.push(format!("srf {c} {row:?} {stream} {gap} {v:02x?}"));
    }
    fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
        self.ev.push(format!("sram {c} {slice} {addr} {v:02x?}"));
    }
    fn diag(&mut self, c: u64, queue: u16, code: ots_isa::Code) {
        self.ev.push(format!("diag {c} {queue} {}", code.as_str()));
    }
    fn lint(&mut self, c: u64, queue: u16, code: &'static str) {
        self.ev.push(format!("lint {c} {queue} {code}"));
    }
    fn dispatch(&mut self, c: u64, queue: u16) {
        self.ev.push(format!("dispatch {c} {queue}"));
    }
    fn wait(&mut self, c: u64, queue: u16, why: Hold, since: u64) {
        self.ev.push(format!("wait {c} {queue} {why:?} {since}"));
    }
    fn fetch(&mut self, c: u64, queue: u16, bytes: u32) {
        self.ev.push(format!("fetch {c} {queue} {bytes}"));
    }
}

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/replay")
}

fn names() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(data_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".txt"))
        .collect();
    v.sort();
    v
}

fn profile_of(name: &str) -> &'static str {
    if name.starts_with("ref") {
        "ref"
    } else {
        "mini"
    }
}

/// A program of the corpus: the images by queue (every `Q` line of a queue appended) and the SRAM words.
#[derive(Clone)]
struct Prog {
    profile: &'static str,
    images: Vec<(u16, Vec<u32>)>,
    words: Vec<(u8, u32, Vec<u8>)>,
}

fn load(name: &str) -> Prog {
    let text = std::fs::read_to_string(data_dir().join(name)).unwrap();
    let mut p = Prog { profile: profile_of(name), images: Vec::new(), words: Vec::new() };
    for l in text.lines() {
        let f: Vec<&str> = l.split(' ').collect();
        let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
        match f[0] {
            "S" => p.words.push((hex(f[1]) as u8, hex(f[2]) as u32, f[3..].iter().map(|b| hex(b) as u8).collect())),
            "Q" => {
                let q = hex(f[1]) as u16;
                let units: Vec<u32> = f[2..].iter().map(|u| hex(u) as u32).collect();
                match p.images.iter_mut().find(|(x, _)| *x == q) {
                    Some((_, u)) => u.extend(units),
                    None => p.images.push((q, units)),
                }
            }
            _ => {}
        }
    }
    p
}

fn machine(p: &Prog) -> Machine {
    let mut m = Machine::new(&Profile::load(p.profile).unwrap(), 0, Box::new(NoLinks)).unwrap();
    for (q, units) in &p.images {
        m.load_image(*q, units).unwrap();
    }
    for (s, a, w) in &p.words {
        m.preload(*s, *a, w).unwrap();
    }
    m
}

/// What a run leaves that `--stats` and the platform read: the IQ levels of every queue and quiescence.
fn levels(m: &Machine) -> String {
    let q = (0..144u16).chain([0xffff]).map(|q| format!("{:?}", m.queue_state(q).unwrap())).collect::<Vec<_>>();
    format!("{};quiescent={}", q.join(";"), m.quiescent())
}

type Outcome = (Result<Vec<String>, RunError>, String);

fn run(mut m: Machine) -> Outcome {
    let mut log = Log::default();
    let r = m.run(MAX_CYCLES, &mut log);
    (r.map(|_| log.ev), levels(&m))
}

fn record(p: &Prog) -> Option<(Machine, Schedule)> {
    let mut m = machine(p);
    m.record_schedule().unwrap();
    m.run(MAX_CYCLES, &mut Log::default()).ok()?;
    let s = m.take_schedule()?;
    Some((m, s))
}

/// The sites of a program: the address of every Read and Write of the initial images, in queue order, and the values the
/// program has there.
fn sites_of(p: &Prog) -> (Vec<Site>, Vec<i32>, Vec<QueueText>) {
    let m = machine(p);
    let words = m.geom().words_per_slice as i64;
    let (mut sites, mut values, mut text) = (Vec::new(), Vec::new(), Vec::new());
    for (q, units) in &p.images {
        text.push(QueueText { queue: *q, units: units.clone() });
        let mut pos = 0;
        while pos < units.len() {
            let w = text_width(units[pos]);
            if pos + w > units.len() {
                break;
            }
            if let Ok(i @ (Insn::Read { addr, .. } | Insn::Write { addr, .. })) =
                decode_text(m.isa(), units[pos], units.get(pos + 1).copied().unwrap_or(0))
            {
                let v = i.entry().vars.iter().find(|v| v.name == "addr").expect("an address operand");
                assert_eq!((v.hi_n, w), (0, if i.entry().width == 64 { 2 } else { 1 }));
                sites.push(Site {
                    queue: *q,
                    unit: pos,
                    nunits: w,
                    lo: u32::from(v.lo),
                    width: u32::from(v.n),
                    vmin: 0,
                    vmax: (words - 1).min((1i64 << v.n) - 1),
                    stride: 1,
                    span: 1,
                });
                values.push(addr as i32);
            }
            pos += w;
        }
    }
    (sites, values, text)
}

/// Write `values` into the images of `p` at the sites.
fn patched(p: &Prog, sites: &[Site], values: &[i32]) -> Prog {
    let mut out = p.clone();
    for (s, v) in sites.iter().zip(values) {
        let units = &mut out.images.iter_mut().find(|(q, _)| *q == s.queue).unwrap().1;
        let mask = ((1u64 << s.width) - 1) << s.lo;
        let w = if s.nunits == 2 {
            ((units[s.unit] as u64) << 32) | units[s.unit + 1] as u64
        } else {
            units[s.unit] as u64
        };
        let n = (w & !mask) | ((*v as u64) << s.lo);
        if s.nunits == 2 {
            units[s.unit] = (n >> 32) as u32;
            units[s.unit + 1] = n as u32;
        } else {
            units[s.unit] = n as u32;
        }
    }
    out
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Another plan: every site takes another address of the same bank (`flip`: another bank for site 0 instead).
fn other_plan(m: &Machine, values: &[i32], rng: &mut Rng) -> Vec<i32> {
    let words = m.geom().words_per_slice as u32;
    values
        .iter()
        .map(|&v| {
            let bank = m.sram_bank(v as u32);
            loop {
                let a = (rng.next() % u64::from(words)) as u32;
                if m.sram_bank(a) == bank {
                    return a as i32;
                }
            }
        })
        .collect()
}

struct Tally {
    programs: usize,
    built: usize,
    plans: usize,
    served: usize,
    refused: usize,
    differ_without_instantiate: usize,
    elements_patched: usize,
}

fn corpus_run() -> Tally {
    let mut t = Tally {
        programs: 0,
        built: 0,
        plans: 0,
        served: 0,
        refused: 0,
        differ_without_instantiate: 0,
        elements_patched: 0,
    };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for n in names() {
        t.programs += 1;
        let p = load(&n);
        let (sites, values, text) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let Some((m0, s)) = record(&p) else { continue };
        let mut ps = match ParamSchedule::build(&m0, s, sites.clone(), &text, &values) {
            Ok(ps) => ps,
            Err(e) => {
                eprintln!("{n}: not parametrised: {e}");
                continue;
            }
        };
        t.built += 1;
        t.elements_patched += ps.shape().patched_elements;
        eprintln!(
            "{n}: patched elements {}, alone in their cycle {}",
            ps.shape().patched_elements,
            ps.shape().solo_elements
        );
        // the stale schedule (no instantiation) on another plan: the mutation of "the plan reaches the elements"
        let first = other_plan(&m0, &values, &mut rng);
        {
            let oracle = run(machine(&patched(&p, &sites, &first)));
            let mut m = machine(&patched(&p, &sites, &first));
            ps.arm(&mut m, false).expect("the images differ in the sites only");
            let mut log = Log::default();
            let r = m.run(MAX_CYCLES, &mut log);
            let stale = (r.map(|_| log.ev), levels(&m));
            if stale != oracle {
                t.differ_without_instantiate += 1;
            }
        }
        let mut ps = ParamSchedule::build(&m0, record(&p).unwrap().1, sites.clone(), &text, &values).unwrap();
        for k in 0..4 {
            let plan = if k == 3 { values.clone() } else { other_plan(&m0, &values, &mut rng) };
            let prog = patched(&p, &sites, &plan);
            let oracle = run(machine(&prog));
            let mut m = machine(&prog);
            t.plans += 1;
            match ps.instantiate(&m, &plan) {
                Err(g) => {
                    t.refused += 1;
                    eprintln!("{n} plan {k}: refused: {g}");
                }
                Ok(_) => {
                    ps.arm(&mut m, false).unwrap();
                    let mut log = Log::default();
                    let r = m.run(MAX_CYCLES, &mut log);
                    let got: Outcome = (r.map(|_| log.ev), levels(&m));
                    assert_eq!(
                        oracle.0.as_ref().unwrap(),
                        got.0.as_ref().unwrap_or_else(|_| panic!("{n} plan {k}: the replay failed")),
                        "{n} plan {k}: events differ"
                    );
                    assert_eq!(oracle.1, got.1, "{n} plan {k}: the state after the run differs");
                    // the per-tick check against the normal driver
                    let mut mc = machine(&prog);
                    ps.arm(&mut mc, true).unwrap();
                    mc.run(MAX_CYCLES, &mut Log::default())
                        .unwrap_or_else(|e| panic!("{n} plan {k}: the check failed: {e}"));
                    t.served += 1;
                }
            }
        }
    }
    t
}

#[test]
fn a_parametrised_schedule_is_the_normal_run_on_other_plans() {
    let t = corpus_run();
    eprintln!(
        "programs {}, parametrised {}, plans {}, served {}, refused {}, stale schedules that differ {}, elements patched {}",
        t.programs, t.built, t.plans, t.served, t.refused, t.differ_without_instantiate, t.elements_patched
    );
    assert!(t.built >= 8, "only {} programs were parametrised", t.built);
    assert!(t.served >= 20, "only {} plans were served", t.served);
    assert!(t.differ_without_instantiate >= 4, "the plans do not change what the programs do: the test cannot fail");
}

#[test]
fn a_plan_off_the_table_a_bank_change_and_other_text_are_refused() {
    let (mut checked, mut bank_refused, mut bank_served, mut grid_checked) = (0, 0, 0, 0);
    for n in names() {
        let p = load(&n);
        let (sites, values, text) = sites_of(&p);
        let Some((m0, s)) = record(&p) else { continue };
        let Ok(mut ps) = ParamSchedule::build(&m0, s, sites.clone(), &text, &values) else { continue };
        if ps.shape().patched_elements == 0 {
            continue;
        }
        let m = machine(&p);
        // a count that is not the table's
        assert!(matches!(ps.instantiate(&m, &values[1..]), Err(Gate::Values(_))), "{n}");
        // a value outside the range of the site
        let mut bad = values.clone();
        bad[0] = sites[0].vmax as i32 + 1;
        assert!(matches!(ps.instantiate(&m, &bad), Err(Gate::Values(_))), "{n}");
        // a grid: the table with stride 2 from 0, the program recorded with even addresses: an odd value is refused
        let even: Vec<i32> = values.iter().map(|v| v & !1).collect();
        let grid: Vec<Site> = sites.iter().map(|s| Site { stride: 2, span: 1, ..s.clone() }).collect();
        let p_even = patched(&p, &sites, &even);
        if let Some((m_even, s_even)) = record(&p_even) {
            let mut gps = ParamSchedule::build(&m_even, s_even, grid, &text, &even).expect("the grid table builds");
            let odd: Vec<i32> = even.iter().map(|v| v + 1).collect();
            assert!(matches!(gps.instantiate(&m, &odd), Err(Gate::Values(_))), "{n}: a value off the grid");
            gps.instantiate(&m, &even).expect("on the grid");
            grid_checked += 1;
        }
        // a memory reference that changes bank: refused when it meets another reference of its queue and cycle in a bank (then the
        // normal driver reports MEM-001), served otherwise (then the replay is the normal run)
        let bank_of = |v: i32| m.sram_bank(v as u32);
        let words = m.geom().words_per_slice as i32;
        for i in 0..sites.len() {
            let Some(other) = (0..words).find(|a| bank_of(*a) != bank_of(values[i]) && (*a as i64) <= sites[i].vmax)
            else {
                continue;
            };
            let mut plan = values.clone();
            plan[i] = other;
            match ps.instantiate(&m, &plan) {
                Err(Gate::Bank(_)) => bank_refused += 1,
                Err(Gate::Decode(_)) | Err(Gate::Values(_)) => {}
                Ok(_) => {
                    if bank_served < 40 {
                        let prog = patched(&p, &sites, &plan);
                        let oracle = run(machine(&prog));
                        let mut mm = machine(&prog);
                        ps.arm(&mut mm, false).unwrap();
                        let mut log = Log::default();
                        let r = mm.run(MAX_CYCLES, &mut log);
                        assert_eq!(
                            oracle.0.as_ref().unwrap(),
                            r.map(|_| log.ev).as_ref().unwrap(),
                            "{n}: a bank change at site {i} was served wrongly"
                        );
                    }
                    bank_served += 1;
                    ps.instantiate(&m, &values).unwrap(); // back to the recorded plan
                }
                Err(g) => panic!("{n}: {g}"),
            }
        }
        // the images: a bit that no site covers
        let (q0, u0) = (p.images[0].0, p.images[0].1[0]);
        let bit = 1u32 << 3;
        let covered = sites.iter().any(|s| {
            let mask = ((1u64 << s.width) - 1) << s.lo;
            s.queue == q0
                && ((s.nunits == 2 && s.unit == 0 && (mask >> 32) as u32 & bit != 0)
                    || (s.unit == 0 && s.nunits == 1 && mask as u32 & bit != 0)
                    || (s.nunits == 2 && s.unit == 1 && 0 != 0))
        });
        let mut other_text = p.clone();
        other_text.images[0].1[0] = u0 ^ bit;
        let mut mt = machine(&other_text);
        let mut fresh = ParamSchedule::build(&m0, record(&p).unwrap().1, sites.clone(), &text, &values).unwrap();
        let g = fresh.arm(&mut mt, false);
        assert!(covered || matches!(g, Err(Gate::Images(_))), "{n}: {g:?}");
        checked += 1;
    }
    eprintln!("bank changes of a lone reference: served {bank_served}, refused {bank_refused}");
    assert!(checked >= 6 && grid_checked >= 4, "only {checked} programs were checked ({grid_checked} for the grid)");
    assert!(
        bank_served >= 100 && bank_refused == 0,
        "a reference that is alone in its cycle cannot meet another in a bank"
    );
}

/// Two references of one queue, a Read and a Write of the same liquid thread, one after the other in a hand-made image of a queue
/// that issues both (`bundle`: in one bundle, so that they issue in one cycle and their banks matter).
fn two_references(bundle: bool) -> Option<Prog> {
    for n in names() {
        let src = load(&n);
        let m = machine(&src);
        for (q, units) in &src.images {
            let (mut read, mut write) = (None, None);
            let mut pos = 0;
            while pos < units.len() {
                let w = text_width(units[pos]);
                if pos + w > units.len() {
                    break;
                }
                let u = [units[pos], units.get(pos + 1).copied().unwrap_or(0)];
                match decode_text(m.isa(), u[0], u[1]) {
                    Ok(Insn::Read { thread: 0, .. }) if read.is_none() && w == 2 => read = Some(u),
                    Ok(i @ Insn::Write { thread: 0, .. })
                        if write.is_none() && w == 2 && m.issued_by(&i, usize::from(*q)) =>
                    {
                        write = Some(u)
                    }
                    _ => {}
                }
                pos += w;
            }
            if let (Some(r), Some(w)) = (read, write) {
                let img =
                    vec![if bundle { r[0] | 0x8000_0000 } else { r[0] & 0x7fff_ffff }, r[1], w[0] & 0x7fff_ffff, w[1]];
                return Some(Prog { profile: src.profile, images: vec![(*q, img)], words: src.words.clone() });
            }
        }
    }
    None
}

#[test]
fn two_references_that_meet_in_a_bank_are_refused_and_the_normal_driver_agrees() {
    let p = two_references(true).expect("a Read and a Write of one queue in the corpus");
    let m0 = machine(&p);
    let bit = m0.geom().bank_bit as u32;
    let (sites, _, text) = sites_of(&p);
    assert_eq!(sites.len(), 2, "the Read and the Write");
    // the recorded plan: the Read in bank 0, the Write in bank 1: no clash
    let values = vec![8, (1 << bit) + 8];
    let base = patched(&p, &sites, &values);
    let (m, s) = record(&base).expect("the recording");
    assert!(!run(machine(&base)).0.unwrap().iter().any(|e| e.contains("MEM-001")), "the recorded plan must not clash");
    let mut ps = ParamSchedule::build(&m, s, sites.clone(), &text, &values).unwrap();
    assert_eq!(ps.shape().solo_elements, 0, "the two references issue in one cycle");
    // the Read moves to the bank of the Write: they meet: refused, and the normal driver reports MEM-001
    let clash = vec![(1 << bit) + 24, values[1]];
    let oracle = run(machine(&patched(&p, &sites, &clash)));
    assert!(oracle.0.as_ref().unwrap().iter().any(|e| e.contains("MEM-001")), "the plan clashes in the normal driver");
    assert!(matches!(ps.instantiate(&m, &clash), Err(Gate::Bank(_))));
    assert_eq!(ps.values(), &values[..]);
    // both move within their banks: served, and the replay is the normal run
    let moved = vec![40, (1 << bit) + 56];
    let prog = patched(&p, &sites, &moved);
    let oracle = run(machine(&prog));
    let mut mm = machine(&prog);
    ps.instantiate(&mm, &moved).unwrap();
    ps.arm(&mut mm, false).unwrap();
    let mut log = Log::default();
    let r = mm.run(MAX_CYCLES, &mut log);
    assert_eq!(oracle.0.as_ref().unwrap(), r.map(|_| log.ev).as_ref().unwrap());
    // both in bank 0: refused too
    let both = vec![8, 16];
    assert!(matches!(ps.instantiate(&machine(&p), &both), Err(Gate::Bank(_))));
    // the same two references one after the other are alone in their cycles: any banks are served
    let p2 = two_references(false).unwrap();
    let (sites2, _, text2) = sites_of(&p2);
    let base2 = patched(&p2, &sites2, &values);
    let (m2, s2) = record(&base2).expect("the recording");
    let mut ps2 = ParamSchedule::build(&m2, s2, sites2, &text2, &values).unwrap();
    assert_eq!(ps2.shape().patched_elements, ps2.shape().solo_elements);
    ps2.instantiate(&machine(&p2), &both).expect("alone in their cycles: served");
}

#[test]
fn a_recording_with_a_bank_clash_is_not_parametrised() {
    // a schedule with a MEM-001 in its operations: equal banks would not give equal checks
    for n in names() {
        let p = load(&n);
        let (sites, values, text) = sites_of(&p);
        let Some((m0, mut s)) = record(&p) else { continue };
        let Some(i) = (0..s.ticks()).find(|&i| s.ops_off[i] != s.ops_off[i + 1]) else { continue };
        s.ops[s.ops_off[i] as usize] = Op::Diag(0, ots_isa::Code::Mem001);
        let e = ParamSchedule::build(&m0, s, sites, &text, &values).err().expect("refused");
        assert!(e.0.contains("MEM-001"), "{e}");
        return;
    }
    panic!("no program");
}

#[test]
fn the_arc_of_the_schedule_is_shared_with_no_copy_between_steps() {
    // after a run the machine drops its hold: the next instantiation writes in place
    let p = load("mini-0.txt");
    let (sites, values, text) = sites_of(&p);
    if let Some((m0, s)) = record(&p) {
        if let Ok(mut ps) = ParamSchedule::build(&m0, s, sites, &text, &values) {
            let before = Arc::as_ptr(ps.schedule());
            let m = machine(&p);
            let _ = ps.instantiate(&m, &values);
            assert_eq!(before, Arc::as_ptr(ps.schedule()));
        }
    }
}

/// The `assembly.md` add example with an `ifetchm` of word `fetch` at cycle 47 and a `write` of address `W` at cycle 48 on the same
/// MEM queue: the fetch reads word `fetch + 1` in cycle 48, so a Write there meets its bank. The words the fetch reads hold padding.
/// With `repeat` a `repeat 2, 1` iterates the fetch twice and the write follows the second iteration (the Write is unit 4 of the queue,
/// not 2).
fn fetch_then_write(fetch: u32, repeat: bool) -> Prog {
    let m0 = Machine::new(&Profile::load("ref").unwrap(), 0, Box::new(NoLinks)).unwrap();
    let lanes = m0.geom().vlanes as usize;
    // the Write of the Repeat variant is three cycles later, so is the data it stores
    let late = if repeat { 3 } else { 0 };
    let pad = vec![0xffu8; lanes];
    let mut words = vec![(4u8, 0x10u32, vec![3u8; lanes]), (5, 0, vec![4u8; lanes])];
    for a in [0xfffu32, 0x1000, 0x7ff, 0x800] {
        words.push((0x2c, a, pad.clone()));
    }
    Prog {
        profile: "ref",
        images: vec![
            (0x04, vec![0x0000_0021 + late, 0x1000_0010, 0]),
            (0x05, vec![0x0000_0021 + late, 0x1000_0000, 0x1000_0000]),
            (
                0x2c,
                if repeat {
                    vec![0x0000_002f, 0x01c0_0000 | fetch, 0x00c0_0021, 1, 0x1040_0010, 0x2000_0000]
                } else {
                    vec![0x0000_002f, 0x01c0_0000 | fetch, 0x1040_0010, 0x2000_0000]
                },
            ),
            (0x58, vec![0x0000_0028 + late, 0x2002_0010, 0]),
        ],
        words,
    }
}

/// The site of the field `addr` of the instruction of the queue `q` at logical unit `unit`.
fn addr_site(p: &Prog, q: u16, unit: usize) -> Site {
    let m = machine(p);
    let units = &p.images.iter().find(|(x, _)| *x == q).unwrap().1;
    let i = decode_text(m.isa(), units[unit], units.get(unit + 1).copied().unwrap_or(0)).unwrap();
    let v = i.entry().vars.iter().find(|v| v.name == "addr").expect("an address operand");
    Site {
        queue: q,
        unit,
        nunits: text_width(units[unit]),
        lo: u32::from(v.lo),
        width: u32::from(v.n),
        vmin: 0,
        vmax: m.geom().words_per_slice as i64 - 1,
        stride: 1,
        span: 1,
    }
}

fn texts(p: &Prog) -> Vec<QueueText> {
    p.images.iter().map(|(q, u)| QueueText { queue: *q, units: u.clone() }).collect()
}

fn mem001(o: &Outcome) -> bool {
    o.0.as_ref().unwrap().iter().any(|e| e.contains("MEM-001"))
}

/// The second read of an `ifetchm` takes the bank of the next word in the next cycle: a patched Write there is not alone, and a
/// plan that moves it into that bank is refused (the model reports OTS-E-MEM-001, the replay would write the word and drop it).
#[test]
fn a_parametrised_schedule_counts_its_own_tables_in_its_bytes() {
    let p = fetch_then_write(0xfff, false);
    let site = addr_site(&p, 0x2c, 2);
    let recorded = patched(&p, std::slice::from_ref(&site), &[0x10]);
    let (m, s) = record(&recorded).expect("the recording");
    let ps = ParamSchedule::build(&m, s, vec![site], &texts(&p), &[0x10]).expect("build");
    let (sched, total) = (ps.schedule().bytes(), ps.bytes());
    // what the schedule's own tables hold, apart from the struct and the instantiated schedule: at least the text of the queues (4 bytes
    // a unit) and the one site
    let own = total - sched - std::mem::size_of::<ParamSchedule>();
    let text_bytes: usize = texts(&p).iter().map(|q| q.units.len() * 4).sum();
    assert!(
        own >= text_bytes + std::mem::size_of::<Site>(),
        "the tables (sites, items, elements, text) are in it: {own} bytes of its own against {text_bytes} of text and a site \
         (total {total}, the schedule's {sched})"
    );
}

#[test]
fn a_write_the_cycle_after_an_ifetchm_is_checked_against_the_fetch_bank() {
    let base = fetch_then_write(0xfff, false);
    let site = addr_site(&base, 0x2c, 2);
    let bit = machine(&base).geom().bank_bit as u32;
    let recorded = patched(&base, std::slice::from_ref(&site), &[0x10]);
    assert!(!mem001(&run(machine(&recorded))), "the recorded plan is clean");
    let (m, s) = record(&recorded).expect("the recording");
    let mut ps = ParamSchedule::build(&m, s, vec![site.clone()], &texts(&base), &[0x10]).expect("build");
    assert_eq!(ps.shape().patched_elements, 1);
    assert_eq!(ps.shape().solo_elements, 0, "the Write meets the second read of the fetch in the next cycle");
    // the bank of word 0x1000 (the second word the fetch reads): the model rejects the write
    let into = (1i32 << bit) + 0x10;
    let oracle = run(machine(&patched(&base, std::slice::from_ref(&site), &[into])));
    assert!(mem001(&oracle), "the model rejects a write into the bank of the fetch's second word");
    let got = ps.instantiate(&machine(&base), &[into]);
    assert!(matches!(got, Err(Gate::Bank(_))), "served: {got:?}");
    assert_eq!(ps.values(), &[0x10]);
    // a new address in the recorded bank is served, and the replay is the normal run
    let prog = patched(&base, std::slice::from_ref(&site), &[0x20]);
    let oracle = run(machine(&prog));
    assert!(!mem001(&oracle));
    let mut mm = machine(&prog);
    ps.instantiate(&mm, &[0x20]).expect("the same bank is served");
    ps.arm(&mut mm, false).unwrap();
    let mut log = Log::default();
    let r = mm.run(MAX_CYCLES, &mut log);
    assert_eq!(oracle.0.as_ref().unwrap(), r.map(|_| log.ev).as_ref().unwrap());
}

/// The same for the iteration of a Repeat of the `ifetchm`: its second read is in the cycle after the iteration.
#[test]
fn a_write_the_cycle_after_a_repeated_ifetchm_is_checked_against_the_fetch_bank() {
    let base = fetch_then_write(0xfff, true);
    let site = addr_site(&base, 0x2c, 4);
    let sites = std::slice::from_ref(&site);
    let bit = machine(&base).geom().bank_bit as u32;
    let recorded = patched(&base, sites, &[0x10]);
    assert!(!mem001(&run(machine(&recorded))), "the recorded plan is clean");
    let (m, s) = record(&recorded).expect("the recording");
    let mut ps = ParamSchedule::build(&m, s, sites.to_vec(), &texts(&base), &[0x10]).expect("build");
    let into = (1i32 << bit) + 0x10;
    assert!(mem001(&run(machine(&patched(&base, sites, &[into])))), "the model rejects the write");
    let got = ps.instantiate(&machine(&base), &[into]);
    assert!(matches!(got, Err(Gate::Bank(_))), "served: {got:?}");
    ps.instantiate(&machine(&base), &[0x20]).expect("the recorded bank is served");
}

/// The site of the field `name` of the instruction of queue `q` at logical unit `unit`, on the range of its width.
fn field_site(p: &Prog, q: u16, unit: usize, name: &str) -> Site {
    let m = machine(p);
    let units = &p.images.iter().find(|(x, _)| *x == q).unwrap().1;
    let i = decode_text(m.isa(), units[unit], units.get(unit + 1).copied().unwrap_or(0)).unwrap();
    let v = i.entry().vars.iter().find(|v| v.name == name).unwrap_or_else(|| {
        panic!("{i:?} has no field {name}: {:?}", i.entry().vars.iter().map(|v| v.name).collect::<Vec<_>>())
    });
    Site {
        queue: q,
        unit,
        nunits: text_width(units[unit]),
        lo: u32::from(v.lo),
        width: u32::from(v.n),
        vmin: 0,
        vmax: (1i64 << v.n) - 1,
        stride: 1,
        span: 1,
    }
}

/// The program of the `assembly.md` add example that the probes below share: two MEM readers feed a VXM add whose result a
/// MEM queue writes. `q2c` is the text of MEM queue 0x2c, `q58` that of the VXM queue and `extra` more words of slice 0x2c
/// (the first four bytes given, the rest 0xff).
fn add_program(q2c: Vec<u32>, q58: Vec<u32>, extra: &[(u32, [u8; 4])]) -> Prog {
    let m0 = Machine::new(&Profile::load("ref").unwrap(), 0, Box::new(NoLinks)).unwrap();
    let lanes = m0.geom().vlanes as usize;
    let mut words = vec![(4u8, 0x10u32, vec![3u8; lanes]), (5, 0, vec![4u8; lanes])];
    for (a, head) in extra {
        let mut w = vec![0xffu8; lanes];
        w[..4].copy_from_slice(head);
        words.push((0x2c, *a, w));
    }
    Prog {
        profile: "ref",
        images: vec![
            (0x04, vec![0x0000_0021, 0x1000_0010, 0]),
            (0x05, vec![0x0000_0021, 0x1000_0000, 0x1000_0000]),
            (0x2c, q2c),
            (0x58, q58),
        ],
        words,
    }
}

/// A site that changes what a run does is refused when a plan changes it, and the run takes the normal driver: the schedule is a
/// function of the memory addresses only. `was` is the recorded value, `now` the one the plan serves.
fn site_refused(base: &Prog, site: &Site, was: i64, now: i64) {
    let sites = std::slice::from_ref(site);
    let recorded = patched(base, sites, &[was as i32]);
    let (m, s) = record(&recorded).expect("the recording");
    let mut ps = ParamSchedule::build(&m, s, sites.to_vec(), &texts(base), &[was as i32]).expect("build");
    // the plan changes the run: the normal driver's result for it is not the recorded one
    let (normal_was, normal_now) = (run(machine(&recorded)), run(machine(&patched(base, sites, &[now as i32]))));
    assert!(normal_was.0.is_ok() && normal_now.0.is_ok(), "both plans run");
    assert_ne!(normal_was.0, normal_now.0, "the plan changes what the model fetches, dispatches or reports");
    let got = ps.instantiate(&machine(base), &[now as i32]);
    assert!(matches!(got, Err(Gate::Site(_))), "served: {got:?}");
    assert_eq!(ps.values(), &[was as i32], "a refused plan changes nothing");
    // the recorded plan is still served
    ps.instantiate(&machine(base), &[was as i32]).expect("the recorded values");
}

/// A site on the address of an `ifetchm` that keeps both banks of its reads: the replay would drop the fetch the model makes.
#[test]
fn a_site_on_an_ifetchm_address_in_the_same_banks_is_refused() {
    // word 0xfc holds a unit that does not decode (tag 6), 0xfe padding: the model fetches the one and reports DEC-003
    let base = add_program(
        vec![0x0000_002f, 0x01c0_00fe, 0xffff_ffff],
        vec![0x0000_0028, 0x2002_0010, 0],
        &[(0xfc, [0, 0, 0, 0x60]), (0xfd, [0xff; 4]), (0xfe, [0xff; 4]), (0xff, [0xff; 4])],
    );
    let site = addr_site(&base, 0x2c, 1);
    site_refused(&base, &site, 0xfe, 0xfc);
}

/// The same across the end of a memory segment: the model refuses the fetch at 0xff, the plan serves the one at 0xfe.
#[test]
fn a_site_on_an_ifetchm_address_across_a_segment_end_is_refused() {
    let base = add_program(
        vec![0x0000_002e, 0x1200_0000, 0x0100_0000, 0x01c0_00ff, 0xffff_ffff],
        vec![0x0000_0028, 0x2002_0010, 0],
        &[(0xfe, [0xff; 4]), (0xff, [0xff; 4])],
    );
    let site = addr_site(&base, 0x2c, 3);
    site_refused(&base, &site, 0xff, 0xfe);
}

/// A site on the count of a `repeat`: the model issues the repeated instruction that many times and reports each.
#[test]
fn a_site_on_a_repeat_count_is_refused() {
    let base = add_program(
        vec![0x0000_002f, 0x01c0_0fff, 0x1040_0010, 0x2000_0000],
        vec![0x0000_0028, 0x2002_0010, 0, 0x00c0_0010, 0xffff_ffff],
        &[(0xfff, [0xff; 4]), (0x1000, [0xff; 4])],
    );
    let site = field_site(&base, 0x58, 3, "n");
    site_refused(&base, &site, 1, 2);
}

/// A site on another field of a Read or a Write than its address is refused too: only the address is checked after the patch.
#[test]
fn a_site_on_a_field_of_a_read_or_a_write_other_than_its_address_is_refused() {
    let base = fetch_then_write(0xfff, false);
    let m = machine(&base);
    let mut tried = 0;
    for (q, unit) in [(0x04u16, 1usize), (0x2c, 2)] {
        let units = &base.images.iter().find(|(x, _)| *x == q).unwrap().1;
        let i = decode_text(m.isa(), units[unit], units.get(unit + 1).copied().unwrap_or(0)).unwrap();
        for v in i.entry().vars.iter().filter(|v| v.name != "addr" && v.hi_n == 0 && v.n > 0) {
            let site = field_site(&base, q, unit, v.name);
            let sites = std::slice::from_ref(&site);
            let w = if site.nunits == 2 {
                (u64::from(units[unit]) << 32) | u64::from(units[unit + 1])
            } else {
                units[unit].into()
            };
            let was = ((w >> site.lo) & ((1u64 << site.width) - 1)) as i64;
            let now = if was == 0 { 1 } else { was - 1 };
            let (rm, rs) = record(&base).expect("the recording");
            let mut ps = ParamSchedule::build(&rm, rs, sites.to_vec(), &texts(&base), &[was as i32]).expect("build");
            let got = ps.instantiate(&machine(&base), &[now as i32]);
            assert!(matches!(got, Err(Gate::Site(_))), "{i:?} field {}: served {got:?}", v.name);
            tried += 1;
        }
    }
    assert!(tried > 0, "the Read and the Write have a field besides their address");
}

/// A site on the address of an `ifetchm` that moves the bank of its second read is refused as a site, whatever its banks: the
/// refusal does not depend on the bank rule, and a fetch that stays where it was is served.
#[test]
fn a_site_on_an_ifetchm_address_that_moves_a_bank_is_refused() {
    let base = fetch_then_write(0xfff, false);
    let write = addr_site(&base, 0x2c, 2);
    let fetch = addr_site(&base, 0x2c, 1);
    let sites = vec![fetch.clone(), write.clone()];
    let recorded = patched(&base, &sites, &[0xfff, 0x10]);
    let (m, s) = record(&recorded).expect("the recording");
    let mut ps = ParamSchedule::build(&m, s, sites.clone(), &texts(&base), &[0xfff, 0x10]).expect("build");
    // word 0x7ff and 0x800 are both in bank 0, the bank of the Write: the second read now meets it
    let oracle = run(machine(&patched(&base, &sites, &[0x7ff, 0x10])));
    assert!(mem001(&oracle), "the model rejects it");
    let got = ps.instantiate(&machine(&base), &[0x7ff, 0x10]);
    assert!(matches!(got, Err(Gate::Site(_))), "served: {got:?}");
    // the fetch unchanged and the Write in its own bank: served
    ps.instantiate(&machine(&base), &[0xfff, 0x20]).expect("a Write in its own bank and the fetch unchanged");
}

/// The real bucket: the 64-step decode sequence of the P(8,4,128) program (`just replay-plan` makes the inputs and runs these
/// tests, which are `#[ignore]`d without them). `ISS_BENCH_DIR` holds `P8_4_128.elf` and `.safetensors` (the all-padding plan), and
/// `REPLAY_PLAN_DIR` the site table, the text arena, the logical text and the values of the steps.
mod real {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::collections::HashMap;
    use std::time::Instant;
    use tsp_iss::driver::replay::{RecordOn, ReplayCache};
    use tsp_iss::driver::replay_param::Run;
    use tsp_iss::observe::NoObserver;

    const LIMIT: u64 = 1 << 30;

    #[derive(Clone, Copy)]
    enum Loc {
        Image(usize, usize),
        Word(usize, usize),
    }

    pub struct Real {
        profile: Profile,
        images: Vec<(u16, Vec<u32>)>,
        words: Vec<(u8, u32, Vec<u8>)>,
        sites: Vec<Site>,
        locs: Vec<Vec<Loc>>,
        site_words: Vec<usize>,
        text: Vec<QueueText>,
        arena: Vec<Run>,
        pub steps: Vec<Vec<i32>>,
        pub padded: Vec<i32>,
    }

    fn dir(var: &str, default: &str) -> PathBuf {
        PathBuf::from(std::env::var(var).unwrap_or_else(|_| default.into()))
    }

    impl Real {
        pub fn open() -> Real {
            let (bench, plans) = (dir("ISS_BENCH_DIR", "/tmp/iss-bench"), dir("REPLAY_PLAN_DIR", "/tmp/replay-plan"));
            let bucket = std::env::var("ISS_BENCH_BUCKET").unwrap_or_else(|_| "P8_4_128".into());
            let obj = std::fs::read(bench.join(format!("{bucket}.elf"))).expect("the bucket object (just replay-plan)");
            let pkg = std::fs::read(bench.join(format!("{bucket}.safetensors")))
                .expect("the bucket weights (just replay-plan)");
            let (name, images, externs) = ots_format::read_object(&obj, Some("ref"), None).unwrap();
            let tensors = ots_format::read_weights(&pkg, Some(&name), None).unwrap();
            let words: Vec<(u8, u32, Vec<u8>)> = ots_format::bind_weights(&externs, &tensors)
                .unwrap()
                .iter()
                .map(|w| (w.slice as u8, w.addr as u32, w.word.to_vec()))
                .collect();
            let widx: HashMap<(u8, u32), usize> = words.iter().enumerate().map(|(i, w)| ((w.0, w.1), i)).collect();
            let images: Vec<(u16, Vec<u32>)> = images.iter().map(|(q, u)| (q, u.to_vec())).collect();
            let upw = 80;
            let mut sites = Vec::new();
            let mut locs = Vec::new();
            let mut site_words = std::collections::BTreeSet::new();
            for l in std::fs::read_to_string(plans.join("sites.txt")).unwrap().lines() {
                let f: Vec<&str> = l.split(' ').collect();
                let n = |i: usize| f[i].parse::<i64>().unwrap();
                sites.push(Site {
                    queue: n(1) as u16,
                    unit: n(2) as usize,
                    nunits: n(3) as usize,
                    lo: n(4) as u32,
                    width: n(5) as u32,
                    vmin: n(6),
                    vmax: n(7),
                    stride: n(8),
                    span: n(9),
                });
                let mut ls = Vec::new();
                for loc in &f[12..] {
                    let p: Vec<&str> = loc.split(':').collect();
                    let x = |i: usize| p[i].parse::<usize>().unwrap();
                    ls.push(match p[0] {
                        "I" => {
                            Loc::Image(images.iter().position(|(q, _)| *q as usize == x(1)).expect("queue image"), x(2))
                        }
                        _ => {
                            let wi = widx[&(x(1) as u8, x(2) as u32)];
                            site_words.insert(wi);
                            Loc::Word(wi, x(3))
                        }
                    });
                }
                locs.push(ls);
            }
            let bin = std::fs::read(plans.join("text.bin")).unwrap();
            let mut text = Vec::new();
            let mut at = 0;
            let u32at = |b: &[u8], i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
            while at < bin.len() {
                let (q, total) = (u32at(&bin, at), u32at(&bin, at + 8) as usize);
                at += 12;
                text.push(QueueText { queue: q as u16, units: (0..total).map(|k| u32at(&bin, at + 4 * k)).collect() });
                at += 4 * total;
            }
            let arena = std::fs::read_to_string(plans.join("arena.txt"))
                .unwrap()
                .lines()
                .map(|l| {
                    let f: Vec<u64> = l.split(' ').map(|x| x.parse().unwrap()).collect();
                    Run {
                        queue: f[0] as u16,
                        start: f[1] as usize,
                        slice: f[2] as u8,
                        addr: f[3] as u32,
                        words: f[4] as u32,
                    }
                })
                .collect();
            let ints = |name: &str| -> Vec<i32> {
                std::fs::read(plans.join(name))
                    .unwrap()
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            };
            let padded = ints("padded.bin");
            let all = ints("steps.bin");
            let steps: Vec<Vec<i32>> = all.chunks_exact(sites.len()).map(|c| c.to_vec()).collect();
            let nsteps = std::env::var("REPLAY_PLAN_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(steps.len());
            let _ = upw;
            Real {
                profile: Profile::load("ref").unwrap(),
                images,
                words,
                sites,
                locs,
                site_words: site_words.into_iter().collect(),
                text,
                arena,
                steps: steps.into_iter().take(nsteps).collect(),
                padded,
            }
        }
    }

    /// One chip's lineage of steps: the host's copy of the images and SRAM words as the patcher leaves them, and the machine of
    /// the last step. Store machines on the heap because this test keeps three lineages live concurrently.
    pub struct Lineage<'a> {
        r: &'a Real,
        images: Vec<(u16, Vec<u32>)>,
        words: Vec<(u8, u32, Vec<u8>)>,
        pub machine: Option<Box<Machine>>,
    }

    impl<'a> Lineage<'a> {
        pub fn new(r: &'a Real) -> Lineage<'a> {
            Lineage { r, images: r.images.clone(), words: r.words.clone(), machine: None }
        }

        fn unit(&self, l: Loc) -> u32 {
            match l {
                Loc::Image(q, u) => self.images[q].1[u],
                Loc::Word(w, k) => u32::from_le_bytes(self.words[w].2[4 * k..4 * k + 4].try_into().unwrap()),
            }
        }

        fn set(&mut self, l: Loc, v: u32) {
            match l {
                Loc::Image(q, u) => self.images[q].1[u] = v,
                Loc::Word(w, k) => self.words[w].2[4 * k..4 * k + 4].copy_from_slice(&v.to_le_bytes()),
            }
        }

        /// The patcher: every site gets its value.
        fn patch(&mut self, values: &[i32]) {
            for (i, s) in self.r.sites.iter().enumerate() {
                let mask = ((1u64 << s.width) - 1) << s.lo;
                let locs = &self.r.locs[i];
                let w = if s.nunits == 2 {
                    ((self.unit(locs[0]) as u64) << 32) | self.unit(locs[1]) as u64
                } else {
                    self.unit(locs[0]) as u64
                };
                let n = (w & !mask) | ((values[i] as u64) << s.lo);
                if s.nunits == 2 {
                    self.set(locs[0], (n >> 32) as u32);
                    self.set(locs[1], n as u32);
                } else {
                    self.set(locs[0], n as u32);
                }
            }
        }

        /// The chip of the next step: the plan patched in, a restarted machine on the SRAM of the last one.
        pub fn load(&mut self, values: &[i32]) -> Box<Machine> {
            self.patch(values);
            let (mut m, load_all_words) = match self.machine.take() {
                Some(mut previous) => {
                    previous.restart_with_links(Box::new(NoLinks));
                    (previous, false)
                }
                None => (Box::new(Machine::new(&self.r.profile, 0, Box::new(NoLinks)).unwrap()), true),
            };
            if load_all_words {
                for (s, a, b) in &self.words {
                    m.preload(*s, *a, b).unwrap();
                }
            } else {
                for &word in &self.r.site_words {
                    let (slice, address, bytes) = &self.words[word];
                    m.preload(*slice, *address, bytes).unwrap();
                }
            }
            for (q, u) in &self.images {
                m.load_image(*q, u).unwrap();
            }
            m
        }
    }

    /// The whole run folded into one hash: every event in order.
    #[derive(Default)]
    pub struct Fold {
        pub h: u64,
        pub n: u64,
        pub diags: Vec<String>,
    }

    impl Fold {
        fn mix(&mut self, v: u64) {
            self.h = (self.h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
            self.n += 1;
        }
        fn bytes(&mut self, b: &[u8]) {
            for c in b.chunks(8) {
                let mut x = [0u8; 8];
                x[..c.len()].copy_from_slice(c);
                self.mix(u64::from_le_bytes(x));
            }
        }
    }

    impl Observer for Fold {
        fn srf(&mut self, c: u64, row: Row, stream: u8, gap: u8, v: &[u8]) {
            self.mix(c ^ (row as u64) << 40 ^ (stream as u64) << 48 ^ (gap as u64) << 56 ^ 1);
            self.bytes(v);
        }
        fn sram(&mut self, c: u64, slice: u8, addr: u32, v: &[u8]) {
            self.mix(c ^ (slice as u64) << 40 ^ (addr as u64) << 48 ^ 2);
            self.bytes(v);
        }
        fn diag(&mut self, c: u64, queue: u16, code: ots_isa::Code) {
            self.mix(c ^ (queue as u64) << 40 ^ 3);
            self.diags.push(format!("{c} {queue} {}", code.as_str()));
        }
        fn lint(&mut self, c: u64, queue: u16, _code: &'static str) {
            self.mix(c ^ (queue as u64) << 40 ^ 4);
        }
        fn dispatch(&mut self, c: u64, queue: u16) {
            self.mix(c ^ (queue as u64) << 40 ^ 5);
        }
        fn fetch(&mut self, c: u64, queue: u16, bytes: u32) {
            self.mix(c ^ (queue as u64) << 40 ^ (bytes as u64) << 20 ^ 6);
        }
    }

    pub fn sram_digest(m: &Machine) -> [u8; 32] {
        let mut h = Sha256::new();
        for (s, a, w) in m.sram().nonzero_words() {
            h.update([s as u8]);
            h.update((a as u32).to_le_bytes());
            h.update(&w[..m.geom().vlanes as usize]);
        }
        h.finalize().into()
    }

    /// What a step leaves: ticks, last cycle, events folded, the diagnostics, the SRAM, the IQ levels.
    #[derive(PartialEq, Eq, Debug)]
    pub struct Seen {
        ticks: u64,
        last: Option<u64>,
        events: (u64, u64),
        diags: Vec<String>,
        sram: [u8; 32],
        levels: String,
    }

    pub fn seen(m: &Machine, s: tsp_iss::machine::RunSummary, f: Fold) -> Seen {
        Seen {
            ticks: s.ticks,
            last: s.last_cycle,
            events: (f.h, f.n),
            diags: f.diags,
            sram: sram_digest(m),
            levels: levels(m),
        }
    }

    fn param(r: &Real) -> (ParamSchedule, Box<Machine>) {
        // the recording of step 0 on a lineage of its own
        let mut lin = Lineage::new(r);
        let mut m = lin.load(&r.steps[0]);
        m.record_schedule().unwrap();
        m.run(LIMIT, &mut NoObserver).unwrap();
        let s = m.take_schedule().expect("recordable");
        let mut ps =
            ParamSchedule::build(&m, s, r.sites.clone(), &r.text, &r.steps[0]).unwrap_or_else(|e| panic!("{e}"));
        ps.set_arena(r.arena.clone());
        (ps, m)
    }

    #[test]
    #[ignore = "needs the inputs of `just replay-plan`"]
    fn shape_and_origins() {
        let r = Real::open();
        let t = Instant::now();
        let (ps, _) = param(&r);
        let sh = ps.shape();
        eprintln!("recorded and parametrised step 0 in {:.1} s: {sh:#?}", t.elapsed().as_secs_f64());
        assert_eq!(sh.sites, r.sites.len());
        assert!(sh.patched_elements > 0);
    }

    /// The exactness gate: for every step of the sequence the instantiated schedule gives the normal driver's run (events
    /// folded, cycle, diagnostics, SRAM, levels), and the normal driver checked tick by tick against the schedule agrees.
    #[test]
    #[ignore = "needs the inputs of `just replay-plan`"]
    fn every_step_of_the_sequence_is_the_normal_run() {
        verify_sequence_with_journal(true);
    }

    /// The separately declared 64 accepted plans; the broader padded-plan test above is retained.
    #[test]
    #[ignore = "needs the default 64 steps from just replay-plan-prep"]
    fn accepted64_sequence_retains_text_write_history() {
        verify_sequence_with_journal(false);
    }

    fn verify_sequence_with_journal(include_padded: bool) {
        let r = Real::open();
        let (mut ps, _rec) = param(&r);
        let (mut ln, mut lp, mut lc) = (Lineage::new(&r), Lineage::new(&r), Lineage::new(&r));
        let mut served = 0;
        let mut padded_fallbacks = 0;
        let mut stats = (0usize, 0usize, 0usize, 0usize);
        let plans: Vec<&Vec<i32>> = if include_padded {
            std::iter::once(&r.padded).chain(r.steps.iter()).collect()
        } else {
            assert_eq!(r.steps.len(), 64, "the exact declared accepted-plan domain");
            r.steps.iter().collect()
        };
        for (s, plan) in plans.iter().copied().enumerate() {
            // the normal driver
            let mut m = ln.load(plan);
            let mut f = Fold::default();
            let sum = m.run(LIMIT, &mut f).unwrap();
            let want = seen(&m, sum, f);
            ln.machine = Some(m);
            // the check: the normal driver against the instantiated schedule, tick by tick
            let mut mc = lc.load(plan);
            let st = match ps.instantiate(&mc, plan) {
                Ok(step) => step,
                Err(gate @ (Gate::Site(_) | Gate::Bank(_))) if include_padded && s == 0 => {
                    eprintln!("padded plan uses the normal driver: {gate}");
                    padded_fallbacks += 1;
                    let mut mp = lp.load(plan);
                    for machine in [&mut mc, &mut mp] {
                        let mut fold = Fold::default();
                        let summary = machine.run(LIMIT, &mut fold).unwrap();
                        assert_eq!(want, seen(machine, summary, fold), "padded plan fallback");
                    }
                    lc.machine = Some(mc);
                    lp.machine = Some(mp);
                    continue;
                }
                Err(gate) => panic!("step {s}: refused: {gate}"),
            };
            ps.arm(&mut mc, true).unwrap_or_else(|g| panic!("step {s}: {g}"));
            let mut f = Fold::default();
            let sum = mc.run(LIMIT, &mut f).unwrap_or_else(|e| panic!("step {s}: the per-tick check failed: {e}"));
            assert_eq!(want, seen(&mc, sum, f), "step {s}: the checked run");
            drop(mc.take_schedule()); // release the schedule
            lc.machine = Some(mc);
            // the replay
            let mut mp = lp.load(plan);
            ps.arm(&mut mp, false).unwrap_or_else(|g| panic!("step {s}: {g}"));
            let mut f = Fold::default();
            let sum = mp.run(LIMIT, &mut f).unwrap_or_else(|e| panic!("step {s}: the replay failed: {e}"));
            assert_eq!(want, seen(&mp, sum, f), "step {s}: the replayed run");
            drop(mp.take_schedule());
            ps.text_gate(&mp).unwrap_or_else(|g| panic!("step {s}: the text changed in the run: {g}"));
            assert_eq!(mp.sram().journal_pages(), None, "the completed replay closes its text journal");
            lp.machine = Some(mp);
            served += 1;
            stats = (stats.0 + st.sites_changed, stats.1 + st.insns, stats.2 + st.elements, stats.3 + st.banks_checked);
            eprintln!(
                "step {s}: cycles {:?}, {} sites changed, {} insns, {} elements, {} banks compared",
                want.last, st.sites_changed, st.insns, st.elements, st.banks_checked
            );
        }
        eprintln!("served {served} of {} plans; copies of the schedule {}; totals {stats:?}", plans.len(), ps.copies);
        assert_eq!(served + padded_fallbacks, plans.len());
        assert!(served >= r.steps.len(), "every live plan must be replayed");
        assert_eq!(ps.copies, 0);
    }

    /// Faults at cycle 0 (the verifier's protocol: `F sram 0` words of the weights and the text), and plans outside the table.
    #[test]
    #[ignore = "needs the inputs of `just replay-plan`"]
    fn faults_and_foreign_plans_are_refused_and_the_normal_driver_answers() {
        let r = Real::open();
        let (mut ps, _rec) = param(&r);
        let mut ln = Lineage::new(&r);
        let plan = &r.steps[1];
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (mut refused, mut text_flips) = (0, 0);
        ps.instantiate(&ln.load(plan), plan).unwrap();
        for k in 0..24 {
            let mut m = ln.load(plan);
            let (s, a, _) = &r.words[(rng.next() % r.words.len() as u64) as usize];
            m.service_flip_sram(*s, *a, (rng.next() % 20) as u8, (rng.next() % 138) as u8).unwrap();
            let g = ps.arm(&mut m, false);
            if k == 0 {
                eprintln!("an ECC fault at cycle 0 on slice {s} word {a:#x}: {}", g.as_ref().unwrap_err());
            }
            assert_eq!(g, Err(Gate::Ecc), "fault {k}");
            refused += 1;
            ln.machine = Some(m);
        }
        // a data bit of the text arena changed without an ECC fault (another program's text): the text gate
        for k in 0..8 {
            let mut m = ln.load(plan);
            let run = r.arena[(rng.next() % r.arena.len() as u64) as usize];
            let mut w = m.sram_word(run.slice, run.addr).unwrap();
            let bit = (rng.next() % 128) as usize;
            w[bit / 8] ^= 1 << (bit % 8);
            m.preload(run.slice, run.addr, &w[..320]).unwrap();
            let g = ps.arm(&mut m, false);
            if k == 0 {
                eprintln!("a data bit of the text arena flipped (no ECC fault): {}", g.as_ref().unwrap_err());
            }
            assert!(matches!(g, Err(Gate::Text(_))), "text flip {k}: {g:?}");
            text_flips += 1;
            ln.machine = Some(m);
        }
        // a plan with a site value off the table
        let mut bad = plan.clone();
        bad[0] = r.sites[0].vmax as i32 + 1;
        let g = ps.instantiate(&ln.load(plan), &bad);
        eprintln!("a site value outside its range: {}", g.as_ref().unwrap_err());
        assert!(matches!(g, Err(Gate::Values(_))));
        // a plan of another length
        let g = ps.instantiate(&ln.load(plan), &plan[1..]);
        eprintln!("a plan with one value too few: {}", g.as_ref().unwrap_err());
        assert!(matches!(g, Err(Gate::Values(_))));
        // Change a legal embedding-map value without changing a memory address.
        let mut fresh = Lineage::new(&r);
        let base = fresh.load(plan);
        let (index, site) = r
            .sites
            .iter()
            .enumerate()
            .find(|(_, site)| {
                let units = &r.text.iter().find(|q| q.queue == site.queue).unwrap().units;
                matches!(
                    decode_text(base.isa(), units[site.unit], units.get(site.unit + 1).copied().unwrap_or(0)),
                    Ok(Insn::Permute { .. })
                )
            })
            .expect("a serving embedding-map site");
        drop(base);
        let mut foreign = plan.clone();
        let value = i64::from(plan[index]);
        foreign[index] =
            if value + site.stride <= site.vmax { value + site.stride } else { value - site.stride } as i32;
        let mut m = fresh.load(&foreign);
        assert!(matches!(ps.instantiate(&m, &foreign), Err(Gate::Site(_))));
        let mut log = Fold::default();
        m.run(LIMIT, &mut log).unwrap();
        assert!(log.diags.is_empty(), "{:?}", log.diags);
        eprintln!("faults refused {refused}, text flips refused {text_flips}");
        assert_eq!(ps.values(), &plan[..]);
    }

    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    fn load_average() -> String {
        std::process::Command::new("uptime")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).split("load average").nth(1).unwrap_or("?").trim().to_string())
            .unwrap_or_else(|_| "?".into())
    }

    #[derive(Default, Clone)]
    struct Phases {
        load: f64,
        plan: f64,
        run: f64,
    }

    /// The A/B of the PoC (opentsp-iss/README.md section 3): the same plans, interleaved rounds.
    #[test]
    #[ignore = "needs the inputs of `just replay-plan`; about ten minutes"]
    fn bench() {
        let r = Real::open();
        let rounds: usize = std::env::var("REPLAY_PLAN_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let (mut ps, rec_machine) = param(&r);
        drop(rec_machine);
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let mut res: HashMap<&str, Vec<Phases>> = HashMap::new(); // per round, per variant: total of the sequence
        let mut per_step: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut first_step: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut inst_ms = Vec::new();
        let mut arm_ms = Vec::new();
        let mut build_ms = Vec::new();
        let (mut checked_steps, mut unchecked_steps) = (Vec::new(), Vec::new());
        let mut loads = Vec::new();
        // the plain key of every step (profile and initial IQ images): how many different programs the shipped replay sees
        {
            let mut lin = Lineage::new(&r);
            let mut keys = std::collections::BTreeSet::new();
            let (mut key_ms, mut gate_ms, mut n) = (0.0, 0.0, 0);
            for plan in &r.steps {
                let mut m = lin.load(plan);
                let t = Instant::now();
                keys.insert(m.schedule_key());
                key_ms += ms(t);
                ps.instantiate(&m, plan).unwrap();
                let t = Instant::now();
                ps.text_gate(&m).expect("the text of the plan");
                gate_ms += ms(t);
                n += 1;
                drop(m.take_schedule());
                lin.machine = Some(m);
            }
            eprintln!(
                "distinct plain keys over {} steps: {}; schedule_key {:.2} ms, text gate {:.2} ms (means of {n})",
                r.steps.len(),
                keys.len(),
                key_ms / n as f64,
                gate_ms / n as f64
            );
        }
        for round in 0..rounds {
            // N the normal driver, S plain replay, P parametrised replay, C the same with the per-tick check on every 8th served step (the
            // checked and the unchecked steps of one run are compared step by step: the cost of one check, amortised over n steps, is
            // (checked - unchecked) / (n * unchecked)), A a second copy of P (the noise floor: P against A is one program twice)
            let all = ["N", "S", "P", "C", "A"];
            let order: Vec<&str> = (0..5).map(|k| all[(k + round) % 5]).collect();
            for v in order {
                let before = load_average();
                let mut ph = Phases::default();
                let mut lin = Lineage::new(&r);
                let cache = Arc::new(ReplayCache::new(512 << 20, 4));
                // the schedule of the variant P, built from its own recording of step 0 inside the timed sequence of every round
                let mut round_ps: Option<ParamSchedule> = None;
                for (s, plan) in r.steps.iter().enumerate() {
                    let step_plan = if v == "S" { &r.steps[0] } else { plan };
                    let t = Instant::now();
                    let mut m = lin.load(step_plan);
                    let l = ms(t);
                    let t = Instant::now();
                    let mut recording = false;
                    match v {
                        "S" => {
                            recording = m.replay_session_begin_in(&cache, RecordOn::Second)
                                == tsp_iss::driver::replay::Began::Recording;
                        }
                        "P" | "C" | "A" if s > 0 => {
                            let ps = round_ps.as_mut().expect("built at step 0");
                            let t0 = Instant::now();
                            ps.instantiate(&m, plan).unwrap_or_else(|g| panic!("step {s}: {g}"));
                            inst_ms.push(ms(t0));
                            let t0 = Instant::now();
                            ps.arm(&mut m, v == "C" && (s - 1) % 8 == 0).unwrap_or_else(|g| panic!("step {s}: {g}"));
                            arm_ms.push(ms(t0));
                        }
                        "P" | "C" | "A" => m.record_schedule().unwrap(),
                        _ => {}
                    }
                    let mut p = ms(t);
                    let t = Instant::now();
                    m.run(LIMIT, &mut NoObserver).unwrap();
                    if v == "S" && recording {
                        m.replay_session_end();
                    }
                    let run = ms(t);
                    if v != "N" && v != "S" && s == 0 {
                        // `ParamSchedule::build` is part of the plan work of the sequence: it is timed in the total
                        let sched = m.take_schedule().expect("recordable");
                        let t0 = Instant::now();
                        let mut built = ParamSchedule::build(&m, sched, r.sites.clone(), &r.text, plan)
                            .unwrap_or_else(|e| panic!("{e}"));
                        built.set_arena(r.arena.clone());
                        let b = ms(t0);
                        build_ms.push(b);
                        p += b;
                        round_ps = Some(built);
                    }
                    drop(m.take_schedule());
                    lin.machine = Some(m);
                    ph.load += l;
                    ph.plan += p;
                    ph.run += run;
                    per_step.entry(v).or_default().push(p + run);
                    if v == "C" && s > 0 {
                        if (s - 1) % 8 == 0 { &mut checked_steps } else { &mut unchecked_steps }.push(p + run);
                    }
                    if s == 0 {
                        first_step.entry(v).or_default().push(p + run);
                    }
                    loads.push(l);
                }
                eprintln!(
                    "round {round} {v}: load {:.0} ms, plan {:.1} ms, run {:.0} ms, total without load {:.0} ms (load average before: {before})",
                    ph.load,
                    ph.plan,
                    ph.run,
                    ph.plan + ph.run
                );
                res.entry(v).or_default().push(ph);
            }
        }
        eprintln!("\nvariant   sequence (plan+run) median   run median   plan median   per-step median (plan+run)");
        for v in ["N", "S", "P", "C", "A"] {
            let rs = &res[v];
            let mut seq: Vec<f64> = rs.iter().map(|p| p.plan + p.run).collect();
            let mut run: Vec<f64> = rs.iter().map(|p| p.run).collect();
            let mut plan: Vec<f64> = rs.iter().map(|p| p.plan).collect();
            let mut st = per_step[v].clone();
            eprintln!(
                "{v}   {:.0} ms   {:.0} ms   {:.1} ms   {:.1} ms",
                median(&mut seq),
                median(&mut run),
                median(&mut plan),
                median(&mut st)
            );
        }
        for v in ["N", "S", "P", "C", "A"] {
            eprintln!(
                "first step of {v} (the one that records, for S the first sight of the key): median {:.1} ms",
                median(first_step.get_mut(v).unwrap())
            );
        }
        eprintln!(
            "instantiate median {:.2} ms, arm median {:.2} ms, ParamSchedule::build median {:.0} ms (in the total of P), host load median {:.1} ms",
            median(&mut inst_ms),
            median(&mut arm_ms),
            median(&mut build_ms),
            median(&mut loads)
        );
        let n = median(&mut res["N"].iter().map(|p| p.plan + p.run).collect::<Vec<_>>());
        let p = median(&mut res["P"].iter().map(|p| p.plan + p.run).collect::<Vec<_>>());
        eprintln!("speed-up P over N on the sequence: {:.2}x", n / p);
        // the cost of the per-tick check: the steps of C that ran in it against the others of the same runs
        let (chk, unchk) = (median(&mut checked_steps), median(&mut unchecked_steps));
        eprintln!(
            "check cost: a checked step {chk:.1} ms (median of {}), an unchecked one {unchk:.1} ms (median of {}): {:.2}x; amortised over 64 steps {:.2} %, over 128 {:.2} %, over 256 {:.2} %",
            checked_steps.len(),
            unchecked_steps.len(),
            chk / unchk,
            100.0 * (chk / unchk - 1.0) / 64.0,
            100.0 * (chk / unchk - 1.0) / 128.0,
            100.0 * (chk / unchk - 1.0) / 256.0
        );
        let tot = |v: &str| -> Vec<f64> { res[v].iter().map(|p| p.plan + p.run).collect() };
        eprintln!(
            "totals (informational): P {:.0} ms, C {:.0} ms (every 8th step checked), A {:.0} ms",
            median(&mut tot("P")),
            median(&mut tot("C")),
            median(&mut tot("A"))
        );
    }

    /// One sequence of `n` steps as the platform would serve it: step 0 records, then every step is served from the schedule, except
    /// that with a policy `(sample_every, check_every)` the first step after the recording and then one in `sample_every` run on the
    /// normal driver (the warm baseline of the measured gain) and one served step in `check_every`, the first included, runs in the
    /// per-tick check. Milliseconds of plan work and run, the load of the machine excluded (the host's cost, as in `bench`).
    fn sequence(r: &Real, policy: Option<(u64, u64)>, n: usize) -> f64 {
        let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
        let mut lin = Lineage::new(r);
        let mut ps: Option<ParamSchedule> = None;
        let mut refused = 0u64;
        let (mut served, mut since_sample, mut have_baseline, mut total) = (0u64, 0u64, false, 0.0);
        for s in 0..n {
            let plan = &r.steps[s % r.steps.len()];
            let mut m = lin.load(plan);
            let t = Instant::now();
            if s == 0 {
                m.record_schedule().unwrap();
                m.run(LIMIT, &mut NoObserver).unwrap();
                let sched = m.take_schedule().expect("recordable");
                let mut built =
                    ParamSchedule::build(&m, sched, r.sites.clone(), &r.text, plan).unwrap_or_else(|e| panic!("{e}"));
                built.set_arena(r.arena.clone());
                ps = Some(built);
            } else if policy.is_some_and(|(every, _)| !have_baseline || since_sample >= every) {
                have_baseline = true;
                since_sample = 0;
                m.run(LIMIT, &mut NoObserver).unwrap();
            } else {
                since_sample += 1;
                let check = policy.is_some_and(|(_, every)| served % every == 0);
                served += 1;
                let ps = ps.as_mut().expect("built at step 0");
                // as the platform does ( (1)): a plan that changes a site that is not the address of a Read or a Write is refused at
                // plan time and the step runs on the normal driver; the refusals are counted and printed with the total
                match ps.instantiate(&m, plan).and_then(|_| ps.arm(&mut m, check)) {
                    Ok(()) => {}
                    Err(_) => refused += 1,
                }
                m.run(LIMIT, &mut NoObserver).unwrap();
            }
            total += ms(t);
            drop(m.take_schedule());
            lin.machine = Some(m);
        }
        eprintln!("  steps refused at plan time (run on the normal driver): {refused} of {n}");
        total
    }

    /// The cost of the default replay devices, the warm sampling and the per-tick check together, against replay with neither
    /// (`REPLAY_PLAN_ROUNDS` interleaved rounds, the order rotating, over the steps of the inputs of `just replay-plan-prep`: at most
    /// 64, because the sequence's context must stay in the 128-slot bucket (with `REPLAY_PLAN_STEPS` 112, 116 or 256 the prep stops on an
    /// assertion or a `StepPlanError`). A step whose plan changes a site that is not the address of a Read or a Write is refused at plan time
    /// and runs on the normal driver, as in the platform ( (1)); the refusals are printed.) P no device, D the defaults
    /// (sample 1 in 32, check 1 in 64), E one in 64 and one in 128.
    #[test]
    #[ignore = "needs the inputs of `just replay-plan`; about five minutes"]
    fn default_devices_cost() {
        let r = Real::open();
        let rounds: usize = std::env::var("REPLAY_PLAN_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let n = r.steps.len();
        let variants: [(&str, Option<(u64, u64)>); 3] = [("P", None), ("D", Some((32, 64))), ("E", Some((64, 128)))];
        let mut totals: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut ratios: HashMap<&str, Vec<f64>> = HashMap::new();
        let mut loads: Vec<String> = Vec::new();
        for round in 0..rounds {
            let mut this: HashMap<&str, f64> = HashMap::new();
            for k in 0..variants.len() {
                let (name, policy) = variants[(k + round) % variants.len()];
                let before = load_average();
                let t = sequence(&r, policy, n);
                loads.push(before.clone());
                eprintln!("round {round} {name}: {t:.0} ms over {n} steps (load average before: {before})");
                this.insert(name, t);
                totals.entry(name).or_default().push(t);
            }
            for (name, _) in &variants[1..] {
                ratios.entry(name).or_default().push(this[name] / this["P"]);
            }
        }
        eprintln!("\nsteps {n}, rounds {rounds}, load average before each sequence: {}", loads.join(", "));
        for (name, policy) in &variants[1..] {
            let rs = ratios.get_mut(name).unwrap();
            let (lo, hi) = (rs.iter().cloned().fold(f64::MAX, f64::min), rs.iter().cloned().fold(0.0, f64::max));
            eprintln!(
                "{name} {policy:?}: median of the pair ratios over P {:.4} ({:+.2} %), range {:.4} to {:.4}; medians P {:.0} ms, {name} {:.0} ms",
                median(rs),
                100.0 * (median(rs) - 1.0),
                lo,
                hi,
                median(totals.get_mut("P").unwrap()),
                median(totals.get_mut(name).unwrap())
            );
        }
    }
}
