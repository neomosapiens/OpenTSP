//! Parametrised replay in the platform (`Platform::start_plan`, `src/platform/param.rs`): a shape recorded on its second
//! sight serves the later plans, the steps are the normal driver's step bit for bit, and what the gate must keep out goes to the
//! normal driver: a fault or a scrub during a run, a word written over the text, a plan the table refuses, a small bucket.
//!
//! The programs are the generated mixed programs of `tests/data/replay/` with a site on the address of every MEM Read and Write.
//! Run: `cargo test -p tsp-iss --features replay --test replay_platform`.
#![cfg(feature = "replay")]
use ots_isa::Insn;
use ots_profile::Profile;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tsp_iss::driver::replay::ReplayCache;
use tsp_iss::driver::replay_param::{QueueText, Run, Site};
use tsp_iss::frontend::{decode_text, text_width};
use tsp_iss::platform::{CheckMode, ParamSpec, ParamWhy, Phase, Platform, Record};

const MAX_CYCLES: u64 = 6000;

thread_local! {
    /// Heap bytes allocated and not yet freed by the calls of this thread (a test that measures resident memory reads it on the
    /// thread that built everything it measures, so the other tests of the file do not move it).
    static HEAP: Cell<isize> = const { Cell::new(0) };
}

/// The system allocator with a per-thread count of the bytes it holds.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        HEAP.with(|h| h.set(h.get() + l.size() as isize));
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        HEAP.with(|h| h.set(h.get() - l.size() as isize));
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        HEAP.with(|h| h.set(h.get() + new as isize - l.size() as isize));
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn heap() -> isize {
    HEAP.with(Cell::get)
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

#[derive(Clone)]
struct Prog {
    profile: &'static str,
    images: Vec<(u16, Vec<u32>)>,
    words: Vec<(u8, u32, Vec<u8>)>,
}

fn load(name: &str) -> Prog {
    let text = std::fs::read_to_string(data_dir().join(name)).unwrap();
    let profile = if name.starts_with("ref") { "ref" } else { "mini" };
    let mut p = Prog { profile, images: Vec::new(), words: Vec::new() };
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

fn profile(p: &Prog) -> Profile {
    Profile::load(p.profile).unwrap()
}

/// The sites of a program (the address of every Read and Write of the initial images), the values it has there and the text.
fn sites_of(p: &Prog) -> (Vec<Site>, Vec<i32>, Vec<QueueText>, u32) {
    let prof = profile(p);
    let g = tsp_iss::Geom::new(&prof);
    let isa = ots_isa::Isa::new(&prof);
    let words = g.words_per_slice;
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
                decode_text(&isa, units[pos], units.get(pos + 1).copied().unwrap_or(0))
            {
                let v = i.entry().vars.iter().find(|v| v.name == "addr").expect("an address operand");
                sites.push(Site {
                    queue: *q,
                    unit: pos,
                    nunits: w,
                    lo: u32::from(v.lo),
                    width: u32::from(v.n),
                    vmin: 0,
                    vmax: (words as i64 - 1).min((1i64 << v.n) - 1),
                    stride: 1,
                    span: 1,
                });
                values.push(addr as i32);
            }
            pos += w;
        }
    }
    (sites, values, text, g.bank_bit as u32)
}

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

/// Another plan: every site takes another address of the bank it had.
fn other_plan(bank_bit: u32, words: u32, values: &[i32], rng: &mut Rng) -> Vec<i32> {
    values
        .iter()
        .map(|&v| {
            let bank = (v as u32 >> bank_bit) & 1;
            loop {
                let a = (rng.next() % u64::from(words)) as u32;
                if (a >> bank_bit) & 1 == bank {
                    return a as i32;
                }
            }
        })
        .collect()
}

/// Run `body` on a thread with a stack large enough for the machines of a served run (a machine of the widest lane count
/// and the replay bookkeeping around it do not fit the 2 MiB of a test thread twice over).
fn on_big_stack(body: fn()) {
    let t = std::thread::Builder::new().stack_size(32 << 20).spawn(body).unwrap();
    if let Err(e) = t.join() {
        std::panic::resume_unwind(e);
    }
}

/// A platform in reset with the program of `p` loaded (the images patched with `values` at the sites).
fn loaded(p: &Prog, cache: &Arc<ReplayCache>) -> Platform {
    let mut pl = Platform::new(&profile(p), 0).unwrap();
    pl.set_replay_cache(cache.clone());
    pl
}

fn fill(pl: &mut Platform, p: &Prog) {
    pl.reset().unwrap();
    for (q, units) in &p.images {
        pl.select_iq(*q).unwrap();
        pl.write_iq_window(units).unwrap();
    }
    for (s, a, w) in &p.words {
        pl.write_sram_word(*s, *a, w).unwrap();
    }
}

/// What a step leaves that the host reads: the cycle count, the diagnostics and the SRAM.
type Outcome = (u64, Vec<Record>, [u8; 32]);

fn outcome(pl: &Platform) -> Outcome {
    (pl.cycle(), pl.diagnostics().to_vec(), pl.sram_digest())
}

/// One step on the normal driver of a platform that knows nothing of shapes.
fn normal_step(prog: &Prog) -> Outcome {
    let mut pl = Platform::new(&profile(prog), 0).unwrap();
    fill(&mut pl, prog);
    pl.start().unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
    outcome(&pl)
}

fn spec(sites: &[Site], text: &[QueueText]) -> ParamSpec {
    ParamSpec { sites: sites.to_vec(), text: text.to_vec(), arena: Vec::new() }
}

fn cache() -> Arc<ReplayCache> {
    Arc::new(ReplayCache::new(512 << 20, 4))
}

#[test]
fn a_shape_is_recorded_on_its_second_sight_and_serves_the_later_plans_as_the_normal_driver_would() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut shaped, mut served_total) = (0, 0);
    for n in names().iter().take(12) {
        let p = load(n);
        let (sites, values, text, bank_bit) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let words = tsp_iss::Geom::new(&profile(&p)).words_per_slice as u32;
        let c = cache();
        let mut pl = loaded(&p, &c);
        pl.set_param_replay(true);
        pl.set_param_min_gain(0.0);
        pl.add_param_shape("bucket", spec(&sites, &text));
        for k in 0..7 {
            let plan = if k == 0 { values.clone() } else { other_plan(bank_bit, words, &values, &mut rng) };
            let prog = patched(&p, &sites, &plan);
            fill(&mut pl, &prog);
            pl.start_plan("bucket", &plan).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
            assert_eq!(outcome(&pl), normal_step(&prog), "{n} step {k} differs from the normal driver");
        }
        let st = pl.replay_status();
        let row = &st.shapes[0];
        if row.state != "ready" {
            eprintln!("{n}: {:?}", row.why);
            continue;
        }
        shaped += 1;
        served_total += st.served;
        // step 0 first sight, step 1 records, step 2 is the warm normal run that is the baseline of the gain, steps 3..6 are served,
        // none refused
        assert_eq!((st.served, st.refused, st.fallback), (4, 0, 0), "{n}");
        let counters = pl.replay_counters();
        assert_eq!((counters.param_served, counters.param_recordings, counters.param_first_sight), (4, 1, 1), "{n}");
    }
    assert!(shaped >= 6, "only {shaped} programs were served");
    eprintln!("programs served {shaped}, steps served {served_total}");
}

#[test]
fn the_device_is_off_by_default_and_start_plan_is_start() {
    let p = load(&names()[0]);
    let (sites, values, text, _) = sites_of(&p);
    let c = cache();
    let mut pl = loaded(&p, &c);
    pl.add_param_shape("bucket", spec(&sites, &text));
    assert!(!pl.param_replay());
    for _ in 0..4 {
        fill(&mut pl, &p);
        pl.start_plan("bucket", &values).unwrap();
        pl.run_to_done(MAX_CYCLES).unwrap();
        assert_eq!(outcome(&pl), normal_step(&p));
    }
    let st = pl.replay_status();
    assert_eq!((st.enabled, st.served, st.refused, st.fallback), (false, 0, 0, 0));
    assert_eq!(st.shapes[0].state, "first");
    let k = c.counters();
    assert_eq!((k.param_served, k.param_recordings, k.param_first_sight), (0, 0, 0));
}

/// A program and a plan sequence whose shape is `ready` after two steps and has a warm baseline after a third: the platform, the
/// program, its sites and the values.
fn ready(rng: &mut Rng, min_gain: f64) -> (Platform, Prog, Vec<Site>, Vec<i32>, Vec<QueueText>, u32, u32) {
    for n in names() {
        let p = load(&n);
        let (sites, values, text, bank_bit) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let words = tsp_iss::Geom::new(&profile(&p)).words_per_slice as u32;
        let c = cache();
        let mut pl = loaded(&p, &c);
        pl.set_param_replay(true);
        pl.set_param_min_gain(min_gain);
        pl.add_param_shape("bucket", spec(&sites, &text));
        for k in 0..2 {
            let plan = if k == 0 { values.clone() } else { other_plan(bank_bit, words, &values, rng) };
            fill(&mut pl, &patched(&p, &sites, &plan));
            pl.start_plan("bucket", &plan).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
        }
        if pl.replay_status().shapes[0].state == "ready" {
            // the first step of a ready shape is the warm normal run that is the baseline of the measured gain
            fill(&mut pl, &p);
            pl.start_plan("bucket", &values).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
            assert_eq!(pl.replay_status().served, 0);
            return (pl, p, sites, values, text, bank_bit, words);
        }
    }
    panic!("no program of the corpus gives a shape");
}

#[test]
fn active_shape_replacements_apply_on_reset_and_restart_and_last_registration_wins() {
    for restart in [false, true] {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (mut pl, p, sites, values, text, _bank, _words) = ready(&mut rng, 0.0);
        fill(&mut pl, &p);
        pl.start_plan("bucket", &values).unwrap();
        pl.advance(1).unwrap();
        assert_eq!(pl.phase(), Phase::Running);
        pl.add_param_shape("bucket", ParamSpec { sites: vec![], text: vec![], arena: vec![] });
        pl.add_param_shape("bucket", spec(&sites, &text));
        assert_eq!(pl.replay_status().shapes[0].state, "ready");
        if restart {
            pl.restart().unwrap();
        } else {
            pl.reset().unwrap();
        }
        assert_eq!(pl.replay_status().shapes[0].state, "first");
        for _ in 0..2 {
            fill(&mut pl, &p);
            pl.start_plan("bucket", &values).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
        }
        assert_eq!(pl.replay_status().shapes[0].state, "ready", "the final valid replacement was applied");
    }
}

#[test]
fn disabling_param_replay_keeps_an_inflight_runs_integrity_and_cleanup() {
    for checked in [false, true] {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (mut pl, p, sites, values, _text, bank_bit, words) = ready(&mut rng, 0.0);
        pl.set_param_check(checked);
        let plan = other_plan(bank_bit, words, &values, &mut rng);
        let prog = patched(&p, &sites, &plan);
        fill(&mut pl, &prog);
        pl.start_plan("bucket", &plan).unwrap();
        pl.advance(3).unwrap();
        assert_eq!(pl.phase(), Phase::Running);
        pl.set_param_replay(false);
        assert!(!pl.param_replay());
        let (slice, addr, ..) = p.words[0].clone();
        pl.service_flip_sram(slice, addr, 0, 1).unwrap();
        assert_eq!(pl.replay_status().fallback, 1, "disabling must retain in-flight service checks");
        pl.run_to_done(MAX_CYCLES).unwrap();
        let served = pl.replay_status().served;
        fill(&mut pl, &prog);
        pl.start_plan("bucket", &plan).unwrap();
        pl.run_to_done(MAX_CYCLES).unwrap();
        assert_eq!(outcome(&pl), normal_step(&prog));
        assert_eq!(pl.replay_status().served, served, "the next run must honor the disabled flag");
    }
}

#[test]
fn a_fault_flip_or_a_scrub_during_a_served_run_sends_it_to_the_normal_driver() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut pl, p, sites, values, _text, bank_bit, words) = ready(&mut rng, 0.0);
    let plan = other_plan(bank_bit, words, &values, &mut rng);
    let prog = patched(&p, &sites, &plan);
    let (slice, addr, ..) = p.words[0].clone();
    // the same calls on a platform that never serves: start, a few cycles, a flip of two bits of one word (an uncorrectable
    // fault that reads of the word report), the rest of the run, a scrub
    let reference = {
        let mut r = Platform::new(&profile(&p), 0).unwrap();
        fill(&mut r, &prog);
        r.start().unwrap();
        r.advance(3).unwrap();
        r.service_flip_sram(slice, addr, 0, 1).unwrap();
        r.service_flip_sram(slice, addr, 0, 2).unwrap();
        r.run_to_done(MAX_CYCLES).unwrap();
        r.service_scrub();
        outcome(&r)
    };
    fill(&mut pl, &prog);
    pl.start_plan("bucket", &plan).unwrap();
    assert_eq!(pl.replay_status().served, 1, "the step is served until the flip");
    pl.advance(3).unwrap();
    assert_eq!(pl.phase(), Phase::Running);
    pl.service_flip_sram(slice, addr, 0, 1).unwrap();
    assert_eq!(pl.replay_status().fallback, 1, "the flip sent the run to the normal driver at once");
    pl.service_flip_sram(slice, addr, 0, 2).unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
    pl.service_scrub();
    assert_eq!(outcome(&pl), reference, "the run is the normal driver's with the same calls");
    let st = pl.replay_status();
    assert_eq!((st.served, st.fallback), (1, 1));
    assert_eq!(pl.replay_counters().param_fallback, 1);
    // the shape serves the next plan again
    let plan2 = other_plan(bank_bit, words, &values, &mut rng);
    let prog2 = patched(&p, &sites, &plan2);
    fill(&mut pl, &prog2);
    pl.start_plan("bucket", &plan2).unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
    assert_eq!(outcome(&pl), normal_step(&prog2));
    assert_eq!(pl.replay_status().served, 2);
}

#[test]
fn a_fault_before_the_start_is_not_served() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (mut pl, p, sites, values, _text, bank_bit, words) = ready(&mut rng, 0.0);
    let plan = other_plan(bank_bit, words, &values, &mut rng);
    let prog = patched(&p, &sites, &plan);
    let (slice, addr, ..) = p.words[0].clone();
    fill(&mut pl, &prog);
    pl.service_flip_sram(slice, addr, 0, 1).unwrap();
    pl.start_plan("bucket", &plan).unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
    let reference = {
        let mut r = Platform::new(&profile(&p), 0).unwrap();
        fill(&mut r, &prog);
        r.service_flip_sram(slice, addr, 0, 1).unwrap();
        r.start().unwrap();
        r.run_to_done(MAX_CYCLES).unwrap();
        outcome(&r)
    };
    assert_eq!(outcome(&pl), reference);
    let st = pl.replay_status();
    assert_eq!((st.served, st.refused), (0, 1));
}

#[test]
fn a_scrub_before_the_start_is_not_served() {
    // a scrub changes no word when no fault is live, so no other gate would see it: a run with a fault flip or a scrub between reset and start is never served, and that is the platform's own rule
    let mut rng = Rng(0x0bad_cafe_1357_9bdf);
    let (mut pl, p, sites, values, _text, bank_bit, words) = ready(&mut rng, 0.0);
    let plan = other_plan(bank_bit, words, &values, &mut rng);
    let prog = patched(&p, &sites, &plan);
    fill(&mut pl, &prog);
    pl.service_scrub();
    pl.start_plan("bucket", &plan).unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
    assert_eq!(outcome(&pl), {
        let mut r = Platform::new(&profile(&p), 0).unwrap();
        fill(&mut r, &prog);
        r.service_scrub();
        r.start().unwrap();
        r.run_to_done(MAX_CYCLES).unwrap();
        outcome(&r)
    });
    let st = pl.replay_status();
    assert_eq!((st.served, st.refused), (0, 1));
}

#[test]
fn a_plan_the_table_refuses_takes_the_normal_driver() {
    let mut rng = Rng(0x0fed_cba9_8765_4321);
    let (mut pl, p, sites, values, _text, _bank_bit, _words) = ready(&mut rng, 0.0);
    let mut plan = values.clone();
    plan[0] = sites[0].vmax as i32 + 1; // off the table's range: the normal driver gets the images with this value as they are
    let prog = patched(&p, &sites, &values);
    fill(&mut pl, &prog);
    pl.start_plan("bucket", &plan).unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
    assert_eq!(outcome(&pl), normal_step(&prog));
    let st = pl.replay_status();
    assert_eq!((st.served, st.refused, st.fallback), (0, 1, 0));
    assert_eq!(pl.replay_counters().param_refused, 1);
}

#[test]
fn a_shape_that_measures_under_the_minimum_gain_is_served_by_the_normal_driver() {
    let mut rng = Rng(0x7777_1111_2222_3333);
    // the minimum is a gain no replay reaches: the five replayed steps that are timed decide it
    let (mut pl, p, sites, values, _text, bank_bit, words) = ready(&mut rng, 1.0e9);
    for k in 0..12 {
        let plan = other_plan(bank_bit, words, &values, &mut rng);
        let prog = patched(&p, &sites, &plan);
        fill(&mut pl, &prog);
        pl.start_plan("bucket", &plan).unwrap();
        pl.run_to_done(MAX_CYCLES).unwrap();
        assert_eq!(outcome(&pl), normal_step(&prog), "step {k}");
    }
    let st = pl.replay_status();
    let row = &st.shapes[0];
    assert_eq!(row.state, "normal");
    assert!(matches!(row.why, Some(ParamWhy::Small { gain, .. }) if gain < 1.0e9), "{:?}", row.why);
    assert!(row.gain.is_some() && row.normal_us.is_some() && row.replay_us.is_some());
    assert_eq!(
        st.served, 11,
        "eleven steps were served: the first ran in the per-tick check and is not timed, ten are (two decisions of five)"
    );
    assert_eq!(row.normal_runs, 2, "the baseline run and the step after the decision");
    assert_eq!(pl.replay_counters().param_small, 1);
}

/// A word of the text arena written over by the program while it runs: the program has a Write whose address is a site; the
/// arena (declared for the test: the first words of the text of the Write's queue, which the host keeps in the SRAM of the slice
/// that queue writes) is the target of one plan.
#[test]
fn a_write_over_the_text_during_a_served_run_sends_it_to_the_normal_driver() {
    on_big_stack(|| a_write_over_the_text_during_a_served_run_sends_it_to_the_normal_driver_body(0));
}

#[test]
fn disabling_midrun_cannot_bypass_the_final_text_gate() {
    on_big_stack(|| a_write_over_the_text_during_a_served_run_sends_it_to_the_normal_driver_body(1));
}

#[test]
fn replacing_an_active_shape_cannot_bypass_the_final_text_gate() {
    on_big_stack(|| a_write_over_the_text_during_a_served_run_sends_it_to_the_normal_driver_body(2));
}

#[test]
fn adding_another_shape_preserves_the_active_text_gate() {
    on_big_stack(|| a_write_over_the_text_during_a_served_run_sends_it_to_the_normal_driver_body(3));
}

fn a_write_over_the_text_during_a_served_run_sends_it_to_the_normal_driver_body(change: u8) {
    let mut rng = Rng(0x5151_5151_a1a1_a1a1);
    let (mut tried, mut hit) = (0, 0);
    'programs: for n in names() {
        let p = load(&n);
        let (sites, values, text0, bank_bit) = sites_of(&p);
        let prof = profile(&p);
        let g = tsp_iss::Geom::new(&prof);
        let (words, upw) = (g.words_per_slice as u32, g.vlanes as usize / 4);
        let isa = ots_isa::Isa::new(&prof);
        let writes: Vec<usize> = (0..sites.len())
            .filter(|&i| {
                let s = &sites[i];
                let units = &p.images.iter().find(|(q, _)| *q == s.queue).unwrap().1;
                s.queue < 144
                    && matches!(
                        decode_text(&isa, units[s.unit], units.get(s.unit + 1).copied().unwrap_or(0)),
                        Ok(Insn::Write { .. })
                    )
            })
            .collect();
        for w in writes.into_iter().take(8) {
            let q = sites[w].queue;
            let slice = tsp_iss::layout::issuer_of(q as usize).slice as u8;
            // the text of the queue is padded to a whole word (as the blocks are), the arena is one word of the slice
            let mut text = text0.clone();
            let t = text.iter_mut().find(|t| t.queue == q).unwrap();
            while t.units.len() < upw {
                t.units.push(ots_isa::PADDING);
            }
            let arena_at = words - 1;
            let arena = vec![Run { queue: q, start: 0, slice, addr: arena_at, words: 1 }];
            // the host writes the text of the plan into the arena before every step
            let with_text = |prog: &Prog| -> Prog {
                let mut units = prog.images.iter().find(|(x, _)| *x == q).unwrap().1.clone();
                while units.len() < upw {
                    units.push(ots_isa::PADDING);
                }
                let mut out = prog.clone();
                out.words.retain(|(s, a, _)| !(*s == slice && *a == arena_at));
                out.words.push((slice, arena_at, units[..upw].iter().flat_map(|u| u.to_le_bytes()).collect()));
                out
            };
            let c = cache();
            let mut pl = loaded(&p, &c);
            pl.set_param_replay(true);
            pl.set_param_min_gain(0.0);
            let shape_spec = ParamSpec { sites: sites.clone(), text, arena };
            pl.add_param_shape("bucket", shape_spec.clone());
            for k in 0..2 {
                let plan = if k == 0 { values.clone() } else { other_plan(bank_bit, words, &values, &mut rng) };
                fill(&mut pl, &with_text(&patched(&p, &sites, &plan)));
                pl.start_plan("bucket", &plan).unwrap();
                pl.run_to_done(MAX_CYCLES).unwrap();
            }
            if pl.replay_status().shapes[0].state != "ready" {
                continue 'programs;
            }
            // the warm normal run that is the baseline of the measured gain
            fill(&mut pl, &with_text(&patched(&p, &sites, &values)));
            pl.start_plan("bucket", &values).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
            tried += 1;
            // the plan whose Write lands on the arena word
            let mut plan = other_plan(bank_bit, words, &values, &mut rng);
            plan[w] = arena_at as i32;
            let prog = with_text(&patched(&p, &sites, &plan));
            let mut r = Platform::new(&prof, 0).unwrap();
            fill(&mut r, &prog);
            r.start().unwrap();
            r.run_to_done(MAX_CYCLES).unwrap();
            // the Write really lands on the text word: the normal run changed it
            let planned = &prog.words.last().unwrap().2;
            if r.read_sram_word(slice, arena_at).unwrap()[..planned.len()] == planned[..] {
                eprintln!("{n} site {w}: the Write (queue {q}, slice {slice}) left the word as it was");
                continue;
            }
            let reference = outcome(&r);
            fill(&mut pl, &prog);
            pl.start_plan("bucket", &plan).unwrap();
            if change != 0 {
                pl.advance(1).unwrap();
                assert_eq!(pl.phase(), Phase::Running);
                match change {
                    1 => pl.set_param_replay(false),
                    2 => {
                        pl.add_param_shape("bucket", shape_spec.clone());
                        assert_eq!(pl.replay_status().shapes[0].state, "ready", "replacement is deferred");
                    }
                    3 => pl.add_param_shape("another", shape_spec.clone()),
                    _ => unreachable!(),
                }
            }
            pl.run_to_done(MAX_CYCLES).unwrap();
            assert_eq!(outcome(&pl), reference, "{n} site {w}: the run is the normal driver's");
            let st = pl.replay_status();
            if st.refused > 0 {
                eprintln!("{n} site {w}: refused");
                continue; // the gate refused the plan (the bank rule): it never started as a served run
            }
            if change == 2 {
                assert_eq!(st.shapes[0].state, "first", "deferred replacement applies after completion");
                assert_eq!((st.served, st.fallback), (0, 0), "the new registration has fresh counters");
            } else {
                assert_eq!(
                    (st.served, st.fallback),
                    (1, 1),
                    "{n} site {w}: the served run was repeated on the normal driver"
                );
            }
            assert_eq!(pl.replay_counters().param_fallback, 1);
            hit += 1;
            if hit == 2 {
                break 'programs;
            }
        }
    }
    eprintln!("shapes tried {tried}, text writes caught {hit}");
    assert!(hit >= 1, "no program of the corpus has a Write that can be aimed at a word of its text");
}

/// One step of a ready shape with another plan of the same banks.
fn step(pl: &mut Platform, p: &Prog, sites: &[Site], values: &[i32], bank_bit: u32, words: u32, rng: &mut Rng) {
    let plan = other_plan(bank_bit, words, values, rng);
    fill(pl, &patched(p, sites, &plan));
    pl.start_plan("bucket", &plan).unwrap();
    pl.run_to_done(MAX_CYCLES).unwrap();
}

#[test]
fn the_served_steps_run_in_the_per_tick_check_one_in_sixty_four_by_default() {
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let (mut pl, p, sites, values, _t, bank_bit, words) = ready(&mut rng, 0.0);
    if std::env::var_os("TSP_ISS_REPLAY_CHECK").is_none() {
        assert_eq!(pl.param_check(), CheckMode::Every(64));
    } else {
        pl.set_param_check_every(64);
    }
    for _ in 0..200 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    let st = pl.replay_status();
    assert!(st.served > 150, "{} served", st.served);
    assert_eq!(st.fallback, 0);
    // the first served step and then every 64th
    assert_eq!(st.checked, st.served.div_ceil(64), "{} served", st.served);
    assert_eq!(st.shapes[0].checked, st.checked);
    // the switches: all, none
    pl.set_param_check(true);
    let before = pl.replay_status();
    for _ in 0..5 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    let all = pl.replay_status();
    assert_eq!(all.checked - before.checked, all.served - before.served);
    pl.set_param_check(false);
    for _ in 0..70 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    assert_eq!(pl.replay_status().checked, all.checked, "none");
}

/// Timings that the clock of the gain gives: a normal run and a replayed one, in nanoseconds; the replay's are read from `fast`.
fn clock(normal: u64, replay: Arc<AtomicU64>) -> Box<dyn FnMut(bool) -> u64 + Send> {
    Box::new(move |served| if served { replay.load(Ordering::SeqCst) } else { normal })
}

#[test]
fn a_load_spike_during_the_probes_does_not_serve_a_shape_by_the_normal_driver() {
    let mut rng = Rng(0x0bad_cafe_f00d_0001);
    let (mut pl, p, sites, values, _t, bank_bit, words) = ready(&mut rng, 1.2);
    pl.set_param_check(false);
    // the baseline is the injected normal time, 1000 ns (a warm normal run is sampled one step in 32)
    let replay = Arc::new(AtomicU64::new(100));
    pl.set_param_timing(Some(clock(1000, replay.clone())));
    for _ in 0..34 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    let served = pl.replay_status().served;
    // the next five served steps are the probes of one decision, and they run in a load spike: gain 0.2
    replay.store(5000, Ordering::SeqCst);
    let mut probes = 0;
    while probes < 5 {
        let before = pl.replay_status().served;
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
        probes += pl.replay_status().served - before;
    }
    replay.store(100, Ordering::SeqCst); // the spike is over: gain 10
    for _ in 0..60 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    let st = pl.replay_status();
    assert_eq!(st.shapes[0].state, "ready", "{:?}", st.shapes[0].why);
    assert!(st.served > served + 50, "{} served, {served} before", st.served);
    // a shape that stays slow is served by the normal driver
    replay.store(5000, Ordering::SeqCst);
    for _ in 0..40 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    assert_eq!(pl.replay_status().shapes[0].state, "normal");
}

/// A step that ran in the per-tick check ran the normal driver as well: its time says nothing of the replay.
#[test]
fn a_checked_step_is_not_timed_for_the_gain() {
    let mut rng = Rng(0x0bad_cafe_f00d_0004);
    let (mut pl, p, sites, values, _t, bank_bit, words) = ready(&mut rng, 1.2);
    pl.set_param_check(true);
    pl.set_param_timing(Some(clock(1000, Arc::new(AtomicU64::new(5000)))));
    for _ in 0..60 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    let st = pl.replay_status();
    assert_eq!(st.shapes[0].state, "ready", "{:?}", st.shapes[0].why);
    assert!(st.shapes[0].replay_us.is_none(), "no checked step was timed");
    assert!(st.checked > 40);
}

#[test]
fn the_first_sight_and_the_recording_run_do_not_set_the_baseline_of_the_gain() {
    let mut rng = Rng(0x0bad_cafe_f00d_0002);
    for n in names() {
        let p = load(&n);
        let (sites, values, text, bank_bit) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let words = tsp_iss::Geom::new(&profile(&p)).words_per_slice as u32;
        let c = cache();
        let mut pl = loaded(&p, &c);
        pl.set_param_replay(true);
        pl.set_param_check(false);
        pl.set_param_min_gain(1.2);
        pl.add_param_shape("bucket", spec(&sites, &text));
        // the cold runs (the first sight, the recording) take ten times as long as a warm one, the replay is 10 % faster than a
        // warm normal run: the gain is 1.1, under the minimum. A baseline from the cold runs would say 11.
        let mut normal_calls = 0;
        pl.set_param_timing(Some(Box::new(move |served| {
            if served {
                900
            } else {
                normal_calls += 1;
                if normal_calls <= 2 {
                    10_000
                } else {
                    1000
                }
            }
        })));
        for k in 0..30 {
            let plan = if k == 0 { values.clone() } else { other_plan(bank_bit, words, &values, &mut rng) };
            fill(&mut pl, &patched(&p, &sites, &plan));
            pl.start_plan("bucket", &plan).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
            if k == 2 && pl.replay_status().shapes[0].state != "ready" {
                break;
            }
        }
        let row = &pl.replay_status().shapes[0].clone();
        if row.served == 0 {
            continue; // a program that cannot be parametrised
        }
        assert_eq!(row.state, "normal", "{n}: a gain of 1.1 is under the minimum");
        assert!(
            matches!(row.why, Some(ParamWhy::Small { gain, .. }) if (gain - 1000.0 / 900.0).abs() < 1e-9),
            "{:?}",
            row.why
        );
        return;
    }
    panic!("no program of the corpus gives a shape");
}

#[test]
fn a_shape_served_by_the_normal_driver_for_its_gain_is_measured_again() {
    let mut rng = Rng(0x0bad_cafe_f00d_0003);
    let (mut pl, p, sites, values, _t, bank_bit, words) = ready(&mut rng, 1.2);
    pl.set_param_check(false);
    let replay = Arc::new(AtomicU64::new(5000));
    pl.set_param_timing(Some(clock(1000, replay.clone())));
    // a warm normal run is sampled one step in 32: the baseline becomes the injected one, then ten probes decide
    for _ in 0..100 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    assert_eq!(pl.replay_status().shapes[0].state, "normal");
    let served = pl.replay_status().served;
    replay.store(100, Ordering::SeqCst); // the load is gone
    for _ in 0..400 {
        step(&mut pl, &p, &sites, &values, bank_bit, words, &mut rng);
    }
    let st = pl.replay_status();
    assert_eq!(st.shapes[0].state, "ready", "{:?}", st.shapes[0].why);
    assert!(st.served > served + 50, "{} served, {served} before", st.served);
}

/// A platform on `cache` with the shape registered for the program `p`, driven until the shape is ready (the first step, the recording).
fn ready_on(p: &Prog, cache: &Arc<ReplayCache>, sites: &[Site], values: &[i32], text: &[QueueText]) -> Platform {
    let mut pl = loaded(p, cache);
    pl.set_param_replay(true);
    pl.set_param_min_gain(0.0);
    pl.add_param_shape("bucket", spec(sites, text));
    for _ in 0..2 {
        fill(&mut pl, &patched(p, sites, values));
        pl.start_plan("bucket", values).unwrap();
        pl.run_to_done(MAX_CYCLES).unwrap();
    }
    pl
}

/// One budget for the process: the parametrised schedules of every platform on a cache and the cache's own schedules together stay
/// within it, and the one used least recently gives way.
#[test]
fn two_platforms_on_one_cache_never_hold_more_than_the_budget() {
    let mut rng = Rng(0x7a7a_7a7a_1234_5678);
    for n in names() {
        let p = load(&n);
        let (sites, values, text, bank_bit) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let words = tsp_iss::Geom::new(&profile(&p)).words_per_slice as u32;
        // what one shape of this program holds
        let big = cache();
        let probe = ready_on(&p, &big, &sites, &values, &text);
        if probe.replay_status().shapes[0].state != "ready" {
            continue;
        }
        let one = probe.replay_status().shapes[0].retained_bytes as usize;
        assert_eq!(big.counters().param_bytes as usize, one, "the cache counts what the platform says it holds");
        assert_eq!(big.counters().retained_bytes as usize, one);
        drop(probe);
        assert_eq!(big.counters().retained_bytes, 0, "a platform that goes gives its room back");
        // room for one and a half shapes: two platforms cannot both hold theirs
        let budget = one + one / 2;
        let c = Arc::new(ReplayCache::new(budget, 4));
        let mut a = ready_on(&p, &c, &sites, &values, &text);
        let mut b = ready_on(&p, &c, &sites, &values, &text);
        let k = c.counters();
        assert!(k.retained_bytes <= k.budget_bytes, "{k:?}");
        assert_eq!(a.replay_status().shapes[0].state, "ready");
        let mut served = 0;
        for round in 0..6 {
            // each platform runs a few steps in turn: the one that runs holds its shape, the other one's is given up
            for pl in [&mut a, &mut b] {
                for step in 0..5 {
                    let plan = other_plan(bank_bit, words, &values, &mut rng);
                    let prog = patched(&p, &sites, &plan);
                    fill(pl, &prog);
                    pl.start_plan("bucket", &plan).unwrap();
                    pl.run_to_done(MAX_CYCLES).unwrap();
                    assert_eq!(outcome(pl), normal_step(&prog), "{n} round {round} step {step}");
                    let k = c.counters();
                    assert!(k.retained_bytes <= k.budget_bytes, "{n} round {round} step {step}: {k:?}");
                    assert!(k.param_bytes <= k.retained_bytes && k.param_shapes <= 1, "{n} round {round}: {k:?}");
                }
            }
            served = a.replay_status().served + b.replay_status().served;
        }
        assert!(c.counters().evictions > 0, "the platform that used its shape last kept it, the other was evicted");
        assert!(served > 0, "the shape that is held serves");
        return;
    }
    panic!("no program of the corpus gives a shape");
}

/// The budget bounds the memory the process holds, not only the cache's account of it: a shape that the cache evicts gives its
/// schedule back at once, also when its platform is idle and never steps that shape again.
#[test]
fn an_evicted_shape_frees_its_memory_while_its_platform_is_idle() {
    for n in names() {
        let p = load(&n);
        let (sites, values, text, _) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let big = cache();
        let probe = ready_on(&p, &big, &sites, &values, &text);
        if probe.replay_status().shapes[0].state != "ready" {
            continue;
        }
        let one = probe.replay_status().shapes[0].retained_bytes as isize;
        drop(probe);
        // room for one and a half shapes: the second to record evicts the first
        let c = Arc::new(ReplayCache::new((one + one / 2) as usize, 4));
        let mk = || {
            let mut pl = loaded(&p, &c);
            pl.set_param_replay(true);
            pl.set_param_min_gain(0.0);
            pl.add_param_shape("bucket", spec(&sites, &text));
            pl
        };
        let step = |pl: &mut Platform| {
            fill(pl, &patched(&p, &sites, &values));
            pl.start_plan("bucket", &values).unwrap();
            pl.run_to_done(MAX_CYCLES).unwrap();
        };
        let (mut a, mut b) = (mk(), mk());
        step(&mut a);
        step(&mut b);
        let before = heap();
        step(&mut a); // records
        let with_a = heap();
        assert!(with_a - before >= one / 2, "{n}: the measurement sees a schedule of {one} bytes: {}", with_a - before);
        step(&mut b); // records and evicts a's schedule, a never steps again
        let with_b = heap();
        let held = with_b - before;
        eprintln!("{n}: one shape {one} bytes, heap with a {}, with a and b {held}", with_a - before);
        assert_eq!(a.replay_status().shapes[0].retained_bytes, 0, "{n}: the evicted shape reports nothing held");
        assert_eq!(b.replay_status().shapes[0].state, "ready");
        let budget = c.counters().budget_bytes as isize;
        assert!(held <= budget, "{n}: the platforms hold {held} bytes of schedules, the budget is {budget}");
        return;
    }
    panic!("no program of the corpus gives a shape");
}

/// A stored schedule of about `bytes` bytes under `key`.
fn stored(key: u8, bytes: usize) -> tsp_iss::driver::replay::Schedule {
    let mut s = tsp_iss::driver::replay::Schedule::default();
    s.key = [key; 32];
    s.cycles = vec![0; (bytes - s.bytes()) / 8];
    s
}

/// The shapes and the ordinary schedules of one cache are one LRU: whichever was used less recently goes first, a stored schedule
/// that a run has hit since counts as used.
#[test]
fn a_stored_schedule_and_a_parametrised_one_are_evicted_in_the_order_of_use() {
    use tsp_iss::driver::replay::{Lookup, RecordOn};
    for n in names() {
        let p = load(&n);
        let (sites, values, text, _) = sites_of(&p);
        if sites.is_empty() {
            continue;
        }
        let big = cache();
        let probe = ready_on(&p, &big, &sites, &values, &text);
        if probe.replay_status().shapes[0].state != "ready" {
            continue;
        }
        let one = probe.replay_status().shapes[0].retained_bytes as usize;
        drop(probe);
        // the stored schedule is older than the shape: it goes
        let c = Arc::new(ReplayCache::new(3 * one, 8));
        c.insert(stored(1, one));
        let pl = ready_on(&p, &c, &sites, &values, &text);
        c.insert(stored(2, one + one / 2));
        assert!(!c.contains(&[1; 32]) && c.contains(&[2; 32]), "{n}: the older stored schedule goes");
        assert_eq!(c.counters().param_shapes, 1, "{n}: the shape stays");
        assert!(c.counters().retained_bytes <= c.counters().budget_bytes);
        drop(pl);
        // the stored schedule was hit after the shape was used: the shape goes
        let c = Arc::new(ReplayCache::new(3 * one, 8));
        c.insert(stored(1, one));
        let pl = ready_on(&p, &c, &sites, &values, &text);
        assert!(matches!(c.begin(&[1; 32], RecordOn::Second), Lookup::Hit(_)));
        c.insert(stored(2, one + one / 2));
        assert!(c.contains(&[1; 32]) && c.contains(&[2; 32]), "{n}: both stored schedules stay");
        assert_eq!(c.counters().param_shapes, 0, "{n}: the shape was used least recently");
        assert!(c.counters().retained_bytes <= c.counters().budget_bytes);
        drop(pl);
        assert_eq!(c.counters().param_bytes, 0);
        return;
    }
    panic!("no program of the corpus gives a shape");
}
