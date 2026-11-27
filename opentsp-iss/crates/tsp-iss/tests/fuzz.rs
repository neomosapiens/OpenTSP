//! The fuzzer of the ISS: random text images on random queues of every unit (catalogue entries
//! with random operands, raw words, padding, truncated parcels, SRAM words, `--chips`, link
//! options, small `--max-cycles`), run on random profiles. Every program must
//! - end with exit 0 or 3 (never exit 4, which is a panic or a broken invariant: a decoder or
//!   legality rule the ISS does not hold shows here, since the units of every kind reach every
//!   queue);
//! - give the same exit code, stdout and stderr when run twice (determinism);
//! - give the same exit code and stdout ticking every cycle (`--tick-every-cycle`);
//! - give `tsp_sim`'s exit code and stdout (stderr only for one chip: the chips of the reference
//!   run a window in parallel, so the order of their messages is a race). A reference run that
//!   exceeds `$ISS_FUZZ_SIM_TIMEOUT` seconds (default 60) is a timeout: it is counted and printed,
//!   and the test fails when more than 10 % of the programs time out (the gate never skips silently).
//!
//! - on several chips, give the same exit code, stdout and stderr with 1 and with 3 chip threads
//!   (`--jobs`; the default runs on every core).
//!
//! Bounded and seeded: `$ISS_FUZZ_N` programs (default 40) from seed `$ISS_FUZZ_SEED` (default 1);
//! `$ISS_FUZZ_SECONDS` = S instead runs rounds of programs, the seed advancing, until S seconds
//! have passed (`just iss-fuzz S`; the time is checked between programs, not only between rounds). A
//! failing program is written to `$ISS_FUZZ_OUT` (default the temporary directory) as
//! `iss-fuzz-<seed>-<round>-<index>.txt` (the input) next to `.args` (its arguments, one per line, the
//! profile first), and shown in the failure; a file that cannot be written fails the test too.
mod common;
use common::prog::{cfg_path, Rng};
use common::*;
use ots_isa::{encode, EntryId, Insn, Isa, Syntax, MAX_VARS};
use ots_profile::Profile;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

const PROFILES: [&str; 9] =
    ["ref", "mini", "ref_slow", "link_tight", "mid_min", "mini_small", "ref_small", "fetch_alt", "c2c_tx_delayed"];

struct Env {
    name: &'static str,
    isa: Isa,
    words: u32,
    slices: Vec<u16>,
    vlanes: usize,
}

fn env(name: &'static str) -> Env {
    let path = cfg_path(name);
    let p = if matches!(name, "ref" | "mini") {
        Profile::load(name).unwrap()
    } else {
        Profile::load_file(name, &path).unwrap()
    };
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let per = v["geometry"]["mem_slices_per_hemisphere"].as_u64().unwrap() as u16;
    Env {
        name,
        isa: Isa::new(&p),
        words: v["sram"]["words_per_slice"].as_u64().unwrap() as u32,
        slices: (0..88u16).filter(|s| s % 44 < per).collect(),
        vlanes: 16 * v["geometry"]["superlanes"].as_u64().unwrap() as usize,
    }
}

/// The units of an instruction with random operand values (the last of 20 tries is kept unchecked
/// unless `legal`: then the first that the profile's decoder accepts).
fn insn_units(r: &mut Rng, e: &Env, legal: bool) -> Vec<u32> {
    let id = r.pick(&EntryId::ALL);
    let en = id.entry();
    let mut f = [0u32; MAX_VARS];
    for _ in 0..20 {
        for (i, v) in en.vars.iter().enumerate() {
            let k = v.kind();
            let full = (1u64 << v.nbits.min(32)) - 1;
            f[i] = match (legal, v.legal) {
                (true, Some(l)) => r.pick(l),
                (true, None) if k.syntax == Syntax::Stream => r.below(64) as u32 / k.group,
                (true, None) => r.below(full.min(48) + 1) as u32,
                (false, _) if r.chance(50) => r.below(full.min(64) + 1) as u32,
                _ => r.below(full + 1) as u32,
            };
        }
        if !legal || e.isa.fields_legal(en, &f) {
            break;
        }
    }
    let w = en.encode_fields(&f[..en.vars.len()]);
    let more = if r.chance(15) { 0x8000_0000 } else { 0 };
    if en.width == 64 {
        vec![((w >> 32) as u32) | more, w as u32]
    } else {
        vec![(w as u32) | more]
    }
}

fn units(r: &mut Rng, e: &Env) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..1 + r.below(24) {
        match r.below(20) {
            0..=8 => out.extend(insn_units(r, e, false)),
            9..=12 => out.extend(insn_units(r, e, true)),
            13 => out.push(r.next() as u32),
            14 => out.push(0xffff_ffff),
            15..=16 => out.extend(vec![encode(&Insn::NOP { n: r.below(60) as u32 }) as u32]),
            17 => out.push(encode(&Insn::Repeat { n: 1 + r.below(4) as u32, d: r.below(4) as u32 }) as u32),
            18 => out.push(encode(&if r.chance(50) { Insn::Sync } else { Insn::Notify }) as u32),
            _ => out.push(encode(&Insn::Deskew) as u32),
        }
    }
    if r.chance(10) {
        out.pop();
    }
    out
}

struct Case {
    args: Vec<String>,
    text: String,
    name: &'static str,
    /// The chips start at a cycle at or above 2^62 (`H start`): `--tick-every-cycle` would tick every idle cycle from
    /// cycle 0, so that comparison is left out.
    far_start: bool,
}

/// Start cycles over the whole unsigned range: at and above 2^62 and 2^63, and the last 2^27 cycles below 2^64 (a run of
/// at most 15 million cycles from there stays below 2^64, where the model and the ISS agree).
const FAR_BASES: [u64; 6] =
    [1 << 62, (1 << 63) - (1 << 20), 1 << 63, (1 << 63) + (1 << 62), u64::MAX - (1 << 27), u64::MAX - (1 << 26)];

fn case(r: &mut Rng, envs: &[Env]) -> Case {
    let e = &envs[r.below(envs.len() as u64) as usize];
    let chips = r.pick(&[1u32, 1, 1, 1, 2, 3, 8]);
    let mut args: Vec<String> = vec![cfg_path(e.name).to_str().unwrap().into(), "--trace".into()];
    if chips > 1 {
        args.extend(["--chips".into(), chips.to_string()]);
    }
    match r.below(8) {
        0 => args.extend(["--link-latency-seed".into(), (1 + r.below(30)).to_string()]),
        1 => args.extend(["--link-jitter-seed".into(), (1 + r.below(30)).to_string()]),
        _ => {}
    }
    // every run is bounded: an idle tick costs microseconds, and one NOP covers 65535 cycles
    let far_start = r.chance(12);
    let start = if far_start { r.pick(&FAR_BASES) + r.below(1 << 20) } else { 0 };
    if far_start {
        // the limit is a cycle above the start: the whole range, or just past a short run
        let max = if r.chance(50) { u64::MAX } else { start + 6000 };
        args.extend(["--max-cycles".into(), max.to_string()]);
    } else {
        args.extend(["--max-cycles".into(), r.pick(&["50", "300", "2500", "6000", "6000", "6000"]).into()]);
    }
    let mut text = String::new();
    for c in 0..chips {
        if chips > 1 {
            writeln!(text, "C {c:x}").unwrap();
        }
        if far_start {
            writeln!(text, "H start {start:x}").unwrap();
        }
        for _ in 0..1 + r.below(10) {
            let q = if r.chance(5) { 0xffff } else { r.below(144) as u16 };
            // a MEM queue of an absent slice takes text too: every instruction there is QUE-002
            write!(text, "Q {q:x}").unwrap();
            for u in units(r, e) {
                write!(text, " {u:x}").unwrap();
            }
            text.push('\n');
        }
        for _ in 0..r.below(4) {
            let (s, a) = (r.pick(&e.slices), r.below(e.words as u64));
            let mut l = format!("S {s:x} {a:x}");
            for _ in 0..e.vlanes {
                write!(l, " {:x}", r.below(256)).unwrap();
            }
            writeln!(text, "{l}").unwrap();
        }
        for _ in 0..r.below(3) {
            writeln!(text, "D {:x} {:x}", r.pick(&e.slices), r.below(e.words as u64)).unwrap();
        }
    }
    Case { args, text, name: e.name, far_start }
}

/// The exit-code slot that stands for a `tsp_sim` run that hit the timeout (not a real exit code).
const TIMEOUT: i32 = -9;

/// `run_sim` with a limit (`$ISS_FUZZ_SIM_TIMEOUT` seconds, default 60): `None` when the reference did not finish.
fn run_sim_limited(args: &[&str], input: &str) -> Option<std::process::Output> {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};
    let limit = Duration::from_secs(env_num("ISS_FUZZ_SIM_TIMEOUT").unwrap_or(60));
    let mut child = Command::new(tsp_sim())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("cannot run tsp_sim");
    let mut si = child.stdin.take().unwrap();
    let data = input.as_bytes().to_vec();
    let w = std::thread::spawn(move || {
        let _ = si.write_all(&data);
    });
    let drain = |mut r: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut v = Vec::new();
            let _ = r.read_to_end(&mut v);
            v
        })
    };
    let out = drain(Box::new(child.stdout.take().unwrap()));
    let err = drain(Box::new(child.stderr.take().unwrap()));
    let started = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break Some(s);
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    let _ = w.join();
    let (out, err) = (out.join().unwrap(), err.join().unwrap());
    status.map(|status| std::process::Output { status, stdout: out, stderr: err })
}

fn run_case(c: &Case) -> Result<i32, String> {
    let args: Vec<&str> = c.args.iter().map(String::as_str).collect();
    let multi = c.args.iter().any(|a| a == "--chips");
    let first = run_iss(&args, &c.text);
    let code = first.status.code().ok_or("killed by a signal")?;
    if code != 0 && code != 3 {
        return Err(format!("exit {code}: {}", text(&first.stderr)));
    }
    let again = run_iss(&args, &c.text);
    if (again.status.code(), &again.stdout, &again.stderr) != (first.status.code(), &first.stdout, &first.stderr) {
        return Err("two runs differ".into());
    }
    if multi {
        for jobs in ["1", "3"] {
            let mut j = args.clone();
            j.extend(["--jobs", jobs]);
            let o = run_bin(iss(), &j, c.text.as_bytes());
            if (o.status.code(), &o.stdout, &o.stderr) != (first.status.code(), &first.stdout, &first.stderr) {
                return Err(format!(
                    "--jobs {jobs} differs from the default (exit {:?}):\n{}",
                    o.status.code(),
                    text(&o.stderr)
                ));
            }
        }
    }
    if !c.far_start {
        let mut every = args.clone();
        every.push("--tick-every-cycle");
        let ticked = run_bin(iss(), &every, c.text.as_bytes());
        if ticked.status.code() != first.status.code() || ticked.stdout != first.stdout {
            return Err(format!(
                "ticking every cycle differs (exit {:?}):\n{}",
                ticked.status.code(),
                text(&ticked.stdout)
            ));
        }
    }
    let Some(sim) = run_sim_limited(&args, &c.text) else {
        return Ok(TIMEOUT);
    };
    let err_ok = multi || norm_stderr(&text(&sim.stderr)) == text(&first.stderr);
    if sim.status.code() != first.status.code() || sim.stdout != first.stdout || !err_ok {
        return Err(format!(
            "differs from tsp_sim (exit {:?}, ours {code}):\n{}\n{}",
            sim.status.code(),
            text(&sim.stdout),
            text(&sim.stderr)
        ));
    }
    Ok(code)
}

fn failure_dir() -> std::path::PathBuf {
    std::env::var_os("ISS_FUZZ_OUT").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir)
}

/// Write a failing case to `dir`: its input and, next to it, its arguments (one per line). The
/// text says where, or that a write failed.
fn save_failure(dir: &std::path::Path, (seed, round, i): (u64, u64, usize), c: &Case) -> String {
    let path = dir.join(format!("iss-fuzz-{seed}-{round}-{i}.txt"));
    let saved = std::fs::write(&path, &c.text)
        .and_then(|()| std::fs::write(path.with_extension("args"), c.args.join("\n") + "\n"));
    match saved {
        Ok(()) => format!("saved to {} (arguments in the .args file next to it)", path.display()),
        Err(e) => format!("NOT saved ({}: {e})", path.display()),
    }
}

fn env_num(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

#[test]
fn random_text_images_never_panic_and_agree() {
    let envs: Vec<Env> = PROFILES.iter().map(|n| env(n)).collect();
    let seed = env_num("ISS_FUZZ_SEED").unwrap_or(1);
    let deadline = env_num("ISS_FUZZ_SECONDS").map(|s| Instant::now() + Duration::from_secs(s));
    let (mut total, mut exits, mut round) = (0usize, BTreeMap::<i32, usize>::new(), 0u64);
    let started = Instant::now();
    loop {
        let n = if deadline.is_some() { 32 } else { env_num("ISS_FUZZ_N").unwrap_or(40) as usize };
        let mut r = Rng::new(seed * 1_000_003 + round * 7919);
        let cases: Vec<Case> = (0..n).map(|_| case(&mut r, &envs)).collect();
        let results = std::sync::Mutex::new(Vec::new());
        let indexed: Vec<(usize, &Case)> = cases.iter().enumerate().collect();
        let bad = parallel(&indexed, |&(i, c)| {
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(()); // out of time: the case is not run, nor counted
            }
            match run_case(c) {
                Ok(code) => {
                    results.lock().unwrap().push(code);
                    Ok(())
                }
                Err(why) => {
                    let note = save_failure(&failure_dir(), (seed, round, i), c);
                    Err(format!("{} {:?} ({note}): {why}\n{}", c.name, c.args, c.text))
                }
            }
        });
        assert!(bad.is_empty(), "{} of {n} programs fail (round {round}); first:\n{}", bad.len(), bad[0]);
        let ran = results.into_inner().unwrap();
        total += ran.len();
        for code in ran {
            *exits.entry(code).or_default() += 1;
        }
        round += 1;
        if deadline.is_none_or(|d| Instant::now() >= d) {
            break;
        }
    }
    let timeouts = exits.remove(&TIMEOUT).unwrap_or(0);
    eprintln!(
        "fuzz: {total} programs in {:.0} s (seed {seed}, {round} rounds), exit codes {exits:?}, {timeouts} tsp_sim timeouts",
        started.elapsed().as_secs_f64()
    );
    if let Err(e) = timeouts_within_limit(timeouts, total) {
        panic!("{e}");
    }
    assert!(exits.get(&0).copied().unwrap_or(0) * 4 >= total - timeouts, "too few runs end normally: {exits:?}");
}

/// The share of programs on which `tsp_sim` may time out before the comparison is said not to cover them: 1 %, as in the Python
/// differentials (no timeout in 2,496 programs was measured).
const TIMEOUT_LIMIT_PERCENT: usize = 1;

fn timeouts_within_limit(timeouts: usize, total: usize) -> Result<(), String> {
    if timeouts * 100 <= total * TIMEOUT_LIMIT_PERCENT {
        Ok(())
    } else {
        Err(format!(
            "{timeouts} of {total} programs timed out on tsp_sim (limit {TIMEOUT_LIMIT_PERCENT} %): the comparison is not covering them"
        ))
    }
}

#[test]
fn more_than_one_percent_of_tsp_sim_timeouts_fails_the_fuzzer() {
    assert!(timeouts_within_limit(0, 2496).is_ok());
    assert!(timeouts_within_limit(1, 100).is_ok(), "exactly 1 %");
    assert!(timeouts_within_limit(2, 100).is_err());
    assert!(timeouts_within_limit(24, 2496).is_ok());
    assert!(timeouts_within_limit(250, 2496).is_err(), "10 % was the old allowance");
}

#[test]
fn a_failing_case_is_saved_with_its_arguments_and_a_write_error_is_reported() {
    let c = Case {
        args: vec![
            "/c/ref.json".into(),
            "--trace".into(),
            "--chips".into(),
            "2".into(),
            "--max-cycles".into(),
            "50".into(),
        ],
        text: "C 0\nQ 0 0\n".into(),
        name: "ref",
        far_start: false,
    };
    let dir = tempfile::tempdir().unwrap();
    let note = save_failure(dir.path(), (7, 2, 31), &c);
    assert!(note.starts_with("saved to ") && note.contains("iss-fuzz-7-2-31.txt"), "{note}");
    assert_eq!(std::fs::read_to_string(dir.path().join("iss-fuzz-7-2-31.txt")).unwrap(), c.text);
    let args = std::fs::read_to_string(dir.path().join("iss-fuzz-7-2-31.args")).unwrap();
    assert_eq!(args.lines().collect::<Vec<_>>(), ["/c/ref.json", "--trace", "--chips", "2", "--max-cycles", "50"]);
    let note = save_failure(&dir.path().join("no/such/dir"), (7, 2, 31), &c);
    assert!(note.starts_with("NOT saved ("), "{note}");
}
