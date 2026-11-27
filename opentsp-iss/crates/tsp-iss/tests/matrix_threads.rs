//! `tsp-iss run --matrix-threads N|auto`: the width of the pool that splits a large matrix product. The default
//! stays 1; `auto` is min(cores / jobs, 6) for a program with an ABC the pool would split, else 1; the results do not depend on
//! the width.
mod common;
use common::*;
use ots_format::text::parse_images;
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::geom::Geom;
use tsp_iss::native::threads::{abc_counts, auto_width, MatrixThreads, AUTO_MAX};

fn programs_dir() -> std::path::PathBuf {
    root().join("opentsp-iss/crates/tsp-iss/benches/programs")
}

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// The `Q` lines of a benchmark program (`tsp-iss run` takes no `S` lines of a text program: the data is zero).
fn q_only(name: &str) -> String {
    std::fs::read_to_string(programs_dir().join(format!("{name}.txt")))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("Q "))
        .map(|l| format!("{l}\n"))
        .collect()
}

fn stats_of(args: &[&str], prog: &str) -> Value {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("p.txt");
    std::fs::write(&p, prog).unwrap();
    let f = d.path().join("s.json");
    let st = format!("--stats={}", f.display());
    let mut a = vec!["run", "--profile", "ref", &st];
    a.extend_from_slice(args);
    a.push(p.to_str().unwrap());
    let o = run_iss(&a, "");
    assert_eq!(o.status.code(), Some(0), "{args:?}: {}", text(&o.stderr));
    serde_json::from_slice(&std::fs::read(&f).unwrap()).unwrap()
}

#[test]
fn the_option_takes_a_width_or_auto() {
    assert_eq!(MatrixThreads::parse("1"), Some(MatrixThreads::Width(1)));
    assert_eq!(MatrixThreads::parse("16"), Some(MatrixThreads::Width(16)));
    assert_eq!(MatrixThreads::parse("auto"), Some(MatrixThreads::Auto));
    for bad in ["", "0", "17", "-1", "four", "Auto", "2.5"] {
        assert_eq!(MatrixThreads::parse(bad), None, "{bad:?}");
    }
}

#[test]
fn a_bad_width_is_a_usage_error_with_a_message() {
    for bad in ["0", "17", "x", "auto2"] {
        let o = run_iss(&["run", "--profile", "ref", "--matrix-threads", bad, "/nonexistent.txt"], "");
        assert_eq!(o.status.code(), Some(2), "{bad}");
        assert!(o.stdout.is_empty());
        let e = text(&o.stderr);
        assert!(e.contains("--matrix-threads") && e.contains("a number from 1 to 16, or auto"), "{bad}: {e}");
    }
    let o = run_iss(&["run", "--profile", "ref", "--matrix-threads"], "");
    assert_eq!(o.status.code(), Some(2), "missing value");
}

#[test]
fn the_default_width_is_one_and_a_number_sets_it() {
    if std::env::var_os("TSP_ISS_THREADS").is_some() {
        return; // the environment of this process sets the default
    }
    let prog = q_only("b1");
    assert_eq!(stats_of(&[], &prog)["matrix_threads"], 1);
    assert_eq!(stats_of(&["--matrix-threads", "1"], &prog)["matrix_threads"], 1);
    assert_eq!(stats_of(&["--matrix-threads", "3"], &prog)["matrix_threads"], 3.min(cores()));
}

#[test]
fn auto_is_cores_over_jobs_up_to_six_for_a_dense_program_and_one_for_the_rest() {
    let dense = q_only("dense_f16");
    let want = cores().clamp(1, AUTO_MAX);
    // an i8 ABC is not split: the kernel is faster than the fork and join (measured; see `threads.rs`)
    assert_eq!(stats_of(&["--matrix-threads", "auto", "--jobs", "1"], &q_only("dense_i8"))["matrix_threads"], 1);
    let st = stats_of(&["--matrix-threads", "auto", "--jobs", "1"], &dense);
    assert_eq!(st["matrix_threads"], want);
    // an adaptive pool says what it decided; a fixed width above 1 is adaptive too, a pool of one thread has nothing to decide
    assert!(["idle", "measuring", "on", "off"].contains(&st["matrix_split"].as_str().unwrap()), "{st}");
    let three = stats_of(&["--matrix-threads", "3"], &dense)["matrix_split"].clone();
    assert!(["measuring", "on", "off"].contains(&three.as_str().unwrap()), "width 3 is adaptive: {three}");
    assert_eq!(stats_of(&["--matrix-threads", "1"], &dense)["matrix_split"], "fixed");
    // jobs is the threads the run's chips use already
    if cores() >= 4 {
        let two = (cores() / 2).clamp(1, AUTO_MAX);
        assert_eq!(
            stats_of(&["--matrix-threads", "auto", "--jobs", "2", "--chips", "2"], &dense)["matrix_threads"],
            two
        );
    }
    // no ABC in B1 (reads) or in the elementwise program: one thread
    for name in ["b1", "vxm_valid", "transc"] {
        assert_eq!(
            stats_of(&["--matrix-threads", "auto", "--jobs", "1"], &q_only(name))["matrix_threads"],
            1,
            "{name}"
        );
    }
}

/// No width above 1 slows a program: the pool of a fixed width measures what a split saves like the one of `auto`, and the
/// estimate of an i8 product (about 1.4 us for a whole plane, less than a fork) keeps it from forking at all, so a dense i8
/// program never measures and `--stats` says `idle`, not `measuring`.
#[test]
fn a_fixed_width_cannot_slow_an_i8_program() {
    if std::env::var_os("TSP_ISS_THREADS").is_some() || cores() < 4 {
        return;
    }
    let i8s = q_only("dense_i8");
    for width in ["4", "6"] {
        let st = stats_of(&["--matrix-threads", width], &i8s);
        assert_eq!(
            st["matrix_split"], "idle",
            "width {width}: an i8 product is below the fork threshold, nothing forks, so nothing is measured: {st}"
        );
    }
    // a dense f16 program does fork and the pool decides
    let st = stats_of(&["--matrix-threads", "4"], &q_only("dense_f16"));
    assert!(["on", "off"].contains(&st["matrix_split"].as_str().unwrap()), "{st}");
}

#[test]
fn abc_counts_find_the_matrix_instructions_of_the_images() {
    let profile = Profile::load("ref").unwrap();
    let count = |name: &str| abc_counts(&profile, &parse_images(&q_only(name), name).unwrap());
    let (f, i) = count("dense_f16");
    assert!(f > 0 && i == 0, "dense_f16: {f} f16 and {i} i8");
    let (f, i) = count("dense_i8");
    assert!(f == 0 && i > 0, "dense_i8: {f} f16 and {i} i8");
    assert_eq!(count("b1"), (0, 0));
    assert_eq!(count("vxm_valid"), (0, 0));
}

#[test]
fn auto_width_follows_the_core_count_the_jobs_and_the_cap() {
    let profile = Profile::load("ref").unwrap();
    let geom = Geom::new(&profile);
    let dense = parse_images(&q_only("dense_f16"), "d").unwrap();
    let i8s = parse_images(&q_only("dense_i8"), "i").unwrap();
    assert_eq!(auto_width(&profile, &geom, &[&i8s], 18, 1), 1, "an i8 program stays on one thread");
    let plain = parse_images(&q_only("b1"), "b").unwrap();
    for (cores, jobs, want) in
        [(18, 1, 6), (18, 3, 6), (18, 6, 3), (18, 18, 1), (4, 1, 4), (1, 1, 1), (8, 0, 6), (2, 4, 1)]
    {
        assert_eq!(auto_width(&profile, &geom, &[&dense], cores, jobs), want, "cores {cores}, jobs {jobs}");
    }
    assert_eq!(auto_width(&profile, &geom, &[&plain], 18, 1), 1);
    assert_eq!(auto_width(&profile, &geom, &[&plain, &dense], 18, 1), 6, "one chip of several is enough");
}

/// The width never changes a byte of the output (each output row of a product is independent).
#[test]
fn the_output_does_not_depend_on_the_width() {
    let d = tempfile::tempdir().unwrap();
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let g = std::process::Command::new(python)
        .arg(programs_dir().join("gen_bench.py"))
        .args(["--small", d.path().to_str().unwrap()])
        .output()
        .expect("python3 (source the toolchain env.sh)");
    assert!(g.status.success(), "{}", text(&g.stderr));
    let cfg = config("ref").to_str().unwrap().to_string();
    for name in ["dense_f16", "dense_i8"] {
        let src = std::fs::read_to_string(d.path().join(format!("{name}.txt"))).unwrap();
        let one = run_bin(iss(), &[cfg.as_str(), "--trace"], src.as_bytes());
        assert_eq!(one.status.code(), Some(0), "{name}");
        for w in ["2", "6", "16"] {
            let mut c = std::process::Command::new(iss());
            c.args([cfg.as_str(), "--trace"]).env("TSP_ISS_THREADS", w);
            let o = run_with_env(&mut c, src.as_bytes());
            assert_eq!(text(&o.stdout), text(&one.stdout), "{name}: {w} threads");
        }
    }
}

fn run_with_env(c: &mut std::process::Command, stdin: &[u8]) -> std::process::Output {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn bench_takes_the_same_option() {
    let p = programs_dir().join("vxm_valid.txt");
    let want = run_iss(&["bench", "--program", p.to_str().unwrap()], "");
    assert_eq!(want.status.code(), Some(0));
    let cycles = |o: &std::process::Output| {
        text(&o.stdout).split_whitespace().find(|w| w.starts_with("cycles=")).map(String::from)
    };
    for w in ["1", "4", "auto"] {
        let o = run_iss(&["bench", "--program", p.to_str().unwrap(), "--matrix-threads", w], "");
        assert_eq!(o.status.code(), Some(0), "{w}: {}", text(&o.stderr));
        assert_eq!(cycles(&o), cycles(&want), "{w}");
    }
    for bad in ["0", "17", "x", ""] {
        let o = run_iss(&["bench", "--program", p.to_str().unwrap(), "--matrix-threads", bad], "");
        assert_eq!(o.status.code(), Some(2), "{bad:?}");
        assert!(text(&o.stderr).contains("usage: tsp-iss bench"), "{bad:?}: {}", text(&o.stderr));
    }
}
