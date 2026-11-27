//! A panic in a worker thread of `tsp-iss vectors` fails that vector and the run goes on; the process never hangs.
//!
//! The panic hook of the binary writes a line on standard error. The process used to hold the lock of standard
//! error for the whole run, so a worker that panicked waited for the lock for ever while the main thread waited
//! for the worker's result. Every run here has a bound, and a run past it fails the test.
mod common;
use common::*;
use serde_json::{json, Value};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The longest a run of these tests may take; a hang fails the test.
const BOUND: Duration = Duration::from_secs(120);
/// Vectors of the golden file that a run takes.
const TAKEN: usize = 6;

fn golden() -> Value {
    let path = spec().join("conformance/model/ref.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn name(doc: &Value, i: usize) -> String {
    doc["vectors"][i]["name"].as_str().unwrap().to_string()
}

/// `tsp-iss vectors` on `doc` with the first `TAKEN` vectors, with `env` set: `(exit code, stdout, stderr)`; fails the
/// test when the process runs past [`BOUND`] (it is killed first).
fn vectors(doc: &Value, jobs: &str, env: &[(&str, String)]) -> (Option<i32>, String, String) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("v.json");
    std::fs::write(&file, serde_json::to_vec(doc).unwrap()).unwrap();
    let cfg = config("ref");
    let mut cmd = Command::new(iss());
    cmd.args(["vectors", file.to_str().unwrap(), "--config", cfg.to_str().unwrap(), "--jobs", jobs]);
    for i in 0..TAKEN {
        cmd.args(["--only", &name(doc, i)]);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
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
            break s;
        }
        if started.elapsed() > BOUND {
            let _ = child.kill();
            let _ = child.wait();
            panic!("tsp-iss vectors ran for more than {} s and was killed: it hangs", BOUND.as_secs());
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    (status.code(), text(&out.join().unwrap()), text(&err.join().unwrap()))
}

fn fails(stdout: &str) -> Vec<&str> {
    stdout.lines().filter(|l| l.starts_with("FAIL ")).collect()
}

#[test]
fn a_panic_in_one_vector_fails_that_vector_and_the_run_ends() {
    let doc = golden();
    let victim = name(&doc, 2);
    for jobs in ["1", "3"] {
        let (code, out, err) = vectors(&doc, jobs, &[("TSP_ISS_VECTORS_PANIC", victim.clone())]);
        assert_eq!(code, Some(1), "jobs {jobs}: {out}{err}");
        let f = fails(&out);
        assert_eq!(f.len(), 1, "jobs {jobs}: {out}");
        assert!(f[0].starts_with(&format!("FAIL {victim}: internal error: ")), "{}", f[0]);
        assert!(f[0].contains("injected panic"), "{}", f[0]);
        assert_eq!(out.lines().last().unwrap(), format!("{TAKEN} vectors: {} pass, 1 FAIL", TAKEN - 1), "{out}");
        // the hook still says where the panic came from
        assert!(err.contains("tsp-iss: internal error: injected panic"), "jobs {jobs}: {err}");
    }
}

#[test]
fn a_vector_that_is_not_the_named_one_is_unaffected() {
    let doc = golden();
    let (code, out, err) = vectors(&doc, "2", &[("TSP_ISS_VECTORS_PANIC", "no-such-vector".to_string())]);
    assert_eq!((code, err.as_str()), (Some(0), ""), "{out}");
    assert_eq!(out.lines().last().unwrap(), format!("{TAKEN} vectors: {TAKEN} pass, 0 FAIL"));
}

#[test]
fn an_expected_event_with_wide_characters_is_reported_not_a_panic() {
    // the report of a difference cuts the two events at 60 bytes; a cut inside a character used to panic
    let mut doc = golden();
    let wide = format!("srf 0 E:0 gap 0 x{}", "é".repeat(30));
    doc["vectors"][1]["expect"]["events"][0] = json!(wide);
    let (code, out, err) = vectors(&doc, "1", &[]);
    assert_eq!(code, Some(1), "{out}{err}");
    let f = fails(&out);
    assert_eq!(f.len(), 1, "{out}");
    assert!(
        f[0].starts_with(&format!("FAIL {}: events: ", name(&doc, 1))) && !f[0].contains("internal error"),
        "{}",
        f[0]
    );
    assert!(!err.contains("internal error"), "{err}");
}
