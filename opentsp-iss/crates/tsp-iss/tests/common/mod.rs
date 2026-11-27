//! Shared helpers of the integration tests: the binary, the specification tree, and the
//! reference harness `tsp_sim` (built on demand with the specification toolchain).
// Every test crate includes this module and uses a subset of it.
#![allow(dead_code)]
pub mod prog;

use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::OnceLock,
};

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..").canonicalize().unwrap()
}

pub fn spec() -> PathBuf {
    root().join("opentsp-arch-spec")
}

pub fn config(name: &str) -> PathBuf {
    spec().join("config").join(format!("{name}.json"))
}

pub fn iss() -> &'static str {
    env!("CARGO_BIN_EXE_tsp-iss")
}

/// The longest a spawned program may run (`$ISS_TEST_TIMEOUT` seconds, default 600): a hung process
/// fails its test instead of the whole run.
fn timeout() -> std::time::Duration {
    let s = std::env::var("ISS_TEST_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(600);
    std::time::Duration::from_secs(s)
}

pub fn run_bin(bin: &str, args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {bin}: {e}"));
    let mut si = child.stdin.take().unwrap();
    let data = stdin.to_vec();
    let w = std::thread::spawn(move || {
        let _ = si.write_all(&data);
    });
    let drain = |mut r: Box<dyn std::io::Read + Send>| {
        std::thread::spawn(move || {
            let mut v = Vec::new();
            let _ = r.read_to_end(&mut v);
            v
        })
    };
    let out = drain(Box::new(child.stdout.take().unwrap()));
    let err = drain(Box::new(child.stderr.take().unwrap()));
    let limit = timeout();
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        if started.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{bin} {args:?} ran for more than {} s and was killed", limit.as_secs());
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    };
    w.join().unwrap();
    Output { status, stdout: out.join().unwrap(), stderr: err.join().unwrap() }
}

/// Extra ISS arguments for every `run_iss`: `--tick-every-cycle` when `$ISS_TICK_EVERY_CYCLE` is set,
/// so the differential tests compare the every-cycle run with `tsp_sim` (`just iss-tick`).
pub fn iss_extra() -> &'static [&'static str] {
    if std::env::var_os("ISS_TICK_EVERY_CYCLE").is_some() {
        &["--tick-every-cycle"]
    } else {
        &[]
    }
}

pub fn run_iss(args: &[&str], stdin: &str) -> Output {
    let all: Vec<&str> = args.iter().chain(iss_extra()).copied().collect();
    run_bin(iss(), &all, stdin.as_bytes())
}

pub fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// The reference harness: `$TSP_SIM`, else `target/spec/tsp_sim`, built with
/// `opentsp-arch-spec/tools/build_sim.sh` after `source ~/.local/opt/opentsp-toolchain/env.sh`.
/// It never skips: if it cannot be had, the test fails loudly.
pub fn tsp_sim() -> &'static Path {
    static SIM: OnceLock<PathBuf> = OnceLock::new();
    SIM.get_or_init(|| {
        if let Ok(p) = std::env::var("TSP_SIM") {
            let p = PathBuf::from(p);
            assert!(p.exists(), "TSP_SIM={} does not exist", p.display());
            return p;
        }
        let bin = root().join("target/spec/tsp_sim");
        let setup =
            r#"source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh" && exec bash "$1" "$2""#;
        let st = Command::new("bash")
            .args(["-c", setup, "tsp-sim-test-setup"])
            .arg(root().join("tools/ensure_sim.sh"))
            .arg(&bin)
            .status()
            .expect("bash");
        assert!(
            st.success() && bin.is_file(),
            "tsp_sim cannot be built from the configured toolchain; set TSP_SIM to an explicitly built reference binary"
        );
        bin
    })
}

pub fn run_sim(args: &[&str], stdin: &str) -> Output {
    run_bin(tsp_sim().to_str().unwrap(), args, stdin.as_bytes())
}

/// Run `f` on every item, on 8 threads; the `Err` texts of the items that failed, in item order.
pub fn parallel<T: Sync>(items: &[T], f: impl Fn(&T) -> Result<(), String> + Sync) -> Vec<String> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(item) = items.get(i) else { break };
                if let Err(e) = f(item) {
                    bad.lock().unwrap().push((i, format!("item {i}: {e}")));
                }
            });
        }
    });
    let mut bad = bad.into_inner().unwrap();
    bad.sort();
    bad.into_iter().map(|b| b.1).collect()
}

/// The reference's stderr with its program name replaced by ours.
pub fn norm_stderr(s: &str) -> String {
    s.replace("tsp_sim:", "tsp-iss:")
}
