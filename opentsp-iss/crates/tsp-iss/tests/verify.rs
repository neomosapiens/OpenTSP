//! `tsp-iss verify`: the object runs without data; a diagnostic of a schedule class
//! exits 1, anything else 0, an input error 2.
mod common;
use common::*;
use std::path::Path;
use tsp_iss::verify::{
    classify, is_schedule_code, judge, read_diagnostics, ChipFound, Class, Diagnostic, Found, Parked, Verdict,
    SCHEDULE_CODES, USAGE,
};

fn data(name: &str) -> String {
    root().join("opentsp-sw-toolchain/crates/ots-asm/tests/data").join(name).to_str().unwrap().to_string()
}

fn verify(args: &[&str]) -> (i32, String, String) {
    let mut a = vec!["verify"];
    a.extend(args);
    let o = run_iss(&a, "");
    (o.status.code().unwrap(), text(&o.stdout), text(&o.stderr))
}

#[test]
fn hazard_object_exits_1_with_one_line_per_diagnostic() {
    let (code, out, err) = verify(&[&data("hazard.elf")]);
    assert_eq!(code, 1, "{err}");
    assert_eq!(
        out,
        "OTS-E-TIM-001 cycle 43 queue 88 schedule\nOTS-E-TIM-001 cycle 48 queue 89 schedule\nOTS-E-TIM-001 cycle 48 queue 44 schedule\n"
    );
    assert_eq!(err, "");
}

#[test]
fn clean_object_exits_0_and_prints_nothing() {
    let (code, out, err) = verify(&[&data("example.elf")]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
}

#[test]
fn config_option_is_checked_like_run() {
    let ok = verify(&[&data("hazard.elf"), "--config", config("ref").to_str().unwrap()]);
    assert_eq!(ok.0, 1, "{}", ok.2);
    let bad = verify(&[&data("hazard.elf"), "--config", "/no/such/config.json"]);
    assert_eq!(bad.0, 2);
    assert!(bad.2.starts_with("tsp-iss: verify: config /no/such/config.json"), "{}", bad.2);
}

#[test]
fn missing_and_truncated_objects_are_input_errors() {
    let (code, out, err) = verify(&["/no/such/object.elf"]);
    assert_eq!((code, out.as_str()), (2, ""));
    assert!(err.starts_with("tsp-iss: verify: /no/such/object.elf: "), "{err}");
    let bytes = std::fs::read(data("hazard.elf")).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let cut = dir.path().join("cut.elf");
    std::fs::write(&cut, &bytes[..bytes.len() - 20]).unwrap();
    let (code, out, err) = verify(&[cut.to_str().unwrap()]);
    assert_eq!((code, out.as_str()), (2, ""));
    assert!(err.starts_with("tsp-iss: verify: ") && err.contains("OTS-E-ASM-"), "{err}");
}

#[test]
fn usage_errors_exit_2() {
    for args in [&[][..], &["a.elf", "b.elf"], &["--bogus", "a.elf"], &["a.elf", "--weights"]] {
        let (code, out, err) = verify(args);
        assert_eq!((code, out.as_str()), (2, ""), "{args:?}");
        assert_eq!(err, format!("tsp-iss: verify: {USAGE}\n"));
        assert!(
            USAGE.contains("--profile")
                && USAGE.contains("--host")
                && USAGE.contains("--allow-parked")
                && USAGE.contains("--max-cycles"),
            "{USAGE}"
        );
    }
}

fn d(cycle: u64, queue: u64, code: &str) -> Diagnostic {
    Diagnostic { cycle, queue, code: code.into(), fetched: false, data: false }
}

fn chip(n: u64, listed: Vec<Diagnostic>) -> ChipFound {
    let schedule = listed.iter().any(|x| classify(x) == Class::Schedule);
    let fetched = listed.iter().filter(|x| x.fetched).count() as u64;
    let data = listed.iter().find(|x| classify(x) == Class::Data && x.code != "OTS-E-IQ-004").map(|x| x.code.clone());
    ChipFound {
        chip: n,
        total: listed.len() as u64,
        listed,
        schedule,
        fetched,
        data,
        parked: Vec::new(),
        released: Vec::new(),
        sent: Vec::new(),
        received: Vec::new(),
    }
}

#[test]
fn diagnostics_json_is_read_for_every_chip_and_classified() {
    let doc = r#"{"chips": [
        {"by_code": {"OTS-E-IQ-004": 1}, "chip": 0,
         "diagnostics": [{"code": "OTS-E-IQ-004", "cycle": 7, "queue": 12, "queue_name": "MEM[12]"}],
         "listed": 1, "total": 1, "truncated": false},
        {"by_code": {"OTS-E-DEC-003": 2, "OTS-E-LNK-002": 1}, "by_code_fetched": {"OTS-E-DEC-003": 2}, "chip": 1,
         "diagnostics": [{"code": "OTS-E-DEC-003", "cycle": 9, "queue": 3, "fetched": true},
                         {"code": "OTS-E-LNK-002", "cycle": 9, "queue": 65535, "queue_name": "CU"}],
         "listed": 2, "total": 3, "truncated": true,
         "parked": [{"queue": 88, "since": 4, "cause": "host-wait"}]}], "schema": 1}"#;
    let f = read_diagnostics(doc).unwrap();
    assert_eq!(f.chips.len(), 2);
    assert_eq!(f.chips[0].listed, vec![d(7, 12, "OTS-E-IQ-004")]);
    assert!(!f.chips[0].schedule);
    assert_eq!(f.chips[1].listed[0], Diagnostic { fetched: true, ..d(9, 3, "OTS-E-DEC-003") });
    assert_eq!(f.chips[1].parked, vec![Parked { queue: 88, since: 4, cause: "host-wait".into() }]);
    assert!(f.chips[1].schedule, "LNK-002 is a schedule code");
    assert_eq!((f.chips[0].fetched, f.chips[1].fetched), (0, 2));
    // the counts per code decide, also when the list is cut; fetched DEC-003 do not count
    let cut = r#"{"chips": [{"by_code": {"OTS-E-TIM-001": 5000000}, "chip": 0, "diagnostics": [], "listed": 0, "total": 5000000, "truncated": true}]}"#;
    assert!(read_diagnostics(cut).unwrap().chips[0].schedule);
    let fetched = r#"{"chips": [{"by_code": {"OTS-E-DEC-003": 4}, "by_code_fetched": {"OTS-E-DEC-003": 4}, "chip": 0, "diagnostics": [], "listed": 0, "total": 4, "truncated": true}]}"#;
    assert!(!read_diagnostics(fetched).unwrap().chips[0].schedule);
    let mixed = r#"{"chips": [{"by_code": {"OTS-E-DEC-003": 5}, "by_code_fetched": {"OTS-E-DEC-003": 4}, "chip": 0, "diagnostics": [], "listed": 0, "total": 5, "truncated": true}]}"#;
    assert!(read_diagnostics(mixed).unwrap().chips[0].schedule, "one DEC-003 is from the object itself");
    // a data code (IQ-004 excepted) in the counts is kept for the verdict, whether the list holds it or not
    let counted = r#"{"chips": [{"by_code": {"OTS-E-SXM-001": 5000000, "OTS-E-IQ-004": 1}, "chip": 0, "diagnostics": [], "listed": 0, "total": 5000001, "truncated": true}]}"#;
    assert_eq!(read_diagnostics(counted).unwrap().chips[0].data.as_deref(), Some("OTS-E-SXM-001"));
    assert_eq!(f.chips[0].data, None, "IQ-004 alone is not a data code that decides");
    assert_eq!(read_diagnostics("not json"), None);
    assert_eq!(read_diagnostics(r#"{"chips": []}"#), None);
}

#[test]
fn the_class_of_a_code_follows_the_table_and_the_origin_of_the_text() {
    for yes in ["OTS-E-IQ-002", "OTS-E-IQ-004", "OTS-E-SXM-001"] {
        assert_eq!(classify(&d(1, 1, yes)), Class::Data, "{yes}");
    }
    for fault in ["OTS-E-ECC-001", "OTS-E-ECC-002", "OTS-E-ECC-003", "OTS-E-LNK-004"] {
        assert_eq!(classify(&d(1, 1, fault)), Class::Fault, "{fault}");
        assert!(!is_schedule_code(fault), "{fault}");
    }
    for no in ["OTS-E-TIM-001", "OTS-E-MEM-001", "OTS-E-RES-001", "OTS-E-QUE-001", "OTS-E-DEC-003"] {
        assert_eq!(classify(&d(1, 1, no)), Class::Schedule, "{no}");
    }
    let fetched = Diagnostic { fetched: true, ..d(1, 1, "OTS-E-DEC-003") };
    assert_eq!(classify(&fetched), Class::DataFetched);
    assert_eq!(Class::DataFetched.label(), "data-fetched");
}

/// The codes of class `schedule` in the table of `microarchitecture.md` section 8, read from the document.
fn table_schedule_codes() -> Vec<String> {
    let text = std::fs::read_to_string(root().join("opentsp-arch-spec/docs/microarchitecture.md")).unwrap();
    let mut codes = Vec::new();
    let mut in_table = false;
    for line in text.lines() {
        if line.starts_with("| Code | Raised when | Queue | Class |") {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }
        if !line.starts_with('|') {
            break;
        }
        let cells: Vec<&str> = line.trim_end().trim_end_matches('|').rsplitn(2, '|').collect();
        let class = cells[0].trim();
        let first = cells[1].split('|').nth(1).unwrap().trim();
        if class.starts_with("schedule") {
            codes.extend(first.split(',').map(|c| format!("OTS-E-{}", c.trim())));
        }
    }
    codes.sort();
    codes
}

#[test]
fn the_schedule_codes_are_the_schedule_class_of_the_specification_table() {
    let table = table_schedule_codes();
    assert_eq!(table.len(), 17, "{table:?}");
    let mut ours: Vec<String> = SCHEDULE_CODES.iter().map(|c| c.to_string()).collect();
    ours.sort();
    assert_eq!(ours, table, "SCHEDULE_CODES and the `schedule` entries of the Class column differ");
}

#[test]
fn the_verdict_is_clean_hazard_or_undecided() {
    let found = |chips| Found { chips };
    assert_eq!(judge(&found(vec![chip(0, vec![])])), Verdict::Clean);
    // the end-of-run record removes no effect: it alone leaves the run clean
    assert_eq!(judge(&found(vec![chip(0, vec![d(9, 1, "OTS-E-IQ-004")])])), Verdict::Clean);
    // a data code with no schedule code after it is undecided, not clean: the effect it removed (a Permute that did not run) can
    // be the instruction that would have raised a schedule code on the data of the real run
    for code in ["OTS-E-SXM-001", "OTS-E-IQ-002"] {
        let v = judge(&found(vec![chip(0, vec![d(3, 1, code), d(9, 1, "OTS-E-IQ-004")])]));
        assert_eq!(v, Verdict::Undecided { chip: 0, data: d(3, 1, code), schedule: None }, "{code}");
    }
    // a data code that the counts show and the list does not (the list was cut) decides the same way
    let mut counted = chip(0, vec![]);
    counted.data = Some("OTS-E-SXM-001".into());
    let v = judge(&found(vec![counted]));
    assert!(matches!(v, Verdict::Undecided { schedule: None, ref data, .. } if data.code == "OTS-E-SXM-001"), "{v:?}");
    // a schedule code first, then data: a hazard
    let hazard = vec![d(5, 2, "OTS-E-TIM-001"), d(6, 2, "OTS-E-SXM-001")];
    assert_eq!(judge(&found(vec![chip(0, hazard)])), Verdict::Hazard);
    // data first: undecided, naming the first data code
    let v = judge(&found(vec![chip(0, vec![d(3, 1, "OTS-E-SXM-001"), d(5, 2, "OTS-E-TIM-001")])]));
    assert_eq!(
        v,
        Verdict::Undecided { chip: 0, data: d(3, 1, "OTS-E-SXM-001"), schedule: Some((0, d(5, 2, "OTS-E-TIM-001"))) }
    );
    // the end-of-run record follows every run-time record whatever its stamp: it precedes nothing
    let late = vec![d(5, 2, "OTS-E-TIM-001"), d(2, 9, "OTS-E-IQ-004")];
    assert_eq!(judge(&found(vec![chip(0, late)])), Verdict::Hazard);
    // across chips: a data code at an earlier cycle on another chip precedes, one at the same cycle does not
    let earlier = judge(&found(vec![chip(0, vec![d(4, 1, "OTS-E-SXM-001")]), chip(1, vec![d(5, 2, "OTS-E-TIM-001")])]));
    assert!(matches!(earlier, Verdict::Undecided { chip: 0, .. }), "{earlier:?}");
    let same = judge(&found(vec![chip(0, vec![d(5, 1, "OTS-E-SXM-001")]), chip(1, vec![d(5, 2, "OTS-E-TIM-001")])]));
    assert_eq!(same, Verdict::Hazard);
    // a fetched DEC-003 is data: it makes the verdict undecided when a schedule code follows, and also when none does (what it would have dispatched
    // was never judged)
    let fetched = Diagnostic { fetched: true, ..d(2, 1, "OTS-E-DEC-003") };
    let alone = judge(&found(vec![chip(0, vec![fetched.clone()])]));
    assert_eq!(alone, Verdict::Undecided { chip: 0, data: fetched.clone(), schedule: None });
    // the count decides when the list is cut
    let mut cut_fetched = chip(0, vec![]);
    cut_fetched.fetched = 7;
    assert!(matches!(judge(&found(vec![cut_fetched])), Verdict::Undecided { schedule: None, .. }));
    let after = judge(&found(vec![chip(0, vec![fetched, d(7, 1, "OTS-E-TIM-001")])]));
    assert!(matches!(after, Verdict::Undecided { .. }), "{after:?}");
    // a schedule count with the list cut before it: a hazard unless a data code is listed
    let mut cut = chip(0, vec![]);
    cut.schedule = true;
    assert_eq!(judge(&found(vec![cut.clone()])), Verdict::Hazard);
    cut.listed = vec![d(1, 1, "OTS-E-SXM-001")];
    assert!(matches!(judge(&found(vec![cut])), Verdict::Undecided { schedule: None, .. }));
}

#[test]
fn the_scratch_directory_is_removed() {
    // the scratch directory is made under $TMPDIR: give the process its own and see it empty afterwards
    let tmp = tempfile::tempdir().unwrap();
    for obj in [data("hazard.elf"), "/no/such/object.elf".to_string()] {
        let o = std::process::Command::new(iss()).args(["verify", &obj]).env("TMPDIR", tmp.path()).output().unwrap();
        assert!(matches!(o.status.code(), Some(1 | 2)), "{}", text(&o.stderr));
        let left: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().filter_map(|e| e.ok()).collect();
        assert!(left.is_empty(), "{left:?}");
    }
}

// ---- the verifier's domain: every chip, both link latency bounds, parked text, three classes ----------------------
//
// The fixtures are in `tests/data/verify/`: each `.ots` is the source of the `.elf` next to it, and the manifests are
// `ots-asm mpack` of the objects named in the test (chip 0 first); `c0`/`c1` are a sender and the receiver of its
// vector at the latest link latency, `c1bad` the receiver one cycle too early, `lat` a receiver at send + 216.

fn fx(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/verify").join(name).to_str().unwrap().to_string()
}

fn host_file(dir: &tempfile::TempDir, text: &str) -> String {
    let p = dir.path().join("host.txt");
    std::fs::write(&p, text).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn verify_takes_a_topology_and_a_manifest_of_more_than_264_chips_needs_it() {
    // : the option is passed to the runs; without it the run's refusal is this command's message
    let (code, out, err) = verify(&[&fx("m265.otsm")]);
    assert_eq!((code, out.as_str()), (2, ""), "{err}");
    assert!(err.starts_with("tsp-iss: verify: ") && err.contains("--topology FILE"), "{err}");
    let (code, out, err) = verify(&["--topology", &fx("topo265.json"), &fx("m265.otsm")]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
}

#[test]
fn a_correct_manifest_is_clean_at_both_latency_bounds() {
    let (code, out, err) = verify(&[&fx("pair.otsm")]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
}

#[test]
fn a_hazard_on_chip_1_decides_the_exit_and_names_chip_and_latency() {
    let (code, out, err) = verify(&[&fx("pair_bad.otsm")]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.contains("chip 1") && out.contains("OTS-E-TIM-001") && out.contains("schedule"), "{out}");
    assert!(out.contains("link-latency"), "{out}");
}

#[test]
fn a_hazard_that_only_the_longest_link_shows_is_found() {
    // the receiver is at send + 216: clean at the default latency (216), a TIM-001 at 217 and above
    let (code, out, err) = verify(&[&fx("lat.otsm")]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.contains("link-latency 228"), "{out}");
    assert!(!out.contains("link-latency 209"), "{out}");
}

#[test]
fn a_lone_object_with_c2c_traffic_is_refused_and_the_manifest_asked_for() {
    for obj in ["c0.elf", "c1.elf"] {
        let (code, out, err) = verify(&[&fx(obj)]);
        assert_eq!((code, out.as_str()), (2, ""), "{obj}");
        assert!(err.starts_with("tsp-iss: verify: ") && err.contains("manifest") && err.contains("mpack"), "{err}");
    }
}

#[test]
fn text_parked_behind_a_host_wait_is_reported_and_fails_unless_allowed() {
    let (code, out, err) = verify(&[&fx("hw_ok.elf")]);
    assert_eq!(code, 1, "{out}{err}");
    assert_eq!(out, "parked queue 0 cycle 0 host-wait\n");
    assert!(err.contains("--host") && err.contains("--allow-parked"), "{err}");
    let (code, out, _) = verify(&[&fx("hw_ok.elf"), "--allow-parked"]);
    assert_eq!((code, out.as_str()), (0, "parked queue 0 cycle 0 host-wait\n"));
}

#[test]
fn a_host_script_releases_the_wait_and_the_text_behind_it_is_judged() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_file(&dir, "H db 5 0\n");
    let (code, out, err) = verify(&[&fx("hw_ok.elf"), "--host", &host]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    // the same wait, then a schedule hazard: found only because the host rings the doorbell
    let (code, out, _) = verify(&[&fx("hw.elf")]);
    assert_eq!((code, out.as_str()), (1, "parked queue 88 cycle 0 host-wait\n"));
    let (code, out, _) = verify(&[&fx("hw.elf"), "--host", &host]);
    assert_eq!(code, 1);
    assert!(out.contains("OTS-E-TIM-001") && out.contains("schedule") && !out.contains("parked"), "{out}");
    let (code, _, err) = verify(&[&fx("hw_ok.elf"), "--host", "/no/such/host.txt"]);
    assert_eq!(code, 2);
    assert!(err.starts_with("tsp-iss: verify: --host /no/such/host.txt"), "{err}");
    // a bad line of the host file is an input error that names the line
    let bad = host_file(&dir, "# the doorbell\nH nonsense 5 0\n");
    let (code, out, err) = verify(&[&fx("hw_ok.elf"), "--host", &bad]);
    assert_eq!((code, out.as_str()), (2, ""));
    assert!(err.starts_with("tsp-iss: verify: --host ") && err.contains("input line 2"), "{err}");
}

#[test]
fn text_parked_behind_a_deskew_recv_is_reported() {
    let (code, out, err) = verify(&[&fx("dr.otsm")]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.contains("parked chip 0 queue 128 cycle 0 deskew-recv"), "{out}");
}

#[test]
fn text_fetched_from_sram_that_does_not_decode_is_data_and_leaves_the_program_undecided() {
    let (code, out, err) = verify(&[&fx("ifm.elf")]);
    // no schedule code, but the fetched text was never judged: undecided, not clean
    assert_eq!(code, 3, "{out}{err}");
    assert!(err.contains("undecided") && err.contains("fetched from SRAM") && err.contains("--weights"), "{err}");
    assert_eq!(out.lines().count(), 160, "{out}");
    assert!(out.lines().all(|l| l.starts_with("OTS-E-DEC-003 ") && l.ends_with(" data-fetched")), "{out}");
}

#[test]
fn a_data_code_before_the_first_schedule_code_leaves_the_verdict_undecided() {
    let (code, out, err) = verify(&[&fx("casc.elf")]);
    assert_eq!(code, 3, "{out}{err}");
    assert!(out.contains("OTS-E-SXM-001 cycle 65 queue 124 data-dependent"), "{out}");
    assert!(err.contains("undecided") && err.contains("OTS-E-SXM-001"), "{err}");
    // with the program's data the run is clean
    let (code, out, err) = verify(&[&fx("casc.elf"), "--weights", &fx("casc.safetensors")]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
}

// ---- latencies per link and per transfer, host-only objects, the cycle limit, data-class OPR-001 -------------------
//
// `tri_c0`, `tri_c1`, `tri_c2` are three chips: chip 1 waits with a DeskewRecv for chip 0's vector (sent at cycle 40 on its link 0), then
// receives chip 2's vector (sent at cycle 30 on its link 1) at a fixed offset after the release. The release is at cycle 252 when the
// link from chip 0 is fast (latency 212 or less) and at 504 when it is slow, and the vector from chip 2 arrives at 239..258 at the
// earliest, so the receive finds it only if its own link is not slow while the other is fast: a hazard that needs one link at each bound.

#[test]
fn a_hazard_that_needs_one_link_fast_and_another_slow_is_found() {
    let (code, out, err) = verify(&[&fx("tri.otsm")]);
    assert_eq!(code, 1, "{out}{err}");
    // names the chip, the cycle, and the directed links at their latencies (chip:link=latency)
    assert!(out.contains("OTS-E-TIM-001 chip 1 cycle 253 queue 128 schedule link-latency 0:0=209+2:1=228"), "{out}");
    // the uniform bounds alone are clean for it
    for l in ["209", "228"] {
        assert!(!out.contains(&format!("link-latency {l}")), "{out}");
    }
}

#[test]
fn the_hazard_of_the_three_chip_manifest_is_one_the_reference_model_reports_too() {
    // the reference harness fails on the same manifest at some per-link draws and per-transfer jitter seeds (the ISS too, identically)
    let input = std::fs::read_to_string(fx("tri.mobjinput")).unwrap();
    let cfg = config("ref");
    let cfg = cfg.to_str().unwrap();
    let (mut sim_fail, mut iss_fail) = (0, 0);
    for (opt, n) in [("--link-latency-seed", 40), ("--link-jitter-seed", 120)] {
        for seed in 0..n {
            let seed = seed.to_string();
            let args = [cfg, "--chips", "3", opt, seed.as_str()];
            let sim = run_sim(&args, &input);
            let iss = run_iss(&args, &input);
            assert_eq!(text_of(&sim), text_of(&iss), "{opt} {seed}");
            sim_fail += usize::from(text(&sim.stdout).contains("E OTS-E-TIM-001"));
            iss_fail += usize::from(text(&iss.stdout).contains("E OTS-E-TIM-001"));
        }
    }
    assert!(sim_fail > 0 && sim_fail == iss_fail, "sim {sim_fail}, iss {iss_fail}");
    // and verify says so
    assert_eq!(verify(&[&fx("tri.otsm")]).0, 1);
}

fn text_of(o: &std::process::Output) -> String {
    format!("{}|{}", text(&o.stdout), o.status.code().unwrap_or(-1))
}

#[test]
fn a_schedule_that_depends_on_the_latencies_and_shows_no_hazard_is_undecided() {
    // chip 1 only waits with a DeskewRecv: released at 252 or 504 depending on the link, nothing after it can fail
    let (code, out, err) = verify(&[&fx("strad.otsm")]);
    assert_eq!((code, out.as_str()), (3, ""), "{err}");
    assert!(
        err.contains("undecided")
            && err.contains("depends on the link latencies")
            && err.contains("chip 1 queue 128")
            && err.contains("cycle 252")
            && err.contains("cycle 504")
            && err.contains("not judged"),
        "{err}"
    );
    // a manifest whose DeskewRecv is released at the same epoch on every latency is judged by the two bounds alone
    let (code, out, err) = verify(&[&fx("dr.otsm")]);
    assert!(!err.contains("depends on the link latencies"), "{out}{err}");
    assert_eq!(code, 1, "{out}{err}"); // parked
}

#[test]
fn a_failing_later_run_still_shows_what_the_earlier_runs_found() {
    // the run at the longest latency needs cycle 504; the run at the shortest ends at 253 and has found three hazards
    //  /  (3): hazard-found beats --max-cycles undecided — exit 1, not 3
    let (code, out, err) = verify(&["--max-cycles", "300", &fx("strad_haz.otsm")]);
    assert_eq!(code, 1, "{out}{err}"); // not 3: hazard-found beats --max-cycles undecided
    assert!(out.lines().count() == 3 && out.lines().all(|l| l.starts_with("OTS-E-TIM-001 chip 1 ")), "{out}");
    assert!(err.contains("--max-cycles 300"), "{err}");
    // with room the same manifest is a hazard at both bounds
    let (code, out, _) = verify(&[&fx("strad_haz.otsm")]);
    assert_eq!(code, 1);
    assert!(out.contains("link-latency 209,228"), "{out}");
}

#[test]
fn max_cycles_is_passed_to_the_run() {
    let (code, out, err) = verify(&["--max-cycles", "10", &data("hazard.elf")]);
    assert_eq!((code, out.as_str()), (3, ""), "{err}");
    assert!(err.starts_with("tsp-iss: verify: the run needs cycle"), "{err}"); // : named as a verify message
    assert!(err.contains("--max-cycles 10"), "{err}");
    // the option needs its value
    assert_eq!(verify(&[&data("hazard.elf"), "--max-cycles"]).0, 2);
}

#[test]
fn an_object_that_only_talks_to_the_host_is_judged_not_refused() {
    // PciSend and PciReceive sit on the C2C queues but name no link
    let (code, out, err) = verify(&[&fx("pci_send.elf")]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    let (code, out, err) = verify(&[&fx("pci_recv.elf")]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.contains("OTS-E-TIM-001 cycle 100 queue 142 schedule") && !err.contains("manifest"), "{out}{err}");
    // the host's vector arrives in time when the host script says so
    let dir = tempfile::tempdir().unwrap();
    let host = host_file(&dir, "H dma 0 1 2 3\n");
    let (code, out, err) = verify(&[&fx("pci_recv.elf"), "--host", &host]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    // a Send in the same text is a link and is still refused
    assert_eq!(verify(&[&fx("tri_c0.elf")]).0, 2);
}

#[test]
fn an_opr_001_of_an_sxselect_selector_is_data_when_the_data_is_given() {
    // the selector comes from an extern: without data it is 0 (valid); the weight package gives it 3 (sel > 2)
    let (code, out, err) = verify(&[&fx("sel.elf")]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    let (code, out, err) = verify(&["--weights", &fx("sel3.safetensors"), &fx("sel.elf")]);
    // the selector's OPR-001 is a data code (printed, never a schedule failure); it removed the select's effect, so the run
    // cannot decide the object (the object is judged by the run without that data, above)
    assert_eq!(out, "OTS-E-OPR-001 cycle 45 queue 125 data-dependent\n", "{err}");
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("undecided") && err.contains("OTS-E-OPR-001"), "{err}");
    // : --weights was given, so the message does not ask for it
    assert!(!err.contains("give --weights"), "{err}");
}

#[test]
fn the_new_keys_of_diagnostics_json_are_read() {
    let doc = r#"{"chips": [
        {"by_code": {"OTS-E-OPR-001": 1}, "by_code_data": {"OTS-E-OPR-001": 1}, "chip": 0,
         "diagnostics": [{"code": "OTS-E-OPR-001", "cycle": 45, "queue": 125, "data": true}],
         "listed": 1, "total": 1, "truncated": false,
         "deskew_released": [[128, 0, 252]], "links_sent": [0, 3], "links_received": [1]}], "schema": 1}"#;
    let f = read_diagnostics(doc).unwrap();
    let c = &f.chips[0];
    assert!(c.listed[0].data && classify(&c.listed[0]) == Class::Data);
    assert!(!c.schedule, "the data-class OPR-001 is not a schedule code");
    assert_eq!((c.released.clone(), c.sent.clone(), c.received.clone()), (vec![(128, 0, 252)], vec![0, 3], vec![1]));
    // without the count of data codes the same code is a schedule code
    let plain = doc.replace(r#""by_code_data": {"OTS-E-OPR-001": 1}, "#, "");
    assert!(read_diagnostics(&plain).unwrap().chips[0].schedule);
}

#[test]
fn a_data_code_that_hides_a_schedule_code_leaves_the_verdict_undecided() {
    // `hide.elf` is `casc.elf` plus a transposer that writes the stream register the Permute writes, in the same cycle. Without
    // data the Permute is suppressed (SXM-001), so the write collision (SRF-002) that the real maps give is never raised: the
    // run holds a data code and no schedule code.
    let (code, out, err) = verify(&[&fx("hide.elf")]);
    assert_eq!(code, 3, "{out}{err}");
    assert!(out.contains("OTS-E-SXM-001 cycle 65 queue 124 data-dependent") && !out.contains("schedule"), "{out}");
    assert!(err.contains("undecided") && err.contains("OTS-E-SXM-001") && err.contains("--weights"), "{err}");
    // maps that are all zero give the same run
    let (code, out, err) = verify(&[&fx("hide.elf"), "--weights", &fx("casc_zero.safetensors")]);
    assert_eq!(code, 3, "{out}{err}");
    // the program's own maps give the collision: ill-formed
    let (code, out, err) = verify(&[&fx("hide.elf"), "--weights", &fx("casc.safetensors")]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(out.contains("OTS-E-SRF-002 cycle 85 queue 120 schedule"), "{out}");
}
