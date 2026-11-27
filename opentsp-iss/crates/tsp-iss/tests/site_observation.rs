//! The optional Sail site stream, including its interleaving with the ordinary harness output.
mod common;
use common::*;

#[test]
fn observed_runs_match_sail() {
    let fetched = "Q 0 01c00010\nS 0 10 ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff\nS 0 11 12 00 00 10 00 00 00 1c ff ff ff ff ff ff ff ff\n";
    for name in ["ref", "mini"] {
        let cfg = config(name);
        for (program, options) in [
            ("Q 58 51c00000 01800001\nH db 7 0\nH csr 4 w 7f0 0\n", vec![]),
            ("Q ffff 00800000\nQ 58 3 01800001 00c00031 01800001\nH inject 9 58 01800001\n", vec![]),
            ("Q 80 52400000 01800001\nH csr 7 w 7f0 0\n", vec![]),
            (fetched, vec![]),
            ("C 0\nQ 58 ffffffff 01800001\nC 1\nQ 59 3 01800001\n", vec!["--chips", "2"]),
        ] {
            for trace in [false, true] {
                let mut args = vec![cfg.to_str().unwrap(), "--observe"];
                args.extend_from_slice(&options);
                if trace {
                    args.push("--trace");
                }
                let oracle = run_sim(&args, program);
                assert!(oracle.status.success(), "{}", text(&oracle.stderr));
                assert!(text(&oracle.stdout).lines().any(|l| l.starts_with("O ")));
                let native = run_bin(iss(), &args, program.as_bytes());
                assert!(native.status.success(), "{}", text(&native.stderr));
                assert_eq!(text(&native.stdout), text(&oracle.stdout), "{args:?} {program}");
            }
        }
    }
}

#[test]
fn faulty_second_beat_reports_before_admission_on_each_replay_attempt() {
    // Two MEM Reads feed a stream Ifetch. Damage two bits in its second beat.
    let program = "Q 0 10000000 0 10000001 0\nQ 58 8 01000000\n";
    for name in ["ref", "mini"] {
        for (mark, code) in [("t", 0), ("p", 5)] {
            let cfg = config(name);
            let args = [cfg.to_str().unwrap(), "--trace", "--observe", "--on-irq", "replay"];
            let program = format!("{program}F srf 9 e 0 15 0 80 {mark}\nF srf 9 e 0 15 0 81 {mark}\n");
            let oracle = run_sim(&args, &program);
            assert_eq!(oracle.status.code(), Some(code), "{}", text(&oracle.stderr));
            assert!(text(&oracle.stdout).contains("T replay start"));
            let out = run_bin(iss(), &args, program.as_bytes());
            assert_eq!(out.status.code(), Some(code));
            assert_eq!(text(&out.stdout), text(&oracle.stdout));
        }
    }
}

#[test]
fn native_observation_survives_statistics_profiling_and_repeated_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let program = "Q 58 01800001 01800002\n";
    let file = tmp.path().join("program.txt");
    let profile = tmp.path().join("profile.json");
    std::fs::write(&file, program).unwrap();
    let args = [
        "run",
        "--profile",
        "mini",
        "--observe",
        "--trace",
        "--stats",
        "--run-profile-json",
        profile.to_str().unwrap(),
        "--replay-runs",
        "2",
        file.to_str().unwrap(),
    ];
    let out = run_bin(iss(), &args, b"");
    assert!(out.status.success(), "{}", text(&out.stderr));
    let oracle = run_sim(&[config("mini").to_str().unwrap(), "--observe", "--trace"], program);
    assert!(oracle.status.success());
    assert_eq!(text(&out.stdout), text(&oracle.stdout));
    assert!(profile.is_file());
}

#[test]
fn help_advertises_observation_and_preserves_error_usage() {
    for args in [vec!["--help"], vec![]] {
        let oracle = run_sim(&args, "");
        let out = run_bin(iss(), &args, b"");
        assert_eq!(out.status.code(), oracle.status.code());
        let help = text(&oracle.stdout).replace(tsp_sim().to_str().unwrap(), "tsp-iss");
        // `--help` continues with the subcommands of tsp-iss after the line of `tsp_sim`
        assert!(text(&out.stdout).starts_with(&help), "{}", text(&out.stdout));
        assert_eq!(text(&out.stdout) == help, args.is_empty());
        assert_eq!(help.contains("--observe"), !args.is_empty());
        assert_eq!(text(&out.stderr).contains("--observe"), text(&oracle.stderr).contains("--observe"));
    }
}
