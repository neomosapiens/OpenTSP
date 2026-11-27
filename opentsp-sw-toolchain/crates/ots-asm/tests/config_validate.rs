//! `--config F` is validated like `tsp-iss` and `tsp_sim` validate it: a configuration the model refuses
//! is refused by every command with the same message and status 2; `--no-validate` is the escape.
//! The messages below are the ones `tsp-iss` prints (`opentsp-iss/crates/tsp-iss/tests/cli_paths.rs` pins the same strings).
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn ots_asm(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output().unwrap()
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// (file name, text) of the configurations the model refuses: 33 superlanes, 16384 words per slice on a 13-bit address, 131072 words per slice,
/// 640 MXM rows.
fn bad_configs() -> Vec<(&'static str, String, &'static str)> {
    let r = std::fs::read_to_string(root().join("opentsp-arch-spec/config/ref.json")).unwrap();
    let swap = |a: &str, b: &str| {
        assert!(r.contains(a), "{a}");
        r.replace(a, b)
    };
    vec![
        (
            "sl33",
            swap("\"superlanes\": 20", "\"superlanes\": 33"),
            "geometry.superlanes: 33 is above the maximum 32 (must be between 1 and 32)",
        ),
        (
            "w16k",
            swap(
                "\"words_per_slice\": 8192,\n    \"bank_bit\": 12",
                "\"words_per_slice\": 16384,\n    \"bank_bit\": 13",
            ),
            "sram.bank_bit: must be < sram.addr_bits (sram.bank_bit = 13, sram.addr_bits = 13): a slice has at most 2^addr_bits words, so the bank bit (log2(words_per_slice) - 1) is below addr_bits",
        ),
        (
            "w128k",
            swap(
                "\"words_per_slice\": 8192,\n    \"bank_bit\": 12,\n    \"addr_bits\": 13",
                "\"words_per_slice\": 131072,\n    \"bank_bit\": 16,\n    \"addr_bits\": 16",
            ),
            "sram.words_per_slice: must be one of 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536",
        ),
        (
            "mxm640",
            swap("\"rows\": 320", "\"rows\": 640"),
            "mxm.rows: 640 is above the maximum 320 (must be between 16 and 320)",
        ),
    ]
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ots-asm-config-validate-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn every_command_refuses_a_configuration_the_model_refuses() {
    let dir = scratch("refuse");
    let data = root().join("opentsp-sw-toolchain/crates/ots-asm/tests/data");
    let ots = data.join("example.ots");
    let elf = data.join("example.elf");
    let otsm = data.join("mini3.otsm");
    for (name, text, message) in bad_configs() {
        let cfg = dir.join(format!("{name}.json"));
        std::fs::write(&cfg, text).unwrap();
        let cfg = cfg.to_str().unwrap();
        let want = format!("ots-asm: config {cfg}: {message}\n");
        let out_file = dir.join("out.bin");
        let out_file = out_file.to_str().unwrap();
        let commands: Vec<Vec<&str>> = vec![
            vec!["insn", "--config", cfg, "nop 1"],
            vec!["word", "--config", cfg, "32", "0"],
            vec!["asm", "--config", cfg, ots.to_str().unwrap()],
            vec!["dis", "--config", cfg, elf.to_str().unwrap()],
            vec!["objinput", "--config", cfg, elf.to_str().unwrap()],
            vec!["wpack", "--profile", "ref", "--config", cfg, "-o", out_file],
            vec!["mpack", "--config", cfg, "-o", out_file, elf.to_str().unwrap()],
            vec!["mobjinput", "--config", cfg, otsm.to_str().unwrap()],
        ];
        for args in commands {
            let o = ots_asm(&args);
            assert_eq!(o.status.code(), Some(2), "{name} {args:?}");
            assert_eq!(String::from_utf8_lossy(&o.stderr), want, "{name} {args:?}");
            assert!(o.stdout.is_empty(), "{name} {args:?}");
        }
        // the escape: the command runs as it did before the check existed
        let o = ots_asm(&["insn", "--no-validate", "--config", cfg, "nop 1"]);
        assert_ne!(o.status.code(), Some(2), "{name}: {}", String::from_utf8_lossy(&o.stderr));
    }
}

#[test]
fn a_valid_configuration_and_a_missing_file_are_as_before() {
    let dir = scratch("valid");
    let ok = dir.join("ref.json");
    std::fs::copy(root().join("opentsp-arch-spec/config/ref.json"), &ok).unwrap();
    let o = ots_asm(&["insn", "--config", ok.to_str().unwrap(), "nop 0xfcb3"]);
    assert_eq!((o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned()), (Some(0), "32 fcb3\n".into()));
    // a file that cannot be read keeps the message of the command (here: the profile loader's)
    let o = ots_asm(&["insn", "--config", "/no/such/file.json", "nop 1"]);
    assert_ne!(o.status.code(), Some(0));
    assert!(!String::from_utf8_lossy(&o.stderr).contains("ots-asm: config /no/such/file.json: "));
}

/// : a profile read with `--no-validate` that lacks the key of a profile variable (`mxm.acc_depth`) is refused by name
/// (`OTS-E-ASM-003`), as `tsp_as.py` refuses it, and no longer falls back to the lane count for the bound of an ACC index.
#[test]
fn a_profile_without_a_variable_key_is_refused_by_name_without_validation() {
    let dir = scratch("novar");
    let r = std::fs::read_to_string(root().join("opentsp-arch-spec/config/ref.json")).unwrap();
    let cfg = dir.join("noacc.json");
    std::fs::write(&cfg, r.replace("\"acc_depth\": 2,", "")).unwrap();
    assert!(!std::fs::read_to_string(&cfg).unwrap().contains("acc_depth"));
    let o = ots_asm(&["insn", "--no-validate", "--config", cfg.to_str().unwrap(), "acc.i32 W:0, replace, 31"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert_eq!(o.status.code(), Some(1), "{err}");
    assert!(err.contains("OTS-E-ASM-003") && err.contains("mxm.acc_depth"), "{err}");
    assert!(o.stdout.is_empty());
    // with the key the same command runs as before (and an index at the depth is refused by the operand check)
    let o = ots_asm(&[
        "insn",
        "--config",
        root().join("opentsp-arch-spec/config/ref.json").to_str().unwrap(),
        "acc.i32 W:0, replace, 1",
    ]);
    assert_eq!(o.status.code(), Some(0));
}
