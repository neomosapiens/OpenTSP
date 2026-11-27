//! Multi-chip manifests through the command line: `mpack` writes the bytes `tsp_as.py mpack`
//! writes, and `mobjinput` prints the `C <chip>` input of `tsp_as.py mobjinput`. The expected files in
//! `tests/data/` (`*.otsm`, `*.mobjinput`) were produced by `opentsp-arch-spec/tools/tsp_as.py` from the
//! objects and packages of `tests/data/`: `ref1` (`example`), `ref2` (`example`, `bind`), `mini3`
//! (`example_mini`, `bind_mini`, `example_mini`) and `mini16` (`example_mini`, `bind_mini` eight times,
//! 16 chips = 8 N, N = 2). The layout and every rejection of the reader are in `ots-format/tests/manifest.rs`,
//! and the differential run (`just asm-diff`, `tests/diff_tsp_as.py`) holds both to `mpack` and `mobjinput` over random chip sets and to
//! seeded mutations of the manifests `tsp_as.py` made.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR")))
}

fn bytes(name: &str) -> Vec<u8> {
    fs::read(data(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn run(args: &[&str]) -> Output {
    let mut waited = 0;
    loop {
        match Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && waited < 100 => {
                waited += 1;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            r => return r.unwrap(),
        }
    }
}

fn text(o: &Output) -> (i32, String, String) {
    (
        o.status.code().unwrap(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ots-asm-manifest-{tag}-{}", std::process::id()));
        fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }
    fn path(&self, n: &str) -> String {
        self.0.join(n).to_str().unwrap().to_string()
    }
    fn write(&self, n: &str, b: &[u8]) -> String {
        let p = self.path(n);
        fs::write(&p, b).unwrap();
        p
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).ok();
    }
}

fn mpack(profile: Option<&str>, out: &str, objs: &[&str]) -> (i32, String, String) {
    let mut a = vec!["mpack"];
    if let Some(p) = profile {
        a.extend(["--profile", p]);
    }
    a.extend(["-o", out]);
    a.extend(objs);
    text(&run(&a))
}

/// `(manifest, objects in chip order, weight packages in chip order)` of the fixtures.
const SETS: [(&str, &str, &[&str], &[&str]); 4] = [
    ("ref1", "ref", &["example.elf"], &["example.safetensors"]),
    ("ref2", "ref", &["example.elf", "bind.elf"], &["example.safetensors", "rich.safetensors"]),
    (
        "mini3",
        "mini",
        &["example_mini.elf", "bind_mini.elf", "example_mini.elf"],
        &["example_mini.safetensors", "rich_mini.safetensors", "example_mini.safetensors"],
    ),
    (
        "mini16",
        "mini",
        &[
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
            "example_mini.elf",
            "bind_mini.elf",
        ],
        &[
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
            "example_mini.safetensors",
            "rich_mini.safetensors",
        ],
    ),
];

#[test]
fn mpack_writes_the_manifest_tsp_as_mpack_writes() {
    let t = Tmp::new("pack");
    for (name, profile, objs, _) in SETS {
        let paths: Vec<PathBuf> = objs.iter().map(|o| data(o)).collect();
        let args: Vec<&str> = paths.iter().map(|p| s(p)).collect();
        let out = t.path(&format!("{name}.otsm"));
        assert_eq!(mpack(Some(profile), &out, &args), (0, String::new(), String::new()), "{name}");
        assert_eq!(fs::read(&out).unwrap(), bytes(&format!("{name}.otsm")), "{name}");
    }
}

#[test]
fn mpack_takes_the_profile_ref_by_default() {
    let t = Tmp::new("default");
    let out = t.path("d.otsm");
    assert_eq!(mpack(None, &out, &[s(&data("example.elf")), s(&data("bind.elf"))]), (0, String::new(), String::new()));
    assert_eq!(fs::read(&out).unwrap(), bytes("ref2.otsm"));
    // and a mini object is then refused, the chip named
    let (rc, o, e) = mpack(None, &t.path("m.otsm"), &[s(&data("example.elf")), s(&data("example_mini.elf"))]);
    assert_eq!((rc, o.as_str()), (1, ""));
    assert!(e.starts_with("mpack: OTS-E-ASM-003: chip 1: "), "{e}");
}

#[test]
fn the_manifest_has_the_layout_of_the_specification() {
    let m = bytes("ref2.otsm");
    let (a, b) = (bytes("example.elf"), bytes("bind.elf"));
    assert_eq!(&m[..8], b"OTSM\x00\x01\x40\x00");
    assert_eq!(&m[8..24], b"ref\0\0\0\0\0\0\0\0\0\0\0\0\0");
    assert_eq!(&m[24..40], &[2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&m[40..52], &a[92..104], "encoding digest prefix");
    assert_eq!(&m[52..64], &a[124..136], "profile digest prefix");
    let e = |off: usize, n: usize| [(off as u32).to_le_bytes(), (n as u32).to_le_bytes()].concat();
    assert_eq!(&m[64..72], e(80, a.len()).as_slice());
    assert_eq!(&m[72..80], e(80 + a.len(), b.len()).as_slice());
    assert_eq!(&m[80..80 + a.len()], a.as_slice());
    assert_eq!(&m[80 + a.len()..], b.as_slice());
}

#[test]
fn mpack_rejections_are_asm_003_exit_1_and_leave_no_file() {
    let t = Tmp::new("rej");
    let ex = s(&data("example.elf")).to_string();
    let out = t.path("o.otsm");
    let junk = t.write("junk.elf", b"OTSO");
    let man = s(&data("ref1.otsm")).to_string();
    let q = t.write("q.txt", b"Q 4 00000021\n");
    // a system of the `--chips` wiring holds 1..264 chips (nine is legal; the limit was eight before the chip field widened)
    let too_many: Vec<&str> = vec![ex.as_str(); 265];
    for (what, profile, objs, want) in [
        ("no object", "ref", vec![], "0 chips"),
        ("265 objects", "ref", too_many, "265 chips"),
        ("a bad object", "ref", vec![ex.as_str(), junk.as_str()], "chip 1: "),
        ("a manifest as chip", "ref", vec![man.as_str()], "chip 0: "),
        ("a text file", "ref", vec![q.as_str()], "chip 0: "),
        ("another profile", "mini", vec![ex.as_str()], "chip 0: "),
        ("an unknown profile", "nosuch", vec![ex.as_str()], "chip 0: "),
    ] {
        let (rc, o, e) = mpack(Some(profile), &out, &objs);
        assert_eq!((rc, o.as_str()), (1, ""), "{what}: {e}");
        assert!(e.starts_with("mpack: OTS-E-ASM-003: ") && e.contains(want), "{what}: {e}");
        assert!(!std::path::Path::new(&out).exists(), "{what}: a file was written");
    }
}

#[test]
fn mpack_takes_up_to_264_chips_without_a_topology_file() {
    let t = Tmp::new("wide");
    let ex = s(&data("example.elf")).to_string();
    for k in [9usize, 264] {
        let out = t.path(&format!("k{k}.otsm"));
        let objs: Vec<&str> = vec![ex.as_str(); k];
        let (rc, o, e) = mpack(Some("ref"), &out, &objs);
        assert_eq!((rc, o.as_str(), e.as_str()), (0, "", ""), "{k} chips");
        let m = fs::read(&out).unwrap();
        assert_eq!(u32::from_le_bytes(m[24..28].try_into().unwrap()) as usize, k, "the chip count of the header");
    }
}

#[test]
fn mpack_usage_and_os_errors() {
    let t = Tmp::new("usage");
    let ex = s(&data("example.elf")).to_string();
    // no -o, `-o` without a value
    for args in [vec!["mpack", "--profile", "ref", ex.as_str()], vec!["mpack", "-o"], vec!["mpack"]] {
        let (rc, o, _) = text(&run(&args));
        assert_eq!((rc, o.as_str()), (2, ""), "{args:?}");
    }
    // an input that cannot be read: exit 1 and no manifest
    let out = t.path("o.otsm");
    let (rc, _, e) = mpack(Some("ref"), &out, &[ex.as_str(), "/nonexistent/chip1.elf"]);
    assert_eq!(rc, 1);
    assert!(e.starts_with("mpack: ") && e.contains("chip1.elf"), "{e}");
    assert!(!std::path::Path::new(&out).exists());
    // an output that cannot be written
    let (rc, _, e) = mpack(Some("ref"), &t.path("no/such/dir/o.otsm"), &[ex.as_str()]);
    assert_eq!(rc, 1);
    assert!(e.starts_with("mpack: ") && e.contains("o.otsm"), "{e}");
}

#[test]
fn mpack_checks_the_configuration_of_the_objects() {
    let t = Tmp::new("config");
    let text_ =
        fs::read_to_string(format!("{}/../../../opentsp-arch-spec/config/ref.json", env!("CARGO_MANIFEST_DIR")))
            .unwrap();
    let cfg = t.write(
        "ref22.json",
        text_.replace("\"mem_slices_per_hemisphere\": 44", "\"mem_slices_per_hemisphere\": 22").as_bytes(),
    );
    // an object of the shipped `ref` is not an object of this configuration. (This configuration is not one the model
    // accepts, so `--no-validate` lets the command reach the check under test; without it status 2, config_validate.rs.)
    let (rc, _, e) = text(&run(&[
        "mpack",
        "--profile",
        "ref",
        "--no-validate",
        "--config",
        &cfg,
        "-o",
        &t.path("c.otsm"),
        s(&data("example.elf")),
    ]));
    assert_eq!(rc, 1);
    assert!(e.starts_with("mpack: OTS-E-ASM-003: chip 0: "), "{e}");
    // `mobjinput` reads a manifest against the same configuration
    let (rc, o, e) = text(&run(&["mobjinput", s(&data("ref1.otsm")), "--no-validate", "--config", &cfg]));
    assert_eq!((rc, o.as_str()), (1, ""));
    assert!(e.starts_with("mobjinput: OTS-E-ASM-003: ") && e.contains("configuration"), "{e}");
}

#[test]
fn mobjinput_prints_the_c_line_and_the_q_lines_of_every_chip() {
    for (name, ..) in SETS {
        let (rc, out, err) = text(&run(&["mobjinput", s(&data(&format!("{name}.otsm")))]));
        assert_eq!((rc, err.as_str()), (0, ""), "{name}");
        assert_eq!(out.as_bytes(), bytes(&format!("{name}.mobjinput")).as_slice(), "{name}");
    }
}

#[test]
fn each_chip_is_the_objinput_of_its_object_after_its_c_line() {
    // `ref2`: chip 0 is `example`, chip 1 is `bind`; with the packages, `objinput --weights` of each
    let (rc, out, _) = text(&run(&[
        "mobjinput",
        s(&data("ref2.otsm")),
        "--weights",
        s(&data("example.safetensors")),
        "--weights",
        s(&data("rich.safetensors")),
    ]));
    assert_eq!(rc, 0);
    let (a, b) = (bytes("example.weights.objinput"), bytes("bind.weights.objinput"));
    let want = [b"C 0\n".as_slice(), &a, b"C 1\n", &b].concat();
    assert_eq!(out.as_bytes(), want.as_slice());
}

#[test]
fn mobjinput_with_one_package_per_chip_adds_the_s_lines() {
    for (name, _, _, pkgs) in SETS {
        let paths: Vec<PathBuf> = pkgs.iter().map(|p| data(p)).collect();
        let man = data(&format!("{name}.otsm"));
        let mut args = vec!["mobjinput", s(&man)];
        for p in &paths {
            args.extend(["--weights", s(p)]);
        }
        let (rc, out, err) = text(&run(&args));
        assert_eq!((rc, err.as_str()), (0, ""), "{name}");
        assert_eq!(out.as_bytes(), bytes(&format!("{name}.weights.mobjinput")).as_slice(), "{name}");
    }
}

#[test]
fn the_chips_are_numbered_in_hex() {
    let (_, out, _) = text(&run(&["mobjinput", s(&data("mini16.otsm"))]));
    let c: Vec<&str> = out.lines().filter(|l| l.starts_with("C ")).collect();
    assert_eq!(
        c,
        [
            "C 0", "C 1", "C 2", "C 3", "C 4", "C 5", "C 6", "C 7", "C 8", "C 9", "C a", "C b", "C c", "C d", "C e",
            "C f"
        ]
    );
}

#[test]
fn the_number_of_packages_must_be_one_per_chip() {
    let m = s(&data("ref2.otsm")).to_string();
    let (w0, w1) = (s(&data("example.safetensors")).to_string(), s(&data("rich.safetensors")).to_string());
    for (what, weights, want) in [
        ("one for two", vec![w0.as_str()], "1 weight packages for 2 chips"),
        ("three for two", vec![w0.as_str(), w1.as_str(), w1.as_str()], "3 weight packages for 2 chips"),
    ] {
        let mut args = vec!["mobjinput", m.as_str()];
        for w in weights {
            args.extend(["--weights", w]);
        }
        let (rc, out, err) = text(&run(&args));
        assert_eq!((rc, out.as_str()), (1, ""), "{what}");
        assert!(err.starts_with("mobjinput: OTS-E-ASM-004: ") && err.contains(want), "{what}: {err}");
    }
}

#[test]
fn packages_in_the_wrong_order_do_not_bind() {
    // chip 1 is `bind`, whose externs example.safetensors does not have; nothing is printed
    let (rc, out, err) = text(&run(&[
        "mobjinput",
        s(&data("ref2.otsm")),
        "--weights",
        s(&data("rich.safetensors")),
        "--weights",
        s(&data("example.safetensors")),
    ]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("mobjinput: OTS-E-ASM-004: "), "{err}");
    // a package for the other profile
    let (rc, out, err) = text(&run(&[
        "mobjinput",
        s(&data("ref2.otsm")),
        "--weights",
        s(&data("example.safetensors")),
        "--weights",
        s(&data("rich_mini.safetensors")),
    ]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("mobjinput: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn mobjinput_of_something_that_is_not_a_manifest_is_asm_003() {
    let t = Tmp::new("notman");
    let cut = bytes("ref2.otsm");
    let cut = t.write("cut.otsm", &cut[..cut.len() - 1]);
    let obj = s(&data("example.elf")).to_string();
    let txt = t.write("q.txt", b"Q 4 00000021\n");
    for (what, path) in
        [("an object", obj), ("text", txt), ("a truncated manifest", cut), ("an empty file", t.write("e.otsm", b""))]
    {
        let (rc, out, err) = text(&run(&["mobjinput", &path]));
        assert_eq!((rc, out.as_str()), (1, ""), "{what}");
        assert!(err.starts_with("mobjinput: OTS-E-ASM-003: "), "{what}: {err}");
    }
}

#[test]
fn mobjinput_ignores_the_profile_option_and_names_its_own_profile() {
    let a = text(&run(&["mobjinput", s(&data("mini3.otsm"))]));
    let b = text(&run(&["mobjinput", "--profile", "ref", s(&data("mini3.otsm"))]));
    assert_eq!(a.0, 0);
    assert_eq!(a, b);
}

#[test]
fn the_manifest_is_read_before_the_packages_and_the_packages_before_the_check() {
    let t = Tmp::new("order");
    // a bad manifest and a missing package: the package file is read first (tsp_as reads every file
    // before it parses anything), so the OS error wins
    let junk = t.write("junk.otsm", b"OTSM");
    let (rc, _, err) = text(&run(&["mobjinput", &junk, "--weights", "/nonexistent/w.safetensors"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("mobjinput: ") && err.contains("w.safetensors"), "{err}");
    // a missing manifest
    let (rc, _, err) = text(&run(&["mobjinput", "/nonexistent/m.otsm"]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("mobjinput: ") && err.contains("m.otsm"), "{err}");
    // the package count is checked after the manifest
    let (rc, _, err) = text(&run(&["mobjinput", &junk, "--weights", s(&data("example.safetensors"))]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("mobjinput: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn mobjinput_usage() {
    for args in [vec!["mobjinput"], vec!["mobjinput", "--weights"], vec!["mobjinput", "--weights", "w.safetensors"]] {
        let (rc, o, e) = text(&run(&args));
        assert_eq!((rc, o.as_str()), (2, ""), "{args:?}");
        assert!(e.contains("usage:"), "{e}");
    }
}

#[test]
fn the_usage_lists_the_manifest_commands() {
    let (rc, _, e) = text(&run(&[]));
    assert_eq!(rc, 2);
    assert!(e.contains("ots-asm mpack") && e.contains("ots-asm mobjinput"), "{e}");
}

#[test]
fn a_repeated_weights_option_is_a_list_and_objinput_still_takes_the_last() {
    // `tsp_as.main` keeps every --weights for mobjinput and the last for objinput
    let (rc, out, _) = text(&run(&[
        "objinput",
        s(&data("example.elf")),
        "--weights",
        s(&data("rich.safetensors")),
        "--weights",
        s(&data("example.safetensors")),
    ]));
    assert_eq!(rc, 0);
    assert_eq!(out.as_bytes(), bytes("example.weights.objinput").as_slice());
}
