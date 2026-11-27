//! Weight packages through the command line: `wpack` writes the bytes `tsp_as.py wpack`
//! writes, and `objinput --weights` turns an object and a package into the harness's `S` and `Q` lines.
//! The expected files in `tests/data/` (`*.safetensors`, `*.weights.objinput`) were produced by
//! `opentsp-arch-spec/tools/tsp_as.py` (`wpack`, `objinput --weights`) from the objects and the byte
//! patterns written here (`pat`); the layout and every rejection of the reader are in
//! `ots-format/tests/weights.rs`, and the differential run (`tests/diff_tsp_as.py`) holds both to
//! mutations of every package.
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

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("ots-asm-weights-{tag}-{}", std::process::id()));
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

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// The bytes of tensor `t`: `n` words of `vl` lanes (the generator of the fixtures).
fn pat(t: usize, n: usize, vl: usize) -> Vec<u8> {
    (0..n).flat_map(|j| (0..vl).map(move |l| ((t * 53 + j * 31 + l * 7 + 1) & 0xff) as u8)).collect()
}

/// The five raw files the fixtures were packed from, under `t`; `(name=dtype:file ...)` arguments of
/// the example and of the rich package.
fn inputs(t: &Tmp, vl: usize) -> (Vec<String>, Vec<String>) {
    let x = t.write("x.bin", &vec![3; vl]);
    let y = t.write("y.bin", &vec![4; vl]);
    let ex = vec![format!("x=i8:{x}"), format!("y=i8:{y}")];
    let rich = vec![
        format!("w0=i8:{}", t.write("t0.bin", &pat(0, 3, vl))),
        format!("raw_t=raw:{}", t.write("t1.bin", &pat(1, 1, vl))),
        format!("f=f16:{}", t.write("t2.bin", &pat(2, 2, vl))),
        format!("abcdefghijklmno=u32:{}", t.write("t3.bin", &pat(3, 1, vl))),
        ex[0].clone(),
        ex[1].clone(),
    ];
    (ex, rich)
}

fn wpack(profile: &str, out: &str, args: &[String]) -> (i32, String, String) {
    let mut a = vec!["wpack", "--profile", profile, "-o", out];
    a.extend(args.iter().map(String::as_str));
    text(&run(&a))
}

#[test]
fn wpack_writes_the_package_tsp_as_wpack_writes() {
    for (suffix, profile, vl) in [("", "ref", 320usize), ("_mini", "mini", 16)] {
        let t = Tmp::new("pack");
        let (ex, rich) = inputs(&t, vl);
        let out = t.path("out.safetensors");
        // example has i8 tensors only; rich has f16 and u32 ones, which warn (stderr only, the package is a byte copy)
        assert_eq!(wpack(profile, &out, &ex), (0, String::new(), String::new()), "example{suffix}");
        assert_eq!(fs::read(&out).unwrap(), bytes(&format!("example{suffix}.safetensors")), "example{suffix}");
        let (code, stdout, stderr) = wpack(profile, &out, &rich);
        assert_eq!((code, stdout.as_str()), (0, ""), "rich{suffix}");
        assert!(!stderr.is_empty() && stderr.lines().all(|l| l.starts_with("wpack: warning: ")), "{stderr}");
        assert_eq!(fs::read(&out).unwrap(), bytes(&format!("rich{suffix}.safetensors")), "rich{suffix}");
    }
}

#[test]
fn wpack_of_no_tensors_writes_a_header_only_package() {
    let t = Tmp::new("empty");
    let out = t.path("e.safetensors");
    assert_eq!(wpack("mini", &out, &[]).0, 0);
    let b = fs::read(&out).unwrap();
    assert!(ots_format::read_weights(&b, Some("mini"), None).unwrap().is_empty());
    assert_eq!(8 + u64::from_le_bytes(b[..8].try_into().unwrap()) as usize, b.len());
}

#[test]
fn wpack_takes_the_word_size_from_the_profile_and_the_config_option() {
    let t = Tmp::new("vl");
    let f = t.write("one.bin", &[1; 16]);
    // 16 bytes are a whole word of mini and not of ref
    let (rc, _, err) = wpack("ref", &t.path("o.safetensors"), &[format!("a=i8:{f}")]);
    assert_eq!(rc, 1);
    assert!(err.starts_with("wpack: OTS-E-ASM-004: "), "{err}");
    assert!(err.contains("320-byte"), "{err}");
    assert_eq!(wpack("mini", &t.path("o.safetensors"), &[format!("a=i8:{f}")]).0, 0);
    // `--config` stands for the profile's file: mini's own file reads the same
    let cfg = format!("{}/../../../opentsp-arch-spec/config/mini.json", env!("CARGO_MANIFEST_DIR"));
    let o = text(&run(&[
        "wpack",
        "--profile",
        "mini",
        "--config",
        &cfg,
        "-o",
        &t.path("c.safetensors"),
        &format!("a=i8:{f}"),
    ]));
    assert_eq!(o.0, 0, "{}", o.2);
    assert_eq!(fs::read(t.path("c.safetensors")).unwrap(), fs::read(t.path("o.safetensors")).unwrap());
}

#[test]
fn wpack_rejections_are_asm_004_exit_1_and_leave_no_file() {
    let t = Tmp::new("rej");
    let ok = t.write("ok.bin", &[7; 16]);
    let short = t.write("short.bin", &[7; 17]);
    let none = t.write("none.bin", &[]);
    let out = t.path("o.safetensors");
    for (what, args, want) in [
        ("odd length", vec![format!("a=i8:{short}")], "not a whole number of 16-byte words"),
        ("empty file", vec![format!("a=i8:{none}")], "there must be one"),
        ("bad name", vec![format!("1a=i8:{ok}")], "identifier"),
        ("long name", vec![format!("abcdefghijklmnop=i8:{ok}")], "identifier"),
        ("empty name", vec![format!("=i8:{ok}")], "identifier"),
        ("unknown dtype", vec![format!("a=f64:{ok}")], "unknown element type"),
        ("empty dtype", vec![format!("a=:{ok}")], "unknown element type"),
        ("second tensor", vec![format!("a=i8:{ok}"), format!("b-=i8:{ok}")], "identifier"),
    ] {
        let (rc, o, e) = wpack("mini", &out, &args);
        assert_eq!((rc, o.as_str()), (1, ""), "{what}: {e}");
        assert!(e.starts_with("wpack: OTS-E-ASM-004: ") && e.contains(want), "{what}: {e}");
        assert!(!std::path::Path::new(&out).exists(), "{what}: a file was written");
    }
    // the check of every file precedes the check of every name (tsp_as reads the files first)
    let (rc, _, e) = wpack("mini", &out, &[format!("1a=i8:{ok}"), format!("b=i8:{short}")]);
    assert_eq!(rc, 1);
    assert!(e.contains("not a whole number"), "{e}");
}

#[test]
fn wpack_duplicate_names_are_asm_004_and_no_file_is_written() {
    let t = Tmp::new("dup");
    let ok = t.write("ok.bin", &[7; 16]);
    let out = t.path("o.safetensors");
    let (rc, o, e) = wpack("mini", &out, &[format!("a=i8:{ok}"), format!("a=i8:{ok}")]);
    assert_eq!((rc, o.as_str()), (1, ""), "{e}");
    assert!(e.starts_with("wpack: OTS-E-ASM-004: ") && e.contains("more than once"), "{e}");
    assert!(!std::path::Path::new(&out).exists(), "a file was written");
    // two names that differ are packed
    assert_eq!(wpack("mini", &out, &[format!("a=i8:{ok}"), format!("b=i8:{ok}")]).0, 0);
}

#[test]
fn wpack_profile_errors_are_asm_003_and_usage_is_exit_2() {
    let t = Tmp::new("usage");
    let ok = t.write("ok.bin", &[7; 16]);
    let (rc, _, e) = wpack("nosuch", &t.path("o.safetensors"), &[format!("a=i8:{ok}")]);
    assert_eq!(rc, 1);
    assert!(e.starts_with("wpack: OTS-E-ASM-003: "), "{e}");
    // no -o (after the files were read), a tensor argument that is not name=dtype:file, `-o` without a value
    let noo = format!("a=i8:{ok}");
    for args in [
        vec!["wpack", "--profile", "mini", noo.as_str()],
        vec!["wpack", "--profile", "mini", "-o", "/nonexistent-dir/o", "a"],
        vec!["wpack", "-o"],
    ] {
        let (rc, o, _) = text(&run(&args));
        assert_eq!((rc, o.as_str()), (2, ""), "{args:?}");
    }
    let (rc, _, e) =
        text(&run(&["wpack", "--profile", "mini", "-o", &t.path("o.safetensors"), &format!("a=i8:{ok}"), "b=i8"]));
    assert_eq!(rc, 2, "{e}");
    assert!(!std::path::Path::new(&t.path("o.safetensors")).exists());
}

#[test]
fn wpack_unreadable_input_or_output_is_an_os_error_exit_1() {
    let t = Tmp::new("io");
    let (rc, _, e) = wpack("mini", &t.path("o.safetensors"), &[format!("a=i8:{}", t.path("missing.bin"))]);
    assert_eq!(rc, 1);
    assert!(e.starts_with("wpack: ") && e.contains("missing.bin"), "{e}");
    let ok = t.write("ok.bin", &[1; 16]);
    let (rc, _, e) = wpack("mini", &t.path("no/such/dir/o.safetensors"), &[format!("a=i8:{ok}")]);
    assert_eq!(rc, 1);
    assert!(e.starts_with("wpack: ") && e.contains("o.safetensors"), "{e}");
}

#[test]
fn objinput_with_weights_prints_the_s_lines_then_the_q_lines_tsp_as_prints() {
    for (obj, pkg, expect) in [
        ("example.elf", "example.safetensors", "example.weights.objinput"),
        ("example.elf", "rich.safetensors", "rich.weights.objinput"),
        ("example_mini.elf", "example_mini.safetensors", "example_mini.weights.objinput"),
        ("example_mini.elf", "rich_mini.safetensors", "rich_mini.weights.objinput"),
        ("bind.elf", "rich.safetensors", "bind.weights.objinput"),
        ("bind_mini.elf", "rich_mini.safetensors", "bind_mini.weights.objinput"),
    ] {
        let (rc, out, err) = text(&run(&["objinput", s(&data(obj)), "--weights", s(&data(pkg))]));
        assert_eq!((rc, err.as_str()), (0, ""), "{obj} {pkg}");
        assert_eq!(out.as_bytes(), bytes(expect).as_slice(), "{obj} {pkg}");
    }
}

#[test]
fn the_example_gets_x_and_y_in_every_lane_and_its_program_as_q_lines() {
    let (rc, out, _) = text(&run(&["objinput", s(&data("example.elf")), "--weights", s(&data("example.safetensors"))]));
    assert_eq!(rc, 0);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[0], format!("S 4 10{}", " 3".repeat(320)));
    assert_eq!(lines[1], format!("S 5 0{}", " 4".repeat(320)));
    assert_eq!(
        &lines[2..],
        bytes("example.objinput")
            .split(|b| *b == b'\n')
            .map(|l| std::str::from_utf8(l).unwrap())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
    );
}

#[test]
fn objinput_with_weights_of_an_object_without_externs_prints_only_the_q_lines() {
    // a text file has no externs and `--profile` (default ref) names the package's profile
    let t = Tmp::new("text");
    let q = t.write("q.txt", b"Q 4 00000021 10000010 00000000\n");
    let (rc, out, err) = text(&run(&["objinput", &q, "--weights", s(&data("rich.safetensors"))]));
    assert_eq!((rc, err.as_str()), (0, ""));
    assert_eq!(out, "Q 4 00000021 10000010 00000000\n");
    // the package must be for that profile
    let (rc, out, err) = text(&run(&["objinput", &q, "--weights", s(&data("rich_mini.safetensors"))]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("objinput: OTS-E-ASM-003: "), "{err}");
    let (rc, _, _) = text(&run(&["objinput", &q, "--profile", "mini", "--weights", s(&data("rich_mini.safetensors"))]));
    assert_eq!(rc, 0);
}

#[test]
fn objinput_with_weights_binding_errors_are_asm_004_and_print_nothing() {
    let t = Tmp::new("bind");
    // a package with only x: y has no tensor
    let x = t.write("x.bin", &[3; 320]);
    let only_x = t.path("only_x.safetensors");
    assert_eq!(wpack("ref", &only_x, &[format!("x=i8:{x}")]).0, 0);
    let (rc, out, err) = text(&run(&["objinput", s(&data("example.elf")), "--weights", &only_x]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("objinput: OTS-E-ASM-004: ") && err.contains("extern y"), "{err}");
    // a tensor of two words for an extern of one
    let two = t.write("two.bin", &[3; 640]);
    let wide = t.path("wide.safetensors");
    assert_eq!(wpack("ref", &wide, &[format!("x=i8:{x}"), format!("y=i8:{two}")]).0, 0);
    let (rc, out, err) = text(&run(&["objinput", s(&data("example.elf")), "--weights", &wide]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("objinput: OTS-E-ASM-004: ") && err.contains("2 words"), "{err}");
    // names are case-sensitive
    let up = t.path("up.safetensors");
    let y = t.write("y.bin", &[4; 320]);
    assert_eq!(wpack("ref", &up, &[format!("X=i8:{x}"), format!("y=i8:{y}")]).0, 0);
    assert_eq!(text(&run(&["objinput", s(&data("example.elf")), "--weights", &up])).0, 1);
}

#[test]
fn objinput_with_weights_format_errors() {
    let t = Tmp::new("fmt");
    let good = bytes("example.safetensors");
    let obj = s(&data("example.elf")).to_string();
    let run_w = |name: &str, b: &[u8]| {
        let p = t.write(name, b);
        text(&run(&["objinput", &obj, "--weights", &p]))
    };
    // a flipped image byte: the digest does not match
    let mut flip = good.clone();
    let last = flip.len() - 1;
    flip[last] ^= 1;
    let (rc, out, err) = run_w("flip.safetensors", &flip);
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("objinput: OTS-E-ASM-004: ") && err.contains("digest"), "{err}");
    // a truncated file, trailing bytes, another magic, another version, a package of another profile
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("truncated", good[..good.len() - 1].to_vec()),
        ("trailing", [good.clone(), vec![0]].concat()),
        ("magic", [b"OTSO".to_vec(), good[4..].to_vec()].concat()),
        ("version", [good[..5].to_vec(), vec![2], good[6..].to_vec()].concat()),
        ("short", good[..31].to_vec()),
        ("empty", vec![]),
        ("mini package", bytes("example_mini.safetensors")),
        ("legacy", b"OTSW0001\x01\0\0\0\x02\0\0\0{}".to_vec()),
    ];
    for (what, b) in cases {
        let (rc, out, err) = run_w(&format!("{what}.safetensors"), &b);
        assert_eq!((rc, out.as_str()), (1, ""), "{what}");
        assert!(err.starts_with("objinput: OTS-E-ASM-003: "), "{what}: {err}");
    }
    // an object that is itself bad is reported before the package is read
    let bad_obj = t.write("bad.elf", b"OTSO");
    let (rc, _, err) = text(&run(&["objinput", &bad_obj, "--weights", &t.path("nonexistent.safetensors")]));
    assert_eq!(rc, 1);
    assert!(err.starts_with("objinput: OTS-E-ASM-003: "), "{err}");
}

#[test]
fn objinput_with_weights_missing_file_is_exit_1_and_a_missing_value_is_usage() {
    let (rc, out, err) = text(&run(&["objinput", s(&data("example.elf")), "--weights", "/nonexistent/w.safetensors"]));
    assert_eq!((rc, out.as_str()), (1, ""));
    assert!(err.starts_with("objinput: ") && err.contains("w.safetensors"), "{err}");
    let (rc, out, _) = text(&run(&["objinput", s(&data("example.elf")), "--weights"]));
    assert_eq!((rc, out.as_str()), (2, ""));
}

#[test]
fn objinput_without_weights_still_prints_only_q_lines_and_dis_ignores_weights() {
    let (rc, out, _) = text(&run(&["objinput", s(&data("example.elf"))]));
    assert_eq!(rc, 0);
    assert_eq!(out.as_bytes(), bytes("example.objinput").as_slice());
    let a = text(&run(&["dis", s(&data("example.elf"))]));
    let b = text(&run(&["dis", s(&data("example.elf")), "--weights", s(&data("example.safetensors"))]));
    assert_eq!(a, b);
}

#[test]
fn the_config_option_stands_for_the_profile_file_when_reading_a_package() {
    let t = Tmp::new("cfg");
    let cfg = format!("{}/../../../opentsp-arch-spec/config/ref.json", env!("CARGO_MANIFEST_DIR"));
    let (rc, out, err) = text(&run(&[
        "objinput",
        s(&data("example.elf")),
        "--config",
        &cfg,
        "--weights",
        s(&data("example.safetensors")),
    ]));
    assert_eq!((rc, err.as_str()), (0, ""));
    assert_eq!(out.as_bytes(), bytes("example.weights.objinput").as_slice());
    // a config that is not valid JSON of a profile: ASM-003
    // (`--no-validate`: without it the configuration is refused first, with status 2, see tests/config_validate.rs)
    let junk = t.write("junk.json", b"{}");
    let (rc, _, err) = text(&run(&[
        "objinput",
        s(&data("example.elf")),
        "--no-validate",
        "--config",
        &junk,
        "--weights",
        s(&data("example.safetensors")),
    ]));
    assert_eq!(rc, 1);
    assert!(err.contains("OTS-E-ASM-003"), "{err}");
}

// ---- wpack --planes and the multi-byte warning ----------------------------------

/// `n` little-endian elements of `w` bytes: element i has byte k = (i * 7 + k * 31 + 1) mod 256.
fn elements(n: usize, w: usize) -> Vec<u8> {
    (0..n).flat_map(|i| (0..w).map(move |k| ((i * 7 + k * 31 + 1) % 256) as u8)).collect()
}

/// The plane words by hand, vector-major (placement (a) of the layout chapter: the `w` plane words of one vector are
/// consecutive words): for vector j of 320 elements and plane k, the word at index j * w + k holds byte k of the
/// elements of vector j.
fn planes_by_hand(raw: &[u8], w: usize) -> Vec<u8> {
    let n = raw.len() / w;
    (0..n / 320).flat_map(|j| (0..w).flat_map(move |k| (0..320).map(move |i| raw[(j * 320 + i) * w + k]))).collect()
}

#[test]
fn planes_of_two_vectors_are_vector_major() {
    // 640 f16 values: vector 0 is 0x0100 + i, vector 1 is 0x0200 + i (low byte i, high byte 1 or 2)
    let t = Tmp::new("planes-two");
    let mut raw = Vec::new();
    for v in 1..=2u16 {
        for i in 0..320u16 {
            raw.extend((v << 8 | (i & 0xff)).to_le_bytes());
        }
    }
    let file = t.write("two.bin", &raw);
    let out = t.path("two.safetensors");
    let spec = format!("x=f16:{file}");
    assert_eq!(
        text(&run(&["wpack", "--profile", "ref", "--planes", "-o", &out, &spec])),
        (0, String::new(), String::new())
    );
    let pkg = fs::read(&out).unwrap();
    // the four plane words are the last four words of the package, in the order v0p0, v0p1, v1p0, v1p1
    let words: Vec<&[u8]> = pkg[pkg.len() - 4 * 320..].chunks(320).collect();
    let want_low: Vec<u8> = (0..320).map(|i| (i & 0xff) as u8).collect();
    assert_eq!(words[0], want_low.as_slice(), "word 0 is plane 0 of vector 0");
    assert!(words[1].iter().all(|&b| b == 1), "word 1 is plane 1 (high bytes) of vector 0");
    assert_eq!(words[2], want_low.as_slice(), "word 2 is plane 0 of vector 1");
    assert!(words[3].iter().all(|&b| b == 2), "word 3 is plane 1 (high bytes) of vector 1");
}

#[test]
fn planes_converts_interleaved_elements_to_plane_words() {
    let t = Tmp::new("planes");
    for (dtype, w, n) in [("f16", 2, 320), ("f32", 4, 640), ("i16", 2, 320), ("i8", 1, 320)] {
        let raw = elements(n, w);
        let file = t.write("el.bin", &raw);
        let hand = t.write("pl.bin", &planes_by_hand(&raw, w));
        let (out_a, out_b) = (t.path("a.safetensors"), t.path("b.safetensors"));
        let spec_a = format!("x={dtype}:{file}");
        let spec_b = format!("x={dtype}:{hand}");
        let a = run(&["wpack", "--profile", "ref", "--planes", "-o", &out_a, &spec_a]);
        assert_eq!(text(&a), (0, String::new(), String::new()), "{dtype}");
        let b = run(&["wpack", "--profile", "ref", "--raw-bytes", "-o", &out_b, &spec_b]);
        assert_eq!(text(&b), (0, String::new(), String::new()), "{dtype}");
        assert_eq!(fs::read(&out_a).unwrap(), fs::read(&out_b).unwrap(), "{dtype}: the package is the plane words");
        if w > 1 {
            // and it is not the byte copy of the interleaved file
            let (out_c, spec_c) = (t.path("c.safetensors"), format!("x={dtype}:{file}"));
            run(&["wpack", "--profile", "ref", "--raw-bytes", "-o", &out_c, &spec_c]);
            assert_ne!(fs::read(&out_a).unwrap(), fs::read(&out_c).unwrap(), "{dtype}");
        }
    }
}

#[test]
fn a_multi_byte_dtype_without_planes_or_raw_bytes_warns_and_copies() {
    let t = Tmp::new("warn");
    let raw = elements(320, 2);
    let file = t.write("el.bin", &raw);
    let spec = format!("x=f16:{file}");
    let out = t.path("w.safetensors");
    let (code, stdout, stderr) = text(&run(&["wpack", "--profile", "ref", "-o", &out, &spec]));
    assert_eq!((code, stdout.as_str()), (0, ""));
    assert!(stderr.starts_with("wpack: warning: x: f16 is 2-byte data"), "{stderr}");
    assert!(stderr.contains("--planes") && stderr.contains("--raw-bytes"), "{stderr}");
    // the bytes are those of the silent forms: still a byte copy
    let quiet = t.path("q.safetensors");
    let q = run(&["wpack", "--profile", "ref", "--raw-bytes", "-o", &quiet, &spec]);
    assert_eq!(text(&q), (0, String::new(), String::new()));
    assert_eq!(fs::read(&out).unwrap(), fs::read(&quiet).unwrap());
    // one-byte and raw tensors never warn
    for spec in [format!("x=i8:{file}"), format!("x=raw:{file}")] {
        let (_, _, stderr) = text(&run(&["wpack", "--profile", "ref", "-o", &out, &spec]));
        assert_eq!(stderr, "", "{spec}");
    }
}

#[test]
fn planes_refuses_raw_ragged_and_conflicting_input() {
    let t = Tmp::new("planes-bad");
    let ragged = t.write("ragged.bin", &elements(300, 2)); // 300 elements, not a whole number of 320
    let whole = t.write("whole.bin", &elements(320, 2));
    let out = t.path("w.safetensors");
    let (code, _, err) = text(&run(&["wpack", "--profile", "ref", "--planes", "-o", &out, &format!("x=f16:{ragged}")]));
    assert_eq!(code, 1);
    assert!(err.contains("OTS-E-ASM-004") && err.contains("not a whole number of 320 elements of 2 bytes"), "{err}");
    let (code, _, err) = text(&run(&["wpack", "--profile", "ref", "--planes", "-o", &out, &format!("x=raw:{whole}")]));
    assert_eq!(code, 1);
    assert!(err.contains("OTS-E-ASM-004") && err.contains("--planes needs an element type"), "{err}");
    let (code, _, err) =
        text(&run(&["wpack", "--profile", "ref", "--planes", "--raw-bytes", "-o", &out, &format!("x=f16:{whole}")]));
    assert_eq!(code, 2);
    assert!(err.contains("--planes and --raw-bytes exclude one another"), "{err}");
}

// Widths follow the assigned element formats, independently of the CLI helper.
const ELEMENT_WIDTHS: &[(&str, usize)] = &[
    ("i8", 1),
    ("u8", 1),
    ("b8", 1),
    ("e4m3", 1),
    ("e5m2", 1),
    ("f16", 2),
    ("bf16", 2),
    ("i16", 2),
    ("u16", 2),
    ("b16", 2),
    ("i32", 4),
    ("u32", 4),
    ("f32", 4),
    ("b32", 4),
];

#[test]
fn every_assigned_element_type_packs_two_vectors_in_byte_plane_order() {
    let mut assigned: Vec<_> = ots_isa::ELTYPE_CODES.iter().map(|&(name, _)| name).collect();
    let mut tested: Vec<_> = ELEMENT_WIDTHS.iter().map(|&(name, _)| name).collect();
    assigned.sort_unstable();
    tested.sort_unstable();
    assert_eq!(assigned, tested, "every assigned type needs an explicit width/layout case");
    let t = Tmp::new("all-element-planes");
    for (profile, lanes) in [("mini", 16), ("ref", 320)] {
        for &(dtype, width) in ELEMENT_WIDTHS {
            let raw = elements(2 * lanes, width);
            let file = t.write("elements.bin", &raw);
            let spec = format!("x={dtype}:{file}");
            let out = t.path("planes.safetensors");
            assert_eq!(
                text(&run(&["wpack", "--profile", profile, "--planes", "-o", &out, &spec])),
                (0, String::new(), String::new()),
                "{profile}/{dtype}"
            );
            let package = fs::read(&out).unwrap();
            // Read the actual Safetensors file with the existing format reader, then independently
            // check every byte of each vector-major plane, including vector 1.
            let tensors = ots_format::read_weights(&package, Some(profile), None).unwrap();
            assert_eq!(tensors.len(), 1);
            let tensor = &tensors["x"];
            assert_eq!(
                u32::from(tensor.dtype),
                ots_isa::ELTYPE_CODES.iter().find(|&&(name, _)| name == dtype).unwrap().1
            );
            let payload = &package[package.len() - raw.len()..];
            for vector in 0..2 {
                for plane in 0..width {
                    for lane in 0..lanes {
                        assert_eq!(
                            payload[(vector * width + plane) * lanes + lane],
                            raw[(vector * lanes + lane) * width + plane],
                            "{profile}/{dtype}/{vector}/{plane}/{lane}"
                        );
                    }
                }
            }
            let raw_out = t.path("raw.safetensors");
            let plain_out = t.path("plain.safetensors");
            let raw_result = text(&run(&["wpack", "--profile", profile, "--raw-bytes", "-o", &raw_out, &spec]));
            assert_eq!(raw_result, (0, String::new(), String::new()));
            let (code, stdout, warning) = text(&run(&["wpack", "--profile", profile, "-o", &plain_out, &spec]));
            assert_eq!((code, stdout.as_str()), (0, ""));
            assert_eq!(
                fs::read(&plain_out).unwrap(),
                fs::read(&raw_out).unwrap(),
                "{profile}/{dtype}: default is byte copy"
            );
            if width > 1 {
                assert!(warning.starts_with(&format!("wpack: warning: x: {dtype} is {width}-byte data")), "{warning}");
                assert!(warning.contains("--planes") && warning.contains("--raw-bytes"), "{warning}");
                assert_ne!(package, fs::read(&raw_out).unwrap(), "{profile}/{dtype}: converted bytes differ");
            } else {
                assert!(warning.is_empty());
                assert_eq!(package, fs::read(&raw_out).unwrap());
            }
        }
    }
}

#[test]
fn every_element_type_rejects_incomplete_vectors_before_writing_a_package() {
    let t = Tmp::new("all-element-ragged");
    for &(dtype, width) in ELEMENT_WIDTHS {
        let mut raw = elements(16, width);
        raw.pop();
        let file = t.write("ragged.bin", &raw);
        let out = t.path("absent.safetensors");
        let spec = format!("x={dtype}:{file}");
        let (code, stdout, err) = text(&run(&["wpack", "--profile", "mini", "--planes", "-o", &out, &spec]));
        assert_eq!((code, stdout.as_str()), (1, ""));
        assert!(
            err.starts_with("wpack: OTS-E-ASM-004: ") && err.contains(&format!("16 elements of {width} byte")),
            "{dtype}: {err}"
        );
        assert!(!Path::new(&out).exists());
    }
}
