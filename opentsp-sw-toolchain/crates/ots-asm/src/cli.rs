//! The command line (`tsp_as.main`): `-o`, `--profile`, `--config` and `--weights` take a value, any
//! other `--x` is a flag, everything else is positional. The commands are `insn`, `word`, `asm`
//! (`--input`, `-o`), `dis`, `objinput` (`--weights`), `wpack`, `mpack` and `mobjinput`; `--weights`
//! may repeat, every value is kept in order.
use crate::{
    load_profile,
    pynum::{hex_alt, py_int},
};
use num_bigint::BigInt;
use ots_format::{
    bind_weights, manifest_input, preload_text, read_object, read_weights,
    text::{is_text_input, parse_images, Images},
    weights::TensorIn,
    write_manifest_for, write_weights, Extern, MAX_CHIPS, MAX_WIRED_CHIPS,
};
use ots_isa::{render, AsmError, Code, Isa};
use std::collections::HashMap;

pub const USAGE: &str = "usage:
  every command: --config F is validated like tsp-iss does (the schema and timing rules); --no-validate skips it
  ots-asm insn [--profile P] [--config F] <text>             one instruction -> \"<width> <hex>\"
  ots-asm word [--profile P] [--config F] <width> <hex>      one word -> instruction text
  ots-asm asm [--config F] [--input] [-o out.elf] <prog.ots>
                                                             assemble; --input prints the Q lines of the lowering,
                                                             -o writes the object (format revision 2)
  ots-asm dis [--profile P] [--config F] <obj.elf | text.txt>
                                                             the listing of an object or of a text image (Q lines)
  ots-asm objinput [--profile P] [--config F] [--weights W.safetensors] <obj.elf | text.txt>
                                                             the harness input: the Q lines of an object or text,
                                                             preceded by the S lines of its externs with --weights
  ots-asm wpack --profile P [--config F] [--planes | --raw-bytes] -o W.safetensors name=dtype:file ...
                                                             pack a weight package (dtype: an element type or raw;
                                                             file: raw bytes, a whole number of vlanes-byte words;
                                                             --planes: file = little-endian elements, converted to plane
                                                             words, vector-major: the w plane words of one vector of vlanes
                                                             elements are consecutive words, plane 0 first, then the next
                                                             vector; a multi-byte dtype without either option warns)
  ots-asm mpack [--profile P] [--config F] [--topology T.json] -o M.otsm <chip0.elf> ...
                                                             pack a multi-chip manifest (default profile ref): K objects
                                                             of profile P in chip order, K = 1..264, or the chips of the
                                                             topology file (at most 512)
  ots-asm patch [--profile P] [--config F] --table T.json --values V.json -o out.elf <obj.elf>
                                                             write the values of V into the operands the patch table T
                                                             names (format in the toolchain README) and re-check the
                                                             parcels it touched; one line per patched site on stdout
  ots-asm mobjinput [--config F] <M.otsm> [--weights chip0.safetensors --weights chip1.safetensors ...]
                                                             manifest -> `tsp_sim --chips K` input: per chip a `C <chip>`
                                                             line, its S lines (one package per chip, in order, or none)
                                                             and its Q lines
";

pub struct Opts {
    pub args: Vec<String>,
    pub opts: HashMap<String, Option<String>>,
    /// Every `--weights` value, in order (`opts` keeps the last one, as `objinput` takes it).
    pub weights: Vec<String>,
}

/// `Err(option)` when an option that takes a value has none.
pub fn parse_args(argv: &[String]) -> Result<Opts, String> {
    let mut args = Vec::new();
    let mut opts = HashMap::new();
    let mut weights = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if matches!(a.as_str(), "-o" | "--profile" | "--config" | "--weights" | "--table" | "--values" | "--topology") {
            let v = argv.get(i + 1).ok_or_else(|| a.clone())?;
            opts.insert(a.clone(), Some(v.clone()));
            if a == "--weights" {
                weights.push(v.clone());
            }
            i += 2;
        } else if a.starts_with("--") {
            opts.insert(a.clone(), None);
            i += 1;
        } else {
            args.push(a.clone());
            i += 1;
        }
    }
    Ok(Opts { args, opts, weights })
}

fn value<'a>(o: &'a Opts, k: &str) -> Option<&'a str> {
    o.opts.get(k).and_then(|v| v.as_deref())
}

/// The canonical text of one word, or `illegal<width> 0x<hex>` (`ISA.disasm`).
pub fn disasm(isa: &Isa, width: &BigInt, w: &BigInt) -> String {
    use num_bigint::ToBigInt;
    let fits = |bits: u32| *w >= BigInt::from(0) && *w < (1u64 << bits).to_bigint().unwrap();
    // a word with bits above its width never round-trips, so it is not an instruction
    let insn = if *width == BigInt::from(32) && fits(32) {
        isa.decode(32, w.to_string().parse::<u64>().unwrap())
    } else if *width == BigInt::from(64) && *w >= BigInt::from(0) && *w < (BigInt::from(1) << 64u32) {
        isa.decode(64, w.to_string().parse::<u64>().unwrap())
    } else {
        None
    };
    match insn {
        Some(i) => render(&i),
        None => format!("illegal{width} {}", hex_alt(w)),
    }
}

fn fail(cmd: &str, e: &AsmError) -> i32 {
    eprintln!("{}", e.render(cmd));
    1
}

/// `asm <prog.ots> [--input] [-o out.elf] [--config F]`: warnings on stderr as they arise, then the Q
/// lines of the lowering on stdout with `--input`, then the object file with `-o`.
///
/// A parcel keeps its instructions in source order (`assembly.md` §4).
fn asm(o: &Opts, rest: &[String]) -> i32 {
    use crate::program::assemble;
    let Some(path) = rest.first() else {
        eprint!("{USAGE}");
        return 2;
    };
    let src = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("asm: OTS-E-ASM-003: {path}: {e}");
            return 1;
        }
    };
    let mut warn = |w: crate::Warning| eprintln!("{}", w.render());
    match assemble(&src, path, value(o, "--config"), &mut warn) {
        Ok((isa, prog)) => {
            let images = prog.images();
            if o.opts.contains_key("--input") {
                print!("{}", ots_format::text::input_text(&images));
            }
            if let Some(out) = value(o, "-o") {
                let bytes = ots_format::write_object(isa.profile(), &images, &prog.externs);
                if let Err(e) = std::fs::write(out, bytes) {
                    eprintln!("asm: OTS-E-ASM-003: {out}: {e}");
                    return 1;
                }
            }
            0
        }
        Err(e) => fail("asm", &e),
    }
}

/// The input of `dis` and `objinput` (`tsp_as.main`): `(profile, images, externs)`. A text file (first
/// non-blank byte `Q`, or blank) is ASCII `Q` lines for `--profile` (default `ref`, not loaded here);
/// anything else is an object, which names its own profile, `--profile` having to equal it.
/// `Err(status)` has printed its diagnostic.
fn input(cmd: &str, o: &Opts, rest: &[String], profile: &str) -> Result<(String, Images, Vec<Extern>), i32> {
    let Some(path) = rest.first() else {
        eprint!("{USAGE}");
        return Err(2);
    };
    let data = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{cmd}: OTS-E-ASM-003: {path}: {e}");
            return Err(1);
        }
    };
    if is_text_input(&data) {
        let Some(txt) = std::str::from_utf8(&data).ok().filter(|t| t.is_ascii()) else {
            return Err(fail(cmd, &AsmError::new(Code::Asm003, format!("{path}: text input is not ASCII"))));
        };
        let images = parse_images(txt, path).map_err(|e| fail(cmd, &e))?;
        Ok((profile.to_string(), images, Vec::new()))
    } else {
        read_object(&data, value(o, "--profile"), value(o, "--config")).map_err(|e| fail(cmd, &e))
    }
}

/// `dis <text.txt | obj.elf> [--profile P] [--config F]`: the listing on stdout; exit 1 when an
/// instruction is annotated.
fn dis(o: &Opts, rest: &[String], profile: &str) -> i32 {
    let (prof, images, externs) = match input("dis", o, rest, profile) {
        Ok(x) => x,
        Err(rc) => return rc,
    };
    let config = value(o, "--config");
    let isa = match load_profile(&prof, config) {
        Ok(p) => Isa::with_where(
            &p,
            match config {
                Some(path) => format!("file {path}"),
                None => format!("profile {}", p.name()),
            },
        ),
        Err(e) => return fail("dis", &e),
    };
    let (listing, bad) = crate::dis::disassemble(&isa, &images, &externs);
    print!("{listing}");
    i32::from(bad > 0)
}

/// `objinput <text.txt | obj.elf> [--profile P] [--config F] [--weights W.safetensors]`: the `Q` lines of
/// the images (the object is checked in full; a text file is not tied to a profile). With `--weights`
/// the `S` lines of the externs bound to the tensors of the package come first: the package must be for
/// the object's profile (for a text file, `--profile`, default `ref`), and nothing is printed when the
/// package is refused or an extern cannot be bound.
fn objinput(o: &Opts, rest: &[String], profile: &str) -> i32 {
    let (prof, images, externs) = match input("objinput", o, rest, profile) {
        Ok(x) => x,
        Err(rc) => return rc,
    };
    let mut out = String::new();
    let cfg = value(o, "--config");
    let raw = match std::fs::read(&rest[0]) {
        Ok(x) => x,
        Err(e) => return fail("objinput", &AsmError::new(Code::Asm003, e.to_string())),
    };
    let mut map = match ots_format::object_preloads(&raw, &prof, &externs, cfg, 1) {
        Ok(x) => x,
        Err(e) => return fail("objinput", &e),
    };
    if let Some(w) = value(o, "--weights") {
        let data = match std::fs::read(w) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("objinput: OTS-E-ASM-003: {w}: {e}");
                return 1;
            }
        };
        if let Err(e) = read_weights(&data, Some(&prof), cfg).and_then(|t| ots_format::merge_preloads(&mut map, t)) {
            return fail("objinput", &e);
        }
    }
    if !map.is_empty() || value(o, "--weights").is_some() {
        match bind_weights(&externs, &map) {
            Ok(p) => out.push_str(&preload_text(&p)),
            Err(e) => return fail("objinput", &e),
        }
    }
    out.push_str(&ots_format::text::input_text(&images));
    print!("{out}");
    0
}

/// `wpack --profile P [--config F] -o W.safetensors name=dtype:file ...`: a weight package. `dtype` is an
/// element type of `encoding.json` or `raw`; `file` holds the words, `vlanes` bytes each, lane 0 first.
///
/// The order of `tsp_as.main`: the profile (`ASM-003`), then every file in argument order (an OS error
/// is exit 1; a length that is not a whole number of words is `ASM-004`), then `-o` (usage when absent),
/// then the package's own checks ([`write_weights`]). The package is written only when all of them
/// pass. An argument that is not `name=dtype:file` is usage (exit 2): `tsp_as.py` ends in a traceback.
/// The bytes of one element of an element type of `encoding.json` (`None` for `raw` or an unknown name).
fn element_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "i8" | "u8" | "b8" | "e4m3" | "e5m2" => Some(1),
        "f16" | "bf16" | "i16" | "u16" | "b16" => Some(2),
        "i32" | "u32" | "f32" | "b32" => Some(4),
        _ => None,
    }
}

/// Split little-endian elements (`w` bytes each) into vector-major plane words: word `j * w + k`
/// holds byte `k` of all `vl` lanes of vector `j`. This rearranges bits without numeric conversion.
/// `None` when the elements are not a whole number of vectors.
fn to_plane_words(raw: &[u8], w: usize, vl: usize) -> Option<Vec<Vec<u8>>> {
    if raw.is_empty() || !raw.len().is_multiple_of(w * vl) {
        return None;
    }
    // vector-major (placement (a) of the layout chapter): the `w` plane words of one vector are consecutive, so plane k of
    // vector j is word j * w + k
    let mut words = Vec::with_capacity(raw.len() / vl);
    for vector in raw.chunks_exact(w * vl) {
        for k in 0..w {
            words.push((0..vl).map(|i| vector[i * w + k]).collect());
        }
    }
    Some(words)
}

fn wpack(o: &Opts, rest: &[String], profile: &str) -> i32 {
    let prof = match load_profile(profile, value(o, "--config")) {
        Ok(p) => p,
        Err(e) => return fail("wpack", &e),
    };
    let vl = prof.vlanes() as usize;
    let planes = o.opts.contains_key("--planes");
    let raw_bytes = o.opts.contains_key("--raw-bytes");
    if planes && raw_bytes {
        eprintln!("ots-asm wpack: --planes and --raw-bytes exclude one another\n{USAGE}");
        return 2;
    }
    let mut specs = Vec::new();
    for a in rest {
        let parsed = a
            .split_once('=')
            .and_then(|(name, spec)| spec.split_once(':').map(|(d, p)| (name, d, p)))
            .filter(|(_, _, p)| !p.is_empty());
        let Some((name, dtype, path)) = parsed else {
            eprintln!("ots-asm wpack: {a:?}: expected name=dtype:file\n{USAGE}");
            return 2;
        };
        specs.push((name, dtype, path));
    }
    let mut tensors = Vec::new();
    for (name, dtype, path) in specs {
        let raw = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("wpack: OTS-E-ASM-003: {path}: {e}");
                return 1;
            }
        };
        let width = element_bytes(dtype);
        let words = if planes {
            // an element-interleaved little-endian file -> plane words (plane k = byte k of each element, plane 0 first)
            let Some(w) = width else {
                return fail(
                    "wpack",
                    &AsmError::new(Code::Asm004, format!("{name}: --planes needs an element type, not {dtype:?}")),
                );
            };
            match to_plane_words(&raw, w, vl) {
                Some(words) => words,
                None => {
                    return fail(
                        "wpack",
                        &AsmError::new(
                            Code::Asm004,
                            format!(
                                "{path}: {} bytes is not a whole number of {vl} elements of {w} byte{}",
                                raw.len(),
                                if w == 1 { "" } else { "s" }
                            ),
                        ),
                    )
                }
            }
        } else {
            if raw.len() % vl != 0 {
                return fail(
                    "wpack",
                    &AsmError::new(
                        Code::Asm004,
                        format!("{path}: {} bytes is not a whole number of {vl}-byte words", raw.len()),
                    ),
                );
            }
            if width.is_some_and(|w| w > 1) && !raw_bytes {
                // the file is a byte copy; a producer that wrote elements one after the other has not made plane words
                eprintln!(
                    "wpack: warning: {name}: {dtype} is {}-byte data but {path} is copied as it is, so it must already hold \
                     plane words (word k of the tensor = byte k of {vl} elements); give --planes to convert an \
                     element-interleaved little-endian file, or --raw-bytes if the file is already plane words",
                    width.unwrap_or(1)
                );
            }
            raw.chunks_exact(vl).map(<[u8]>::to_vec).collect()
        };
        tensors.push(TensorIn { name: name.to_string(), dtype: (dtype != "raw").then(|| dtype.to_string()), words });
    }
    let Some(out) = value(o, "-o") else {
        eprint!("ots-asm wpack: -o <W.safetensors> is required\n{USAGE}");
        return 2;
    };
    match write_weights(&prof, &tensors) {
        Ok(bytes) => match std::fs::write(out, bytes) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("wpack: OTS-E-ASM-003: {out}: {e}");
                1
            }
        },
        Err(e) => fail("wpack", &e),
    }
}

/// The `chips` of a topology file (`tsp_sim --topology`).
fn topology_chips(path: &str) -> Result<u64, String> {
    let text = std::fs::read(path).map_err(|e| e.to_string())?;
    let v: serde_json::Value = serde_json::from_slice(&text).map_err(|_| "not a JSON object".to_string())?;
    match v.get("chips").and_then(serde_json::Value::as_u64) {
        Some(k) if (1..=MAX_CHIPS).contains(&k) => Ok(k),
        _ => Err(format!("\"chips\" must be an integer 1..{MAX_CHIPS}")),
    }
}

/// `mpack [--profile P] [--config F] -o M.otsm <chip0.elf> ...`: a manifest of the objects, in chip
/// order, for `P` (default `ref`).
///
/// The order of `tsp_as.main`: `-o` (usage when absent), then every object file in argument order (an OS
/// error is exit 1), then the manifest's own checks ([`write_manifest_for`]: the chip count, then every
/// object, the chip named; `ASM-003`). The manifest is written only when all of them pass (`tsp_as.py`
/// creates the file first and leaves it empty when a check fails).
fn mpack(o: &Opts, rest: &[String], profile: &str) -> i32 {
    let Some(out) = value(o, "-o") else {
        eprint!("ots-asm mpack: -o <M.otsm> is required\n{USAGE}");
        return 2;
    };
    let mut objects = Vec::new();
    for path in rest {
        match std::fs::read(path) {
            Ok(b) => objects.push(b),
            Err(e) => {
                eprintln!("mpack: OTS-E-ASM-003: {path}: {e}");
                return 1;
            }
        }
    }
    let mut max_chips = MAX_WIRED_CHIPS;
    if let Some(t) = value(o, "--topology") {
        // the topology file gives the chip count, and the objects must be that many; the harness checks the rest of the file
        match topology_chips(t) {
            Ok(k) if k == objects.len() as u64 => max_chips = MAX_CHIPS,
            Ok(k) => {
                eprintln!(
                    "mpack: OTS-E-ASM-003: --topology {t} names {k} chips but {} objects were given",
                    objects.len()
                );
                return 1;
            }
            Err(m) => {
                eprintln!("mpack: OTS-E-ASM-003: --topology {t}: {m}");
                return 1;
            }
        }
    }
    match write_manifest_for(profile, &objects, value(o, "--config"), max_chips) {
        Ok(bytes) => match std::fs::write(out, bytes) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("mpack: OTS-E-ASM-003: {out}: {e}");
                1
            }
        },
        Err(e) => fail("mpack", &e),
    }
}

/// `mobjinput <M.otsm> [--config F] [--weights W.safetensors ...]`: the input of `tsp_sim --chips K` (see
/// [`manifest_input`]). The manifest file is read first, then every package in argument order (an OS
/// error is exit 1), then the manifest is checked; nothing is printed when anything fails. `--profile` is
/// not used: the manifest names its profile.
fn mobjinput(o: &Opts, rest: &[String]) -> i32 {
    let Some(path) = rest.first() else {
        eprint!("{USAGE}");
        return 2;
    };
    let data = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("mobjinput: OTS-E-ASM-003: {path}: {e}");
            return 1;
        }
    };
    let mut packages = Vec::new();
    for w in &o.weights {
        match std::fs::read(w) {
            Ok(b) => packages.push(b),
            Err(e) => {
                eprintln!("mobjinput: OTS-E-ASM-003: {w}: {e}");
                return 1;
            }
        }
    }
    match manifest_input(&data, &packages, value(o, "--config")) {
        Ok(text) => {
            print!("{text}");
            0
        }
        Err(e) => fail("mobjinput", &e),
    }
}

/// The configuration of `--config F` is validated as `tsp_sim` and `tsp-iss` validate it (`Profile::validate`: the schema,
/// its relations and the timing rules), so the assemblers do not accept a profile the model refuses.
/// `--no-validate` skips the check. A file that cannot be read is left to the command, which reports it as before.
/// Prints `ots-asm: config F: <message>` and returns 2, the status and message of `tsp-iss`, or `None`.
fn check_config(o: &Opts) -> Option<i32> {
    if o.opts.contains_key("--no-validate") {
        return None;
    }
    let path = value(o, "--config")?;
    // the limit of `tsp_config_load`: a file of 2^25 bytes or more is refused
    let bytes = std::fs::read(path).ok()?;
    let result = if bytes.len() >= 1 << 25 {
        Err("file too large".to_string())
    } else {
        ots_profile::validate_bytes(&bytes).map_err(|e| e.message)
    };
    match result {
        Ok(()) => None,
        Err(m) => {
            eprintln!("ots-asm: config {path}: {m}");
            Some(2)
        }
    }
}

/// `patch <obj.elf> --table T.json --values V.json -o out.elf [--profile P] [--config F]`: the values of a values file
/// written into the operands the patch table names, the parcels that hold them checked again ([`crate::patch`]).
fn patch(o: &Opts, rest: &[String]) -> i32 {
    let (Some(obj), Some(table), Some(values), Some(out)) =
        (rest.first(), value(o, "--table"), value(o, "--values"), value(o, "-o"))
    else {
        eprint!("{USAGE}");
        return 2;
    };
    let read = |p: &str| {
        std::fs::read(p).map_err(|e| {
            eprintln!("patch: {p}: {e}");
            1
        })
    };
    let (data, table, values) = match (read(obj), read(table), read(values)) {
        (Ok(d), Ok(t), Ok(v)) => (d, t, v),
        (Err(rc), ..) | (_, Err(rc), _) | (.., Err(rc)) => return rc,
    };
    let run = || {
        let sites = crate::patch::parse_table(&table)?;
        let values = crate::patch::parse_values(&values)?;
        crate::patch::patch(&data, value(o, "--profile"), value(o, "--config"), &sites, &values)
    };
    match run() {
        Ok((bytes, report)) => {
            if let Err(e) = std::fs::write(out, bytes) {
                eprintln!("patch: {out}: {e}");
                return 1;
            }
            for l in report {
                println!("{l}");
            }
            0
        }
        Err(e) => fail("patch", &e),
    }
}

/// Run the command line (without the program name); returns the exit status: 0, 1 for a
/// diagnostic, 2 for usage.
pub fn run(argv: &[String]) -> i32 {
    if matches!(argv, [arg] if arg == "--help" || arg == "help") {
        print!("{USAGE}");
        return 0;
    }
    let o = match parse_args(argv) {
        Ok(o) => o,
        Err(opt) => {
            eprintln!("ots-asm: option {opt} needs a value\n{USAGE}");
            return 2;
        }
    };
    let Some(cmd) = o.args.first().map(String::as_str) else {
        eprint!("{USAGE}");
        return 2;
    };
    let rest = &o.args[1..];
    let profile = value(&o, "--profile").unwrap_or("ref");
    if let Some(code) = check_config(&o) {
        return code;
    }
    match cmd {
        "insn" => {
            let config = value(&o, "--config");
            let isa = match load_profile(profile, config) {
                Ok(p) => Isa::with_where(
                    &p,
                    match config {
                        Some(path) => format!("file {path}"),
                        None => format!("profile {}", p.name()),
                    },
                ),
                Err(e) => return fail(cmd, &e),
            };
            match crate::parse_insn(&isa, &rest.join(" "), &HashMap::new(), None) {
                Ok((i, w)) => {
                    println!("{} {w:x}", i.entry().width);
                    0
                }
                Err(e) => fail(cmd, &e),
            }
        }
        "word" => {
            let config = value(&o, "--config");
            let isa = match load_profile(profile, config) {
                Ok(p) => Isa::with_where(
                    &p,
                    match config {
                        Some(path) => format!("file {path}"),
                        None => format!("profile {}", p.name()),
                    },
                ),
                Err(e) => return fail(cmd, &e),
            };
            let (Some(width), Some(w)) =
                (rest.first().and_then(|s| py_int(s, 10)), rest.get(1).and_then(|s| py_int(s, 16)))
            else {
                eprintln!("word: OTS-E-ASM-001: expected <width> <hex>, two integers");
                return 1;
            };
            let bits: u32 = if width == BigInt::from(32) {
                32
            } else if width == BigInt::from(64) {
                64
            } else {
                eprintln!("word: OTS-E-ASM-001: width {width} is not 32 or 64");
                return 1;
            };
            if w < BigInt::from(0) || (&w >> bits) != BigInt::from(0) {
                eprintln!("word: OTS-E-ASM-001: value {} does not fit in {width} bits", hex_alt(&w));
                return 1;
            }
            println!("{}", disasm(&isa, &width, &w));
            0
        }
        "asm" => asm(&o, rest),
        "dis" => dis(&o, rest, profile),
        "objinput" => objinput(&o, rest, profile),
        "wpack" => wpack(&o, rest, profile),
        "mpack" => mpack(&o, rest, profile),
        "mobjinput" => mobjinput(&o, rest),
        "patch" => patch(&o, rest),
        _ => {
            eprint!("{USAGE}");
            2
        }
    }
}
