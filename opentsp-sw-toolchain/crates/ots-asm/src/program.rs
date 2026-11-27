//! Program files `.ots` (`assembly.md` §3): the line grammar, the externs, the bundles, and the cycle
//! view they make, lowered to the text of each queue. A port of `Program`, `assemble` and
//! `extern_decl` of `opentsp-arch-spec/tools/tsp_as.py`.
//!
//! The source is bytes read as Latin-1 and split at `\n`; a tab or a byte above 0x7F on any line
//! (comments included) is `OTS-E-ASM-001`, a CR is an ordinary character. Warnings go to the
//! caller as they arise (address warnings while a bundle is read, AGT and slot warnings when it
//! closes, the `iq_depth` warning after the whole file); the first error aborts.
use crate::checks::{agt_bundle, bundle_checks, Agt, Group, Warning};
use crate::insn::{parse_insn, sat_dec, sat_hex};
use crate::load_profile;
use ots_format::{check_extern, text::Images, Extern};
use ots_isa::{queue_id, queue_name, AsmError, Code, Isa};
use regex::Regex;
use std::{collections::HashMap, sync::OnceLock};

/// The architecture a program names.
pub const ARCH: &str = "opentsp-tsa";

/// One instruction of the cycle view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub cycle: u64,
    pub queue: u16,
    /// The 64-bit word (a 32-bit layout in bits 31..0).
    pub word: u64,
}

/// An assembled program: the profile name, the externs in declaration order, and the cycle view in
/// source order.
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub profile: String,
    pub externs: Vec<Extern>,
    pub records: Vec<Record>,
}

impl Program {
    /// The records in the order of the text: stably by (cycle, queue), so the instructions of one
    /// parcel (one queue, one bundle) stay in **source order** (`assembly.md` §4;
    /// `tsp_as.py` sorts the same way).
    pub fn ordered(&self) -> Vec<Record> {
        let mut r = self.records.clone();
        r.sort_by_key(|x| (x.cycle, x.queue));
        r
    }

    /// The text of each queue: the lowering of the cycle view (`ots_format::text::lower`).
    pub fn images(&self) -> Images {
        let entries: Vec<(u64, u16, u64)> = self.ordered().iter().map(|r| (r.cycle, r.queue, r.word)).collect();
        ots_format::text::lower(&entries).expect("a program has valid queues")
    }
}

/// Python's `str.isspace` on ASCII: the characters `\s` matches in a `str` pattern.
const NOT_SPACE: &str = "[^ \\t\\n\\r\\x0b\\x0c\\x1c-\\x1f]";

struct Res {
    arch: Regex,
    profile: Regex,
    ext: Regex,
    section: Regex,
    bundle: Regex,
    slot: Regex,
    extern_decl: Regex,
}

fn res() -> &'static Res {
    static R: OnceLock<Res> = OnceLock::new();
    R.get_or_init(|| {
        let re = |p: String| Regex::new(&format!("\\A(?:{p})\\z")).unwrap();
        let hex = "0x[0-9a-fA-F]+";
        Res {
            arch: re(format!("\\.arch +({NOT_SPACE}+)")),
            profile: re(format!("\\.profile +({NOT_SPACE}+)")),
            ext: re("\\.extern +(.*)".into()),
            section: re("\\.section +text".into()),
            bundle: re("bundle +@cycle=([0-9]+) *\\{".into()),
            slot: re("([^ :][^:]*?) *: *(.+)".into()),
            extern_decl: re(format!("([A-Za-z_][A-Za-z0-9_]*) *, *slice=([0-9]+) *, *addr=({hex}) *, *size=({hex})")),
        }
    })
}

/// `.extern <name>, slice=<n>, addr=0x<hex>, size=0x<hex>` (`extern_decl`).
fn extern_decl(text: &str, isa: &Isa, known: &[Extern]) -> Result<Extern, AsmError> {
    let Some(m) = res().extern_decl.captures(text) else {
        return Err(AsmError::new(
            Code::Asm001,
            format!("expected '.extern <name>, slice=<n>, addr=0x<hex>, size=0x<hex>', got {text:?}"),
        ));
    };
    let name = m[1].to_string();
    if name.len() > ots_format::externs::NAME_MAX {
        return Err(AsmError::new(
            Code::Asm002,
            format!("extern name {name:?} longer than {} characters", ots_format::externs::NAME_MAX),
        ));
    }
    let x = Extern { name, slice: sat_dec(&m[2]), addr: sat_hex(&m[3]), size: sat_hex(&m[4]) };
    check_extern(&x, isa.profile(), known)?;
    Ok(x)
}

/// The text before the first `;`, without trailing spaces (`strip_comment`).
fn strip_comment(line: &str) -> &str {
    line.split(';').next().unwrap().trim_end_matches(' ')
}

#[derive(PartialEq, Clone, Copy)]
enum State {
    Start,
    Arch,
    Profile,
    Text,
    Bundle,
}

/// Assemble program text. `src` is the file's bytes; `path` is what diagnostics call it; `config`
/// is `--config`. Returns the instruction set of the program's profile and the program, or the
/// first error, located (`path:line`, `path:end` or `path:1`).
pub fn assemble(
    src: &[u8],
    path: &str,
    config: Option<&str>,
    warn: &mut dyn FnMut(Warning),
) -> Result<(Isa, Program), AsmError> {
    let r = res();
    let mut prog = Program::default();
    let mut isa: Option<Isa> = None;
    let mut state = State::Start;
    let mut cycle: u64 = 0;
    let mut last_cycle: i128 = -1;
    let mut used: Vec<Group> = Vec::new();
    let mut agt = Agt::new();
    let mut syms: HashMap<String, Extern> = HashMap::new();
    for (i, raw) in src.split(|&b| b == b'\n').enumerate() {
        let ln = i + 1;
        let at = |e: AsmError| e.at(format!("{path}:{ln}"));
        // §1 applies to the whole line, comments included
        if raw.contains(&b'\t') {
            return Err(at(AsmError::new(Code::Asm001, "tab character (only spaces separate tokens)")));
        }
        if !raw.is_ascii() {
            return Err(at(AsmError::new(Code::Asm001, "non-ASCII character (a line is ASCII)")));
        }
        let raw = std::str::from_utf8(raw).expect("ASCII");
        let s = strip_comment(raw).trim_matches(' ');
        if s.is_empty() {
            continue;
        }
        let a = |code: Code, msg: &str| at(AsmError::new(code, msg));
        if state == State::Bundle {
            if s == "}" {
                let isa = isa.as_ref().unwrap();
                bundle_checks(isa, &used, cycle, path)?;
                agt_bundle(&mut agt, &used, cycle, warn, path);
                state = State::Text;
                last_cycle = cycle as i128;
                continue;
            }
            // any text before the first ":" is a queue name (unknown names: ASM-005)
            let Some(m) = r.slot.captures(s) else {
                return Err(a(Code::Asm001, &format!("expected '<queue>: <instruction>' or '}}', got {s:?}")));
            };
            let isa = isa.as_ref().unwrap();
            let q = queue_id(&m[1])
                .ok_or_else(|| a(Code::Asm005, &format!("unknown queue {:?} (or index out of range)", &m[1])))?;
            let (insn, w) = parse_insn(isa, &m[2], &syms, if q <= 87 { Some(q) } else { None }).map_err(at)?;
            // placement and the slot rule are checked when the bundle closes (bundle_checks)
            let e = insn.entry();
            let nw = e.sram_words as u64; // IfetchM reads a and a+1
            let f = insn.fields();
            for (vi, v) in e.vars.iter().enumerate() {
                if v.kind().name == "sram_addr" && (f[vi] as u64) + nw > isa.profile().words_per_slice() {
                    let a = f[vi] as u64;
                    let what = if nw == 1 {
                        format!("address {a:#x}")
                    } else {
                        format!("word {:#x} (of {a:#x}..{:#x})", a + nw - 1, a + nw - 1)
                    };
                    warn(Warning {
                        where_: format!("{path}:{ln}"),
                        code: Code::Opr001,
                        msg: format!(
                            "{what} is beyond words_per_slice {} of profile {}; the reference faults at run time",
                            isa.profile().words_per_slice(),
                            isa.profile().name()
                        ),
                    });
                }
            }
            match used.iter_mut().find(|g| g.queue == q) {
                Some(g) => {
                    g.insns.push(insn);
                    g.lines.push(ln);
                }
                None => used.push(Group { queue: q, insns: vec![insn], lines: vec![ln] }),
            }
            prog.records.push(Record { cycle, queue: q, word: w });
            continue;
        }
        if let Some(m) = r.arch.captures(s) {
            if state != State::Start {
                return Err(a(Code::Asm001, ".arch must be the first directive"));
            }
            if &m[1] != ARCH {
                return Err(a(Code::Asm003, &format!("unknown architecture {:?} (expected {ARCH})", &m[1])));
            }
            state = State::Arch;
            continue;
        }
        if let Some(m) = r.profile.captures(s) {
            if state != State::Arch {
                return Err(a(Code::Asm001, ".profile must follow .arch, once"));
            }
            let prof = load_profile(&m[1], config).map_err(at)?;
            isa = Some(Isa::with_where(
                &prof,
                match config {
                    Some(path) => format!("file {path}"),
                    None => format!("profile {}", prof.name()),
                },
            ));
            prog.profile = m[1].to_string();
            state = State::Profile;
            continue;
        }
        if let Some(m) = r.ext.captures(s) {
            if state != State::Profile {
                return Err(a(Code::Asm001, ".extern must follow .profile and precede .section text"));
            }
            let x = extern_decl(&m[1], isa.as_ref().unwrap(), &prog.externs).map_err(at)?;
            syms.insert(x.name.clone(), x.clone());
            prog.externs.push(x);
            continue;
        }
        if s.starts_with('.') {
            if r.section.is_match(s) {
                if state != State::Profile {
                    return Err(a(Code::Asm001, ".section text must follow .arch and .profile, once"));
                }
                state = State::Text;
                continue;
            }
            return Err(a(Code::Asm001, &format!("unknown directive {s:?}")));
        }
        if let Some(m) = r.bundle.captures(s) {
            if state != State::Text {
                return Err(a(Code::Asm001, "bundle outside .section text, or nested"));
            }
            let c = sat_dec(&m[1]);
            if c as i128 <= last_cycle || c >= 1 << 32 {
                return Err(a(
                    Code::Asm001,
                    &format!("bundle @cycle={}: cycles must increase strictly and be below 2^32", &m[1]),
                ));
            }
            cycle = c;
            state = State::Bundle;
            used.clear();
            continue;
        }
        return Err(a(Code::Asm001, &format!("unexpected {s:?}")));
    }
    if state == State::Bundle {
        return Err(AsmError::new(Code::Asm001, "unterminated bundle").at(format!("{path}:end")));
    }
    if matches!(state, State::Start | State::Arch | State::Profile) {
        return Err(
            AsmError::new(Code::Asm001, "a program needs .arch, .profile and .section text").at(format!("{path}:1"))
        );
    }
    let isa = isa.unwrap();
    // an image longer than the instruction queue: a loader rejects it as an initial image
    let cap = isa.profile().iq_depth() / 4;
    for (q, us) in prog.images().iter() {
        if us.len() as u64 > cap {
            warn(Warning {
                where_: path.to_string(),
                code: Code::Asm003,
                msg: format!(
                    "the text of {} is {} units, more than icu.iq_depth ({cap} units): a loader rejects it as an initial image; fetch the rest with ifetch",
                    queue_name(q).unwrap_or("?"),
                    us.len()
                ),
            });
        }
    }
    Ok((isa, prog))
}
