//! One instruction of assembly text: the template matcher, the operand values and the reject
//! precedence of `assembly.md` §1 and §5. A port of `ISA._token`, `_compile`, `_literal`,
//! `_value`, `_joint` and `parse` of `opentsp-arch-spec/tools/tsp_as.py`, with the regular
//! expressions built the same way, so every template is tried and the first error of the first
//! matching entry is the one reported.
use ots_format::Extern;
use ots_isa::{
    queue_name, AsmError, Code, Entry, EntryId, Insn, Isa, Kind, Syntax, Var, BOOLEAN_CODES, ENTRIES, MAX_VARS,
    STREAM_ROW_BIT, STREAM_ROW_CODES,
};
use regex::Regex;
use std::{collections::HashMap, sync::OnceLock};

/// Only the space character separates tokens (`assembly.md` §1): a tab is an error.
const SPC: &str = " +";
const SEP: &str = " *, *";
/// Sail `hex_bits_N`: leading zeros and `A`-`F` accepted.
const HEX_TOK: &str = "0x[0-9a-fA-F]+";
/// Sail `dec_bits_N`: leading zeros accepted.
const DEC_TOK: &str = "[0-9]+";
const STREAM_TOK: &str = "[EW]:[0-9]+";
/// A symbol is `name` or `name+0xoff`, no spaces around `+`.
const SYM_TOK: &str = "[A-Za-z_][A-Za-z0-9_]*(?:\\+0x[0-9a-fA-F]+)?";

/// One entry's compiled template: the whole template anchored, one capture group per operand,
/// `order` the index into `Entry::vars` of the operand each group holds (template order).
struct Compiled {
    id: EntryId,
    re: Regex,
    order: Vec<usize>,
}

/// Python's `re.escape` (3.7 and later): which characters it escapes decides the length of an
/// escaped name, and `tsp_as` sorts the names of a kind by that length.
fn py_escape_len(s: &str) -> usize {
    s.chars().map(|c| if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(c) { 2 } else { c.len_utf8() }).sum()
}

fn token(k: &Kind) -> String {
    match k.syntax {
        Syntax::Name => {
            // longest escaped name first (a stable sort, as `sorted(..., reverse=True)`)
            let mut names: Vec<&str> = k.names.iter().map(|(n, _)| *n).collect();
            names.sort_by_key(|n| std::cmp::Reverse(py_escape_len(n)));
            names.iter().map(|n| regex::escape(n)).collect::<Vec<_>>().join("|")
        }
        Syntax::Stream => STREAM_TOK.to_string(),
        Syntax::Dec => DEC_TOK.to_string(),
        Syntax::Hex if k.symbol => format!("{SYM_TOK}|{HEX_TOK}"),
        Syntax::Hex => HEX_TOK.to_string(),
    }
}

/// A template literal: `", "` is the separator, `" "` one or more spaces, the rest is escaped.
fn literal(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(c) = rest.chars().next() {
        if rest.starts_with(", ") {
            out.push_str(SEP);
            rest = &rest[2..];
        } else if c == ' ' {
            out.push_str(SPC);
            rest = &rest[1..];
        } else {
            out.push_str(&regex::escape(&c.to_string()));
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

fn compile(e: &'static Entry) -> Compiled {
    let t = e.assembly;
    let mut pat = String::from("\\A");
    let mut order = Vec::new();
    let mut pos = 0;
    while let Some(open) = t[pos..].find('{').map(|i| i + pos) {
        let close = t[open..]
            .find('}')
            .map(|i| i + open)
            .unwrap_or_else(|| panic!("{}: unclosed `{{` in template {t:?}", e.name));
        let name = &t[open + 1..close];
        let vi = e
            .vars
            .iter()
            .position(|v| v.name == name)
            .unwrap_or_else(|| panic!("{}: template names the unknown operand {name:?}", e.name));
        pat.push_str(&literal(&t[pos..open]));
        pat.push('(');
        pat.push_str(&token(e.vars[vi].kind()));
        pat.push(')');
        order.push(vi);
        pos = close + 1;
    }
    pat.push_str(&literal(&t[pos..]));
    pat.push_str("\\z");
    assert!(order.len() == e.vars.len(), "{}: template does not name every operand once", e.name);
    let re = regex::RegexBuilder::new(&pat)
        .size_limit(1 << 26)
        .build()
        .unwrap_or_else(|er| panic!("{}: {pat}: {er}", e.name));
    Compiled { id: e.id, re, order }
}

/// The patterns depend on the catalogue only (the profile enters through the operand bounds), so
/// they are built once.
fn compiled() -> &'static [Compiled] {
    static C: OnceLock<Vec<Compiled>> = OnceLock::new();
    C.get_or_init(|| ENTRIES.iter().map(compile).collect())
}

/// The value of a run of decimal digits, saturating at `u64::MAX` (every field is at most 32 bits,
/// so a saturated value is an out-of-range one, as with Python's unbounded integers).
pub(crate) fn sat_dec(s: &str) -> u64 {
    s.bytes().fold(0u64, |a, b| a.saturating_mul(10).saturating_add((b - b'0') as u64))
}

/// The value of `0x` and hex digits, saturating at `u64::MAX`.
pub(crate) fn sat_hex(s: &str) -> u64 {
    s[2..].bytes().fold(0u64, |a, b| a.saturating_mul(16).saturating_add((b as char).to_digit(16).unwrap() as u64))
}

/// `json.dumps(text, ensure_ascii=False)` as `tsp_as.py` quotes the text of OTS-E-DEC-003: double quotes, `\"` and `\\`,
/// the short escapes `\n \r \t \b \f`, any other control character below U+0020 as `\u00XX`, everything else as it is.
fn json_quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn err(code: Code, msg: String) -> AsmError {
    AsmError::new(code, msg)
}

/// The value of one operand token (`_value`). The checks run in `tsp_as`'s order: stream number,
/// group base, symbol, the field's width (a signed field then needs its range and returns), then
/// `max_expr`, `values`/`codes` (`DEC-003` for a name kind, `OPR-001` otherwise), the codes of an extension the profile does not
/// declare (`DEC-003`), `not_values`, `max`.
fn value(
    isa: &Isa,
    e: &Entry,
    v: &Var,
    tok: &str,
    syms: &HashMap<String, Extern>,
    queue: Option<u16>,
) -> Result<u32, AsmError> {
    let k = v.kind();
    let name = format!("{} operand {}", e.mnemonic, v.name);
    let nbits = v.nbits as u32;
    let x: u64 = match k.syntax {
        Syntax::Name => {
            k.names.iter().find(|(n, _)| *n == tok).map(|(_, c)| *c as u64).expect("a name token the pattern matched")
        }
        Syntax::Stream => {
            let (row, n) = tok.split_once(':').unwrap();
            let n = sat_dec(n);
            if n > 31 {
                return Err(err(Code::Opr001, format!("{name}: stream {tok} out of range 0..31")));
            }
            let g = k.group as u64;
            if !n.is_multiple_of(g) {
                return Err(err(
                    Code::Opr001,
                    format!("{name}: {tok} is not a {g}-stream group base ({row}:0 or {row}:16)"),
                ));
            }
            let code = STREAM_ROW_CODES.iter().find(|(r, _)| *r == row).expect("a stream row").1 as u64;
            ((code << STREAM_ROW_BIT) + n) / g
        }
        Syntax::Dec => sat_dec(tok),
        Syntax::Hex if tok.starts_with("0x") => sat_hex(tok),
        Syntax::Hex => {
            // a symbol: `name` or `name+0xoff`
            let (sym, off) = tok.split_once('+').map_or((tok, 0), |(s, o)| (s, sat_hex(o)));
            let Some(ext) = syms.get(sym) else {
                return Err(err(Code::Asm002, format!("{name}: undefined symbol {sym:?}")));
            };
            let nw = e.sram_words as u64; // IfetchM reads a and a+1
            if off.saturating_add(nw) > ext.size {
                let what = if nw == 1 {
                    format!("offset {off:#x}")
                } else {
                    format!("words {off:#x}..{:#x}", off.saturating_add(nw) - 1)
                };
                return Err(err(Code::Asm002, format!("{name}: {what} outside {sym} (size {:#x})", ext.size)));
            }
            if let Some(q) = queue {
                if q as u64 != ext.slice {
                    return Err(err(
                        Code::Asm002,
                        format!(
                            "{name}: {sym} is in MEM[{}], not on queue {}",
                            ext.slice,
                            queue_name(q).unwrap_or("?")
                        ),
                    ));
                }
            }
            ext.addr.saturating_add(off)
        }
    };
    let fits = nbits >= 64 || x >> nbits == 0;
    if k.signed {
        if !fits {
            return Err(err(Code::Opr001, format!("{name}: {tok} does not fit {nbits} bits")));
        }
        let s = if x >> (nbits - 1) & 1 == 1 { x as i64 - (1i64 << nbits) } else { x as i64 };
        let (lo, hi) = isa.signed_range(v);
        if !(lo <= s && s <= hi) {
            return Err(err(Code::Opr001, format!("{name}: {s} out of range {lo}..{hi}")));
        }
        return Ok(x as u32);
    }
    if !fits {
        return Err(err(Code::Opr001, format!("{name}: {tok} does not fit the {nbits}-bit field")));
    }
    if let Some(me) = v.max_expr {
        let m = isa.max_of(v).unwrap();
        if x as i64 > m {
            let expr = if me.offset == 0 { me.var.to_string() } else { format!("{}-{}", me.var, me.offset) };
            return Err(err(Code::Opr001, format!("{name}: {tok} above {expr} = {m} ({})", isa.where_config())));
        }
    }
    if let Some(l) = v.legal {
        if !l.contains(&(x as u32)) {
            let code = if k.syntax == Syntax::Name { Code::Dec003 } else { Code::Opr001 };
            return Err(err(code, format!("{name}: {tok} is not legal here")));
        }
    }
    for (ext, codes) in v.ext_codes {
        if codes.contains(&(x as u32)) && !isa.profile().has_extension(ext) {
            return Err(err(
                Code::Dec003,
                format!(
                    "{name}: {tok} needs the extension {ext}, which the configuration in force ({}) does not declare",
                    isa.where_config()
                ),
            ));
        }
    }
    if v.not_values.contains(&(x as u32)) || v.max.is_some_and(|m| x as i64 > m) {
        let range = match v.max {
            Some(m) => format!("0..{m}"),
            None => format!("{nbits} bits"),
        };
        let nv = if v.not_values.is_empty() {
            String::new()
        } else {
            format!(", not {}", v.not_values.iter().map(u32::to_string).collect::<Vec<_>>().join(" "))
        };
        return Err(err(Code::Opr001, format!("{name}: {tok} is excluded (range: {range}{nv})")));
    }
    Ok(x as u32)
}

/// Joint constraints (`_joint`): an operand's `bool_values` apply when the `type` operand is one
/// of the boolean types.
fn joint(e: &Entry, f: &[u32; MAX_VARS], toks: &[&str; MAX_VARS]) -> Result<(), AsmError> {
    let Some(t) = e.vars.iter().position(|v| v.name == "type") else { return Ok(()) };
    if !BOOLEAN_CODES.contains(&f[t]) {
        return Ok(());
    }
    for (i, v) in e.vars.iter().enumerate() {
        if let Some(b) = v.bool_values {
            if !b.contains(&f[i]) {
                let tname = e.vars[t].kind().names.iter().find(|(_, c)| *c == f[t]).map_or("?", |(n, _)| n);
                return Err(err(
                    Code::Dec003,
                    format!("{}: {} is not legal for boolean type {tname}", e.mnemonic, toks[i]),
                ));
            }
        }
    }
    Ok(())
}

/// The instruction of one line of text and its word (a 32-bit layout in bits 31..0), or the
/// diagnostic `tsp_as.ISA.parse` reports.
///
/// - `syms`: the externs, for the symbols of an SRAM address; `queue`: the MEM queue the
///   instruction is written on (a symbol must lie in that queue's slice), or `None`.
/// - A tab or a non-ASCII character is `OTS-E-ASM-001`; leading and trailing spaces are ignored.
/// - Every entry's template is tried. One entry accepting the text gives the result; two is
///   `OTS-E-ASM-001`; none gives the first error of the first matching entry, or `OTS-E-DEC-003`
///   when no template matches.
pub fn parse_insn(
    isa: &Isa,
    text: &str,
    syms: &HashMap<String, Extern>,
    queue: Option<u16>,
) -> Result<(Insn, u64), AsmError> {
    if text.contains('\t') {
        return Err(err(Code::Asm001, "tab character (only spaces separate tokens)".into()));
    }
    if !text.is_ascii() {
        return Err(err(Code::Asm001, "non-ASCII character".into()));
    }
    let text = text.trim_matches(' '); // leading and trailing spaces are ignored (§1)
    let mut hits: Vec<(Insn, u64)> = Vec::new();
    let mut first_error: Option<AsmError> = None;
    for c in compiled() {
        let Some(m) = c.re.captures(text) else { continue };
        let e = c.id.entry();
        let mut f = [0u32; MAX_VARS];
        let mut toks = [""; MAX_VARS];
        let mut failed = None;
        for (g, &vi) in c.order.iter().enumerate() {
            let tok = m.get(g + 1).unwrap().as_str();
            toks[vi] = tok;
            match value(isa, e, &e.vars[vi], tok, syms, queue) {
                Ok(x) => f[vi] = x,
                Err(er) => {
                    failed = Some(er);
                    break;
                }
            }
        }
        if failed.is_none() {
            failed = joint(e, &f, &toks).err();
        }
        match failed {
            None => hits.push((Insn::from_fields(c.id, &f), e.encode_fields(&f))),
            Some(er) => {
                first_error.get_or_insert(er);
            }
        }
    }
    match hits.len() {
        1 => Ok(hits.pop().unwrap()),
        0 => {
            Err(first_error
                .unwrap_or_else(|| err(Code::Dec003, format!("no instruction matches {}", json_quote(text)))))
        }
        _ => Err(err(
            Code::Asm001,
            format!("{text:?} is ambiguous: {:?}", hits.iter().map(|h| h.0.entry().name).collect::<Vec<_>>()),
        )),
    }
}

/// `ISA.decode` as `tsp_as` defines it: a word decodes to the first entry whose pattern
/// fits and whose canonical text parses back to exactly that word. `Isa::decode` reaches the same
/// answer by field checks; the property test holds the two together.
pub fn decode_by_reparse(isa: &Isa, width: u8, w: u64) -> Option<Insn> {
    let none = HashMap::new();
    for e in ENTRIES.iter() {
        if e.width != width || w & e.mask != e.fixed {
            continue;
        }
        let mut f = [0u32; MAX_VARS];
        for (i, v) in e.vars.iter().enumerate() {
            f[i] = v.extract(w);
        }
        let Some(text) = ots_isa::render_fields(e, &f) else { continue };
        if let Ok((_, pw)) = parse_insn(isa, &text, &none, None) {
            if pw == w {
                return Some(Insn::from_fields(e.id, &f));
            }
        }
    }
    None
}
