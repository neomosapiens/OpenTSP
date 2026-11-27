//! `ots-asm patch <obj.elf> --table <table.json> --values <values.json> -o <out.elf>`: rewrite operand fields of an
//! object's text at the sites of a patch table, and re-run the parcel checks on every parcel that was touched
//! (the object format does not change).
//!
//! A patch table is a JSON object `{"format": 1, "sites": [ ... ]}`. A site names an operand of one instruction of the
//! object's text and the values the host may put there:
//!
//! | key | meaning |
//! |---|---|
//! | `name` | the site's name (letters, digits and `_ . : -`), unique in the table |
//! | `queue` | the queue id, as a number |
//! | `unit` | the index of the instruction's first text unit in the queue's text (counting every unit, NOP units included) |
//! | `op` | the mnemonic of the instruction that must stand there (a catalogue entry's `mnemonic`) |
//! | `field` | the operand's name in the catalogue (`addr`, `map`, ...) |
//! | `kind` | the operand's kind in the catalogue (`sram_addr`, `map_id`, ...) |
//! | `min`, `max` | the range of values the site accepts, inclusive |
//! | `stride`, `span` | optional (1 and 1): a value v is accepted when `(v - min) % stride < span` |
//!
//! A values file is a JSON object `{"<site name>": <non-negative integer>, ...}`. A name that is not a site, a value
//! outside the site's range or grid, and a site whose instruction is not the one the table names are refused (nothing
//! is written). A site without a value keeps the object's operand. The patched object is then checked as the assembler
//! checks a bundle: every parcel that holds a patched instruction must decode (`OTS-E-DEC-003`), fit the queue's slots
//! (`OTS-E-QUE-001`), be placed on a queue of its unit (`OTS-E-QUE-002`) and obey the bank rule (`OTS-E-MEM-001`: a
//! patched read and a write of one thread in the same bank are refused, so a patch that swaps a MEM bank into a clash
//! does not produce an object). The opcode, the length and the timing of the text never change.
use ots_format::{object::replace_images, read_object};
use ots_isa::{decode_text_units, queue_name, AsmError, Code, Insn, Isa};
use ots_profile::json::{parse, Json};

/// The table format this command reads.
pub const FORMAT: u64 = 1;

/// One site of a patch table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    pub name: String,
    pub queue: u16,
    pub unit: usize,
    pub op: String,
    pub field: String,
    pub kind: String,
    pub min: u64,
    pub max: u64,
    pub stride: u64,
    pub span: u64,
}

fn err(code: Code, msg: impl Into<String>) -> AsmError {
    AsmError::new(code, msg)
}

fn num(j: &Json, what: &str) -> Result<u64, AsmError> {
    match j {
        Json::Num(t) if t.bytes().all(|b| b.is_ascii_digit()) => {
            t.parse().map_err(|_| err(Code::Asm001, format!("{what}: {t} is too large")))
        }
        _ => Err(err(Code::Asm001, format!("{what}: expected a non-negative integer"))),
    }
}

fn text<'a>(j: &'a Json, what: &str) -> Result<&'a str, AsmError> {
    j.as_str().ok_or_else(|| err(Code::Asm001, format!("{what}: expected a string")))
}

fn member<'a>(o: &'a Json, key: &str, what: &str) -> Result<&'a Json, AsmError> {
    o.get(key).ok_or_else(|| err(Code::Asm001, format!("{what}: missing {key:?}")))
}

fn is_name(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
}

/// The sites of a patch table (the text of `--table`).
pub fn parse_table(data: &[u8]) -> Result<Vec<Site>, AsmError> {
    let j = parse(data).map_err(|at| err(Code::Asm001, format!("patch table: not JSON (byte {at})")))?;
    if !matches!(j, Json::Obj(_)) {
        return Err(err(Code::Asm001, "patch table: expected a JSON object"));
    }
    let f = num(member(&j, "format", "patch table")?, "patch table format")?;
    if f != FORMAT {
        return Err(err(Code::Asm001, format!("patch table: format {f}, this tool reads format {FORMAT}")));
    }
    let Json::Arr(items) = member(&j, "sites", "patch table")? else {
        return Err(err(Code::Asm001, "patch table: \"sites\" must be an array"));
    };
    let mut sites: Vec<Site> = Vec::with_capacity(items.len());
    for (i, s) in items.iter().enumerate() {
        let what = format!("patch table site {i}");
        let name = text(member(s, "name", &what)?, &what)?.to_string();
        if !is_name(&name) {
            return Err(err(Code::Asm001, format!("{what}: name {name:?} is not made of letters, digits and _ . : -")));
        }
        if sites.iter().any(|x| x.name == name) {
            return Err(err(Code::Asm001, format!("{what}: site {name} is declared twice")));
        }
        let what = format!("patch table site {name}");
        let queue = num(member(s, "queue", &what)?, &format!("{what} queue"))?;
        let queue =
            u16::try_from(queue).map_err(|_| err(Code::Asm001, format!("{what}: queue {queue} out of range")))?;
        let unit = num(member(s, "unit", &what)?, &format!("{what} unit"))?;
        let (min, max) = (
            num(member(s, "min", &what)?, &format!("{what} min"))?,
            num(member(s, "max", &what)?, &format!("{what} max"))?,
        );
        if min > max {
            return Err(err(Code::Asm001, format!("{what}: min {min} is above max {max}")));
        }
        let opt = |k: &str| s.get(k).map(|v| num(v, &format!("{what} {k}"))).transpose();
        let (stride, span) = (opt("stride")?.unwrap_or(1), opt("span")?.unwrap_or(1));
        if span == 0 || span > stride {
            return Err(err(Code::Asm001, format!("{what}: stride {stride} and span {span}")));
        }
        sites.push(Site {
            name,
            queue,
            unit: unit as usize,
            op: text(member(s, "op", &what)?, &what)?.to_string(),
            field: text(member(s, "field", &what)?, &what)?.to_string(),
            kind: text(member(s, "kind", &what)?, &what)?.to_string(),
            min,
            max,
            stride,
            span,
        });
    }
    Ok(sites)
}

/// `(name, value)` of a values file, in file order.
pub fn parse_values(data: &[u8]) -> Result<Vec<(String, u64)>, AsmError> {
    let j = parse(data).map_err(|at| err(Code::Asm001, format!("patch values: not JSON (byte {at})")))?;
    let Json::Obj(members) = j else {
        return Err(err(Code::Asm001, "patch values: expected a JSON object {\"site\": value, ...}"));
    };
    let mut out: Vec<(String, u64)> = Vec::with_capacity(members.len());
    for (k, v) in &members {
        if out.iter().any(|(n, _)| n == k) {
            return Err(err(Code::Asm001, format!("patch values: {k:?} is given twice")));
        }
        out.push((k.clone(), num(v, &format!("patch value {k}"))?));
    }
    Ok(out)
}

/// One instruction of a queue's text: its first unit, its units and what it decodes to.
struct Placed {
    start: usize,
    insn: Result<Insn, Code>,
    more: bool,
}

/// The instructions of a queue's text in order, each with its bundle bit (a parcel runs to the first instruction
/// whose bit is clear).
fn walk(isa: &Isa, units: &[u32]) -> Vec<Placed> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < units.len() {
        let used = decode_text_units(units[i] & 0x7FFF_FFFF).unwrap_or(1) as usize;
        let insn = isa.decode_text(&units[i..]).map(|(x, _)| x);
        // a wide word without its second unit is one undecodable unit
        let used = if insn.is_err() && i + used > units.len() { 1 } else { used };
        out.push(Placed { start: i, insn, more: units[i] >> 31 == 1 });
        i += used;
    }
    out
}

/// The parcel checks of the instruction starting at `unit` of queue `q`: the first diagnostic, if any.
fn check_parcel(isa: &Isa, q: u16, units: &[u32], unit: usize) -> Option<(Code, String)> {
    let placed = walk(isa, units);
    let at = placed.iter().position(|p| p.start == unit)?;
    let mut lo = at;
    while lo > 0 && placed[lo - 1].more {
        lo -= 1;
    }
    let mut hi = at;
    while hi + 1 < placed.len() && placed[hi].more {
        hi += 1;
    }
    let parcel = &placed[lo..=hi];
    let opts: Vec<Option<Insn>> = parcel.iter().map(|p| p.insn.as_ref().ok().copied()).collect();
    let shares = isa.group_codes(q, &opts);
    let qn = queue_name(q).unwrap_or("?");
    // an instruction that no longer decodes is named before the slot rule of the parcel it breaks
    if let Some(p) = parcel.iter().find(|p| p.insn.is_err()) {
        if let Err(code) = p.insn {
            return Some((code, format!("unit {} of {qn} does not decode", p.start)));
        }
    }
    for ((p, share), opt) in parcel.iter().zip(&shares).zip(&opts) {
        if let Some(code) = share {
            let why = match code {
                Code::Mem001 => "references of one thread address the same bank, or one is a gather or scatter",
                _ => "the parcel does not fit the queue's slots",
            };
            return Some((*code, format!("parcel of {qn} at unit {}: {why}", p.start)));
        }
        match (&p.insn, opt) {
            (Err(code), _) => return Some((*code, format!("unit {} of {qn} does not decode", p.start))),
            (Ok(i), _) => {
                if let Some(code) = isa.placement(i.id(), q) {
                    return Some((code, format!("{} cannot issue on {qn}", i.entry().mnemonic)));
                }
            }
        }
    }
    None
}

/// Apply `values` to the object `data` at `sites`; the patched object and one line per patched site.
pub fn patch(
    data: &[u8],
    profile: Option<&str>,
    config: Option<&str>,
    sites: &[Site],
    values: &[(String, u64)],
) -> Result<(Vec<u8>, Vec<String>), AsmError> {
    let (prof_name, mut images, externs) = read_object(data, profile, config)?;
    let prof = ots_format::load_profile(&prof_name, config)?;
    let isa = Isa::new(&prof);
    for (name, _) in values {
        if !sites.iter().any(|s| &s.name == name) {
            return Err(err(Code::Asm001, format!("patch values: {name:?} is not a site of the table")));
        }
    }
    let mut report = Vec::new();
    let mut touched: Vec<(u16, usize, String)> = Vec::new();
    for s in sites {
        let Some((_, v)) = values.iter().find(|(n, _)| *n == s.name) else { continue };
        let site = |m: String| err(Code::Opr001, format!("site {}: {m}", s.name));
        if *v < s.min || *v > s.max {
            return Err(site(format!("value {v} outside [{}, {}]", s.min, s.max)));
        }
        if (*v - s.min) % s.stride >= s.span {
            return Err(site(format!("value {v:#x} is not on the grid of stride {} from {:#x}", s.stride, s.min)));
        }
        let mut units = images.get(s.queue).cloned().ok_or_else(|| {
            err(Code::Asm001, format!("site {}: the object has no text for queue {}", s.name, s.queue))
        })?;
        if !walk(&isa, &units).iter().any(|p| p.start == s.unit) {
            return Err(err(
                Code::Asm001,
                format!(
                    "site {}: unit {} of queue {} is not the first unit of an instruction",
                    s.name, s.unit, s.queue
                ),
            ));
        }
        let (insn, used) =
            units.get(s.unit..).filter(|u| !u.is_empty()).and_then(|u| isa.decode_text(u).ok()).ok_or_else(|| {
                err(
                    Code::Asm001,
                    format!("site {}: no instruction starts at unit {} of queue {}", s.name, s.unit, s.queue),
                )
            })?;
        let e = insn.entry();
        if e.mnemonic != s.op {
            return Err(err(
                Code::Asm001,
                format!("site {}: unit {} of queue {} holds {}, not {}", s.name, s.unit, s.queue, e.mnemonic, s.op),
            ));
        }
        let Some(k) = e.vars.iter().position(|x| x.name == s.field) else {
            return Err(err(Code::Asm001, format!("site {}: {} has no operand {:?}", s.name, e.mnemonic, s.field)));
        };
        let var = &e.vars[k];
        if var.kind().name != s.kind {
            return Err(err(
                Code::Asm001,
                format!(
                    "site {}: operand {} of {} is of kind {}, not {}",
                    s.name,
                    s.field,
                    e.mnemonic,
                    var.kind().name,
                    s.kind
                ),
            ));
        }
        if var.nbits < 32 && *v >> var.nbits != 0 {
            return Err(site(format!("value {v} does not fit the {}-bit field", var.nbits)));
        }
        let old = insn.fields()[k];
        let mut f = insn.fields();
        f[k] = *v as u32;
        let word = e.encode_fields(&f[..e.vars.len()]);
        let more = units[s.unit] & 0x8000_0000;
        if used == 2 {
            units[s.unit] = ((word >> 32) as u32) | more;
            units[s.unit + 1] = word as u32;
        } else {
            units[s.unit] = (word as u32) | more;
        }
        images.insert(s.queue, units);
        report.push(format!("{} = {v:#x} (was {old:#x}; queue {}, unit {})", s.name, s.queue, s.unit));
        touched.push((s.queue, s.unit, s.name.clone()));
    }
    for (q, unit, name) in touched {
        let units = images.get(q).expect("a patched queue has text");
        if let Some((code, msg)) = check_parcel(&isa, q, units, unit) {
            return Err(err(code, format!("the patch of site {name} makes an invalid parcel: {msg}")));
        }
    }
    Ok((replace_images(data, &prof, &images, &externs)?, report))
}
