//! Text images: per-queue sequences of 32-bit text units (`encoding.json` `text_format`).
use ots_isa::{
    queue_key, AsmError, Code, BUNDLE_BIT, CONTROL, DATA_QUEUES, NARROW_TAGS, PADDING, UNDECODABLE, WIDE_TAGS,
};
use std::{collections::BTreeMap, fmt};

/// The largest NOP count a unit holds.
pub const NOP_MAX: u64 = 0xFFFF;

/// The text of each queue that has any: queues in ascending order, the control channel last.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Images {
    map: BTreeMap<(bool, u16), Vec<u32>>,
}

impl Images {
    pub fn new() -> Images {
        Images::default()
    }

    /// Append units to a queue's text.
    pub fn extend(&mut self, q: u16, units: &[u32]) {
        self.map.entry(queue_key(q)).or_default().extend_from_slice(units);
    }

    /// Replace a queue's text.
    pub fn insert(&mut self, q: u16, units: Vec<u32>) {
        self.map.insert(queue_key(q), units);
    }

    pub fn get(&self, q: u16) -> Option<&Vec<u32>> {
        self.map.get(&queue_key(q))
    }

    /// `(queue, units)` in text order.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &[u32])> {
        self.map.iter().map(|((_, q), u)| (*q, u.as_slice()))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Drop queues with no unit.
    pub fn drop_empty(mut self) -> Images {
        self.map.retain(|_, u| !u.is_empty());
        self
    }
}

/// An entry the lowering cannot take (`tsp_as.lower` raises `ValueError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowerError {
    /// The queue is neither 0..143 nor 0xffff.
    BadQueue(u16),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::BadQueue(q) => write!(f, "queue {q:#x} is neither 0..143 nor 0xffff"),
        }
    }
}

impl std::error::Error for LowerError {}

/// The layout width of a cycle-view word (`assembly.md` §4, lowering): 32 iff bits 63..32 are zero
/// and the tag nibble in 31..28 is a narrow tag; else 64. The queue does not matter.
pub fn layout_width(w0: u64) -> u8 {
    if w0 >> 32 == 0 && is_narrow_nibble((w0 >> 28) & 0xF) {
        32
    } else {
        64
    }
}

fn is_narrow_nibble(n: u64) -> bool {
    NARROW_TAGS.iter().any(|t| *t as u64 == n)
}

fn is_wide_nibble(n: u64) -> bool {
    WIDE_TAGS.iter().any(|t| *t as u64 == n)
}

/// The text units of one cycle-view word (64 bits; a 32-bit layout in bits 31..0): one unit when
/// bits 63..32 are zero and bits 31..28 are a narrow tag; two (high half, low half) when bits 63..60
/// are a wide tag; else the undecodable unit.
pub fn word_units(w: u64) -> Vec<u32> {
    if w >> 32 == 0 && is_narrow_nibble((w >> 28) & 0xF) {
        vec![w as u32]
    } else if is_wide_nibble(w >> 60) {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![UNDECODABLE]
    }
}

/// A NOP word: tag 0, opcode 0, reserved bits zero, `n` = bits 15..0 not zero.
pub fn is_nop_word(w: u64) -> bool {
    w >> 16 == 0 && w & 0xFFFF != 0
}

/// The text of each queue for a cycle view `(cycle, queue, 64-bit word)`.
///
/// Per queue: sort by cycle (source order within a cycle); the entries of one cycle are one parcel,
/// bundle bit 1 on every instruction but the last; before each parcel NOPs (each at most 65535) for
/// `g = cycle - previous cycle - 1` (previous = -1 at the start). A NOP entry that is alone in its
/// cycle is a timing instruction, not a dispatched one: it makes no parcel; the gaps already carry
/// its time, so every other entry keeps its cycle. A NOP together with other entries stays in the
/// parcel (a slot-rule violation).
pub fn lower(entries: &[(u64, u16, u64)]) -> Result<Images, LowerError> {
    let mut by_q: BTreeMap<(bool, u16), BTreeMap<u64, Vec<u64>>> = BTreeMap::new();
    for &(c, q, w) in entries {
        if q >= DATA_QUEUES && q != CONTROL {
            return Err(LowerError::BadQueue(q));
        }
        by_q.entry(queue_key(q)).or_default().entry(c).or_default().push(w);
    }
    let mut images = Images::new();
    for ((_, q), cycles) in by_q {
        let mut units: Vec<u32> = Vec::new();
        let mut prev: i128 = -1;
        for (c, ws) in cycles {
            if ws.len() == 1 && is_nop_word(ws[0]) {
                continue;
            }
            let mut gap = c as i128 - prev - 1;
            while gap > 0 {
                let n = gap.min(NOP_MAX as i128);
                units.push(n as u32);
                gap -= n;
            }
            for (k, w) in ws.iter().enumerate() {
                let mut u = word_units(*w);
                if k < ws.len() - 1 {
                    u[0] |= BUNDLE_BIT;
                }
                units.extend(u);
            }
            prev = c as i128;
        }
        if !units.is_empty() {
            images.insert(q, units);
        }
    }
    Ok(images)
}

/// One instruction of a raised cycle view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Raised {
    pub cycle: i64,
    pub queue: u16,
    /// The instruction's 64-bit word (a 32-bit layout in bits 31..0).
    pub word: u64,
    /// 1 or 2 text units.
    pub units: u8,
}

/// The inverse of [`lower`], for text that never waits: `cycle` = the previous parcel's cycle + 1 +
/// the NOPs before the parcel (-1 before the first); the instructions of a parcel share its cycle. A
/// trailing parcel whose last instruction has the bundle bit set is still listed; NOPs after the last
/// parcel are dropped.
pub fn raise_images(images: &Images) -> Vec<Raised> {
    let mut out = Vec::new();
    for (q, us) in images.iter() {
        let (mut i, mut last, mut nops) = (0usize, -1i64, 0i64);
        let mut parcel: Vec<(u64, u8)> = Vec::new();
        while i < us.len() {
            let u = us[i];
            if u == PADDING {
                i += 1;
                continue;
            }
            let tag = (u >> 28) & 7;
            let n: u8 = if WIDE_TAGS.contains(&(tag as u8)) { 2 } else { 1 };
            let w = if n == 2 && i + 1 < us.len() {
                (((u & 0x7FFF_FFFF) as u64) << 32) | us[i + 1] as u64
            } else {
                (u & 0x7FFF_FFFF) as u64
            };
            let nop = parcel.is_empty() && u & BUNDLE_BIT == 0 && tag == 0 && u & 0x0FFF_0000 == 0 && u & 0xFFFF != 0;
            if nop {
                nops += (u & 0xFFFF) as i64;
            } else {
                parcel.push((w, n));
                if u & BUNDLE_BIT == 0 {
                    last = last + 1 + nops;
                    out.extend(parcel.drain(..).map(|(word, units)| Raised { cycle: last, queue: q, word, units }));
                    nops = 0;
                }
            }
            i += n as usize;
        }
        if !parcel.is_empty() {
            out.extend(parcel.drain(..).map(|(word, units)| Raised { cycle: last + 1 + nops, queue: q, word, units }));
        }
    }
    out
}

/// The images without padding units.
pub fn drop_padding(images: &Images) -> Images {
    let mut out = Images::new();
    for (q, us) in images.iter() {
        out.insert(q, us.iter().copied().filter(|u| *u != PADDING).collect());
    }
    out
}

/// The harness input lines of the text images: `Q <queue> <unit> ...` (hex, at most `per_line`
/// units a line; several lines for one queue append).
pub fn q_lines(images: &Images, per_line: usize) -> Vec<String> {
    let mut out = Vec::new();
    for (q, us) in images.iter() {
        for chunk in us.chunks(per_line) {
            let mut l = format!("Q {q:x}");
            for u in chunk {
                l.push_str(&format!(" {u:08x}"));
            }
            out.push(l);
        }
    }
    out
}

/// The harness input of the images: the `Q` lines, each ending in a line feed.
pub fn input_text(images: &Images) -> String {
    q_lines(images, 64).into_iter().map(|l| l + "\n").collect()
}

/// Python's `str.split()` on ASCII text: runs of white space (space, `\t\n\v\f\r` and `\x1c..\x1f`).
fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f')
}

/// Images from the harness text form `Q <queue> <unit> ...` (`assembly.md` §4). Blank lines are
/// skipped; any other line, a queue that is not 0..8f or ffff, or a unit that is not hex of at most 32
/// bits is `ASM-003`. Several lines for one queue append. The text is ASCII (the caller refuses
/// anything else).
pub fn parse_images(text: &str, path: &str) -> Result<Images, AsmError> {
    let mut images = Images::new();
    for (ln, line) in text.split('\n').enumerate() {
        let ln = ln + 1;
        let t: Vec<&str> = line.split(is_space).filter(|s| !s.is_empty()).collect();
        if t.is_empty() {
            continue;
        }
        if t[0] != "Q" || t.len() < 2 {
            return Err(AsmError::new(
                Code::Asm003,
                format!("{path}:{ln}: expected 'Q <queue> <unit> ...', got {:?}", line.trim()),
            ));
        }
        for x in &t[1..] {
            if !x.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(AsmError::new(Code::Asm003, format!("{path}:{ln}: {x:?} is not a hex value")));
            }
        }
        let sig = |s: &str| s.trim_start_matches('0').len();
        let q = if sig(t[1]) <= 4 { u32::from_str_radix(t[1], 16).unwrap_or(u32::MAX) } else { u32::MAX };
        if !(q < DATA_QUEUES as u32 || q == CONTROL as u32) || t[2..].iter().any(|x| sig(x) > 8) {
            return Err(AsmError::new(
                Code::Asm003,
                format!("{path}:{ln}: queue {:?} is not 0..8f or ffff, or a unit has more than 32 bits", t[1]),
            ));
        }
        let units: Vec<u32> = t[2..].iter().map(|x| u32::from_str_radix(x, 16).unwrap()).collect();
        images.extend(q as u16, &units);
    }
    Ok(images.drop_empty())
}

/// An input file is text when its first non-blank byte (blank: space, tab, CR, LF) is `Q`, or it
/// has none.
pub fn is_text_input(data: &[u8]) -> bool {
    match data.iter().find(|b| !matches!(b, b' ' | b'\t' | b'\r' | b'\n')) {
        None => true,
        Some(b) => *b == b'Q',
    }
}
