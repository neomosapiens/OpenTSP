//! The shapes of the generated tables (`build.rs` fills them from `encoding.json`).
use ots_profile::TimingClass;

/// How an operand kind is written in text (`operand_kinds[].syntax`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Syntax {
    /// One of a list of names (the value is the name's code); the empty name is allowed.
    Name,
    /// `E:n` or `W:n`.
    Stream,
    /// Decimal digits.
    Dec,
    /// `0x` and hex digits (or a symbol when `Kind::symbol`).
    Hex,
}

#[derive(Clone, Copy, Debug)]
pub struct Kind {
    pub name: &'static str,
    pub syntax: Syntax,
    /// `(name, code)` of a `Syntax::Name` kind.
    pub names: &'static [(&'static str, u32)],
    /// Two's complement (`simm16`).
    pub signed: bool,
    /// A symbol `name` or `name+0xoff` is accepted in place of a number (`sram_addr`).
    pub symbol: bool,
    /// `stream_group16`: the field holds `(32 * row + id) / group`.
    pub group: u32,
}

/// `max_expr`: the field is at most `<profile variable> - offset`.
#[derive(Clone, Copy, Debug)]
pub struct MaxExpr {
    pub var: &'static str,
    /// The index of this operand's bound in [`MAX_EXPR_BOUNDS`](crate::MAX_EXPR_BOUNDS): an [`Isa`](crate::Isa) holds the value of each once.
    pub idx: u8,
    pub offset: u32,
}

/// One operand of an entry: its place in the word and its legal values.
#[derive(Clone, Copy, Debug)]
pub struct Var {
    pub name: &'static str,
    /// Index into [`crate::KINDS`].
    pub kind: u8,
    /// Least significant bit and width of the field.
    pub lo: u8,
    pub n: u8,
    /// The bits above the field (`hi_location`); `hi_n == 0`: none.
    pub hi_lo: u8,
    pub hi_n: u8,
    /// The bits above those (`top_location`); `top_n == 0`: none.
    pub top_lo: u8,
    pub top_n: u8,
    /// Total width, `n + hi_n + top_n`.
    pub nbits: u8,
    /// `extension_codes`: the values that are legal only in a profile that declares the named extension.
    pub ext_codes: &'static [(&'static str, &'static [u32])],
    /// `values` or `codes`: the only legal values, when present.
    pub legal: Option<&'static [u32]>,
    pub not_values: &'static [u32],
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub max_expr: Option<MaxExpr>,
    /// Legal values when the entry's `type` is a boolean type.
    pub bool_values: Option<&'static [u32]>,
    /// Footprint expressions of a stream operand (`porting-guide.md` §2.1), as written.
    pub streams: Option<&'static str>,
    pub align: Option<&'static str>,
}

/// One catalogue entry (`encoding.json` `instructions[]`).
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub id: crate::EntryId,
    pub name: &'static str,
    pub mnemonic: &'static str,
    pub slice: &'static str,
    pub units: &'static [&'static str],
    pub timing: TimingClass,
    /// The template, with `{variable}` for each operand.
    pub assembly: &'static str,
    /// 32 or 64.
    pub width: u8,
    /// The match pattern's fixed bits (`-` read as 0) and the mask of the fixed positions.
    pub fixed: u64,
    pub mask: u64,
    /// The bits of all operand fields; every other non-fixed bit is reserved (must be zero).
    pub var_mask: u64,
    pub vars: &'static [Var],
    /// Queues the entry may be issued on: bit `i` is data queue `i`, bit 144 the control channel.
    pub queues: [u64; 3],
    /// SRAM words accessed from an `sram_addr` operand (`sram_access`; 1 unless listed).
    pub sram_words: u8,
    /// Indices into [`crate::MEM_SLOT_NAMES`] the entry takes; `None`: the whole queue.
    pub slots: Option<&'static [u8]>,
    pub all_threads: bool,
    pub indirect: bool,
    pub agt_loopable: bool,
    /// Position in the AGT `config` order (Countdown, Step, StreamStep, Iterations).
    pub agt_config: Option<u8>,
}

/// A station gap: a configuration key, plus or minus the MEM quad.
#[derive(Clone, Copy, Debug)]
pub struct StationExpr {
    pub key: &'static str,
    /// -1: `key - quad`, 0: `key`, 1: `key + quad`.
    pub quad: i8,
}

/// Where a unit reads and writes each stream row (`stations`): index 0 is row E, 1 is row W.
#[derive(Clone, Copy, Debug)]
pub struct Station {
    pub unit: &'static str,
    pub read: [Option<StationExpr>; 2],
    pub write: [Option<StationExpr>; 2],
}
