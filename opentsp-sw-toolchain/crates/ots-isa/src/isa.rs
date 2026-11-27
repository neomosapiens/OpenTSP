//! Decode, encode, render, placement, slots and stations. A port of the `ISA` class of
//! `tools/tsp_as.py` (decode by field checks instead of by render-then-parse; the corpus tests
//! hold the two together).
use crate::{
    Code, Entry, EntryId, Insn, Kind, Profile, Syntax, Var, AGT_LEVELS, BOOLEAN_CODES, CONTROL, ENTRIES, KINDS,
    MAX_EXPR_BOUNDS, MAX_VARS, MEM_SLOT_PATH, MEM_SLOT_PER_THREAD, MEM_SLOT_QUEUES, MEM_THREADS, NARROW_TAGS, STATIONS,
    STREAM_ROW_BIT, STREAM_ROW_CODES, WIDE_TAGS,
};

fn mask(n: u32) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

impl Var {
    /// The operand's value in word `w` (`extract`).
    pub fn extract(&self, w: u64) -> u32 {
        let mut x = (w >> self.lo) & mask(self.n as u32);
        if self.hi_n > 0 {
            x |= ((w >> self.hi_lo) & mask(self.hi_n as u32)) << self.n;
        }
        if self.top_n > 0 {
            x |= ((w >> self.top_lo) & mask(self.top_n as u32)) << (self.n + self.hi_n);
        }
        x as u32
    }

    /// The bits of operand value `x` in the word (`place`).
    pub fn place(&self, x: u32) -> u64 {
        let x = x as u64;
        let mut w = (x & mask(self.n as u32)) << self.lo;
        if self.hi_n > 0 {
            w |= ((x >> self.n) & mask(self.hi_n as u32)) << self.hi_lo;
        }
        if self.top_n > 0 {
            w |= ((x >> (self.n + self.hi_n)) & mask(self.top_n as u32)) << self.top_lo;
        }
        w
    }

    pub fn kind(&self) -> &'static Kind {
        &KINDS[self.kind as usize]
    }
}

impl Entry {
    /// The word of this entry with these operand values.
    pub fn encode_fields(&self, f: &[u32]) -> u64 {
        self.vars.iter().enumerate().fold(self.fixed, |w, (i, v)| w | v.place(f[i]))
    }
}

/// The entry of an id.
pub fn entry(id: EntryId) -> &'static Entry {
    &ENTRIES[id as usize]
}

/// The word of an instruction (32-bit layouts in bits 31..0).
pub fn encode(insn: &Insn) -> u64 {
    insn.entry().encode_fields(&insn.fields())
}

fn render_token(k: &Kind, x: u32) -> Option<String> {
    Some(match k.syntax {
        Syntax::Name => k.names.iter().find(|(_, c)| *c == x)?.0.to_string(),
        Syntax::Stream => {
            let x = x as u64 * k.group as u64;
            let row = STREAM_ROW_CODES.iter().find(|(_, c)| *c as u64 == (x >> STREAM_ROW_BIT) & 1)?.0;
            format!("{row}:{}", x & mask(STREAM_ROW_BIT))
        }
        Syntax::Dec => x.to_string(),
        Syntax::Hex => format!("{x:#x}"),
    })
}

/// The canonical text of an entry's operand values, as the Sail assembly mapping prints it;
/// `None` if a name kind has no name for a value.
pub fn render_fields(e: &Entry, f: &[u32]) -> Option<String> {
    let mut out = e.assembly.to_string();
    for (i, v) in e.vars.iter().enumerate() {
        let t = render_token(v.kind(), f[i])?;
        out = out.replace(&format!("{{{}}}", v.name), &t);
    }
    Some(out)
}

/// The canonical text of an instruction.
pub fn render(insn: &Insn) -> String {
    render_fields(insn.entry(), &insn.fields()).expect("an Insn holds named values")
}

/// The placement diagnostic of an entry on a queue (`QUE-002`), or `None` when it may issue there
/// (`control_channel`, then `units`).
pub fn placement(id: EntryId, q: u16) -> Option<Code> {
    let bit = if q == CONTROL { 144 } else { q as usize };
    if bit <= 144 && ENTRIES[id as usize].queues[bit / 64] >> (bit % 64) & 1 == 1 {
        None
    } else {
        Some(Code::Que002)
    }
}

/// The slots an instruction takes on a MEM queue: a bit per (slot, thread), see [`Isa::slots`].
pub type SlotSet = u64;

/// A station's access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// The profile-dependent half of the instruction set.
#[derive(Clone, Debug)]
pub struct Isa {
    profile: Profile,
    /// `file <path>` under `--config`, else `profile <name>`.
    where_label: String,
    words: u64,
    bank_bit: u32,
    mem_slices: u64,
    mxm_planes: u64,
    /// The bound `variable - offset` of each `max_expr` operand ([`MAX_EXPR_BOUNDS`]), evaluated once here: a decode checks it per bounded
    /// operand.
    bounds: Vec<i64>,
}

impl Isa {
    pub fn new(profile: &Profile) -> Isa {
        Isa::with_where(profile, format!("profile {}", profile.name()))
    }

    /// Like [`Isa::new`], but the refusal messages name `where_label`.
    pub fn with_where(profile: &Profile, where_label: String) -> Isa {
        Isa {
            profile: profile.clone(),
            where_label,
            words: profile.words_per_slice(),
            bank_bit: profile.bank_bit() as u32,
            mem_slices: profile.uint("geometry.mem_slices_per_hemisphere").unwrap_or(44),
            mxm_planes: profile.uint("mxm.planes_per_hemisphere").unwrap_or(2),
            bounds: MAX_EXPR_BOUNDS
                .iter()
                .map(|(n, off)| profile.variable(n).unwrap_or(profile.vlanes()) as i64 - i64::from(*off))
                .collect(),
        }
    }

    pub fn where_config(&self) -> &str {
        &self.where_label
    }

    /// The placement diagnostic of an entry on a queue under this profile (`ISA.placement` of
    /// `tsp_as`): `QUE-002` when the queue is that of an absent MEM slice (`q < 88` and
    /// `q mod 44 >= geometry.mem_slices_per_hemisphere`) or of an absent MXM plane (the index of its plane in its
    /// hemisphere is at least `mxm.planes_per_hemisphere`), else [`placement`].
    pub fn placement(&self, id: EntryId, q: u16) -> Option<Code> {
        if q != CONTROL && q < 88 && (q as u64) % 44 >= self.mem_slices {
            return Some(Code::Que002);
        }
        if (104..=115).contains(&q) {
            let k = if q <= 109 { u64::from(q > 106) } else { u64::from(q > 112) };
            if k >= self.mxm_planes {
                return Some(Code::Que002);
            }
        }
        placement(id, q)
    }

    /// `geometry.mem_slices_per_hemisphere` (44 when the profile has no such key).
    pub fn mem_slices(&self) -> u64 {
        self.mem_slices
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The upper bound of an unsigned operand from `max` and `max_expr` (`max_of`).
    pub fn max_of(&self, v: &Var) -> Option<i64> {
        let mut m = v.max;
        if let Some(e) = v.max_expr {
            let val = self.bounds[e.idx as usize];
            m = Some(m.map_or(val, |m| m.min(val)));
        }
        m
    }

    /// The range of a signed operand: `min`..`max` (`-32768..32767` unless given), and when there is a `max_expr` its value
    /// bounds the magnitude (`-e..e`, with `e` the smaller of `max` and the expression).
    pub fn signed_range(&self, v: &Var) -> (i64, i64) {
        let (mut lo, mut hi) = (v.min.unwrap_or(-(1 << 15)), v.max.unwrap_or((1 << 15) - 1));
        if v.max_expr.is_some() {
            hi = self.max_of(v).unwrap();
            lo = lo.max(-hi);
        }
        (lo, hi)
    }

    /// Is `x` a legal value of operand `v`? The checks `tsp_as` makes on a parsed operand, as they
    /// apply to a field of a word: a name kind needs a name for `x`; a signed field its range
    /// (`-32768..32767` unless `min`/`max`); else `max_expr`, `values`/`codes`, `not_values`, `max`.
    pub fn field_legal(&self, v: &Var, x: u32) -> bool {
        let k = v.kind();
        let nbits = v.nbits as u32;
        if k.syntax == Syntax::Name && !k.names.iter().any(|(_, c)| *c == x) {
            return false;
        }
        if nbits < 32 && (x as u64) >> nbits != 0 {
            return false;
        }
        if k.signed {
            let s = if (x as u64) >> (nbits - 1) & 1 == 1 { x as i64 - (1i64 << nbits) } else { x as i64 };
            let (lo, hi) = self.signed_range(v);
            return lo <= s && s <= hi;
        }
        // (`max` is checked below, so the bound of the expression alone is enough here)
        if let Some(e) = v.max_expr {
            if x as i64 > self.bounds[e.idx as usize] {
                return false;
            }
        }
        if let Some(l) = v.legal {
            if !l.contains(&x) {
                return false;
            }
        }
        // a code of an extension is legal only in a profile that declares it
        if v.ext_codes.iter().any(|(ext, codes)| codes.contains(&x) && !self.profile.has_extension(ext)) {
            return false;
        }
        if v.not_values.contains(&x) || v.max.is_some_and(|m| x as i64 > m) {
            return false;
        }
        true
    }

    /// Are all operand values of an entry legal, jointly too: when `type` is a boolean type, every
    /// operand with `bool_values` takes one of them?
    pub fn fields_legal(&self, e: &Entry, f: &[u32; MAX_VARS]) -> bool {
        if !e.vars.iter().enumerate().all(|(i, v)| self.field_legal(v, f[i])) {
            return false;
        }
        if let Some(t) = e.vars.iter().position(|v| v.name == "type") {
            if BOOLEAN_CODES.contains(&f[t]) {
                return e.vars.iter().enumerate().all(|(i, v)| v.bool_values.is_none_or(|b| b.contains(&f[i])));
            }
        }
        true
    }

    /// The instruction a word of this width decodes to (the first entry whose match pattern fits,
    /// whose reserved bits are zero and whose operands are all legal), or `None`.
    pub fn decode(&self, width: u8, w: u64) -> Option<Insn> {
        for e in ENTRIES.iter() {
            if e.width != width || w & e.mask != e.fixed {
                continue;
            }
            // bits that are neither fixed nor an operand are reserved: tsp_as re-parses the text and
            // the re-encoded word has them clear
            let all = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
            if w & all & !e.mask & !e.var_mask != 0 {
                continue;
            }
            let mut f = [0u32; MAX_VARS];
            for (i, v) in e.vars.iter().enumerate() {
                f[i] = v.extract(w);
            }
            if self.fields_legal(e, &f) {
                return Some(Insn::from_fields(e.id, &f));
            }
        }
        None
    }

    /// The instruction that starts a queue's text units (one unit, or two for a 64-bit layout) by
    /// the decode rule of `encoding.json`: clear the bundle bit; the tag in bits 30..28 of the first
    /// unit gives the layout; tag 6 or 7, a wide tag without a second unit, or a word that matches no
    /// entry is `DEC-003`. Returns the instruction and the units it uses. The queue does not matter
    /// (placement does: [`placement`]).
    pub fn decode_text(&self, units: &[u32]) -> Result<(Insn, u8), Code> {
        let u0 = units.first().copied().ok_or(Code::Dec003)? & 0x7FFF_FFFF;
        let tag = ((u0 >> 28) & 7) as u8;
        let (width, w, used) = if NARROW_TAGS.contains(&tag) {
            (32, u0 as u64, 1)
        } else if WIDE_TAGS.contains(&tag) && units.len() > 1 {
            (64, ((u0 as u64) << 32) | units[1] as u64, 2)
        } else {
            return Err(Code::Dec003);
        };
        self.decode(width, w).map(|i| (i, used)).ok_or(Code::Dec003)
    }

    /// [`Isa::decode_text`] of a raised word (`raise_images`): its units are rebuilt from it.
    pub fn decode_word(&self, w0: u64, nunits: u8) -> Result<Insn, Code> {
        let mut units = if nunits == 1 {
            vec![(w0 & 0xFFFF_FFFF) as u32]
        } else {
            vec![(w0 >> 32) as u32, (w0 & 0xFFFF_FFFF) as u32]
        };
        if nunits == 1 && !NARROW_TAGS.contains(&(((w0 >> 28) & 7) as u8)) {
            units = vec![(w0 & 0x7FFF_FFFF) as u32];
        }
        self.decode_text(&units).map(|(i, _)| i)
    }

    /// The slots an instruction takes on a MEM queue (`mem_slots`): a bit per (slot, thread), the
    /// slot index being the position in [`MEM_SLOT_NAMES`], 8 bits per slot, bit 0 for "no thread" and
    /// bit `1 + t` for thread `t`. `None`: it needs the whole queue.
    pub fn slots(&self, insn: &Insn) -> Option<SlotSet> {
        let e = insn.entry();
        let names = e.slots?;
        let th = insn.get("thread").unwrap_or(0).min(6) as u64;
        let mut set = 0u64;
        for &s in names {
            let s = s as usize;
            if MEM_SLOT_PER_THREAD[s] {
                if e.all_threads {
                    for t in 0..MEM_THREADS as u64 {
                        set |= 1 << (s * 8 + 1 + t as usize);
                    }
                } else {
                    set |= 1 << (s * 8 + 1 + th as usize);
                }
            } else {
                set |= 1 << (s * 8);
            }
        }
        Some(set)
    }

    /// Bank used in this dispatch cycle; `None` for an indirect reference.
    /// IfetchM's second word uses the following cycle. Cross-cycle hazards are runtime checks.
    /// A word outside the slice addresses no bank.
    fn banks(&self, insn: &Insn) -> Option<u8> {
        let e = insn.entry();
        if e.indirect {
            return None;
        }
        let f = insn.fields();
        let a = e.vars.iter().position(|v| v.kind().name == "sram_addr").map_or(0, |i| f[i]) as u64;
        Some(if a < self.words { 1 << ((a >> self.bank_bit) & 1) } else { 0 })
    }

    /// The model's group diagnostics for the records of one queue and cycle: `group` has `None` for
    /// a record that does not decode. One code (or `None`) per record: `QUE-001` on every record when
    /// they do not fit the slots (two in one slot, a whole-queue record with others, an undecodable
    /// record among several, several records on a non-MEM queue); otherwise `MEM-001` on both
    /// references of a same-thread R + W pair in one bank or with an indirect reference.
    pub fn group_codes(&self, q: u16, group: &[Option<Insn>]) -> Vec<Option<Code>> {
        let n = group.len();
        if n <= 1 {
            return vec![None; n];
        }
        let bit = if q == CONTROL { 144 } else { q as usize };
        let in_slot_queues = bit <= 144 && MEM_SLOT_QUEUES[bit / 64] >> (bit % 64) & 1 == 1;
        let all_que = || vec![Some(Code::Que001); n];
        if !in_slot_queues || group.iter().any(Option::is_none) {
            return all_que();
        }
        let insns: Vec<&Insn> = group.iter().map(|i| i.as_ref().unwrap()).collect();
        let mut taken = 0u64;
        let mut slot_sets = Vec::with_capacity(n);
        for i in &insns {
            match self.slots(i) {
                Some(s) if taken & s == 0 => {
                    taken |= s;
                    slot_sets.push(s);
                }
                _ => return all_que(),
            }
        }
        // references by (path, thread): the last record of a key wins, as in tsp_as
        let mut refs: Vec<(u8, u8, usize, Option<u8>)> = Vec::new();
        for (k, i) in insns.iter().enumerate() {
            let e = i.entry();
            let tags = self.banks(i);
            for (s, path) in MEM_SLOT_PATH.iter().enumerate() {
                let Some(p) = path else { continue };
                for t in 0..8u8 {
                    if slot_sets[k] >> (s * 8 + t as usize) & 1 == 1 && t > 0 {
                        // (slot, thread t-1) of a per-thread slot (a path slot always is)
                        match refs.iter_mut().find(|r| r.0 == *p && r.1 == t) {
                            Some(r) => *r = (*p, t, k, tags),
                            None => refs.push((*p, t, k, tags)),
                        }
                    }
                }
            }
            let _ = e;
        }
        let mut codes = vec![None; n];
        for &(p, t, k, b) in &refs {
            if let Some(&(_, _, _, ob)) = refs.iter().find(|r| r.0 == 1 - p && r.1 == t) {
                if b != Some(0) && ob != Some(0) && (b.is_none() || ob.is_none() || b.unwrap() & ob.unwrap() != 0) {
                    codes[k] = Some(Code::Mem001);
                }
            }
        }
        codes
    }

    /// The gap at which a unit reads or writes a stream row (`stations`): `row_east` selects row E.
    /// `quad` is `(queue mod 44) div 4` for a MEM unit. `None`: the unit has no station there (an
    /// operand on that row is `DIR-001`), or the gap is outside 0..43.
    pub fn station(&self, unit: &str, access: Access, row_east: bool, quad: i64) -> Option<u32> {
        let s = STATIONS.iter().find(|s| s.unit == unit)?;
        let expr = match access {
            Access::Read => s.read,
            Access::Write => s.write,
        }[if row_east { 0 } else { 1 }]?;
        let base = self.profile.uint(expr.key)? as i64;
        let gap = base + expr.quad as i64 * quad;
        (0..=43).contains(&gap).then_some(gap as u32)
    }

    /// Number of AGT levels (`agt.levels`).
    pub fn agt_levels(&self) -> u32 {
        AGT_LEVELS
    }
}
