//! The extern declaration of a program and an object (`assembly.md` §3, §4): a name for a range
//! of words of one MEM slice that the loader fills from a weight package.
use ots_isa::{AsmError, Code};
use ots_profile::Profile;
use std::collections::{BTreeMap, HashMap, HashSet};

/// `.extern <name>, slice=<n>, addr=0x<hex>, size=0x<hex>`. The fields are as wide as the text
/// lets them be (decimal and hex of any length saturate at `u64::MAX`), so the extern rules
/// (OTS-E-ASM-002) see the value that was written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extern {
    pub name: String,
    pub slice: u64,
    pub addr: u64,
    pub size: u64,
}

/// The longest extern name (`assembly.md` §3: at most 15 characters).
pub const NAME_MAX: usize = 15;

/// `[A-Za-z_][A-Za-z0-9_]*`, the whole text.
pub fn is_ident(s: &str) -> bool {
    let mut b = s.bytes();
    matches!(b.next(), Some(c) if c.is_ascii_alphabetic() || c == b'_')
        && b.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

/// The extern rules shared by the text assembler and the object reader (`tsp_as.check_extern`,
/// `OTS-E-ASM-002`), in `tsp_as`'s order: the name is an identifier of at most 15 characters and
/// not declared yet (`known`: the externs so far, in declaration order); the slice exists in the
/// profile (`sram.slices`, and `slice mod 44 < geometry.mem_slices_per_hemisphere`); the words
/// `addr .. addr+size-1` are inside `words_per_slice`, and `size` is not 0; no overlap with an
/// extern of the same slice.
pub fn check_extern(x: &Extern, profile: &Profile, known: &[Extern]) -> Result<(), AsmError> {
    check_with(
        x,
        profile,
        |n| known.iter().any(|o| o.name == n),
        |x| {
            known
                .iter()
                .find(|o| {
                    o.slice == x.slice
                        && (x.addr as u128) < (o.addr as u128 + o.size as u128)
                        && (o.addr as u128) < (x.addr as u128 + x.size as u128)
                })
                .map(|o| o.name.clone())
        },
    )
}

/// The rules of [`check_extern`], with the two questions about the earlier externs asked of the caller:
/// is this name declared, and which extern (the first in declaration order) overlaps this range.
fn check_with(
    x: &Extern,
    profile: &Profile,
    declared: impl Fn(&str) -> bool,
    overlapped: impl Fn(&Extern) -> Option<String>,
) -> Result<(), AsmError> {
    let bad = |m: String| AsmError::new(Code::Asm002, m);
    let name = &x.name;
    // the whole name is an identifier: `tsp_as` uses `IDENT.fullmatch` (a final line feed is rejected)
    if !is_ident(name) {
        return Err(bad(format!("extern name {name:?} is not an identifier")));
    }
    if name.len() > NAME_MAX {
        return Err(bad(format!("extern name {name:?} longer than {NAME_MAX} characters")));
    }
    if declared(name) {
        return Err(bad(format!("extern {name:?} declared twice")));
    }
    if x.slice >= profile.uint("sram.slices").unwrap_or(0) {
        return Err(bad(format!("extern {name}: slice {} out of range", x.slice)));
    }
    let mem = profile.uint("geometry.mem_slices_per_hemisphere").unwrap_or(44);
    if x.slice % 44 >= mem {
        return Err(bad(format!(
            "extern {name}: slice {} does not exist in profile {} (slice mod 44 >= geometry.mem_slices_per_hemisphere)",
            x.slice,
            profile.name()
        )));
    }
    let words = profile.words_per_slice();
    if x.size == 0 || (x.addr as u128) + (x.size as u128) > words as u128 {
        return Err(bad(format!(
            "extern {name}: words {:#x}..{:#x} outside 0..{:#x}",
            x.addr,
            (x.addr as u128 + x.size as u128).wrapping_sub(1),
            words.wrapping_sub(1)
        )));
    }
    if let Some(o) = overlapped(x) {
        return Err(bad(format!("extern {name} overlaps {o} in slice {}", x.slice)));
    }
    Ok(())
}

/// The externs accepted so far, answering the rules of [`check_extern`] without a scan of all of them: the
/// names are in a hash set, and the ranges of each slice in an ordered map. `check_extern(x, profile, known)`
/// with `known` the externs pushed, in order, gives the same verdict and the same message as `check(x, profile)`.
/// Only externs that passed `check` are pushed, so the ranges of a slice are disjoint and sorted by address.
#[derive(Default)]
pub struct ExternIndex {
    names: HashSet<String>,
    /// per slice: start address -> (end address, order of declaration, name)
    ranges: HashMap<u64, BTreeMap<u64, (u64, usize, String)>>,
    n: usize,
}

impl ExternIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of externs pushed.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `check_extern` against the externs pushed so far.
    pub fn check(&self, x: &Extern, profile: &Profile) -> Result<(), AsmError> {
        check_with(
            x,
            profile,
            |n| self.names.contains(n),
            |x| {
                let slice = self.ranges.get(&x.slice)?;
                // the range check came first, so `end` is at most `words_per_slice`
                let end = ((x.addr as u128) + (x.size as u128)).min(u64::MAX as u128) as u64;
                // the ranges that start before `x` ends, nearest first; a range that ends at or before `x.addr`
                // ends the walk (the ranges are disjoint, so every lower one ends earlier)
                let mut first: Option<(usize, &str)> = None;
                for (_, (stop, order, name)) in slice.range(..end).rev() {
                    if *stop <= x.addr {
                        break;
                    }
                    if first.is_none_or(|(o, _)| *order < o) {
                        first = Some((*order, name));
                    }
                }
                first.map(|(_, n)| n.to_string())
            },
        )
    }

    /// Records an extern that passed [`ExternIndex::check`].
    pub fn push(&mut self, x: &Extern) {
        self.names.insert(x.name.clone());
        self.ranges.entry(x.slice).or_default().insert(x.addr, (x.addr + x.size, self.n, x.name.clone()));
        self.n += 1;
    }
}
