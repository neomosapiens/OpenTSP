//! Queue names, aliases and units (`encoding.json` `queues`, `units`).
use crate::{CONTROL, QUEUE_ALIASES, QUEUE_NAMES, UNITS};
use std::{borrow::Cow, collections::HashMap, sync::OnceLock};

/// The number of data queues; the control channel is [`CONTROL`].
pub const DATA_QUEUES: u16 = 144;

/// `MEM[007]` is `MEM[7]`: every `[digits]` has its leading zeros removed (at least one digit stays).
pub fn canonical_queue_name(name: &str) -> Cow<'_, str> {
    if !name.contains('[') {
        return Cow::Borrowed(name);
    }
    let b = name.as_bytes();
    let mut out = String::with_capacity(name.len());
    let mut i = 0;
    let mut changed = false;
    while i < b.len() {
        if b[i] == b'[' {
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && b.get(j) == Some(&b']') {
                let digits = &name[i + 1..j];
                let t = digits.trim_start_matches('0');
                let t = if t.is_empty() { "0" } else { t };
                changed |= t.len() != digits.len();
                out.push('[');
                out.push_str(t);
                out.push(']');
                i = j + 1;
                continue;
            }
        }
        // names are ASCII or not; copy one char
        let ch = name[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(name)
    }
}

fn by_name() -> &'static HashMap<&'static str, u16> {
    static M: OnceLock<HashMap<&'static str, u16>> = OnceLock::new();
    M.get_or_init(|| QUEUE_NAMES.iter().chain(QUEUE_ALIASES.iter()).map(|(n, q)| (*n, *q)).collect())
}

/// The queue id of a queue name (canonical or alias; an index may have leading zeros), or `None`
/// (`OTS-E-ASM-005` in a program).
pub fn queue_id(name: &str) -> Option<u16> {
    by_name().get(canonical_queue_name(name).as_ref()).copied()
}

/// The canonical name of a queue id (the inverse of [`queue_id`]).
pub fn queue_name(q: u16) -> Option<&'static str> {
    static M: OnceLock<HashMap<u16, &'static str>> = OnceLock::new();
    M.get_or_init(|| QUEUE_NAMES.iter().map(|(n, q)| (*q, *n)).collect()).get(&q).copied()
}

/// The queues of a unit name (`MEM_W`, `VXM`, `any`, ...).
pub fn unit_queues(unit: &str) -> Option<&'static [u16]> {
    UNITS.iter().find(|(n, _)| *n == unit).map(|(_, q)| *q)
}

/// Is `q` a queue (0..143 or the control channel)?
pub fn is_queue(q: u32) -> bool {
    q < DATA_QUEUES as u32 || q == CONTROL as u32
}

/// The sort key of the text order: data queues ascending, the control channel last.
pub fn queue_key(q: u16) -> (bool, u16) {
    (q == CONTROL, q)
}
