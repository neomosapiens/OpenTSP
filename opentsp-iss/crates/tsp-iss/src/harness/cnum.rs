//! C's `strtoull`, `strtol` and `strtoll` on byte strings, because the options and the input fields
//! of `tsp_sim` are parsed with them and "the same inputs are accepted" includes their quirks:
//! leading white space, a sign, `0x` hex and leading-zero octal in base 0, saturation on overflow.

/// The result of a C conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CNum {
    /// The magnitude, saturated at `u64::MAX`, before the sign is applied.
    pub mag: u64,
    pub neg: bool,
    /// Bytes consumed; 0 when there were no digits (`endptr == nptr`).
    pub end: usize,
    /// The value did not fit (`ERANGE`), for the signed conversions: outside the `i64` range.
    pub overflow: bool,
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn digit(b: u8) -> Option<u32> {
    (b as char).to_digit(16)
}

/// The common scanner. `base` is 0 or 16; `signed` selects the overflow rule.
fn scan(s: &[u8], base: u32, signed: bool) -> CNum {
    let mut i = 0;
    while i < s.len() && is_space(s[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let has_prefix = i + 2 < s.len() && s[i] == b'0' && (s[i + 1] | 0x20) == b'x' && digit(s[i + 2]).is_some();
    let radix = match base {
        16 => 16,
        _ if has_prefix => 16,
        _ if i < s.len() && s[i] == b'0' => 8,
        _ => 10,
    };
    if has_prefix && radix == 16 {
        i += 2;
    }
    let start = i;
    let mut mag: u128 = 0;
    let mut sat = false;
    while i < s.len() {
        let Some(d) = digit(s[i]).filter(|d| *d < radix) else { break };
        if !sat {
            mag = mag * radix as u128 + d as u128;
            if mag > u64::MAX as u128 {
                sat = true;
            }
        }
        i += 1;
    }
    if i == start {
        // no digits: nothing was converted (a "0x" with no hex digit after it parses the "0")
        return CNum { mag: 0, neg: false, end: 0, overflow: false };
    }
    let mut mag = if sat { u64::MAX } else { mag as u64 };
    if !signed && neg && !sat {
        mag = mag.wrapping_neg(); // strtoull negates modulo 2^64
    }
    let overflow =
        if signed { sat || if neg { mag > i64::MAX as u64 + 1 } else { mag > i64::MAX as u64 } } else { sat };
    CNum { mag, neg, end: i, overflow }
}

/// `strtoull(s, &e, 0)`.
pub fn strtoull0(s: &[u8]) -> CNum {
    scan(s, 0, false)
}

/// `strtoull(s, &e, 16)`.
pub fn strtoull16(s: &[u8]) -> CNum {
    scan(s, 16, false)
}

/// `strtol(s, &e, 0)` / `strtoll(s, &e, 0)` on a 64-bit `long`: the value, saturated, and the rest.
pub fn strtol0(s: &[u8]) -> (i64, usize, bool) {
    let n = scan(s, 0, true);
    let v = if n.neg {
        if n.mag > i64::MAX as u64 + 1 {
            i64::MIN
        } else {
            (n.mag as i128).wrapping_neg() as i64
        }
    } else if n.mag > i64::MAX as u64 {
        i64::MAX
    } else {
        n.mag as i64
    };
    (v, n.end, n.overflow)
}
