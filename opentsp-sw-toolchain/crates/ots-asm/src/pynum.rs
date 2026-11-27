//! Python's `int(text, base)` for the command line (`tsp_as.main` calls it on `word` arguments):
//! surrounding whitespace, a sign, `0x` for base 16, and single underscores between digits.
use num_bigint::{BigInt, BigUint, Sign};

pub fn py_int(s: &str, base: u32) -> Option<BigInt> {
    let s = s.trim_matches(|c: char| c.is_ascii_whitespace() || c == '\x0b');
    let (neg, s) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    let prefixed = base == 16 && (s.starts_with("0x") || s.starts_with("0X"));
    let t = if prefixed { &s[2..] } else { s };
    // an underscore sits between two digits, or right after a prefix
    let mut prev_us = !prefixed;
    let mut digits = String::new();
    for c in t.chars() {
        if c == '_' {
            if prev_us {
                return None;
            }
            prev_us = true;
        } else if c.is_digit(base) && c.is_ascii() {
            digits.push(c);
            prev_us = false;
        } else {
            return None;
        }
    }
    if digits.is_empty() || prev_us {
        return None;
    }
    let mag = BigUint::parse_bytes(digits.as_bytes(), base)?;
    Some(BigInt::from_biguint(if neg { Sign::Minus } else { Sign::Plus }, mag))
}

/// Python's `f"{x:#x}"`.
pub fn hex_alt(x: &BigInt) -> String {
    let m = x.magnitude().to_str_radix(16);
    if x.sign() == Sign::Minus {
        format!("-0x{m}")
    } else {
        format!("0x{m}")
    }
}
