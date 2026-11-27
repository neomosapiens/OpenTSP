//! Tests of `harness/cnum`, moved from the module's in-source test block (they use the crate's public API).
use tsp_iss::harness::cnum::*;

#[test]
fn base_zero_forms() {
    assert_eq!(strtoull0(b"10"), CNum { mag: 10, neg: false, end: 2, overflow: false });
    assert_eq!(strtoull0(b"0x10").mag, 16);
    assert_eq!(strtoull0(b"0X1f").mag, 31);
    assert_eq!(strtoull0(b"010").mag, 8);
    assert_eq!(strtoull0(b"08"), CNum { mag: 0, neg: false, end: 1, overflow: false });
    assert_eq!(strtoull0(b"0x"), CNum { mag: 0, neg: false, end: 1, overflow: false });
    assert_eq!(strtoull0(b"0xg").end, 1);
    assert_eq!(strtoull0(b"5x").end, 1);
    assert_eq!(strtoull0(b"").end, 0);
    assert_eq!(strtoull0(b"x").end, 0);
    assert_eq!(strtoull0(b"  7").mag, 7);
    assert_eq!(strtoull0(b"  7").end, 3);
}

#[test]
fn overflow_saturates() {
    assert!(!strtoull0(b"18446744073709551615").overflow);
    let n = strtoull0(b"18446744073709551616");
    assert!(n.overflow && n.mag == u64::MAX && n.end == 20);
    assert!(strtoull16(b"10000000000000000").overflow);
    assert!(!strtoull16(b"ffffffffffffffff").overflow);
    assert!(strtoull16(b"fffffffffffffffffffffffff").overflow);
}

#[test]
fn hex_base_accepts_a_prefix() {
    assert_eq!(strtoull16(b"ff").mag, 255);
    assert_eq!(strtoull16(b"0xff").mag, 255);
    assert_eq!(strtoull16(b"0x").end, 1);
    assert_eq!(strtoull16(b"0xz").end, 1);
    assert_eq!(strtoull16(b"0x5").end, 3);
    assert_eq!(strtoull16(b"zz").end, 0);
    assert_eq!(strtoull16(b"\x0b5").mag, 5);
    assert_eq!(strtoull16(b" -5").mag, u64::MAX - 4);
}

#[test]
fn signed() {
    assert_eq!(strtol0(b"-5"), (-5, 2, false));
    assert_eq!(strtol0(b"+5"), (5, 2, false));
    assert_eq!(strtol0(b" 3"), (3, 2, false));
    assert_eq!(strtol0(b""), (0, 0, false));
    assert_eq!(strtol0(b"-"), (0, 0, false));
    assert_eq!(strtol0(b"9223372036854775807"), (i64::MAX, 19, false));
    assert_eq!(strtol0(b"9223372036854775808"), (i64::MAX, 19, true));
    assert_eq!(strtol0(b"-9223372036854775808"), (i64::MIN, 20, false));
    assert_eq!(strtol0(b"-9223372036854775809"), (i64::MIN, 20, true));
    assert_eq!(strtol0(b"0x7"), (7, 3, false));
}

#[test]
fn saturation_of_the_unsigned_and_signed_conversions_with_a_sign() {
    // strtoull saturates before the negation: a negative number that does not fit is ULLONG_MAX, not negated
    let n = strtoull0(b"-18446744073709551616");
    assert!(n.overflow && n.mag == u64::MAX && n.neg && n.end == 21);
    // a number beyond u64 is out of range for strtol on either side
    assert_eq!(strtol0(b"99999999999999999999"), (i64::MAX, 20, true));
    assert_eq!(strtol0(b"-99999999999999999999"), (i64::MIN, 21, true));
    // a prefix with the hex digits missing parses the zero only (base 0), and base 16 takes the prefix off
    assert_eq!(strtol0(b"0xg"), (0, 1, false));
    assert_eq!(strtoull16(b"0x0"), CNum { mag: 0, neg: false, end: 3, overflow: false });
}
