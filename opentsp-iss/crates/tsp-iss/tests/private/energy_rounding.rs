use super::Exact;
use num_bigint::BigUint;

#[test]
fn final_overflow_rounding_uses_halfway_threshold_without_clamping() {
    let threshold = ((BigUint::from(1u8) << 54usize) - 1u8) << 970usize;
    let below = Exact::ratio(&threshold - 1u8, 1u8.into());
    let tie = Exact::ratio(threshold.clone(), 1u8.into());
    let above = Exact::ratio(threshold + 1u8, 1u8.into());
    assert_eq!(below.value(), f64::MAX);
    assert_eq!(tie.value(), f64::INFINITY);
    assert_eq!(above.value(), f64::INFINITY);
}
#[test]
fn final_subnormal_rounding_is_ties_to_even() {
    let denominator = BigUint::from(1u8) << 1075usize;
    for (numerator, bits) in [(1u8, 0u64), (3, 2), (5, 2), (7, 4)] {
        assert_eq!(Exact::ratio(numerator.into(), denominator.clone()).value().to_bits(), bits);
    }
    let just_above_half = Exact::ratio((BigUint::from(1u8) << 1075usize) + 1u8, BigUint::from(1u8) << 2150usize);
    assert_eq!(just_above_half.value().to_bits(), 1);
}
