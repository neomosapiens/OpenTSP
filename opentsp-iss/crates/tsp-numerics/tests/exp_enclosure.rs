// Exact rational Taylor bounds, independently of the fixed-point recurrence.
use num_bigint::BigInt;
use tsp_numerics::ziv;

fn oracle(p: i64, a: i64, b: i64) -> (BigInt, BigInt) {
    let scale = BigInt::from(1) << p as usize;
    if a == 0 {
        return (scale.clone(), scale);
    }
    let aa = a.abs();
    let mut numerator = BigInt::from(1);
    let mut denominator = BigInt::from(1);
    let mut power = BigInt::from(1);
    for k in 0..10000i64 {
        let next_power = &power * aa;
        let next_den = &denominator * b * (k + 1);
        // For 0 <= a/b <= 1/2 the remaining tail is below twice the next term.
        if &next_power * &scale * 32 < next_den {
            let upper_num = &numerator * &next_den + &next_power * &denominator * 2;
            let upper_den = &denominator * &next_den;
            if a > 0 {
                return (&numerator * &scale / &denominator, (&upper_num * &scale + &upper_den - 1) / &upper_den);
            }
            return (&upper_den * &scale / &upper_num, (&denominator * &scale + &numerator - 1) / &numerator);
        }
        numerator = numerator * b * (k + 1) + &next_power;
        denominator = next_den;
        power = next_power;
    }
    panic!("oracle did not converge");
}

#[test]
fn exp_small_encloses_exact_rational_bounds_at_every_ziv_precision() {
    for p in [64i64, 128, 256, 512, 1024, 2048, 4096] {
        let scale = BigInt::from(1) << p as usize;
        for a in -4..=4i64 {
            let y = &scale * a / 8;
            let (lo, hi) = ziv::exp_small(&y, p);
            let (reference_lo, reference_hi) = oracle(p, a, 8);
            assert!(lo <= reference_lo && hi >= reference_hi, "p={p}, y={a}/8");
        }
    }
}
