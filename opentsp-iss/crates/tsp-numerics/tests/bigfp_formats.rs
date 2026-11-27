// The exact public rounding helper supports IEEE and finite-only formats.
use num_bigint::BigInt;
use tsp_numerics::bigfp::fround_big;
use tsp_numerics::fp::{fround, FMT16, FMT32, FMT_BF16, FMT_E4M3, FMT_E5M2};

#[test]
fn finite_only_top_exponent_retains_each_finite_value() {
    for (value, bits) in
        [(256, 0x78), (288, 0x79), (320, 0x7a), (352, 0x7b), (384, 0x7c), (416, 0x7d), (448, 0x7e), (464, 0x7e)]
    {
        for negative in [false, true] {
            assert_eq!(
                fround_big(FMT_E4M3, negative, &BigInt::from(value), 0),
                bits | if negative { 0x80 } else { 0 },
                "value={value}"
            );
        }
    }
}

#[test]
fn all_declared_formats_match_scalar_rounding_on_signed_dyadic_lattice() {
    for format in [FMT16, FMT32, FMT_BF16, FMT_E4M3, FMT_E5M2] {
        for negative in [false, true] {
            for exponent in -20..=20 {
                for magnitude in 0..=512u128 {
                    assert_eq!(
                        fround_big(format, negative, &BigInt::from(magnitude), exponent as i64),
                        fround(format, negative, magnitude, exponent),
                        "format={format:?}, magnitude={magnitude}, exponent={exponent}"
                    );
                }
            }
        }
    }
}
