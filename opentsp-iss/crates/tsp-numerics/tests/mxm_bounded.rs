use tsp_numerics::mxm::{f16_dot_prepared_bounded, f16_dot_reference, f16_row_max_exponent, F16Act};

fn planes(values: &[u16]) -> (Vec<u8>, Vec<u8>) {
    values.iter().map(|value| (value.to_le_bytes()[0], value.to_le_bytes()[1])).unzip()
}

fn compare(weights: &[u16], input: &[u16]) {
    let (wlo, whi) = planes(weights);
    let (xlo, xhi) = planes(input);
    let bound = f16_row_max_exponent(&whi);
    let expected_bound = weights.iter().map(|value| ((value >> 10) & 31) as u8).max().unwrap_or(0);
    assert_eq!(bound, expected_bound);
    let act = F16Act::new(&xlo, &xhi);
    assert_eq!(
        f16_dot_prepared_bounded(&wlo, &whi, &act, bound),
        f16_dot_reference(&wlo, &whi, &xlo, &xhi),
        "weights max exponent {bound}, length {}",
        weights.len(),
    );
}

#[test]
fn bounded_dot_compares_with_exact_reference() {
    for size in [1, 7, 8, 9, 16, 319, 320, 511, 512] {
        for weight in [0x03ff, 0x43ff, 0x47ff, 0x4bff, 0x7bff, 0x7c00, 0x7fff, 0x8000] {
            for x in [0, 0x0001, 0x3c00, 0x47ff, 0xc7ff, 0x7c00] {
                compare(&vec![weight; size], &vec![x; size]);
                let signed: Vec<_> = (0..size).map(|i| if i % 2 == 1 { weight ^ 0x8000 } else { weight }).collect();
                compare(&signed, &vec![x; size]);
            }
        }
    }
}
