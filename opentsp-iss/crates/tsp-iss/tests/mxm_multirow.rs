use tsp_iss::operands::{NO_PLANES, ZERO_VEC};
use tsp_iss::slices::mxm::WType;
use tsp_iss::slices::mxm_check::{mxm_check, Verdict};

#[test]
fn all_formats_detect_both_four_row_syndrome_aliases() {
    for t in [WType::I8, WType::U8, WType::F16, WType::Fp8, WType::Bfp8, WType::Bf16] {
        for nominal in [[0u8, 0, 0, 1], [0, 1, 1, 0]] {
            let mut weights = vec![ZERO_VEC; 32];
            let mut x = NO_PLANES;
            x[0][0] = 1; // Smallest finite value, or integer 1; checker product is 1.
            if t == WType::Bfp8 {
                x[1][0] = 8; // Minimum four-bit scale, giving units of 2^-17.
            }
            for (r, value) in nominal.into_iter().enumerate() {
                weights[if t.is_pair() { 2 * r } else { r }][0] = value;
                if t == WType::Bfp8 {
                    weights[2 * r + 1][0] = 8;
                }
            }
            let run = |faults: &[(usize, u32)]| mxm_check(&weights, 16, 16, t, t, 1, &x, faults);
            assert_eq!(run(&[]), Verdict::Clean, "{t:?}");
            assert_eq!(run(&[(0, 0)]), Verdict::Corrected(0), "{t:?}");
            assert_eq!(run(&[(0, 0), (0, 1)]), Verdict::Corrected(0), "{t:?}");
            assert_eq!(run(&[(0, 0), (0, 0)]), Verdict::Clean, "{t:?}");
            assert_eq!(run(&[(16, 0)]), Verdict::Clean, "{t:?}");
            assert_eq!(run(&[(0, 0), (1, 0), (2, 0), (3, 0)]), Verdict::Uncorrectable, "{t:?}");
        }
    }
}

#[test]
fn detects_four_row_aliases_of_subset_checks() {
    for nominal in [[0u8, 0, 0, 1], [0, 1, 1, 0]] {
        let mut weights = vec![ZERO_VEC; 16];
        for (row, value) in nominal.into_iter().enumerate() {
            weights[row][0] = value;
        }
        let mut x = NO_PLANES;
        x[0][0] = 1;
        let faults = [(0, 0), (1, 0), (2, 0), (3, 0)];
        assert_eq!(mxm_check(&weights, 16, 16, WType::I8, WType::I8, 0, &x, &faults), Verdict::Uncorrectable);
    }
}

#[test]
fn classifies_final_changed_rows_not_fault_entries() {
    let weights = vec![ZERO_VEC; 16];
    let x = NO_PLANES;
    let run = |faults: &[(usize, u32)]| mxm_check(&weights, 16, 16, WType::I8, WType::I8, 0, &x, faults);
    assert_eq!(run(&[(0, 0), (0, 1)]), Verdict::Corrected(0));
    assert_eq!(run(&[(0, 0), (0, 0)]), Verdict::Clean);
    assert_eq!(run(&[(16, 0)]), Verdict::Clean);
    assert_eq!(run(&[(0, 0), (1, 0)]), Verdict::Uncorrectable);
}
