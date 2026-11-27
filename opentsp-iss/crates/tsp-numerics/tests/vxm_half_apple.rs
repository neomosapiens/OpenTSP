use tsp_numerics::lanes::{self, LaneOp};
use tsp_numerics::planes::apply_planes;
use tsp_numerics::types::EType;

#[test]
fn every_first_operand_and_selected_second_operands_match_f16_rounding() {
    let n = 127;
    let zeros = vec![0; n];
    for opcode in [0, 3, 6] {
        let opt = LaneOp::Bin(opcode, EType::F16);
        for selected in [0, 0x8000, 1, 0x03ff, 0x0400, 0x3400, 0x3c01, 0x7bff, 0x7c00, 0x7c01, 0xfc00] {
            for start in (0..=65535).step_by(n) {
                let values: Vec<u16> = (0..n).map(|j| (start + j).min(65535) as u16).collect();
                let a0: Vec<_> = values.iter().map(|v| v.to_le_bytes()[0]).collect();
                let a1: Vec<_> = values.iter().map(|v| v.to_le_bytes()[1]).collect();
                let [l, h] = (selected as u16).to_le_bytes();
                let b0 = vec![l; n];
                let b1 = vec![h; n];
                let (mut lo, mut hi, mut keep0, mut keep1) = (vec![0; n], vec![0; n], vec![90; n], vec![67; n]);
                apply_planes(
                    opt,
                    2,
                    2,
                    &[&a0, &a1, &zeros, &zeros],
                    &[&b0, &b1, &zeros, &zeros],
                    &[&zeros; 4],
                    &mut [&mut lo, &mut hi, &mut keep0, &mut keep1],
                );
                for i in 0..n {
                    let expected = lanes::apply(opt, u32::from(values[i]), selected, 0);
                    assert_eq!(
                        u32::from(u16::from_le_bytes([lo[i], hi[i]])),
                        expected,
                        "op={opcode} a={:x} b={selected:x} lane={i}",
                        values[i]
                    );
                }
                assert!(keep0.iter().all(|&b| b == 90) && keep1.iter().all(|&b| b == 67));
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[test]
fn native_half_preserves_status_and_falls_back_for_flush_to_zero() {
    use core::arch::asm;

    fn fpcr() -> u64 {
        let value;
        unsafe { asm!("mrs {value}, fpcr", value = out(reg) value, options(nostack, nomem)) };
        value
    }
    fn fpsr() -> u64 {
        let value;
        unsafe { asm!("mrs {value}, fpsr", value = out(reg) value, options(nostack, nomem)) };
        value
    }
    fn set_mode(value: u64) {
        unsafe { asm!("msr fpcr, {value}", "isb", value = in(reg) value, options(nostack, nomem)) };
    }
    fn set_flags(value: u64) {
        unsafe { asm!("msr fpsr, {value}", value = in(reg) value, options(nostack, nomem)) };
    }
    struct Restore(u64, u64);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_mode(self.0);
            set_flags(self.1);
        }
    }

    assert!(std::arch::is_aarch64_feature_detected!("fp16"));
    let _restore = Restore(fpcr(), fpsr());
    let zero = [0; 16];
    let mut first_lo = [1; 16];
    let mut first_hi = [0; 16];
    first_lo[0] = 0x01;
    first_hi[0] = 0x7c; // signalling NaN would raise the invalid flag in hardware
    let second_lo = [1; 16];
    let mut lo = [0; 16];
    let mut hi = [0; 16];
    let mut unused_a = [0x29; 16];
    let mut unused_b = [0x62; 16];
    for mode in [0, 1_u64 << 19] {
        set_mode(mode);
        assert_eq!(fpcr(), mode);
        set_flags(1 << 4);
        let psr = fpsr();
        apply_planes(
            LaneOp::Bin(0, EType::F16),
            2,
            2,
            &[&first_lo, &first_hi, &zero, &zero],
            &[&second_lo, &zero, &zero, &zero],
            &[&zero; 4],
            &mut [&mut lo, &mut hi, &mut unused_a, &mut unused_b],
        );
        assert_eq!(fpsr(), psr, "status changed with FPCR {mode:x}");
        assert_eq!(fpcr(), mode, "control changed");
        assert_eq!((unused_a, unused_b), ([0x29; 16], [0x62; 16]));
        assert_eq!(u16::from_le_bytes([lo[0], hi[0]]), 0x7e00);
        assert_eq!(u16::from_le_bytes([lo[1], hi[1]]), 2, "FPCR {mode:x}");
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[test]
#[should_panic(expected = "planes_to_words: lengths")]
fn native_half_preserves_existing_validation_of_short_input_planes() {
    let zero = [0; 16];
    let mut o = [[0; 16]; 4];
    let [a, b, c, d] = &mut o;
    apply_planes(
        LaneOp::Bin(0, EType::F16),
        2,
        2,
        &[&zero, &zero, &zero, &[]],
        &[&zero; 4],
        &[&zero; 4],
        &mut [a, b, c, d],
    );
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[test]
#[should_panic(expected = "words_to_planes: lengths")]
fn native_half_preserves_existing_validation_of_short_output_planes() {
    let zero = [0; 16];
    let mut o = [[0; 16]; 3];
    let [a, b, c] = &mut o;
    apply_planes(LaneOp::Bin(0, EType::F16), 2, 2, &[&zero; 4], &[&zero; 4], &[&zero; 4], &mut [a, b, c, &mut []]);
}
