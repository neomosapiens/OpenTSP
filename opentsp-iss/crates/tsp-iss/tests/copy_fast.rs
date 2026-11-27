//! `copy::copy` against `copy_from_slice`: every length, odd offsets, and the bytes around the target untouched.
use tsp_iss::copy::copy;

#[test]
fn copy_equals_copy_from_slice_for_every_length_and_offset() {
    let src: Vec<u8> = (0..700u32).map(|i| (i * 37 + 11) as u8 | 1).collect();
    for len in 0..=400usize {
        for (so, doff) in [(0usize, 0usize), (1, 0), (0, 3), (5, 7), (15, 16)] {
            let mut got = vec![0xAAu8; 500];
            let mut want = got.clone();
            copy(&mut got[doff..doff + len], &src[so..so + len]);
            want[doff..doff + len].copy_from_slice(&src[so..so + len]);
            assert_eq!(got, want, "len {len} src offset {so} dst offset {doff}");
        }
    }
}

#[test]
#[should_panic(expected = "lengths differ")]
fn copy_of_unequal_lengths_panics() {
    let mut d = [0u8; 16];
    copy(&mut d, &[1u8; 17]);
}
