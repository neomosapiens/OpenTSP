use super::sram_hint_tail;

#[test]
fn sram_hint_covers_final_physical_line_without_duplicate() {
    for size in 0..=640 {
        for alignment in 0..64 {
            let starts: Vec<_> = (0..size).step_by(64).collect();
            let mut physical: Vec<_> = starts.iter().map(|offset| (alignment + offset) / 64).collect();
            if let Some(offset) = sram_hint_tail(alignment, size) {
                assert!(offset < size);
                physical.push((alignment + offset) / 64);
            }
            let expected: Vec<_> = if size == 0 { Vec::new() } else { (0..=(alignment + size - 1) / 64).collect() };
            assert_eq!(physical, expected, "alignment {alignment} size {size}");
        }
    }
}
