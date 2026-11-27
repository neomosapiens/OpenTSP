//! A data replacement is identified using its own write gap, at ordinary and boundary times.
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::RowId;
use tsp_iss::srf::Srf;

#[test]
fn only_an_intervening_same_diagonal_data_write_suppresses_a_spare_report() {
    for cycle in [100u64, (1u64 << 63) + 100, u64::MAX - 10] {
        for row in [RowId::E, RowId::W] {
            for gap in [1u8, 5, 20] {
                for (write_gap, shift, replaced) in [
                    (gap - 1, 0i64, false),
                    (gap, 0, false),
                    (gap + 1, 0, true),
                    (gap + 3, 0, true),
                    (gap + 4, 0, false),
                    (gap + 1, 1, false),
                ] {
                    let mut srf = Srf::new(16);
                    let mut sp = [0u32; 30];
                    sp[0] = 1;
                    sp[1] = 1;
                    assert!(!srf.report(row, gap, cycle, 1, 1, &sp));
                    let write_cycle =
                        u64::try_from(i128::from(cycle) + i128::from(write_gap) - i128::from(gap) + i128::from(shift))
                            .unwrap();
                    srf.write(row, 0, write_gap, write_cycle, &[7; MAX_LANES]);
                    let read_cycle = cycle + 4;
                    assert_eq!(
                        srf.spare(row, 0, gap + 4, read_cycle),
                        u32::from(!replaced),
                        "cycle={cycle} row={row:?} report_gap={gap} data_gap={write_gap} shift={shift}"
                    );
                    assert_eq!(
                        srf.spare(row, 1, gap + 4, read_cycle),
                        1,
                        "a write to another stream does not replace its report"
                    );
                    let other = if row == RowId::E { RowId::W } else { RowId::E };
                    assert_eq!(srf.spare(other, 0, gap + 4, read_cycle), 0);
                }
            }
        }
    }
}
