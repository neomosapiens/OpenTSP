//! Tests of `layout`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::{Access, Isa};
use ots_profile::Profile;
use tsp_iss::layout::*;
use tsp_iss::Geom;

fn geom(name: &str) -> (Geom, Isa) {
    let p = Profile::load(name).unwrap();
    (Geom::new(&p), Isa::new(&p))
}

// model: unit_tests/test_slices.sail test_queue_units
#[test]
fn every_data_queue_issues_to_its_documented_unit() {
    use UnitKind::*;
    for n in 0..144usize {
        let i = issuer_of(n);
        let ok = match n {
            0..=43 => i.unit == MemW && i.slice as usize == n,
            44..=87 => i.unit == MemE && i.slice as usize == n,
            88..=103 => i.unit == Vxm,
            104..=109 => i.unit == MxmW && i.plane == if n <= 106 { 0 } else { 1 },
            110..=115 => i.unit == MxmE && i.plane == if n <= 112 { 2 } else { 3 },
            _ => i.sxm == if n >= 130 { 1 } else { 0 },
        };
        assert!(ok, "queue {n}");
    }
    for h in 0..2usize {
        let mut count = std::collections::HashMap::new();
        for n in 116 + 14 * h..=129 + 14 * h {
            let i = issuer_of(n);
            assert_eq!(i.sxm as usize, h);
            let k = i.unit.name().replace("_W_", "_").replace("_E_", "_");
            *count.entry(k).or_insert(0) += 1;
        }
        let get = |k: &str| count[k];
        assert_eq!(
            (
                get("SXM_DST"),
                get("SXM_TSP"),
                get("SXM_SHF"),
                get("SXM_PRM"),
                get("SXM_SEL"),
                get("C2C_TX"),
                get("C2C_RX")
            ),
            (4, 2, 2, 1, 1, 2, 2)
        );
    }
    assert_eq!(issuer_of(126).unit, C2cWTx);
    assert_eq!(issuer_of(128).unit, C2cWRx);
    assert_eq!(issuer_of(140).unit, C2cETx);
    assert_eq!(issuer_of(143).unit, C2cERx);
}

// The gaps of encoding.json `stations` against the model's layout, on every data
// queue and both rows.
#[test]
fn layout_matches_encoding_json_stations() {
    for name in ["ref", "mini"] {
        let (g, isa) = geom(name);
        for q in 0..144usize {
            let i = issuer_of(q);
            let quad = if matches!(i.unit, UnitKind::MemW | UnitKind::MemE) { mem_quad(q) } else { 0 };
            for (row, east) in [(RowId::E, true), (RowId::W, false)] {
                let rd = isa.station(i.unit.name(), Access::Read, east, quad);
                let wr = isa.station(i.unit.name(), Access::Write, east, quad);
                assert_eq!(
                    read_gap(&g, i.unit, i.slice as usize, row).map(u32::from),
                    rd,
                    "{name}: queue {q} read row {row:?}"
                );
                assert_eq!(
                    write_gap(&g, i.unit, i.slice as usize, row).map(u32::from),
                    wr,
                    "{name}: queue {q} write row {row:?}"
                );
            }
        }
    }
}

#[test]
fn absent_slices_have_no_station() {
    let (mut g, _) = geom("ref");
    g.mem_slices = 8;
    assert!(
        queue_present(&g, 7)
            && !queue_present(&g, 8)
            && !queue_present(&g, 43)
            && queue_present(&g, 44)
            && queue_present(&g, 100)
    );
    assert_eq!(read_gap(&g, UnitKind::MemW, 8, RowId::E), None);
    assert_eq!(write_gap(&g, UnitKind::MemE, 52, RowId::W), None);
    assert!(read_gap(&g, UnitKind::MemW, 7, RowId::E).is_some());
}

#[test]
fn mxm_queues() {
    assert!((104..116).all(is_mxm_queue));
    assert!(!is_mxm_queue(103) && !is_mxm_queue(116) && !is_mxm_queue(144));
}
