//! Tests of `geom`, moved from the module's in-source test block (they use the crate's public API).
use ots_profile::Profile;
use tsp_iss::geom::*;

#[test]
fn queue_index_maps_the_control_channel() {
    assert_eq!(queue_index(0), Some(0));
    assert_eq!(queue_index(143), Some(143));
    assert_eq!(queue_index(144), None);
    assert_eq!(queue_index(0xfffe), None);
    assert_eq!(queue_index(0xffff), Some(CONTROL_INDEX));
}

#[test]
fn shipped_profiles() {
    let r = Geom::new(&Profile::load("ref").unwrap());
    assert_eq!((r.vlanes, r.words_per_slice, r.iq_units, r.quads), (320, 8192, 320, 11));
    let mp = Profile::load("mini").unwrap();
    let m = Geom::new(&mp);
    assert_eq!((m.vlanes, m.words_per_slice, m.mxm_rows), (16, 256, mp.uint("mxm.rows")));
    assert!(r.slice_present(0) && r.slice_present(87) && !r.slice_present(88));
    let small = Geom { mem_slices: 8, quads: 2, ..r };
    assert!(small.slice_present(7) && !small.slice_present(8) && !small.slice_present(43));
    assert!(small.slice_present(44) && small.slice_present(51) && !small.slice_present(52));
    assert!(small.check_lanes().is_ok());
    assert!(Geom { vlanes: 528, ..small.clone() }.check_lanes().is_err());
    assert!(small.check_link_latency().is_ok());
    assert!(Geom { link_latency: 300, ..small }.check_link_latency().is_err());
}

#[test]
fn the_sram_address_width_is_in_the_geometry() {
    let dir = format!("{}/../../../opentsp-arch-spec/config", env!("CARGO_MANIFEST_DIR"));
    for (name, file, bits, words, bank) in [
        ("ref", "ref.json", 13, 8192, 12),
        ("mini", "mini.json", 13, 256, 7),
        ("w14", "test/w14.json", 14, 16384, 13),
        ("w15", "test/w15.json", 15, 32768, 14),
        ("w16", "test/w16.json", 16, 65536, 15),
    ] {
        let p = Profile::load_file(name, format!("{dir}/{file}")).unwrap();
        p.validate().unwrap();
        let g = Geom::new(&p);
        assert_eq!((g.addr_bits, g.words_per_slice, g.bank_bit), (bits, words, bank), "{name}");
        assert!(g.words_per_slice <= 1 << g.addr_bits, "{name}: a slice has at most 2^addr_bits words");
    }
}
