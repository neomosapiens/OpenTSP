//! Ports of the profile unit tests of the Sail model (`unit_tests/test_profile.sail`):
//! `sram.words_per_slice`, `geometry.mem_slices_per_hemisphere` and `mxm.rows`.
//! Like the Sail tests they are profile-generic (every expected value is computed from the
//! configuration) and run on `ref`, `mini` and every profile of `config/test/`.
use ots_isa::Insn;
use ots_profile::Profile;
use tsp_iss::layout::{issuer_of, mem_slice_present, queue_present, read_gap, write_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::NoObserver;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::{Diag, Machine};

const PROFILES: [&str; 7] = ["ref", "mini", "link_tight", "mid_min", "mini_small", "ref_slow", "ref_small"];

fn machine(name: &str) -> Machine {
    let profile = if matches!(name, "ref" | "mini") {
        Profile::load(name)
    } else {
        Profile::load_file(
            name,
            format!("{}/../../../opentsp-arch-spec/config/test/{name}.json", env!("CARGO_MANIFEST_DIR")),
        )
    }
    .unwrap();
    Machine::new(&profile, 0, Box::new(NoLinks)).unwrap()
}

/// `slice_test_set_issue_context` + `execute_test`: execute `i` as dispatched at cycle `c` on queue `q`.
fn exec(m: &mut Machine, q: usize, c: u64, i: &Insn) -> Exec {
    let ctx = Issue {
        cycle: c,
        queue: q,
        element: 0,
        addr_offset: 0,
        stream_offset: 0,
        thread: 0,
        lat: m.latency_of_instr(i),
    };
    m.execute(i, &ctx, &mut NoObserver)
}

// model: unit_tests/test_profile.sail test_f3_mem_slices
#[test]
fn test_f3_mem_slices() {
    // The queue map has 144 ids in every profile; MEM slice s exists when (s mod 44) < the key,
    // and an absent slice has no station.
    for name in PROFILES {
        let m = machine(name);
        let g = m.geom();
        let mut present = 0;
        for q in 0..144usize {
            let want = q > 87 || (q % 44) < g.mem_slices as usize;
            assert_eq!(queue_present(g, q), want, "{name} queue {q}");
            if q <= 87 {
                assert_eq!(mem_slice_present(g, q), want, "{name} slice {q}");
                if want {
                    present += 1;
                }
                // A MEM slice reads and writes both rows at its quad's gaps, or at none when it is absent.
                let unit = issuer_of(q).unit;
                for row in [RowId::E, RowId::W] {
                    assert_eq!(read_gap(g, unit, q, row).is_some(), want, "{name} queue {q}");
                    assert_eq!(write_gap(g, unit, q, row).is_some(), want, "{name} queue {q}");
                }
            }
        }
        assert_eq!(present, 2 * g.mem_slices, "{name}");
        assert_eq!(g.mem_slices % 4, 0, "{name}");
        // The slices nearest the VXM always exist (the key is at least 4: one quad).
        assert!(mem_slice_present(g, 0) && mem_slice_present(g, 44), "{name}");
    }
}

// model: unit_tests/test_profile.sail test_f3_sram_geometry
#[test]
fn test_f3_sram_geometry() {
    // words_per_slice is a power of two from 256 to 8192 and the bank bit is its top address bit.
    for name in PROFILES {
        let m = machine(name);
        let words = m.geom().words_per_slice;
        assert!((256..=8192).contains(&words), "{name}");
        assert_eq!(1u64 << (m.geom().bank_bit + 1), words, "{name}");
        let at = |a: u64| a as u32;
        assert_eq!(m.sram_index(at(words - 1)), Some(words - 1), "{name}");
        assert_eq!(m.sram_bank(at(words - 1)), 1, "{name}");
        assert_eq!(m.sram_bank(at(words / 2 - 1)), 0, "{name}");
        assert_eq!(m.sram_bank(at(words / 2)), 1, "{name}");
        // An address from words_per_slice on is outside the slice (the field has 13 bits).
        if words < 8192 {
            assert_eq!(m.sram_index(at(words)), None, "{name}");
        }
    }
}

/// Feed the 16 streams W:0..15 to the MXM input of the instruction issued at cycle `c`, read at
/// `c + d_skew` (`f3_feed16`).
fn f3_feed16(m: &mut Machine, c: u64, d_skew: u64) {
    let t = c + d_skew;
    let g = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).expect("no MXM read gap");
    assert!(g > 0 && t > 0);
    let zero: Vec320 = [0; tsp_iss::geom::MAX_LANES];
    for s in 0..16u8 {
        m.edit_state(|state| state.srf.write(RowId::W, s, g - 1, t - 1, &zero)).unwrap();
    }
}

const I8: u32 = 0;
const F16: u32 = 1;
const U8: u32 = 2;

// model: unit_tests/test_profile.sail test_f3_mxm_lw_capacity
#[test]
fn test_f3_mxm_lw_capacity() {
    // The weight buffer holds mxm.rows stream values; the load that would pass that is OTS-E-RES-002.
    for name in PROFILES {
        for t in [I8, F16] {
            let mut m = machine(name);
            let rows = m.geom().mxm_rows.expect("mxm.rows") as usize;
            let n = rows / 16;
            let lw = Insn::LW { r#type: t, src: 2 }; // W:0, a group of 16
            let d_skew = m.latency_of_instr(&lw).d_skew as u64;
            for i in 0..=n {
                let c = 100 + 4 * i as u64;
                f3_feed16(&mut m, c, d_skew);
                if i < n {
                    assert_eq!(exec(&mut m, 104, c, &lw), Exec::Issued, "{name} type {t} load {i}");
                    m.unit_state_flush(c, &mut NoObserver);
                    assert_eq!(m.mxm().planes[0].buf_rows, 16 * (i + 1), "{name} type {t} load {i}");
                } else {
                    assert_eq!(exec(&mut m, 104, c, &lw), Exec::Diagnostic(Diag::Res002), "{name} type {t}");
                    assert_eq!(m.mxm().planes[0].buf_rows, rows, "{name} type {t}");
                }
            }
        }
    }
}

// model: unit_tests/test_profile.sail test_f3_mxm_iw_capacity
#[test]
fn test_f3_mxm_iw_capacity() {
    // An installed matrix holds mxm.rows i8 rows or mxm.rows / 2 f16 rows.
    for name in PROFILES {
        for t in [U8, F16] {
            let mut m = machine(name);
            let rows = m.geom().mxm_rows.expect("mxm.rows") as usize;
            let n = rows / 16;
            let iws = Insn::IWS { r#type: t, src: 2 };
            let d_skew = m.latency_of_instr(&iws).d_skew as u64;
            for i in 0..=n {
                let c = 100 + 4 * i as u64;
                f3_feed16(&mut m, c, d_skew);
                if i < n {
                    assert_eq!(exec(&mut m, 104, c, &iws), Exec::Issued, "{name} type {t} install {i}");
                    m.unit_state_flush(c, &mut NoObserver);
                    // i8 and u8 rows are stream values; f16 rows take two each.
                    let want = if t == F16 { 8 * (i + 1) } else { 16 * (i + 1) };
                    assert_eq!(m.mxm().planes[0].rows, want, "{name} type {t} install {i}");
                } else {
                    assert_eq!(exec(&mut m, 104, c, &iws), Exec::Diagnostic(Diag::Res002), "{name} type {t}");
                }
            }
            assert_eq!(m.mxm().planes[0].rows, if t == F16 { rows / 2 } else { rows }, "{name} type {t}");
        }
    }
}
