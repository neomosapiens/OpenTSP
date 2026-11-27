//! The exact sums of an ABC (the wide accumulator's input) exist only for a profile that declares `wide_acc`: a profile without
//! the extension keeps no wide data in its ABC entries, so an ABC costs it no allocation for them and no reference count to move.
use ots_isa::Insn;
use ots_profile::Profile;
use std::sync::Arc;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::Observer;
use tsp_iss::slices::mxm::WType;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::Machine;

#[derive(Default)]
struct Rec;
impl Observer for Rec {}

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
    m.execute(i, &ctx, &mut Rec)
}

/// West plane 0 with `rows` installed rows of type `t`, ready, and an activation for an ABC issued at cycle 30.
fn plane0(profile: &str, t: WType, rows: usize, planes: u8) -> Machine {
    let profile = if profile == "ref" {
        Profile::load("ref").unwrap()
    } else {
        let path = format!("{}/../../../opentsp-arch-spec/config/test/{profile}.json", env!("CARGO_MANIFEST_DIR"));
        Profile::load_file(profile, &path).unwrap()
    };
    let mut m = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.wtype = t;
        p.rows = rows;
        p.ready_iw = 0;
        p.weights = Arc::new(vec![[0u8; MAX_LANES]; 2 * rows]);
    })
    .unwrap();
    let g = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
    for id in 0..planes {
        let mut v = [0u8; MAX_LANES];
        for (l, b) in v.iter_mut().enumerate() {
            *b = (l % 5) as u8;
        }
        m.edit_state(|state| state.srf.write(RowId::W, id, g - 1, 29, &v)).unwrap();
    }
    m
}

fn abc(m: &mut Machine, t: u32) {
    assert_eq!(exec(m, 105, 30, &Insn::ABC { r#type: t, src: 32 }), Exec::Issued);
    m.unit_state_flush(30, &mut Rec);
}

#[test]
fn a_profile_without_wide_acc_keeps_no_wide_data_in_an_abc_entry() {
    let mut m = plane0("ref", WType::F16, 4, 2);
    abc(&mut m, 1);
    assert!(m.mxm().planes[0].abc[0].wide.is_none(), "no wide_acc: the entry holds no exact sums");
}

#[test]
fn an_integer_abc_keeps_no_wide_data_even_with_wide_acc() {
    let mut m = plane0("ref_ext", WType::I8, 4, 1);
    abc(&mut m, 0);
    assert!(m.mxm().planes[0].abc[0].wide.is_none());
}

#[test]
fn a_profile_with_wide_acc_keeps_the_exact_sums_of_an_f16_abc() {
    let mut m = plane0("ref_ext", WType::F16, 4, 2);
    abc(&mut m, 1);
    let wide = m.mxm().planes[0].abc[0].wide.as_ref().expect("wide_acc: the entry holds the exact sums");
    assert_eq!(wide.len(), MAX_LANES);
}
