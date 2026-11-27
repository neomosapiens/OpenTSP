//! The MXM structure keys (`mxm.planes_per_hemisphere`, `mxm.acc_depth`, `mxm.f16_rows`, `mxm.fp8_rate`) in the ISS: what the
//! geometry reads, which queues and planes exist, how many accumulators a plane has, which ACC indices execute, and which
//! staged parts of one plane clash (`slices/mxm/mxm_regs.sail`, `postlude/tick.sail` `mxm_clash`). The end-to-end programs
//! are `opentsp-arch-spec/test/directed/mxm_keys.py`.
use ots_isa::{Code, Insn};
use ots_profile::Profile;
use std::sync::Arc;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{mxm_plane, queue_present};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::Observer;
use tsp_iss::slices::mxm::{AbcEntry, Mxm, Staged, Value, WType};
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::Machine;

fn profile(file: &str) -> Profile {
    let path = format!("{}/../../../opentsp-arch-spec/config/{file}.json", env!("CARGO_MANIFEST_DIR"));
    let name = file.rsplit('/').next().unwrap();
    Profile::load_file(name, path).unwrap()
}

fn machine(file: &str) -> Machine {
    Machine::new(&profile(file), 0, Box::new(NoLinks)).unwrap()
}

struct Quiet;
impl Observer for Quiet {}

fn exec(m: &mut Machine, q: usize, cycle: u64, i: &Insn) -> Exec {
    let ctx =
        Issue { cycle, queue: q, element: 0, addr_offset: 0, stream_offset: 0, thread: 0, lat: m.latency_of_instr(i) };
    m.execute(i, &ctx, &mut Quiet)
}

#[test]
fn the_geometry_reads_the_four_keys() {
    let g = |f: &str| {
        let m = machine(f);
        (m.geom().mxm_planes, m.geom().mxm_acc_depth, m.geom().mxm_f16_rows, m.geom().mxm_fp8_rate)
    };
    assert_eq!(g("ref"), (2, 2, 160, 1));
    assert_eq!(g("mini"), (2, 2, 8, 1));
    assert_eq!(g("test/acc8"), (2, 8, 160, 1));
    assert_eq!(g("test/f16_tandem"), (2, 2, 320, 1));
    assert_eq!(g("test/fp8_rate2"), (2, 2, 160, 2));
    assert_eq!(g("test/one_plane"), (1, 2, 160, 1));
    assert_eq!(g("test/mxm_wide"), (2, 32, 16, 2));
}

#[test]
fn the_registers_have_two_planes_per_hemisphere_and_acc_depth_accumulators() {
    let m = machine("test/one_plane");
    assert_eq!(m.mxm().planes.len(), 2);
    let m = machine("test/mxm_wide");
    assert_eq!(m.mxm().planes.len(), 4);
    assert!(m.mxm().planes.iter().all(|p| p.acc.len() == 32 && p.acc_poison.len() == 32));
    assert_eq!(machine("ref").mxm().planes[0].acc.len(), 2);
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_planes_per_hemisphere
#[test]
fn the_queues_of_an_absent_plane_do_not_exist_and_the_east_plane_follows_the_west_ones() {
    let one = machine("test/one_plane");
    let g = one.geom();
    for q in [104, 105, 106, 110, 111, 112] {
        assert!(queue_present(g, q), "queue {q}");
    }
    for q in [107, 108, 109, 113, 114, 115] {
        assert!(!queue_present(g, q), "queue {q}");
    }
    assert_eq!((mxm_plane(g, 104), mxm_plane(g, 110)), (0, 1));
    let r = machine("ref");
    assert!((104..=115).all(|q| queue_present(r.geom(), q)));
    assert_eq!((mxm_plane(r.geom(), 107), mxm_plane(r.geom(), 110), mxm_plane(r.geom(), 113)), (1, 2, 3));
}

fn entry(q: usize, t: WType) -> AbcEntry {
    AbcEntry { dispatch: 30, commit: 50, ptype: t, q, product: Arc::new([0; MAX_LANES]), wide: None, poison: 0 }
}

fn staged_abc(q: usize, plane: usize, t: WType) -> Staged {
    Staged { q, plane, value: Value::Abc(vec![entry(q, t)]) }
}

fn bad(m: &Mxm) -> Vec<usize> {
    let mut b = [false; 145];
    m.clashing_queues(&mut b);
    (0..145).filter(|&q| b[q]).collect()
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_fp8_rate
#[test]
fn a_plane_accepts_fp8_rate_fp8_abcs_in_one_cycle() {
    let mut m = Mxm::with(2, 2, 1);
    m.staged = vec![staged_abc(105, 0, WType::Fp8), staged_abc(106, 0, WType::Fp8)];
    assert_eq!(bad(&m), vec![105, 106], "rate 1: two ABCs of one plane clash");
    let mut m = Mxm::with(2, 2, 2);
    m.staged = vec![staged_abc(105, 0, WType::Fp8), staged_abc(106, 0, WType::Fp8)];
    assert_eq!(bad(&m), Vec::<usize>::new(), "rate 2: two FP8 ABCs are accepted");
    m.staged.push(staged_abc(104, 0, WType::Fp8));
    assert_eq!(bad(&m), vec![104, 105, 106], "rate 2: a third clashes with both");
    m.staged = vec![staged_abc(105, 0, WType::Fp8), staged_abc(106, 0, WType::I8)];
    assert_eq!(bad(&m), vec![105, 106], "rate 2: an FP8 ABC and an i8 ABC clash");
    m.staged = vec![staged_abc(105, 0, WType::Fp8), staged_abc(108, 1, WType::Fp8)];
    assert_eq!(bad(&m), Vec::<usize>::new(), "ABCs of two planes never clash");
}

#[test]
fn each_accumulator_is_a_part_of_its_own() {
    let acc = |q: usize, a: usize| Staged { q, plane: 0, value: Value::Acc(a, Arc::new([0; MAX_LANES]), 0) };
    let mut m = Mxm::with(2, 8, 1);
    m.staged = vec![acc(104, 3), acc(106, 3)];
    assert_eq!(bad(&m), vec![104, 106], "the same accumulator of one plane clashes");
    m.staged = vec![acc(104, 3), acc(106, 7)];
    assert_eq!(bad(&m), Vec::<usize>::new(), "two accumulators do not clash");
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_acc_depth
#[test]
fn an_acc_index_below_the_depth_executes_and_one_at_the_depth_is_refused() {
    let mut m = machine("test/acc8");
    m.edit_state(|state| state.mxm.planes[0].abc = vec![entry(105, WType::I8)]).unwrap();
    m.edit_state(|state| state.mxm.planes[0].rows = 16).unwrap();
    let acc = |a: u32| Insn::ACC { r#type: 2, dst: 8, mode: 0, acc: a };
    assert_eq!(exec(&mut m, 106, 60, &acc(7)), Exec::Issued);
    assert!(matches!(m.mxm().staged.last().unwrap().value, Value::Acc(7, ..)), "accumulator 7 is the staged part");
    assert_eq!(exec(&mut m, 106, 61, &acc(8)), Exec::Diagnostic(Code::Opr001), "the depth is not an index");
}

// model: unit_tests/test_vxm_mxm.sail test_mxm_f16_capacity
#[test]
fn an_f16_capacity_outside_half_the_rows_to_all_of_them_is_refused() {
    let text =
        std::fs::read_to_string(format!("{}/../../../opentsp-arch-spec/config/ref.json", env!("CARGO_MANIFEST_DIR")))
            .unwrap();
    let with = |rows: u64| {
        let cfg = text.replace("\"f16_rows\": 160", &format!("\"f16_rows\": {rows}"));
        Profile::from_bytes("ref", cfg.as_bytes()).unwrap().timing_config_error()
    };
    assert_eq!(with(160), None);
    assert_eq!(with(320), None);
    assert_eq!(with(152).as_deref(), Some("mxm.f16_rows: 152 must be between mxm.rows / 2 and mxm.rows (160 and 320)"));
    assert!(with(328).is_some(), "above the rows");
}
