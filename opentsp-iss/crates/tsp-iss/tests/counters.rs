//! The activity counters: what each unit class did, counted where an instruction has
//! finished executing and exposed in the `activity` object of `tsp-iss run --stats`. They are non-normative: they never
//! change a trace or a cycle count.
mod common;
use common::*;
use ots_isa::{EntryId, Insn};
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{issuer_of, read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::Observer;
use tsp_iss::slices::mxm::WType;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::Machine;

fn machine(name: &str) -> Machine {
    let mut m = Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap();
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap();
    m
}
/// A machine as an embedder gets it from the library with no setting of its own.
fn machine_as_built(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

struct Quiet;
impl Observer for Quiet {}

fn exec(m: &mut Machine, q: usize, cycle: u64, i: &Insn) -> Exec {
    let ctx =
        Issue { cycle, queue: q, element: 0, addr_offset: 0, stream_offset: 0, thread: 0, lat: m.latency_of_instr(i) };
    m.execute(i, &ctx, &mut Quiet)
}

fn vec_of(f: impl Fn(usize) -> u8) -> Vec320 {
    let mut v = [0u8; MAX_LANES];
    for (l, b) in v.iter_mut().enumerate().take(320) {
        *b = f(l);
    }
    v
}

/// Counting is opt-in for the library too: a chip an embedder builds without asking counts nothing, so a serving path that forgets the
/// setting does not pay for counters nobody reads (`tsp-iss run` asks for them with `--stats` and `--energy`).
#[test]
fn a_machine_built_with_the_library_default_counts_nothing() {
    let mut m = machine_as_built("ref");
    assert!(!m.activity().enabled());
    assert_eq!(exec(&mut m, 4, 10, &Insn::Read { addr: 3, dst: E0, thread: 0 }), Exec::Issued);
    assert_eq!(
        (m.activity().mem_total().reads, m.activity().issued.iter().sum::<u64>(), m.activity().srf_writes),
        (0, 0, [0, 0])
    );
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap();
    exec(&mut m, 4, 11, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    assert_eq!(m.activity().mem_total().reads, 1, "asked for, it counts");
}

const E0: u32 = 0;
const W16: u32 = 32 + 16;

#[test]
fn a_mem_read_counts_the_bytes_of_its_active_tiles_on_its_slice() {
    let mut m = machine("ref");
    assert_eq!(m.activity().mem_total().reads, 0);
    assert_eq!(exec(&mut m, 4, 10, &Insn::Read { addr: 3, dst: E0, thread: 0 }), Exec::Issued);
    let s = m.activity().mem[4];
    assert_eq!((s.reads, s.writes, s.read_bytes, s.write_bytes), (1, 0, 320, 0), "20 tiles of 16 bytes");
    assert_eq!(m.activity().mem_total().reads, 1);
    assert_eq!(m.activity().srf_writes, [1, 0], "one vector scheduled on row E");
    // SetVL 3 on the queue leaves tiles 0..2
    assert_eq!(exec(&mut m, 4, 11, &Insn::SetVL { n: 3 }), Exec::Issued);
    exec(&mut m, 4, 12, &Insn::Read { addr: 3, dst: W16, thread: 0 });
    let s = m.activity().mem[4];
    assert_eq!((s.reads, s.read_bytes), (2, 320 + 48), "3 tiles of 16 bytes");
    assert_eq!(m.activity().srf_writes, [1, 1], "the write on row W");
    // a Read on another slice is counted there
    exec(&mut m, 5, 13, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    assert_eq!((m.activity().mem[5].reads, m.activity().mem[4].reads), (1, 2));
    // a Read at an address outside the segment raises a diagnostic and counts nowhere
    m.edit_state(|state| state.mem_segment[4] = 16).unwrap();
    assert!(matches!(exec(&mut m, 4, 14, &Insn::Read { addr: 20, dst: E0, thread: 0 }), Exec::Diagnostic(_)));
    assert_eq!((m.activity().mem[4].reads, m.activity().issued[EntryId::Read.index()]), (2, 3));
}

#[test]
fn a_cycle_in_which_a_unit_class_executes_twice_is_one_active_cycle() {
    let mut m = machine("ref");
    exec(&mut m, 4, 10, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    exec(&mut m, 5, 10, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    exec(&mut m, 6, 11, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    assert_eq!(m.activity().mem_total().reads, 3);
    assert_eq!(m.activity().active_cycles[0], 2, "cycles 10 and 11");
    // a restart of the run (the cycles count from 0 again) starts the bookkeeping over, the counts stay
    m.reset();
    exec(&mut m, 4, 11, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    assert_eq!((m.activity().mem_total().reads, m.activity().active_cycles[0]), (4, 3));
}

#[test]
fn a_mem_write_counts_the_tiles_of_its_thread() {
    let mut m = machine("ref");
    let rg = read_gap(m.geom(), UnitKind::MemW, 0, RowId::W).unwrap();
    m.edit_state(|state| state.srf.write(RowId::W, 16, rg - 1, 49, &vec_of(|_| 0x55))).unwrap();
    assert_eq!(exec(&mut m, 0, 50, &Insn::Write { addr: 5, src: W16, thread: 0 }), Exec::Issued);
    let s = m.activity().mem[0];
    assert_eq!((s.reads, s.writes, s.write_bytes), (0, 1, 320));
    // thread 1 of a slice split at 7: tiles 7..19, thirteen of them
    m.edit_state(|state| state.masks.mem_thread_split[0] = 7).unwrap();
    m.edit_state(|state| state.srf.write(RowId::W, 16, rg - 1, 59, &vec_of(|_| 0x33))).unwrap();
    assert_eq!(exec(&mut m, 0, 60, &Insn::Write { addr: 6, src: W16, thread: 1 }), Exec::Issued);
    let s = m.activity().mem[0];
    assert_eq!((s.writes, s.write_bytes), (2, 320 + 13 * 16));
    // a missing operand is a diagnostic: nothing is counted
    assert!(matches!(exec(&mut m, 0, 70, &Insn::Write { addr: 7, src: W16, thread: 0 }), Exec::Diagnostic(_)));
    assert_eq!(m.activity().mem[0].writes, 2);
    assert_eq!(m.activity().issued[EntryId::Write.index()], 2);
}

#[test]
fn a_vxm_instruction_counts_active_lanes_times_the_operations_per_lane() {
    let mut m = machine("ref");
    let g = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let a = vec_of(|l| l as u8);
    let add = Insn::Add { r#type: 0, dst: 8, lhs: 0, rhs: 1, sg4: 0, lhs_chain: 0, rhs_chain: 0 };
    let sg4 = Insn::Max { r#type: 0, dst: 8, lhs: 0, rhs: 4, sg4: 1, lhs_chain: 0, rhs_chain: 0 };
    // the operands reach the VXM in the cycle the instruction executes: dispatch + d_skew
    let feed = |m: &mut Machine, i: &Insn, c: u64, streams: std::ops::Range<u8>| {
        let at = c + m.latency_of_instr(i).d_skew as u64;
        for s in streams {
            m.edit_state(|state| state.srf.write(RowId::E, s, g - 1, at - 1, &a)).unwrap();
        }
    };
    feed(&mut m, &add, 40, 0..2);
    assert_eq!(exec(&mut m, 88, 40, &add), Exec::Issued);
    let i8_code = 0; // EType::I8
    assert_eq!(m.activity().vxm[EntryId::Add.index()][i8_code], 320, "every lane once");
    // SetVL 5 on the VXM queue: 80 lanes
    exec(&mut m, 88, 41, &Insn::SetVL { n: 5 });
    feed(&mut m, &add, 50, 0..2);
    assert_eq!(exec(&mut m, 88, 50, &add), Exec::Issued);
    assert_eq!(m.activity().vxm[EntryId::Add.index()][i8_code], 320 + 80);
    // an SG4 group of four i8 operands: four operations per lane
    m.edit_state(|state| state.masks.queue_vl[88] = 20).unwrap();
    feed(&mut m, &sg4, 60, 0..8);
    assert_eq!(exec(&mut m, 88, 60, &sg4), Exec::Issued);
    assert_eq!(m.activity().vxm[EntryId::Max.index()][i8_code], 320 * 4);
    assert_eq!(m.activity().vxm_ops_total(), 320 + 80 + 1280);
    let j = m.activity().to_json();
    assert_eq!(j["vxm"]["by_class"]["add_sub"], 400, "Add");
    assert_eq!(j["vxm"]["by_class"]["minmax"], 1280, "Max");
    assert_eq!(j["vxm"]["by_type"]["i8"], 1680);
    // one plane per i8 result, four for the SG4 group: the stream writes follow the planes scheduled
    assert_eq!(m.activity().srf_writes, [1 + 1 + 4, 0]);
}

#[test]
fn an_abc_counts_macs_of_the_installed_rows_by_type() {
    let mut m = machine("ref");
    // plane 0 (west): 100 installed i8 rows, ready
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.wtype = WType::I8;
        p.rows = 100;
        p.ready_iw = 0;
        p.weights = std::sync::Arc::new(vec![[0u8; MAX_LANES]; 100]);
    })
    .unwrap();
    let g = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
    m.edit_state(|state| state.srf.write(RowId::W, 0, g - 1, 29, &vec_of(|l| (l % 7) as u8))).unwrap();
    assert_eq!(exec(&mut m, 105, 30, &Insn::ABC { r#type: 0, src: 32 }), Exec::Issued);
    assert_eq!(m.activity().mxm.abc, [1, 0, 0, 0]);
    assert_eq!(m.activity().mxm.macs, [100 * 320, 0, 0, 0], "100 rows over 320 lanes");
    assert_eq!(m.activity().mxm.abc_by_plane, [1, 0, 0, 0]);
    assert_eq!(m.activity().macs_total(), 32_000);
    // an ABC with no installed matrix is a diagnostic: not counted
    assert!(matches!(exec(&mut m, 108, 31, &Insn::ABC { r#type: 0, src: 32 }), Exec::Diagnostic(_)));
    assert_eq!(m.activity().mxm.abc, [1, 0, 0, 0]);
}

/// An installed matrix of `rows` rows of type `t` on west plane 0, ready, and an activation on W:0 (and W:1) for cycle `at`.
fn plane0_with(rows: usize, t: WType, at: u64, planes: u8) -> Machine {
    let mut m = machine("ref");
    m.edit_state(|state| {
        let p = &mut state.mxm.planes[0];
        p.wtype = t;
        p.rows = rows;
        p.ready_iw = 0;
        p.weights = std::sync::Arc::new(vec![[0u8; MAX_LANES]; 2 * rows]);
    })
    .unwrap();
    let g = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
    for id in 0..planes {
        m.edit_state(|state| state.srf.write(RowId::W, id, g - 1, at - 1, &vec_of(|l| (l % 5) as u8))).unwrap();
    }
    m
}

#[test]
fn pair_types_are_counted_with_f16() {
    for (code, ty) in [(1, WType::F16), (4, WType::Bf16), (5, WType::Bfp8)] {
        let mut m = plane0_with(16, ty, 30, 2);
        assert_eq!(exec(&mut m, 105, 30, &Insn::ABC { r#type: code, src: 32 }), Exec::Issued, "{ty:?}");
        assert_eq!(m.activity().mxm.abc, [0, 0, 1, 0], "{ty:?}");
    }
}

#[test]
fn macs_are_counted_under_the_abc_type_and_acc_counts_its_lanes() {
    // f16: 50 row pairs, two activation planes: 50 x 320 f16 MACs
    let mut m = plane0_with(50, WType::F16, 30, 2);
    assert_eq!(exec(&mut m, 105, 30, &Insn::ABC { r#type: 1, src: 32 }), Exec::Issued);
    assert_eq!((m.activity().mxm.abc, m.activity().mxm.macs), ([0, 0, 1, 0], [0, 0, 50 * 320, 0]), "i8 u8 f16 fp8");
    // u8 activations on i8 weights count as u8
    let mut m = plane0_with(7, WType::I8, 30, 1);
    assert_eq!(exec(&mut m, 105, 30, &Insn::ABC { r#type: 2, src: 32 }), Exec::Issued);
    assert_eq!((m.activity().mxm.abc, m.activity().mxm.macs), ([0, 1, 0, 0], [0, 7 * 320, 0, 0]));
    // fp8: ABC8 on fp8 weights
    let mut m = plane0_with(9, WType::Fp8, 30, 1);
    assert_eq!(exec(&mut m, 105, 30, &Insn::ABC8 { fmt: 0, src: 32 }), Exec::Issued);
    assert_eq!((m.activity().mxm.abc, m.activity().mxm.macs), ([0, 0, 0, 1], [0, 0, 0, 9 * 320]));
    // ACC produces mxm.rows (320) accumulator lanes whatever the installed rows: an ABC result of 100 rows
    let mut m = machine("ref");
    m.edit_state(|state| {
        state.mxm.planes[0].abc = vec![tsp_iss::slices::mxm::AbcEntry {
            dispatch: 30,
            commit: 40,
            ptype: WType::I8,
            q: 105,
            product: std::sync::Arc::new([0; MAX_LANES]),
            wide: None,
            poison: 0,
        }]
    })
    .unwrap();
    m.edit_state(|state| state.mxm.planes[0].rows = 100).unwrap();
    assert_eq!(exec(&mut m, 106, 40, &Insn::ACC { r#type: 2, dst: 8, mode: 0, acc: 0 }), Exec::Issued);
    assert_eq!((m.activity().mxm.acc, m.activity().mxm.acc_lanes), (1, 320));
    // LW and IWS move 16 rows each; IW installs the buffered rows
    let mut m = machine("ref");
    let g = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
    for id in 0..16u8 {
        m.edit_state(|state| state.srf.write(RowId::W, id, g - 1, 49, &vec_of(|_| 1))).unwrap();
    }
    assert_eq!(exec(&mut m, 104, 50, &Insn::LW { r#type: 0, src: 2 }), Exec::Issued);
    assert_eq!((m.activity().mxm.lw, m.activity().mxm.lw_rows), (1, 16));
}

#[test]
fn the_totals_of_two_chips_add() {
    let mut a = machine("ref");
    let mut b = machine("ref");
    exec(&mut a, 4, 10, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    exec(&mut b, 4, 10, &Insn::Read { addr: 3, dst: E0, thread: 0 });
    exec(&mut b, 4, 11, &Insn::Read { addr: 4, dst: E0, thread: 0 });
    let mut t = a.activity().clone();
    t.add(b.activity());
    assert_eq!((t.mem[4].reads, t.mem[4].read_bytes, t.srf_writes[0]), (3, 960, 3));
    assert_eq!(t.active_cycles[0], 3);
}

// ---- through the command line ------------------------------------------------------------------------------------

/// A read, a loop of four iterations, then another read that waits for the loop (queue 4 = MEM slice 4).
const LOOP_THEN_READ: &str = "Q 4 10000010 00000000 11c00000 00040000 10000011 04000000\n";

fn run(prog: &std::path::Path, extra: &[&str]) -> (std::process::Output, Option<Value>) {
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("stats.json");
    let st = format!("--stats={}", file.display());
    let mut args = vec!["run", "--profile", "ref"];
    args.extend_from_slice(extra);
    args.push(&st);
    args.push(prog.to_str().unwrap());
    let out = run_iss(&args, "");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let doc = std::fs::read(&file).ok().map(|b| serde_json::from_slice(&b).unwrap());
    (out, doc)
}

fn programs_dir() -> std::path::PathBuf {
    root().join("opentsp-iss/crates/tsp-iss/benches/programs")
}

#[test]
fn run_reads_the_s_lines_of_the_dense_programs_so_that_stats_counts_them() {
    for (name, abc_key) in [("dense_f16", "f16"), ("dense_i8", "i8")] {
        let (out, doc) = run(&programs_dir().join(format!("{name}.txt")), &[]);
        assert_eq!(out.status.code(), Some(0), "{name}");
        let s = doc.unwrap();
        assert_eq!(s["diagnostics"], 0, "{name}: the program raises no diagnostic once its matrices are loaded");
        let abc = &s["activity"]["mxm"]["abc_by_type"][abc_key];
        assert!(abc["abc"].as_u64().unwrap() > 0 && abc["macs"].as_u64().unwrap() > 0, "{name}: {abc}");
    }
    // an S line is checked like the S line of the harness, and the message names the line of the file
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("bad.txt");
    std::fs::write(&p, "Q 4 10000010 00000000 11c00000\nS 99 0 1 2\n").unwrap();
    let o = run_iss(&["run", "--profile", "ref", p.to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("line 2: slice out of range"), "{}", text(&o.stderr));
    // a word the S line preloads is in the SRAM at the end
    std::fs::write(&p, "Q 4 10000010 00000000 11c00000\nS 0 0 1 2\n").unwrap();
    let o = run_iss(&["run", "--profile", "ref", "--dump", "0:0", p.to_str().unwrap()], "");
    assert!(text(&o.stdout).contains("W 0 0 01 02 00"), "{}", text(&o.stdout));
}

#[test]
fn bench_reports_the_elements_executed_beside_the_dispatches() {
    let o = run_iss(&["bench", "--elements", "--program", programs_dir().join("vxm_valid.txt").to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(0), "{}", text(&o.stderr));
    let line = text(&o.stdout);
    let field = |k: &str| -> f64 {
        line.split_whitespace().find_map(|f| f.strip_prefix(&format!("{k}="))).unwrap().parse().unwrap()
    };
    // 16 queues repeat their instruction 20000 times each, with 48 dispatches in all
    assert_eq!(field("dispatches"), 48.0);
    assert_eq!(field("elements"), 320_000.0, "{line}");
    let want = field("seconds") * 1e9 / 320_000.0;
    assert!((field("ns_per_element") - want).abs() < 0.01 * want + 0.01, "{line}");
    // without the flag the line is as it was
    let o = run_iss(&["bench", "--program", programs_dir().join("vxm_valid.txt").to_str().unwrap()], "");
    assert!(!text(&o.stdout).contains("elements="), "{}", text(&o.stdout));
}

#[test]
fn the_stats_document_has_an_activity_object_per_chip_and_in_total() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("loop.txt");
    std::fs::write(&p, LOOP_THEN_READ).unwrap();
    let (_, doc) = run(&p, &[]);
    let s = doc.unwrap();
    let a = &s["activity"];
    let reads = a["mem"]["reads"].as_u64().unwrap();
    assert!(reads >= 2, "the program reads at least twice: {a}");
    assert_eq!(a["srf"]["writes_east"].as_u64().unwrap() + a["srf"]["writes_west"].as_u64().unwrap(), reads);
    assert_eq!(a["mem"]["read_bytes"].as_u64().unwrap(), 320 * reads);
    assert_eq!(a["mem"]["per_slice"][0]["slice"], 4);
    assert_eq!(a["active_cycles"]["MEM"].as_u64().unwrap(), reads, "one reference per cycle");
    assert_eq!(a["issued"]["Read"].as_u64().unwrap(), reads);
    assert_eq!(s["per_chip"][0]["activity"], *a, "one chip: the total is the chip's");
    for k in ["vxm", "mxm", "sxm", "c2c", "host"] {
        assert!(a[k].is_object(), "{k}: {a}");
    }
}

#[test]
fn counting_changes_neither_the_trace_nor_the_results() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("loop.txt");
    std::fs::write(&p, LOOP_THEN_READ).unwrap();
    let plain = run_iss(&["run", "--profile", "ref", "--trace", p.to_str().unwrap()], "");
    let (with, _) = run(&p, &["--trace"]);
    assert_eq!(plain.stdout, with.stdout, "the trace is the same with --stats");
    assert_eq!(plain.status.code(), with.status.code());
}

// ---- counting is opt-in --------------------------------------------------------------------------------------------

#[test]
fn a_chip_that_does_not_count_counts_nothing_and_counts_again_when_asked() {
    let mut m = machine("ref");
    assert!(m.activity().enabled(), "a machine built by the library counts unless the process default says otherwise");
    m.edit_state(|state| state.activity.set_enabled(false)).unwrap();
    let read = Insn::Read { addr: 1, dst: E0, thread: 0 };
    assert_eq!(exec(&mut m, 4, 10, &read), Exec::Issued);
    assert_eq!(exec(&mut m, 4, 11, &read), Exec::Issued);
    let a = m.activity();
    assert_eq!((a.mem_total().reads, a.srf_writes, a.active_cycles[0]), (0, [0, 0], 0), "nothing counted while off");
    assert_eq!(a.issued[EntryId::Read.index()], 0);
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap();
    exec(&mut m, 4, 12, &read);
    let a = m.activity();
    assert_eq!(
        (a.mem_total().reads, a.srf_writes, a.active_cycles[0], a.issued[EntryId::Read.index()]),
        (1, [1, 0], 1, 1)
    );
}

/// One instruction of each unit class, issued with counting off or on: the six arms of the dispatch all honour the flag.
fn one_of_each_unit(on: bool) -> (Vec<Exec>, Machine) {
    let mut m = machine("ref");
    m.edit_state(|state| state.activity.set_enabled(on)).unwrap();
    let g = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
    let a = vec_of(|l| l as u8);
    let add = Insn::Add { r#type: 0, dst: 8, lhs: 0, rhs: 1, sg4: 0, lhs_chain: 0, rhs_chain: 0 };
    let at = 40 + m.latency_of_instr(&add).d_skew as u64;
    for s in 0..2 {
        m.edit_state(|state| state.srf.write(RowId::E, s, g - 1, at - 1, &a)).unwrap();
    }
    let gw = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
    for id in 0..16u8 {
        m.edit_state(|state| state.srf.write(RowId::W, id, gw - 1, 49, &vec_of(|_| 1))).unwrap();
    }
    let gs = read_gap(m.geom(), issuer_of(120).unit, issuer_of(120).slice as usize, RowId::W).unwrap();
    for r in 0..16u8 {
        let skew = m.latency_of_instr(&Insn::Transpose { dst: 0, src: 0 }).d_skew as u64;
        m.edit_state(|state| state.srf.write(RowId::W, r, gs - 1, 30 + skew - 1, &a)).unwrap();
    }
    let r = vec![
        exec(&mut m, 4, 10, &Insn::Read { addr: 1, dst: E0, thread: 0 }), // MEM
        exec(&mut m, 4, 11, &Insn::SetVL { n: 3 }),                       // CU
        exec(&mut m, 88, 40, &add),                                       // VXM
        exec(&mut m, 104, 50, &Insn::LW { r#type: 0, src: 2 }),           // MXM
        exec(&mut m, 30, 5, &Insn::Deskew),                               // C2C
        exec(&mut m, 120, 30, &Insn::Transpose { dst: 1, src: 2 }),       // SXM
    ];
    (r, m)
}

#[test]
fn every_unit_class_honours_the_counting_flag() {
    let (off_r, off) = one_of_each_unit(false);
    let (on_r, on) = one_of_each_unit(true);
    assert_eq!(off_r, on_r, "counting does not change what the instructions do");
    assert!(on_r.iter().all(|x| *x == Exec::Issued), "{on_r:?}");
    let issued = |m: &Machine| m.activity().issued.iter().sum::<u64>();
    assert_eq!(issued(&on), 6, "one of each unit class counts when on");
    assert_eq!(
        (issued(&off), off.activity().active_cycles, off.activity().srf_writes),
        (0, [0; 6], [0, 0]),
        "none counts when off"
    );
}

#[test]
fn counting_off_gives_the_same_execution_as_counting_on() {
    let run_with = |on: bool| {
        let mut m = machine("ref");
        m.edit_state(|state| state.activity.set_enabled(on)).unwrap();
        let read = Insn::Read { addr: 1, dst: E0, thread: 0 };
        (0..4).map(|c| exec(&mut m, 4, 10 + c, &read)).collect::<Vec<_>>()
    };
    assert_eq!(run_with(false), run_with(true));
}

#[test]
fn a_serving_platform_counts_nothing_until_asked_and_keeps_the_setting_across_restarts() {
    use tsp_iss::platform::Platform;
    let profile = Profile::load("mini").unwrap();
    let mut p = Platform::new(&profile, 0).unwrap();
    assert!(!p.activity().enabled(), "the platform does not count by default");
    p.set_activity_counting(true);
    assert!(p.activity().enabled());
    p.restart().unwrap();
    assert!(p.activity().enabled(), "a restart keeps the setting");
    p.reset().unwrap();
    assert!(p.activity().enabled(), "a reset keeps the setting");
    p.set_activity_counting(false);
    p.reset().unwrap();
    assert!(!p.activity().enabled());
}

#[test]
fn run_counts_for_stats_and_for_energy() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("loop.txt");
    std::fs::write(&p, LOOP_THEN_READ).unwrap();
    // --stats counts
    let (_, doc) = run(&p, &[]);
    assert!(doc.unwrap()["activity"]["mem"]["reads"].as_u64().unwrap() >= 2);
    // --energy alone prices real counters (it needs them)
    let out = run_iss(&["run", "--profile", "ref", "--energy", &shipped_table_path(), p.to_str().unwrap()], "");
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(!text(&out.stderr).contains("0.0000e0 J"), "energy of counted activity: {}", text(&out.stderr));
}

fn shipped_table_path() -> String {
    root().join("opentsp-iss/perf/energy.yaml").to_str().unwrap().to_string()
}

// ---- the energy table and the priced run -------------------------------------------------------------------------

use tsp_iss::slices::activity::Activity;
use tsp_iss::slices::energy::{estimate, estimate_run, Table};

fn shipped_table() -> String {
    std::fs::read_to_string(root().join("opentsp-iss/perf/energy.yaml")).unwrap()
}

/// A table of round numbers that every test below can price by hand.
fn round_table() -> String {
    let t = shipped_table();
    // zero every event and cycle energy, then set the few the hand calculation uses
    let mut out = String::new();
    let mut section = String::new();
    let mut sub = String::new();
    for line in t.lines() {
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if trimmed.starts_with('#') || trimmed.is_empty() || !trimmed.contains(':') {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let (k, v) = trimmed.split_once(':').unwrap();
        if indent == 0 {
            section = k.to_string();
            sub.clear();
        } else if indent == 2 {
            sub = k.to_string();
        }
        let zero = v.trim().parse::<f64>().is_ok() && (section == "events" || section == "cycles" || k == "static_w");
        let line = if zero {
            let val = match (section.as_str(), sub.as_str(), k) {
                ("events", _, "mem_read_pj_per_byte") => "2.0",
                ("events", "mxm_mac_pj", "i8") => "0.5",
                ("events", "vxm_width_scale", _) | ("events", _, "vxm_float_scale") => "1.0",
                ("cycles", "MEM", "active_pj") => "10",
                ("cycles", "MEM", "idle_pj") => "1",
                ("static_w", _, _) | (_, _, "static_w") => "0.5",
                _ => "0",
            };
            format!("{}{k}: {val}", " ".repeat(indent))
        } else {
            line.to_string()
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

#[test]
fn the_shipped_table_parses_and_labels_every_value_an_estimate() {
    let text = shipped_table();
    let t = Table::parse(&text).expect("the shipped table parses and passes the checks");
    let _ = t;
    assert!(text.contains("status: estimate"));
    assert!(text.contains("EVERY VALUE BELOW IS AN ESTIMATE"));
    // both published ranges are stated, and neither is picked
    for needle in ["low: 35", "high: 350", "average: 185", "maximum: 300", "status: unresolved"] {
        assert!(text.contains(needle), "{needle}");
    }
    assert!(text.contains("PWR-001 semantics (per-bank power-down, wake cycles) are not implemented"));
}

#[test]
fn a_table_that_is_not_an_estimate_or_misses_a_value_is_refused() {
    let t = shipped_table();
    let e = Table::parse(&t.replace("status: estimate", "status: measured")).unwrap_err();
    assert!(e.contains("`status` must be `estimate`"), "{e}");
    let e = Table::parse(
        &t.replace("    mxm_mac_pj:\n", "    mxm_mac_pj_gone:\n").replace("  mxm_mac_pj:\n", "  mxm_mac_pj_gone:\n"),
    )
    .unwrap_err();
    assert!(e.contains("events.mxm_mac_pj.i8"), "{e}");
    let e = Table::parse(&t.replace("mem_read_pj_per_byte: 0.40", "mem_read_pj_per_byte: -1")).unwrap_err();
    assert!(e.contains("mem_read_pj_per_byte") && e.contains("0 or more"), "{e}");
    let e = Table::parse(&t.replace("static_w: 30", "static_w: lots")).unwrap_err();
    assert!(e.contains("static_w") && e.contains("not a number"), "{e}");
    let e = Table::parse("schema: 1\n  status: estimate\n").unwrap_err();
    assert!(e.contains("indentation"), "{e}");
    let e = Table::parse("schema: 1\nschema: 2\n").unwrap_err();
    assert!(e.contains("duplicate key"), "{e}");
    let e = Table::parse("schema: 1\nevents:\nstatus: estimate\n").unwrap_err();
    assert!(e.contains("no value and no keys"), "{e}");
    let e = Table::parse("- a\n").unwrap_err();
    assert!(e.contains("lists"), "{e}");
}

#[test]
fn counters_become_joules_and_watts_by_the_table() {
    let t = Table::parse(&round_table()).unwrap();
    // The priced chip has the structure the table is calibrated for (its `reference` block), so every class scale is 1 and the
    // static power is `static_w` as it stands (`Activity::new` has the same structure: 88 MEM slices, 4 planes, 16 links).
    let reference = tsp_iss::slices::activity::Structure {
        vlanes: 320,
        mem_slices: 88,
        mxm_planes: 4,
        mxm_rows: 320,
        c2c_links: 16,
    };
    let mut a = Activity::with_structure(320, 20, reference);
    // 1000 bytes read, 2000 i8 MACs, MEM active in 5 of 20 cycles
    a.mem[3].reads = 4;
    a.mem[3].read_bytes = 1000;
    a.mxm.macs[0] = 2000;
    a.active_cycles[0] = 5;
    let e = estimate(&t, &a, 0, 20, 900.0);
    // pJ: 1000 * 2 + 2000 * 0.5 + 5 * 10 + 15 * 1 = 3065; the other classes charge 0 here
    let dynamic = 3065e-12;
    let seconds = 20.0 / 900e6;
    let fixed = 0.5 * seconds;
    let j = &e["joules"];
    assert!((j["mem"].as_f64().unwrap() - 2000e-12).abs() < 1e-18, "{j}");
    assert!((j["mxm"].as_f64().unwrap() - 1000e-12).abs() < 1e-18, "{j}");
    assert!((j["active_cycles"].as_f64().unwrap() - 50e-12).abs() < 1e-18);
    assert!((j["idle_cycles"].as_f64().unwrap() - 15e-12).abs() < 1e-18);
    assert!((j["total"].as_f64().unwrap() - (dynamic + fixed)).abs() < 1e-18, "{j}");
    let w = e["watts"]["average"].as_f64().unwrap();
    assert!((w - (dynamic + fixed) / seconds).abs() < 1e-6, "{w}");
    assert_eq!(e["status"], "estimate");
    // the clock is an argument: twice the clock, half the time, the same joules, twice the dynamic watts
    let e2 = estimate(&t, &a, 0, 20, 1800.0);
    assert!((e2["joules"]["mem"].as_f64().unwrap() - 2000e-12).abs() < 1e-18);
    assert!((e2["watts"]["dynamic"].as_f64().unwrap() / e["watts"]["dynamic"].as_f64().unwrap() - 2.0).abs() < 1e-9);
    // an empty run has no watts
    assert!(estimate(&t, &Activity::new(320, 20, 320), 0, 0, 900.0)["watts"]["average"].is_null());
}

#[test]
fn vxm_energy_scales_with_the_width_and_the_float_type() {
    let shipped = Table::parse(&shipped_table()).unwrap();
    let mut a = Activity::new(320, 20, 320);
    let add = EntryId::Add.index();
    a.vxm[add][0] = 1000; // i8
    let one = estimate(&shipped, &a, 0, 1, 900.0)["joules"]["vxm"].as_f64().unwrap();
    assert!((one - 1000.0 * 0.04e-12).abs() < 1e-18, "i8 add_sub is 0.04 pJ per element op: {one}");
    a.vxm[add][0] = 0;
    a.vxm[add][3] = 1000; // f32: width scale 4.8, float scale 3.0
    let f32e = estimate(&shipped, &a, 0, 1, 900.0)["joules"]["vxm"].as_f64().unwrap();
    assert!((f32e - 1000.0 * 0.04e-12 * 4.8 * 3.0).abs() < 1e-18, "{f32e}");
}

#[test]
fn the_run_estimate_adds_the_chips_and_states_the_unresolved_calibration() {
    let t = Table::parse(&shipped_table()).unwrap();
    let mut a = Activity::new(320, 20, 320);
    a.mem[0].read_bytes = 3200;
    let r = estimate_run(&t, &[(a.clone(), 0, 100), (a, 0, 50)], 900.0);
    let (c0, c1) = (&r["per_chip"][0], &r["per_chip"][1]);
    let sum = c0["joules"]["total"].as_f64().unwrap() + c1["joules"]["total"].as_f64().unwrap();
    assert!((r["joules"].as_f64().unwrap() - sum).abs() < 1e-18);
    assert_eq!(r["cycles"], 100, "the longest chip");
    assert_eq!(r["calibration"]["published_range_w"]["low"], 35);
    assert_eq!(r["calibration"]["product_brief_w"]["average"], 185);
    assert!(r["calibration"]["status"].as_str().unwrap().starts_with("unresolved"));
    assert!(r["pwr_001"].as_str().unwrap().starts_with("not implemented"));
    // the full-activity bound of the shipped estimate lies inside both published ranges (that is how it was chosen)
    let w = c0["full_activity_watts"].as_f64().unwrap();
    assert!((35.0..=300.0).contains(&w), "{w}");
}

/// The `ref` profile's structure and `mini`'s (16 lanes, one superlane, 16 matrix rows; the same MEM slices, planes and links).
fn structure_of(profile: &str) -> (u64, u64, tsp_iss::slices::activity::Structure) {
    let p = Profile::load(profile).unwrap();
    let g = tsp_iss::Geom::new(&p);
    let st = tsp_iss::slices::activity::Structure {
        vlanes: g.vlanes,
        mem_slices: 2 * g.mem_slices,
        mxm_planes: 2 * g.mxm_planes,
        mxm_rows: g.mxm_rows.unwrap(),
        c2c_links: g.links,
    };
    (g.vlanes, g.superlanes, st)
}

fn chip(profile: &str) -> Activity {
    let (v, sl, st) = structure_of(profile);
    let mut a = Activity::with_structure(v, sl, st);
    // the same work on either chip: 100 cycles, MEM and VXM active in 40 of them, 1000 bytes read
    a.active_cycles[0] = 40;
    a.active_cycles[1] = 40;
    a.mem[3].reads = 4;
    a.mem[3].read_bytes = 1000;
    a
}

#[test]
fn a_smaller_chip_is_priced_smaller_than_the_reference_chip() {
    let t = Table::parse(&shipped_table()).unwrap();
    let (r, m) = (estimate(&t, &chip("ref"), 0, 100, 900.0), estimate(&t, &chip("mini"), 0, 100, 900.0));
    let w = |e: &Value, k: &str| e["watts"][k].as_f64().unwrap();
    // the reference chip pays the table's static power as it stands, every class at scale 1
    assert_eq!(w(&r, "static"), 30.0);
    assert!(r["class_scale"].as_object().unwrap().values().all(|v| v.as_f64() == Some(1.0)), "{}", r["class_scale"]);
    // mini: MEM 16/320 of the lanes, VXM and SXM 16/320, MXM (16 rows of 16 lanes) 1/400 of the planes' size, C2C and CU the same
    let sc = |c: &str| m["class_scale"][c].as_f64().unwrap();
    assert!((sc("MEM") - 0.05).abs() < 1e-12 && (sc("VXM") - 0.05).abs() < 1e-12 && (sc("SXM") - 0.05).abs() < 1e-12);
    assert!((sc("MXM") - (16.0 * 16.0) / (320.0 * 320.0)).abs() < 1e-12);
    assert_eq!((sc("C2C"), sc("CU")), (1.0, 1.0));
    let want_static = 30.0 * (0.35 * 0.05 + 0.10 * 0.05 + 0.30 * (256.0 / 102400.0) + 0.05 * 0.05 + 0.15 + 0.05);
    assert!((w(&m, "static") - want_static).abs() < 1e-9, "{} against {want_static}", w(&m, "static"));
    // the same program no longer costs the same on a 16-lane chip and a 320-lane chip (it was 36.0 and 36.1 W)
    assert!(w(&m, "dynamic") < 0.5 * w(&r, "dynamic"), "{} against {}", w(&m, "dynamic"), w(&r, "dynamic"));
    assert!(w(&m, "average") < 0.6 * w(&r, "average"), "{} against {}", w(&m, "average"), w(&r, "average"));
    // the full-activity bound reads the structure of the chip: the reference one stays in both published ranges
    let (fr, fm) = (r["full_activity_watts"].as_f64().unwrap(), m["full_activity_watts"].as_f64().unwrap());
    assert!((35.0..=300.0).contains(&fr), "{fr}");
    assert!(fm < 0.5 * fr, "mini's bound {fm} against ref's {fr}");
}

/// : the energy structure of a machine reads the planes and links of its profile, as it reads the lanes and MEM slices.
#[test]
fn the_energy_structure_of_a_machine_follows_the_planes_and_links_of_its_profile() {
    let of = |name: &str| {
        let path = format!("{}/../../../opentsp-arch-spec/config/{name}.json", env!("CARGO_MANIFEST_DIR"));
        let p = Profile::load_file(name.rsplit('/').next().unwrap(), path).unwrap();
        let mut m = Machine::new(&p, 0, Box::new(NoLinks)).unwrap();
        m.edit_state(|state| state.activity.set_enabled(true)).unwrap();
        m
    };
    let structure = |name: &str| *of(name).activity().structure();
    let (r, one) = (structure("ref"), structure("test/one_plane"));
    assert_eq!((r.mxm_planes, one.mxm_planes), (4, 2), "two hemispheres, `mxm.planes_per_hemisphere` planes each");
    assert_eq!(r.c2c_links, 16);
    // the widest-chip constructor says what its own comment says: sixteen links, the reference profile's
    assert_eq!(Activity::new(320, 20, 320).structure().c2c_links, 16);
    // a one-plane chip has half the MXM of `ref`: its full-activity bound and its static power come out smaller
    let t = Table::parse(&shipped_table()).unwrap();
    let est = |name: &str| estimate(&t, of(name).activity(), 0, 100, 900.0);
    let (er, e1) = (est("ref"), est("test/one_plane"));
    assert!((er["class_scale"]["MXM"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    assert!((e1["class_scale"]["MXM"].as_f64().unwrap() - 0.5).abs() < 1e-12, "{}", e1["class_scale"]);
    assert!(e1["full_activity_watts"].as_f64().unwrap() < er["full_activity_watts"].as_f64().unwrap());
}

#[test]
fn the_energy_command_prices_mini_and_ref_differently() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("loop.txt");
    std::fs::write(&p, LOOP_THEN_READ).unwrap();
    let avg = |profile: &str| {
        let f = d.path().join(format!("{profile}.json"));
        let st = format!("--stats={}", f.display());
        let out =
            run_iss(&["run", "--profile", profile, "--energy", &shipped_table_path(), &st, p.to_str().unwrap()], "");
        assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
        let doc: Value = serde_json::from_slice(&std::fs::read(&f).unwrap()).unwrap();
        doc["energy"]["watts"].as_f64().unwrap()
    };
    let (r, m) = (avg("ref"), avg("mini"));
    assert!(m < 0.6 * r, "mini {m} W against ref {r} W for the same program");
}

#[test]
fn a_table_whose_structure_or_static_shares_are_wrong_is_refused() {
    let t = shipped_table();
    let e = Table::parse(&t.replace(
        "  MEM: 0.35
",
        "  MEM: 0.40
",
    ))
    .unwrap_err();
    assert!(e.contains("static_share") && e.contains("add up to 1"), "{e}");
    let e = Table::parse(&t.replace(
        "  mxm_planes: 4
  mxm_rows",
        "  mxm_planes_gone: 4
  mxm_rows",
    ))
    .unwrap_err();
    assert!(e.contains("reference.mxm_planes"), "{e}");
    let e = Table::parse(&t.replace(
        "  vlanes: 320
",
        "  vlanes: 0
",
    ))
    .unwrap_err();
    assert!(e.contains("reference.vlanes") && e.contains("at least 1"), "{e}");
}

#[test]
fn the_command_line_prices_a_run_and_refuses_bad_arguments() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("loop.txt");
    std::fs::write(&p, LOOP_THEN_READ).unwrap();
    let table = root().join("opentsp-iss/perf/energy.yaml");
    let stats = d.path().join("s.json");
    let out = run_iss(
        &[
            "run",
            "--profile",
            "ref",
            "--energy",
            table.to_str().unwrap(),
            "--clock-mhz",
            "900",
            &format!("--stats={}", stats.display()),
            p.to_str().unwrap(),
        ],
        "",
    );
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    let err = text(&out.stderr);
    assert!(err.contains("energy (estimate, 900.0 MHz"), "{err}");
    let doc: Value = serde_json::from_slice(&std::fs::read(&stats).unwrap()).unwrap();
    let e = &doc["energy"];
    assert_eq!(e["status"], "estimate");
    assert!(e["joules"].as_f64().unwrap() > 0.0 && e["watts"].as_f64().unwrap() > 0.0);
    assert_eq!(e["per_chip"][0]["cycles"], doc["last_cycle"].as_u64().unwrap() + 1);
    // the same run without the table has no energy and prints the same results
    let plain = run_iss(&["run", "--profile", "ref", p.to_str().unwrap()], "");
    assert_eq!(plain.stdout, out.stdout, "pricing the run changes nothing it prints");
    // a bad table, a missing table, a bad clock and a clock without a table
    let bad = d.path().join("bad.yaml");
    std::fs::write(&bad, "schema: 1\nstatus: measured\n").unwrap();
    let o = run_iss(&["run", "--profile", "ref", "--energy", bad.to_str().unwrap(), p.to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("`status` must be `estimate`"), "{}", text(&o.stderr));
    let o = run_iss(&["run", "--profile", "ref", "--energy", "/nonexistent/e.yaml", p.to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("cannot read"), "{}", text(&o.stderr));
    for clock in ["0", "-3", "fast", "inf"] {
        let o = run_iss(
            &[
                "run",
                "--profile",
                "ref",
                "--energy",
                table.to_str().unwrap(),
                "--clock-mhz",
                clock,
                p.to_str().unwrap(),
            ],
            "",
        );
        assert_eq!(o.status.code(), Some(2), "{clock}");
        assert!(text(&o.stderr).contains("expected a clock in MHz"), "{clock}: {}", text(&o.stderr));
    }
    let o = run_iss(&["run", "--profile", "ref", "--clock-mhz", "900", p.to_str().unwrap()], "");
    assert_eq!(o.status.code(), Some(2));
    assert!(text(&o.stderr).contains("--clock-mhz needs --energy"), "{}", text(&o.stderr));
}

/// A replayed run executes the same instructions, so it counts the same (needs `--features replay`:
/// `just iss-replay-test`-style build; the test compiles to nothing without it).
#[cfg(feature = "replay")]
#[test]
fn a_replayed_run_has_the_counters_of_the_recorded_one() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("loop.txt");
    std::fs::write(&p, LOOP_THEN_READ).unwrap();
    let (_, plain) = run(&p, &[]);
    let (_, replayed) = run(&p, &["--replay-runs", "2"]);
    let (plain, replayed) = (plain.unwrap(), replayed.unwrap());
    assert!(
        replayed["replay"]["hits"].as_u64().unwrap() >= 1,
        "the second run was served by the schedule: {}",
        replayed["replay"]
    );
    assert_eq!(plain["activity"], replayed["activity"]);
}

#[test]
fn energy_retains_a_complete_unsigned_run_length_and_finite_estimates() {
    let table = Table::parse(&shipped_table()).unwrap();
    let a = machine("mini").activity().clone();
    let cycles = 1u128 << 64;
    let one = estimate(&table, &a, 0, cycles, 1000.0);
    assert_eq!(one["cycles"].to_string(), "18446744073709551616");
    assert!(one["watts"]["average"].as_f64().unwrap().is_finite());
    let run = estimate_run(&table, &[(a.clone(), 0, cycles), (a, 0, cycles - 1)], 1000.0);
    assert_eq!(run["cycles"].to_string(), "18446744073709551616");
    assert!(run["watts"].as_f64().unwrap().is_finite());
    assert!(run["joules"].as_f64().unwrap().is_finite());
}

#[test]
fn each_active_class_counts_a_maximum_cycle_without_a_marker_collision() {
    let add = Insn::Add { r#type: 0, dst: 8, lhs: 0, rhs: 1, sg4: 0, lhs_chain: 0, rhs_chain: 0 };
    let cases = [
        (4, Insn::Read { addr: 1, dst: 0, thread: 0 }, "MEM"),
        (4, Insn::SetVL { n: 1 }, "CU"),
        (88, add, "VXM"),
        (104, Insn::LW { r#type: 0, src: 2 }, "MXM"),
        (120, Insn::Transpose { dst: 1, src: 2 }, "SXM"),
        (126, Insn::HostNotify { n: 0 }, "C2C"),
    ];
    for cycle in [100u64, u64::MAX - 1, u64::MAX] {
        for (q, instruction, class) in cases {
            let mut m = machine("ref");
            let at = cycle - u64::from(m.latency_of_instr(&instruction).d_skew);
            match class {
                "VXM" => {
                    let gap = read_gap(m.geom(), UnitKind::Vxm, 0, RowId::E).unwrap();
                    for stream in 0..2 {
                        m.edit_state(|state| state.srf.write(RowId::E, stream, gap - 1, cycle - 1, &vec_of(|_| 1)))
                            .unwrap();
                    }
                }
                "MXM" | "SXM" => {
                    let unit = issuer_of(q);
                    let gap = read_gap(m.geom(), unit.unit, unit.slice as usize, RowId::W).unwrap();
                    for stream in 0..16 {
                        m.edit_state(|state| state.srf.write(RowId::W, stream, gap - 1, cycle - 1, &vec_of(|_| 1)))
                            .unwrap();
                    }
                }
                _ => {}
            }
            assert_eq!(exec(&mut m, q, at, &instruction), Exec::Issued, "{cycle} {instruction:?}");
            assert_eq!(
                m.activity().active_cycles
                    [tsp_iss::slices::activity::CLASSES.iter().position(|x| *x == class).unwrap()],
                1,
                "{cycle} {instruction:?}"
            );
        }
    }
}

#[test]
fn activity_markers_distinguish_zero_maximum_duplicates_restart_and_disabled_counting() {
    let mut m = machine("mini");
    let instruction = Insn::SetVL { n: 1 };
    let cu = tsp_iss::slices::activity::CLASSES.iter().position(|x| *x == "CU").unwrap();
    for cycle in [0u64, 0, u64::MAX - 1, u64::MAX - 1, u64::MAX, u64::MAX] {
        assert_eq!(exec(&mut m, 4, cycle, &instruction), Exec::Issued);
    }
    assert_eq!(m.activity().active_cycles[cu], 3);
    m.edit_state(|state| state.activity.set_enabled(false)).unwrap();
    exec(&mut m, 4, u64::MAX, &instruction);
    m.edit_state(|state| state.activity.set_enabled(true)).unwrap();
    exec(&mut m, 4, u64::MAX, &instruction);
    assert_eq!(m.activity().active_cycles[cu], 3, "enable toggles preserve the last counted stamp");
    m.edit_state(|state| state.activity.restart()).unwrap();
    exec(&mut m, 4, u64::MAX, &instruction);
    assert_eq!(m.activity().active_cycles[cu], 4, "restart clears stamps, retaining counts");
    m.edit_state(|state| state.activity.restart()).unwrap();
    exec(&mut m, 4, 0, &instruction);
    assert_eq!(m.activity().active_cycles[cu], 5, "zero remains a distinct admitted first cycle");
}
