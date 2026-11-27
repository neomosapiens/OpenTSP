//! LW appends its 16 rows to the plane's weight buffer when the cycle ends instead of staging a copy of the whole
//! buffer (a copy of up to 320 rows per LW). What the planes hold afterwards must be what the copying version held:
//! the same rows, the installed weights untouched by a later LW, two queues that both stage the buffer in one cycle
//! dropping both changes, and the oldest of two stagings by one queue standing.
use ots_isa::Insn;
use ots_profile::Profile;
use std::sync::Arc;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::layout::{read_gap, RowId, UnitKind};
use tsp_iss::links::NoLinks;
use tsp_iss::observe::Observer;
use tsp_iss::slices::{Exec, Issue};
use tsp_iss::srf::Vec320;
use tsp_iss::{Diag, Machine};

fn machine() -> Machine {
    Machine::new(&Profile::load("ref").unwrap(), 0, Box::new(NoLinks)).unwrap()
}

#[derive(Default)]
struct Rec;
impl Observer for Rec {
    fn diag(&mut self, _c: u64, _queue: u16, _code: Diag) {}
}

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

const I8: u32 = 0;
const LWQ: usize = 104; // plane 0's LW queue; plane 0's other queues are 105 and 106
const GROUP: u32 = 2; // W row, streams 0..15: field (32 + 0) / 16

fn lw() -> Insn {
    Insn::LW { r#type: I8, src: GROUP }
}

fn group(base: u8) -> [Vec320; 16] {
    std::array::from_fn(|k| [base.wrapping_add(k as u8); MAX_LANES])
}

/// Feed streams 0..15 of plane 0's input row for a read at cycle `t`.
fn feed(m: &mut Machine, vals: &[Vec320; 16], t: u64) {
    let g = read_gap(m.geom(), UnitKind::MxmW, 0, RowId::W).unwrap();
    for (k, v) in vals.iter().enumerate() {
        m.edit_state(|state| state.srf.write(RowId::W, k as u8, g - 1, t - 1, v)).unwrap();
    }
}

/// Issue an LW at dispatch cycle `c` with `vals` on the streams, and end the cycle.
fn load(m: &mut Machine, c: u64, vals: &[Vec320; 16]) {
    let d_skew = m.latency_of_instr(&lw()).d_skew as u64;
    feed(m, vals, c + d_skew);
    assert_eq!(exec(m, LWQ, c, &lw()), Exec::Issued);
    m.unit_state_flush(c + d_skew, &mut Rec);
}

fn lane0(rows: &[Vec320]) -> Vec<u8> {
    rows.iter().map(|r| r[0]).collect()
}

#[test]
fn successive_loads_extend_one_buffer_in_place() {
    let mut m = machine();
    load(&mut m, 10, &group(1));
    let first = Arc::as_ptr(&m.mxm().planes[0].buf);
    load(&mut m, 30, &group(100));
    load(&mut m, 50, &group(200));
    let p = &m.mxm().planes[0];
    assert_eq!(p.buf_rows, 48);
    let want: Vec<u8> =
        (0..16).map(|k| 1 + k).chain((0..16).map(|k| 100 + k)).chain((0..16).map(|k| 200 + k)).collect();
    assert_eq!(lane0(&p.buf), want);
    assert_eq!(Arc::as_ptr(&p.buf), first, "the buffer was copied instead of extended");
}

#[test]
fn a_load_after_an_install_leaves_the_installed_weights_alone() {
    let mut m = machine();
    load(&mut m, 10, &group(1));
    load(&mut m, 30, &group(50));
    let ready = m.mxm().planes[0].ready_lw;
    assert_eq!(exec(&mut m, LWQ, ready.try_into().unwrap(), &Insn::IW { r#type: I8 }), Exec::Issued);
    m.unit_state_flush(ready.try_into().unwrap(), &mut Rec);
    assert_eq!((m.mxm().planes[0].buf_rows, m.mxm().planes[0].rows), (0, 32));
    load(&mut m, (ready + 20).try_into().unwrap(), &group(9));
    let p = &m.mxm().planes[0];
    assert_eq!(p.buf_rows, 16);
    assert_eq!(lane0(&p.buf), (0..16).map(|k| 9 + k).collect::<Vec<u8>>());
    let w: Vec<u8> = (0..16).map(|k| 1 + k).chain((0..16).map(|k| 50 + k)).collect();
    assert_eq!(p.rows, 32);
    assert_eq!(lane0(&p.weights), w, "the installed weights changed with the buffer");
}

#[test]
fn two_queues_staging_the_buffer_in_one_cycle_drop_both() {
    let mut m = machine();
    load(&mut m, 10, &group(1));
    let d_skew = m.latency_of_instr(&lw()).d_skew as u64;
    let c = 40;
    feed(&mut m, &group(70), c + d_skew);
    assert_eq!(exec(&mut m, LWQ, c, &lw()), Exec::Issued);
    assert_eq!(exec(&mut m, LWQ + 1, c, &lw()), Exec::Issued);
    m.unit_state_flush(c + d_skew, &mut Rec);
    let p = &m.mxm().planes[0];
    assert_eq!(p.buf_rows, 16);
    assert_eq!(lane0(&p.buf), (0..16).map(|k| 1 + k).collect::<Vec<u8>>());
}

#[test]
fn of_two_loads_staged_by_one_queue_in_one_cycle_the_oldest_stands() {
    let mut m = machine();
    load(&mut m, 10, &group(1));
    let d_skew = m.latency_of_instr(&lw()).d_skew as u64;
    let c = 40;
    feed(&mut m, &group(70), c + d_skew);
    assert_eq!(exec(&mut m, LWQ, c, &lw()), Exec::Issued);
    feed(&mut m, &group(120), c + d_skew);
    assert_eq!(exec(&mut m, LWQ, c, &lw()), Exec::Issued);
    m.unit_state_flush(c + d_skew, &mut Rec);
    let p = &m.mxm().planes[0];
    assert_eq!(p.buf_rows, 32);
    let want: Vec<u8> = (0..16).map(|k| 1 + k).chain((0..16).map(|k| 70 + k)).collect();
    assert_eq!(lane0(&p.buf), want);
}
