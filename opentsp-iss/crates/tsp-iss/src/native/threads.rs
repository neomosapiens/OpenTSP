//! `tsp-iss run --matrix-threads N|auto`: the width of the pool that splits a large matrix product (`slices/pool.rs`).
//!
//! `N` is the width (1 to 16, also clamped to the cores). `auto` is for a deployment that runs few programs at a time and
//! gives each the cores it can use: **min(cores / jobs, 6)** threads when the program holds an **f16** MXM ABC whose estimated
//! serial time is at least [`SPLIT_MIN_NS`], else 1. The f16 ABC of a dense matrix program scales (about 3x at six threads,
//! measured on `benches/programs/dense_f16.txt`, see `opentsp-iss/README.md`); a program of a few sparse ABCs, as a decode
//! step of a language model is, gains nothing and would take the cores of the other workers. An **i8** ABC is not split by
//! `auto`: the kernel costs about 1.4 us, less than the fork and join of the pool, and the measured width-4 run of
//! `benches/programs/dense_i8.txt` was slower than width 1 under a fixed pool while `slices/mxm.rs` estimated an i8 row at one
//! nanosecond per lane (about 70 times too much; it is now one per 64 lanes, so an i8 product never forks). `jobs` is the
//! threads the run's chips already use (`--jobs`, else the cores, one per chip at most), so a process per core, each with `jobs = 1`, gets one thread each. The default
//! stays 1 (also without the option, whatever `TSP_ISS_THREADS` says for the library).
//!
//! The queues that hold ABCs are the MXM queues of the layout (`layout::issuer_of`); only the instruction text the program
//! loads into them at the start is counted, so a program that fetches its matrix text later (Ifetch of MXM text) is not seen and
//! gets width 1.
//!
//! `auto` also makes the pool **adaptive** ([`crate::slices::pool::Pool::new_adaptive`]): the static test only says that a
//! program has f16 ABCs; whether splitting them pays depends on the activations (a sparse f16 product costs about 5 us, the fork
//! and join 3 us: P8 gets no speedup from threads and burns 2.7 times the CPU time). The pool measures its first forks and
//! answers `width() == 1`, waking no worker, while a split saves less than `SAVE_MIN_NS`, and probes again after a while.
use crate::geom::{Geom, DATA_QUEUES};
use crate::layout::{issuer_of, UnitKind};
use crate::slices::pool::MAX_WIDTH;
use crate::Machine;
use ots_format::text::Images;
use ots_isa::Insn;
use ots_profile::Profile;

/// The estimated serial time, in nanoseconds, below which `slices/mxm.rs` does not split an ABC over the pool (its
/// `SPLIT_MIN_NS`: 3000, a join of four threads costs about half of that). A mirror: `tests/matrix_threads.rs` checks the
/// decision against a real split.
pub const SPLIT_MIN_NS: usize = 3_000;
/// The width `auto` stops at: the f16 scaling measured at 6 threads is about 3x, and 12 threads give little more.
pub const AUTO_MAX: usize = 6;

/// `--matrix-threads`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixThreads {
    Width(usize),
    Auto,
}

impl MatrixThreads {
    /// `4`, `auto`.
    pub fn parse(s: &str) -> Option<MatrixThreads> {
        if s == "auto" {
            return Some(MatrixThreads::Auto);
        }
        s.parse::<usize>().ok().filter(|n| (1..=MAX_WIDTH).contains(n)).map(MatrixThreads::Width)
    }
}

/// Is data queue `q` an MXM queue (the layout of the queues, `layout::issuer_of`, not a list of numbers kept here)?
fn is_mxm_queue(q: usize) -> bool {
    matches!(issuer_of(q).unit, UnitKind::MxmW | UnitKind::MxmE)
}

/// The MXM ABCs of the images of one chip: (f16, the others: i8, u8 and fp8), counted from the parcels the instruction queues hold (the IQs'
/// own parse, so the alignment of two-unit instructions is the machine's).
pub fn abc_counts(profile: &Profile, images: &Images) -> (usize, usize) {
    let Ok(mut m) = Machine::new(profile, 0, Box::new(crate::links::NoLinks)) else { return (0, 0) };
    let mut counts = (0, 0);
    for (q, units) in images.iter() {
        if q == 0xffff || q as usize >= DATA_QUEUES || !is_mxm_queue(q as usize) {
            continue; // only the MXM queues hold ABCs
        }
        if m.load_image(q, units).is_err() {
            continue;
        }
        let qi = q as usize;
        loop {
            let (units, f16, other) = {
                let h = m.iq.head(qi);
                if !h.complete {
                    break;
                }
                let (mut f16, mut other) = (0, 0);
                for i in h.items.iter().filter_map(|i| i.as_ref().ok()) {
                    match i {
                        Insn::ABC { r#type: 1, .. } => f16 += 1, // the weight type code of f16 (`slices/mxm.rs wtype`)
                        Insn::ABC { .. } | Insn::ABC8 { .. } => other += 1,
                        _ => {}
                    }
                }
                (h.units, f16, other)
            };
            counts.0 += f16;
            counts.1 += other;
            m.iq.pop(qi, units);
        }
    }
    counts
}

/// Does an ABC of this profile pay for a split? An f16 row costs at least 30 ns (`slices/mxm.rs`), so the lower bound of its
/// estimate is `rows * 30`, to be at least [`SPLIT_MIN_NS`]. An i8 ABC does not (the kernel is faster than the join).
pub fn splits(geom: &Geom, f16: bool) -> bool {
    let rows = geom.mxm_rows.unwrap_or(0) as usize;
    f16 && rows * 30 >= SPLIT_MIN_NS
}

/// The width `auto` gives a run on `cores` cores with `jobs` chip threads: [`AUTO_MAX`] or `cores / jobs`, whichever is
/// less (at least 1), when some chip's program holds an ABC that the pool would split, else 1.
pub fn auto_width(profile: &Profile, geom: &Geom, chips: &[&Images], cores: usize, jobs: usize) -> usize {
    let dense = chips.iter().any(|im| {
        let (f16, other) = abc_counts(profile, im);
        (f16 > 0 && splits(geom, true)) || (other > 0 && splits(geom, false))
    });
    if dense {
        (cores / jobs.max(1)).clamp(1, AUTO_MAX)
    } else {
        1
    }
}
