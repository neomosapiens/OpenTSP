//! Cost of the units' instructions per catalogue entry and slice (cargo feature `opprof`, off by default: the module
//! is not compiled without it). `Machine::execute` times each call; `dump` prints the table to standard error when a
//! run ends, with the clock's own cost (measured here, back to back) taken off every sample.
use ots_isa::ENTRIES;
use std::cell::RefCell;
use std::time::Instant;

struct Table {
    calls: Vec<u64>,
    ns: Vec<u128>,
}

thread_local! {
    static TABLE: RefCell<Table> = RefCell::new(Table { calls: vec![0; ENTRIES.len()], ns: vec![0; ENTRIES.len()] });
}

/// Time `f` and charge it to entry `idx`.
pub fn timed<R>(idx: usize, f: impl FnOnce() -> R) -> R {
    let t0 = Instant::now();
    let r = f();
    let d = t0.elapsed().as_nanos();
    TABLE.with(|t| {
        let mut t = t.borrow_mut();
        t.calls[idx] += 1;
        t.ns[idx] += d;
    });
    r
}

fn clock_cost_ns() -> f64 {
    const N: u32 = 2_000_000;
    let mut sum: u128 = 0;
    for _ in 0..N {
        let t0 = Instant::now();
        sum += t0.elapsed().as_nanos();
    }
    sum as f64 / N as f64
}

/// Print the table (by slice, then by total time) and reset it.
pub fn dump() {
    let over = clock_cost_ns();
    let t = TABLE.with(|t| {
        let mut t = t.borrow_mut();
        let out = Table { calls: t.calls.clone(), ns: t.ns.clone() };
        t.calls.iter_mut().for_each(|x| *x = 0);
        t.ns.iter_mut().for_each(|x| *x = 0);
        out
    });
    let adj = |i: usize| (t.ns[i] as f64 - over * t.calls[i] as f64).max(0.0);
    let mut rows: Vec<usize> = (0..ENTRIES.len()).filter(|&i| t.calls[i] > 0).collect();
    rows.sort_by(|&a, &b| adj(b).total_cmp(&adj(a)));
    let total: f64 = rows.iter().map(|&i| adj(i)).sum();
    let calls: u64 = rows.iter().map(|&i| t.calls[i]).sum();
    eprintln!(
        "opprof: clock cost {over:.1} ns per sample (subtracted); {calls} executions, {:.3} ms in unit semantics",
        total / 1e6
    );
    eprintln!("opprof slice entry calls ns_per_call total_ms share_pct");
    for slice in ["MEM", "MXM", "VXM", "SXM", "ICU", "CU", "C2C"] {
        let (mut c, mut n) = (0u64, 0.0f64);
        for &i in &rows {
            if ENTRIES[i].slice == slice {
                c += t.calls[i];
                n += adj(i);
            }
        }
        if c > 0 {
            eprintln!("opprof {slice} * {c} {:.1} {:.3} {:.1}", n / c as f64, n / 1e6, 100.0 * n / total);
        }
    }
    for &i in &rows {
        let e = &ENTRIES[i];
        eprintln!(
            "opprof {} {} {} {:.1} {:.3} {:.1}",
            e.slice,
            e.name,
            t.calls[i],
            adj(i) / t.calls[i] as f64,
            adj(i) / 1e6,
            100.0 * adj(i) / total
        );
    }
}
