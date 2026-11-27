//! The parse writes each parcel's slot record in the place where the IQ keeps it (`classify_into`), instead of
//! returning a 300-byte value that is copied twice. The record must be what the by-value reading of the model's
//! `classify_items` gives: the reference below is that function as it was written before, kept here as the oracle.
use ots_isa::{Code as Diag, EntryId, Insn, ENTRIES, MAX_VARS};
use tsp_iss::driver::classify::{classify_into, classify_items, place, CycleRecs};

fn reference(q: usize, items: &[Result<Insn, Diag>]) -> CycleRecs {
    let n = items.len();
    let mut r = CycleRecs { n, ..CycleRecs::none() };
    match items {
        [] => r,
        [x] => match x {
            Err(d) => CycleRecs { ok: false, err: Some(*d), ..r },
            Ok(i) => {
                if q > 87 {
                    CycleRecs { single: Some(*i), ..r }
                } else {
                    let mut y = r;
                    if place(&mut y, *i) {
                        y
                    } else {
                        CycleRecs { single: Some(*i), ..r }
                    }
                }
            }
        },
        _ => {
            if q > 87 {
                return CycleRecs { ok: false, ..r };
            }
            let mut ok = true;
            for y in items {
                match y {
                    Err(_) => ok = false,
                    Ok(i) => {
                        let fit = place(&mut r, *i);
                        ok &= fit;
                    }
                }
            }
            let fetchm = matches!(r.ifetch.map(|i| i.id()), Some(EntryId::IfetchM));
            if fetchm && (r.refs[0].is_some() || r.refs[1].is_some()) {
                ok = false;
            }
            CycleRecs { ok, ..r }
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One instruction of every entry, with the thread tag 0 to 3 where the entry has one.
fn pool() -> Vec<Result<Insn, Diag>> {
    let mut v = Vec::new();
    for e in ENTRIES.iter() {
        for x in [0u32, 1, 2, 3] {
            let mut f = [0u32; MAX_VARS];
            for (i, var) in e.vars.iter().enumerate() {
                f[i] = if var.name == "thread" { x } else { 0 };
            }
            v.push(Ok(Insn::from_fields(e.id, &f)));
        }
    }
    v.push(Err(Diag::Dec003));
    v
}

#[test]
fn the_record_written_in_place_is_the_by_value_record() {
    let pool = pool();
    let mut r = Rng(0xC1A5_51F1_0000_0001);
    let mut distinct = std::collections::HashSet::new();
    let hot: Vec<_> = pool
        .iter()
        .filter(|x| {
            matches!(
                x.as_ref().map(|i| i.id()),
                Ok(EntryId::Read
                    | EntryId::Write
                    | EntryId::Gather
                    | EntryId::Scatter
                    | EntryId::Ifetch
                    | EntryId::IfetchM)
            )
        })
        .copied()
        .collect();
    for round in 0..200_000 {
        let q = [0usize, 7, 43, 87, 88, 100, 143, 144][(r.next() % 8) as usize];
        let n = (r.next() % 6) as usize; // 0 to 5 instructions
                                         // the later rounds favour the references and the fetch, whose slots collide, over the rest
        let items: Vec<_> = (0..n)
            .map(|_| {
                let k = r.next() as usize;
                if round % 2 == 0 {
                    pool[k % pool.len()]
                } else {
                    hot[k % hot.len()]
                }
            })
            .collect();
        let want = reference(q, &items);
        assert_eq!(classify_items(q, &items), want, "classify_items, queue {q}, {items:?}");
        let mut got = CycleRecs::none();
        classify_into(q, &items, &mut got);
        assert_eq!(got, want, "classify_into, queue {q}, {items:?}");
        distinct.insert((
            want.ok,
            want.single.is_some(),
            want.ifetch.is_some(),
            want.cfg.is_some(),
            want.err.is_some(),
            n.min(3),
        ));
    }
    assert!(distinct.len() >= 12, "the sample must reach the different shapes of record, {}", distinct.len());
}
