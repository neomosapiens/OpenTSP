//! The decoder's legality rules against `tsp_sim`: the ISS decodes with the
//! generated decoder of `ots-isa`, which mirrors the Sail `when` guards (`binop_legal`,
//! `unop_legal`, `cmp_legal`, `ty_valid`, the operand ranges). A word the decoder accepts but the
//! model rejects (or the other way round) would make an `unreachable!` or an `expect` of a slice
//! fire (exit 4) or hide a DEC-003.
//!
//! For every catalogue entry the operand fields are swept: the small fields (up to 4 bits: types,
//! modes, flags, names) over their whole product, the others over a few boundary values, capped
//! per entry (the rest drawn at random); a word with a reserved bit set is added now and then. Each
//! word is the only instruction of a queue its entry may issue on, up to 145 to a program. Exit
//! code and the whole `--trace` stdout must be byte-identical, and a profile must reach both
//! DEC-003 and execution of the entries (diagnostics other than DEC-003, QUE-002).
//! (`$ISS_DECODE_CAP` words per entry, default 400.)
mod common;
use common::prog::*;
use common::*;
use ots_isa::{EntryId, MAX_VARS};
use std::collections::BTreeMap;
use std::fmt::Write as _;

fn words(r: &mut Rng, en: &ots_isa::Entry, cap: usize) -> Vec<u64> {
    let small: Vec<usize> = (0..en.vars.len()).filter(|i| en.vars[*i].nbits <= 4).collect();
    let pick = |r: &mut Rng, i: usize| {
        let v = &en.vars[i];
        let full = (1u64 << v.nbits.min(32)) - 1;
        match r.below(5) {
            0 => 0,
            1 => 1.min(full) as u32,
            2 => full as u32,
            3 => full.min(63) as u32,
            _ => r.below(full.min(1 << 20) + 1) as u32,
        }
    };
    let space: usize = small.iter().map(|i| 1usize << en.vars[*i].nbits).product();
    let mut out = Vec::new();
    for k in 0..space.min(cap).max(1) {
        let mut f = [0u32; MAX_VARS];
        let mut rem = k;
        for (i, v) in en.vars.iter().enumerate() {
            f[i] = if small.contains(&i) {
                // the whole product of the small fields when it fits the cap, else a draw
                if space <= cap {
                    let d = (rem % (1 << v.nbits)) as u32;
                    rem >>= v.nbits;
                    d
                } else {
                    r.below(1 << v.nbits) as u32
                }
            } else {
                pick(r, i)
            };
        }
        let mut w = en.encode_fields(&f[..en.vars.len()]);
        let all = if en.width >= 64 { u64::MAX } else { (1u64 << en.width) - 1 };
        let reserved = all & !en.mask & !en.var_mask;
        if reserved != 0 && r.chance(8) {
            let bits: Vec<u32> = (0..64).filter(|b| reserved >> b & 1 == 1).collect();
            w |= 1u64 << r.pick(&bits);
        }
        out.push(w);
    }
    out
}

fn units(en: &ots_isa::Entry, w: u64) -> Vec<u32> {
    if en.width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

fn profile(name: &str, seed: u64) {
    let e = env(name);
    let cfg = cfg_path(name).to_str().unwrap().to_string();
    let cap = std::env::var("ISS_DECODE_CAP").ok().and_then(|s| s.parse().ok()).unwrap_or(400);
    let mut r = Rng::new(seed);
    let mut progs: Vec<BTreeMap<u16, Vec<u32>>> = Vec::new();
    let mut total = 0usize;
    for id in EntryId::ALL {
        let en = id.entry();
        let queues = queues_of(&e, id);
        for w in words(&mut r, en, cap) {
            let start = r.below(queues.len() as u64) as usize;
            let mut slot = None;
            'find: for k in 0..queues.len() {
                let q = queues[(start + k) % queues.len()];
                for (p, prog) in progs.iter().enumerate() {
                    if !prog.contains_key(&q) {
                        slot = Some((p, q));
                        break 'find;
                    }
                }
            }
            let (p, q) = slot.unwrap_or_else(|| {
                progs.push(BTreeMap::new());
                (progs.len() - 1, queues[start])
            });
            progs[p].insert(q, units(en, w));
            total += 1;
        }
    }
    let texts: Vec<String> = progs
        .iter()
        .map(|p| {
            let mut t = String::new();
            for (q, us) in p {
                write!(t, "Q {q:x}").unwrap();
                for u in us {
                    write!(t, " {u:x}").unwrap();
                }
                t.push('\n');
            }
            t
        })
        .collect();
    let diags = std::sync::Mutex::new(BTreeMap::<String, usize>::new());
    let bad = parallel(&texts, |t| {
        let args = [cfg.as_str(), "--trace", "--max-cycles", "400"];
        let ours = run_iss(&args, t);
        let theirs = run_sim(&args, t);
        {
            let mut d = diags.lock().unwrap();
            for l in text(&theirs.stdout).lines().filter(|l| l.starts_with("T diag")) {
                *d.entry(l.rsplit(' ').next().unwrap().to_string()).or_default() += 1;
            }
        }
        if ours.status.code() == theirs.status.code() && ours.stdout == theirs.stdout {
            return Ok(());
        }
        Err(format!(
            "({name}):\n{t}\nours (exit {:?}):\n{}\ntsp_sim (exit {:?}):\n{}",
            ours.status.code(),
            text(&ours.stdout),
            theirs.status.code(),
            text(&theirs.stdout)
        ))
    });
    let diags = diags.into_inner().unwrap();
    eprintln!(
        "decode_diff {name}: {total} words in {} programs, {} differ; diagnostics {diags:?}",
        texts.len(),
        bad.len()
    );
    assert!(
        bad.is_empty(),
        "{} programs differ on {name}; first:\n{}",
        bad.len(),
        bad[0].chars().take(6000).collect::<String>()
    );
    let dec = diags.get("OTS-E-DEC-003").copied().unwrap_or(0);
    let other: usize =
        diags.iter().filter(|(c, _)| !matches!(c.as_str(), "OTS-E-DEC-003" | "OTS-E-QUE-002")).map(|(_, n)| n).sum();
    assert!(
        dec >= total / 20 && other >= total / 4,
        "{name}: {dec} DEC-003, {other} other diagnostics of {total} words"
    );
}

#[test]
fn every_entrys_operand_sweep_decodes_as_tsp_sim_does_ref() {
    profile("ref", 301);
}

#[test]
fn every_entrys_operand_sweep_decodes_as_tsp_sim_does_mini() {
    profile("mini", 302);
}
