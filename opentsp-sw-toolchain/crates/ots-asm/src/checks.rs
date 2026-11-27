//! The static checks at the close of a bundle (`assembly.md` §3, `microarchitecture.md` §3 and §4):
//! the slot rule, placement and the bank rule as errors ([`bundle_checks`]); the AGT placement
//! rules as warnings ([`agt_bundle`]). Ports of `bundle_checks` and `agt_bundle` of
//! `opentsp-arch-spec/tools/tsp_as.py`.
use ots_isa::{queue_name, AsmError, Code, Insn, Isa, AGT_LEVELS, MEM_SLOT_PATH, MEM_SLOT_PER_THREAD, MEM_THREADS};
use std::collections::HashMap;

/// A warning: the program is still valid. `where_` is `path:line` (or `path`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    pub where_: String,
    pub code: Code,
    pub msg: String,
}

impl Warning {
    /// `<where>: <code> (warning): <message>`, the line `tsp_as.py` prints.
    pub fn render(&self) -> String {
        format!("{}: {} (warning): {}", self.where_, self.code.as_str(), self.msg)
    }
}

/// The instructions of one queue in the open bundle, in source order, with their source lines.
#[derive(Clone, Debug)]
pub struct Group {
    pub queue: u16,
    pub insns: Vec<Insn>,
    pub lines: Vec<usize>,
}

fn qname(q: u16) -> &'static str {
    queue_name(q).unwrap_or("?")
}

/// The dispatch checks of one bundle, in the model's order per queue (queues in order of their
/// first line): the slot rule (`QUE-001`, at the line of the queue's last instruction), then each
/// instruction's placement (`QUE-002`), then the bank rule (`MEM-001`, at the first clashing
/// instruction). The first failure is returned.
pub fn bundle_checks(isa: &Isa, used: &[Group], cycle: u64, path: &str) -> Result<(), AsmError> {
    let fail =
        |code: Code, g: &Group, k: usize, msg: String| AsmError::new(code, msg).at(format!("{path}:{}", g.lines[k]));
    for g in used {
        let some: Vec<Option<Insn>> = g.insns.iter().map(|i| Some(*i)).collect();
        let codes = isa.group_codes(g.queue, &some);
        let q = qname(g.queue);
        if codes.contains(&Some(Code::Que001)) {
            let ms: Vec<&str> = g.insns.iter().map(|i| i.entry().mnemonic).collect();
            return Err(fail(
                Code::Que001,
                g,
                g.insns.len() - 1,
                format!(
                    "{} on {q} in bundle @cycle={cycle} do not fit the queue's slots (one Ifetch/IfetchM, one R and one W reference per thread, one AGT configuration record, or one other instruction alone; microarchitecture.md sec. 3)",
                    ms.join(", ")
                ),
            ));
        }
        for (k, i) in g.insns.iter().enumerate() {
            if let Some(code) = isa.placement(i.id(), g.queue) {
                return Err(fail(code, g, k, format!("{} cannot issue on {q}", i.entry().mnemonic)));
            }
        }
        if let Some(k) = codes.iter().position(|c| *c == Some(Code::Mem001)) {
            return Err(fail(
                Code::Mem001,
                g,
                k,
                format!("R and W references of one thread on {q} in bundle @cycle={cycle} address the same bank (or one is a gather/scatter)"),
            ));
        }
    }
    Ok(())
}

/// One open AGT configuration: the cycle it was last continued, its level, the position in the
/// `config` order of the last field written (-1: none) and the number of records received.
#[derive(Clone, Copy, Debug)]
struct Open {
    cycle: u64,
    level: u32,
    pos: i64,
    nrec: u32,
}

/// The open AGT configurations of a program: per (queue, dispatch path R = 0, W = 1).
#[derive(Clone, Debug, Default)]
pub struct Agt {
    open: HashMap<(u16, u8), Open>,
}

impl Agt {
    pub fn new() -> Agt {
        Agt::default()
    }
}

/// AGT placement (`assembly.md` §3; `encoding.json` `agt`; `microarchitecture.md` §4). Each dispatch
/// path of a MEM queue has its own AGT. The model reports a stray configuration record and a fifth
/// level as `OPR-001` at run time, and a new reference on a path whose configuration is open as
/// `QUE-001`; the assembler warns about them statically, in the model's order within a cycle: a
/// cycle of the queue without a configuration record ends the open configurations; the
/// configuration record goes to the configuring path (both: the one with fewer records, R on a
/// tie); then each reference alone on its path opens that path.
pub fn agt_bundle(agt: &mut Agt, used: &[Group], cycle: u64, warn: &mut dyn FnMut(Warning), path: &str) {
    let mut say = |ln: usize, code: Code, msg: String| warn(Warning { where_: format!("{path}:{ln}"), code, msg });
    // no records on the queue in this cycle, or a gap
    agt.open.retain(|&(q, _), st| st.cycle + 1 == cycle && used.iter().any(|g| g.queue == q));
    for g in used {
        let q = g.queue;
        let qn = qname(q);
        let cfg = g.insns.iter().position(|i| i.entry().agt_config.is_some());
        match cfg {
            None => {
                for p in 0..2u8 {
                    agt.open.remove(&(q, p));
                }
            }
            Some(k) => {
                let e = g.insns[k].entry();
                let ln = g.lines[k];
                let open: Vec<u8> = (0..2u8).filter(|p| agt.open.contains_key(&(q, *p))).collect();
                if open.is_empty() {
                    say(
                        ln,
                        Code::Opr001,
                        format!("{} on {qn} @cycle={cycle} configures no loop (no reference alone on its path, nor a configuration, on this queue in the previous cycle)", e.mnemonic),
                    );
                } else {
                    let p = if open.len() == 2 {
                        if agt.open[&(q, 1)].nrec < agt.open[&(q, 0)].nrec {
                            1
                        } else {
                            0
                        }
                    } else {
                        open[0]
                    };
                    let st = agt.open[&(q, p)];
                    let fp = e.agt_config.unwrap() as i64;
                    // a field at or before the last one written starts the next level
                    let level = if fp <= st.pos { st.level + 1 } else { st.level };
                    if level >= AGT_LEVELS {
                        say(ln, Code::Opr001, format!("{} on {qn} @cycle={cycle} needs a level beyond {AGT_LEVELS}; the loop is aborted at run time", e.mnemonic));
                        agt.open.remove(&(q, p));
                    } else {
                        agt.open.insert((q, p), Open { cycle, level, pos: fp, nrec: st.nrec + 1 });
                    }
                }
                for p in 0..2u8 {
                    if let Some(st) = agt.open.get_mut(&(q, p)) {
                        st.cycle = cycle; // the configurations continue into this cycle
                    }
                }
            }
        }
        // the references of each path: one entry for each slot of the instruction on that path, and an
        // instruction that takes a per-thread slot on every thread (IfetchM) counts once per thread
        let mut on_path: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for (k, i) in g.insns.iter().enumerate() {
            let e = i.entry();
            for &s in e.slots.unwrap_or(&[]) {
                let s = s as usize;
                if let Some(p) = MEM_SLOT_PATH[s] {
                    let n = if MEM_SLOT_PER_THREAD[s] && e.all_threads { MEM_THREADS as usize } else { 1 };
                    for _ in 0..n {
                        on_path[p as usize].push(k);
                    }
                }
            }
        }
        let held: Vec<bool> = (0..2u8).map(|p| agt.open.contains_key(&(q, p))).collect();
        for p in 0..2usize {
            if held[p] {
                for &k in &on_path[p] {
                    say(
                        g.lines[k],
                        Code::Que001,
                        format!("{} on {qn} @cycle={cycle}: its path's AGT configuration is open; only configuration records get through", g.insns[k].entry().mnemonic),
                    );
                }
            } else if on_path[p].len() == 1 && g.insns[on_path[p][0]].entry().agt_loopable {
                agt.open.insert((q, p as u8), Open { cycle, level: 0, pos: -1, nrec: 0 });
            }
        }
    }
}
