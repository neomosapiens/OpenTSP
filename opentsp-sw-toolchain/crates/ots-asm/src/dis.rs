//! The disassembler (`assembly.md` §4 "Disassembly"): the program text of text images, and the number
//! of instructions the model reports at decode or placement. A port of `disassemble` of
//! `opentsp-arch-spec/tools/tsp_as.py`.
//!
//! The images are raised with no waits ([`ots_format::text::raise_images`]); the listing has one
//! bundle per cycle that holds an instruction, and in it the queues in order of their first
//! instruction. Each instruction the model reports is annotated with the model's code, in the
//! model's order: for two or more instructions in one parcel the slot rule ([`Isa::group_codes`]:
//! `QUE-001` on each when they do not fit, else `MEM-001` on a clashing same-thread read and write);
//! for a single instruction `DEC-003` (the word does not decode), then `QUE-002` ([`Isa::placement`]).
//! For an instruction with no annotation, assembling the listing lowers to the same cycle view
//! (`assemble(disassemble(i))`, with a parcel in source order).
use crate::program::ARCH;
use ots_format::{
    text::{raise_images, Images, Raised},
    Extern,
};
use ots_isa::{queue_name, render, Code, Insn, Isa};

/// `illegal32 0x…` or `illegal64 0x…` of a raised word that does not decode.
fn illegal(r: &Raised) -> String {
    if r.units == 1 {
        format!("illegal32 {:#x}", r.word & 0xFFFF_FFFF)
    } else {
        format!("illegal64 {:#x}", r.word)
    }
}

/// The listing of `images` under `isa` (with the externs of an object, in the order given), and the
/// number of annotated instructions. The command exits 1 when that number is not zero.
pub fn disassemble(isa: &Isa, images: &Images, externs: &[Extern]) -> (String, usize) {
    let mut out = vec![format!(".arch {ARCH}"), format!(".profile {}", isa.profile().name())];
    out.extend(
        externs
            .iter()
            .map(|x| format!(".extern {}, slice={}, addr={:#x}, size={:#x}", x.name, x.slice, x.addr, x.size)),
    );
    out.push(".section text".to_string());
    let mut bad = 0;
    let entries = raise_images(images);
    let mut cycles: Vec<i64> = entries.iter().map(|e| e.cycle).collect();
    cycles.sort_unstable();
    cycles.dedup();
    for c in cycles {
        out.push(format!("bundle @cycle={c} {{"));
        let rs: Vec<&Raised> = entries.iter().filter(|e| e.cycle == c).collect();
        let mut queues: Vec<u16> = Vec::new();
        for r in &rs {
            if !queues.contains(&r.queue) {
                queues.push(r.queue);
            }
        }
        for q in queues {
            let group: Vec<&Raised> = rs.iter().copied().filter(|r| r.queue == q).collect();
            let dec: Vec<Result<Insn, Code>> = group.iter().map(|r| isa.decode_word(r.word, r.units)).collect();
            let opts: Vec<Option<Insn>> = dec.iter().map(|d| d.as_ref().ok().copied()).collect();
            let shares = isa.group_codes(q, &opts);
            let qn = queue_name(q).unwrap_or("?");
            for ((r, d), share) in group.iter().zip(&dec).zip(&shares) {
                let line = match (share, d) {
                    // the model checks the group before decoding each instruction
                    (Some(s), _) => {
                        bad += 1;
                        let t = match d {
                            Ok(i) => render(i),
                            Err(_) => illegal(r),
                        };
                        format!("{t}   ; {}", s.as_str())
                    }
                    (None, Err(code)) => {
                        bad += 1;
                        format!("{}   ; {}", illegal(r), code.as_str())
                    }
                    (None, Ok(i)) => match isa.placement(i.id(), q) {
                        Some(pl) => {
                            bad += 1;
                            format!("{}   ; {}", render(i), pl.as_str())
                        }
                        None => render(i),
                    },
                };
                out.push(format!("  {qn}: {line}"));
            }
        }
        out.push("}".to_string());
    }
    let mut s = out.join("\n");
    s.push('\n');
    (s, bad)
}
