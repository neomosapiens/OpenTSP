//! Directed programs for the rare corners of the dispatch driver: the control channel's Sync and
//! Notify against data queues, Syncs that no Notify releases (the machine must finish with the
//! queues still parked), a Sync behind an AGT loop or a Repeat, AGT configurations cut short, a
//! Repeat on a parked queue, `IfetchM` past a `SetSegment`, and `Ifetch` with nothing to fetch. Each
//! program runs on `tsp_sim` and on the ISS with `--trace --lint` and, on the multi-chip path, with
//! `--chips 2`: exit code and stdout must be byte-identical.
mod common;
use common::*;
use ots_isa::{encode, Insn};
use std::fmt::Write as _;

const CTRL: u16 = 0xffff;

fn units(i: &Insn) -> Vec<u32> {
    let w = encode(i);
    if i.entry().width == 64 {
        vec![(w >> 32) as u32, w as u32]
    } else {
        vec![w as u32]
    }
}

/// `Q <queue> <units>` lines for the instructions of each queue, one instruction after the other.
fn q_text(queues: &[(u16, Vec<Insn>)]) -> String {
    let mut out = String::new();
    for (q, prog) in queues {
        write!(out, "Q {q:x}").unwrap();
        for u in prog.iter().flat_map(units) {
            write!(out, " {u:x}").unwrap();
        }
        out.push('\n');
    }
    out
}

fn nop(n: u32) -> Insn {
    Insn::NOP { n }
}
fn setvl(n: u32) -> Insn {
    Insn::SetVL { n }
}
fn read(addr: u32, dst: u32, thread: u32) -> Insn {
    Insn::Read { addr, dst, thread }
}

/// A named program: the instructions of each queue it uses.
type Program = (&'static str, Vec<(u16, Vec<Insn>)>);

fn programs() -> Vec<Program> {
    let agt = |extra: Vec<Insn>| {
        let mut v = vec![Insn::Countdown { d: 1 }, Insn::Iterations { n: 2 }, Insn::Step { a: 1 }, read(0, 0, 0)];
        v.extend(extra);
        v
    };
    vec![
        (
            "control Sync, nothing releases it",
            vec![(CTRL, vec![Insn::Sync]), (0, vec![setvl(3), setvl(4)]), (100, vec![nop(4), setvl(2)])],
        ),
        (
            "control Notify releases a data Sync",
            vec![
                (CTRL, vec![nop(5), Insn::Notify]),
                (0, vec![Insn::Sync, setvl(2)]),
                (100, vec![Insn::Sync, setvl(3)]),
            ],
        ),
        (
            "control Notify before any Sync",
            vec![(CTRL, vec![Insn::Notify, nop(3), Insn::Notify]), (3, vec![nop(9), Insn::Sync, setvl(2)])],
        ),
        (
            "control Sync then Notify on a data queue",
            vec![
                (CTRL, vec![nop(2), Insn::Sync, setvl(2)]),
                (44, vec![nop(6), Insn::Notify, setvl(2)]),
                (88, vec![setvl(2), setvl(3)]),
            ],
        ),
        (
            "a control Sync and a data Sync, one Notify",
            vec![(CTRL, vec![Insn::Sync, Insn::Notify]), (7, vec![Insn::Sync, setvl(2)])],
        ),
        ("data Sync with no Notify", vec![(0, vec![Insn::Sync, setvl(2)]), (87, vec![setvl(3)])]),
        ("Sync behind an AGT loop", vec![(0, agt(vec![Insn::Sync, setvl(2)])), (CTRL, vec![nop(30), Insn::Notify])]),
        ("Sync behind an AGT loop, never released", vec![(0, agt(vec![Insn::Sync, setvl(2)]))]),
        (
            "Repeat parked by a Sync",
            vec![
                (0, vec![setvl(2), Insn::Repeat { n: 4, d: 2 }, Insn::Sync, setvl(3)]),
                (CTRL, vec![nop(12), Insn::Notify]),
            ],
        ),
        ("Repeat parked by a Sync, never released", vec![(0, vec![setvl(2), Insn::Repeat { n: 4, d: 2 }, Insn::Sync])]),
        (
            "AGT configuration cut short by a Sync",
            vec![
                (0, vec![Insn::Countdown { d: 2 }, Insn::Iterations { n: 3 }, Insn::Sync]),
                (CTRL, vec![nop(8), Insn::Notify]),
            ],
        ),
        (
            "an AGT loop and a Sync on the other queue of the node",
            vec![(0, agt(vec![])), (1, vec![Insn::Sync, setvl(4)]), (CTRL, vec![nop(3), Insn::Notify])],
        ),
        (
            "Repeat on the control channel",
            vec![
                (CTRL, vec![setvl(3), Insn::Repeat { n: 3, d: 1 }, nop(4), Insn::Notify]),
                (0, vec![Insn::Sync, setvl(2)]),
            ],
        ),
        (
            "control channel executes memory and vector instructions",
            vec![(
                CTRL,
                vec![
                    Insn::Config { v: 3 },
                    setvl(2),
                    Insn::SetSegment { n: 4 },
                    read(0, 0, 0),
                    Insn::Deskew,
                    Insn::RuntimeDeskew { t: 5 },
                ],
            )],
        ),
        (
            "IfetchM past the segment",
            vec![(
                0,
                vec![
                    Insn::SetSegment { n: 3 },
                    Insn::IfetchM { addr: 1 },
                    Insn::IfetchM { addr: 2 },
                    Insn::IfetchM { addr: 5 },
                    setvl(2),
                ],
            )],
        ),
        (
            "Ifetch with nothing on the streams, on several queue kinds",
            vec![
                (0, vec![Insn::Ifetch { stream: 0 }]),
                (100, vec![Insn::Ifetch { stream: 2 }]),
                (CTRL, vec![Insn::Ifetch { stream: 4 }]),
                (130, vec![Insn::Ifetch { stream: 3 }]),
            ],
        ),
        (
            "deskew then Sync, Notify from a data queue",
            vec![
                (0, vec![Insn::Deskew, Insn::Sync, setvl(2)]),
                (87, vec![Insn::RuntimeDeskew { t: 20 }, Insn::Notify, setvl(3)]),
            ],
        ),
        (
            "two Syncs on one queue and two Notifies",
            vec![
                (0, vec![Insn::Sync, setvl(2), Insn::Sync, setvl(3)]),
                (CTRL, vec![nop(4), Insn::Notify, nop(40), Insn::Notify]),
            ],
        ),
    ]
}

fn check(profile: &str, extra: &[&str]) {
    let cfg = config(profile).to_str().unwrap().to_string();
    for (name, qs) in programs() {
        let prog = q_text(&qs);
        let mut args = vec![cfg.as_str(), "--trace", "--lint"];
        args.extend(extra);
        let ours = run_iss(&args, &prog);
        let theirs = run_sim(&args, &prog);
        assert_eq!(
            ours.status.code(),
            theirs.status.code(),
            "{name} ({profile} {extra:?}): exit code\n{prog}\nours: {}\ntsp_sim: {}",
            text_of(&ours.stderr),
            text_of(&theirs.stderr)
        );
        assert_eq!(text_of(&ours.stdout), text_of(&theirs.stdout), "{name} ({profile} {extra:?}):\n{prog}");
    }
}

fn text_of(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[test]
fn directed_driver_corners_match_tsp_sim_ref() {
    check("ref", &[]);
}

#[test]
fn directed_driver_corners_match_tsp_sim_mini() {
    check("mini", &[]);
}

#[test]
fn directed_driver_corners_match_tsp_sim_on_two_chips() {
    check("mini", &["--chips", "2"]);
}

#[test]
fn directed_driver_corners_stop_at_max_cycles_like_tsp_sim() {
    check("ref", &["--max-cycles", "9"]);
}
