//! A chip's link port is asked twice per tick whether the harness has anything to do (`next_event`,
//! `arrivals`). While no vector has ever entered an inbound FIFO the answers are fixed (no event, no
//! arrival), so `PlanLinks` gives them without taking the lock of the shared `ChipLinks`; from the first vector
//! on it asks `ChipLinks`. These tests hold that to the locked answer on random traffic, and check that the
//! idle answer does not take the lock.
use ots_profile::Profile;
use std::sync::{mpsc, Arc};
use std::time::Duration;
use tsp_iss::geom::{Geom, MAX_LANES};
use tsp_iss::harness::cli::LinkMode;
use tsp_iss::harness::links::*;
use tsp_iss::links::LinkPort;

fn geom() -> Geom {
    Geom::new(&Profile::load("ref").unwrap())
}

fn vecmsg(arrival: u64, link: usize, tag: u8) -> VecMsg {
    let mut b = [0u8; MAX_LANES];
    b[0] = tag;
    VecMsg {
        arrival: arrival.into(),
        chip: 0,
        link,
        state: VState::Pending,
        taken_at: 0,
        ptype: 0,
        ctl: None,
        fec_c: 0,
        fec_u: 0,
        b,
        poison: 0,
    }
}

#[test]
fn an_idle_port_answers_without_the_lock() {
    let g = geom();
    let mut p = PlanLinks::loopback(&g);
    let state = p.state();
    let held = state.lock().unwrap(); // the harness is inside the chip's links: a locking answer would wait for it
    let (tx, rx) = mpsc::channel();
    let t = std::thread::spawn(move || {
        let ev = p.next_event();
        let arr = p.arrivals(1_000_000);
        tx.send((ev, arr)).unwrap();
    });
    let got = rx.recv_timeout(Duration::from_secs(5));
    drop(held);
    t.join().unwrap();
    assert_eq!(got, Ok((None, Vec::new())), "an idle port has no event and no arrival, and does not wait for the lock");
}

#[test]
fn the_port_agrees_with_the_locked_state_on_random_traffic() {
    let g = geom();
    let mut seen_event = 0;
    let mut seen_arrival = 0;
    for seed in 1..=40u64 {
        let plan = Arc::new(LinkPlan::new(LinkMode::Nominal, &g, 1).unwrap());
        let mut p = PlanLinks::new(plan, 0, &g);
        let state = p.state();
        let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut rnd = move |n: u64| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d) % n
        };
        let mut now = 0u64;
        for step in 0..120 {
            now += rnd(6);
            // two links and mostly deliveries: a FIFO of 16 overruns, which is what makes events and arrivals
            match rnd(8) {
                0..=3 => state.lock().unwrap().deliver(vecmsg(now + rnd(30), rnd(2) as usize, step as u8)),
                4 => {
                    let mut v = [0u8; MAX_LANES];
                    v[0] = step as u8;
                    p.send(now, rnd(2) as u8, &v);
                }
                5 => {
                    let l = rnd(2) as u8;
                    if p.recv_ready(now, l) {
                        p.recv(now, l);
                    }
                }
                _ => {}
            }
            let want_ev = state.lock().unwrap().next_event();
            assert_eq!(p.next_event(), want_ev, "seed {seed} step {step}: next_event");
            if want_ev.is_some() {
                seen_event += 1;
            }
            // `arrivals` consumes: the port gets the first look, the locked state then has nothing left to give
            // unless the port skipped it
            let got = p.arrivals(now);
            let left = state.lock().unwrap().process_arrivals(now);
            assert!(left.is_empty(), "seed {seed} step {step}: arrivals left {left:?} after the port gave {got:?}");
            seen_arrival += got.len();
        }
    }
    assert!(seen_event > 20 && seen_arrival > 5, "events {seen_event}, arrivals {seen_arrival}");
}

#[test]
fn vectors_delivered_by_the_harness_wake_the_port() {
    // the harness puts vectors in from another chip (`deliver`) and the chip itself never sent one: 17 on a FIFO of
    // 16 make an overrun the port must report, so the first delivery must end the idle answer
    let g = geom();
    let mut p = PlanLinks::loopback(&g);
    assert_eq!(p.next_event(), None);
    for k in 0..17u64 {
        p.state().lock().unwrap().deliver(vecmsg(10 + k, 0, k as u8));
    }
    assert_eq!(p.next_event(), Some(10 + 16), "the 17th arrival finds the FIFO full");
    assert_eq!(p.arrivals(40), vec![10 + 16]);
}
