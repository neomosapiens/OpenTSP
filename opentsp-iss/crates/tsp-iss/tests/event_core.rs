//! The event-driven core's structures: the set of non-empty fill FIFOs, the occupancy masks of the commit and
//! execute rings and the wake calendar follow the state they summarise.
use ots_profile::Profile;
use tsp_iss::driver::Driver;
use tsp_iss::links::NoLinks;
use tsp_iss::machine::Machine;
use tsp_iss::observe::NoObserver;

fn machine() -> Machine {
    Machine::new(&Profile::load("ref").unwrap(), 0, Box::new(NoLinks)).unwrap()
}

fn has(set: &[u64; 3], q: usize) -> bool {
    set[q >> 6] >> (q & 63) & 1 == 1
}

#[test]
fn a_fetch_enters_the_fill_set_and_the_last_unit_delivered_leaves_it() {
    let mut m = machine();
    assert_eq!(m.iq().fill_queues(), [0; 3]);
    let v = [0x11u8; 320];
    m.edit_state(|state| state.iq.fetch(7, 0, &v, &v)).unwrap();
    m.edit_state(|state| state.iq.fetch(144, 0, &v, &v)).unwrap();
    assert!(has(&m.iq().fill_queues(), 7) && has(&m.iq().fill_queues(), 144));
    let mut c = 0;
    while !m.iq().fill[7].is_empty() {
        m.edit_state(|state| state.iq.deliver(7, c)).unwrap();
        c += 1;
    }
    assert!(!has(&m.iq().fill_queues(), 7) && has(&m.iq().fill_queues(), 144));
    m.edit_state(|state| state.iq.clear()).unwrap();
    assert_eq!(m.iq().fill_queues(), [0; 3]);
}

#[test]
fn the_calendar_returns_a_queue_at_its_wake_cycle_and_not_before() {
    let mut d = Driver::new(1);
    d.cal_take(0); // every queue starts due
    d.wake[5] = 40;
    d.cal_schedule(5, 40);
    d.wake[9] = 1000;
    d.cal_schedule(9, 1000);
    assert!(!has(&d.cal_take(39), 5));
    assert!(has(&d.cal_take(41), 5), "a tick past the wake cycle still visits the queue");
    assert!(!has(&d.cal_take(999), 9));
    assert!(has(&d.cal_take(1000), 9), "a wake beyond the ring comes back from the overflow");
    d.touch(3);
    assert!(has(&d.cal_take(1001), 3));
}

#[test]
fn an_empty_tick_leaves_the_rings_empty() {
    let mut m = machine();
    m.tick(0, &mut NoObserver);
    assert_eq!(m.driver().pending_occ, 0);
    assert_eq!(m.driver().exec_occ, 0);
}

/// Wakes far beyond the ring come back at their cycle across many ticks, in any order, and a wake already in the
/// past when its slot comes round is returned at the next tick: the calendar is compared with a plain list.
#[test]
fn far_wakes_return_at_their_cycle_over_many_ticks() {
    let mut d = Driver::new(1);
    d.cal_take(0);
    let mut model: Vec<(u64, usize)> = Vec::new();
    let mut x = 12345u64;
    let mut rnd = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    let mut c = 0u64;
    for _ in 0..400 {
        for _ in 0..rnd(4) {
            let q = rnd(144) as usize;
            let w = c + 1 + rnd(1500);
            d.wake[q] = w.into();
            d.cal_schedule(q, w.into());
            model.retain(|&(_, m)| m != q);
            model.push((w, q));
        }
        c += 1 + rnd(120);
        let got = d.cal_take(c);
        let mut want = [0u64; 3];
        for &(w, q) in &model {
            if w <= c {
                want[q >> 6] |= 1 << (q & 63);
            }
        }
        for q in 0..144 {
            if has(&want, q) {
                assert!(has(&got, q), "queue {q} due at or before {c} but not returned");
            }
        }
        model.retain(|&(w, _)| w > c);
    }
}
