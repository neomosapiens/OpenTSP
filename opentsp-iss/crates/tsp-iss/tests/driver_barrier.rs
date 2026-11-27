//! Tests of `driver/barrier`, moved from the module's in-source test block (they use the crate's public API).
use tsp_iss::driver::barrier::*;

const LAT: u64 = 35;

// test_barrier_multiple_windows: every Sync opens its own window, closed by the first Notify
// at or after it.
#[test]
fn test_barrier_multiple_windows() {
    let mut b = Barrier::with_latency(LAT);
    b.sync(144, 5);
    b.notify(10);
    assert!(!b.barrier_parked(4));
    assert_eq!(b.barrier_parked(20), 20 < 10 + LAT);
    assert!(b.barrier_parked(9 + LAT) && !b.barrier_parked(10 + LAT));
    b.sync(144, 100);
    assert!(b.barrier_parked(1000), "no Notify yet: open-ended");
    b.notify(110);
    assert!(b.barrier_parked(100) && b.barrier_parked(109 + LAT));
    assert!(!b.barrier_parked(110 + LAT));
}

// test_queue_sync: a data-queue Sync parks that queue only; a Notify on any queue releases it
#[test]
fn test_queue_sync() {
    let mut b = Barrier::with_latency(LAT);
    b.sync(88, 10);
    b.notify(20);
    assert!(b.queue_parked(88, 30));
    assert!(!b.queue_parked(89, 30));
    assert!(!b.queue_parked(88, 20 + LAT));
    assert!(!b.barrier_parked(30));
}

#[test]
fn a_control_sync_parks_every_data_queue_and_a_notify_in_the_same_cycle_releases_it() {
    let mut b = Barrier::with_latency(LAT);
    b.notify(7);
    b.sync(3, 7); // dispatched later in the Notify's cycle: released by it
    assert_eq!(b.release_at[3], Some(7 + u128::from(LAT)));
    b.sync(144, 9);
    assert!(b.queue_parked(0, 9) && b.queue_parked(143, 12) && !b.queue_parked(0, 8));
    assert!(b.stuck(144) && !b.stuck(3));
    b.notify(9);
    assert!(!b.stuck(144) && b.queue_parked(5, 9 + LAT - 1) && !b.queue_parked(5, 9 + LAT));
    b.retire(9 + LAT);
    assert_eq!(b.parked_since[144], None);
}

#[test]
fn default_reset_and_the_queue_view_of_the_control_barrier() {
    let mut b = Barrier::default();
    assert!(!b.queue_parked(3, 0) && !b.barrier_parked(0) && !b.stuck(3) && !b.stuck(144));
    b.sync(144, 5);
    assert!(b.stuck(144), "a Sync with no Notify never resumes by itself");
    assert!(b.queue_parked(3, 6), "the control channel's Sync parks every data queue");
    assert!(b.barrier_parked(6));
    b.sync(7, 5);
    assert!(b.stuck(7));
    b.notify(4); // before the Syncs: releases nothing
    assert!(b.stuck(144) && b.stuck(7));
    b.notify(5);
    assert!(!b.stuck(144) && !b.stuck(7));
    // reset forgets the Syncs, keeps the latency
    let mut c = Barrier::with_latency(LAT);
    c.sync(9, 1);
    c.notify(1);
    c.reset();
    assert!(!c.queue_parked(9, 2) && !c.stuck(9));
    c.sync(9, 3);
    c.notify(3);
    assert!(c.queue_parked(9, 3 + LAT - 1) && !c.queue_parked(9, 3 + LAT));
    // retire clears a window that has closed, so that a later Sync can park again
    c.retire(3 + LAT);
    assert!(!c.queue_parked(9, 3) && !c.stuck(9));
}

/// `retire` skips its scan until the earliest release cycle: each release retires at its own cycle, whichever way
/// it was set (a Notify after the Sync, a Sync in the Notify's cycle), and a window opened after a retire is seen.
#[test]
fn retire_takes_each_window_at_its_own_release_cycle() {
    let mut b = Barrier::with_latency(LAT);
    b.sync(10, 1);
    b.sync(20, 2);
    b.notify(2); // both released at 2 + LAT
    b.retire(2 + LAT - 1);
    assert!(b.parked_since[10].is_some() && b.parked_since[20].is_some(), "not yet");
    b.retire(2 + LAT);
    assert!(b.parked_since[10].is_none() && b.parked_since[20].is_none());
    // a Sync that no Notify has released yet is released by a later Notify, after a retire found nothing to take
    b.sync(30, 50);
    b.retire(50 + LAT + 10);
    assert!(b.parked_since[30].is_some(), "no release yet");
    b.notify(100); // queue 30 at 100 + LAT
    b.retire(100 + LAT - 1);
    assert!(b.parked_since[30].is_some());
    b.retire(100 + LAT);
    assert!(b.parked_since[30].is_none());
    // a Sync in the cycle of the last Notify is released at once and retires in time
    b.notify(200);
    b.sync(40, 200);
    b.retire(200 + LAT);
    assert!(b.parked_since[40].is_none() && b.release_at[40].is_none());
    // a window that opens after everything retired is retired in its turn
    b.sync(50, 300);
    b.notify(300);
    b.retire(300 + LAT - 1);
    assert!(b.parked_since[50].is_some());
    b.retire(300 + LAT);
    assert!(b.parked_since[50].is_none());
}
