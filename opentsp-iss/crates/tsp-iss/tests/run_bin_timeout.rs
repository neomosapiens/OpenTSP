//! `run_bin` of the test helpers kills a program that runs past `$ISS_TEST_TIMEOUT` and fails the test
//! (a hung process must not stall the whole run), and still returns the output of one that finishes.
mod common;
use common::*;

#[test]
fn a_hung_program_is_killed_and_fails_the_test() {
    // the only test of this crate: nothing else reads the variable meanwhile
    std::env::set_var("ISS_TEST_TIMEOUT", "1");
    let started = std::time::Instant::now();
    let r = std::panic::catch_unwind(|| run_bin("/bin/sleep", &["60"], b""));
    let msg = *r.expect_err("a program past the limit must fail the test").downcast::<String>().unwrap();
    assert!(msg.contains("ran for more than 1 s and was killed"), "{msg}");
    assert!(started.elapsed().as_secs() < 20, "took {:?}", started.elapsed());
    let o = run_bin("/bin/cat", &[], b"abc");
    assert_eq!((o.status.code(), o.stdout), (Some(0), b"abc".to_vec()));
}
