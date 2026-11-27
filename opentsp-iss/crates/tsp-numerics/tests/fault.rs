//! A model fault leaves the numerics as an unwinding panic whose payload is `ModelFault`, without
//! running the panic hook (a library user sees no message; `tsp_iss::Machine` turns it into an error).
use std::panic::catch_unwind;
use std::sync::atomic::{AtomicUsize, Ordering};
use tsp_numerics::fault::{model_fault, ModelFault};

static HOOK_RUNS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn a_model_fault_carries_its_message_and_does_not_run_the_panic_hook() {
    // one test body: the panic hook is process-global
    std::panic::set_hook(Box::new(|_| {
        HOOK_RUNS.fetch_add(1, Ordering::SeqCst);
    }));
    let fault = catch_unwind(|| model_fault("fp_transcendental: Ziv loop did not converge"));
    let after_fault = HOOK_RUNS.load(Ordering::SeqCst);
    let plain = catch_unwind(|| panic!("a plain panic"));
    let after_plain = HOOK_RUNS.load(Ordering::SeqCst);
    let _ = std::panic::take_hook();

    let payload = fault.expect_err("a fault unwinds");
    assert_eq!(payload.downcast::<ModelFault>().unwrap().0, "fp_transcendental: Ziv loop did not converge");
    assert_eq!(after_fault, 0, "the hook ran for a model fault");
    assert!(plain.is_err());
    assert_eq!(after_plain, 1, "the hook runs for a plain panic");
}
