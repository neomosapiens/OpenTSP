//! Model faults (the Sail `internal_error`) reach a library caller as `RunError::Internal` naming the
//! cycle, and the compatible-mode binary exits 4 with the message; the host floating-point
//! self-check does the same for an environment the f32 arithmetic cannot rely on.
use ots_isa::Insn;
use ots_profile::Profile;
use std::ffi::OsString;
use tsp_iss::harness::main_with;
use tsp_iss::links::NoLinks;
use tsp_iss::machine::{fault_boundary, Machine, RunError};
use tsp_iss::observe::NoObserver;
use tsp_numerics::fault::model_fault;

fn machine(name: &str) -> Machine {
    Machine::new(&Profile::load(name).unwrap(), 0, Box::new(NoLinks)).unwrap()
}

#[test]
fn a_held_parcel_reaching_dispatch_is_an_internal_error_naming_the_cycle() {
    let mut m = machine("ref");
    // a parcel is dispatched only when nothing holds it; here queue 3 is busy until cycle 100
    m.edit_state(|state| state.driver.queue_busy_until[3] = 100).unwrap();
    match m.dispatch_check(3, 7, &Insn::Deskew, None) {
        Err(RunError::Internal(msg)) => {
            assert!(msg.starts_with("model fault at cycle 7: "), "{msg}");
            assert!(msg.contains("dispatch of a held instruction") && msg.contains("queue 3"), "{msg}");
        }
        other => panic!("{other:?}"),
    }
    // the same parcel on a free queue is a plain check: no fault
    assert!(m.dispatch_check(4, 7, &Insn::Deskew, None).is_ok());
}

#[test]
fn a_model_fault_raised_below_the_kernels_is_an_internal_error_at_its_cycle() {
    // the numerics raise `ModelFault` (Ziv non-convergence, an index that does not fit): the boundary of a tick
    let r = fault_boundary(5, || -> u32 { model_fault("fp_transcendental: Ziv loop did not converge") });
    assert_eq!(
        r,
        Err(RunError::Internal("model fault at cycle 5: fp_transcendental: Ziv loop did not converge".into()))
    );
    assert_eq!(fault_boundary(5, || 3), Ok(3));
}

#[test]
#[should_panic(expected = "a plain bug")]
fn any_other_panic_passes_the_boundary_unchanged() {
    let _ = fault_boundary(5, || -> u32 { panic!("a plain bug") });
}

#[test]
fn the_message_is_the_one_tsp_sim_prints() {
    // tsp_sim: "model fault at cycle N: <message>" (c_emulator/tsp_sim.c check_model)
    assert_eq!(
        RunError::fault(12, "configuration record").to_string(),
        "model fault at cycle 12: configuration record"
    );
}

/// Set flush-to-zero in this thread's floating-point control register.
#[cfg(target_arch = "aarch64")]
fn set_ftz(on: bool) {
    // SAFETY: reads and writes FPCR of this thread only (bit 24 is FZ); no memory is touched.
    unsafe {
        let old: u64;
        std::arch::asm!("mrs {0}, fpcr", out(reg) old, options(nomem, nostack));
        let new = if on { old | (1 << 24) } else { old & !(1 << 24) };
        std::arch::asm!("msr fpcr, {0}", in(reg) new, options(nomem, nostack));
    }
}

/// Set flush-to-zero and denormals-are-zero in MXCSR.
#[cfg(target_arch = "x86_64")]
fn set_ftz(on: bool) {
    // SAFETY: reads and writes MXCSR of this thread only (FTZ is bit 15, DAZ bit 6).
    unsafe {
        let mut old: u32 = 0;
        std::arch::asm!("stmxcsr [{0}]", in(reg) &mut old, options(nostack));
        let new = if on { old | (1 << 15) | (1 << 6) } else { old & !((1 << 15) | (1 << 6)) };
        std::arch::asm!("ldmxcsr [{0}]", in(reg) &new, options(nostack, readonly));
    }
}

#[test]
fn a_flush_to_zero_host_fails_the_run_and_the_binary_exits_4() {
    let mut m = machine("ref");
    set_ftz(true);
    let r = m.run(10, &mut NoObserver);
    let cfg = format!("{}/../../../opentsp-arch-spec/config/mini.json", env!("CARGO_MANIFEST_DIR"));
    let args = [OsString::from(cfg)];
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = main_with(&args, &mut "S 1 1 7\nD 1 1\n".as_bytes(), &mut out, &mut err);
    set_ftz(false);
    match r {
        Err(RunError::Internal(msg)) => assert!(msg.contains("floating-point"), "{msg}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(code, 4);
    let err = String::from_utf8(err).unwrap();
    assert!(err.starts_with("tsp-iss: host floating-point"), "{err}");
    // restored: the run works again
    assert!(machine("ref").run(10, &mut NoObserver).is_ok());
}
