use crate::links::NoLinks;
use crate::machine::{Machine, RunError};
use crate::observe::NoObserver;
use ots_profile::Profile;

#[test]
fn finishing_a_run_with_missing_text_history_fails_closed() {
    let profile = Profile::load("mini").unwrap();
    let mut machine: Machine = Machine::new(&profile, 0, Box::new(NoLinks)).unwrap();
    machine.rp.text_watch = Some(vec![(0, 64, 1)]);
    machine.sram.journal_begin();
    machine.sram.journal_end();

    let result = machine.run(16, &mut NoObserver);
    assert!(
        matches!(result, Err(RunError::Internal(ref why)) if why.contains("write history was lost")),
        "an active text watch must reject missing history: {result:?}"
    );
}
