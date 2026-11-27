//! The dispatch reads an instruction's kind from the instruction itself, not from the catalogue row. Each
//! of those readings equals the catalogue's for every entry.
use ots_isa::{EntryId, Insn, ENTRIES, MAX_VARS};
use tsp_iss::driver::classify::{is_agt_config, ref_path, ref_thread};

fn insns() -> Vec<Insn> {
    let mut v = Vec::new();
    for e in ENTRIES.iter() {
        for x in [0u32, 1, 2, 3] {
            let mut f = [0u32; MAX_VARS];
            for (i, var) in e.vars.iter().enumerate() {
                f[i] = if var.name == "thread" { x } else { 0 };
            }
            v.push(Insn::from_fields(e.id, &f));
        }
    }
    v
}

#[test]
fn agt_config_is_the_catalogues_agt_config() {
    for i in insns() {
        assert_eq!(is_agt_config(&i), i.entry().agt_config.is_some(), "{:?}", i);
    }
}

#[test]
fn reference_thread_and_path_are_the_catalogues() {
    for i in insns() {
        let is_ref = matches!(i.id(), EntryId::Read | EntryId::Write | EntryId::Gather | EntryId::Scatter);
        let thread = if is_ref { i.get("thread").unwrap_or(0) as u8 & 1 } else { 0 };
        assert_eq!(ref_thread(&i), thread, "{:?}", i);
        let path = match i.id() {
            EntryId::Read | EntryId::Gather => Some(0),
            EntryId::Write | EntryId::Scatter => Some(1),
            _ => None,
        };
        assert_eq!(ref_path(&i), path, "{:?}", i);
    }
}
