//! The profile-dependent half of the instruction set and the small public helpers: stations,
//! absent MEM slices, the group rule outside MEM queues, operand widths, the encoding digest and
//! the diagnostic text.
use ots_isa::{
    encoding_digest12, entry, is_queue, placement, Access, AsmError, Code, EntryId, Isa, CONTROL, DATA_QUEUES,
    ENCODING_JSON, ENTRIES,
};
use ots_profile::Profile;
use sha2::{Digest, Sha256};

fn ref_isa() -> Isa {
    Isa::new(&Profile::load("ref").unwrap())
}

/// `ref` with `section.key` set to `v`, or removed when `v` is `None`.
fn with(key: &str, v: Option<u64>) -> Profile {
    let mut c = Profile::load("ref").unwrap().value().clone();
    let (sec, k) = key.split_once('.').unwrap();
    match v {
        Some(v) => c[sec][k] = v.into(),
        None => drop(c[sec].as_object_mut().unwrap().remove(k)),
    }
    Profile::from_bytes("ref", c.to_string().as_bytes()).unwrap()
}

#[test]
fn diagnostics_render_as_tsp_as_prints_them() {
    let e = AsmError::new(Code::Que002, "not on this queue");
    assert_eq!(e.to_string(), format!("{}: not on this queue", Code::Que002.as_str()));
    assert_eq!(e.render("asm"), format!("asm: {}: not on this queue", Code::Que002.as_str()));
    let at = e.clone().at("p.ots:3");
    assert_eq!(at.render("asm"), format!("p.ots:3: {}: not on this queue", Code::Que002.as_str()));
    assert_ne!(at, e);
}

#[test]
fn the_encoding_digest_is_that_of_the_shipped_encoding_json() {
    let path = format!("{}/../../../opentsp-arch-spec/isa-db/generated/encoding.json", env!("CARGO_MANIFEST_DIR"));
    let disk = std::fs::read(&path).unwrap();
    assert_eq!(ENCODING_JSON, &disk[..], "the built-in copy differs from {path}");
    let d: [u8; 32] = Sha256::digest(&disk).into();
    assert_eq!(encoding_digest12()[..], d[..12]);
}

#[test]
fn entry_ids_index_the_catalogue() {
    for (i, e) in ENTRIES.iter().enumerate() {
        assert_eq!(e.id.index(), i);
        assert!(std::ptr::eq(entry(e.id), e) && std::ptr::eq(e.id.entry(), e));
    }
    assert_eq!(entry(EntryId::Sync).name, EntryId::Sync.entry().name);
}

#[test]
fn queues_are_the_data_queues_and_the_control_channel() {
    assert!(is_queue(0) && is_queue(DATA_QUEUES as u32 - 1) && is_queue(CONTROL as u32));
    assert!(!is_queue(DATA_QUEUES as u32) && !is_queue(CONTROL as u32 - 1) && !is_queue(u32::MAX));
}

#[test]
fn stations_follow_the_layout_keys_of_the_profile() {
    let isa = ref_isa();
    // ref: mem_down_gap_base 24, mem_up_gap_base 18, mxm_in_gap 43, mxm_acc_gap 2
    assert_eq!(isa.station("MEM_E", Access::Read, true, 2), Some(26));
    assert_eq!(isa.station("MEM_E", Access::Write, false, 2), Some(16));
    assert_eq!(isa.station("MEM_W", Access::Read, true, 3), Some(15));
    assert_eq!(isa.station("MXM_E", Access::Read, true, 0), Some(43));
    assert_eq!(isa.station("MXM_E", Access::Write, false, 0), Some(2));
    // no station on that row, no such unit, or the gap leaves 0..43
    assert_eq!(isa.station("MXM_E", Access::Read, false, 0), None);
    assert_eq!(isa.station("CU", Access::Write, true, 0), None);
    assert_eq!(isa.station("NOPE", Access::Read, true, 0), None);
    assert_eq!(isa.station("MEM_E", Access::Read, true, 20), None);
    assert_eq!(isa.station("MEM_W", Access::Read, true, 19), None);
    // a profile without the key has no station
    let isa = Isa::new(&with("layout.mxm_in_gap", None));
    assert_eq!(isa.station("MXM_E", Access::Read, true, 0), None);
    assert_eq!(isa.agt_levels(), 4);
}

#[test]
fn queues_of_absent_mem_slices_refuse_every_entry() {
    let isa = ref_isa();
    assert_eq!(isa.mem_slices(), 44);
    let small = Isa::new(&with("geometry.mem_slices_per_hemisphere", Some(40)));
    assert_eq!(small.mem_slices(), 40);
    let mem = ENTRIES
        .iter()
        .find(|e| placement(e.id, 0).is_none() && placement(e.id, 44).is_none())
        .expect("an entry of the MEM queues");
    for q in [0u16, 39, 44, 83] {
        assert_eq!(small.placement(mem.id, q), None, "{} on {q}", mem.name);
    }
    for q in [40u16, 43, 84, 87] {
        assert_eq!(isa.placement(mem.id, q), placement(mem.id, q), "{} on {q}", mem.name);
        assert_eq!(small.placement(mem.id, q), Some(Code::Que002), "{} on {q}", mem.name);
    }
    // the control channel and the queues above the MEM slices are not slices
    assert_eq!(small.placement(EntryId::Sync, CONTROL), placement(EntryId::Sync, CONTROL));
    assert_eq!(small.placement(mem.id, 100), placement(mem.id, 100));
}

#[test]
fn a_group_outside_the_mem_queues_is_que_001_on_every_record() {
    let isa = ref_isa();
    let sync = isa.decode(32, 0x0040_0000);
    assert!(sync.is_some());
    assert_eq!(isa.group_codes(CONTROL, &[]), vec![]);
    assert_eq!(isa.group_codes(CONTROL, &[sync]), vec![None]);
    assert_eq!(isa.group_codes(CONTROL, &[None]), vec![None]);
    assert_eq!(isa.group_codes(CONTROL, &[sync, sync]), vec![Some(Code::Que001); 2]);
    assert_eq!(isa.group_codes(200, &[sync, sync, sync]), vec![Some(Code::Que001); 3]);
}

#[test]
fn a_field_wider_than_its_operand_is_illegal() {
    let isa = ref_isa();
    let mut n = 0;
    for e in ENTRIES.iter() {
        for v in e.vars.iter().filter(|v| v.nbits < 32) {
            assert!(!isa.field_legal(v, 1 << v.nbits), "{}.{}", e.name, v.name);
            n += 1;
        }
    }
    assert!(n > 0);
}
