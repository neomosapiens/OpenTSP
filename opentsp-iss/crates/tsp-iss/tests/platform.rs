//! The platform interface of `platform.md` §1–§7, through the public API of `tsp_iss::platform`:
//! every register and flow of §6, the error FIFO with its overflow flag and the sticky first error,
//! and whole programs loaded through the IQ windows and the SRAM aperture, run, and compared with
//! `Machine::run`, the golden vectors and `tsp_sim`.
mod common;
use common::*;
use ots_format::weights::TensorIn;
use ots_format::{bind_weights, read_weights, write_weights, Extern};
use ots_profile::Profile;
use serde_json::Value;
use tsp_iss::platform::*;
use tsp_iss::{Geom, LoadError, Machine, NoLinks, NoObserver};

const READ_WRITE: [(u16, &[u32]); 2] = [(4, &[0x14, 0x1000_0001, 0]), (0, &[0x1a, 0x1040_0003, 0])];

fn profile(name: &str) -> Profile {
    Profile::load(name).unwrap()
}

fn platform(name: &str) -> Platform {
    Platform::new(&profile(name), 0).unwrap()
}

fn load(p: &mut Platform, q: u16, units: &[u32]) {
    p.select_iq(q).unwrap();
    p.write_iq_window(units).unwrap();
}

/// A MEM Read of slice 4 address 1 onto E:0 and a MEM Write of slice 0 address 3 from it, with the
/// word of slice 4 preloaded: the word reaches slice 0 and the chip is done at cycle 28.
fn copy_program(p: &mut Platform) {
    for (q, units) in READ_WRITE {
        load(p, q, units);
    }
    p.write_sram_word(4, 1, &[9; 16]).unwrap();
}

fn config_json(name: &str) -> Value {
    serde_json::from_slice(&std::fs::read(config(name)).unwrap()).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn units_of(s: &str) -> Vec<u32> {
    s.split_whitespace().map(|u| u32::from_str_radix(u, 16).unwrap()).collect()
}

fn key_pair(k: &str) -> (u8, u32) {
    let (a, b) = k.split_once(':').unwrap();
    (a.parse().unwrap(), b.parse().unwrap())
}

fn vectors(profile: &str) -> Vec<Value> {
    let path = spec().join("conformance/model").join(format!("{profile}.json"));
    let doc: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    doc["vectors"].as_array().unwrap().clone()
}

fn vector(profile: &str, name: &str) -> Value {
    vectors(profile).into_iter().find(|v| v["name"] == name).unwrap()
}

/// Load a vector's program and data through the windows and the aperture.
fn load_vector(p: &mut Platform, v: &Value) {
    for (q, units) in v["text"].as_object().unwrap() {
        load(p, q.parse().unwrap(), &units_of(units.as_str().unwrap()));
    }
    for (k, w) in v["sram"].as_object().unwrap() {
        let (s, a) = key_pair(k);
        p.write_sram_word(s, a, &unhex(w.as_str().unwrap())).unwrap();
    }
}

/// The vectors that run on a chip with no fabric: those without a C2C unit (the golden vectors
/// loop every link back).
fn has_c2c(v: &Value) -> bool {
    v["units"].as_array().unwrap().iter().any(|u| u == "C2C")
}

/// A vector with a host script drives the host module (`Machine::set_host_script`); the platform windows do not carry it.
fn has_host(v: &Value) -> bool {
    v.get("host").and_then(|h| h.as_array()).is_some_and(|h| !h.is_empty())
}

fn rec(cycle: u64, queue: u16, code: &str) -> (u64, u16, String) {
    (cycle, queue, code.to_string())
}

fn recs(rs: &[Record]) -> Vec<(u64, u16, String)> {
    rs.iter().map(|r| (r.cycle, r.queue, r.code.as_str().to_string())).collect()
}

fn pop_all(p: &mut Platform) -> Vec<Record> {
    std::iter::from_fn(|| p.err_pop()).collect()
}

/// The reports `microarchitecture.md` §8 marks data-dependent, by queue: every IQ-002 (an IQ overflow) and
/// IQ-004 (text left at completion, an end-of-run record), Gather and Scatter (MEM) and `sxselect` (the SEL queues) for
/// OPR-001 (the filter keeps every MEM queue: a documented superset), a C2C receive queue for TIM-001, every SXM-001, and
/// every DEC-003 of text.
fn detected(queue: u16, code: &str) -> bool {
    match code {
        "OTS-E-SXM-001" | "OTS-E-DEC-003" | "OTS-E-IQ-002" | "OTS-E-IQ-004" => true,
        "OTS-E-OPR-001" => queue < 88 || queue == 125 || queue == 139,
        "OTS-E-TIM-001" => [128, 129, 142, 143].contains(&queue),
        _ => false,
    }
}

// ---- identification ----

#[test]
fn identification_registers_report_the_implementation() {
    for name in ["ref", "mini"] {
        let p = platform(name);
        let c = config_json(name);
        assert_eq!(p.id(), PLATFORM_ID);
        assert_eq!(p.spec_version(), 1, "0.1: major 0, minor 1");
        assert_eq!(p.profile_digest(), profile(name).digest12());
        assert_eq!(p.icu_fetch_latency(), c["icu"]["fetch_latency"].as_u64().unwrap());
        assert_eq!(p.icu_fetch_bytes(), c["icu"]["fetch_bytes_per_cycle"].as_u64().unwrap());
        assert_eq!(p.icu_iq_depth(), c["icu"]["iq_depth"].as_u64().unwrap());
    }
    assert_ne!(platform("ref").profile_digest(), platform("mini").profile_digest());
}

#[test]
fn c2c_peers_are_read_write_before_start_and_configuration() {
    let mut p = platform("mini");
    let links = Geom::new(&profile("mini")).links as u8;
    assert_eq!(p.c2c_peer(0), Ok(None));
    p.set_c2c_peer(0, Some(5)).unwrap();
    p.set_c2c_peer(links - 1, Some(63)).unwrap();
    assert_eq!(p.c2c_peer(0), Ok(Some(5)));
    assert_eq!(p.c2c_peer(links - 1), Ok(Some(63)));
    assert_eq!(p.c2c_peer(links), Err(PlatformError::Link(links)));
    assert_eq!(p.set_c2c_peer(links, Some(1)), Err(PlatformError::Link(links)));
    p.set_c2c_peer(0, None).unwrap();
    assert_eq!(p.c2c_peer(0), Ok(None));
    p.set_c2c_peer(0, Some(5)).unwrap();
    p.start().unwrap();
    assert_eq!(p.set_c2c_peer(0, Some(6)), Err(PlatformError::WrongPhase(Phase::Running)));
    assert_eq!(p.c2c_peer(0), Ok(Some(5)));
    p.reset().unwrap();
    assert_eq!(p.c2c_peer(0), Ok(Some(5)), "reset keeps the wiring");
}

#[test]
fn the_peer_decides_whether_a_receive_is_a_detected_error() {
    // the mini vector coverage-548: a Receive of chip 0 on queue 128 with nothing sent
    let v = vector("mini", "coverage-548");
    let links = Geom::new(&profile("mini")).links as u8;
    let run = |peers: bool| {
        let mut p = platform("mini");
        if peers {
            for l in 0..links {
                p.set_c2c_peer(l, Some(0)).unwrap();
            }
        }
        load_vector(&mut p, &v);
        p.start().unwrap();
        p.run_to_done(1000).unwrap();
        p
    };
    // the peer is the sender: the Receive finds no vector (TIM-001), which hardware detects
    let mut p = run(true);
    assert_eq!(recs(p.diagnostics()), [rec(10, 128, "OTS-E-TIM-001")]);
    assert_eq!(recs(&pop_all(&mut p)), [rec(10, 128, "OTS-E-TIM-001")]);
    // no peer: the chip_id check fails (OPR-001 on a receive queue), an object-validity code
    let p = run(false);
    assert_eq!(recs(p.diagnostics()), [rec(10, 128, "OTS-E-OPR-001")]);
    assert_eq!(p.err_count(), 0);
    assert!(!p.status().error);
}

// ---- start, stop, done ----

#[test]
fn start_runs_to_done_and_done_shows_in_status() {
    let mut p = platform("mini");
    copy_program(&mut p);
    assert_eq!(p.phase(), Phase::Reset);
    assert_eq!(p.status(), Status::default());
    assert_eq!(p.cycle(), 0);
    p.start().unwrap();
    assert_eq!(p.status(), Status { running: true, ..Status::default() });
    assert_eq!(p.status().bits(), Status::RUNNING);
    assert_eq!(p.run_to_done(1000), Ok(28));
    assert_eq!(p.phase(), Phase::Done);
    assert_eq!(p.status(), Status { quiescent: true, ..Status::default() });
    assert_eq!(p.status().bits(), Status::QUIESCENT);
    assert_eq!(p.cycle(), 28);
    assert_eq!(p.read_sram_word(0, 3).unwrap()[..16], [9; 16]);
    // a chip that is done starts again only after a reset
    assert_eq!(p.start(), Err(PlatformError::WrongPhase(Phase::Done)));
    assert_eq!(p.advance(1), Err(PlatformError::WrongPhase(Phase::Done)));
}

#[test]
fn an_empty_chip_is_done_after_cycle_zero() {
    let mut p = platform("ref");
    p.start().unwrap();
    assert_eq!(p.run_to_done(1), Ok(1));
    assert_eq!(p.cycle(), 1);
}

#[test]
fn advance_in_pieces_equals_one_run() {
    let mut whole = platform("mini");
    copy_program(&mut whole);
    whole.start().unwrap();
    whole.run_to_done(1000).unwrap();
    for step in [1, 7, 13] {
        let mut p = platform("mini");
        copy_program(&mut p);
        p.start().unwrap();
        let mut at = 0;
        while p.phase() == Phase::Running {
            p.advance(step).unwrap();
            if p.phase() == Phase::Running {
                at += step;
                assert_eq!(p.cycle(), at, "the counter counts every cycle while running");
            }
        }
        assert_eq!(p.cycle(), whole.cycle(), "step {step}");
        assert_eq!(p.read_sram_word(0, 3).unwrap(), whole.read_sram_word(0, 3).unwrap());
    }
}

#[test]
fn run_to_done_past_the_limit_leaves_the_chip_running() {
    let mut p = platform("mini");
    copy_program(&mut p);
    p.start().unwrap();
    assert_eq!(p.run_to_done(10), Err(PlatformError::NotDone { cycle: 10 }));
    assert_eq!(p.phase(), Phase::Running);
    assert_eq!(p.cycle(), 10);
    assert_eq!(p.run_to_done(1000), Ok(28));
}

#[test]
fn stop_freezes_the_chip_in_the_current_cycle() {
    let mut p = platform("mini");
    copy_program(&mut p);
    p.start().unwrap();
    p.advance(22).unwrap();
    p.stop();
    assert_eq!(p.status(), Status::default(), "neither running nor quiescent");
    assert_eq!(p.phase(), Phase::Stopped);
    assert_eq!(p.cycle(), 22);
    // the SRAM, the counter and the error registers stay readable
    assert_eq!(p.read_sram_word(0, 3).unwrap(), vec![0; 16]);
    assert_eq!(p.read_sram_word(4, 1).unwrap()[..16], [9; 16]);
    assert_eq!(p.advance(100), Err(PlatformError::WrongPhase(Phase::Stopped)));
    assert_eq!(p.cycle(), 22);
    assert_eq!(p.start(), Err(PlatformError::WrongPhase(Phase::Stopped)));
    // a stop that finds the chip idle or done changes nothing
    let mut q = platform("mini");
    q.stop();
    assert_eq!(q.phase(), Phase::Reset);
    q.start().unwrap();
    q.run_to_done(10).unwrap();
    q.stop();
    assert_eq!(q.phase(), Phase::Done);
}

#[test]
fn reset_clears_the_sram_the_iqs_the_counter_and_the_status() {
    let mut p = platform("mini");
    copy_program(&mut p);
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    p.set_irq_enable(IrqEnable { quiescent: true, error: true });
    p.select_iq(77).unwrap();
    p.reset().unwrap();
    assert_eq!(p.phase(), Phase::Reset);
    assert_eq!(p.status(), Status::default());
    assert_eq!(p.cycle(), 0);
    assert_eq!(p.selected_iq(), 0);
    assert_eq!(p.sram_addr(), SramAddr::default());
    assert_eq!(p.read_sram_word(4, 1).unwrap(), vec![0; 16]);
    assert_eq!(p.read_sram_word(0, 3).unwrap(), vec![0; 16]);
    assert_eq!(p.irq_enable(), IrqEnable { quiescent: true, error: true }, "interrupt enables are configuration");
    // every IQ is empty: the chip does nothing and is done after cycle 0
    p.start().unwrap();
    assert_eq!(p.run_to_done(1000), Ok(1));
    assert_eq!(p.read_sram_word(0, 3).unwrap(), vec![0; 16]);
}

// ---- IQ load window ----

#[test]
fn the_window_appends_in_order_and_drops_padding() {
    let (mut a, mut b) = (platform("mini"), platform("mini"));
    for (q, units) in READ_WRITE {
        load(&mut a, q, units);
    }
    // the same images, appended in pieces with padding between
    load(&mut b, 4, &[0x14, 0xffff_ffff]);
    load(&mut b, 0, &[0x1a]);
    load(&mut b, 4, &[0x1000_0001, 0xffff_ffff, 0]);
    load(&mut b, 0, &[0xffff_ffff, 0x1040_0003, 0]);
    for p in [&mut a, &mut b] {
        p.write_sram_word(4, 1, &[9; 16]).unwrap();
        p.start().unwrap();
        assert_eq!(p.run_to_done(1000), Ok(28));
        assert_eq!(p.read_sram_word(0, 3).unwrap()[..16], [9; 16]);
    }
}

#[test]
fn the_window_selects_data_queues_and_the_control_channel_only() {
    let mut p = platform("mini");
    assert_eq!(p.selected_iq(), 0);
    p.select_iq(143).unwrap();
    assert_eq!(p.selected_iq(), 143);
    p.select_iq(0xffff).unwrap();
    assert_eq!(p.selected_iq(), 0xffff);
    assert_eq!(p.select_iq(144), Err(PlatformError::Load(LoadError::Queue(144))));
    assert_eq!(p.select_iq(0x1000), Err(PlatformError::Load(LoadError::Queue(0x1000))));
    assert_eq!(p.selected_iq(), 0xffff, "a refused selection changes nothing");
}

#[test]
fn an_append_beyond_iq_depth_is_refused_whole_and_sets_load_error() {
    let mut p = platform("mini");
    let units = (p.icu_iq_depth() / 4) as usize;
    p.select_iq(3).unwrap();
    p.write_iq_window(&vec![1; units - 1]).unwrap();
    assert!(!p.status().load_error);
    // two units do not fit in the one slot left: neither is taken
    let err = LoadError::ImageTooLong { queue: 3, units: units + 1, limit: units };
    assert_eq!(p.write_iq_window(&[1, 2]), Err(PlatformError::Load(err)));
    assert!(p.status().load_error);
    assert_eq!(p.status().bits(), Status::LOAD_ERROR);
    // padding does not count, and the slot is still free
    p.write_iq_window(&[0xffff_ffff, 1]).unwrap();
    let err = LoadError::ImageTooLong { queue: 3, units: units + 1, limit: units };
    assert_eq!(p.write_iq_window(&[1]), Err(PlatformError::Load(err)));
    // the bit is not a stop: the host reads it, the chip still starts
    p.start().unwrap();
    assert!(p.status().running && p.status().load_error);
    // only a reset clears it
    p.reset().unwrap();
    assert!(!p.status().load_error);
    p.select_iq(3).unwrap();
    p.write_iq_window(&vec![1; units]).unwrap();
}

#[test]
fn the_windows_are_written_before_start_only() {
    let mut p = platform("mini");
    p.start().unwrap();
    assert_eq!(p.write_iq_window(&[1]), Err(PlatformError::WrongPhase(Phase::Running)));
    p.run_to_done(100).unwrap();
    assert_eq!(p.write_iq_window(&[1]), Err(PlatformError::WrongPhase(Phase::Done)));
    p.reset().unwrap();
    p.write_iq_window(&[1]).unwrap();
}

// ---- SRAM aperture ----

#[test]
fn the_aperture_moves_one_tile_of_one_word() {
    let mut p = platform("ref");
    let a = SramAddr { slice: 7, word: 200, superlane: 2 };
    p.set_sram_addr(a).unwrap();
    assert_eq!(p.sram_addr(), a);
    let tile: [u8; 16] = std::array::from_fn(|i| 0xa0 + i as u8);
    p.write_sram_data(&tile).unwrap();
    assert_eq!(p.read_sram_data(), Ok(tile));
    // only that tile of the word changed
    let word = p.read_sram_word(7, 200).unwrap();
    assert_eq!(word.len(), 320);
    assert_eq!(&word[32..48], &tile);
    assert!(word[..32].iter().chain(&word[48..]).all(|b| *b == 0));
    // the neighbouring tile is another access
    p.set_sram_addr(SramAddr { superlane: 3, ..a }).unwrap();
    assert_eq!(p.read_sram_data(), Ok([0; 16]));
    p.write_sram_data(&[1; 16]).unwrap();
    p.set_sram_addr(a).unwrap();
    assert_eq!(p.read_sram_data(), Ok(tile));
}

#[test]
fn the_aperture_refuses_addresses_the_chip_does_not_have() {
    let mut p = platform("ref");
    let wps = Geom::new(&profile("ref")).words_per_slice as u32;
    let ok = SramAddr { slice: 87, word: wps - 1, superlane: 19 };
    p.set_sram_addr(ok).unwrap();
    for (bad, err) in [
        (SramAddr { slice: 88, ..ok }, PlatformError::Load(LoadError::SliceRange(88))),
        (SramAddr { word: wps, ..ok }, PlatformError::Load(LoadError::Address(u64::from(wps)))),
        (SramAddr { superlane: 20, ..ok }, PlatformError::Superlane(20)),
    ] {
        assert_eq!(p.set_sram_addr(bad), Err(err));
        assert_eq!(p.sram_addr(), ok, "a refused address changes nothing");
    }
    let sparse = Profile::load_file("mini_small", config("test/mini_small")).unwrap();
    let mut q = Platform::new(&sparse, 0).unwrap();
    let absent = SramAddr { slice: 8, word: 0, superlane: 0 };
    assert_eq!(q.set_sram_addr(absent), Err(PlatformError::Load(LoadError::SliceAbsent(8))));
    q.set_sram_addr(SramAddr { slice: 7, ..absent }).unwrap();
}

#[test]
fn a_burst_is_a_sequence_of_tile_accesses() {
    let mut p = platform("ref");
    let bytes: Vec<u8> = (1..=40).collect();
    p.write_sram_word(3, 9, &bytes).unwrap();
    let word = p.read_sram_word(3, 9).unwrap();
    assert_eq!(word[..40], bytes[..]);
    assert!(word[40..].iter().all(|b| *b == 0), "the last tile is padded with zeros");
    // tiles beyond the burst keep their bytes
    p.set_sram_addr(SramAddr { slice: 3, word: 9, superlane: 3 }).unwrap();
    p.write_sram_data(&[7; 16]).unwrap();
    p.write_sram_word(3, 9, &[5; 16]).unwrap();
    let word = p.read_sram_word(3, 9).unwrap();
    assert_eq!(word[..16], [5; 16]);
    assert_eq!(word[16..40], bytes[16..]);
    assert_eq!(word[48..64], [7; 16]);
    assert_eq!(
        p.write_sram_word(3, 9, &[0; 321]),
        Err(PlatformError::Load(LoadError::TooManyBytes { bytes: 321, lanes: 320 }))
    );
}

#[test]
fn aperture_writes_equal_the_s_lines_of_the_harness() {
    for name in ["ref", "mini"] {
        let prof = profile(name);
        let g = Geom::new(&prof);
        let vl = g.vlanes as usize;
        let mut p = Platform::new(&prof, 0).unwrap();
        let mut m = Machine::new(&prof, 0, Box::new(NoLinks)).unwrap();
        let mut x = 0x2545_f491_4f6c_dd1du64;
        for i in 0..60u32 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let slice = [0u8, 1, 43, 44, 87, 20][(x >> 60) as usize % 6];
            let word = (i * 3 + (x >> 20) as u32 % 3) % g.words_per_slice as u32; // a fresh word each time
            let len = [vl, vl, vl / 2, 16, 1, 0, 33][i as usize % 7].min(vl);
            let bytes: Vec<u8> = (0..len).map(|k| (x >> (k % 8 * 8)) as u8 ^ k as u8).collect();
            m.preload(slice, word, &bytes).unwrap();
            p.write_sram_word(slice, word, &bytes).unwrap();
            assert_eq!(p.read_sram_word(slice, word).unwrap(), m.sram_word(slice, word).unwrap()[..vl], "{name} {i}");
        }
    }
}

#[test]
fn the_aperture_is_refused_while_the_chip_runs() {
    let mut p = platform("mini");
    copy_program(&mut p);
    p.start().unwrap();
    assert_eq!(p.read_sram_data(), Err(PlatformError::WrongPhase(Phase::Running)));
    assert_eq!(p.write_sram_data(&[1; 16]), Err(PlatformError::WrongPhase(Phase::Running)));
    assert_eq!(p.write_sram_word(0, 0, &[1]), Err(PlatformError::WrongPhase(Phase::Running)));
    assert_eq!(p.read_sram_word(0, 0), Err(PlatformError::WrongPhase(Phase::Running)));
    // the address register may still be written; after done and after stop the aperture is open
    p.set_sram_addr(SramAddr { slice: 0, word: 3, superlane: 0 }).unwrap();
    p.run_to_done(1000).unwrap();
    assert_eq!(p.read_sram_data(), Ok([9; 16]));
    p.write_sram_data(&[3; 16]).unwrap();
    assert_eq!(p.read_sram_data(), Ok([3; 16]));
}

#[test]
fn weights_are_loaded_like_any_data_through_the_aperture() {
    // an `.safetensors` package bound to the extern `src` (slice 4, address 1): the program copies it
    let prof = profile("mini");
    let externs = [Extern { name: "src".into(), slice: 4, addr: 1, size: 1 }];
    let tensors = [TensorIn { name: "src".into(), dtype: None, words: vec![vec![9; 16]] }];
    let package = read_weights(&write_weights(&prof, &tensors).unwrap(), Some("mini"), None).unwrap();
    let mut p = platform("mini");
    for (q, units) in READ_WRITE {
        load(&mut p, q, units);
    }
    for w in bind_weights(&externs, &package).unwrap() {
        p.write_sram_word(w.slice as u8, w.addr as u32, w.word).unwrap();
    }
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    assert_eq!(p.read_sram_word(0, 3).unwrap(), vec![9; 16]);
}

// ---- diagnostics ----

/// A MEM queue whose units are all `tag 7`: every unit reports DEC-003, one per cycle from cycle 10.
fn undecodable(n: usize) -> Vec<u32> {
    let mut units = vec![10];
    units.extend(vec![0x7000_0000; n]);
    units
}

#[test]
fn detected_reports_enter_the_fifo_in_order_and_the_first_is_sticky() {
    // mini coverage-487: eight DEC-003 of one queue
    let v = vector("mini", "coverage-487");
    let mut p = platform("mini");
    load_vector(&mut p, &v);
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    let all: Vec<_> = v["expect"]["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| rec(d["cycle"].as_u64().unwrap(), d["queue"].as_u64().unwrap() as u16, d["code"].as_str().unwrap()))
        .collect();
    assert_eq!(all.len(), 8);
    assert_eq!(recs(p.diagnostics()), all);
    assert_eq!(p.err_count(), 8);
    assert!(p.status().error && !p.status().fifo_overflow);
    assert_eq!(p.status().bits(), Status::QUIESCENT | Status::ERROR);
    let first = p.err_first().unwrap();
    assert_eq!((first.cycle, first.queue, first.code.as_str()), (15, 0, "OTS-E-DEC-003"));
    // the host reads until the count is 0; the oldest record comes first
    assert_eq!(recs(&pop_all(&mut p)), all);
    assert_eq!(p.err_count(), 0);
    assert_eq!(p.err_pop(), None);
    // the sticky register and the status keep the error
    assert_eq!(p.err_first(), Some(first));
    assert!(p.status().error);
    // only a reset clears them
    p.reset().unwrap();
    assert_eq!(p.err_first(), None);
    assert!(!p.status().error);
    assert!(p.diagnostics().is_empty());
}

#[test]
fn a_full_fifo_of_sixteen_does_not_overflow() {
    let mut p = platform("mini");
    load(&mut p, 0, &undecodable(ERR_FIFO_DEPTH));
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    assert_eq!(ERR_FIFO_DEPTH, 16);
    assert_eq!(p.diagnostics().len(), 16);
    assert_eq!(p.err_count(), 16);
    assert!(!p.status().fifo_overflow);
}

#[test]
fn the_seventeenth_record_sets_overflow_and_is_dropped_and_not_counted() {
    let mut p = platform("mini");
    load(&mut p, 0, &undecodable(20));
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    // the simulator's list keeps all twenty; the FIFO holds the first sixteen
    let all = p.diagnostics().to_vec();
    assert_eq!(all.len(), 20);
    assert_eq!(p.err_count(), 16, "the dropped records are not counted");
    assert!(p.status().fifo_overflow && p.status().error);
    assert_eq!(p.status().bits(), Status::QUIESCENT | Status::ERROR | Status::FIFO_OVERFLOW);
    assert_eq!(p.err_first(), Some(all[0]), "the first error survives the overflow");
    let popped = pop_all(&mut p);
    assert_eq!(popped, all[..16]);
    assert_eq!(p.err_count(), 0);
    // draining does not forget the loss, nor the first error
    assert!(p.status().fifo_overflow);
    assert_eq!(p.err_first(), Some(all[0]));
    // a reset clears the flag
    p.reset().unwrap();
    assert!(!p.status().fifo_overflow);
    assert_eq!(p.err_first(), None);
    assert_eq!(p.err_count(), 0);
}

#[test]
fn records_after_a_pop_fill_the_freed_space() {
    let mut p = platform("mini");
    load(&mut p, 0, &undecodable(20));
    p.start().unwrap();
    p.advance(10 + 17).unwrap(); // the records of cycles 10..=26 (the 17th lands at 26)
    assert_eq!(p.err_count(), 16);
    assert!(p.status().fifo_overflow);
    let oldest = p.err_pop().unwrap();
    assert_eq!(oldest.cycle, 10);
    p.run_to_done(1000).unwrap();
    // the dropped record is gone; the later ones find room
    // the record of cycle 27 took the freed place, the records of cycles 28 and 29 were dropped
    assert_eq!(p.err_count(), 16);
    let rest = pop_all(&mut p);
    assert_eq!((rest[0].cycle, rest[14].cycle, rest[15].cycle), (11, 25, 27));
    assert_eq!(p.err_first().unwrap(), oldest);
}

#[test]
fn only_the_data_dependent_codes_reach_the_fifo() {
    // mini directed-002 raises RES-001 (an object-validity code): in the full list, not in the FIFO
    let v = vector("mini", "directed-002");
    let mut p = platform("mini");
    load_vector(&mut p, &v);
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    assert_eq!(recs(p.diagnostics()), [rec(10, 88, "OTS-E-RES-001")]);
    assert_eq!(p.err_count(), 0);
    assert_eq!(p.err_first(), None);
    assert!(!p.status().error);
    assert!(!p.irq_pending());
}

/// Run the vector `name` of `profile` on a platform and return it done.
fn run_vector(profile_name: &str, name: &str) -> Platform {
    let mut p = platform(profile_name);
    load_vector(&mut p, &vector(profile_name, name));
    p.start().unwrap();
    p.run_to_done(1 << 20).unwrap();
    p
}

#[test]
fn an_iq_overflow_iq002_reaches_the_fifo() {
    // ref generated-088: two IQ-002 on queue 7, then other reports (QUE-001, TIM-001, DEC-003)
    let mut p = run_vector("ref", "generated-088");
    let all = recs(p.diagnostics());
    let det: Vec<_> = all.iter().filter(|r| detected(r.1, &r.2)).cloned().collect();
    assert_eq!(det[..2], [rec(14, 7, "OTS-E-IQ-002"), rec(15, 7, "OTS-E-IQ-002")]);
    assert_eq!(p.err_count(), det.len());
    assert!(p.status().error);
    assert_eq!(p.err_first().map(|r| (r.cycle, r.code.as_str())), Some((14, "OTS-E-IQ-002")));
    assert_eq!(recs(&pop_all(&mut p)), det);
}

#[test]
fn text_left_at_completion_iq004_reaches_the_fifo_as_an_end_of_run_record() {
    // ref generated-100: an IQ-004 on queue 46 at cycle 216, raised by the completion check after the run-time reports
    let mut p = run_vector("ref", "generated-100");
    let all = recs(p.diagnostics());
    assert_eq!(all.last(), Some(&rec(216, 46, "OTS-E-IQ-004")));
    let det: Vec<_> = all.iter().filter(|r| detected(r.1, &r.2)).cloned().collect();
    assert_eq!(det.last(), Some(&rec(216, 46, "OTS-E-IQ-004")), "the IQ-004 is the last detected report");
    assert_eq!(recs(&pop_all(&mut p)), det);
    assert!(p.err_first().is_some());
    // a truncated parcel on queue 0 (bundle bit set, no second unit) and a DEC-003 at cycle 49 on queue 1: the IQ-004 record
    // carries the stamp of the queue's last activity and follows the run-time record, so the FIFO is not monotonic in
    // cycle (`microarchitecture.md` §8: an end-of-run record)
    let mut p = platform("mini");
    load(&mut p, 0, &[0x9000_0010, 0]);
    load(&mut p, 1, &[0x0000_0031, 0x6fff_ffff]);
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    let want = [rec(49, 1, "OTS-E-DEC-003"), rec(0, 0, "OTS-E-IQ-004")];
    assert_eq!(recs(p.diagnostics()), want);
    assert_eq!(recs(&pop_all(&mut p)), want, "the FIFO holds the records in reporting order, the IQ-004 last");
}

#[test]
fn error_and_done_raise_the_interrupts_that_are_enabled() {
    let v = vector("mini", "coverage-487");
    let mut p = platform("mini");
    load_vector(&mut p, &v);
    assert_eq!(p.irq_enable(), IrqEnable::default());
    p.set_irq_enable(IrqEnable { quiescent: true, error: false });
    p.start().unwrap();
    assert!(!p.irq_pending());
    p.advance(16).unwrap();
    assert!(p.status().error && !p.irq_pending(), "an error does not raise the done interrupt");
    p.set_irq_enable(IrqEnable { quiescent: false, error: true });
    assert!(p.irq_pending());
    p.set_irq_enable(IrqEnable { quiescent: false, error: false });
    assert!(!p.irq_pending());
    p.set_irq_enable(IrqEnable { quiescent: true, error: false });
    p.run_to_done(1000).unwrap();
    assert!(p.irq_pending());
    p.set_irq_enable(IrqEnable::default());
    assert!(!p.irq_pending());
}

// ---- whole programs ----

#[test]
fn every_golden_vector_runs_through_the_platform_as_through_the_machine() {
    for name in ["ref", "mini"] {
        let prof = profile(name);
        let g = Geom::new(&prof);
        let vl = g.vlanes as usize;
        let mut ran = 0;
        for v in vectors(name).iter().filter(|v| !has_c2c(v) && !has_host(v)) {
            let id = format!("{name}/{}", v["name"].as_str().unwrap());
            let mut p = Platform::new(&prof, 0).unwrap();
            load_vector(&mut p, v);
            p.start().unwrap();
            let cycles = p.run_to_done(1 << 20).unwrap();
            assert_eq!(p.phase(), Phase::Done, "{id}");

            // the golden diagnostics, in order, and the golden words
            let want: Vec<_> = v["expect"]["diagnostics"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| {
                    rec(d["cycle"].as_u64().unwrap(), d["queue"].as_u64().unwrap() as u16, d["code"].as_str().unwrap())
                })
                .collect();
            assert_eq!(recs(p.diagnostics()), want, "{id}: diagnostics");
            for (k, w) in v["expect"]["words"].as_object().unwrap() {
                let (s, a) = key_pair(k);
                assert_eq!(hex(&p.read_sram_word(s, a).unwrap()), w.as_str().unwrap(), "{id}: word {k}");
            }

            // the FIFO: the detected reports, the first sixteen, with the overflow flag
            let det: Vec<_> = want.iter().filter(|r| detected(r.1, &r.2)).cloned().collect();
            assert_eq!(p.status().fifo_overflow, det.len() > ERR_FIFO_DEPTH, "{id}");
            assert_eq!(
                p.err_first().map(|r| (r.cycle, r.queue, r.code.as_str().to_string())),
                det.first().cloned(),
                "{id}"
            );
            assert_eq!(recs(&pop_all(&mut p)), det[..det.len().min(ERR_FIFO_DEPTH)], "{id}: fifo");

            // start to done is Machine::run
            let mut m = Machine::new(&prof, 0, Box::new(NoLinks)).unwrap();
            for (q, units) in v["text"].as_object().unwrap() {
                m.load_image(q.parse().unwrap(), &units_of(units.as_str().unwrap())).unwrap();
            }
            for (k, w) in v["sram"].as_object().unwrap() {
                let (s, a) = key_pair(k);
                m.preload(s, a, &unhex(w.as_str().unwrap())).unwrap();
            }
            let sum = m.run(1 << 20, &mut NoObserver).unwrap();
            assert_eq!(cycles, sum.last_cycle.unwrap() + 1, "{id}: cycles");
            for (k, _) in v["expect"]["words"].as_object().unwrap() {
                let (s, a) = key_pair(k);
                assert_eq!(p.read_sram_word(s, a).unwrap(), m.sram_word(s, a).unwrap()[..vl], "{id}: {k}");
            }
            ran += 1;
        }
        assert!(ran >= 400, "{name}: only {ran} vectors ran");
    }
}

/// The harness input of a vector, as `tsp_sim` reads it, and the `W` lines it must print.
fn harness_text(v: &Value) -> String {
    let mut s = String::new();
    for (k, w) in v["sram"].as_object().unwrap() {
        let (sl, a) = key_pair(k);
        s += &format!("S {sl:x} {a:x}");
        for b in unhex(w.as_str().unwrap()) {
            s += &format!(" {b:02x}");
        }
        s += "\n";
    }
    for (q, units) in v["text"].as_object().unwrap() {
        s += &format!("Q {:x} {}\n", q.parse::<u16>().unwrap(), units.as_str().unwrap());
    }
    for k in v["expect"]["words"].as_object().unwrap().keys() {
        let (sl, a) = key_pair(k);
        s += &format!("D {sl:x} {a:x}\n");
    }
    s
}

#[test]
fn a_program_through_the_windows_equals_tsp_sim() {
    for name in ["ref", "mini"] {
        let prof = profile(name);
        let cfg = config(name);
        let mut compared = 0;
        for v in vectors(name)
            .iter()
            .filter(|v| !has_c2c(v) && !v["expect"]["words"].as_object().unwrap().is_empty())
            .step_by(25)
        {
            let id = format!("{name}/{}", v["name"].as_str().unwrap());
            let sim = run_sim(&[cfg.to_str().unwrap()], &harness_text(v));
            assert_eq!(sim.status.code(), Some(0), "{id}");
            let out = text(&sim.stdout);
            let mut p = Platform::new(&prof, 0).unwrap();
            load_vector(&mut p, v);
            p.start().unwrap();
            p.run_to_done(1 << 20).unwrap();
            let mut want = Vec::new();
            for k in v["expect"]["words"].as_object().unwrap().keys() {
                let (s, a) = key_pair(k);
                let mut line = format!("W {s:x} {a:x}");
                let word = p.read_sram_word(s, a).unwrap();
                for b in word.iter().copied().chain(std::iter::repeat(0)).take(320) {
                    line += &format!(" {b:02x}");
                }
                want.push(line);
            }
            let got: Vec<&str> = out.lines().filter(|l| l.starts_with("W ")).collect();
            assert_eq!(
                got,
                want.iter().map(String::as_str).collect::<Vec<_>>(),
                "{id}: words of tsp_sim vs the platform"
            );
            let errors = out
                .lines()
                .find(|l| l.starts_with("E "))
                .map(|l| l.split(' ').nth(3).unwrap().parse::<usize>().unwrap());
            assert_eq!(errors, Some(p.diagnostics().len()), "{id}: the E line");
            compared += 1;
        }
        assert!(compared >= 8, "{name}: only {compared} programs compared");
    }
}

// model: core/queues.sail diag_report: the platform FIFO mirrors the chip's diagnostic log (core/csr.sail diag_record).
#[test]
fn the_fifo_mirrors_the_chip_diagnostic_log() {
    let word = |r: &Record| -> u64 {
        let q = if r.queue == 0xffff { 144 } else { r.queue as u64 };
        q << 56 | tsp_iss::driver::diag_number(r.code).unwrap() << 48 | r.cycle
    };
    let mut p = platform("mini");
    load(&mut p, 0, &undecodable(20));
    p.start().unwrap();
    p.run_to_done(1000).unwrap();
    let (log, overflow) = p.chip_diag_log();
    assert_eq!(log.len(), ERR_FIFO_DEPTH, "the log holds the first sixteen records");
    assert_eq!(log.len(), p.err_count());
    assert!(overflow && p.status().fifo_overflow, "the overflow flag of the log is the FIFO's");
    // pop the FIFO: each record is the next entry of the log, and the pop is the chip's DIAG_POP
    let mut i = 0;
    while let Some(r) = p.err_pop() {
        assert_eq!(log[i], word(&r), "record {i}");
        i += 1;
        assert_eq!(p.chip_diag_log().0, log[i..], "the chip log lost its oldest entry too");
    }
    assert_eq!(i, ERR_FIFO_DEPTH);
    assert!(p.chip_diag_log().0.is_empty());
}
