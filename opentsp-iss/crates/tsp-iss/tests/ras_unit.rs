//! The unit tests of the error CSRs (`model/unit_tests/test_ras.sail`), ported: the counters, the log and its
//! overflow, the first-error copy, the severity, the clearing rules, the error interrupt and the ICU CSR space, on the
//! machine itself, on every profile with a different log depth.
mod common;
use ots_profile::Profile;
use tsp_iss::observe::NoObserver;
use tsp_iss::{Machine, NoLinks};

const PROFILES: [&str; 5] = ["ref", "mini", "test/ref_small", "test/mini_small", "test/mid_min"];

fn each(f: impl Fn(&mut Machine, u64, u64)) {
    for name in PROFILES {
        let p = Profile::from_bytes(name.rsplit('/').next().unwrap(), &std::fs::read(common::config(name)).unwrap())
            .unwrap();
        let depth = p.uint("ras.log_depth").unwrap();
        let fetch = p.uint("icu.fetch_latency").unwrap();
        let mut m = Machine::new(&p, 0, Box::new(NoLinks)).unwrap();
        f(&mut m, depth, fetch);
    }
}

fn w0(c: u64, kind: u64, q: u64) -> u64 {
    c | kind << 48 | q << 52
}

/// One corrected data error of a stream register: queue 88, stream 3, gap 21, superlane 2, position 9.
fn corrected(m: &mut Machine, c: u64) {
    m.ras_ecc_event(c, 88, 0, false, true, 3, 21, 2, 9, &mut NoObserver);
}

// model: unit_tests/test_ras.sail test_ras_record_and_registers
#[test]
fn ras_record_and_registers() {
    each(|m, depth, _| {
        corrected(m, 62);
        assert_eq!(m.csr_read(16), 1);
        assert_eq!(m.csr_read(17), 1);
        assert_eq!((m.csr_read(18), m.csr_read(19), m.csr_read(20)), (1, 0, 1));
        assert_eq!(m.csr_read(32), w0(62, 1, 88));
        assert_eq!(m.csr_read(33), (1 + 3 * 2 + 21 * 64) | (2 << 43) | (9 << 48));
        assert_eq!((m.csr_read(240), m.csr_read(241)), (m.csr_read(32), m.csr_read(33)));
        assert_eq!((m.csr_read(48), m.csr_read(50)), (0, 0));
        m.ras_ecc_event(70, 88, 1, true, false, 5, 9, 4, 77, &mut NoObserver);
        assert_eq!(m.csr_read(16), if depth > 1 { 3 } else { 3 + 512 });
        assert_eq!(m.csr_read(17), 2);
        assert_eq!(m.csr_read(19), 1);
        assert_eq!((m.csr_read(50), m.csr_read(48)), (1, 1));
        if depth > 1 {
            assert_eq!(m.csr_read(34), w0(70, 2, 88));
            assert_eq!(m.csr_read(35), (5 + 9 * 128) | (4 << 43) | (77 << 48));
        }
        assert_eq!(m.csr_read(240), w0(62, 1, 88));
    });
}

// model: unit_tests/test_ras.sail test_ras_text_event_and_control_queue
#[test]
fn ras_text_event_and_control_queue() {
    each(|m, depth, _| {
        m.ras_ecc_event(11, 144, 2, false, false, 0, 5, 0, 3, &mut NoObserver);
        assert_eq!(m.csr_read(32), w0(11, 3, 144));
        assert_eq!(m.csr_read(33), (5 * 256 + 144) | (3 << 48));
        assert_eq!(m.csr_read(16), 4);
        m.ras_ecc_event(12, 3, 2, true, false, 0, 6, 0, 9, &mut NoObserver);
        assert_eq!(m.csr_read(16), if depth > 1 { 12 } else { 12 + 512 });
        assert_eq!(m.csr_read(50), 1);
    });
}

// model: unit_tests/test_ras.sail test_ras_log_overflow_pop_and_clear
#[test]
fn ras_log_overflow_pop_and_clear() {
    each(|m, depth, _| {
        for i in 0..=depth {
            corrected(m, 100 + i);
        }
        assert_eq!(m.csr_read(20), depth);
        assert_eq!(m.csr_read(18), depth + 1);
        assert_eq!(m.csr_read(16), 1 + 512);
        assert_eq!(m.csr_read(32 + 2 * (depth - 1)), w0(100 + depth - 1, 1, 88));
        m.csr_write(21, 0, 200, &mut NoObserver);
        assert_eq!(m.csr_read(20), depth - 1);
        if depth > 1 {
            assert_eq!(m.csr_read(32), w0(101, 1, 88));
        }
        assert_eq!(m.csr_read(32 + 2 * (depth - 1)), 0);
        assert_eq!(m.csr_read(240), w0(100, 1, 88));
        m.csr_write(16, 1, 201, &mut NoObserver);
        assert_eq!(m.csr_read(16), 512);
        m.csr_write(16, 512, 202, &mut NoObserver);
        assert_eq!((m.csr_read(16), m.csr_read(17)), (0, 0));
        assert_eq!(m.csr_read(240), 0);
        m.csr_write(18, 5, 203, &mut NoObserver);
        assert_eq!(m.csr_read(18), 0);
        corrected(m, 300);
        assert_eq!(m.csr_read(240), w0(300, 1, 88));
    });
}

// model: unit_tests/test_ras.sail test_ras_saturation_and_mask
#[test]
fn ras_saturation_and_mask() {
    each(|m, _, _| {
        m.ras_set_sbe_count(0xFFFF_FFFF);
        corrected(m, 5);
        assert_eq!(m.csr_read(18), 0xFFFF_FFFF);
        m.csr_write(22, 1, 6, &mut NoObserver);
        assert_eq!(m.csr_read(22), 1);
        m.csr_write(16, 1, 7, &mut NoObserver);
        corrected(m, 8);
        assert_eq!(m.csr_read(50), 1);
        corrected(m, 9);
        assert_eq!(m.csr_read(50), 1);
        m.csr_write(16, 1, 10, &mut NoObserver);
        corrected(m, 11);
        assert_eq!(m.csr_read(50), 2);
        m.csr_write(49, 0xFFFE, 12, &mut NoObserver);
        m.csr_write(16, 1, 13, &mut NoObserver);
        corrected(m, 14);
        assert_eq!((m.csr_read(50), m.csr_read(48)), (2, 1));
    });
}

// model: unit_tests/test_ras.sail test_ras_matrix_and_link_events
#[test]
fn ras_matrix_and_link_events() {
    each(|m, depth, _| {
        m.ras_mxm_event(200, 105, 3, Some(17), &mut NoObserver);
        assert_eq!(m.csr_read(16), 16);
        assert_eq!((m.csr_read(17), m.csr_read(18), m.csr_read(19)), (1, 1, 0));
        assert_eq!(m.csr_read(32), w0(200, 5, 105));
        assert_eq!(m.csr_read(33), 3 * 65536 + 17);
        assert_eq!((m.csr_read(50), m.csr_read(48)), (0, 0));
        m.ras_mxm_event(210, 105, 1, None, &mut NoObserver);
        assert_eq!(m.csr_read(16), if depth > 1 { 48 } else { 48 + 512 });
        assert_eq!((m.csr_read(17), m.csr_read(19)), (2, 1));
        assert_eq!((m.csr_read(50), m.csr_read(48)), (1, 1));
        if depth > 1 {
            assert_eq!(m.csr_read(34), w0(210, 6, 105));
            assert_eq!(m.csr_read(35), 65536 + 65535);
        }
    });
    each(|m, depth, _| {
        m.ras_link_event(300, 5, 9, false, &mut NoObserver);
        assert_eq!((m.csr_read(16), m.csr_read(17), m.csr_read(18)), (64, 1, 1));
        assert_eq!((m.csr_read(32), m.csr_read(33)), (w0(300, 7, 4095), 5 * 65536 + 9));
        assert_eq!(m.csr_read(50), 0);
        m.ras_link_event(301, 5, 10, true, &mut NoObserver);
        assert_eq!(m.csr_read(16), if depth > 1 { 192 } else { 192 + 512 });
        assert_eq!((m.csr_read(17), m.csr_read(19), m.csr_read(50)), (2, 1, 1));
        if depth > 1 {
            assert_eq!((m.csr_read(34), m.csr_read(35)), (w0(301, 8, 4095), 5 * 65536 + 10));
        }
        assert_eq!(m.csr_read(240), w0(300, 7, 4095));
    });
    each(|m, _, _| {
        m.csr_write(22, 64, 0, &mut NoObserver);
        m.ras_link_event(400, 0, 1, false, &mut NoObserver);
        assert_eq!(m.csr_read(50), 1);
    });
}

// model: unit_tests/test_ras.sail test_ras_fetch_rt_and_unmapped
#[test]
fn ras_fetch_rt_and_unmapped() {
    each(|m, _, fetch| {
        assert_eq!((m.csr_read(256), m.csr_read(256 + 143)), (fetch, fetch));
        assert_eq!(m.csr_read(256 + 144), 0);
        m.csr_write(256, 77, 1, &mut NoObserver);
        assert_eq!(m.csr_read(256), fetch);
        assert_eq!((m.csr_read(23), m.csr_read(24)), (0, 0));
    });
}

#[test]
fn management_fault_survives_mask_ack_and_distinguishes_storage() {
    each(|m, _, _| {
        let o = &mut NoObserver;
        m.csr_write(0x16, 0, 0, o);
        m.ras_ecc_event(4, 0, 1, false, false, 0, 7, 0, 1, o);
        assert_eq!(m.service_ras().fatal_count, 0);
        m.ras_scrub_event(5, 0, true, 7, 0, 3, o);
        m.csr_write(0x10, u64::MAX, 6, o);
        assert_eq!(m.csr_read(0x10), 0);
        assert_eq!(m.service_ras().sram_word, Some((0, 7)));
        m.ras_ecc_event(7, 0, 0, true, false, 0, 7, 0, 1, o);
        assert_eq!(m.service_ras().sram_word, None);
        assert_eq!(m.service_ras().fatal_count, 2);
    });
}

#[test]
fn management_scan_catches_a_stored_report_before_raw_output() {
    each(|m, _, _| {
        m.service_flip_sram(0, 7, 0, 137).unwrap();
        m.service_scrub(5);
        assert_eq!(m.service_ras().fatal_count, 1);
        assert_eq!(m.service_ras().sram_word, None);
        assert_eq!(m.csr_read(0x10), 0);
    });
}

#[test]
fn management_scan_repairs_data_before_raw_read() {
    each(|m, _, _| {
        let original = m.sram_word(0, 7).unwrap();
        m.service_flip_sram(0, 7, 0, 3).unwrap();
        assert_ne!(m.sram_word(0, 7).unwrap(), original);
        m.service_scrub(6);
        assert_eq!(m.sram_word(0, 7).unwrap(), original);
        assert_eq!(m.service_ras().events, 1);
        assert_eq!(m.service_ras().fatal_count, 0);
        m.service_scrub(7);
        assert_eq!(m.service_ras().events, 1);
    });
}

// model: unit_tests/test_ras.sail test_srf_report_eviction
// The array keeps the 256 latest reports; the report that a new register's report overwrites is "live" when it is still in its row at the new report's cycle (register diag + c <= 44): the lint OTS-W-RAS-001.
#[test]
fn spare_report_array_evicts_the_oldest_and_says_when_it_was_live() {
    use tsp_iss::layout::RowId;
    use tsp_iss::srf::Srf;
    let sp = [0u32; 30];
    let mut s = Srf::new(320);
    // 256 reports at cycles 100..355 on gap 20 of row E: distinct diagonals, so 256 registers
    for k in 0..256u64 {
        assert!(!s.report(RowId::E, 20, 100 + k, 1, 1, &sp), "report {k} fits");
    }
    // a report on a register that has one is merged and evicts nothing
    assert!(!s.report(RowId::E, 20, 100, 1, 1, &sp));
    // the 257th register overwrites the slot of the oldest (the one of cycle 100, diagonal 20 - 100): it left its row long before
    // cycle 360 (register 20 - 100 + 360 = 280 > 44), so nothing live was lost
    assert!(!s.report(RowId::E, 20, 360, 1, 1, &sp));
    // a fresh array whose oldest report (cycle 100, gap 20) is still in its row when the 257th register is written at cycle 120
    // (register 20 - 100 + 120 = 40 <= 44): live, and the lint says so; at cycle 125 it has left (45 > 44)
    for (c, live) in [(120u64, true), (125, false)] {
        let mut s = Srf::new(320);
        for k in 0..256u64 {
            // the first report is at cycle 100, the others write other rows' registers (gap 21) so that all 256 stay distinct
            let (g, cyc) = if k == 0 { (20, 100) } else { (21, 100 + k) };
            assert!(!s.report(RowId::E, g, cyc, 1, 1, &sp));
        }
        assert_eq!(s.report(RowId::W, 5, c, 1, 1, &sp), live, "cycle {c}");
    }
}

fn dw(c: u64, code: u64, q: u64) -> u64 {
    c | code << 48 | q << 56
}

// model: unit_tests/test_ras.sail test_diag_log_records_class_data_only
#[test]
fn diag_log_records_class_data_only() {
    use ots_isa::Code;
    each(|m, _, _| {
        let depth = m_depth(m);
        let o = &mut NoObserver;
        // OPR-001 (code 7) on MEM queue 5 is class data; QUE-002 and OPR-001 on a VXM queue are not
        m.diag(o, 5, 10, Code::Opr001);
        m.diag(o, 5, 11, Code::Que002);
        m.diag(o, 88, 12, Code::Opr001);
        assert_eq!(m.csr_read(0x400), 1);
        assert_eq!(m.csr_read(0x410), dw(10, 7, 5));
        // DEC-003 of the control channel (queue field 144), IQ-002, IQ-004, SXM-001, TIM-001 of a C2C receive queue (128)
        m.diag(o, 144, 13, Code::Dec003);
        m.diag(o, 7, 14, Code::Iq002);
        m.diag(o, 144, 15, Code::Iq004);
        m.diag(o, 124, 16, Code::Sxm001);
        m.diag(o, 128, 17, Code::Tim001);
        m.diag(o, 8, 18, Code::Tim001); // a MEM queue: not logged
        assert_eq!(m.csr_read(0x400), depth.min(6));
        if depth >= 2 {
            assert_eq!(m.csr_read(0x411), dw(13, 0, 144));
        }
        if depth >= 5 {
            assert_eq!(m.csr_read(0x414), dw(16, 12, 124));
        }
        if depth >= 6 {
            assert_eq!(m.csr_read(0x415), dw(17, 3, 128));
        }
    });
}

// model: unit_tests/test_ras.sail test_diag_log_overflow_pop_and_clear
#[test]
fn diag_log_overflow_pop_and_clear() {
    use ots_isa::Code;
    each(|m, _, _| {
        let depth = m_depth(m);
        let o = &mut NoObserver;
        for i in 0..=depth {
            m.diag(o, 3, 100 + i, Code::Dec003);
        }
        assert_eq!(m.csr_read(0x400), depth, "the log holds ras.diag_depth entries");
        assert_eq!(m.csr_read(0x402), 1, "the record that found the log full was dropped");
        assert_eq!(m.csr_read(0x410), dw(100, 0, 3));
        assert_eq!(m.csr_read(0x410 + depth), 0, "the address after the last entry reads 0");
        m.csr_write(0x400, 0, 0, o);
        m.csr_write(0x410, 0, 0, o);
        assert_eq!((m.csr_read(0x400), m.csr_read(0x410)), (depth, dw(100, 0, 3)), "read-only");
        m.csr_write(0x401, 1, 0, o);
        assert_eq!(m.csr_read(0x400), depth - 1);
        if depth >= 2 {
            assert_eq!(m.csr_read(0x410), dw(101, 0, 3), "the later entries move down");
        }
        assert_eq!(m.csr_read(0x402), 1, "a pop does not clear the overflow flag");
        m.csr_write(0x402, 0, 0, o);
        assert_eq!(m.csr_read(0x402), 1, "a write of 0 does not clear it");
        m.csr_write(0x402, 1, 0, o);
        assert_eq!(m.csr_read(0x402), 0, "a write of 1 clears the overflow flag");
        for _ in 0..depth {
            m.csr_write(0x401, 1, 0, o);
        }
        assert_eq!((m.csr_read(0x400), m.csr_read(0x410)), (0, 0), "popping an empty log does nothing");
    });
}

/// `ras.diag_depth` of the profile the machine was built for: the entries the log holds.
fn m_depth(m: &Machine) -> u64 {
    m.geom().ras_diag_depth
}
