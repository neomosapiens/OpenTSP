//! Tests of `harness/output`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::Code;
use tsp_iss::geom::MAX_LANES;
use tsp_iss::harness::output::*;
use tsp_iss::observe::{Observer, Row};

#[test]
fn trace_lines_and_summary() {
    let mut buf = Vec::new();
    {
        let mut o = TraceObserver::new(&mut buf, true);
        o.lint = true;
        o.srf(7, Row::West, 3, 12, &[]);
        o.sram(8, 87, 4095, &[]);
        o.lint(9, 0xffff, "OTS-W-IQ-001");
        o.diag(9, 0xffff, Code::Que001);
        o.diag(10, 5, Code::Que002);
        o.srf(11, Row::East, 0, 0, &[1, 0xab]);
        let mut w = [0u8; MAX_LANES];
        w[1] = 0xab;
        o.finish(&[(0x57, 0xfff, w)]);
        assert!(o.error.is_none());
    }
    let s = String::from_utf8(buf).unwrap();
    let mut lines = s.lines();
    assert_eq!(lines.next(), Some("T srf 7 W:3 gap 12"));
    assert_eq!(lines.next(), Some("T sram 8 slice 87 addr 4095"));
    assert_eq!(lines.next(), Some("T lint 9 queue 65535 OTS-W-IQ-001"));
    assert_eq!(lines.next(), Some(format!("T diag 9 queue 65535 {}", Code::Que001.as_str()).as_str()));
    assert_eq!(lines.next(), Some(format!("T diag 10 queue 5 {}", Code::Que002.as_str()).as_str()));
    assert_eq!(lines.next(), Some("T srf 11 E:0 gap 0 01ab"));
    assert_eq!(lines.next(), Some(format!("E {} 9 2", Code::Que001.as_str()).as_str()));
    let w = lines.next().unwrap();
    assert!(w.starts_with("W 57 fff 00 ab 00"), "{w}");
    assert_eq!(w.split(' ').count(), 3 + 320);
    assert_eq!(lines.next(), None);
}

#[test]
fn without_trace_only_the_summary() {
    let mut buf = Vec::new();
    let mut o = TraceObserver::new(&mut buf, false);
    o.srf(1, Row::East, 0, 0, &[]);
    o.diag(4, 2, Code::Que002);
    o.finish(&[]);
    drop(o);
    assert_eq!(String::from_utf8(buf).unwrap(), format!("E {} 4 1\n", Code::Que002.as_str()));
    let mut buf = Vec::new();
    TraceObserver::new(&mut buf, true).finish(&[]);
    assert_eq!(buf, b"E - -1 0\n");
}

/// Actual diagnostic cycles retain their full unsigned timestamp; only absence prints -1.
#[test]
fn the_e_line_prints_the_complete_unsigned_first_cycle() {
    for (cycle, shown) in [
        (4u64, "4"),
        ((1 << 63) - 1, "9223372036854775807"),
        (1 << 63, "9223372036854775808"),
        (u64::MAX - 5, "18446744073709551610"),
        (u64::MAX, "18446744073709551615"),
    ] {
        let mut buf = Vec::new();
        let mut o = TraceObserver::new(&mut buf, true);
        o.diag(cycle, 2, Code::Que002);
        o.finish(&[]);
        drop(o);
        let out = String::from_utf8(buf).unwrap();
        assert!(out.contains(&format!("T diag {cycle} queue 2 ")), "{out}");
        assert!(out.ends_with(&format!("E {} {shown} 1\n", Code::Que002.as_str())), "{out}");
    }
}

#[test]
fn lint_lines_without_trace_and_not_counted() {
    let mut buf = Vec::new();
    let mut o = TraceObserver::new(&mut buf, false);
    o.lint(3, 4, "OTS-W-IQ-001");
    o.lint = true;
    o.lint(5, 6, "OTS-W-IQ-001");
    o.finish(&[]);
    drop(o);
    assert_eq!(String::from_utf8(buf).unwrap(), "T lint 5 queue 6 OTS-W-IQ-001\nE - -1 0\n");
}

#[test]
fn fixed_width_hex_preserves_every_byte_and_dump_width() {
    for vlanes in [16, 320, MAX_LANES] {
        let mut bytes = [0u8; MAX_LANES];
        for (index, byte) in bytes[..vlanes].iter_mut().enumerate() {
            *byte = index as u8;
        }
        let mut out = Vec::new();
        let mut observer = TraceObserver::traced(&mut out);
        observer.srf(42, Row::East, 3, 9, &bytes[..vlanes]);
        observer.finish_lanes(&[(87, 15, bytes)], vlanes);

        let mut expected = String::from("T srf 42 E:3 gap 9 ");
        for byte in &bytes[..vlanes] {
            expected.push_str(&format!("{byte:02x}"));
        }
        expected.push_str("\nE - -1 0\nW 57 f");
        for byte in &bytes[..vlanes.max(320)] {
            expected.push_str(&format!(" {byte:02x}"));
        }
        expected.push('\n');
        assert_eq!(out, expected.as_bytes());
    }
}
