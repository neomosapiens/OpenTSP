//! The text layer: `lower` (cycle view -> per-queue text), `raise_images` (its inverse),
//! `q_lines`, `parse_images`, `is_text_input`. Golden cases come from `tools/tsp_as.py`
//! (`opentsp-sw-toolchain/tests/gen_text_fixtures.py`); the hand-written cases are the corners.
use ots_format::text::{
    drop_padding, input_text, is_nop_word, is_text_input, layout_width, lower, parse_images, q_lines, raise_images,
    word_units, Images, LowerError, Raised,
};
use ots_isa::{Code, BUNDLE_BIT, CONTROL, NARROW_TAGS, PADDING, UNDECODABLE};

fn images(list: &[(u16, &[u32])]) -> Images {
    let mut im = Images::new();
    for (q, us) in list {
        im.extend(*q, us);
    }
    im
}

fn units_of(im: &Images, q: u16) -> Vec<u32> {
    im.get(q).map(|u| u.to_vec()).unwrap_or_default()
}

#[test]
fn nop_gaps_above_65535_split_into_units() {
    // sync at cycle 70000: g = 70000 NOP cycles = 65535 + 4465 (0x1171)
    let im = lower(&[(70000, 1, 0x0040_0000)]).unwrap();
    assert_eq!(units_of(&im, 1), vec![0xFFFF, 0x1171, 0x0040_0000]);
    // g = 65535 exactly: one unit; the next gap 65536: 65535 then 1
    let im = lower(&[(65535, 1, 0x0040_0000), (65535 + 65536 + 1, 1, 0x0080_0000)]).unwrap();
    assert_eq!(units_of(&im, 1), vec![0xFFFF, 0x0040_0000, 0xFFFF, 0x0000_0001, 0x0080_0000]);
}

#[test]
fn a_lone_nop_makes_no_parcel() {
    assert!(lower(&[(0, 0, 5)]).unwrap().is_empty());
    assert!(lower(&[]).unwrap().is_empty());
    // the gaps already carry its time: a sync after a lone NOP keeps its own cycle
    let im = lower(&[(2, 0, 5), (4, 0, 0x0040_0000)]).unwrap();
    assert_eq!(units_of(&im, 0), vec![4, 0x0040_0000]);
    // a NOP together with another entry stays in the parcel (a slot-rule violation)
    let im = lower(&[(3, 0, 5), (3, 0, 0x0040_0000)]).unwrap();
    assert_eq!(units_of(&im, 0), vec![3, 0x8000_0005, 0x0040_0000]);
}

#[test]
fn the_bundle_bit_is_set_on_every_instruction_of_a_parcel_but_the_last() {
    let im = lower(&[(0, 5, 0x0040_0000), (0, 5, 0x0080_0000), (0, 5, 0x0040_0000)]).unwrap();
    assert_eq!(units_of(&im, 5), vec![0x8040_0000, 0x8080_0000, 0x0040_0000]);
    // a wide word: the bit goes on its first unit
    let im = lower(&[(0, 5, 0x1234_5678_9abc_def0), (0, 5, 0x0040_0000)]).unwrap();
    assert_eq!(units_of(&im, 5), vec![0x9234_5678, 0x9abc_def0, 0x0040_0000]);
    assert_eq!(BUNDLE_BIT, 0x8000_0000);
}

#[test]
fn a_word_that_fits_no_layout_lowers_to_the_undecodable_unit() {
    assert_eq!(UNDECODABLE, 0x6FFF_FFFF);
    assert_eq!(word_units(0x6FFF_FFFF0), vec![UNDECODABLE]);
    assert_eq!(word_units(0x0_8000_0000), vec![UNDECODABLE], "bundle bit in the word");
    assert_eq!(word_units(0x1_0000_0000), vec![UNDECODABLE], "tag 0 in the high word");
    assert_eq!(word_units(0x6000_0000_0000_0000), vec![UNDECODABLE]);
    assert_eq!(word_units(0x7000_0000), vec![UNDECODABLE]);
    assert_eq!(word_units(0x0040_0000), vec![0x0040_0000]);
    assert_eq!(word_units(0x5000_0000), vec![0x5000_0000]);
    assert_eq!(word_units(0x1000_0000_0000_0001), vec![0x1000_0000, 1]);
    assert_eq!(word_units(0x4000_0000_0000_0001), vec![0x4000_0000, 1]);
}

#[test]
fn queues_are_in_ascending_order_and_the_control_channel_is_last() {
    let im =
        lower(&[(0, CONTROL, 0x0040_0000), (2, 3, 0x0080_0000), (0, 3, 0x0040_0000), (0, 143, 0x0040_0000)]).unwrap();
    let qs: Vec<u16> = im.iter().map(|(q, _)| q).collect();
    assert_eq!(qs, vec![3, 143, CONTROL]);
    assert_eq!(units_of(&im, 3), vec![0x0040_0000, 1, 0x0080_0000]);
}

#[test]
fn a_queue_that_is_not_a_queue_is_refused() {
    assert!(lower(&[(0, 144, 0x0040_0000)]).is_err());
    assert!(lower(&[(0, 0xFFFE, 0x0040_0000)]).is_err());
    let e = lower(&[(0, 3, 0x0040_0000), (1, 0x90, 0x0040_0000)]).unwrap_err();
    assert_eq!(e, LowerError::BadQueue(0x90));
    assert_eq!(e.to_string(), "queue 0x90 is neither 0..143 nor 0xffff");
}

#[test]
fn the_layout_width_is_32_only_for_a_narrow_tag_with_bits_63_32_clear() {
    for n in 0..16u64 {
        let want = if NARROW_TAGS.contains(&(n as u8)) { 32 } else { 64 };
        assert_eq!(layout_width(n << 28 | 0x0040_0000), want, "tag nibble {n:x}");
        assert_eq!(layout_width(1 << 32 | n << 28), 64, "tag nibble {n:x} with bit 32");
    }
    assert!(NARROW_TAGS.contains(&0));
}

#[test]
fn raise_inverts_lower_for_text_that_never_waits() {
    let entries =
        [(0u64, 3u16, 0x0040_0000u64), (0, 3, 0x0080_0000), (4, 3, 0x1234_5678_9abc_def0), (9, CONTROL, 0x0040_0000)];
    let im = lower(&entries).unwrap();
    let r = raise_images(&im);
    let want: Vec<Raised> = vec![
        Raised { cycle: 0, queue: 3, word: 0x0040_0000, units: 1 },
        Raised { cycle: 0, queue: 3, word: 0x0080_0000, units: 1 },
        Raised { cycle: 4, queue: 3, word: 0x1234_5678_9abc_def0, units: 2 },
        Raised { cycle: 9, queue: CONTROL, word: 0x0040_0000, units: 1 },
    ];
    assert_eq!(r, want);
}

#[test]
fn raise_skips_padding_and_counts_nops() {
    let im = images(&[(2, &[PADDING, 3, 0x0040_0000, PADDING, 2, 0x0080_0000][..])]);
    let r = raise_images(&im);
    assert_eq!(r.iter().map(|r| (r.cycle, r.word)).collect::<Vec<_>>(), vec![(3, 0x0040_0000), (6, 0x0080_0000)]);
    assert_eq!(drop_padding(&im).get(2).unwrap(), &[3, 0x0040_0000, 2, 0x0080_0000]);
    // a trailing parcel whose last instruction has the bundle bit set is still listed
    let im = images(&[(1, &[0x8040_0000][..])]);
    assert_eq!(raise_images(&im), vec![Raised { cycle: 0, queue: 1, word: 0x0040_0000, units: 1 }]);
    // NOPs after the last parcel are dropped
    let im = images(&[(1, &[0x0040_0000, 5][..])]);
    assert_eq!(raise_images(&im).len(), 1);
    // a truncated wide instruction keeps its two-unit width
    let im = images(&[(1, &[0x1000_0001][..])]);
    assert_eq!(raise_images(&im), vec![Raised { cycle: 0, queue: 1, word: 0x1000_0001, units: 2 }]);
}

#[test]
fn q_lines_are_at_most_64_units_ascending_with_the_control_channel_last() {
    let units: Vec<u32> = (0..130).map(|i| 0x0000_0001 + i).collect();
    let im = images(&[(CONTROL, &[0x0040_0000][..]), (7, &units[..])]);
    let lines = q_lines(&im, 64);
    assert_eq!(lines.len(), 4);
    assert!(lines[0].starts_with("Q 7 00000001 00000002 "));
    assert_eq!(lines[0].split(' ').count(), 2 + 64);
    assert_eq!(lines[2].split(' ').count(), 2 + 2);
    assert_eq!(lines[3], "Q ffff 00400000");
    assert_eq!(input_text(&im), lines.iter().map(|l| format!("{l}\n")).collect::<String>());
}

#[test]
fn parse_images_reads_q_lines_and_refuses_the_rest() {
    let ok = parse_images("Q 3 00400000 1\n\nQ 3 80000005\n  Q ffff 400000  \nQ 8f\n", "in.txt").unwrap();
    assert_eq!(units_of(&ok, 3), vec![0x0040_0000, 1, 0x8000_0005]);
    assert_eq!(units_of(&ok, CONTROL), vec![0x0040_0000]);
    assert!(ok.get(0x8f).is_none(), "a queue with no unit is dropped");
    let code = |t: &str| parse_images(t, "in.txt").unwrap_err().code;
    for bad in ["X 3 1", "Q", "Q 90 1", "Q 144 1", "Q 3 zz", "Q 3 100000000", "Q fffe 1", "Q g 1", "q 3 1", "Q 3 0x1"] {
        assert_eq!(code(bad), Code::Asm003, "{bad:?}");
    }
    let e = parse_images("Q 3 1\nQ 3 zz\n", "in.txt").unwrap_err();
    assert!(e.msg.starts_with("in.txt:2:"), "{}", e.msg);
    // leading zeros: a 32-bit value may be written with more digits, a queue likewise
    assert!(parse_images("Q 0003 000000001", "t").is_ok());
    assert!(parse_images("Q 0000ffff 1", "t").is_ok());
    // ... but a queue of more than four significant digits is not one, even when it truncates to one
    assert_eq!(code("Q 10003 1"), Code::Asm003);
    assert_eq!(code("Q 1000000000000003 1"), Code::Asm003);
    // ASCII whitespace other than the space also separates tokens (Python str.split)
    assert!(parse_images("Q\t3\x0b1", "t").is_ok());
}

#[test]
fn text_input_is_recognised_by_its_first_non_blank_byte() {
    assert!(is_text_input(b""));
    assert!(is_text_input(b"  \n\t\r Q 0 1"));
    assert!(!is_text_input(b"OTSO"));
    assert!(!is_text_input(b"\0Q"));
    assert!(!is_text_input(b"\x0bQ"));
}

#[test]
fn nop_words() {
    assert!(is_nop_word(1) && is_nop_word(0xFFFF));
    assert!(!is_nop_word(0) && !is_nop_word(0x1_0000));
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap()
}

#[test]
fn golden_cases_of_tsp_as() {
    let path = format!("{}/tests/data/lower.txt", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap();
    type Case = (Vec<(u64, u16, u64)>, Vec<String>, Vec<Raised>);
    let mut cases: Vec<Case> = Vec::new();
    for l in text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let t: Vec<&str> = l.split(' ').collect();
        match t[0] {
            "C" => cases.push((vec![], vec![], vec![])),
            "E" => cases.last_mut().unwrap().0.push((t[1].parse().unwrap(), hex(t[2]) as u16, hex(t[3]))),
            "Q" => cases.last_mut().unwrap().1.push(l.to_string()),
            "R" => cases.last_mut().unwrap().2.push(Raised {
                cycle: t[1].parse().unwrap(),
                queue: hex(t[2]) as u16,
                word: hex(t[3]),
                units: t[4].parse().unwrap(),
            }),
            other => panic!("{other}"),
        }
    }
    assert!(cases.len() > 250);
    for (i, (entries, qlines, raised)) in cases.iter().enumerate() {
        let im = lower(entries).unwrap();
        assert_eq!(&q_lines(&im, 64), qlines, "case {i}: lower of {entries:x?}");
        assert_eq!(&raise_images(&im), raised, "case {i}: raise_images");
        // the text form round-trips through parse_images
        assert_eq!(parse_images(&input_text(&im), "x").unwrap(), im, "case {i}: parse_images(input_text)");
    }
}
