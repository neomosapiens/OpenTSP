//! Tests of `harness/input`, moved from the module's in-source test block (they use the crate's public API).
use ots_isa::PADDING;
use ots_profile::Profile;
use tsp_iss::geom::Geom;
use tsp_iss::harness::input::*;

fn geom(name: &str) -> Geom {
    Geom::new(&Profile::load(name).unwrap())
}

fn parse_ok(input: &str, chips: u32) -> Program {
    let g = geom("ref");
    parse(input.as_bytes(), Limits { geom: &g, chips, topo: None }).unwrap()
}

fn parse_err(input: &str, chips: u32) -> InputError {
    let g = geom("ref");
    parse(input.as_bytes(), Limits { geom: &g, chips, topo: None }).unwrap_err()
}

#[test]
fn loaders_apply_the_checks_of_the_lines() {
    let g = geom("ref");
    let mut c = ChipInput::default();
    // padding is dropped, several calls append, an empty image leaves no queue
    c.add_image(&g, 8, &[1, PADDING, 2]).unwrap();
    c.add_image(&g, 8, &[3]).unwrap();
    c.add_image(&g, 9, &[PADDING]).unwrap();
    assert_eq!(c.images.get(8), Some(&vec![1, 2, 3]));
    assert_eq!(c.images.get(9), None);
    // the control channel is a queue, 0x90 is not
    c.add_image(&g, 0xffff, &[4]).unwrap();
    assert!(c.add_image(&g, 0x90, &[1]).unwrap_err().contains("out of range"));
    // the IQ holds iq_depth bytes of text, as `Q` lines count them
    let full = vec![7u32; g.iq_units];
    c.add_image(&g, 1, &full).unwrap();
    assert!(c.add_image(&g, 1, &[1]).unwrap_err().contains("icu.iq_depth"));
    c.add_image(&g, 1, &[PADDING]).unwrap();
    // preloads: slice, address and length
    c.add_preload(&g, 3, 5, &[1, 2]).unwrap();
    assert_eq!(c.preloads, vec![Preload { slice: 3, addr: 5, bytes: vec![1, 2] }]);
    assert!(c.add_preload(&g, 88, 0, &[]).unwrap_err().contains("slice out of range"));
    assert!(c.add_preload(&g, 0, g.words_per_slice, &[]).unwrap_err().contains("address out of range"));
    assert!(c
        .add_preload(&g, 0, 0, &[0; 321])
        .unwrap_err()
        .contains("an S line has more bytes than the 320 lanes of this profile"));
    assert_eq!(c.preloads.len(), 1);
}

#[test]
fn grammar() {
    let p = parse_ok("# c\n\n  S 3 5 01 0x02\t03\r\n\tD 3 5\nQ 8f 1 ffffffff 2\nQ 8f 3\nQ ffff 0x4\nS 0 0\n", 1);
    assert_eq!(
        p.chips[0].preloads,
        vec![Preload { slice: 3, addr: 5, bytes: vec![1, 2, 3] }, Preload { slice: 0, addr: 0, bytes: vec![] }]
    );
    assert_eq!(p.dumps, vec![Dump { chip: 0, slice: 3, addr: 5 }]);
    assert_eq!(p.chips[0].images.get(0x8f), Some(&vec![1, 2, 3]));
    assert_eq!(p.chips[0].images.get(0xffff), Some(&vec![4]));
}

#[test]
fn chips_select_where_lines_go() {
    let p = parse_ok("S 0 0 1\nC 2\nS 0 0 2\nD 0 0\nC 0\nD 1 1\n", 3);
    assert_eq!(p.chips[0].preloads.len(), 1);
    assert_eq!(p.chips[2].preloads[0].bytes, vec![2]);
    assert_eq!(p.dumps, vec![Dump { chip: 2, slice: 0, addr: 0 }, Dump { chip: 0, slice: 1, addr: 1 }]);
}

#[test]
fn errors_name_the_line() {
    let cases: &[(&str, usize, &str)] = &[
        ("\n\nQ 90 1\n", 3, "queue out of range"),
        ("Q 10000 1", 1, "queue out of range"),
        ("S 58 0\n", 1, "slice out of range"),
        ("S 0 2000\n", 1, "address out of range"),
        ("S 0 0 100\n", 1, "byte out of range"),
        ("Q 0 100000000\n", 1, "text unit out of range"),
        ("Z 0 0 0 0\n", 1, "unknown line"),
        ("S\n", 1, "unknown line"),
        ("S1 2\n", 1, "unknown line"),
        ("S 1\n", 1, "missing address"),
        ("Q \n", 1, "missing queue"),
        ("C \t\n", 1, "missing chip"),
        ("C 1\n", 1, "chip out of range"),
        ("C 0 0\n", 1, "too many fields"),
        ("D 0 0 0\n", 1, "too many fields"),
        ("S 0 0 +1\n", 1, "expected a hex number"),
        ("S 0 0 -1\n", 1, "expected a hex number"),
        ("S 0 0 zz\n", 1, "expected a hex number"),
        ("S 0 0 0x\n", 1, "expected a hex number"),
        ("S 0 0 1x\n", 1, "expected a hex number"),
        ("S 0 0 10000000000000000\n", 1, "number out of range"),
        ("S 0 0 1\nS 0 0 1\nq 0\n", 3, "unknown line"),
    ];
    for (input, line, msg) in cases {
        let e = parse_err(input, 1);
        assert_eq!((e.line, e.msg.starts_with(msg)), (*line, true), "{input:?}: {e}");
    }
    let long = format!("S 0 0 {}", vec!["1"; 321].join(" "));
    assert_eq!(parse_err(&long, 1).msg, "an S line has more bytes than the 320 lanes of this profile");
}

#[test]
fn missing_slice_is_reported_before_the_range() {
    // `need` reads the slice and the address before either is range-checked
    assert_eq!(parse_err("S 58\n", 1).msg, "missing address");
    assert_eq!(parse_err("D 58 0\n", 1).msg, "slice out of range (0..57 hex)");
}

#[test]
fn image_length_counts_units_without_padding() {
    let full = vec!["0"; 320].join(" ");
    parse_ok(&format!("Q 0 {full}\nQ 0 ffffffff\nQ 1 {full} ffffffff\n"), 1);
    let e = parse_err(&format!("Q 0 {full}\nQ 0 1\n"), 1);
    assert_eq!((e.line, &e.msg[..]), (2, "initial IQ image longer than icu.iq_depth bytes"));
    // separate chips have separate images
    parse_ok(&format!("Q 0 {full}\nC 1\nQ 0 {full}\n"), 2);
}

#[test]
fn absent_slices_are_input_errors() {
    let mut g = geom("ref");
    g.mem_slices = 8;
    let lim = Limits { geom: &g, chips: 1, topo: None };
    assert!(parse(b"S 7 0 1\nS 2c 0 1\n", lim).is_ok());
    let e = parse(b"S 8 0 1\n", lim).unwrap_err();
    assert!(e.msg.starts_with("slice does not exist"), "{e}");
    assert!(parse(b"D 34 0\n", lim).is_err());
}

#[test]
fn nul_ends_a_line_and_a_missing_final_newline_is_fine() {
    let p = parse_ok("S 0 0 1\0 zz\nD 0 0", 1);
    assert_eq!(p.chips[0].preloads[0].bytes, vec![1]);
    assert_eq!(p.dumps.len(), 1);
    assert_eq!(parse_ok("", 1), Program { chips: vec![ChipInput::default()], starts: vec![0], ..Program::default() });
}

#[test]
fn other_white_space_inside_a_number_field() {
    // strtoull skips white space that the field reader does not: a vertical tab before a
    // number is accepted, as in tsp_sim
    let p = parse_ok("S 0 0 \x0b5\n", 1);
    assert_eq!(p.chips[0].preloads[0].bytes, vec![5]);
}
