//! The operands that hold an SRAM address, a segment bound or an address stride follow `sram.addr_bits` (13 on `ref`, 14 to 16 on the test
//! profiles `w14`, `w15`, `w16`): the largest address, SetSegment and Step of each profile assemble, one past them is refused, and the words
//! disassemble to the text. The expected words and messages are those of `opentsp-arch-spec/tools/tsp_as.py`.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn ots_asm(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output().unwrap()
}

fn config(name: &str) -> String {
    let p: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../opentsp-arch-spec/config").join(name);
    p.to_str().unwrap().to_string()
}

fn insn(cfg: &str, text: &str) -> (Option<i32>, String) {
    let o = ots_asm(&["insn", "--config", &config(cfg), text]);
    let out = if o.status.success() { o.stdout } else { o.stderr };
    (o.status.code(), String::from_utf8_lossy(&out).trim_end().to_string())
}

#[test]
fn the_largest_operands_of_each_width_assemble() {
    for (cfg, text, word) in [
        ("ref.json", "read 0x1fff, E:0, 0x0", "64 10001fff00000000"),
        ("ref.json", "ifetchm 0x1fff", "32 1c01fff"),
        ("ref.json", "setsegment 0x2000", "64 1200000020000000"),
        ("ref.json", "step 0x1fff", "64 118000001fff0000"),
        ("ref.json", "step 0xe001", "64 11800000e0010000"),
        ("test/w14.json", "read 0x3fff, E:0, 0x0", "64 10003fff00000000"),
        ("test/w14.json", "step 0x3fff", "64 118000003fff0000"),
        ("test/w15.json", "write 0x7fff, W:3, 0x1", "64 10607fff8c000000"),
        ("test/w16.json", "read 0xffff, E:0, 0x0", "64 1000ffff00000000"),
        ("test/w16.json", "ifetchm 0xffff", "32 1c0ffff"),
        ("test/w16.json", "setsegment 0x10000", "64 1200000100000000"),
        ("test/w16.json", "step 0x7fff", "64 118000007fff0000"),
        ("test/w16.json", "step 0x8001", "64 1180000080010000"),
    ] {
        assert_eq!(insn(cfg, text), (Some(0), word.to_string()), "{cfg} {text}");
    }
}

#[test]
fn one_past_the_largest_operand_is_refused_with_the_bound() {
    for (cfg, text, msg) in [
        (
            "ref.json",
            "read 0x2000, E:0, 0x0",
            "OTS-E-OPR-001: read operand addr: 0x2000 above sram_words-1 = 8191 (file {cfgpath})",
        ),
        (
            "ref.json",
            "ifetchm 0x2000",
            "OTS-E-OPR-001: ifetchm operand addr: 0x2000 above sram_words-1 = 8191 (file {cfgpath})",
        ),
        (
            "ref.json",
            "setsegment 0x2001",
            "OTS-E-OPR-001: setsegment operand n: 0x2001 above sram_words = 8192 (file {cfgpath})",
        ),
        ("ref.json", "step 0x2000", "OTS-E-OPR-001: step operand a: 8192 out of range -8191..8191"),
        (
            "test/w14.json",
            "read 0x4000, E:0, 0x0",
            "OTS-E-OPR-001: read operand addr: 0x4000 above sram_words-1 = 16383 (file {cfgpath})",
        ),
        ("test/w14.json", "step 0x4000", "OTS-E-OPR-001: step operand a: 16384 out of range -16383..16383"),
        (
            "test/w16.json",
            "read 0x10000, E:0, 0x0",
            "OTS-E-OPR-001: read operand addr: 0x10000 does not fit the 16-bit field",
        ),
        (
            "test/w16.json",
            "setsegment 0x10001",
            "OTS-E-OPR-001: setsegment operand n: 0x10001 above sram_words = 65536 (file {cfgpath})",
        ),
        ("test/w16.json", "step 0x8000", "OTS-E-OPR-001: step operand a: -32768 out of range -32767..32767"),
    ] {
        let (code, out) = insn(cfg, text);
        assert_eq!(code, Some(1), "{cfg} {text}");
        // under --config the message names the file in force, as tsp_as.py does
        let cfgpath = config(cfg);
        assert_eq!(out, format!("insn: {}", msg.replace("{cfgpath}", &cfgpath)), "{cfg} {text}");
    }
}

#[test]
fn the_words_disassemble_on_the_profile_that_holds_them_and_not_on_a_narrower_one() {
    // read 0xffff, E:0, 0x0: the address needs 16 bits
    let word = "1000ffff00000000";
    let o = ots_asm(&["word", "--config", &config("test/w16.json"), "64", word]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim_end(), "read 0xffff, E:0, 0x0");
    let o = ots_asm(&["word", "--config", &config("ref.json"), "64", word]);
    assert!(
        !String::from_utf8_lossy(&o.stdout).starts_with("read"),
        "a 13-bit profile does not decode a 16-bit address"
    );
}
