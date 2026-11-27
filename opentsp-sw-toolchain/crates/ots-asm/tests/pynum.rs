//! The integers of the `word` command line, which `tsp_as.main` reads with Python's `int(text, 10)` (the width)
//! and `int(text, 16)` (the word): surrounding whitespace, a sign, `0x` for base 16, single underscores between
//! digits. A width other than 32 and 64, a negative word and a word wider than its width are OTS-E-ASM-001.
use std::process::{Command, Output};

/// Run the binary; a missing binary (relinked by a concurrent build in the shared target directory) is retried
/// for a few seconds before it is an error.
fn run(args: &[&str]) -> Output {
    let mut waited = 0;
    loop {
        match Command::new(env!("CARGO_BIN_EXE_ots-asm")).args(args).output() {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && waited < 100 => {
                waited += 1;
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            r => return r.unwrap(),
        }
    }
}

fn word(width: &str, hex: &str) -> (i32, String) {
    let o = run(&["word", width, hex]);
    (o.status.code().unwrap(), String::from_utf8_lossy(&o.stdout).into_owned())
}

#[test]
fn the_width_accepts_what_python_int_accepts() {
    // every spelling of 32 and of 64 gives the output of the plain one
    for (width, spellings) in [
        ("32", [" +32 ", " \t\n\r\x0b\x0c32\x0c", "3_2", "032", "+0_32"]),
        ("64", [" +64 ", "6_4", "064", "\x0c64\n", "0_064"]),
    ] {
        let plain = word(width, "40");
        assert_eq!(plain.0, 0);
        for s in spellings {
            assert_eq!(word(s, "40"), plain, "width {s:?}");
        }
    }
    // a width that reads as another number than 32 or 64 is OTS-E-ASM-001, whatever the spelling
    for s in ["-0", "-3", "5", "1_0", "123456789012345678901234567890"] {
        assert_eq!(word(s, "1"), (1, String::new()), "width {s:?}");
    }
}

#[test]
fn the_word_accepts_what_python_int_accepts_and_prints_alternate_hex() {
    // every spelling of a value gives the output of the plain one; a negative value is OTS-E-ASM-001
    for (s, plain) in [
        ("0X1F", "1f"),
        ("0x_1f", "1f"),
        ("0x1_f", "1f"),
        ("1_0", "10"),
        (" +5 ", "5"),
        ("-0", "0"),
        ("00400000", "400000"),
        ("0x400000", "400000"),
    ] {
        let want = word("32", plain);
        assert_eq!(want.0, 0);
        assert_eq!(word("32", s), want, "word {s:?}");
    }
    // the value as Python prints it: the alternate hex of the integer as it was read
    assert_eq!(word("64", "ffffffffffffffff").1, "illegal64 0xffffffffffffffff\n");
    assert_eq!(word("32", "ffffffff").1, "illegal32 0xffffffff\n");
    for s in ["-0x1f", "-1", "-0x1", "ffffffffffffffffffffffffffffff"] {
        assert_eq!(word("64", s), (1, String::new()), "word {s:?}");
    }
}

#[test]
fn what_python_int_refuses_is_asm_001() {
    for s in ["", "  ", "-", "+-1", "_1", "1_", "1__0", "0x1f", "12a", "\x1c5", "1.0", "\u{e9}", "0x"] {
        assert_eq!(word(s, "1"), (1, String::new()), "width {s:?}");
    }
    for s in ["", "  ", "-", "+-1", "_1", "1_", "1__0", "0x", "0x_", "0_x1", "0x1g", "1.0", "\u{e9}", "0x0x1"] {
        assert_eq!(word("16", s), (1, String::new()), "word {s:?}");
    }
}
