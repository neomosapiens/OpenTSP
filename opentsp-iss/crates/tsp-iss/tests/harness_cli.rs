//! Tests of `harness/cli`, moved from the module's in-source test block (they use the crate's public API).
use std::ffi::OsString;
use tsp_iss::harness::cli::*;

fn p(a: &[&str]) -> Result<Options, CliError> {
    parse(&a.iter().map(OsString::from).collect::<Vec<_>>())
}

#[test]
fn defaults() {
    let o = p(&["c.json"]).unwrap();
    assert_eq!((o.trace, o.max_cycles, o.chips), (false, DEFAULT_MAX_CYCLES, 1));
    assert_eq!(o.link_mode(), Ok(LinkMode::Nominal));
    assert_eq!(p(&[]), Err(CliError::Usage));
}

#[test]
fn max_cycles() {
    assert_eq!(p(&["c", "--max-cycles", "0x10"]).unwrap().max_cycles, 16);
    assert_eq!(p(&["c", "--max-cycles", "010"]).unwrap().max_cycles, 8);
    assert_eq!(p(&["c", "--max-cycles", "18446744073709551615"]).unwrap().max_cycles, u64::MAX);
    for bad in ["-1", "+5", " 5", "", "18446744073709551616", "5x", "0x", "08", "1 "] {
        assert_eq!(p(&["c", "--max-cycles", bad]), Err(CliError::Usage), "{bad:?}");
    }
    assert_eq!(p(&["c", "--max-cycles"]), Err(CliError::Usage));
}

#[test]
fn chips() {
    for (a, k) in [
        ("1", 1),
        ("8", 8),
        ("9", 9),
        ("12", 12),
        ("16", 16),
        ("64", 64),
        ("264", 264),
        ("0x2", 2),
        (" 3", 3),
        ("+2", 2),
    ] {
        assert_eq!(p(&["c", "--chips", a]).unwrap().chips, k, "{a:?}");
    }
    for bad in ["0", "265", "-1", "", "x", "2x"] {
        assert!(matches!(p(&["c", "--chips", bad]), Err(CliError::Message(_))), "{bad:?}");
    }
}

#[test]
fn link_latency_map_option() {
    // hidden (not in the usage line): the latency of directed links, for `tsp-iss verify`
    assert_eq!(
        p(&["c", "--link-latency-map", "0:1=209,2:3=228"]).unwrap().link_mode(),
        Ok(LinkMode::Map(vec![(0, 1, 209), (2, 3, 228)]))
    );
    assert_eq!(
        p(&["c", "--link-latency-map", "1:0=216", "--link-latency", "9"]).unwrap().link_mode(),
        Err(CliError::Usage)
    );
    assert_eq!(
        p(&["c", "--link-jitter-seed", "1", "--link-latency-map", "1:0=216"]).unwrap().link_mode(),
        Err(CliError::Usage)
    );
    for bad in ["", "1", "1:0", "1:0=", "a:0=209", "1:0=209,", "-1:0=209", "1:0=209;2:0=209"] {
        assert_eq!(p(&["c", "--link-latency-map", bad]), Err(CliError::Usage), "{bad:?}");
    }
    assert!(!tsp_iss::harness::cli::USAGE.contains("link-latency-map"));
}

#[test]
fn link_options_and_exclusion() {
    assert_eq!(p(&["c", "--link-latency", "9"]).unwrap().link_mode(), Ok(LinkMode::Fixed(9)));
    assert_eq!(p(&["c", "--link-latency-seed", "0x5"]).unwrap().link_mode(), Ok(LinkMode::Seed(5)));
    assert_eq!(p(&["c", "--link-jitter-seed", "0"]).unwrap().link_mode(), Ok(LinkMode::Jitter(0)));
    // an empty string reads as 0 (strtol) where errno stays clear (glibc), and is EINVAL on the BSD libc, as in tsp_sim
    if cfg!(target_vendor = "apple") {
        assert_eq!(p(&["c", "--link-latency-seed", ""]), Err(CliError::Usage));
    } else {
        assert_eq!(p(&["c", "--link-latency-seed", ""]).unwrap().link_mode(), Ok(LinkMode::Seed(0)));
    }
    assert_eq!(p(&["c", "--link-latency", "1", "--link-jitter-seed", "1"]).unwrap().link_mode(), Err(CliError::Usage));
    assert_eq!(p(&["c", "--link-latency", "1", "--link-latency-seed", "1"]).unwrap().link_mode(), Err(CliError::Usage));
    assert_eq!(
        p(&["c", "--link-latency-seed", "1", "--link-jitter-seed", "1"]).unwrap().link_mode(),
        Err(CliError::Usage)
    );
    // the last of a repeated option wins
    assert_eq!(p(&["c", "--link-latency", "1", "--link-latency", "2"]).unwrap().link_mode(), Ok(LinkMode::Fixed(2)));
    for bad in ["-1", "1x", "9223372036854775808"] {
        assert_eq!(p(&["c", "--link-latency", bad]), Err(CliError::Usage), "{bad:?}");
    }
}

#[test]
fn unknown_arguments() {
    assert_eq!(p(&["c", "--bogus"]), Err(CliError::Usage));
    assert_eq!(p(&["c", "extra"]), Err(CliError::Usage));
    assert!(p(&["c", "--trace", "--trace"]).unwrap().trace);
    assert!(p(&["c", "--lint"]).unwrap().lint && !p(&["c"]).unwrap().lint);
}

#[test]
fn subcommands() {
    let a = |s: &str| vec![OsString::from(s)];
    assert!(
        is_subcommand(&a("info")) && is_subcommand(&a("run")) && !is_subcommand(&a("ref.json")) && !is_subcommand(&[])
    );
}
