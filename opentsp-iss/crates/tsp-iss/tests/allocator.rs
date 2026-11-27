//! The `tsp-iss` binary uses mimalloc on macOS (`src/main.rs`): the system allocator costs about 7 % of a run there.
//! The allocator changes no result (every other test and the exactness lines run through it); this checks that it is
//! the one linked, so a build that drops it is noticed. Elsewhere the default allocator stays and nothing is checked.
#[test]
fn the_binary_links_mimalloc_on_macos() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let bin = std::fs::read(env!("CARGO_BIN_EXE_tsp-iss")).unwrap();
    let needle = b"mimalloc: warning";
    assert!(bin.windows(needle.len()).any(|w| w == needle), "the binary has no mimalloc in it");
}
