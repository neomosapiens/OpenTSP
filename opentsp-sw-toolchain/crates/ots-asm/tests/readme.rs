//! The first `asm` block of every README of the repository assembles (a README once showed an `.extern` syntax
//! that `ots-asm asm` refused with OTS-E-ASM-001).
use ots_asm::assemble;
use std::path::Path;

fn first_asm_block(md: &str) -> Option<String> {
    let mut lines = md.lines();
    lines.by_ref().find(|l| l.trim_end() == "```asm")?;
    Some(lines.take_while(|l| !l.starts_with("```")).map(|l| format!("{l}\n")).collect())
}

#[test]
fn first_asm_block_of_each_readme_assembles() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let mut checked = 0;
    for rel in [
        "README.md",
        "opentsp-sw-toolchain/README.md",
        "opentsp-iss/README.md",
        "opentsp-arch-spec/README.md",
        "opentsp-vllm/README.md",
    ] {
        let Ok(md) = std::fs::read_to_string(root.join(rel)) else { continue };
        let Some(block) = first_asm_block(&md) else { continue };
        let r = assemble(block.as_bytes(), rel, None, &mut |_| {});
        assert!(
            r.is_ok(),
            "{rel}: the first asm block does not assemble: {:?}\n{block}",
            r.err().map(|e| e.render("asm"))
        );
        checked += 1;
    }
    assert!(checked >= 1, "no README has an asm block any more: update this test");
}

#[test]
fn block_extraction_takes_the_first_block_only() {
    let md = "x\n```asm\n.arch a\n```\ntext\n```asm\n.arch b\n```\n";
    assert_eq!(first_asm_block(md).as_deref(), Some(".arch a\n"));
    assert_eq!(first_asm_block("no block"), None);
}
