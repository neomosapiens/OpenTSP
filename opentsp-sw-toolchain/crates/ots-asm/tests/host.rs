//! The four host instructions through the command line: the expected files in `tests/data/`
//! (`host.elf`, `host.q`, `host.dis`) were produced by `opentsp-arch-spec/tools/tsp_as.py` from `host.ots`
//! (`asm --input -o`, `dis`). The placement rules (PciReceive on the C2C RX queues, PciSend on the TX queues,
//! HostWait and HostNotify on any data queue) are the generated tables'; the differential run
//! (`tests/diff_tsp_as.py`) holds the assembler to `tsp_as.py` on every mutation of the corpora.
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR")))
}

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

fn text(o: &Output) -> (i32, String, String) {
    (
        o.status.code().unwrap(),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ots-asm-host-{tag}-{}", std::process::id()));
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_program_of_the_host_instructions_lowers_and_packs_like_tsp_as() {
    let d = tmp("ok");
    let out = d.join("host.elf");
    let src = data("host.ots");
    let (rc, stdout, stderr) = text(&run(&["asm", src.to_str().unwrap(), "--input", "-o", out.to_str().unwrap()]));
    assert_eq!((rc, stderr.as_str()), (0, ""));
    assert_eq!(stdout, fs::read_to_string(data("host.q")).unwrap());
    assert_eq!(fs::read(&out).unwrap(), fs::read(data("host.elf")).unwrap());
    let (rc, listing, err) = text(&run(&["dis", data("host.elf").to_str().unwrap()]));
    assert_eq!((rc, err.as_str()), (0, ""));
    assert_eq!(listing, fs::read_to_string(data("host.dis")).unwrap());
    fs::remove_file(&out).ok();
    fs::remove_dir(&d).ok();
}

#[test]
fn a_host_instruction_on_a_queue_that_cannot_issue_it_is_que_002() {
    let d = tmp("bad");
    let head = ".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=10 {\n";
    for (line, code) in [
        ("VXM[0]: pci_send W:3", "OTS-E-QUE-002"),
        ("VXM[0]: pci_receive W:24", "OTS-E-QUE-002"),
        ("C2C.W.RX[0]: pci_send W:3", "OTS-E-QUE-002"),
        ("C2C.W.TX[0]: pci_receive W:24", "OTS-E-QUE-002"),
        ("CU: host_notify 0x1", "OTS-E-QUE-002"),
        ("VXM[0]: host_notify 0x0", "OTS-E-OPR-001"),
        ("VXM[0]: host_wait 0x40", "OTS-E-OPR-001"),
        ("C2C.W.RX[0]: pci_receive W:23", "OTS-E-OPR-001"),
    ] {
        let p = d.join("bad.ots");
        fs::write(&p, format!("{head}  {line}\n}}\n")).unwrap();
        let (rc, stdout, stderr) = text(&run(&["asm", p.to_str().unwrap()]));
        assert_eq!((rc, stdout.as_str()), (1, ""), "{line}");
        assert!(stderr.starts_with(&format!("{}:5: {code}", p.display())), "{line}: {stderr}");
        fs::remove_file(&p).ok();
    }
    fs::remove_dir(&d).ok();
}
