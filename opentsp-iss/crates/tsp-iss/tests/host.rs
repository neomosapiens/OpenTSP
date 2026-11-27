//! The host path: the CSR space, the host's script (`H` and `R` lines), the host I/O instructions
//! PciReceive, PciSend, HostWait and HostNotify, MSI-X, the start cycle, text injection and the csr packet of the
//! control unit. Every chip has its own host interface (`H` and `R` lines act on the chip of the preceding `C` line).
//! Every case runs the reference `tsp_sim` on the same arguments and input and compares exit code and stdout
//! (stderr with the program name normalised); the cases that name an output line also assert it, so a reference
//! that stopped printing it would not pass them silently.
mod common;
use common::*;
use tsp_iss::geom::MAX_LANES;

fn cfg(name: &str) -> String {
    config(name).to_str().unwrap().to_string()
}

/// Both harnesses on the same arguments and input; the results must agree. Returns our stdout.
fn both(args: &[&str], input: &str) -> String {
    let ours = run_iss(args, input);
    let theirs = run_sim(args, input);
    assert_eq!(
        ours.status.code(),
        theirs.status.code(),
        "exit code, args {args:?}, input {input:?}\nours: {}\ntsp_sim: {}",
        text(&ours.stderr),
        text(&theirs.stderr)
    );
    assert_eq!(text(&ours.stdout), text(&theirs.stdout), "stdout, args {args:?}, input {input:?}");
    assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "stderr, args {args:?}, input {input:?}");
    text(&ours.stdout)
}

// HostWait 3 at cycle 10 on queue 88, HostNotify 5 at 11.
const WAIT_NOTIFY: &str = "Q 58 0000000a 51c00003 52000005\n";

// model: core/csr.sail host_wait, host_ring; slices/c2c/c2c_insts.sail execute C2C_HOSTNOTIFY
#[test]
fn doorbell_releases_hostwait_and_notify_raises_msix() {
    let c = cfg("ref");
    let out = both(&[&c, "--trace"], &format!("{WAIT_NOTIFY}H db 32 3\nR 30\nR 32\n"));
    assert!(out.contains("T irq 51 msix 5\n"), "the notify runs the cycle after the ring (0x32 = 50): {out}");
    assert!(out.contains("X 030 20\n") && out.contains("X 032 1\n"), "cause bit 5, one interrupt: {out}");
    // no ring: the queue parks for good, the run ends cleanly and nothing is notified
    let out = both(&[&c, "--trace"], &format!("{WAIT_NOTIFY}R 30\n"));
    assert!(!out.contains("T irq") && out.contains("X 030 0\n"), "{out}");
    // a ring before the wait is kept until a HostWait takes it
    let out = both(&[&c, "--trace"], &format!("{WAIT_NOTIFY}H db 1 3\nR 300\nR 303\n"));
    assert!(out.contains("T irq 11 msix 5\n"), "the wait finds the doorbell set and does not park: {out}");
}

// model: core/csr.sail host_msix, csr_write
#[test]
fn irq_mask_and_cause_are_registers_of_the_chip() {
    let c = cfg("ref");
    let out = both(&[&c, "--trace"], &format!("{WAIT_NOTIFY}H csr 0 w 31 0\nH db 100 3\nR 30\nR 32\nR 31\n"));
    assert!(!out.contains("T irq"), "vector 5 masked: no interrupt: {out}");
    assert!(out.contains("X 030 20\n") && out.contains("X 032 0\n") && out.contains("X 031 0\n"), "{out}");
    // the cause register clears by writing ones
    let out = both(&[&c, "--trace"], &format!("{WAIT_NOTIFY}H db 32 3\nH csr 69 w 30 20\nR 30\n"));
    assert!(out.contains("X 030 0\n"), "{out}");
    // the CSRs read 0 outside the registers and as written for the doorbells; a read prints a T csr line
    let out = both(&[&c, "--trace"], "R 0\nR 300\nR 123\nH csr 5 r 0\nH csr 7 r 301\n");
    assert!(out.contains("T csr 5 000 0\n") && out.contains("T csr 7 301 0\n"), "{out}");
}

// model: c_emulator/tsp_sim.c host_process (the ingress FIFO); slices/c2c/c2c_insts.sail execute C2C_PCIRECV
#[test]
fn dma_vector_reaches_pcireceive() {
    let c = cfg("ref");
    // PciReceive S24 on RX queue 142 after a NOP; the vector is pushed at cycle 0
    let prog = "Q 8e 64 5140e000\n";
    let out = both(&[&c, "--trace"], &format!("{prog}H dma 0 1 2 3\n"));
    assert!(out.contains("T dma 0 in\n"), "{out}");
    assert!(out.contains("T srf 101 W:24 gap 5 010203"), "the vector lands on the stream: {out}");
    // nothing arrived: TIM-001, as Receive
    let out = both(&[&c, "--trace"], prog);
    assert!(out.contains("OTS-E-TIM-001"), "{out}");
}

fn ingress_depth() -> usize {
    serde_json::from_slice::<serde_json::Value>(&std::fs::read(config("ref")).unwrap()).unwrap()["host"]
        ["ingress_depth"]
        .as_u64()
        .unwrap() as usize
}

// model: c_emulator/tsp_sim.c host_process, host_recv (DMA back pressure); OTS-E-HIO-002 is retired
#[test]
fn a_full_fifo_makes_the_dma_engine_wait_and_nothing_is_dropped() {
    let c = cfg("ref");
    let depth = ingress_depth();
    // host.ingress_depth vectors fit; the next one waits in the DMA engine (host.cycles_per_vector = 10 apart)
    let dmas: String = (0..=depth).map(|i| format!("H dma {:x} {}\n", i * 10, i + 1)).collect();
    let out = both(&[&c, "--trace"], &dmas);
    assert!(out.contains(&format!("T dma {} wait\n", depth * 10)), "{out}");
    assert!(!out.contains(" drop\n") && !out.contains("OTS-E-HIO-002") && out.contains("E - -1 0\n"), "{out}");
    // a PciReceive on RX queue 142 at cycle 200 (after the FIFO filled at 150) frees a slot: the waiting vector enters
    // the cycle after it, and a second PciReceive then takes the vectors in order
    let prog = "Q 8e c8 5140e000\n";
    let out = both(&[&c, "--trace"], &format!("{prog}{dmas}"));
    let wait = format!("T dma {} wait\n", depth * 10);
    let after = out.split(&wait).nth(1).expect("the engine waited");
    assert!(after.contains("T dma 201 in\n"), "the waiting vector enters the cycle after the PciReceive (200): {out}");
}

// model: slices/c2c/c2c_insts.sail execute C2C_PCISEND (HIO-001), host.cycles_per_vector = 10 on ref
#[test]
fn pcisend_accepts_one_vector_per_interval() {
    let c = cfg("ref");
    // three Reads fill streams 0, 1, 2 (queue 44, from vector host-6); PciSend at cycles 100, 105 (inside the
    // interval: HIO-001) and 110 (exactly one interval later: accepted)
    let text =
        "Q 2c 4b 10000001 80000000 4 10000002 84000000 4 10000003 88000000\nQ 7e 64 51808000 4 51808400 4 51808800\n";
    let out = both(&[&c, "--trace"], text);
    assert!(out.contains("T dma 100 out\n") && out.contains("T dma 110 out\n"), "{out}");
    assert!(!out.contains("T dma 105"), "{out}");
    assert_eq!(out.matches("T diag 105 queue 126 OTS-E-HIO-001").count(), 1, "{out}");
}

// model: postlude/harness_api.sail model_set_start, model_inject_unit
#[test]
fn start_cycle_and_injected_text() {
    let c = cfg("ref");
    let out = both(&[&c, "--trace"], &format!("{WAIT_NOTIFY}H start 64\nH db 200 3\n"));
    assert!(out.contains("T irq "), "{out}");
    let out = both(&[&c, "--trace"], "H inject 14 58 52000009\n");
    assert!(out.contains("T irq 20 msix 9\n"), "the injected HostNotify 9 dispatches at its cycle: {out}");
}

// model: c_emulator/tsp_sim.c read_host_line, read_pk_line
#[test]
fn bad_host_lines_and_remote_csr() {
    let c = cfg("ref");
    for (name, input) in [
        ("unknown word", "H frob 1\n"),
        ("no word", "H\n"),
        ("doorbell range", "H db 1 40\n"),
        ("csr range", "H csr 1 r 1000\n"),
        ("csr rw", "H csr 1 x 10\n"),
        ("csr value", "H csr 1 w 10\n"),
        ("too many dma bytes", &format!("H dma 0 {}\n", vec!["1"; 321].join(" "))),
        ("inject queue", "H inject 0 90 1\n"),
        ("R range", "R 1000\n"),
        ("routed csr read is gone", "H csr 5 r 1:0\n"),
        ("routed csr write is gone", "H csr 5 w 1:0 3\n"),
        ("P csr rw", "P csr 1 0 x 10\n"),
        ("P csr value", "P csr 1 0 w 10\n"),
        ("P csr range", "P csr 1 0 r 1000\n"),
        ("missing cycle", "H db\n"),
    ] {
        let args: Vec<&str> = vec![&c, "--chips", "2"];
        let ours = run_iss(&args, input);
        let theirs = run_sim(&args, input);
        assert_eq!(ours.status.code(), theirs.status.code(), "{name}: exit code");
        assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "{name}: message");
        assert_eq!(ours.status.code(), Some(2), "{name}: an input error");
    }
    // R lines of each chip
    both(&[&c, "--trace", "--chips", "2"], "R 0\nC 1\nR 0\nR 300\n");
}

// model: c_emulator/tsp_sim.c read_pk_line, ctl_process, ctl_depart; microarchitecture.md section 10 "Packet types"
#[test]
fn ctrl_packet_waits_for_a_gap_and_is_taken_in_by_the_peer() {
    let c = cfg("ref");
    // vector host-024: a Transmit (a sync packet) at cycle 200 on link 0; P ctrl at 202 must wait for the gap: the link
    // accepted a vector 2 cycles before, so the packet leaves at 224 (200 + c2c.cycles_per_vector) and arrives 216 later
    let prog = "Q 7e c8 50c00000\nQ 8e 1a1 50408000 1a 50408400\n";
    let out = both(&[&c, "--trace"], &format!("{prog}P ctrl ca 0\n"));
    assert!(out.contains("T pkt 440 ctrl 0\n"), "{out}");
    // a ctrl packet never enters the receive FIFO: the Receive finds nothing of it (its first Receive takes the sync one)
    let out = both(&[&c, "--trace"], "P ctrl 0 0\nP ctrl 0 0\nP ctrl 1 0\n");
    assert_eq!(out.matches("T pkt ").count(), 3, "{out}");
    assert!(
        out.contains("T pkt 216 ctrl 0\n") && out.contains("T pkt 218 ctrl 0\n") && out.contains("T pkt 220 ctrl 0\n"),
        "{out}"
    );
    // bad P lines
    for (name, input) in [
        ("packet type", "P frob 1 0\n"),
        ("no type", "P\n"),
        ("missing link", "P ctrl 1\n"),
        ("extra field", "P ctrl 1 0 0\n"),
        ("link out of range", "P ctrl 1 10\n"),
    ] {
        let ours = run_iss(&[&c], input);
        let theirs = run_sim(&[&c], input);
        assert_eq!(ours.status.code(), Some(2), "{name}");
        assert_eq!(ours.status.code(), theirs.status.code(), "{name}");
        assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)), "{name}: message");
    }
    // a link that goes nowhere (two chips: only link 0 is connected)
    let args: Vec<&str> = vec![&c, "--chips", "2"];
    let ours = run_iss(&args, "P ctrl 1 5\n");
    let theirs = run_sim(&args, "P ctrl 1 5\n");
    assert_eq!(ours.status.code(), Some(2));
    assert_eq!(text(&ours.stderr), norm_stderr(&text(&theirs.stderr)));
}

// model: c_emulator/tsp_sim.c run_chips (--window), ctl_process; microarchitecture.md section 14
#[test]
fn the_csr_packet_is_one_hop_and_the_window_changes_nothing() {
    let c = cfg("ref");
    // the control unit of chip 0 reads CHIP_ID of the chip at the far end of link 0, writes its doorbell 0 and reads it
    // back; the host of chip 1 reads the doorbell with `R`
    let script = "P csr 5 0 r 0\nP csr 9 0 w 300 1\nP csr 12c 0 r 300\nP ctrl 3 0\nC 1\nP ctrl 4 0\nR 300\n";
    let base = both(&[&c, "--trace", "--chips", "2"], script);
    // the request leaves chip 0 at its cycle, arrives one link latency later, the reply the same after that
    assert!(base.contains("T csr 437 1 000 1\n") && base.contains("T csr 732 1 300 1\n"), "{base}");
    assert!(base.contains("X 300 1\n"), "the write reached the CSR of chip 1: {base}");
    for w in ["1", "5", "24", "100", "216"] {
        let o = both(&[&c, "--trace", "--chips", "2", "--window", w], script);
        assert_eq!(o, base, "--window {w}");
    }
    // `--window` has no lower bound any more (the rule is causal); 0 is a usage error
    let args: Vec<&str> = vec![&c, "--chips", "2", "--window", "0"];
    let (ours, theirs) = (run_iss(&args, ""), run_sim(&args, ""));
    assert_eq!(ours.status.code(), Some(2));
    assert_eq!(ours.status.code(), theirs.status.code());
    // a csr packet is refused when `host.remote_csr` is off: every shipped profile has it on, so the line is accepted
    let o = both(&[&c, "--trace"], "P csr 5 0 r 0\n");
    assert!(o.contains("T csr 437 0 000 0\n"), "a loopback link answers with CHIP_ID of this chip: {o}");
}

// model: core/csr.sail csr_read; c_emulator/tsp_sim.c host_setup (J-9: the host interface on every chip)
#[test]
fn every_chip_has_its_own_host_interface() {
    let c = cfg("ref");
    // the host of chip 0 feeds a vector, chip 0 sends it to chip 1, the host of chip 1 reads it with PciSend
    let input = std::fs::read_to_string(root().join("opentsp-iss/crates/tsp-iss/tests/data/host_two_chip.in")).unwrap();
    let out = both(&[&c, "--trace", "--chips", "2"], &input);
    let (c0, c1) = out.split_once("C 1\n").expect("two chips");
    assert!(c0.contains("T dma 0 in\n") && !c0.contains("Y "), "{c0}");
    assert!(c1.contains("T dma 576 out\n") && c1.contains("Y 576 66 6d 74 7b"), "{c1}");
    // the other chip's host lines: `C 1` then `H` is no longer an input error, and each chip has its own CSR space
    let out = both(&[&c, "--trace", "--chips", "2"], "R 0\nH csr 5 w 31 1\nC 1\nR 0\nR 31\nH db 7 2\nR 302\n");
    let (c0, c1) = out.split_once("C 1\n").expect("two chips");
    assert!(c0.contains("X 000 0\n") && c1.contains("X 000 1\n"), "CHIP_ID per chip: {out}");
    assert!(c1.contains("X 031 ffff\n") && c1.contains("X 302 1\n"), "chip 1's own mask and doorbell: {out}");
    assert!(!c0.contains("X 302"), "{out}");
}

// model: core/csr.sail csr_write (a write of IRQ_MASK delivers the newly enabled vectors), host_msix, host_deliver
#[test]
fn unmasking_delivers_the_interrupt_that_was_raised_while_masked() {
    let c = cfg("ref");
    // HostNotify 3 at cycle 100; IRQ_MASK is cleared at 0 and restored at 0x78 (120)
    let out = both(&[&c, "--trace"], "Q 58 64 52000003\nH csr 0 w 31 0\nH csr 78 w 31 ffff\nR 30\nR 32\n");
    assert!(out.contains("T irq 120 msix 3\n") && out.matches("T irq").count() == 1, "{out}");
    assert!(out.contains("X 030 8\n") && out.contains("X 032 1\n"), "the cause bit stays; one interrupt: {out}");
    // enabling again delivers again (the cause bit is still set); clearing it first does not
    let out = both(
        &[&c, "--trace"],
        "Q 58 64 52000003\nH csr 0 w 31 0\nH csr 78 w 31 ffff\nH csr 80 w 31 0\nH csr 88 w 31 ffff\nR 32\n",
    );
    assert_eq!(out.matches("T irq").count(), 2, "{out}");
    let out = both(&[&c, "--trace"], "Q 58 64 52000003\nH csr 0 w 31 0\nH csr 78 w 30 8\nH csr 80 w 31 ffff\nR 32\n");
    assert!(!out.contains("T irq") && out.contains("X 032 0\n"), "{out}");
    // a vector that is still masked is not delivered
    let out = both(&[&c, "--trace"], "Q 58 64 52000003\nH csr 0 w 31 0\nH csr 78 w 31 7\nR 32\n");
    assert!(!out.contains("T irq"), "{out}");
}

// model: slices/c2c/c2c_insts.sail Send and Transmit (OTS-E-LNK-003), c_emulator/tsp_sim.c c2c_packet_busy
#[test]
fn a_transmit_released_into_a_packet_in_flight_is_lnk_003() {
    let c = cfg("ref");
    // P ctrl at cycle 0x64 (100) leaves at once and occupies the link for its cycles; a Transmit at 101 is released into it
    let out = both(&[&c, "--trace"], "Q 7e 65 50c00000\nP ctrl 64 0\n");
    assert!(out.contains("OTS-E-LNK-003"), "{out}");
    assert!(!out.contains("OTS-E-LNK-001"), "{out}");
    // a Transmit at 100 + ctrl cycles is released after it: no diagnostic
    let out = both(&[&c, "--trace"], "Q 7e 6c 50c00000\nP ctrl 64 0\n");
    assert!(!out.contains("OTS-E-LNK-003"), "{out}");
    // a Transmit before the packet is made leaves first, and the packet waits for the vector's cycles
    let out = both(&[&c, "--trace"], "Q 7e 64 50c00000\nP ctrl 65 0\n");
    assert!(!out.contains("OTS-E-LNK-003"), "{out}");
}

/// What a library host's run reports: the stream writes, the DMA events and the vectors PciSend gave the host.
#[derive(Default)]
struct Seen {
    srf: Vec<(u64, u8, Vec<u8>)>,
    dma: Vec<(u64, &'static str)>,
}

impl tsp_iss::Observer for Seen {
    fn srf(&mut self, c: u64, _row: tsp_iss::Row, stream: u8, _gap: u8, v: &[u8]) {
        self.srf.push((c, stream, v.to_vec()));
    }
    fn host_dma(&mut self, c: u64, what: tsp_iss::observe::HostDma) {
        self.dma.push((c, what.word()));
    }
}

// model: c_emulator/tsp_sim.c host_setup (the `H dma` lines); `Machine::host_extend` is the same lines given while the chips run
#[test]
fn host_extend_gives_the_events_of_a_script_that_had_them_up_front() {
    use tsp_iss::host::{HostEvent, HostKind};
    let profile = ots_profile::Profile::load("mini").unwrap();
    let vl = tsp_iss::Geom::new(&profile).vlanes as usize;
    let dma = |cycle: u64, k: u8| {
        let mut b = Box::new([0u8; MAX_LANES]);
        b[..vl].fill(k);
        HostEvent { cycle, kind: HostKind::Dma(b) }
    };
    // PciReceive S24 and S25 on RX queue 142 at cycles 300 and 310
    let build = || {
        let mut m = tsp_iss::Machine::new(&profile, 0, Box::new(tsp_iss::NoLinks)).unwrap();
        m.load_image(142, &[300, 0x5140_e000, 9, 0x5140_e400]).unwrap();
        m
    };
    let events = vec![dma(200, 2), dma(190, 1), dma(190, 3)];
    let mut scripted = build();
    scripted.set_host_script(events.clone(), 0);
    let mut want = Seen::default();
    scripted.run(100_000, &mut want).unwrap();
    assert_eq!(
        want.dma,
        [(190, "in"), (200, "in"), (210, "in")],
        "the scenario is sound: paced 10 apart, in request order"
    );
    assert_eq!(
        want.srf.iter().map(|x| x.1).collect::<Vec<_>>(),
        [24, 25],
        "and the program takes both vectors: {:?}",
        want.srf
    );
    // the same events given after the chip ran to cycle 150, in the same order
    let mut late = build();
    let mut got = Seen::default();
    late.run_until(150, 100_000, &mut got).unwrap();
    late.host_extend(events);
    late.run(100_000, &mut got).unwrap();
    assert_eq!(got.dma, want.dma);
    assert_eq!(got.srf, want.srf);
    assert_eq!(late.host().egress, scripted.host().egress);
    assert_eq!((late.host().dma_pending(), late.host().ingress_len()), (0, 1));
}
