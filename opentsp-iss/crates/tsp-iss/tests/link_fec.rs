//! C2C link FEC and link error reporting (`microarchitecture.md` §17): the flit map, the `F link` input line, the link
//! port's events and poison mask, and the link counters of the CSR space. The whole path through the machine is held to
//! `tsp_sim` by `opentsp-arch-spec/test/directed/link_fec.py` run with the ISS, and by the golden vectors (`ecc-*`).
use ots_profile::Profile;
use tsp_iss::geom::{Geom, MAX_LANES};
use tsp_iss::harness::input::{parse, Limits};
use tsp_iss::harness::links::PlanLinks;
use tsp_iss::host::Host;
use tsp_iss::links::{link_flit_mask, link_flits, LinkEvent, LinkFault, LinkPort};

fn geom(name: &str) -> Geom {
    Geom::new(&Profile::load(name).unwrap())
}

fn err(input: &str, chips: u32) -> String {
    let g = geom("ref");
    parse(input.as_bytes(), Limits { geom: &g, chips, topo: None }).unwrap_err().msg
}

// model: unit_tests/test_c2c.sail test_link_fec
// model: slices/c2c/c2c_insts.sail link_flit_mask
#[test]
fn the_flit_map_covers_the_superlanes_of_a_vector() {
    // Five payload bytes in the header, 39 bodies of eight, three in the tail.
    assert_eq!(link_flits(320), 41);
    let all = (1u32 << 20) - 1;
    assert_eq!(link_flit_mask(0, 320, 20), all, "the header covers every superlane");
    assert_eq!(link_flit_mask(40, 320, 20), all, "the tail covers every superlane");
    assert_eq!(link_flit_mask(1, 320, 20), 1);
    assert_eq!(link_flit_mask(2, 320, 20), 3, "bytes 13..20 span superlanes 0 and 1");
    assert_eq!(link_flit_mask(3, 320, 20), 2);
    assert_eq!(link_flit_mask(39, 320, 20), 1 << 19);
    let mut covered = 0u32;
    for f in 1..40 {
        covered |= link_flit_mask(f, 320, 20);
    }
    assert_eq!(covered, all, "the payload flits together cover every superlane");
}

// model: c_emulator/tsp_sim.c read_link_fault
#[test]
fn an_f_link_line_is_read_and_checked_in_the_order_of_the_harness() {
    let g = geom("ref");
    let p = parse(
        b"F link 13c 0 2 uncorr\nF link 13c 0 3 corr t\nF link 13c 0 1 corr p\n",
        Limits { geom: &g, chips: 1, topo: None },
    )
    .unwrap();
    assert_eq!(
        p.link_faults,
        vec![
            LinkFault { chip: 0, cycle: 0x13c, link: 0, flit: 2, unc: true },
            LinkFault { chip: 0, cycle: 0x13c, link: 0, flit: 3, unc: false },
            LinkFault { chip: 0, cycle: 0x13c, link: 0, flit: 1, unc: false },
        ]
    );
    assert!(err("F link 10 99 1 corr\n", 1).contains("link out of range (>= c2c.links)"));
    assert!(err("F link 10 3 1 corr\n", 2).contains("F link: that link is not connected"));
    assert!(err("F link 10 0 2a uncorr\n", 1).contains("flit out of range"));
    assert!(err("F link 10 0 1 maybe\n", 1).contains("F link: expected corr or uncorr"));
    assert!(err("F link 10 0 1\n", 1).contains("F link: expected corr or uncorr"));
    assert!(err("F link 10 0 1 corr x\n", 1).contains("F: expected t or p at the end of the line"));
    assert!(err("F link 10 0 1 corr\nF link 10 0 1 uncorr\n", 1).contains("two lines mark the same flit of one vector"));
    assert!(err("F net 10 0 1 corr\n", 1).contains("expected sram, srf, iq, link or mxm"));
}

// model: c_emulator/tsp_sim.c c2c_send, process_arrivals, c2c_recv
#[test]
fn a_marked_vector_reports_its_flits_at_arrival_and_poisons_the_receive() {
    let g = geom("ref");
    let mut port = PlanLinks::loopback(&g);
    let faults = [(0x20u64, 4u8, true), (0x20, 2, false), (0x20, 9, true), (0x30, 0, false)];
    port.state().lock().unwrap().set_faults(faults.iter().map(|&(cycle, flit, unc)| LinkFault {
        chip: 0,
        cycle,
        link: 0,
        flit,
        unc,
    }));
    let v = [7u8; MAX_LANES];
    port.send(0x20, 0, &v);
    port.send(0x30, 0, &v);
    port.send(0x40, 0, &v); // no fault
    let arrival = 0x20 + g.link_latency;
    assert_eq!(port.link_events(arrival - 1), vec![], "nothing before the arrival");
    let ev = port.link_events(arrival);
    let want: Vec<LinkEvent> = [(2u8, false), (4, true), (9, true)]
        .iter()
        .map(|&(flit, unc)| LinkEvent::Fec { cycle: arrival, link: 0, unc, flit })
        .collect();
    assert_eq!(ev, want, "by flit, corrected and uncorrectable together");
    assert!(port.recv_ready(arrival, 0));
    port.recv(arrival, 0);
    // Body 4 spans bytes 29..36 (superlanes 1,2), body 9 bytes 69..76 (superlane 4).
    assert_eq!(port.recv_bad(0), (1 << 1) | (1 << 2) | (1 << 4));
    // the second vector had a corrected flit only: no poison
    let second = 0x30 + g.link_latency;
    assert_eq!(port.link_events(second), vec![LinkEvent::Fec { cycle: second, link: 0, unc: false, flit: 0 }]);
    port.recv(second, 0);
    assert_eq!(port.recv_bad(0), 0);
    port.recv(0x40 + g.link_latency, 0);
    assert_eq!(port.recv_bad(0), 0, "an unmarked vector is clean");
    assert_eq!(port.state().lock().unwrap().unused_fault(), None);
}

// model: c_emulator/tsp_sim.c finish_check
#[test]
fn a_line_that_marked_no_vector_is_reported_in_input_order() {
    let g = geom("ref");
    let mut port = PlanLinks::loopback(&g);
    let f = |cycle, link, flit| LinkFault { chip: 0, cycle, link, flit, unc: false };
    port.state().lock().unwrap().set_faults([f(5, 0, 1), f(0x20, 1, 1), f(0x21, 0, 2)]);
    port.send(0x21, 0, &[0u8; MAX_LANES]);
    assert_eq!(port.state().lock().unwrap().unused_fault(), Some((0, 5)), "the first unused line in input order");
}

// model: core/csr.sail csr_read
// model: postlude/harness_api.sail model_link_fec
#[test]
fn the_link_counters_live_at_0x200_and_0x210_and_reset() {
    let mut h = Host::new();
    h.link_fec(3, false);
    h.link_fec(3, false);
    h.link_fec(3, true);
    h.link_fec(15, true);
    assert_eq!(h.csr_read(0x203, 0, 64), 2);
    assert_eq!(h.csr_read(0x213, 0, 64), 1);
    assert_eq!(h.csr_read(0x21f, 0, 64), 1);
    assert_eq!(h.csr_read(0x200, 0, 64), 0);
    assert_eq!(h.csr_read(0x220, 0, 64), 0, "past the 16 links");
    h.reset();
    assert_eq!(h.csr_read(0x203, 0, 64), 0);
    assert_eq!(h.csr_read(0x213, 0, 64), 0);
}
