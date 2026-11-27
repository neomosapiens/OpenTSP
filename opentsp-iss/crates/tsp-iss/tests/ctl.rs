//! The control unit's packet rules (`slices/c2c/c2c_insts.sail` packet types; `c_emulator/tsp_sim.c` `ctl_candidate`,
//! `ctl_depart`, `c2c_packet_busy`): the types, the cycles a packet occupies a link and the causal rule of a link
//! shared by the typed packets (`provenance.yaml` `host.link_share`).
use tsp_iss::ctl::{self, CtlPkt, Op, Window};
use tsp_iss::geom::Geom;

fn window(name: &str) -> (Window, u64, u64) {
    let p = ots_profile::Profile::load(name).unwrap();
    let g = Geom::new(&p);
    (Window::new(g.cycles_per_vector, g.vlanes), g.cycles_per_vector, g.vlanes)
}

fn pkt(id: u64) -> CtlPkt {
    CtlPkt { op: Op::Ctrl, origin: 0, target: 0, addr: 0, val: 0, id }
}

// model: unit_tests/test_c2c.sail test_c2c_packet_types
#[test]
fn test_c2c_packet_types() {
    for name in ["ref", "mini"] {
        let (w, n, vlanes) = window(name);
        let vf = 1 + vlanes.div_ceil(8);
        assert!(ctl::queued(ctl::PT_DATA) && ctl::queued(ctl::PT_SYNC));
        assert!(!ctl::queued(ctl::PT_CSR) && !ctl::queued(ctl::PT_CTRL));
        assert_eq!((w.cycles(ctl::PT_DATA), w.cycles(ctl::PT_SYNC)), (n, n));
        let (csr, ctrl) = (w.cycles(ctl::PT_CSR), w.cycles(ctl::PT_CTRL));
        assert!(csr * vf >= n * 3 && (csr - 1) * vf < n * 3, "{name}");
        assert!(ctrl * vf >= n * 2 && (ctrl - 1) * vf < n * 2, "{name}");
        assert!(csr <= n && ctrl <= csr && ctrl >= 1, "{name}");
        assert_eq!((ctl::PT_DATA, ctl::PT_CSR, ctl::PT_SYNC, ctl::PT_CTRL), (0, 1, 2, 3));
    }
}

// model: c_emulator/tsp_sim.c ctl_candidate, ctl_depart, c2c_packet_busy; slices/c2c/c2c_insts.sail OTS-E-LNK-003
#[test]
fn a_packet_leaves_when_the_link_is_free_and_the_rule_is_causal() {
    let (mut w, n, _) = window("ref");
    let k = w.cycles(ctl::PT_CTRL);
    // a vector accepted at 90: the link is busy until 90 + n; the packet ready at 100 leaves at 90 + n
    w.accepted(0, 90);
    w.add(pkt(0), ctl::PT_CTRL, 0, 100);
    assert_eq!(w.wake(), Some(u128::from(90 + n)));
    // the departure is certain once the chip has ticked every cycle below it: at 90 + n - 1 it is not, at 90 + n it is
    assert!(w.depart(None, Some(u128::from(90 + n - 1)), u64::MAX).unwrap().is_empty());
    let out = w.depart(None, Some(u128::from(90 + n)), u64::MAX).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, u128::from(90 + n));
    // a vector accepted at or after the packet's cycle does not delay it (the control unit knows nothing of what comes
    // after the cycle it leaves in): the packet leaves at 100 and the vector is released into its cycles (LNK-003)
    let (mut w, _, _) = window("ref");
    w.accepted(0, 100 + k - 1);
    w.add(pkt(0), ctl::PT_CTRL, 0, 100);
    let out = w.depart(None, None, u64::MAX).unwrap();
    assert_eq!(out[0].1, 100);
    assert!(
        !w.packet_busy(99, 0) && w.packet_busy(100, 0) && w.packet_busy(100 + k - 1, 0) && !w.packet_busy(100 + k, 0)
    );
    // a vector accepted before the ready cycle does delay it, by its cycles
    let (mut w, n, _) = window("ref");
    w.accepted(0, 99);
    w.add(pkt(0), ctl::PT_CTRL, 0, 100);
    assert_eq!(w.depart(None, None, u64::MAX).unwrap()[0].1, u128::from(99 + n));
}

#[test]
fn packets_of_a_link_leave_in_order_and_occupy_it() {
    let (mut w, _, _) = window("ref");
    let k = w.cycles(ctl::PT_CTRL);
    w.add(pkt(5), ctl::PT_CTRL, 0, 10);
    w.add(pkt(3), ctl::PT_CTRL, 0, 10);
    w.add(pkt(4), ctl::PT_CTRL, 1, 10);
    let out = w.depart(None, None, u64::MAX).unwrap();
    let order: Vec<(u64, usize, u128)> = out.iter().map(|(p, c)| (p.pkt.id, p.link, *c)).collect();
    // links in ascending order; on link 0 the lower packet number first, the second after the first's cycles
    assert_eq!(order, vec![(3, 0, 10), (5, 0, u128::from(10 + k)), (4, 1, 10)]);
    // a reply sorts after the request of the same number
    assert!(CtlPkt { op: Op::Reply, ..pkt(7) }.seq() > pkt(7).seq());
}
