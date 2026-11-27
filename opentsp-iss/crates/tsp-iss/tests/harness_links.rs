//! Tests of `harness/links`, moved from the module's in-source test block (they use the crate's public API).
use ots_profile::Profile;
use std::sync::Arc;
use tsp_iss::geom::{Geom, MAX_LANES};
use tsp_iss::harness::cli::LinkMode;
use tsp_iss::harness::links::*;
use tsp_iss::harness::topology::Topology;
use tsp_iss::links::LinkPort;

fn geom() -> Geom {
    Geom::new(&Profile::load("ref").unwrap())
}

#[test]
fn one_chip_loops_back() {
    for l in 0..16 {
        assert_eq!(link_peer(1, 16, 0, l), Some((0, l)));
    }
    assert_eq!(link_peer(1, 4, 0, 4), None);
}

#[test]
fn a_node_is_fully_connected_and_symmetric() {
    for k in 2..=8usize {
        for c in 0..k {
            let mut reached = vec![false; k];
            for l in 0..16 {
                match link_peer(k, 16, c, l) {
                    Some((p, pl)) => {
                        assert!(l <= 6 && p != c, "k {k} chip {c} link {l}");
                        assert_eq!(link_peer(k, 16, p, pl), Some((c, l)), "k {k} chip {c} link {l}");
                        reached[p] = true;
                    }
                    None => assert!(l >= k - 1 || l > 6, "k {k} chip {c} link {l}"),
                }
            }
            assert_eq!(reached.iter().filter(|r| **r).count(), k - 1);
        }
    }
}

#[test]
fn nodes_are_joined_by_global_links_symmetrically() {
    for n in 2..=8usize {
        let chips = 8 * n;
        let mut pairs = std::collections::BTreeSet::new();
        for c in 0..chips {
            for l in 7..=10 {
                if let Some((p, pl)) = link_peer(chips, 16, c, l) {
                    assert!((7..=10).contains(&pl));
                    assert_eq!(link_peer(chips, 16, p, pl), Some((c, l)), "n {n} chip {c} link {l}");
                    assert_ne!(c / 8, p / 8);
                    pairs.insert((c / 8, p / 8));
                }
            }
        }
        // every ordered pair of nodes is joined
        assert_eq!(pairs.len(), n * (n - 1), "n {n}");
    }
    assert_eq!(link_peer(16, 16, 0, 11), None);
    assert_eq!(link_peer(8, 16, 0, 7), None);
}

#[test]
fn a_last_node_that_is_not_full_is_wired_symmetrically() {
    for chips in [9usize, 12, 15, 17, 20, 100, 263, 264] {
        let mut pairs = std::collections::BTreeSet::new();
        for c in 0..chips {
            for l in 0..16 {
                if let Some((p, pl)) = link_peer(chips, 16, c, l) {
                    assert!(p < chips, "chips {chips} chip {c} link {l} reaches chip {p}");
                    assert_eq!(link_peer(chips, 16, p, pl), Some((c, l)), "chips {chips} chip {c} link {l}");
                    if c / 8 != p / 8 {
                        pairs.insert((c / 8, p / 8));
                    }
                }
            }
        }
        // two nodes are joined when the chip of the global port of the pair exists in both
        let nodes = chips.div_ceil(8);
        assert!(pairs.len() <= nodes * (nodes - 1), "chips {chips}");
        if chips % 8 == 0 {
            assert_eq!(pairs.len(), nodes * (nodes - 1), "chips {chips}");
        }
    }
    // 12 chips: a node of 8 and a node of 4; the global link 7 of chip 0 reaches chip 8 on link 7, the local links of the small node end at link 2
    assert_eq!(link_peer(12, 16, 0, 7), Some((8, 7)));
    assert_eq!(link_peer(12, 16, 8, 0), Some((9, 2)));
    assert_eq!(link_peer(12, 16, 8, 3), None);
    assert_eq!(link_peer(12, 16, 0, 8), None);
}

#[test]
fn thirty_three_nodes_join_every_pair() {
    let mut pairs = std::collections::BTreeSet::new();
    for c in 0..264usize {
        for l in 7..=10 {
            if let Some((p, pl)) = link_peer(264, 16, c, l) {
                assert_eq!(link_peer(264, 16, p, pl), Some((c, l)));
                pairs.insert((c / 8, p / 8));
            }
        }
    }
    assert_eq!(pairs.len(), 33 * 32);
    assert_eq!(link_peer(264, 16, 0, 7), Some((15, 10)));
}

#[test]
fn the_jitter_state_of_the_links_below_16_is_the_earlier_one() {
    for (chip, link) in [(0usize, 0usize), (1, 3), (63, 15)] {
        assert_eq!(
            jitter_state(5, chip, link),
            5 ^ ((chip * 16 + link + 1) as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        );
    }
    let mut seen = std::collections::BTreeSet::new();
    for chip in 0..512 {
        for link in 0..128 {
            assert!(seen.insert(jitter_state(0, chip, link)), "chip {chip} link {link}");
        }
    }
}

fn topo(text: &str) -> Result<Topology, String> {
    Topology::parse(text.as_bytes(), &geom())
}

#[test]
fn a_topology_file_gives_the_adjacency_and_the_latencies() {
    let t = topo(r#"{"chips": 3, "latency": 212, "links": [{"a": [0, 3], "b": [2, 9], "latency": 220}, {"a": [1, 0], "b": [1, 0]}, {"a": [1, 5], "b": [0, 4]}]}"#)
        .unwrap();
    assert_eq!(t.chips, 3);
    assert_eq!((t.peer(0, 3), t.peer(2, 9)), (Some((2, 9)), Some((0, 3))));
    assert_eq!(t.peer(1, 0), Some((1, 0)), "a link that names the same end twice loops back");
    assert_eq!(t.peer(0, 0), None);
    let g = geom();
    let plan = LinkPlan::with_topology(LinkMode::Nominal, &g, Arc::new(t.clone())).unwrap();
    assert_eq!((plan.latency(0, 3), plan.latency(2, 9)), (220, 220));
    assert_eq!((plan.latency(1, 5), plan.latency(1, 0)), (212, 212), "the file's default");
    assert_eq!(plan.window(), 212);
    assert_eq!(plan.peer(0, 3), Some((2, 9)));
    assert_eq!(plan.peer(0, 0), None);
    // a link option overrides every latency of the file
    let plan = LinkPlan::with_topology(LinkMode::Fixed(210), &g, Arc::new(t)).unwrap();
    assert_eq!((plan.latency(0, 3), plan.latency(1, 5)), (210, 210));
    // the file's jitter seed is the mode when no link option is given
    let t = topo(r#"{"chips": 2, "jitter_seed": 7, "links": [{"a": [0, 0], "b": [1, 0]}]}"#).unwrap();
    let plan = LinkPlan::with_topology(LinkMode::Nominal, &g, Arc::new(t)).unwrap();
    assert_eq!(plan.mode, LinkMode::Jitter(7));
    assert_eq!(plan.jitter_state(1, 0), jitter_state(7, 1, 0));
}

#[test]
fn a_topology_file_is_checked() {
    let bad = [
        (r#"{"chips":1,"links":[{"z":1,"a":[0,0],"a":[0,0],"b":[0,0]}]}"#, "links[0]: unknown key \"z\""),
        (r#"{"chips":1,"links":[{"a":[0,0],"a":[0,0],"b":[0,0]}],"chips":1}"#, "duplicate key \"chips\""),
        (r#"{"chips":1,"unknown":0,"links":[{"a":[0,0],"a":[0,0],"b":[0,0]}]}"#, "unknown key \"unknown\""),
        (r#"{"chips":1,"z":0,"a":0,"links":[]}"#, "unknown key \"z\""),
        (r#"{"chips":"\u0000","chips":1,"links":[]}"#, "not a JSON object"),
        ("{\"chips\": 2, \"links\": []}\0x", "not a JSON object"),
        (r#"{"chips\u0000suffix": 2, "links": []}"#, "not a JSON object"),
        (r#"{"chips": 2, "links": [{"a\u0000suffix": [0, 0], "b": [1, 0]}]}"#, "not a JSON object"),
        (r#"{"chips": "2\u0000suffix", "links": []}"#, "not a JSON object"),
        (r#"{"chips": 2, "links": [{"a": [0, 0], "b": [2, 0]}]}"#, "chip: 2 is outside 0..1"),
        (r#"{"chips": 2, "links": [{"a": [0, 16], "b": [1, 0]}]}"#, "link: 16 is outside 0..15"),
        (
            r#"{"chips": 3, "links": [{"a": [0, 0], "b": [1, 0]}, {"a": [2, 0], "b": [0, 0]}]}"#,
            "links[1]: chip 0 link 0 is joined twice",
        ),
        (r#"{"chips": 2, "links": [], "x": 1}"#, "unknown key \"x\""),
        (r#"{"links": []}"#, "\"chips\" is missing"),
        (r#"{"chips": 513, "links": []}"#, "chips: 513 is outside 1..512"),
        (r#"{"chips": 1.5, "links": []}"#, "chips must be an integer"),
        (r#"{"chips": 2}"#, "\"links\" must be an array"),
        (r#"{"chips": 2, "links": [{"a": [0, 0]}]}"#, "links[0].b must be [chip, link]"),
        (
            r#"{"chips": 2, "links": [{"a": [0, 0], "b": [1, 0], "latency": 5}]}"#,
            "links[0].latency: 5 is outside 209..228",
        ),
        (
            r#"{"chips": 2, "cycles_per_vector": 7, "links": []}"#,
            "cycles_per_vector 7 is not the profile's c2c.cycles_per_vector 24",
        ),
        (r#"{"chips": 2, "links": [{"a": [0, 0], "b": [1, 0], "z": 1}]}"#, "links[0]: unknown key \"z\""),
        ("[1]", "not a JSON object"),
        ("{", "not a JSON object"),
        // : trailing text and a repeated key are refused, as tsp_sim refuses them
        (r#"{"chips": 2, "links": []} x"#, "not a JSON object"),
        (r#"{"chips": 2, "chips": 3, "links": []}"#, "duplicate key \"chips\""),
        (r#"{"chips": 2, "links": [{"a": [0, 0], "b": [1, 0], "b": [1, 1]}]}"#, "duplicate key \"b\""),
    ];
    for (text, msg) in bad {
        assert_eq!(topo(text).unwrap_err(), msg, "{text}");
    }
    assert!(topo(r#"{"chips": 512, "links": []}"#).is_ok());
}

#[test]
fn topology_number_boundaries_preserve_semantic_diagnostics_and_precedence() {
    // These diagnostics are the C harness contract, including binary64 overflow,
    // signed underflow, and integer-valued floats beyond the former 9e15 guard.
    for literal in ["1e400", "-1e400", "1e99999", "1.5", "null", "true", "\"1\""] {
        let fields = [
            (format!(r#"{{"chips":{literal},"links":[]}}"#), "chips"),
            (format!(r#"{{"chips":1,"links":[],"latency":{literal}}}"#), "latency"),
            (format!(r#"{{"chips":1,"links":[],"jitter_seed":{literal}}}"#), "jitter_seed"),
            (format!(r#"{{"chips":1,"links":[],"cycles_per_vector":{literal}}}"#), "cycles_per_vector"),
            (format!(r#"{{"chips":1,"links":[{{"a":[{literal},0],"b":[0,0]}}]}}"#), "chip"),
            (format!(r#"{{"chips":1,"links":[{{"a":[0,{literal}],"b":[0,0]}}]}}"#), "link"),
            (format!(r#"{{"chips":1,"links":[{{"a":[0,0],"b":[0,0],"latency":{literal}}}]}}"#), "links[0].latency"),
        ];
        for (text, field) in fields {
            assert_eq!(topo(&text).unwrap_err(), format!("{field} must be an integer"), "{text}");
        }
    }
    for (literal, printed) in [
        ("9000000000000000", "9000000000000000"),
        ("1e16", "10000000000000000"),
        ("9223372036854775808", "9223372036854775808"),
        ("-9223372036854775808", "-9223372036854775808"),
        ("1e-400", "0"),
        ("-1e-400", "-0"),
    ] {
        let text = format!(r#"{{"chips":{literal},"links":[]}}"#);
        assert_eq!(topo(&text).unwrap_err(), format!("chips: {printed} is outside 1..512"), "{text}");
    }
    for (text, expected) in [
        (r#"{"chips":1,"links":[],"junk":1e400}"#, "unknown key \"junk\""),
        (r#"{"chips":1e400,"chips":1,"links":[]}"#, "duplicate key \"chips\""),
        (r#"{"chips":1e400,"links":[],"junk":0}"#, "unknown key \"junk\""),
        (r#"{"chips":1,"links":[{"junk":1e400,"a":[0,0],"a":[0,0],"b":[0,0]}]}"#, "links[0]: unknown key \"junk\""),
        (r#"{"chips":1,"links":[{"a":[1e400,0],"a":[0,0],"b":[0,0]}]}"#, "duplicate key \"a\""),
        (r#"{"chips":1e400,"links":[{"junk":0}]}"#, "chips must be an integer"),
        (r#"{"chips":1e400,"links":[],"junk":"\u0000"}"#, "not a JSON object"),
        (r#"{"chips":1e400,"links":[]}x"#, "not a JSON object"),
        (r#"{"chips":01,"links":[]}"#, "not a JSON object"),
        (r#"{"chips":1.,"links":[]}"#, "not a JSON object"),
        (r#"{"chips":NaN,"links":[]}"#, "not a JSON object"),
    ] {
        assert_eq!(topo(text).unwrap_err(), expected, "{text}");
    }
    // The normative strict syntax reader caps value depth at 128. Validating
    // with the lossless reader must not accidentally widen that boundary.
    for depth in [126, 127, 128, 129, 200] {
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let text = format!(r#"{{"chips":1,"links":[],"junk":{nested}}}"#);
        let expected = if depth <= 127 { "unknown key \"junk\"" } else { "not a JSON object" };
        assert_eq!(topo(&text).unwrap_err(), expected, "depth {depth}");
    }
    let t = topo(r#"{"chips":1e0,"links":[{"a":[-1e-400,0.0],"b":[0e0,0]}],"latency":216.0,"jitter_seed":1e-400,"cycles_per_vector":2.4e1}"#).unwrap();
    assert_eq!(t.chips, 1);
    assert_eq!(t.peer(0, 0), Some((0, 0)));
    assert_eq!(t.latency_default, Some(216));
    assert_eq!(t.jitter_seed, Some(0));
}

#[test]
fn splitmix_known_values() {
    // the reference values of splitmix64 from seed 0
    let mut x = 0u64;
    assert_eq!(splitmix(&mut x), 0xE220_A839_7B1D_CDAF);
    assert_eq!(splitmix(&mut x), 0x6E78_9E6A_A1B9_65F4);
    assert_eq!(splitmix(&mut x), 0x06C4_5D18_8009_454F);
}

#[test]
fn plan_latencies() {
    let g = geom();
    let p = LinkPlan::new(LinkMode::Nominal, &g, 1).unwrap();
    assert_eq!((p.latency(0, 0), p.window()), (216, 216));
    let p = LinkPlan::new(LinkMode::Fixed(210), &g, 2).unwrap();
    assert_eq!((p.latency(1, 15), p.window()), (210, 210));
    let p = LinkPlan::new(LinkMode::Jitter(5), &g, 2).unwrap();
    assert_eq!((p.latency(0, 0), p.window()), (209, 209));
    assert_eq!(p.jitter_state(0, 0), 5 ^ 0xD1B5_4A32_D192_ED03);
    let a = LinkPlan::new(LinkMode::Seed(7), &g, 8).unwrap();
    let b = LinkPlan::new(LinkMode::Seed(7), &g, 8).unwrap();
    let c = LinkPlan::new(LinkMode::Seed(8), &g, 8).unwrap();
    let all = |p: &LinkPlan| {
        (0..8).flat_map(|c| (0..16).map(move |l| (c, l))).map(|(c, l)| p.latency(c, l)).collect::<Vec<_>>()
    };
    assert_eq!(all(&a), all(&b));
    assert_ne!(all(&a), all(&c));
    assert!(all(&a).iter().all(|l| (209..=228).contains(l)));
    assert!(a.window() >= 209);
}

#[test]
fn plan_latency_map() {
    let g = geom();
    let p = LinkPlan::new(LinkMode::Map(vec![(0, 0, 209), (2, 1, 228)]), &g, 3).unwrap();
    assert_eq!((p.latency(0, 0), p.latency(2, 1)), (209, 228));
    // a link not named has the nominal latency
    assert_eq!((p.latency(1, 0), p.latency(0, 1)), (216, 216));
    assert_eq!(p.window(), 209);
    for (m, text) in [
        (vec![(0, 0, 208)], "--link-latency-map 0:0=208: the latency must be in [209, 228]"),
        (vec![(0, 0, 229)], "--link-latency-map 0:0=229: the latency must be in [209, 228]"),
        (vec![(3, 0, 216)], "--link-latency-map 3:0=216: no such chip or link"),
        (vec![(0, 16, 216)], "--link-latency-map 0:16=216: no such chip or link"),
    ] {
        assert_eq!(LinkPlan::new(LinkMode::Map(m), &g, 3).unwrap_err(), text);
    }
}

#[test]
fn plan_latency_map_names_links_up_to_the_links_of_the_profile() {
    // a link at or above `c2c.links` of the profile does not exist, however wide the widest profile is
    let mut g = geom();
    g.links = 20;
    let p = LinkPlan::new(LinkMode::Map(vec![(0, 19, 228)]), &g, 3).unwrap();
    assert_eq!(p.latency(0, 19), 228);
    assert_eq!(
        LinkPlan::new(LinkMode::Map(vec![(0, 20, 216)]), &g, 3).unwrap_err(),
        "--link-latency-map 0:20=216: no such chip or link"
    );
}

#[test]
fn plan_refuses_out_of_range_latency() {
    let g = geom();
    assert_eq!(LinkPlan::new(LinkMode::Fixed(208), &g, 1).unwrap_err(), "--link-latency must be in [209, 228]");
    assert!(LinkPlan::new(LinkMode::Fixed(229), &g, 1).is_err());
    assert!(LinkPlan::new(LinkMode::Fixed(209), &g, 1).is_ok());
    assert!(LinkPlan::new(LinkMode::Fixed(228), &g, 1).is_ok());
}

fn vecmsg(arrival: u64, tag: u8) -> VecMsg {
    let mut b = [0u8; MAX_LANES];
    b[0] = tag;
    VecMsg {
        arrival: arrival.into(),
        chip: 0,
        link: 0,
        state: VState::Pending,
        taken_at: 0,
        ptype: 0,
        ctl: None,
        fec_c: 0,
        fec_u: 0,
        b,
        poison: 0,
    }
}

// model: c_emulator/tsp_sim.c fifo_push
#[test]
fn a_fifo_is_in_arrival_order_and_ties_keep_the_order_of_insertion() {
    let g = geom();
    let plan = Arc::new(LinkPlan::new(LinkMode::Nominal, &g, 1).unwrap());
    let mut p = PlanLinks::new(plan.clone(), 0, &g);
    for (a, t) in [(10, 1), (12, 2), (11, 3), (10, 4), (9, 5), (12, 6)] {
        p.state().lock().unwrap().deliver(vecmsg(a, t));
    }
    assert_eq!(p.state().lock().unwrap().inbound_len(0), 6);
    let mut order = Vec::new();
    while p.recv_ready(20, 0) {
        order.push(p.recv(20, 0)[0]);
    }
    assert_eq!(order, vec![5, 1, 4, 3, 2, 6], "arrival 9, 10, 10, 11, 12, 12; a tie keeps the order of insertion");
    // nothing is ready before its arrival cycle
    let mut early = PlanLinks::new(plan, 0, &g);
    early.state().lock().unwrap().deliver(vecmsg(30, 1));
    assert!(!early.recv_ready(29, 0) && early.recv_ready(30, 0));
}

#[test]
fn a_looped_back_vector_waits_for_its_latency_and_is_taken_once() {
    let g = geom();
    let mut p = PlanLinks::loopback(&g);
    assert_eq!(p.peer(5), Some(0));
    assert_eq!(p.peer(16), None);
    let mut v = [0u8; MAX_LANES];
    v[3] = 77;
    p.send(100, 5, &v);
    assert!(!p.recv_ready(100 + 215, 5) && !p.recv_ready(1000, 4));
    assert!(p.recv_ready(100 + 216, 5));
    assert_eq!(p.recv(0, 5)[3], 77);
    assert!(!p.recv_ready(10_000, 5));
    assert_eq!(p.recv(0, 5), [0; MAX_LANES]);
}

#[test]
fn a_vector_for_another_chip_is_held_for_the_harness_and_delivered_on_the_far_link() {
    let g = geom();
    let plan = Arc::new(LinkPlan::new(LinkMode::Fixed(210), &g, 3).unwrap());
    let a = PlanLinks::new(plan.clone(), 0, &g);
    let b = PlanLinks::new(plan.clone(), 1, &g);
    // chip 0 link 0 reaches chip 1 on its link K - 2 - 0 = 1
    assert_eq!((a.peer(0), b.peer(1)), (Some(1), Some(0)));
    let mut a2 = a.clone();
    a2.send(50, 0, &[9; MAX_LANES]);
    let out = a.state().lock().unwrap().take_outbound();
    assert_eq!(out.len(), 1);
    assert_eq!((out[0].arrival, out[0].chip, out[0].link), (260, 1, 1));
    assert!(a.state().lock().unwrap().take_outbound().is_empty());
    b.state().lock().unwrap().deliver(out[0].clone());
    let mut b2 = b.clone();
    assert!(!b2.recv_ready(259, 1) && b2.recv_ready(260, 1));
    assert_eq!(b2.recv(0, 1)[0], 9);
}

// model: c_emulator/tsp_sim.c c2c_send
#[test]
fn jitter_draws_in_range_and_delivers_in_order_one_vector_per_cycle() {
    let g = geom();
    let plan = Arc::new(LinkPlan::new(LinkMode::Jitter(3), &g, 2).unwrap());
    let mut a = PlanLinks::new(plan, 0, &g);
    // the same send cycle ten times: arrivals never decrease and are one cycle apart at least
    for _ in 0..10 {
        a.send(1000, 0, &[1; MAX_LANES]);
    }
    let out = a.state().lock().unwrap().take_outbound();
    assert_eq!(out.len(), 10);
    assert!((1000 + 209..=1000 + 228).contains(&out[0].arrival));
    for w in out.windows(2) {
        assert!(w[1].arrival > w[0].arrival, "{} then {}", w[0].arrival, w[1].arrival);
    }
    // the same seed draws the same latencies
    let plan2 = Arc::new(LinkPlan::new(LinkMode::Jitter(3), &g, 2).unwrap());
    let mut b = PlanLinks::new(plan2, 0, &g);
    for _ in 0..10 {
        b.send(1000, 0, &[1; MAX_LANES]);
    }
    let again = b.state().lock().unwrap().take_outbound();
    assert_eq!(out.iter().map(|m| m.arrival).collect::<Vec<_>>(), again.iter().map(|m| m.arrival).collect::<Vec<_>>());
}

// model: c_emulator/tsp_sim.c link_arrival (one in-order clamp per link over every packet type)
#[test]
fn jitter_keeps_one_order_per_link_over_every_packet_type() {
    use tsp_iss::ctl::{CtlPkt, Op, PT_CTRL, PT_DATA};
    let g = geom();
    let pkt = CtlPkt { op: Op::Ctrl, origin: 0, target: 0, addr: 0, val: 0, id: 0 };
    let k = tsp_iss::ctl::Window::new(g.cycles_per_vector, g.vlanes).cycles(PT_CTRL);
    let mut swapped = Vec::new();
    for seed in 1..=40u64 {
        // a ctrl packet leaves at cycle 0, a vector is sent as soon as the packet has left the link
        let plan = Arc::new(LinkPlan::new(LinkMode::Jitter(seed), &g, 2).unwrap());
        let mut a = PlanLinks::new(plan.clone(), 0, &g);
        a.ctl_add(pkt, PT_CTRL, 0, 0);
        a.ctl_depart(None, None, u64::MAX).unwrap();
        a.send_packet(k, 0, PT_DATA, &[1; MAX_LANES]);
        let out = a.state().lock().unwrap().take_outbound();
        assert_eq!((out[0].ptype, out[1].ptype), (PT_CTRL, PT_DATA));
        if out[1].arrival <= out[0].arrival {
            swapped.push((seed, "ctrl first", out[0].arrival, out[1].arrival));
        }
        // a vector is sent at 0, a ctrl packet made at 1 leaves when the vector's cycles are over
        let mut b = PlanLinks::new(plan, 0, &g);
        b.send_packet(0, 0, PT_DATA, &[1; MAX_LANES]);
        b.ctl_add(pkt, PT_CTRL, 0, 1);
        b.ctl_depart(None, None, u64::MAX).unwrap();
        let out = b.state().lock().unwrap().take_outbound();
        assert_eq!((out[0].ptype, out[1].ptype), (PT_DATA, PT_CTRL));
        if out[1].arrival <= out[0].arrival {
            swapped.push((seed, "data first", out[0].arrival, out[1].arrival));
        }
    }
    assert!(swapped.is_empty(), "a packet arrived before an earlier one of another type: {swapped:?}");
}

// model: c_emulator/tsp_sim.c link_resolve, link_next_event, process_arrivals
#[test]
fn a_full_receive_fifo_drops_the_arrival_and_a_receive_in_the_arrival_cycle_takes_after_it() {
    let mut g = geom();
    g.rx_depth = 2;
    let lat = g.link_latency;
    for (take_at, dropped) in [(None, true), (Some(lat + 6), true), (Some(lat + 5), false)] {
        let mut p = PlanLinks::loopback(&g);
        for c in [0, 3, 6] {
            p.send(c, 2, &[c as u8; MAX_LANES]);
        }
        p.send(0, 3, &[0; MAX_LANES]); // another link is independent
                                       // only the third arrival on link 2 can overrun
        assert_eq!(p.next_event(), Some(u128::from(lat) + 6));
        assert!(p.arrivals(lat + 5).is_empty());
        if let Some(t) = take_at {
            assert!(p.recv_ready(t, 2));
            assert_eq!(p.recv(t, 2)[0], 0);
        }
        assert_eq!(p.arrivals(lat + 6), if dropped { vec![lat + 6] } else { vec![] }, "{take_at:?}");
        assert_eq!(p.next_event(), None);
        let left = p.state().lock().unwrap().inbound_len(2);
        assert_eq!(left, if take_at.is_some() { 1 } else { 2 } + usize::from(!dropped));
    }
}

#[test]
fn an_unconnected_system_has_window_one() {
    // no link at all (c2c.links = 0): nothing is connected, so any window is correct
    let mut g = geom();
    g.links = 0;
    assert_eq!(LinkPlan::new(LinkMode::Nominal, &g, 2).unwrap().window(), 1);
}

// model: c_emulator/tsp_sim.c fifo_trim: a long run of taken vectors is retired and the data stays right
#[test]
fn many_taken_vectors_are_retired_without_losing_or_repeating_data() {
    let g = geom();
    let lat = g.link_latency;
    let mut p = PlanLinks::loopback(&g);
    for i in 0..1600u64 {
        let tag = (i % 251) as u8;
        p.send(i, 2, &[tag; MAX_LANES]);
        let t = i + lat;
        assert!(p.recv_ready(t, 2), "vector {i}");
        assert_eq!(p.recv(t, 2)[0], tag, "vector {i}");
        assert!(p.arrivals(t).is_empty());
        assert!(!p.recv_ready(t, 2), "a vector is taken once");
    }
    assert_eq!(p.state().lock().unwrap().inbound_len(2), 0);
    assert_eq!(p.next_event(), None);
}

// model: c_emulator/tsp_sim.c link_next_event (one link): the Receives that execute at or after an arrival count
// toward the occupancy it sees, up to 64 of them are remembered
#[test]
fn an_overrun_counts_the_takes_it_knows_up_to_64() {
    let mut g = geom();
    g.rx_depth = 100;
    let lat = g.link_latency;
    let mut p = PlanLinks::loopback(&g);
    // 70 vectors taken at cycles far after the arrivals that follow
    for i in 0..70u64 {
        p.send(i, 2, &[1; MAX_LANES]);
        assert!(p.recv_ready(lat + 1000 + i, 2));
        p.recv(lat + 1000 + i, 2);
    }
    assert_eq!(p.next_event(), None);
    // 40 more arrivals, none taken: each sees the 64 remembered takes and the ones before it, so the 37th
    // (64 + 36 = 100 >= depth) is the first to overrun; with all 70 counted it would be the 31st
    for j in 0..40u64 {
        p.send(100 + j, 2, &[2; MAX_LANES]);
    }
    assert_eq!(p.next_event(), Some(u128::from(lat) + 100 + 36));
}

// model: c_emulator/tsp_sim.c process_arrivals, link_next_event: the drops decided by a Receive's look are
// reported later, in cycle order, and the next event is the earlier of a report and an overrun
#[test]
fn reports_are_due_in_cycle_order_and_the_next_event_is_the_earlier_of_report_and_overrun() {
    let mut g = geom();
    g.rx_depth = 1;
    let lat = g.link_latency;
    let mut p = PlanLinks::loopback(&g);
    assert_eq!(p.next_event(), None);
    // two links overrun on their second vector at the same cycle: both are reported, in one call
    for l in [2u8, 5] {
        p.send(0, l, &[7; MAX_LANES]);
        p.send(0, l, &[7; MAX_LANES]);
    }
    assert_eq!(p.next_event(), Some(u128::from(lat)));
    assert_eq!(p.arrivals(lat), vec![lat, lat]);
    assert_eq!(p.next_event(), None);
    // a Receive's look decides the drop of link 1 at lat + 60 (a report not yet returned); link 2 would overrun at lat + 50
    let mut q = PlanLinks::loopback(&g);
    for c in [60u64, 60] {
        q.send(c, 1, &[1; MAX_LANES]);
    }
    for c in [50u64, 50] {
        q.send(c, 2, &[1; MAX_LANES]);
    }
    assert_eq!(q.next_event(), Some(u128::from(lat) + 50), "an overrun before the drop to come");
    assert!(q.recv_ready(lat + 60, 1));
    assert_eq!(q.next_event(), Some(u128::from(lat) + 50), "the overrun is still the earlier event");
    assert!(q.arrivals(lat + 10).is_empty());
    assert_eq!(q.arrivals(lat + 50), vec![lat + 50]);
    assert_eq!(q.next_event(), Some(u128::from(lat) + 60), "the report of link 1 waits for its cycle");
    assert!(q.arrivals(lat + 59).is_empty());
    assert_eq!(q.arrivals(lat + 60), vec![lat + 60]);
    assert_eq!(q.next_event(), None);
    // an overrun and a report in the other order: the report is first
    let mut r = PlanLinks::loopback(&g);
    for c in [0u64, 0] {
        r.send(c, 1, &[1; MAX_LANES]);
    }
    for c in [40u64, 40] {
        r.send(c, 2, &[1; MAX_LANES]);
    }
    assert!(r.recv_ready(lat, 1));
    assert_eq!(r.next_event(), Some(u128::from(lat)));
}

// model: c_emulator/tsp_sim.c c2c_send (the model checks chip_id first: no peer, no send)
#[test]
fn a_send_on_an_unconnected_link_goes_nowhere() {
    let g = geom();
    let plan = Arc::new(LinkPlan::new(LinkMode::Nominal, &g, 2).unwrap());
    let mut p = PlanLinks::new(plan, 0, &g);
    assert_eq!(p.peer(15), None, "chip 0 of 2 has one connected link");
    p.send(0, 15, &[1; MAX_LANES]);
    let st = p.state();
    assert!(st.lock().unwrap().take_outbound().is_empty());
    p.send(0, 0, &[1; MAX_LANES]);
    assert_eq!(st.lock().unwrap().take_outbound().len(), 1);
}

/// : the per-tick scans of a chip's inbound FIFOs (`idle`, `next_event`) walk the profile's links, not the 128 allocated.
#[test]
fn the_inbound_scans_cover_the_links_of_the_profile() {
    let g = geom();
    assert_eq!(g.links, 16);
    let msg = |link: usize, i: u64| VecMsg {
        arrival: 1000 + u128::from(i),
        chip: 0,
        link,
        state: VState::Pending,
        taken_at: 0,
        ptype: 0,
        ctl: None,
        fec_c: 0,
        fec_u: 0,
        b: [0; MAX_LANES],
        poison: 0,
    };
    let overfull = |link: usize| {
        let plan = Arc::new(LinkPlan::new(LinkMode::Nominal, &g, 1).unwrap());
        let mut c = ChipLinks::new(plan, 0, &g);
        for i in 0..=g.rx_depth {
            c.deliver(msg(link, i));
        }
        c.next_event()
    };
    // a FIFO of the profile's links that must overrun is seen ...
    assert_eq!(overfull(15), Some(1000 + u128::from(g.rx_depth)));
    // ... one beyond them is not looked at (no link of the profile leads there: `link_peer` has no peer above `c2c.links`)
    assert_eq!(overfull(16), None);
}
