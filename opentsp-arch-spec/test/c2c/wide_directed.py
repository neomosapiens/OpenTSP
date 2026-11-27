"""Directed test of the wide chip and link numbers and of the topology file (`tsp_sim --topology`, `tsp-iss --topology`):
chip ids above 63 and links above 15 in the words of Send and Receive (bits 8..6 of `chip_id` in bits 2..0, bits 6..4 of `link` in bits 5..3),
the wiring of `--chips K` for a K that is not 8 N, and the errors of a bad topology file. Every run goes through `tsp_sim` and `tsp-iss`
and the two outputs must be identical (with `-` in place of the ISS only `tsp_sim` runs, as in `model_test.sh`). The last section runs a 264 chip
system (33 nodes) on the ISS alone: two transfers over global links whose chips have `chip_id` above 63.

One vector crosses one link: the sending chip's east TX queue sends the word that a west slice read onto E:0, the receiving chip's west RX queue
takes it at the arrival cycle and a MEM Write stores it. The test checks the stored word, that no chip reports a diagnostic, and the cycle of
the arrival (the latency of that link).
usage: wide_directed.py <tsp_sim> <tsp-iss or -> <config.json> <config.json with 24 links>"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM, ISS, CFG, WIDE = sys.argv[1:5]
HAVE_ISS = ISS != "-"
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def setup(cfg_path):
    cfg = json.load(open(cfg_path))
    return dict(cfg=cfg, L=cfg["c2c"]["link_latency"], TX=cfg["layout"]["c2c_tx_gap"], RX=cfg["layout"]["c2c_rx_gap"],
                DC=cfg["timing"]["c2c"]["d_func"], DM=cfg["timing"]["mem"]["d_func"], VL=16 * cfg["geometry"]["superlanes"])


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def c2c(op, chip=0, s=0, link=0):
    """The 32-bit word of Send (0) or Receive (1): the low 6 bits of the chip and 4 of the link in their own fields, the rest in the low bits."""
    return ((5 << 28) | (op << 22) | ((chip & 63) << 16) | (s << 10) | ((link & 15) << 6) | (((link >> 4) & 7) << 3) | ((chip >> 6) & 7))
SEND, RECEIVE = 0, 1
WRX, ETX = 128, 140


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q


def hop(P, src, ls, dst, ld, lat, chip_of_src=None, chip_of_dst=None):
    """The programs of one transfer: {chip: (records, sram, dumps)}. The word names `chip_of_src` as the chip at the far end (the test of a wrong id)."""
    val = bytes((7 * l + 3) & 0xFF for l in range(P["VL"]))
    T0 = P["DM"] + P["TX"] - gap_mem(0, "E") + 1
    a = [(T0 - P["DM"] - (P["TX"] - gap_mem(0, "E")), 0, 1, mem(0, 1, 0), 0),
         (T0, ETX, 1, c2c(SEND, dst if chip_of_dst is None else chip_of_dst, 0, ls), 0)]
    rc = T0 + lat
    b = [(rc, WRX, 1, c2c(RECEIVE, src if chip_of_src is None else chip_of_src, 0, ld), 0),
         (rc + P["DC"] + (gap_mem(0, "E") - P["RX"]), 0, 1, mem(1, 20, 0), 0)]
    return {src: (a, {(0, 1): val}, []), dst: (b, {}, [(0, 20)])}, val, T0, rc


def text(progs, K):
    t = []
    for k in range(K):
        recs, sram, dumps = progs.get(k, ([], {}, []))
        t += [f"C {k:x}"]
        t += [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
        t += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    return "\n".join(t) + "\n"


def run(tool, cfg, args, progs, K):
    p = subprocess.run([tool, cfg, "--trace"] + args, input=text(progs, K), capture_output=True, text=True, timeout=1200)
    return p


def words_of(out, VL):
    parts, cur = {}, None
    for l in out.splitlines():
        if l.startswith("C "): cur = int(l.split()[1]); parts[cur] = []
        elif cur is not None: parts[cur].append(l)
    res = {}
    for k, part in parts.items():
        res[k] = ([l for l in part if l.startswith("T diag")],
                  {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in part) if t and t[0] == "W"})
    return res


def both(cfg, args, progs, K, P, what):
    """Run both tools; the outputs must be identical. The decoded result of the receiving chips is returned."""
    a = run(SIM, cfg, args, progs, K)
    if HAVE_ISS:
        b = run(ISS, cfg, args, progs, K)
        check(f"{what}: tsp_sim and tsp-iss agree (exit code and every output line)", (a.returncode, a.stdout) == (b.returncode, b.stdout),
              (a.returncode, b.returncode, a.stderr[:200], b.stderr[:200]))
    return a, words_of(a.stdout, P["VL"]) if a.returncode == 0 else {}


def both_cli(args):
    """The same arguments and no input on tsp_sim and tsp-iss (with no ISS, the ISS answer is the simulator's with its name replaced)."""
    a = subprocess.run([SIM, CFG] + args, input="", capture_output=True, text=True)
    if not HAVE_ISS:
        class R: pass
        b = R(); b.returncode = a.returncode; b.stderr = a.stderr.replace("tsp_sim: ", "tsp-iss: ", 1); b.stdout = a.stdout
        return a, b
    return a, subprocess.run([ISS, CFG] + args, input="", capture_output=True, text=True)


def write_topo(d, name, obj):
    p = os.path.join(d, name)
    open(p, "w").write(json.dumps(obj))
    return p


Pm = setup(CFG); Pw = setup(WIDE)
with tempfile.TemporaryDirectory() as d:
    # 1. Links above 15: a topology with 24 links per chip joins chip 0 link 20 to chip 1 link 17 (latency 210) and chip 1 link 23 to chip 2 link 5.
    topo = write_topo(d, "wide.json", {"chips": 3, "links": [{"a": [0, 20], "b": [1, 17], "latency": 210}, {"a": [1, 23], "b": [2, 5]}]})
    progs, val, T0, rc = hop(Pw, 0, 20, 1, 17, 210)
    r, res = both(WIDE, ["--topology", topo], progs, 3, Pw, "link 20 of chip 0 to link 17 of chip 1")
    check(f"the vector crosses link 20 (latency 210: sent at {T0}, received at {rc}), is stored and no chip reports a diagnostic",
          r.returncode == 0 and res.get(1, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res), r.stdout[-300:] + r.stderr[:200])
    progs, val, T0, rc = hop(Pw, 1, 23, 2, 5, Pw["L"])
    r, res = both(WIDE, ["--topology", topo], progs, 3, Pw, "link 23 of chip 1 to link 5 of chip 2 (the default latency)")
    check("the vector crosses link 23 and arrives after the profile's link latency", res.get(2, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res))
    # the receive one cycle early finds nothing: the latency of that link is 210, not the default
    progs, val, T0, rc = hop(Pw, 0, 20, 1, 17, 210)
    progs[1][0][0] = (rc - 1,) + progs[1][0][0][1:]
    r, res = both(WIDE, ["--topology", topo], progs, 3, Pw, "a Receive one cycle before the arrival over link 17")
    check("a Receive one cycle before the arrival is OTS-E-TIM-001 (the link's own latency applies)",
          any("TIM-001" in x for x in res.get(1, ([], {}))[0]) and res[1][1].get((0, 20)) != val, r.stdout[-200:])
    # the same word on a profile with 16 links names no link (OTS-E-OPR-001), also with a link number that needs the upper bits
    progs, val, T0, rc = hop(Pm, 0, 20, 1, 17, Pm["L"])
    r, res = both(CFG, ["--chips", "2"], progs, 2, Pm, "a link number above c2c.links - 1 on a 16-link profile")
    check("link 20 of a profile with 16 links is OTS-E-OPR-001 on the sender", any("OPR-001" in x for x in res.get(0, ([], {}))[0]), r.stdout[-200:])

    # 2. Chip ids above 63: 66 chips, chip 0 link 3 joined to chip 65 link 4.
    topo = write_topo(d, "c66.json", {"chips": 66, "links": [{"a": [0, 3], "b": [65, 4]}]})
    progs, val, T0, rc = hop(Pm, 0, 3, 65, 4, Pm["L"])
    r, res = both(CFG, ["--topology", topo], progs, 66, Pm, "chip 0 to chip 65")
    check("a vector is sent to chip 65 (chip_id 0x41: bit 6 in the low bits of the word) and stored there", res.get(65, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res), r.stdout[-200:] + r.stderr[:200])
    progs, val, T0, rc = hop(Pm, 0, 3, 65, 4, Pm["L"], chip_of_dst=1)       # 1 is chip 65 without the upper bit: not the peer of link 3
    r, res = both(CFG, ["--topology", topo], progs, 66, Pm, "the word names chip 1, the link reaches chip 65")
    check("chip 1 is not the chip at the far end of link 3 (65 without bit 6): OTS-E-OPR-001 on the sender", any("OPR-001" in x for x in res.get(0, ([], {}))[0]) and res[65][1].get((0, 20)) != val, r.stdout[-200:])

    # 3. --chips 12: a node of 8 and a node of 4; the global link 7 of chip 0 reaches chip 8 on link 7, local link 0 of chip 8 reaches chip 9 on link 2.
    progs, val, T0, rc = hop(Pm, 0, 7, 8, 7, Pm["L"])
    r, res = both(CFG, ["--chips", "12"], progs, 12, Pm, "--chips 12: global link 7 of chip 0")
    check("--chips 12 joins chip 0 to chip 8 over global link 7", res.get(8, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res), r.stdout[-200:] + r.stderr[:200])
    progs, val, T0, rc = hop(Pm, 8, 0, 9, 2, Pm["L"])
    r, res = both(CFG, ["--chips", "12"], progs, 12, Pm, "--chips 12: local link 0 of chip 8")
    check("--chips 12: in the node of 4 chips, local link 0 of chip 8 reaches chip 9 on link 2", res.get(9, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res))
    progs, val, T0, rc = hop(Pm, 0, 8, 9, 2, Pm["L"])
    r, res = both(CFG, ["--chips", "12"], progs, 12, Pm, "--chips 12: a global link that has no peer")
    check("--chips 12: global link 8 of chip 0 reaches a chip that does not exist: OTS-E-OPR-001", any("OPR-001" in x for x in res.get(0, ([], {}))[0]))

    # 3b. The generators of tools/gen_topology.py: dragonfly 12 is the wiring of --chips 12; a torus and a tray join the links they name.
    gen = lambda *a: subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "gen_topology.py"), *a], capture_output=True, text=True, check=True).stdout
    for name, args in (("dragonfly 12", ["dragonfly", "12"]), ("node8", ["node8"]), ("torus 3", ["torus", "3"]), ("tray8", ["tray8"])):
        open(os.path.join(d, "g.json"), "w").write(gen(*args))
    open(os.path.join(d, "g12.json"), "w").write(gen("dragonfly", "12"))
    progs, val, T0, rc = hop(Pm, 0, 7, 8, 7, Pm["L"])
    a1 = run(SIM, CFG, ["--topology", os.path.join(d, "g12.json")], progs, 12); a2 = run(SIM, CFG, ["--chips", "12"], progs, 12)
    check("the generated dragonfly 12 and --chips 12 give the same output from tsp_sim", a1.returncode == 0 and a1.stdout == a2.stdout, (a1.stderr[:200], a2.stderr[:200]))
    if HAVE_ISS:
        b1 = run(ISS, CFG, ["--topology", os.path.join(d, "g12.json")], progs, 12)
        check("tsp-iss gives that output for the generated topology as well", a1.stdout == b1.stdout, b1.stderr[:200])
    topo = write_topo(d, "torus.json", json.loads(gen("torus", "3")))
    progs, val, T0, rc = hop(Pm, 0, 1, 1, 4, Pm["L"])
    r, res = both(CFG, ["--topology", topo], progs, 9, Pm, "torus 3: east link 1 of chip 0")
    check("torus 3: east copy 1 of chip 0 (link 1) reaches the west copy 1 (link 4) of chip 1", res.get(1, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res), r.stdout[-200:])
    progs, val, T0, rc = hop(Pm, 0, 11, 3, 8, Pm["L"])
    r, res = both(CFG, ["--topology", topo], progs, 9, Pm, "torus 3: south link 2 of chip 0")
    check("torus 3: south copy 2 of chip 0 (link 11) reaches the north copy 2 (link 8) of chip 3, the chip below", res.get(3, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res), r.stdout[-200:])
    topo = write_topo(d, "tray.json", json.loads(gen("tray8")))
    progs, val, T0, rc = hop(Pm, 7, 1, 0, 3, Pm["L"])
    r, res = both(CFG, ["--topology", topo], progs, 8, Pm, "tray8: link 1 of chip 7")
    check("tray8: link 1 of chip 7 reaches link 3 of chip 0 (the ring closes)", res.get(0, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res), r.stdout[-200:])

    # 3c. A relay over the boundary of the two nodes of --chips 12 (chip 0 -> chip 8 over global link 7 -> chip 9 over local link 0 of chip 8): software routing.
    #     The cost of the hop is the one of relay_directed.py: the link latency plus d_func(c2c) + (c2c_tx_gap - c2c_rx_gap) at the relay chip.
    L, HOP = Pm["L"], Pm["DC"] + Pm["TX"] - Pm["RX"]
    val = bytes((7 * l + 3) & 0xFF for l in range(Pm["VL"]))
    T0 = Pm["DM"] + Pm["TX"] - gap_mem(0, "E") + 1
    rb = T0 + L; sb = rb + HOP; rc = sb + L
    a_ = [(T0 - Pm["DM"] - (Pm["TX"] - gap_mem(0, "E")), 0, 1, mem(0, 1, 0), 0), (T0, ETX, 1, c2c(SEND, 8, 0, 7), 0)]
    b_ = [(rb, WRX, 1, c2c(RECEIVE, 0, 0, 7), 0), (sb, ETX, 1, c2c(SEND, 9, 0, 0), 0)]
    c_ = [(rc, WRX, 1, c2c(RECEIVE, 8, 0, 2), 0), (rc + Pm["DC"] + (gap_mem(0, "E") - Pm["RX"]), 0, 1, mem(1, 20, 0), 0)]
    progs = {0: (a_, {(0, 1): val}, []), 8: (b_, {}, []), 9: (c_, {}, [(0, 20)])}
    r, res = both(CFG, ["--chips", "12"], progs, 12, Pm, "relay 0 -> 8 -> 9 on --chips 12")
    check(f"the relay over two nodes delivers the vector at chip 9: Send at {T0}, Receive at {rc} = {rc - T0} = 2 x {L} + {HOP} cycles later, no diagnostic",
          res.get(9, ([], {}))[1].get((0, 20)) == val and all(res[k][0] == [] for k in res) and rc - T0 == 2 * L + HOP, r.stdout[-200:] + r.stderr[:200])

    # 4. The errors of a topology file: the same message and exit code from both tools (the program name differs).
    bad = [
        ("a chip out of range", {"chips": 2, "links": [{"a": [0, 0], "b": [2, 0]}]}, "chip: 2 is outside 0..1"),
        ("a link out of range", {"chips": 2, "links": [{"a": [0, 16], "b": [1, 0]}]}, "link: 16 is outside 0..15"),
        ("an end used twice", {"chips": 3, "links": [{"a": [0, 0], "b": [1, 0]}, {"a": [0, 0], "b": [2, 0]}]}, "chip 0 link 0 is joined twice"),
        ("an unknown key", {"chips": 2, "links": [], "jitter": 1}, 'unknown key "jitter"'),
        ("no chips", {"links": []}, '"chips" is missing'),
        ("too many chips", {"chips": 513, "links": []}, "chips: 513 is outside 1..512"),
        ("a latency outside the profile's range", {"chips": 2, "links": [{"a": [0, 0], "b": [1, 0], "latency": 5}]}, "latency: 5 is outside"),
        ("a cycles_per_vector that is not the profile's", {"chips": 2, "links": [], "cycles_per_vector": 7}, "is not the profile's c2c.cycles_per_vector"),
    ]
    for name, obj, frag in bad:
        t = write_topo(d, "bad.json", obj)
        a, b = both_cli(["--topology", t])
        ma = a.stderr.replace("tsp_sim: ", "", 1); mb = b.stderr.replace("tsp-iss: ", "", 1)
        check(f"a topology with {name}: exit 2 and '{frag}' from both tools", a.returncode == b.returncode == 2 and frag in ma and ma == mb, (ma, mb, a.returncode, b.returncode))
    t = write_topo(d, "ok.json", {"chips": 2, "links": []})
    a, b = both_cli(["--topology", t, "--chips", "2"])
    check("--topology with --chips is refused by both tools with the same message", a.returncode == b.returncode == 2 and a.stderr.replace("tsp_sim: ", "", 1) == b.stderr.replace("tsp-iss: ", "", 1) and "do not combine" in a.stderr, (a.stderr, b.stderr))
    a, b = both_cli(["--topology", os.path.join(d, "none.json")])
    check("a topology file that does not exist is exit 2 from both tools", a.returncode == b.returncode == 2 and "none.json" in a.stderr and "none.json" in b.stderr, (a.stderr, b.stderr))
    open(os.path.join(d, "notjson"), "w").write("{")
    a, b = both_cli(["--topology", os.path.join(d, "notjson")])
    check("a topology file that is not JSON is exit 2 with the same message", a.returncode == b.returncode == 2 and a.stderr.replace("tsp_sim: ", "", 1) == b.stderr.replace("tsp-iss: ", "", 1) and "not a JSON object" in a.stderr, (a.stderr, b.stderr))
    # : trailing text after the object and a repeated key are refused by both tools (the C parser took the text and the first key)
    for name, raw, frag in (("trailing text", '{"chips": 2, "links": []} x', "not a JSON object"),
                             ("unknown and duplicate link keys", '{"chips":1,"links":[{"z":1,"a":[0,0],"a":[0,0],"b":[0,0]}]}', 'links[0]: unknown key "z"'),
                             ("outer and nested duplicate keys", '{"chips":1,"links":[{"a":[0,0],"a":[0,0],"b":[0,0]}],"chips":1}', 'duplicate key "chips"'),
                             ("an unknown outer key and nested duplicate", '{"chips":1,"unknown":0,"links":[{"a":[0,0],"a":[0,0],"b":[0,0]}]}', 'unknown key "unknown"'),
                             ("unknown keys in file order", '{"chips":1,"z":0,"a":0,"links":[]}', 'unknown key "z"'),
                             ("NUL hidden by a duplicate key", r'{"chips":"\u0000","chips":1,"links":[]}', "not a JSON object"),
                             ("NUL before trailing bytes", '{"chips": 2, "links": []}\0x', "not a JSON object"),
                             ("an escaped NUL in a top-level key", r'{"chips\u0000suffix": 2, "links": []}', "not a JSON object"),
                             ("an escaped NUL in a link key", r'{"chips": 2, "links": [{"a\u0000suffix": [0, 0], "b": [1, 0]}]}', "not a JSON object"),
                             ("an escaped NUL in a value", r'{"chips": "2\u0000suffix", "links": []}', "not a JSON object"),
                             ("a leading-zero number", '{"chips": 02, "links": []}', "not a JSON object"),
                             ("non-JSON whitespace", '{"chips": 2,\v"links": []}', "not a JSON object"),
                             ("a repeated chips key", '{"chips": 2, "chips": 3, "links": []}', 'duplicate key "chips"'),
                             ("a repeated key of a link", '{"chips": 2, "links": [{"a": [0, 0], "b": [1, 0], "b": [1, 1]}]}', 'duplicate key "b"')):
        open(os.path.join(d, "raw.json"), "w").write(raw)
        a, b = both_cli(["--topology", os.path.join(d, "raw.json")])
        ma = a.stderr.replace("tsp_sim: ", "", 1); mb = b.stderr.replace("tsp-iss: ", "", 1)
        check(f"a topology with {name}: exit 2 and '{frag}' from both tools", a.returncode == b.returncode == 2 and frag in ma and ma == mb, (ma, mb, a.returncode, b.returncode))
    for k in (265, 0):
        a, b = both_cli(["--chips", str(k)])
        check(f"--chips {k} is refused by both tools with the same message", a.returncode == b.returncode == 2 and a.stderr.replace("tsp_sim: ", "", 1) == b.stderr.replace("tsp-iss: ", "", 1) and "--chips must be 1..264" in a.stderr, (a.stderr, b.stderr))

# 5. A system of 264 chips (33 nodes of 8: the largest wiring of ISCA22 sec. 2.2), on the ISS alone (264 processes of tsp_sim are not run).
#    Two transfers over global links: the first that leaves chip 0 (link 7), and the last pair of chips (the highest chip id, which has bits 8..6 = 4).
if HAVE_ISS:
    with tempfile.TemporaryDirectory() as d:
        t264 = json.loads(gen("dragonfly", "264"))
        ends = {tuple(e["a"]): tuple(e["b"]) for e in t264["links"]}
        ends.update({tuple(e["b"]): tuple(e["a"]) for e in t264["links"]})
        far = max(k for k in ends if k[0] >= 256 and k[1] >= 7)            # a global link of a chip with chip_id >= 256
        pairs = [((0, 7), ends[(0, 7)]), (far, ends[far])]
        progs = {}
        for (sc, sl), (dc, dl) in pairs:
            pr, val, T0, rc = hop(Pm, sc, sl, dc, dl, Pm["L"])
            progs.update(pr)
        r = run(ISS, CFG, ["--chips", "264"], progs, 264)
        res = words_of(r.stdout, Pm["VL"]) if r.returncode == 0 else {}
        check(f"264 chips on the ISS: chip 0 link 7 to chip {pairs[0][1][0]} link {pairs[0][1][1]} and chip {pairs[1][0][0]} link {pairs[1][0][1]} to chip {pairs[1][1][0]} link {pairs[1][1][1]}: both vectors arrive, no diagnostic on any chip",
              r.returncode == 0 and len(res) == 264 and all(res[dc][1].get((0, 20)) == val and res[dc][0] == [] for _, (dc, _) in pairs) and all(res[k][0] == [] for k in res),
              r.stderr[:300] + r.stdout[-200:])
        # the same system from the generated topology file
        topo = write_topo(d, "t264.json", t264)
        r2 = run(ISS, CFG, ["--topology", topo], progs, 264)
        check("the generated topology file of the 264 chips gives the output of --chips 264", r2.returncode == 0 and r2.stdout == r.stdout, r2.stderr[:300])

print("wide: all pass" if ok else "wide: FAILURES")
sys.exit(0 if ok else 1)
