#!/usr/bin/env python3
"""The multi-chip regression corpus of the ISS (ISS side).

Each case is a multi-chip input for `tsp_sim` / `tsp-iss` (`C <chip>` sections of Q, S and D lines)
that sends vectors over the chip-to-chip links and receives them, the arguments it runs with and
the stdout and exit code the reference simulator gives. `tests/multichip_corpus.rs` runs the ISS on
every case and compares, so the corpus guards multi-chip behaviour without `tsp_sim`.

Files here:
  cases.tsv     name, profile (config/<profile>.json), arguments (with --chips), expected exit code,
                one case a line
  <name>.in     the input
  <name>.out    stdout of `tsp_sim <config> --trace <arguments> < <name>.in`

Regenerate the expectations from the reference simulator (inputs unchanged; each case is run three
times and must give the same stdout every time):

  source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
  TSP_SIM=/path/to/tsp_sim python3 gen_corpus.py

Rewrite the inputs as well (a seeded generator; the same seeds give the same files), then the
expectations:

  TSP_SIM=/path/to/tsp_sim python3 gen_corpus.py --make-inputs

`tsp_sim` is built by opentsp-arch-spec/tools/build_sim.sh. The specification is frozen, so the
expectations change only when this generator or the harness does.

The programs follow `test/c2c/c2c_directed.py`: a Transmit (the notification vector) or a Send (a
vector read from MEM) on a TX queue, the Receive on the peer chip around the arrival of the vector
(at several cycles when the latency is drawn from a seed), a Deskew or RuntimeDeskew now and then,
and on the nominal-latency cases a MEM Write that drains the received vector so it shows in the
trace and the dump.
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.normpath(os.path.join(HERE, "../../../../../../opentsp-arch-spec"))
sys.path.insert(0, os.path.join(SPEC, "tools"))
from tsp_as import lower_records, q_lines  # noqa: E402  the normative lowering (assembly.md sec. 4)

TX = (126, 127, 140, 141)
RX = (128, 129, 142, 143)
SEND, RECEIVE, TRANSMIT = 0, 1, 3


def c2c(op, chip=0, s=0, link=0):
    return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)


def rtdeskew(t):
    return (5 << 28) | (4 << 22) | t


DESKEW = c2c(2)


def mem(op, addr, s):
    return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)


def link_peer(chips, links, c, l):
    """c_emulator/tsp_sim.c link_peer: the far end (chip, link) of chip c's link l, or None."""
    if l >= links:
        return None
    if chips == 1:
        return (c, l)
    nn = 1 if chips <= 8 else -(-chips // 8)
    node, k = (0, c) if chips <= 8 else divmod(c, 8)
    nk = chips if chips <= 8 else min(8, chips - 8 * node)           # chips in this node
    if l <= 6:
        if l + 1 >= nk:
            return None
        return (node * 8 + (k + l + 1) % nk, nk - 2 - l)
    if l <= 10 and nn > 1:
        p = 4 * k + (l - 7)
        if p + 1 >= nn:
            return None
        m, q = (node + p + 1) % nn, nn - 2 - p
        peer = m * 8 + q // 4
        if peer >= chips:
            return None
        return (peer, 7 + q % 4)
    return None


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    west = slice_ < 44
    up = (row == "E") == west
    return 18 - q if up else 24 + q


# (name, profile, chips, arguments, events, seed, tight)
CASES = [
    ("ref_k2_nominal", "ref", 2, [], 6, 101, False),
    ("ref_k2_latency209", "ref", 2, ["--link-latency", "209"], 6, 102, False),
    ("ref_k2_latency228", "ref", 2, ["--link-latency", "228"], 6, 103, False),
    ("ref_k2_latency_seed", "ref", 2, ["--link-latency-seed", "7"], 5, 104, False),
    ("ref_k2_jitter_seed", "ref", 2, ["--link-jitter-seed", "3"], 5, 105, False),
    ("ref_k2_max_cycles_ok", "ref", 2, ["--max-cycles", "400"], 6, 106, False),
    ("ref_k2_max_cycles_stop", "ref", 2, ["--max-cycles", "300"], 6, 106, False),
    ("ref_k3_nominal", "ref", 3, [], 5, 107, False),
    ("ref_k5_latency_seed", "ref", 5, ["--link-latency-seed", "11"], 5, 108, False),
    ("ref_k8_nominal", "ref", 8, [], 8, 109, False),
    ("ref_k8_latency220", "ref", 8, ["--link-latency", "220"], 8, 110, False),
    ("ref_k8_latency_seed", "ref", 8, ["--link-latency-seed", "2"], 6, 111, False),
    ("ref_k8_jitter_seed", "ref", 8, ["--link-jitter-seed", "9"], 6, 112, False),
    ("ref_k16_nominal", "ref", 16, [], 8, 113, False),
    ("ref_k16_latency215", "ref", 16, ["--link-latency", "215"], 8, 114, False),
    ("ref_k16_latency_seed", "ref", 16, ["--link-latency-seed", "5"], 6, 115, False),
    ("ref_k16_jitter_seed", "ref", 16, ["--link-jitter-seed", "21"], 6, 116, False),
    ("mini_k2_nominal", "mini", 2, [], 8, 117, False),
    ("mini_k8_nominal", "mini", 8, [], 10, 118, False),
    ("mini_k16_jitter_seed", "mini", 16, ["--link-jitter-seed", "4"], 10, 119, False),
    ("tight_k2_burst", "test/link_tight", 2, [], 12, 120, True),
    ("tight_k8_burst", "test/link_tight", 8, ["--link-latency", "212"], 14, 121, True),
    ("tight_k16_jitter_seed", "test/link_tight", 16, ["--link-jitter-seed", "6"], 14, 122, True),
]


class Chip:
    """The records and SRAM image of one chip; a (queue, cycle) holds one instruction."""

    def __init__(self):
        self.recs, self.sram, self.dumps, self.used = [], {}, [], set()

    def add(self, cycle, queue, word):
        while (queue, cycle) in self.used:
            cycle += 1
        self.used.add((queue, cycle))
        self.recs.append((cycle, queue, 1, word, 0))
        return cycle

    def text(self, k, c, vl):
        out = [f"C {c:x}"] if k > 1 else []
        out += [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in self.sram.items()]
        out += q_lines(lower_records(self.recs))
        out += [f"D {sl:x} {a:x}" for sl, a in self.dumps]
        return out


def make_case(spec, cfg):
    name, profile, k, args, n_events, seed, tight = spec
    r = random.Random(seed)
    links = cfg["c2c"]["links"]
    vl = 16 * cfg["geometry"]["superlanes"]
    tx_gap, rx_gap = cfg["layout"]["c2c_tx_gap"], cfg["layout"]["c2c_rx_gap"]
    d_mem, d_c2c = cfg["timing"]["mem"]["d_func"], cfg["timing"]["c2c"]["d_func"]
    nominal = cfg["c2c"]["link_latency"]
    fixed = None
    if "--link-latency" in args:
        fixed = int(args[args.index("--link-latency") + 1])
    drawn = "--link-latency-seed" in args or "--link-jitter-seed" in args
    lat = fixed or nominal
    chips = [Chip() for _ in range(k)]
    valid = [(c, l) for c in range(k) for l in range(links) if link_peer(k, links, c, l)]
    if tight:
        valid = [(c, l) for c, l in valid if l < 2]
    for ev in range(n_events):
        c, l = valid[r.randrange(len(valid))]
        glob = [(c2, l2) for c2, l2 in valid if l2 >= 7]
        if k == 16 and ev % 2 == 0 and glob:
            c, l = glob[r.randrange(len(glob))]
        peer, pl = link_peer(k, links, c, l)
        t = 40 + r.randrange(40)
        txq = TX[r.randrange(4)]
        rxq = RX[r.randrange(4)]
        burst = r.randrange(2, 5) if tight else 1
        data = not tight and r.random() < 0.4
        sender, receiver = chips[c], chips[peer]
        stream = 1 + ev
        for b in range(burst):
            tb = t + b
            if data and b == 0:
                # a vector read from MEM onto the TX queue's row, one value per lane
                row = "W" if txq in (126, 127) else "E"
                slice_ = (44 if row == "W" else 0) + r.randrange(4)
                addr = 16 + ev
                val = bytes((31 * ev + 7 * ln + 3) & 0xFF for ln in range(vl))
                sender.sram[(slice_, addr)] = val
                reader_cycle = tb
                sender.add(reader_cycle - d_mem - (tx_gap - gap_mem(slice_, row)), slice_,
                           mem(0, addr, (32 if row == "W" else 0) + stream))
                sender.add(tb, txq, c2c(SEND, peer, (32 if row == "W" else 0) + stream, l))
            else:
                sender.add(tb, txq, c2c(TRANSMIT, peer, 0, l))
        rrow = "E" if rxq in (128, 129) else "W"
        rs = (32 if rrow == "W" else 0) + stream
        if drawn:
            # the latency is drawn per link (or per vector): try the receive at several cycles
            for step in range(0, 24, 4):
                receiver.add(t + 209 + step, rxq, c2c(RECEIVE, c, rs, pl))
        elif tight:
            if r.random() < 0.6:
                for b in range(burst):
                    receiver.add(t + lat + b * 3 - r.randrange(2), rxq, c2c(RECEIVE, c, rs, pl))
        else:
            at = receiver.add(t + lat, rxq, c2c(RECEIVE, c, rs, pl))
            if r.random() < 0.3:
                receiver.add(at - 1 - r.randrange(2), rxq, c2c(RECEIVE, c, rs, pl))  # before arrival
            if data:
                # drain the received vector to MEM: the slice nearest the RX row's writer
                dslice = (0 if rrow == "E" else 44) + r.randrange(4)
                daddr = 200 + ev
                g = gap_mem(dslice, rrow)
                receiver.add(at + d_c2c + (g - rx_gap), dslice, mem(1, daddr, rs))
                receiver.dumps.append((dslice, daddr))
        if r.random() < 0.2:
            sender.add(t + 1 + r.randrange(30), txq, DESKEW)
        if r.random() < 0.15:
            sender.add(t + 1 + r.randrange(10), txq, rtdeskew(r.randrange(30)))
    text = []
    for c, chip in enumerate(chips):
        text += chip.text(k, c, vl)
    return "\n".join(text) + "\n"


def run_sim(sim, cfg_path, args, text):
    p = subprocess.run([sim, cfg_path, "--trace", *args], input=text.encode(), capture_output=True, timeout=600)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-inputs", action="store_true", help="rewrite the inputs and cases.tsv as well")
    ap.add_argument("--tsp-sim", default=os.environ.get("TSP_SIM"), help="the reference simulator (default $TSP_SIM)")
    a = ap.parse_args()
    if not a.tsp_sim or not os.path.exists(a.tsp_sim):
        sys.exit("gen_corpus.py: set TSP_SIM (or --tsp-sim) to a built tsp_sim")
    cfgs = {}
    for spec in CASES:
        profile = spec[1]
        cfgs.setdefault(profile, (os.path.join(SPEC, "config", profile + ".json"),))
    rows = []
    if a.make_inputs:
        for spec in CASES:
            cfg = json.load(open(cfgs[spec[1]][0]))
            with open(os.path.join(HERE, spec[0] + ".in"), "w") as f:
                f.write(make_case(spec, cfg))
    total = 0
    for name, profile, k, args, _, _, _ in CASES:
        args = args + ["--chips", str(k)]
        text = open(os.path.join(HERE, name + ".in")).read()
        cfg_path = cfgs[profile][0]
        runs = [run_sim(a.tsp_sim, cfg_path, args, text) for _ in range(3)]
        if any(x != runs[0] for x in runs):
            sys.exit(f"gen_corpus.py: {name}: tsp_sim is not deterministic on this input")
        code, out = runs[0]
        arrivals = sum(1 for l in out.decode().splitlines() if l.startswith("T srf") and int(l.split()[2]) >= 200)
        if code != 3 and arrivals < 2:  # exit 3: the run was stopped by --max-cycles, nothing is printed
            sys.exit(f"gen_corpus.py: {name}: only {arrivals} stream writes after cycle 200: no link traffic")
        with open(os.path.join(HERE, name + ".out"), "wb") as f:
            f.write(out)
        rows.append((name, profile, " ".join(args), code))
        total += len(out) + len(text)
        print(f"{name}: exit {code}, {len(out)} bytes out, {arrivals} stream writes after cycle 200")
    with open(os.path.join(HERE, "cases.tsv"), "w") as f:
        f.write("# name\tprofile\targuments\texit code (see gen_corpus.py)\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"{len(rows)} cases, {total} bytes")


if __name__ == "__main__":
    main()
