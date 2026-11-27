#!/usr/bin/env python3
"""Write a topology file for `tsp_sim --topology`, `tsp-iss --topology` and `ots-asm mpack --topology` (topology.md sec. 7).

usage: gen_topology.py node8
       gen_topology.py dragonfly K       the wiring of `--chips K` (K = 1..264): nodes of 8 chips, the last may hold fewer
       gen_topology.py torus R           R x R chips in a two-dimensional torus (R >= 3), each pair of neighbours joined by three
                                         parallel links (ISCA22 sec. 4.4): link 3 d + t of a chip, direction d = east, west, north,
                                         south, copy t = 0..2; chip y R + x; needs a profile with at least 12 links
       gen_topology.py tray8             8 chips in a ring, each neighbour pair joined by two parallel links (links 0, 1 to the
                                         next chip, 2, 3 to the previous one); needs at least 4 links
Options: --latency N (the default latency of the file), --jitter-seed S, --cycles-per-vector N. The JSON goes to standard output.

The shapes other than the node and the group level are opentsp choices: the published sources name the radix-8 torus and the
triple-connected links (ISCA22 sec. 4.4) and give no link numbering."""
import json, sys


def dragonfly_peer(K, c, l):
    """The far end (chip, link) of chip c's link l in the wiring of `tsp_sim --chips K`, or None (c_emulator/tsp_sim.c dragonfly_peer)."""
    if K == 1:
        return (c, l)
    nn = 1 if K <= 8 else (K + 7) // 8
    node, k = (0, c) if K <= 8 else (c // 8, c % 8)
    nk = K if K <= 8 else min(8, K - 8 * node)
    if l <= 6:
        if l >= nk - 1:
            return None
        return (node * 8 + (k + l + 1) % nk, nk - 2 - l)
    if l <= 10 and nn > 1:
        p = 4 * k + (l - 7)
        if p >= nn - 1:
            return None
        m, q = (node + p + 1) % nn, nn - 2 - p
        peer = m * 8 + q // 4
        if peer >= K:
            return None
        return (peer, 7 + q % 4)
    return None


def dragonfly(K, links=16):
    if not 1 <= K <= 264:
        raise SystemExit("gen_topology: dragonfly K needs K = 1..264")
    out = []
    for c in range(K):
        for l in range(links):
            p = dragonfly_peer(K, c, l)
            if p is not None and (c, l) <= p:
                out.append({"a": [c, l], "b": list(p)})
    return K, out


def torus(R):
    if R < 3:
        raise SystemExit("gen_topology: torus R needs R >= 3 (two neighbours in one direction would be the same chip)")
    out = []
    for y in range(R):
        for x in range(R):
            c, east, south = y * R + x, y * R + (x + 1) % R, ((y + 1) % R) * R + x
            for t in range(3):
                out.append({"a": [c, t], "b": [east, 3 + t]})          # east link t to the west link t of the east neighbour
                out.append({"a": [c, 9 + t], "b": [south, 6 + t]})     # south link t to the north link t of the south neighbour
    return R * R, out


def tray8():
    out = []
    for i in range(8):
        for t in range(2):
            out.append({"a": [i, t], "b": [(i + 1) % 8, 2 + t]})
    return 8, out


def main(argv):
    opts, args, i = {}, [], 1
    while i < len(argv):
        if argv[i] in ("--latency", "--jitter-seed", "--cycles-per-vector") and i + 1 < len(argv):
            opts[argv[i]] = int(argv[i + 1]); i += 2
        else:
            args.append(argv[i]); i += 1
    try:
        if args == ["node8"]:
            chips, links = dragonfly(8)
        elif len(args) == 2 and args[0] == "dragonfly":
            chips, links = dragonfly(int(args[1]))
        elif len(args) == 2 and args[0] == "torus":
            chips, links = torus(int(args[1]))
        elif args == ["tray8"]:
            chips, links = tray8()
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except ValueError:
        print(__doc__, file=sys.stderr)
        return 2
    t = {"chips": chips}
    if "--latency" in opts:
        t["latency"] = opts["--latency"]
    if "--jitter-seed" in opts:
        t["jitter_seed"] = opts["--jitter-seed"]
    if "--cycles-per-vector" in opts:
        t["cycles_per_vector"] = opts["--cycles-per-vector"]
    t["links"] = links
    print(json.dumps(t))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
