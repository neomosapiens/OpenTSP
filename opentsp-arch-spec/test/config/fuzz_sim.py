"""Fuzz tsp_sim: no input may crash it.

Each iteration builds a program (queue text, Q lines) from the catalogue's encodings, and then one of:
  valid    a random configuration inside config/schema.json, the program must run;
  mutated  a shipped profile with random keys replaced by hostile values;
  garbage  a shipped profile with corrupted program lines or corrupted configuration text.
Allowed outcomes: exit 0; exit 2 (input error) with one line on stderr, only where the input
can be invalid; exit 3 (over the cycle budget). Anything else is a failure: a signal, exit 1
(a Sail assert or pattern failure), exit 4 (a model or harness invariant), or a timeout. Each
failing input is saved in <outdir>.

usage: fuzz_sim.py <tsp_sim> <config-dir> <encoding.json> <seconds> <seed> <jobs> <outdir>
"""
import concurrent.futures as cf, copy, json, os, random, re, subprocess, sys, tempfile, time

sim, cfgdir, encpath, seconds, seed, jobs, outdir = sys.argv[1:8]
seconds, seed, jobs = int(seconds), int(seed), int(jobs)
enc = json.load(open(encpath))
PATS = [(i["width"], i["match"]) for i in (enc["instructions"] if "instructions" in enc else enc)]
PROFILES = [json.load(open(os.path.join(cfgdir, n))) for n in ("ref.json", "mini.json")]
HOSTILE = [-1, 0, 1, 2, 3, 15, 16, 43, 44, 63, 64, 127, 128, 255, 256, 4095, 4096, 65535, 65536, 2**31, 2**63, 2**70,
           -2**63, 1.5, 1e3, "5", "", None, True, [], {}]
GAPS = ["vxm_gap", "mem_up_gap_base", "mem_down_gap_base", "mxm_in_gap", "mxm_acc_gap", "sxm_dst_gap",
        "sxm_turn_in_gap", "sxm_turn_out_gap", "c2c_tx_gap", "c2c_rx_gap"]


def leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    else:
        yield path


def put(cfg, path, value):
    for k in path[:-1]:
        cfg = cfg[k]
    cfg[path[-1]] = value


def valid_config(r):
    """A random configuration the schema accepts, biased to the bounds."""
    c = copy.deepcopy(r.choice(PROFILES))
    c["geometry"]["superlanes"] = r.choice([1, 2, 3, 4, 8, 20, 21, 31, 32, r.randrange(1, 33)])   # any count from 1 to 32
    # Any power of two from 256 to 8192 (bank_bit = log2 - 1), any quad count, any row count
    w = r.choice([256, 512, 1024, 2048, 4096, 8192])
    c["sram"]["words_per_slice"], c["sram"]["bank_bit"] = w, w.bit_length() - 2
    c["geometry"]["mem_slices_per_hemisphere"] = r.choice([4, 8, 12, 24, 40, 44, 4 * r.randrange(1, 12)])
    c["mxm"]["rows"] = r.choice([16, 32, 160, 304, 320, 16 * r.randrange(1, 21)])
    c["mxm"]["f16_rows"] = r.choice([c["mxm"]["rows"] // 2, c["mxm"]["rows"]])      # between half the rows and all of them
    quads = c["geometry"]["mem_slices_per_hemisphere"] // 4
    def row(lo):
        # d_skew < d_func outside the CU (schema_invariants.py D_FUNC_MIN_CLASSES, ,
        # and the model's timing table check)
        f = r.choice([lo, 1, 5, 20, 63, r.randrange(lo, 64)])
        sk = r.choice([0, 1, 15, r.randrange(16)])
        return f, (sk if lo == 0 else min(sk, f - 1))
    for k, t in c["timing"].items():
        if k != "overrides" and isinstance(t, dict):
            t["d_func"], t["d_skew"] = row(0 if k == "cu" else 1)
    sl = c["geometry"]["superlanes"]
    c["timing"]["sxm_cross"]["d_func"] = max(c["timing"]["sxm_cross"]["d_func"], sl + c["timing"]["sxm_cross"]["d_skew"])
    c["timing"]["mxm_abc"]["d_func"] = max(c["timing"]["mxm_abc"]["d_func"], sl + c["timing"]["mxm_abc"]["d_skew"])
    # the scalar and per-class keys of the timing basis (schema: sxm_map_ready 1..63, occupancy_f16 1..ABC d_func, rtdeskew_floor 1..65535)
    c["timing"]["sxm_map_ready"] = r.choice([sl, max(sl, 20), 63, r.randrange(sl, 64)])
    c["timing"]["mxm_abc"]["occupancy_f16"] = r.randrange(1, c["timing"]["mxm_abc"]["d_func"] + 1)
    c["timing"]["c2c"]["rtdeskew_floor"] = r.choice([1, 100, 65535, r.randrange(1, 65536)])
    # timing.overrides: a few entries with their own row; Read, Write,
    # Gather, Scatter and IfetchM keep the MEM class's d_skew (IfetchM by an override).
    ovr = [{"entry": "IfetchM", "d_func": r.randrange(64), "d_skew": c["timing"]["mem"]["d_skew"]}]
    for e in r.sample(["Add", "Mul", "Reduce", "ABC", "ACC", "LW", "Transpose", "Distribute", "Send", "Receive"], r.randrange(4)):
        f, sk = row(1)
        if e == "ABC":
            f = max(f, sk + sl, c["timing"]["mxm_abc"]["occupancy_f16"])   # effective ABC row satisfies causality and occupancy
        ovr.append({"entry": e, "d_func": f, "d_skew": sk})
    if r.random() < 0.3:
        ovr.append({"entry": "Shift", "d_func": r.randrange(max(sl, 1), 64), "d_skew": 0})
    c["timing"]["overrides"] = ovr
    c["icu"] = {"fetch_latency": r.choice([1, 4, 64, r.randrange(1, 65)]), "fetch_bytes_per_cycle": r.choice([4, 16, 64, 4 * r.randrange(1, 17)]),
                "iq_depth": r.choice([64, 1280, 4096, 4 * r.randrange(16, 1025)]),
                "fetch_staging_depth": r.choice([1, 8, 64, r.randrange(1, 65)])}
    rate = c["icu"]["fetch_bytes_per_cycle"]
    minimum = max(b + k - (16 * (b * sl + k)) // rate for b in range(2) for k in range(sl))
    c["icu"]["fetch_latency"] = max(minimum, c["icu"]["fetch_latency"])
    c["barrier"]["release_latency"] = r.choice([0, 1, 35, 4095, r.randrange(4096)])
    # 1 <= link_latency_min <= link_latency <= link_latency_max < hac_period <= 65535 and
    # c2c.links 1..16 (schema_invariants.py BOUNDS, C2C_RELATIONS; ).
    c["c2c"]["links"] = r.choice([1, 2, 7, 11, 16, r.randrange(1, 17)])
    p = r.choice([2, 3, 252, 256, 65535, r.randrange(2, 65536)])
    hi = r.choice([1, p - 1, r.randrange(1, p)])
    lo = r.choice([1, hi, r.randrange(1, hi + 1)])
    c["c2c"].update(hac_period=p, link_latency_max=hi, link_latency_min=lo,
                    link_latency=r.choice([lo, hi, r.randrange(lo, hi + 1)]))
    if r.random() < 0.5:  # another legal floorplan
        up = r.randrange(12, 20)
        vx = up + r.randrange(1, 3)
        dn = vx + r.randrange(1, 3)
        tx = dn + quads + r.randrange(0, 2)
        dst = tx + 1
        turn = dst + 1
        l = c["layout"]
        rx = r.randrange(1, up + 1 - quads)
        out = r.randrange(0 + 1, rx) if rx > 1 else None
        if out is not None:
            l.update(mem_up_gap_base=up, vxm_gap=vx, mem_down_gap_base=dn, c2c_tx_gap=tx, sxm_dst_gap=dst,
                     sxm_turn_in_gap=turn, mxm_in_gap=r.randrange(turn + 1, 44), c2c_rx_gap=rx, sxm_turn_out_gap=out,
                     mxm_acc_gap=r.randrange(0, out))
    return c


def word(r, m):
    v = 0
    for ch in m:
        v = (v << 1) | (r.getrandbits(1) if ch == "-" else int(ch))
    return v


def mem_slice(r, cfg):
    """A MEM slice for an S or D line: mostly one that exists (an absent slice is an input error)."""
    g = cfg.get("geometry") if isinstance(cfg.get("geometry"), dict) else {}
    m = g.get("mem_slices_per_hemisphere")
    if isinstance(m, int) and 4 <= m <= 44 and r.random() < 0.9:
        return r.choice([0, 44]) + r.randrange(m)
    return r.randrange(88)


def program(r, cfg):
    wps = cfg["sram"]["words_per_slice"] if isinstance(cfg.get("sram"), dict) and isinstance(cfg["sram"].get("words_per_slice"), int) and cfg["sram"]["words_per_slice"] > 0 else 256
    lines = []
    for _ in range(r.randint(0, 30)):
        n = r.choice([16, 32, 320, r.randrange(1, 321)])
        lines.append("S %x %x %s" % (mem_slice(r, cfg), r.randrange(min(wps, 8192)), " ".join("%x" % r.randrange(256) for _ in range(n))))
    icu = cfg.get("icu") if isinstance(cfg.get("icu"), dict) else {}
    cap = icu["iq_depth"] // 4 if isinstance(icu.get("iq_depth"), int) and icu["iq_depth"] > 0 else 320
    # The text of a few queues (units; the queue's initial instruction queue): instructions of
    # the catalogue's encodings (one unit for a 32-bit layout, two for a 64-bit layout, the bundle
    # bit random), NOP units, padding, and fully random units.
    for _ in range(r.randint(1, 12)):
        q = r.randrange(144, 0x10000) if r.random() < 0.03 else r.choice([r.randrange(144), 0xffff, r.randrange(88), r.randrange(88, 144)])
        units = []
        for _ in range(r.randint(1, 60)):
            w, m = r.choice(PATS)
            x = word(r, m)
            k = r.random()
            if k < .05:
                u = [0xFFFFFFFF]
            elif k < .15:
                u = [r.randrange(1, 70000) & 0xFFFF or 1]                     # NOP n
            elif k < .20:
                u = [r.getrandbits(32), r.getrandbits(32)][:r.randint(1, 2)]
            elif w > 32:
                u = [(x >> 32) & 0x7FFFFFFF, x & 0xFFFFFFFF]
            else:
                u = [x & 0x7FFFFFFF]
            if len(u) and r.random() < .3:
                u[0] |= 0x80000000
            units += u
        units = units[:max(cap - r.choice([0, 0, 0, 1, -2]), 1)]
        lines.append("Q %x %s" % (q, " ".join("%x" % x for x in units)))
    for _ in range(4):
        lines.append("D %x %x" % (mem_slice(r, cfg), r.randrange(min(wps, 8192))))
    return lines


def corrupt_lines(r, lines):
    out = list(lines)
    for _ in range(r.randint(1, 5)):
        k = r.randrange(len(out))
        how = r.randrange(5)
        if how == 0:
            out[k] = out[k].replace(" ", "  ", 1) + " ff"
        elif how == 1:
            out[k] = " ".join(out[k].split()[:r.randint(0, 3)])
        elif how == 2:
            out[k] = r.choice("SQPDCXs") + out[k][1:]
        elif how == 3:
            out[k] = out[k].replace(r.choice("0123456789abcdef"), r.choice(["-", "ffffffffffffffffffff", "g", "0x"]), 1)
        else:
            out[k] = "".join(chr(r.randrange(1, 256)) for _ in range(r.randrange(40)))
    return out


def one(it):
    r = random.Random((seed << 32) ^ it)
    mode = r.choice(["valid", "valid", "valid", "mutated", "garbage"])
    args, may_reject = [], False
    if mode == "valid":
        cfg = valid_config(r)
        text = json.dumps(cfg)
        lines = program(r, cfg)
    elif mode == "mutated":
        may_reject = True
        cfg = copy.deepcopy(r.choice(PROFILES))
        for _ in range(r.randint(1, 3)):
            put(cfg, r.choice(list(leaves(cfg))), r.choice(HOSTILE))
        text = json.dumps(cfg)
        lines = program(r, cfg)
    else:
        may_reject = True
        cfg = r.choice(PROFILES)
        text = json.dumps(cfg)
        lines = program(r, cfg)
        if r.random() < .5:
            lines = corrupt_lines(r, lines)
        else:
            b = bytearray(text.encode())
            for _ in range(r.randint(1, 6)):
                b[r.randrange(len(b))] = r.randrange(1, 256)
            text = b.decode("latin-1")
    if r.random() < .3:
        args = ["--chips", str(r.choice([1, 2, 3, 64 if mode == "garbage" else 2]))]
    if r.random() < .1:
        args += ["--trace"]
    args += ["--max-cycles", str(r.choice([10**5, 10**6]))]
    inp = "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w", encoding="latin-1") as fh:
            fh.write(text)
        try:
            p = subprocess.run([sim, path] + args, input=inp.encode("latin-1"), capture_output=True, timeout=300)
            rc, err = p.returncode, p.stderr.decode("latin-1")
        except subprocess.TimeoutExpired:
            rc, err = "timeout", ""
    # A valid configuration must load: in valid mode any configuration error is a failure.
    # A valid-format program may still be an input error (a preload line longer than a
    # smaller profile's lanes, an SRAM address beyond its words), which tsp_sim reports as
    # "tsp_sim: input line N: ..."; any other exit 2 in valid mode is a failure.
    bad = rc not in (0, 2, 3)
    if rc == 2 and not may_reject and not err.startswith("tsp_sim: input line "):
        bad = True
    # One line on stderr, except for a garbage configuration whose key holds a control character:
    # both programs print the key raw, so the message spans several lines (known, recorded in the
    # ledger of 2026-11-16: escaping the key is a harness change that needs an approved CR).
    ctrl_key = re.search(r'\\u00[01][0-9a-f]|\\[nrt]', text) is not None
    if rc == 2 and err.count("\n") != 1 and not (ctrl_key and "config " in err):
        bad = True
    if bad:
        base = os.path.join(outdir, f"fail_{seed}_{it}")
        open(base + ".json", "w", encoding="latin-1").write(text)
        open(base + ".in", "w", encoding="latin-1").write(inp)
    return mode, rc, args, err[:200].replace("\n", " | "), bad


os.makedirs(outdir, exist_ok=True)
start, it, stats, fails = time.time(), 0, {}, 0
with cf.ThreadPoolExecutor(jobs) as ex:
    pending = set()
    while time.time() - start < seconds or pending:
        while time.time() - start < seconds and len(pending) < jobs:
            pending.add(ex.submit(one, it))
            it += 1
        done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
        for f in done:
            mode, rc, args, err, bad = f.result()
            stats[(mode, rc)] = stats.get((mode, rc), 0) + 1
            if bad:
                fails += 1
                print("FAIL", mode, rc, args, err, flush=True)
print(f"fuzz: {it} runs in {time.time() - start:.0f}s, {fails} failures")
for k in sorted(stats, key=str):
    print(" ", k, stats[k])
sys.exit(1 if fails else 0)
