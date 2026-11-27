"""Directed tests of the timing table (microarchitecture.md sec. 1): the keys that are not
hard-coded latencies any more, the f16 MACC occupancy and its measured throughput ratio, the
per-entry override rows, the C2C vector rate and the Repeat spacing.
Every key is moved in a copy of the profile and the trace is compared with the unmoved one.
usage: timing_rows.py <tsp_sim> <config.json>"""
import copy, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines     # the normative lowering (assembly.md sec. 4)

SIM, CFG = sys.argv[1], sys.argv[2]
CJ = json.load(open(CFG))
TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]
ROWS = CJ["mxm"]["rows"]
VL = 16 * SL
TMP = tempfile.mkdtemp(prefix="timing_rows_")
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def variant(edit):
    """The profile with edit(config dict) applied, written to a temporary file."""
    c = copy.deepcopy(CJ)
    edit(c)
    p = os.path.join(TMP, "v%d.json" % len(os.listdir(TMP)))
    json.dump(c, open(p, "w"))
    return p


def run(cfg, recs, sram={}):
    lines = ["S %x %x " % (sl, a) + " ".join("%x" % b for b in v) for (sl, a), v in sram.items()]
    lines += q_lines(lower_records(recs))
    r = subprocess.run([SIM, cfg, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout, r.stderr


def diags(o): return [l for l in o.splitlines() if "diag" in l]
def srfs(o): return [int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf")]


def lat(cfg_json, entry, cls, ty=None):
    """(d_func, d_skew) of a catalogue entry at element type `ty`: its `<entry>.<ty>` row, else its override row, else its class row."""
    keys = ([entry + "." + ty] if ty else []) + [entry]
    for k in keys:
        for o in cfg_json["timing"]["overrides"]:
            if o["entry"] == k:
                return o["d_func"], o["d_skew"]
    return cfg_json["timing"][cls]["d_func"], cfg_json["timing"][cls]["d_skew"]


DM = TM["mem"]["d_func"]


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)
def repeat(n, d): return (3 << 22) | (n << 4) | d
def E(i): return i
def W(i): return 32 + i


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    up = (row == "E") == (slice_ < 44)
    return 18 - q if up else 24 + q


def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle, dm=DM):
    """Read SRAM[slice_][addr] onto stream so that it reaches reader_gap in read_cycle."""
    sram[(slice_, addr)] = val
    g = gap_mem(slice_, row)
    recs.append((read_cycle - dm - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))


def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


# ---------------------------------------------------------------------------------------
#  (a): timing.sxm_map_ready, the delay from a LoadMap's execute cycle to the first cycle a
# Permute may use the map (it was `superlanes`, hard-coded).
def h1_perm(d, s, m): return (4 << 60) | (1 << 54) | (d << 46) | (s << 40) | (m << 30)
def h1_loadwide(m, s, s2): return (4 << 60) | (6 << 54) | (s << 40) | (m << 30) | (s2 << 24)


def permute_diags(cfg, gap):
    """LoadMapWide at L, Permute `gap` cycles later on the east permuter (queue 138)."""
    cj = json.load(open(cfg))
    dm = cj["timing"]["mem"]["d_func"]
    L = 80
    sram, recs = {}, []
    feed(recs, sram, 4, 1, bytes((VL - 1 - l) & 255 for l in range(VL)), 0, "E", 40, L, dm)
    feed(recs, sram, 5, 1, bytes((VL - 1 - l) >> 8 for l in range(VL)), 1, "E", 40, L, dm)
    recs.append((L, 138, 1, h1_loadwide(3, E(0), E(1)), 0))
    feed(recs, sram, 6, 2, bytes(l % 251 + 1 for l in range(VL)), 2, "E", 40, L + gap, dm)
    recs.append((L + gap, 138, 1, h1_perm(W(5), E(2), 3), 0))
    rc, o, err = run(cfg, recs, sram)
    return rc, [l.split(" ", 3)[3] if len(l.split()) > 3 else l for l in diags(o)], err


def test_map_ready():
    n = TM.get("sxm_map_ready")
    check("the profile has timing.sxm_map_ready, and it equals geometry.superlanes (the former hard-coded delay)", n == SL, (n, SL))
    if n is None:
        return
    rc, d, _ = permute_diags(CFG, n)
    check("a Permute sxm_map_ready cycles after its LoadMapWide uses the map", rc == 0 and d == [], (rc, d))
    if n > 1:
        rc, d, _ = permute_diags(CFG, n - 1)
        check("a Permute one cycle earlier is RES-001", rc == 0 and d == ["queue 138 OTS-E-RES-001"], (rc, d))
    for k in (SL, n + 7):
        cfg = variant(lambda c, k=k: c["timing"].__setitem__("sxm_map_ready", k))
        rc, d, e = permute_diags(cfg, k)
        check("sxm_map_ready = %d: a Permute %d cycles after the load uses the map" % (k, k), rc == 0 and d == [], (rc, d, e))
        if k > 1:
            rc, d, e = permute_diags(cfg, k - 1)
            check("sxm_map_ready = %d: a Permute %d cycles after the load is RES-001" % (k, k - 1), rc == 0 and d == ["queue 138 OTS-E-RES-001"], (rc, d, e))


# ---------------------------------------------------------------------------------------
# The f16 MACC occupancy (timing.mxm_abc.occupancy_f16) and the measured throughput ratio of f16
# against i8 (the published rule: "these operations require two cycles, reducing throughput by 75%
# relative to INT8 operations"; isa-db/provenance.yaml).
def burst(cfg, t, n):
    """Install weights of type t, then dispatch n ABCs on consecutive cycles from one queue; returns
    the diagnostics of the ABCs."""
    cj = json.load(open(cfg))
    dm = cj["timing"]["mem"]["d_func"]
    sram, recs, T = {}, [], 200
    one = (0x3C00).to_bytes(2, "little")  # f16 1.0
    for r in range(16):
        v = bytearray(VL)
        v[0] = one[r % 2] if t == 1 else r + 1
        feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes(v), r, "W", 43, T, dm)
    recs.append((T, 104, 1, mxm(1, t, 32 | 0, 1), 0))  # iw.<t> W:0, from streams
    width = 2 if t == 1 else 1
    T2 = T + 2
    for k in range(n):
        base = 24 + width * k
        for w in range(width):
            v = bytearray(VL)
            v[0] = one[w] if t == 1 else 2
            feed(recs, sram, 60 + w, 5 + k, bytes(v), base + w, "W", 43, T2 + k, dm)
        recs.append((T2 + k, 105, 1, mxm(2, t, 32 | base), 0))  # abc.<t> W:base
    rc, o, err = run(cfg, recs, sram)
    return rc, [d for d in diags(o)], err


INSTALLS = min(2, ROWS // 16)                  # the profile's plane holds `mxm.rows` stream values (mini: 16, one install)


def installed_rows(cfg, t):
    """The output lanes that are nonzero after two IW installs of 16 stream values each, type t (0 i8, 1 f16, 2 u8): one ABC
    against an activation of 2 (or f16 1.0) and an ACC; the output lane r is nonzero for each installed row r
    (as in the directed accumulation check). The second install lands after the rows the first one made, so the lanes show how many
    rows 16 stream values make: 16 for an integer type, 8 for f16 (HC34 p.28)."""
    cj = json.load(open(cfg))
    dm = cj["timing"]["mem"]["d_func"]
    sram, recs, T = {}, [], 200
    one = (0x3C00).to_bytes(2, "little")
    for k in range(INSTALLS):
        for r in range(16):
            v = bytearray(VL)
            v[0] = one[r % 2] if t == 1 else 16 * k + r + 1
            feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + 16 * k + r, bytes(v), r, "W", 43, T + k, dm)
        recs.append((T + k, 104, 1, mxm(1, t, 32, 1), 0))
    T2 = T + 1 + INSTALLS
    for w in range(2 if t == 1 else 1):
        v = bytearray(VL)
        v[0] = one[w] if t == 1 else 2
        feed(recs, sram, 60 + w, 5, bytes(v), 24 + w, "W", 43, T2, dm)
    recs.append((T2, 105, 1, mxm(2, t, 32 | 24), 0))
    recs.append((T2 + 20, 106, 1, mxm(3, 3 if t == 1 else 2, 8, 0, 0 if t == 1 else 1), 0))
    for b in range(4):
        drain(recs, 4 * b, 9, 8 + b, "E", 2, T2 + 40)
    lines = ["S %x %x " % (sl, a) + " ".join("%x" % b for b in v) for (sl, a), v in sram.items()]
    lines += q_lines(lower_records(recs)) + ["D %x 9" % (4 * b) for b in range(4)]
    r = subprocess.run([SIM, cfg, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    words = {(int(x[1], 16), int(x[2], 16)): bytes(int(h, 16) for h in x[3:]) for x in (l.split() for l in r.stdout.splitlines()) if x and x[0] == "W"}
    lanes = [int.from_bytes(bytes(words.get((4 * b, 9), bytes(VL))[l] for b in range(4)), "little") for l in range(VL)]
    return [i for i, x in enumerate(lanes) if x], [l for l in diags(r.stdout)]


def test_f16_rate():
    n = 4                                      # an f16 ABC takes a stream pair: 4 of them fill streams 24..31
    occ = TM["mxm_abc"].get("occupancy_f16")
    check("the profile defines a positive FP16 initiation interval", isinstance(occ, int) and occ >= 1, occ)
    if occ is None:
        return
    _, d_i8, _ = burst(CFG, 2, n)
    _, d_f16, _ = burst(CFG, 1, n)
    acc_i8 = n - len([x for x in d_i8 if "RES-001" in x])
    acc_f16 = n - len([x for x in d_f16 if "RES-001" in x])
    check("integer ABCs issue on every cycle", acc_i8 == n, d_i8)
    check("f16 ABCs issue one per occupancy_f16 = %d cycles" % occ, acc_f16 == -(-n // occ), (acc_f16, d_f16))
    # Measure per-plane initiation and row count; FP16 additionally reserves its partner plane.
    lanes_i8, d1 = installed_rows(CFG, 2)
    lanes_f16, d2 = installed_rows(CFG, 1)
    check("%d integer stream values make %d rows and %d f16 stream values %d (HC34 p.28), in the lanes below the vector length" % (16 * INSTALLS, 16 * INSTALLS, 16 * INSTALLS, 8 * INSTALLS),
          lanes_i8 == list(range(min(16 * INSTALLS, VL))) and lanes_f16 == list(range(min(8 * INSTALLS, VL))) and d1 == [] and d2 == [], (lanes_i8, lanes_f16, d1, d2))
    rows_i8, rows_f16 = 16, 8                  # per 16 stream values, as measured above
    ratio = 0.5 * (acc_f16 / n * rows_f16) / (acc_i8 / n * rows_i8)
    if os.path.basename(CFG) in ("ref.json", "mini.json"):
        check("the f16 dot-product rate is 25.0% of the integer rate, the published reduction of 75%", abs(ratio - 0.25) < 1e-9, ratio)
    for k in (1, 3, 4):
        cfg = variant(lambda c, k=k: c["timing"]["mxm_abc"].__setitem__("occupancy_f16", k))
        rc, dd, e = burst(cfg, 1, n)
        acc = n - len([x for x in dd if "RES-001" in x])
        check("occupancy_f16 = %d: %d of %d back-to-back f16 ABCs accepted" % (k, -(-n // k), n), rc == 0 and acc == -(-n // k), (rc, acc, dd, e))
        rc, dd, e = burst(cfg, 2, n)
        check("occupancy_f16 = %d leaves integer ABCs at one per cycle" % k, rc == 0 and not [x for x in dd if "RES-001" in x], (rc, dd, e))


# ---------------------------------------------------------------------------------------
#  (c): timing.c2c.rtdeskew_floor, the shortest hold of a RuntimeDeskew (it was max(t, 1)).
RTDESKEW = lambda t: (5 << 28) | (4 << 22) | t


def rtdeskew_read_cycle(cfg, t):
    """A Read due the cycle after a RuntimeDeskew t on queue 0: the cycle it reaches the SRF, minus d_func."""
    cj = json.load(open(cfg))
    dm = cj["timing"]["mem"]["d_func"]
    rc, o, err = run(cfg, [(10, 0, 1, RTDESKEW(t), 0), (11, 0, 1, mem(0, 0, E(0)), 0)], {(0, 0): bytes(VL)})
    return rc, [c - dm for c in srfs(o)], diags(o), err


def test_rtdeskew_floor():
    f = TM["c2c"].get("rtdeskew_floor")
    check("the profile has timing.c2c.rtdeskew_floor = 1 (a RuntimeDeskew holds its queue at least one cycle)", f == 1, f)
    if f is None:
        return
    for t in (0, 1, 5):
        rc, r, d, _ = rtdeskew_read_cycle(CFG, t)
        check("RuntimeDeskew %d holds max(%d, floor %d) cycles: the next Read dispatches at %d" % (t, t, f, max(11, 10 + max(t, f))),
              rc == 0 and r == [max(11, 10 + max(t, f))] and d == [], (rc, r, d))
    for k in (4, 9):
        cfg = variant(lambda c, k=k: c["timing"]["c2c"].__setitem__("rtdeskew_floor", k))
        for t in (0, 3, 12):
            rc, r, d, e = rtdeskew_read_cycle(cfg, t)
            want = max(11, 10 + max(t, k))
            check("rtdeskew_floor = %d: RuntimeDeskew %d, the next Read dispatches at %d" % (k, t, want), rc == 0 and r == [want] and d == [], (rc, r, d, e))



# ---------------------------------------------------------------------------------------
# : every catalogue entry has an explicit row that is reachable. The six override rows of the
# shipped profiles are in the profile; an entry's row is the one that sets its commit cycle, and
# moving it moves only that entry.
SHIPPED = {"Transpose": (5, 0), "Permute": (20, 0), "Exp": (5, 2), "Log": (5, 2), "RSqrt": (5, 2), "TanH": (5, 2)}
EXPQ = (2 << 60) | (22 << 54) | (1 << 52) | (8 << 46)        # exp.f16 E:8 <- E:0 (stream pair), queue 88


def commit(cfg, what, T=100):
    """(rc, diagnostics, cycle of the last SRF write) for one instance of `what` dispatched at T. The inputs are read at
    the entry's own d_skew in the profile `cfg`."""
    cj = json.load(open(cfg))
    dm = cj["timing"]["mem"]["d_func"]
    sram, recs = {}, []
    if what == "Transpose":
        ds = lat(cj, "Transpose", "sxm")[1]
        for r in range(16):
            feed(recs, sram, 44 + r, 1 + r, bytes(((l * (r + 3)) % 251) for l in range(VL)), r, "W", 40, T + ds, dm)
        recs.append((T, 120, 1, (4 << 60) | (4 << 54) | (E(0) << 46) | (W(0) << 40), 0))
    elif what == "Permute":
        ds = lat(cj, "Permute", "sxm_cross")[1]
        L = T - 40
        feed(recs, sram, 4, 1, bytes((VL - 1 - l) & 255 for l in range(VL)), 0, "E", 40, L, dm)
        feed(recs, sram, 5, 1, bytes((VL - 1 - l) >> 8 for l in range(VL)), 1, "E", 40, L, dm)
        recs.append((L, 138, 1, h1_loadwide(3, E(0), E(1)), 0))
        feed(recs, sram, 6, 2, bytes(l % 251 + 1 for l in range(VL)), 2, "E", 40, T + ds, dm)
        recs.append((T, 138, 1, h1_perm(W(5), E(2), 3), 0))
    elif what == "Exp":
        ds = lat(cj, "Exp", "vxm")[1]
        for b in range(2):
            feed(recs, sram, 4 * b, 1, bytes(VL), b, "E", 21, T + ds, dm)
        recs.append((T, 88, 1, EXPQ, 0))
    elif what == "And":                       # the class row of the VXM, for the comparison
        ds = lat(cj, "And", "vxm")[1]
        feed(recs, sram, 0, 1, bytes(VL), 0, "E", 21, T + ds, dm)
        feed(recs, sram, 1, 2, bytes(VL), 4, "E", 21, T + ds, dm)
        recs.append((T, 88, 1, (2 << 60) | (11 << 54) | (0 << 52) | (8 << 46) | (0 << 40) | (4 << 34) | (2 << 22), 0))
    elif what in ("Max.i8", "Max.f32"):       # max.<type> E:8 <- E:0, E:4 on queue 88: one stream for i8, a stream group of four for f32
        ty = what.split(".")[1]
        ds = lat(cj, "Max", "vxm", ty)[1]
        n, t = (1, 0) if ty == "i8" else (4, 3)
        for b in range(n):
            feed(recs, sram, b, 1, bytes(VL), b, "E", 21, T + ds, dm)
            feed(recs, sram, 4 + b, 2, bytes(VL), 4 + b, "E", 21, T + ds, dm)
        recs.append((T, 88, 1, (2 << 60) | (9 << 54) | (t << 52) | (8 << 46) | (0 << 40) | (4 << 34), 0))
    rc, o, err = run(cfg, recs, sram)
    d = diags(o)
    c = srfs(o)
    return rc, d, (max(c) if c else None)


def test_override_rows():
    got = {o["entry"]: (o["d_func"], o["d_skew"]) for o in TM["overrides"]}
    if CJ["geometry"]["superlanes"] in (20,):        # the shipped profiles
        check("the shipped profile carries the six override rows of the evidence", all(got.get(k) == v for k, v in SHIPPED.items()), got)
    for what, cls in (("Transpose", "sxm"), ("Permute", "sxm_cross"), ("Exp", "vxm")):
        df, ds = lat(CJ, what, cls)
        rc, d, c = commit(CFG, what)
        check("%s dispatched at 100 commits at 100 + its row's d_func = %d" % (what, 100 + df), rc == 0 and d == [] and c == 100 + df, (rc, d, c))
        # moving the entry's row moves the entry (d_func and d_skew)
        cfg = variant(lambda c_, w=what, df=df, ds=ds: c_["timing"].__setitem__("overrides", [o for o in c_["timing"]["overrides"] if o["entry"] != w] + [{"entry": w, "d_func": df + 3, "d_skew": ds}]))
        rc, d, c2 = commit(cfg, what)
        check("%s: its row moved to d_func %d commits 3 cycles later" % (what, df + 3), rc == 0 and d == [] and c2 == 100 + df + 3, (rc, d, c2))
        causal_df = max(df, ds + 1 + CJ["geometry"]["superlanes"]) if what == "Permute" else df
        cfg = variant(lambda c_, w=what, df=causal_df, ds=ds: c_["timing"].__setitem__("overrides", [o for o in c_["timing"]["overrides"] if o["entry"] != w] + [{"entry": w, "d_func": df, "d_skew": ds + 1}]))
        rc, d, c2 = commit(cfg, what)
        check("%s: its causal row moved to d_skew %d reads its operands one cycle later and commits at 100 + d_func" % (what, ds + 1), rc == 0 and d == [] and c2 == 100 + causal_df, (rc, d, c2))
    # an override of one entry leaves the other entries of its class alone: the Exp row moved, And (vxm class row) not
    cfg = variant(lambda c_: c_["timing"].__setitem__("overrides", [o for o in c_["timing"]["overrides"] if o["entry"] != "Exp"] + [{"entry": "Exp", "d_func": 9, "d_skew": 2}]))
    rc, d, c2 = commit(cfg, "And")
    rc0, d0, c0 = commit(CFG, "And")
    check("an override of Exp does not move And (the class row of the VXM)", rc == 0 and c2 == c0 and c0 == 100 + TM["vxm"]["d_func"], (rc, d, c2, c0))




# ---------------------------------------------------------------------------------------
# Per-type rows: an override row named "<Entry>.<type>" is the row of that element type only. The row of an instruction is
# the typed row, else the entry's row, else its class row; the commit cycle follows the row and the other types stay put.
def rows_variant(rows):
    """The profile with its rows of Max (typed or not) replaced by `rows`."""
    def edit(c_):
        keep = [o for o in c_["timing"]["overrides"] if o["entry"] != "Max" and not o["entry"].startswith("Max.")]
        c_["timing"]["overrides"] = keep + [{"entry": e, "d_func": f, "d_skew": k} for e, f, k in rows]
    return variant(edit)


def test_typed_rows():
    cdf, cds = TM["vxm"]["d_func"], TM["vxm"]["d_skew"]
    for what in ("Max.i8", "Max.f32"):                  # the profile's own rows: the typed row if it has one, else the class row
        want = lat(CJ, "Max", "vxm", what.split(".")[1])[0]
        rc, d, c = commit(CFG, what)
        check("%s under the profile's own rows commits at 100 + %d" % (what, want), rc == 0 and d == [] and c == 100 + want, (rc, d, c))
    cfg = rows_variant([("Max.i8", cdf + 3, cds)])
    for what, extra in (("Max.i8", 3), ("Max.f32", 0)):
        rc, d, c = commit(cfg, what)
        check("a Max.i8 row at d_func %d: %s commits at 100 + %d" % (cdf + 3, what, cdf + extra), rc == 0 and d == [] and c == 100 + cdf + extra, (rc, d, c))
    cfg = rows_variant([("Max.f32", cdf + 4, cds + 1)])
    for what, extra in (("Max.i8", 0), ("Max.f32", 4)):
        rc, d, c = commit(cfg, what)
        check("a Max.f32 row at d_func %d, d_skew %d: %s commits at 100 + %d" % (cdf + 4, cds + 1, what, cdf + extra), rc == 0 and d == [] and c == 100 + cdf + extra, (rc, d, c))
    cfg = rows_variant([("Max", cdf + 2, cds)])
    for what in ("Max.i8", "Max.f32"):
        rc, d, c = commit(cfg, what)
        check("a Max row (no type) at d_func %d covers %s" % (cdf + 2, what), rc == 0 and d == [] and c == 100 + cdf + 2, (rc, d, c))
    cfg = rows_variant([("Max", cdf + 2, cds), ("Max.i8", cdf + 5, cds)])
    for what, extra in (("Max.i8", 5), ("Max.f32", 2)):
        rc, d, c = commit(cfg, what)
        check("a Max row and a Max.i8 row: the typed row wins for %s (d_func %d)" % (what, cdf + extra), rc == 0 and d == [] and c == 100 + cdf + extra, (rc, d, c))
    rc, d, c = commit(rows_variant([("Max.i8", cdf + 5, cds)]), "And")
    check("a Max.i8 row does not move And", rc == 0 and d == [] and c == 100 + cdf, (rc, d, c))


# ---------------------------------------------------------------------------------------
# Repeat n, d re-issues the previous instruction n more times with d cycles
# between iterations (ISCA20 Table I), so iteration k issues at p + k (d + 1), p the cycle of the
# repeated instruction. (NOP N is a delay of N cycles between parcels: its spacing N + 1 is unchanged.)
def repeat_reads(cfg, p, at, n, d):
    """Read cycles of a Read at p and a Repeat n, d dispatched at `at` on queue 0."""
    cj = json.load(open(cfg))
    dm = cj["timing"]["mem"]["d_func"]
    rc, o, err = run(cfg, [(p, 0, 1, mem(0, 1, E(0)), 0), (at, 0, 1, repeat(n, d), 0)], {(0, 1): bytes(VL)})
    return rc, [c - dm for c in srfs(o)], diags(o)


def test_repeat_spacing():
    for n, d, p, at, want in ((3, 1, 10, 11, [10, 12, 14, 16]),
                              (2, 3, 10, 11, [10, 14, 18]),
                              (2, 4, 10, 17, [10, 17, 22]),      # the first iteration is never before the Repeat's own cycle
                              (1, 15, 10, 11, [10, 26]),
                              (50, 1, 10, 11, [10 + 2 * k for k in range(51)]),
                              (3, 0, 10, 11, [10, 11, 12, 13]),    # d = 0: no cycle between, consecutive cycles
                              (2, 0, 10, 12, [10, 12, 13]),
                              (50, 0, 10, 11, [10 + k for k in range(51)])):
        rc, r, dg = repeat_reads(CFG, p, at, n, d)
        check("Read at %d, Repeat %d,%d at %d: reads at p + k (d + 1), the first not before %d" % (p, n, d, at, at),
              rc == 0 and r == want and dg == [], (rc, r, dg))



test_map_ready()
test_override_rows()
test_typed_rows()
test_repeat_spacing()
test_f16_rate()
test_rtdeskew_floor()

print("timing_rows: " + ("all passed" if ok else "FAILED"))
sys.exit(0 if ok else 1)
