"""Directed end-to-end tests of poison propagation through the MXM and over C2C (microarchitecture.md sec. 15 and 16):
an uncorrectable superlane of an ABC activation poisons every superlane of the product, one of a weight row r poisons the
superlane r / 16 (f16: row r is buffer rows 2r and 2r + 1), ACC carries the poison to the accumulator and into an
accumulating ACC, Send passes the poisoned superlanes with the vector and Receive writes them poisoned. The poison is
observed where a consumer reports it: a VXM add reads the result and raises `T ecc ... unc` for each poisoned superlane.
The MXM scenarios need 16 MEM slices to feed 16 weight streams in one cycle (a profile with mem_slices_per_hemisphere
below 44 skips them). Runs against the Sail emulator or the ISS.
usage: ras_poison.py <tsp_sim|tsp-iss|oracle> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
_rows = {o["entry"]: (o["d_func"], o["d_skew"]) for o in TM["overrides"]}
MDF, MDS = _rows.get("Max.i8") or _rows.get("Max") or (DV, VXM_DS)      # max.i8: its row keyed by the type, else the entry's, else the class row
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap) - MEM_DS, slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


A = bytes((13 * l + 5) & 0xFF for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
T = 60
VXMQ = 88
FETCH_LAT = CJ["icu"]["fetch_latency"]



MEMS = CJ["geometry"]["mem_slices_per_hemisphere"]
LAY = CJ["layout"]; TX_GAP = LAY["c2c_tx_gap"]; RX_GAP = LAY["c2c_rx_gap"]; ACC_GAP = LAY["mxm_acc_gap"]
D_ABC, D_ACC = TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"]; D_C2C = TM["c2c"]["d_func"]
LAT = CJ["c2c"]["link_latency"]
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)
def W(i): return 32 + i
def c2c(op, chip=0, s=0, link=0): return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)
SEND, RECEIVE = 0, 1
WTX, ERX = 126, 142
X = bytes((11 * l + 3) & 0x7F for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
def fsram(cycle, slice_, addr, sl, bit): return f"F sram {cycle:x} {slice_:x} {addr:x} {sl:x} {bit:x}"
def dbl(slice_, addr, sl): return [fsram(0, slice_, addr, sl, 5), fsram(0, slice_, addr, sl, 9)]   # an uncorrectable word


def consumer(recs, sram, row, stream, avail, b_stream):
    """A VXM add of `row`:`stream` and `row`:`b_stream` (the second from a MEM slice) executing at cycle `avail`."""
    r = (lambda i: i) if row == "E" else W
    feed(recs, sram, 1 if row == "E" else 45, 2, B, b_stream, row, 21, avail)
    recs.append((avail - VXM_DS, VXMQ, 1, add_i8(r(16), r(stream), r(b_stream)), 0))


def unc_at(p, queue, where, queues=None):
    """The superlanes reported uncorrectable by `queue` on `where` (e.g. "E:8"), by cycle: {cycle: [sl, ...]}."""
    out = {}
    for l in lines(p, "ecc"):
        t = l.split()
        if t[3] == "queue" and int(t[4]) in (queues or (queue,)) and t[5] == "unc" and t[6] == where:
            out.setdefault(int(t[2]), []).append(int(t[8]))
    return out


def mxm_prog(nrows=16, second=None):
    """16 (or 32) int8 weight rows on W:0..15 (IWS), one activation on W:24 (ABC), ACC i32 into accumulator 1 -> E:8, and a
    VXM consumer of E:8. `second` = (mode of the second ACC): a second ABC (clean or not) and an ACC of it, with its consumer.
    Returns (recs, sram, [consumer exec cycles], rows' (slice, addr))."""
    recs, sram, places = [], {}, []
    T = 200
    for blk in range(nrows // 16):
        Tb = T + 2 * blk
        for r in range(16):
            sl_ = 44 + 4 * (r % 11) + (r // 11)
            feed(recs, sram, sl_, 3 + r + 16 * blk, bytes((3 * (r + 16 * blk) + l) & 0x7F for l in range(VL)), r, "W", 43, Tb)
            places.append((sl_, 3 + r + 16 * blk))
        recs.append((Tb, 104, 1, mxm(1, 0, 32 | 0, 1), 0))                       # iws.i8 W:0
    T2 = T + 2 * (nrows // 16) + 2
    feed(recs, sram, 60, 5, X, 24, "W", 43, T2)
    recs.append((T2, 105, 1, mxm(2, 0, 32 | 24, 0), 0))                          # abc.i8 W:24
    T3 = T2 + D_ABC
    recs.append((T3, 106, 1, mxm(3, 2, 8, 0, 1), 0))                             # acc.i32 E:8, replace, accumulator 1
    cons = [T3 + D_ACC + (21 - ACC_GAP)]
    consumer(recs, sram, "E", 8, cons[0], 12)
    if second is not None:
        S = 4
        feed(recs, sram, 61, 5, X, 24, "W", 43, T2 + S)
        recs.append((T2 + S, 105, 1, mxm(2, 0, 32 | 24, 0), 0))
        recs.append((T2 + S + D_ABC, 106, 1, mxm(3, 2, 8, second, 1), 0))
        c2 = T2 + S + D_ABC + D_ACC + (21 - ACC_GAP)
        cons.append(c2)
        # a second consumer: another VXM queue, other B operand slice
        feed(recs, sram, 2, 2, B, 20, "E", 21, c2)
        recs.append((c2 - VXM_DS, VXMQ + 1, 1, add_i8(24, 8, 20), 0))
    return recs, sram, cons, places


_ORACLE = []
def oracle_go(recs, sram, extra):
    """SIM = `oracle`: the same scenario on the independent Python oracle (test/oracle), for the differential check of the poison rules."""
    import types
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, ".."))
    from oracle import machine as mc, diff_oracle as do
    if not _ORACLE:
        _ORACLE.append(mc.Machine(CJ))
    faults = [("sram",) + tuple(int(x, 16) for x in l.split()[2:]) for l in extra]
    prog = types.SimpleNamespace(recs=list(recs), units={}, pre=dict(sram), faults=faults, dump=set())
    return types.SimpleNamespace(returncode=0, stderr="", stdout="\n".join(do.oracle_output(_ORACLE[0], prog)) + "\n")


def go(recs, sram, extra=(), opts=(), nchips=1):
    if SIM == "oracle":
        return oracle_go(recs, sram, extra)
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + list(extra)
    p = subprocess.run([SIM, CFG, "--trace", *opts], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-400:])
    return p
def lines(p, kind): return [l for l in p.stdout.splitlines() if l.startswith(f"T {kind} ")]
def diags(p): return [l.split()[-1] for l in lines(p, "diag")]
ALL = list(range(SL))

if MEMS >= 44:
    recs, sram, cons, places = mxm_prog()
    p = go(recs, sram)
    check("MXM: a clean run reports nothing", lines(p, "ecc") == [] and diags(p) == [], (lines(p, "ecc")[:3], diags(p)))

    # 1. An uncorrectable activation poisons every superlane of the product.
    p = go(recs, sram, dbl(60, 5, SL - 1))
    check("MXM: an uncorrectable activation: one OTS-E-ECC-001 at the ABC, and the consumer of the ACC result reports every superlane",
          diags(p) == ["OTS-E-ECC-001"] * 2 and list(unc_at(p, 105, "W:24").values()) == [[SL - 1]] and
          list(unc_at(p, VXMQ, "E:8").values()) == [ALL], (diags(p), lines(p, "ecc")))

    # 2. An uncorrectable weight row r poisons the superlane r / 16 (not the superlane that held the fault).
    for r in ([5, 12] if SL > 1 else [5]):
        sl_, a = places[r]
        p = go(recs, sram, dbl(sl_, a, SL - 1))
        check(f"MXM: weight row {r} uncorrectable in superlane {SL - 1}: the product is poisoned in superlane {r // 16} only",
              list(unc_at(p, VXMQ, "E:8").values()) == [[r // 16]] and diags(p) == ["OTS-E-ECC-001"] * 2, (diags(p), lines(p, "ecc")))
    # one flip is corrected: nothing is poisoned
    sl_, a = places[5]
    p = go(recs, sram, [fsram(0, sl_, a, 0, 5)])
    check("MXM: a single flip of a weight row is corrected, nothing is poisoned", unc_at(p, VXMQ, "E:8") == {} and diags(p) == [], (lines(p, "ecc"), diags(p)))

    # 3. 32 weight rows (two loads): row 20 is superlane 1.
    if CJ["mxm"]["rows"] >= 32 and SL > 1:
        recs32, sram32, cons32, places32 = mxm_prog(32)
        p = go(recs32, sram32, dbl(*places32[20], 0))
        check("MXM: weight row 20 of a second load poisons the superlane 1 of the product",
              list(unc_at(p, VXMQ, "E:8").values()) == [[1]], (lines(p, "ecc"), diags(p)))

    # 4. ACC: replace takes the product's poison; an accumulating ACC keeps the accumulator's poison; a replacing ACC of a clean product clears it.
    rec2, sram2, cons2, _ = mxm_prog(second=1)
    p = go(rec2, sram2, dbl(60, 5, 0))
    u = unc_at(p, VXMQ, "E:8", (VXMQ, VXMQ + 1))
    check("ACC: a poisoned product replaces the accumulator: poisoned result; an accumulating ACC of a clean product stays poisoned (inherited)",
          sorted(u) == [cons2[0], cons2[1]] and all(v == ALL for v in u.values()) and diags(p) == ["OTS-E-ECC-001"] * 3, (u, diags(p)))
    rec3, sram3, cons3, _ = mxm_prog(second=0)
    p = go(rec3, sram3, dbl(60, 5, 0))
    u = unc_at(p, VXMQ, "E:8", (VXMQ, VXMQ + 1))
    check("ACC: a replacing ACC of a clean product clears the accumulator's poison", sorted(u) == [cons3[0]] and diags(p) == ["OTS-E-ECC-001"] * 2, (u, diags(p)))

# 5. C2C: a poisoned vector sent on a link arrives poisoned in the same superlanes; the Receive does not report it again.
def c2c_prog(chip_to=0, chip_from=0):
    recs, sram = [], {}
    T = 100
    feed(recs, sram, 44, 1, bytes((7 * l + 3) & 0xFF for l in range(VL)), 0, "W", TX_GAP, T)
    recs.append((T, WTX, 1, c2c(SEND, chip_to, W(0)), 0))
    Tr = T + LAT
    recs.append((Tr, ERX, 1, c2c(RECEIVE, chip_from, W(1)), 0))
    cons = Tr + D_C2C + (21 - RX_GAP)
    consumer(recs, sram, "W", 1, cons, 4)
    return recs, sram, T, Tr, cons

recs, sram, T, Tr, cons = c2c_prog()
p = go(recs, sram)
check("C2C: a clean vector over the link is not poisoned", lines(p, "ecc") == [] and diags(p) == [], (lines(p, "ecc")[:3], diags(p)))
sls = [0] if SL == 1 else [0, SL - 1]
p = go(recs, sram, dbl(44, 1, sls[-1]))
check(f"C2C: a vector with an uncorrectable superlane ({sls[-1]}) is reported by the Send, arrives poisoned in superlane {sls[-1]} only, "
      "and is reported by its consumer, not by the Receive",
      diags(p) == ["OTS-E-ECC-001"] * 2 and unc_at(p, WTX, "W:0") != {} and unc_at(p, ERX, "W:1") == {}
      and list(unc_at(p, VXMQ, "W:1").values()) == [[sls[-1]]], (diags(p), lines(p, "ecc")))
p = go(recs, sram, [fsram(0, 44, 1, 0, 5)])
check("C2C: a single flip is corrected at the Send; the vector arrives clean", unc_at(p, VXMQ, "W:1") == {} and diags(p) == [], (lines(p, "ecc"), diags(p)))
if SL > 1:
    p = go(recs, sram, dbl(44, 1, 0) + dbl(44, 1, SL - 1))
    check("C2C: two uncorrectable superlanes arrive as the same two", list(unc_at(p, VXMQ, "W:1").values()) == [[0, SL - 1]], lines(p, "ecc"))

# 6. A stored poisoned word is not kept poisoned . The first consumer of an uncorrectable word reports it and the
#    VXM writes a poisoned result back to SRAM; one more flip of a check bit of that stored word (c[8], bit 0x88) turns its check pattern into
#    the syndrome of a single-bit error, so the next read reports it as CORRECTED (and changes a data bit) instead of reporting the poison
#    again. This documents the behaviour of the model (core/ecc.sail): it is the specified consequence of poison being an overlay on a
#    SECDED code (the overlay is at distance 1 from the single-error syndromes, so no syndrome choice can avoid it), not a defect to fix.
#    microarchitecture.md says poison is reported by every consumer only while the stored word is undisturbed.
def go_uncaptured(recs, sram, extra=()):
    """As go(), but not one of the programs tools/gen_model_vectors.py captures for the frozen vectors (the capture hooks go(); this scenario documents a
    behaviour and must not renumber the vectors that follow it)."""
    if SIM == "oracle":
        return oracle_go(recs, sram, extra)
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + list(extra)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-400:])
    return p


def launder(flip):
    recs, sram = [], {}
    T0 = 40
    feed(recs, sram, 0, 0x10, A, 0, "E", 21, T0)                                   # slice 0 reads its preloaded word onto E:0 for the VXM
    recs.append((T0 - MDS, VXMQ, 1, VXM(9, 0, 8, 0, 0), 0))                      # max.i8 E:8, E:0, E:0 (the poison travels with the value)
    Tc = T0 - MDS + MDF                                                          # the cycle E:8 is committed at the VXM station
    drain(recs, 44, 9, 8, "E", 21, Tc)                                             # east slice 44 stores it (still poisoned)
    Tr = Tc + 60
    feed(recs, {}, 44, 9, B, 12, "W", 21, Tr)                                      # read the stored word back onto W:12 (nothing is preloaded)
    recs.append((Tr - MDS, VXMQ + 1, 1, VXM(9, 0, W(16), W(12), W(12)), 0))     # second consumer
    extra = dbl(0, 0x10, 0) + ([fsram(Tc + 30, 44, 9, 0, 0x88)] if flip else [])
    return go_uncaptured(recs, sram, extra), T0, Tr


p, T0, Tr = launder(False)
check(" base: the first consumer and the second consumer of the stored poisoned word both report it uncorrectable",
      diags(p) == ["OTS-E-ECC-001"] * 2 and sorted(unc_at(p, VXMQ, "E:0", (VXMQ,))) == [T0] and list(unc_at(p, VXMQ + 1, "W:12", (VXMQ + 1,))) == [Tr],
      (lines(p, "ecc"), diags(p)))
p, T0, Tr = launder(True)
second = sorted({l for l in lines(p, "ecc") if f"queue {VXMQ + 1} " in l and "W:12" in l})      # the harness prints a `T ecc` line twice (trace and event list)
check(" documented: one flip of check bit c[8] of the stored poisoned word makes the second consumer report it corrected, with no second diagnostic",
      diags(p) == ["OTS-E-ECC-001"] and len(second) == 1 and second[0].split()[5] == "corr" and unc_at(p, VXMQ + 1, "W:12", (VXMQ + 1,)) == {},
      (lines(p, "ecc"), diags(p)))


def launder_flip(bit):
    """The launder scenario with one flip of `bit` of the stored poisoned word (0..127 a data bit, 128 and above a check bit)."""
    recs, sram = [], {}
    T0 = 40
    feed(recs, sram, 0, 0x10, A, 0, "E", 21, T0)
    recs.append((T0 - MDS, VXMQ, 1, VXM(9, 0, 8, 0, 0), 0))
    Tc = T0 - MDS + MDF
    drain(recs, 44, 9, 8, "E", 21, Tc)
    Tr = Tc + 60
    feed(recs, {}, 44, 9, B, 12, "W", 21, Tr)
    recs.append((Tr - MDS, VXMQ + 1, 1, VXM(9, 0, W(16), W(12), W(12)), 0))
    return go_uncaptured(recs, sram, dbl(0, 0x10, 0) + [fsram(Tc + 30, 44, 9, 0, bit)]), Tr


def second_line(p):
    return sorted({l for l in lines(p, "ecc") if f"queue {VXMQ + 1} " in l and "W:12" in l})


# A flip of a data bit of the stored poisoned word launders it too, and the correction it triggers changes a second data bit (the
# flipped bit 5 and the "corrected" bit 4 are both wrong afterwards); a flip of data bit 127 gives syndrome 3 plus an uncorrectable
# neighbour (position 139) and the word stays uncorrectable. So "every neighbour is correctable" is false: the neighbours with a
# syndrome above the last position (136) are not.
p, Tr = launder_flip(5)
second = second_line(p)
check("launder, data bit: one flip of data bit 5 of the stored poisoned word makes the second consumer report it corrected at position 9 "
      "(a second data bit is corrected wrongly), with no second diagnostic",
      diags(p) == ["OTS-E-ECC-001"] and len(second) == 1 and second[0].split()[5] == "corr" and second[0].split()[-1] == "9", (lines(p, "ecc"), diags(p)))
p, Tr = launder_flip(127)
second = second_line(p)
check("launder, data bit 127: one flip of data bit 127 leaves the word uncorrectable at position 139 and the second consumer reports it again",
      diags(p) == ["OTS-E-ECC-001"] * 2 and len(second) == 1 and second[0].split()[5] == "unc" and second[0].split()[-1] == "139", (lines(p, "ecc"), diags(p)))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
