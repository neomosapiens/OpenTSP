"""Directed tests of the superlane count (microarchitecture.md sec. 4, topology.md "Profile parameters", encoding.md "Growth limits of the
geometry-tied fields"): `geometry.superlanes` is any count from 1 to 32. The tests are profile-generic: every expected value is
computed from the configuration. They cover what the count reaches outside the vector width itself:

 - `SetVL n` is legal for n = 1 .. 32 on every profile (a 6-bit field; a value above the profile's superlane count is the count, the
   clamp) and is OTS-E-DEC-003 for 0 and for 33 .. 63; the length gates the tiles of a MEM Read (superlanes at or beyond n drive zero)
   and the last superlane is the last 16 lanes of the vector;
 - `Config v` splits the tiles of a MEM slice into liquid threads (tiles 0..v are thread 0), and a value of superlanes - 1 or more leaves
   every tile in thread 0;
 - an ECC event in the last superlane is reported with that superlane in the trace and in the error log record (the 5-bit superlane
   field of the record holds 0..31).

usage: superlanes.py <tsp_sim|tsp-iss> <config.json>
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines

SIM, CFG = sys.argv[1], sys.argv[2]
CJ = json.load(open(CFG))
SL = CJ["geometry"]["superlanes"]
VL = 16 * SL
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def mem(op, addr=0, s=0, thread=0): return (1 << 60) | (op << 54) | (thread << 53) | (addr << 32) | (s << 26)
def setvl(n): return (6 << 22) | n
def config(v): return (5 << 22) | v


def run(recs, sram=None, extra=(), csr=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in (sram or {}).items()]
    ls += q_lines(lower_records(list(recs))) + list(extra) + [f"R {a:x}" for a in csr]
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-400:])
    return p


def lines(p, kind): return [l.split() for l in p.stdout.splitlines() if l.split()[:2] == ["T", kind]]
def diags(p): return [(int(t[2]), int(t[4]), t[5][6:]) for t in lines(p, "diag")]
def srf(p): return [bytes.fromhex(t[6]) for t in lines(p, "srf")]
def xs(p): return {int(t[1], 16): int(t[2], 16) for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "X"}


WORD = bytes((i * 7 + 1) % 251 + 1 for i in range(VL))          # no zero byte: a zero in the result is an inactive tile
ADDR, STREAM = 3, 5
SRAM = {(0, ADDR): WORD}


def read_at(cycle, thread=0): return (cycle, 0, 1, mem(0, ADDR, STREAM, thread), 0)


# 1. SetVL n for n = 1 .. 32: legal on every profile; the Read after it gives the first min(n, superlanes) superlanes and zero above.
ns = sorted({1, max(1, SL // 2), max(1, SL - 1), SL, min(32, SL + 1), 20, 32})
for n in ns:
    p = run([(10, 0, 1, setvl(n), 0), read_at(12)], SRAM)
    got = srf(p)
    k = min(n, SL)
    want = WORD[:16 * k] + bytes(VL - 16 * k)
    check(f"SetVL {n}: no diagnostic, and a Read gives superlanes 0..{k - 1} and zero above", diags(p) == [] and got == [want],
          (diags(p), [g.hex()[:40] for g in got]))

# 2. The last superlane is the last 16 lanes of the vector (the default length gives all of them).
p = run([read_at(12)], SRAM)
got = srf(p)
check(f"a Read with the default length gives all {SL} superlanes; superlane {SL - 1} is lanes {16 * (SL - 1)}..{VL - 1}",
      got == [WORD] and got[0][16 * (SL - 1):] == WORD[16 * (SL - 1):], [g.hex()[:40] for g in got])

# 3. SetVL 0 and SetVL 33 .. 63 do not decode (OTS-E-DEC-003, no effect).
for n in (0, 33, 63):
    p = run([(10, 0, 1, setvl(n), 0), read_at(12)], SRAM)
    d = diags(p)
    check(f"SetVL {n} is OTS-E-DEC-003 and leaves the length alone (the Read still gives every superlane)",
          d == [(10, 0, "DEC-003")] and srf(p) == [WORD], (d, [g.hex()[:40] for g in srf(p)]))

# 4. Config v on a MEM queue: tiles 0..v are thread 0, the rest thread 1; v >= superlanes - 1 leaves every tile in thread 0.
for v in sorted({v for v in (0, SL // 2 - 1, SL - 2) if 0 <= v <= SL - 2}):
    for t in (0, 1):
        p = run([(8, 0, 1, config(v), 0), read_at(12, t)], SRAM)
        tiles = range(0, v + 1) if t == 0 else range(v + 1, SL)
        want = b"".join(WORD[16 * g:16 * g + 16] if g in tiles else bytes(16) for g in range(SL))
        check(f"Config {v}: a thread-{t} Read gives tiles {tiles.start}..{tiles.stop - 1} only", diags(p) == [] and srf(p) == [want],
              (diags(p), [g.hex()[:40] for g in srf(p)]))
for v in sorted({SL - 1, SL, 63, 4095}):
    p = run([(8, 0, 1, config(v), 0), read_at(12, 0), read_at(13, 1)], SRAM)
    got = srf(p)
    check(f"Config {v} (>= {SL - 1}): every tile is in thread 0, a thread-1 Read gives zero", diags(p) == [] and got == [WORD, bytes(VL)],
          (diags(p), [g.hex()[:40] for g in got]))

# 5. An ECC event in the last superlane carries that superlane: the trace line and the error log record (word 1, bits 47..43). The program
# is the VXM add of ras.py: slice 0 addr 1 on E:0 and slice 1 addr 2 on E:4 into the VXM, the sum written to slice 44.
DM, MEM_DS = CJ["timing"]["mem"]["d_func"], CJ["timing"]["mem"]["d_skew"]
VXM_DS = CJ["timing"]["vxm"]["d_skew"]
VXMQ, T = 88, 60


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    west = slice_ < 44
    up = (row == "E") == west
    return 18 - q if up else 24 + q


def feed(recs, sram, slice_, addr, val, stream, reader_gap, read_cycle):
    g = gap_mem(slice_, "E")
    sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, stream), 0))


def vxm(op, d, a, b): return (2 << 60) | (op << 54) | (d << 46) | (a << 40) | (b << 34)


A = bytes((13 * l + 5) & 0xFF for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
POS = [q for q in range(3, 137) if q & (q - 1)]           # the codeword position of data bit j (ras.py)
for sl in sorted({0, SL - 1}):
    recs, sram = [], {}
    feed(recs, sram, 0, 1, A, 0, 21, T + VXM_DS)
    feed(recs, sram, 1, 2, B, 4, 21, T + VXM_DS)
    recs.append((T, VXMQ, 1, vxm(0, 8, 0, 4), 0))
    p = run(recs, sram, extra=[f"F sram 0 0 1 {sl:x} 5"], csr=(0x14, 0x21))
    want = f"T ecc {T + VXM_DS} queue {VXMQ} corr E:0 sl {sl} pos {POS[5]}"
    got = [" ".join(t) for t in lines(p, "ecc")]
    x = xs(p)
    check(f"a bit flipped in superlane {sl} of a stored word is corrected and reported with sl {sl} (trace)", got == [want] and diags(p) == [], (got, diags(p)))
    check(f"the error log holds one entry whose superlane field is {sl}", x.get(0x14) == 1 and (x.get(0x21, 0) >> 43) & 31 == sl,
          {hex(k): hex(v) for k, v in x.items()})

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
