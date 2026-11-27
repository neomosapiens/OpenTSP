"""Directed end-to-end test of the bank-conflict report in the spare bits (microarchitecture.md sec. 15.5): a Read and a Write dispatched in one
cycle on the same bank of one MEM slice are OTS-E-MEM-001 (neither runs) and set, in the commit cycle (dispatch + d_func) at the write gap
of the slice, on the row of the Read's destination stream, S[2] of superlane 0, the parity S[3], the write address on S[4..16] and the read
address on S[17..29]. A Gather bounds error that meets it in the same register keeps its auxiliary information (the priority) and S[2] is still
noted ("both noted"). The first consumer logs it (`T ecc <c> queue <q> spare <row>:<stream> sl 0 pos <S[0..3]>`).
Runs against the Sail emulator or the ISS. Profile-portable: ref, mini and the test profiles.
usage: ras_bank.py <tsp_sim|tsp-iss|oracle> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
ok = True
if CJ["geometry"]["mem_slices_per_hemisphere"] < 44:
    print("PASS all (skipped: the programs use MEM slices up to 44; this profile has fewer)"); sys.exit(0)


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)
def iterations(n): return (1 << 60) | (7 << 54) | (n << 16)
def step(a): return (1 << 60) | (6 << 54) | (a << 16)
def gather(d, m, k=0, t=0): return (1 << 60) | (3 << 54) | (t << 53) | (k << 48) | (d << 42) | (m << 36)
def scatter(s, m, k=0, t=0): return (1 << 60) | (4 << 54) | (t << 53) | (k << 48) | (s << 42) | (m << 36)
def setsegment(n): return (1 << 60) | (8 << 54) | (n << 16)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap) - MEM_DS, slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))



RD = 21; TG = 100; GAP = gap_mem(0, "E"); TV = TG + DM + (RD - GAP); VXMQ = 88
B = CJ["sram"]["bank_bit"]
A = bytes((13 * l + 5) & 0xFF for l in range(VL)); C = bytes((7 * l + 91) & 0xFF for l in range(VL))
SUM = bytes((x + y) & 0xFF for x, y in zip(A, C)); WORD = bytes((3 * l + 17) & 0xFF for l in range(VL))
RA, WA = 0x13, 0x2B                        # read address (bits 0, 1, 4), write address (bits 0, 1, 3, 5): same bank (bank 0)
GBAD = 0x25                                # the Gather's bad tile address (superlane 0)


def program(kind, reads, ra=RA, wa=WA):
    recs, sram = [], {}
    if kind in ("conflict", "apart"):
        recs.append((TG, 0, 1, mem(0, ra, 2), 0)); recs.append((TG, 0, 1, mem(1, wa if kind == "conflict" else wa | (1 << B), 6), 0))
    if kind == "both":
        recs.append((TG, 0, 1, mem(0, ra, 2), 0)); recs.append((TG, 0, 1, mem(1, wa, 6), 0))
        sram[(1, 1)] = WORD
        recs.append((10, 1, 1, setsegment(4), 0))
        m = bytearray(VL)
        for g in range(SL): m[16 * g] = 1
        m[0] = GBAD
        feed(recs, sram, 4, 0, bytes(m), 4, "E", GAP, TG)
        recs.append((TG, 1, 1, gather(2, 4), 0))
    d = 0
    if kind == "loop":
        # a Write at TG opens its loop (Iterations 3: references at TG, TG + 1, TG + 2); a new Read at TG + 1 meets the loop's reference
        d = 1
        recs.append((TG, 0, 1, mem(1, wa, 6), 0)); recs.append((TG + 1, 0, 1, iterations(3), 0)); recs.append((TG + 1, 0, 1, mem(0, ra, 2), 0))
    if kind == "step":
        # the same loop with Step 1: references at TG (address WA), TG + 1 (WA + 1), TG + 2 (WA + 2); a new Read at TG + 2 meets the third
        d = 2
        recs.append((TG, 0, 1, mem(1, wa, 6), 0)); recs.append((TG + 1, 0, 1, step(1), 0)); recs.append((TG + 2, 0, 1, iterations(3), 0))
        recs.append((TG + 2, 0, 1, mem(0, ra, 2), 0))
    a, b = reads
    feed(recs, sram, 8, 1, A, a, "E", RD, TV + d)
    feed(recs, sram, 12, 2, C, b, "E", RD, TV + d)
    recs.append((TV + d - VXM_DS, VXMQ, 1, add_i8(8, a, b), 0))
    drain(recs, 44, 9, 8, "E", RD, TV + d - VXM_DS + DV)
    return recs, sram



_ORACLE = []
def oracle_go(recs, sram):
    import types
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, ".."))
    from oracle import machine as mc, diff_oracle as do
    if not _ORACLE:
        _ORACLE.append(mc.Machine(CJ))
    prog = types.SimpleNamespace(recs=list(recs), units={}, pre=dict(sram), faults=[], dump={(44, 9)})
    return types.SimpleNamespace(returncode=0, stderr="", stdout="\n".join(do.oracle_output(_ORACLE[0], prog)) + "\n")


def go(kind, reads, **kw):
    recs, sram = program(kind, reads, **kw)
    if SIM == "oracle":
        return oracle_go(recs, sram)
    dumps, extra = [(44, 9)], []
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + extra
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (kind, reads, p.returncode, p.stderr, p.stdout[-600:])
    return p


def ln(p, prefix): return [l for l in p.stdout.splitlines() if l.startswith(prefix)]
SPARE = lambda p: [l for l in ln(p, "T ecc") if " spare " in l]
sig = lambda s, pos, d=0: f"T ecc {TV + d} queue {VXMQ} spare E:{s} sl 0 pos {pos}"
MEM001 = lambda p: [l for l in ln(p, "T diag") if "OTS-E-MEM-001" in l]

# 1. A same-bank Read + Write pair: MEM-001 as before; S[2] and the parity S[3] (pos 12) on E:2 and E:3 of superlane 0.
p = go("conflict", (2, 3))
check("bank: a same-bank Read + Write pair is still OTS-E-MEM-001 (twice: one per instruction)", len(MEM001(p)) == 2, ln(p, "T diag"))
check("bank: S[2] and S[3] (pos 12) of superlane 0 are logged by the consumer of E:2 and E:3", SPARE(p) == [sig(2, 12), sig(3, 12)], SPARE(p))
# 2. The addresses: the write address 0x2B (bits 0, 1, 3, 5: streams 4, 5, 7, 9) and the read address 0x13 (bits 0, 1, 4: streams 17, 18, 21).
p = go("conflict", (4, 7))
check("bank: the write address bits on S[4] and S[7]", SPARE(p) == [sig(4, 12), sig(7, 12)], SPARE(p))
p = go("conflict", (6, 8))
check("bank: the write address bits that are clear (S[6], S[8]) carry nothing", not SPARE(p), SPARE(p))
p = go("conflict", (17, 21))
check("bank: the read address bits on S[17] and S[21]", SPARE(p) == [sig(17, 12), sig(21, 12)], SPARE(p))
p = go("conflict", (19, 20))
check("bank: the read address bits that are clear (S[19], S[20]) carry nothing", not SPARE(p), SPARE(p))
if B == 12:
    p = go("conflict", (29, 28), ra=0x1013, wa=0x102B)
    check("bank: S[29] (bit 12 of the read address 0x1013) is logged; S[28] (bit 11) is not", SPARE(p) == [sig(29, 12)], SPARE(p))
# 3. Different banks: no conflict, no report.
p = go("apart", (2, 3))
check("bank: a Read + Write pair on different banks: no MEM-001, no report", not MEM001(p) and not SPARE(p), (ln(p, "T diag"), SPARE(p)))
# 4. Gather bounds error and a bank conflict in one register: S[2] is noted, the auxiliary data are the Gather's (map E:4: id bit 2 = S[20];
#    bad address 0x25: bits 0, 2, 5 = S[4], S[6], S[9]); S[0] and S[2]: pos 5 and the parity S[3] = 0.
p = go("both", (2, 3))
check("bank: both noted: S[0] and S[2] (pos 5), the parity S[3] is 0 so E:3 logs nothing", SPARE(p) == [sig(2, 5)], SPARE(p))
p = go("both", (4, 5))
check("bank: both noted: the Gather's address bit S[4] (pos 5) is logged, the bank's write-address bit S[5] is not", SPARE(p) == [sig(4, 5)], SPARE(p))
p = go("both", (17, 20))
check("bank: both noted: the bank's read-address bit S[17] is dropped by the priority; the Gather's map-id bit S[20] is logged", SPARE(p) == [sig(20, 5)], SPARE(p))

# 5. A new Read meets the reference of an AGT loop (a Write with Iterations 3) in the next cycle: the loop's reference is blocked too, the
#    report holds the loop reference's address (0x2B) and the new Read's (0x13).
p = go("loop", (2, 3))
check("bank: loop: a new Read meets a loop reference: MEM-001 and S[2], S[3] (pos 12) one cycle later", MEM001(p) and SPARE(p) == [sig(2, 12, 1), sig(3, 12, 1)], (ln(p, "T diag"), SPARE(p)))
p = go("loop", (4, 17))
check("bank: loop: the loop reference's write address (S[4]) and the Read's address (S[17])", SPARE(p) == [sig(4, 12, 1), sig(17, 12, 1)], SPARE(p))
p = go("loop", (6, 19))
check("bank: loop: the address bits that are clear carry nothing", not SPARE(p), SPARE(p))

# 6. The loop's address steps: its reference at TG + 2 has the address WA + 2 = 0x2D (bits 0, 2, 3, 5: streams 4, 6, 7, 9; not 5).
p = go("step", (4, 5))
check("bank: step: the loop reference's address is the stepped one (0x2D): S[4] set, S[5] clear", SPARE(p) == [sig(4, 12, 2)], SPARE(p))
p = go("step", (6, 7))
check("bank: step: S[6] and S[7] of 0x2D are set", SPARE(p) == [sig(6, 12, 2), sig(7, 12, 2)], SPARE(p))

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
