"""Directed tests of the SRAM address width (microarchitecture.md sec. 3, topology.md "Profile parameters", encoding.md "Growth limits of the
geometry-tied fields"): `sram.addr_bits` is 13 to 16 and `sram.words_per_slice` any power of two from 256 up to 2^addr_bits. The tests are
profile-generic: every expected value is computed from the configuration (ref: 13 bits, 8192 words; test/w14, w15, w16: 14 to 16 bits). They
cover what the width reaches in each address-carrying field:

 - `Read` and `Write` carry the address in bits 47..32; the last word of a slice is reachable, and an address that needs more bits than the
   profile has is OTS-E-DEC-003 (so every word that was legal on 13 bits still means what it meant);
 - the bank bit is the top bit of the words: a Read and a Write of one slice in one cycle conflict (OTS-E-MEM-001) in one bank and do not
   conflict across the banks;
 - `IfetchM` carries the address in bits 15..0, and its second word (a + 1) must exist;
 - `Gather` takes its address from the low addr_bits bits of the little-endian halfword of the map, the bits above are ignored;
 - `SetSegment n` is legal up to 2^addr_bits (bits 32..16) and OTS-E-DEC-003 above;
 - `Step a` is legal for |a| up to min(2^addr_bits - 1, 32767) (the 16-bit field) and OTS-E-DEC-003 beyond, and a loop whose stride reaches
   the far end of the slice reads the right words.

usage: sram_width.py <tsp_sim|tsp-iss> <config.json>
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines

SIM, CFG = sys.argv[1], sys.argv[2]
CJ = json.load(open(CFG))
SL = CJ["geometry"]["superlanes"]
VL = 16 * SL
AB = CJ["sram"]["addr_bits"]
W = CJ["sram"]["words_per_slice"]
BB = CJ["sram"]["bank_bit"]
DM = CJ["timing"]["mem"]["d_func"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def mem(op, addr=0, s=0, thread=0): return (1 << 60) | (op << 54) | (thread << 53) | (addr << 32) | (s << 26)
def gather(d, m, k=0, t=0): return (1 << 60) | (3 << 54) | (t << 53) | (k << 48) | (d << 42) | (m << 36)
def step(a): return (1 << 60) | (6 << 54) | ((a & 0xFFFF) << 16)
def iterations(n): return (1 << 60) | (7 << 54) | (n << 16)
def setsegment(n): return (1 << 60) | (8 << 54) | (n << 16)
def ifetchm(a): return (7 << 22) | a


def run(recs, sram=None, extra=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in (sram or {}).items()]
    ls += q_lines(lower_records(list(recs))) + list(extra)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-400:])
    return p


def lines(p, kind): return [l.split() for l in p.stdout.splitlines() if l.split()[:2] == ["T", kind]]
def diags(p): return [(int(t[2]), int(t[4]), t[5][6:]) for t in lines(p, "diag")]
def codes(p): return [d[2] for d in diags(p)]
def srf(p): return [bytes.fromhex(t[6]) for t in lines(p, "srf")]
def short(vs): return [v.hex()[:24] for v in vs]


def word(seed): return bytes((seed * 31 + l * 7) % 251 + 1 for l in range(VL))   # no zero byte


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    west = slice_ < 44
    up = (row == "E") == west
    return 18 - q if up else 24 + q


STREAM = 5
LAST, HALF = W - 1, W // 2

# 1. Read of the last word of the slice and of a word that only the bank bit distinguishes.
seed = {(0, LAST): word(1), (0, HALF): word(2), (0, 3): word(3)}
p = run([(10, 0, 1, mem(0, LAST, STREAM), 0), (12, 0, 1, mem(0, HALF, STREAM), 0), (14, 0, 1, mem(0, 3, STREAM), 0)], seed)
check(f"Read of words {LAST:#x}, {HALF:#x}, 3 of a {W}-word slice ({AB}-bit address) gives each word, no diagnostic",
      diags(p) == [] and srf(p) == [word(1), word(2), word(3)], (diags(p), short(srf(p))))

# 2. An address of more than addr_bits bits does not decode (OTS-E-DEC-003); 16 bits is the most the field holds.
if AB < 16:
    p = run([(10, 0, 1, mem(0, 1 << AB, STREAM), 0)], seed)
    check(f"a Read whose address needs {AB + 1} bits is OTS-E-DEC-003 on a {AB}-bit profile", codes(p) == ["DEC-003"], diags(p))
    p = run([(10, 0, 1, mem(1, (1 << AB) | 5, STREAM), 0)], seed)
    check(f"a Write whose address needs {AB + 1} bits is OTS-E-DEC-003 on a {AB}-bit profile", codes(p) == ["DEC-003"], diags(p))

# 3. An address inside the field but outside the slice is OTS-E-OPR-001 (only when the slice has fewer words than the field addresses).
if W < (1 << AB):
    p = run([(10, 0, 1, mem(0, W, STREAM), 0)], seed)
    check(f"a Read of address {W} (the slice has {W} words) is OTS-E-OPR-001", codes(p) == ["OPR-001"], diags(p))

# 4. The bank bit is the top bit of the words: a Read and a Write in one cycle conflict in one bank, not across the banks.
RD, WR_SAME, WR_OTHER = LAST, LAST - 1, LAST ^ (1 << BB)
p = run([(40, 0, 1, mem(0, RD, 2), 0), (40, 0, 1, mem(1, WR_SAME, 6), 0)], seed)
check(f"a Read of {RD:#x} and a Write of {WR_SAME:#x} in one cycle (one bank) are OTS-E-MEM-001", "MEM-001" in codes(p), diags(p))
p = run([(40, 0, 1, mem(0, RD, 2), 0), (40, 0, 1, mem(1, WR_OTHER, 6), 0)], seed)
check(f"a Read of {RD:#x} and a Write of {WR_OTHER:#x} in one cycle (banks differ at bit {BB}) raise no MEM-001", "MEM-001" not in codes(p), diags(p))

# 5. IfetchM carries its address in bits 15..0; it needs word a + 1 as well.
pad = bytes([0xFF]) * VL
seed_f = {(0, LAST - 1): pad, (0, LAST): pad}
p = run([(10, 0, 1, ifetchm(LAST - 1), 0)], seed_f)
check(f"IfetchM {LAST - 1:#x} (words {LAST - 1:#x} and {LAST:#x}) raises no diagnostic", diags(p) == [], diags(p))
p = run([(10, 0, 1, ifetchm(LAST), 0)], seed_f)
check(f"IfetchM {LAST:#x} (word {W:#x} does not exist) is OTS-E-OPR-001", codes(p) == ["OPR-001"], diags(p))

# 6. Gather: the address is the low addr_bits bits of the map halfword; bits above are ignored. Slice 1 gathers from its own word; the map is
# read from slice 4 onto E:4 so that it arrives in the cycle of the Gather (as in ras_bank.py).
TG = 100
HIGH = ((1 << 16) - 1) & ~((1 << AB) - 1)
for field in sorted({LAST, LAST | HIGH}) if CJ["geometry"]["mem_slices_per_hemisphere"] >= 8 else []:      # the map reader is slice 4: needs two quads
    m = bytearray(VL)
    for g in range(SL):
        m[16 * g], m[16 * g + 1] = field & 0xFF, field >> 8
    sram = {(1, LAST): word(9), (4, 0): bytes(m)}
    recs = [(TG - DM - (gap_mem(1, "E") - gap_mem(4, "E")), 4, 1, mem(0, 0, 4), 0), (TG, 1, 1, gather(2, 4), 0)]
    p = run(recs, sram)
    want = bytearray(VL)
    for g in range(SL):
        want[16 * g] = word(9)[16 * g]
    check(f"Gather with map address {field:#06x} reads word {LAST:#x} (bits above {AB} ignored), no diagnostic",
          diags(p) == [] and srf(p)[-1:] == [bytes(want)], (diags(p), short(srf(p))))

# 7. SetSegment n is legal up to 2^addr_bits and OTS-E-DEC-003 above; a segment of the whole field leaves the slice reachable and a shorter
# segment makes the last word an OTS-E-OPR-001.
p = run([(8, 0, 1, setsegment(1 << AB), 0), (12, 0, 1, mem(0, LAST, STREAM), 0)], seed)
check(f"SetSegment {1 << AB} (2^{AB}) is legal; the last word is still reachable", diags(p) == [] and srf(p) == [word(1)], (diags(p), short(srf(p))))
p = run([(8, 0, 1, setsegment((1 << AB) + 1), 0)], seed)
check(f"SetSegment {(1 << AB) + 1} (above 2^{AB}) is OTS-E-DEC-003", codes(p) == ["DEC-003"], diags(p))
p = run([(8, 0, 1, setsegment(LAST), 0), (12, 0, 1, mem(0, LAST, STREAM), 0)], seed)
check(f"SetSegment {LAST} makes a Read of word {LAST:#x} OTS-E-OPR-001", codes(p) == ["OPR-001"], diags(p))

# 8. Step a: |a| <= min(2^addr_bits - 1, 32767) decodes, beyond is OTS-E-DEC-003. (A Step alone is a configuration with nothing to configure:
# OTS-E-OPR-001 on every profile; only the decode diagnostic tells the two cases apart.)
LIM = min((1 << AB) - 1, 32767)
for a in (LIM, -LIM):
    p = run([(8, 0, 1, step(a), 0)], seed)
    check(f"Step {a} decodes", "DEC-003" not in codes(p), diags(p))
for a in (LIM + 1, -(LIM + 1)):
    p = run([(8, 0, 1, step(a), 0)], seed)
    check(f"Step {a} is OTS-E-DEC-003", codes(p) == ["DEC-003"], diags(p))

# 9. A loop whose stride reaches the far end of the slice (the stride is the largest one that stays inside the slice): Read at 0 with Step FAR and
# two iterations reads words 0 and FAR; a Read at FAR with Step -FAR reads FAR then 0.
FAR = min(LIM, LAST)
seed_l = {(0, 0): word(4), (0, FAR): word(5)}
T0 = 30
p = run([(T0, 0, 1, mem(0, 0, STREAM), 0), (T0 + 1, 0, 1, step(FAR), 0), (T0 + 2, 0, 1, iterations(2), 0)], seed_l)
check(f"a Read loop with Step {FAR} reads words 0 and {FAR:#x}", diags(p) == [] and srf(p) == [word(4), word(5)], (diags(p), short(srf(p))))
p = run([(T0, 0, 1, mem(0, FAR, STREAM), 0), (T0 + 1, 0, 1, step(-FAR), 0), (T0 + 2, 0, 1, iterations(2), 0)], seed_l)
check(f"a Read loop at {FAR:#x} with Step {-FAR} reads words {FAR:#x} and 0", diags(p) == [] and srf(p) == [word(5), word(4)], (diags(p), short(srf(p))))

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
