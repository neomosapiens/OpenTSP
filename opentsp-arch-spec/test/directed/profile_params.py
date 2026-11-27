"""Directed tests of the profile parameters (microarchitecture.md
sec. 1 "Profile parameters", validation.md "Which keys keep conformance"): sram.words_per_slice
(any power of two from 256 to 8192, bank_bit = log2 - 1), geometry.mem_slices_per_hemisphere
(MEM slice s exists when s mod 44 < the key; an absent slice's queue reports OTS-E-QUE-002
for every instruction, and S and D lines for it are input errors) and mxm.rows (the weight
buffer and an installed matrix hold mxm.rows stream values; beyond that is OTS-E-RES-002; an
ABC produces mxm.rows lanes).

The tests are profile-generic: every expected value is computed here from the configuration.
They run on ref, mini and the test profiles of config/test/ (which have fewer MEM slices, fewer
words and fewer MXM rows). The MXM tests that stream 16 operands need 16 MEM slices to feed
them and are skipped, with a message, on a profile that has fewer.

usage: profile_params.py <tsp_sim> <config.json>
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import tsp_as as A

SIM, CFG = sys.argv[1], sys.argv[2]
cfg = json.load(open(CFG))
VL = 16 * cfg["geometry"]["superlanes"]
MEMS = cfg["geometry"]["mem_slices_per_hemisphere"]
WPS = cfg["sram"]["words_per_slice"]
BANK = cfg["sram"]["bank_bit"]
ROWS = cfg["mxm"]["rows"]
DM = cfg["timing"]["mem"]["d_func"]
D_ABC = cfg["timing"]["mxm_abc"]["d_func"]
D_ACC = cfg["timing"]["mxm_acc"]["d_func"]
LG = cfg["layout"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def present(s):
    return s % 44 < MEMS


def run(images, sram=None, dumps=(), extra=()):
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in (sram or {}).items()]
    lines += list(extra) + A.q_lines(images) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    return subprocess.run([SIM, CFG, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)


def diags(r):
    out = []
    for ln in r.stdout.splitlines():
        t = ln.split()
        if t[:2] == ["T", "diag"]:
            out.append((int(t[2]), int(t[4]), t[5]))
    return out


def srf_count(r):
    return sum(1 for ln in r.stdout.splitlines() if ln.startswith("T srf"))


def words(r):
    return {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL]
            for t in (ln.split() for ln in r.stdout.splitlines()) if t and t[0] == "W"}


def mem(op, addr=0, s=0):
    return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)


def mxm(op, t, s=0, mode=0, acc=0):
    return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)


def image_of(recs):
    return A.lower_records([(c, q, 1, w, 0) for c, q, w in recs])


Q2 = "OTS-E-QUE-002"
print(f"profile {os.path.basename(CFG)}: {MEMS} MEM slices per hemisphere, {WPS} words, bank bit {BANK}, {ROWS} MXM rows, {VL} lanes")

# ---------------------------------------------------------------------------------------
# 1. The queue map is the same in every profile; a MEM queue of an absent slice reports
#    OTS-E-QUE-002 for its instruction and does nothing; the others run.
ZERO = bytes(VL)
recs = [(40, q, mem(0, 0, q % 4)) for q in range(88)]
r = run(image_of(recs), {(s, 0): bytes([1 + s % 7] * VL) for s in range(88) if present(s)})
absent = [s for s in range(88) if not present(s)]
check("the queue map has 144 queues: Q lines for every id are accepted", run({q: [0] for q in range(0, 144, 7)}).returncode == 0, r.stderr)
check("a Read on every MEM queue: QUE-002 exactly on the absent slices' queues (and nothing else)",
      r.returncode == 0 and sorted(diags(r)) == [(40, s, Q2) for s in absent] and srf_count(r) == 88 - len(absent),
      (r.returncode, r.stderr[:100], diags(r)[:4], srf_count(r)))

if absent:
    wa, ea = MEMS, 44 + MEMS            # the first absent slice of each hemisphere
    rd, wr = mem(0, 1, 0), mem(1, 2, 1)
    for a in (wa, ea, 43, 87):
        image = image_of([(40, a, rd), (40, a, wr), (50, a, 1 << 22), (60, a, rd), (70, a, (3 << 22) | (1 << 4) | 1)])
        image[a ^ 1 if a ^ 1 not in image else 0] = [0x60000000]       # an illegal word on another absent queue
        r = run(image)
        expect = [(40, a, Q2)] * 2 + [(50, a, Q2), (60, a, Q2), (70, a, Q2)]
        check(f"absent queue {a}: a Read + Write parcel is two QUE-002, Sync does not park, Repeat is QUE-002 too",
              r.returncode == 0 and [d for d in diags(r) if d[1] == a] == expect, (diags(r), r.stderr[:100]))
    r = run({wa: [0x60000000], ea: [0x60000000]})
    check("an instruction that does not decode is QUE-002 on an absent queue (not DEC-003)",
          sorted(diags(r)) == [(0, wa, Q2), (0, ea, Q2)], diags(r))
    # A present slice next to it is unaffected: the same word is DEC-003 there.
    r = run({MEMS - 1: [0x60000000]})
    check("the last present slice's queue: an illegal word is DEC-003", [d[2] for d in diags(r) if d[1] == MEMS - 1] == ["OTS-E-DEC-003"], diags(r))

# ---------------------------------------------------------------------------------------
# 2. The harness: S and D for an absent slice are input errors; so is an address from
#    words_per_slice on; the last present slice and the last word are fine.
def rejects(lines, needle):
    r = subprocess.run([SIM, CFG], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
    return r.returncode == 2 and needle in r.stderr, (r.returncode, r.stderr.strip()[:120])


last = MEMS - 1
check("S, D at the last word of the last present slice of each hemisphere are accepted",
      all(subprocess.run([SIM, CFG], input=f"S {s:x} {WPS - 1:x} 1 2 3\nD {s:x} {WPS - 1:x}\n", capture_output=True, text=True).returncode == 0
          for s in (last, 44 + last)))
for tag in "SD":
    res, info = rejects([f"{tag} 0 {WPS:x} 1"] if tag == "S" else [f"D 0 {WPS:x}"], "address")
    check(f"{tag} address words_per_slice ({WPS}) is an input error", res, info)
    for a in absent[:1] + absent[-1:]:
        res, info = rejects([f"{tag} {a:x} 0" + (" 1" if tag == "S" else "")], "slice")
        check(f"{tag} line for absent slice {a} is an input error", res, info)

# ---------------------------------------------------------------------------------------
# 3. SRAM addressing: only words_per_slice words exist, and the bank is address bit bank_bit.
BNK = 1 << BANK
rd = lambda a: mem(0, a, 0)
wr = lambda a: mem(1, a, 1)
if WPS < 8192:          # the address field is 13 bits: on a full slice no address is beyond the end
    r = run(image_of([(40, 0, rd(WPS - 1)), (50, 0, rd(WPS))]), {(0, WPS - 1): bytes([9] * VL)})
    check("Read of the last word is fine, Read of address words_per_slice is OPR-001", diags(r) == [(50, 0, "OTS-E-OPR-001")] and srf_count(r) == 1, diags(r))
else:
    r = run(image_of([(40, 0, rd(WPS - 1))]), {(0, WPS - 1): bytes([9] * VL)})
    check("Read of the last word (8191, the largest 13-bit address) is fine", diags(r) == [] and srf_count(r) == 1, diags(r))


def pair(ra, wa_):
    r = run(image_of([(40, 0, rd(ra)), (40, 0, wr(wa_))]))
    return [d[2] for d in diags(r) if d[2] in ("OTS-E-MEM-001", "OTS-E-QUE-001")]


check(f"Read and Write in different banks (addresses 0 and {BNK}) do not clash", pair(0, BNK) == [], pair(0, BNK))
check(f"Read and Write in the same bank (addresses 0 and {BNK >> 1}) are MEM-001 x2", pair(0, BNK >> 1) == ["OTS-E-MEM-001"] * 2, pair(0, BNK >> 1))
check(f"bank bit {BANK} is the top address bit: 2^(bank_bit+1) = words_per_slice", 2 ** (BANK + 1) == WPS, (BANK, WPS))
if BANK < 12:
    check(f"a Write beyond the slice (address {2 * BNK}) uses no bank: no clash with a Read", pair(0, 2 * BNK) == [], pair(0, 2 * BNK))

# ---------------------------------------------------------------------------------------
# 4. MXM: mxm.rows stream values in the buffer and in an installed matrix.
slices16 = [s for s in range(88) if present(s)][:16]
if len(slices16) < 16:
    print(f"SKIP MXM streaming tests: {len(slices16)} MEM slices cannot feed 16 operand streams")
else:
    def gap_mem(s, row):
        q, west = (s % 44) // 4, s < 44
        up = (row == "E") == west
        return LG["mem_up_gap_base"] - q if up else LG["mem_down_gap_base"] + q

    def feed16(recs, sram, addr, read_cycle, vals):
        """16 streams W:0..15 holding vals[k] (a word of VL bytes), visible at the MXM input gap in read_cycle."""
        for k in range(16):
            s = slices16[k]
            sram[(s, addr)] = vals[k]
            recs.append((read_cycle - DM - (LG["mxm_in_gap"] - gap_mem(s, "W")), s, mem(0, addr, 32 + k)))

    def weight_row(r):
        return bytes([(r % 100) + 1] + [0] * (VL - 1))

    T0, STEP = 200, 3
    for t, name in ((0, "i8"), (1, "f16")):
        # LW: ROWS / 16 loads fit, the next is RES-002 (stream values: ROWS i8 rows or ROWS / 2 f16 rows).
        n = ROWS // 16
        recs, sram = [], {}
        for g in range(n + 1):
            feed16(recs, sram, 1 + g, T0 + STEP * g, [weight_row(16 * g + k) for k in range(16)])
            recs.append((T0 + STEP * g, 104, mxm(0, t, 32, 0)))
        r = run(image_of(recs), sram)
        check(f"LW {name}: {n} loads of 16 stream values fill the buffer of {ROWS}, load {n + 1} is RES-002",
              r.returncode == 0 and diags(r) == [(T0 + STEP * n, 104, "OTS-E-RES-002")], (r.stderr[:100], diags(r)))
    # IW from streams: the same capacity for an installed matrix.
    recs, sram = [], {}
    for g in range(n + 1):
        feed16(recs, sram, 1 + g, T0 + STEP * g, [weight_row(16 * g + k) for k in range(16)])
        recs.append((T0 + STEP * g, 104, mxm(1, 0, 32, 1)))
    r = run(image_of(recs), sram)
    check(f"IW from streams: {n} installs fill a matrix of {ROWS} rows, install {n + 1} is RES-002",
          r.returncode == 0 and diags(r) == [(T0 + STEP * n, 104, "OTS-E-RES-002")], (r.stderr[:100], diags(r)))
    # Install ROWS rows, one ABC: the product has ROWS lanes (lane r = weight r * x), whatever the stream width.
    recs, sram = [], {}
    for g in range(n):
        feed16(recs, sram, 1 + g, T0 + STEP * g, [weight_row(16 * g + k) for k in range(16)])
        recs.append((T0 + STEP * g, 104, mxm(1, 0, 32, 1)))
    T2 = T0 + STEP * n
    x = bytes([1] + [0] * (VL - 1))
    sram[(slices16[0], 200)] = x
    recs.append((T2 - DM - (LG["mxm_in_gap"] - gap_mem(slices16[0], "W")), slices16[0], mem(0, 200, 32 + 24)))
    recs.append((T2, 105, mxm(2, 0, 32 + 24)))
    T3 = T2 + D_ABC
    recs.append((T3, 106, mxm(3, 2, 8, 0, 0)))
    for b in range(4):                                  # ACC writes E:8..11 at the ACC gap at T3 + d_func; quad 0 reads E at mem_up_gap_base
        s = b
        recs.append((T3 + D_ACC + (gap_mem(s, "E") - LG["mxm_acc_gap"]), s, mem(1, 9, 8 + b)))
    r = run(image_of(recs), sram, [(b, 9) for b in range(4)])
    w = words(r)
    got = [int.from_bytes(bytes(w[(b, 9)][l] for b in range(4)), "little") for l in range(VL)]
    exp = [(l % 100) + 1 if l < ROWS else 0 for l in range(VL)]
    check(f"ABC after {ROWS} installed rows: product lane l = weight l for l < {ROWS}, zero from lane {ROWS} on ({VL} lanes seen)",
          r.returncode == 0 and got == exp and diags(r) == [], (got[:20], exp[:20], diags(r)[:3]))

print("profile_params: " + ("all passed" if ok else "FAILED"))
sys.exit(0 if ok else 1)
