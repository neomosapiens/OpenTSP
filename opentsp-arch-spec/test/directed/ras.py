"""Directed end-to-end tests of the SECDED ECC on SRAM words, stream registers and data
paths (microarchitecture.md sec. 15), on the Sail emulator: a single flipped bit of an SRAM word or of a
stream register is corrected in the reader's copy and traced (`T ecc ... corr`), a double flip is
detected (OTS-E-ECC-001, `T ecc ... unc`) and poisons the results of the instruction, the MEM Write stores
the stream's bits unchecked, Gather and Scatter check their superlane of the stored word. Every latency and
gap is read from the profile; a vector is vlanes = 16 x superlanes bytes. Profile-portable: ref, mini and the test profiles.
usage: ras.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
def W(i): return 32 + i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)      # a Write reads its stream operand at dispatch + d_skew
    recs.append((commit_cycle + (g - writer_gap) - MEM_DS, slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def run(recs, sram, dumps, faults=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(faults)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr)
    return p


def lines(p, kind): return [l for l in p.stdout.splitlines() if l.startswith(f"T {kind} ")]
def diags(p): return lines(p, "diag")
def words(p): return {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "W"}


# The (137, 128) code written down independently: data bit j has codeword position POS[j], the j-th integer
# >= 3 that is not a power of two; a check bit c[i] (i < 8) is position 2^i, c[8] is position 0 (overall parity).
POS = [p for p in range(3, 137) if p & (p - 1)]
assert len(POS) == 128 and POS[0] == 3 and POS[127] == 136
def pos_of(bit):                       # the syndrome position the check names for a single flip of `bit`
    return POS[bit] if bit < 128 else (1 << (bit - 128) if bit < 136 else 0)
def pos_of2(a, b):                     # the syndrome of a double flip (the raw XOR of the two positions)
    return pos_of(a) ^ pos_of(b)
def flip_bytes(v, sl, bits):           # flip data bits of superlane sl in a vector
    v = bytearray(v)
    for b in bits:
        assert b < 128
        v[16 * sl + b // 8] ^= 1 << (b % 8)
    return bytes(v)


A = bytes((13 * l + 5) & 0xFF for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
def add(a, b): return bytes((x + y) & 0xFF for x, y in zip(a, b))
T = 60
VXMQ = 88


def vxm_program(a=A, b=B, t=T):
    """slice 0 addr 1 (A) on E:0 and slice 1 addr 2 (B) on E:4 read by the VXM at gap 21 in t + d_skew; the add.i8 result
    on E:8 is written to slice 44 addr 9 (a Write, unchecked)."""
    recs, sram = [], {}
    feed(recs, sram, 0, 1, a, 0, "E", 21, t + VXM_DS)
    feed(recs, sram, 1, 2, b, 4, "E", 21, t + VXM_DS)
    recs.append((t, VXMQ, 1, add_i8(E(8), E(0), E(4)), 0))
    drain(recs, 44, 9, 8, "E", 21, t + DV)
    return recs, sram


def faults_sram(cycle, slice_, addr, sl, bit): return f"F sram {cycle:x} {slice_:x} {addr:x} {sl:x} {bit:x}"
def faults_srf(cycle, row, stream, gap, sl, bit): return f"F srf {cycle:x} {row} {stream:x} {gap:x} {sl:x} {bit:x}"


# 1. No fault: the program's result, no ECC line and no diagnostic.
recs, sram = vxm_program()
p = run(recs, sram, [(44, 9)])
check("no fault: add.i8 result is the sum, no T ecc line, no diagnostic",
      words(p)[(44, 9)] == add(A, B) and lines(p, "ecc") == [] and diags(p) == [], (diags(p), lines(p, "ecc")))

# 2. SRAM single flips (a data bit, a check bit, the overall parity bit c[8], the spare bit): the Read forwards the stored
#    word unchecked, the VXM corrects its copy: the result is the sum, one `T ecc corr`; a spare-bit flip is no correction and no ECC error (the consumer logs it as a spare-bit error report).
for name, bit, last in [("data bit 5", 5, 0), ("data bit 127", 127, 0), ("data bit 77 of the last superlane", 77, SL - 1),
                        ("check bit c[3]", 128 + 3, 0), ("overall parity bit c[8]", 136, 0)]:
    recs, sram = vxm_program()
    p = run(recs, sram, [(44, 9)], [faults_sram(0, 0, 1, last, bit)])
    want = f"T ecc {T + VXM_DS} queue {VXMQ} corr E:0 sl {last} pos {pos_of(bit)}"
    check(f"SRAM single flip, {name}: the sum is correct and one `T ecc corr` (pos {pos_of(bit)}), no diagnostic",
          words(p)[(44, 9)] == add(A, B) and lines(p, "ecc") == [want] and diags(p) == [], (lines(p, "ecc"), diags(p)))
recs, sram = vxm_program()
p = run(recs, sram, [(0, 1), (44, 9)], [faults_sram(0, 0, 1, 0, 137)])
# Spare-bit error reports (microarchitecture.md sec. 15.5): the spare bit has no code, so the flip is neither a correction nor an uncorrectable word and the result is
# the sum; the consumer finds the bit set and logs it as an error report (`T ecc ... spare`).
check("SRAM spare-bit flip is no ECC correction (the spare bit has no code); the consumer logs it as a spare-bit report",
      [l for l in lines(p, "ecc") if " spare " not in l] == [] and len([l for l in lines(p, "ecc") if " spare " in l]) == 1 and diags(p) == [] and words(p)[(44, 9)] == add(A, B))
p = run(recs, sram, [(0, 1)], [faults_sram(0, 0, 1, 0, 5)])
check("the SRAM word still holds the flipped bit after the consumer corrected its copy (no write-back)", words(p)[(0, 1)] == flip_bytes(A, 0, [5]), words(p)[(0, 1)][:4])

# 3. SRAM double flip: the VXM reports one OTS-E-ECC-001 (every superlane acts), the lanes of the bad superlane are computed
#    from the stored bits, the result superlane is poisoned, and the next consumer of the result reports it again.
bits2 = [5, 77]
recs, sram = vxm_program()
p = run(recs, sram, [(44, 9)], [faults_sram(0, 0, 1, 0, bits2[0]), faults_sram(0, 0, 1, 0, bits2[1])])
syn = pos_of2(*bits2)
check("SRAM double flip: one OTS-E-ECC-001 at the VXM and one `T ecc unc` with the raw syndrome",
      diags(p) == [f"T diag {T + VXM_DS} queue {VXMQ} OTS-E-ECC-001"] and lines(p, "ecc") == [f"T ecc {T + VXM_DS} queue {VXMQ} unc E:0 sl 0 pos {syn}"], (diags(p), lines(p, "ecc")))
w = words(p)[(44, 9)]
check("the double flip does not stop the instruction: the good superlanes hold the sum, the bad one the sum of the stored bits",
      w == add(flip_bytes(A, 0, bits2), B) and (SL == 1 or w[16:] == add(A, B)[16:]), w[:16].hex())

# SetVL on the consumer masks output bytes while operand integrity still covers
# the whole input. Keep the feeding MEM queues at their full vector length.
if SL > 1:
    masked, initial = vxm_program()
    masked.insert(0, (0, VXMQ, 1, (6 << 22) | 1, 0))  # setvl 1
    fault = [faults_sram(0, 0, 1, SL - 1, bit) for bit in bits2]
    result = run(masked, initial, [(44, 9)], fault)
    expected = add(A, B)[:16] + bytes(VL - 16)
    check("SetVL 1 masks the output and reports an uncorrectable inactive input",
          words(result)[(44, 9)] == expected
          and diags(result) == [f"T diag {T + VXM_DS} queue {VXMQ} OTS-E-ECC-001"]
          and lines(result, "ecc") == [f"T ecc {T + VXM_DS} queue {VXMQ} unc E:0 sl {SL - 1} pos {syn}"],
          (diags(result), lines(result, "ecc")))

# the poisoned result is stored by the Write and read again: a second add reports ECC-001 again
T2 = T + 80
recs, sram = vxm_program()
feed(recs, sram, 44, 9, bytes(VL), 8, "W", 21, T2 + VXM_DS)           # the same word (a Read of the stored 138 bits) on W:8
feed(recs, sram, 45, 3, bytes(VL), 9, "W", 21, T2 + VXM_DS)
recs.append((T2, VXMQ, 1, add_i8(W(16), W(8), W(9)), 0))
drain(recs, 0, 20, 16, "W", 21, T2 + DV)
sram[(44, 9)] = bytes(VL)
p = run(recs, sram, [(44, 9)], [faults_sram(0, 0, 1, 0, bits2[0]), faults_sram(0, 0, 1, 0, bits2[1])])
check("the next consumer of the poisoned result reports OTS-E-ECC-001 again, and the Write stored the poison unchecked",
      diags(p) == [f"T diag {T + VXM_DS} queue {VXMQ} OTS-E-ECC-001", f"T diag {T2 + VXM_DS} queue {VXMQ} OTS-E-ECC-001"], (diags(p), lines(p, "ecc")))

# 4. A flip in a stream register (F srf): the value written by the Read at gap 18 in commit cycle c is in register 19 at c + 1
#    and in register 21 at c + 3 (the reader's gap). A flip in register 19 reaches the reader (corrected), a flip in register 22
#    at c + 4 is downstream of the reader: it never sees it.
c = T + VXM_DS - 3
recs, sram = vxm_program()
p = run(recs, sram, [(44, 9)], [faults_srf(c + 1, "e", 0, 19, 0, 40)])
check("register flip upstream of the reader: corrected at the VXM, the sum is right, one `T ecc corr`",
      words(p)[(44, 9)] == add(A, B) and lines(p, "ecc") == [f"T ecc {T + VXM_DS} queue {VXMQ} corr E:0 sl 0 pos {pos_of(40)}"] and diags(p) == [], (lines(p, "ecc"), diags(p)))
p = run(recs, sram, [(44, 9)], [faults_srf(c + 4, "e", 0, 22, 0, 40)])
check("register flip downstream of the reader's gap: the VXM sees no error", lines(p, "ecc") == [] and diags(p) == [] and words(p)[(44, 9)] == add(A, B), (lines(p, "ecc"), diags(p)))

# 5. The MEM Write stores the stream's 138 bits unchecked: a flip in a register, a Write, then a Read and a consumer corrects.
#    (The register flip reaches the east slice's Write; the stored word is the flipped one; the W dump shows it.)
recs = []; sram = {(0, 1): A}
recs.append((T - 3, 0, 1, mem(0, 1, E(0)), 0))
drain(recs, 44, 5, 0, "E", 18, T - 3 + DM)               # slice 44 reads E at gap 24: 6 cycles after the commit at gap 18
p = run(recs, sram, [(44, 5)], [faults_srf(T - 3 + DM + 1, "e", 0, 19, 0, 9)])
check("the MEM Write does not check: the stored word is the stream's flipped bits",
      diags(p) == [] and lines(p, "ecc") == [] and words(p).get((44, 5)) == flip_bytes(A, 0, [9]), (diags(p), lines(p, "ecc"), words(p)))

# 6. Gather (needs a MEM quad 1: slices 4 and 5 exist when geometry.mem_slices_per_hemisphere >= 8)
if CJ["geometry"]["mem_slices_per_hemisphere"] >= 8:
  # 6. Gather checks its tile's superlane of the stored word: a single flip is corrected, a double flip is ECC-001 on the MEM queue.
  m = bytearray(VL)
  for g in range(SL):
      m[16 * g] = 40 + g; m[16 * g + 1] = 0
  for kind, fl, want_diag in [("single", [0], False), ("double", [0, 1], True)]:
      sram2 = {(4, 1): bytes(m)}
      for g in range(SL): sram2[(0, 40 + g)] = bytes([g] * VL)
      TG = 30
      recs = [(TG + MEM_DS - DM - 1, 4, 1, mem(0, 1, E(1)), 0), (TG, 0, 1, (1 << 60) | (3 << 54) | (E(2) << 42) | (E(1) << 36), 0)]
      recs.append((TG + DM + 6, 44, 1, mem(1, 7, E(2)), 0))
      p = run(recs, sram2, [(44, 7)], [faults_sram(0, 0, 40, 0, b) for b in fl])
      ev = lines(p, "ecc")
      want_ev = [f"T ecc {TG + MEM_DS} queue 0 {'unc' if want_diag else 'corr'} sram 0:{40} sl 0 pos {pos_of2(*fl) if want_diag else pos_of(fl[0])}"]
      got = words(p).get((44, 7), b"")
      # tile 0 takes byte 0 of its superlane (value 0): corrected to 0, or (uncorrectable) the stored byte with bits 0 and 1 flipped = 3.
      check(f"Gather of a word with a {kind} flip: {'one OTS-E-ECC-001 on the MEM queue and ' if want_diag else 'no diagnostic and '}`T ecc {'unc' if want_diag else 'corr'} sram`; lane 0 is {3 if want_diag else 0}",
            ev == want_ev and diags(p) == ([f"T diag {TG + MEM_DS} queue 0 OTS-E-ECC-001"] if want_diag else []) and len(got) == VL and got[0] == (3 if want_diag else 0), (ev, diags(p), got[:2]))

  # 7. Scatter: the tile checks the old superlane (single flip: corrected), replaces its byte and regenerates the check bits.
  sram3 = {(4, 1): bytes(m), (0, 40): bytes([0x55] * VL), (5, 5): bytes([0xaa] * VL)}
  for g in range(1, SL): sram3[(0, 40 + g)] = bytes([g] * VL)
  TS = 30
  recs = [(TS + MEM_DS - DM - 1, 4, 1, mem(0, 1, E(1)), 0), (TS + MEM_DS - DM - 1, 5, 1, mem(0, 5, E(3)), 0),
          (TS, 0, 1, (1 << 60) | (4 << 54) | (E(3) << 42) | (E(1) << 36), 0)]
  p = run(recs, sram3, [(0, 40)], [faults_sram(0, 0, 40, 0, 9)])
  want = bytes([0xaa]) + bytes([0x55] * (VL - 1))
  check("Scatter of a byte into a word with a single flip: the tile corrects, replaces its byte; one `T ecc corr sram`, no diagnostic, the word is clean",
        diags(p) == [] and lines(p, "ecc") == [f"T ecc {TS + MEM_DS} queue 0 corr sram 0:40 sl 0 pos {pos_of(9)}"] and words(p).get((0, 40)) == want,
        (diags(p), lines(p, "ecc"), words(p).get((0, 40), b"")[:4].hex()))

  # 8. The map stream is an operand like any other: the map word (slice 4, addr 1) of tile tg (1; 0 on a one-superlane profile) with a
  #    single flip is corrected, so the Gather and the Scatter use address 40 + tg; with a double flip the tile makes no access and
  #    the instruction reports ECC-001. The word at 40 + tg holds 5 everywhere, the word the flip would name (xor 8) holds 0.
  tg = 1 if SL > 1 else 0
  wa = (40 + tg) ^ 8
  for kind, fl in [("single", [3]), ("double", [3, 11])]:
      unc = len(fl) == 2
      sram4 = {(4, 1): bytes(m)}
      for g in range(SL): sram4[(0, 40 + g)] = bytes([g] * VL)
      sram4[(0, 40 + tg)] = bytes([5] * VL)
      TG = 30
      recs = [(TG + MEM_DS - DM - 1, 4, 1, mem(0, 1, E(1)), 0), (TG, 0, 1, (1 << 60) | (3 << 54) | (E(2) << 42) | (E(1) << 36), 0)]
      recs.append((TG + DM + 6, 44, 1, mem(1, 7, E(2)), 0))
      p = run(recs, sram4, [(44, 7)], [faults_sram(0, 4, 1, tg, b) for b in fl])
      got = words(p).get((44, 7), b"")
      want_ev = [f"T ecc {TG + MEM_DS} queue 0 {'unc' if unc else 'corr'} E:1 sl {tg} pos {pos_of2(*fl) if unc else pos_of(fl[0])}"]
      check(f"Gather with a {kind} flip in the map word of tile {tg}: " + (f"one OTS-E-ECC-001, the tile makes no access (lane {16 * tg} is 0)" if unc else f"corrected, the tile reads address {40 + tg} (lane {16 * tg} is 5), no diagnostic"),
            lines(p, "ecc") == want_ev and diags(p) == ([f"T diag {TG + MEM_DS} queue 0 OTS-E-ECC-001"] if unc else []) and len(got) == VL and got[16 * tg] == (0 if unc else 5),
            (lines(p, "ecc"), diags(p), got[16 * tg:16 * tg + 1]))
      sram5 = {(4, 1): bytes(m), (5, 5): bytes([0xaa] * VL)}
      for g in range(SL): sram5[(0, 40 + g)] = bytes([g] * VL)
      sram5[(0, 40 + tg)] = bytes([5] * VL)
      recs = [(TS + MEM_DS - DM - 1, 4, 1, mem(0, 1, E(1)), 0), (TS + MEM_DS - DM - 1, 5, 1, mem(0, 5, E(3)), 0),
              (TS, 0, 1, (1 << 60) | (4 << 54) | (E(3) << 42) | (E(1) << 36), 0)]
      p = run(recs, sram5, [(0, 40 + tg), (0, wa)], [faults_sram(0, 4, 1, tg, b) for b in fl])
      wr = words(p).get((0, 40 + tg), b""); wx = words(p).get((0, wa), b"")
      check(f"Scatter with a {kind} flip in the map word of tile {tg}: " + ("one OTS-E-ECC-001, the tile writes nothing" if unc else f"corrected, the tile writes address {40 + tg} and not {wa}"),
            lines(p, "ecc") == want_ev and diags(p) == ([f"T diag {TS + MEM_DS} queue 0 OTS-E-ECC-001"] if unc else [])
            and wx[16 * tg] == 0 and wr[16 * tg] == (5 if unc else 0xaa) and wr[16 * tg + 1] == 5,
            (lines(p, "ecc"), diags(p), wr[16 * tg:16 * tg + 2], wx[16 * tg:16 * tg + 2]))

# 9. SXM Shift and Permute: an output superlane takes lanes of other superlanes, so one uncorrectable input superlane poisons
#    every output superlane (core/ecc.sail ecc_dep_all). The result is stored by a Write and read again by an add.i8, which
#    reports the poison once per superlane (a `T ecc unc` line per superlane). On a one-superlane profile there is nothing to
#    spread, so the scenario needs superlanes > 1.
SXM_X = TM["sxm_cross"]["d_func"]
PERM_X = {o["entry"]: o["d_func"] for o in TM["overrides"]}.get("Permute", SXM_X)      # Permute's own row, else the class row
def h1_perm(d, s, m): return (4 << 60) | (1 << 54) | (d << 46) | (s << 40) | (m << 30)
def h1_loadwide(m, s, s2): return (4 << 60) | (6 << 54) | (s << 40) | (m << 30) | (s2 << 24)
def poison_consumer(recs, sram, slice_, addr, row, T2):
    """read the stored result (slice_ addr) on `row`:6 and a second operand on `row`:4 at the VXM in T2 + d_skew and add them."""
    r = E if row == "E" else W
    feed(recs, sram, slice_, addr, bytes(VL), 6, row, 21, T2 + VXM_DS)
    feed(recs, sram, 1 if row == "E" else 45, 2, B, 4, row, 21, T2 + VXM_DS)
    recs.append((T2, VXMQ, 1, add_i8(r(16), r(6), r(4)), 0))
if SL > 1 and CJ["geometry"]["mem_slices_per_hemisphere"] >= 8:
    X = bytes((l % 200) + 1 for l in range(VL))
    def spread(kind, flips):
        recs, sram = [], {}
        if kind == "Shift":
            Tx = 80
            feed(recs, sram, 44, 1, X, 0, "W", 40, Tx)
            recs.append((Tx, 122, 1, (4 << 60) | (0 << 54) | (E(5) << 46) | (W(0) << 40) | (0 << 39) | (1 << 30), 0))
            drain(recs, 0, 9, 5, "E", 4, Tx + SXM_X)
            src, dst, row = (44, 1), (0, 9), "E"
        else:
            Tx = 80
            rev = [VL - 1 - l for l in range(VL)]
            feed(recs, sram, 4, 1, bytes(e & 255 for e in rev), 0, "E", 40, Tx - SL)
            feed(recs, sram, 5, 1, bytes(e >> 8 for e in rev), 1, "E", 40, Tx - SL)
            recs.append((Tx - SL, 138, 1, h1_loadwide(3, E(0), E(1)), 0))
            feed(recs, sram, 6, 2, X, 20, "E", 40, Tx)
            recs.append((Tx, 138, 1, h1_perm(W(5), E(20), 3), 0))
            drain(recs, 44, 9, 5, "W", 4, Tx + PERM_X)
            src, dst, row = (6, 2), (44, 9), "W"
        T2 = Tx + (PERM_X if kind == "Permute" else SXM_X) + 60
        poison_consumer(recs, sram, dst[0], dst[1], row, T2)
        return run(recs, sram, [], [faults_sram(0, src[0], src[1], 0, b) for b in flips]), T2
    for kind, row in (("Shift", "E"), ("Permute", "W")):
        p, T2 = spread(kind, [5, 77])
        ev = [l for l in lines(p, "ecc") if f"queue {VXMQ} unc {row}:6 sl " in l]
        check(f"{kind}: one uncorrectable input superlane poisons every output superlane (the add reads {SL} bad superlanes)",
              len(ev) == SL and [d.split()[-1] for d in diags(p)] == ["OTS-E-ECC-001"] * 2 and diags(p)[-1].startswith(f"T diag {T2 + VXM_DS} "),
              (len(ev), diags(p)))
        p, T2 = spread(kind, [5])
        check(f"{kind}: a single flip of the input is corrected, nothing is poisoned",
              [l for l in lines(p, "ecc") if " unc " in l] == [] and diags(p) == [], (lines(p, "ecc")[:3], diags(p)))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
