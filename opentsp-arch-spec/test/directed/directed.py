"""Directed end-to-end tests of the source-aligned behaviour on the Sail emulator (tsp_sim).
Each scenario is a full program; expected results are computed independently here.
Covers: AGT loops (one and two levels), one-map Gather and Scatter, Repeat (issue and
errors), per-queue and chip-wide barriers, MXM matmul (i8 and u8, IW from streams, both
accumulators), SXM shift + select, typed VXM ops (u8 saturation, i16, bool), Convert,
SetSegment, and regressions of the rules of microarchitecture.md (test/c2c covers C2C).
Domain: any profile with 44 MEM slices per hemisphere: the superlane count (vector length, tile maps and splits), words per slice, bank bit
and MXM rows are read from the profile (mini: 1 superlane, 256 words; scenarios written for a larger geometry print SKIP with the reason and
are counted in the last line). A profile with fewer MEM slices (the quads the scenarios route through do not exist) is reported as skipped.
usage: directed.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines     # the normative lowering (assembly.md sec. 4)
SIM=sys.argv[1]; CFG=sys.argv[2]
# Shift and Permute cross superlanes, so they have their own timing class.
# Every latency, gap and constant below is read from the profile: the same checks must pass
# on `ref` and on the test profile `ref_slow` (longer VXM, MEM and SXM latencies).
CJ = json.load(open(CFG)); TM = CJ["timing"]
if CJ["geometry"]["mem_slices_per_hemisphere"] < 44:
    print("PASS all (skipped: the scenarios route streams through MEM slices up to 87; this profile has fewer)"); sys.exit(0)
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL     # a vector is vlanes bytes; the programs use SL tiles
WORDS = CJ["sram"]["words_per_slice"]; BANK1 = 1 << CJ["sram"]["bank_bit"]      # words per slice; the lowest address of SRAM bank 1
ROWS = CJ["mxm"]["rows"]                    # weight rows of a plane; a plane buffers rows // 16 LWs
SXM_X = TM["sxm_cross"]["d_func"]
DSX = TM["sxm"]["d_func"]                 # SXM select (and the other in-superlane units) result latency
# An entry's row is its timing.overrides row, else its class row (core/timing.sail entry_latency): Transpose has its own.
OVR = {o["entry"]: (o["d_func"], o["d_skew"]) for o in TM["overrides"]}
DTR = OVR.get("Transpose", (DSX, TM["sxm"]["d_skew"]))[0]      # Transpose's result latency
PERM_X = OVR.get("Permute", (SXM_X, TM["sxm_cross"]["d_skew"]))[0]      # Permute's result latency
# A VXM instruction dispatched at T reads its operands at T + d_skew and commits at T + d_func.
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]     # a Read/Gather commits at dispatch + d_func; a Write writes at dispatch + d_skew
D_LW, D_IW = TM["mxm_lw"]["d_func"], TM["mxm_iw"]["d_func"]
D_ABC, D_ACC = TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"]
BAR = CJ["barrier"]["release_latency"]   # a Sync is released BAR cycles after the Notify that releases it
def S(*cs): return [c + DM for c in cs]       # the commit cycles of MEM Reads/Gathers dispatched at cs
def run(recs, sram={}, dumps=[]):
    lines=[f"S {sl:x} {a:x} "+" ".join(f"{b:x}" for b in v[:VL]) for (sl,a),v in sram.items()]      # words are written for 20 superlanes: a narrower profile keeps the first vlanes bytes
    lines+=q_lines(lower_records(recs))+[f"D {sl:x} {a:x}" for sl,a in dumps]
    o=subprocess.run([SIM,CFG,"--trace"],input="\n".join(lines)+"\n",capture_output=True,text=True,timeout=600).stdout
    words={(int(t[1],16),int(t[2],16)):bytes(int(x,16) for x in t[3:])[:VL] for t in (l.split() for l in o.splitlines()) if t and t[0]=='W'}
    return o, words
def mem(op, addr=0, s=0): return (1<<60)|(op<<54)|(addr<<32)|(s<<26)
def agt(op, v): return (1<<60)|(op<<54)|((v&0xFFFF)<<16)
def E(i): return i
def W(i): return 32+i
ok=True
SKIPPED=[]
def check(name, cond, info="", needs=""):
    """`needs`: why this scenario does not apply to this profile (a geometry the scenario is written for); it is then reported as SKIP, never as PASS."""
    global ok
    if needs:
        print("SKIP "+name+" ("+needs+")"); SKIPPED.append(name); return
    print(("PASS " if cond else "FAIL ")+name, "" if cond else info); ok &= cond
# 1. AGT read loop: west MEM slice 4 (quad 1) reads addr 16, 18, 20 onto E:0; west slice 0 writes E:0 each cycle.
#    Slice 4 writes E at gap 17; slice 0 reads E at gap 18: transit 1.
sram={(4,16+2*k):bytes([k+1]*VL) for k in range(3)}
T=10
recs=[(T,4,1,mem(0,16,E(0)),0),(T+1,4,1,agt(5,1),0),(T+2,4,1,agt(6,2),0),(T+3,4,1,agt(7,3),0)]
# references at T (ref 0), T+2, T+4: period 1 + countdown, running while the configuration
# arrives (the Step at T+2 applies to the reference of T+2)
cyc=[T,T+2,T+4]
for k,c in enumerate(cyc):
    recs.append((c+DM+1,0,1,mem(1,100+k,E(0)),0))
o,wd=run(recs,sram,[(0,100+k) for k in range(3)])
check("AGT read loop: 3 references, stride 2, countdown 1", all(wd.get((0,100+k),b'')[:1]==bytes([k+1]) for k in range(3)), [l for l in o.splitlines() if 'diag' in l][:3])
# 2. Gather: one map stream, per tile address. Map word at slice 8 (quad 2 west, writes E gap 16): tile g address = 40+g
m=bytearray(VL)
for g in range(SL): m[16*g]=40+g; m[16*g+1]=0
sram2={(8,1):bytes(m)}
for g in range(SL): sram2[(0,40+g)]=bytes([g]*VL)
# Gather on slice 0 reads map E:1 at gap 18 at cycle T (slice 8 writes gap 16: transit 2) -> read at T-5-2
T=30
recs=[(T-DM-2,8,1,mem(0,1,E(1)),0),(T,0,1,(1<<60)|(3<<54)|(E(2)<<42)|(E(1)<<36),0)]
# result on E:2 at gap 18 written by slice 0 at T+5; consumer: west slice 20 (quad 5) reads E? west MEM reads E at 18-q -> upstream.
# Consumer on E downstream of 18: VXM reads at 21 -> instead use east MEM quad q reading E at 24+q: slice 44 reads E at 24 -> transit 6
recs.append((T+DM+6,44,1,mem(1,7,E(2)),0))
o,wd=run(recs,sram2,[(44,7)])
w=wd.get((44,7),b'')
check("Gather k=0: tile g takes lane 16g of word SRAM[40+g]", len(w)==VL and all(w[16*g:16*g+16]==bytes([g]+[0]*15) for g in range(SL)), [l for l in o.splitlines() if 'diag' in l][:3])
# 3. Repeat as the queue's first instruction has nothing to repeat.
o,_=run([(10,88,1,0x0<<28|3<<22|(0x002<<4)|0x2,0)])
check("Repeat with no previous instruction is OTS-E-RES-001", [l for l in o.splitlines() if 'diag' in l]==["T diag 10 queue 88 OTS-E-RES-001"], o[:200])
# 4. Data-queue Sync parks only that queue
o,_=run([(5,0,1,0x0<<28|1<<22,0),(20,1,1,0x0<<28|2<<22,0),(30,0,1,mem(0,0,E(0)),0),(30,2,1,mem(0,0,E(1)),0),(60,0,1,mem(0,0,E(2)),0)])
d=[l for l in o.splitlines() if 'diag' in l]
sr=sorted(int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf"))
# queue 0: Sync at 5; its Read (text cycle 30) waits for the release at 20+35 = 55 and dispatches then; the
# next parcel is one NOP-prefix gap later (29 cycles after the previous dispatch + 1): 55+1+29 = 85. Queue 2 is not parked.
check("Queue Sync parks queue 0 only: its Read waits to 55 (no diagnostic), queue 2 reads at 30", d==[] and sr==S(30, 20+BAR, 20+BAR+30), (d,sr))
# 5. Scatter out of segment: no effect
o,wd=run([(5,0,1,agt(8,0x10)&~(0xFFFF<<16)|(0x10<<16),0)],{},[])
check("SetSegment decodes and runs", "illegal" not in o and 'diag' not in o, o[:200])

# ---------------------------------------------------------------------------------------
# Station gaps (opentsp-arch-spec/docs/topology.md sec. 4): west MEM quad q reads/
# writes E at 18-q and W at 24+q; east MEM quad q: W at 18-q, E at 24+q; VXM 21; MXM in
# 43, ACC out 2; SXM DST 39, TSP/SHF/PRM/SEL in 40, out 4. Transit = reader gap - writer gap.
def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    west = slice_ < 44
    up = (row == "E") == west
    return 18 - q if up else 24 + q
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2<<60)|(op<<54)|((ty&3)<<52)|(d<<46)|(a<<40)|(b<<34)|(m<<28)|(mode<<24)|(tx<<22)
TY = {"i8":0,"f16":1,"i32":2,"f32":3,"u8":4,"i16":5,"u16":6,"u32":7,"b8":8,"b16":9,"b32":10}
def vxm_op(op, t, d, a, b=0, mode=0): c = TY[t]; return VXM(op, c & 3, d, a, b, 0, mode, c >> 2)
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    """Read SRAM[slice_][addr] onto stream so it reaches reader_gap in read_cycle."""
    g = gap_mem(slice_, row)
    sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))

# 6. Typed VXM: u8 add_sat saturates, i16 add wraps, b8 and, through the full dataflow.
for name, t, op, xa, xb, expect in [
    ("VXM add_sat.u8 saturates at 255", "u8", 1, [200], [100], [255]),
    ("VXM add.i16 wraps", "i16", 0, [0xff, 0x7f], [0x01, 0x00], [0x00, 0x80]),
    ("VXM and.b8 is canonical true", "b8", 11, [5], [0x80], [0xff]),
]:
    w = len(xa); sram = {}; recs = []
    T = 40
    for b in range(w):   # operands on E:0.. (a) and E:4.. (b) from west slices, read by the VXM at 21 at T + d_skew
        feed(recs, sram, 4 * b, 1, bytes([xa[b]] * VL), b, "E", 21, T + VXM_DS)
        feed(recs, sram, 4 * b + 1, 2, bytes([xb[b]] * VL), 4 + b, "E", 21, T + VXM_DS)
    recs.append((T, 88, 1, vxm_op(op, t, 8, 0, 4), 0))
    for b in range(w):   # result on E:8.. at gap 21, commit T+5; east slice 44+4b reads E at 24+b
        drain(recs, 44 + 4 * b, 9, 8 + b, "E", 21, T + DV)
    o, wd = run(recs, sram, [(44 + 4 * b, 9) for b in range(w)])
    got = [wd.get((44 + 4 * b, 9), b"\0")[0] for b in range(w)]
    check(name, got == expect, (got, expect, [l for l in o.splitlines() if "diag" in l][:3]))

# 7. Convert u8 -> f32: 200 -> 0x43480000 (bytes 00 00 48 43 on 4 streams).
sram = {}; recs = []; T = 40
feed(recs, sram, 0, 1, bytes([200] * VL), 0, "E", 21, T + VXM_DS)
c = TY["u8"]
recs.append((T, 89, 1, VXM(34, c & 3, 8, 0, 0, 0, TY["f32"], c >> 2), 0))
for b in range(4):
    drain(recs, 44 + 4 * b, 9, 8 + b, "E", 21, T + DV)
o, wd = run(recs, sram, [(44 + 4 * b, 9) for b in range(4)])
got = [wd.get((44 + 4 * b, 9), b"\0")[0] for b in range(4)]
check("Convert u8 -> f32 (200.0)", got == [0x00, 0x00, 0x48, 0x43], (got, [l for l in o.splitlines() if "diag" in l][:3]))

# 8. Two-level AGT write loop: west slice 0 writes E:2 to addresses 0,1,2 | 10,11,12
#    (inner step 1 x3, outer step 8: 2 + 8 = 10). The stream E:2 carries six values
#    produced by west slice 8 (quad 2, gap 16) one per reference cycle.
T = 100
cfg = [(5, 1), (6, 1), (7, 3), (5, 1), (6, 8), (7, 2)]   # inner: countdown 1 step 1 iters 3; outer: countdown 1 step 8 iters 2
recs = [(T, 0, 1, mem(1, 0, E(2)), 0)] + [(T + 1 + i, 0, 1, agt(op, v), 0) for i, (op, v) in enumerate(cfg)]
# reference cycles (period 1 + countdown, concurrent with the configuration): T, T+2, T+4;
# the level boundary waits Issue + inner + outer countdown = 3: T+7, then T+9, T+11
ref_cycles = [T, T + 2, T + 4, T + 7, T + 9, T + 11]
sram = {}
for k, c in enumerate(ref_cycles):
    feed(recs, sram, 8, 40 + k, bytes([0x30 + k] * VL), 2, "E", 18, c)
o, wd = run(recs, sram, [(0, a) for a in (0, 1, 2, 10, 11, 12)])
got = [wd.get((0, a), b"\0")[0] for a in (0, 1, 2, 10, 11, 12)]
check("AGT two-level write loop (step 1 x3, outer step 8)", got == [0x30 + k for k in range(6)], (got, [l for l in o.splitlines() if "diag" in l][:4]))

# 9. Scatter one map: each tile g of E:1 (from west slice 4) goes to SRAM[slice 0][60+g].
m = bytearray(VL); d = bytearray(VL)
for g in range(SL):
    m[16 * g] = 60 + g; d[16 * g:16 * g + 16] = bytes([g + 1] * 16)
sram = {}; recs = []; T = 50
feed(recs, sram, 4, 1, bytes(d), 1, "E", 18, T)
feed(recs, sram, 8, 1, bytes(m), 2, "E", 18, T)
recs.append((T, 0, 1, (1 << 60) | (4 << 54) | (E(1) << 42) | (E(2) << 36), 0))
o, wd = run(recs, sram, [(0, 60 + g) for g in range(SL)])
check("Scatter k=0: lane 16g of src to SRAM[map_g]", all(wd.get((0, 60 + g), b"") == bytes(g + 1 if l == 16 * g else 0 for l in range(VL)) for g in range(SL)),
      [l for l in o.splitlines() if "diag" in l][:3])

# 10. Repeat issues: Read at 10 then Repeat 2,3 at 11 -> reads at 10, 14, 18 (d cycles between iterations: iteration k at
#     10 + 4k, counted from the repeated Read) -> commits at 15, 19, 23.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, (3 << 22) | (2 << 4) | 3, 0)], {(0, 0): bytes(VL)})
srf = [int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf")]
check("Repeat 2,3 re-issues the Read at 14 and 18", srf == S(10, 14, 18), srf)

# 11. Chip-wide barrier: control Sync at 5, Notify at 20 -> MEM Read at 54 parked, at 55 runs.
o, _ = run([(5, 0xFFFF, 1, 1 << 22, 0), (20, 0xFFFF, 1, 2 << 22, 0), (19 + BAR, 1, 1, mem(0, 0, E(0)), 0), (20 + BAR, 2, 1, mem(0, 0, E(1)), 0)])
d = [l for l in o.splitlines() if "diag" in l]
sr = sorted(int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf"))
check("Chip barrier: a Read due at 54 waits until Notify + 35 = 55 (no diagnostic); the one due at 55 is not late", d == [] and sr == S(20 + BAR, 20 + BAR), (d, sr))

# 12. MXM matmul, west plane 0: weights row r = r+1 in lane 0 only (16 rows, IW from streams),
#     activations x = 2 in lane 0 -> product[r] = 2(r+1). u8 then i8 (signed -1 weights).
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)
for name, t, wv, xv, sign in [("MXM u8 matmul via IW from streams", 2, lambda r: r + 1, 2, 1), ("MXM i8 matmul (signed)", 0, lambda r: 0xff, 3, -1)]:
    sram = {}; recs = []; T = 200
    for r in range(16):   # IW from streams W:0..15 read at gap 43 at T; producers: east slices (W row) quad q gap 18-q
        sl = 44 + 4 * (r % 11) + (r // 11)
        v = bytearray(VL); v[0] = wv(r)
        feed(recs, sram, sl, 3 + r, bytes(v), r, "W", 43, T)
    recs.append((T, 104, 1, mxm(1, t, 32 | 0, 1), 0))           # iw.t W:0 (from streams)
    T2 = T + 2
    v = bytearray(VL); v[0] = xv
    feed(recs, sram, 60, 5, bytes(v), 24, "W", 43, T2)
    recs.append((T2, 105, 1, mxm(2, t, 32 | 24), 0))            # abc.t W:24
    T3 = T2 + 20
    recs.append((T3, 106, 1, mxm(3, 2, 8, 0, 1), 0))            # acc.i32 E:8, replace, accumulator 1
    for b in range(4):                                          # ACC writes E at gap 2 at T3+20; west slice 4b reads E at 18-b
        drain(recs, 4 * b, 9, 8 + b, "E", 2, T3 + 20)
    o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
    words = [wd.get((4 * b, 9), bytes(VL)) for b in range(4)]
    vals = [int.from_bytes(bytes(words[b][l] for b in range(4)), "little", signed=True) for l in range(16)]
    exp = [sign * xv * (wv(r) if sign > 0 else 1) for r in range(16)]
    check(name, vals == exp, (vals, exp, [l for l in o.splitlines() if "diag" in l][:4]))

# 13. SXM shift + select (west): shift W:0 up 1 on a shifter, select it against W:0 where
#     sel W:1 != 0; result to E:5 at gap 4.
x = bytes(range(VL)) if False else bytes((l % 200) + 1 for l in range(VL))
sel = bytes((1 if l % 2 else 0) for l in range(VL))
sram = {}; recs = []; T = 80
feed(recs, sram, 44, 1, x, 0, "W", 40, T)
recs.append((T, 122, 1, (4 << 60) | (0 << 54) | (W(2) << 46) | (W(0) << 40) | (0 << 39) | (1 << 30), 0))   # shift W:0 up 1 -> W:2? (turn out to E)
d = [l for l in run(recs, sram)[0].splitlines() if "diag" in l]
check("SXM shift writes the turned-round row (W in -> E out); W destination is a direction error", any("OTS-E-DIR-001" in l for l in d), d)
# Shift W:0 up by 1 -> E:5 (turned round at gap 4): dst[l] = x[l-1], dst[0] = 0.
sram = {}; recs = []; T = 80
feed(recs, sram, 44, 1, x, 0, "W", 40, T)
recs.append((T, 122, 1, (4 << 60) | (0 << 54) | (E(5) << 46) | (W(0) << 40) | (0 << 39) | (1 << 30), 0))
drain(recs, 0, 9, 5, "E", 4, T + SXM_X)
o, wd = run(recs, sram, [(0, 9)])
got = wd.get((0, 9), b"")
exp = bytes([0]) + x[:VL - 1]
check("SXM shift up 1: dst[l] = src[l-1]", got == exp, (got[:8].hex(), exp[:8].hex(), [l for l in o.splitlines() if "diag" in l][:3]))
# sxselect a, b, c, sel, dst: dst[l] = a[l] / b[l] / c[l] for sel[l] =
# 0 / 1 / 2 (unshifted, north-shifted, south-shifted; ISCA 2020 Fig. 8); other values OPR-001.
def sxsel(a, b, c, m, d): return (4 << 60) | (7 << 54) | (d << 46) | (a << 40) | (b << 24) | (m << 18) | (c << 12)
y = bytes(((l * 7) % 250) + 3 for l in range(VL))
z = bytes(((l * 13) % 240) + 5 for l in range(VL))
sel3 = bytes(l % 3 for l in range(VL))
sram = {}; recs = []; T = 80
feed(recs, sram, 44, 1, x, 0, "W", 40, T)      # a = W:0
feed(recs, sram, 45, 2, y, 1, "W", 40, T)      # b = W:1
feed(recs, sram, 48, 3, sel3, 2, "W", 40, T)   # sel = W:2
feed(recs, sram, 49, 4, z, 3, "W", 40, T)      # c = W:3
recs.append((T, 125, 1, sxsel(W(0), W(1), W(3), W(2), E(5)), 0))
drain(recs, 0, 9, 5, "E", 4, T + DSX)
o, wd = run(recs, sram, [(0, 9)])
got = wd.get((0, 9), b"")
exp = bytes((x, y, z)[sel3[l]][l] for l in range(VL))
check("sxselect: dst = a, b or c by sel 0, 1, 2", got == exp, (got[:8].hex(), exp[:8].hex(), [l for l in o.splitlines() if "diag" in l][:3]))
sram = {}; recs = []
feed(recs, sram, 44, 1, x, 0, "W", 40, T)
feed(recs, sram, 45, 2, y, 1, "W", 40, T)
feed(recs, sram, 48, 3, bytes([3] + [0] * (VL - 1)), 2, "W", 40, T)
feed(recs, sram, 49, 4, z, 3, "W", 40, T)
recs.append((T, 125, 1, sxsel(W(0), W(1), W(3), W(2), E(5)), 0))
o, _ = run(recs, sram)
d = [l for l in o.splitlines() if "diag" in l]
check("sxselect with sel 3 is OPR-001", d == ["T diag 80 queue 125 OTS-E-OPR-001"], d)
# A lane whose sel is above 2 is zero, the other lanes are selected, and the
# instruction reports one OPR-001 (an exception to "a diagnostic means no effect").
sel4 = bytes((3 if l % 5 == 4 else l % 3) for l in range(VL))
sram = {}; recs = []
feed(recs, sram, 44, 1, x, 0, "W", 40, T)
feed(recs, sram, 45, 2, y, 1, "W", 40, T)
feed(recs, sram, 48, 3, sel4, 2, "W", 40, T)
feed(recs, sram, 49, 4, z, 3, "W", 40, T)
recs.append((T, 125, 1, sxsel(W(0), W(1), W(3), W(2), E(5)), 0))
drain(recs, 0, 9, 5, "E", 4, T + DSX)
o, wd = run(recs, sram, [(0, 9)])
got = wd.get((0, 9), b"")
exp = bytes(0 if sel4[l] > 2 else (x, y, z)[sel4[l]][l] for l in range(VL))
d = [l for l in o.splitlines() if "diag" in l]
check("sxselect: lanes with sel > 2 are zero, the others selected, one OPR-001",
      got == exp and d == ["T diag 80 queue 125 OTS-E-OPR-001"], (got[:10].hex(), exp[:10].hex(), d))

# ---------------------------------------------------------------------------------------
# Regressions of the rules of microarchitecture.md.
def diags(o): return [l for l in o.splitlines() if "diag" in l]
def srfs(o): return [int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf")]
SYNC, NOTIFY = 1 << 22, 2 << 22
def nop(n): return n
def repeat(n, d): return (3 << 22) | (n << 4) | d
def rep_feed(t0):
    """A value on E:2 in every cycle t0 .. t0 + 51 at the readers: Repeat n, d re-issues at p + k (d + 1), at best every
    second cycle, so two slices of one quad (8 and 9, one gap) alternate, each a Read and a Repeat 25, 1."""
    return [(t0, 8, 1, mem(0, 1, E(2)), 0), (t0 + 1, 8, 1, repeat(25, 1), 0), (t0 + 1, 9, 1, mem(0, 1, E(2)), 0), (t0 + 2, 9, 1, repeat(25, 1), 0)]
Z = {(0, a): bytes(VL) for a in range(0, 32)}

# a rejected Notify has no effect; the barrier is causal.
o, _ = run([(5, 0, 1, SYNC, 0), (20, 1, 1, NOTIFY | 1, 0), (60, 0, 1, mem(0, 0, E(0)), 0)])
check("rejected Notify (DEC-003) releases nothing: the parked Read never dispatches", diags(o) == ["T diag 20 queue 1 OTS-E-DEC-003"] and srfs(o) == [], (diags(o), srfs(o)))
o, _ = run([(5, 0, 1, SYNC | 1, 0), (20, 1, 1, NOTIFY, 0), (21, 0, 1, mem(0, 0, E(0)), 0)])
check("rejected Sync (DEC-003) parks nothing", diags(o) == ["T diag 5 queue 0 OTS-E-DEC-003"] and srfs(o) == S(21), (diags(o), srfs(o)))
o, _ = run([(5, 1, 1, NOTIFY, 0), (5, 0, 1, SYNC, 0), (4 + BAR, 0, 1, mem(0, 0, E(0)), 0), (5 + BAR, 0, 1, mem(0, 1, E(1)), 0)])
check("Notify on a lower or higher queue in the Sync's cycle releases it at 40; the Read due at 39 waits to 40, the next follows in the next cycle", diags(o) == [] and srfs(o) == S(5 + BAR, 6 + BAR), (diags(o), srfs(o)))
o, _ = run([(5, 2, 1, SYNC, 0), (5, 1, 1, NOTIFY, 0), (4 + BAR, 2, 1, mem(0, 0, E(0)), 0), (5 + BAR, 2, 1, mem(0, 1, E(1)), 0)])
check("same-cycle Notify on a lower queue releases a Sync on a higher one", diags(o) == [] and srfs(o) == S(5 + BAR, 6 + BAR), (diags(o), srfs(o)))
# a parked queue delays AGT references and Repeat iterations.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, agt(5, 4), 0), (12, 0, 1, agt(7, 4), 0), (18, 0xFFFF, 1, SYNC, 0), (18, 1, 1, NOTIFY, 0)], Z)
check("AGT references wait out a barrier, then resume shifted", srfs(o) == S(10, 15, 18 + BAR, 23 + BAR), (srfs(o), diags(o)))
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, repeat(3, 4), 0), (17, 0xFFFF, 1, SYNC, 0), (17, 1, 1, NOTIFY, 0)], Z)
check("Repeat iterations wait out a barrier, then resume shifted", srfs(o) == S(10, 15, 17 + BAR, 22 + BAR), (srfs(o), diags(o)))

# a word that fits no layout is lowered to the undecodable unit (tag 6): DEC-003.
o, _ = run([(5, 0, 1, mem(0, 0, E(0)) | 1, 0), (6, 104, 1, (1 << 32) | mxm(0, 0, 0), 0)])
check("reserved bit is DEC-003; the upper word of a 32-bit layout is lowered to the undecodable unit: DEC-003", diags(o) == ["T diag 5 queue 0 OTS-E-DEC-003", "T diag 6 queue 104 OTS-E-DEC-003"], diags(o))

# the queue stays busy through the cycle of the last reference.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, agt(5, 1), 0), (12, 0, 1, mem(0, 9, E(1)), 0)], Z)
check("a Read due in the cycle of the last AGT reference waits for the path (last + 1 + countdown = 14)", diags(o) == [] and srfs(o) == S(10, 12, 14), (srfs(o), diags(o)))

# AGT failure cases: stray configuration records; a fifth level aborts the loop; the loop
# opens at dispatch, so a reference 0 whose execution fails still runs its loop.
o, _ = run([(5, 0, 1, agt(5, 2), 0), (20, 0, 1, mem(0, 0, E(0)), 0), (21, 0, 1, agt(7, 0), 0), (23, 0, 1, agt(7, 1), 0)], Z)
check("stray AGT config is OPR-001", diags(o) == ["T diag 5 queue 0 OTS-E-OPR-001", "T diag 23 queue 0 OTS-E-OPR-001"], diags(o))
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0)] + [(11 + i, 0, 1, agt(7, 2), 0) for i in range(5)], Z)
# the loop runs while it is configured (references at 10..14); the fifth level at 15 aborts it
check("a fifth level is OPR-001 and aborts the loop", diags(o) == ["T diag 15 queue 0 OTS-E-OPR-001"] and srfs(o) == S(10, 11, 12, 13, 14), (srfs(o), diags(o)))
o, _ = run([(10, 0, 1, mem(1, 0, E(3)), 0), (11, 0, 1, agt(7, 2), 0)])
check("A reference 0 that fails still opens its loop: references at 10 and 11 are TIM-001", diags(o) == ["T diag 10 queue 0 OTS-E-TIM-001", "T diag 11 queue 0 OTS-E-TIM-001"], diags(o))

# Repeat repeats the instruction, not its AGT configuration.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, agt(7, 2), 0), (14, 0, 1, repeat(1, 2), 0)], Z)
check("Repeat skips AGT configuration records", srfs(o) == S(10, 11, 14), (srfs(o), diags(o)))

# NOP N holds its queue for N cycles.
o, _ = run([(10, 0, 1, nop(5), 0), (14, 0, 1, mem(0, 0, E(0)), 0), (15, 0, 1, mem(0, 0, E(1)), 0)], Z)
check("a lone NOP entry makes no parcel: the Reads keep their cycles (14, 15)", diags(o) == [] and srfs(o) == S(14, 15), (diags(o), srfs(o)))

# a strided address is bounds-checked, not wrapped.
o, _ = run([(10, 0, 1, mem(0, WORDS - 1, E(0)), 0), (11, 0, 1, agt(6, 2), 0), (12, 0, 1, agt(7, 1), 0)], {(0, WORDS - 1): bytes(VL)})
check("stride past the top of the slice is OPR-001", diags(o) == ["T diag 11 queue 0 OTS-E-OPR-001"], diags(o))
o, _ = run([(10, 0, 1, mem(0, 1, E(0)), 0), (11, 0, 1, agt(6, -2), 0), (12, 0, 1, agt(7, 1), 0)], Z)
check("stride below address 0 is OPR-001", diags(o) == ["T diag 11 queue 0 OTS-E-OPR-001"], diags(o))

# one Read plus one Write per cycle on a MEM queue, in different
# banks; any other pair of records is QUE-001 for every record, and neither dispatches.
sram = {(4, 1): bytes([0x55] * VL), (0, 2): bytes([0x66] * VL)}
recs = [(30 - DM, 4, 1, mem(0, 1, E(0)), 0), (31, 0, 1, mem(1, BANK1 | 3, E(0)), 0), (31, 0, 1, mem(0, 2, E(1)), 0)]
o, wd = run(recs, sram, [(0, BANK1 | 3)])
check("Read + Write in different banks dispatch together", diags(o) == [] and wd.get((0, BANK1 | 3), b"\0")[0] == 0x55 and srfs(o).count(31 + DM) == 1, (srfs(o), diags(o)))
recs = [(30 - DM, 4, 1, mem(0, 1, E(0)), 0), (31, 0, 1, mem(1, 3, E(0)), 0), (31, 0, 1, mem(0, 2, E(1)), 0)]
o, wd = run(recs, sram, [(0, 3)])
check("Read + Write in one bank are MEM-001, neither runs", diags(o) == ["T diag 31 queue 0 OTS-E-MEM-001"] * 2 and wd.get((0, 3), b"\0")[0] == 0 and 31 + DM not in srfs(o), (srfs(o), diags(o)))
o, _ = run([(31, 0, 1, mem(0, 2, E(1)), 0), (31, 0, 1, mem(0, BANK1 | 3, E(2)), 0)], Z)
check("two Reads are QUE-001 for each, neither runs", diags(o) == ["T diag 31 queue 0 OTS-E-QUE-001"] * 2 and srfs(o) == [], (srfs(o), diags(o)))

# Patching an object (porting-guide.md sec. 5, "Patching an assembled object"; assembly.md sec. 5): the bank rule is checked by the
# assembler on source text, so an operand patched into the units of an assembled object bypasses it. The simulator
# still enforces it at dispatch: both references of the parcel are dropped and the Read's stream commit disappears.
# A patcher must therefore re-check the bank rule on every parcel it touches; bank_clash() is that check.
def sim_lines(lines):
    return subprocess.run([SIM, CFG, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600).stdout
def bank_clash(a, b): return (a ^ b) & BANK1 == 0                     # two SRAM word addresses in one bank
def patch_write_address(qline, old_addr, new_addr):
    """Rewrite the address field (unit bits 12..0 of the first unit of a MEM word) of the Write whose address is old_addr in a Q line."""
    toks = qline.split(); old_u0 = (mem(1, old_addr, E(0)) >> 32) & 0x7FFFFFFF; hits = 0
    for i in range(2, len(toks)):
        u = int(toks[i], 16)
        if u & 0x7FFFFFFF == old_u0:
            toks[i] = f"{(u & 0x80000000) | ((mem(1, new_addr, E(0)) >> 32) & 0x7FFFFFFF):08x}"; hits += 1
    assert hits == 1, hits
    return " ".join(toks)
sram_lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
legal = [(30 - DM, 4, 1, mem(0, 1, E(0)), 0), (31, 0, 1, mem(1, BANK1 | 3, E(0)), 0), (31, 0, 1, mem(0, 2, E(1)), 0)]
q_legal = q_lines(lower_records(legal))
o = sim_lines(sram_lines + q_legal + [f"D 0 {BANK1 | 3:x}"])
check("the legal parcel (Write in bank 1, Read in bank 0) dispatches both references", diags(o) == [] and srfs(o).count(31 + DM) == 1, (srfs(o), diags(o)))
patched = [patch_write_address(l, BANK1 | 3, 3) if l.startswith("Q 0 ") else l for l in q_legal]
check("the patcher's re-check flags the patched parcel (Read 2 and Write 3 share a bank) and passes the legal one",
      bank_clash(2, 3) and not bank_clash(2, BANK1 | 3), (bank_clash(2, 3), bank_clash(2, BANK1 | 3)))
o = sim_lines(sram_lines + patched + ["D 0 3"])
check("the same text patched to one bank: MEM-001 for both references, no Read commit, no Write (what the re-check prevents)",
      diags(o) == ["T diag 31 queue 0 OTS-E-MEM-001"] * 2 and 31 + DM not in srfs(o), (srfs(o), diags(o)))
a, b = (10, 90, 1, vxm_op(0, "i8", 1, 0, 0), 0), (10, 90, 1, vxm_op(0, "i8", 2, 0, 0), 0)
oa, ob = run([a, b])[0], run([b, a])[0]
check("duplicate instructions in one parcel: order independent, none dispatches", diags(oa) == diags(ob) == ["T diag 10 queue 90 OTS-E-QUE-001"] * 2, (diags(oa), diags(ob)))

# MXM and SXM unit state is read as of the start of the cycle; two
# same-cycle instructions that change the same state of one plane (or map) are both
# OTS-E-RES-003 and neither takes effect, whatever their queue order. ACC uses the newest
# ABC product committed by its dispatch cycle.
def mxm_prog(t_iw, abc_q, acc_q, abc2=None, acc_at=None, xv=2, t=2):
    sram = {}; recs = []; T = 200
    for r in range(16):
        sl = 44 + 4 * (r % 11) + (r // 11)
        v = bytearray(VL); v[0] = r + 1
        feed(recs, sram, sl, 3 + r, bytes(v), r, "W", 43, T)
    recs.append((T, 104, 1, mxm(1, t, 32 | 0, 1), 0))
    T2 = T + 2
    v = bytearray(VL); v[0] = xv
    feed(recs, sram, 60, 5, bytes(v), 24, "W", 43, T2)
    recs.append((T2, abc_q, 1, mxm(2, t, 32 | 24), 0))
    if abc2 is not None:
        v = bytearray(VL); v[0] = abc2[1]
        feed(recs, sram, 61, 6, bytes(v), 25, "W", 43, abc2[0])
        recs.append((abc2[0], abc_q, 1, mxm(2, t, 32 | 25), 0))
    T3 = acc_at or T2 + 20
    recs.append((T3, acc_q, 1, mxm(3, 2, 8, 0, 1), 0))
    for b in range(4):
        drain(recs, 4 * b, 9, 8 + b, "E", 2, T3 + 20)
    o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
    words = [wd.get((4 * b, 9), bytes(VL)) for b in range(4)]
    return [int.from_bytes(bytes(words[b][l] for b in range(4)), "little", signed=True) for l in range(4)], diags(o)
# ABC#2 dispatched in the cycle ACC reads: ACC sees ABC#1's committed product either way.
ra = mxm_prog(1, 105, 106, abc2=(222, 5), acc_at=222)
rb = mxm_prog(1, 106, 105, abc2=(222, 5), acc_at=222)
check("ACC uses the product committed by its cycle, queue order irrelevant", ra == rb == ([2, 4, 6, 8], []), (ra, rb))
# ACC after ABC#2 commits sees ABC#2's product.
r = mxm_prog(1, 105, 106, abc2=(222, 5), acc_at=242)
check("a later ACC sees the newer ABC product", r == ([5, 10, 15, 20], []), r)
# ACC with no ABC since reset.
o, _ = run([(10, 106, 1, mxm(3, 2, 8, 0, 1), 0)])
check("ACC with no committed ABC is RES-001", diags(o) == ["T diag 10 queue 106 OTS-E-RES-001"], diags(o))
# two IW-from-streams on two queues of one plane in one cycle: both RES-003, nothing installed.
def two_iw(qa, qb):
    sram = {}; recs = []; T = 200
    for r in range(16):
        sl = 44 + 4 * (r % 11) + (r // 11)
        feed(recs, sram, sl, 3 + r, bytes(VL), r, "W", 43, T)
    recs += [(T, qa, 1, mxm(1, 2, 32 | 0, 1), 0), (T, qb, 1, mxm(1, 2, 32 | 0, 1), 0), (T + 5, 106, 1, mxm(2, 2, 32 | 0), 0)]
    return diags(run(recs, sram)[0])
da, db = two_iw(104, 105), two_iw(105, 104)
check("two installs on one plane in one cycle are RES-003 for both", sorted(da) == sorted(db) and da[:2] == ["T diag 200 queue 104 OTS-E-RES-003", "T diag 200 queue 105 OTS-E-RES-003"] and "T diag 205 queue 106 OTS-E-RES-001" in da, (da, db))

# two writes to one row/stream/gap in one cycle are SRF-002 on each writer queue; the
# lowest (queue, plane) writer keeps its value, whatever the record order.
def collide(order):
    sram = {}; recs = []; T = 40
    feed(recs, sram, 0, 1, bytes([3] * VL), 0, "E", 21, T + VXM_DS)
    feed(recs, sram, 1, 2, bytes([5] * VL), 4, "E", 21, T + VXM_DS)
    ops = [(T, 89, 1, vxm_op(0, "i8", 8, 4, 4), 0), (T, 88, 1, vxm_op(0, "i8", 8, 0, 0), 0)]
    recs += ops if order else ops[::-1]
    drain(recs, 44, 9, 8, "E", 21, T + DV)
    o, wd = run(recs, sram, [(44, 9)])
    return wd.get((44, 9), b"\0")[0], diags(o)
ca, cb = collide(True), collide(False)
check("SRF-002 on both writers, the lowest queue's value is kept", ca == cb == (6, [f"T diag {40 + DV} queue 88 OTS-E-SRF-002", f"T diag {40 + DV} queue 89 OTS-E-SRF-002"]), (ca, cb))

# Gather/Scatter move one element (lane k) per tile.
def gs(op, d, m, k=0, t=0): return (1 << 60) | (op << 54) | (t << 53) | (k << 48) | (d << 42) | (m << 36)
m = bytearray(VL)
for g in range(SL): m[16 * g] = 40 + g
sram = {(8, 1): bytes(m)}
for g in range(SL): sram[(0, 40 + g)] = bytes((g * 16 + l) & 0xff for l in range(VL))
T = 30
recs = [(T - DM - 2, 8, 1, mem(0, 1, E(1)), 0), (T, 0, 1, gs(3, E(2), E(1), k=5), 0), (T + DM + 6, 44, 1, mem(1, 7, E(2)), 0)]
o, wd = run(recs, sram, [(44, 7)])
w = wd.get((44, 7), b"")
exp = bytearray(VL)
for g in range(SL): exp[16 * g + 5] = sram[(0, 40 + g)][16 * g + 5]
check("Gather k=5: lane 16g+5 = byte 16g+5 of SRAM[addr_g], other lanes zero", w == bytes(exp), (w[:40].hex(), diags(o)))
m = bytearray(VL); d = bytes(range(1, 256)) + bytes(65)
for g in range(SL): m[16 * g] = 60 + g
sram = {(0, 60 + g): bytes([0xEE] * VL) for g in range(SL)}; recs = []; T = 50
feed(recs, sram, 4, 1, d, 1, "E", 18, T)
feed(recs, sram, 8, 1, bytes(m), 2, "E", 18, T)
recs.append((T, 0, 1, gs(4, E(1), E(2), k=2), 0))
o, wd = run(recs, sram, [(0, 60 + g) for g in range(SL)])
ok18 = all(wd.get((0, 60 + g), b"") == bytes(d[l] if l == 16 * g + 2 else 0xEE for l in range(VL)) for g in range(SL))
check("Scatter k=2: only byte 16g+2 of SRAM[addr_g] changes", ok18, diags(o))

# SetVL gates SRAM writes; liquid threads (Config SL on a MEM queue, thread
# tag in bit 53): a tile executes only instructions of its own thread.
SETVL = lambda n: (6 << 22) | n
CONFIG = lambda v: (5 << 22) | v
def write_t(addr, s, t): return mem(1, addr, s) | (t << 53)
def vl_case(extra, t):
    sram = {(0, 9): bytes([0xEE] * VL)}; recs = list(extra); T = 50
    feed(recs, sram, 4, 1, bytes([7] * VL), 1, "E", 18, T)
    recs.append((T, 0, 1, write_t(9, E(1), t), 0))
    o, wd = run(recs, sram, [(0, 9)])
    return wd.get((0, 9), b""), diags(o)
w, dg = vl_case([(10, 0, 1, SETVL(3), 0)], 0)
check("SetVL 3: a Write changes tiles 0..2 only", w == bytes([7] * 48 + [0xEE] * (VL - 48)) and dg == [], (w[:64].hex(), dg), needs=SL < 4 and "needs at least 4 superlanes")
w, dg = vl_case([(10, 0, 1, CONFIG(6), 0)], 1)
check("Config SL=6: a thread-1 Write changes tiles 7..19 only", w == bytes([0xEE] * 112 + [7] * 208) and dg == [], (w[:128].hex(), dg), needs=SL < 20 and "written for the 7 / 13 tile split of 20 superlanes")
w, dg = vl_case([(10, 0, 1, CONFIG(6), 0)], 0)
check("Config SL=6: a thread-0 Write changes tiles 0..6 only", w == bytes([7] * 112 + [0xEE] * 208) and dg == [], (w[:128].hex(), dg), needs=SL < 20 and "written for the 7 / 13 tile split of 20 superlanes")
sram = {(4, 1): bytes([9] * VL)}
o, wd = run([(10, 4, 1, CONFIG(1), 0), (20, 4, 1, mem(0, 1, E(0)) | (1 << 53), 0), (20 + DM + 1, 0, 1, mem(1, 3, E(0)), 0)], sram, [(0, 3)])
check("a thread-1 Read drives only thread-1 tiles (zero elsewhere)", wd.get((0, 3), b"") == bytes([0] * 32 + [9] * (VL - 32)), (wd.get((0, 3), b"")[:48].hex(), diags(o)), needs=SL < 3 and "needs at least 3 superlanes")
# Both liquid threads of slice 4 Read to E:0 in one cycle. Each tile drives its own
# superlane, so the two results merge by tile: one commit, no SRF-002.
sram = {(4, 1): bytes([0x11] * VL), (4, 2): bytes([0x22] * VL)}
o, wd = run([(10, 4, 1, CONFIG(9), 0), (20, 4, 1, mem(0, 1, E(0)), 0), (20, 4, 1, mem(0, 2, E(0)) | (1 << 53), 0), (20 + DM + 1, 0, 1, mem(1, 3, E(0)), 0)], sram, [(0, 3)])
check("two liquid threads' Reads to one stream merge by tile (one commit, no SRF-002)",
      wd.get((0, 3), b"") == bytes([0x11] * 160 + [0x22] * 160) and diags(o) == [] and srfs(o).count(20 + DM) == 1,
      (wd.get((0, 3), b"")[::16].hex(), diags(o), srfs(o)), needs=SL < 20 and "written for the 10 / 10 tile split of 20 superlanes")

# StreamStep s: each reference of the level uses stream id + s more.
sram = {(4, 16 + k): bytes([k + 1] * VL) for k in range(3)}
recs = [(10, 4, 1, mem(0, 16, E(0)), 0), (11, 4, 1, agt(5, 4), 0), (12, 4, 1, agt(6, 1), 0), (13, 4, 1, agt(11, 2), 0), (14, 4, 1, agt(7, 3), 0)]
cyc = [10, 15, 20]
for k, c in enumerate(cyc):
    recs.append((c + DM + 1, k, 1, mem(1, 100, E(2 * k)), 0))
o, wd = run(recs, sram, [(k, 100) for k in range(3)])
check("StreamStep 2: references go to E:0, E:2, E:4", [wd.get((k, 100), b"\0")[0] for k in range(3)] == [1, 2, 3] and diags(o) == [], ([wd.get((k, 100), b"\0")[0] for k in range(3)], diags(o)))
o, _ = run([(10, 4, 1, mem(0, 16, E(30)), 0), (11, 4, 1, agt(11, 2), 0), (12, 4, 1, agt(7, 1), 0)], sram)
check("a stepped stream past E:31 is OPR-001", diags(o) == ["T diag 11 queue 4 OTS-E-OPR-001"], diags(o))

# a non-configuration instruction in the cycle after the last configuration record ends the
# configuration; the loop is then running, so that instruction waits for its path.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, agt(5, 3), 0), (12, 0, 1, mem(0, 1, E(1)), 0)], Z)
check("the Read that ends a configuration waits for the loop's path: last reference 14 + 1 + countdown 3 = 18", diags(o) == [] and srfs(o) == S(10, 14, 18), (srfs(o), diags(o)))

# Directed checks for barriers, AGT and matrix state.
o, _ = run([(5, 0xFFFF, 1, SYNC, 0), (5, 0, 1, NOTIFY, 0), (4 + BAR, 1, 1, mem(0, 0, E(0)), 0), (5 + BAR, 1, 1, mem(0, 1, E(1)), 0)], Z)
check("a Notify in the Sync's own cycle releases it at +35 (the Read due at 39 dispatches at 40)", diags(o) == [] and srfs(o) == S(5 + BAR, 6 + BAR), (diags(o), srfs(o)))
o, _ = run([(5, 0xFFFF, 1, SYNC, 0), (20, 0xFFFF, 1, NOTIFY, 0), (100, 0xFFFF, 1, NOTIFY, 0), (19 + BAR, 1, 1, mem(0, 0, E(0)), 0), (25 + BAR, 1, 1, mem(0, 1, E(1)), 0)], Z)
check("the first Notify after a Sync releases it (a Read due at 54 dispatches at 55; the next 5 cycles later)", diags(o) == [] and srfs(o) == S(20 + BAR, 26 + BAR), (diags(o), srfs(o)))
o, _ = run([(5, 0, 1, SYNC, 0), (20, 0, 1, NOTIFY, 0), (30, 0, 1, mem(0, 0, E(0)), 0), (20 + BAR, 0, 1, mem(0, 1, E(1)), 0)], Z)
check("a Notify on its own parked queue is not parked (the Read due at 30 dispatches at 55, the next 25 cycles later)", diags(o) == [] and srfs(o) == S(20 + BAR, 45 + BAR), (diags(o), srfs(o)))
sram = {(4, 20 - 2 * k): bytes([k + 1] * VL) for k in range(3)}
recs = [(10, 4, 1, mem(0, 20, E(0)), 0), (11, 4, 1, agt(5, 1), 0), (12, 4, 1, agt(6, -2), 0), (13, 4, 1, agt(7, 3), 0)]
recs += [(c + DM + 1, 0, 1, mem(1, 100 + k, E(0)), 0) for k, c in enumerate([10, 12, 14])]
o, wd = run(recs, sram, [(0, 100 + k) for k in range(3)])
check("a negative Step reads 20, 18, 16", [wd.get((0, 100 + k), b"\0")[0] for k in range(3)] == [1, 2, 3] and diags(o) == [], diags(o))

def mxm_words(recs, sram, base_cycle, n):
    """Drain ACC results committed at base_cycle + j (j < n) on E:8+4j.. into slices 4b+j, address 9."""
    for j in range(n):
        for b in range(4):
            drain(recs, 4 * b + j, 9, 8 + 4 * j + b, "E", 2, base_cycle + j)
    o, wd = run(recs, sram, [(4 * b + j, 9) for b in range(4) for j in range(n)])
    out = [[int.from_bytes(bytes(wd.get((4 * b + j, 9), bytes(VL))[l] for b in range(4)), "little", signed=True) for l in range(4)] for j in range(n)]
    return out, diags(o)
# Accumulator 1 replace, accumulator 1 accumulate, then accumulator 0 accumulate (starts at 0).
sram = {}; recs = []; T = 200
for r in range(16):
    v = bytearray(VL); v[0] = r + 1
    feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes(v), r, "W", 43, T)
recs.append((T, 104, 1, mxm(1, 2, 32, 1), 0))
v = bytearray(VL); v[0] = 2
feed(recs, sram, 60, 5, bytes(v), 24, "W", 43, T + 2)
recs.append((T + 2, 105, 1, mxm(2, 2, 32 | 24), 0))
T3 = T + 22
recs += [(T3, 106, 1, mxm(3, 2, 8, 0, 1), 0), (T3 + 1, 106, 1, mxm(3, 2, 12, 1, 1), 0), (T3 + 2, 104, 1, mxm(3, 2, 16, 1, 0), 0)]
r = mxm_words(recs, sram, T3 + 20, 3)
check("two accumulators; accumulate adds", r == ([[2, 4, 6, 8], [4, 8, 12, 16], [2, 4, 6, 8]], []), r)
# The LW buffer holds exactly `rows` stream values: rows // 16 LWs (20 on ref) fill it, the next is RES-002.
KLW = ROWS // 16
sram = {}; recs = []; T = 200
for k in range(KLW + 1):
    for r in range(16):
        feed(recs, sram, 44 + r, 1 + k, bytes([k] * VL), r, "W", 43, T + k)
    recs.append((T + k, 104, 1, mxm(0, 0, 32), 0))
o, _ = run(recs, sram)
check(f"{KLW} LWs fill the buffer; the next is RES-002", diags(o) == [f"T diag {T + KLW} queue 104 OTS-E-RES-002"], diags(o))
# IW from an f16 buffer installs half as many rows as stream values: stale buffer entries
# beyond the last LW are not installed.
sram = {}; recs = []; T = 200
one = (0x3C00).to_bytes(2, "little")
def lwfeed(base, val, t, tag):
    for r in range(16):
        st = base + r; v = bytearray(VL); v[0] = val(st)
        feed(recs, sram, 44 + (st % 44), tag + st, bytes(v), st, "W", 43, t)
lwfeed(0, lambda st: 0, T, 100); lwfeed(16, lambda st: one[st % 2], T + 1, 100)
recs += [(T, 104, 1, mxm(0, 1, 32), 0), (T + 1, 104, 1, mxm(0, 1, 32 | 16), 0), (T + 25, 104, 1, mxm(1, 1), 0)]
lwfeed(0, lambda st: 0, T + 30, 200)
recs += [(T + 30, 104, 1, mxm(0, 1, 32), 0), (T + 55, 104, 1, mxm(1, 1), 0)]
T2 = T + 57
for k in range(2):
    v = bytearray(VL); v[0] = one[k]
    sram[(80 + k, 5)] = bytes(v)
    recs.append((T2 - DM - (43 - gap_mem(80 + k, "W")), 80 + k, 1, mem(0, 5, 32 + 24 + k), 0))
recs += [(T2, 105, 1, mxm(2, 1, 32 | 24), 0), (T2 + 20, 106, 1, mxm(3, 3, 8, 0, 0), 0)]
for b in range(4):
    drain(recs, 4 * b, 9, 8 + b, "E", 2, T2 + 40)
o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
lanes = [bytes(wd.get((4 * b, 9), b"\1" * VL)[l] for b in range(4)) for l in range(16)]
check("IW.f16 from the buffer installs 8 rows per 16 stream values", all(x == bytes(4) for x in lanes) and diags(o) == [], (lanes[:10], diags(o)), needs=KLW < 2 and "needs a buffer of at least 2 LWs (mxm.rows >= 32)")

# ---------------------------------------------------------------------------------------
# diagnostics (g1d): the check order per (queue, cycle) of microarchitecture.md sec. 8,
# one test per rule; SRAM write timing; the barrier window; the reset state
# (§11); the output and trace protocol (§12); Repeat field limits.
import re
RD0 = mem(0, 0, E(0))                  # a valid Read on any MEM queue
BADW = mem(0, 0, E(0)) | 1             # reserved bit 0 set: DEC-003
MXMW = mxm(0, 0, 0)                    # an MXM LW word: valid, but no MEM unit has it
def dg(recs, sram={}): return diags(run(recs, sram)[0])
def at(c, q, code, n=1): return [f"T diag {c} queue {q} OTS-E-{code}"] * n
# Rule 1: two or more instructions in one parcel that do not fit the slots are QUE-001 for each,
# whatever they are: an instruction that does not decode reports no DEC code (generated-068).
check("R1.1 two instructions, one undecodable: QUE-001 x2, no DEC-003", dg([(10, 0, 1, BADW, 0), (10, 0, 1, RD0, 0)]) == at(10, 0, "QUE-001", 2), dg([(10, 0, 1, BADW, 0), (10, 0, 1, RD0, 0)]))
check("R1.1 three instructions: QUE-001 x3", dg([(10, 0, 1, RD0, 0)] * 3) == at(10, 0, "QUE-001", 3), dg([(10, 0, 1, RD0, 0)] * 3))
# Rule 2: a lone instruction that does not decode is reported (DEC-003) when it dispatches: a parked
# queue holds it like any other instruction.
check("R1.2 a lone NOP entry makes no parcel: the undecodable word keeps its cycle", dg([(8, 0, 1, nop(5), 0), (10, 0, 1, BADW, 0)]) == at(10, 0, "DEC-003"), dg([(8, 0, 1, nop(5), 0), (10, 0, 1, BADW, 0)]))
check("R1.2 undecodable on a parked queue waits for the release (55): DEC-003 then", dg([(5, 0, 1, SYNC, 0), (8, 0, 1, BADW, 0), (20, 1, 1, NOTIFY, 0)]) == at(20 + BAR, 0, "DEC-003"), dg([(5, 0, 1, SYNC, 0), (8, 0, 1, BADW, 0), (20, 1, 1, NOTIFY, 0)]))
# Rule 3: a parked queue holds its head parcel (no diagnostic); the instruction dispatches after the
# release (Notify at 20, release 55) and then meets its own checks.
check("R1.3 a wrong-unit word on a parked queue waits, then QUE-002 at the release", dg([(5, 0, 1, SYNC, 0), (8, 0, 1, MXMW, 0), (20, 1, 1, NOTIFY, 0)]) == at(20 + BAR, 0, "QUE-002"), dg([(5, 0, 1, SYNC, 0), (8, 0, 1, MXMW, 0), (20, 1, 1, NOTIFY, 0)]))
# Rule 4: a Repeat that runs holds the queue: the next instruction waits (no QUE-001), then meets its checks.
rep_run = [(10, 0, 1, RD0, 0), (11, 0, 1, repeat(3, 4), 0), (12, 0, 1, MXMW, 0)]
check("R1.4 wrong unit while a Repeat runs waits for the last iteration (25): QUE-002 at 26", dg(rep_run, Z) == at(26, 0, "QUE-002"), dg(rep_run, Z))
check("R1.5 wrong unit on a free queue: QUE-002", dg([(8, 0, 1, MXMW, 0)]) == at(8, 0, "QUE-002"), dg([(8, 0, 1, MXMW, 0)]))
# Rule 6: a running AGT loop (and its tail) holds the path (or, for a whole-queue instruction, the queue):
# the instruction waits, then meets its checks.
agt_run = [(10, 0, 1, RD0, 0), (11, 0, 1, agt(5, 4), 0), (12, 0, 1, agt(7, 2), 0)]   # 2 refs, period 5: at 10 and 15, tail to 20
o = run(agt_run + [(14, 0, 1, MXMW, 0)], Z)[0]
check("R1.6 wrong unit while an AGT loop runs waits for the loop and its tail (20): QUE-002 at 20", diags(o) == at(20, 0, "QUE-002") and srfs(o) == S(10, 15), (diags(o), srfs(o)))
o = run(agt_run + [(14, 0, 1, RD0, 0)], Z)[0]
check("R1.6 a Read while an AGT loop runs waits for its path: it dispatches at 20 (srf 25)", diags(o) == [] and srfs(o) == S(10, 15, 20), (diags(o), srfs(o)))
o = run(agt_run + [(15, 0, 1, MXMW, 0)], Z)[0]
check("R1.6 an instruction due in a reference's own cycle waits too", diags(o) == at(20, 0, "QUE-002"), diags(o))
# Rule 7: a Read + Write pair waits as one parcel, then dispatches together (MEM-001 and the instructions' own checks follow).
o = run([(5, 0, 1, SYNC, 0), (8, 0, 1, mem(0, 1, E(0)), 0), (8, 0, 1, mem(1, BANK1, E(0)), 0), (20, 1, 1, NOTIFY, 0)], Z)[0]
check("R1.7 Read + Write pair on a parked queue waits for the release (55), then dispatches together", diags(o) == at(20 + BAR, 0, "TIM-001") and srfs(o) == S(20 + BAR), (diags(o), srfs(o)))
check("R1.7 Read + Write pair in one bank: MEM-001 x2 (the bank rule, not a hold)", dg([(8, 0, 1, mem(0, 1, E(0)), 0), (8, 0, 1, mem(1, 2, E(0)), 0)]) == at(8, 0, "MEM-001", 2),
      dg([(8, 0, 1, mem(0, 1, E(0)), 0), (8, 0, 1, mem(1, 2, E(0)), 0)]))
# Rule 8: the instruction's own checks come last, at the cycle it issues; a failing Repeat
# iteration reports at its cycle and the next instruction waits for it.
o = run([(10, 0, 1, mem(1, 0, E(0)), 0), (11, 0, 1, repeat(1, 2), 0), (12, 0, 1, RD0, 0)], Z)[0]
check("R1.8 Repeat iteration reports at 13 (p + d + 1); the Read due at 12 waits for it (dispatches at 14, srf 19)", diags(o) == at(10, 0, "TIM-001") + at(13, 0, "TIM-001") and srfs(o) == S(14), (diags(o), srfs(o)))
# Control channel: two instructions QUE-001 each; decode; unit; never parked.
check("control channel: two instructions in a parcel are QUE-001 x2", dg([(8, 0xFFFF, 1, SYNC, 0), (8, 0xFFFF, 1, NOTIFY, 0)]) == at(8, 65535, "QUE-001", 2), dg([(8, 0xFFFF, 1, SYNC, 0), (8, 0xFFFF, 1, NOTIFY, 0)]))
check("control channel is not parked by its own Sync", dg([(5, 0xFFFF, 1, SYNC, 0), (8, 0xFFFF, 1, NOTIFY, 0)]) == [], dg([(5, 0xFFFF, 1, SYNC, 0), (8, 0xFFFF, 1, NOTIFY, 0)]))
check("control channel: Repeat is QUE-002", dg([(8, 0xFFFF, 1, repeat(1, 1), 0), (10, 0xFFFF, 1, repeat(1, 1), 0)]) == at(8, 65535, "QUE-002") + at(10, 65535, "QUE-002"),
      dg([(8, 0xFFFF, 1, repeat(1, 1), 0), (10, 0xFFFF, 1, repeat(1, 1), 0)]))
# Order within one cycle: SRF-002, control channel, queues 0..143.
o = run([(10, 0, 1, RD0, 0), (10, 1, 1, RD0, 0), (10 + DM, 0xFFFF, 1, 1 << 32, 0), (10 + DM, 3, 1, BADW, 0)], {(0, 0): bytes([1] * VL), (1, 0): bytes([2] * VL)})[0]
check("same-cycle diagnostic and dispatch order: SRF-002 (queue order), control, queues", diags(o) == at(10 + DM, 0, "SRF-002") + at(10 + DM, 1, "SRF-002") + at(10 + DM, 65535, "DEC-003") + at(10 + DM, 3, "DEC-003"), diags(o))

# A Write updates SRAM in its execute cycle t + d_skew (when it reads its stream);
# only stream results wait d_func. Write at T on slice 0 addr 2; Read of addr 2 at T + delta.
def wr_rd(delta):
    sram = {(0, 2): bytes([0x09] * VL)}; recs = []; T = 40
    feed(recs, sram, 4, 1, bytes([0x07] * VL), 0, "E", 18, T)
    recs.append((T, 0, 1, mem(1, 2, E(0)), 0))
    recs.append((T + delta, 0, 1, mem(0, 2, E(1)), 0))
    drain(recs, 44, 7, 1, "E", 18, T + delta + DM)
    o, wd = run(recs, sram, [(44, 7)])
    return wd.get((44, 7), b"\0")[0], diags(o), [l.rsplit(" ", 1)[0] for l in o.splitlines() if l.startswith("T sram") and " slice 0 " in l]   # without the value
r0, r1, r5 = wr_rd(0), wr_rd(1), wr_rd(DM)
check("R2 Write + Read of one address in one cycle: same bank, MEM-001 x2, nothing written", r0[1][:2] == at(40, 0, "MEM-001", 2) and r0[2] == [], r0)
check("R2 Read one cycle after a Write sees the new word (SRAM written at execute, MEM d_skew 0)", r1 == (0x07, [], [f"T sram {40 + MEM_DS} slice 0 addr 2"]), r1)
check("R2 Read d_func cycles after a Write sees the new word", r5 == (0x07, [], [f"T sram {40 + MEM_DS} slice 0 addr 2"]), r5)
m = bytearray(VL)
for g in range(SL): m[16 * g] = 60 + g
sram = {(0, 60): bytes([0xEE] * VL)}; recs = []; T = 50
feed(recs, sram, 4, 1, bytes([0x33] * VL), 1, "E", 18, T)
feed(recs, sram, 8, 1, bytes(m), 2, "E", 18, T)
recs += [(T, 0, 1, gs(4, E(1), E(2)), 0), (T + 1, 0, 1, mem(0, 60, E(3)), 0)]
drain(recs, 44, 7, 3, "E", 18, T + 1 + DM)
o, wd = run(recs, sram, [(44, 7)])
check("R2 Scatter updates SRAM at execute (MEM d_skew 0): a Read at +1 sees byte 0", wd.get((44, 7), b"\0")[0] == 0x33 and [l for l in o.splitlines() if l.startswith("T sram")][0].startswith("T sram 50 slice 0 addr 60 "), (wd.get((44, 7), b"")[:2], diags(o)))

# A control-channel Sync at c parks every data queue from c (it dispatches before queues
# 0..143); a data-queue Sync at c parks its queue from c + 1. Released at N + 35 for the
# first Notify N at or after the Sync. A parked queue waits (no diagnostic).
o = run([(8, 0xFFFF, 1, SYNC, 0), (8, 0, 1, RD0, 0), (8, 1, 1, NOTIFY, 0), (7 + BAR, 0, 1, RD0, 0), (8 + BAR, 0, 1, RD0, 0)], Z)[0]
check("R4 control Sync parks a data-queue instruction in its own cycle: released at 43; the next Reads follow their gaps (77, 78)", diags(o) == [] and srfs(o) == S(8 + BAR, 7 + 2 * BAR, 8 + 2 * BAR), (diags(o), srfs(o)))
o = run([(8, 0, 1, SYNC, 0), (9, 0, 1, RD0, 0), (20, 1, 1, NOTIFY, 0)], Z)[0]
check("R4 data-queue Sync at c parks its queue from c + 1: the Read waits for the release (55)", diags(o) == [] and srfs(o) == S(20 + BAR), (diags(o), srfs(o)))
check("R4 data-queue Sync at c does not park another queue at c", dg([(8, 0, 1, SYNC, 0), (8, 1, 1, RD0, 0), (9, 1, 1, RD0, 0)], Z) == [], dg([(8, 0, 1, SYNC, 0), (8, 1, 1, RD0, 0), (9, 1, 1, RD0, 0)], Z))

# Reset state (microarchitecture.md sec. 11). SRAM is zero; SetVL is every superlane;
# every tile is thread 0 (a thread-1 Read before any Config drives zeros); the segment is
# the whole slice.
o, wd = run([], {}, [(5, 77)])
check("SRAM is zero at reset", wd.get((5, 77)) == bytes(VL), wd.get((5, 77), b"")[:8])
sram = {(4, 1): bytes([9] * VL)}
o, wd = run([(20, 4, 1, mem(0, 1, E(0)) | (1 << 53), 0), (20 + DM + 1, 0, 1, mem(1, 3, E(0)), 0)], sram, [(0, 3)])
check("every tile is thread 0 at reset: a thread-1 Read drives zeros", wd.get((0, 3)) == bytes(VL) and diags(o) == [], (wd.get((0, 3), b"")[:8], diags(o)))
o, wd = run([(20, 4, 1, mem(0, 1, E(0)), 0), (20 + DM + 1, 0, 1, mem(1, 3, E(0)), 0)], sram, [(0, 3)])
check("SetVL is full at reset: a thread-0 Read drives all 20 tiles", wd.get((0, 3)) == bytes([9] * VL), wd.get((0, 3), b"")[:8])
o, _ = run([(20, 4, 1, mem(0, WORDS - 1, E(0)), 0)])
check("the segment is the whole slice at reset (the last address is in range)", diags(o) == [], diags(o))

# output protocol (microarchitecture.md sec. 12).
o, _ = run([(10, 0, 1, RD0, 0), (11, 0, 1, mem(1, 4, E(5)), 0), (12, 3, 1, BADW, 0)], {}, [(0, 4)])
lines = o.splitlines()
VLANES = 16 * json.load(open(CFG))["geometry"]["superlanes"]
HEXV = " [0-9a-f]{%d}" % (2 * VLANES)      # a trace value: a space, then vlanes bytes as lower-case hex pairs, lane 0 first
grammar = [r"T srf \d+ [EW]:\d+ gap \d+" + HEXV, r"T sram \d+ slice \d+ addr \d+" + HEXV, r"T diag \d+ queue \d+ OTS-E-[A-Z]{3}-\d{3}", r"E (OTS-E-[A-Z]{3}-\d{3} \d+|- -1) \d+", r"W [0-9a-f]+ [0-9a-f]+( [0-9a-f]{2}){320}"]
check("every output line matches the sec. 12 grammar", all(any(re.fullmatch(g, l) for g in grammar) for l in lines), lines)
# The trace carries values. T srf: the committed vector (after any merge); T sram: the word written.
# The programs are written for 20 superlanes (like the other scenarios of this file); on a narrower profile the golden vectors check the values.
if VLANES == 320:
    def tvals(o, kind): return [(int(t[2]), t[-1]) for t in (l.split() for l in o.splitlines()) if t[:2] == ["T", kind]]
    pat = bytes((7 * l + 3) & 0xFF for l in range(VLANES))
    Rd = lambda a, e: mem(0, a, E(e))
    o, _ = run([(10, 0, 1, mem(0, 4, E(5)), 0)], {(0, 4): pat}, [])
    check("T srf carries the committed vector (lane 0 first, lower-case hex)", tvals(o, "srf") == [(10 + DM, pat.hex())], (tvals(o, "srf"), o[:120]))
    sram = {}; recs = []
    feed(recs, sram, 4, 1, pat, 1, "E", 18, 50)
    recs.append((50, 0, 1, mem(1, 9, E(1)), 0))
    o, wd = run(recs, sram, [(0, 9)])
    check("T sram carries the word written (the Write's source vector)", tvals(o, "sram") == [(50 + MEM_DS, pat.hex())] and wd[(0, 9)] == pat, (tvals(o, "sram"), diags(o)))
    sram = {(0, 9): bytes([0xEE] * VLANES)}; recs = [(10, 0, 1, SETVL(3), 0)]
    feed(recs, sram, 4, 1, bytes([7] * VLANES), 1, "E", 18, 50)
    recs.append((50, 0, 1, write_t(9, E(1), 0), 0))
    o, wd = run(recs, sram, [(0, 9)])
    check("T sram of a masked Write is the whole word once merged (tiles 0..2 new, the rest old)", tvals(o, "sram") == [(50 + MEM_DS, (bytes([7] * 48) + bytes([0xEE] * (VLANES - 48))).hex())], tvals(o, "sram"))
    sram = {(4, 1): bytes([0x11] * VLANES), (4, 2): bytes([0x22] * VLANES)}
    o, wd = run([(10, 4, 1, CONFIG(9), 0), (20, 4, 1, mem(0, 1, E(0)), 0), (20, 4, 1, mem(0, 2, E(0)) | (1 << 53), 0)], sram, [])
    check("T srf of two merged liquid threads' Reads is the merged vector, on one line", tvals(o, "srf") == [(20 + DM, (bytes([0x11] * 160) + bytes([0x22] * (VLANES - 160))).hex())], tvals(o, "srf"))
    m = bytearray(VLANES)
    for g in range(SL): m[16 * g] = 60 + g
    sram = {(0, 60 + g): bytes([0xEE] * VLANES) for g in range(SL)}; recs = []; T = 50
    feed(recs, sram, 4, 1, pat, 1, "E", 18, T)
    feed(recs, sram, 8, 1, bytes(m), 2, "E", 18, T)
    recs.append((T, 0, 1, gs(4, E(1), E(2), k=2), 0))
    o, wd = run(recs, sram, [(0, 60 + g) for g in range(SL)])
    tv = [v for c, v in tvals(o, "sram")]
    exp = [bytes(pat[l] if l == 16 * g + 2 else 0xEE for l in range(VLANES)).hex() for g in range(SL)]
    check("T sram of a Scatter: one line per active tile, each the whole word written", tv == exp, (len(tv), diags(o)))
check("E line: first diagnostic (code, cycle) and the count; then W lines", [l for l in lines if l[0] in "EW"][0] == "E OTS-E-TIM-001 11 2" and lines[-1].startswith("W 0 4 "), lines)
check("no diagnostic: E - -1 0", run([], {}, [])[0].splitlines() == ["E - -1 0"], run([], {}, [])[0])

# Repeat n 12 bits (1..4095), d 4 bits (0..15): d = 0 is "no cycles between" (consecutive cycles); n = 0 does not decode.
o, _ = run([(10, 0, 1, RD0, 0), (11, 0, 1, repeat(1, 15), 0)], Z)
check("Repeat d = 15 (the largest) re-issues at p + 16", srfs(o) == S(10, 26) and diags(o) == [], (srfs(o), diags(o)))
o, _ = run([(10, 0, 1, RD0, 0), (11, 0, 1, repeat(1, 0), 0)], Z)
check("Repeat d = 0 re-issues at p + 1 (consecutive cycles)", srfs(o) == S(10, 11) and diags(o) == [], (srfs(o), diags(o)))
check("Repeat n = 0 is DEC-003, d = 0 is not", dg([(10, 0, 1, repeat(1, 0), 0), (12, 0, 1, repeat(0, 1), 0)]) == at(10, 0, "RES-001") + at(12, 0, "DEC-003"),
      dg([(10, 0, 1, repeat(1, 0), 0), (12, 0, 1, repeat(0, 1), 0)]))
# AGT: worked examples of the loop rules, checked
# for addresses and cycles. A Write on west slice 0 (queue 0) at T;
# every reference shows up as "T sram <cycle> slice 0 addr <a>" in the cycle it issues.
# Records: Countdown = agt(5, d), Step = agt(6, a), Iterations = agt(7, n).
def agt_refs(cfg, A=16, T=40):
    recs = [(T, 0, 1, mem(1, A, E(2)), 0)] + [(T + 1 + i, 0, 1, agt(op, v), 0) for i, (op, v) in enumerate(cfg)]
    # E:2 feed: west slice 8 (quad 2, gap 16; slice 0 reads E at gap 18, 7 cycles later)
    # reads in every cycle from T - 8 (rep_feed: two slices alternate). So E:2 has a value at slice 0 in T-1 .. T+50.
    recs += rep_feed(T - DM - 3)
    o, _ = run(recs, {(8, 1): bytes(VL), (9, 1): bytes(VL)})
    got = [(int(t[2]) - T, int(t[6]) - A) for t in (l.split() for l in o.splitlines()) if t[:2] == ["T", "sram"] and t[4] == "0"]
    return got, diags(o)
CD, ST, IT = 5, 6, 7
def agt_check(name, cfg, exp):
    got, d = agt_refs(cfg)
    check(name, got == exp and d == [], (got, exp, d))
# "Op(A),Step(2),AGT->Op(A),Op(A+2)"
agt_check("AGT ex.1 Op(A),Step(2) -> Op(A),Op(A+2)", [(ST, 2)], [(0, 0), (1, 2)])
# "Op(A),Cdwn(1), AGT ->Op(A),NOP,Op(A)"
agt_check("AGT ex.2 Op(A),Cdwn(1) -> Op(A),NOP,Op(A)", [(CD, 1)], [(0, 0), (2, 0)])
# "Op(A),Step(0),Cdwn(1)->Op(A),Op(A),NOP,Op(A),Op(A)"
agt_check("AGT ex.3 Op(A),Step(0),Cdwn(1) -> Op(A),Op(A),NOP,Op(A),Op(A)", [(ST, 0), (CD, 1)], [(0, 0), (1, 0), (3, 0), (4, 0)])
# "Op(A),Iter(3),Cdwn(2),Step(1) ->Op(A),Op(A),Op(A),NOP,NOP,Op(A+1),Op(A+1),Op(A+1)"
agt_check("AGT ex.4 Op(A),Iter(3),Cdwn(2),Step(1) -> 3xA, 2 NOP, 3x(A+1)", [(IT, 3), (CD, 2), (ST, 1)],
          [(0, 0), (1, 0), (2, 0), (5, 1), (6, 1), (7, 1)])
# Level boundary: "Wait(countdown_a);" is not governed by the If, so at a boundary the
# inner and the outer waits both apply: period 1 + 3 inside, 1 + 3 + 7 across the boundary.
agt_check("AGT level boundary waits inner + outer countdown", [(CD, 3), (ST, 1), (IT, 2), (CD, 7), (ST, 16), (IT, 2)],
          [(0, 0), (4, 1), (15, 17), (19, 18)])
# Iterations n makes n references; n < 1 counts as 1 (decision). Iter(0)/Iter(1) close level 0
# at once; Step(5) opens level 1 (defaults countdown 0, numIter 2).
agt_check("AGT Iterations 0 counts as 1 reference", [(IT, 0), (ST, 5)], [(0, 0), (2, 5)])
agt_check("AGT Iterations 1 is 1 reference", [(IT, 1), (ST, 5)], [(0, 0), (2, 5)])
# FIG. 10: the two-cubed dataset, element (x, y, z) at x + 2y + 4z, traversed by
# "Rd, SetStep(s0), SetStep(s1), SetStep(s2)" (three levels of the default 2 references).
for order, steps in [("XYZ", (1, 1, 1)), ("XZY", (1, 3, -3)), ("YXZ", (2, -1, 1)),
                     ("ZXY", (4, -3, -3)), ("YZX", (2, 2, -5)), ("ZYX", (4, -2, -5))]:
    pos = {"X": 0, "Y": 0, "Z": 0}; exp = []
    for k in range(8):
        for j, dim in enumerate(order):
            pos[dim] = (k >> j) & 1
        exp.append(pos["X"] + 2 * pos["Y"] + 4 * pos["Z"])
    got, d = agt_refs([(ST, s) for s in steps])
    check(f"AGT FIG. 10 traversal {order} (steps {steps})", [a for _, a in got] == exp and [c for c, _ in got] == list(range(8)) and d == [], (got, exp, d))

# ---------------------------------------------------------------------------------------
# Dispatch: each MEM queue has a read path (Read, Gather) and a write path
# (Write, Scatter), one per port of the pseudo-dual-port SRAM, each with its
# own busy state and AGT. Per cycle: one R and one W per thread, R and W of
# one thread in different banks, plus one Ifetch, plus one configuration record. A
# configuration record goes to the path that is configuring; with both (a pair), to the
# one with fewer records so far, R on a tie: a pair's records alternate R, W, R, W.
def mem_t(op, addr, s, t): return mem(op, addr, s) | (t << 53)
def refcycles(o, row_gap):
    return [int(l.split()[2]) - DM for l in o.splitlines() if l.startswith("T srf") and " ".join(l.split()[3:6]) == row_gap]
def sram_writes(o, sl):
    return [(int(l.split()[2]), int(l.split()[6])) for l in o.splitlines() if l.startswith("T sram") and l.split()[4] == str(sl)]
# A pair on west slice 43: read 10: read 0x1000, step 24 (a stride of 24 words from 0x1000
# for 112 vectors) and write 10: write 0x00ff, step 1. Iterations n makes n references, so
# the 112 vectors are written "iterations 112". Each loop runs while its configuration arrives
# With the fields its path has received so far: period 1 (countdown 0); R gets
# step 24 at T+1 (reference at T+1, then its 2 default iterations are used up and it waits)
# and iterations 112 at T+3 (references T+3 .. T+112); W gets step 1 at T+2 (reference at
# T+2) and iterations 112 at T+4 (references T+4 .. T+113).
# The write's data (W:16) comes from west slice 1 (quad 0: W gap 24; slice 43, quad 10,
# reads W at 34: transit 10; it writes E at 18 - 10 = 8): one Read of address k on slice 1
# 15 cycles before the k-th write reference.
T = 40
sram = {(43, 0x1000 + 24 * k): bytes([(k % 250) + 1] * VL) for k in range(112)}
sram.update({(1, k): bytes([(k % 200) + 3] * VL) for k in range(112)})
pair = [(T, 43, 1, mem(0, 0x1000, E(0)), 0), (T, 43, 1, mem(1, 0x00FF, W(16)), 0),
        (T + 1, 43, 1, agt(6, 24), 0), (T + 2, 43, 1, agt(6, 1), 0), (T + 3, 43, 1, agt(7, 112), 0), (T + 4, 43, 1, agt(7, 112), 0)]
rd_exp = [T, T + 1] + list(range(T + 3, T + 113))
wr_exp = [T, T + 2] + list(range(T + 4, T + 114))
wr_feed = [(c - DM - 10, 1, 1, mem(0, k, W(16)), 0) for k, c in enumerate(wr_exp)]
o, wd = run(pair + wr_feed, sram, [(43, 0xFF + k) for k in range(112)])
rd = refcycles(o, "E:0 gap 8")
wr = sram_writes(o, 43)
check("dispatch pair: read and write loops on one slice, 112 references each",
      rd == rd_exp and wr == [(c, 0xFF + k) for k, c in enumerate(wr_exp)] and diags(o) == [],
      (len(rd), len(wr), rd[:4], wr[:4], diags(o)[:3]), needs=WORDS < 8192 and "the loops walk 112 x 24 words from address 0x1000: needs 8192 words per slice")

check("dispatch pair: the write stores the 112 source vectors in order",
      all(wd.get((43, 0xFF + k), b"\0")[0] == (k % 200) + 3 for k in range(112)), [wd.get((43, 0xFF + k), b"\0")[0] for k in range(4)],
      needs=WORDS < 8192 and "the loops walk 112 x 24 words from address 0x1000: needs 8192 words per slice")
o2, _ = run([pair[1], pair[0]] + pair[2:] + wr_feed, sram)
check("dispatch pair: the loops do not depend on the order of the pair's records",
      refcycles(o2, "E:0 gap 8") == rd and sram_writes(o2, 43) == wr and diags(o2) == [], diags(o2)[:3],
      needs=WORDS < 8192 and "the loops walk 112 x 24 words from address 0x1000: needs 8192 words per slice")
# A lone loopable instruction takes every configuration record, on its own path. The
# Write's data (E:0) comes from slice 8 (write gap 16; slice 4 reads E at 17), which runs
# the same one-level loop 6 cycles earlier.
# Iterations 3, period 1: references at t, t + 1 (it runs while configuring), t + 2.
o, _ = run([(4, 8, 1, mem(0, 1, E(0)), 0), (5, 8, 1, agt(7, 3), 0),
            (5 + DM, 4, 1, mem(1, 0, E(0)), 0), (6 + DM, 4, 1, agt(7, 3), 0)], {(8, 1): bytes([5] * VL)})
check("dispatch configuration after a lone Write loops the Write (W path)", diags(o) == [] and sram_writes(o, 4) == [(5 + DM, 0), (6 + DM, 0), (7 + DM, 0)], (diags(o), sram_writes(o, 4)))
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (10, 0, 1, mem(1, BANK1, E(1)), 0), (11, 0, 1, agt(7, 1), 0), (11, 0, 1, agt(7, 1), 0)], Z)
check("dispatch two configuration records in one cycle are QUE-001", diags(o) == ["T diag 10 queue 0 OTS-E-TIM-001"] + ["T diag 11 queue 0 OTS-E-QUE-001"] * 2, diags(o))
# Ifetch issues alongside a MEM reference (one instruction fetch from streams and one read
# and one write operation per thread). Its stream E:0 (two vectors, this cycle and the next) reaches slice 4's
# read gap (17) from slice 8 (gap 16), the vectors Read in consecutive cycles.
T = 30
# The fetched text is padding (0xFFFFFFFF units, dropped), so the fetch adds nothing to queue 4's text.
sram = {(8, 1): bytes([0xFF] * VL), (8, 2): bytes([0xFF] * VL), (4, 16): bytes([9] * VL)}
IFETCH = lambda s: (4 << 22) | (s << 10)
recs = [(T - DM - 1, 8, 1, mem(0, 1, E(0)), 0), (T - DM, 8, 1, mem(0, 2, E(0)), 0),
        (T, 4, 1, IFETCH(E(0)), 0), (T, 4, 1, mem(0, 16, E(5)), 0)]
o, _ = run(recs, sram)
check("dispatch Ifetch + Read on one MEM queue in one cycle both issue", diags(o) == [] and refcycles(o, "E:5 gap 17") == [T], (diags(o), refcycles(o, "E:5 gap 17")))
o, _ = run(recs + [(T, 4, 1, mem(1, BANK1, E(6)), 0)], sram)
check("dispatch Ifetch + Read + Write: three records issue (the Write lacks data: TIM-001)", diags(o) == [f"T diag {T} queue 4 OTS-E-TIM-001"] and refcycles(o, "E:5 gap 17") == [T], diags(o))
o, _ = run([(T, 4, 1, IFETCH(E(0)), 0), (T, 4, 1, IFETCH(E(2)), 0)], sram)
check("dispatch two Ifetches in one cycle are QUE-001", diags(o) == [f"T diag {T} queue 4 OTS-E-QUE-001"] * 2, diags(o))
# One R per thread: two Reads on threads 0 and 1 issue together (Config splits the tiles:
# thread 0 has tiles 0..9, thread 1 tiles 10..19; the other tiles' superlanes are zero).
# Two Reads of one thread are QUE-001.
sram = {(0, 1): bytes([1] * VL), (0, 2): bytes([2] * VL)}
recs = [(5, 0, 1, (5 << 22) | 9, 0), (10, 0, 1, mem_t(0, 1, E(0), 0), 0), (10, 0, 1, mem_t(0, 2, E(1), 1), 0)]
o0, wd0 = run(recs + [(10 + DM + 6, 44, 1, mem(1, 7, E(0)), 0)], sram, [(44, 7)])
o1, wd1 = run(recs + [(10 + DM + 6, 44, 1, mem(1, 7, E(1)), 0)], sram, [(44, 7)])
w0, w1 = wd0.get((44, 7), b""), wd1.get((44, 7), b"")
check("dispatch Reads of threads 0 and 1 issue in one cycle; tiles 0..9 / 10..19",
      diags(o0) == [] and diags(o1) == [] and w0 == bytes([1] * 160 + [0] * 160) and w1 == bytes([0] * 160 + [2] * 160),
      (diags(o0), diags(o1), w0[:1], w0[160:161], w1[:1], w1[160:161]), needs=SL < 20 and "written for the 10 / 10 tile split of 20 superlanes")
o, _ = run([(10, 0, 1, mem_t(0, 1, E(0), 0), 0), (10, 0, 1, mem_t(0, BANK1 | 2, E(1), 0), 0)], Z)
check("dispatch two Reads of one thread are QUE-001", diags(o) == ["T diag 10 queue 0 OTS-E-QUE-001"] * 2, diags(o))
o, _ = run([(10, 0, 1, mem_t(0, 1, E(0), 0), 0), (10, 0, 1, mem_t(1, 1, E(1), 1), 0)], Z)
check("dispatch Read and Write of different threads may share a bank", "MEM-001" not in o and "QUE-001" not in o, diags(o))
# An AGT loop occupies only its own path: a Write issues while a Read loop runs, a Read
# does not. The reference cycles are taken from a run without the extra record.
base = [(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, agt(5, 3), 0), (12, 0, 1, agt(6, 1), 0), (13, 0, 1, agt(7, 3), 0)]
o, _ = run(base, Z)
r = refcycles(o, "E:0 gap 18")
check("dispatch base Read loop makes 3 references, period 1 + countdown 3", r == [10, 14, 18] and diags(o) == [], (r, diags(o)))
# On a profile this block was not written for (mini) the loop makes no references and the check above fails; the
# scenarios below take the ref cycles so that they run, and tools/gen_model_vectors.py keeps only their programs.
r = r if len(r) > 2 else [10, 14, 18]
mid = r[1] + 1 if r[2] - r[1] > 1 else r[1]
o, wd = run(base + [(mid, 0, 1, mem(1, BANK1 | 5, E(1)), 0)], Z)
check("dispatch a Write issues while a Read loop runs (W path free)", diags(o) == [f"T diag {mid} queue 0 OTS-E-TIM-001"], diags(o))
o, _ = run(base + [(mid, 0, 1, mem(0, BANK1 | 5, E(1)), 0)], Z)
check("dispatch a Read while a Read loop runs waits for the path (loop ends 18, tail 3: free at 22)", diags(o) == [] and refcycles(o, "E:0 gap 18") == r and refcycles(o, "E:1 gap 18") == [22], (diags(o), refcycles(o, "E:1 gap 18")))
SEG = agt(8, 0x10) & ~(0xFFFF << 16) | (0x10 << 16)
o, _ = run(base + [(mid, 0, 1, SEG, 0), (mid + 1, 0, 1, mem(0, 5, E(1)), 0)], Z)
check("dispatch a whole-queue instruction (SetSegment) while a loop runs waits for the loop and its tail; the next instruction follows it", diags(o) == [] and refcycles(o, "E:1 gap 18") == [23], (diags(o), refcycles(o, "E:1 gap 18")))
# The bank rule covers loop references: a Write in the bank of the Read loop's reference of
# the same cycle is MEM-001 with that reference; neither runs, and the loop goes on.
o, _ = run(base + [(r[1], 0, 1, mem(1, 9, E(1)), 0)], Z)
check("dispatch a Write in the bank of a same-cycle loop reference: both MEM-001", diags(o) == [f"T diag {r[1]} queue 0 OTS-E-MEM-001"] * 2 and refcycles(o, "E:0 gap 18") == [r[0], r[2]], (diags(o), refcycles(o, "E:0 gap 18")))
# A pair whose R path is busy waits as a whole (a parcel dispatches together).
o, _ = run(base + [(mid, 0, 1, mem(0, BANK1 | 5, E(1)), 0), (mid, 0, 1, mem(1, 6, E(2)), 0)], Z)
check("dispatch a pair waits for both paths: Read and Write dispatch together at 22 (the Write lacks data: TIM-001)", diags(o) == ["T diag 22 queue 0 OTS-E-TIM-001"] and refcycles(o, "E:1 gap 18") == [22], (diags(o), refcycles(o, "E:1 gap 18")))

# A path's loop runs while its configuration arrives; the other path is
# free (a Write in the other bank, in the parcel of the last configuration record, here without data:
# TIM-001). The parcel dispatches together.
cfg_run = [(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, agt(5, 3), 0), (12, 0, 1, agt(7, 3), 0)]
o, _ = run(cfg_run + [(12, 0, 1, mem(1, BANK1 | 5, E(1)), 0)], Z)
check("dispatch a Write on the W path in the parcel of an R configuration record issues", diags(o) == ["T diag 12 queue 0 OTS-E-TIM-001"] and refcycles(o, "E:0 gap 18") == [10, 14, 18], (diags(o), refcycles(o, "E:0 gap 18")))

# ---------------------------------------------------------------------------------------
# Regression cases for surviving mutations: one check per surviving mutant,
# each on the boundary the mutant moves. Expected values are computed here from the rules
# of microarchitecture.md (sec. 5 VXM, sec. 6 MXM, sec. 7 SXM, sec. 4 MEM, sec. 2 SRF).
WID = {"i8": 1, "u8": 1, "b8": 1, "f16": 2, "i16": 2, "u16": 2, "b16": 2, "i32": 4, "u32": 4, "f32": 4, "b32": 4}
def vx1(op, t, lanes, mode=0, dt=None):
    """One VXM op on queue 88 at T = 40 over the full dataflow: lane l of the operand (E:0..)
    is lanes[l % len(lanes)]; the result (E:8.., width of dt or t) is drained to east slices
    44 + 4b, address 9. Returns (result element per lane, diagnostics)."""
    ws, wd = WID[t], WID[dt or t]
    sram = {}; recs = []; T = 40
    df, ds = OVR.get({21: "TanH", 22: "Exp", 23: "RSqrt", 24: "Log", 33: "Reduce"}.get(op), (DV, VXM_DS))      # the op's own row, else the VXM class row
    for b in range(ws):
        feed(recs, sram, 4 * b, 1, bytes((lanes[l % len(lanes)] >> (8 * b)) & 255 for l in range(VL)), b, "E", 21, T + ds)
    recs.append((T, 88, 1, vxm_op(op, t, 8, 0, 0, mode), 0))
    for b in range(wd):
        drain(recs, 44 + 4 * b, 9, 8 + b, "E", 21, T + df)
    o, wds = run(recs, sram, [(44 + 4 * b, 9) for b in range(wd)])
    words = [wds.get((44 + 4 * b, 9), bytes(VL)) for b in range(wd)]
    return [int.from_bytes(bytes(words[b][l] for b in range(wd)), "little") for l in range(VL)], diags(o)
CVT, RED, RSQRT, EXP = 34, 33, 23, 22
F32 = {"200.0": 0x43480000, "-1.0": 0xBF800000, "256.0": 0x43800000, "255.5": 0x437F8000, "+inf": 0x7F800000,
       "-inf": 0xFF800000, "qnan": 0x7FC00000, "snan": 0x7F800001, "+0": 0, "-0": 0x80000000, "0.5": 0x3F000000, "1.5": 0x3FC00000}
# float -> unsigned saturates at the unsigned maximum (200.0 -> u8 200, not 127).
src = ["200.0", "-1.0", "256.0", "255.5", "+inf", "-inf", "qnan", "0.5", "1.5"]
r, d = vx1(CVT, "f32", [F32[k] for k in src], mode=TY["u8"], dt="u8")
check("Convert f32 -> u8: 200 -> 200, -1 -> 0, 256 -> 255, 255.5 -> 255, +Inf 255, -Inf 0, NaN 0, 0.5 -> 0, 1.5 -> 2",
      r[:9] == [200, 0, 255, 255, 255, 0, 0, 0, 2] and d == [], (r[:9], d))
f16_200, f16_inf = 0x5A40, 0x7C00
r, d = vx1(CVT, "f16", [f16_200, f16_inf, 0x7E00, 0xBC00], mode=TY["u16"], dt="u16")
check("Convert f16 -> u16: 200 -> 200, +Inf -> 65535, NaN -> 0, -1 -> 0", r[:4] == [200, 65535, 0, 0] and d == [], (r[:4], d))
# float -> bool is value != 0, and NaN and Inf count as true.
src = ["qnan", "snan", "+inf", "-inf", "+0", "-0", "0.5"]
r, d = vx1(CVT, "f32", [F32[k] for k in src], mode=TY["b8"], dt="b8")
check("Convert f32 -> b8: NaN, sNaN, +Inf, -Inf true; +0, -0 false", r[:7] == [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0xFF] and d == [], (r[:7], d))
r, d = vx1(CVT, "f16", [0x7E00, 0xFC00, 0x8000, 0x0001], mode=TY["b16"], dt="b16")
check("Convert f16 -> b16: NaN, -Inf, subnormal true; -0 false", r[:4] == [0xFFFF, 0xFFFF, 0, 0xFFFF] and d == [], (r[:4], d))
# integer transcendentals at their special inputs: rsqrt.i8(0) = 127 (saturated
# +Inf), exp.i8(0) = 1, rsqrt.i8(1) = 1, rsqrt.i8(-1) = 0 (NaN -> 0).
r, d = vx1(RSQRT, "i8", [0, 1, 4, 0xFF])
check("rsqrt.i8: 0 -> 127 (saturated +Inf), 1 -> 1, 4 -> 0 (0.5 ties to even), -1 -> 0 (NaN)", r[:4] == [127, 1, 0, 0] and d == [], (r[:4], d))
r, d = vx1(EXP, "i8", [0, 1, 0xFF, 0x80, 0x7F])
check("exp.i8: 0 -> 1, 1 -> 3, -1 -> 0, -128 -> 0, 127 -> 127", r[:5] == [1, 3, 0, 0, 127] and d == [], (r[:5], d))
# fp Reduce add folds lanes 16g..16g+15 in ascending order, rounding at each step
# (microarchitecture.md sec. 5): [1, 1, 0 x 13, 2^24] -> 2 + 2^24 = 0x4b800001 (descending
# order would absorb each 1 into 2^24: 0x4b800000).
lanes = [0x3F800000, 0x3F800000] + [0] * 13 + [0x4B800000]
r, d = vx1(RED, "f32", lanes, mode=0)
check("reduce.add.f32 is ascending-lane, rounded per step: lane 16g = 0x4b800001", all(r[16 * g] == 0x4B800001 and r[16 * g + 1:16 * g + 16] == [0] * 15 for g in range(SL)) and d == [], (hex(r[0]), d))
lanes = [0x4B800000] + [0x3F800000] * 15
r, d = vx1(RED, "f32", lanes, mode=0)
check("reduce.add.f32 [2^24, 1 x 15]: every step rounds back to 2^24 (ties to even)", r[0] == 0x4B800000 and d == [], (hex(r[0]), d))

# MXM helpers: IW from streams of 16 rows (row r value val(r) in lane 0) on plane 0.
def iws_rows(recs, sram, T, base, val=lambda r: r + 1):
    for r in range(16):
        v = bytearray(VL); v[0] = val(r) & 255
        feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), base + r, bytes(v), r, "W", 43, T)
# rows // 16 IW-from-streams installs fill the weight rows (20 on ref: 320 rows); the next is RES-002.
for n, exp in ((KLW, []), (KLW + 1, [f"T diag {200 + KLW} queue 104 OTS-E-RES-002"])):
    sram = {}; recs = []
    for k in range(n):
        iws_rows(recs, sram, 200 + k, 100 + 16 * k)
        recs.append((200 + k, 104, 1, mxm(1, 0, 32, 1), 0))
    o, _ = run(recs, sram)
    check(f"{n} IWS on one plane: " + (f"all accepted ({ROWS} rows)" if n == KLW else "the next is RES-002"), diags(o) == exp, diags(o))
# IW from the buffer waits for the LW commit: LW at 200 (d_func 20) -> IW at 219 is
# RES-001, at 220 it is accepted.
for dt, exp in ((19, ["T diag 219 queue 104 OTS-E-RES-001"]), (20, [])):
    sram = {}; recs = []
    iws_rows(recs, sram, 200, 3)
    recs += [(200, 104, 1, mxm(0, 0, 32), 0), (200 + dt, 104, 1, mxm(1, 0), 0)]
    o, _ = run(recs, sram)
    check(f"IW at LW + {dt}: " + ("RES-001" if dt == 19 else "accepted"), diags(o) == exp, diags(o))
# an IWS opens a new matrix only after an ABC dispatched before it (strictly
# earlier cycle); an ABC in the IWS's own cycle uses the old matrix and the IWS appends.
# A = rows 0..15 (r + 1) at 200; ABC (x = 1) and IWS B (rows = 50) at 205; IWS C (rows = 100)
# at 208; ABC at 211 sees A + B + C (48 rows, one matrix): lanes 0..15 = 1..16, 16..31 = 0x32,
# 32..47 = 0x64. Had the same-cycle ABC closed the append, C would start a new matrix.
sram = {}; recs = []; T = 200
iws_rows(recs, sram, T, 3); recs.append((T, 104, 1, mxm(1, 0, 32, 1), 0))
v = bytearray(VL); v[0] = 1; feed(recs, sram, 86, 5, bytes(v), 24, "W", 43, T + 5)
recs.append((T + 5, 105, 1, mxm(2, 0, 32 | 24), 0))
iws_rows(recs, sram, T + 5, 40, val=lambda r: 50); recs.append((T + 5, 106, 1, mxm(1, 0, 32, 1), 0))
iws_rows(recs, sram, T + 8, 70, val=lambda r: 100); recs.append((T + 8, 104, 1, mxm(1, 0, 32, 1), 0))
v = bytearray(VL); v[0] = 1; feed(recs, sram, 87, 6, bytes(v), 25, "W", 43, T + 11); recs.append((T + 11, 106, 1, mxm(2, 0, 32 | 25), 0))
recs.append((T + 40, 105, 1, mxm(3, 2, 8, 0, 1), 0))
for b in range(4): drain(recs, 4 * b, 9, 8 + b, "E", 2, T + 60)
o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
w0 = wd.get((0, 9), bytes(VL))
check("an ABC in the IWS's cycle does not close the append: 48 rows 1..16, 50 x 16, 100 x 16",
      list(w0[:48]) == list(range(1, 17)) + [50] * 16 + [100] * 16 and diags(o) == [], (w0[:48].hex(), diags(o)), needs=KLW < 3 and "needs a buffer of at least 3 LWs (mxm.rows >= 48)")
# f16 MXM: weights {0, +Inf} x activations {1, -1} in one row -> +Inf + -Inf =
# the canonical NaN 0x7fc00000 in f32 (MA 6).
sram = {}; recs = []; T = 200
for st in range(16):
    v = bytearray(VL)
    if st == 1: v[0] = 0x7C; v[1] = 0x7C
    feed(recs, sram, 44 + 4 * (st % 11) + (st // 11), 3 + st, bytes(v), st, "W", 43, T)
recs.append((T, 104, 1, mxm(1, 1, 32, 1), 0))
for k, b in enumerate([(0x00, 0x00), (0x3C, 0xBC)]):
    v = bytearray(VL); v[0] = b[0]; v[1] = b[1]; feed(recs, sram, 60 + k, 5, bytes(v), 24 + k, "W", 43, T + 2)
recs += [(T + 2, 105, 1, mxm(2, 1, 32 | 24), 0), (T + 22, 106, 1, mxm(3, 3, 8, 0, 1), 0)]
for b in range(4): drain(recs, 4 * b, 9, 8 + b, "E", 2, T + 42)
o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
l0 = int.from_bytes(bytes(wd.get((4 * b, 9), bytes(VL))[0] for b in range(4)), "little")
check("f16 dot +Inf + -Inf is the canonical NaN", l0 == 0x7FC00000 and diags(o) == [], (hex(l0), diags(o)))
# The assembly chapter's cancellation example on the actual MXM pipeline.
# Reuse the scheduled word pair for weight row 0 and the activation pair.
sram[48, 4] = bytes([0x3c] * 3 + [0] * 317)  # high bytes of three unit weights
sram[61, 5] = bytes([0x68, 0x3c, 0xe8] + [0] * 317)
o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
lane0 = bytes(wd[4 * b, 9][0] for b in range(4))
check("single-round f16 dot against ones preserves the unit at 2048", lane0 == bytes.fromhex("0000803f") and not diags(o),
      (lane0.hex(), diags(o)))
# Rotate n writes n^2 streams from dst: dst = 32 - n^2 is the last legal one.
def rot(dst, n=4):
    sram = {}; recs = []; T = 80
    feed(recs, sram, 44, 1, bytes((l % 250) + 1 for l in range(VL)), 0, "W", 40, T)
    recs.append((T, 120, 1, (4 << 60) | (3 << 54) | (dst << 46) | ((32 | 0) << 40) | (n << 30), 0))
    o, _ = run(recs, sram)
    return sorted(int(l.split()[3].split(":")[1]) for l in o.splitlines() if l.startswith(f"T srf {80 + DSX} E:")), diags(o)
r = rot(16)
check("Rotate n=4 dst E:16 commits E:16..31", r == (list(range(16, 32)), []), r)
r = rot(17)
check("Rotate n=4 dst E:17 is OPR-002", r == ([], ["T diag 80 queue 120 OTS-E-OPR-002"]), r)
# SetSegment n: addresses 0..n-1 are in segment, n is OPR-001 (n = 16 and the
# full slice).
def seg(n, a):
    o, _ = run([(5, 0, 1, (1 << 60) | (8 << 54) | (n << 16), 0), (10, 0, 1, mem(0, a, E(0)), 0)], {(0, a): bytes([7] * VL)})
    return srfs(o), diags(o)
check(f"SetSegment {WORDS}: address {WORDS - 1} is in segment", seg(WORDS, WORDS - 1) == ([10 + DM], []), seg(WORDS, WORDS - 1))
check("SetSegment 16: address 15 is in segment", seg(16, 15) == ([10 + DM], []), seg(16, 15))
check("SetSegment 16: address 16 is OPR-001", seg(16, 16) == ([], ["T diag 10 queue 0 OTS-E-OPR-001"]), seg(16, 16))
# a Read + Write pair is not repeatable and clears the Repeat target.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (20, 0, 1, mem(0, 1, E(1)), 0), (20, 0, 1, mem(1, BANK1 | 1, E(2)), 0), (25, 0, 1, repeat(1, 2), 0)], Z)
check("Repeat after a Read + Write pair is RES-001", srfs(o) == S(10, 20) and diags(o) == ["T diag 20 queue 0 OTS-E-TIM-001", "T diag 25 queue 0 OTS-E-RES-001"], (srfs(o), diags(o)))
# NOP is not a Repeat target: Read, NOP, Repeat re-issues the Read.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (12, 0, 1, nop(1), 0), (14, 0, 1, repeat(1, 2), 0)], Z)
check("Read, NOP, Repeat 1,2 at 14 re-issues the Read at 14 (p + d + 1 = 13 is past)", srfs(o) == S(10, 14) and diags(o) == [], (srfs(o), diags(o)))
# Sync, Notify, Deskew and RuntimeDeskew act at dispatch (barrier, queue hold), which
# an iteration does not repeat, so they are not Repeat targets: a Repeat after one is RES-001.
RTDESKEW = lambda t: (5 << 28) | (4 << 22) | t
o, _ = run([(10, 88, 1, NOTIFY, 0), (11, 88, 1, repeat(1, 1), 0)])
check("Repeat after Notify is RES-001", diags(o) == ["T diag 11 queue 88 OTS-E-RES-001"], diags(o))
o, _ = run([(10, 88, 1, RTDESKEW(1), 0), (11, 88, 1, repeat(1, 1), 0)])
check("Repeat after RuntimeDeskew is RES-001", diags(o) == ["T diag 11 queue 88 OTS-E-RES-001"], diags(o))
o, _ = run([(10, 88, 1, SYNC, 0), (12, 1, 1, NOTIFY, 0), (12 + BAR, 88, 1, repeat(1, 1), 0)])
check("Repeat after a released Sync is RES-001", diags(o) == at(12 + BAR, 88, "RES-001"), diags(o))
# An address outside the slice accesses no SRAM bank, so it clashes with no
# other reference at dispatch; its execution reports OPR-001. An AGT reference stepped to
# words_per_slice next to a Write of the same thread, and an IfetchM whose second word is outside the
# slice next to a Write in the other bank, both dispatch.
WPS0 = json.load(open(CFG))["sram"]["words_per_slice"]
o, _ = run([(10, 0, 1, mem(0, WPS0 - 2, E(0)), 0), (11, 0, 1, agt(6, 2), 0), (11, 0, 1, mem(1, 0x10, E(1)), 0)], Z)
check("an AGT reference stepped out of the slice clashes with no bank: OPR-001, the Write runs",
      sorted(diags(o)) == sorted(["T diag 11 queue 0 OTS-E-OPR-001", "T diag 11 queue 0 OTS-E-TIM-001"]), diags(o))
o, _ = run([(10, 0, 1, (7 << 22) | (WPS0 - 1), 0), (10, 0, 1, mem(1, 0x10, E(1)), 0)], Z)
check("IfetchM a+1 outside the slice uses only a's bank: OPR-001, the Write runs",
      sorted(diags(o)) == sorted(["T diag 10 queue 0 OTS-E-OPR-001", "T diag 10 queue 0 OTS-E-TIM-001"]), diags(o))
# (microarchitecture.md sec. 4 and sec. 8, "Order within a queue's cycle"):
# a configuration record is checked before the cycle's references, so a Countdown with a
# Read in its cycle is a stray OPR-001; a Read with no configuration after it leaves its path
# free; in a slot cycle the Ifetch executes before the AGT reference due that cycle.
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (10, 0, 1, agt(5, 1), 0)], Z)
check("a Countdown in its Read's own cycle is a stray OPR-001", diags(o) == ["T diag 10 queue 0 OTS-E-OPR-001"], diags(o))
o, _ = run([(10, 0, 1, mem(0, 0, E(0)), 0), (11, 0, 1, mem(0, 1, E(1)), 0)], Z)
check("a Read with no configuration record after it leaves its path free", diags(o) == [], diags(o))
o, _ = run([(10, 0, 1, mem(0, WPS0 - 2, E(0)), 0), (11, 0, 1, agt(6, 1), 0), (12, 0, 1, agt(7, 3), 0), (12, 0, 1, (4 << 22) | (2 << 10), 0)], Z)
check("in a slot cycle the Ifetch (TIM-001) reports before the AGT reference (OPR-001)",
      diags(o) == ["T diag 12 queue 0 OTS-E-TIM-001", "T diag 12 queue 0 OTS-E-OPR-001"], diags(o))
# A C2C RX queue fetches at the TX gap of its hemisphere's arriving row (W:0 from west
# slice 0 at gap 24 reaches gap 38 at T and again, a second vector, at T + 1).
TXG = json.load(open(CFG))["layout"]["c2c_tx_gap"]
sram = {}; recs = []; T = 60
for b in range(2):                                                   # one stream, two consecutive vectors
    feed(recs, sram, 0, 1 + b, bytes([0xFF] * VL), 0, "W", TXG, T + b)      # padding text: the fetch adds nothing
recs.append((T, 128, 1, (4 << 22) | (32 << 10), 0))
o, _ = run(recs, sram)
check("Ifetch W:0 on C2C RX queue 128 reads at the TX gap", diags(o) == [], diags(o))
# the SRF upstream-diagnosis horizon: VXM writes E:8 at gap 21 at 45; west slice 4
# (reads E at 17, upstream) Writes E:8 at 67, 68 (DIR-001: the value is still live
# downstream) and at 69 (TIM-001: it has left the row).
for rc, code in ((40 + DV + 22, "DIR-001"), (40 + DV + 23, "DIR-001"), (40 + DV + 24, "TIM-001")):
    sram = {}; recs = []
    feed(recs, sram, 0, 1, bytes([3] * VL), 0, "E", 21, 40 + VXM_DS)
    recs += [(40, 88, 1, vxm_op(0, "i8", 8, 0, 0), 0), (rc, 4, 1, mem(1, 9, E(8)), 0)]
    o, _ = run(recs, sram)
    check(f"upstream Write of E:8 at {rc} is {code}", diags(o) == [f"T diag {rc} queue 4 OTS-E-{code}"], diags(o))
# East hemisphere array control: two ACCs on plane 2 in one cycle are RES-003 for both and
# neither commits (the drains then find no data: TIM-001).
sram = {}; recs = []; T = 200
for r in range(16):
    v = bytearray(VL); v[0] = r + 1; feed(recs, sram, 4 * (r % 11) + (r // 11), 3 + r, bytes(v), r, "E", 43, T)
recs.append((T, 110, 1, mxm(1, 2, 0, 1), 0))
v = bytearray(VL); v[0] = 2; feed(recs, sram, 16, 5, bytes(v), 24, "E", 43, T + 2)
recs += [(T + 2, 111, 1, mxm(2, 2, 24), 0), (T + 22, 111, 1, mxm(3, 2, 32 | 8, 0, 1), 0), (T + 22, 112, 1, mxm(3, 2, 32 | 8, 0, 1), 0)]
for b in range(4): drain(recs, 44 + 4 * b, 9, 8 + b, "W", 2, T + 42)
o, _ = run(recs, sram)
check("east plane: two ACCs in one cycle are RES-003, neither commits",
      diags(o) == at(222, 111, "RES-003") + at(222, 112, "RES-003") + [f"T diag {255 + k} queue {56 - 4 * k} OTS-E-TIM-001" for k in range(4)], diags(o))
# SXM maps are per hemisphere: LoadMap of map 0 on the west PRM (124) and the east
# PRM (138) in one cycle do not clash.
sram = {}; recs = []
feed(recs, sram, 44, 1, bytes(VL), 0, "W", 40, 80)
feed(recs, sram, 0, 1, bytes(VL), 0, "E", 40, 80)
recs += [(80, 124, 1, (4 << 60) | (5 << 54) | (32 << 40), 0), (80, 138, 1, (4 << 60) | (5 << 54) | (0 << 40), 0)]
o, _ = run(recs, sram)
check("LoadMap of one map index on both hemispheres in one cycle: no clash", diags(o) == [], diags(o))

# ABC waits for the IW commit (ready_iw). With the shipped profiles (mxm_iw d_func
# 1) a same-cycle ABC already fails on "no rows installed", so the boundary is only
# observable with a longer, schema-legal IW latency: run this pair on CFG with
# timing.mxm_iw.d_func = 3. IWS at 200 -> rows ready at 203: ABC at 202 is RES-001, at 203
# it is accepted.
import json, os, tempfile
with open(CFG) as fh: cfg_iw3 = json.load(fh)
cfg_iw3["timing"]["mxm_iw"]["d_func"] = 3
fd, CFG_IW3 = tempfile.mkstemp(suffix=".json")
with os.fdopen(fd, "w") as fh: json.dump(cfg_iw3, fh)
for dt, exp in ((2, ["T diag 202 queue 105 OTS-E-RES-001"]), (3, [])):
    sram = {}; recs = []
    iws_rows(recs, sram, 200, 3); recs.append((200, 104, 1, mxm(1, 0, 32, 1), 0))
    v = bytearray(VL); v[0] = 2; feed(recs, sram, 86, 5, bytes(v), 24, "W", 43, 200 + dt)
    recs.append((200 + dt, 105, 1, mxm(2, 0, 32 | 24), 0))
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()] + q_lines(lower_records(recs))
    o = subprocess.run([SIM, CFG_IW3, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600).stdout
    check(f"(mxm_iw d_func 3) ABC at IWS + {dt}: " + ("RES-001" if dt == 2 else "accepted"), diags(o) == exp, diags(o))
os.remove(CFG_IW3)

# ================================================================================
# the SXM OR merge. HC34 p.31 draws the SEL and TSP outputs of one hemisphere
# entering an OR whose output enters register 5 of the other row (a turn-out write at gap 4
# is in register 5). A Transpose (TSP, queue 120) and an sxselect (SEL, queue 125) writing
# E:5 in the same commit cycle store the OR, with no SRF-002; a Shift (queue 122) with an
# sxselect is still SRF-002 on both queues, and the lower queue's value is kept.
def g8_feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    """As feed() above (that name is reused for a list later in this file)."""
    g = gap_mem(slice_, row)
    sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def g8_inputs(T):
    sram = {}; recs = []; ins = []
    for r in range(16):
        val = bytes(((l * (r + 3) + 7 * r) % 251) & 0xF0 for l in range(VL)) if r != 1 else bytes(VL)
        ins.append(val)
        g8_feed(recs, sram, 44 + r, 1 + r, val, r, "W", 40, T)  # W:0..15 (W:1 = 0: sel picks a)
    return sram, recs, ins
def g8_sel(recs, sram, ins, T):
    """The sxselect that commits in the cycle a Transpose dispatched at T commits (T + DTR): it dispatches DTR - DSX
    cycles later than T, so its two operands W:0 and W:1 are fed again for its own read cycle."""
    Ts = T + DTR - DSX
    if Ts != T:
        g8_feed(recs, sram, 44, 1, ins[0], 0, "W", 40, Ts)
        g8_feed(recs, sram, 45, 2, ins[1], 1, "W", 40, Ts)
    recs.append((Ts, 125, 1, sxsel(W(0), W(0), W(0), W(1), E(5)), 0))   # = W:0, low nibbles 0
def g8_run(second):
    T = 80
    sram, recs, ins = g8_inputs(T)
    g8_sel(recs, sram, ins, T)
    recs.append(second(T))
    drain(recs, 0, 9, 5, "E", 4, T + DTR)
    o, wd = run(recs, sram, [(0, 9)])
    return ins, wd.get((0, 9), b""), diags(o)
ins, got, d = g8_run(lambda T: (T, 120, 1, (4 << 60) | (4 << 54) | (E(0) << 46) | (W(0) << 40), 0))
tsp5 = bytes(ins[l % 16][16 * (l // 16) + 5] for l in range(VL))
exp = bytes(a | b for a, b in zip(ins[0], tsp5))
check("Transpose + sxselect to one stream in one cycle: OR, no SRF-002", got == exp and exp != ins[0] and exp != tsp5 and d == [], (got[:8].hex(), exp[:8].hex(), d))
# The Shift (timing class sxm_cross) dispatches SXM_X - 5 cycles before the sxselect so
# both commit at T + 5; it reads a copy of W:0 fed on W:16 at its own read cycle.
def g8_shift(T):
    Ts = T + DSX - SXM_X
    _, _, ins0 = g8_inputs(T)
    g8_feed(g8_shift.recs, g8_shift.sram, 60, 1, ins0[0], 16, "W", 40, Ts)
    return (Ts, 122, 1, (4 << 60) | (0 << 54) | (E(5) << 46) | (W(16) << 40) | (0 << 39) | (1 << 30), 0)
def g8_run_shift():
    T = 80
    sram, recs, ins = g8_inputs(T)
    g8_shift.recs, g8_shift.sram = recs, sram
    recs.append((T, 125, 1, sxsel(W(0), W(0), W(0), W(1), E(5)), 0))
    recs.append(g8_shift(T))
    drain(recs, 0, 9, 5, "E", 4, T + DSX)
    o, wd = run(recs, sram, [(0, 9)])
    return ins, wd.get((0, 9), b""), diags(o)
ins, got, d = g8_run_shift()
check("Shift + sxselect to one stream in one cycle: SRF-002 on both, queue 122 kept", got == bytes([0]) + ins[0][:VL - 1] and d == [f"T diag {80 + DSX} queue 122 OTS-E-SRF-002", f"T diag {80 + DSX} queue 125 OTS-E-SRF-002"], (got[:8].hex(), d))
# =======================================================================================
# the local-memory form of Ifetch (ISCA20 Table I: "Fetch instructions from
# streams or local memory") and the optional IQ occupancy check
# (ISCA20 sec. III-A3: "It is imperative that IQs never go empty").
import json as _json
IFETCHM = lambda a: (7 << 22) | a
WPS = _json.load(open(CFG))["sram"]["words_per_slice"]
def padw(sl, *addrs): return {(sl, a): bytes([0xFF] * VL) for a in addrs}   # text words of padding units: a fetch adds nothing
o, _ = run([(10, 4, 1, IFETCHM(16), 0)], padw(4, 16, 17))
check("ifetchm on a MEM queue issues", diags(o) == [], diags(o))
o, _ = run([(10, 88, 1, IFETCHM(16), 0)])
check("ifetchm on a VXM queue is QUE-002", diags(o) == ["T diag 10 queue 88 OTS-E-QUE-002"], diags(o))
o, _ = run([(10, 4, 1, IFETCHM(WPS - 1), 0), (11, 4, 1, IFETCHM(WPS - 2), 0)], padw(4, WPS - 2, WPS - 1))
check("ifetchm whose second word is past the slice is OPR-001", diags(o) == ["T diag 10 queue 4 OTS-E-OPR-001"], diags(o))
o, _ = run([(5, 4, 1, agt(8, 0x10), 0), (10, 4, 1, IFETCHM(15), 0), (11, 4, 1, IFETCHM(14), 0)], padw(4, 14, 15))
check("ifetchm is bounded by the segment", diags(o) == ["T diag 10 queue 4 OTS-E-OPR-001"], diags(o))
o, _ = run([(10, 4, 1, IFETCHM(16), 0), (10, 4, 1, mem(0, 0, E(0)), 0)], Z)
check("ifetchm + Read on one MEM queue are QUE-001 (one R port)", diags(o) == ["T diag 10 queue 4 OTS-E-QUE-001"] * 2 and refcycles(o, "E:0 gap 17") == [], (diags(o), refcycles(o, "E:0 gap 17")))
o, _ = run([(10, 4, 1, IFETCHM(16), 0), (10, 4, 1, IFETCH(E(0)), 0)])
check("ifetchm + Ifetch in one cycle are QUE-001", diags(o) == ["T diag 10 queue 4 OTS-E-QUE-001"] * 2, diags(o))
o, _ = run([(10, 4, 1, (7 << 22) | (1 << 13), 0)])
check("ifetchm with bit 13 set does not decode", diags(o) == ["T diag 10 queue 4 OTS-E-DEC-003"], diags(o))

# =======================================================================================
# The MEM slot rule (microarchitecture.md sec. 3, sec. 8 rule 1/7); IfetchM on
# the R port; Repeat from the repeated issue; the AGT trailing waits.
# The IQ tests are in frontend.py.
T1 = 1 << 53                               # liquid thread 1
GATHER = (1 << 60) | (3 << 54) | (2 << 42) | (1 << 36)
ZP = dict(Z); ZP.update(padw(4, *range(0, 40), BANK1 - 2, BANK1 - 1, BANK1, BANK1 + 1))     # Z, and padding text words of queue 4's slice
def codes(recs, sram=ZP): return [l.split()[-1][6:] for l in dg(recs, sram) if "QUE-001" in l or "MEM-001" in l]
R1 = [  # (name, records at cycle 20, the QUE/MEM codes expected)
    ("Read t0 + Read t1", [mem(0, 1, E(0)), mem(0, 1, E(1)) | T1], []),
    ("Write t0 + Write t1", [mem(1, 1, E(0)), mem(1, BANK1, E(1)) | T1], []),
    ("Read t0 + Write t0, other banks", [mem(0, 1, E(0)), mem(1, BANK1, E(1))], []),
    ("Read t0 + Write t1, one bank (threads never clash)", [mem(0, 1, E(0)), mem(1, 2, E(1)) | T1], []),
    ("Read t0 + Write t0, one bank", [mem(0, 1, E(0)), mem(1, 2, E(1))], ["MEM-001"] * 2),
    ("Gather t0 + Write t0 (indirect)", [GATHER, mem(1, BANK1, E(3))], ["MEM-001"] * 2),
    ("Read t0 + Read t0", [mem(0, 1, E(0)), mem(0, 2, E(1))], ["QUE-001"] * 2),
    ("Write t0 + Write t0", [mem(1, 1, E(0)), mem(1, BANK1, E(1))], ["QUE-001"] * 2),
    ("Ifetch + Read t0", [IFETCH(E(0)), mem(0, 1, E(1))], []),
    ("Countdown + Read t0", [agt(5, 4), mem(0, 1, E(0))], []),
    ("Countdown + Iterations (two config records)", [agt(5, 4), agt(7, 2)], ["QUE-001"] * 2),
    ("SetSegment (whole queue) + Read", [agt(8, 0x10), mem(0, 1, E(0))], ["QUE-001"] * 2),
    ("NOP + Read (a NOP inside a parcel is a slot violation)", [nop(1), mem(0, 1, E(0))], ["QUE-001"] * 2),
    ("Ifetch + NOP", [IFETCH(E(0)), nop(1)], ["QUE-001"] * 2),
    ("R t0, R t1, W t0, W t1, Ifetch, Countdown (every slot once)", [mem(0, 1, E(0)), mem(0, 1, E(1)) | T1, mem(1, BANK1, E(2)), mem(1, BANK1, E(3)) | T1, IFETCH(E(4)), agt(5, 4)], []),
    ("every slot + a second Read t0", [mem(0, 1, E(0)), mem(0, 1, E(1)) | T1, mem(1, BANK1, E(2)), mem(0, 3, E(5)), IFETCH(E(4))], ["QUE-001"] * 5),
    ("Read + undecodable record", [mem(0, 1, E(0)), BADW], ["QUE-001"] * 2),
]
for name, ws, exp in R1:
    got = codes([(20, 0, 1, w, 0) for w in ws])
    check(f"R1 slot rule: {name}", got == exp, got)
# IfetchM takes the R port of every tile (QUE-001 with an R-type reference of either
# thread) and its words a, a+1 must be in banks other than a same-cycle Write's (MEM-001).
IFR1 = [
    ("IfetchM + Read t0", [IFETCHM(16), mem(0, 1, E(1))], ["QUE-001"] * 2),
    ("IfetchM + Read t1", [IFETCHM(16), mem(0, 1, E(1)) | T1], ["QUE-001"] * 2),
    ("IfetchM + Gather", [IFETCHM(16), GATHER], ["QUE-001"] * 2),
    ("IfetchM + Ifetch", [IFETCHM(16), IFETCH(E(0))], ["QUE-001"] * 2),
    ("IfetchM bank 0 + Write bank 1", [IFETCHM(16), mem(1, BANK1, E(1))], []),
    ("IfetchM bank 0 + Write t0 bank 0", [IFETCHM(16), mem(1, 2, E(1))], ["MEM-001"] * 2),
    ("IfetchM bank 0 + Write t1 bank 0", [IFETCHM(16), mem(1, 2, E(1)) | T1], ["MEM-001"] * 2),
    ("IfetchM across banks + Write bank 1 is legal in the first cycle", [IFETCHM(BANK1 - 1), mem(1, BANK1 + 5, E(1))], []),
    ("IfetchM a, a+1 across banks + Write bank 0", [IFETCHM(BANK1 - 1), mem(1, 5, E(1))], ["MEM-001"] * 2),
]
for name, ws, exp in IFR1:
    got = codes([(20, 4, 1, w, 0) for w in ws])
    check(f"{name}", got == exp, got)
o, _ = run([(20, 4, 1, IFETCHM(16), 0), (20, 4, 1, mem(1, 2, E(1)), 0)], Z)
check("a clashing IfetchM + Write: neither runs", not any(l.startswith("T sram") for l in o.splitlines()), [l for l in o.splitlines() if l.startswith("T ")][:4])
# IfetchM while the R path's AGT loop runs waits for the R path (it needs it).
o = dg([(10, 4, 1, mem(0, 1, E(0)), 0), (11, 4, 1, agt(5, 4), 0), (12, 4, 1, agt(7, 2), 0), (13, 4, 1, IFETCHM(16), 0)], ZP)
check("IfetchM while the R path's loop runs waits for the path: no diagnostic", [l for l in o if "QUE" in l] == [], o)
# Iteration k of Repeat n, d at p + k (d + 1) (p: the repeated instruction's cycle; d cycles
# between iterations): Read at 10, Repeat 3,1 at 11 reads at 10, 12, 14, 16.
o, _ = run([(10, 0, 1, mem(0, 1, E(0)), 0), (11, 0, 1, repeat(3, 1), 0)], Z)
check("Read at 10, Repeat 3,1 at 11: iterations at 12, 14, 16", srfs(o) == S(10, 12, 14, 16) and diags(o) == [], (srfs(o), diags(o)))
o, _ = run([(10, 0, 1, mem(0, 1, E(0)), 0), (12, 0, 1, repeat(2, 4), 0)], Z)
check("Read at 10, Repeat 2,4 at 12: iterations at 15, 20", srfs(o) == S(10, 15, 20), (srfs(o), diags(o)))
# A Write loop (Countdown 3, 2 references: 40, 44) runs its trailing Wait(3) after the
# last reference: its path is held through 47.
TT = 40
fd = rep_feed(TT - 8)
Z2 = {(8, 1): bytes(VL), (9, 1): bytes(VL)}
for nx in (TT + 5, TT + 7, TT + 8):
    o, _ = run(fd + [(TT, 0, 1, mem(1, 16, E(2)), 0), (TT + 1, 0, 1, agt(5, 3), 0), (nx, 0, 1, mem(1, 32, E(2)), 0)], Z2)
    got = [w for w in sram_writes(o, 0) if w[1] == 32]
    check(f"a Write due at T+{nx - TT} after a loop ending T+4 (Wait 3) waits for the path: it writes at T+8", got == [(TT + 8, 32)] and not [l for l in diags(o) if "QUE-001" in l], (got, diags(o)))

# =======================================================================================
# VXM ALU chaining is explicit. A VXM operand reads the VXM result committed
# at the VXM station in the previous cycle only when its chain bit is set (".chain" in the
# text; bits 21/20/19 for the first/second/third source); every other read follows the
# flow rule, so a VXM instruction may name its own source as destination (ISCA20 Listing 1)
# and one ALU's result does not replace another ALU's operand. Expected values are computed
# here from the operands fed, not from the model's rule.
_cfg = _json.load(open(CFG))
DF, DS = _cfg["timing"]["vxm"]["d_func"], _cfg["timing"]["vxm"]["d_skew"]
# the row of max.i8: its row keyed by the type, else the entry's row, else the class row (microarchitecture.md section 1)
_rows = {o["entry"]: (o["d_func"], o["d_skew"]) for o in _cfg["timing"]["overrides"]}
MDF, MDS = _rows.get("Max.i8") or _rows.get("Max") or (DF, DS)
VG = _cfg["layout"]["vxm_gap"]
CH = lambda a=0, b=0, m=0: (a << 21) | (b << 20) | (m << 19)
def h1_feed(recs, sram, slice_, addr, val, stream, read_cycle):
    """West MEM slice_ reads SRAM[addr] onto E:stream so that it is at the VXM in read_cycle."""
    sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (VG - gap_mem(slice_, "E")), slice_, 1, mem(0, addr, E(stream)), 0))
def h1_drain(recs, addr, stream, commit_cycle):
    """East MEM slice 44 writes E:stream, as committed by the VXM in commit_cycle, to SRAM[addr]."""
    recs.append((commit_cycle + gap_mem(44, "E") - VG, 44, 1, mem(1, addr, E(stream)), 0))
# (a) ISCA20 Listing 1: z = g.add(x_strm, y_strm, stream='S_0'); x on S_0, y on S_4, one
# vector per cycle on consecutive cycles: add.i8 E:0, E:0, E:4 at T..T+11.
N = 12; T = 60; sram = {}; recs = []
for k in range(N):
    h1_feed(recs, sram, 4, 16 + k, bytes([k + 1] * VL), 0, T + k + DS)
    h1_feed(recs, sram, 5, 16 + k, bytes([100] * VL), 4, T + k + DS)
    recs.append((T + k, 88, 1, vxm_op(0, "i8", 0, 0, 4), 0))
    h1_drain(recs, 100 + k, 0, T + k + DF)
o, wd = run(recs, sram, [(44, 100 + k) for k in range(N)])
got = [wd.get((44, 100 + k), b"\0")[0] for k in range(N)]
check("ISCA20 Listing 1: add.i8 E:0, E:0, E:4 on consecutive cycles = x + y", got == [k + 1 + 100 for k in range(N)] and diags(o) == [], (got, diags(o)))
# (b) max.i8 E:0, E:0, E:4 with Repeat 54, 1 (destination = first source),
# as max.i8 E:0, E:0, E:4 at T with Repeat 54, 1 at T + 1. Operand x(c) = c mod 120 + 1 is
# fed for every read cycle c, b = 0; each issue at i must give x(i + d_skew).
T = 60; sram = {}; recs = []
xs = lambda c: c % 120 + 1      # i8: below 127 over the 55 issues, which are two cycles apart
for c in range(T + MDS, T + 112 + MDS):
    h1_feed(recs, sram, 4, c - T, bytes([xs(c)] * VL), 0, c)
    h1_feed(recs, sram, 5, c - T, bytes(VL), 4, c)
recs.append((T, 88, 1, vxm_op(9, "i8", 0, 0, 4), 0)); recs.append((T + 1, 88, 1, (3 << 22) | (54 << 4) | 1, 0))
# Repeat 54, 1 re-issues at T + 2 .. T + 108 (iteration k at p + k (d + 1),
# microarchitecture.md sec. 3), so the 55 issues are T, T + 2 .. T + 108:
want_issues = [T + 2 * j for j in range(55)]
for i in want_issues:
    h1_drain(recs, (0x400 + i) % WORDS, 0, i + MDF)
o, wd = run(recs, sram, [(44, (0x400 + i) % WORDS) for i in want_issues])
issues = [int(l.split()[2]) - MDF for l in o.splitlines() if l.startswith("T srf") and l.split()[3:6] == ["E:0", "gap", str(VG)]]
bad = [(i, wd.get((44, (0x400 + i) % WORDS), b"\0")[0], xs(i + MDS)) for i in want_issues if wd.get((44, (0x400 + i) % WORDS), b"\0")[0] != xs(i + MDS)]
check("max.i8 E:0, E:0, E:4 rep 54, 1: 55 issues every second cycle, each = its own operand", issues == want_issues and bad == [] and diags(o) == [], (issues[:4], bad[:6], diags(o)))
# (b2) The same example as ONE instruction on ONE ALU queue,
# max.i8.sg4 E:0, E:0, E:4 does an
# int8 max on each of the four streams of the groups E:0..3 and E:4..7 ("four INT8 operations
# per cycle" per ALU, ScaledML 2020). Stream b of the first group carries x_b(c), the second
# group 60; every issue i gives max(x_b(i + d_skew), 60) on stream b. Computed here.
SG4 = 1 << 18
T = 60; sram = {}; recs = []
xb = lambda b, c: (c + 37 * b) % 120 + 1
for c in range(T + MDS, T + 112 + MDS):
    for b in range(4):
        h1_feed(recs, sram, 4 + b, c - T, bytes([xb(b, c)] * VL), b, c)
        h1_feed(recs, sram, 8 + b, c - T, bytes([60] * VL), 4 + b, c)
recs.append((T, 88, 1, vxm_op(9, "i8", 0, 0, 4) | SG4, 0)); recs.append((T + 1, 88, 1, (3 << 22) | (54 << 4) | 1, 0))
want_issues = [T + 2 * j for j in range(55)]
for i in want_issues:
    for b in range(4):
        drain(recs, 44 + b, (0x400 + i) % WORDS, b, "E", VG, i + MDF)
o, wd = run(recs, sram, [(44 + b, (0x400 + i) % WORDS) for i in want_issues for b in range(4)])
bad = [(i, b) for i in want_issues for b in range(4) if wd.get((44 + b, (0x400 + i) % WORDS), b"\0")[0] != max(xb(b, i + MDS), 60)]
check("max.i8.sg4 E:0, E:0, E:4 rep 54, 1 on one ALU queue: four int8 maxes per issue",
      bad == [] and diags(o) == [] and {q for (_, q, *_) in recs if 88 <= q <= 103} == {88}, (bad[:6], diags(o)))
# A group operand that is not SG4-aligned is OTS-E-OPR-002.
o, _ = run([(60, 88, 1, vxm_op(9, "i8", 0, 1, 4) | SG4, 0)])
check("max.i8.sg4 with a source group at E:1 is OPR-002 (at its execute cycle)", diags(o) == [f"T diag {60 + MDS} queue 88 OTS-E-OPR-002"], diags(o))
# (c) Cross-queue: queue 89 writes an unrelated result to E:0 every cycle while queue 88 adds
# E:0 + E:4 into E:8. Queue 88 must see the MEM operands.
N = 12; T = 60; sram = {}; recs = []
for k in range(N):
    h1_feed(recs, sram, 4, 16 + k, bytes([k + 1] * VL), 0, T + k + DS)
    h1_feed(recs, sram, 5, 16 + k, bytes([100] * VL), 4, T + k + DS)
    h1_feed(recs, sram, 6, 16 + k, bytes([50] * VL), 12, T + k + DS)
    recs.append((T + k, 88, 1, vxm_op(0, "i8", 8, 0, 4), 0))
    recs.append((T + k, 89, 1, vxm_op(0, "i8", 0, 12, 12), 0))
    h1_drain(recs, 100 + k, 8, T + k + DF)
o, wd = run(recs, sram, [(44, 100 + k) for k in range(N)])
got = [wd.get((44, 100 + k), b"\0")[0] for k in range(N)]
check("cross-queue: another ALU's result on E:0 does not replace queue 88's operand", got == [k + 101 for k in range(N)] and diags(o) == [], (got, diags(o)))
# (d) Explicit chaining: neg.i8 E:8, E:0 on queue 88 at T (x = 5); add.i8 E:12, E:8.chain,
# E:8.chain on queue 89 at T + d_func - d_skew + 1 (it reads at T + d_func + 1) gives -10. Without the chain bits the same
# add reads the MEM value 7 that flows on E:8 (14).
for chain, want in ((1, 0xF6), (0, 14)):
    T = 60; sram = {}; recs = []
    h1_feed(recs, sram, 4, 1, bytes([5] * VL), 0, T + DS)
    recs.append((T, 88, 1, vxm_op(15, "i8", 8, 0), 0))
    t2 = T + DF - DS + 1
    h1_feed(recs, sram, 5, 2, bytes([7] * VL), 8, t2 + DS)
    recs.append((t2, 89, 1, vxm_op(0, "i8", 12, 8, 8) | CH(chain, chain), 0))
    h1_drain(recs, 0x30, 12, t2 + DF)
    o, wd = run(recs, sram, [(44, 0x30)])
    got = wd.get((44, 0x30), b"\0")[0]
    check(f"chain bits {chain}: add.i8 E:12, E:8{'.chain' if chain else ''}, E:8{'.chain' if chain else ''} = {want:#x}", got == want and diags(o) == [], (got, diags(o)))
# (e) A chained operand with no VXM result at the station in the previous cycle is OTS-E-TIM-001.
o, _ = run([(60, 89, 1, vxm_op(0, "i8", 12, 8, 8) | CH(1, 0), 0)])
check("a chained operand with no VXM result one cycle earlier is TIM-001", diags(o) == [f"T diag {60 + DS} queue 89 OTS-E-TIM-001"], diags(o))
# (f) A chain bit on an operand the instruction does not have does not decode.
o, _ = run([(60, 89, 1, vxm_op(15, "i8", 12, 8) | CH(0, 1), 0)])
check("neg with a second-operand chain bit does not decode", diags(o) == ["T diag 60 queue 89 OTS-E-DEC-003"], diags(o))
# The map carries the Permute mask (HC34 p.31 "arbitrary mapping of input + mask"). A map
# entry >= vlanes zero-fills its output lane and may repeat; the entries below vlanes must be
# distinct (OTS-E-SXM-001). East permuter: queue 138 reads E at gap 40 and writes W at gap 4.
def h1_perm(d, s, m): return (4 << 60) | (1 << 54) | (d << 46) | (s << 40) | (m << 30)
def h1_loadwide(m, s, s2): return (4 << 60) | (6 << 54) | (s << 40) | (m << 30) | (s2 << 24)
F2_SL = json.load(open(CFG))["geometry"]["superlanes"]
def h1_permute(pm):
    T = 80; sram = {}; recs = []
    feed(recs, sram, 4, 1, bytes(e & 255 for e in pm), 0, "E", 40, T)
    feed(recs, sram, 5, 1, bytes(e >> 8 for e in pm), 1, "E", 40, T)
    recs.append((T, 138, 1, h1_loadwide(3, E(0), E(1)), 0))
    G = max(10, F2_SL)      # a map is ready for a Permute `superlanes` cycles after its LoadMap
    feed(recs, sram, 6, 2, bytes(l % 251 + 1 for l in range(VL)), 2, "E", 40, T + G)
    recs.append((T + G, 138, 1, h1_perm(W(5), E(2), 3), 0))
    drain(recs, 44, 9, 5, "W", 4, T + G + PERM_X)
    return run(recs, sram, [(44, 9)])
_rev = [VL - 1 - l for l in range(VL)]
o, wd = h1_permute(_rev)
check("Permute by a full bijection", diags(o) == [] and wd.get((44, 9)) == bytes((VL - 1 - l) % 251 + 1 for l in range(VL)), diags(o))
_pm = [1023 if l % 14 == 0 else VL if l % 7 == 0 else VL - 1 - l for l in range(VL)]
o, wd = h1_permute(_pm)
check("Permute map entries >= vlanes zero-fill and may repeat",
      diags(o) == [] and wd.get((44, 9)) == bytes(0 if l % 7 == 0 else (VL - 1 - l) % 251 + 1 for l in range(VL)), diags(o))
_pm = [VL if l % 7 == 0 else VL - 1 - l for l in range(VL)]; _pm[1] = _pm[2]
o, wd = h1_permute(_pm)
check("a repeated unmasked Permute entry is still SXM-001", diags(o)[:1] == [f"T diag {80 + max(10, F2_SL)} queue 138 OTS-E-SXM-001"], diags(o))

# =======================================================================================
# End-to-end checks aimed at mutants of the model that the other tests do not kill
# (validation.md sec. 3). Each expectation is the rule of microarchitecture.md sec. 3 / sec. 8 (dispatch,
# AGT), sec. 5 (SXM OR merge, chaining), sec. 9 (Ifetch, IQ); the independent oracle agrees
# with every one. Being in this file, each program is also a golden vector.
def h6_srf(o, tag): return [int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf") and l.split()[3:6] == tag.split()]
FEED_W16 = mem(0, 0, W(16))                  # slice 1 reads SRAM[1][0] onto W:16 (a busy neighbour)
# Slice 1 (W gap 24) feeds queue 43's writes (W gap 34): a Read at c arrives at c + d_func + 10. fw(a, b) feeds
# arrivals a .. b - 1; the scenarios below were written with 15 cycles of lead, so a profile whose MEM d_func is
# longer shifts the whole scenario by SH (sh() moves the records, SS() the expected commit cycles).
SH = max(0, DM + 10 - 15)
def fw(a, b): return [(c - DM - 10, 1, 1, FEED_W16, 0) for c in range(a, b)]
def sh(recs): return [(c + SH, *r) for c, *r in recs]
def SS(*cs): return S(*[c + SH for c in cs])
H6Z = {(1, 0): bytes([9] * VL), (43, 0): bytes([5] * VL)}
# F4-AGT Q12: two references on one path (Read t0 + Read t1) open no loop, so the
# Iterations record of the next cycle has no loop to configure: OPR-001, both reads run once.
o, _ = run([(10, 0, 1, mem(0, 1, E(0)), 0), (10, 0, 1, mem(0, 2, E(1)) | T1, 0), (11, 0, 1, agt(7, 3), 0)], {(0, 1): bytes([1] * VL), (0, 2): bytes([1] * VL)})
check("two refs on one path open no loop: the Iterations record is OPR-001", diags(o) == at(11, 0, "OPR-001") and srfs(o) == S(10, 10), (diags(o), srfs(o)))
# Q14: a running W loop holds the whole queue: a whole-queue instruction (SetSegment) during it waits.
o, _ = run(rep_feed(37 - DM) + [(40, 0, 1, mem(1, 0x10, E(2)), 0), (41, 0, 1, agt(5, 12), 0), (42, 0, 1, agt(7, 3), 0), (45, 0, 1, agt(8, 0x10), 0)], {(8, 1): bytes(VL), (9, 1): bytes(VL)})
check("a whole-queue instruction while the queue's W loop runs waits (no diagnostic)", [l for l in diags(o) if "queue 0 " in l] == [], diags(o))
# Q17: queue 43 runs an R loop (refs 10, 14, ..., Countdown 3, 6 iterations). A new Write at
# 15 in the loop's bank opens a W loop; at 22 a second new Write is rejected (QUE-001: the W
# path is busy) and waits for it; it must not block the R loop's references.
recs = fw(15, 60)
recs += [(10, 43, 1, mem(0, 0, E(0)), 0), (11, 43, 1, agt(5, 3), 0), (12, 43, 1, agt(7, 6), 0),
         (15, 43, 1, mem(1, BANK1 | 0xff, W(16)), 0), (16, 43, 1, agt(5, 5), 0), (17, 43, 1, agt(7, 3), 0), (22, 43, 1, mem(1, 9, W(16)), 0)]
o, _ = run(sh(recs), H6Z)
check("a new Write on the busy W path waits (W loop 15, 21, 27, tail 5: it writes at 33) and does not block the R loop (refs 15..35 every 4)",
      diags(o) == [] and h6_srf(o, "E:0 gap 8") == SS(10, 14, 18, 22, 26, 30) and (33 + SH, 9) in sram_writes(o, 43), (diags(o), h6_srf(o, "E:0 gap 8"), sram_writes(o, 43)))
# Q20: an R loop and a W loop of one queue in one bank clash at the cycles both are due
# (18, 22) with no new record in the cycle: MEM-001 on both, twice.
recs = fw(16, 30)
recs += [(10, 43, 1, mem(0, 0, E(0)), 0), (11, 43, 1, agt(5, 3), 0), (12, 43, 1, agt(7, 6), 0),
         (16, 43, 1, mem(1, 0xff & (BANK1 - 1), W(16)), 0), (17, 43, 1, agt(5, 1), 0), (18, 43, 1, agt(7, 4), 0)]
o, _ = run(sh(recs), H6Z, [(43, 0xff & (BANK1 - 1))])
check("two loop references in one bank clash without new records: MEM-001 x2 at 18 and 22",
      diags(o) == at(18 + SH, 43, "MEM-001", 2) + at(22 + SH, 43, "MEM-001", 2), diags(o))
# Q21: a loop reference blocked by the bank rule still holds its path: a whole-queue instruction waits.
recs = fw(33, 50)
recs += [(30, 43, 1, mem(0, 0, E(0)), 0), (31, 43, 1, agt(5, 3), 0), (32, 43, 1, agt(7, 3), 0),
         (33, 43, 1, mem(1, 0x40, W(16)), 0), (34, 43, 1, agt(5, 4), 0), (35, 43, 1, agt(7, 2), 0), (38, 43, 1, agt(8, 0x10), 0)]
recs = sh(recs)
o, _ = run(recs, H6Z)
check("a blocked loop reference holds its path: MEM-001 x2 at 38, SetSegment waits (no diagnostic)",
      diags(o) == at(38 + SH, 43, "MEM-001", 2), diags(o))
# The trailing waits hold the queue by themselves: a wrong-unit word due in that cycle waits for both
# paths (R loop 30, 34, 38 + tail 3, W loop 33, 38 + tail 4: free at 43) and is then QUE-002.
o, _ = run(recs[:-1] + [(38 + SH, 43, 1, MXMW, 0)], H6Z)
check("a wrong-unit word due in a blocked reference's cycle waits for the paths (43), then QUE-002",
      diags(o) == at(38 + SH, 43, "MEM-001", 2) + at(43 + SH, 43, "QUE-002"), diags(o))
# Q22: Ifetch (from streams) on a MEM queue issues while the queue's R loop runs.
o, _ = run([(15 - DM, 8, 1, mem(0, 2, E(0)), 0), (14 - DM, 8, 1, mem(0, 1, E(0)), 0), (10, 4, 1, mem(0, 0, E(5)), 0),
            (11, 4, 1, agt(5, 3), 0), (12, 4, 1, agt(7, 3), 0), (15, 4, 1, IFETCH(E(0)), 0)], {(8, 1): bytes([0xFF] * VL), (8, 2): bytes([0xFF] * VL)})
check("Ifetch on a MEM queue issues while its R loop runs", diags(o) == [] and h6_srf(o, "E:5 gap 17") == S(10, 14, 18), (diags(o), h6_srf(o, "E:5 gap 17")))
# Q23: the configuration-record count restarts with each new loop: a loop opened at 40 takes
# its Iterations 3 as its first record (refs 40, 41, 42; commits 45..47).
recs = [(10, 43, 1, mem(0, 0, E(0)), 0), (11, 43, 1, agt(5, 1), 0), (12, 43, 1, agt(6, 1), 0), (13, 43, 1, agt(7, 2), 0)]
recs += fw(40, 44)
recs += [(40, 43, 1, mem(0, 0, E(0)), 0), (40, 43, 1, mem(1, BANK1 | 0x40, W(16)), 0), (41, 43, 1, agt(7, 3), 0)]
o, _ = run(sh(recs), H6Z)
check("a new loop counts its configuration records from zero (refs 40..42)", diags(o) == [] and h6_srf(o, "E:0 gap 8") == SS(10, 12, 40, 41, 42), (diags(o), h6_srf(o, "E:0 gap 8")))
# Only a VXM operand with its chain bit set reads a value
# written at its own gap in the previous cycle. A MEM Write on west slice 1 (reads E at 18)
# one cycle after west slice 0 committed E:0 at gap 18 finds nothing upstream: DIR-001, no write.
o, wd = run([(10, 0, 1, mem(0, 1, E(0)), 0), (10 + DM + 1, 1, 1, mem(1, 9, E(0)), 0)], {(0, 1): bytes([7] * VL)}, [(1, 9)])
check("a MEM unit never takes a chained read: the writer's own gap one cycle later is DIR-001",
      diags(o) == at(10 + DM + 1, 1, "DIR-001") and wd.get((1, 9)) == bytes(VL), (diags(o), wd.get((1, 9), b"")[:4].hex()))
# F4-Q43 / F4-V2ONLY Q45: the OR merge takes exactly one SEL and one TSP (HC34 p.31). SEL +
# TSP + TSP to one stream, and TSP + TSP, are SRF-002 on every writer (16 per Transpose).
TSP_W0 = (4 << 60) | (4 << 54) | (E(0) << 46) | (W(0) << 40)
for name, extra, exp_q in (("SEL + TSP + TSP", [True], [120] * 16 + [121] * 16 + [125]),
                           ("TSP + TSP", [], [120] * 16 + [121] * 16)):
    sram, recs, ins = g8_inputs(80)
    if extra:
        g8_sel(recs, sram, ins, 80)
    recs += [(80, 120, 1, TSP_W0, 0), (80, 121, 1, TSP_W0, 0)]
    drain(recs, 0, 9, 5, "E", 4, 80 + DTR)
    o, wd = run(recs, sram, [(0, 9)])
    check(f"{name} to one stream in one cycle: SRF-002 on every writer, no merge",
          diags(o) == [f"T diag {80 + DTR} queue {q} OTS-E-SRF-002" for q in exp_q], (len(diags(o)), diags(o)[-2:]))
# F4-Q50: IfetchM issues on every MEM queue, east included (44..87).
for q in (44, 50, 87):
    o, _ = run([(10, q, 1, IFETCHM(5), 0)], padw(q, 5, 6))
    check(f"IfetchM on MEM queue {q} issues", diags(o) == [], diags(o))
# F4-V2ONLY Q55: Ifetch reads one stream, two vectors (2 x vlanes bytes: this cycle and the next); with only one vector present it is TIM-001.
o, _ = run([(29 - DM, 8, 1, mem(0, 1, E(0)), 0), (30, 4, 1, IFETCH(E(0)), 0)], {(8, 1): bytes([7] * VL)})
check("Ifetch missing its second beat reports TIM-001 at the second edge", diags(o) == at(31, 4, "TIM-001"), diags(o))
# F4-Q78: a 64-bit MEM word on the control channel decodes like any other text; placement rejects it.
o, _ = run([(10, 0xFFFF, 1, mem(0, 0, E(0)), 0)])
check("a 64-bit MEM word on the control channel is QUE-002", diags(o) == at(10, 65535, "QUE-002"), diags(o))
# F4-V2ONLY (C2C source constants in a golden vector): the HAC epoch is 252 cycles (ISCA22
# footnote 4), a link carries a vector in the nominal 216 cycles (ISCA22 Table 2), and link
# 15 exists (16 links, ISCA20 sec. II). Literal source values, not the profile's.
HAC, LNOM = 252, 216
o, _ = run([(HAC + 10, 88, 1, (5 << 28) | (2 << 22), 0), (2 * HAC - 1, 88, 1, vxm_op(15, "i8", 8, 0), 0)])
check("Deskew at a non-zero epoch holds its queue to 2 x 252: the VXM op due one cycle earlier waits and executes at 2 x 252", diags(o) == at(2 * HAC + VXM_DS, 88, "TIM-001"), diags(o))
o, _ = run([(10, 140, 1, (5 << 28) | (3 << 22) | (15 << 6), 0), (10 + LNOM - 1, 128, 1, (5 << 28) | (1 << 22) | (15 << 6), 0), (10 + LNOM, 128, 1, (5 << 28) | (1 << 22) | (15 << 6), 0)])
check("link 15 loops back in 216 cycles", diags(o) == at(9 + LNOM, 128, "TIM-001"), diags(o))

# =======================================================================================
# Per-tile partial effects and Permute map readiness.
# Gather and Scatter: each active tile acts alone; a tile whose effective address is outside
# the segment or the slice drives zero (Gather) or writes nothing (Scatter), the good tiles
# act, and the instruction reports one OPR-001. A Permute map is ready `superlanes` cycles
# after the execute cycle of its LoadMap or LoadMapWide: earlier is RES-001, a map that is
# not bijective is SXM-001; both leave the Permute without effect. Distribute is unchanged.
SETSEG = lambda n: (1 << 60) | (8 << 54) | (n << 16)
# Gather: tile g takes SRAM[0][40 + g]; SetSegment 50 makes tiles 10..19 bad.
m = bytearray(VL)
for g in range(SL): m[16 * g] = 40 + g
sram = {(8, 1): bytes(m)}
for g in range(SL): sram[(0, 40 + g)] = bytes((g * 16 + l) & 0xff for l in range(VL))
T = 30
recs = [(5, 0, 1, SETSEG(50), 0), (T - DM - 2, 8, 1, mem(0, 1, E(1)), 0), (T, 0, 1, gs(3, E(2), E(1), k=5), 0), (T + DM + 6, 44, 1, mem(1, 7, E(2)), 0)]
o, wd = run(recs, sram, [(44, 7)])
w = wd.get((44, 7), b"")
exp = bytearray(VL)
for g in range(min(10, SL)): exp[16 * g + 5] = sram[(0, 40 + g)][16 * g + 5]
check("Gather: tiles with an address outside the segment drive zero, the good tiles' lanes are written, one OPR-001",
      w == bytes(exp) and diags(o) == [f"T diag {T} queue 0 OTS-E-OPR-001"], (w[:40].hex(), diags(o)), needs=SL < 11 and "needs tiles 10.. to be outside SetSegment 50: at least 11 superlanes")
# Scatter: tile g writes to SRAM[0][60 + g]; SetSegment 62 leaves tiles 0 and 1 good.
m = bytearray(VL); d = bytes(range(1, 256)) + bytes(65)
for g in range(SL): m[16 * g] = 60 + g
sram = {(0, 60 + g): bytes([0xEE] * VL) for g in range(SL)}; recs = [(5, 0, 1, SETSEG(62), 0)]; T = 50
feed(recs, sram, 4, 1, d, 1, "E", 18, T)
feed(recs, sram, 8, 1, bytes(m), 2, "E", 18, T)
recs.append((T, 0, 1, gs(4, E(1), E(2), k=2), 0))
o, wd = run(recs, sram, [(0, 60 + g) for g in range(SL)])
okf2 = all(wd.get((0, 60 + g), b"") == bytes(d[l] if (l == 16 * g + 2 and g < 2) else 0xEE for l in range(VL)) for g in range(SL))
check("Scatter: the good tiles write, the tiles with an address outside the segment do not, one OPR-001",
      okf2 and diags(o) == [f"T diag {T} queue 0 OTS-E-OPR-001"], diags(o), needs=SL < 3 and "needs tiles 2.. to be outside SetSegment 62: at least 3 superlanes")

# Permute map readiness. loads: [(cycle, entries)], each a LoadMapWide of map 3 on the east
# permuter (queue 138); the Permute is dispatched at pcycle. The result goes to SRAM[44][9].
def f2_perm(loads, pcycle):
    sram = {}; recs = []
    for i, (c, pm) in enumerate(loads):
        feed(recs, sram, 4 + 4 * i, 1, bytes(e & 255 for e in pm), 2 * i, "E", 40, c)
        feed(recs, sram, 5 + 4 * i, 1, bytes(e >> 8 for e in pm), 2 * i + 1, "E", 40, c)
        recs.append((c, 138, 1, h1_loadwide(3, E(2 * i), E(2 * i + 1)), 0))
    feed(recs, sram, 30, 2, bytes(l % 251 + 1 for l in range(VL)), 20, "E", 40, pcycle)
    recs.append((pcycle, 138, 1, h1_perm(W(5), E(20), 3), 0))
    drain(recs, 44, 9, 5, "W", 4, pcycle + PERM_X)
    return run(recs, sram, [(44, 9)])
F2_REV = [VL - 1 - l for l in range(VL)]
F2_BAD = [l for l in range(VL)]; F2_BAD[7] = F2_BAD[8]
F2_OUT = bytes((VL - 1 - l) % 251 + 1 for l in range(VL))
L0 = 80
if F2_SL > 1:
    o, wd = f2_perm([(L0, F2_REV)], L0 + F2_SL - 1)
    check("a Permute one cycle before its map is ready is RES-001 and has no effect",
          diags(o)[:1] == [f"T diag {L0 + F2_SL - 1} queue 138 OTS-E-RES-001"] and wd.get((44, 9)) != F2_OUT, diags(o))
o, wd = f2_perm([(L0, F2_REV)], L0 + F2_SL)
check("a Permute `superlanes` cycles after its LoadMapWide uses the map", diags(o) == [] and wd.get((44, 9)) == F2_OUT, diags(o))
o, wd = f2_perm([(L0, F2_BAD)], L0 + F2_SL)
check("a ready map that is not bijective: SXM-001 and no effect",
      diags(o)[:1] == [f"T diag {L0 + F2_SL} queue 138 OTS-E-SXM-001"] and wd.get((44, 9)) != bytes(l % 251 + 1 for l in range(VL)), diags(o))
# the clock restarts at every LoadMap, and the flag is that of the latest map
o, wd = f2_perm([(L0, F2_BAD), (L0 + 3, F2_REV)], L0 + 3 + F2_SL)
check("a bijective map loaded over a non-bijective one is usable `superlanes` cycles after the second load", diags(o) == [] and wd.get((44, 9)) == F2_OUT, diags(o))
if F2_SL > 3:
    o, wd = f2_perm([(L0, F2_REV), (L0 + 3, F2_REV)], L0 + 2 + F2_SL)
    check("the second LoadMapWide restarts the clock: a Permute `superlanes` after the first is RES-001",
          diags(o)[:1] == [f"T diag {L0 + 2 + F2_SL} queue 138 OTS-E-RES-001"], diags(o))
    o, wd = f2_perm([(L0, F2_BAD), (L0 + 3, F2_REV)], L0 + 3 + F2_SL - 1)
    check("a map loaded again is not ready before `superlanes` cycles, whatever the older map was",
          diags(o)[:1] == [f"T diag {L0 + 3 + F2_SL - 1} queue 138 OTS-E-RES-001"], diags(o))
o, wd = f2_perm([], 90)
check("a reset map (identity) is bijective and ready at once", diags(o) == [] and wd.get((44, 9)) == bytes(l % 251 + 1 for l in range(VL)), diags(o))
# Distribute: unchanged, a map is usable the cycle after its LoadMap. East distributor (queue 130).
sram = {}; recs = []
feed(recs, sram, 8, 1, bytes(15 - l % 16 for l in range(VL)), 0, "E", 39, L0)
feed(recs, sram, 12, 1, bytes(l % 251 + 1 for l in range(VL)), 2, "E", 39, L0 + 1)
recs.append((L0, 130, 1, (4 << 60) | (5 << 54) | (E(0) << 40) | (3 << 30), 0))
recs.append((L0 + 1, 130, 1, (4 << 60) | (2 << 54) | (E(5) << 46) | (E(2) << 40) | (3 << 30), 0))
o, _ = run(recs, sram)
check("Distribute runs the cycle after its LoadMap (no RES-001)", diags(o) == [] and srfs(o) != [], diags(o))
# : the boundary of that rule, pinned. A Distribute on another queue in the execute cycle of a LoadMap of the same
# map sees the old map (the load takes effect at the end of its cycle, like Permute's, so the result does not depend on the
# order of the queues) and raises nothing; from the next cycle it sees the new one. There is no window in between: nothing
# can be "early" for a superlane-local unit, so no RES-001 exists for Distribute. East permuter (138) loads, distributor (130) uses.
def h1_dist(d, s, m): return (4 << 60) | (2 << 54) | (d << 46) | (s << 40) | (m << 30)
def h1_load(m, s): return (4 << 60) | (5 << 54) | (s << 40) | (m << 30)
D8_REV = [15 - l % 16 for l in range(VL)]
D8_IN = bytes(l % 251 + 1 for l in range(VL))
D8_NEW = bytes(D8_IN[(l // 16) * 16 + D8_REV[l]] for l in range(VL)).hex()
D8_OLD = bytes(D8_IN[l] if l < 16 else 0 for l in range(VL)).hex()      # the reset map is the identity: entries >= 16 zero-fill
def distribute_weight_rows(load_cycle, dist_cycle):
    sram = {}; recs = []
    feed(recs, sram, 4, 1, bytes(D8_REV), 0, "E", 40, load_cycle)
    recs.append((load_cycle, 138, 1, h1_load(3, E(0)), 0))
    feed(recs, sram, 6, 2, D8_IN, 2, "E", 39, dist_cycle)
    recs.append((dist_cycle, 130, 1, h1_dist(E(5), E(2), 3), 0))
    o, _ = run(recs, sram)
    vals = [l.split()[-1] for l in o.splitlines() if l.startswith("T srf") and l.split()[3] == "E:5"]
    return o, vals
o, vals = distribute_weight_rows(L0, L0)
check("a Distribute in the execute cycle of a LoadMap of its map uses the old map and raises nothing (as a Permute does)",
      vals == [D8_OLD] and not [d for d in diags(o) if "TIM" not in d], (vals[:1], diags(o)))
o, vals = distribute_weight_rows(L0, L0 + 1)
check("a Distribute one cycle after the LoadMap uses the new map and raises nothing",
      vals == [D8_NEW] and not [d for d in diags(o) if "TIM" not in d], (vals[:1], diags(o)))

# ---------------------------------------------------------------------------------------
# Two wrong-result mutants of the Sail model that the other layers do not kill: the stream ring and the f16 dot product
# of negative zeros. Directed tests only.
def cr2_acc_lanes(wd):
    return [int.from_bytes(bytes(wd.get((4 * b, 9), bytes(VL))[l] for b in range(4)), "little") for l in range(16)]

# Stream ring: a write at the same gap and stream whose diagonal differs by 32 (a commit 32 cycles later) must not
# displace a value still in flight (a ring of 32 slots aliases them; the model keeps 64, proof obligation P-SRF-3).
# East slice 85 (quad 10) writes W:24 at gap 8; the ABC reads it at gap 43, 35 cycles later; a second Read of slice 85
# commits `delta` cycles after the first, before the ABC's read. delta 32 aliases, 33 does not (control).
for delta, ring_name in ((33, "control: 33 cycles apart"), (32, "32 cycles apart")):
    sram = {}; recs = []; T = 200; T2 = T + 2
    iws_rows(recs, sram, T, 3); recs.append((T, 104, 1, mxm(1, 2, 32, 1), 0))
    v = bytearray(VL); v[0] = 2
    sram[(85, 5)] = bytes(v); recs.append((T2 - 35 - DM, 85, 1, mem(0, 5, 32 + 24), 0))
    sram[(85, 6)] = bytes([0x55] * VL); recs.append((T2 - 35 + delta - DM, 85, 1, mem(0, 6, 32 + 24), 0))
    recs.append((T2, 105, 1, mxm(2, 2, 32 | 24), 0))
    T3 = T2 + D_ABC
    recs.append((T3, 106, 1, mxm(3, 2, 8, 0, 1), 0))
    for b in range(4): drain(recs, 4 * b, 9, 8 + b, "E", 2, T3 + D_ACC)
    o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
    check(f"stream ring: two live diagonals, {ring_name}, the ABC still reads the first value (2 x (r+1))",
          cr2_acc_lanes(wd) == [2 * (r + 1) for r in range(16)] and diags(o) == [], (cr2_acc_lanes(wd), diags(o)[:3]))

# Sign of an exact zero in the f16 dot product: row 0 = {-0 in all 320 columns} x {+0} is -0 (0x80000000, every product
# is negative zero); one +0 product, or a cancellation 1 x 1 + 1 x -1, is +0 (0x00000000).
def cr2_f16_zero(w_hi, x_hi):
    sram = {}; recs = []; T = 200
    for st in range(16):
        feed(recs, sram, 44 + 4 * (st % 11) + (st // 11), 3 + st, w_hi if st == 1 else bytes(VL), st, "W", 43, T)
    recs.append((T, 104, 1, mxm(1, 1, 32, 1), 0))
    feed(recs, sram, 60, 5, bytes(VL), 24, "W", 43, T + 2)
    feed(recs, sram, 61, 5, x_hi, 25, "W", 43, T + 2)
    T3 = T + 2 + D_ABC
    recs += [(T + 2, 105, 1, mxm(2, 1, 32 | 24), 0), (T3, 106, 1, mxm(3, 3, 8, 0, 1), 0)]
    for b in range(4): drain(recs, 4 * b, 9, 8 + b, "E", 2, T3 + D_ACC)
    o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
    return cr2_acc_lanes(wd), diags(o)
l, d = cr2_f16_zero(bytes([0x80] * VL), bytes(VL))
check("f16 dot product of all negative zeros (-0 x +0 in 320 columns) is -0 = 0x80000000", l[0] == 0x80000000 and l[1:] == [0] * 15 and d == [], ([hex(x) for x in l[:2]], d))
l, d = cr2_f16_zero(bytes([0x80] + [0] * (VL - 1)), bytes(VL))
check("control: -0 x +0 in one column and +0 x +0 elsewhere is +0", l[0] == 0 and d == [], ([hex(x) for x in l[:2]], d))
l, d = cr2_f16_zero(bytes([0x3C, 0x3C] + [0] * 318), bytes([0x3C, 0xBC] + [0] * 318))
check("control: 1 x 1 + 1 x -1 cancels to +0 (0x00000000)", l[0] == 0 and d == [], ([hex(x) for x in l[:2]], d))

# ---------------------------------------------------------------------------------------
# : f16 and i8 TanH, Exp, RSqrt and Log against an independent reference. The reference below is written here with
# mpmath alone (it shares no code with the model or with test/oracle/numerics.py): the exact value at 60 and at 120
# decimal digits, rounded once to nearest even (f16: gradual underflow, overflow to infinity from 65520; i8: to the nearest
# integer, saturated), with the special values of microarchitecture.md section 5. The inputs are fixed: every i8 value, and for
# f16 the specials, the neighbours of every power of two, the inputs whose exact result lies closest to a rounding midpoint
# (found once by a scan of all 65536 patterns, relative distance 1e-8 to 1e-7), and a stride sample of the whole range. The
# f32 sweep stays a separate tool (tools/check_transcendental_f32.sh).
try:
    import mpmath as _mpm
except ImportError:
    _mpm = None
check("mpmath is available for the transcendental reference", _mpm is not None)
D3_OPS = {"tanh": 21, "exp": 22, "rsqrt": 23, "log": 24}
def transcendental_reference_value(op, x):
    return {"tanh": _mpm.tanh, "exp": _mpm.exp, "rsqrt": lambda z: 1 / _mpm.sqrt(z), "log": _mpm.log}[op](x)
def round_f16_bits(v):
    """Round the mpf v to f16 bits, to nearest even; None when v is within 1e-30 (relative) of a midpoint."""
    if v == 0: return 0
    neg = v < 0; a = abs(v)
    if a >= 65520: return 0xFC00 if neg else 0x7C00
    ex = _mpm.frexp(a)[1] - 1                     # floor(log2(a))
    e = max(ex, -14); q = _mpm.mpf(2) ** (e - 10)
    k = int(_mpm.floor(a / q)); rem = a - k * q; half = q / 2
    if abs(rem - half) < a * _mpm.mpf(10) ** -30: return None
    if rem > half: k += 1
    if k == 0: return 0x8000 if neg else 0
    if e == -14 and k < 1024: bits = k
    else:
        if k == 2048: k = 1024; e += 1
        bits = ((e + 15) << 10) | (k - 1024)
        if e + 15 >= 31: bits = 0x7C00
    return bits | (0x8000 if neg else 0)
def transcendental_f16_reference(op, b):
    neg = bool(b & 0x8000); e = (b >> 10) & 31; m = b & 1023; NAN = 0x7E00
    if e == 31:
        if m: return NAN
        return {"tanh": 0xBC00 if neg else 0x3C00, "exp": 0 if neg else 0x7C00,
                "rsqrt": NAN if neg else 0, "log": NAN if neg else 0x7C00}[op]
    if e == 0 and m == 0:
        return {"tanh": b, "exp": 0x3C00, "rsqrt": 0xFC00 if neg else 0x7C00, "log": 0xFC00}[op]
    out = []
    for dps in (60, 120):
        _mpm.mp.dps = dps
        x = (_mpm.mpf(m) * _mpm.mpf(2) ** -24) if e == 0 else (1 + _mpm.mpf(m) / 1024) * _mpm.mpf(2) ** (e - 15)
        x = -x if neg else x
        if op in ("rsqrt", "log") and x < 0: return NAN
        if op == "log" and x == 1: return 0
        out.append(round_f16_bits(transcendental_reference_value(op, x)))
    assert out[0] is not None and out[0] == out[1], (op, hex(b), out)
    return out[0]
def transcendental_i8_reference(op, x):
    v = x - 256 if x > 127 else x
    if v == 0: return {"exp": 1, "rsqrt": 127, "log": 0x80, "tanh": 0}[op]
    if op in ("rsqrt", "log") and v < 0: return 0
    out = []
    for dps in (60, 120):
        _mpm.mp.dps = dps
        r = transcendental_reference_value(op, _mpm.mpf(v))
        a = abs(r); k = int(_mpm.floor(a)); rem = a - k
        if abs(rem - _mpm.mpf(1) / 2) < _mpm.mpf(10) ** -30 and a != _mpm.mpf(1) / 2: out.append(None); continue
        if rem > _mpm.mpf(1) / 2 or (abs(rem - _mpm.mpf(1) / 2) < _mpm.mpf(10) ** -30 and k % 2 == 1): k += 1
        out.append(max(-128, min(127, -k if r < 0 else k)) & 0xFF)
    assert out[0] == out[1] and out[0] is not None, (op, v, out)
    return out[0]
D3_HARD = {"tanh": [0x2745, 0xA745, 0x3EC4, 0xBEC4, 0x3D61, 0x2D86], "exp": [0x1F79, 0x25CF, 0xA954, 0x8C00, 0xA57F, 0xA922, 0xC13B, 0x1AFD],
           "rsqrt": [0x0879, 0x1079, 0x2879, 0x4879], "log": [0x305F, 0x4131, 0x1D78, 0x1935, 0x6051, 0x396F, 0x2DBB, 0x423E]}
D3_SPECIAL = [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE00, 0xFFFF, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3C00, 0xBC00,
              0x4880, 0xC880, 0x5BE0, 0x5BE1, 0xCC00, 0xCC47, 0xCC48]
D3_POW2 = [(e << 10) + d for e in range(1, 31) for d in (-1, 0, 1) if 0 <= (e << 10) + d < 0x7C00] + [0x0002, 0x0004, 0x0100, 0x0200]
D3_POW2 += [0x8000 | x for x in D3_POW2]
D3_EDGES = [0x4C63, 0x4C64, 0x4C65, 0x4C66, 0xCC40, 0xCC41, 0xCC42, 0xCC43, 0xC9A0, 0xC9A1, 0xC9A2, 0x4740, 0x4741, 0x4742]   # near exp overflow and underflow, tanh saturation
if _mpm is not None:
    for opname, op in D3_OPS.items():
        inputs = sorted(set(D3_SPECIAL + D3_POW2 + D3_EDGES + D3_HARD[opname] + list(range(5, 65536, 61))))
        bad = []
        for i in range(0, len(inputs), VL):                  # one vector (vlanes lanes) per run
            chunk = inputs[i:i + VL]
            r, d = vx1(op, "f16", chunk)
            if d: bad.append(("diag", d[:2]))
            for l, b in enumerate(chunk):
                want = transcendental_f16_reference(opname, b)
                if r[l] != want: bad.append((hex(b), hex(r[l]), hex(want)))
        check(f": f16 {opname} equals the mpmath reference on {len(inputs)} fixed inputs (specials, power-of-two neighbours, hardest-to-round, stride sample)", not bad, bad[:4])
        r, d = [], []
        for i in range(0, 256, VL):
            ri, di = vx1(op, "i8", list(range(i, min(i + VL, 256))))
            r += ri[:min(VL, 256 - i)]; d += di
        bad = [(x, r[x], transcendental_i8_reference(opname, x)) for x in range(256) if r[x] != transcendental_i8_reference(opname, x)]
        check(f": i8 {opname} equals the mpmath reference on all 256 inputs", not bad and not d, (bad[:4], d))

# ---------------------------------------------------------------------------------------
# : what a diagnosed instruction does. The rule (microarchitecture.md section 8, "Effect of a diagnosed instruction"):
# a diagnosed instruction has no effect, except Gather, Scatter and sxselect, which act per tile or per lane and report one
# OPR-001 (the stagger makes an all-good check impossible before the first tile acts). Part 1 sweeps instructions of every
# unit that write a stream or SRAM and checks that the diagnosed one wrote nothing; the three exceptions are pinned above
# ("Gather ... one OPR-001", "Scatter ... one OPR-001", "sxselect: lanes with sel > 2 ..."). Part 2 checks that the Sail
# markers `DIAG-EFFECT`, the header of core/diagnostics.sail and the table of section 8 name the same three instructions.
def stream_trace_lines(o, rowstream):
    return [l for l in o.splitlines() if l.startswith("T srf") and l.split()[3] == rowstream]
X9 = bytes((l * 5) % 251 + 1 for l in range(VL))
DIAGNOSTIC_CYCLE = 80
# a Permute that is early (RES-001) and one with a map that is not bijective (SXM-001): nothing is written to W:5
def run_permute_case(loads, pcycle):
    sram = {}; recs = []
    for i, (c, pm) in enumerate(loads):
        feed(recs, sram, 4 + 4 * i, 1, bytes(e & 255 for e in pm), 2 * i, "E", 40, c)
        feed(recs, sram, 5 + 4 * i, 1, bytes(e >> 8 for e in pm), 2 * i + 1, "E", 40, c)
        recs.append((c, 138, 1, h1_loadwide(3, E(2 * i), E(2 * i + 1)), 0))
    feed(recs, sram, 30, 2, X9, 20, "E", 40, pcycle)
    recs.append((pcycle, 138, 1, h1_perm(W(5), E(20), 3), 0))
    return run(recs, sram)[0]
if F2_SL > 1:
    o = run_permute_case([(L0, F2_REV)], L0 + F2_SL - 1)
    check("a Permute diagnosed RES-001 (early) writes nothing", "OTS-E-RES-001" in " ".join(diags(o)) and not stream_trace_lines(o, "W:5"), diags(o))
o = run_permute_case([(L0, F2_BAD)], L0 + F2_SL)
check("a Permute diagnosed SXM-001 (map not bijective) writes nothing", "OTS-E-SXM-001" in " ".join(diags(o)) and not stream_trace_lines(o, "W:5"), diags(o))
# Rotate whose last output stream is beyond stream 31 (OPR-002): none of the nine outputs is written
sram = {}; recs = []
feed(recs, sram, 44, 1, X9, 0, "W", 40, DIAGNOSTIC_CYCLE)
recs.append((DIAGNOSTIC_CYCLE, 120, 1, (4 << 60) | (3 << 54) | (E(30) << 46) | (W(0) << 40) | (3 << 30), 0))
o, _ = run(recs, sram)
check("a Rotate diagnosed OPR-002 (destination past stream 31) writes none of its streams",
      diags(o) == [f"T diag {DIAGNOSTIC_CYCLE} queue 120 OTS-E-OPR-002"] and not any(stream_trace_lines(o, f"E:{k}") for k in range(30, 32)), diags(o))
# a VXM add whose operands are not on the stream registers (TIM-001) writes nothing
o, _ = run([(DIAGNOSTIC_CYCLE, 88, 1, vxm_op(0, "i8", 9, 0, 1), 0)])
check("a VXM add diagnosed TIM-001 writes nothing", [d.split()[-1] for d in diags(o) if "queue 88" in d] == ["OTS-E-TIM-001"] and not stream_trace_lines(o, "E:9"), diags(o))
# a MEM Write with no value at the reader's gap (TIM-001) leaves the word as it was
old = bytes([0x55] * VL)
o, wd = run([(DIAGNOSTIC_CYCLE, 0, 1, mem(1, 9, E(0)), 0)], {(0, 9): old}, [(0, 9)])
check("a MEM Write diagnosed TIM-001 leaves the word unchanged", diags(o) == [f"T diag {DIAGNOSTIC_CYCLE} queue 0 OTS-E-TIM-001"] and wd.get((0, 9)) == old, (diags(o), wd.get((0, 9), b"")[:4].hex()))
# Part 2: the three exceptions are named alike in the model, in its header and in the specification
import re as _effect_re
_arch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_marks = set()
for _dp, _dn, _fn in os.walk(os.path.join(_arch, "model", "slices")):
    for _f in _fn:
        if _f.endswith(".sail"):
            _marks |= set(_effect_re.findall(r"DIAG-EFFECT: (\w+) (?:per-tile|per-lane)", open(os.path.join(_dp, _f)).read()))
_hdr = _effect_re.findall(r"DIAG-EFFECT exceptions: ([\w, ]+)", open(os.path.join(_arch, "model", "core", "diagnostics.sail")).read())
_doc = open(os.path.join(_arch, "docs", "microarchitecture.md")).read()
_rows = set(_effect_re.findall(r"^\| `(\w+)` \| (?:per tile|per lane) \|", _doc, _effect_re.M))
_want = {"gather", "scatter", "sxselect"}
check("the Sail markers DIAG-EFFECT name Gather, Scatter and sxselect and nothing else", _marks == _want, sorted(_marks))
check("the header of core/diagnostics.sail lists the same exceptions", len(_hdr) == 1 and {x.strip() for x in _hdr[0].split(",")} == _want, _hdr)
check("the table 'Effect of a diagnosed instruction' in microarchitecture.md lists the same exceptions", _rows == _want, sorted(_rows))

print("directed:", ("all pass" if ok else "FAILURES") + (f" ({len(SKIPPED)} scenarios skipped: they are written for a larger geometry)" if SKIPPED else ""))
sys.exit(0 if ok else 1)
