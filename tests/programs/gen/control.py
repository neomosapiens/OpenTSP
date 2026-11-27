"""Category F of the design's sec. 2.2: barriers and Repeat (the families
sync_notify, barrier_pipeline, barrier_pre_notify_park and repeat_nop families), with opentsp's semantics
(microarchitecture.md sec. 3): a parked parcel waits and dispatches at the first unparked cycle, the parcels behind
it keep their spacing, a Notify releases every Sync dispatched at or before it from Notify + barrier.release_latency,
and a Repeat with nothing to repeat is RES-001."""
import random

from opentsp_tools.programs.case import Kit
from basic import PROFILES, rand_word


def wrap_add(a, b):
    return bytes((u + v) & 0xFF for u, v in zip(a, b))


def park_waits(p, seed=1):
    """MEM[4] parks itself with a Sync at cycle 0; its two Reads are due at 2 and 3. The control channel notifies
    at cycle 4, so the queue is released at R = 4 + release_latency: the first Read dispatches at R (it waits;
    it is not skipped) and the second at R + 1 (it keeps its spacing). MEM[44] and MEM[45] write the two
    values at the cycles that follow from R, so the words are right only if both Reads ran late, in order."""
    k = Kit(p, "control/park_waits", "a parked Read waits for the release and keeps its order", seed,
            module="control", tags=("sim", "e2e"))
    rng = random.Random(seed)
    x0, x1 = rand_word(rng, p), rand_word(rng, p)
    k.extern_word("x0", 4, 0x10, x0)
    k.extern_word("x1", 4, 0x11, x1)
    b = k.b
    b.raw(0, 4, "sync", slot="insn")
    n = 4
    b.raw(n, 0xFFFF, "notify")
    r = n + p.release_latency
    v0 = b.read(r, [4], [0x10], "E", 0, addr_text=["x0"], text_cycle=2)
    v1 = b.read(r + 1, [4], [0x11], "E", 1, addr_text=["x1"], text_cycle=3)
    k.last = max(b.write(v0, [44], [0x30]), b.write(v1, [45], [0x30]))
    k.expect(44, 0x30, x0)
    k.expect(45, 0x30, x1)
    return k.done("park_waits")


def barrier_pipeline(p, seed=1):
    """Two phases joined by data-queue Syncs and one Notify. Phase 1: z = x + y (MEM[4], MEM[5] -> VXM[0] -> row W
    -> MEM[7] in west quad 1). MEM[6] parks at 0, MEM[7] parks right after its Write, and MEM[4] notifies one cycle
    later, releasing both at R = N + release_latency. Phase 2: MEM[6] and MEM[7] read w and z (their parcels are
    due right after the Notify, so both wait and dispatch at R), VXM[1] adds them and MEM[46] writes
    r = z + w = (x + y) + w. Zero diagnostics."""
    k = Kit(p, "control/barrier_pipeline", "two phases joined by Sync and Notify", seed, module="control",
            tags=("sim", "e2e"))
    rng = random.Random(seed)
    x, y, w = (rand_word(rng, p) for _ in range(3))
    k.extern_word("x", 4, 0x10, x)
    k.extern_word("y", 5, 0x10, y)
    k.extern_word("w", 6, 0x10, w)
    b = k.b
    b.raw(0, 6, "sync", slot="insn")
    vx = b.read(1, [4], [0x10], "E", 0, addr_text=["x"])
    vy = b.read(1, [5], [0x10], "E", 1, addr_text=["y"])
    vz = b.vxm(0, "add.i8", "W", 0, [vx, vy])
    t_wz = b.write(vz, [7], [0x20])
    b.raw(t_wz + 1, 7, "sync", slot="insn")
    n = t_wz + 1
    b.raw(n, 4, "notify", slot="insn")
    r = n + p.release_latency
    v_w = b.read(r, [6], [0x10], "E", 4, addr_text=["w"], text_cycle=n + 1)
    v_z = b.read(r, [7], [0x20], "E", 5, text_cycle=n + 2)
    v_r = b.vxm(1, "add.i8", "E", 8, [v_w, v_z])
    k.last = b.write(v_r, [46], [0x20])
    k.expect(7, 0x20, wrap_add(x, y))
    k.expect(46, 0x20, wrap_add(wrap_add(x, y), w))
    return k.done("barrier_pipeline")


def repeat(p, seed=1):
    """Repeat on MEM and on VXM (iteration k at p + k (d + 1); d = 0 is consecutive cycles, ISCA20 Table I): MEM[4] reads x once and repeats it three
    times (four equal vectors on E:0); MEM[5] reads y0..y3 explicitly; VXM[0] adds once and repeats three times,
    so z_k = x + y_k on four consecutive cycles; MEM[44] writes the four results to four words, and MEM[45] writes
    once and repeats, so its word holds the last, z_3. Negative: a Repeat that is the first instruction of VXM[2]
    (only NOP timing before it) and a Repeat right after a Notify on VXM[3] have nothing to repeat: RES-001 at
    their dispatch, and no effect."""
    k = Kit(p, "control/repeat", "Repeat on MEM and VXM; RES-001 after a NOP and after a Notify", seed,
            module="control", tags=("sim", "e2e"))
    rng = random.Random(seed)
    x = rand_word(rng, p)
    ys = [rand_word(rng, p) for _ in range(4)]
    k.extern_word("x", 4, 0x10, x)
    for i, yi in enumerate(ys):
        k.extern_word(f"y{i}", 5, 0x10 + i, yi)
    b = k.b
    t0 = 0
    vx = [b.read(t0 + i, [4], [0x10], "E", 0, addr_text=["x"], emit=(i == 0)) for i in range(4)]
    b.raw(t0 + 1, 4, "repeat 0x3, 0x0", slot=None)
    vy = [b.read(t0 + i, [5], [0x10 + i], "E", 4, addr_text=[f"y{i}"]) for i in range(4)]
    vz = [b.vxm(0, "add.i8", "E", 8, [vx[i], vy[i]], emit=(i == 0)) for i in range(4)]
    t_add = b.when(vx[0], "add.i8", 88)
    b.raw(t_add + 1, 88, "repeat 0x3, 0x0", slot=None)
    zs = [wrap_add(x, yi) for yi in ys]
    for i in range(4):
        k.last = max(k.last, b.write(vz[i], [44], [0x30 + i]))
        b.write(vz[i], [45], [0x30], emit=(i == 0))
        k.expect(44, 0x30 + i, zs[i])
    b.raw(b.when(vz[0], "write", 45) + 1, 45, "repeat 0x3, 0x0", slot=None)
    k.expect(45, 0x30, zs[3])
    tn = k.last + 2
    b.raw(tn, 90, "repeat 0x1, 0x1")
    b.raw(tn, 91, "notify")
    b.raw(tn + 1, 91, "repeat 0x1, 0x1")
    k.diags += [(tn, 90, "OTS-E-RES-001"), (tn + 1, 91, "OTS-E-RES-001")]
    k.last = tn + 1
    return k.done("repeat")


CASES = [
    ("control/park_waits", park_waits, PROFILES, {}),
    ("control/barrier_pipeline", barrier_pipeline, PROFILES, {}),
    ("control/repeat", repeat, PROFILES, {}),
]
