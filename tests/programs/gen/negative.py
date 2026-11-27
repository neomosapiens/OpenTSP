"""Category G of the design's sec. 2.2: the runner's example of an expected-diagnostics case (the
timing_violation family). The three stream-timing errors, at separate
cycles so that each raises exactly one report group (microarchitecture.md sec. 2 and sec. 8)."""
import random

from opentsp_tools.programs.case import Kit
from basic import PROFILES, rand_word
from opentsp_tools import stations


def stream_timing(p, seed=1):
    """1. Early read (TIM-001): the VXM add is dispatched one cycle before its operands reach gap 21; it executes at
          dispatch + d_skew, finds the value still upstream on its diagonal, and has no effect.
       2. Upstream read (DIR-001): MEM[8] (west quad 2, row E gap 16) writes a stream whose only live value was
          committed at gap 17 by slice 4, one cycle earlier, so every live write is downstream of the reader.
       3. Collision (SRF-002): slices 4 and 5 (one quad, one gap) read onto E:24 in one cycle; each queue reports
          SRF-002 at the commit cycle, ascending queue order, and the lowest queue's value (x) is kept, which
          MEM[44] then writes to SRAM."""
    k = Kit(p, "negative/stream_timing", "TIM-001, DIR-001 and SRF-002", seed, module="negative")
    rng = random.Random(seed)
    x, y = rand_word(rng, p), rand_word(rng, p)
    k.extern_word("x", 4, 0x10, x)
    k.extern_word("y", 5, 0x20, y)
    b = k.b
    d_mem, _ = p.timing("read")
    _, skew_vxm = p.timing("add.i8")
    # 1. early read
    vx = b.read(0, [4], [0x10], "E", 0, addr_text=["x"])
    b.read(0, [5], [0x20], "E", 4, addr_text=["y"])
    t_ok = b.when(vx, "add.i8", 88)
    b.raw(t_ok - 1, 88, "add.i8 E:8, E:0, E:4")
    k.diags.append((t_ok - 1 + skew_vxm, 88, "OTS-E-TIM-001"))
    # 2. upstream read
    t2 = t_ok + 12
    v2 = b.read(t2, [4], [0x10], "E", 16, addr_text=["x"])
    if stations.need_gap(p, 8, "E", "read") >= v2.gap:
        raise stations.ScheduleError("slice 8 is not upstream of slice 4 on row E")
    b.raw(v2.cycle + 1, 8, "write 0x30, E:16, 0x0", slot="W")
    k.diags.append((v2.cycle + 1, 8, "OTS-E-DIR-001"))
    # 3. collision, then the kept value written downstream
    t3 = v2.cycle + 12
    b.raw(t3, 4, "read x, E:24, 0x0", slot="R")
    b.raw(t3, 5, "read y, E:24, 0x0", slot="R")
    g = stations.need_gap(p, 4, "E", "write")
    k.diags += [(t3 + d_mem, 4, "OTS-E-SRF-002"), (t3 + d_mem, 5, "OTS-E-SRF-002")]
    b.t.write("E", 24, g, t3 + d_mem, "kept")
    t_w = stations.meet(p, g, t3 + d_mem, "write", 44, "E")
    b.t.read("E", 24, stations.need_gap(p, 44, "E", "read"), t_w, "kept")
    b.raw(t_w, 44, "write 0x30, E:24, 0x0", slot="W")
    k.expect(44, 0x30, x)
    k.last = t_w
    return k.done("stream_timing")


CASES = [
    ("negative/stream_timing", stream_timing, PROFILES, {}),
]
