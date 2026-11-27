"""Flow: scheduling of VXM operations whose operands are vectors resident in MEM or values on streams. A `Vec` is
a vector in the SRAM of one quad (plane k on the quad's k-th slice, one word address); `Flow.vxm` reads every `Vec`
operand so that it meets the operation (the sec. 1 equation), picks the first free dispatch cycle when no stream
operand fixes it, and `Flow.store` writes a result back to a quad. Rows follow the hemispheres: a west quad reaches
the VXM on row E and a VXM result bound for the west is on row W (and the east mirrored)."""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from opentsp_tools.programs import data
from opentsp_tools import stations
from opentsp_tools.stations import ScheduleError

LAST_GAP = 64                       # beyond every station: a reader at no particular station
READ_ROW = {"W": "E", "E": "W"}      # the row on which a hemisphere's MEM reaches the VXM


@dataclass
class Vec:
    slices: List[int]
    addr: int
    ready: int                        # the first cycle at which a Read sees the vector
    hemi: str
    names: Optional[List[str]] = None  # extern names of the planes, when bound from a weight package
    patch: Optional[list] = None      # patch sites of the planes' read address, [(name, lo, hi[, stride, span])] per slice

    @property
    def width(self):
        return len(self.slices)


class Flow:
    def __init__(self, k):
        self.k, self.b, self.p = k, k.b, k.p
        self.avoid = False        # see stream_for
        self.streams = {"E": 0, "W": 0}
        self.next_addr = {}
        self.base = 0x80          # first word `addr` hands out (a program with weights below it moves it up)
        self.low = {"E": 0, "W": 0}     # streams below these are left to installs that run beside other stages

    def fold(self, quad):
        """The quad a vector of the caller's quad number is kept on: the caller counts the 11 quads of `ref`, a profile with fewer
        has the vectors of the surplus quads share the quads it has (other words, same slices)."""
        return quad % max(1, self.p.slices_per_hemisphere // 4)

    def addr(self, hemi, quad, bank):
        """The next free word of a quad in the given bank (words below `base`, 0x80 by default, are left to the kernels'
        fixed layouts and the weights)."""
        key = (hemi, self.fold(quad), bank)
        a = self.next_addr.get(key, self.base)
        self.next_addr[key] = a + 1
        return a + (bank << self.p.bank_bit)

    def stream(self, row, width):
        """The first of `width` aligned streams of `row`, taken in rotation: a value shares its streams only with the
        values that are `streams / width` allocations older."""
        lo = self.low[row]
        base = max(lo, -(-self.streams[row] // width) * width)
        if base + width > self.p.streams:
            base = lo
        self.streams[row] = base + width
        return base

    def stream_for(self, row, width, g_w, c_w):
        """`stream(row, width)` that skips the groups on which a value committed at gap `g_w` in cycle `c_w` would meet another
        writer downstream of it on its diagonal, or take over a read from its producer (`Tracker.stream_free`, with the reader
        at the last gap): a value placed in the window of another that was emitted earlier but runs at the same time (a matrix
        installed ahead of its use). With no such neighbour the first group of the rotation is taken, as `stream` does; with `avoid` False (the default: only a
        caller that places work ahead of the cursor, emit_block with EARLY_INSTALL, sets it) the check is skipped."""
        if not self.avoid:
            return self.stream(row, width)
        tr = self.b.t
        for _ in range(self.p.streams // width + 1):
            s = self.stream(row, width)
            if all(tr.stream_free(row, s + k, g_w, c_w, LAST_GAP) for k in range(width)):
                return s
        raise ScheduleError(f"no free stream group of {width} on row {row} for a value committed at gap {g_w} in cycle {c_w}")

    def const(self, name, arr, hemi, quad):
        """An extern vector (bank 0) bound from the case's weight package."""
        a = self.addr(hemi, quad, 0)
        slices, names = self.k.place(name, arr, self.fold(quad), a, hemi)
        return Vec(slices, a, 0, hemi, names)

    def table(self, name, arrs, hemi, quad):
        """Vectors `arrs` (all of one type) as consecutive words of the quad's slices, one extern per plane
        (`<name>p<k>`, one word per vector) bound from the case's weight package; returns one Vec per vector."""
        planes = [data.to_planes(a) for a in arrs]
        base = self.addr(hemi, quad, 0)
        self.next_addr[(hemi, self.fold(quad), 0)] = base + len(arrs)
        first = (0 if hemi == "W" else 44) + 4 * self.fold(quad)
        slices = list(range(first, first + len(planes[0])))
        for k, s in enumerate(slices):
            self.b.prog.extern(f"{name}p{k}", s, base, len(arrs))
            self.k.weights[f"{name}p{k}"] = ("raw", b"".join(pl[k] for pl in planes))
        return [Vec(slices, base + i, 0, hemi, [f"{name}p{k}+{i:#x}" for k in range(len(slices))])
                for i in range(len(arrs))]

    def splat(self, name, value, hemi, quad):
        """An extern f32 vector of one value in every lane."""
        return self.const(name, np.full(self.p.vlanes, value, np.float32), hemi, quad)

    def bound(self, v, mnemonic, queue):
        """The first dispatch cycle of `mnemonic` on `queue` that can read vec v."""
        g_w = stations.need_gap(self.p, v.slices[0], READ_ROW[v.hemi], "write")
        g_r = stations.need_gap(self.p, queue, READ_ROW[v.hemi], "read")
        if g_r <= g_w:
            raise ScheduleError(f"queue {queue} is not downstream of slice {v.slices[0]}")
        d_func, _ = self.p.timing("read")
        _, d_skew = self.p.timing(mnemonic)
        return max(v.ready, 0) + (g_r - g_w) + d_func - d_skew

    def transit(self, queue, row, alu):
        """Cycles from a VXM result on `row` (committed by VXM[alu]) to its read at `queue`'s station."""
        return stations.need_gap(self.p, queue, row, "read") - stations.need_gap(self.p, 88 + alu, row, "write")

    def read(self, v, t, mnemonic, queue, row=None):
        """Read vec v onto a free stream group so that it meets `mnemonic` dispatched at t on `queue`; `row`
        defaults to the row on which the vec's hemisphere reaches the VXM."""
        row = row or READ_ROW[v.hemi]
        t_rd = self.b.read_cycle_for(t, mnemonic, queue, v.slices[0], row)
        g_w, c_w = stations.commit(self.p, t_rd, "read", v.slices[0], row)
        return self.b.read_for(t, mnemonic, queue, v.slices, [v.addr] * v.width, row, self.stream_for(row, v.width, g_w, c_w),
                               addr_text=v.names, patch=v.patch)

    def readable(self, v, t, mnemonic, queue, row=None):
        """Whether `read(v, t, mnemonic, queue)` finds the R slot of every slice of the vec free and no Write of the same
        bank in that cycle (OTS-E-MEM-001): the cycles at which a Write lands depend on the profile's latencies, so a caller
        that may start a read later asks before it reads."""
        row = row or READ_ROW[v.hemi]
        t_rd = self.b.read_cycle_for(t, mnemonic, queue, v.slices[0], row)
        tr = self.b.t
        return all(tr.free(s, t_rd, "R") and tr.bank_ok(s, t_rd, "R", v.addr) for s in v.slices)

    def vxm(self, alu, op, srcs, to="E", chain=(), width=None, tag=None, at=0, stream=None):
        """`op` on VXM[alu]. A `Vec` operand is read to meet the operation, a Value is met on its diagonal (a chained
        Value in the cycle after its commit). The dispatch cycle is fixed by the Values, else the first cycle from
        `at` at which the operands can be read and the queue is free. The result goes onto row `to` (the hemisphere
        it is bound for), on `stream` or a free stream group."""
        queue, mnem = 88 + alu, op          # mnem keeps the element type: a typed timing row applies
        _, d_skew = self.p.timing(mnem)
        forced, lower = set(), at
        for i, s in enumerate(srcs):
            if isinstance(s, Vec):
                lower = max(lower, self.bound(s, mnem, queue))
            elif i in chain:
                forced.add(s.cycle + 1 - d_skew)
            else:
                forced.add(self.b.when(s, mnem, queue))
        if len(forced) > 1:
            raise ScheduleError(f"{op}: stream operands meet the VXM at {sorted(forced)}")
        fixed = bool(forced)
        t = forced.pop() if fixed else lower
        if fixed and t < lower:
            raise ScheduleError(f"{op}: a vector operand is not ready before {lower}, the stream operands fix {t}")
        while not fixed and not self._free(t, queue, srcs, mnem):
            t += 1
        read = {}                                       # one Read per vector, even for an operand used twice
        for s in srcs:
            if isinstance(s, Vec) and id(s) not in read:
                read[id(s)] = self.read(s, t, mnem, queue)
        vals = [read[id(s)] if isinstance(s, Vec) else s for s in srcs]
        w = width if width is not None else vals[0].width
        if stream is None:
            g_w, c_w = stations.commit(self.p, t, mnem, queue, to)
            stream = self.stream_for(to, w, g_w, c_w)
        return self.b.vxm(alu, op, to, stream, vals, chain=chain, cycle=t, width=w, tag=tag)

    def _free(self, t, queue, srcs, mnem):
        tr = self.b.t
        return all(self.readable(s, t, mnem, queue) for s in srcs if isinstance(s, Vec)) and tr.free(queue, t)

    def store(self, v, quad, expect=None, bank=1, hemi=None):
        """Write value v to a quad of `hemi` (default: the hemisphere its row leads to, row E: east; an MXM result
        is upstream of both hemispheres' MEM and may go to either); returns the Vec. With `expect` (an array of the
        value's element type) the words are dumped and expected."""
        hemi = hemi or v.row
        a = self.addr(hemi, quad, bank)
        base = (0 if hemi == "W" else 44) + 4 * self.fold(quad)
        slices = list(range(base, base + v.width))
        t = self.b.write(v, slices, [a] * v.width)
        if expect is not None:
            for s, plane in zip(slices, data.to_planes(expect)):
                self.k.expect(s, a, plane)
        self.k.last = max(self.k.last, t)
        return Vec(slices, a, t + 1, hemi)
