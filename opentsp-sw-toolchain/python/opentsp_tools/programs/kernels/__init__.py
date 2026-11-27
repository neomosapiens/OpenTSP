"""Reusable emitters . A Builder couples the program text with the tracker: every emitted
instruction claims its slot, every stream write and read is recorded, and `finish` checks that each read sees its
intended producer. Values are vectors of `width` byte planes on consecutive streams of one row, all planes at one
gap and commit cycle (topology.md sec. 3: lane = element, a w-byte element uses w aligned streams)."""
import itertools
from dataclasses import dataclass

from opentsp_tools import emit
from opentsp_tools import patch as tpatch
from opentsp_tools import sched
from opentsp_tools import stations
from opentsp_tools.stations import ScheduleError
from opentsp_tools.queues import agt_refs, qname

_ids = itertools.count()


@dataclass
class Value:
    row: str
    stream: int          # the first stream; planes are stream .. stream + width - 1
    width: int
    gap: int             # writer gap
    cycle: int           # commit cycle at that gap
    tag: str
    first: int = 0       # index, within the producer's planes, of this value's first plane

    def planes(self, start, width):
        """The `width` planes from plane `start` of this value, as a value of their own (a 16-stream result
        stored quad by quad)."""
        if start < 0 or start + width > self.width:
            raise ScheduleError(f"planes {start}..{start + width - 1} of a {self.width}-plane value")
        return Value(self.row, self.stream + start, width, self.gap, self.cycle, self.tag, self.first + start)

    def at(self, g):
        """The cycle in which the value is at gap g (downstream only)."""
        if g <= self.gap:
            raise ScheduleError(f"gap {g} is not downstream of {self.gap}")
        return self.cycle + g - self.gap

    def operand(self, chain=False):
        return f"{self.row}:{self.stream}" + (".chain" if chain else "")


class Builder:
    def __init__(self, p, comment=None):
        self.p = p
        self.prog = emit.Program(p, comment)
        self.t = sched.Tracker(p)

    def _write(self, row, stream, width, gap, cycle, tag=None):
        tag = tag or f"v{next(_ids)}"
        for k in range(width):
            self.t.write(row, stream + k, gap, cycle, f"{tag}.{k}")
        return Value(row, stream, width, gap, cycle, tag)

    def _read(self, v, queue, mnemonic, t_dispatch):
        g = stations.need_gap(self.p, queue, v.row, "read")
        _, d_skew = self.p.timing(mnemonic)
        for k in range(v.width):
            self.t.read(v.row, v.stream + k, g, t_dispatch + d_skew, f"{v.tag}.{v.first + k}")

    def when(self, v, mnemonic, queue):
        """The dispatch cycle at which `mnemonic` on `queue` reads value v (the sec. 1 equation)."""
        return stations.meet(self.p, v.gap, v.cycle, mnemonic, queue, v.row)

    # -- MEM
    def _addr(self, k, a, addr_text, patch, emit):
        """(operand text, patch.Decl or None) of plane k's address `a`: a patch site `patch[k]` = (name, lo, hi[, stride,
        span[, any_bank]]) carries `a` as its placeholder (`patch.Decl`; `any_bank` needs stride and span first)."""
        if not patch or patch[k] is None:
            return (addr_text[k] if addr_text else f"{a:#x}"), None
        if addr_text or not emit:
            raise tpatch.PatchError(f"patch site {patch[k][0]}: needs emitted text and no address text")
        d = tpatch.Decl(patch[k][0], "addr", patch[k][1], patch[k][2], a, *patch[k][3:])
        return d.token, d

    def _bank(self, queue, cycle, path, a, thread, site):
        if site is not None and site.any_bank:
            self.t.bank_any(queue, cycle, path, thread)
        else:
            self.t.bank(queue, cycle, path, a, thread)

    def read(self, cycle, slices, addrs, row, stream, thread=0, addr_text=None, tag=None, emit=True, text_cycle=None,
             patch=None):
        """Read one word per plane: plane k from slices[k] at addrs[k], all in one cycle; the slices must share
        a gap on `row` (one quad). Returns the Value. `emit=False` tracks without text (a Repeat iteration);
        `text_cycle` writes the text at another cycle than the dispatch tracked (a parcel that waits on a barrier);
        `patch[k]` = (name, lo, hi[, stride, span[, any_bank]]) makes plane k's address a patch site with range lo..hi (opentsp_tools.patch)."""
        gaps = {stations.need_gap(self.p, s, row, "write") for s in slices}
        if len(gaps) != 1:
            raise ScheduleError(f"planes on slices {slices} would commit at different gaps {gaps}")
        for k, (s, a) in enumerate(zip(slices, addrs)):
            self.t.slot(s, cycle, "R", thread)
            text, site = self._addr(k, a, addr_text, patch, emit)
            self._bank(s, cycle, "R", a, thread, site)
            if emit:
                self.prog.add(cycle if text_cycle is None else text_cycle, qname(s),
                              f"read {text}, {row}:{stream + k}, {thread:#x}", site=site)
        g, c = stations.commit(self.p, cycle, "read", slices[0], row)
        return self._write(row, stream, len(slices), g, c, tag)

    def read_cycle_for(self, t_consumer, consumer, consumer_queue, slice_, row):
        """The cycle of the Read on `slice_` whose value meets `consumer` dispatched at `t_consumer` on
        `consumer_queue`."""
        g_w = stations.need_gap(self.p, slice_, row, "write")
        g_r = stations.need_gap(self.p, consumer_queue, row, "read")
        if g_r <= g_w:
            raise ScheduleError(f"slice {slice_} on row {row} is not upstream of queue {consumer_queue}")
        d_func, _ = self.p.timing("read")
        _, d_skew = self.p.timing(consumer)
        return t_consumer + d_skew - (g_r - g_w) - d_func

    def read_for(self, t_consumer, consumer, consumer_queue, slices, addrs, row, stream, **kw):
        """Read so that the value meets `consumer` dispatched at `t_consumer` on `consumer_queue`."""
        return self.read(self.read_cycle_for(t_consumer, consumer, consumer_queue, slices[0], row), slices, addrs, row,
                         stream, **kw)

    def write(self, v, slices, addrs, thread=0, cycle=None, emit=True, patch=None, window=False):
        """Write plane k of v to slices[k] at addrs[k]; the dispatch cycle follows from v unless given. `patch` as
        in `read`. `window`: the Write is a KV pool write, its cycle a write window of the slice (Tracker.window)."""
        t = self.when(v, "write", slices[0]) if cycle is None else cycle
        for k, (s, a) in enumerate(zip(slices, addrs)):
            if self.when(v, "write", s) != t:
                raise ScheduleError(f"plane {k} of {v.tag} does not meet slice {s} at {t}")
            self.t.slot(s, t, "W", thread)
            if window:
                self.t.window(s, t, a)
            text, site = self._addr(k, a, None, patch, emit)
            self._bank(s, t, "W", a, thread, site)
            if emit:
                self.prog.add(t, qname(s), f"write {text}, {v.row}:{v.stream + k}, {thread:#x}", site=site)
        self._read(v, slices[0], "write", t)
        return t

    # -- VXM
    def vxm(self, alu, op, dst_row, dst_stream, srcs, width=None, cycle=None, chain=(), tag=None, emit=True):
        """One VXM operation on queue VXM[alu]: `op` is the mnemonic with its type (`add.i8`); the sources are
        Values met on their diagonals (or, for an index in `chain`, the previous result at the station)."""
        queue = 88 + alu
        mnem = op          # with its element type: a typed timing row applies
        _, d_skew = self.p.timing(mnem)
        times = {self.when(v, mnem, queue) for i, v in enumerate(srcs) if i not in chain}
        # A chained operand reads the result committed at the VXM station in the previous cycle.
        for i in chain:
            if srcs[i].gap != stations.need_gap(self.p, queue, srcs[i].row, "write"):
                raise ScheduleError(f"{op}: chained operand {srcs[i].tag} is not a VXM result")
            times.add(srcs[i].cycle + 1 - d_skew)
        if cycle is None:
            if len(times) != 1:
                raise ScheduleError(f"{op}: operands meet the VXM at different cycles {times}")
            cycle = times.pop()
        elif times and times != {cycle}:
            raise ScheduleError(f"{op}: operands meet the VXM at {times}, not {cycle}")
        self.t.slot(queue, cycle)
        for i, v in enumerate(srcs):
            if i not in chain:
                self._read(v, queue, mnem, cycle)
        ops = ", ".join(v.operand(i in chain) for i, v in enumerate(srcs))
        w = width if width is not None else srcs[0].width
        if emit:
            self.prog.add(cycle, qname(queue), f"{op} {dst_row}:{dst_stream}, {ops}")
        g, c = stations.commit(self.p, cycle, mnem, queue, dst_row)
        return self._write(dst_row, dst_stream, w, g, c, tag)

    # -- MXM, SXM: any other data-queue instruction with stream operands
    def unit(self, queue, op, text, srcs=(), dst=None, cycle=None, tag=None, site=None):
        """One instruction of any data queue that reads streams and may write one result: `op` is the mnemonic whose
        timing class applies (`lw`, `iw`, `abc`, `acc`, `shift`, `sxselect`), `text` the whole instruction text,
        `srcs` the Values it reads (every plane of each, all met at one cycle on the queue's read station) and `dst`
        = (row, first stream, width) the result, committed at the queue's write station d_func after the dispatch.
        `site` is the patch.Decl of the token `text` carries (opentsp_tools.patch). With no source (IW from the buffer)
        `cycle` is required. Returns (dispatch cycle, result Value or None)."""
        d_func, d_skew = self.p.timing(op)
        times = {self.when(v, op, queue) for v in srcs}
        if cycle is None:
            if len(times) != 1:
                raise ScheduleError(f"{op}: operands meet queue {queue} at different cycles {sorted(times)}")
            cycle = times.pop()
        elif times and times != {cycle}:
            raise ScheduleError(f"{op}: operands meet queue {queue} at {sorted(times)}, not {cycle}")
        self.t.slot(queue, cycle)
        for v in srcs:
            self._read(v, queue, op, cycle)
        self.prog.add(cycle, qname(queue), text, site=site)
        if dst is None:
            return cycle, None
        row, stream, width = dst
        g, c = stations.commit(self.p, cycle, op, queue, row)
        return cycle, self._write(row, stream, width, g, c, tag)

    # -- AGT loops (microarchitecture.md sec. 4)
    def read_loop(self, cycle, slice_, addr, row, stream, records, addr_text=None, thread=0, patch=None):
        """A Read at `cycle` followed by AGT configuration records on the next cycles; returns one Value per
        reference (with its cycle, address and stream from agt_refs). `slice_` may be a list of the slices of one
        quad: every reference then reads all of them at one word (a Value of that many planes). `patch` as in `read`
        applies to the first reference, the one the text carries."""
        slices = [slice_] if isinstance(slice_, int) else list(slice_)
        refs = agt_refs(cycle, addr, stream, records)
        out = []
        for i, (c, a, s) in enumerate(refs):
            if not 0 <= a < self.p.words or not 0 <= s < self.p.streams:
                raise ScheduleError(f"AGT reference {i} at {c}: address {a} or stream {s} out of range")
            out.append(self.read(c, slices, [a] * len(slices), row, s, thread=thread, emit=(i == 0),
                                 addr_text=addr_text if i == 0 else None, patch=patch if i == 0 else None))
        for s in slices:
            self._config(cycle, s, records)
        return out

    def write_loop(self, values, slice_, addr, records, thread=0):
        """A Write of values[0] followed by AGT records; reference k must meet values[k] (checked)."""
        t0 = self.when(values[0], "write", slice_)
        refs = agt_refs(t0, addr, values[0].stream, records)
        if len(refs) != len(values):
            raise ScheduleError(f"write loop makes {len(refs)} references for {len(values)} values")
        for i, ((c, a, s), v) in enumerate(zip(refs, values)):
            if s != v.stream or self.when(v, "write", slice_) != c:
                raise ScheduleError(f"write reference {i} at {c} on stream {s} misses {v.tag} (stream {v.stream})")
            self.write(v, [slice_], [a], thread=thread, cycle=c, emit=(i == 0))
        self._config(t0, slice_, records)
        return refs

    def _config(self, cycle, slice_, records):
        for j, (field, val) in enumerate(records):
            self.raw(cycle + 1 + j, slice_, f"{field} {val & 0xFFFF:#x}", slot="config")   # simm16: two's complement

    # -- anything else (control, AGT configuration, negative cases): no tracking beyond the slot
    def raw(self, cycle, queue, text, slot="insn", thread=0):
        if slot:
            self.t.slot(queue, cycle, slot, thread)
        self.prog.add(cycle, qname(queue), text)

    def finish(self):
        self.t.check()
        return self.prog.text()
