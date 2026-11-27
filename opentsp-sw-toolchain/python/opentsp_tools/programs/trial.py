"""A trial emission: everything a kernel records in the tracker and the program while the trial is open is journaled, so
that a ScheduleError, or a read that would see another producer, undoes it and the caller can try another start cycle.
The kernels need not know: the tracker's and the program's recording methods are wrapped on the instances."""
import copy

from opentsp_tools.stations import ScheduleError


class Trial:
    def __init__(self, f, extra=None):
        """`f`: the Flow (its builder, tracker, program and kit); `extra`: objects whose __dict__ is restored too."""
        self.f, self.extra = f, list(extra or [])
        self.log = []
        self.diags = set()

    def __enter__(self):
        f, tr, prog = self.f, self.f.b.t, self.f.b.prog
        self.saved = [(o, copy.copy(o.__dict__)) for o in [f.k] + self.extra]
        self.flow_state = {n: copy.deepcopy(getattr(f, n)) for n in ("streams", "next_addr", "low", "taken", "used", "vectors")
                           if hasattr(f, n)}
        log, diags = self.log, self.diags
        o_slot, o_bank, o_bank_any, o_window = tr.slot, tr.bank, tr.bank_any, tr.window
        o_write, o_read, o_add = tr.write, tr.read, prog.add

        def slot(queue, cycle, kind="insn", thread=0):
            o_slot(queue, cycle, kind, thread)
            log.append(("slot", (queue, cycle), (kind, thread) if kind in ("R", "W") else kind))

        def bank(queue, cycle, path, addr, thread=0):
            key = (queue, cycle, thread)
            prev = dict(tr.banks[key]) if key in tr.banks else None
            o_bank(queue, cycle, path, addr, thread)
            log.append(("bank", key, prev))

        def bank_any(queue, cycle, path, thread=0):
            key = (queue, cycle, thread)
            prev = dict(tr.banks[key]) if key in tr.banks else None
            o_bank_any(queue, cycle, path, thread)
            log.append(("bank", key, prev))

        def window(queue, cycle, addr):
            key = (queue, cycle)
            had = key in tr.windows
            o_window(queue, cycle, addr)
            if not had:
                log.append(("window", key, None))

        def write(row, stream, gap, cycle, tag):
            o_write(row, stream, gap, cycle, tag)
            log.append(("write", (row, stream, gap, cycle), (row, stream, cycle - gap)))
            diags.add((row, stream, cycle - gap))

        def read(row, stream, gap, cycle, want):
            o_read(row, stream, gap, cycle, want)
            log.append(("read", None, (row, stream, cycle - gap)))
            diags.add((row, stream, cycle - gap))

        def add(cycle, queue, text, site=None):
            o_add(cycle, queue, text, site)
            log.append(("add", cycle, site))

        self.orig = (o_slot, o_bank, o_bank_any, o_window, o_write, o_read, o_add)
        tr.slot, tr.bank, tr.bank_any, tr.window, tr.write, tr.read, prog.add = slot, bank, bank_any, window, write, read, add
        return self

    def _unwrap(self):
        tr, prog = self.f.b.t, self.f.b.prog
        tr.slot, tr.bank, tr.bank_any, tr.window, tr.write, tr.read, prog.add = self.orig

    def consistent(self):
        """Every read on a diagonal the trial touched still sees its intended producer."""
        tr = self.f.b.t
        for d in self.diags:
            row, stream, base = d
            for gap, want in tr.diag_reads.get(d, []):
                if tr.producer(row, stream, gap, base + gap) != want:
                    return False
        return True

    def rollback(self):
        tr, prog = self.f.b.t, self.f.b.prog
        for kind, key, val in reversed(self.log):
            if kind == "slot":
                used = tr.slots[key]
                used.discard(val)
                if not used:
                    del tr.slots[key]
            elif kind == "bank":
                if val is None:
                    tr.banks.pop(key, None)
                else:
                    tr.banks[key] = val
            elif kind == "window":
                tr.windows.pop(key, None)
            elif kind == "write":
                del tr.writes[key]
                tr.diag[val].pop()
                if not tr.diag[val]:
                    del tr.diag[val]
            elif kind == "read":
                tr.reads.pop()
                tr.diag_reads[val].pop()
                if not tr.diag_reads[val]:
                    del tr.diag_reads[val]
            elif kind == "add":
                if val is not None:
                    raise ScheduleError("a trial cannot undo a patch site")
                prog.bundles[key].pop()
                if not prog.bundles[key]:
                    del prog.bundles[key]
        for o, d in self.saved:
            o.__dict__.clear()
            o.__dict__.update(d)
        for n, v in self.flow_state.items():
            setattr(self.f, n, v)
        self.log.clear()
        self.diags.clear()

    def __exit__(self, exc_type, exc, tb):
        self._unwrap()
        if exc_type is ScheduleError:
            self.rollback()
            self.ok = False
            return True
        if exc_type is None and not self.consistent():
            self.rollback()
            self.ok = False
            return False
        self.ok = exc_type is None
        return False
