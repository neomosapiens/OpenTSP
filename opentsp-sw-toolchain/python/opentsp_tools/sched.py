"""The resource tracker, checked at generation time : the slot rule per (queue, cycle), the MEM
bank rule per thread, and stream occupancy along diagonals, so that every intended read resolves to its intended
producer and no two writers meet at one gap and cycle (OTS-E-SRF-002). A violation is a ScheduleError."""
from .stations import ScheduleError

MEM_SLOTS = ("R", "W", "config", "fetch")
ANY = "any"            # the bank of a reference patched into either bank


class Tracker:
    def __init__(self, p):
        self.p = p
        self.slots = {}       # (queue, cycle) -> set of slot keys
        self.banks = {}       # (queue, cycle, thread) -> {path: bank or None}
        self.windows = {}     # (queue, cycle) -> bank of a pool write window
        self.writes = {}      # (row, stream, gap, cycle) -> tag
        self.diag = {}        # (row, stream, cycle - gap) -> [(gap, tag)]
        self.reads = []       # (row, stream, gap, cycle, want)
        self.diag_reads = {}  # (row, stream, cycle - gap) -> [(gap, want)]

    # -- slot rule (encoding.json mem_slots): a MEM queue takes one R and one W per thread, one config and one
    #    fetch per parcel; every other queue one instruction.
    def slot(self, queue, cycle, kind="insn", thread=0):
        used = self.slots.setdefault((queue, cycle), set())
        key = (kind, thread) if kind in ("R", "W") else kind
        if queue <= 87 and kind not in MEM_SLOTS and kind != "insn":
            raise ScheduleError(f"unknown slot {kind}")
        if key in used or "insn" in used or (kind == "insn" and used):
            raise ScheduleError(f"queue {queue} cycle {cycle}: slot {key} already taken ({sorted(map(str, used))})")
        used.add(key)

    def free(self, queue, cycle, kind="insn", thread=0):
        """Whether `slot(queue, cycle, kind, thread)` would succeed."""
        used = self.slots.get((queue, cycle), set())
        key = (kind, thread) if kind in ("R", "W") else kind
        return key not in used and "insn" not in used and not (kind == "insn" and used)

    def window(self, queue, cycle, addr):
        """A write window of a pool slice : the Write's address is patched, but stays in the pool's bank
        (patch.check_mem_range), so no other reference of the slice in that cycle, of any thread or path, may address
        that bank, whatever value the host patches in. Raises when one is recorded already; `bank` refuses later ones."""
        b = self.p.bank(addr) if 0 <= addr < self.p.words else None
        for thread in (0, 1):
            if b in [v for path, v in self.banks.get((queue, cycle, thread), {}).items() if path != "W"]:
                raise ScheduleError(f"queue {queue} cycle {cycle}: a pool write window in bank {b} meets another reference")
        if self.windows.setdefault((queue, cycle), b) != b:
            raise ScheduleError(f"queue {queue} cycle {cycle}: two write windows in banks {self.windows[(queue, cycle)]} and {b}")

    def bank(self, queue, cycle, path, addr, thread=0):
        """An R and a W reference of one thread in one cycle must address different banks (OTS-E-MEM-001); a cycle in a
        pool write window takes no reference but that Write to the window's bank."""
        b = self.p.bank(addr) if 0 <= addr < self.p.words else None
        if path == "R" and b is not None and self.windows.get((queue, cycle)) == b:
            raise ScheduleError(f"queue {queue} cycle {cycle}: a Read of bank {b} in a pool write window")
        refs = self.banks.setdefault((queue, cycle, thread), {})
        other = refs.get("W" if path == "R" else "R", "none")
        if other != "none" and b is not None and other in (b, ANY):
            raise ScheduleError(f"queue {queue} cycle {cycle}: R and W of thread {thread} in bank {b}")
        refs[path] = b

    def bank_any(self, queue, cycle, path, thread=0):
        """A reference whose address, patched by the host, may be in either bank (`patch.Decl.any_bank`): it clashes
        with a reference of the other path whichever bank that is in."""
        refs = self.banks.setdefault((queue, cycle, thread), {})
        other = refs.get("W" if path == "R" else "R", "none")
        if other not in ("none", None):
            raise ScheduleError(f"queue {queue} cycle {cycle}: R and W of thread {thread} (patched {path} in either bank)")
        refs[path] = ANY

    # -- streams
    def write(self, row, stream, gap, cycle, tag):
        key = (row, stream, gap, cycle)
        if key in self.writes:
            raise ScheduleError(f"SRF-002: {tag} and {self.writes[key]} commit {row}:{stream} gap {gap} at {cycle}")
        self.writes[key] = tag
        self.diag.setdefault((row, stream, cycle - gap), []).append((gap, tag))

    def producer(self, row, stream, gap, cycle):
        """The tag whose value a reader at `gap` in `cycle` sees: the last writer upstream on its diagonal."""
        best = None
        for g, tag in self.diag.get((row, stream, cycle - gap), []):
            if g < gap and (best is None or g > best[0]):
                best = (g, tag)
        return best[1] if best else None

    def read(self, row, stream, gap, cycle, want):
        self.reads.append((row, stream, gap, cycle, want))
        self.diag_reads.setdefault((row, stream, cycle - gap), []).append((gap, want))

    def stream_free(self, row, stream, g_w, c_w, g_r):
        """Whether a value committed on `stream` of `row` at gap g_w in cycle c_w, read at gap g_r downstream, fits:
        no writer of the same register and cycle, none between the two gaps on its diagonal (it would be the value
        the reader sees), and no recorded read further downstream that it would take over from its producer."""
        if (row, stream, g_w, c_w) in self.writes:
            return False
        d = c_w - g_w
        writers = self.diag.get((row, stream, d), [])
        if any(g_w < g < g_r for g, _ in writers):
            return False
        for g_read, _ in self.diag_reads.get((row, stream, d), []):
            if g_read > g_w and not any(g_w < g < g_read for g, _ in writers):
                return False
        return True

    def bank_ok(self, queue, cycle, path, addr, thread=0):
        """Whether `bank(queue, cycle, path, addr, thread)` would succeed."""
        b = self.p.bank(addr) if 0 <= addr < self.p.words else None
        other = self.banks.get((queue, cycle, thread), {}).get("W" if path == "R" else "R", "none")
        return other == "none" or b is None or other not in (b, ANY)

    def check(self):
        for row, stream, gap, cycle, want in self.reads:
            got = self.producer(row, stream, gap, cycle)
            if got != want:
                raise ScheduleError(f"read of {row}:{stream} at gap {gap} cycle {cycle} sees {got}, wants {want}")
