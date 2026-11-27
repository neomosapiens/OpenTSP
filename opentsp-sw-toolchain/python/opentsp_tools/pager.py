"""The instruction pager : turns a cycle view whose lowered queue images exceed `icu.iq_depth` into a
fetch-driven one. `page(cycle_view, profile, tracker)` returns (cycle_view', blocks, externs):

  cycle_view'  the entries that stay in the initial images (whole parcels), plus the `ifetch`/`ifetchm` entries that
               bring the rest, and the `read` entries on MEM slices that put each `ifetch` payload on a stream pair;
  blocks       the text that fetches bring: one payload of 2 * vlanes bytes per fetch, the units of the queue's
               lowering after its initial image, the last one padded with 0xFFFFFFFF;
  externs      the `.extern` declarations binding each block to SRAM words, bound from a weight package as raw
               tensors (dtype 255, assembly.md sec. 4).

Model of the front end (microarchitecture.md sec. 3, model/slices/cu/cu_iq.sail): an IfetchM that executes at cycle t
reads its second word at t + 1 and delivers from t + 1 + icu.fetch_latency. Ifetch reads its one stream operand
at t and t + 1 and delivers from the same boundary, at icu.fetch_bytes_per_cycle / 4 units per cycle, payloads in the order
the fetches executed, before the dispatch of that cycle; a unit that finds iq_depth / 4 units in the IQ is lost
(the diagnostic OTS-E-IQ-002); a parcel that completes after the cycle it is due is the lint OTS-W-IQ-001. Dispatch cycles are the cycles of
the view (no queue waits), so the occupancy at every arrival is computed exactly. Per queue the planner takes, for
each block, the latest fetch cycle at which every parcel of the block is complete by its due cycle, the queue
dispatches nothing else and the fetch finds its resources: a free stream (two consecutive cycles) and two free MEM
slices of one quad for `ifetch`, an idle R path for `ifetchm`. A fetch is itself text, so the plan is iterated until the fetch cycles are a
fixed point, then checked by a cycle-by-cycle model (`iq_findings`). It fails with a ScheduleError where no legal cycle
exists.

Constraints on the program. `page` refuses a program that breaks one with a ScheduleError naming the queue and the cycle;
tests/programs/gen/test_pager.py pins each refusal (the machine events behind them are in assembly.md, "Program-data
packages"):

  Barriers (Sync, Notify, Deskew, RuntimeDeskew). A queue behind a barrier no longer dispatches at the cycles of the view: a Sync
      parks it until the release cycle (a Notify at cycle N releases at N + barrier.release_latency), a Deskew or a RuntimeDeskew holds it
      to the next HAC epoch boundary (c2c.hac_period). The plan computes the occupancy of the IQ at every arrival from the cycles of the
      view, so after a barrier it is unknown: "after a barrier the cycles of the view are not dispatch cycles". `allow_waits=True` lifts the
      refusal for a caller that has already put the actual dispatch cycles into the view.
  Holds. A Repeat holds its queue through its last iteration, an AGT loop holds its path through its trailing countdowns, and a parcel
      that is due inside a hold waits. A Repeat on a queue that needs paging is refused (the plan does not model the head parcel's hold,
      "repeat on a paged queue"). Elsewhere the pager works around holds: a fetch is never placed in a cycle where the queue or the
      path it needs is held (`_windows`, `held`), and an `ifetchm` stays out of AGT loops.
  Control channel. The CU queue has no fetch and is never paged: a CU image longer than `icu.iq_depth` / 4 units is refused.
  Slices that host text. The words from `floor` (default: half of the slice) upwards of every MEM slice the pager uses are free
      (or the `regions` given); the pager takes its words from the top of the slice, outside every extern. A slice whose words
      are read onto a stream as text must run no SetVL or Config while it does (both gate or split the stream a Read puts out).
  Resources. A fetch needs a free stream (two consecutive cycles) and two free MEM slices of one quad for `ifetch`, an idle R path for
      `ifetchm`; where no cycle is legal the planner fails with the queue, the block and the cycles it tried."""
import bisect
from dataclasses import dataclass

from . import emit, patch, refasm, sched, stations
from .queues import agt_refs, qname
from .stations import ScheduleError

tsp_as = refasm.load()

PAD = tsp_as.PADDING
NOP_MAX = tsp_as.NOP_MAX
CU = tsp_as.CONTROL
LAST_MEM = 87
WAITS = ("sync", "notify", "deskew", "runtime_deskew")
REF_R, REF_W = ("read", "gather"), ("write", "scatter")
CONFIG = ("countdown", "step", "streamstep", "iterations")
MAX_ITERATIONS = 40


# -- the cycle view ------------------------------------------------------------------------------------------------
class CycleView:
    """The entries of a program: (cycle, queue id, instruction text), in source order within a cycle, and its
    externs (name, slice, addr, size). `added` lists the entries the pager added."""

    def __init__(self, entries, externs, comment=None, sites=()):
        self.entries = list(entries)
        self.externs = list(externs)
        self.comment = comment
        self.sites = list(sites)         # the patch.Site records of the entries' tokens
        self.added = []
        self.full = None                 # a paged view: the whole text (the cut entries, the fetches), for `locate`

    @classmethod
    def of(cls, prog):
        """The view of an emit.Program."""
        return cls([(c, tsp_as.queue_id(q), t) for c in sorted(prog.bundles) for q, t in prog.bundles[c]],
                   prog.externs, prog.comment, prog.sites)

    def program(self, p):
        prog = emit.Program(p, self.comment)
        prog.externs = list(self.externs)
        prog.sites = list(self.sites)
        for c, q, t in sorted(self.entries, key=lambda e: e[0]):      # stable: source order within a cycle
            prog.add(c, qname(q), t)
        return prog


def encode(view, p):
    """{queue: [(cycle, [word, ...])]} of a view: the assembler's words, one parcel per cycle; a NOP alone in its
    cycle makes no parcel (tsp_as.lower)."""
    prog = tsp_as.assemble(view.program(p).text(), config=p.path)[1]
    out = {}
    for c, q, _, w, _ in prog.records:
        ps = out.setdefault(q, [])
        if ps and ps[-1][0] == c:
            ps[-1][1].append(w)
        else:
            ps.append((c, [w]))
    for q in out:
        out[q] = [(c, ws) for c, ws in out[q] if not (len(ws) == 1 and tsp_as.is_nop_word(ws[0]))]
    return out


def layout(parcels):
    """(units, table) of a queue's parcels [(cycle, [word])]: the lowering of tsp_as.lower, and for each parcel
    (cycle, first unit including its NOP prefix, end unit)."""
    units, table, prev = [], [], -1
    for c, ws in parcels:
        u0 = len(units)
        gap = c - prev - 1
        while gap > 0:
            n = min(gap, NOP_MAX)
            units.append(n)
            gap -= n
        for k, w in enumerate(ws):
            u = tsp_as.word_units(w)
            if k < len(ws) - 1:
                u[0] |= tsp_as.BUNDLE_BIT
            units += u
        table.append((c, u0, len(units)))
        prev = c
    return units, table


def _table(counts):
    """The table of `layout` from {cycle: units of the parcel's instructions}, without building the units."""
    table, n, prev = [], 0, -1
    for c in sorted(counts):
        u0 = n
        gap = c - prev - 1
        if gap > 0:
            n += -(-gap // NOP_MAX)
        n += counts[c]
        table.append((c, u0, n))
        prev = c
    return table


# -- the fetch model -----------------------------------------------------------------------------------------------
def arrive(state, ready, n_units, rate):
    """The delivery cycles of the n_units of a payload that is ready at cycle `ready`, behind the payloads before it:
    `rate` units per cycle in order (iq_deliver). state = (last cycle, units delivered in it)."""
    cur, used = state
    out = []
    for _ in range(n_units):
        c = max(ready, cur)
        if c == cur and used >= rate:
            c = cur + 1
        if c != cur:
            used = 0
        used += 1
        cur = c
        out.append(c)
    return out, (cur, used)


class Geometry:
    def __init__(self, p):
        self.cap = p.icu["iq_depth"] // 4
        self.block = p.vlanes // 2                      # units of one payload (2 * vlanes bytes)
        self.rate = p.icu["fetch_bytes_per_cycle"] // 4
        self.latency = p.icu["fetch_latency"]
        self.staging = p.icu["fetch_staging_depth"]

    def ready(self, q, t, skew):
        """The cycle from which the payload of a fetch dispatched at `t` on queue `q` enters the IQ: the execute cycle
        (`t` + skew) + 1 + icu.fetch_latency: both fetch forms read their second
        vector in the cycle after the first."""
        return t + skew + 1 + self.latency


class _FullIQ(ScheduleError):
    """A legal fetch arrived too early for the current initial-image cut."""


@dataclass
class Block:
    """One fetch: `queue`'s fetch number `index`, executing at `cycle`, bringing `tensors`: [(extern name, slice, first
    word address, bytes)]; ifetchm: one tensor of two words; ifetch: two tensors of one word each (the stream s and
    s + 1 words)."""
    queue: int
    index: int
    cycle: int
    kind: str
    tensors: list
    vlanes: int = 0


def tensor_words(blocks):
    """{extern name: [word bytes]} of the blocks, for tsp_as.write_weights / bind_weights."""
    return {n: [data[k:k + b.vlanes] for k in range(0, len(data), b.vlanes)]
            for b in blocks for n, _, _, data in b.tensors}


def with_block_bytes(blocks, data):
    """Copies of `blocks` with the text of the blocks in `data` {(queue, index): bytes of the block's units} (the patched
    text of a block): an `ifetchm` block is one tensor of two words, an `ifetch` block two of one word each."""
    out = []
    for b in blocks:
        raw = data.get((b.queue, b.index))
        if raw is None:
            out.append(b)
            continue
        v = b.vlanes
        parts = [raw] if b.kind == "ifetchm" else [raw[:v], raw[v:]]
        out.append(Block(b.queue, b.index, b.cycle, b.kind, [(n, s, a, d) for (n, s, a, _), d in zip(b.tensors, parts)], v))
    return out


def weights(blocks):
    """The blocks as the weights of a program case: {extern name: ("raw", bytes)}."""
    return {n: ("raw", data) for b in blocks for n, _, _, data in b.tensors}


# -- busy windows --------------------------------------------------------------------------------------------------
def _mn(text):
    return text.split()[0]


def _windows(texts, is_mem):
    """[(first cycle, last cycle, kind)] of one queue, kind R, W (an AGT loop on that path, from its first reference
    to its last reference plus the trailing countdowns, or one reference) or Q (a Repeat: the whole queue)."""
    out, cs = [], sorted(texts)
    for i, c in enumerate(cs):
        mns = [_mn(t) for t in texts[c]]
        if "repeat" in mns:
            prev = next((x for x in reversed(cs[:i]) if any(_mn(t) not in ("nop", "repeat") + CONFIG
                                                             for t in texts[x])), c)
            n, d = (int(v, 0) for v in texts[c][mns.index("repeat")].replace(",", " ").split()[1:3])
            out.append((c, max(prev + d + 1, c) + max(n - 1, 0) * (d + 1) + 1, "Q"))     # iteration k at p + k (d + 1)
        if not is_mem:
            continue
        paths = [k for k, names in (("R", REF_R), ("W", REF_W)) if any(m in names for m in mns)]
        recs, k = [], 1
        while c + k in texts and any(_mn(t) in CONFIG for t in texts[c + k]):
            t = next(t for t in texts[c + k] if _mn(t) in CONFIG).split()
            recs.append((t[0], int(t[1], 0)))
            k += 1
        for j, path in enumerate(paths):
            mine = recs if len(paths) == 1 else recs[j::2]
            if not mine:
                out.append((c, c, path))
                continue
            last = agt_refs(c, 0, 0, mine)[-1][0]
            trail = sum(v for f, v in mine if f == "countdown")
            out.append((c, max(last + trail, c + len(mine)) + 1, path))
    return out


# -- the allocator of text words -----------------------------------------------------------------------------------
class _Words:
    """Free SRAM words for text blocks, from the top of each slice downwards to `floor`, outside every extern. With
    `regions` ({slice: (first word, end word)}, the planner's `tx<slice>` regions) a slice's blocks are taken only from its
    region, from its end downwards, and a slice without a region has no words."""

    def __init__(self, p, externs, floor, regions=None):
        self.p, self.floor, self.regions = p, floor, regions
        self.top, self.used = {}, {}
        for _, s, a, n in externs:
            self.used.setdefault(s, []).append((a, a + n))

    def bounds(self, s):
        """(lowest word, end word) the blocks of slice s may take, or None."""
        if self.regions is None:
            return self.floor, self.p.words
        return self.regions.get(s)

    def peek(self, s, n, k=1):
        """The address of the k-th block of n words that would be taken from slice s, or None."""
        bounds = self.bounds(s)
        if bounds is None:
            return None
        lo, hi = bounds
        a = self.top.get(s, hi)
        for _ in range(k):
            a -= n
            a -= a % n
            while a >= lo:
                clash = [lo_ for lo_, hi_ in self.used.get(s, []) if lo_ < a + n and a < hi_]
                if not clash:
                    break
                a = min(clash) - n
                a -= a % n
            if a < lo:
                return None
        return a

    def refuse(self, s):
        if self.regions is None:
            raise ScheduleError(f"{qname(s)}: no free word for text between {self.floor} and the top of the slice")
        where = "has no text region" if self.bounds(s) is None else \
            f"has no free word in its text region {self.bounds(s)[0]}..{self.bounds(s)[1] - 1}"
        raise ScheduleError(f"{qname(s)}: no free word for text: the slice {where}")

    def take(self, s, n):
        a = self.peek(s, n)
        if a is None:
            self.refuse(s)
        self.top[s] = a
        return a


# -- the planner ---------------------------------------------------------------------------------------------------
class _Pager:
    def __init__(self, view, p, tracker, floor, allow_waits, regions=None):
        self.view, self.p, self.t, self.allow_waits = view, p, tracker, allow_waits
        self.g = Geometry(p)
        self.words = _Words(p, view.externs, p.words // 2 if floor is None else floor, regions)
        self.texts, self.counts = {}, {}
        for c, q, t in view.entries:
            self.texts.setdefault(q, {}).setdefault(c, []).append(t)
        for q, ps in encode(view, p).items():
            self.counts[q] = {c: sum(len(tsp_as.word_units(w)) for w in ws) for c, ws in ps}
        self.windows = {q: _windows(ts, q <= LAST_MEM) for q, ts in self.texts.items()}
        self.tainted = {q for q, ts in self.texts.items() if q <= LAST_MEM and
                        any(_mn(t) in ("setvl", "config") for cs in ts.values() for t in cs)}
        self.added, self.externs, self.blocks, self.paged = [], [], {}, {}
        self.initial_cap = {}
        self.stream_fetch = set()
        self.trial = None
        self.trial_words = {}
        self.d_func = p.timing("read")[0]
        self.d_skew = p.timing("ifetch")[1]
        self.hosts = {}

    # entries and windows
    def add(self, c, q, text, units):
        self.added.append((c, q, text))
        self.texts.setdefault(q, {}).setdefault(c, []).append(text)
        self.counts.setdefault(q, {})
        self.counts[q][c] = self.counts[q].get(c, 0) + units

    def held(self, q, c, kinds):
        return any(lo <= c <= hi and (k in kinds or k == "Q") for lo, hi, k in self.windows.get(q, []))

    def occupied(self, q, c):
        return c in self.texts.get(q, {})

    def size(self, q):
        t = _table(self.counts.get(q, {}))
        return t[-1][2] if t else 0

    # resources of one fetch
    def host_free(self, s, c):
        return (s not in self.blocks and s not in self.tainted and not self.occupied(s, c) and not self.held(s, c, ("R", "W"))
                and self.t.free(s, c, "R", 0)
                and (self.trial is None or self.trial.free(s, c, "insn")))

    def candidates(self, q):
        if q not in self.hosts:
            p, out = self.p, []
            for row in "EW":
                g_r = stations.gap(p, q, row, "read")
                if g_r is None:
                    continue
                for base in (0, 44):
                    for quad in range((p.slices_per_hemisphere + 3) // 4):
                        for off in (0, 2):
                            a, b = base + 4 * quad + off, base + 4 * quad + off + 1
                            if q in (a, b):
                                continue
                            ga, gb = stations.gap(p, a, row, "write"), stations.gap(p, b, row, "write")
                            if ga is not None and ga == gb and ga < g_r:
                                out.append((g_r - ga, row, a, b, ga, g_r))
            self.hosts[q] = [c[1:] for c in sorted(out)]
        return self.hosts[q]

    def stream_plan(self, q, t, j):
        """(row, slice A, slice B, g_w, g_r, read cycle, stream) of an `ifetch` of queue q at t, or None. The one stream
        operand carries two vectors in consecutive cycles: A's word is read at the read cycle, B's one cycle later, both
        onto `stream` (a pair of slices of one quad, one write gap)."""
        if t < 0 or self.occupied(q, t) or self.held(q, t, ()) or not self.t.free(q, t, "insn"):
            return None
        cands = self.candidates(q)
        for k in range(len(cands)):
            row, a, b, g_w, g_r = cands[(j + k) % len(cands)]
            r = t + self.d_skew - (g_r - g_w) - self.d_func
            if r < 0 or self.words.peek(a, 1, 1 + self.trial_words.get(a, 0)) is None or self.words.peek(b, 1, 1 + self.trial_words.get(b, 0)) is None:
                continue
            if not (self.host_free(a, r) and self.host_free(b, r + 1)):
                continue
            for s in range(self.p.streams - 1, -1, -1):
                if all(tr.stream_free(row, s, g_w, r + i + self.d_func, g_r)
                       for tr in (self.t, self.trial) if tr is not None for i in (0, 1)):
                    return row, a, b, g_w, g_r, r, s
        return None

    def mem_plan(self, q, t, j):
        if (t < 0 or self.occupied(q, t) or not self.t.free(q, t, "fetch")
                or any(self.held(q, c, ("R", "W") if c == t else ("R",))
                       or not all(self.t.free(q, c, "R", th) for th in (0, 1)) for c in (t, t + 1))):
            return None
        address = self.words.peek(q, 2, j + 1)
        if address is None:
            return None
        bank = self.p.bank(address + 1)
        # A second-cycle Write is legal in the other bank. Other instructions
        # are conservatively excluded; an unknown/patchable W bank conflicts.
        if any(_mn(text) != "write" for text in self.texts.get(q, {}).get(t + 1, [])):
            return None
        if self.t.windows.get((q, t + 1)) == bank:
            return None
        for th in (0, 1):
            wb = self.t.banks.get((q, t + 1, th), {}).get("W", "none")
            if wb in (bank, sched.ANY):
                return None
        return address

    def placement(self, q, t, j):
        return self.mem_plan(q, t, j) is not None if q <= LAST_MEM and q not in self.stream_fetch else self.stream_plan(q, t, j) is not None

    # planning one queue
    def times(self, q, table, i_len, m, prev):
        # Each fixed-point attempt gets an independent reservation ledger.
        # Planned stream fetches must not steal one another's producers or
        # SRAM words before commit publishes them to the real tracker.
        self.trial = sched.Tracker(self.p)
        self.trial_words = {}
        try:
            return self._times(q, table, i_len, m, prev)
        finally:
            self.trial = None
            self.trial_words = {}

    def _times(self, q, table, i_len, m, prev):
        g, n = self.g, table[-1][2]
        cycles, ends = [c for c, _, _ in table], [e for _, _, e in table]
        state, out = (-1, 0), []
        for j in range(m):
            lo = i_len + j * g.block
            hi = min(n, lo + g.block)
            first, last = bisect.bisect_right(ends, lo), bisect.bisect_right(ends, hi)
            need = {ends[i] - 1 - lo: cycles[i] for i in range(first, last)}
            tail = bisect.bisect_right(ends, hi - 1)          # the parcel that holds the block's last unit
            need.setdefault(hi - 1 - lo, cycles[min(tail, len(cycles) - 1)])

            def ok(t):
                arr, _ = arrive(state, g.ready(q, t, self.d_skew), g.block, g.rate)
                return all(arr[i] <= due for i, due in need.items())

            # the fetch must itself sit in text that precedes its block
            t_hi = min(min(need.values()) - g.ready(q, 0, self.d_skew), cycles[first - 1] if first else -1)
            if t_hi < 0 or not ok(0):
                raise ScheduleError(f"{qname(q)}: block {j} (units {lo}..{hi - 1}, due from cycle "
                                    f"{min(need.values())}) cannot arrive in time: no fetch at cycle >= 0")
            a, b = 0, t_hi
            while a < b:
                mid = (a + b + 1) // 2
                a, b = (mid, b) if ok(mid) else (a, mid - 1)
            start = a if j >= len(prev) else min(a, prev[j])
            if q <= LAST_MEM and q not in self.stream_fetch and self.words.peek(q, 2, j + 1) is None:
                self.words.refuse(q)
            floor = out[-1] + 2 if out else 0
            t = start
            while t >= floor and not self.placement(q, t, j):
                t -= 1
            if t < floor:
                raise ScheduleError(f"{qname(q)}: no legal fetch cycle for block {j} in cycles {floor}..{start} "
                                    f"(latest arrival {a}): the queue dispatches there or no stream pair, "
                                    f"slice or R path is free")
            arr, state = arrive(state, g.ready(q, t, self.d_skew), g.block, g.rate)
            for i in range(hi - lo):
                idx = bisect.bisect_left(cycles, arr[i])
                popped = table[idx - 1][2] if idx else 0
                if lo + i + 1 - popped > g.cap:
                    raise _FullIQ(f"{qname(q)}: block {j}, fetch at cycle {t}: unit {lo + i} arrives at "
                                        f"{arr[i]} in a full IQ (OTS-E-IQ-002)")
            if q > LAST_MEM or q in self.stream_fetch:
                row, a, b, g_w, g_r, r, stream = self.stream_plan(q, t, j)
                for i, sl in enumerate((a, b)):
                    self.trial.slot(sl, r + i, "insn")
                    self.trial_words[sl] = self.trial_words.get(sl, 0) + 1
                    tag = f"trial.{q}.{j}.{i}"
                    self.trial.write(row, stream, g_w, r + i + self.d_func, tag)
                    self.trial.read(row, stream, g_r, t + self.d_skew + i, tag)
            out.append(t)
        return out

    def plan(self, q):
        try:
            return self.plan_initial_image(q)
        except ScheduleError as local_error:
            if q > LAST_MEM:
                raise
            # A dense SRAM schedule can have no two-cycle local-fetch gap.
            # Ifetch occupies only the queue's fetch slot and reads text from
            # upstream slices. No plan has been committed at this point.
            self.stream_fetch.add(q)
            try:
                return self.plan_initial_image(q)
            except ScheduleError as stream_error:
                raise ScheduleError(f"{local_error}; stream-fetch alternative: {stream_error}") from stream_error

    def plan_initial_image(self, q):
        """Try smaller initial images when an early legal fetch would overflow.

        Each attempt removes at least one unit from the capacity bound; at most
        cap attempts are possible. The chosen cut is also used by finish.
        """
        for capacity in range(self.g.cap, 0, -1):
            self.initial_cap[q] = capacity
            try:
                return self.plan_at_capacity(q)
            except _FullIQ:
                if capacity == 1:
                    raise
        raise ScheduleError(f"{qname(q)}: no initial image fits")

    def plan_at_capacity(self, q):
        """The fetch cycles of queue q (a fixed point of the plan), and its initial image's parcel count."""
        g, fetches = self.g, []
        base = dict(self.counts[q])
        for _ in range(MAX_ITERATIONS):
            counts = dict(base)
            for t in fetches:
                counts[t] = counts.get(t, 0) + 1
            table = _table(counts)
            ends = [e for _, _, e in table]
            k = bisect.bisect_right(ends, self.initial_cap[q])
            if k == 0:
                raise ScheduleError(f"{qname(q)}: the first parcel does not fit icu.iq_depth")
            i_len = ends[k - 1]
            m = -(-(ends[-1] - i_len) // g.block)
            new = self.times(q, table, i_len, m, fetches)
            if new == fetches:
                for j, t in enumerate(fetches):             # a fetch must be in text that precedes its own block
                    if ends[bisect.bisect_left([c for c, _, _ in table], t)] > i_len + j * g.block:
                        raise ScheduleError(f"{qname(q)}: the fetch of block {j} at cycle {t} lies in its own block")
                return fetches
            fetches = new
        raise ScheduleError(f"{qname(q)}: the fetch plan does not converge in {MAX_ITERATIONS} iterations")

    def commit(self, q, fetches):
        tr = self.t
        blocks = []
        for j, t in enumerate(fetches):
            if q <= LAST_MEM and q not in self.stream_fetch:
                a = self.words.take(q, 2)
                name = f"tx{q}b{j}"
                self.externs.append((name, q, a, 2))
                tr.slot(q, t, "fetch")
                for th in (0, 1):
                    tr.slot(q, t, "R", th)
                    tr.slot(q, t + 1, "R", th)
                    tr.bank(q, t, "R", a, th)
                    tr.bank(q, t + 1, "R", a + 1, th)
                self.add(t, q, f"ifetchm {name}", 1)
                blocks.append(Block(q, j, t, "ifetchm", [(name, q, a, None)], self.p.vlanes))
                continue
            plan = self.stream_plan(q, t, j)
            if plan is None:
                raise ScheduleError(f"{qname(q)}: fetch {j} at cycle {t} lost its resources")
            row, a, b, g_w, g_r, r, s = plan
            tensors = []
            for i, (sl, nm) in enumerate(((a, f"tx{q}b{j}a"), (b, f"tx{q}b{j}b"))):
                addr = self.words.take(sl, 1)
                self.externs.append((nm, sl, addr, 1))
                tr.slot(sl, r + i, "R", 0)
                tr.bank(sl, r + i, "R", addr, 0)
                tag = f"text.{q}.{j}.{i}"
                tr.write(row, s, g_w, r + i + self.d_func, tag)
                tr.read(row, s, g_r, t + self.d_skew + i, tag)
                self.add(r + i, sl, f"read {nm}, {row}:{s}, 0x0", 2)
                tensors.append((nm, sl, addr, None))
            tr.slot(q, t, "insn")
            self.add(t, q, f"ifetch {row}:{s}", 1)
            blocks.append(Block(q, j, t, "ifetch", tensors, self.p.vlanes))
        self.blocks[q] = blocks

    def run(self):
        g, view = self.g, self.view
        over = [q for q in sorted(self.counts) if self.size(q) > g.cap]
        if not over:
            return view, [], []
        if CU in over:
            raise ScheduleError(f"CU: {self.size(CU)} units exceed icu.iq_depth ({g.cap}); the control channel has "
                                f"no fetch")
        if not self.allow_waits:
            for q, ts in self.texts.items():
                for c, cs in ts.items():
                    if any(_mn(t) in WAITS for t in cs):
                        raise ScheduleError(f"{qname(q)} cycle {c}: {_mn(cs[0])}: after a barrier the cycles of the "
                                            f"view are not dispatch cycles, so the occupancy cannot be planned")
        for q in over:
            for c, cs in sorted(self.texts[q].items()):
                if any(_mn(t) == "repeat" for t in cs):
                    raise ScheduleError(f"{qname(q)} cycle {c}: repeat on a paged queue: the occupancy plan does not "
                                        f"model the head parcel's hold during a Repeat")
        for q in [q for q in over if q > LAST_MEM]:
            self.commit(q, self.plan(q))
        # A stream fetch for a MEM queue can add host Reads to another MEM
        # queue. Revisit unpaged queues until none overflow; host_free prevents
        # edits to a committed queue. At most 88 MEM queues can be committed.
        while True:
            pending = [q for q in sorted(self.counts) if q <= LAST_MEM and q not in self.blocks and self.size(q) > g.cap]
            if not pending:
                break
            q = pending[0]
            self.commit(q, self.plan(q))
        return self.finish()

    def finish(self):
        """Cut every paged queue's lowering into its initial image and its blocks."""
        p, g = self.p, self.g
        full = CycleView(self.view.entries + self.added, self.view.externs + self.externs, self.view.comment,
                         self.view.sites)
        parcels = encode(full, p)
        keep, blocks = list(full.entries), []
        for q in sorted(self.blocks):
            units, table = layout(parcels[q])
            ends = [e for _, _, e in table]
            k = bisect.bisect_right(ends, self.initial_cap[q])
            i_len = ends[k - 1]
            tail = units[i_len:]
            bl = self.blocks[q]
            if len(bl) != -(-len(tail) // g.block):
                raise ScheduleError(f"{qname(q)}: {len(bl)} fetches for {len(tail)} units of text")
            for j, blk in enumerate(bl):
                chunk = tail[j * g.block:(j + 1) * g.block]
                data = b"".join(u.to_bytes(4, "little") for u in chunk + [PAD] * (g.block - len(chunk)))
                v = p.vlanes
                blk.tensors = [(n, s, a, d) for (n, s, a, _), d in
                               zip(blk.tensors, [data] if blk.kind == "ifetchm" else [data[:v], data[v:]])]
            cut = table[k][0]
            keep = [e for e in keep if not (e[1] == q and e[0] >= cut)]
            blocks += bl
        view2 = CycleView(keep, full.externs, full.comment, full.sites)
        view2.added = list(self.added)
        view2.full = full
        externs = list(self.externs)
        lints = iq_findings(view2, blocks, p)
        if lints:
            raise ScheduleError(f"the plan has IQ lints or diagnostics {lints[:4]}")
        return view2, blocks, externs


def page(view, p, tracker, floor=None, allow_waits=False, regions=None):
    """(cycle_view', blocks, externs): see the module docstring. `tracker` is the program's sched.Tracker; the
    reads and stream writes of the fetches are recorded in it. `regions` ({slice: (first word, end word)}) confines the
    text blocks of each slice to its region (the planner's `tx<slice>`); by default they come from the top of the slice
    down to `floor`. The patch sites of the view (patch.Decl records) stay in the text as tokens; `locate` finds them
    in the paged program."""
    view2, blocks, externs = _Pager(view, p, tracker, floor, allow_waits, regions).run()
    if view2 is view:
        view2.full = view
    return view2, blocks, externs


def locate(view2, blocks, p):
    """The patch.Paged table of a paged program (the result of `page`): the sites are located in the whole text, the
    initial images and the blocks (patch.split_position gives each site's place), after checking that the text the
    machine will see (`follow`) is that text."""
    full = view2.full
    prog = tsp_as.assemble(view2.program(p).text(), config=p.path)[1]
    seen = follow(view2, blocks, p, prog)
    want = patch.lower(full, p)
    for q in sorted(set(want) | set(seen)):
        if want.get(q) != seen.get(q):
            raise patch.PatchError(f"{qname(q)}: the text of the paged program (initial image and blocks) is not the "
                                   f"lowering of the whole program")
    paged = patch.locate_paged(full, p, {q: len(prog.images[q]) for q in {b.queue for b in blocks}},
                               Geometry(p).block)
    return paged


def page_builder(b, **kw):
    """Page the program of a Builder in place (its text and externs); returns the blocks."""
    view = CycleView.of(b.prog)
    view2, blocks, _ = page(view, b.p, b.t, **kw)
    b.prog.bundles = {}
    for c, q, t in view2.entries:
        b.prog.add(c, qname(q), t)
    b.prog.externs = list(view2.externs)
    return blocks


# -- following the fetches, and the occupancy model ----------------------------------------------------------------
def _isa(p):
    return tsp_as.ISA(p.asm_name, p.path)


_fetch_cache = {}


def _fetch_entry(isa, q, w, n):
    """The decode (code, entry, fields) of a word if it is an Ifetch or IfetchM, else None (cached: decoding is
    slow)."""
    if n != 1 or (w >> 28) & 7 != 0:
        return None
    key = (isa.profile, w)
    if key not in _fetch_cache:
        d = isa.decode_word(q, w, n)
        _fetch_cache[key] = d if d[1] is not None and d[1]["name"] in ("Ifetch", "IfetchM") else None
    return _fetch_cache[key]


def _block_units(blk):
    data = b"".join(t[3] for t in blk.tensors)
    return [int.from_bytes(data[k:k + 4], "little") for k in range(0, len(data), 4)]


def follow(view2, blocks, p, prog=None):
    """{queue: units} of the full text of every queue: the initial image, then, for each fetch the text brings in the
    order it executes, the next block of the queue (padding dropped). An ifetchm must name the words of its block.
    `prog` is the assembled view2, when the caller has it."""
    prog = prog or tsp_as.assemble(view2.program(p).text(), config=p.path)[1]
    isa = _isa(p)
    out = {q: list(us) for q, us in prog.images.items()}
    # extern name -> the stream (row offset 32 for W) its Read puts it on; a Read in cut text is in `added`
    bound = {}
    for _, q, t in view2.entries + view2.added:
        f = t.replace(",", " ").split()
        if q <= LAST_MEM and f[0] == "read" and len(f) > 2 and ":" in f[2]:
            row, s = f[2].split(":")
            bound[f[1]] = int(s) + (32 if row == "W" else 0)
    by_q = {}
    for b in blocks:
        by_q.setdefault(b.queue, []).append(b)
    for q, bl in by_q.items():
        us, used, i = out.setdefault(q, []), 0, 0
        while i < len(us):
            tag = (us[i] >> 28) & 7
            n = 2 if tag in (1, 2, 4) else 1
            w = (((us[i] & 0x7FFFFFFF) << 32) | us[i + 1]) if n == 2 and i + 1 < len(us) else (us[i] & 0x7FFFFFFF)
            d = _fetch_entry(isa, q, w, n)
            if d is not None:
                if used >= len(bl):
                    raise ScheduleError(f"{qname(q)}: more fetches in its text than blocks ({len(bl)})")
                blk = bl[used]
                if blk.kind == "ifetchm" and d[2]["addr"] != blk.tensors[0][2]:
                    raise ScheduleError(f"{qname(q)}: fetch {used} reads word {d[2]['addr']}, its block is at "
                                        f"{blk.tensors[0][2]}")
                if blk.kind == "ifetch":
                    got = [bound.get(t[0]) for t in blk.tensors]
                    if got != [d[2]["stream"]] * 2:
                        raise ScheduleError(f"{qname(q)}: fetch {used} takes stream {d[2]['stream']} twice, "
                                            f"the Reads of its block put {[t[0] for t in blk.tensors]} "
                                            f"on streams {got}")
                us += [u for u in _block_units(blk) if u != PAD]
                used += 1
            i += n
        if used != len(bl):
            raise ScheduleError(f"{qname(q)}: {len(bl)} blocks, {used} fetches in its text")
    return out


def _parse(units):
    """[(nops, first unit, end unit, [(word, units)])] of a queue's text: its parcels (as tsp_as.raise_images)."""
    out, i, nops, parcel, u0 = [], 0, 0, [], 0
    while i < len(units):
        u = units[i]
        tag = (u >> 28) & 7
        n = 2 if tag in (1, 2, 4) else 1
        w = (((u & 0x7FFFFFFF) << 32) | units[i + 1]) if n == 2 and i + 1 < len(units) else (u & 0x7FFFFFFF)
        if not parcel and u & tsp_as.BUNDLE_BIT == 0 and tag == 0 and u & 0x0FFF0000 == 0 and u & 0xFFFF != 0:
            nops += u & 0xFFFF
        else:
            parcel.append((w, n))
            if u & tsp_as.BUNDLE_BIT == 0:
                out.append((nops, u0, i + n, parcel))
                nops, parcel, u0 = 0, [], i + n
        i += n
    return out


MODEL_CYCLES = 4000000        # a guard of the queue model below: a program of a chip that waits for its input from a chain's earlier chips starts late (stage 12 of 12 at about 1.2 M cycles)


def iq_findings(view2, blocks, p):
    """[(cycle, queue, code)] what the front end reports for the program (view2, blocks): the lint OTS-W-IQ-001 (underflow)
    the diagnostic OTS-E-IQ-002 (a unit that finds the IQ full is lost) and the diagnostic OTS-E-IQ-003
    (an Ifetch that finds the queue's staging FIFO of icu.fetch_staging_depth payloads full fetches nothing),
    from a cycle-by-cycle model of every paged queue (cu_iq.sail: a fetch delivers `rate` units per cycle from its
    execute cycle + latency, a full IQ loses a unit; the head parcel dispatches at its earliest cycle once complete).
    Queues are assumed free (not parked, busy or repeating) in the cycle before a late completion."""
    g, isa = Geometry(p), _isa(p)
    prog = tsp_as.assemble(view2.program(p).text(), config=p.path)[1]
    streams = follow(view2, blocks, p, prog)
    by_q = {}
    for b in blocks:
        by_q.setdefault(b.queue, []).append(b)
    lints = []
    for q in sorted(by_q):
        parcels = _parse(streams[q])
        image = len(prog.images[q])
        fifo, filled, held, last, pi, used = [], image, image, -1, 0, 0
        dry, lost = False, False
        c = 0
        while pi < len(parcels) and not lost:
            # deliver
            budget = g.rate
            complete_before = pi < len(parcels) and parcels[pi][2] <= filled and held > 0
            while budget and fifo and fifo[0][0] <= c:
                ready, units = fifo[0]
                u = units.pop(0)
                budget -= 1
                if u != PAD:
                    if held >= g.cap:
                        lints.append((c, q, "OTS-E-IQ-002"))
                        lost = True
                    else:
                        held += 1
                        filled += 1
                if not units:
                    fifo.pop(0)
            nops, u0, u1, ps = parcels[pi]
            earliest = last + 1 + nops
            if not lost and u1 <= filled and not complete_before and c > earliest:
                dry = True
            if not lost and c >= earliest and u1 <= filled:
                if dry:
                    lints.append((c, q, "OTS-W-IQ-001"))
                    dry = False
                for w, n in ps:
                    if _fetch_entry(isa, q, w, n) is not None and len(fifo) >= g.staging:
                        lints.append((c, q, "OTS-E-IQ-003"))        # the staging FIFO is full: nothing is fetched
                        lost = True
                    elif _fetch_entry(isa, q, w, n) is not None:
                        fifo.append((g.ready(q, c, p.timing("ifetch")[1]), _block_units(by_q[q][used])))
                        used += 1
                held -= u1 - u0
                last, pi = c, pi + 1
            c += 1
            if c > MODEL_CYCLES:
                raise ScheduleError(f"{qname(q)}: the model does not finish within {MODEL_CYCLES} cycles")
    return lints
