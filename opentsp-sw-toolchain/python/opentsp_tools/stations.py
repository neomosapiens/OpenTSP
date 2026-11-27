"""Station gaps (encoding.json `stations`, topology.md sec. 4) and the scheduling equation of the design's sec. 1:

    T_C = T_P + d_func(P) + (G_C - g_P) - d_skew(C),   G_C > g_P on the chosen row.

A value committed at writer gap g in cycle c is at gap G > g exactly in cycle c + (G - g), downstream only.
"""
import re

from .profile import encoding


class ScheduleError(Exception):
    """A generation-time violation: the generator is wrong, never the program."""


def unit_of(queue):
    units = encoding()["units"]
    for name, qs in units.items():
        if name in ("text", "any"):
            continue
        if queue in qs:
            return name
    raise ScheduleError(f"queue {queue} has no unit")


def gap(p, queue, row, kind):
    """The gap at which `queue` reads (kind 'read') or writes ('write') stream row `row`, or None."""
    unit = unit_of(queue)
    expr = encoding()["stations"][unit][kind][row]
    if expr is None:
        return None
    if unit.startswith("MEM") and queue % 44 >= p.slices_per_hemisphere:
        return None
    m = re.fullmatch(r"layout\.(\w+)(?: ([-+]) quad)?", expr)
    if not m:
        raise ScheduleError(f"station expression {expr!r}")
    g = p.layout[m.group(1)]
    if m.group(2):
        quad = (queue % 44) // 4
        g = g - quad if m.group(2) == "-" else g + quad
    return g if 0 <= g <= 43 else None


def need_gap(p, queue, row, kind):
    g = gap(p, queue, row, kind)
    if g is None:
        raise ScheduleError(f"queue {queue} cannot {kind} row {row} (no station)")
    return g


def commit(p, t_dispatch, mnemonic, queue, row, dtype=None):
    """(gap, cycle) at which the result of `mnemonic` (of element type `dtype`, when it has a typed timing row) dispatched at t on `queue`
    is committed on `row`."""
    d_func, _ = p.timing(mnemonic, dtype)
    return need_gap(p, queue, row, "write"), t_dispatch + d_func


def consumer_dispatch(p, t_producer, producer, producer_queue, consumer, consumer_queue, row, producer_type=None, consumer_type=None):
    """The dispatch cycle at which `consumer` on `consumer_queue` reads, on `row`, the value that `producer`
    dispatched at `t_producer` on `producer_queue` writes. Raises when the reader is not downstream. The two types name the typed timing
    rows of the producer and the consumer (none: the entry's row)."""
    g_p, t_commit = commit(p, t_producer, producer, producer_queue, row, producer_type)
    return meet(p, g_p, t_commit, consumer, consumer_queue, row, consumer_type)


def meet(p, g_p, t_commit, consumer, consumer_queue, row, consumer_type=None):
    """The dispatch cycle of `consumer` that reads a value committed at (g_p, t_commit) on `row`."""
    g_c = need_gap(p, consumer_queue, row, "read")
    if g_c <= g_p:
        raise ScheduleError(f"queue {consumer_queue} reads row {row} at gap {g_c}, not downstream of gap {g_p}")
    _, d_skew = p.timing(consumer, consumer_type)
    return t_commit + (g_c - g_p) - d_skew
