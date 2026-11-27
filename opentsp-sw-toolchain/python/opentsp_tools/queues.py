"""Queue names and the references of an AGT loop: what a scheduler and the pager share about the machine's queues."""

from .stations import ScheduleError


def qname(queue):
    if queue == 0xFFFF:
        return "CU"
    if queue <= 87:
        return f"MEM[{queue}]"
    if 88 <= queue <= 103:
        return f"VXM[{queue - 88}]"
    if 104 <= queue <= 115:
        return f"MXM[{(queue - 104) // 3}].{('LW', 'ABC', 'ACC')[(queue - 104) % 3]}"
    if 116 <= queue <= 125 or 130 <= queue <= 139:
        side, k = ("W", queue - 116) if queue < 130 else ("E", queue - 130)
        for unit, base, n in (("DST", 0, 4), ("TSP", 4, 2), ("SHF", 6, 2), ("PRM", 8, 1), ("SEL", 9, 1)):
            if base <= k < base + n:
                return f"SXM.{side}.{unit}" + (f"[{k - base}]" if n > 1 else "")
    for first, name in ((126, "C2C.W.TX"), (128, "C2C.W.RX"), (140, "C2C.E.TX"), (142, "C2C.E.RX")):
        if first <= queue < first + 2:
            return f"{name}[{queue - first}]"
    raise ScheduleError(f"no name for queue {queue}")


AGT_FIELDS = ("countdown", "step", "streamstep", "iterations")


def agt_refs(p0, addr, stream, records, limit=100000):  # limit: cycles after p0 a loop may take
    """(cycle, address, stream) of every reference of an AGT loop whose reference 0 dispatches at p0 and whose
    configuration records [(field, value)] arrive on cycles p0 + 1, p0 + 2, ... (microarchitecture.md sec. 4):
    fields go countdown -> step -> streamstep -> iterations within a level, a field at or before the last one
    written starts the next level (at most 4); defaults 0, 0, 0, 2 (iterations and countdown below 1 count as 1
    and 0 respectively); a record applies before the reference of its own cycle; the next reference is made by the
    innermost level with references left, 1 + (countdowns of the wrapped levels) + (its countdown) cycles after
    the previous one, adding its step and stream step; with every level exhausted the loop waits while the
    configuration is open and ends once it is closed."""
    levels, last_field = [], None
    refs = [(p0, addr, stream)]
    counts = []
    a, s, last = addr, stream, p0
    close = p0 + 1 + len(records)
    c = p0
    while c < p0 + limit:
        c += 1
        if c < close:
            field, val = records[c - p0 - 1]
            fi = AGT_FIELDS.index(field)
            if not levels or last_field is None or fi <= last_field:
                if len(levels) == 4:
                    raise ScheduleError("a fifth AGT level is OTS-E-OPR-001")
                levels.append({"countdown": 0, "step": 0, "streamstep": 0, "iterations": 2})
                counts.append(1)
            levels[-1][field] = val
            last_field = fi
        k = next((i for i, lv in enumerate(levels) if counts[i] < max(1, lv["iterations"])), None)
        if k is None:
            if c >= close:
                return refs
            continue
        wait = 1 + sum(lv["countdown"] for lv in levels[:k]) + levels[k]["countdown"]
        if c < last + wait:
            continue
        for j in range(k):
            counts[j] = 1
        counts[k] += 1
        a += levels[k]["step"]
        s += levels[k]["streamstep"]
        last = c
        refs.append((c, a, s))
    raise ScheduleError("AGT loop does not end")
