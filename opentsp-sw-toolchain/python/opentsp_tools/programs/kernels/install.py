"""install: a weight matrix into an MXM plane : `nlw` LW cycles, one stream per MEM slice of the
plane's hemisphere (16 slices, slice j holds word `addr + i` for LW i), then the IW once the last LW has committed."""
import numpy as np

from opentsp_tools.programs import data
from opentsp_tools.programs import geometry
from opentsp_tools.stations import ScheduleError

ADDR_W = 0x10        # first word of a matrix slice: row 8 i + j div 2 is word ADDR_W + i of slice j


def plane_queues(plane):
    """(first slice of the hemisphere, stream row, LW queue) of MXM plane 0..3: the LW, ABC and ACC queues are
    consecutive."""
    if not 0 <= plane < 4:
        raise ScheduleError(f"no MXM plane {plane}")
    return (0, "W", 104 + 3 * plane) if plane < 2 else (44, "E", 104 + 3 * plane)


def group(p, plane):
    """The 16 MEM slices an LW of `plane` reads, in stream order: slice j of the group holds plane j mod 2 of the rows 8 i + j div 2.
    The plane's own hemisphere when it has 16 slices, else its slices and the first ones of the other hemisphere
    (`geometry.Geometry.install_group`); on `ref` the 16 slices from `plane_queues(plane)[0]`."""
    return geometry.Geometry.of(p).install_group(plane)


def install(k, wtype, nlw, weighted, addr, t0=30, plane=0, prefix="w", text=None):
    """LW i dispatches at t0 + i on the plane's LW queue and takes word `addr + i` of each of the 16 slices of the group (named
    `<prefix>j` when `weighted`: the matrix is bound from a weight package, else preloaded; `text(j, i)` names word i
    of slice j otherwise). Returns the cycle of the first ABC the installed matrix serves."""
    b, p = k.b, k.p
    geometry.check_install(p, nlw, f"the matrix of {nlw * 8} f16 rows")
    _, row, lw = plane_queues(plane)
    slices = group(p, plane)
    # a slice of the other hemisphere is further from the matrix input: its read leaves earlier than the LW by its extra stations
    t0 += max(0, -min(b.read_cycle_for(t0, "lw", lw, s, row) for s in slices))
    for i in range(nlw):
        vs = [b.read_for(t0 + i, "lw", lw, [slices[j]], [addr + i], row, j,
                         addr_text=[text(j, i) if text else f"{prefix}{j}+{i:#x}"] if weighted else None)
              for j in range(16)]
        b.unit(lw, "lw", f"lw.{wtype} {row}:0", vs)
    t_iw = t0 + nlw - 1 + p.timing("lw")[0]
    b.unit(lw, "iw", f"iw.{wtype}", cycle=t_iw)
    return t_iw + p.timing("iw")[0]


def weight_planes(w, vlanes):
    """The 16 slices' words of an f16 matrix w (rows x columns, a multiple of 8 rows; the columns beyond w's are
    zero): slice j holds plane j mod 2 (low or high byte) of rows 8 i + j div 2 at word i (the layout of LW)."""
    padded = np.zeros((len(w), vlanes), np.float16)
    padded[:, :w.shape[1]] = w
    planes = [data.to_planes(row) for row in padded]
    return [[planes[8 * i + j // 2][j % 2] for i in range(len(w) // 8)] for j in range(16)]


def bind_weights(k, w, plane=0, prefix="w"):
    """An f16 matrix as one extern per slice of the plane's hemisphere (`<prefix>j`, one word per 8 rows at ADDR_W),
    bound from the case's weight package."""
    slices = group(k.p, plane)
    for j, words in enumerate(weight_planes(w, k.p.vlanes)):
        k.b.prog.extern(f"{prefix}{j}", slices[j], ADDR_W, len(words))
        k.weights[f"{prefix}{j}"] = ("raw", b"".join(words))
