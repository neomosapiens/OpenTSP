"""The geometry a generated program is written for, derived from the profile: lanes, tile, chunk, rounds, packing.

The kernels' shapes follow from a few facts of the profile (`superlanes`, `mxm.rows`, the MEM slices per hemisphere, the words per slice).
`Geometry.of(profile)` turns them into the sizes the generators use, so that no generator names a profile or writes the 320 lanes, the 256-lane
tile or the 128-slot chunk of `ref` into its text:

- `lanes`: the lanes of a vector; the unit that sums an MXM dot product has that many (it decides the sign of an exact zero sum).
- `tile_superlanes` and `tile`: the largest power of two of superlanes up to 16 and its lanes (256 on `ref`, 20 superlanes; 128 on 8
  superlanes). An activation of more features than a tile is several tiles; a power of two of superlanes is what the butterfly of a
  reduction needs, and 16 is the width the wide kernels were validated at, so a larger chip leaves the superlanes above the tile idle.
- `head_tile`: the lanes of an attention vector, two heads of 64 (128), or the tile when that is smaller.
- `chunk`: the slots of one score vector: the largest power of two up to 128 that the MXM can install as f16 K rows (`mxm.rows / 2`) and
  that the vector can hold (`lanes`): 128 on `ref`, 64 on `test/sl8`.
- `rounds`: the butterfly rounds over a tile (4 on `ref`); `chunk_rounds`: those over a chunk (3 on `ref`).
- `pack`: the vocabulary rows of the packed embedding and head tables per word, `lanes // 64` (5 on `ref`, 2 on 8 superlanes, 8 on 32).
- `install_rows`: the f16 rows one MXM plane holds (`mxm.rows / 2`, 160 on `ref`).
- `install_group(plane)`: the 16 MEM slices an LW reads: the plane's own hemisphere when it has 16 slices, else its slices and the first
  ones of the other hemisphere (a MEM slice writes both rows; the matrix input station reads one).

`Needs` states what a program family asks of a geometry as lower bounds and names the first one the profile does not meet with both numbers
("needs N words per slice, profile has M"). A profile larger than the program's own always fits."""
from dataclasses import dataclass

from opentsp_tools.stations import ScheduleError

SEG_LANES = 64            # lanes of one vocabulary row, of one head: the unit the packed tables and the attention are built from
TILE_SUPERLANES_MAX = 16  # the widest tile the wide kernels are written for
CHUNK_MAX = 128           # slots of a score vector the kernels are written for


class ProfileUnfit(ScheduleError):
    """The profile cannot hold the program: the message names the first unmet requirement and the profile's value."""


def pow2_floor(n):
    """The largest power of two not above n (n >= 1)."""
    assert n >= 1, n
    return 1 << (n.bit_length() - 1)


def log2(n):
    assert n >= 1 and n & (n - 1) == 0, f"{n} is not a power of two"
    return n.bit_length() - 1


@dataclass(frozen=True)
class Geometry:
    name: str
    superlanes: int
    mxm_rows: int
    slices_per_hemisphere: int
    words: int                  # words per MEM slice
    bank_bit: int

    @classmethod
    def of(cls, p):
        """The geometry of a profile object of either kind (`opentsp_tools.profile.Profile` or `layout.Profile`)."""
        if hasattr(p, "superlanes"):
            sl = p.superlanes
        else:
            sl, rem = divmod(p.vlanes, 16)
            assert rem == 0, f"{p.name}: {p.vlanes} lanes are not a whole number of superlanes"
        per = p.slices_per_hemisphere if hasattr(p, "slices_per_hemisphere") else p.mem_per_hemisphere
        words = p.words if hasattr(p, "words") else p.words_per_slice
        return cls(p.name, sl, p.mxm_rows, per, words, p.bank_bit)

    # -- lanes, tiles, chunks ----------------------------------------------------------------------------------------
    @property
    def lanes(self):
        return 16 * self.superlanes

    @property
    def tile_superlanes(self):
        return min(TILE_SUPERLANES_MAX, pow2_floor(self.superlanes))

    @property
    def tile(self):
        return 16 * self.tile_superlanes

    @property
    def head_tile(self):
        return min(self.tile, 2 * SEG_LANES)

    @property
    def rounds(self):
        return log2(self.tile_superlanes)

    @property
    def install_rows(self):
        return self.mxm_rows // 2

    @property
    def chunk(self):
        return min(CHUNK_MAX, pow2_floor(self.install_rows), pow2_floor(self.lanes))

    @property
    def chunk_rounds(self):
        return log2(self.chunk // 16) if self.chunk >= 16 else 0

    @property
    def pack(self):
        return self.lanes // SEG_LANES

    @property
    def bank_words(self):
        return 1 << self.bank_bit

    @property
    def quads(self):
        return self.slices_per_hemisphere // 4

    # -- the slices an install reads -----------------------------------------------------------------------------------
    def install_group(self, plane):
        """The 16 MEM slices an LW of MXM plane `plane` (0, 1 west; 2, 3 east) reads, in stream order: the plane's hemisphere first."""
        own, other = (0, 44) if plane < 2 else (44, 0)
        n = self.slices_per_hemisphere
        group = list(range(own, own + min(16, n)))
        if len(group) < 16:
            group += list(range(other, other + 16 - len(group)))
        return group


def of(p):
    return Geometry.of(p)


@dataclass(frozen=True)
class Need:
    """One lower bound: the profile has to offer at least `value` of `name`."""
    name: str                   # "superlanes", "mxm.rows", "words per slice", ...
    value: int
    unit: str
    why: str


def message(profile, what, need, has):
    """The refusal text, one place for every program family."""
    return (f"profile {profile} does not fit {what}: needs {need.value} {need.unit}, profile has {has} ({need.why}); "
            f"first unmet requirement: {need.name}")


def check_install(p, nlw, what):
    """Raise `ProfileUnfit` when `nlw` LW cycles (16 stream values each) do not fit the weight buffer of one MXM plane (`mxm.rows` stream
    values: 8 f16 rows per LW), with both numbers; `what` names the matrix."""
    if 16 * nlw > p.mxm_rows:
        raise ProfileUnfit(message(p.name, what, Need("mxm.rows", 16 * nlw, "MXM rows", f"{nlw} LW cycles of 16 stream values fill the weight buffer"),
                                   p.mxm_rows))


class Needs:
    """What a program family needs of a geometry: a list of (Need, getter(geometry)); the first one the profile misses is reported."""

    def __init__(self, what, rows):
        self.what, self.rows = what, rows

    def unmet(self, p):
        g = Geometry.of(p)
        for need, get in self.rows:
            has = get(g)
            if has < need.value:
                return need, has
        return None

    def check(self, p):
        miss = self.unmet(p)
        if miss is not None:
            raise ProfileUnfit(message(p.name, self.what, *miss))
        return p
