"""Patch sites: operand fields of generated program text
that the host rewrites between runs, so that one bucket program serves different MEM addresses or a different Permute
map. No opcode, length or cycle changes: a site is a numeric operand, which has a fixed width in the encoding.

Recording. An emitter marks an operand with `Decl(name, field, lo, hi, placeholder)`: the text carries the token
`@name` in the operand's place, `emit.Program.add(..., site=decl)` records a `Site` (the instruction's queue, cycle and
index among that queue's entries of the cycle), and `Program.text` writes the placeholder value in the token's
place, so a program with sites assembles to the text of the same program with literals.

Locating. `locate(view, p)` assembles the program twice, the sites all 0 and all ones (the field's width), and
requires the bits that differ between the two lowerings to be exactly the bits the records predict (the unit of the
recorded instruction from the lowering's own layout, the field's bits from encoding.json): a site that does not differ,
a difference that no site records, or a changed text length is a PatchError. When the all-zero or all-ones assembly puts
a read and a write of one bundle in one bank (OTS-E-MEM-001), a MEM address site takes its placeholder's bank bit with the
other bits at 0 and at all ones, and one more assembly flips the bank bit of every MEM site (sites whose flip would meet
a static reference of the bundle in one bank are left out of it): every bit of a field is then verified, the bank bit by
the flip. The result is a `Table`.

Applying. `apply(image, table, values)` returns a copy of the image with each site's field set, after range-checking
every value. The image is {queue: bytearray} of little-endian 32-bit text units. A field is addressed in the
instruction's 64-bit word: for a two-unit instruction the high 32 bits are its first unit, so a field may span both
(the Permute `map`, bits 33..30). `patches.npz` (`save`, `load`) holds the table: `format` (1), `names`, `queue`, `unit`
(of the instruction's first unit), `nunits`, `lo`, `width`, `vmin`, `vmax`, `placeholder`, `fixed` (the instruction's
fixed bits: opcode, tag).

Positions are in the lowering of the view as given. A paged program (opentsp_tools/pager.py) keeps its patch sites as tokens
through paging; `locate_paged` locates them in the whole text and `Paged` maps each site's units, with
`split_position`, into the initial image of its queue or one of its text blocks (a two-unit field may straddle a block
boundary: two blocks are then patched); `apply_paged` patches the images and the blocks.

The checked path. `apply` writes units and checks nothing but the ranges: a read moved into the bank of its parcel's write
gives a valid image that fails only when the model dispatches it (assembly.md, "Patching an assembled object"). For a program
whose sites are all in the text of its object (not paged: a site in a text block lives in a weight package, which `ots-asm
patch` does not touch) `asm_table` writes the located table in the format of `ots-asm patch` and `apply_object` patches an
assembled object with it: the toolchain command writes the same bits and re-runs the parcel checks (decode, slots, queue, bank
rule) on every parcel it touched, so a patch that makes an invalid parcel produces no object."""
import numbers
import os
import re
from dataclasses import dataclass

import numpy as np

from . import profile as tprofile, refasm

tsp_as = refasm.load()

FORMAT = 1
TOKEN = re.compile(r"@([A-Za-z0-9_.:\-]+)")
NARROW_BITS = 28         # a one-unit instruction keeps its tag in bits 30..28 and its bundle bit in bit 31
WIDE_BITS = 60           # a two-unit instruction keeps its tag in bits 63..60


class PatchError(Exception):
    pass


@dataclass(frozen=True)
class Decl:
    """What an emitter declares: the operand `field` (a variable name of the instruction in encoding.json), the range
    [lo, hi] of values the host may patch in, and the placeholder the text carries. A site of a grid of blocks (a KV
    pool's block addresses) also gives `stride` and `span`: a value v is accepted when (v - lo) % stride < span, so an
    address of another layer's words, or of the middle of a block, is refused. The default accepts every value.
    A MEM `addr` site stays in one bank unless `any_bank`: its values then lie in both banks (the embedding lookup
    reads its row in bank 0 or a zero word in bank 1), the tracker takes the reference as in both banks, and the
    range need only be inside the SRAM."""
    name: str
    field: str
    lo: int
    hi: int
    placeholder: int
    stride: int = 1
    span: int = 1
    any_bank: bool = False

    def __post_init__(self):
        if not re.fullmatch(TOKEN.pattern[1:], self.name):
            raise PatchError(f"site name {self.name!r} is not made of letters, digits and _ . : -")
        if self.lo >= self.hi:
            raise PatchError(f"site {self.name}: range [{self.lo}, {self.hi}] has fewer than two values")
        if not self.lo <= self.placeholder <= self.hi:
            raise PatchError(f"site {self.name}: placeholder {self.placeholder} outside its range [{self.lo}, {self.hi}]")
        if not 1 <= self.span <= self.stride:
            raise PatchError(f"site {self.name}: stride {self.stride} and span {self.span}")
        if (self.placeholder - self.lo) % self.stride >= self.span:
            raise PatchError(f"site {self.name}: placeholder {self.placeholder} is off the grid of stride {self.stride} "
                             f"from {self.lo}")

    @property
    def token(self):
        return "@" + self.name


@dataclass(frozen=True)
class Site:
    """A recorded site: instruction number `index` (source order) of `queue`'s entries at `cycle`, whose mnemonic is
    `op`."""
    name: str
    queue: int
    cycle: int
    index: int
    op: str
    field: str
    lo: int
    hi: int
    placeholder: int
    stride: int = 1
    span: int = 1


def record(prog, cycle, queue, text, decls):
    """Record the sites of one instruction about to be added to `prog` (an emit.Program) at `cycle` on the queue named
    `queue`; their tokens must be in `text`."""
    decls = [decls] if isinstance(decls, Decl) else list(decls)
    q = tsp_as.queue_id(queue)
    index = sum(1 for qq, _ in prog.bundles.get(cycle, []) if qq == queue)
    op = text.split()[0]
    for d in decls:
        try:
            field_info(op, d.field)
        except PatchError as e:
            raise PatchError(f"site {d.name}: {e}") from None
        if d.field == "addr":
            check_mem_range(prog.p, d)
        if d.token not in text:
            raise PatchError(f"site {d.name}: its token {d.token} is not in {text!r}")
        if any(s.name == d.name for s in prog.sites):
            raise PatchError(f"site {d.name} is recorded twice")
        prog.sites.append(Site(d.name, q, cycle, index, op, d.field, d.lo, d.hi, d.placeholder, d.stride, d.span))


def check_mem_range(p, d):
    """A MEM address site stays inside the SRAM and inside one bank (unless `any_bank`), so the bank of every patched
    reference is the placeholder's and the tracker's bank rule (OTS-E-MEM-001) holds for every value of the range."""
    if d.hi >= p.words:
        raise PatchError(f"site {d.name}: range [{d.lo:#x}, {d.hi:#x}] exceeds the {p.words} words of a slice")
    if d.any_bank:
        return
    banks = {p.bank(d.lo), p.bank(d.hi)} if d.hi - d.lo < 1 << p.bank_bit else {0, 1}
    if banks != {p.bank(d.placeholder)}:
        raise PatchError(f"site {d.name}: range [{d.lo:#x}, {d.hi:#x}] is not in one bank (bank_bit {p.bank_bit})")


def resolve(text, values):
    """`text` with each token replaced by its value (hex): `values` is {name: int}."""
    def sub(m):
        if m.group(1) not in values:
            raise PatchError(f"site token @{m.group(1)} has no record")
        return f"{values[m.group(1)]:#x}"
    return TOKEN.sub(sub, text)


# -- the fields of the encoding -----------------------------------------------------------------------------------
_fields = {}

# The fields a site may be: operands whose value selects data, never the length, the cycles or the configuration of
# the run. An AGT loop count, `config`, `setvl`, a thread bit or a Repeat count would change them.
MEM_ADDR = {("read", "addr"), ("write", "addr")}
ALLOWED = {("read", "addr"), ("write", "addr"), ("permute", "map"), ("distribute", "map"), ("loadmap", "map")}


def field_info(op, field):
    """(lo, width, fixed) of operand `field` of instruction `op`: its bits in the 64-bit word and the instruction's
    fixed bits (the match pattern's 0 and 1). The field must be in ALLOWED."""
    key = (op, field)
    if key not in ALLOWED:
        raise PatchError(f"{op} {field}: not a patch site (allowed: "
                         f"{', '.join(f'{o} {f}' for o, f in sorted(ALLOWED))})")
    if key not in _fields:
        try:
            e = tprofile.entry(op)
        except KeyError:
            raise PatchError(f"no instruction {op!r}") from None
        v = next((v for v in e["variables"] if v["name"] == field), None)
        if v is None:
            raise PatchError(f"{op} has no operand {field!r} (it has {[v['name'] for v in e['variables']]})")
        if "hi_location" in v or "top_location" in v:
            raise PatchError(f"{op} {field}: a field in two parts is not supported")
        kind = tprofile.encoding()["operand_kinds"][v["kind"]]
        if kind["syntax"] != "hex" or kind.get("signed"):
            raise PatchError(f"{op} {field}: kind {v['kind']} is not an unsigned number")
        hi, _, lo = v["location"].partition("-")
        hi, lo = int(hi), int(lo or hi)
        match = e["match"]
        fixed = sum(1 << (len(match) - 1 - i) for i, ch in enumerate(match) if ch in "01")
        _fields[key] = (lo, hi - lo + 1, fixed)
    return _fields[key]


# -- the table ------------------------------------------------------------------------------------------------------
COLUMNS = ("queue", "unit", "nunits", "lo", "width", "vmin", "vmax", "placeholder", "fixed")


class Table:
    """The located sites, one row each. `unit` is the index of the instruction's first unit in the queue's text, `lo`
    and `width` the field in the instruction's word, `vmin`, `vmax` its range and `fixed` the instruction's fixed
    bits. `stride` and `span` (default: one and one, every value of the range) as in Decl."""

    def __init__(self, names, queue, unit, nunits, lo, width, vmin, vmax, placeholder, fixed, stride=None, span=None):
        self.names = np.array(list(names), dtype=str)
        n = len(self.names)
        self.stride = np.array(np.ones(n) if stride is None else stride, dtype=np.int64)
        self.span = np.array(np.ones(n) if span is None else span, dtype=np.int64)
        cols = dict(zip(COLUMNS, (queue, unit, nunits, lo, width, vmin, vmax, placeholder, fixed)))
        for c, v in cols.items():
            setattr(self, c, np.array(v, dtype=np.uint64 if c == "fixed" else np.int64))
        self.index = {str(n): i for i, n in enumerate(self.names)}
        self.validate()

    def __len__(self):
        return len(self.names)

    def validate(self):
        if len(self.index) != len(self.names):
            raise PatchError("duplicate site names")
        for c in COLUMNS:
            if getattr(self, c).shape != (len(self),):
                raise PatchError(f"column {c} has the wrong length")
        used = {}
        for i, name in enumerate(self.names):
            n, lo, w = int(self.nunits[i]), int(self.lo[i]), int(self.width[i])
            if n not in (1, 2) or w < 1 or lo < 0:
                raise PatchError(f"site {name}: {n} units, field {w} bits from bit {lo}")
            limit = WIDE_BITS if n == 2 else NARROW_BITS
            if lo + w > limit:
                raise PatchError(f"site {name}: bits {lo}..{lo + w - 1} reach the tag of a {n}-unit instruction "
                                 f"(bit {limit} and above)")
            field = ((1 << w) - 1) << lo
            if field & int(self.fixed[i]):
                raise PatchError(f"site {name}: bits {lo}..{lo + w - 1} overlap the opcode of the instruction")
            if not 0 <= self.vmin[i] <= self.placeholder[i] <= self.vmax[i] < 1 << w:
                raise PatchError(f"site {name}: range [{self.vmin[i]}, {self.vmax[i]}] and placeholder "
                                 f"{self.placeholder[i]} do not fit {w} bits")
            if not 1 <= self.span[i] <= self.stride[i] or (self.placeholder[i] - self.vmin[i]) % self.stride[i] >= self.span[i]:
                raise PatchError(f"site {name}: stride {self.stride[i]}, span {self.span[i]} and placeholder "
                                 f"{self.placeholder[i]} from {self.vmin[i]}")
            for k in range(n):
                key = (int(self.queue[i]), int(self.unit[i]) + k)
                span = (field >> 32, field & 0xFFFFFFFF)[k] if n == 2 else field
                if used.get(key, 0) & span:
                    raise PatchError(f"site {name} overlaps another site")
                used[key] = used.get(key, 0) | span

    def records(self):
        """The sites as the `sites` list of a case file."""
        return [{"name": str(self.names[i]), "queue": int(self.queue[i]), "unit": int(self.unit[i]),
                 "nunits": int(self.nunits[i]), "lo": int(self.lo[i]), "width": int(self.width[i]),
                 "min": int(self.vmin[i]), "max": int(self.vmax[i]), "placeholder": int(self.placeholder[i]),
                 **({"stride": int(self.stride[i]), "span": int(self.span[i])} if self.stride[i] > 1 else {})}
                for i in range(len(self))]


def save(table, path):
    np.savez(path, format=np.int64(FORMAT), names=table.names, stride=table.stride, span=table.span,
             **{c: getattr(table, c) for c in COLUMNS})


def load(path):
    with np.load(path) as z:
        if int(z["format"]) != FORMAT:
            raise PatchError(f"{path}: patch table format {int(z['format'])}, expected {FORMAT}")
        grid = {c: z[c] for c in ("stride", "span") if c in z}
        return Table(list(z["names"]), *[z[c] for c in COLUMNS], **grid)


# -- lowering and locating -----------------------------------------------------------------------------------------
def _assemble(view, p, values=None):
    """({queue: units}, records) of the view with each site's value from `values` (default: its placeholder)."""
    prog = tsp_as.assemble(view.program(p).text(values), config=p.path)[1]
    return prog.images, prog.records


def lower(view, p, values=None):
    """{queue: [unit, ...]}: the lowering of the view with the sites set to `values` ({name: int}; a site not named
    keeps its placeholder)."""
    return _assemble(view, p, values)[0]


def _instructions(records, images):
    """{(queue, cycle, index): (first unit, units, word)} of the assembled program, by the lowering rule of
    tsp_as.lower, checked against the lowering itself."""
    by_q = {}
    for c, q, _, w, _ in records:
        by_q.setdefault(q, {}).setdefault(c, []).append(w)
    out = {}
    for q, cycles in by_q.items():
        units, prev = [], -1
        for c in sorted(cycles):
            ws = cycles[c]
            if len(ws) == 1 and tsp_as.is_nop_word(ws[0]):
                continue
            gap = c - prev - 1
            while gap > 0:
                n = min(gap, tsp_as.NOP_MAX)
                units.append(n)
                gap -= n
            for k, w in enumerate(ws):
                u = tsp_as.word_units(w)
                out[(q, c, k)] = (len(units), len(u), w)
                if k < len(ws) - 1:
                    u[0] |= tsp_as.BUNDLE_BIT
                units += u
            prev = c
        if units != images.get(q):
            raise PatchError(f"queue {q}: the layout of the assembler's records is not its lowering")
    return out


def _bits(unit, nunits, lo, width, mask):
    """{(unit, bit)} of the field's bits that are set in `mask`, in the queue's text: a two-unit instruction holds
    bits 63..32 in its first unit."""
    out = set()
    for b in range(lo, lo + width):
        if mask >> (b - lo) & 1:
            out.add((unit + (0 if b >= 32 else 1), b % 32) if nunits == 2 else (unit, b))
    return out


def _probe_max(s, width, p):
    """The largest value the placeholder assemblies give a site: all ones of its field, and for a MEM address the largest address of the
    profile (the field has 16 bits, an address needs sram.addr_bits of them)."""
    full = (1 << width) - 1
    return min(full, (1 << p.addr_bits) - 1) if (s.op, s.field) in MEM_ADDR else full


def locate(view, p):
    """The Table of the view's recorded sites, after the self-check described in the module docstring."""
    sites = view.sites
    names = [s.name for s in sites]
    if len(set(names)) != len(names):
        raise PatchError("a site name is recorded twice")
    tokens = {n for _, _, t in view.entries if "@" in t for n in TOKEN.findall(t)}
    for s in sites:
        if s.name not in tokens:
            raise PatchError(f"site {s.name} is recorded, but its token is not in the text")
    info = {}
    for s in sites:
        lo, width, fixed = _site_info(s)
        if (s.op, s.field) in MEM_ADDR and s.hi >= 1 << p.addr_bits:
            raise PatchError(f"site {s.name}: range [{s.lo}, {s.hi}] does not fit the {p.addr_bits}-bit SRAM address of the profile")
        info[s.name] = (lo, width, fixed)
    pairs = {s.name: (0, _probe_max(s, info[s.name][1], p)) for s in sites}
    flips = None
    try:
        ia, ib = (_assemble(view, p, {n: v[k] for n, v in pairs.items()})[0] for k in (0, 1))
    except tsp_as.AsmError as e:
        if e.code != "OTS-E-MEM-001":
            raise
        # Every field at 0 or all ones put a read and a write of one bundle in one bank: a MEM address keeps its
        # placeholder's bank bit, with every other bit at 0 and then at all ones, and the bank bits are verified by one
        # more assembly with the bank of every MEM site flipped (a read and a write then still differ in bank).
        bank = 1 << p.bank_bit
        mem = [s for s in sites if (s.op, s.field) in MEM_ADDR]
        for s in mem:
            full, keep = _probe_max(s, info[s.name][1], p), s.placeholder & bank
            pairs[s.name] = (keep, (full & ~bank) | keep)
        ia, ib = (_assemble(view, p, {n: v[k] for n, v in pairs.items()})[0] for k in (0, 1))
        flips = _flipped_banks(view, p, mem, bank)
    base_images, base_records = _assemble(view, p)
    where = _instructions(base_records, base_images)
    for q in sorted(set(ia) | set(ib)):
        if len(ia.get(q, [])) != len(ib.get(q, [])):
            raise PatchError(f"queue {q}: the text length depends on the placeholder values")
    observed = {(q, u, bit) for q in ia for u, (x, y) in enumerate(zip(ia[q], ib[q])) for bit in range(32)
                if (x ^ y) >> bit & 1}
    predicted, rows = {}, []
    for s in sites:
        lo, width, fixed = info[s.name]
        pos = where.get((s.queue, s.cycle, s.index))
        if pos is None:
            raise PatchError(f"site {s.name}: no instruction at queue {s.queue}, cycle {s.cycle}, index {s.index}")
        unit, n, word = pos
        if lo + width > (WIDE_BITS if n == 2 else NARROW_BITS):
            raise PatchError(f"site {s.name}: the {s.op} {s.field} field does not fit the {n}-unit instruction there")
        if (word >> lo) & ((1 << width) - 1) != s.placeholder:
            raise PatchError(f"site {s.name}: the instruction at queue {s.queue}, cycle {s.cycle} carries "
                             f"{(word >> lo) & ((1 << width) - 1):#x} in bits {lo}..{lo + width - 1}, "
                             f"not the placeholder {s.placeholder:#x}")
        for u, bit in _bits(unit, n, lo, width, pairs[s.name][0] ^ pairs[s.name][1]):
            predicted[(s.queue, u, bit)] = s.name
        rows.append((s.name, s.queue, unit, n, lo, width, s.lo, s.hi, s.placeholder, fixed, s.stride, s.span))
    if flips is not None:
        seen = {(q, u, bit) for q in flips[0] for u, (x, y) in enumerate(zip(base_images[q], flips[0][q]))
                for bit in range(32) if (x ^ y) >> bit & 1}
        want = {}
        for s in flips[1]:
            lo, width, _ = info[s.name]
            unit, n, _ = where[(s.queue, s.cycle, s.index)]
            want.update({(s.queue, u, bit): s.name for u, bit in _bits(unit, n, lo, width, 1 << p.bank_bit)})
        wrong = sorted((want.get(k, "?"), k) for k in seen ^ set(want))
        if wrong:
            raise PatchError(f"site {wrong[0][0]}: the bank bit does not differ between the placeholder assembly and the "
                             f"one with every MEM site's bank flipped, at (queue, unit, bit) {wrong[0][1]} "
                             f"({len(wrong)} bits)")
    missing = sorted((v, k) for k, v in predicted.items() if k not in observed)
    if missing:
        raise PatchError(f"site {missing[0][0]} does not differ between placeholder assemblies at "
                         f"(queue, unit, bit) {missing[0][1]} ({len(missing)} bits of the sites)")
    extra = sorted(observed - set(predicted))
    if extra:
        raise PatchError(f"{len(extra)} bits differ between placeholder assemblies that no site records, the first at "
                         f"(queue, unit, bit) {extra[0]}")
    return Table(*[[r[i] for r in rows] for i in range(10)], stride=[r[10] for r in rows], span=[r[11] for r in rows])


def _flipped_banks(view, p, mem, bank):
    """(lowering of the view with the bank bit of every MEM site in `mem` flipped, the sites whose flip is in it): all at
    once, which keeps a read and a write of one bundle in different banks; when a static reference of a bundle then meets
    a flipped site in one bank (OTS-E-MEM-001) the flipped assembly is made without the sites that cannot flip."""
    todo = list(mem)
    while todo:
        try:
            return _assemble(view, p, {s.name: s.placeholder ^ bank for s in todo})[0], todo
        except tsp_as.AsmError as e:
            if e.code != "OTS-E-MEM-001":
                raise
            todo = todo[:-1]
    return _assemble(view, p)[0], []


def _site_info(s):
    try:
        lo, width, fixed = field_info(s.op, s.field)
    except PatchError as e:
        raise PatchError(f"site {s.name}: {e}") from None
    if not 0 <= s.lo <= s.placeholder <= s.hi < 1 << width:
        raise PatchError(f"site {s.name}: range [{s.lo}, {s.hi}] with placeholder {s.placeholder} does not fit the "
                         f"{width}-bit {s.field} field")
    return lo, width, fixed


# -- the host patcher ----------------------------------------------------------------------------------------------
def to_image(units):
    """{queue: bytearray}: the little-endian bytes of {queue: [unit, ...]}."""
    return {q: bytearray(b"".join(u.to_bytes(4, "little") for u in us)) for q, us in sorted(units.items())}


def to_units(image):
    return {q: [int.from_bytes(b[k:k + 4], "little") for k in range(0, len(b), 4)] for q, b in image.items()}


def apply(image, table, values, strict=True):
    """A copy of `image` with the fields of the sites named in `values` ({name: int}) set. Every value is checked first:
    an unknown name, a value outside the site's range, and (with `strict`) a site without a value are refused with
    the site's name, and nothing is changed."""
    unknown = sorted(n for n in values if n not in table.index)
    if unknown:
        raise PatchError(f"unknown patch sites {unknown[:5]}")
    if strict:
        absent = [str(n) for n in table.names if n not in values]
        if absent:
            raise PatchError(f"no value for patch sites {absent[:5]}" + (f" and {len(absent) - 5} more"
                                                                         if len(absent) > 5 else ""))
    for n, v in values.items():
        i = table.index[n]
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise PatchError(f"site {n}: value {v!r} is not an integer")
        if not table.vmin[i] <= v <= table.vmax[i]:
            raise PatchError(f"site {n}: value {int(v)} outside [{int(table.vmin[i])}, {int(table.vmax[i])}]")
        if (int(v) - int(table.vmin[i])) % int(table.stride[i]) >= int(table.span[i]):
            raise PatchError(f"site {n}: value {int(v):#x} is not on the grid of stride {int(table.stride[i])} from "
                             f"{int(table.vmin[i]):#x}")
    out = {q: bytearray(b) for q, b in image.items()}
    for n, v in values.items():
        i = table.index[n]
        q, u, k = int(table.queue[i]), int(table.unit[i]), int(table.nunits[i])
        lo, w = int(table.lo[i]), int(table.width[i])
        b = out[q]
        word = int.from_bytes(b[4 * u:4 * u + 4], "little")
        if k == 2:
            word = (word << 32) | int.from_bytes(b[4 * u + 4:4 * u + 8], "little")
        new = (word & ~(((1 << w) - 1) << lo)) | (int(v) << lo)
        if (new ^ word) & int(table.fixed[i]) or (new ^ word) >> (WIDE_BITS if k == 2 else NARROW_BITS):
            raise PatchError(f"site {n}: the patch would change the opcode or the length of the instruction")
        if k == 2:
            b[4 * u:4 * u + 4] = (new >> 32).to_bytes(4, "little")
            b[4 * u + 4:4 * u + 8] = (new & 0xFFFFFFFF).to_bytes(4, "little")
        else:
            b[4 * u:4 * u + 4] = new.to_bytes(4, "little")
    return out


def split_position(unit, image_len, block_units):
    """(None, unit) when the unit of a paged queue's logical text is in its initial image of `image_len` units, else
    (block number, unit in the block) for blocks of `block_units` units."""
    if unit < image_len:
        return None, unit
    return (unit - image_len) // block_units, (unit - image_len) % block_units


# -- paged programs ------------------------------------------------------------------------------------------------
@dataclass
class Paged:
    """The sites of a paged program: `table` holds the positions in each queue's whole text (initial image, then the
    blocks), `image_len[q]` the units of paged queue q that stay in its initial image (a queue not listed is not paged),
    `block_units` the units of a block."""
    table: Table
    image_len: dict
    block_units: int

    def where(self, i):
        """[(block, unit in the image or block)] of the units of site i: block None is the initial image."""
        q, u = int(self.table.queue[i]), int(self.table.unit[i])
        out = []
        for k in range(int(self.table.nunits[i])):
            if q in self.image_len:
                out.append(split_position(u + k, self.image_len[q], self.block_units))
            else:
                out.append((None, u + k))
        return out


def locate_paged(full, p, image_len, block_units):
    """The Paged table of `full`, the whole text of a paged program (every entry, the cut ones and the fetches
    included), with the self-check of `locate` on its whole lowering."""
    return Paged(locate(full, p), dict(image_len), block_units)


def apply_paged(images, blocks, paged, values, strict=True):
    """(images, blocks) copies with the sites of `values` set: `images` {queue: bytearray}, `blocks` {(queue, j):
    bytearray of one block, padding included}. Checks as `apply`."""
    logical = {q: bytearray(b) for q, b in images.items()}
    by_q = {}
    for (q, j), b in blocks.items():
        by_q.setdefault(q, {})[j] = b
    for q, bl in by_q.items():
        if q not in paged.image_len or len(images[q]) != 4 * paged.image_len[q]:
            raise PatchError(f"queue {q}: its initial image is not the one the sites were located in")
        for j in sorted(bl):
            if len(bl[j]) != 4 * paged.block_units:
                raise PatchError(f"queue {q} block {j}: {len(bl[j]) // 4} units, not {paged.block_units}")
            logical[q] += bl[j]
    out = apply(logical, paged.table, values, strict)
    new_images, new_blocks = {}, {}
    for q, b in out.items():
        n = 4 * paged.image_len[q] if q in by_q else len(b)
        new_images[q] = bytearray(b[:n])
        for j in sorted(by_q.get(q, {})):
            lo = n + j * 4 * paged.block_units
            new_blocks[(q, j)] = bytearray(b[lo:lo + 4 * paged.block_units])
    return new_images, new_blocks


# -- the checked path: `ots-asm patch` ------------------------------------------------------------------------------
def asm_table(table, sites):
    """The patch table of `ots-asm patch` (opentsp-sw-toolchain/README.md, "Patching an object") for the located `table`:
    `sites` are the recorded Sites of the view it was located in, which give each site's mnemonic and operand name; the
    operand's kind comes from the encoding. A site of a paged program is outside the object: the table of its
    initial images only is what the command can patch."""
    by_name = {s.name: s for s in sites}
    rows = []
    for i, name in enumerate(table.names):
        s = by_name[str(name)]
        kind = next(v["kind"] for v in tprofile.entry(s.op)["variables"] if v["name"] == s.field)
        row = {"name": str(name), "queue": int(table.queue[i]), "unit": int(table.unit[i]), "op": s.op, "field": s.field,
               "kind": kind, "min": int(table.vmin[i]), "max": int(table.vmax[i])}
        if table.stride[i] > 1:
            row["stride"], row["span"] = int(table.stride[i]), int(table.span[i])
        rows.append(row)
    return {"format": 1, "sites": rows}


def apply_object(obj, table, sites, values, p, asm=None):
    """The bytes of the object `obj` (bytes of an `.elf`) with the sites of `values` ({name: int}) patched by
    `ots-asm patch` (`asm`: the executable, default $OTS_ASM or target/release/ots-asm): the values are range-checked and
    every parcel that holds a patched instruction is checked again. A refusal is a PatchError carrying the command's
    diagnostic (OTS-E-MEM-001 for a read moved into the bank of its write)."""
    import json
    import subprocess
    import tempfile
    asm = asm or os.environ.get("OTS_ASM") or os.path.join(tprofile.ROOT, "target", "release", "ots-asm")
    with tempfile.TemporaryDirectory(prefix="ots-patch-") as d:
        paths = {n: os.path.join(d, n) for n in ("in.elf", "table.json", "values.json", "out.elf")}
        with open(paths["in.elf"], "wb") as f:
            f.write(obj)
        with open(paths["table.json"], "w") as f:
            json.dump(asm_table(table, sites), f)
        with open(paths["values.json"], "w") as f:
            json.dump({n: int(v) for n, v in values.items()}, f)
        r = subprocess.run([asm, "patch", "--config", p.path, "--table", paths["table.json"], "--values",
                            paths["values.json"], "-o", paths["out.elf"], paths["in.elf"]], capture_output=True, text=True)
        if r.returncode != 0:
            raise PatchError(r.stderr.strip() or f"ots-asm patch exited {r.returncode}")
        with open(paths["out.elf"], "rb") as f:
            return f.read()
