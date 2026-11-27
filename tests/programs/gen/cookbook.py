#!/usr/bin/env python3
"""The generated blocks of 

usage: cookbook.py [--update] [--doc PATH]

Every block of the document between `<!-- cookbook:NAME -->` and `<!-- /cookbook:NAME -->` is produced here: the cycle-annotated
listings from the programs the toolchain kernels emit on `ref` (the text of the program, so a change of a kernel or of the
profile changes the listing), and the division tables from the ISS measurement (division_domains.json) and from the numpy
twin. `test_cookbook.py` regenerates every block and compares it with the document, so no number in the document can drift
from the code. Without options the blocks are checked (exit 1 and a diff on any difference); `--update` rewrites them."""
import argparse
import difflib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import division  # noqa: E402
import division_domains  # noqa: E402
import model  # noqa: E402
from opentsp_tools import pager
from opentsp_tools import profile as tprofile
from opentsp_tools.programs.kernels import reduce_bcast as k_rb, softmax as k_softmax
from opentsp_tools.programs.kernels.flow import Flow

DOC = os.path.join(tprofile.ROOT, "docs", "kernel-cookbook.md")
BUNDLE = re.compile(r"^bundle @cycle=(\d+) \{$")
LINE = re.compile(r"^  ([A-Za-z0-9_.\[\]]+): (.*)$")


def instructions(text):
    """[(cycle, queue, instruction)] of a program text, in the order of the text."""
    out, cycle = [], None
    for ln in text.splitlines():
        m = BUNDLE.match(ln)
        if m:
            cycle = int(m.group(1))
            continue
        m = LINE.match(ln)
        if m and cycle is not None:
            out.append((cycle, m.group(1), m.group(2)))
    return out


def timing_of(p, instr):
    """(mnemonic, d_func, d_skew) of an instruction text, or None when the mnemonic has no timing row."""
    head = instr.split(" ", 1)[0]                      # the mnemonic with its type, as the profile's typed rows are keyed
    try:
        return (head.split(".", 1)[0],) + tuple(p.timing(head))
    except KeyError:                                   # no such mnemonic; an entry with typed rows asked without a type is UntypedTiming, not this
        return None


def listing(p, text, notes, hide, head=None, since=None):
    """The annotated listing of the program `text`: one line per instruction that no `hide` rule removes, with its cycle,
    the cycles since the previous listed line, the instruction and the first of `notes` ((regex of 'QUEUE: instruction',
    note)) that matches; a note may contain {d_func} and {d_skew} of the instruction's mnemonic. Consecutive hidden
    instructions collapse into one summary line (the `hide` rule's label and count). Every listed line must have a note: a
    kernel that emits an instruction the notes do not know fails here, not silently."""
    rows = [f"{'cycle':>5} {'+dt':>4}  {'queue: instruction':<50} ; meaning"]
    pending, last, run = {}, None, None      # run: [note, first row index, cycles] of equal consecutive notes

    def end_run():
        nonlocal run
        if run and len(run[2]) > 1:
            more = run[2][1:]
            when = "same cycle" if set(more) == {run[2][0]} else "cycles " + ", ".join(map(str, more))
            rows.append(f"      ... {len(more)} more, same meaning ({when})")
        run = None

    def flush():
        for label, cycles in pending.items():
            at = f"cycle {cycles[0]}" if len(set(cycles)) == 1 else f"cycles {min(cycles)} to {max(cycles)}"
            rows.append(f"  ... {len(cycles)} {label} ({at})")
        pending.clear()
    for cycle, queue, instr in instructions(text):
        if since is not None and cycle < since:
            continue
        line = f"{queue}: {instr}"
        label = next((lab for rx, lab in hide if re.match(rx, line)), None)
        if label:
            end_run()
            pending.setdefault(label, []).append(cycle)
            continue
        if pending:
            end_run()
        flush()
        note = next((n for rx, n in notes if re.match(rx, line)), None)
        if note is None:
            raise ValueError(f"no note for the instruction '{line}' at cycle {cycle}")
        t = timing_of(p, instr)
        fields = {"d_func": t[1], "d_skew": t[2]} if t else {}
        note = note.format(**fields)
        if run and run[0] == note:
            run[2].append(cycle)
            last = cycle
            continue
        end_run()
        delta = "" if last is None else f"+{cycle - last}"
        rows.append(f"{cycle:>5} {delta:>4}  {line:<50} ; {note}")
        run = [note, len(rows) - 1, [cycle]]
        last = cycle
    end_run()
    flush()
    return "\n".join(rows) + "\n"


def code(body, lang=""):
    return f"```{lang}\n{body}```\n"


# -- division ----------------------------------------------------------------------------------------------------------

DIV_HIDE = [(r"MEM\[\d+\]: read", "MEM reads of the operands"), (r"MEM\[\d+\]: write", "MEM writes of the result")]
DIV_NOTES = [
    (r"VXM\[\d+\]: mul\.f32 E:8, E:16, E:16", "x*x: both operands are the one stream; rounded to f32 (overflow gives inf, "
     "underflow a subnormal or 0); d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[\d+\]: rsqrt\.f32 E:12, E:8\.chain", "rsqrt of the chained product, one cycle after it commits: 1/|x|; d_func={d_func}"),
    (r"VXM\[\d+\]: rsqrt\.f32 E:8, E:16", "rsqrt(x): 1/sqrt(x), NaN for x < 0; d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[\d+\]: mul\.f32 E:12, E:8\.chain, E:8\.chain", "the chained result twice, squared: 1/x for x > 0; d_func={d_func}"),
    (r"VXM\[\d+\]: mul\.f32 E:8, E:16, E:20", "m = x*y0 (y0 the seed, a second operand stream); d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[\d+\]: sub\.f32 E:12, E:24, E:8\.chain", "2 - m: the constant 2 is read to meet it, m is chained; d_func={d_func}"),
    (r"VXM\[\d+\]: mul\.f32 E:8, E:28, E:12\.chain", "y1 = y0*(2 - m): the seed read again to meet it; d_func={d_func}"),
]


def division_listing(idiom):
    p = tprofile.load("ref")
    x = np.linspace(1.0, 100.0, p.vlanes, dtype=np.float32)
    k, _ = division.program(p, "cookbook/division", [(idiom, x)], "cookbook")
    text, _ = k.done("cookbook")
    return code(listing(p, text, DIV_NOTES, DIV_HIDE, since=0), "text")


def division_table():
    with open(division.DOMAINS) as f:
        return division_domains.render_table(json.load(f))


def fmt32(v):
    return f"{float(v):.8g}"


def division_specials():
    """The results on special values, from the numpy twin (the ISS and tsp_sim agree: vxm/division_edges)."""
    x = division.fill(division.SPECIAL_BITS, len(division.SPECIAL_BITS))
    rows = ["| x | " + " | ".join(f"`{i}`" for i in division.IDIOMS) + " |", "|---|" + "---|" * len(division.IDIOMS)]
    res = {i: division.twin(i, x) for i in division.IDIOMS}
    for j, v in enumerate(x):
        rows.append(f"| {fmt32(v)} | " + " | ".join(fmt32(res[i][j]) for i in division.IDIOMS) + " |")
    return "\n".join(rows) + "\n"


# -- the kernels -------------------------------------------------------------------------------------------------------

MATVEC_HIDE = [(r"MEM\[\d+\]: read w\d+\+", "weight reads from the 16 slices of the matrix (the install)")]
MATVEC_NOTES = [
    (r"MEM\[\d+\]: read xp", "read the f32 activation x (four byte planes) to meet the VXM narrowing"),
    (r"MXM\[0\]\.LW: lw", "load weights into the MXM buffer, one LW per cycle, from the weight words read earlier"),
    (r"VXM\[0\]: conv_f32_f16\.f32", "narrow x to f16 on the VXM, result on row W towards the MXM; d_func={d_func} d_skew={d_skew}"),
    (r"MXM\[0\]\.LW: iw", "install the weights: IW makes the loaded matrix the active one"),
    (r"MXM\[0\]\.ABC: abc", "activation broadcast and compute: the f16 vector against the 64 weight rows; d_func={d_func} d_skew={d_skew}"),
    (r"MXM\[0\]\.ACC: acc", "accumulate: the exact dot products rounded once to f32 onto row E, lane l = row l; d_func={d_func}"),
    (r"MEM\[\d+\]: read bp", "read the f32 bias to meet the add"),
    (r"VXM\[1\]: add\.f32", "y = W x + b in f32 (the matrix result and the bias meet on the VXM); d_func={d_func} d_skew={d_skew}"),
    (r"MEM\[\d+\]: write", "store y (four byte planes) in the east quad"),
]


def matvec_listing():
    p = tprofile.load("ref")
    text, _ = model.matvec_f16(p)
    return code(listing(p, text, MATVEC_NOTES, MATVEC_HIDE), "text")


KERNEL_HIDE = [(r"MEM\[\d+\]: (read|write)", "MEM reads and writes"),
               (r"SXM\.[A-Z]\.(PRM|DST\[\d\]): loadmap", "map loads"),
               (r"SXM\.W\.SEL: sxselect", "select moves out of the SXM")]
RB_NOTES = [
    (r"VXM\[0\]: reduce\.add\.f32 E:8, E:0\b", "per-superlane sum of the 16 lanes of the vector read; d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[0\]: reduce\.add\.f32 E:8, E:24\.chain", "per-superlane sum of the squares, chained from the multiply; d_func={d_func}"),
    (r"VXM\[0\]: reduce\.max\.f32", "per-superlane maximum of the 16 lanes; d_func={d_func} d_skew={d_skew}"),
    (r"SXM\.W\.PRM: permute W:\d+, 0x1", "first butterfly stage: the superlane totals rotate by 2 superlanes (one byte plane per instruction); d_func={d_func}"),
    (r"SXM\.W\.PRM: permute W:\d+, 0x2", "second stage: rotate by 1 superlane; d_func={d_func}"),
    (r"SXM\.W\.DST\[\d\]: distribute", "distribute: the segment's total reaches every superlane of the segment; d_func={d_func}"),
    (r"VXM\[[12]\]: (add|max)\.f32", "combine: this round's sum or maximum with the rotated copy; d_func={d_func} d_skew={d_skew}"),
]
LN_NOTES = RB_NOTES + [
    (r"VXM\[2\]: mul\.f32 E:12, E:4, E:8", "mean = total * (1/n) (the total comes back from the SXM as a vector in MEM); d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[3\]: sub\.f32", "centred = x - mean (mean chained); d_func={d_func}"),
    (r"VXM\[4\]: mul\.f32 E:24, W:0, W:0", "centred squared (centred was stored and is read twice as one stream); d_func={d_func}"),
    (r"VXM\[5\]: mul\.f32", "variance = sum of squares * (1/n); d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[6\]: add\.f32", "variance + epsilon (chained); d_func={d_func}"),
    (r"VXM\[7\]: rsqrt\.f32", "rsqrt: the inverse standard deviation (chained); d_func={d_func}"),
    (r"VXM\[8\]: mul\.f32", "centred * rsqrt (centred read to meet it); d_func={d_func}"),
    (r"VXM\[9\]: mul\.f32", "* gamma (chained); d_func={d_func}"),
    (r"VXM\[10\]: add\.f32", "+ beta (chained): the LayerNorm result; d_func={d_func}"),
]
SM_NOTES = RB_NOTES + [
    (r"VXM\[0\]: reduce\.max\.f32", "per-superlane maximum of the 16 lanes of the scores read; d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[0\]: reduce\.add\.f32 E:8, E:16\.chain", "per-superlane sum of the exponentials, chained from exp; d_func={d_func}"),
    (r"VXM\[3\]: sub\.f32", "s - max (the broadcast maximum, a vector read to meet it); d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[4\]: exp\.f32", "exp (chained): the exponentials, kept in MEM; d_func={d_func}"),
    (r"VXM\[5\]: mul\.f32", "total*total (the broadcast sum of the exponentials); d_func={d_func} d_skew={d_skew}"),
    (r"VXM\[6\]: rsqrt\.f32", "rsqrt of the square: 1/total (chained); total is in [1, n], inside the domain of rsqrt_sq; d_func={d_func}"),
    (r"VXM\[7\]: mul\.f32", "exp * (1/total): the softmax (the kept exponentials read to meet it); d_func={d_func} d_skew={d_skew}"),
]


def layernorm_listing():
    p = tprofile.load("ref")
    text, _ = model.layernorm(p)
    return code(listing(p, text, LN_NOTES, KERNEL_HIDE), "text")


def softmax_one(p, seed=1):
    k = model.kit(p, "softmax1", "softmax over 64-lane segments, one vector, f32", seed)
    maps = (k_rb.load_maps(k, 60), k_rb.load_distribute_map(k, 80))
    f = Flow(k)
    s = model.lanes(k, p.vlanes, 3)
    v = k_softmax.softmax(f, f.const("s0", s, "W", 1), maps, 0x10, (1 << p.bank_bit) + 0x40, e_quad=1, at=k.last)
    f.store(v, 3, s * 0)           # the expectation is not used here: the program is only listed
    return k.done("softmax1")


def softmax_listing():
    p = tprofile.load("ref")
    text, _ = softmax_one(p)
    return code(listing(p, text, SM_NOTES, KERNEL_HIDE), "text")


GEOMETRY_PROFILES = ["ref", "test/ref_slow", "test/sl32", "test/sl8", "test/ref_small", "test/mid_min", "mini"]


def geometry_table():
    """The sizes the serving generators derive from each profile (`opentsp_tools.programs.geometry`, `layout.Shape`) and what the batched serving step does
    with it: the capability check's verdict, from the code."""
    sys.path.insert(0, os.path.join(tprofile.ROOT, "tests", "showcase", "tinystories"))
    from opentsp_vllm.models.tinystories import layout
    from opentsp_tools.programs import geometry
    from opentsp_tools.programs import needs
    rows = ["| profile | lanes | tile | chunk | pack | head rows | model | batched serving step |", "|---|---|---|---|---|---|---|---|"]
    for name in GEOMETRY_PROFILES:
        p = tprofile.load(name)
        g, lp = geometry.of(p), layout.load_profile(name)
        cfg = layout.serve_model(lp, 2)
        sh = layout.Shape.of(lp, cfg)
        try:
            needs.SERVE.check(p)
            needs.model_needs(cfg).check(p)
            verdict = "fits"
        except geometry.ProfileUnfit as e:
            msg = str(e)
            verdict = "refused: " + msg[msg.index("needs "):msg.index(" (")]
        model = "2-layer test model" if cfg.vocab == 50257 else f"small model ({cfg.vocab} tokens)"
        rows.append(f"| `{name}` | {g.lanes} | {g.tile} | {g.chunk} | {sh.pack} | {sh.head_rows} | {model} | {verdict} |")
    return "\n".join(rows) + "\n"


def pager_constraints():
    """The constraints section of the pager's own documentation (opentsp_tools/pager.py), word for word."""
    doc = pager.__doc__
    return code(doc[doc.index("Constraints on the program."):].rstrip() + "\n", "text")


BLOCKS = {
    "geometry-table": geometry_table,
    "pager-constraints": pager_constraints,
    "division-table": division_table,
    "division-specials": division_specials,
    "listing-rsqrt-sq": lambda: division_listing("rsqrt_sq"),
    "listing-rsqrt-pair": lambda: division_listing("rsqrt_pair"),
    "listing-newton": lambda: division_listing("newton_sq"),
    "listing-matvec": matvec_listing,
    "listing-layernorm": layernorm_listing,
    "listing-softmax": softmax_listing,
}


def blocks_in(doc):
    return {m.group(1): m for m in re.finditer(r"<!-- cookbook:([a-z-]+) -->\n(.*?)<!-- /cookbook:\1 -->", doc, re.S)}


def render(doc):
    """The document with every block regenerated; the names must be the BLOCKS exactly."""
    found = blocks_in(doc)
    if set(found) != set(BLOCKS):
        raise ValueError(f"blocks in the document {sorted(found)} differ from the generators {sorted(BLOCKS)}")
    return re.sub(r"<!-- cookbook:([a-z-]+) -->\n(.*?)<!-- /cookbook:\1 -->",
                  lambda m: f"<!-- cookbook:{m.group(1)} -->\n{BLOCKS[m.group(1)]()}<!-- /cookbook:{m.group(1)} -->", doc, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--doc", default=DOC)
    a = ap.parse_args()
    old = open(a.doc).read()
    new = render(old)
    if a.update:
        open(a.doc, "w").write(new)
        print(f"cookbook: {len(BLOCKS)} blocks written to {a.doc}")
        return 0
    if old != new:
        sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "document", "regenerated"))
        return 1
    print(f"cookbook: {len(BLOCKS)} blocks equal the generators")
    return 0


if __name__ == "__main__":
    sys.exit(main())
