"""Writes `.ots` text (assembly.md sec. 3): directives, externs, bundles in cycle order. It emits text, never
words, so that both assemblers stay on the tested path."""

from . import patch

ARCH = "opentsp-tsa"
SCHEMA = "opentsp.program-case"


def word_hex(b):
    return bytes(b).hex()


def key(slice_, addr):
    return f"{slice_}:{addr}"


def case(name, p, weights=None, preload=None, dump=(), words=None, diagnostics=(), golden=None,
         cycles=200, tags=("sim",), asm_warnings=(), asm_error=None, sites=None):
    """A case file (schema opentsp.program-case version 1). `weights`: {extern: (dtype, bytes)},
    the bytes a whole number of vlanes-byte words, inlined as hex; `preload`, `words`: {(slice, addr): bytes};
    `diagnostics`: [(cycle, queue, code)] in order; `sites`: the patch.Table of the program's patch sites."""
    base = name.split("/")[-1]
    c = {
        "schema": SCHEMA, "version": 1, "spec_version": "0.1",
        "name": name, "profile": p.name, "program": f"{base}.{p.tag}.ots",
        "tags": list(tags),
        "weights": {n: {"dtype": dt, "words": len(b) // p.vlanes, "hex": word_hex(b)}
                    for n, (dt, b) in (weights or {}).items()},
        "preload": {key(*k): word_hex(v) for k, v in (preload or {}).items()},
        "dump": [key(*k) for k in dump],
        "expect": {
            "asm_warnings": list(asm_warnings),
            "diagnostics": [{"cycle": cy, "queue": q, "code": code} for cy, q, code in diagnostics],
            "words": {key(*k): word_hex(v) for k, v in (words or {}).items()},
            "decoded": None,
        },
        "golden": golden or {"source": "hand"},
        "budget": {"cycles": cycles},
    }
    if sites is not None and len(sites):
        c["sites"] = sites.records()
    if asm_error:
        c["expect"]["asm_error"] = asm_error     # both assemblers must refuse the program with this code
    for n, (dt, b) in (weights or {}).items():
        if len(b) % p.vlanes or not b:
            raise ValueError(f"weight {n}: {len(b)} bytes is not a whole number of {p.vlanes}-byte words")
    return c


class Program:
    def __init__(self, p, comment=None):
        self.p = p
        self.comment = comment
        self.externs = []
        self.bundles = {}    # cycle -> [(queue, text)]
        self.sites = []      # patch.Site, in the order added

    def extern(self, name, slice_, addr, size):
        self.externs.append((name, slice_, addr, size))

    def add(self, cycle, queue, text, site=None):
        """`site`: the patch.Decl (or a list of them) of the tokens `text` carries."""
        if cycle < 0:
            raise ValueError(f"negative cycle {cycle} for {queue}: {text}")
        if site is not None:
            patch.record(self, cycle, queue, text, site)
        self.bundles.setdefault(cycle, []).append((queue, text))

    def text(self, values=None):
        """The `.ots` text; a patch site's token is written as its value in `values` ({name: int}) or its placeholder."""
        vals = {s.name: s.placeholder for s in self.sites}
        vals.update(values or {})
        out = []
        if self.comment:
            out += ["; " + line for line in self.comment.strip().split("\n")]
        out += [f".arch {ARCH}", f".profile {self.p.asm_name}"]
        for name, s, a, n in self.externs:
            out.append(f".extern {name}, slice={s}, addr={a:#x}, size={n:#x}")
        out.append(".section text")
        for c in sorted(self.bundles):
            out.append(f"bundle @cycle={c} {{")
            out += [f"  {q}: {patch.resolve(t, vals) if '@' in t else t}" for q, t in self.bundles[c]]
            out.append("}")
        return "\n".join(out) + "\n"
