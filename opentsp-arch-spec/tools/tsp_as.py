#!/usr/bin/env python3
"""tsp_as: the reference assembler and disassembler of the TSP reference architecture.

It implements opentsp-arch-spec/docs/assembly.md exactly. The instruction grammar
is not written here: it is derived from isa-db/generated/encoding.json (per-instruction
templates, match patterns, fields and the operand-kind table), which is checked against
the Sail model's encdec and assembly mappings by tools/isa_db_check.sh.

usage:
  tsp_as.py asm  <prog.ots> [-o <obj.elf>] [--input]     assemble a program file (--input: print the
                                                           harness Q lines of its lowering)
  tsp_as.py dis  <obj.elf | input.txt> [--profile P]      disassemble to program text
  tsp_as.py insn [--profile P] <text>                      one instruction -> "<width> <hex>"
  tsp_as.py word [--profile P] <width> <hex>               one word -> instruction text
  tsp_as.py objinput <obj.elf> [--profile P] [--weights <w.safetensors>]
                                                           binary object -> tsp_sim input: the Q lines of
                                                           its text images (with the S lines of its externs)
  tsp_as.py wpack --profile P -o <w.safetensors> name=dtype:file ...  pack a weight package (dtype: an
                                                           element type or raw; file: raw bytes)
  tsp_as.py mpack --profile P [--topology T.json] -o <m.otsm> <chip0.elf> ... pack a multi-chip manifest:
                                                           K objects of profile P, chip order (K = 1..264, or
                                                           the chips of the topology file, at most 512)
  tsp_as.py mobjinput <m.otsm> [--weights <chip0.safetensors> --weights <chip1.safetensors> ...]
                                                           manifest -> tsp_sim --chips K input: per chip a
                                                           "C <chip>" line, its S lines (one weight package
                                                           per chip, in order) and its Q lines

Every command takes --config <profile.json> (a profile file other than config/<P>.json). It is validated as tsp_sim and tsp-iss
validate it: the file against config/schema.json (types, bounds, relations), then the timing rules; a refused file prints
"tsp_as: config <file>: <key>: <reason>" and exits 2. --no-validate skips the check.

Diagnostics are printed as "<file>:<line>: <code>: <message>" and the exit status is 1. An argument that is not well formed
(a <width> or <hex> that is not an integer, a missing argument) or a file that cannot be read is a coded diagnostic too
(OTS-E-ASM-001, OTS-E-ASM-003); the tool never prints a traceback.
"""
import ctypes, ctypes.util, functools, hashlib, json, os, re, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..")
ENCODING = os.path.join(SPEC, "isa-db", "generated", "encoding.json")
SPEC_VERSION = (0, 1)
ARCH = "opentsp-tsa"


def load_encoding(path=ENCODING):
    db = json.load(open(path))
    if db.get("spec_version") != "%d.%d" % SPEC_VERSION:
        raise AsmError("OTS-E-ASM-003", f"{path}: spec version {db.get('spec_version')}, not 0.1")
    return db


# Record-level tables, all read from encoding.json: nothing about queues,
# units, the control channel, the decode rule or boolean types is hard-coded here.
class _Tables:
    def __init__(self, db):
        self.control = db["control_channel"]["queue"]
        self.control_set = tuple(db["control_channel"]["instructions"])
        self.units = {u: set(qs) for u, qs in db["units"].items() if u != "text"}
        self.names = dict(db["queues"]["names"])
        self.aliases = dict(db["queues"]["aliases"])
        self.inverse = {q: n for n, q in self.names.items()}
        self.decode_rule = db["decode_rule"]
        self.boolean_codes = set(db["boolean_codes"])
        self.stream = db["stream_encoding"]
        self.agt = db["agt"]
        self.mem_slots = db["mem_slots"]
        self.slot_queues = set().union(*(self.units[u] for u in self.mem_slots["units"]))
        self.sram_access = db["sram_access"]["words"]
        self.profile_variables = db["profile_variables"]
        self.diagnostics = db["diagnostics"]
        self.eltype_codes = db["eltype_codes"]


T = None


def tables():
    global T
    if T is None:
        T = _Tables(load_encoding())
    return T


def __getattr__(name):          # CONTROL and CONTROL_CHANNEL, for callers and tests
    if name == "CONTROL":
        return tables().control
    if name == "CONTROL_CHANNEL":
        return tables().control_set
    raise AttributeError(name)

# Decoded queue and extent records from the ELF containers.
RAW_DTYPE = 255                               # a tensor of raw bytes (no element type)
IMAGE = struct.Struct("<HHI")                 # queue, 0, number of 32-bit text units
EXTERN = struct.Struct("<16sHHIII")           # name, slice, 0, addr, size, 0


class AsmError(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg = code, msg


# ---------------------------------------------------------------------------------------
# Lexical rules (assembly.md sec. 1). Separators are those of the Sail library mappings:
# spc() = one or more spaces, sep() = opt_spc() "," def_spc() = [ ]* "," [ ]*. Only the
# space character separates tokens; a tab is an error.
SPC, SEP = r" +", r" *, *"
HEX_TOK = r"0x[0-9a-fA-F]+"          # Sail hex_bits_N: leading zeros and A-F accepted
DEC_TOK = r"[0-9]+"                  # Sail dec_bits_N: leading zeros accepted
STREAM_TOK = r"[EW]:[0-9]+"
SYM_TOK = r"[A-Za-z_][A-Za-z0-9_]*(?:\+0x[0-9a-fA-F]+)?"   # no spaces around "+" (sec. 1)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")      # used with fullmatch: `$` would also match before a final line feed
TENSOR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,14}")   # a tensor name in a weight package (assembly.md sec. 4)


def span(location):
    a, _, b = location.partition("-")
    return int(a), int(b or a)


def field_parts(v):
    """The bit ranges of a field, least significant first: location, then hi_location, then top_location."""
    return [span(v[k]) for k in ("location", "hi_location", "top_location") if k in v]


def field_bits(v):
    return sum(a - b + 1 for a, b in field_parts(v))


def place(v, x):
    w, shift = 0, 0
    for a, b in field_parts(v):
        n = a - b + 1
        w |= ((x >> shift) & ((1 << n) - 1)) << b
        shift += n
    return w


def extract(v, w):
    x, shift = 0, 0
    for a, b in field_parts(v):
        n = a - b + 1
        x |= ((w >> b) & ((1 << n) - 1)) << shift
        shift += n
    return x


def check_profile_name(name):
    if not re.fullmatch(r"[a-z0-9_]{1,16}", name or ""):
        raise AsmError("OTS-E-ASM-003", f"unknown profile {name!r}")


def load_profile(name, config=None):
    check_profile_name(name)
    path = config or os.path.join(SPEC, "config", f"{name}.json")
    if not os.path.exists(path):
        raise AsmError("OTS-E-ASM-003", f"unknown profile {name!r}")
    cfg = json.load(open(path))
    if cfg.get("spec_version") != "%d.%d" % SPEC_VERSION:
        raise AsmError("OTS-E-ASM-003", f"profile {name!r} is spec version {cfg.get('spec_version')}, not 0.1")
    from opentsp_reference.schema import causality_errors
    errors = causality_errors(cfg)
    if errors:
        raise AsmError("OTS-E-ASM-003", "; ".join(errors))
    return cfg


# ---------------------------------------------------------------------------------------
# Queue names (assembly.md sec. 3, topology.md sec. 5), from encoding.json `queues`.
def queue_id(name):
    """The queue id of a queue name, or AsmError OTS-E-ASM-005. An index may have leading zeros."""
    t = tables()
    canon = re.sub(r"\[0*([0-9]+)\]", r"[\1]", name)
    if canon in t.names:
        return t.names[canon]
    if canon in t.aliases:
        return t.aliases[canon]
    raise AsmError("OTS-E-ASM-005", f"unknown queue {name!r} (or index out of range)")


def queue_name(q):
    """The canonical name of queue id q (the inverse of queue_id)."""
    n = tables().inverse.get(q)
    if n is None:
        raise AsmError("OTS-E-ASM-005", f"queue id {q:#x} out of range")
    return n


# ---------------------------------------------------------------------------------------
class ISA:
    """The instruction grammar and encoder, derived from encoding.json."""

    def __init__(self, profile="ref", config=None, encoding=ENCODING):
        if config and CHECK_CAPS:
            check_config(config)             # the CLI refuses an invalid profile; so does the API (ConfigError)
        self.cfg = load_profile(profile, config)
        self.profile = profile
        self.config = config                 # the file in force, if any (: named in the refusals)
        db = load_encoding(encoding)
        self.t = _Tables(db)
        self.pvars = {n: self._config_expr(d["expr"]) for n, d in self.t.profile_variables.items()}
        self.vlanes = self.pvars["vlanes"]
        self.words = self.cfg["sram"]["words_per_slice"]
        self.kinds = db["operand_kinds"]
        self.entries = db["instructions"]
        self.narrow_tags = sorted(self.t.decode_rule["narrow_tags"])
        self.wide_tags = sorted(self.t.decode_rule["wide_tags"])
        self.compiled = [self._compile(e) for e in self.entries]

    def where(self):
        """The configuration in force, for the messages: the file given by --config, else the named profile."""
        return f"file {self.config}" if self.config else f"profile {self.profile}"

    def _config_expr(self, expr):
        """A profile variable: factors joined by `*`, each an integer, a `<config.path>` or `<int> ^ <config.path>` (encoding.json profile_variables)."""
        val = 1
        for f in expr.split("*"):
            f = f.strip()
            if re.fullmatch(r"[0-9]+", f):
                val *= int(f)
            else:
                base, _, key = f.partition("^")
                x = self.cfg
                for k in (key if key else f).strip().split("."):
                    try:
                        x = x[k]
                    except (KeyError, TypeError):    # only reachable without validation (--no-validate): name the key, no traceback
                        raise AsmError("OTS-E-ASM-003", f"profile {self.profile}: {(key if key else f).strip()}: a key of a profile variable is missing") from None
                val *= int(base) ** x if key else x
        return val

    def max_of(self, v):
        """The upper bound of an unsigned field from `max` and `max_expr` (`name` or `name-<int>`)."""
        m = v.get("max")
        if "max_expr" in v:
            name, _, off = v["max_expr"].partition("-")
            e = self.pvars[name.strip()] - (int(off) if off else 0)
            m = e if m is None else min(m, e)
        return m

    # -- grammar ----------------------------------------------------------------------
    def _token(self, v):
        k = self.kinds[v["kind"]]
        if k["syntax"] == "name":
            return "|".join(sorted(map(re.escape, k["names"]), key=len, reverse=True))
        if k["syntax"] == "stream":
            return STREAM_TOK
        if k["syntax"] == "dec":
            return DEC_TOK
        return SYM_TOK + "|" + HEX_TOK if k.get("symbol") else HEX_TOK

    def _compile(self, e):
        """A regex of the entry's template. Each operand position accepts every token of its
        lexical class, so an out-of-range value is reported as such, not as a syntax error."""
        vs = {v["name"]: v for v in e["variables"]}
        pat, order, pos = "", [], 0
        for m in re.finditer(r"\{(\w+)\}", e["assembly"]):
            pat += self._literal(e["assembly"][pos:m.start()])
            pat += "(" + self._token(vs[m.group(1)]) + ")"
            order.append(vs[m.group(1)])
            pos = m.end()
        pat += self._literal(e["assembly"][pos:])
        return re.compile(pat + r"\Z"), order

    @staticmethod
    def _literal(s):
        out = ""
        for part in re.split(r"(, | )", s):
            out += SEP if part == ", " else SPC if part == " " else re.escape(part)
        return out

    # -- operand values ---------------------------------------------------------------
    def _value(self, e, v, tok, syms, queue):
        k = self.kinds[v["kind"]]
        name = f"{e['mnemonic']} operand {v['name']}"
        nbits = field_bits(v)
        if k["syntax"] == "name":
            x = k["names"][tok]
        elif k["syntax"] == "stream":
            row, n = tok.split(":")
            n = int(n, 10)
            if n > 31:
                raise AsmError("OTS-E-OPR-001", f"{name}: stream {tok} out of range 0..31")
            g = k.get("group", 1)
            if n % g:
                raise AsmError("OTS-E-OPR-001", f"{name}: {tok} is not a {g}-stream group base ({row}:0 or {row}:16)")
            x = ((self.t.stream["row_codes"][row] << self.t.stream["row_bit"]) + n) // g
        elif k["syntax"] == "dec":
            x = int(tok, 10)
        elif tok.startswith("0x"):
            x = int(tok, 16)
        else:                                         # a symbol: name or name+0xoff
            sym, _, off = tok.partition("+")
            if sym not in syms:
                raise AsmError("OTS-E-ASM-002", f"{name}: undefined symbol {sym!r}")
            ext = syms[sym]
            off = int(off, 16) if off else 0
            nw = self.t.sram_access.get(e["name"], 1)       # IfetchM reads a and a+1 (F7)
            if off + nw > ext["size"]:
                what = f"offset {off:#x}" if nw == 1 else f"words {off:#x}..{off + nw - 1:#x}"
                raise AsmError("OTS-E-ASM-002", f"{name}: {what} outside {sym} (size {ext['size']:#x})")
            if queue is not None and queue != ext["slice"]:
                raise AsmError("OTS-E-ASM-002", f"{name}: {sym} is in MEM[{ext['slice']}], not on queue {queue_name(queue)}")
            x = ext["addr"] + off
        if k.get("signed"):
            if x >= 1 << nbits:
                raise AsmError("OTS-E-OPR-001", f"{name}: {tok} does not fit {nbits} bits")
            s = x - (1 << nbits) if x >> (nbits - 1) else x
            lo, hi = v.get("min", -(1 << 15)), v.get("max", (1 << 15) - 1)
            if "max_expr" in v:                       # a signed field: the expression bounds the magnitude
                hi = self.max_of(v)
                lo = max(lo, -hi)
            if not lo <= s <= hi:
                raise AsmError("OTS-E-OPR-001", f"{name}: {s} out of range {lo}..{hi}")
            return x
        if x >= 1 << nbits:
            raise AsmError("OTS-E-OPR-001", f"{name}: {tok} does not fit the {nbits}-bit field")
        if "max_expr" in v and x > self.max_of(v):
            raise AsmError("OTS-E-OPR-001", f"{name}: {tok} above {v['max_expr']} = {self.max_of(v)} ({self.where()})")
        legal = v.get("values", v.get("codes"))
        if legal is not None and x not in legal:
            code = "OTS-E-DEC-003" if k["syntax"] == "name" else "OTS-E-OPR-001"
            raise AsmError(code, f"{name}: {tok} is not legal here")
        declared = self.cfg.get("extensions", [])
        for ext, ext_codes in v.get("extension_codes", {}).items():
            if x in ext_codes and ext not in declared:
                raise AsmError("OTS-E-DEC-003", f"{name}: {tok} needs the extension {ext}, which the configuration in force ({self.where()}) does not declare")
        if x in v.get("not_values", []) or x > v.get("max", x):
            raise AsmError("OTS-E-OPR-001", f"{name}: {tok} is excluded (range: {self._range_text(v)})")
        return x

    @staticmethod
    def _range_text(v):
        s = f"0..{v['max']}" if "max" in v else f"{field_bits(v)} bits"
        return s + (", not " + " ".join(map(str, v["not_values"])) if v.get("not_values") else "")

    def _joint(self, e, fields, toks):
        """Joint constraints: a field's bool_values apply when the type is one of boolean_codes."""
        names = {v["name"]: v for v in e["variables"]}
        t = names.get("type")
        if t is None:
            return
        tname = {c: n for n, c in self.kinds[t["kind"]]["names"].items()}[fields["type"]]
        for v in e["variables"]:
            if "bool_values" in v and fields["type"] in self.t.boolean_codes and fields[v["name"]] not in v["bool_values"]:
                raise AsmError("OTS-E-DEC-003", f"{e['mnemonic']}: {toks[v['name']]} is not legal for boolean type {tname}")

    # -- assemble / disassemble one instruction ---------------------------------------
    def parse(self, text, syms=None, queue=None):
        """(entry, word) for one instruction text, or AsmError."""
        if "\t" in text:
            raise AsmError("OTS-E-ASM-001", "tab character (only spaces separate tokens)")
        if not text.isascii():
            raise AsmError("OTS-E-ASM-001", "non-ASCII character")
        text = text.strip(" ")                        # leading and trailing spaces are ignored (sec. 1)
        syms = syms or {}
        hits, errors = [], []
        for e, (rx, order) in zip(self.entries, self.compiled):
            m = rx.match(text)
            if not m:
                continue
            try:
                fields = {v["name"]: self._value(e, v, tok, syms, queue) for v, tok in zip(order, m.groups())}
                self._joint(e, fields, {v["name"]: tok for v, tok in zip(order, m.groups())})
                hits.append((e, self.encode(e, fields)))
            except AsmError as err:
                errors.append(err)
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AsmError("OTS-E-ASM-001", f"{text!r} is ambiguous: {[h[0]['name'] for h in hits]}")
        if errors:
            raise errors[0]
        # the text in double quotes, as the Rust assembler quotes it (Debug form): one text for both
        raise AsmError("OTS-E-DEC-003", f"no instruction matches {json.dumps(text, ensure_ascii=False)}")

    @staticmethod
    def encode(e, fields):
        w = int(e["match"].replace("-", "0"), 2)
        for v in e["variables"]:
            w |= place(v, fields[v["name"]])
        return w

    def _matches(self, e, w):
        m = e["match"]
        fixed = int(m.replace("-", "0"), 2)
        mask = int("".join("0" if c == "-" else "1" for c in m), 2)
        return w & mask == fixed

    def width_of(self, w0, queue=None):
        """The layout width of a cycle-view word (assembly.md sec. 4, lowering): 32 iff bits
        63..32 are zero and the tag in 31..28 is a narrow tag; else 64. The queue does not matter."""
        return 32 if w0 >> 32 == 0 and (w0 >> 28) & 0xF in self.narrow_tags else 64

    def decode_text(self, q, units):
        """(code, entry, fields) of the instruction that starts a queue's text units (one unit,
        or two for a 64-bit layout), by encoding.json decode_rule: the slice tag in bits 30..28 of
        the first unit gives the layout; the bundle bit (bit 31) is cleared first; a tag that is
        neither narrow nor wide, or a word that matches no entry, is OTS-E-DEC-003 (code None:
        decoded). The queue does not change the decode (placement does)."""
        u0 = units[0] & 0x7FFFFFFF
        tag = (u0 >> 28) & 7
        if tag in self.narrow_tags:
            width, w = 32, u0
        elif tag in self.wide_tags and len(units) > 1:
            width, w = 64, (u0 << 32) | units[1]
        else:
            return "OTS-E-DEC-003", None, None
        d = self.decode(width, w)
        if d is None:
            return "OTS-E-DEC-003", None, None
        return None, d[0], d[1]

    def decode_word(self, q, w0, nunits):
        """decode_text of a raised word (raise_images): its units are rebuilt from it."""
        units = [w0 & 0xFFFFFFFF] if nunits == 1 else [w0 >> 32, w0 & 0xFFFFFFFF]
        if nunits == 1 and (w0 >> 28) & 7 not in self.narrow_tags:
            units = [w0 & 0x7FFFFFFF]
        return self.decode_text(q, units)

    def placement(self, e, q):
        """The placement diagnostic of entry e on queue q, or None (control_channel, units)."""
        if q == self.t.control:
            return None if e["name"] in self.t.control_set else "OTS-E-QUE-002"
        if q < 88 and q % 44 >= self.cfg["geometry"]["mem_slices_per_hemisphere"]:
            return "OTS-E-QUE-002"                  # the queue of an absent MEM slice
        if 104 <= q <= 115 and ((0 if q <= 106 else 1) if q <= 109 else (0 if q <= 112 else 1)) >= self.cfg["mxm"]["planes_per_hemisphere"]:
            return "OTS-E-QUE-002"                  # the queue of an absent MXM plane
        return None if any(q in self.t.units.get(u, ()) for u in e["units"]) else "OTS-E-QUE-002"

    def decode(self, width, w):
        """(entry, fields) of a word, or None if it does not decode."""
        for e in self.entries:
            if e["width"] != width or not self._matches(e, w):
                continue
            fields = {v["name"]: extract(v, w) for v in e["variables"]}
            try:
                text = self.render(e, fields)
                if self.parse(text)[1] == w:
                    return e, fields
            except AsmError:
                continue
        return None

    def render(self, e, fields):
        """The canonical text (as the Sail assembly mapping prints it)."""
        out = e["assembly"]
        for v in e["variables"]:
            k, x = self.kinds[v["kind"]], fields[v["name"]]
            if k["syntax"] == "name":
                t = {c: n for n, c in k["names"].items()}.get(x)
                if t is None:
                    raise AsmError("OTS-E-DEC-003", f"{e['name']}: code {x} of {v['name']} has no name")
            elif k["syntax"] == "stream":
                x *= k.get("group", 1)
                rb = self.t.stream["row_bit"]
                row = {c: r for r, c in self.t.stream["row_codes"].items()}[(x >> rb) & 1]
                t = f"{row}:{x & ((1 << rb) - 1)}"
            elif k["syntax"] == "dec":
                t = str(x)
            else:
                t = hex(x)
            out = out.replace("{" + v["name"] + "}", t)
        return out

    def disasm(self, width, w):
        d = self.decode(width, w)
        if d is None:
            return f"illegal{width} {w:#x}"
        return self.render(*d)

    def unit_ok(self, e, q):
        return self.placement(e, q) is None

    # -- records per cycle (microarchitecture.md sec. 3 "Instruction front end") ----------
    def slots(self, e, f):
        """The slots entry e with fields f takes on a MEM queue (encoding.json `mem_slots`), as
        (slot, thread) keys, or None when it needs the whole queue."""
        ms = self.t.mem_slots
        names = ms["slot_of"].get(e["name"])
        if names is None:
            return None
        if e["name"] in ms.get("all_threads", []):
            return [(n, t) if n in ms["per_thread"] else (n, None)
                    for n in names for t in (range(ms["threads"]) if n in ms["per_thread"] else [None])]
        th = f.get("thread", 0)
        return [(n, th if n in ms["per_thread"] else None) for n in names]

    def group_codes(self, q, group):
        """The model's group diagnostics for the records of one queue and cycle: group is a list
        of (entry, fields) with entry None for a record that does not decode. Returns one code
        (or None) per record: OTS-E-QUE-001 on every record when they do not fit the slots
        (two in one slot, a whole-queue record with others, an undecodable record among
        several, several records on a non-MEM queue); otherwise OTS-E-MEM-001 on both
        references of a same-thread R + W pair in one bank or with an indirect reference."""
        n = len(group)
        if n <= 1:
            return [None] * n
        if q not in self.t.slot_queues or any(e is None for e, _ in group):
            return ["OTS-E-QUE-001"] * n
        taken = set()
        for e, f in group:
            sl = self.slots(e, f)
            if sl is None or taken & set(sl):
                return ["OTS-E-QUE-001"] * n
            taken |= set(sl)
        ms, bank = self.t.mem_slots, self.cfg["sram"]["bank_bit"]
        def tags(e, f):
            """Bank used in this dispatch cycle, or None for an indirect reference.
            IfetchM's second word uses the following cycle. A word outside the
            slice addresses no bank; cross-cycle hazards remain runtime checks."""
            if e["name"] in ms["indirect"]:
                return None
            a = next(f[v["name"]] for v in e["variables"] if v["kind"] == "sram_addr")
            return {(a >> bank) & 1} if a < self.words else set()
        refs = {}
        for k, (e, f) in enumerate(group):
            for slot, th in self.slots(e, f):
                if slot in ms["paths"]:
                    refs[(ms["paths"][slot], th)] = (k, tags(e, f))
        codes = [None] * n
        for (p, th), (k, b) in refs.items():
            o = refs.get((1 - p, th))
            if o is not None and b != set() and o[1] != set() and (b is None or o[1] is None or b & o[1]):
                codes[k] = "OTS-E-MEM-001"
        return codes


# ---------------------------------------------------------------------------------------
# Lowering (assembly.md sec. 4): the
# cycle view of a program (cycle, queue, word) -> the text of each queue. This is the
# normative function; every test generator and the assembler use it.
PADDING = 0xFFFFFFFF          # a text unit that is never enqueued
UNDECODABLE = 0x6FFFFFFF      # the unit a word that fits no layout is lowered to (tag 6: OTS-E-DEC-003)
NOP_MAX = 0xFFFF
BUNDLE_BIT = 0x80000000
CONTROL = 0xFFFF


def word_units(w):
    """The text units of one cycle-view word (64 bits; a 32-bit layout in bits 31..0)."""
    if not 0 <= w < 1 << 64:
        raise ValueError(f"word {w:#x} is not 64 bits")
    if w >> 32 == 0 and (w >> 28) & 0xF in (0, 3, 5):
        return [w]
    if (w >> 60) in (1, 2, 4):
        return [w >> 32, w & 0xFFFFFFFF]
    return [UNDECODABLE]


def is_nop_word(w):
    """A NOP word: tag 0, opcode 0, reserved bits zero, n = bits 15..0 != 0."""
    return w >> 16 == 0 and w & 0xFFFF != 0


def lower(entries):
    """{queue: [unit, ...]} for entries = [(cycle, queue, word), ...] (queue 0..143 or 0xFFFF;
    the result has the queues in ascending order, the control channel last, and omits queues
    without text).
    Per queue: sort by cycle (source order within a cycle); the entries of one cycle are one
    parcel, bundle bit 1 on every instruction but the last; before each parcel NOPs (each at most
    65535) for g = cycle - previous cycle - 1 (previous = -1 at the start). A NOP entry that is
    alone in its cycle is a timing instruction, not a dispatched one (it needs no dispatch
    path): it makes no parcel; the gaps already carry its time, so every other entry keeps its
    cycle. A NOP together with other entries stays in the parcel (a slot-rule violation).
    Theorem (tested, test/directed/frontend.py): for a program whose entries never meet a wait
    condition (microarchitecture.md sec. 3), the model dispatches every entry at its cycle."""
    by_q = {}
    for i, (c, q, w) in enumerate(entries):
        if not (0 <= q <= 143 or q == CONTROL):
            raise ValueError(f"queue {q:#x} is neither 0..143 nor 0xffff")
        if c < 0:
            raise ValueError(f"cycle {c} is negative")
        by_q.setdefault(q, []).append((c, i, w))
    images = {}
    for q in sorted(by_q, key=lambda q: (q == CONTROL, q)):
        cycles = {}
        for c, _, w in sorted(by_q[q]):
            cycles.setdefault(c, []).append(w)
        units, prev = [], -1
        for c in sorted(cycles):
            ws = cycles[c]
            if len(ws) == 1 and is_nop_word(ws[0]):
                continue
            gap = c - prev - 1
            while gap > 0:
                n = min(gap, NOP_MAX)
                units.append(n)
                gap -= n
            for k, w in enumerate(ws):
                u = word_units(w)
                if k < len(ws) - 1:
                    u[0] |= BUNDLE_BIT
                units += u
            prev = c
        if units:
            images[q] = units
    return images


def lower_records(recs):
    """lower() for the 5-tuples (cycle, queue, word_count, word0, word1) the test generators
    author: only a word count of 1 and a zero word 1 can be expressed (text has no such fields)."""
    for c, q, wc, w0, w1 in recs:
        if wc != 1 or w1 != 0:
            raise ValueError(f"record ({c}, {q:#x}) has word count {wc} / word1 {w1:#x}: text cannot express it")
    return lower([(c, q, w0) for c, q, _, w0, _ in recs])


def raise_images(images):
    """The inverse of lower(), for text that never waits: [(cycle, queue, word, units)], where
    cycle = the previous parcel's cycle + 1 + the NOPs before the parcel (-1 before the first),
    word = the instruction's 64-bit word (a 32-bit layout in bits 31..0), units = 1 or 2. The
    instructions of a parcel share its cycle. A trailing parcel whose last instruction has the
    bundle bit set is still listed; NOPs after the last parcel are dropped."""
    out = []
    for q in sorted(images, key=lambda q: (q == CONTROL, q)):
        us, i, last, nops, parcel = images[q], 0, -1, 0, []
        while i < len(us):
            u = us[i]
            if u == PADDING:
                i += 1
                continue
            tag = (u >> 28) & 7
            n = 2 if tag in (1, 2, 4) else 1
            w = (((u & 0x7FFFFFFF) << 32) | us[i + 1]) if n == 2 and i + 1 < len(us) else (u & 0x7FFFFFFF)
            nop = not parcel and u & BUNDLE_BIT == 0 and tag == 0 and u & 0x0FFF0000 == 0 and u & 0xFFFF != 0
            if nop:
                nops += u & 0xFFFF
            else:
                parcel.append((w, n))
                if u & BUNDLE_BIT == 0:
                    last = last + 1 + nops
                    out += [(last, q, pw, pn) for pw, pn in parcel]
                    parcel, nops = [], 0
            i += n
        if parcel:
            out += [(last + 1 + nops, q, pw, pn) for pw, pn in parcel]
    return out


def drop_padding(images):
    return {q: [u for u in us if u != PADDING] for q, us in images.items()}


def q_lines(images, per_line=64):
    """The tsp_sim input lines of the text images: `Q <queue> <unit> ...` (hex, at most per_line
    units a line; several lines for one queue append)."""
    out = []
    for q in sorted(images, key=lambda q: (q == CONTROL, q)):
        us = images[q]
        for k in range(0, len(us), per_line):
            out.append(f"Q {q:x} " + " ".join(f"{u:08x}" for u in us[k:k + per_line]))
    return out


# ---------------------------------------------------------------------------------------
# Program files (assembly.md sec. 3).
class Program:
    """An assembled program: the cycle view (records: (cycle, queue, 1, word, 0), a stable sort
    by cycle and queue: the instructions of one parcel in source order) and its lowering (images: {queue: text units})."""
    def __init__(self):
        self.profile, self.externs, self.records = None, {}, []

    @property
    def images(self):
        return lower_records(self.records)


def strip_comment(line):
    return line.split(";", 1)[0].rstrip(" ")


def assemble(src, path="<input>", config=None, warn=None):
    """Assemble program text. Returns (isa, Program); raises AsmError with a location."""
    prog, isa = Program(), None
    state = "start"      # start -> arch -> profile -> text -> bundle
    cycle, last_cycle, used, lines = None, -1, {}, {}
    agt = {}             # (queue, path) -> [cycle, level, field position, records] of an open AGT configuration
    for ln, raw in enumerate(src.split("\n"), 1):
        def err(code, msg):
            e = AsmError(code, msg)
            e.where = f"{path}:{ln}"
            return e
        # Sec. 1 applies to the whole line, comments included.
        if "\t" in raw:
            raise err("OTS-E-ASM-001", "tab character (only spaces separate tokens)")
        if not raw.isascii():
            raise err("OTS-E-ASM-001", "non-ASCII character (a line is ASCII)")
        line = strip_comment(raw)
        s = line.strip(" ")
        if not s:
            continue
        try:
            if state == "bundle":
                if s == "}":
                    bundle_checks(isa, used, lines, cycle, path)
                    agt_bundle(isa, agt, used, lines, cycle, warn, path)
                    state, last_cycle = "text", cycle
                    continue
                # Any text before the first ":" is a queue name (unknown names: OTS-E-ASM-005).
                m = re.fullmatch(r"([^ :][^:]*?) *: *(.+)", s)
                if not m:
                    raise AsmError("OTS-E-ASM-001", f"expected '<queue>: <instruction>' or '}}', got {s!r}")
                q = queue_id(m.group(1))
                e, w = isa.parse(m.group(2), prog.externs, q if 0 <= q <= 87 else None)
                # Placement and the slot rule are checked when the bundle closes, in the model's
                # order (bundle_checks).
                nw = isa.t.sram_access.get(e["name"], 1)       # IfetchM reads a and a+1 (F7)
                for v in e["variables"]:
                    if v["kind"] == "sram_addr" and extract(v, w) + nw - 1 >= isa.words and warn:
                        a = extract(v, w)
                        what = f"address {a:#x}" if nw == 1 else f"word {a + nw - 1:#x} (of {a:#x}..{a + nw - 1:#x})"
                        warn(f"{path}:{ln}: OTS-E-OPR-001 (warning): {what} is beyond "
                             f"words_per_slice {isa.words} of profile {isa.profile}; the reference faults at run time")
                used.setdefault(q, []).append((e, {v["name"]: extract(v, w) for v in e["variables"]}))
                lines.setdefault(q, []).append(ln)
                prog.records.append((cycle, q, 1, w, 0))
                continue
            m = re.fullmatch(r"\.arch +(\S+)", s)
            if m:
                if state != "start":
                    raise AsmError("OTS-E-ASM-001", ".arch must be the first directive")
                if m.group(1) != ARCH:
                    raise AsmError("OTS-E-ASM-003", f"unknown architecture {m.group(1)!r} (expected {ARCH})")
                state = "arch"
                continue
            m = re.fullmatch(r"\.profile +(\S+)", s)
            if m:
                if state != "arch":
                    raise AsmError("OTS-E-ASM-001", ".profile must follow .arch, once")
                isa = ISA(m.group(1), config)
                prog.profile, state = m.group(1), "profile"
                continue
            m = re.fullmatch(r"\.extern +(.*)", s)
            if m:
                if state != "profile":
                    raise AsmError("OTS-E-ASM-001", ".extern must follow .profile and precede .section text")
                x = extern_decl(m.group(1), isa, prog.externs)
                prog.externs[x["name"]] = x
                continue
            if s.startswith("."):
                if re.fullmatch(r"\.section +text", s):
                    if state != "profile":
                        raise AsmError("OTS-E-ASM-001", ".section text must follow .arch and .profile, once")
                    state = "text"
                    continue
                raise AsmError("OTS-E-ASM-001", f"unknown directive {s!r}")
            m = re.fullmatch(r"bundle +@cycle=([0-9]+) *\{", s)
            if m:
                if state != "text":
                    raise AsmError("OTS-E-ASM-001", "bundle outside .section text, or nested")
                digits = m.group(1).lstrip("0") or "0"     # bounded before int(): Python refuses over 4,300 digits
                cycle = int(digits, 10) if len(digits) <= 10 else 1 << 32
                if cycle <= last_cycle or cycle >= 1 << 32:
                    raise AsmError("OTS-E-ASM-001", f"bundle @cycle={m.group(1)}: cycles must increase strictly and be below 2^32")
                state, used, lines = "bundle", {}, {}
                continue
            raise AsmError("OTS-E-ASM-001", f"unexpected {s!r}")
        except AsmError as e:
            e.where = getattr(e, "where", f"{path}:{ln}")
            raise
    if state == "bundle":
        e = AsmError("OTS-E-ASM-001", "unterminated bundle")
        e.where = f"{path}:end"
        raise e
    if state in ("start", "arch", "profile"):
        e = AsmError("OTS-E-ASM-001", "a program needs .arch, .profile and .section text")
        e.where = f"{path}:1"
        raise e
    prog.records.sort(key=lambda r: (r[0], r[1]))   # stable: cycle, then queue id; the instructions of one parcel stay in source order (sec. 4)
    if warn:
        cap = isa.cfg["icu"]["iq_depth"] // 4
        for q, us in prog.images.items():
            if len(us) > cap:
                warn(f"{path}: OTS-E-ASM-003 (warning): the text of {queue_name(q)} is {len(us)} units, more than "
                     f"icu.iq_depth ({cap} units): a loader rejects it as an initial image; fetch the rest with ifetch")
    return isa, prog


def bundle_checks(isa, used, lines, cycle, path):
    """The static dispatch checks of one bundle, in the model's order per queue
    (microarchitecture.md sec. 3 and sec. 8; model/postlude/tick.sail classify, dispatch):
    the slot rule (OTS-E-QUE-001), then each record's placement (OTS-E-QUE-002), then the
    bank rule (OTS-E-MEM-001)."""
    def fail(code, q, k, msg):
        e = AsmError(code, msg)
        e.where = f"{path}:{lines[q][k]}"
        raise e
    for q in sorted(used, key=lambda q: min(lines[q])):
        g = used[q]
        codes = isa.group_codes(q, g)
        if "OTS-E-QUE-001" in codes:
            fail("OTS-E-QUE-001", q, len(g) - 1, f"{', '.join(x[0]['mnemonic'] for x in g)} on {queue_name(q)} in bundle "
                 f"@cycle={cycle} do not fit the queue's slots (one Ifetch/IfetchM, one R and one W reference per "
                 "thread, one AGT configuration record, or one other instruction alone; microarchitecture.md sec. 3)")
        for k, (e, _) in enumerate(g):
            code = isa.placement(e, q)
            if code:
                fail(code, q, k, f"{e['mnemonic']} cannot issue on {queue_name(q)}")
        if "OTS-E-MEM-001" in codes:
            fail("OTS-E-MEM-001", q, codes.index("OTS-E-MEM-001"), f"R and W references of one thread on {queue_name(q)} "
                 f"in bundle @cycle={cycle} address the same bank (or one is a gather/scatter)")


# AGT placement (assembly.md sec. 3; encoding.json `agt`; microarchitecture.md sec. 4). Each
# dispatch path (R = 0, W = 1) of a MEM queue has its own AGT. The model reports a stray
# configuration record and a fifth level as OTS-E-OPR-001 at run time, and a new reference on
# a path whose configuration is open as OTS-E-QUE-001; the assembler warns about them
# statically. It follows the model's order within a cycle (model/postlude/tick.sail tick,
# dispatch_slots): a cycle of the queue without a configuration record ends the open
# configurations; the configuration record goes to the configuring path (both: the one with
# fewer records, R on a tie); then each reference alone on its path opens that path.
def agt_bundle(isa, agt, used, lines, cycle, warn, path):
    a, ms = isa.t.agt, isa.t.mem_slots
    def say(ln, msg):
        if warn:
            warn(f"{path}:{ln}: {msg}")
    for (q, p) in [k for k, st in agt.items() if st[0] != cycle - 1 or k[0] not in used]:
        del agt[(q, p)]                              # no records on q in this cycle, or a gap
    for q, group in used.items():
        cfg = [(k, e) for k, (e, _) in enumerate(group) if e["name"] in a["config"]]
        if not cfg:
            for p in (0, 1):
                agt.pop((q, p), None)
        else:
            k, e = cfg[0]
            ln = lines[q][k]
            open_ = [p for p in (0, 1) if (q, p) in agt]
            if not open_:
                say(ln, f"OTS-E-OPR-001 (warning): {e['mnemonic']} on {queue_name(q)} @cycle={cycle} configures no loop "
                        "(no reference alone on its path, nor a configuration, on this queue in the previous cycle)")
            else:
                p = (1 if agt[(q, 1)][3] < agt[(q, 0)][3] else 0) if len(open_) == 2 else open_[0]
                _, level, pos, nrec = agt[(q, p)]
                fp = a["config"].index(e["name"])
                if fp <= pos:
                    level, pos = level + 1, -1
                if level >= a["levels"]:
                    say(ln, f"OTS-E-OPR-001 (warning): {e['mnemonic']} on {queue_name(q)} @cycle={cycle} needs a level "
                            f"beyond {a['levels']}; the loop is aborted at run time")
                    del agt[(q, p)]
                else:
                    agt[(q, p)] = [cycle, level, fp, nrec + 1]
            for p in (0, 1):                         # the configurations continue into this cycle
                if (q, p) in agt:
                    agt[(q, p)][0] = cycle
        on_path = {0: [], 1: []}
        for k, (e, f) in enumerate(group):
            for slot, _ in isa.slots(e, f) or ():
                if slot in ms["paths"]:
                    on_path[ms["paths"][slot]].append((k, e))
        held = {p for p in (0, 1) if (q, p) in agt}
        for p, refs in on_path.items():
            if p in held:
                for k, e in refs:
                    say(lines[q][k], f"OTS-E-QUE-001 (warning): {e['mnemonic']} on {queue_name(q)} @cycle={cycle}: its "
                                     "path's AGT configuration is open; only configuration records get through")
            elif len(refs) == 1 and refs[0][1]["name"] in a["loopable"]:
                agt[(q, p)] = [cycle, 0, -1, 0]


def extern_decl(text, isa, known):
    """`.extern <name>, slice=<n>, addr=0x<hex>, size=0x<hex>` (assembly.md sec. 3)."""
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)" + SEP + r"slice=([0-9]+)" + SEP + r"addr=(" + HEX_TOK + ")" + SEP
                     + r"size=(" + HEX_TOK + ")", text)
    if not m:
        raise AsmError("OTS-E-ASM-001", f"expected '.extern <name>, slice=<n>, addr=0x<hex>, size=0x<hex>', got {text!r}")
    name, sl, addr, size = m.group(1), int(m.group(2)), int(m.group(3), 16), int(m.group(4), 16)
    if len(name) > 15:
        raise AsmError("OTS-E-ASM-002", f"extern name {name!r} longer than 15 characters")
    x = {"name": name, "slice": sl, "addr": addr, "size": size}
    check_extern(x, isa, known)
    return x


def check_extern(x, isa, known):
    """The extern rules shared by the text assembler and the object reader (OTS-E-ASM-002)."""
    name, sl, addr, size = x["name"], x["slice"], x["addr"], x["size"]
    if not IDENT.fullmatch(name):
        raise AsmError("OTS-E-ASM-002", f"extern name {name!r} is not an identifier")
    if len(name) > 15:
        raise AsmError("OTS-E-ASM-002", f"extern name {name!r} longer than 15 characters")
    if name in known:
        raise AsmError("OTS-E-ASM-002", f"extern {name!r} declared twice")
    if sl >= isa.cfg["sram"]["slices"]:
        raise AsmError("OTS-E-ASM-002", f"extern {name}: slice {sl} out of range")
    if sl % 44 >= isa.cfg["geometry"]["mem_slices_per_hemisphere"]:
        raise AsmError("OTS-E-ASM-002", f"extern {name}: slice {sl} does not exist in profile {isa.profile} "
                       f"(slice mod 44 >= geometry.mem_slices_per_hemisphere)")
    if size == 0 or addr + size > isa.words:
        raise AsmError("OTS-E-ASM-002", f"extern {name}: words {addr:#x}..{addr + size - 1:#x} outside 0..{isa.words - 1:#x}")
    for o in known.values():
        if o["slice"] == sl and addr < o["addr"] + o["size"] and o["addr"] < addr + size:
            raise AsmError("OTS-E-ASM-002", f"extern {name} overlaps {o['name']} in slice {sl}")


# ---------------------------------------------------------------------------------------
# Objects (assembly.md sec. 4).
def digest(data, n):
    return hashlib.sha256(data).digest()[:n]


def interface_digests(profile, config=None):
    """(encoding digest, profile digest): the first 12 bytes of SHA-256 of isa-db/generated/
    encoding.json and of the profile configuration file, byte for byte."""
    if config and CHECK_CAPS:
        check_config(config)                 # an object is written only for a profile the CLI accepts
    path = config or os.path.join(SPEC, "config", f"{profile}.json")
    return digest(open(ENCODING, "rb").read(), 12), digest(open(path, "rb").read(), 12)


ELF_HEADER = struct.Struct("<16sHHIIIIIHHHHHH")
ELF_SECTION = struct.Struct("<10I")
ELF_NOTE = struct.Struct("<III")
ELF_COMPRESSION = struct.Struct("<III")
ELF_IDENT = b"\x7fELF\x01\x01\x01" + bytes(9)
ELF_NOTE_NAME = b"OpenTSP\0"
ELF_ABI = struct.Struct("<HH16s32s32s")
ELF_FLAGS_COMPRESSED = 0x800
ELF_MAX_SECTION = 64 * 1024 * 1024


@functools.lru_cache(maxsize=1)
def _zstd_lib():
    libname = os.environ.get("OPENTSP_ZSTD_LIBRARY") or ctypes.util.find_library("zstd")
    if not libname and sys.platform == "darwin":
        libname = next((f"{root}/opt/zstd/lib/libzstd.dylib" for root in ("/opt/homebrew", "/usr/local")
                        if os.path.isfile(f"{root}/opt/zstd/lib/libzstd.dylib")), None)
    if not libname:
        raise AsmError("OTS-E-ASM-003", "libzstd is required for OpenTSP ELF")
    try:
        lib = ctypes.CDLL(libname)
    except OSError as error:
        raise AsmError("OTS-E-ASM-003", f"cannot load libzstd: {error}") from None
    for name, args in {
        "ZSTD_compressBound": [ctypes.c_size_t],
        "ZSTD_compress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int],
        "ZSTD_decompress": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t],
        "ZSTD_findFrameCompressedSize": [ctypes.c_char_p, ctypes.c_size_t],
    }.items():
        func = getattr(lib, name)
        func.argtypes, func.restype = args, ctypes.c_size_t
    lib.ZSTD_isError.argtypes, lib.ZSTD_isError.restype = [ctypes.c_size_t], ctypes.c_uint
    return lib


def _elf_compress(data):
    lib = _zstd_lib()
    out = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(data)))
    used = lib.ZSTD_compress(out, len(out), data, len(data), 3)
    if lib.ZSTD_isError(used):
        raise AsmError("OTS-E-ASM-003", "ELF Zstandard compression failed")
    return ELF_COMPRESSION.pack(2, len(data), 4) + out.raw[:used]


def _elf_decompress(data):
    if len(data) < ELF_COMPRESSION.size:
        raise AsmError("OTS-E-ASM-003", "truncated ELF compression header")
    codec, size, alignment = ELF_COMPRESSION.unpack_from(data)
    if codec != 2 or size > ELF_MAX_SECTION or alignment != 4:
        raise AsmError("OTS-E-ASM-003", "invalid OpenTSP ELF compression")
    frame = data[ELF_COMPRESSION.size:]
    lib = _zstd_lib()
    if lib.ZSTD_findFrameCompressedSize(frame, len(frame)) != len(frame):
        raise AsmError("OTS-E-ASM-003", "expected one complete ELF Zstandard frame")
    out = ctypes.create_string_buffer(size)
    used = lib.ZSTD_decompress(out, size, frame, len(frame))
    if used != size:
        raise AsmError("OTS-E-ASM-003", "ELF decompressed section length differs")
    return out.raw


def _write_elf(sections):
    body = bytearray(ELF_HEADER.size)
    strings = bytearray(b"\0")
    directory = bytearray(ELF_SECTION.size)
    # The section name table is always the last section.
    for name, kind, flags, contents, alignment in sections:
        name_at = len(strings)
        strings += name.encode("ascii") + b"\0"
        body += bytes((-len(body)) % alignment)
        directory += ELF_SECTION.pack(name_at, kind, flags, 0, len(body), len(contents), 0, 0, alignment, 0)
        body += contents
    name_at = len(strings)
    strings += b".shstrtab\0"
    directory += ELF_SECTION.pack(name_at, 3, 0, 0, len(body), len(strings), 0, 0, 1, 0)
    body += strings
    body += bytes((-len(body)) % 4)
    shoff = len(body)
    body += directory
    body[:ELF_HEADER.size] = ELF_HEADER.pack(ELF_IDENT, 1, 0, 1, 0, 0, shoff, 0, ELF_HEADER.size,
                                           0, 0, ELF_SECTION.size, len(sections) + 2, len(sections) + 1)
    return bytes(body)


def _read_elf(data):
    def bad(message):
        raise AsmError("OTS-E-ASM-003", message)

    if len(data) < ELF_HEADER.size or data[:16] != ELF_IDENT:
        bad("expected an OpenTSP ELF32 program; rebuild old object files")
    ident, etype, machine, ver, entry, phoff, shoff, flags, eh, phsize, phnum, shsize, shnum, strings = ELF_HEADER.unpack_from(data)
    if (etype, machine, ver, entry, phoff, flags, eh, phsize, phnum, shsize) != (1, 0, 1, 0, 0, 0, ELF_HEADER.size, 0, 0, ELF_SECTION.size):
        bad("invalid OpenTSP ELF header")
    if not (4 <= shnum <= 9 and 0 < strings < shnum and ELF_HEADER.size <= shoff and
            shoff % 4 == 0 and shoff + shnum * shsize == len(data)):
        bad("invalid OpenTSP ELF section directory")
    table = [ELF_SECTION.unpack_from(data, shoff + i * shsize) for i in range(shnum)]
    if any(table[0]):
        bad("invalid ELF null section")
    entries = []
    for i, section in enumerate(table[1:], 1):
        name, kind, flags, addr, offset, size, link, info, align, ent = section
        if addr or link or info or ent or align not in (1, 4) or offset % align or not (ELF_HEADER.size <= offset <= offset + size <= shoff):
            bad("ELF section outside the payload or invalid alignment")
        if kind not in (1, 3, 7) or flags not in (0, ELF_FLAGS_COMPRESSED) or (kind != 1 and flags):
            bad("unsupported OpenTSP ELF section")
        entries.append((offset, offset + size))
    ordered = sorted(entries)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        bad("overlapping ELF sections")
    if table[strings][1] != 3:
        bad("invalid ELF section-name table")
    so, ss = table[strings][4:6]
    names = data[so:so + ss]
    out = {}
    for n, kind, flags, _, offset, size, *_ in table[1:]:
        end = names.find(b"\0", n)
        if not 0 < n < end:
            bad("invalid ELF section name")
        name = names[n:end]
        types = {b".note.opentsp": 7, b".tsp.iq": 1, b".tsp.load": 1,
                 b".tsp.text": 1, b".tsp.const": 1, b".tsp.zero": 1, b".tsp.recipe": 1, b".shstrtab": 3}
        if types.get(name) != kind or name in out or size > ELF_MAX_SECTION:
            bad("unknown, repeated or oversized OpenTSP ELF section")
        payload = data[offset:offset + size]
        out[name] = _elf_decompress(payload) if flags else payload
    if not {b".note.opentsp", b".tsp.iq", b".tsp.load", b".shstrtab"} <= out.keys():
        bad("OpenTSP ELF lacks a required section")
    return out


def elf_identity(data):
    """SHA-256 of the decoded OpenTSP sections excluding the name table."""
    digest = hashlib.sha256()
    for name, payload in sorted(_read_elf(data).items()):
        if name == b".shstrtab":
            continue
        digest.update(name + b"\0")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def full_interface_digests(profile, config=None):
    path = config or os.path.join(SPEC, "config", f"{profile}.json")
    with open(ENCODING, "rb") as f:
        enc = hashlib.file_digest(f, "sha256").digest()
    with open(path, "rb") as f:
        cfg = hashlib.file_digest(f, "sha256").digest()
    return enc, cfg


def write_object(prog, config=None, *, text=b"", constants=b"", zeroes=None, recipe=b""):
    """Serialize a resolved program as an OpenTSP ELF32 object."""
    check_profile_name(prog.profile)
    if config and CHECK_CAPS:
        check_config(config)
    enc, prof = full_interface_digests(prog.profile, config)
    meta = ELF_ABI.pack(1, 0, prog.profile.encode(), enc, prof)
    note = ELF_NOTE.pack(len(ELF_NOTE_NAME), len(meta), 1) + ELF_NOTE_NAME + meta
    iq = bytearray(struct.pack("<I", len(prog.images)))
    for q, units in prog.images.items():
        iq.extend(IMAGE.pack(q, 0, len(units)))
    for units in prog.images.values():
        for unit in units:
            iq.extend(struct.pack("<I", unit))
    loads = bytearray(struct.pack("<I", len(prog.externs)))
    for x in prog.externs.values():
        loads.extend(EXTERN.pack(x["name"].encode(), x["slice"], 0, x["addr"], x["size"], 0))
    sections = [(".note.opentsp", 7, 0, note, 4),
                (".tsp.iq", 1, ELF_FLAGS_COMPRESSED, _elf_compress(bytes(iq)), 4),
                (".tsp.load", 1, ELF_FLAGS_COMPRESSED, _elf_compress(bytes(loads)), 4)]
    if text:
        sections.append((".tsp.text", 1, ELF_FLAGS_COMPRESSED, _elf_compress(text), 4))
    if constants:
        sections.append((".tsp.const", 1, ELF_FLAGS_COMPRESSED, _elf_compress(constants), 4))
    if zeroes:
        records = b"".join(struct.pack("<16sI", name.encode(), phases) for name, phases in sorted(zeroes.items()))
        sections.append((".tsp.zero", 1, ELF_FLAGS_COMPRESSED, _elf_compress(records), 4))
    if recipe:
        sections.append((".tsp.recipe", 1, ELF_FLAGS_COMPRESSED, _elf_compress(recipe), 4))
    return _write_elf(sections)



def _ascii_field(raw, what):
    """A NUL-padded ASCII field: the text, then only NUL bytes (OTS-E-ASM-003 otherwise)."""
    text, nul, rest = raw.partition(b"\0")
    if rest.strip(b"\0") or not text:
        raise AsmError("OTS-E-ASM-003", f"{what}: not NUL-padded text")
    try:
        return text.decode("ascii")
    except UnicodeDecodeError:
        raise AsmError("OTS-E-ASM-003", f"{what}: not ASCII") from None


def read_object(data, profile=None, config=None):
    """Read a resolved OpenTSP ELF32 program."""
    sections = _read_elf(data)
    note = sections[b".note.opentsp"]
    base = ELF_NOTE.size + len(ELF_NOTE_NAME)
    if (len(note) != base + ELF_ABI.size or ELF_NOTE.unpack_from(note) != (len(ELF_NOTE_NAME), ELF_ABI.size, 1)
            or note[ELF_NOTE.size:base] != ELF_NOTE_NAME):
        raise AsmError("OTS-E-ASM-003", "invalid OpenTSP ELF ABI note")
    rev, reserved, raw_name, enc_digest, profile_digest = ELF_ABI.unpack_from(note, base)
    if (rev, reserved) != (1, 0):
        raise AsmError("OTS-E-ASM-003", "unsupported OpenTSP ELF ABI")
    prof = _ascii_field(raw_name, "profile")
    if profile is not None and prof != profile:
        raise AsmError("OTS-E-ASM-003", f"object is for profile {prof!r}, not {profile!r}")
    load_profile(prof, config)
    enc, cur = full_interface_digests(prof, config)
    if enc != enc_digest:
        raise AsmError("OTS-E-ASM-003", "object was built against another encoding.json")
    if cur != profile_digest:
        raise AsmError("OTS-E-ASM-003", f"object was built against another configuration of profile {prof!r}")
    data = sections[b".tsp.iq"]
    loads = sections[b".tsp.load"]
    if len(data) < 4 or len(loads) < 4:
        raise AsmError("OTS-E-ASM-003", "truncated OpenTSP section")
    nimg, = struct.unpack_from("<I", data)
    next_, = struct.unpack_from("<I", loads)
    if nimg > 145 or len(data) < 4 + nimg * IMAGE.size or len(loads) != 4 + next_ * EXTERN.size:
        raise AsmError("OTS-E-ASM-003", "ELF section length does not match its directory")
    table = [IMAGE.unpack_from(data, 4 + k * IMAGE.size) for k in range(nimg)]
    nunits = sum(n for _, _, n in table)
    if len(data) != 4 + nimg * IMAGE.size + 4 * nunits:
        raise AsmError("OTS-E-ASM-003", "ELF IQ section length differs")
    isa = ISA(prof, config)          # an unknown profile, or one of another version: OTS-E-ASM-003
    cap = isa.cfg["icu"]["iq_depth"] // 4
    images, p, prev = {}, 4 + nimg * IMAGE.size, None
    for q, z, n in table:
        if z != 0 or q not in isa.t.inverse or n == 0:
            raise AsmError("OTS-E-ASM-003", f"bad image entry (queue {q:#x}, {n} units)")
        key = (q == CONTROL, q)
        if prev is not None and prev >= key:
            raise AsmError("OTS-E-ASM-003", f"image of queue {q:#x}: not in canonical order (ascending queue, control channel last) or duplicated")
        prev = key
        us = list(struct.unpack_from(f"<{n}I", data, p))
        p += 4 * n
        if sum(1 for u in us if u != PADDING) > cap:
            raise AsmError("OTS-E-ASM-003", f"image of queue {q:#x}: {n} units do not fit icu.iq_depth ({cap} units)")
        images[q] = us
    # The parcels of an image must fit the queue's slots (microarchitecture.md sec. 3, "Parcels
    # and slots"; encoding.json `mem_slots`), as in the assembler.
    groups = {}
    for c, q, w, nu in raise_images(images):
        groups.setdefault((q, c), []).append((w, nu))
    for (q, c), g in groups.items():
        if len(g) > 1:
            dec = [isa.decode_word(q, w, nu)[1:] for w, nu in g]
            if "OTS-E-QUE-001" in isa.group_codes(q, dec):
                raise AsmError("OTS-E-ASM-003", f"{len(g)} instructions in one parcel of queue {q:#x} do not fit the queue's "
                               "slots (microarchitecture.md sec. 3, Instruction front end)")
    known = {}
    exts = []
    p = 4
    for i in range(next_):
        n, sl, z, a, s_, z2 = EXTERN.unpack_from(loads, p)
        p += EXTERN.size
        if z or z2:
            raise AsmError("OTS-E-ASM-003", f"extern entry {i}: non-zero pad field")
        x = {"name": _ascii_field(n, f"extern entry {i} name"), "slice": sl, "addr": a, "size": s_}
        try:
            check_extern(x, isa, known)
        except AsmError as e:
            raise AsmError("OTS-E-ASM-003", f"extern entry {i}: {e.msg}") from None
        known[x["name"]] = x
        exts.append(x)
    return prof, images, exts


SAFETENSORS_MAX_HEADER = 16 * 1024 * 1024


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def write_weights(profile, tensors, config=None):
    """Save named physical words as Safetensors U8 arrays with profile metadata."""
    isa = ISA(profile, config)
    vl = isa.pvars["vlanes"]
    codes = isa.t.eltype_codes
    attributes, images = {"opentsp.profile": profile}, bytearray()
    header = {"__metadata__": attributes}
    ordered = {}
    for name, dtype, words in tensors:
        if not TENSOR_NAME.fullmatch(name):
            raise AsmError("OTS-E-ASM-004", f"tensor name {name!r} is not an identifier of at most 15 characters")
        if not words or any(len(w) != vl for w in words):
            raise AsmError("OTS-E-ASM-004", f"tensor {name}: every word must be {vl} bytes, and there must be one")
        if dtype is not None and dtype not in codes:
            raise AsmError("OTS-E-ASM-004", f"tensor {name}: unknown element type {dtype!r}")
        if name in ordered:
            raise AsmError("OTS-E-ASM-004", f"tensor name {name!r} occurs more than once")
        ordered[name] = dtype, words
    for name, (dtype, words) in sorted(ordered.items()):
        start = len(images)
        img = b"".join(words)
        images += img
        header[name] = {"dtype": "U8", "shape": [len(words), vl], "data_offsets": [start, len(images)]}
        attributes[f"dtype.{name}"] = str(RAW_DTYPE if dtype is None else codes[dtype])
        attributes[f"sha256.{name}"] = hashlib.sha256(img).hexdigest()
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + images


def read_weights(data, profile=None, config=None):
    """Read profile-bound raw SRAM word arrays from Safetensors."""
    def bad(message):
        raise AsmError("OTS-E-ASM-003", message)

    if len(data) < 8 or data[:4] == b"OTSW":
        bad("expected OpenTSP Safetensors; rebuild old weight packages")
    size, = struct.unpack_from("<Q", data)
    if not 0 < size <= min(SAFETENSORS_MAX_HEADER, len(data) - 8) or data[8:9] != b"{":
        bad("invalid Safetensors header length")
    try:
        header = json.loads(data[8:8 + size], object_pairs_hook=_unique_json)
    except (ValueError, RecursionError) as error:
        bad(f"invalid Safetensors metadata: {error}")
    attrs = header.pop("__metadata__", None)
    if not isinstance(attrs, dict) or any(not isinstance(v, str) for v in attrs.values()):
        bad("invalid OpenTSP Safetensors metadata")
    name = attrs.get("opentsp.profile")
    if not isinstance(name, str) or (profile is not None and name != profile):
        bad("Safetensors profile differs from the object")
    isa = ISA(name, config)
    vl = isa.pvars["vlanes"]
    codes = set(isa.t.eltype_codes.values()) | {RAW_DTYPE}
    data = memoryview(data)[8 + size:]
    tensors = {}
    ranges = []
    for name, item in header.items():
        offsets = item.get("data_offsets") if isinstance(item, dict) else None
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int or x < 0 for x in offsets):
            bad("invalid SRAM tensor offsets")
        start, end = offsets
        ranges.append((start, end, name, item))
    expected = 0
    for start, end, name, item in sorted(ranges):
        if not TENSOR_NAME.fullmatch(name) or item.get("dtype") != "U8":
            bad("invalid SRAM tensor")
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(x) is not int or x < 0 for x in shape):
            bad("invalid SRAM tensor shape")
        if shape[1] != vl or not shape[0] or not start < end <= len(data) or end - start != shape[0] * vl:
            bad(f"tensor {name}: invalid image length")
        dtype = attrs.get(f"dtype.{name}", "")
        if not re.fullmatch(r"\+?[0-9]{1,32}", dtype) or int(dtype) not in codes:
            bad(f"tensor {name}: invalid element type metadata")
        dt = int(dtype)
        if start != expected:
            bad("Safetensors payload has holes or overlaps")
        img = data[start:end]
        if hashlib.sha256(img).hexdigest() != attrs.get(f"sha256.{name}"):
            raise AsmError("OTS-E-ASM-004", f"tensor {name}: image does not match its digest")
        tensors[name] = (dt, [bytes(img[i:i + vl]) for i in range(0, len(img), vl)])
        expected = end
    if expected != len(data):
        bad("Safetensors payload has trailing bytes")
    return tensors


def object_preloads(data, profile, externs, config=None, *, event=1):
    """Resolve embedded data for cold load (1) or variant install (2)."""
    if is_text_input(data):
        return {}
    sections = _read_elf(data)
    regions = {x["name"]: x for x in externs}
    out = {}
    present = set()
    for section in (b".tsp.text", b".tsp.const"):
        if section in sections:
            for name, tensor in read_weights(sections[section], profile, config).items():
                if name in present or name not in regions or len(tensor[1]) != regions[name]["size"]:
                    raise AsmError("OTS-E-ASM-004", f"invalid embedded SRAM extent {name}")
                present.add(name)
                out[name] = tensor
    zs = sections.get(b".tsp.zero", b"")
    if len(zs) % 20:
        raise AsmError("OTS-E-ASM-003", "invalid ELF zero range directory")
    if zs:
        vl = ISA(profile, config).pvars["vlanes"]
        zero = bytes(vl)
        for raw, phases in struct.iter_unpack("<16sI", zs):
            name = _ascii_field(raw, "zero range")
            if name in present or name not in regions or phases not in (1, 3):
                raise AsmError("OTS-E-ASM-004", f"invalid embedded zero range {name}")
            present.add(name)
            if phases & event:
                out[name] = (RAW_DTYPE, [zero] * regions[name]["size"])
    return out


def merge_preloads(destination, external):
    for name, tensor in external.items():
        if name in destination:
            raise AsmError("OTS-E-ASM-004", f"duplicate tensor {name} across weight packages")
        destination[name] = tensor
    return destination


def bind_weights(externs, tensors):
    """The SRAM preloads (slice, addr, word) the loader writes before cycle 0: every extern
    takes the tensor of its name, of exactly its size (OTS-E-ASM-004 otherwise)."""
    pre = []
    for x in externs:
        t = tensors.get(x["name"])
        if t is None:
            raise AsmError("OTS-E-ASM-004", f"extern {x['name']}: no tensor of that name in the weight package")
        if len(t[1]) != x["size"]:
            raise AsmError("OTS-E-ASM-004", f"extern {x['name']}: tensor has {len(t[1])} words, the extern {x['size']}")
        pre += [(x["slice"], x["addr"] + k, w) for k, w in enumerate(t[1])]
    return pre


def preload_text(pre):
    """The harness S lines of SRAM preloads (lane 0 first)."""
    return "".join(f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in w) + "\n" for sl, a, w in pre)


def input_text(images):
    """The tsp_sim input lines of text images (c_emulator/tsp_sim.c): the Q lines."""
    return "".join(line + "\n" for line in q_lines(images))


def parse_images(text, path="<input>"):
    """Images from the tsp_sim text form `Q <queue> <unit> ...` (assembly.md sec. 4). Blank lines
    are skipped; any other line, a queue that is not 0..8f or ffff, or a unit that is not hex of at
    most 32 bits is OTS-E-ASM-003. Several lines for one queue append."""
    images = {}
    for ln, line in enumerate(text.split("\n"), 1):
        t = line.split()
        if not t:
            continue
        if t[0] != "Q" or len(t) < 2:
            raise AsmError("OTS-E-ASM-003", f"{path}:{ln}: expected 'Q <queue> <unit> ...', got {line.strip()!r}")
        for x in t[1:]:
            if not re.fullmatch(r"[0-9a-fA-F]+", x):
                raise AsmError("OTS-E-ASM-003", f"{path}:{ln}: {x!r} is not a hex value")
        q = int(t[1], 16)
        if not (0 <= q <= 143 or q == CONTROL) or any(int(x, 16) >> 32 for x in t[2:]):
            raise AsmError("OTS-E-ASM-003", f"{path}:{ln}: queue {t[1]!r} is not 0..8f or ffff, or a unit has more than 32 bits")
        images.setdefault(q, []).extend(int(x, 16) for x in t[2:])
    return {q: us for q, us in images.items() if us}


def is_text_input(data):
    """An input file is text when its first non-blank byte is 'Q' (or it is blank)."""
    s = data.lstrip(b" \t\r\n")
    return s[:1] == b"Q" or s == b""


def disassemble(isa, images, externs=()):
    """Program text of text images (raised with no waits, raise_images), and the number of
    instructions the model reports at decode or placement. Each such instruction is annotated
    with the model's code, in the model's order: for two or more instructions in one parcel, the
    slot rule (ISA.group_codes: QUE-001 on each when they do not fit, else MEM-001 on a clashing
    same-thread R + W pair); for a single instruction DEC-003 (decode_word), then QUE-002
    (placement). For instructions with no annotation, assemble(disassemble(i)) lowers to the
    same cycle view."""
    out = [f".arch {ARCH}", f".profile {isa.profile}"]
    out += [f".extern {x['name']}, slice={x['slice']}, addr={x['addr']:#x}, size={x['size']:#x}" for x in externs]
    out += [".section text"]
    bad = 0
    entries = raise_images(images)
    for c in sorted({e[0] for e in entries}):
        out.append(f"bundle @cycle={c} {{")
        rs = [e for e in entries if e[0] == c]
        for q in sorted({r[1] for r in rs}, key=lambda q: min(i for i, r in enumerate(rs) if r[1] == q)):
            group = [r for r in rs if r[1] == q]
            dec = []
            for _, _, w0, nu in group:
                code, e, f = isa.decode_word(q, w0, nu)
                if code:
                    width = 32 if nu == 1 else 64
                    text = f"illegal{width} {(w0 & 0xFFFFFFFF if width == 32 else w0):#x}   ; {code}"
                else:
                    text = isa.render(e, f)
                dec.append((code, e, w0, text, f))
            shares = isa.group_codes(q, [(d[1], d[4]) for d in dec])
            qn = queue_name(q)
            for (code, e, w0, text, _), share in zip(dec, shares):
                if share:                         # the model checks the group before decoding each instruction
                    bad += 1
                    text = text.split("   ;")[0] + f"   ; {share}"
                elif code:
                    bad += 1
                else:
                    pl = isa.placement(e, q)
                    if pl:
                        bad += 1
                        text += f"   ; {pl}"
                out.append(f"  {qn}: {text}")
        out.append("}")
    return "\n".join(out) + "\n", bad


# ---------------------------------------------------------------------------------------
# Manifests (.otsm; assembly.md sec. 4).
MMAGIC = b"OTSM"
# magic, major, minor, header bytes, profile, chip count K, topology, format revision, 0,
# encoding digest, profile digest: the same 64 bytes and digests as an object header
MHEADER = struct.Struct("<4sBBH16sIIII12s12s")
MENTRY = struct.Struct("<II")                 # offset and size of one embedded .elf object
MFORMAT_REVISION = 2                          # bumped by every change to the manifest format
TOPOLOGY_NODE = 0                             # the `tsp_sim --chips K` wiring (microarchitecture.md sec. 10)


MAX_WIRED_CHIPS = 264                         # the `--chips K` wiring: 33 nodes of 8 chips (ISCA22 sec. 2.2)
MAX_CHIPS = 512                               # chip_id is 9 bits


def chip_count_ok(k):
    """K chips of a manifest, as a reader takes them: 1..512 (the wiring is the harness's: `--chips K` for 1..264, a
    topology file up to 512)."""
    return 1 <= k <= MAX_CHIPS


def read_topology_chips(path):
    """The chip count of a topology file (`tsp_sim --topology`, topology.md sec. 7); the harness checks the rest."""
    try:
        t = json.load(open(path))
    except ValueError:
        raise AsmError("OTS-E-ASM-003", f"--topology {path}: not a JSON object") from None
    k = t.get("chips") if isinstance(t, dict) else None
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_CHIPS:
        raise AsmError("OTS-E-ASM-003", f"--topology {path}: \"chips\" must be an integer 1..{MAX_CHIPS}")
    return k


def write_manifest(profile, objects, config=None, max_chips=MAX_WIRED_CHIPS):
    """A manifest (.otsm): the header, K (offset, size) entries, then the K embedded .elf objects in
    chip order, back to back. Every object is read first (read_object), so it must be a valid
    object of `profile`. `max_chips`: 264 for the `--chips K` wiring, up to 512 for a topology file."""
    if not 1 <= len(objects) <= min(max_chips, MAX_CHIPS):
        what = "a topology file names" if max_chips > MAX_WIRED_CHIPS else "a manifest of the --chips wiring holds"
        raise AsmError("OTS-E-ASM-003", f"{len(objects)} chips: {what} 1..{min(max_chips, MAX_CHIPS)} objects")
    for i, o in enumerate(objects):
        try:
            read_object(o, profile, config)
        except AsmError as e:
            raise AsmError(e.code, f"chip {i}: {e.msg}") from None
    enc, prof = interface_digests(profile, config)
    off = MHEADER.size + len(objects) * MENTRY.size
    table = b""
    for o in objects:
        table += MENTRY.pack(off, len(o))
        off += len(o)
    head = MHEADER.pack(MMAGIC, *SPEC_VERSION, MHEADER.size, profile.encode(), len(objects),
                        TOPOLOGY_NODE, MFORMAT_REVISION, 0, enc, prof)
    return head + table + b"".join(objects)


def read_manifest(data, profile=None, config=None):
    """(profile, [object bytes, ...]) of a manifest. Rejects every other magic, version, revision,
    topology, a chip count the wiring cannot take, other digests, an entry that does not follow its
    predecessor, a size that does not match the file, and an embedded object that is not a valid
    object of the manifest's profile (OTS-E-ASM-003)."""
    if len(data) < MHEADER.size or data[:4] != MMAGIC:
        raise AsmError("OTS-E-ASM-003", "not an OTSM manifest (bad magic)")
    magic, major, minor, hsize, prof, k, topo, rev, z0, denc, dprof = MHEADER.unpack_from(data)
    if (major, minor) != SPEC_VERSION or hsize != MHEADER.size:
        raise AsmError("OTS-E-ASM-003", f"manifest is spec version {major}.{minor}, this tool reads 0.1 only")
    if rev != MFORMAT_REVISION or z0 != 0:
        raise AsmError("OTS-E-ASM-003", f"manifest format revision {rev} (reserved {z0:#x}); this tool reads revision {MFORMAT_REVISION}")
    if topo != TOPOLOGY_NODE:
        raise AsmError("OTS-E-ASM-003", f"manifest topology {topo}: 0 is the only topology of 0.1")
    prof = _ascii_field(prof, "manifest profile")
    if profile is not None and prof != profile:
        raise AsmError("OTS-E-ASM-003", f"manifest is for profile {prof!r}, not {profile!r}")
    load_profile(prof, config)
    enc, cur = interface_digests(prof, config)
    if denc != enc:
        raise AsmError("OTS-E-ASM-003", "manifest was built against another encoding.json (queue map or encodings)")
    if dprof != cur:
        raise AsmError("OTS-E-ASM-003", f"manifest was built against another configuration of profile {prof!r}")
    if not chip_count_ok(k):
        raise AsmError("OTS-E-ASM-003", f"manifest chip count {k}: must be 1..{MAX_CHIPS}")
    if len(data) < MHEADER.size + k * MENTRY.size:
        raise AsmError("OTS-E-ASM-003", "manifest size does not match its header")
    expect, objs = MHEADER.size + k * MENTRY.size, []
    for i in range(k):
        off, size = MENTRY.unpack_from(data, MHEADER.size + i * MENTRY.size)
        if off != expect or size == 0 or off + size > len(data):
            raise AsmError("OTS-E-ASM-003", f"manifest entry {i}: offset or size does not follow the previous entry")
        o = data[off:off + size]
        try:
            read_object(o, prof, config)
        except AsmError as e:
            raise AsmError(e.code, f"manifest entry {i}: {e.msg}") from None
        objs.append(o)
        expect = off + size
    if expect != len(data):
        raise AsmError("OTS-E-ASM-003", "manifest size does not match its entries")
    return prof, objs


def manifest_input(data, weight_packages, config=None):
    """The tsp_sim input of a manifest (with --chips K): for every chip i a "C i" line, the S lines
    of its weight package (when weight packages are given: exactly one per chip, in chip order) and
    the Q lines of its images, as `objinput` prints them for one object."""
    prof, objs = read_manifest(data, None, config)
    if weight_packages and len(weight_packages) != len(objs):
        raise AsmError("OTS-E-ASM-004", f"{len(weight_packages)} weight packages for {len(objs)} chips: one per chip")
    out = ""
    for i, o in enumerate(objs):
        _, images, exts = read_object(o, prof, config)
        out += f"C {i:x}\n"
        w = object_preloads(o, prof, exts, config)
        if weight_packages:
            merge_preloads(w, read_weights(weight_packages[i], prof, config))
            out += preload_text(bind_weights(exts, w))
        elif w:
            out += preload_text(bind_weights([x for x in exts if x["name"] in w], w))
        out += input_text(images)
    return out


# ---------------------------------------------------------------------------------------
# --config validation: the checks of tsp_sim and tsp-iss (c_emulator/tsp_config_check.c, ots-profile `Profile::validate`) on a profile file.
class _Obj(list):
    """A JSON object as [(key, value)] in file order, so that a duplicate key can be refused."""

    def get(self, k):
        for a, b in self:
            if a == k:
                return b
        return None


class _Num(str):
    """A JSON number as written."""


class ConfigError(Exception):
    pass


def _int_value(v):
    """An integer as Sail reads it: an optional '-' and decimal digits only, at most 18 of them."""
    if not isinstance(v, _Num):
        return None
    d = v[1:] if v.startswith("-") else v
    return int(v) if d.isdigit() and d.isascii() and len(d) <= 18 else None


def _schema_check(sch, v, key, known):
    k = key or "(top level)"
    dot = lambda n: f"{key}.{n}" if key else n
    for kw in sch:
        if kw not in known:
            raise ConfigError(f"schema: unsupported keyword \"{kw}\" at {key}")
    t = sch.get("type")
    if t == "object" or (t is None and "properties" in sch):
        if not isinstance(v, _Obj):
            raise ConfigError(f"{k}: expected an object")
        props = sch.get("properties", {})
        for n, (name, f) in enumerate(v):
            if any(g == name for g, _ in v[:n]):
                raise ConfigError(f"{dot(name)}: duplicate key")
            if name not in props:
                if sch.get("additionalProperties") is False:
                    raise ConfigError(f"{dot(name)}: unknown key")
            else:
                _schema_check(props[name], f, dot(name), known)
        for r in sch.get("required", []):
            if v.get(r) is None:
                raise ConfigError(f"{dot(r)}: missing")
    elif t == "array":
        if not isinstance(v, list) or isinstance(v, _Obj):
            raise ConfigError(f"{k}: expected an array")
        if "items" in sch:
            for n, e in enumerate(v):
                _schema_check(sch["items"], e, f"{key}[{n}]", known)
    elif t == "integer":
        x = _int_value(v)
        if x is None:
            raise ConfigError(f"{k}: expected an integer (decimal digits, at most 18)")
        between = f" (must be between {sch['minimum']} and {sch['maximum']})" if "minimum" in sch and "maximum" in sch else ""
        if "minimum" in sch and x < sch["minimum"]:
            raise ConfigError(f"{k}: {x} is below the minimum {sch['minimum']}{between}")
        if "maximum" in sch and x > sch["maximum"]:
            raise ConfigError(f"{k}: {x} is above the maximum {sch['maximum']}{between}")
        if "const" in sch and x != sch["const"]:
            raise ConfigError(f"{k}: must be {sch['const']}")
        if "multipleOf" in sch and x % sch["multipleOf"] != 0:
            raise ConfigError(f"{k}: {x} is not a multiple of {sch['multipleOf']}")
    elif t == "string":
        if not isinstance(v, str) or isinstance(v, _Num):
            raise ConfigError(f"{k}: expected a string")
        if "const" in sch and v != sch["const"]:
            raise ConfigError(f"{k}: must be \"{sch['const']}\"")
    elif t == "boolean":
        if not isinstance(v, bool):
            raise ConfigError(f"{k}: expected true or false")
    elif t is not None:
        raise ConfigError(f"schema: unsupported type {t} at {k}")
    for sub in sch.get("allOf", []):
        _schema_check(sub, v, key, known)
    if "anyOf" in sch:
        first = None
        for sub in sch["anyOf"]:
            try:
                _schema_check(sub, v, key, known)
                break
            except ConfigError as e:
                if str(e).startswith("schema:"):
                    raise
                first = first or str(e)
        else:
            if "x-opentsp-why" in sch:
                raise ConfigError(f"{k}: {sch['x-opentsp-why']}")
            consts = [s.get("const") for s in sch["anyOf"]]
            if consts and all(isinstance(c, int) and not isinstance(c, bool) for c in consts):
                raise ConfigError(f"{k}: must be one of {', '.join(str(c) for c in consts)}")
            raise ConfigError(f"{k}: none of the allowed alternatives ({first})")


def _lookup(cfg, dotted):
    for p in dotted.split("."):
        if not isinstance(cfg, _Obj):
            return None
        cfg = cfg.get(p)
    return cfg


def _relation_check(sch, cfg):
    for r in sch.get("x-opentsp-relations", []):
        a, b, op, off = r["a"], r["b"], r["op"], r.get("offset", 0)
        if r.get("quads", 0):
            off += r["quads"] * (_int_value(_lookup(cfg, "geometry.mem_slices_per_hemisphere")) // 4)
        x, y = _int_value(_lookup(cfg, a)), _int_value(_lookup(cfg, b))
        ok = {"<": x < y + off, "<=": x <= y + off, "==": x == y + off, "!=": x != y + off}[op]
        if not ok:
            raise ConfigError(f"{a}: must be {op} {b}{' %+d' % off if off else ''} ({a} = {x}, {b} = {y}): {r['why']}")


_TIMING_KEYS = {"CU": "cu", "MEM": "mem", "VXM": "vxm", "MXM_LW": "mxm_lw", "MXM_IW": "mxm_iw", "MXM_ABC": "mxm_abc", "MXM_ACC": "mxm_acc",
                "SXM": "sxm", "SXM_CROSS": "sxm_cross", "C2C": "c2c"}


# The extensions of this version (the model's extension_error, postlude/timing_check.sail).
_EXTENSIONS = ("bf16", "wide_acc", "block_fp8")


def _timing_error(cfg):
    """The rules of the model's timing_config_error (postlude/timing_check.sail) on the effective rows, the first violation."""
    db = load_encoding()
    return _timing_error_plain(_plain(cfg), db)


def _timing_error_plain(cfg, db):
    t, sl = cfg["timing"], cfg["geometry"]["superlanes"]
    exts = cfg.get("extensions", [])
    for k, x in enumerate(exts):
        if x not in _EXTENSIONS:
            return f"extensions: {x} is not an extension of this version"
        if x in exts[k + 1:]:
            return f"extensions: {x} is listed more than once"
    entries = {i["name"]: _TIMING_KEYS[i["timing"]] for i in db["instructions"]}
    # An override row's entry is "<entry>" or "<entry>.<element type>" (the row of that entry at that type); an entry takes
    # the element types of the codes of its `type` operand.
    takes = {i["name"]: {c for v in i["variables"] if v["name"] == "type" and v["kind"] == "eltype_code" for c in v["codes"]}
             for i in db["instructions"]}
    typed = {f"{n}.{tn}": (n, tn, code in takes[n]) for n in entries for tn, code in db["eltype_codes"].items()}
    ov = {}
    names = [r["entry"] for r in t["overrides"]]
    for r in t["overrides"]:
        e = r["entry"]
        if e not in entries:
            if e not in typed:
                return f"timing.overrides: {e} is not a catalogue entry"
            if not typed[e][2]:
                return f"timing.overrides: {e}: {typed[e][0]} does not take element type {typed[e][1]}"
        if names.count(e) > 1:
            return f"timing.overrides: {e} is listed more than once"
        if e in entries:
            ov[e] = r
    row = lambda n: ov.get(n) or t[entries[n]]
    key = lambda n: f"timing.overrides.{n}" if n in ov else f"timing.{entries[n]}"
    slices = []
    for i in db["instructions"]:
        if i["slice"] not in slices:
            slices.append(i["slice"])
    for i in sorted(db["instructions"], key=lambda i: (slices.index(i["slice"]), i["name"].lower())):
        n, r = i["name"], row(i["name"])
        if entries[n] != "cu" and not r["d_skew"] < r["d_func"]:
            return (f"{key(n)}: {n}: d_skew {r['d_skew']} must be below d_func {r['d_func']} "
                    "(an instruction commits after it executes at dispatch + d_skew)")
    for r in t["overrides"]:
        if r["entry"] not in entries and not r["d_skew"] < r["d_func"]:
            return (f"timing.overrides.{r['entry']}: {r['entry']}: d_skew {r['d_skew']} must be below d_func {r['d_func']} "
                    "(an instruction commits after it executes at dispatch + d_skew)")
    for n in ("Shift", "Permute", "ABC", "ABC8"):
        r = row(n)
        if r["d_func"] < r["d_skew"] + sl:
            return (f"{key(n)}: {n}: d_func {r['d_func']} must be at least geometry.superlanes {sl} plus d_skew {r['d_skew']} "
                    "(later-superlane operands must arrive before output)")
    icu = cfg["icu"]
    if icu["fetch_bytes_per_cycle"] > 0 and any(1 + icu["fetch_latency"] + 16 * (b * sl + k) // icu["fetch_bytes_per_cycle"] <= b + k
                                               for b in range(2) for k in range(sl)):
        return "icu.fetch_bytes_per_cycle: instruction text delivery precedes source tile capture"
    if t["sxm_map_ready"] < sl:
        return "timing.sxm_map_ready: must be at least geometry.superlanes (map dissemination)"
    if t["mxm_abc"]["occupancy_f16"] > row("ABC")["d_func"]:
        return (f"timing.mxm_abc.occupancy_f16: {t['mxm_abc']['occupancy_f16']} must be at most ABC's d_func {row('ABC')['d_func']} "
                "(macc_busy finds the f16 entry that still holds the plane only while it has not committed)")
    acc_bits = cfg["mxm"]["acc_bits"]
    if acc_bits not in (32, 128):
        return f"mxm.acc_bits: {acc_bits} must be 32 or 128"
    if ("wide_acc" in exts) != (acc_bits == 128):
        return "extensions: wide_acc is declared exactly when mxm.acc_bits is 128"
    s0 = row("Read")["d_skew"]
    for n in ("Read", "Write", "Gather", "Scatter", "IfetchM"):
        if row(n)["d_skew"] != s0:
            return (f"{key(n)}: {n}: d_skew {row(n)['d_skew']} must equal Read's {s0} "
                    "(Read, Write, Gather, Scatter and IfetchM share the SRAM ports' dispatch rule)")
    rows, f16 = cfg["mxm"]["rows"], cfg["mxm"]["f16_rows"]
    if not (2 * f16 >= rows and f16 <= rows):
        return f"mxm.f16_rows: {f16} must be between mxm.rows / 2 and mxm.rows ({rows // 2} and {rows})"
    return None


def validate_config(path):
    """Raise ConfigError (`<key>: <reason>`) when the profile file `path` is refused by tsp_sim and tsp-iss; an unreadable file is left to the command."""
    try:
        data = open(path, "rb").read()
    except OSError:
        return
    if len(data) >= 1 << 25:
        raise ConfigError("file too large")
    if b"\0" in data:
        raise ConfigError("NUL byte in file")

    def bad(c):
        raise ConfigError("not valid JSON (NaN or Infinity)")
    try:
        cfg = json.loads(data.decode("utf-8"), object_pairs_hook=_Obj, parse_int=_Num, parse_float=_Num, parse_constant=bad)
    except (ValueError, RecursionError) as e:
        raise ConfigError(f"not valid JSON ({e})") from None
    schema = json.load(open(os.path.join(SPEC, "config", "schema.json")), object_pairs_hook=_Obj)
    sch = _plain(schema)
    known = {"$schema", "type", "properties", "items", "required", "additionalProperties", "const", "minimum", "maximum", "multipleOf",
             "anyOf", "allOf", "description", "default", "x-opentsp-relations", "x-opentsp-why"}
    _schema_check(sch, cfg, "", known)
    _relation_check(sch, cfg)
    err = _timing_error(cfg)
    if err:
        raise ConfigError(err)


_CAP_SECTIONS = ("geometry", "sram", "mxm")
CHECK_CAPS = True          # main() switches it off for --no-validate, as it skips the full validation


_CHECKED = {}


def check_config(path):
    """The full validation of the command line (validate_config) for the import surface (ISA, assemble, write_object): ConfigError with
    its message. The result is kept per file (path, size, modification time), as a script builds many ISAs from one profile."""
    try:
        st = os.stat(path)
    except OSError:
        return
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key in _CHECKED:
        return
    validate_config(path)
    _CHECKED[key] = True


def check_caps(path):
    """The caps on the keys that size the instruction grammar and the memory (the geometry, sram and mxm sections of the schema: 1 to 32
    superlanes, the words per slice and their bank bit, the MEM slices, the MXM rows), as validate_config checks them: ConfigError with
    its message. A file that cannot be read or parsed is left to the caller."""
    try:
        cfg = json.loads(open(path, "rb").read().decode("utf-8"), object_pairs_hook=_Obj, parse_int=_Num, parse_float=_Num)
        schema = _plain(json.load(open(os.path.join(SPEC, "config", "schema.json")), object_pairs_hook=_Obj))
    except (OSError, ValueError, RecursionError):
        return
    if not isinstance(cfg, _Obj):
        return
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k in _CAP_SECTIONS}
    schema["required"] = [k for k in schema.get("required", []) if k in _CAP_SECTIONS]
    schema.pop("x-opentsp-relations", None)
    known = {"$schema", "type", "properties", "items", "required", "additionalProperties", "const", "minimum", "maximum", "multipleOf",
             "anyOf", "allOf", "description", "default", "x-opentsp-relations", "x-opentsp-why"}
    _schema_check(schema, _Obj([(k, v) for k, v in cfg if k in _CAP_SECTIONS]), "", known)


def _plain(x):
    """The schema as dicts and lists (its numbers as ints)."""
    if isinstance(x, _Obj):
        return {k: _plain(v) for k, v in x}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, _Num):
        return int(x)
    return x


def main(argv):
    args, opts = [], {}
    wlist = []                                  # every --weights, in order (mobjinput: one per chip)
    i = 1
    while i < len(argv):
        if argv[i] in ("-o", "--profile", "--config", "--weights", "--topology"):
            if i + 1 >= len(argv):
                print(f"tsp_as: option {argv[i]} needs a value", file=sys.stderr)
                return 2
            opts[argv[i]] = argv[i + 1]
            if argv[i] == "--weights":
                wlist.append(argv[i + 1])
            i += 2
        elif argv[i].startswith("--"):
            opts[argv[i]] = True
            i += 1
        else:
            args.append(argv[i])
            i += 1
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    cmd, rest = args[0], args[1:]
    profile = opts.get("--profile", "ref")
    global CHECK_CAPS
    saved, CHECK_CAPS = CHECK_CAPS, "--no-validate" not in opts
    try:
        return _run(cmd, rest, args, opts, wlist, profile)
    finally:
        CHECK_CAPS = saved


def _run(cmd, rest, args, opts, wlist, profile):
    if "--config" in opts and "--no-validate" not in opts:
        try:
            validate_config(opts["--config"])
        except ConfigError as e:
            print(f"tsp_as: config {opts['--config']}: {e}", file=sys.stderr)
            return 2
    try:
        if cmd == "insn":
            e, w = ISA(profile, opts.get("--config")).parse(" ".join(rest)) if len(rest) != 1 else \
                ISA(profile, opts.get("--config")).parse(rest[0])
            print(e["width"], f"{w:x}")
        elif cmd == "word":
            isa = ISA(profile, opts.get("--config"))
            try:
                width, w = int(rest[0], 10), int(rest[1], 16)
            except (ValueError, IndexError):
                raise AsmError("OTS-E-ASM-001", "expected <width> <hex>, two integers") from None
            if width not in (32, 64):
                raise AsmError("OTS-E-ASM-001", f"width {width} is not 32 or 64")
            if w < 0 or w >> width:
                raise AsmError("OTS-E-ASM-001", f"value {w:#x} does not fit in {width} bits")
            print(isa.disasm(width, w))
        elif cmd == "asm":
            src = open(rest[0], "rb").read().decode("latin-1")    # non-ASCII is reported per line
            isa, prog = assemble(src, rest[0], opts.get("--config"), warn=lambda m: print(m, file=sys.stderr))
            if "--input" in opts:
                sys.stdout.write(input_text(prog.images))
            if "-o" in opts:
                open(opts["-o"], "wb").write(write_object(prog, opts.get("--config")))
        elif cmd in ("dis", "objinput"):
            data = open(rest[0], "rb").read()
            if is_text_input(data):
                try:
                    txt = data.decode("ascii")
                except UnicodeDecodeError:
                    raise AsmError("OTS-E-ASM-003", f"{rest[0]}: text input is not ASCII") from None
                images, exts, prof = parse_images(txt, rest[0]), [], profile
            else:
                prof, images, exts = read_object(data, opts.get("--profile"), opts.get("--config"))
            if cmd == "objinput":
                config = opts.get("--config")
                w = object_preloads(data, prof, exts, config)
                if "--weights" in opts:
                    merge_preloads(w, read_weights(open(opts["--weights"], "rb").read(), prof, config))
                if w or "--weights" in opts:
                    sys.stdout.write(preload_text(bind_weights(exts, w)))
                sys.stdout.write(input_text(images))
                return 0
            text, bad = disassemble(ISA(prof, opts.get("--config")), images, exts)
            sys.stdout.write(text)
            return 1 if bad else 0
        elif cmd == "mpack":
            # mpack --profile P -o out.otsm chip0.elf chip1.elf ... (K objects of profile P)
            objs = [open(f, "rb").read() for f in rest]
            if "--topology" in opts:               # the topology file gives the chip count: the objects must be that many
                k = read_topology_chips(opts["--topology"])
                if len(objs) != k:
                    raise AsmError("OTS-E-ASM-003", f"--topology {opts['--topology']} names {k} chips but {len(objs)} objects were given")
                data = write_manifest(profile, objs, opts.get("--config"), MAX_CHIPS)
            else:
                data = write_manifest(profile, objs, opts.get("--config"))
            open(opts["-o"], "wb").write(data)
        elif cmd == "mobjinput":
            # mobjinput <m.otsm> [--weights chip0.safetensors --weights chip1.safetensors ...]: the `C <chip>` input
            sys.stdout.write(manifest_input(open(rest[0], "rb").read(), [open(f, "rb").read() for f in wlist], opts.get("--config")))
        elif cmd == "wpack":
            # wpack --profile P -o out.safetensors name=dtype:file ... (dtype an element type or raw;
            # file: the raw bytes, a whole number of vlanes-byte words, lane 0 first)
            vl = ISA(profile, opts.get("--config")).pvars["vlanes"]
            specs = []
            for a in rest:                     # every argument is checked before any file is read
                name, eq, spec = a.partition("=")
                dtype, colon, path = spec.partition(":")
                if not (eq and colon and path):
                    print(f"tsp_as: wpack: {a!r}: expected name=dtype:file", file=sys.stderr)
                    print(__doc__, file=sys.stderr)
                    return 2
                specs.append((name, dtype, path))
            ts = []
            for name, dtype, path in specs:
                raw = open(path, "rb").read()
                if len(raw) % vl:
                    raise AsmError("OTS-E-ASM-004", f"{path}: {len(raw)} bytes is not a whole number of {vl}-byte words")
                ts.append((name, None if dtype == "raw" else dtype, [raw[k:k + vl] for k in range(0, len(raw), vl)]))
            if "-o" not in opts:
                print("tsp_as: wpack: -o <w.safetensors> is required", file=sys.stderr)
                print(__doc__, file=sys.stderr)
                return 2
            data = write_weights(profile, ts, opts.get("--config"))   # validates everything first
            with open(opts["-o"], "wb") as fh:
                fh.write(data)
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except AsmError as e:
        print(f"{getattr(e, 'where', cmd)}: {e.code}: {e.msg}", file=sys.stderr)
        return 1
    except OSError as e:                       # a file that is missing or cannot be read or written
        print(f"{cmd}: OTS-E-ASM-003: {e.filename or ''}{': ' if e.filename else ''}{e.strerror or e}", file=sys.stderr)
        return 1
    except IndexError:                         # a missing argument
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
