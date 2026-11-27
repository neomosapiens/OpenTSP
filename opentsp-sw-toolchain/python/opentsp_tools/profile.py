"""Profiles: config/<name>.json of the specification. No latency, gap or size is hard-coded anywhere else."""
import json
import os

from . import paths
ROOT, SPEC = paths.ROOT, paths.SPEC
ENCODING = os.path.join(SPEC, "isa-db", "generated", "encoding.json")

_encoding = None


def encoding():
    global _encoding
    if _encoding is None:
        with open(ENCODING) as f:
            _encoding = json.load(f)
    return _encoding


def entry(mnemonic):
    """The encoding.json record of an instruction, by its assembly mnemonic (`read`, `add`, ...)."""
    for i in encoding()["instructions"]:
        if i["mnemonic"] == mnemonic:
            return i
    raise KeyError(mnemonic)


def _type_field(e):
    """The index of the element type among the dot-separated fields of an instruction text (`{type}` of the assembly template), or None."""
    fields = e["assembly"].split(" ")[0].split(".")
    return next((i for i, f in enumerate(fields) if f.startswith("{type}")), None)


class UntypedTiming(ValueError):
    """A timing was asked of an entry that has typed rows without the element type."""

    def __str__(self):
        return self.args[0]


class Profile:
    def __init__(self, name):
        self.name = name                                  # "ref", "mini", "test/mid_min", ...
        self.path = os.path.join(SPEC, "config", name + ".json")
        with open(self.path) as f:
            self.cfg = json.load(f)
        g, s = self.cfg["geometry"], self.cfg["sram"]
        self.asm_name = os.path.basename(name)            # the `.profile` name; test profiles need --config
        self.superlanes = g["superlanes"]
        self.vlanes = 16 * g["superlanes"]
        self.streams = g["streams_per_row"]
        self.positions = g["srf_positions"]
        self.slices_per_hemisphere = g["mem_slices_per_hemisphere"]
        self.words = s["words_per_slice"]
        self.bank_bit = s["bank_bit"]
        self.addr_bits = s["addr_bits"]                   # the width of an SRAM address: a slice has at most 2^addr_bits words
        self.mxm_rows = self.cfg["mxm"]["rows"]
        self.layout = self.cfg["layout"]
        self.release_latency = self.cfg["barrier"]["release_latency"]
        self.icu = self.cfg["icu"]

    @property
    def is_test(self):
        return "/" in self.name

    @property
    def tag(self):
        """The profile as it appears in file names: test/mid_min -> mid_min."""
        return self.asm_name

    def timing(self, mnemonic, dtype=None):
        """(d_func, d_skew) of an instruction: the row `<entry>.<dtype>` of `timing.overrides` when `dtype` (the element type, spelled as the
        assembly spells it: `i8`, `f32`) is given and such a row exists, else the entry's own `timing.overrides` row, else its class row
        (one row per catalogue entry, a row per element type). An operation written as the assembly writes it names the row of its
        timing type: the field `{type}` of the assembly template (Sail `timing_type_of`), so `timing("max.i8")` and
        `timing("exp.i8.sg4")` take i8, and `timing("convert.u8.f32")` takes the *source* type u8 (not the destination f32 in
        `{dtype}`). `dtype=""` asks for the entry's own row on purpose. Without any type, an entry that the profile gives typed
        rows is refused (`UntypedTiming`): the entry's row would be a guess for the type of the caller, and a schedule built on
        it disagrees with the model."""
        if dtype is None and "." in mnemonic:
            parts = mnemonic.split(".")
            mnemonic = parts[0]
            at = _type_field(entry(mnemonic))
            dtype = parts[at] if at is not None and at < len(parts) else None
        e = entry(mnemonic)
        t = self.cfg["timing"]
        overrides = t.get("overrides", [])
        names = (e["name"], e["mnemonic"])
        if dtype is None and any(o["entry"].rpartition(".")[0] in names for o in overrides):
            raise UntypedTiming(f"timing({mnemonic!r}): profile {self.name} has typed rows for {e['name']}; name the element type "
                                f"(`timing({mnemonic + '.<type>'!r})`), or `dtype=\"\"` for the entry's own row")
        if dtype:
            keys = (e["name"] + "." + dtype, e["mnemonic"] + "." + dtype)
            for o in overrides:
                if o["entry"] in keys:
                    return o["d_func"], o["d_skew"]
        for o in overrides:
            if o["entry"] in names:
                return o["d_func"], o["d_skew"]
        row = t[e["timing"].lower()]
        return row["d_func"], row["d_skew"]

    def bank(self, addr):
        return (addr >> self.bank_bit) & 1

    def slices(self, hemisphere):
        """The present MEM queues of a hemisphere ('W' or 'E')."""
        base = 0 if hemisphere == "W" else 44
        return list(range(base, base + self.slices_per_hemisphere))


_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = Profile(name)
    return _cache[name]
