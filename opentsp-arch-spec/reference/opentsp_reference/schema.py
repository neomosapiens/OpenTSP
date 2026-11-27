"""Complete the configuration schema that Sail generates (--output-schema) with the
invariants Sail's schema output cannot express .

Sail writes one JSON Schema property per `config` key, with the bounds of its type. It
does not emit: closed objects (unknown keys are accepted), the specification version,
constraints that relate two keys (bank_bit to words_per_slice; the order of the layout
gaps; the C2C latency range and the HAC period), bounds tighter than a key's Sail type, or the message to print when a value is rejected. This script adds them, so that
config/schema.json is the complete contract between a profile and the model. Every harness
(c_emulator/tsp_sim.c through c_emulator/tsp_config_check.c) checks the configuration
against it before the model runs; test/config/fuzz_sim.py tests that a profile which
passes it does not reach a model assert (a test, not a proof).

usage: schema_invariants.py <schema.json>     (rewrites the file in place)
       schema_invariants.py --check <config.json>...    (exit 1 on a violated invariant; an `advisory:` line is only a warning)
       schema_invariants.py --normalise <config.json> <out.json>   (write the profile with every omitted defaulted key set to its default:
                                    the form Sail's own backends need, because they reject a `config` key that the file leaves out)
"""
import json, sys

SPEC_VERSION = "0.1"

# The element types that exist only in a profile that declares an extension (core/types.sail: bf16 is defined with the extension bf16).
TYPE_EXTENSION = {"bf16": "bf16"}

# bank_bit is the upper-most SRAM address bit of the words (with ISCA 2020 13-bit addressing that is bit 12). opentsp decision: a profile with
# fewer or more words keeps that rule, so bank_bit = log2(words_per_slice) - 1 and both banks hold
# half of the slice. A lower bit would interleave the banks instead; a higher bit would put
# every address in bank 0 and serialise every Read/Write pair. words_per_slice is
# any power of two from 256 to 65536; sram.addr_bits (13 to 16) is the width of the address, so a slice has at most 2^addr_bits words:
# the relation bank_bit < addr_bits below says so.
BANK_BIT = {2 ** (b + 1): b for b in range(7, 16)}

# Layout gaps: the floorplan order of ISCA 2020 Fig. 4, read outward from the VXM at the
# centre of a row. read_gap/write_gap (core/layout.sail) place MEM quad q (0 .. Q - 1, with
# Q = geometry.mem_slices_per_hemisphere / 4 quads per hemisphere, 11 in ref) at
# mem_up_gap_base - q and mem_down_gap_base + q; those must stay inside 0..43, and the units
# must keep their order, or some units become unreachable (OTS-E-DIR-001 from a valid
# program). Each entry: a op b + offset (+ Q if the entry has "quads": 1, - Q if -1).
MEM_SLICES = "geometry.mem_slices_per_hemisphere"
RELATIONS = [
    ("sram.bank_bit", "<", "sram.addr_bits", 0, "a slice has at most 2^addr_bits words, so the bank bit (log2(words_per_slice) - 1) is below addr_bits"),
    ("layout.c2c_rx_gap", "<", "layout.mem_up_gap_base", 1, "the C2C receiver is outside the last MEM quad on the inward row", -1),
    ("layout.mem_up_gap_base", "<", "layout.vxm_gap", 0, "MEM (inward) is between the C2C receiver and the VXM"),
    ("layout.vxm_gap", "<", "layout.mem_down_gap_base", 0, "the VXM is between the inward and outward MEM gaps"),
    ("layout.mem_down_gap_base", "<", "layout.c2c_tx_gap", 1, "the last MEM quad (outward) is before the C2C transmitter", -1),
    ("layout.c2c_tx_gap", "<", "layout.sxm_dst_gap", 0, "the C2C transmitter is before the SXM distributor"),
    ("layout.sxm_dst_gap", "<", "layout.sxm_turn_in_gap", 0, "the SXM distributor is before the SXM turn"),
    ("layout.sxm_turn_in_gap", "<", "layout.mxm_in_gap", 0, "the SXM turn is before the MXM input"),
    ("layout.mxm_acc_gap", "<", "layout.sxm_turn_out_gap", 0, "the MXM result enters the row before the SXM turn output"),
    ("layout.sxm_turn_out_gap", "<", "layout.c2c_rx_gap", 0, "the SXM turn output is before the C2C receiver"),
]


def close(node):
    """additionalProperties: false on every object schema Sail generated. complete() calls
    this before it adds the anyOf alternatives, which constrain some fields only."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
        for v in node.values():
            close(v)
    elif isinstance(node, list):
        for v in node:
            close(v)


# Bounds Sail does not emit, because of how c2c_insts.sail and timing.sail type these keys
# (link_latency* are nats, links and hac_period are checked by an assert, d_func is
# range(0, 63) for every class). opentsp decisions:
#   c2c.links 1..128: the link field of Send/Receive/Transmit is 7 bits, 4 and 3 more in the
#     low reserved bits (c2c_insts.sail c2c_links; ISCA20 sec. II: 16 links on the published
#     chip) and a chip has at least one link (N-2).
#   c2c.link_latency, link_latency_min, link_latency_max 1..65535: a link takes at least
#     one cycle (tsp_sim --chips needs it, , ), and no configuration can push
#     an arrival past the emulator's cycle range.
#   c2c.hac_period 1..65535: the hand-shake period is positive (the model asserts it).
#   c2c.cycles_per_vector 1..65535 and c2c.rx_depth 1..65535: a link accepts at
#     least one vector per cycle_per_vector cycles (1 = no bandwidth limit beyond one per
#     cycle) and a receive FIFO holds at least one vector; the upper bound keeps every
#     link time inside the emulator's cycle range.
#   timing.<class>.d_func >= 1 for every class except cu (decision timing.d_func_min,
#     ): tick (postlude/tick.sail) applies the commits due at cycle c BEFORE it
#     dispatches the records of c, so a stream result scheduled at dispatch + 0 would land
#     in a ring slot already drained and commit 128 cycles late. A result therefore
#     commits at dispatch + 1 at the earliest (ISCA20 sec. III: an instruction's result is
#     available d_func after dispatch; no source gives d_func 0). The MXM LW/IW/ABC
#     classes keep the same minimum so that every stage's readiness (ready_lw, ready_iw,
#     the ABC commit) is strictly after its dispatch. The control unit's instructions write
#     no stream and have no readiness, so timing.cu.d_func keeps 0..63.
D_FUNC_MIN_CLASSES = ["c2c", "mem", "mxm_abc", "mxm_acc", "mxm_iw", "mxm_lw", "sxm", "sxm_cross", "vxm"]
BOUNDS = {
    ("c2c", "links"): (1, 128),
    ("c2c", "link_latency"): (1, 65535),
    ("c2c", "link_latency_min"): (1, 65535),
    ("c2c", "link_latency_max"): (1, 65535),
    ("c2c", "hac_period"): (1, 65535),
    ("c2c", "cycles_per_vector"): (1, 65535),
    ("c2c", "rx_depth"): (1, 65535),
    ("timing", "sxm_map_ready"): (1, 63),
    ("timing", "mxm_abc", "occupancy_f16"): (1, 63),
    ("timing", "c2c", "rtdeskew_floor"): (1, 65535),
    **{("timing", k, "d_func"): (1, 63) for k in D_FUNC_MIN_CLASSES},
}

# The widths mxm.acc_bits takes (core/geometry.sail: 32, or 128 with the extension wide_acc).
ACC_BITS = (32, 128)

# Keys a profile may leave out: the loader (c_emulator/tsp_config_check.c, the Rust ots-profile) reads a missing key as its default, so a
# profile written before the key existed loads unchanged. ras.diag_depth is the entries of the chip diagnostic log; 16 is the depth of
# every shipped profile (the minimum depth of the platform error FIFO). The schema keyword is the standard "default"; the key leaves the
# "required" list of its object.
DEFAULTS = {("ras", "diag_depth"): 16}

def with_defaults(cfg):
    """The normalised configuration: cfg with every omitted key that has a default set to it (a new dictionary). The configuration digest of the
    golden vectors is taken over this form, so a profile that leaves a defaulted key out has the digest of the same profile with the default."""
    out = json.loads(json.dumps(cfg))
    for (section, key), value in DEFAULTS.items():
        if isinstance(out.get(section), dict):
            out[section].setdefault(key, value)
    return out


# Keys that must be a multiple of a unit: the instruction queue holds and fetches 32-bit
# text units (assembly.md, instruction text): a fill rate or a depth that
# is not a multiple of 4 bytes would split a unit.
MULTIPLES = {
    ("icu", "fetch_bytes_per_cycle"): 4,
    ("icu", "iq_depth"): 4,
    # MEM slices come in quads (the unit of distance from the VXM); LW and IW from
    # streams move 16 stream values at a time.
    ("geometry", "mem_slices_per_hemisphere"): 4,
    ("mxm", "rows"): 16,
    # An f16 row is two stream values, and LW and IW from streams move 16 of them: f16 capacities come in 8 rows.
    ("mxm", "f16_rows"): 8,
}

# Relations between the C2C keys (, N-2). The harness draws each link's latency
# in [link_latency_min, link_latency_max] around the nominal link_latency (ISCA22 Table 2:
# min/mean/max 209/216/228). Deskew determinism (c2c_insts.sail; microarchitecture.md
# sec. 10) needs every link latency below the HAC period: a vector sent at an epoch
# boundary b has arrived by b + P (ISCA22 sec. 3.2; footnote 4: P = 252 > 228).
C2C_RELATIONS = [
    ("c2c.link_latency_min", "<=", "c2c.link_latency", 0, "the nominal link latency is inside the published range"),
    ("c2c.link_latency", "<=", "c2c.link_latency_max", 0, "the nominal link latency is inside the published range"),
    ("c2c.link_latency_max", "<", "c2c.hac_period", 0, "every link latency is below the HAC period, so Deskew is deterministic"),
]

# Shift and Permute (timing class sxm_cross). Cycle numbers are superlane-0 times
# (topology.md sec. 3): superlane k of an operand reaches the unit k cycles after superlane 0.
# An output lane of superlane j may take its value from superlane k > j, so the result can be
# complete only geometry.superlanes cycles after dispatch. A smaller d_func would describe a
# unit that uses data before it arrives.
# An instruction executes (reads its operands) at dispatch + d_skew and commits at dispatch +
# d_func; tick applies a cycle's commits before it
# executes that cycle's elements, so every class outside the control unit needs d_skew < d_func.
# The same rules on the per-entry rows (timing.overrides), and the shared d_skew of the SRAM
# references, are checked by the model itself (postlude/timing_check.sail timing_config_error,
# which every harness runs after loading the configuration).
TIMING_RELATIONS = [
    ("geometry.superlanes", "<=", "timing.sxm_cross.d_func", 0,
     "Shift and Permute read lanes of later superlanes, which arrive up to superlanes - 1 cycles after superlane 0"),
    # An ABC product sums the lanes of every superlane in the same way.
    ("geometry.superlanes", "<=", "timing.mxm_abc.d_func", 0,
     "ABC sums lanes of every superlane, which arrive up to superlanes - 1 cycles after superlane 0"),
    # An f16 ABC holds the plane only while it has not committed (macc_busy).
    ("timing.mxm_abc.occupancy_f16", "<=", "timing.mxm_abc.d_func", 0,
     "an f16 ABC holds its plane for at most d_func cycles, the time before its product commits"),
] + [
    (f"timing.{k}.d_skew", "<", f"timing.{k}.d_func", 0,
     "an instruction executes at dispatch + d_skew and commits after that, at dispatch + d_func")
    for k in D_FUNC_MIN_CLASSES
]


def complete(schema):
    close(schema)
    p = schema["properties"]
    for path, (lo, hi) in BOUNDS.items():
        node = p
        for k in path[:-1]:
            node = node[k]["properties"]
        node[path[-1]].update(minimum=lo, maximum=hi)
    for path, m in MULTIPLES.items():
        p[path[0]]["properties"][path[1]]["multipleOf"] = m
    # Sail types mxm.acc_bits as the range 32..128; the accumulator is 32 or 128 bits wide, nothing between.
    p["mxm"]["properties"]["acc_bits"]["anyOf"] = [{"type": "integer", "const": w} for w in ACC_BITS]
    for (section, key), value in DEFAULTS.items():
        p[section]["properties"][key]["default"] = value
        if key in p[section]["required"]:      # idempotent: a completed schema has it optional already
            p[section]["required"].remove(key)
    p["spec_version"]["const"] = SPEC_VERSION
    p["spec_version"]["x-opentsp-why"] = f"this model is specification version {SPEC_VERSION}"
    schema["anyOf"] = [
        {"type": "object", "properties": {"sram": {"type": "object", "properties": {
            "words_per_slice": {"type": "integer", "const": w}, "bank_bit": {"type": "integer", "const": b}}}}}
        for w, b in sorted(BANK_BIT.items())]
    schema["x-opentsp-why"] = "sram.bank_bit must be the upper-most address bit: log2(words_per_slice) - 1 (7 for 256 words, 12 for 8192, 15 for 65536)"
    schema["x-opentsp-relations"] = [relation(*r) for r in RELATIONS + C2C_RELATIONS + TIMING_RELATIONS]
    p["icu"]["x-opentsp-why"] = "For beat b=0,1 and tile k=0..S-1, 1+fetch_latency+floor(16*(b*S+k)/fetch_bytes_per_cycle) > b+k; S=geometry.superlanes."
    return schema


def relation(a, op, b, off, why, quads=0):
    """The schema entry of a relation; "quads" (+1 or -1) is present only when it is not 0."""
    r = {"a": a, "op": op, "b": b, "offset": off}
    if quads:
        r["quads"] = quads
    r["why"] = why
    return r


def lookup(cfg, key):
    for k in key.split("."):
        cfg = cfg[k]
    return cfg


def causality_errors(cfg):
    """Check effective cross-superlane rows, including instruction overrides.

    Python profile consumers share this check. Sail and Rust check independently.
    This does not replace the full configuration schema validator.
    """
    errors = []
    sl = cfg["geometry"]["superlanes"]
    icu = cfg["icu"]
    rate = icu["fetch_bytes_per_cycle"]
    if rate > 0 and any(1 + icu["fetch_latency"] + 16 * (beat * sl + tile) // rate <= beat + tile
                        for beat in range(2) for tile in range(sl)):
        errors.append("icu.fetch_bytes_per_cycle: instruction text delivery precedes source tile capture")
    timing = cfg["timing"]
    overrides = {r["entry"]: r for r in timing["overrides"]}
    for name, cls in (("Shift", "sxm_cross"), ("Permute", "sxm_cross"),
                      ("ABC", "mxm_abc"), ("ABC8", "mxm_abc")):
        row = overrides.get(name, timing[cls])
        if row["d_func"] - row["d_skew"] < sl:
            errors.append(f"{name}: d_func - d_skew must be at least geometry.superlanes ({sl})")
    if timing["sxm_map_ready"] < sl:
        errors.append("timing.sxm_map_ready: must be at least geometry.superlanes")
    return errors


def check(cfg):
    """The invariants added above, for configs (a self-test of the profiles)."""
    errs = causality_errors(cfg)
    if cfg.get("spec_version") != SPEC_VERSION:
        errs.append("spec_version")
    if BANK_BIT.get(cfg["sram"]["words_per_slice"]) != cfg["sram"]["bank_bit"]:
        errs.append("sram.bank_bit")
    for path, (lo, hi) in BOUNDS.items():
        if not lo <= lookup(cfg, ".".join(path)) <= hi:
            errs.append(".".join(path))
    for path, m in MULTIPLES.items():
        if lookup(cfg, ".".join(path)) % m:
            errs.append(".".join(path))
    if lookup(cfg, "mxm.acc_bits") not in ACC_BITS:
        errs.append("mxm.acc_bits")
    names = cfg["extensions"]
    if any(x not in ("bf16", "wide_acc", "block_fp8") for x in names) or len(set(names)) != len(names):
        errs.append("extensions (only bf16, wide_acc and block_fp8, each once)")
    if ("wide_acc" in names) != (lookup(cfg, "mxm.acc_bits") == 128):
        errs.append("extensions (wide_acc is declared exactly when mxm.acc_bits is 128)")
    # A timing row keyed by an element type that only an extension defines is never used by a profile without it. The simulators read the
    # row without a message, so this check is the author's only notice.
    for r in cfg["timing"]["overrides"]:
        entry, _, tname = r["entry"].rpartition(".")
        ext = TYPE_EXTENSION.get(tname)
        if entry and ext and ext not in names:
            errs.append(f"timing.overrides {r['entry']}: element type {tname} needs the extension {ext}, which the profile does not declare")
    # mxm.f16_rows lies between half the rows and all of them (the weight storage of the model holds 2 * f16_rows stream values,
    # at least the i8 rows; a relation of the model's configuration check, not a schema relation).
    rows, f16 = cfg["mxm"]["rows"], cfg["mxm"]["f16_rows"]
    if not (2 * f16 >= rows and f16 <= rows):
        errs.append(f"mxm.f16_rows: {f16} must be between mxm.rows / 2 and mxm.rows ({rows // 2} and {rows})")
    ops = {"<": lambda x, y: x < y, "<=": lambda x, y: x <= y}
    quads = lookup(cfg, MEM_SLICES) // 4
    for a, op, b, off, why, *q in RELATIONS + C2C_RELATIONS + TIMING_RELATIONS:
        off = off + (q[0] if q else 0) * quads
        if not ops[op](lookup(cfg, a), lookup(cfg, b) + off):
            errs.append(f"{a} {op} {b} {off:+d} ({why})")
    return errs


def advisories(cfg):
    """Warnings about profiles the schema admits but whose shape is probably not intended. They are not schema relations and never make
    --check fail. mxm.rows > vlanes (16 x superlanes): the extra rows count toward the buffer capacity (RES-002) but a product has
    only vlanes lanes, so the rows beyond them cannot be observed."""
    out = []
    vlanes = 16 * cfg["geometry"]["superlanes"]
    if cfg["mxm"]["rows"] > vlanes:
        out.append(f"mxm.rows ({cfg['mxm']['rows']}) exceeds vlanes ({vlanes}): rows beyond vlanes only affect buffer capacity (RES-002), "
                   "their product lanes are not observable")
    return out


def main(argv=None):
    """Validate profiles, normalise defaults, or complete a schema in place."""
    import argparse
    import os
    import stat
    from pathlib import Path
    import tempfile

    parser = argparse.ArgumentParser(description=main.__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="validate one or more profile files")
    modes.add_argument("--normalise", action="store_true", help="write defaulted INPUT to OUTPUT")
    parser.add_argument("paths", nargs="+", help="schema file, profile files, or INPUT OUTPUT")
    args = parser.parse_args(argv)
    if args.normalise and len(args.paths) != 2:
        parser.error("--normalise requires INPUT OUTPUT")
    if not args.check and not args.normalise and len(args.paths) != 1:
        parser.error("schema completion requires one file")
    try:
        if args.check:
            bad = False
            for name in args.paths:
                with open(name) as source:
                    cfg = json.load(source)
                if not isinstance(cfg, dict):
                    raise ValueError("profile must be a JSON object")
                for error in check(cfg):
                    print(f"{name}: {error}")
                    bad = True
                for warning in advisories(cfg):
                    print(f"{name}: advisory: {warning}")
            return int(bad)
        with open(args.paths[0]) as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("input must be a JSON object")
        value = with_defaults(value) if args.normalise else complete(value)
        target = Path(args.paths[1] if args.normalise else args.paths[0]).resolve()
        payload = json.dumps(value, indent=2) + "\n"
        # Validate/serialize before opening the destination; failure preserves its old contents.
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as output:
            temp = Path(output.name)
            try:
                if target.exists():
                    os.fchmod(output.fileno(), stat.S_IMODE(target.stat().st_mode))
                output.write(payload)
                output.close()
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
