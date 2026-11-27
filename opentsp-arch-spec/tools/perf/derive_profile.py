#!/usr/bin/env python3
"""A schema-valid profile from the `ref` template by stated, overridable derivation rules .

usage: derive_profile.py [--superlanes N] [--rows R] [--slices-per-hemisphere M] [--template ref.json]
                         [--set KEY=VALUE ...] [--out FILE] [--allow-invalid]

This is a tool, not a model change. A profile is any legal value of the independent keys of config/schema.json, and `mini` and
the test profiles say so; no relation of the schema derives a latency from the geometry. The rules below are a convention this
tool offers so that "a chip with N superlanes, R MXM rows and M MEM slices per hemisphere" has a starting profile. Each default
is labelled "derived default", is overridable with --set, and is printed with the numbers it changed. The profile goes to stdout
(or --out); the rules and the changes go to stderr, so that `derive_profile.py ... > profile.json` stays a profile.

Rules (S = superlanes, Q = M / 4 MEM quads per hemisphere; ref has S = 20, R = 320, M = 44, Q = 11):
  R1  mxm.rows = min(16 S, 320), one row per lane (ref: 320; the schema allows 16 to 320 in steps of 16, so 320 is a cap: the
      planes are not square above 20 superlanes, and S = 21 to 32 all get 320 rows unless --rows says otherwise), and
      mxm.f16_rows = mxm.rows / 2 at every width (the ratio of ref, 160 of 320, HC34 p.28; the keys allow rows / 2 to rows, so --set
      mxm.f16_rows=<rows> gives the tandem capacity), also when --set changes mxm.rows
  R2  timing.sxm_map_ready = timing.mxm_abc.d_func - d_skew = timing.sxm_cross.d_func - d_skew = S, and the same for the
      per-entry rows of Shift, Permute, ABC and ABC8 (the schema relations need each at least S; ref has exactly S)
  R3  the units outside the MEM quads keep their distance to them: with d = 11 - Q quads fewer than ref, the outward gaps
      (c2c_tx, sxm_dst, sxm_turn_in, mxm_in) move d stations toward the VXM and the inward gaps (c2c_rx, sxm_turn_out, mxm_acc)
      move d stations away from it; the VXM and the MEM bases stay (ref: no change)
  R4  sram.bank_bit = log2(sram.words_per_slice) - 1 (the schema relation) and sram.addr_bits = max(13, log2(sram.words_per_slice)) (a
      slice of 2^k words needs k address bits, and ref has 13) when --set changes the words
  R5  with --superlanes, the link and host rates keep their published byte rates, which do not depend on the width of the chip:
      c2c.cycles_per_vector = ceil(16 S / 13.89), a link of 4 lanes at 25 Gb/s = 12.5 GB/s at the assumed 900 MHz (ref: 24); and
      host.cycles_per_vector = ceil(16 S / 32), 32 bytes per cycle of the PCIe Gen4 x16 link (ref: 10). A wider vector takes more
      cycles on the same link (S = 32: 37 and 16).
  R6  with --set, `extensions` (a JSON list) and mxm.acc_bits keep the schema relation: wide_acc is listed exactly when mxm.acc_bits is 128, and
      whichever of the two is not set by hand follows the other
Nothing else changes: the other keys are those of the template, so the latencies of the functional units stay circuit choices.

The profile is checked against config/schema.json (with the relations of schema_invariants.py, which this tool only reads). A
profile outside the schema, for example more than 32 superlanes (the schema allows 1 to 32), is refused with the reason;
--allow-invalid prints it anyway, labelled NOT SCHEMA-VALID.
"""
import argparse
import copy
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))  # opentsp-arch-spec/tools: schema_invariants (read only)
from opentsp_reference import schema as schema_invariants  # noqa: E402
sys.path.insert(0, HERE)
import derive_peaks  # noqa: E402

DEFAULT_TEMPLATE = os.path.join(SPEC, "config", "ref.json")
SCHEMA = os.path.join(SPEC, "config", "schema.json")
REF_QUADS = 11
MIN_ADDR_BITS = 13                           # the address width of ref (ISCA20 13-bit addressing)
CLOCK_MHZ = 900                              # the assumed clock of derive_peaks.py, which the byte rates below are stated at
LINK_BYTES_PER_CYCLE = 12.5e9 / (CLOCK_MHZ * 1e6)   # one link: 4 lanes x 25 Gb/s (the published figure, microarchitecture.md section 10)
HOST_BYTES_PER_CYCLE = 32                    # the PCIe Gen4 x16 link at 900 MHz (microarchitecture.md section 14; 31.5 GB/s of payload)
OUTWARD = ["c2c_tx_gap", "sxm_dst_gap", "sxm_turn_in_gap", "mxm_in_gap"]
INWARD = ["c2c_rx_gap", "sxm_turn_out_gap", "mxm_acc_gap"]
ROW_ENTRIES = ["Shift", "Permute", "ABC", "ABC8"]
DERIVED = "derived default"


class DeriveError(Exception):
    """An input the tool refuses; main prints it and exits 2."""


def get(cfg, path):
    for part in path.split("."):
        cfg = cfg[part]
    return cfg


def put(cfg, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        cfg = cfg[part]
    cfg[parts[-1]] = value


def flatten(node, prefix=""):
    """Every scalar of a profile as dotted path -> value; the override rows as timing.overrides.<Entry>.<field>."""
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif prefix == "extensions.":
        out["extensions"] = node
    elif isinstance(node, list):
        for row in node:
            for k, v in row.items():
                if k != "entry":
                    out[f"{prefix}{row['entry']}.{k}"] = v
    else:
        out[prefix[:-1]] = node
    return out


def schema_max(*path):
    with open(SCHEMA) as fh:
        node = json.load(fh)["properties"]
    for part in path[:-1]:
        node = node[part]["properties"]
    return node[path[-1]]["maximum"]


def set_f16_rows(cfg, why):
    """Rule R1: the f16 capacity is half the rows (the ratio of ref), at every width."""
    half = cfg["mxm"]["rows"] // 2
    if cfg["mxm"]["f16_rows"] != half:
        put(cfg, "mxm.f16_rows", half)
        why["mxm.f16_rows"] = f"R1 {DERIVED}: an f16 matrix of half the rows (HC34 p.28); the keys allow half the rows to all of them"


def derive(template, superlanes=None, rows=None, slices=None):
    """Apply the rules to a copy of the template: (profile, {dotted path: rule label})."""
    cfg = copy.deepcopy(template)
    why = {}

    def setv(path, value, rule):
        put(cfg, path, value)
        why[path] = rule

    s = superlanes if superlanes is not None else cfg["geometry"]["superlanes"]
    if superlanes is not None:
        setv("geometry.superlanes", s, "argument --superlanes")
    if slices is not None:
        setv("geometry.mem_slices_per_hemisphere", slices, "argument --slices-per-hemisphere")
    if rows is not None:
        setv("mxm.rows", rows, "argument --rows")
    elif superlanes is not None:
        cap = schema_max("mxm", "rows")
        setv("mxm.rows", min(16 * s, cap),
             f"R1 {DERIVED}: one MXM row per lane, 16 x superlanes" + (f", capped at {cap} rows" if 16 * s > cap else ""))
    if superlanes is not None:
        vlanes = 16 * s
        setv("c2c.cycles_per_vector", max(1, math.ceil(vlanes / LINK_BYTES_PER_CYCLE - 1e-9)),
             f"R5 {DERIVED}: ceil(vector bytes / the published link rate of 12.5 GB/s at 900 MHz)")
        setv("host.cycles_per_vector", max(1, math.ceil(vlanes / HOST_BYTES_PER_CYCLE)),
             f"R5 {DERIVED}: ceil(vector bytes / 32 bytes per cycle)")
    set_f16_rows(cfg, why)
    setv("timing.sxm_map_ready", s, f"R2 {DERIVED}: superlanes")
    for cls in ("mxm_abc", "sxm_cross"):
        setv(f"timing.{cls}.d_func", s + cfg["timing"][cls]["d_skew"], f"R2 {DERIVED}: superlanes + d_skew")
    for row in cfg["timing"]["overrides"]:
        if row["entry"] in ROW_ENTRIES:
            row["d_func"] = s + row["d_skew"]
            why[f"timing.overrides.{row['entry']}.d_func"] = f"R2 {DERIVED}: superlanes + d_skew"
    quads = cfg["geometry"]["mem_slices_per_hemisphere"] // 4
    d = REF_QUADS - quads
    for gap in OUTWARD:
        setv(f"layout.{gap}", cfg["layout"][gap] - d, f"R3 {DERIVED}: keeps its distance to the last MEM quad ({d} quads fewer than ref)")
    for gap in INWARD:
        setv(f"layout.{gap}", cfg["layout"][gap] + d, f"R3 {DERIVED}: keeps its distance to the last MEM quad ({d} quads fewer than ref)")
    return cfg, why


def apply_sets(cfg, why, sets):
    """--set KEY=VALUE: an override of any scalar key of the template (and of the override rows)."""
    flat = flatten(cfg)
    by_hand = set()
    for text in sets:
        if "=" not in text:
            raise DeriveError(f"--set {text!r}: expected KEY=VALUE")
        key, raw = text.split("=", 1)
        try:
            value = json.loads(raw)
        except ValueError:
            raise DeriveError(f"--set {key}: {raw!r} is not a JSON value (an integer or true/false)")
        if key not in flat:
            raise DeriveError(f"--set {key}: the template has no such scalar key")
        if key == "extensions":
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise DeriveError(f"--set extensions: expected a JSON list of extension names, got {raw!r}")
        elif isinstance(value, bool) != isinstance(flat[key], bool) or not isinstance(value, (bool, int)):
            raise DeriveError(f"--set {key}: expected {type(flat[key]).__name__}, got {raw!r}")
        parts = key.split(".")
        if key == "extensions":
            cfg["extensions"] = value
        elif parts[0] == "timing" and parts[1] == "overrides":
            entry, field = ".".join(parts[2:-1]), parts[-1]      # an entry may have a dot in its name (Max.i8)
            rows = [r for r in cfg["timing"]["overrides"] if r["entry"] == entry]
            if not rows:
                raise DeriveError(f"--set {key}: the template has no timing row {entry!r}")
            rows[0][field] = value
        else:
            put(cfg, key, value)
        why[key] = "override: --set"
        by_hand.add(key)
    words = cfg["sram"]["words_per_slice"]
    if "sram.words_per_slice" in by_hand and words > 1 and words & (words - 1) == 0:
        if "sram.bank_bit" not in by_hand:
            put(cfg, "sram.bank_bit", words.bit_length() - 2)
            why["sram.bank_bit"] = f"R4 {DERIVED}: log2(sram.words_per_slice) - 1"
        if "sram.addr_bits" not in by_hand:
            put(cfg, "sram.addr_bits", max(MIN_ADDR_BITS, words.bit_length() - 1))
            why["sram.addr_bits"] = f"R4 {DERIVED}: max({MIN_ADDR_BITS}, log2(sram.words_per_slice))"
    # wide_acc is declared exactly when mxm.acc_bits is 128 (the schema relation): the one that is not set by hand follows the other
    if "extensions" in by_hand and "mxm.acc_bits" not in by_hand:
        bits = 128 if "wide_acc" in cfg["extensions"] else 32
        if cfg["mxm"]["acc_bits"] != bits:
            put(cfg, "mxm.acc_bits", bits)
            why["mxm.acc_bits"] = f"R6 {DERIVED}: 128 exactly when `extensions` lists wide_acc"
    elif "mxm.acc_bits" in by_hand and "extensions" not in by_hand:
        ext = [x for x in cfg["extensions"] if x != "wide_acc"]
        if cfg["mxm"]["acc_bits"] == 128:
            ext.append("wide_acc")
        if sorted(ext) != sorted(cfg["extensions"]):
            cfg["extensions"] = ext
            why["extensions"] = f"R6 {DERIVED}: wide_acc is listed exactly when mxm.acc_bits is 128"
    if "mxm.rows" in by_hand and "mxm.f16_rows" not in by_hand:
        set_f16_rows(cfg, why)


def validate(node, schema, path, errors):
    """The JSON Schema keywords config/schema.json uses: type, const, minimum, maximum, multipleOf, properties, required,
    additionalProperties, items, anyOf, allOf. Errors are 'dotted.path: reason'."""
    where = path or "(profile)"
    t = schema.get("type")
    if t == "integer" and (isinstance(node, bool) or not isinstance(node, int)):
        errors.append(f"{where}: must be an integer")
        return
    if t == "boolean" and not isinstance(node, bool):
        errors.append(f"{where}: must be true or false")
        return
    if t == "string" and not isinstance(node, str):
        errors.append(f"{where}: must be a string")
        return
    if t == "object" and not isinstance(node, dict):
        errors.append(f"{where}: must be an object")
        return
    if t == "array" and not isinstance(node, list):
        errors.append(f"{where}: must be an array")
        return
    if "const" in schema and node != schema["const"]:
        errors.append(f"{where}: must be {schema['const']}")
    if isinstance(node, int) and not isinstance(node, bool):
        if "minimum" in schema and node < schema["minimum"]:
            errors.append(f"{where}: {node} is below the minimum {schema['minimum']}")
        if "maximum" in schema and node > schema["maximum"]:
            errors.append(f"{where}: {node} is above the maximum {schema['maximum']}")
        if "multipleOf" in schema and node % schema["multipleOf"]:
            errors.append(f"{where}: {node} must be a multiple of {schema['multipleOf']}")
    if isinstance(node, dict):
        for k in schema.get("required", []):
            if k not in node:
                errors.append(f"{path + '.' if path else ''}{k}: missing")
        props = schema.get("properties", {})
        for k, v in node.items():
            if k in props:
                validate(v, props[k], f"{path + '.' if path else ''}{k}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path + '.' if path else ''}{k}: unknown key")
    if isinstance(node, list) and "items" in schema:
        for i, v in enumerate(node):
            validate(v, schema["items"], f"{path}[{i}]", errors)
    alts = schema.get("anyOf")
    if alts:
        trials = []
        for alt in alts:
            sub = []
            validate(node, alt, path, sub)
            trials.append(sub)
        if all(trials):
            consts = [a["const"] for a in alts if "const" in a]
            if len(consts) == len(alts):
                errors.append(f"{where}: {node} is not one of {', '.join(str(c) for c in sorted(consts))}")
            else:
                errors.append(f"{where}: {schema.get('x-opentsp-why', 'matches none of the allowed alternatives')}")
    for sub in schema.get("allOf", []):
        validate(node, sub, path, errors)


def schema_errors(cfg):
    """Every way the profile leaves config/schema.json (keywords) and the relations of schema_invariants.check (read only)."""
    with open(SCHEMA) as fh:
        schema = json.load(fh)
    errors = []
    validate(cfg, schema, "", errors)
    if not errors:
        try:
            errors += schema_invariants.check(cfg)
        except (KeyError, TypeError) as e:
            errors.append(f"relation check failed on {e}")
    return errors


def report(template, cfg, why, valid_errors):
    flat_t, flat_c = flatten(template), flatten(cfg)
    lines = ["derive_profile: a profile derived from the template by stated rules; a tool, not a model change.",
             f"every rule default is a '{DERIVED}' and is overridable with --set KEY=VALUE.", "", "rules:"]
    lines += [line for line in __doc__.split("Rules (")[1].split("Nothing else changes")[0].splitlines()[1:] if line.strip()]
    changed = [(k, flat_t.get(k), v) for k, v in flat_c.items() if flat_t.get(k) != v]
    lines += ["", "numbers changed from the template:" if changed else "no number changed from the template."]
    for k, old, new in changed:
        lines.append(f"  {k}: {old} -> {new}   [{why.get(k, 'follows its rule')}]")
    if valid_errors:
        lines += ["", "NOT SCHEMA-VALID:"] + [f"  {e}" for e in valid_errors]
    else:
        lines += ["", "schema-valid (config/schema.json and the relations of schema_invariants.py)."]
    return "\n".join(lines)


def positive_int(flag):
    def parse(text):
        try:
            v = int(text)
        except ValueError:
            v = 0
        if v <= 0:
            raise argparse.ArgumentTypeError(f"{flag} must be a positive integer (got {text!r})")
        return v
    return parse


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], epilog="rules: see the module docstring (--help prints the head)")
    ap.add_argument("--superlanes", type=positive_int("--superlanes"))
    ap.add_argument("--rows", type=positive_int("--rows"))
    ap.add_argument("--slices-per-hemisphere", type=positive_int("--slices-per-hemisphere"))
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a key (repeatable)")
    ap.add_argument("--out")
    ap.add_argument("--allow-invalid", action="store_true", help="print a profile that is outside the schema, labelled")
    args = ap.parse_args(argv)
    try:
        try:
            with open(args.template) as fh:
                template = json.load(fh)
        except OSError as e:
            raise DeriveError(f"{args.template}: cannot read the template ({e.strerror})")
        except ValueError as e:
            raise DeriveError(f"{args.template}: not valid JSON ({e})")
        try:
            cfg, why = derive(template, args.superlanes, args.rows, args.slices_per_hemisphere)
            apply_sets(cfg, why, args.set)
        except (KeyError, TypeError, AttributeError) as e:
            raise DeriveError(f"{args.template}: the template lacks a key the rules need ({e}); it must be a full profile such as ref.json")
    except DeriveError as e:
        print(f"derive_profile: {e}", file=sys.stderr)
        return 2
    errors = schema_errors(cfg)
    print(report(template, cfg, why, errors), file=sys.stderr)
    if errors and not args.allow_invalid:
        print("derive_profile: refused: the derived profile is outside the schema (use --allow-invalid to print it anyway)",
              file=sys.stderr)
        return 2
    text = json.dumps(cfg, indent=2) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    derive_peaks.run(main, "derive_profile")
