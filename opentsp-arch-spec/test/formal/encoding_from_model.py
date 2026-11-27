#!/usr/bin/env python3
"""The instruction encoding masks as the Sail model has them, against the catalogue (isa-db/generated/encoding.json).

usage: encoding_from_model.py ISA_QUERY CONFIG [encoding.json]

encoding_exact.py checks the catalogue against itself. This asks the model. For every catalogue entry E, words that the model's own
decoder (isa_query `e32` / `e64`: the entry the word belongs to) assigns to E are found by filling the wildcard bits of E's match pattern
at random; from each such word every bit is flipped in turn and the model is asked again. A bit that moves the word out of E in some probe
is a bit the model treats as fixed for E; a bit that never does is free. Then, for each entry:

  1. a bit that is fixed in the catalogue but free in the model (flipping it in every probe keeps the word in E) is an error: the model
     accepts both values where the catalogue says one;
  2. a bit that is a wildcard in the catalogue but moves the word out of E in every probe is legal only inside a field whose legal values the
     catalogue restricts (a `values`, `not_values`, `codes`, `max` or `min` entry, an alignment): otherwise the model fixes a bit that the
     catalogue calls free, which is an error;
  3. entries are found: every entry must be reached by some word (the catalogue has no entry the model cannot decode).

A bit flip is a probe, not a proof: free here means free in every probe of the run (PROBES fills per entry, a fixed seed), so a legality that
depends on two fields at once can hide, and the exhaustive32 check (`just model-exhaustive`) remains the complete one for 32-bit words. The run
prints, per width, the entries, probes and bits compared.
"""
from pathlib import Path
import json
import random
import subprocess
import sys

SPEC = Path(__file__).resolve().parents[2]
DEFAULT = SPEC / "isa-db/generated/encoding.json"
PROBES = 24            # words per entry from which the bits are flipped
ATTEMPTS = 4000        # random fills tried per entry to find those words
SEED = 20261119
# The keys with which the catalogue restricts the legal values of a field to a subset of its range.
RESTRICTIONS = ("values", "not_values", "bool_values", "codes", "max", "min", "max_expr", "align", "streams")


class Model:
    def __init__(self, exe, cfg):
        self.process = subprocess.Popen([exe, cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def entry(self, width, word):
        self.process.stdin.write("e%d %x\n" % (width, word))
        self.process.stdin.flush()
        return self.process.stdout.readline().rstrip("\n").split("\t")[0]

    def close(self):
        self.process.stdin.close()
        self.process.wait()
        self.process.stdout.close()


def span(text):
    high, _, low = str(text).partition("-")
    return int(high), int(low or high)


def field_bits(variable):
    """bit position -> True for every bit of the variable's location(s)."""
    bits = set()
    for key in ("location", "hi_location", "top_location"):
        if key in variable:
            msb, lsb = span(variable[key])
            bits.update(range(lsb, msb + 1))
    return bits


def restricted(variable):
    return any(key in variable for key in RESTRICTIONS)


def pattern_masks(entry):
    """(fixed mask, fixed value) of the catalogue's match pattern."""
    width, match = entry["width"], entry["match"]
    fixed = int("".join("0" if c == "-" else "1" for c in match), 2)
    value = int("".join("1" if c == "1" else "0" for c in match), 2)
    return fixed, value, width


def probe_words(model, entry, rng):
    """Up to PROBES words that the model assigns to `entry`, by random fills of the wildcard bits."""
    fixed, value, width = pattern_masks(entry)
    free = ((1 << width) - 1) & ~fixed
    found = []
    for attempt in range(ATTEMPTS):
        bias = rng.choice((0.5, 0.1, 0.9)) if attempt else 0.0
        fill = 0
        for bit in range(width):
            if free >> bit & 1 and rng.random() < bias:
                fill |= 1 << bit
        word = value | fill
        if word not in found and model.entry(width, word) == entry["name"]:
            found.append(word)
            if len(found) == PROBES:
                break
    return found


def compare(entry, words, model):
    """Problems of one entry and the numbers (bits fixed in the model, bits compared)."""
    fixed, _value, width = pattern_masks(entry)
    name = entry["name"]
    if not words:
        return ["%s: no word with a free-bit fill decodes to this entry in the model" % name], 0
    moved = [0] * width      # in how many probes flipping the bit leaves the entry
    for word in words:
        for bit in range(width):
            if model.entry(width, word ^ (1 << bit)) != name:
                moved[bit] += 1
    problems = []
    owners = {}
    for variable in entry["variables"]:
        for bit in field_bits(variable):
            owners[bit] = variable
    for bit in range(width):
        in_catalogue = bool(fixed >> bit & 1)
        if in_catalogue and moved[bit] == 0:
            problems.append("%s: bit %d is fixed in the catalogue but the model keeps the word in the entry when it is flipped (%d probes)"
                            % (name, bit, len(words)))
        if not in_catalogue and moved[bit] == len(words):
            variable = owners.get(bit)
            if variable is None or not restricted(variable):
                problems.append("%s: bit %d is free in the catalogue (field %s, no restriction listed) but flipping it leaves the entry in every one of %d probes"
                                % (name, bit, variable["name"] if variable else "none", len(words)))
    return problems, sum(1 for count in moved if count)


def run(model, catalogue):
    rng = random.Random(SEED)
    problems, probes, flips = [], 0, 0
    per_width = {}
    for entry in catalogue["instructions"]:
        words = probe_words(model, entry, rng)
        found, fixed_in_model = compare(entry, words, model)
        problems.extend(found)
        probes += len(words)
        flips += len(words) * entry["width"]
        count = per_width.setdefault(entry["width"], [0, 0])
        count[0] += 1
        count[1] += entry["width"]
    return problems, probes, flips, per_width


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    catalogue = json.loads(Path(argv[2] if len(argv) > 2 else DEFAULT).read_text())
    model = Model(argv[0], argv[1])
    try:
        problems, probes, flips, per_width = run(model, catalogue)
    finally:
        model.close()
    for width, (entries, bits) in sorted(per_width.items()):
        print("encoding_model: %d-bit entries: %d, bit positions compared: %d" % (width, entries, bits))
    print("encoding_model: %d entries, %d probe words, %d bit flips, %d problem(s)" % (len(catalogue["instructions"]), probes, flips, len(problems)))
    for line in problems:
        print("encoding_model: " + line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
