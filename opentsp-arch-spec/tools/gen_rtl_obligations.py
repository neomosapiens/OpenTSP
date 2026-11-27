#!/usr/bin/env python3
"""The RTL obligation of every golden vector (platform.md section 7.1), generated into conformance/rtl/<profile>.json.

The golden vectors of conformance/model/ are frozen and carry no such field, so the scope of each vector for a silicon RTL is
recorded beside them. The value of a vector is the verdict of `tsp-iss verify` (the one implementation of the well-formed
decision) on the vector's own program and data, with the vector's host script:

  pass        well-formed (verify exit 0): the RTL reproduces the five comparisons of 7.1 on this vector
  none        no obligation: ill-formed (verify exit 1: the run raises a schedule code that no data code precedes), a run with
              injected faults (compared only by an RTL with the test mode of 5.1), or an `F mxm` vector (model and ISS only)
  undecided   data-dependent (verify exit 3): the vector's data raise a data code, so the schedule codes of the run cannot be
              told from its cascade. It binds an RTL as a cascade test (7.1, "Cascades") only when it has a `witness`: the
              same program on other data that `verify` finds well-formed (the identity map for the two SXM-001 vectors)

usage: gen_rtl_obligations.py [--iss PATH] [--profile ref|mini|all_ext]... [--check] [-j N]
  --check   compare the committed files with a fresh run and exit 1 when they differ (nothing is written)
"""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "opentsp-arch-spec"
VALUES = ("pass", "none", "undecided")
# Shipped profiles use --profile; the existing combined-extension dump uses --config.
PROFILE_ISS = {
    "ref": ["--profile", "ref"],
    "mini": ["--profile", "mini"],
    "all_ext": ["--config", str(SPEC / "config/test/all_ext.json")],
}

# The two SXM-001 cascade vectors of each profile (the vector names are those of one profile): a data code (SXM-001: a map that is not bijective) comes first and a schedule code
# (TIM-001) follows it. Their witness is the same program with the identity map in the two slices that hold the map (low bytes
# in the first, high bytes in the second): the generator runs it through `verify` and requires it to be well-formed, which shows
# that the schedule code of the vector is the cascade of its data code and not a fault of the object.
WITNESS_MAP = {("ref", "directed-239"): ["4:1", "5:1"], ("ref", "directed-262"): ["4:1", "5:1"],
               ("mini", "directed-256"): ["4:1", "5:1"], ("mini", "directed-313"): ["4:1", "5:1"]}


def witness_vector(profile, vec):
    """The vector with the identity map in the slices of WITNESS_MAP, or None for a vector that has no witness."""
    slots = WITNESS_MAP.get((profile, vec["name"]))
    if slots is None:
        return None
    lanes = len(bytes.fromhex(vec["sram"][slots[0]]))
    w = json.loads(json.dumps(vec))
    w["sram"][slots[0]] = bytes(i & 255 for i in range(lanes)).hex()
    w["sram"][slots[1]] = bytes(i >> 8 for i in range(lanes)).hex()
    return w


def harness_text(vec):
    """The harness input of a vector: its Q and S lines (the host script goes to `verify --host`; the dump list is not needed)."""
    lines = [f"Q {int(q):x} {units}" for q, units in vec["text"].items()]
    lines += [f"S {int(k.split(':')[0]):x} {int(k.split(':')[1]):x} " + " ".join(f"{b:x}" for b in bytes.fromhex(w))
              for k, w in vec["sram"].items()]
    return "\n".join(lines) + "\n"


def judge(iss, profile, vec):
    """(obligation, why) of one vector."""
    host = vec.get("host", [])
    faults = [h for h in host if h.startswith("F ")]
    if any(h.startswith("F mxm ") for h in faults):
        return "none", "F mxm: model and ISS only (the 96-bit exact sum has no counterpart in an RTL)"
    if faults:
        return "none", "fault run: compared only by an RTL with the test mode of platform.md 5.1"
    with tempfile.TemporaryDirectory() as d:
        text = Path(d) / "vector.txt"
        text.write_text(harness_text(vec))
        cmd = [str(iss), "verify", str(text), *PROFILE_ISS[profile]]
        script = [h for h in host if h.startswith("H ") or h.startswith("C ")]
        if script:
            hf = Path(d) / "host.txt"
            hf.write_text("\n".join(script) + "\n")
            cmd += ["--host", str(hf)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    first = {}
    for line in r.stdout.splitlines():
        t = line.split()
        if t and t[0].startswith("OTS-E-") and t[-1] in ("schedule", "data-dependent", "data-fetched"):
            first.setdefault("schedule" if t[-1] == "schedule" else "data", f"{t[0]} at cycle {t[2]} queue {t[4]}")
    if r.returncode == 0:
        return "pass", ""
    if r.returncode == 1:
        if "schedule" in first:
            return "none", "ill-formed: " + first["schedule"]
        return "none", "ill-formed: text parked behind a wait that nothing releases"
    if r.returncode == 3:
        if "schedule" in first:
            return "undecided", "data-dependent: the data code " + first.get("data", "?") + " precedes the schedule code " + first["schedule"]
        return "undecided", "data-dependent: the data code " + first.get("data", "?") + " may hide a schedule code"
    if r.returncode == 2 and "cannot be judged alone" in r.stderr:
        return "undecided", "C2C text: a lone object that sends or receives on a link is judged only as the manifest of its program"
    raise SystemExit(f"gen_rtl_obligations: tsp-iss verify exited {r.returncode}: {r.stderr.strip()}")


def build(iss, profile, jobs):
    # The test-only all_ext dump remains outside the freeze set, even if a stray model/all_ext exists.
    path = SPEC / ("conformance/model-extra" if profile == "all_ext" else "conformance/model") / f"{profile}.json"
    raw = path.read_bytes()
    doc = json.loads(raw)
    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        results = list(ex.map(lambda v: judge(iss, profile, v), doc["vectors"]))
    vectors = {v["name"]: {"obligation": o, "why": w} for v, (o, w) in zip(doc["vectors"], results)}
    for v in doc["vectors"]:
        wv = witness_vector(profile, v)
        if wv is not None:
            if judge(iss, profile, wv)[0] != "pass" or vectors[v["name"]]["obligation"] != "undecided":
                raise SystemExit(f"gen_rtl_obligations: the witness of {v['name']} is not well-formed, or the vector is not undecided")
            vectors[v["name"]]["witness"] = {"identity_map_in": WITNESS_MAP[(profile, v["name"])]}
    counts = {k: sum(1 for x in vectors.values() if x["obligation"] == k) for k in VALUES}
    return {
        "schema": "opentsp.conformance.rtl-obligation",
        "spec_version": doc["spec_version"],
        "profile": profile,
        "vectors_sha256": hashlib.sha256(raw).hexdigest(),
        "note": "Generated by tools/gen_rtl_obligations.py from tsp-iss verify; meaning in platform.md 7.1 and bring-up.md section 2.",
        "counts": counts,
        "vectors": vectors,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iss", default=str(ROOT / "target/release/tsp-iss"))
    ap.add_argument("--profile", action="append", choices=tuple(PROFILE_ISS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args(argv)
    bad = 0
    default = ["ref", "mini"]
    if (SPEC / "conformance/model-extra/all_ext.json").is_file() or (SPEC / "conformance/rtl/all_ext.json").is_file():
        default.append("all_ext")
    for profile in a.profile or default:
        out = SPEC / "conformance/rtl" / f"{profile}.json"
        text = json.dumps(build(a.iss, profile, a.j), indent=1) + "\n"
        if a.check:
            if not out.exists() or out.read_text() != text:
                print(f"gen_rtl_obligations: {out.relative_to(ROOT)} is stale: run opentsp-arch-spec/tools/gen_rtl_obligations.py")
                bad = 1
            else:
                print(f"ok: {out.relative_to(ROOT)}")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
            print(f"wrote {out.relative_to(ROOT)}")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
