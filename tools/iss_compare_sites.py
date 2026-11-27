"""Compare the complete Sail and ISS run for each shipped vector with site observation enabled: exit code, stdout and
stderr (the program name at the start of a line aside). A run that exceeds the timeout is counted and printed, and the
comparison fails if any run times out."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from difflib import unified_diff
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "opentsp-arch-spec"


def address(key):
    s, a = map(int, key.split(":"))
    return f"{s:x} {a:x}"


def program(vector):
    lines = [f"S {address(key)} {bytes.fromhex(word).hex(' ')}" for key, word in vector["sram"].items()]
    lines.extend(f"Q {int(q):x} {units}" for q, units in vector["text"].items())
    lines.extend(f"D {address(key)}" for key in vector["dump"])
    lines.extend(vector.get("host", []))
    return ("\n".join(lines) + "\n").encode()


TIMEOUT_SECONDS = 600


def normalise(stderr):
    """stderr with the program name ('tsp_sim:' or 'tsp-iss:') at the start of a line replaced by 'prog:'."""
    return re.sub(rb"(?m)^(tsp_sim|tsp-iss):", b"prog:", stderr)


def run(binary, args, stdin):
    """(returncode, stdout, normalised stderr), or None when the run timed out. A non-zero exit is a result."""
    try:
        r = subprocess.run([binary, *args], input=stdin, capture_output=True, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return None
    return r.returncode, r.stdout, normalise(r.stderr)


def compare(profile, vector, binaries):
    """True when both runs finished and agree, False when one timed out (counted by the caller), else ValueError."""
    stdin = program(vector)
    args = [SPEC / "config" / f"{profile}.json", "--trace", "--observe", "--lint"]
    sail, iss = [run(binary, args, stdin) for binary in binaries]
    if sail is None or iss is None:
        return False
    if sail[0] != 0 or iss[0] != 0:
        raise ValueError(f"{profile}/{vector['name']}: simulator failed (Sail {sail[0]}, ISS {iss[0]})")
    if sail[0] != iss[0] or sail[2] != iss[2]:
        raise ValueError(f"{profile}/{vector['name']}: exit code or stderr differ\n"
                         f"Sail {sail[0]} {sail[2][:200]!r}\nISS  {iss[0]} {iss[2][:200]!r}")
    if sail[1] != iss[1]:
        delta = unified_diff(sail[1].decode().splitlines(True), iss[1].decode().splitlines(True),
                             fromfile="Sail", tofile="ISS")
        raise ValueError(f"{profile}/{vector['name']}\n{''.join(delta)}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sail", type=Path, default=ROOT / "target/spec/tsp_sim")
    parser.add_argument("--iss", type=Path, default=ROOT / "target/release/tsp-iss")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for profile in ("ref", "mini"):
            with (SPEC / "conformance/model" / f"{profile}.json").open() as inp:
                vectors = json.load(inp)["vectors"]
            results = [pool.submit(compare, profile, v, (args.sail, args.iss)) for v in vectors]
            finished = [result.result() for result in results]
            timeouts = finished.count(False)
            print(f"{profile}: {len(vectors) - timeouts} complete observed runs matched, {timeouts} timeouts", flush=True)
            if timeouts:
                sys.exit(f"{profile}: {timeouts} of {len(vectors)} runs timed out; the comparison does not cover them")


if __name__ == "__main__":
    main()
