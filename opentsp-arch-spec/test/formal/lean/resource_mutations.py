#!/usr/bin/env python3
"""Counterfactual tests of the production-generated resource proofs.

Mutates only the supplied disposable Lean export, never the production model.
Each mutation is paired with its exact production Sail statement in the receipt.
This tests proof sensitivity after generation, not correctness of the Sail exporter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SPEC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SPEC / "tools"))
from fetch_port_proof import extract  # noqa: E402

MUTATIONS = [
    ("repeat_reservation", "repeat_issue", "IFETCHM(_) => reserve_local_fetch(q, i, c)",
     "IFETCHM(_) => ()", "| .IFETCHM _ => (reserve_local_fetch q i c)", "| .IFETCHM _ => (pure ())"),
    ("fetch_hold", "fetchm_hold_until", "c + 2", "c + 1", "(c +i 2)", "(c +i 1)"),
    ("second_stamp", "fetchm_bank_note", "bn_at = c + 1", "bn_at = c + 2", "bn_at := (c +i 1)", "bn_at := (c +i 2)"),
    ("patrol_priority", "scrub_arbitrate", "not(present) | taken", "not(present)",
     "((not present) || taken)", "(not present)"),
]


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def generated_body(text, name):
    match = re.search(r"^def " + re.escape(name) + r"\b.*?(?=^def |\Z)", text, re.M | re.S)
    if match is None:
        raise ValueError("missing generated definition " + name)
    return match


def proof_rejection(code, log):
    """A known theorem rejection only; mixed build/signal/infra errors fail closed."""
    if re.search('^[ \\t]*(?:maximum recursion depth has been reached|maximum number of heartbeats|(?:\\(deterministic\\) )?timeout at [^\\n]*maximum number of heartbeats)', log, re.I | re.M):
        return False
    if re.search(r"^error: .*: [Tt]actic .*failed with a nested error:", log, re.M):
        return False
    if code != 1:
        return False
    theorem = re.compile(r"^error: Proofs/Resources\.lean:[0-9]+:[0-9]+: "
                         r"(?:Application type mismatch|Type mismatch|unsolved goals|"
                         r"[Tt]actic .*failed|omega could not prove the goal:)")
    wrappers = {"error: Lean exited with code 1", "error: build failed"}
    diagnostics = [line for line in log.splitlines()
                   if re.search(r"error:|fatal:|exception:", line, re.I)]
    if re.search(r"segmentation fault|permission denied|no space left|killed by signal|signal [0-9]+", log, re.I):
        return False
    return bool(diagnostics) and any(theorem.match(line) for line in diagnostics) and all(
        theorem.match(line) or line in wrappers for line in diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    project, evidence = args.project.resolve(), args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    path = project / "Out/Tick.lean"
    original = path.read_text()
    sail = (args.model / "postlude/tick.sail").read_text()
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".elan/bin") + ":" + env["PATH"]
    records = []

    def check(tag):
        log = evidence / (tag + ".log")
        start = time.monotonic()
        with log.open("w") as out:
            process = subprocess.Popen(["lake", "build", "Proofs"], cwd=project,
                                       stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise RuntimeError(tag + " exceeded 90-second bound")
        return {"returncode": code, "seconds": time.monotonic()-start,
                "log": str(log), "log_sha256": sha(log.read_text())}

    try:
        for name, function, old, new, lean_old, lean_new in MUTATIONS:
            source = extract(sail, [function])
            if source.count(old) != 1:
                raise ValueError(name + ": production statement count changed")
            match = generated_body(original, function)
            body = match.group()
            if body.count(lean_old) != 1:
                raise ValueError(name + ": generated statement count changed")
            mutant_body = body.replace(lean_old, lean_new)
            mutant = original[:match.start()] + mutant_body + original[match.end():]
            path.write_text(mutant)
            result = check(name)
            result.update({"name": name, "function": function,
                           "production_body_sha256": sha(source), "production_old": old,
                           "production_new": new, "generated_old": lean_old, "generated_new": lean_new,
                           "generated_base_sha256": sha(original), "generated_mutant_sha256": sha(mutant)})
            log = Path(result["log"]).read_text()
            result["killed"] = proof_rejection(result["returncode"], log)
            records.append(result)
            (evidence / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            if not result["killed"]:
                raise RuntimeError(name + " was not killed by a resource theorem")
            path.write_text(original)
    finally:
        path.write_text(original)
        restored = check("restored_green")
        (evidence / "restored.json").write_text(json.dumps(restored, indent=2) + "\n")
    print(json.dumps({"mutants": len(records), "killed": sum(x["killed"] for x in records),
                      "restored_green": restored["returncode"] == 0}, indent=2))
    return int(restored["returncode"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
