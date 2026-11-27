#!/usr/bin/env python3
"""Pin the actual native cap build inputs before and after generation/linking."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(spec, model, lib):
    files = {
        "model/" + str(p.relative_to(model)): sha(p)
        for p in sorted(model.rglob("*")) if p.is_file()
    }
    for name in ("numeric_cap.sail", "numeric_cap_driver.c", "build_numeric_cap.sh"):
        files["driver/" + name] = sha(spec / "test/formal" / name)
    files["stubs.c"] = sha(spec / "test/unit_tests/stubs.c")
    for path in sorted(lib.iterdir()):
        if path.is_file() and path.suffix in (".c", ".h"):
            files["sail-runtime/" + path.name] = sha(path)
    for name in ("GMP_LIBS", "Z_LIBS"):
        path = Path(os.environ.get(name, ""))
        if path.is_file():
            files[name] = sha(path)
    for name in ("sail", "cc"):
        files["tool/" + name] = sha(Path(shutil.which(name)).resolve())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "finish"))
    parser.add_argument("spec", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("lib", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    current = snapshot(args.spec, args.model, args.lib)
    before_path = args.work / "inputs-before.json"
    if args.phase == "before":
        before_path.write_text(json.dumps(current, indent=2) + "\n")
        return
    before = json.loads(before_path.read_text())
    if current != before:
        raise ValueError("native cap inputs changed during the actual build")
    binary = args.work / "numeric_cap_driver.partial"
    receipt = {
        "inputs_before": before, "inputs_after": current,
        "input_identity_equal": True,
        "generated_project_sha256": sha(args.work / "cap.sail_project"),
        "generated_c_sha256": sha(args.work / "model.c"),
        "binary_sha256": sha(binary),
        "compiler_command": args.command,
        "compiler": (args.work / "compiler.txt").read_text(),
        "generator": (args.work / "generator.txt").read_text(),
        "trust": "Sail generator, compiler, GMP/runtime/OS execution; not a kernel proof",
    }
    (args.work / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
