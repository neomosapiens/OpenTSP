"""Run llvm-cov and enforce a common per-crate and total line/branch floor."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def below_floor(report, crates, minimum):
    counts = {crate: [0, 0, 0, 0] for crate in [*crates, "TOTAL"]}
    for file in report["data"][0]["files"]:
        crate = next((name for name in crates if f"/crates/{name}/src/" in file["filename"]), None)
        if crate is None:
            continue
        lines = file["summary"]["lines"]
        branches = file["summary"]["branches"]
        for key in (crate, "TOTAL"):
            count = counts[key]
            count[0] += lines["covered"]
            count[1] += lines["count"]
            count[2] += branches["covered"]
            count[3] += branches["count"]

    below = []
    for name, (lc, ln, bc, bn) in counts.items():
        line_pct = 100.0 * lc / max(ln, 1)
        branch_pct = 100.0 * bc / max(bn, 1)
        print(f"{name:12} lines {line_pct:6.2f}% ({lc}/{ln})  branches {branch_pct:6.2f}% ({bc}/{bn})")
        if ln == 0 or line_pct < minimum or branch_pct < minimum:
            below.append(name)
    return below


def main(label, crates):
    try:
        minimum = float(os.environ.get("COVERAGE_MIN") or "90")
    except ValueError:
        minimum = float("nan")
    if not 0 <= minimum <= 100:
        sys.exit(f"{label}: COVERAGE_MIN must be a number from 0 to 100")

    os.chdir(Path(__file__).resolve().parents[1])
    toolchain = os.environ.get("COVERAGE_TOOLCHAIN") or "nightly"
    preflight = subprocess.run(
        ["rustup", "run", toolchain, "rustc", "--version"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if preflight.returncode:
        sys.exit(f"{label}: toolchain '{toolchain}' is not installed "
                 f"(rustup toolchain install {toolchain} --profile minimal -c llvm-tools-preview)")

    cargo = ["cargo", f"+{toolchain}", "llvm-cov"]
    package_args = [part for crate in crates for part in ("-p", crate)]
    output = os.environ.get("COVERAGE_JSON")
    destination = Path(output) if output else None
    # Use a new report per run. An output left by an earlier run is not evidence.
    with TemporaryDirectory(dir=destination.parent if destination else None) as scratch:
        report = Path(scratch) / "coverage.json"
        try:
            subprocess.run([*cargo, "clean", "--workspace"], check=True)
            subprocess.run([*cargo, "--branch", *package_args, "--json", "--summary-only",
                            "--output-path", str(report)], check=True)
        except subprocess.CalledProcessError as error:
            sys.exit(error.returncode)
        try:
            below = below_floor(json.loads(report.read_text()), crates, minimum)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            sys.exit(f"{label}: invalid coverage report: {error}")
        if destination:
            report.replace(destination)
    if below:
        sys.exit(f"{label}: below {minimum}% lines or branches: {', '.join(below)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
