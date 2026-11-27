#!/usr/bin/env python3
"""Run one bucket program through the in-process ISS binding (the library that serving runs), and time its steps.

    python3 opentsp-iss/perf/binding_p8.py --lib PATH/lib_tsp_iss.dylib [--dir DIR] [--bucket P8_4_128] [--runs N] [--json]

`--lib` is the cdylib of `opentsp-vllm/native` (`cargo build --release --manifest-path opentsp-vllm/native/Cargo.toml`, or the
profile-guided one `just iss-pgo-binding` writes); it is loaded by path, whichever tree built it. `--dir` holds
`<bucket>.elf` and `<bucket>.safetensors` (written by `tools/iss_bench_prep.py`; default `ISS_BENCH_DIR`, else /tmp/iss-bench; the
fixture of `just iss-perf-gate` is such a directory). The chip is built, the package and the program loaded once, then
`Chip.step` runs the program N times (each on a restarted chip, as a serving step does). It prints one line: the wall seconds
of the step (`min`, `median`), the cycles and the SHA-256 prefix of the output words; with `--json` the same as one JSON
object with every run. The SHA-256 covers the output words and every SRAM word the steps changed (`Chip.sram_changes`), so a
binding built another way must print the same one. The step is the host's own call, so the figures are the speed the serving process sees. It is the
training run of the profile-guided build of the binding (`tools/iss_build_pgo.sh --binding`) and the program of its A/B."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time


def load_binding(lib):
    """The binding module of the library at `lib`, imported under its own name from a scratch directory (the module's
    initialiser is `_tsp_iss`, so the file must be named `_tsp_iss.abi3.so`)."""
    here = tempfile.mkdtemp(prefix="iss-binding-")
    os.symlink(os.path.abspath(lib), os.path.join(here, "_tsp_iss.abi3.so"))
    sys.path.insert(0, here)
    import _tsp_iss

    return _tsp_iss


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--dir", default=os.environ.get("ISS_BENCH_DIR", "/tmp/iss-bench"))
    ap.add_argument("--bucket", default="P8_4_128")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--profile", default="ref")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    for ext in (".elf", ".safetensors"):
        path = os.path.join(a.dir, a.bucket + ext)
        if not os.path.isfile(path):
            sys.exit(f"binding_p8: {path} not found (python3 tools/iss_bench_prep.py --out {a.dir} {a.bucket})")
    m = load_binding(a.lib)
    chip = m.Chip(a.profile)
    chip.load_package(os.path.join(a.dir, a.bucket + ".safetensors"))
    chip.load_program(a.bucket, os.path.join(a.dir, a.bucket + ".elf"))
    secs, digest, cycles = [], hashlib.sha256(), None
    for _ in range(a.runs):
        t = time.perf_counter()
        words, cycles, _ = chip.step(a.bucket)
        secs.append(time.perf_counter() - t)
        digest.update(words)
        for slice_, addr, data in chip.sram_changes():
            digest.update(b"%d:%d:" % (slice_, addr) + data)
    digest = digest.hexdigest()[:16]
    doc = {"bucket": a.bucket, "runs": a.runs, "cycles": cycles, "min": min(secs), "median": statistics.median(secs), "output_sha256": digest, "seconds": secs}
    if a.json:
        print(json.dumps(doc))
    else:
        print(f"{a.bucket}: {a.runs} steps, min {doc['min'] * 1e3:.1f} ms, median {doc['median'] * 1e3:.1f} ms, {cycles} cycles, output {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
