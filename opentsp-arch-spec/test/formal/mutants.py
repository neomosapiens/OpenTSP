#!/usr/bin/env python3
"""One-line mutants of the model and the proof-gate mutation ratchet .

`prove_mutants.sh` copies the specification tree to a scratch directory, applies one mutant from
MUTANTS, runs `test/formal/srf_ring_test.sh`, `test/formal/lanes_test.sh`, `test/formal/prove_extra.sh` and `model_prove.sh` there (a mutant of kind
`lean` runs the Lean check only) and records whether a gate failed (the mutant is killed) or passed (it survives). A run that
fails without naming an obligation or a test (a build failure, a compiler error, a crashed generator, a signal) is an `error`, and a run
whose only failures are solver `timeout` or `unknown` answers is a `timeout`: neither proves anything, neither counts as a kill, both are repeated. The result is three numbers, not one score: `proofs: a/N, tests: b/N, survivors: c/N` (kills by an SMT
obligation, kills by a bounded-exhaustive C test, mutants that no gate kills; `lean: l/N` is added when a Lean statement kills a mutant).
The outcomes are compared with `mutants_expected.txt`:

  * a mutant that survives but is listed `kill` is a regression of the gate: the run fails;
  * a mutant that is killed but is listed `survive` means an obligation landed: the run fails
    until the list is tightened to `kill` (a ratchet only moves one way).

The mutated files are never the tree's own: the model is frozen (AGENTS.md section 8).
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import tarfile
import concurrent.futures
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

SPEC = Path(__file__).resolve().parents[2]
EXPECTED = Path(__file__).resolve().with_name("mutants_expected.txt")
COPIED = ("model", "config", "tools", "test/formal", "test/unit_tests", "test/export", "c_emulator")

# name, file under model/, old text (must occur exactly once), new text, area.
MUTANTS = [
    ("mut_srf", "core/streams.sail",
     "slot_index = emod_int(d, 64)", "slot_index = emod_int(d, 32)", "stream timing: the live SRF ring aliases"),
    ("mut_bank", "postlude/tick.sail",
     "      if blocked then {\n        report(q, c, MEM_BANK);", "      if false & blocked then {\n        report(q, c, MEM_BANK);",
     "hazard freedom: a bank-blocked AGT reference runs anyway (killed by bank_agt_clash.sh)"),
    ("m_s1", "core/numerics/fp.sail",
     "rem > half | (rem == half & tmod_int(lo, 2) == 1)", "rem >= half", "numerics: ties round up instead of to even"),
    ("m_s2", "core/numerics/lanes.sail",
     "op == 1 then bits_t(t, sat_t(t, a + b))", "op == 1 then bits_t(t, sat_t(t, b + b))",
     "numerics: VXM saturating add computes b + b"),
    ("m_s3", "slices/c2c/c2c_insts.sail",
     "Some(a) => a + c2c_cycles_per_vector > c,", "Some(a) => a + c2c_cycles_per_vector > c + 1,",
     "C2C: the link busy window is one cycle longer"),
    ("m_k1", "postlude/tick.sail",
     "(Some(s), Some(r)) => s <= c & c < r,", "(Some(s), Some(r)) => s <= c & c <= r,",
     "barrier window (covered by p_barrier_window)"),
    ("m_k2", "slices/cu/cu_iq.sail",
     "iq_head[q] = ring(iq_head[q] + n);", "iq_head[q] = ring(iq_head[q] + n + 1);",
     "IQ pop (covered by p_iq_pop_conservation)"),
    ("m_n1", "core/numerics/fp.sail",
     "f.ieee & ex + f.bias >= pow2i(f.ebits) - 1 then inf_bits(f, neg)", "f.ieee & ex + f.bias > pow2i(f.ebits) - 1 then inf_bits(f, neg)",
     "numerics: the overflow threshold is one binade too high (p_ovf_*)"),
    ("m_n2", "core/numerics/fp.sail",
     "let ee = if p + e < f.emin then f.emin else p + e;", "let ee = if p + e < f.emin then f.emin + 1 else p + e;",
     "numerics: subnormal results are rounded to the wrong spacing (p_rne_sub_*)"),
    ("m_n3", "core/numerics/fp.sail",
     "if m == pow2i(f.mbits + 1) then { m = tdiv_int(m, 2); ex = ex + 1 };", "if m == pow2i(f.mbits + 1) then { m = tdiv_int(m, 2); ex = ex + 2 };",
     "numerics: a carry out of the significand advances the exponent twice (p_rne_*, p_tie_*)"),
    ("m_n4", "core/numerics/fp.sail",
     "bias = 15, emin = -14, width = 16, ieee = true }\nlet FMT32", "bias = 16, emin = -15, width = 16, ieee = true }\nlet FMT32",
     "numerics: a binary16 whose bias and smallest exponent are consistent with each other but not IEEE (p_f16_decode, p_f16_anchors, p_ovf_f16, the solver cross-check)"),
    ("m_b1", "core/numerics/fp.sail",
     "(if f.ebits == 8 then 0x00007FC0 else 0x00007E00)", "(if f.ebits == 8 then 0x00007FC1 else 0x00007E00)",
     "numerics: the bfloat16 canonical NaN pattern (p_bf16_specials)"),
    ("m_b2", "core/numerics/fp.sail",
     "bias = 127, emin = -126, width = 16, ieee = true }", "bias = 126, emin = -126, width = 16, ieee = true }",
     "numerics: a bfloat16 whose bias is not the binary32 bias (p_bf16_decode, p_bf16_anchors)"),
    ("m_w1", "slices/mxm/mxm_insts.sail",
     "(a.wsp == 1 & b.wsp == 2) | (a.wsp == 2 & b.wsp == 1) then struct { wv = 0, wsp = 3", "(a.wsp == 1 & b.wsp == 1) | (a.wsp == 2 & b.wsp == 1) then struct { wv = 0, wsp = 3",
     "wide accumulator: +Inf plus -Inf is not NaN (p_wide_sticky)"),
    ("m_w2", "slices/mxm/mxm_insts.sail",
     "else if w.wv < 0 then fround(FMT32, true, 0 - w.wv, -48)", "else if w.wv < 0 then fround(FMT32, true, 0 - w.wv, -47)",
     "wide accumulator: the read-out of a negative lane uses the wrong unit (p_wide_replace)"),
    ("m_f1", "slices/cu/cu_iq.sail",
     "iq_len[q] = max(iq_len[q] - n, 0)", "iq_len[q] = max(iq_len[q] - n + 1, 0)",
     "IQ pop leaves one unit too many in the length (covered by p_iq_pop_accounting)"),
    ("m_f2", "slices/cu/cu_iq.sail",
     "    iq_len[q] = iq_len[q] + 1;\n    true", "    iq_len[q] = iq_len[q] + 2;\n    true",
     "IQ push counts two units (covered by p_iq_push_accounting)"),
    ("m_x1", "slices/cu/cu_iq.sail",
     "function iq_pop(q : range(0, 144), n : nat) -> unit = {\n",
     "function iq_pop(q : range(0, 144), n : nat) -> unit = {\n"
     "  if iq_len[q] > 0 then internal_error(__FILE__, __LINE__, \"mutant\");\n",
     "IQ pop raises internal_error (the SMT generator crashes: fails closed)"),
    ("m_x2", "slices/cu/cu_iq.sail",
     "function iq_pop(q : range(0, 144), n : nat) -> unit = {\n",
     "function iq_pop(q : range(0, 144), n : nat) -> unit = {\n"
     "  if iq_len[q] > 0 then not_implemented(\"mutant\");\n",
     "IQ pop raises a plain throw"),
    # Mutants in the areas no obligation reaches (expected `survive`: the rows record the gaps; an obligation that lands turns a row to `kill`).
    ("m_t1", "core/timing.sail",
     'if tname == "" then entry_latency(name, t)',
     'if tname == "" then entry_latency(name ^ ".", t)',
     "typed timing: empty tname must use bare key; mutant looks up name+'.' (killed by unit_mini Permute assert)"),
    ("m_ws1", "slices/mxm/mxm_insts.sail",
     "private function wtype_valid(h : bits(1), t : bits(2)) -> bool = h == 0b0 | t == 0b00 | t == 0b01",
     "private function wtype_valid(h : bits(1), t : bits(2)) -> bool = h == 0b0 | t == 0b00 | t == 0b11",
     "wtype_split path: bit-12 with bits 21..20 = 01 (bfp8) is refused as invalid (killed by unit_mini)"),
    ("m_be1", "slices/mxm/mxm_insts.sail",
     "if n >= 8 then n - 16 else n",
     "if n >= 8 then n - 15 else n",
     "bfp8 block scale: a negative scale wraps from the wrong bias (killed by unit_mini mini_ext)"),
    ("m_bf1", "slices/mxm/mxm_insts.sail",
     "let m = a.mag * b.mag * pow2i(a.e + b.e + blk_exp(ws, k) + blk_exp(xs, k) + 48);",
     "let m = a.mag * b.mag * pow2i(a.e + b.e + blk_exp(ws, k) + blk_exp(xs, k) + 47);",
     "bfp8_exact: the exact product unit is off by one power of two (killed by unit_mini mini_ext)"),
    ("m_r1", "postlude/tick.sail",
     "  match i { IFETCHM(_) => reserve_local_fetch(q, i, c), _ => () }\n}",
     "  match i { IFETCHM(_) => (), _ => () }\n}",
     "Repeat: the second-beat reservation of an IfetchM iteration is dropped"),
    ("m_a1", "core/numerics/fp.sail",
     "else (if a.e < b.e then a.e else b.e);", "else (if a.e > b.e then a.e else b.e);",
     "numerics: fadd_core aligns the operands to the larger exponent"),
    ("m_a2", "core/numerics/fp.sail",
     "if s == 0 then zero_bits(f, false)", "if s == 0 then zero_bits(f, true)",
     "numerics: an exact zero sum of opposite signs is -0"),
    ("m_a3", "core/numerics/fp.sail",
     "let neg = a.neg != b.neg;\n  if a.cls == FNaN", "let neg = a.neg == b.neg;\n  if a.cls == FNaN",
     "numerics: fp_mul takes the sign of the equal-sign case"),
    ("m_a4", "core/numerics/fp.sail",
     "bias = 127, emin = -126, width = 32", "bias = 126, emin = -126, width = 32",
     "numerics: the binary32 bias constant"),
    ("m_a5", "core/numerics/lanes.sail",
     "else if op == 7 then bits_t(t, sat_t(t, a * b))", "else if op == 7 then bits_t(t, wrap_t(t, a * b))",
     "numerics: the saturating multiply wraps"),
    ("m_a6", "core/numerics/lanes.sail",
     "else if op == 9 then bits_t(t, if a > b then a else b)", "else if op == 9 then bits_t(t, if a < b then a else b)",
     "numerics: max computes min"),
    ("m_a7", "core/numerics/lanes.sail",
     "else if op == 10 then bits_t(t, if a < b then a else b)", "else if op == 10 then bits_t(t, if a > b then a else b)",
     "numerics: min computes max"),
    ("m_z1", "core/numerics/transcendental.sail",
     "    if is_none(res) & p <= 4096 then {\n      let (lo, hi, sh) = tr_interval(op, a, p);\n      res = round_iv(f, lo, hi, sh);\n      p = p * 2\n    }",
     "    if is_none(res) & p <= 64 then {\n      let (lo, hi, sh) = tr_interval(op, a, p);\n      res = round_iv(f, lo, hi, sh);\n      p = p * 2\n    }",
     "numerics: Ziv tr_eval cap p<=4096 -> p<=64 (killed by unit_mini test_transcendental_converges)"),
    ("m_l0", "postlude/tick.sail",
     "if srf_report(r, g, c + latency_of(T_MEM).d_func, 1, xs, sp) then lint(q, c, LINT_RAS_EVICT)",
     "if not(srf_report(r, g, c + latency_of(T_MEM).d_func, 1, xs, sp)) then lint(q, c, LINT_RAS_EVICT)",
     "RAS: the spare-report eviction lint is raised when nothing was evicted"),
    ("m_c1", "slices/mxm/mxm_insts.sail",
     "else if wrong == 1 then (1, loc)", "else if wrong == 1 then (1, max(tdiv_int(loc, 2), 0))",
     "MXM integrity: a corrected single-row error reports the wrong lane (row shifted right)"),
    ("m_c2", "slices/mxm/mxm_insts.sail",
     "(vec, vec) =\n  if 0 <= r & 2 * r + 1 < mxm_wvals then (ws[2 * r], ws[2 * r + 1])",
     "(vec, vec) =\n  if 0 <= r & 2 * r + 3 < mxm_wvals then (ws[2 * r], ws[2 * r + 1])",
     "MXM sum check: the bound of an f16 weight row is one row short"),
    # Mutants that only a Lean statement kills: the proofs of test/formal/lean/proofs run on the mutated model (kind `lean`).
    ("m_l1", "slices/cu/cu_iq.sail",
     "iq_unit_write(q, ring(iq_head[q] + iq_len[q]), u);", "iq_unit_write(q, ring(iq_head[q] + iq_len[q] + 1), u);",
     "IQ push writes the unit to the wrong slot (stated by the Lean push theorem)"),
    ("m_l2", "slices/cu/cu_iq.sail",
     "iq_tag_write(q, ring(iq_head[q] + iq_len[q]), tag);", "iq_tag_write(q, ring(iq_head[q] + iq_len[q]), tag + 1);",
     "IQ push writes the wrong source tag (stated by the Lean push theorem)"),
    ("m_l3", "postlude/tick.sail",
     "    _                  => false,\n  }\n\nprivate function parked_one", "    _                  => true,\n  }\n\nprivate function parked_one",
     "a queue that never issued a Sync is in a barrier window (stated by the Lean barrier theorems)"),
]
KIND = {"m_k1": "lean", "m_k2": "lean", "m_f1": "lean", "m_f2": "lean", "m_l1": "lean", "m_l2": "lean", "m_l3": "lean", "m_r1": "lean", "m_a1": "lean", "m_a2": "lean", "m_a3": "lean"}
KIND["m_c1"] = "lean"
NAMES = [m[0] for m in MUTANTS]


def apply_mutant(model_root, name):
    """Apply the named mutant under `model_root` (a copy of model/). The old text must occur exactly once."""
    for mutant, rel, old, new, _area in MUTANTS:
        if mutant == name:
            path = Path(model_root) / rel
            text = path.read_text()
            if text.count(old) != 1:
                raise ValueError("mutant %s: %d occurrences of its old text in %s, need exactly 1" %
                                 (name, text.count(old), rel))
            path.write_text(text.replace(old, new))
            return rel
    raise KeyError(name)


def read_expected(text):
    """Parse `name kill|survive` lines (# comments); every mutant must be listed exactly once."""
    expected = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2 or fields[1] not in ("kill", "survive"):
            raise ValueError("bad expected line: " + line)
        if fields[0] in expected:
            raise ValueError("duplicate expected entry: " + fields[0])
        expected[fields[0]] = fields[1]
    if set(expected) != set(NAMES):
        raise ValueError("expected list and mutant table differ: missing=%s unexpected=%s" %
                         (sorted(set(NAMES) - set(expected)), sorted(set(expected) - set(NAMES))))
    return expected


def compare(expected, actual):
    """Return (regressions, improvements) for the names present in `actual` (`kill` or `survive`)."""
    regressions = sorted(n for n, a in actual.items() if a == "survive" and expected[n] == "kill")
    improvements = sorted(n for n, a in actual.items() if a == "kill" and expected[n] == "survive")
    return regressions, improvements


def scratch_copy(dest):
    for rel in COPIED:
        target = Path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SPEC / rel, target, ignore=shutil.ignore_patterns("__pycache__"))


# Process groups of the gates that are running: an interrupt ends them all (a gate is a tree of bash, sail and z3 processes).
RUNNING = set()
LOCK = threading.Lock()
INTERRUPTED = threading.Event()


def stop_all():
    INTERRUPTED.set()
    with LOCK:
        groups = list(RUNNING)
    # A container PID 1 need not reap orphaned grandchildren. Linux adoption
    # is process-wide; all waiting and signals remain scoped to OUR groups.
    libc, previous, reaping_error = None, None, None
    if sys.platform == "linux":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        prior = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(prior), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
            reaping_error = OSError(ctypes.get_errno(), "cannot enable owned-child reaping")
        else:
            previous = prior.value
    try:
        for pid in groups:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while libc is not None:
            live = False
            for group in groups:
                while True:
                    try:
                        child, _ = os.waitpid(-group, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if child == 0:
                        break
                try:
                    os.killpg(group, 0)
                    live = True
                except ProcessLookupError:
                    pass
            if not live:
                break
            if time.monotonic() >= deadline:
                for group in groups:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Give adopted children a bounded second chance to be reaped.
                if time.monotonic() >= deadline + 2:
                    raise RuntimeError("owned gate descendants remain after cancellation")
            time.sleep(0.02)
    finally:
        if libc is not None and previous is not None and libc.prctl(36, previous, 0, 0, 0) != 0:
            raise OSError("cannot restore child-subreaper setting")
    if reaping_error is not None:
        raise reaping_error


def input_manifest(root):
    """Only source inputs; proof-generated projects/build products are separate receipts."""
    files = {}
    for relative in COPIED:
        for path in sorted((Path(root) / relative).rglob("*")):
            if not path.is_file() or any(x in {"__pycache__", ".lake"} for x in path.parts):
                continue
            if path.name == "sail_smt_cache" or path.suffix in {".pyc", ".o", ".olean", ".ilean"}:
                continue
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


TOOL_VERSIONS = {}
TOOL_VERSION_LOCK = threading.Lock()


def tool_identity(kind=None):
    """Resolve the actual pinned native tool entrypoints, including Lean's PATH prefix."""
    result = {}
    env = dict(os.environ, PATH=str(Path.home() / ".elan/bin") + ":" + os.environ["PATH"])
    names = ("sail", "z3", "cc", "python3")
    if kind != "gate":
        names += ("lake", "lean")
    for name in names:
        path = shutil.which(name, path=env["PATH"])
        if path is None:
            raise ValueError("missing compiler input tool " + name)
        actual = Path(path).resolve()
        content = actual.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        native = content[:4] in {b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
                                 b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
                                 b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}
        cacheable = native and actual.name not in {"elan", "rustup", "mise"}
        key = (path, str(actual), digest)
        # Only native non-proxy binaries memoize successful version text. Every call still
        # resolves and hashes the actual binary; a changed path/hash is a new key.
        with TOOL_VERSION_LOCK:
            if not cacheable or key not in TOOL_VERSIONS:
                version = subprocess.run([path, "--version"], capture_output=True, text=True, env=env, timeout=10)
                if version.returncode:
                    raise ValueError("cannot identify compiler input tool " + name)
                if Path(path).resolve() != actual or hashlib.sha256(actual.read_bytes()).hexdigest() != digest:
                    raise ValueError("compiler input tool changed during version probe " + name)
                version_text = (version.stdout + version.stderr).strip()
                if cacheable:
                    TOOL_VERSIONS[key] = version_text
            else:
                version_text = TOOL_VERSIONS[key]
        result[name] = {"path": path, "resolved": str(actual),
                        "sha256": digest, "version": version_text}
    return result


def archive_inputs(tree, manifest, path, base_relative=None, base_content=None):
    with tarfile.open(path, "w:gz") as archive:
        for relative in sorted(manifest):
            archive.add(Path(tree) / relative, arcname=relative, recursive=False)
        if base_relative is not None:
            info = tarfile.TarInfo("base/" + base_relative)
            info.size = len(base_content)
            archive.addfile(info, io.BytesIO(base_content))


def generated_lean_inputs(tree):
    work = Path(tree) / "lean-work"
    paths = [work / name for name in ("Env.lean", "proof.sail_project", "profile.json")]
    project = work / "gen/out"
    paths += [project / name for name in ("lakefile.toml", "lake-manifest.json", "lean-toolchain")]
    paths += list((project / "Out").glob("*.lean")) + list((project / "Proofs").glob("*.lean"))
    return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


def survivor_extra_jobs(name):
    """Bound extra-suite parallelism to the two long expected survivors only."""
    if name not in {"m_c2", "m_l0"}:
        return "1"
    raw = os.environ.get("PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS", "1")
    try:
        jobs = int(raw)
    except ValueError as error:
        raise ValueError("PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS must be an integer from 1 to 8") from error
    if not 1 <= jobs <= 8:
        raise ValueError("PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS must be an integer from 1 to 8")
    return str(jobs)


def run_one(name, workdir, profiles, log_dir):
    tree = Path(workdir) / name
    base_manifest = input_manifest(SPEC)
    relative = "model/" + next(rel for mutant, rel, *_ in MUTANTS if mutant == name)
    base_content = (SPEC / relative).read_bytes()
    scratch_copy(tree)
    copied_manifest = input_manifest(tree)
    if copied_manifest != base_manifest:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        archive_inputs(tree, copied_manifest, Path(log_dir) / (name + ".inputs.tar.gz"), relative, base_content)
        (Path(log_dir) / (name + ".inputs.json")).write_text(json.dumps(
            {"mutant": name, "base": base_manifest, "copied": copied_manifest,
             "source_drift": True, "error": "source inputs drifted while copying"}, indent=2) + "\n")
        shutil.rmtree(tree)
        return name, "error", "source inputs drifted while copying", 0
    apply_mutant(tree / "model", name)
    inputs_before = input_manifest(tree)
    log = Path(log_dir) / (name + ".log")
    receipt_path = Path(log_dir) / (name + ".inputs.json")
    archive_path = Path(log_dir) / (name + ".inputs.tar.gz")
    archive_inputs(tree, inputs_before, archive_path, relative, base_content)
    try:
        tools_before = tool_identity(KIND.get(name, "gate"))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt_path.write_text(json.dumps({"mutant": name, "before": inputs_before,
                                           "error": str(error)}, indent=2) + "\n")
        shutil.rmtree(tree)
        return name, "error", "cannot identify proof tool inputs: " + str(error), 0
    receipt = {"mutant": name, "base": base_manifest, "before": inputs_before,
               "tools_before": tools_before, "mutation": next(list(m) for m in MUTANTS if m[0] == name),
               "archive": archive_path.name, "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
               "generated_input_policy": "Generated Lean compiler sources retained separately; build caches are excluded."}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    # One pass at the default integer width and no reach report: a mutant is killed by an obligation failing, which the
    # default pass shows; set PROVE_MUTANTS_WIDE=1 to run the 256-bit pass too.
    env = dict(os.environ, PROVE_PROFILES=profiles, PROVE_REACH="0", PROVE_WIDE=os.environ.get("PROVE_MUTANTS_WIDE", "0"),
               PROVE_FAIL_FAST="1", PROVE_EXTRA_JOBS=survivor_extra_jobs(name))
    start = time.time()
    with log.open("w") as out:
        with LOCK:
            if INTERRUPTED.is_set():
                raise KeyboardInterrupt
            # The cheapest checks first, each stopping the run when it fails: the bounded-exhaustive tests of the SRF ring (srf_ring_test.sh)
            # and of the VXM lane kernels (lanes_test.sh), the extra obligations of test/formal/props/ (prove_extra.sh, failing fast),
            # then the model's own gate. Any of them failing kills the mutant. A mutant of kind `lean` runs the Lean check alone
            # (the proofs of test/formal/lean/proofs on the mutated model; its own work directory, inside the scratch copy).
            unit_mini_killers = {"m_s3", "m_t1", "m_ws1", "m_be1", "m_bf1", "m_z1"}
            unit_gate = 'bash "$1/test/formal/unit_mini.sh" && ' if name in unit_mini_killers else ""
            bank_gate = 'bash "$1/test/formal/bank_agt_clash.sh" "$1" && ' if name == "mut_bank" else ""
            lean_gate = 'OPENTSP_MODEL_DIR="$1/model" LEAN_CHECK_DIR="$1/lean-work" bash "$1/test/formal/lean/lean_check.sh" mini'
            # FP mutations are judged first by the numeric properties they target. An unrelated
            # expensive wide query must not prevent reaching an actual numeric counterexample.
            # Retain every suite exactly once, and keep timeout/error classification unchanged.
            target = next(rel for mutant, rel, *_ in MUTANTS if mutant == name)
            extra_suites = " numerics ties frontend wide hold fptheory" if target == "core/numerics/fp.sail" else ""
            normal_gate = (
                'bash "$1/test/formal/srf_ring_test.sh" mini && '
                + 'bash "$1/test/formal/lanes_test.sh" && '
                + 'bash "$1/test/formal/prove_extra.sh"' + extra_suites + ' && '
                + 'bash "$1/tools/model_prove.sh"'
            )
            gate = lean_gate if KIND.get(name) == "lean" else unit_gate + bank_gate + normal_gate
            receipt["gate"] = ["bash", "-c", gate, "prove_mutants", str(tree)]
            receipt["environment"] = {key: env[key] for key in
                                      ("PATH", "GMP_PREFIX", "GMP_LIBS", "Z_LIBS", "PYTHON", "OPENTSP_TOOLCHAIN",
                                       "PROVE_PROFILES", "PROVE_REACH", "PROVE_WIDE", "PROVE_FAIL_FAST",
                                       "PROVE_EXTRA_JOBS", "PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS") if key in env}
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            process = subprocess.Popen(["bash", "-c", gate, "prove_mutants", str(tree)], env=env, stdout=out,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            RUNNING.add(process.pid)
        try:
            code = process.wait()
        finally:
            with LOCK:
                RUNNING.discard(process.pid)
    if INTERRUPTED.is_set():
        raise KeyboardInterrupt
    # The producer prints only the last 30 lines. Keep the full theorem diagnostics
    # so a native KILL can be audited without trusting a missing/truncated witness.
    if KIND.get(name) == "lean":
        proof_log = tree / "lean-work" / "proofs.log"
        if proof_log.is_file():
            shutil.copyfile(proof_log, Path(log_dir) / (name + ".proofs.log"))
    receipt["after"] = input_manifest(tree)
    try:
        receipt["tools_after"] = tool_identity(KIND.get(name, "gate"))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["tools_after"] = {"error": str(error)}
    receipt["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    proof_log_copy = Path(log_dir) / (name + ".proofs.log")
    if proof_log_copy.is_file():
        receipt["proofs_log"] = proof_log_copy.name
        receipt["proofs_log_sha256"] = hashlib.sha256(proof_log_copy.read_bytes()).hexdigest()
    receipt["exit_code"] = code
    receipt["source_drift"] = receipt["before"] != receipt["after"]
    receipt["tool_drift"] = receipt["tools_before"] != receipt["tools_after"]
    receipt["generated_lean_inputs"] = generated_lean_inputs(tree)
    if receipt["generated_lean_inputs"]:
        generated_archive = Path(log_dir) / (name + ".generated-inputs.tar.gz")
        archive_inputs(tree, receipt["generated_lean_inputs"], generated_archive)
        receipt["generated_archive"] = generated_archive.name
        receipt["generated_archive_sha256"] = hashlib.sha256(generated_archive.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    text = log.read_text()
    shutil.rmtree(tree)            # the scratch copy of this mutant (a few hundred kilobytes); the log stays
    if receipt["source_drift"] or receipt["tool_drift"]:
        return name, "error", "proof inputs drifted during the gate; retained receipt and exact pre-run inputs", time.time() - start
    # Mathematical counterexamples require the producer's exact clean solver protocol.
    fail_lines = [line for line in text.splitlines() if line.startswith("FAIL ")]
    verdict = re.compile(r"^FAIL (.+?): expected (sat|unsat), got (sat|unsat|timeout|unknown) \(exit 0\)$")
    failed, gave_up, invalid = [], [], []
    for line in fail_lines:
        match = verdict.fullmatch(line)
        if match is None or match[2] == match[3]:
            invalid.append(line)
        elif match[3] in {"timeout", "unknown"}:
            gave_up.append(match[1])
        else:
            failed.append(match[1])
    failed, gave_up = sorted(set(failed)), sorted(set(gave_up))
    native_error = re.compile(r"^error: Proofs/(AgtExecutionFrames|AgtOriginReset|AgtOriginState|AgtOrigins|AgtTimeBound|AgtTimeClosure|AgtTimedHistory|AllocationHistory|BankDomain|Barrier|Binade|CallerBounds|Capacity|CarrierDomains|Common|CombinedMachineInvariant|ControlConsumption|DispatchSlots|DistinctWriterSeen|DotBounds|DotSemantics|Enclosure|EndpointBounds|EventAgtNext|EventBarrierHistory|EventBarrierLazy|EventBoundaries|EventCensus|EventDispatch|EventFrames|EventParcelMonotone|EventQueries|EventQuiet|EventScrubPrefix|EventSearch|EventStageQuiet|EventTimedView|EventUnparked|Events|ExecuteProducer|ExecuteRingFrames|ExecutionHistory|ExecutionTimingBounds|FailureAdmission|FailureEffects|FormatNearest|FpSigns|Frontend|GrantConsumption|GridBounds|HistoryComposition|HistoryContent|HistoryFrames|HoldAdmission|HoldExecutionFrames|HoldQueryRel|Initialized|InputDecode|Intervals|IqPop|IqPush|IqReach|LaneArithmetic|LanePlacement|LanePolicies|Ln2Certificates|LogSeries|Logarithm|MachineInit|MxmChecksum|MxmIntegrity|MxmNominalRows|MxmRowClassifier|Numerics|OrderedDelivery|OrderedFetch|OrderedImage|OverlayHistory|ParcelBound|ParcelFrame|Patrol|PatrolCycleFrames|PatrolFrames|PatrolMemory|PatrolOwnedCycle|PatrolOwnership|PatrolProtected|PatrolValidity|Positivity|PublicNumerics|QueueConsumption|RangeReduction|Reciprocal|ReduceSemantics|RepairCell|RepairView|Reset|ResourceAdmission|ResourceComplete|ResourceExecuteFrames|ResourceFrames|ResourceGeometry|ResourceInvariant|ResourcePresence|ResourceRasFrames|ResourceReach|ResourceReadLoops|ResourceReady|ResourceScan|Resources|RootEnclosure|Rounding|ScalarSemantics|ScrubCatchup|SignedRounding|SlotAdmission|SramStorage|SrfBitPrimitives|SrfCommitDue|SrfCommitJournal|SrfCommitPolicy|SrfPolicyHistory|SrfPairPolicy|SrfCommittedHistory|SrfFaultHistory|SrfHistoryFrames|SrfLifetime|SrfLoopTable|SrfMerge|SrfMergeSemantics|SrfNearest|SrfPoisonFlow|SrfReportFlow|SrfSpareSemantics|SrfState|SrfUpstream|SrfValueFrames|Storage|StorageProduct|StorageRep|TagCoverage|TaggedCheck|TextCheckProduct|Wide128Bounds|ZivDomains)\.lean:[0-9]+:[0-9]+: "
                              r"(?:Application type mismatch|Type mismatch|unsolved goals|[Tt]actic .*failed|omega could not prove the goal:|Tactic `decide` proved that the proposition)")
    native_wrappers = {"error: Lean exited with code 1", "error: build failed"}
    native_nested = re.compile(r"^error: .*: [Tt]actic .*failed with a nested error:")
    native_resource = re.compile('^[ \\t]*(?:maximum recursion depth has been reached|maximum number of heartbeats|(?:\\(deterministic\\) )?timeout at [^\\n]*maximum number of heartbeats)', re.I)
    refused_marker = "lean_check: a proof does not check" in text.splitlines()
    for line in text.splitlines():
        if native_nested.match(line) or native_resource.match(line):
            invalid.append(line)
        elif line.startswith("ERROR SMT "):
            invalid.append(line)
        elif re.search(r"error:|fatal:|exception:", line, re.I):
            if not (refused_marker and (native_error.match(line) or line in native_wrappers)):
                invalid.append(line)
    if code < 0 or code >= 128:
        return name, "error", "gate stopped by a signal (exit %d): rerun" % code, time.time() - start
    if invalid or code not in (0, 1) or (code == 0 and (failed or gave_up or refused_marker)):
        return name, "error", "gate error (exit %d): invalid solver/protocol verdict: %s" % (code, "; ".join(invalid) or "inconsistent verdict status"), time.time() - start
    # Sail $[test] may splice "Assertion failed:" onto the previous "Testing …" line (no newline); still a kill.
    asserted = []
    for line in text.splitlines():
        if "Assertion failed:" in line:
            asserted.append(line.split("Assertion failed:", 1)[1].strip())
    # Sail $[test] may abort with a pattern-match failure instead of Assertion failed (m_ws1).
    matched = [line.strip() for line in text.splitlines() if line.startswith("Pattern match failure")]
    refused = [line for line in text.splitlines() if line.startswith("lean_check: a proof does not check")]
    if code and not failed and not asserted and not matched and not refused and gave_up:
        # Only solver timeouts: a mutant that slows an obligation proves nothing either way. Its own verdict; the run fails so that it is
        # repeated with a longer limit or looked at.
        return name, "timeout", "solver timeout or unknown on %s: rerun" % ",".join(gave_up), time.time() - start
    if code and not failed and not asserted and not matched and not refused:
        # The gate failed without naming an obligation, a test or a proof: a build failure of the mutated copy, a compiler error, a
        # crashed generator, a signal. It proved nothing either way, so the mutant is neither killed nor surviving, and the run
        # fails so that it is repeated.
        how = "gate stopped by a signal (exit %d): rerun" % code if code < 0 or code >= 128 else "gate error (exit %d): rerun" % code
        return name, "error", how, time.time() - start
    # A kill names its kind: FAIL is an SMT obligation that no longer holds (a proof), TEST is an assertion of a
    # bounded-exhaustive C test (srf_ring_test.sh, lanes_test.sh), which is evidence but not a proof, LEAN is a theorem of
    # test/formal/lean/proofs that no longer checks.
    how = ("FAIL " + ",".join(failed) if failed else
           ("TEST assertion: " + asserted[0] if asserted else
            ("TEST " + matched[0] if matched else
             ("LEAN a proof does not check" if refused else ""))))
    return name, ("kill" if code else "survive"), how, time.time() - start


def tally(actual, hows):
    """(proofs, tests, lean, survivors, N): kills by kind, mutants that no gate killed, and the mutants that ran to a verdict."""
    ran = {n: o for n, o in actual.items() if o not in ("error", "timeout")}
    kills = {"FAIL ": 0, "TEST ": 0, "LEAN ": 0}
    for name, outcome in ran.items():
        if outcome == "kill":
            for prefix in kills:
                if hows.get(name, "").startswith(prefix):
                    kills[prefix] += 1
    survivors = sum(1 for o in ran.values() if o == "survive")
    return kills["FAIL "], kills["TEST "], kills["LEAN "], survivors, len(ran)


def summary(actual, hows):
    proofs, tests, lean, survivors, total = tally(actual, hows)
    return "proofs: %d/%d, tests: %d/%d, %ssurvivors: %d/%d" % (proofs, total, tests, total,
                                                                  "lean: %d/%d, " % (lean, total) if lean else "", survivors, total)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect-current", action="store_true",
                        help="run every mutant and require the outcomes of mutants_expected.txt exactly")
    parser.add_argument("--only", action="append", default=[], metavar="NAME", help="run only this mutant (repeatable)")
    parser.add_argument("--profiles", default="mini", help="profiles to prove each mutant on (default: mini)")
    parser.add_argument("--jobs", type=int, default=1, help="mutants proved concurrently (default 1)")
    parser.add_argument("--list", action="store_true", help="print the table and exit")
    parser.add_argument("--check-table", action="store_true",
                        help="apply every mutant to a scratch copy of model/ without proving (each must match once)")
    parser.add_argument("--log-dir", default=None, help="keep the per-mutant logs here (default: a scratch directory)")
    args = parser.parse_args(argv)
    expected = read_expected(EXPECTED.read_text())
    if args.list:
        for name, rel, _old, _new, area in MUTANTS:
            print("%-9s %-8s %-5s %s  %s" % (name, expected[name], KIND.get(name, "gate"), rel, area))
        return 0
    if args.check_table:
        with tempfile.TemporaryDirectory() as tmp:
            for name in NAMES:
                shutil.copytree(SPEC / "model", Path(tmp) / name)
                print("%s: applies to %s" % (name, apply_mutant(Path(tmp) / name, name)))
        return 0
    names = args.only or NAMES
    for name in names:
        if name not in NAMES:
            print("prove_mutants: unknown mutant " + name, file=sys.stderr)
            return 2
    try:
        for name in names:
            survivor_extra_jobs(name)
    except ValueError as error:
        print("prove_mutants: " + str(error), file=sys.stderr)
        return 2
    # Tool-owned scratch directory; nothing is deleted from the tree.
    root = Path(os.environ.get("PROVE_MUTANTS_WORK") or tempfile.mkdtemp(prefix="prove_mutants."))
    root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print("prove_mutants: profiles %s, %d mutants, jobs %d, scratch %s" % (args.profiles, len(names), args.jobs, root), flush=True)
    actual, hows = {}, {}
    # Probe serially before starting compiler jobs, not repeatedly under their load.
    with TOOL_VERSION_LOCK:
        TOOL_VERSIONS.clear()
    try:
        tool_identity("lean" if any(KIND.get(n, "gate") != "gate" for n in names) else "gate")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("prove_mutants: cannot identify proof tool inputs: " + str(error), file=sys.stderr)
        return 3
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
    try:
        futures = [pool.submit(run_one, n, root, args.profiles, log_dir) for n in names]
        for future in concurrent.futures.as_completed(futures):
            name, outcome, how, seconds = future.result()
            actual[name] = outcome
            hows[name] = how
            print("%-9s %-8s (expected %-8s) %5.0f s  %s" % (name, outcome.upper(), expected[name], seconds, how), flush=True)
    except KeyboardInterrupt:
        stop_all()
        pool.shutdown(wait=True, cancel_futures=True)
        for name in names:
            shutil.rmtree(root / name, ignore_errors=True)       # scratch copies of the mutants that were cut short
        print("prove_mutants: interrupted; the running gates were stopped, logs are in %s" % log_dir, file=sys.stderr)
        return 130
    pool.shutdown()
    errors = sorted(n for n, o in actual.items() if o in ("error", "timeout"))
    print(summary(actual, hows))
    for name in errors:
        print("prove_mutants: %s: the gate of %s ended without a verdict (%s); its outcome is unknown, rerun it" %
              ("TIMEOUT" if actual[name] == "timeout" else "ERROR", name, hows[name]), file=sys.stderr)
    regressions, improvements = compare(expected, {n: o for n, o in actual.items() if o not in ("error", "timeout")})
    for name in regressions:
        print("prove_mutants: REGRESSION: %s survives but is a required kill" % name, file=sys.stderr)
    for name in improvements:
        print("prove_mutants: %s is now killed: change it to `kill` in test/formal/mutants_expected.txt "
              "(the ratchet tightens)" % name, file=sys.stderr)
    if errors:
        return 3
    if args.expect_current and (regressions or improvements):
        return 1
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
