"""The mutation ratchet: the mutant table applies exactly, and the comparison fails on the right outcomes."""
from pathlib import Path
import contextlib
import io
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import hashlib
import json
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants
from test_solver_protocol import outer_cli_error_output, outer_extra_error_output

REAL = mutants.SPEC
REAL_TOOL_IDENTITY = mutants.tool_identity


class Table(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = Path(cls.tmp.name) / "model"
        shutil.copytree(REAL / "model", cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fresh(self):
        copy = Path(tempfile.mkdtemp(dir=self.tmp.name))
        shutil.copytree(self.model, copy / "model")
        return copy / "model"

    def test_every_mutant_changes_exactly_one_place(self):
        for name in mutants.NAMES:
            with self.subTest(mutant=name):
                model = self.fresh()
                rel = mutants.apply_mutant(model, name)
                changed = [p for p in model.rglob("*.sail") if p.read_text() != (self.model / p.relative_to(model)).read_text()]
                self.assertEqual([p.relative_to(model).as_posix() for p in changed], [rel])

    def test_a_pattern_that_occurs_twice_or_not_at_all_is_rejected(self):
        model = self.fresh()
        path = model / "core/streams.sail"
        path.write_text(path.read_text() + "\n" + "slot_index = emod_int(d, 64)\n")
        with self.assertRaisesRegex(ValueError, "2 occurrences"):
            mutants.apply_mutant(model, "mut_srf")
        path.write_text(path.read_text().replace("slot_index = emod_int(d, 64)", "x"))
        with self.assertRaisesRegex(ValueError, "0 occurrences"):
            mutants.apply_mutant(model, "mut_srf")

    def test_unknown_mutant(self):
        with self.assertRaises(KeyError):
            mutants.apply_mutant(self.fresh(), "nope")

    def test_mutant_names_are_unique(self):
        self.assertEqual(len(set(mutants.NAMES)), len(mutants.NAMES))

    def test_the_ratchet_list_names_every_mutant(self):
        expected = mutants.read_expected(mutants.EXPECTED.read_text())
        self.assertEqual(set(expected), set(mutants.NAMES))
        for name in ("m_l0", "m_c2"):
            self.assertEqual(expected[name], "survive", name)         # the recorded gaps
        self.assertEqual(expected["m_c1"], "kill")  # actual CR45 indexed-row classifier native rejection
        for name in ("m_a1", "m_a2", "m_a3", "m_a4"):
            self.assertEqual(expected[name], "kill")  # actual native Numerics source mutations
        self.assertEqual(expected["m_r1"], "kill")  # actual production native Resources theorem
        self.assertEqual(expected["mut_bank"], "kill")  # bank_agt_clash.sh
        self.assertEqual(expected["m_z1"], "kill")  # unit_mini / test_transcendental_converges
        self.assertEqual(expected["m_s3"], "kill")  # unit_mini / test_c2c_link_busy


class ToolIdentity(unittest.TestCase):
    def probe(self, kind, missing=()):
        def locate(name, **_kwargs):
            return None if name in missing else sys.executable
        completed = subprocess.CompletedProcess([], 0, "pinned-test-version", "")
        with patch.object(mutants.shutil, "which", side_effect=locate) as calls, patch.object(mutants.subprocess, "run", return_value=completed):
            result = mutants.tool_identity(kind)
        return result, [x.args[0] for x in calls.call_args_list]

    def test_gate_records_only_tools_it_actually_uses(self):
        result, calls = self.probe("gate", missing={"lean", "lake"})
        self.assertEqual(set(result), {"sail", "z3", "cc", "python3"})
        self.assertEqual(set(calls), set(result))
        self.assertNotIn("lean", result)

    def test_native_requires_both_native_tools(self):
        for name in ("lean", "lake"):
            with self.subTest(tool=name), self.assertRaisesRegex(ValueError, "missing compiler input tool " + name):
                self.probe("lean", missing={name})

    def test_required_common_tool_still_fails_closed(self):
        for kind in ("gate", "lean"):
            for name in ("sail", "z3", "cc", "python3"):
                with self.subTest(kind=kind, tool=name), self.assertRaisesRegex(ValueError, "missing compiler input tool " + name):
                    self.probe(kind, missing={name})


class Compare(unittest.TestCase):
    def setUp(self):
        self.expected = {n: "survive" for n in mutants.NAMES}
        self.expected["m_k1"] = "kill"

    def test_new_survivor_is_a_regression(self):
        actual = dict(self.expected, m_k1="survive")
        self.assertEqual(mutants.compare(self.expected, actual), (["m_k1"], []))

    def test_new_kill_is_an_improvement(self):
        actual = dict(self.expected, mut_srf="kill")
        self.assertEqual(mutants.compare(self.expected, actual), ([], ["mut_srf"]))

    def test_unchanged_outcomes_pass(self):
        self.assertEqual(mutants.compare(self.expected, dict(self.expected)), ([], []))

    def test_bad_lists(self):
        full = "\n".join("%s survive" % n for n in mutants.NAMES)
        mutants.read_expected(full)
        for bad in (full + "\nm_k1 kill", full.replace("m_k1 survive", "m_k1 maybe"), full.replace("m_k1 survive", ""),
                    full + "\nextra kill"):
            with self.subTest(bad=bad[-20:]):
                with self.assertRaises(ValueError):
                    mutants.read_expected(bad)


class Run(unittest.TestCase):
    """The whole run with a stand-in model_prove.sh that fails only when the barrier mutant is present."""

    FAKE = '#!/usr/bin/env bash\nset -euo pipefail\nhere="$(cd "$(dirname "$0")/.." && pwd)"\n' \
           'if grep -q "c <= r" "$here/model/postlude/tick.sail"; then echo "FAIL p_barrier_window: expected unsat, got sat (exit 0)"; ' \
           'exit 1; fi\necho "profiles: $PROVE_PROFILES"\n'

    # The extra obligations (test/formal/prove_extra.sh) run first and fail only when the tie rule of fround is the mutant's.
    FAKE_EXTRA = '#!/usr/bin/env bash\nset -euo pipefail\nhere="$(cd "$(dirname "$0")/../.." && pwd)"\n' \
                 'if grep -q "rem >= half" "$here/model/core/numerics/fp.sail"; then ' \
                 'echo "FAIL p_tie_f16_k1: expected unsat, got sat (exit 0)"; exit 1; fi\necho "extra ok"\n'

    # The SRF ring test fails only for the mutant that folds the 64 slots to 32.
    FAKE_SRF = '#!/usr/bin/env bash\nset -euo pipefail\nhere="$(cd "$(dirname "$0")/../.." && pwd)"\n' \
               'if grep -q "emod_int(d, 32)" "$here/model/core/streams.sail"; then ' \
               'echo "Assertion failed: srf_ring: a write on a live diagonal was lost to another live diagonal (A)"; exit 1; fi\necho "srf ok"\n'

    # The lane kernel test fails only for the mutant that adds b to b in the saturating add.
    FAKE_LANES = '#!/usr/bin/env bash\nset -euo pipefail\nhere="$(cd "$(dirname "$0")/../.." && pwd)"\n' \
                 'if grep -q "sat_t(t, b + b)" "$here/model/core/numerics/lanes.sail"; then ' \
                 'echo "Assertion failed: lanes: saturating add"; exit 1; fi\necho "lanes ok"\n'

    # The bank test fails only when the mutant disables bank-conflict reporting.
    FAKE_BANK = ('#!/usr/bin/env bash\nset -euo pipefail\necho "profiles: $PROVE_PROFILES"\nhere="$(cd "$(dirname "$0")/../.." && pwd)"\n'
                 'if grep -Fq "if false & blocked then" "$here/model/postlude/tick.sail"; then '
                 'echo "Assertion failed: bank conflict was not reported"; exit 1; fi\necho "bank ok"\n')

    # The Lean check (kind `lean`) fails only for the mutants that write the wrong slot or tag, or open a barrier window without a Sync.
    FAKE_LEAN = '#!/usr/bin/env bash\nset -euo pipefail\nhere="$(cd "$(dirname "$0")/../../.." && pwd)"\n' \
                'echo "work: ${LEAN_CHECK_DIR:-none} model: ${OPENTSP_MODEL_DIR:-none}"\n' \
                'if grep -q "u);" "$OPENTSP_MODEL_DIR/slices/cu/cu_iq.sail" && grep -q "+ 1), u);" "$OPENTSP_MODEL_DIR/slices/cu/cu_iq.sail"; then ' \
                'echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -q "tag + 1);" "$OPENTSP_MODEL_DIR/slices/cu/cu_iq.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -q "_                  => true," "$OPENTSP_MODEL_DIR/postlude/tick.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "match i { IFETCHM(_) => (), _ => () }" "$OPENTSP_MODEL_DIR/postlude/tick.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "else (if a.e > b.e then a.e else b.e);" "$OPENTSP_MODEL_DIR/core/numerics/fp.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "if s == 0 then zero_bits(f, true)" "$OPENTSP_MODEL_DIR/core/numerics/fp.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "let neg = a.neg == b.neg;" "$OPENTSP_MODEL_DIR/core/numerics/fp.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "s <= c & c <= r" "$OPENTSP_MODEL_DIR/postlude/tick.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "iq_head[q] + n + 1" "$OPENTSP_MODEL_DIR/slices/cu/cu_iq.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "max(iq_len[q] - n + 1, 0)" "$OPENTSP_MODEL_DIR/slices/cu/cu_iq.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'if grep -Fq "iq_len[q] = iq_len[q] + 2;" "$OPENTSP_MODEL_DIR/slices/cu/cu_iq.sail"; then echo "lean_check: a proof does not check"; exit 1; fi\n' \
                'echo "lean_check: ok profile=$1 theorems=8"\n'

    # Mini unit tests: m_s3 (C2C busy), m_t1 (Permute bare key), m_ws1 (wtype_split), m_be1/m_bf1 (bfp8).
    FAKE_UNIT = ('#!/usr/bin/env bash\nset -euo pipefail\nhere="$(cd "$(dirname "$0")/../.." && pwd)"\n'
                 'if grep -q "a + c2c_cycles_per_vector > c + 1" "$here/model/slices/c2c/c2c_insts.sail"; then '
                 'echo "Assertion failed: c2c: link busy window"; exit 1; fi\n'
                 'if grep -Fq "entry_latency(name ^" "$here/model/core/timing.sail"; then '
                 'echo "Assertion failed: unit_tests/test_vxm_mxm.sail:681.74-681.75"; exit 1; fi\n'
                 'if grep -Fq "p <= 64" "$here/model/core/numerics/transcendental.sail"; then '
                 'echo "Assertion failed: unit_tests/test_slices.sail:692.114-692.115"; exit 1; fi\n'
                 'if grep -q "t == 0b11" "$here/model/slices/mxm/mxm_insts.sail"; then '
                 'echo "Pattern match failure in wtype_split_backwards"; exit 1; fi\n'
                 'if grep -q "n - 15" "$here/model/slices/mxm/mxm_insts.sail"; then '
                 'echo "Assertion failed: unit_tests/test_ext_types.sail:187.41-187.42"; exit 1; fi\n'
                 'if grep -q "blk_exp(ws, k) + blk_exp(xs, k) + 47" "$here/model/slices/mxm/mxm_insts.sail"; then '
                 'echo "Assertion failed: unit_tests/test_ext_types.sail:187.41-187.42"; exit 1; fi\n'
                 'echo "unit_mini ok"\n')


    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "spec"
        shutil.copytree(REAL / "model", self.root / "model")
        (self.root / "config").mkdir(parents=True)
        (self.root / "tools").mkdir()
        (self.root / "test/formal").mkdir(parents=True)
        (self.root / "test/unit_tests").mkdir(parents=True)
        (self.root / "test/export").mkdir(parents=True)
        (self.root / "test/formal/lean").mkdir(parents=True)
        (self.root / "c_emulator").mkdir()
        (self.root / "tools/model_prove.sh").write_text(self.FAKE)
        (self.root / "test/formal/prove_extra.sh").write_text(self.FAKE_EXTRA)
        (self.root / "test/formal/srf_ring_test.sh").write_text(self.FAKE_SRF)
        (self.root / "test/formal/lanes_test.sh").write_text(self.FAKE_LANES)
        (self.root / "test/formal/bank_agt_clash.sh").write_text(self.FAKE_BANK)
        (self.root / "test/formal/unit_mini.sh").write_text(self.FAKE_UNIT)
        (self.root / "test/formal/lean/lean_check.sh").write_text(self.FAKE_LEAN)
        for attr in ("SPEC", "EXPECTED"):
            self.addCleanup(setattr, mutants, attr, getattr(mutants, attr))
        tool_patch = patch.object(mutants, "tool_identity", return_value={"test-tool": {"sha256": "fixed"}})
        tool_patch.start()
        self.addCleanup(tool_patch.stop)
        mutants.SPEC = self.root
        self.expected = Path(self.tmp.name) / "expected.txt"
        mutants.EXPECTED = self.expected
        self.work = Path(self.tmp.name) / "work"
        self.addCleanup(os.environ.pop, "PROVE_MUTANTS_WORK", None)
        os.environ["PROVE_MUTANTS_WORK"] = str(self.work)

    def expect(self, kills):
        self.expected.write_text("".join("%s %s\n" % (n, "kill" if n in kills else "survive") for n in mutants.NAMES))

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = mutants.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_survivor_extra_parallelism_reaches_only_two_actual_children(self):
        extra = self.root / "test/formal/prove_extra.sh"
        extra.write_text('#!/bin/sh\nprintf "EXTRA_JOBS=%s\\n" "${PROVE_EXTRA_JOBS:-unset}"\n')
        with patch.dict(os.environ, {"PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS": "8", "PROVE_EXTRA_JOBS": "7"}):
            for name, wanted in [("m_l0", "8"), ("m_c2", "8"), ("m_s1", "1")]:
                with self.subTest(mutant=name):
                    mutants.run_one(name, self.work, "mini", self.work)
                    log = (self.work / (name + ".log")).read_text()
                    self.assertIn("EXTRA_JOBS=" + wanted, log)
                    receipt = json.loads((self.work / (name + ".inputs.json")).read_text())
                    self.assertEqual(receipt["environment"]["PROVE_EXTRA_JOBS"], wanted)

    def test_survivor_parallelism_defaults_to_one(self):
        extra = self.root / "test/formal/prove_extra.sh"
        extra.write_text('#!/bin/sh\nprintf "EXTRA_JOBS=%s\\n" "${PROVE_EXTRA_JOBS:-unset}"\n')
        with patch.dict(os.environ):
            os.environ.pop("PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS", None)
            os.environ["PROVE_EXTRA_JOBS"] = "7"
            mutants.run_one("m_l0", self.work, "mini", self.work)
        self.assertIn("EXTRA_JOBS=1", (self.work / "m_l0.log").read_text())

    def test_survivor_parallelism_invalid_configuration_is_rejected(self):
        self.expect(set())
        for value in ("0", "9", "not-an-integer"):
            with self.subTest(value=value), patch.dict(os.environ, {"PROVE_MUTANTS_SURVIVOR_EXTRA_JOBS": value}):
                code, _, error = self.run_main("--only", "m_l0")
                self.assertEqual(code, 2)
                self.assertIn("must be an integer from 1 to 8", error)

    def test_input_archive_and_before_after_identity(self):
        name, outcome, _, _ = mutants.run_one("m_k1", self.work, "mini", self.work)
        self.assertEqual(outcome, "kill")
        receipt = json.loads((self.work / (name + ".inputs.json")).read_text())
        self.assertFalse(receipt["source_drift"])
        self.assertFalse(receipt["tool_drift"])
        self.assertEqual(receipt["before"], receipt["after"])
        archive = self.work / receipt["archive"]
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), receipt["archive_sha256"])
        with tarfile.open(archive) as contents:
            for relative, digest in receipt["before"].items():
                self.assertEqual(hashlib.sha256(contents.extractfile(relative).read()).hexdigest(), digest)
            original = contents.extractfile("base/model/postlude/tick.sail").read()
            changed = contents.extractfile("model/postlude/tick.sail").read()
            self.assertIn(b"s <= c & c < r", original)
            self.assertIn(b"s <= c & c <= r", changed)
        self.assertIn("lean_check.sh", receipt["gate"][2])

    def test_source_drift_overrides_a_counterexample_marker(self):
        script = self.root / "test/formal/lean/lean_check.sh"
        script.write_text(self.FAKE_LEAN.replace('if grep -q',
            'echo "// deliberate source drift" >> "$here/model/postlude/tick.sail"\nif grep -q', 1))
        _, outcome, reason, _ = mutants.run_one("m_k1", self.work, "mini", self.work)
        self.assertEqual(outcome, "error")
        self.assertIn("inputs drifted", reason)
        receipt = json.loads((self.work / "m_k1.inputs.json").read_text())
        self.assertTrue(receipt["source_drift"])
        self.assertTrue((self.work / receipt["archive"]).exists())

    def test_generated_lean_sources_exclude_build_outputs(self):
        tree = self.work / "generated"
        directory = tree / "lean-work/gen/out/Out"
        directory.mkdir(parents=True)
        (directory / "Tick.lean").write_text("-- compiler input")
        (directory / "Tick.olean").write_text("build output")
        found = mutants.generated_lean_inputs(tree)
        self.assertEqual(set(found), {"lean-work/gen/out/Out/Tick.lean"})

    def test_run_one_reports_kill_and_survive(self):
        name, outcome, how, _ = mutants.run_one("m_k1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_k1", "kill"))
        self.assertEqual(how, "LEAN a proof does not check")
        name, outcome, how, _ = mutants.run_one("m_s3", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_s3", "kill"))
        self.assertIn("TEST assertion: c2c: link busy window", how)
        name, outcome, how, _ = mutants.run_one("m_ws1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_ws1", "kill"))
        self.assertIn("Pattern match failure", how)
        name, outcome, how, _ = mutants.run_one("m_be1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_be1", "kill"))
        self.assertIn("TEST assertion", how)
        self.assertIn("test_ext_types", how)
        name, outcome, how, _ = mutants.run_one("m_bf1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_bf1", "kill"))
        self.assertIn("TEST assertion", how)
        self.assertIn("test_ext_types", how)
        name, outcome, how, _ = mutants.run_one("m_t1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_t1", "kill"))
        self.assertIn("TEST assertion", how)
        self.assertIn("test_vxm_mxm", how)
        name, outcome, how, _ = mutants.run_one("m_z1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_z1", "kill"))
        self.assertIn("TEST assertion", how)
        self.assertIn("test_slices", how)

    def test_expect_current_passes_when_outcomes_match(self):
        self.expect({"m_k1", "m_k2", "m_f1", "m_f2", "m_s1", "mut_srf", "m_s2", "m_s3", "mut_bank", "m_t1", "m_ws1", "m_be1", "m_bf1", "m_z1", "m_l1", "m_l2", "m_l3", "m_r1", "m_a1", "m_a2", "m_a3"})
        code, out, _ = self.run_main("--expect-current", "--jobs", "2")
        self.assertEqual(code, 0, out)
        n = len(mutants.NAMES)
        # Controlled semantic producers: numeric SMT, bounded tests, eleven native routes.
        self.assertIn("proofs: 1/%d, tests: 9/%d, lean: 11/%d, survivors: %d/%d" % (n, n, n, n - 21, n), out)

    def test_the_summary_separates_proofs_tests_lean_and_survivors(self):
        actual = {"a": "kill", "b": "kill", "c": "kill", "d": "survive", "e": "error", "f": "kill"}
        hows = {"a": "FAIL p_x", "b": "TEST assertion: y", "c": "LEAN a proof does not check", "d": "", "e": "gate error (exit 1): rerun",
                "f": "FAIL p_z,p_w"}
        self.assertEqual(mutants.summary(actual, hows), "proofs: 2/5, tests: 1/5, lean: 1/5, survivors: 1/5")
        self.assertEqual(mutants.summary({"a": "kill", "d": "survive"}, {"a": "FAIL p_x", "d": ""}),
                         "proofs: 1/2, tests: 0/2, survivors: 1/2")

    def test_frontend_native_routes_use_actual_target_and_isolated_gate(self):
        expected = {"m_k1": ("Barrier", "p_barrier_window"),
                    "m_k2": ("IqPop", "p_iq_pop_conservation"),
                    "m_f1": ("IqPop", "p_iq_pop_accounting"),
                    "m_f2": ("IqPush", "p_iq_push_accounting")}
        for name in expected:
            with self.subTest(name=name):
                self.assertEqual(mutants.KIND.get(name), "lean")
                _, result, reason, _ = mutants.run_one(name, self.work, "mini", self.work)
                self.assertEqual((result, reason), ("kill", "LEAN a proof does not check"))
                receipt = json.loads((self.work / (name + ".inputs.json")).read_text())
                self.assertIn("lean_check.sh", receipt["gate"][2])
                self.assertNotIn("prove_extra.sh", receipt["gate"][2])
                self.assertNotIn("model_prove.sh", receipt["gate"][2])

    def test_a_lean_mutant_runs_the_lean_check_alone_on_its_own_model(self):
        name, outcome, how, _ = mutants.run_one("m_l1", self.work, "mini", self.work)
        self.assertEqual((name, outcome, how), ("m_l1", "kill", "LEAN a proof does not check"))
        self.assertIn("lean-work", (self.work / "m_l1.log").read_text())
        self.assertNotIn("srf ok", (self.work / "m_l1.log").read_text())
        self.assertEqual(mutants.run_one("m_r1", self.work, "mini", self.work)[1], "kill")
        self.assertIn("lean-work", (self.work / "m_r1.log").read_text())
        self.assertNotIn("srf ok", (self.work / "m_r1.log").read_text())

    def test_a_build_failure_of_the_mutated_copy_is_an_error_not_a_kill(self):
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (self.root / "tools/model_prove.sh").write_text('#!/usr/bin/env bash\necho "Type error: unbound identifier" >&2\nexit 1\n')
        name, outcome, how, _ = mutants.run_one("mut_bank", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("mut_bank", "error"))
        self.assertIn("gate error (exit 1)", how)

    def test_a_solver_timeout_is_its_own_verdict_not_a_proof_kill(self):
        # : check_smt_obligations prints FAIL for any answer but the wanted one, `timeout` and `unknown` included; a mutant that only
        # slows an obligation is not killed by it.
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (self.root / "tools/model_prove.sh").write_text(
            '#!/usr/bin/env bash\necho "FAIL p_f16_anchors: expected unsat, got timeout (exit 0)"\nexit 1\n')
        name, outcome, how, _ = mutants.run_one("mut_bank", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("mut_bank", "timeout"))
        self.assertIn("p_f16_anchors", how)
        self.assertEqual(mutants.tally({"mut_bank": outcome}, {"mut_bank": how}), (0, 0, 0, 0, 0))
        (self.root / "tools/model_prove.sh").write_text(
            '#!/usr/bin/env bash\necho "FAIL p_a: expected unsat, got unknown (exit 0)"\n'
            'echo "FAIL p_b: expected unsat, got sat (exit 0)"\nexit 1\n')
        name, outcome, how, _ = mutants.run_one("mut_bank", self.work, "mini", self.work)
        self.assertEqual(outcome, "kill")
        self.assertEqual(how, "FAIL p_b")

    def test_every_registered_native_module_rejection_is_classified(self):
        # Use the production registration so adding a proof root cannot silently
        # leave the producer or consumer's semantic-rejection list behind.
        production = (REAL / "test/formal/lean/lean_check.sh").read_text()
        sys.path.insert(0, str(REAL / "test/formal/lean"))
        import check_axioms
        roots = set(re.findall(r'"Proofs\.([A-Za-z0-9_]+)"', production))
        proofdir = REAL / "test/formal/lean/proofs"
        semantic = {name for name in roots
                    if check_axioms.declared_theorems((proofdir / (name + ".lean")).read_text())}
        producer = re.search(r"theorem_error=.*?Proofs/\(([^)]+)\)", production)[1]
        self.assertEqual(semantic, set(producer.split("|")))
        self.assertIn("PatrolCycleFrames", roots)
        for module in sorted(roots | {"UnregisteredProof"}):
            with self.subTest(module=module):
                driver = self.root / "test/formal/lean/lean_check.sh"
                driver.write_text(
                    "#!/usr/bin/env bash\n"
                    + f"echo 'error: Proofs/{module}.lean:44:1: unsolved goals'\n"
                    + "echo 'error: Lean exited with code 1'\n"
                    + "echo 'error: build failed'\n"
                    + "echo 'lean_check: a proof does not check'\nexit 1\n"
                )
                expected = "kill" if module in semantic else "error"
                self.assertEqual(mutants.run_one("m_l1", self.work, "mini", self.work)[1], expected)

    def test_nested_native_resource_diagnostics_are_not_kills(self):
        header = "error: Proofs/Resources.lean:44:1: Tactic simp failed with a nested error:\n"
        wrappers = "error: Lean exited with code 1\nerror: build failed\nlean_check: a proof does not check\n"
        for continuation in ("dependency loader failed\n", "maximum recursion depth has been reached\n",
                             "(deterministic) timeout at whnf, maximum number of heartbeats (200000) has been reached\n"):
            with self.subTest(continuation=continuation):
                driver = self.root / "test/formal/lean/lean_check.sh"
                driver.write_text("#!/usr/bin/env bash\ncat <<'RESOURCE'\n" + header + continuation + wrappers + "RESOURCE\nexit 1\n")
                self.assertEqual(mutants.run_one("m_l1", self.work, "mini", self.work)[1], "error")

    def test_a_crashed_lean_check_is_an_error_not_a_kill(self):
        (self.root / "test/formal/lean/lean_check.sh").write_text('#!/usr/bin/env bash\necho "lake: no network" >&2\nexit 1\n')
        self.assertEqual(mutants.run_one("m_l1", self.work, "mini", self.work)[1], "error")

    def test_actual_cli_outer_error_is_not_a_proof_kill(self):
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        for producer in (outer_cli_error_output, outer_extra_error_output):
            with self.subTest(producer=producer.__name__):
                output = producer()
                self.assertIn("FAIL ", output)
                self.assertIn("ERROR SMT ", output)
                (self.root / "tools/model_prove.sh").write_text(
                    "#!/usr/bin/env bash\ncat <<'PROTOCOL_EVIDENCE'\n" + output + "PROTOCOL_EVIDENCE\nexit 1\n")
                self.assertEqual(mutants.run_one("mut_bank", self.work, "mini", self.work)[1], "error")

    def test_mixed_solver_infrastructure_never_counts_as_a_kill(self):
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        for extra in ('ERROR SMT p_other: stderr=broken', 'fatal: solver dependency missing',
                      'FAIL p_other: expected unsat, got unsat (exit 0)'):
            with self.subTest(extra=extra):
                (self.root / "tools/model_prove.sh").write_text(
                    "#!/usr/bin/env bash\nprintf '%s\\n' 'FAIL p_good: expected unsat, got sat (exit 0)' '" + extra + "'\nexit 1\n")
                self.assertEqual(mutants.run_one("mut_bank", self.work, "mini", self.work)[1], "error")

    def test_an_extra_obligation_kills_the_tie_mutant(self):
        name, outcome, how, _ = mutants.run_one("m_s1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_s1", "kill"))
        self.assertIn("FAIL p_tie_f16_k1", how)

    def test_the_srf_ring_test_kills_the_slot_mutant(self):
        name, outcome, how, _ = mutants.run_one("mut_srf", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("mut_srf", "kill"))
        self.assertEqual(how, "TEST assertion: srf_ring: a write on a live diagonal was lost to another live diagonal (A)")

    def test_the_lane_test_kills_the_saturating_add_mutant(self):
        name, outcome, how, _ = mutants.run_one("m_s2", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_s2", "kill"))
        self.assertIn("TEST assertion: lanes", how)

    def test_a_gate_stopped_by_a_signal_is_an_error_not_a_kill(self):
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (self.root / "tools/model_prove.sh").write_text('#!/usr/bin/env bash\nkill -TERM $$\n')
        name, outcome, how, _ = mutants.run_one("mut_bank", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("mut_bank", "error"))
        self.assertIn("signal", how)
        self.expect(set())
        code, out, err = self.run_main("--expect-current", "--only", "mut_bank")
        self.assertEqual(code, 3)
        self.assertIn("proofs: 0/0, tests: 0/0, survivors: 0/0", out)
        self.assertIn("ERROR: the gate of mut_bank ended without a verdict (gate stopped by a signal", err)
        self.assertNotIn("is now killed", err)

    def test_expect_current_fails_on_a_new_survivor(self):
        self.expect({"m_k1", "m_l0"})
        code, _, err = self.run_main("--expect-current")
        self.assertEqual(code, 1)
        self.assertIn("REGRESSION: m_l0 survives but is a required kill", err)

    def test_expect_current_fails_when_a_survivor_is_now_killed(self):
        self.expect(set())
        code, _, err = self.run_main("--expect-current")
        self.assertEqual(code, 1)
        self.assertIn("m_k1 is now killed", err)

    def test_without_expect_current_only_regressions_fail(self):
        self.expect(set())
        self.assertEqual(self.run_main("--only", "m_k1")[0], 0)
        self.expect({"m_k1", "m_l0"})
        self.assertEqual(self.run_main("--only", "m_l0")[0], 1)

    def test_gate_run_without_native_tools_keeps_matching_receipts(self):
        completed = subprocess.CompletedProcess([], 0, "pinned-test-version", "")
        def locate(name, **_kwargs):
            return None if name in {"lean", "lake"} else sys.executable
        with patch.object(mutants, "tool_identity", side_effect=REAL_TOOL_IDENTITY), patch.object(mutants.shutil, "which", side_effect=locate), patch.object(mutants.subprocess, "run", return_value=completed):
            name, outcome, how, _ = mutants.run_one("m_s1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_s1", "kill"))
        self.assertIn("p_tie_f16_k1", how)
        receipt = json.loads((self.work / "m_s1.inputs.json").read_text())
        self.assertEqual(receipt["tools_before"], receipt["tools_after"])
        self.assertEqual(set(receipt["tools_before"]), {"sail", "z3", "cc", "python3"})
        self.assertFalse(receipt["tool_drift"])

    def test_actual_run_refuses_missing_required_tools(self):
        completed = subprocess.CompletedProcess([], 0, "pinned-test-version", "")
        for mutant, absent in [("m_s1", "z3"), ("m_l1", "lean"), ("m_l1", "lake")]:
            def locate(name, **_kwargs):
                return None if name == absent else sys.executable
            with self.subTest(mutant=mutant, absent=absent), patch.object(mutants, "tool_identity", side_effect=REAL_TOOL_IDENTITY), patch.object(mutants.shutil, "which", side_effect=locate), patch.object(mutants.subprocess, "run", return_value=completed):
                _, outcome, how, _ = mutants.run_one(mutant, self.work, "mini", self.work)
            self.assertEqual(outcome, "error")
            self.assertIn("cannot identify proof tool inputs", how)

    def test_profiles_reach_the_gate(self):
        self.expect(set())
        self.run_main("--only", "mut_bank", "--profiles", "mini ref")
        self.assertIn("profiles: mini ref", (self.work / "logs/mut_bank.log").read_text())

    def test_unknown_mutant_is_rejected(self):
        self.expect(set())
        self.assertEqual(self.run_main("--only", "nope")[0], 2)

    def test_scratch_copies_are_removed_and_logs_kept(self):
        self.expect(set())
        self.run_main("--only", "mut_bank")
        self.assertFalse((self.work / "mut_bank").exists())
        self.assertTrue((self.work / "logs/mut_bank.log").exists())

    def test_interrupt_stops_the_whole_gate_process_tree(self):
        self.expect(set())
        # Both selected gates must reach the sleeper; this test targets cancellation, not bank semantics.
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        pidfile = Path(self.tmp.name) / "sleeper.pid"
        (self.root / "tools/model_prove.sh").write_text(
            '#!/usr/bin/env bash\nsleep 40 &\necho $! >> "%s"\nwait\n' % pidfile)
        for state in (mutants.INTERRUPTED,):
            self.addCleanup(state.clear)
        timer = threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.time()
        code, _, err = self.run_main("--jobs", "2", "--only", "mut_bank", "--only", "m_a4")
        self.assertLess(time.time() - started, 20, "the interrupt did not stop the gates promptly")
        self.assertEqual(code, 130)
        self.assertIn("interrupted", err)
        pids = [int(x) for x in pidfile.read_text().split()]
        self.assertEqual(len(pids), 2)
        deadline = time.time() + 10
        alive = pids
        while alive and time.time() < deadline:
            alive = [pid for pid in alive if subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0]
            time.sleep(0.2)
        self.assertEqual(alive, [], "the gate's child processes outlived the interrupt")
        self.assertFalse((self.work / "mut_bank").exists() or (self.work / "m_a4").exists(), "scratch copies left behind")

    def test_the_tree_is_not_touched(self):
        self.expect({"m_k1"})
        before = (self.root / "model/postlude/tick.sail").read_text()
        self.run_main("--only", "m_k1")
        self.assertEqual((self.root / "model/postlude/tick.sail").read_text(), before)

    def test_fp_mutants_run_their_numeric_obligations_before_unrelated_wide_queries(self):
        suites = "numerics ties frontend wide hold fptheory"
        script = ('#!/usr/bin/env bash\nset -euo pipefail\n'
                  'if [ "$*" != "' + suites + '" ]; then '
                  'echo "FAIL p_wide_replace: expected unsat, got timeout (exit 0)"; exit 1; fi\n'
                  'echo "FAIL p_ovf_f16: expected unsat, got sat (exit 0)"; exit 1\n')
        (self.root / "test/formal/prove_extra.sh").write_text(script)
        name, outcome, how, _ = mutants.run_one("m_n1", self.work, "mini", self.work)
        self.assertEqual((name, outcome), ("m_n1", "kill"))
        self.assertIn("FAIL p_ovf_f16", how)

    def test_solver_protocol_errors_are_not_proof_kills(self):
        (self.root / "test/formal/bank_agt_clash.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        bad_results = (
            'FAIL p_example: expected unsat, got (error "invalid SMT expression") (exit 1)',
            'FAIL p_example: expected unsat, got sat (exit 7)',
        )
        for result in bad_results:
            with self.subTest(result=result):
                (self.root / "tools/model_prove.sh").write_text(
                    "#!/usr/bin/env bash\nprintf '%s\\n' '" + result + "'\nexit 1\n")
                self.assertEqual(mutants.run_one("mut_bank", self.work, "mini", self.work)[1], "error")


class ToolVersionCache(unittest.TestCase):
    def setUp(self):
        if hasattr(mutants, "TOOL_VERSIONS"):
            mutants.TOOL_VERSIONS.clear()

    def test_concurrent_identity_executes_one_probe_per_binary(self):
        completed = subprocess.CompletedProcess([], 0, "actual-version", "")
        def execute(*_args, **_kwargs):
            time.sleep(0.02)
            return completed
        with patch.object(mutants.shutil, "which", return_value=sys.executable), patch.object(mutants.subprocess, "run", side_effect=execute) as runs:
            with mutants.concurrent.futures.ThreadPoolExecutor(8) as workers:
                rows = list(workers.map(lambda _: REAL_TOOL_IDENTITY("gate"), range(8)))
        self.assertEqual(runs.call_count, 1)
        self.assertTrue(all(row == rows[0] for row in rows))

    def test_changed_binary_or_path_requires_fresh_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/"a", Path(tmp)/"b"
            a.write_bytes(b"\x7fELFbinary-one")
            b.write_bytes(b"\x7fELFbinary-one")
            completed = subprocess.CompletedProcess([], 0, "actual-version", "")
            with patch.object(mutants.subprocess, "run", return_value=completed) as runs:
                with patch.object(mutants.shutil, "which", return_value=str(a)):
                    first = REAL_TOOL_IDENTITY("gate")
                    a.write_bytes(b"\x7fELFbinary-two")
                    changed = REAL_TOOL_IDENTITY("gate")
                with patch.object(mutants.shutil, "which", return_value=str(b)):
                    moved = REAL_TOOL_IDENTITY("gate")
            self.assertEqual(runs.call_count, 3)
            self.assertNotEqual(first["sail"]["sha256"], changed["sail"]["sha256"])
            self.assertNotEqual(first["sail"]["path"], moved["sail"]["path"])

    def test_unchanged_shell_wrapper_reprobes_backend_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp)/"solver-wrapper"
            wrapper.write_text("#!/bin/sh\nexec underlying-solver \"$@\"\n")
            backend = ["backend-one"]
            def execute(*_args, **_kwargs):
                return subprocess.CompletedProcess([], 0, backend[0], "")
            with patch.object(mutants.shutil, "which", return_value=str(wrapper)), patch.object(mutants.subprocess, "run", side_effect=execute):
                before = REAL_TOOL_IDENTITY("gate")
                backend[0] = "backend-two"
                after = REAL_TOOL_IDENTITY("gate")
            self.assertEqual(before["z3"]["sha256"], after["z3"]["sha256"])
            self.assertNotEqual(before["z3"]["version"], after["z3"]["version"])

    def test_failed_probe_is_not_cached(self):
        completed = subprocess.CompletedProcess([], 0, "actual-version", "")
        with patch.object(mutants.shutil, "which", return_value=sys.executable), patch.object(mutants.subprocess, "run", side_effect=[subprocess.TimeoutExpired("native-tool", 10), completed]) as runs:
            with self.assertRaises(subprocess.TimeoutExpired):
                REAL_TOOL_IDENTITY("gate")
            row = REAL_TOOL_IDENTITY("gate")
        self.assertEqual(runs.call_count, 2)
        self.assertEqual(row["sail"]["version"], "actual-version")




@unittest.skipUnless(sys.platform == "linux", "Linux child adoption")
class OwnedChildReaping(unittest.TestCase):
    def tearDown(self):
        mutants.INTERRUPTED.clear()

    def gate(self):
        process = subprocess.Popen(["bash", "-c", "sleep 40 & wait"], start_new_session=True)
        with mutants.LOCK:
            mutants.RUNNING.add(process.pid)
        self.addCleanup(mutants.RUNNING.discard, process.pid)
        time.sleep(0.1)
        return process

    def test_denied_adoption_still_terminates_owned_group(self):
        process = subprocess.Popen(["sleep", "40"], start_new_session=True)
        with mutants.LOCK:
            mutants.RUNNING.add(process.pid)
        self.addCleanup(mutants.RUNNING.discard, process.pid)
        class Denied:
            def prctl(self, *_args): return -1
        with patch("ctypes.CDLL", return_value=Denied()):
            with self.assertRaises(OSError):
                mutants.stop_all()
        process.wait()
        with self.assertRaises(ProcessLookupError):
            os.killpg(process.pid, 0)

    def test_foreign_child_is_not_consumed(self):
        process = self.gate()
        foreign = subprocess.Popen(["bash", "-c", "exit 23"])
        time.sleep(0.1)
        mutants.stop_all()
        process.wait()
        self.assertEqual(foreign.wait(), 23)


if __name__ == "__main__":
    unittest.main()
