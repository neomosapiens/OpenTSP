"""Build scripts print Sail's diagnostics when it fails; model_check.sh --check never writes the schema."""
from pathlib import Path
import json
import re
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SPEC = Path(__file__).resolve().parents[2]


class FakeSail(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "bin").mkdir()
        (self.root / "share/sail/lib").mkdir(parents=True)
        self.env = dict(os.environ, PATH="%s%s%s" % (self.root / "bin", os.pathsep, os.environ["PATH"]))

    def sail(self, body):
        path = self.root / "bin/sail"
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run_script(self, name, *args, cwd=None):
        return subprocess.run(["bash", str(SPEC / "tools" / name), *args], env=self.env, capture_output=True, text=True,
                              cwd=cwd or self.root)


class Diagnostics(FakeSail):
    def test_a_failing_sail_build_shows_its_diagnostics(self):
        self.sail('echo "Error: boom in model/core/streams.sail line 38" >&2\nexit 1\n')
        for script in ("isa_exhaustive.sh", "isa_db_check.sh", "model_unit_tests.sh"):
            with self.subTest(script=script):
                result = self.run_script(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("sail failed", result.stderr)
                self.assertIn("boom in model/core/streams.sail line 38", result.stderr)

    def test_no_script_discards_sail_diagnostics(self):
        for script in SPEC.glob("tools/*.sh"):
            for number, line in enumerate(script.read_text().splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                self.assertFalse("sail " in line and "2>/dev/null" in line, "%s:%d" % (script.name, number))


class NumericalWorkerAccounting(FakeSail):
    def setUp(self):
        super().setUp()
        self.sail("exit 0\n")
        compiler = self.root / "bin/cc"
        compiler.write_text("""#!/bin/bash
set -eu
while [ "$1" != "-o" ]; do shift; done
cat > "$2" <<'WORKER'
#!/bin/bash
case "${0##*/}" in
    f16_exhaustive) count=$(($3 * 65536)) ;;
    f32_sample) count=$(($3 * 4)) ;;
esac
case "$WORKER_RESULT" in
    short) count=$((count - 1)) ;;
    invalid) printf "wrong %s 0\n" "$count"; exit 0 ;;
esac
printf "ok %s 0\n" "$count"
[ "$WORKER_RESULT" != fail ]
WORKER
chmod +x "$2"
""")
        compiler.chmod(0o755)
        self.env.update(JOBS="2", F16_X_MAX="2", F32_PAIRS="2")
        self.env.pop("F16_X_INSN", None)

    def run_numerical(self, mode, script):
        return subprocess.run(
            ["bash", str(SPEC / "test/numerics" / script), "add"],
            capture_output=True, text=True, env=dict(self.env, WORKER_RESULT=mode), timeout=10,
        )

    def test_distinguishes_complete_workers_from_incomplete_or_failed_workers(self):
        for script in ("f16_exhaustive.sh", "f32_sample.sh"):
            for mode in ("pass", "short", "invalid", "fail"):
                with self.subTest(script=script, mode=mode):
                    result = self.run_numerical(mode, script)
                    output = result.stdout + result.stderr
                    if mode == "pass":
                        self.assertEqual(result.returncode, 0, output)
                    else:
                        self.assertEqual(result.returncode, 1, output)


class ExhaustiveWorkerAccounting(FakeSail):
    def test_failed_worker_with_a_plausible_summary_fails(self):
        self.sail("exit 0\n")
        compiler = self.root / "bin/cc"
        compiler.write_text("""#!/bin/bash
set -eu
while [ "$1" != "-o" ]; do shift; done
cat > "$2" <<'WORKER'
#!/bin/bash
printf 'ok 0 0\n'
exit 5
WORKER
chmod +x "$2"
""")
        compiler.chmod(0o755)
        helper = self.root / "bin/skip-python-helper"
        helper.write_text("#!/bin/sh\nexit 0\n")
        helper.chmod(0o755)
        self.env.update(PYTHON=str(helper), JOBS="2")
        result = self.run_script("isa_exhaustive.sh")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("worker", result.stderr)


class ExportBackends(FakeSail):
    def test_empty_and_invalid_selection_are_rejected_before_export(self):
        log = self.root / "generation-would-run"
        self.sail(f"touch '{log}'\nexit 1\n")
        for selection in ("", "   ", "lean unknown", "lean lean"):
            with self.subTest(selection=selection):
                result = subprocess.run(
                    ["bash", str(SPEC / "test/export/export_check.sh")], timeout=10,
                    env=dict(self.env, EXPORT_CHECK_BACKENDS=selection), capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("export_check:", result.stderr)
                self.assertFalse(log.exists())


class ExtraObligationsWidth(FakeSail):
    """prove_extra.sh proves every suite once, and again at 256-bit integers on request, requiring the same verdicts."""

    def setUp(self):
        super().setUp()
        sys.path.insert(0, str(SPEC / "tools"))
        import check_smt_obligations as c
        source = (SPEC / "test/formal/props/ties.sail").read_text()
        self.names = c.obligations(source)
        self.with_twins = c.obligations(source + "\n" + c.twin_source(source))
        self.calls = self.root / "calls"
        # The fake generator writes one query per obligation whose verdict is the expected one, except that a query
        # generated at width 256 is wrong when the file `wrong256` exists. The default pass has the twins, the wide pass has not.
        lines = ['echo "$*" >> "%s"' % self.calls, 'wide=0; case "$*" in *256*) wide=1;; esac',
                 'wrong=0; [ $wide = 1 ] && [ -f "%s" ] && wrong=1' % (self.root / "wrong256")]
        for name, wanted in self.with_twins.items():
            right, wrong = ("false", "true") if wanted == "unsat" else ("true", "false")
            write = 'if [ $wrong = 1 ]; then v=%s; else v=%s; fi; printf "(assert %%s)\\n(check-sat)\\n" "$v" > %s.smt2' % (wrong, right, name)
            lines.append('if [ $wide = 0 ]; then %s; fi' % write if name.startswith("twin_") else write)
        self.sail("\n".join(lines) + "\nexit 0\n")
        self.env["PROVE_ALLOW_Z3"] = "1"

    def run_extra(self, **env):
        return subprocess.run(["bash", str(SPEC / "test/formal/prove_extra.sh"), "ties"], env=dict(self.env, **env),
                              capture_output=True, text=True)

    def test_the_default_run_is_one_pass_at_the_default_width(self):
        result = self.run_extra()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)
        self.assertNotIn("256", self.calls.read_text())

    def test_the_wide_run_adds_a_pass_at_256_bits(self):
        result = self.run_extra(PROVE_EXTRA_WIDE="1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(len(calls), 2)
        self.assertIn("--smt-int-size 256", calls[1])
        self.assertIn("at both integer widths", result.stdout)

    def test_a_different_verdict_at_256_bits_fails_the_run(self):
        (self.root / "wrong256").write_text("")
        result = self.run_extra(PROVE_EXTRA_WIDE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("integer width 256", result.stdout)


class PropsProject(unittest.TestCase):
    """props_project.sh assembles the proof project; with PROPS_TWINS=1 it adds the twin of every property to the props module."""

    def build(self, **env):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        result = subprocess.run(["bash", str(SPEC / "test/formal/props_project.sh"), tmp.name], capture_output=True, text=True,
                                env=dict(os.environ, **env))
        self.assertEqual(result.returncode, 0, result.stderr)
        return Path(tmp.name)

    def test_without_twins_the_project_is_the_models(self):
        work = self.build()
        self.assertFalse((work / "twins.sail").exists())
        self.assertNotIn("twins.sail", (work / "props.sail_project").read_text())

    def test_with_twins_every_property_has_one(self):
        work = self.build(PROPS_TWINS="1")
        project = (work / "props.sail_project").read_text()
        self.assertIn("    twins.sail,\n}", project)
        twins = (work / "twins.sail").read_text()
        sys.path.insert(0, str(SPEC / "tools"))
        import check_smt_obligations as c
        properties = [n for n, v in c.obligations((work / "obligations.txt").read_text()).items() if v == "unsat"]
        self.assertGreaterEqual(len(properties), 29)
        for name in properties:
            self.assertIn("function twin_%s(" % name, twins)
        # the obligation texts stay the model's: the twins are not part of the inventory
        self.assertNotIn("twin_", (work / "obligations.txt").read_text())


class SchemaCheck(FakeSail):
    def setUp(self):
        super().setUp()
        self.spec = self.root / "spec"
        for rel in ("tools", "config", "model"):
            shutil.copytree(SPEC / rel, self.spec / rel)
        self.schema = self.spec / "config/schema.json"
        self.real = self.root / "generated.json"
        shutil.copy(self.schema, self.real)
        # The fake typechecker succeeds and writes the "generated" schema where Sail is told to.
        self.sail('out=""; while [ $# -gt 0 ]; do [ "$1" = --output-schema ] && out="$2"; shift; done\n'
                  '[ -n "$out" ] && cp "$GENERATED" "$out"\nexit 0\n')
        self.env["GENERATED"] = str(self.real)

    def run_check(self, *args):
        return subprocess.run(["bash", str(self.spec / "tools/model_check.sh"), *args], env=self.env,
                              capture_output=True, text=True)

    def perturb(self):
        """Change a description in the generated schema: a semantic difference the invariants pass keeps."""
        text = self.real.read_text()
        marker = '"description": "'
        self.assertIn(marker, text)
        self.real.write_text(text.replace(marker, marker + "CHANGED ", 1))

    def digest(self):
        return hashlib.sha256(self.schema.read_bytes()).hexdigest()

    def test_check_accepts_the_matching_schema_and_writes_nothing(self):
        before = self.digest()
        result = self.run_check("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("schema check ok", result.stdout)
        self.assertEqual(self.digest(), before)
        self.assertEqual([p.name for p in (self.spec / "config").glob(".schema.*")], [])

    def test_check_reports_a_difference_and_leaves_the_file_alone(self):
        self.perturb()
        before = self.digest()
        result = self.run_check("--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("differs from config/schema.json", result.stderr)
        self.assertEqual(self.digest(), before)

    def test_default_mode_still_replaces_the_schema(self):
        before = self.digest()
        self.perturb()
        self.assertEqual(self.run_check().returncode, 0)
        self.assertNotEqual(self.digest(), before)
        self.assertIn("CHANGED ", self.schema.read_text())

    def test_bad_argument(self):
        result = self.run_check("--nope")
        self.assertEqual(result.returncode, 2)


class ProfileCheckedFirst(FakeSail):
    """model_prove.sh runs the schema invariants on a profile before Sail reads it: a profile that breaks a relation of two keys
    is refused with the message of the relation, not by a type error about another key."""

    def test_a_profile_that_breaks_a_relation_is_refused_before_sail_runs(self):
        import json
        root = self.root / "spec"
        (root / "config/test").mkdir(parents=True)
        for name in ("tools", "model", "test"):
            (root / name).symlink_to(SPEC / name)
        for name in ("mini.json", "schema.json"):
            shutil.copy(SPEC / "config" / name, root / "config" / name)
        cfg = json.loads((SPEC / "config/mini.json").read_text())
        cfg["sram"]["words_per_slice"], cfg["sram"]["addr_bits"] = 16384, 13          # bank_bit < addr_bits no longer holds
        (root / "config/test/broken.json").write_text(json.dumps(cfg))
        calls = self.root / "sail_calls"
        self.sail('echo "$*" >> "%s"\nexit 0\n' % calls)
        result = subprocess.run(["bash", str(root / "tools/model_prove.sh")], capture_output=True, text=True, cwd=self.root,
                                env=dict(self.env, PROVE_PROFILES="test/broken", PROVE_ALLOW_Z3="1", PROVE_REACH="0", PROVE_WIDE="0"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sram.bank_bit", result.stdout + result.stderr)
        self.assertIn("breaks a schema invariant", result.stderr)
        self.assertFalse(calls.exists() and "--config" in calls.read_text(), "sail was run on the broken profile")


class ConfigNormalised(FakeSail):
    """The Sail backends do not support `config` keys that are absent from the file (`Runtime configuration is not supported when generating
    code for target 'smt'`), so the proof, extra-proof, export and Lean flows hand Sail a profile normalised by
    `schema_invariants.with_defaults`: a profile without the optional `ras.diag_depth` reaches Sail with the key (16)."""

    def setUp(self):
        super().setUp()
        import json
        self.spec = self.root / "spec"
        (self.spec / "config/test").mkdir(parents=True)
        for name in ("tools", "model", "test"):
            (self.spec / name).symlink_to(SPEC / name)
        for name in ("schema.json", "ref.json"):
            shutil.copy(SPEC / "config" / name, self.spec / "config" / name)
        cfg = json.loads((SPEC / "config/mini.json").read_text())
        del cfg["ras"]["diag_depth"]
        for name in ("mini.json", "test/nodd.json"):
            (self.spec / "config" / name).write_text(json.dumps(cfg))
        self.seen = self.root / "seen"
        # a stand-in sail that records the `ras.diag_depth` of every --config file it is given, then fails
        self.sail('prev=""; for a in "$@"; do if [ "$prev" = --config ]; then python3 -c \'import json,sys; print(json.load(open(sys.argv[1]))["ras"].get("diag_depth", "MISSING"))\' "$a" >> "%s"; fi; prev="$a"; done\nexit 1\n' % self.seen)

    def run_one(self, script, *args, **env):
        subprocess.run(["bash", str(self.spec / script), *args], capture_output=True, text=True, cwd=self.root,
                       env=dict(self.env, PROVE_ALLOW_Z3="1", PROVE_REACH="0", PROVE_WIDE="0", LEAN_CHECK_DIR=str(self.root / "lean"),
                                EXPORT_CHECK_BACKENDS="lean", **env))
        return self.seen.read_text().split() if self.seen.exists() else []

    def check(self, script, *args, **env):
        got = self.run_one(script, *args, **env)
        self.assertTrue(got, "%s gave sail no --config" % script)
        self.assertEqual(set(got), {"16"}, "%s: sail was given %s" % (script, got))

    def test_model_prove(self):
        self.check("tools/model_prove.sh", PROVE_PROFILES="test/nodd")

    def test_prove_extra(self):
        self.check("test/formal/prove_extra.sh", "frontend")

    def test_export_check(self):
        self.check("test/export/export_check.sh", "test/nodd")

    def test_lean_check(self):
        # This config-forwarding test stops in the fake Sail generator. It must
        # exercise that path even on hosts without a Lean installation.
        self.env["HOME"] = str(self.root)
        lake = self.root / "bin/lake"
        lake.write_text("#!/bin/sh\nexit 99\n")
        lake.chmod(0o755)
        self.check("test/formal/lean/lean_check.sh", "test/nodd")


class ExtraObligationsKillLimit(unittest.TestCase):
    """prove_extra.sh must not kill a solver process before the solver's effective timeout: its kill limit is the gate's own
    KILL_AFTER (305 s, above the 300 s the toolchain's z3 wrapper grants), both in the limits it sets and in the line it prints."""

    def test_kill_limit_is_the_gates_own(self):
        import re
        sys.path.insert(0, str(SPEC / "tools"))
        import check_smt_obligations as c
        text = (SPEC / "test/formal/prove_extra.sh").read_text()
        limits = re.findall(r"c\.TIMEOUT, c\.KILL_AFTER = (\d+), (\d+)", text)
        printed = re.findall(r"--solver-info \"\$spec/test/formal/z3_version.txt\" (\d+) (\d+)", text)
        self.assertEqual(len(limits), 1)
        self.assertEqual(len(printed), 1)
        for _timeout, kill in limits + printed:
            self.assertEqual(int(kill), c.KILL_AFTER)



class Bf16NumericsObligations(unittest.TestCase):
    """: the numerics extra-proof suite must name bf16 obligations (FMT_BF16), not only f16/e5m2/e4m3."""

    REQUIRED = (
        "p_exact_bf16",
        "p_rne_bf16_k3",
        "p_rne_sub_bf16_j2",
        "p_ovf_bf16",
        "p_bf16_decode",
        "p_bf16_specials",
        "p_bf16_anchors",
    )

    def test_numerics_txt_lists_bf16(self):
        names = (SPEC / "test/formal/props/numerics.txt").read_text().split()
        missing = [n for n in self.REQUIRED if n not in names]
        self.assertEqual(missing, [], "numerics.txt missing bf16 obligations: %s" % missing)

    def test_numerics_sail_defines_each(self):
        body = (SPEC / "test/formal/props/numerics.sail").read_text()
        missing = [n for n in self.REQUIRED if ("function %s(" % n) not in body]
        self.assertEqual(missing, [], "numerics.sail missing bf16 functions: %s" % missing)



class ExtensionObligationCoverage(unittest.TestCase):
    """: hold/wide obligations; m_t1/m_ws1/m_be1/m_bf1 kill via unit_mini; hold binds tick.sail."""

    SURVIVE = ()
    KILLED = ("m_t1", "m_ws1", "m_be1", "m_bf1")
    HOLD = ("p_hold_free", "p_hold_parked_barrier")
    WIDE = ("p_wide_replace", "p_wide_sticky", "p_wide_wrap")

    def test_cr34_35_mutants_are_killed(self):
        rows = {}
        for line in (SPEC / "test/formal/mutants_expected.txt").read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, outcome = line.split()
            rows[name] = outcome
        missing = [n for n in self.SURVIVE if rows.get(n) != "survive"]
        self.assertEqual(missing, [], "expected survive rows for %s" % (missing,))
        missing_k = [n for n in self.KILLED if rows.get(n) != "kill"]
        self.assertEqual(missing_k, [], "expected kill rows for %s" % (missing_k,))
        body = (SPEC / "test/formal/mutants.py").read_text()
        for n in self.SURVIVE + self.KILLED:
            self.assertIn('("%s"' % n, body, "mutants.py must define %s" % n)
        self.assertIn("unit_mini_killers", body)
        killers = body.split("unit_mini_killers", 1)[1].split("}", 1)[0]
        for n in self.KILLED:
            self.assertIn('"%s"' % n, killers, n)
        self.assertIn('"m_s3"', killers)
        for n in self.SURVIVE:
            self.assertNotIn('"%s"' % n, killers, n)
        um = (SPEC / "test/formal/unit_mini.sh").read_text()
        self.assertIn("mini_ext.json", um)
        self.assertIn("unit_mini ok: mini_ext", um)
        self.assertIn("mini_dot.json", um)
        self.assertIn("unit_mini ok: mini_dot", um)
        self.assertTrue((SPEC / "config/test/mini_dot.json").is_file(), "mini_dot.json must be committed")

    def test_hold_suite_self_contained(self):
        """: hold proves via sail -smt on path_view props; no missing hold_check.py."""
        script = (SPEC / "test/formal/prove_extra.sh").read_text()
        self.assertNotIn("hold_check.py", script)
        self.assertRegex(script, r"\bhold\b")

    def test_hold_suite_uses_hold_decide(self):
        sail = (SPEC / "test/formal/props/hold.sail").read_text()
        inv = (SPEC / "test/formal/props/hold.txt").read_text().split()
        pe = (SPEC / "test/formal/prove_extra.sh").read_text()
        self.assertIn("hold_decide", sail)
        self.assertNotIn("function hold_decide", sail)  # model binding, not props twin
        self.assertNotIn("struct path_view", sail)
        self.assertNotIn("struct hold_view", sail)
        self.assertNotIn("hold_flat", sail)
        self.assertIn("option(mem_path)", sail)
        missing = [n for n in self.HOLD if n not in inv or ("function %s(" % n) not in sail]
        self.assertEqual(missing, [], "hold suite missing: %s" % missing)
        hold_arm = pe.split("hold)", 1)[1].split("*)", 1)[0]
        self.assertIn("postlude", hold_arm)
        self.assertRegex(pe, r"\bhold\b")

    def test_wide_wrap_obligation(self):
        sail = (SPEC / "test/formal/props/wide.sail").read_text()
        inv = (SPEC / "test/formal/props/wide.txt").read_text().split()
        missing = [n for n in self.WIDE if n not in inv or ("function %s(" % n) not in sail]
        self.assertEqual(missing, [], "wide suite missing: %s" % missing)
        self.assertIn("wrap_s", sail)
        self.assertIn("128", sail)



class Mz1ObservableKill(unittest.TestCase):
    """: m_z1 is an observable Ziv-cap mutant killed by unit_mini."""

    def test_m_z1_is_kill_via_unit_mini(self):
        rows = {}
        comments = {}
        for line in (SPEC / "test/formal/mutants_expected.txt").read_text().splitlines():
            raw = line
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, outcome = line.split()
            rows[name] = outcome
            if "#" in raw:
                comments[name] = raw.split("#", 1)[1]
        self.assertEqual(rows.get("m_z1"), "kill")
        self.assertRegex(comments.get("m_z1", ""), r"(?i)unit_mini|test_slices|p<=64")
        body = (SPEC / "test/formal/mutants.py").read_text()
        self.assertIn("p <= 64", body)
        killers = body.split("unit_mini_killers", 1)[1].split("}", 1)[0]
        self.assertIn('"m_z1"', killers)










class WidenedProofWiring(unittest.TestCase):
    """/07: prove-widened recipe and Sail config normalisation are wired (no prove run)."""

    def test_prove_widened_recipe_names_profiles(self):
        just = (SPEC.parent / "justfile").read_text() + "\n" + (SPEC.parent / "tools/just/formal.just").read_text() if (SPEC.parent / "tools/just/formal.just").is_file() else (SPEC.parent / "justfile").read_text()
        # also search formal-proofs and validation
        blob = just + "\n" + (SPEC.parent / "opentsp-arch-spec/docs/validation.md").read_text()
        self.assertRegex(blob, r"(?m)^prove-widened\s*:")
        for name in ("w16", "one_plane", "acc8", "c2c_wide"):
            self.assertIn(name, blob, name)

    def test_model_prove_normalises_diag_depth(self):
        text = (SPEC / "tools/model_prove.sh").read_text()
        self.assertRegex(text, r"with_defaults|--normalise|normalis")
        self.assertIn("diag_depth", text)
        # ConfigNormalised unit class already exists for the same concern when present
        scripts = (SPEC / "test/formal/test_scripts.py").read_text()
        self.assertTrue("ConfigNormalised" in scripts or "diag_depth" in text)



class TypedBf16RequiresExtension(unittest.TestCase):
    """: a typed row for bf16 on a profile without the bf16 extension is refused by schema_invariants."""

    def test_max_bf16_on_ref_is_refused(self):
        sys.path.insert(0, str(SPEC / "tools"))
        from opentsp_reference import schema as si
        cfg = json.loads((SPEC / "config/ref.json").read_text())
        self.assertNotIn("bf16", cfg.get("extensions", []))
        cfg["timing"]["overrides"] = list(cfg["timing"]["overrides"]) + [
            {"entry": "Max.bf16", "d_func": 5, "d_skew": 2}
        ]
        errs = si.check(cfg)
        self.assertTrue(any("bf16" in e and ("extension" in e or "Max.bf16" in e) for e in errs), errs)



class AllExtVectorsRecipe(unittest.TestCase):
    """: a just recipe regenerates model-extra/all_ext and the RTL sidecar without touching frozen vectors."""

    def test_formal_proofs_just_has_all_ext_vectors(self):
        text = (SPEC.parent / "tools/just/formal-proofs.just").read_text()
        self.assertRegex(text, r"(?m)^all-ext-vectors\s*:")
        self.assertIn("gen_all_ext_vectors.py", text)
        self.assertIn("gen_rtl_obligations.py", text)
        self.assertIn("--profile all_ext", text)



class InductionAssuranceBoundary(unittest.TestCase):
    """: validation.md must not count frontend_induction as independent assurance of Sail state."""

    def test_validation_says_not_independent_assurance(self):
        text = (SPEC.parent / "opentsp-arch-spec/docs/validation.md").read_text()
        self.assertIn("frontend_induction.smt2", text)
        self.assertRegex(text, r"not independent assurance of the model")



class DefaultProveExtraSuites(unittest.TestCase):
    """: the default prove_extra suite list includes hold and wide (inventory; no SMT run)."""

    def test_default_suites_name_hold_and_wide(self):
        text = (SPEC / "test/formal/prove_extra.sh").read_text()
        # default assignment lists both suites
        match = re.search(r'suites="\$\{\*:-([^}]+)\}"', text)
        self.assertIsNotNone(match)
        self.assertTrue({"domains", "packing", "hold", "wide"} <= set(match.group(1).split()))
        self.assertIn("p_wide_wrap", (SPEC / "test/formal/props/wide.txt").read_text().split())
        self.assertIn("p_hold_free", (SPEC / "test/formal/props/hold.txt").read_text().split())



class EncodingSourceBoundary(unittest.TestCase):
    """Validation documents the common-mode risk of the shared encoding catalogue."""

    def test_validation_names_encoding_source_limit(self):
        text = (SPEC.parent / "opentsp-arch-spec/docs/validation.md").read_text()
        _, heading, boundary = text.partition("### Independent encoding source")
        self.assertTrue(heading)
        boundary = boundary.split("\n## ", 1)[0]
        self.assertIn("share generated encoding data", boundary)
        self.assertIn("common-mode encoding error", boundary)
        self.assertIn("do not make the whole encoding independent", boundary)


class AllProfilesProofRecipe(unittest.TestCase):
    """: prove-profiles asks model_prove for every schema-valid profile (inventory; SMT run is separate)."""

    def test_prove_profiles_sets_prove_profiles_all(self):
        just = (SPEC.parent / "tools/just/formal.just").read_text()
        self.assertRegex(just, r"(?m)^prove-profiles\s*:")
        # the recipe body must set PROVE_PROFILES=all for model_prove.sh
        block = just.split("prove-profiles:", 1)[1].split("\n\n", 1)[0]
        self.assertIn("PROVE_PROFILES=all", block)
        model = (SPEC / "tools/model_prove.sh").read_text()
        self.assertIn('if [ "$profiles" = all ]; then', model)
        # validation states the remaining SMT cost / PARTLY
        val = (SPEC.parent / "opentsp-arch-spec/docs/validation.md").read_text()
        self.assertIn("prove-profiles", val)
        self.assertRegex(val, r"every schema-valid profile|PROVE_PROFILES=all")


class ExpectedMutantOutcomes(unittest.TestCase):
    """: the original surviving behavioural mutants are either killed or recorded as survive."""

    def test_original_five_have_explicit_outcomes(self):
        rows = {}
        for line in (SPEC / "test/formal/mutants_expected.txt").read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, outcome = line.split()
            rows[name] = outcome
        # Audit  five: mut_srf, m_s1, m_s2, m_s3, and the bank clash (mut_bank)
        self.assertEqual(rows["mut_srf"], "kill")
        self.assertEqual(rows["m_s1"], "kill")
        self.assertEqual(rows["m_s2"], "kill")
        self.assertEqual(rows["m_s3"], "kill")  # unit_mini / test_c2c_link_busy
        self.assertEqual(rows["mut_bank"], "kill")  # bank_agt_clash.sh / directed Q20
        blob = (SPEC / "test/formal/mutants_expected.txt").read_text()
        bank = [ln for ln in blob.splitlines() if ln.strip().startswith("mut_bank")][0]
        self.assertRegex(bank, r"(?i)bank_agt_clash|Q20|AGT")



class MutationAssuranceDisclosure(unittest.TestCase):
    """: validation.md records the proof-gate ratchet and PARTLY residuals."""

    def test_validation_discloses_mutation_scope(self):
        text = (SPEC.parent / "opentsp-arch-spec/docs/validation.md").read_text()
        self.assertIn("Proof-gate mutation", text)
        self.assertIn("mut_bank", text)
        self.assertRegex(text, r"mut_bank.*kill|kill.*mut_bank")
        self.assertIn("m_l0", text)
        self.assertIn("m_c2", text)
        self.assertIn("exact substitution count", text)  #  PARTLY


class AstOnlyVerdict(unittest.TestCase):
    """AST-only prove_extra must not claim SMT proved (false PASS under PROVE_EXTRA_AST_ONLY)."""

    def test_final_message_distinguishes_ast_only(self):
        src = (SPEC / "test/formal/prove_extra.sh").read_text()
        self.assertIn('PROVE_EXTRA_AST_ONLY', src)
        self.assertIn('ast-only: suites type-checked', src)
        # The SMT success line must be gated so AST-only cannot print it
        self.assertIn('[ "${PROVE_EXTRA_AST_ONLY:-0}" != 1 ]', src)



class WideAstOnly(unittest.TestCase):
    """: wide suite can be type-checked without SMT (AST-only) before a quiet prove."""

    def test_wide_inventory_lists_wrap(self):
        inv = [ln.split("#", 1)[0].strip() for ln in (SPEC / "test/formal/props/wide.txt").read_text().splitlines()]
        inv = [x for x in inv if x]
        self.assertEqual(inv, ["p_wide_replace", "p_wide_sticky", "p_wide_wrap"])

    def test_prove_extra_supports_ast_only_for_wide(self):
        src = (SPEC / "test/formal/prove_extra.sh").read_text()
        self.assertIn("PROVE_EXTRA_AST_ONLY", src)
        self.assertIn("ast-only: suites type-checked", src)
        self.assertIn("wide)", src)  # suite case arm



class ModelReachingObligations(unittest.TestCase):
    """Hold obligations bind the model driver and retain exact unit-test mutation controls."""

    KILLED = ("m_t1", "m_ws1", "m_be1", "m_bf1")

    def test_extension_mutants_killed_by_unit_tests(self):
        rows = {}
        for line in (SPEC / "test/formal/mutants_expected.txt").read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, outcome = line.split()
            rows[name] = outcome
        for n in self.KILLED:
            self.assertEqual(rows.get(n), "kill", n)

    def test_hold_suite_binds_model_tick_hold_decide(self):
        """prove_extra hold loads postlude and calls model hold_decide (no props twin)."""
        sail = (SPEC / "test/formal/props/hold.sail").read_text()
        pe = (SPEC / "test/formal/prove_extra.sh").read_text()
        tick = (SPEC / "model/postlude/tick.sail").read_text()
        self.assertIn("function hold_decide", tick)
        self.assertIn("option(mem_path)", tick)
        self.assertIn("hold_decide", sail)
        self.assertIn("option(mem_path)", sail)
        self.assertNotIn("function hold_decide", sail)
        self.assertNotIn("option(bits(1))", sail)
        self.assertNotIn("struct path_view", sail)
        self.assertNotIn("struct hold_view", sail)
        hold_arm = pe.split("hold)", 1)[1].split("*)", 1)[0]
        self.assertIn("postlude", hold_arm)
        self.assertIn('"slices", "postlude"', hold_arm)
        self.assertIn("files props.sail", hold_arm)



class AstOnlyWithoutDir(unittest.TestCase):
    """PROVE_EXTRA_AST_ONLY alone must not fall through to sail -smt."""

    def test_ast_only_gates_smt_even_without_ast_dir(self):
        src = (SPEC / "test/formal/prove_extra.sh").read_text()
        # Must skip SMT when AST_ONLY=1 regardless of AST_DIR
        self.assertRegex(
            src,
            r'\[ "\$\{PROVE_EXTRA_AST_ONLY:-0\}" != 1 \] \|\| continue',
        )
        # The SMT invocation must come after an AST_ONLY continue that is not nested only under AST_DIR
        smt = src.find("sail -smt --all-modules")
        # Find a top-level AST_ONLY continue before smt that is not indented under AST_DIR-only block alone
        before = src[:smt]
        self.assertIn('[ "${PROVE_EXTRA_AST_ONLY:-0}" != 1 ] || continue', before)
        # Stronger: after the AST_DIR block, a second AST_ONLY guard must exist before sail -smt
        self.assertIn(
            'AST_ONLY must skip SMT even when PROVE_EXTRA_AST_DIR is unset',
            src,
        )



class ProveExtraQuiet(unittest.TestCase):
    """: a quiet wrapper refuses SMT when host load is high or f16_exhaustive is live."""

    def test_wrapper_exists_and_gates_load(self):
        script = SPEC.parent / "tools/prove_extra_quiet.sh"
        self.assertTrue(script.is_file(), "missing tools/prove_extra_quiet.sh")
        text = script.read_text()
        self.assertIn("f16_exhaustive", text)
        self.assertIn("prove_extra.sh", text)
        self.assertIn("LOAD_MAX", text)

    def test_wrapper_refuses_when_load_max_zero(self):
        import os, subprocess
        script = SPEC.parent / "tools/prove_extra_quiet.sh"
        env = os.environ.copy()
        env["LOAD_MAX"] = "0"
        r = subprocess.run(["bash", str(script), "hold"], cwd=str(SPEC.parent), env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("refuse", r.stderr)



class ProofSuiteDocumentation(unittest.TestCase):
    """The validation chapter names the wide wrap and hold suites among prove_extra."""

    def test_validation_names_wide_wrap_and_hold(self):
        text = (SPEC.parent / "opentsp-arch-spec/docs/validation.md").read_text()
        self.assertIn("props/wide.sail", text)
        self.assertIn("props/hold.sail", text)
        self.assertRegex(text, r"128-bit wrap|p_wide_wrap|wrap")
        self.assertRegex(text, r"hold|tick\.sail")



class ProveExtraQuietRequiresSuite(unittest.TestCase):
    """Quiet wrapper must not default to all suites (accidental numerics under a load gate pass)."""

    def test_wrapper_requires_explicit_suite_args(self):
        text = (SPEC.parent / "tools/prove_extra_quiet.sh").read_text()
        self.assertIn('usage:', text.lower())
        self.assertRegex(text, r'\[ \$# -eq 0 \]|\[ \$# -lt 1 \]|require.*suite|usage:')



class WideQfbvStrategy(unittest.TestCase):
    """Exercise the actual wide-only query rewrite, preserving assertions and fail-closed protocol."""

    def rewrite(self, directory, suite="wide"):
        import re
        from unittest.mock import patch
        text = (SPEC / "test/formal/prove_extra.sh").read_text()
        code = text.split("# WIDE_QFBV_NORMALIZER_BEGIN", 1)[1].split("# WIDE_QFBV_NORMALIZER_END", 1)[0]
        import textwrap
        with patch.object(sys, "argv", ["driver", str(SPEC), suite + ".sail", str(directory)]):
            exec(textwrap.dedent(code), {"Path": Path, "sys": sys, "re": re})

    def test_exact_query_preservation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p_wide_replace.smt2"
            original = "(declare-const v (_ BitVec 8))\n(assert (= v v))\n(check-sat)\n"
            p.write_text(original)
            self.rewrite(td)
            self.assertEqual(p.read_text().replace("(check-sat-using qfbv)", "(check-sat)"), original)
            self.assertEqual(p.with_suffix(".smt2.original").read_text(), original)

    def test_other_suite_unchanged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p.smt2"; p.write_text("(check-sat)\n")
            self.rewrite(td, "frontend")
            self.assertEqual(p.read_text(), "(check-sat)\n")
            self.assertFalse(p.with_suffix(".smt2.original").exists())

    def test_malformed_command_inventory_fails(self):
        import tempfile
        for raw in ["(assert true)\n", "(check-sat)\n(check-sat)\n"]:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "p.smt2"; p.write_text(raw)
                with self.assertRaises(ValueError):
                    self.rewrite(td)
                self.assertEqual(p.read_text(), raw)

if __name__ == "__main__":
    unittest.main()
