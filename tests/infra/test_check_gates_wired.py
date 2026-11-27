"""Tests of tools/check_gates_wired.py (run by `just check-docs`): a check or recipe that no gate calls fails, wiring it or listing it
with a reason passes, a stale or malformed list entry fails, a name that is only printed, a comment, a docstring or a directory that is
only named wires nothing, a CI workflow is not a local gate, Rust tests need a cargo run that builds their features, and the real tree
passes."""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/check_gates_wired.py"
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    fails += 0 if ok else 1


def run(files, allow=None):
    """Run the checker on a scratch tree holding `files` ({path: text}); returns (exit status, output)."""
    with tempfile.TemporaryDirectory() as d:
        for path, text in files.items():
            p = pathlib.Path(d, path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        args = [sys.executable, str(TOOL), "--root", d]
        if allow is not None:
            pathlib.Path(d, "tools").mkdir(exist_ok=True)
            pathlib.Path(d, "tools/gates_allow.txt").write_text(allow)
        else:
            pathlib.Path(d, "tools").mkdir(exist_ok=True)
            pathlib.Path(d, "tools/gates_allow.txt").write_text("")
        r = subprocess.run(args, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


BASE_FILES = {"tests/foo/test_a.py": "print('ok')\n"}
BASE = {
    "justfile": "test:\n    python3 tests/foo/test_a.py\n",
    **BASE_FILES,
}
ORPHAN = {"tests/foo/test_orphan.py": "print('nobody calls me')\n"}

rc, out = run(BASE)
check("a tree whose test is called by a root recipe passes", rc == 0, out)

rc, out = run({**BASE, **ORPHAN})
check("a test that no gate calls fails and is named", rc == 1 and "tests/foo/test_orphan.py" in out, out)

for subproject in ("opentsp-sw-toolchain", "opentsp-vllm"):
    test = f"{subproject}/tests/test_orphan.py"
    rc, out = run({**BASE, test: "print('nobody calls me')\n"})
    check(f"an orphaned test in {subproject} is reported", rc == 1 and test in out, out)
    wired = {**BASE, test: "", "justfile": BASE["justfile"] + f"    python3 -m pytest {subproject}/tests\n"}
    rc, out = run(wired)
    check(f"the mapped collector selects tests from {subproject}", rc == 0, out)

wired = {**BASE, **ORPHAN, "justfile": BASE["justfile"] + "    python3 tests/foo/test_orphan.py\n"}
rc, out = run(wired)
check("wiring it into a root recipe passes", rc == 0, out)

rc, out = run({**BASE, **ORPHAN}, "tests/foo/test_orphan.py | manual | needs the vLLM environment; run by hand before a release\n")
check("listing it with a reason and an outside need passes", rc == 0, out)

rc, out = run(wired, "tests/foo/test_orphan.py | manual | needs the vLLM environment\n")
check("a listed entry that is wired now fails: remove it", rc == 1 and "wired now" in out, out)

rc, out = run({**BASE, **ORPHAN}, "tests/foo/test_orphan.py | sometimes | run by hand\n")
check("a reason outside the list fails", rc == 1 and "reason" in out, out)

rc, out = run({**BASE, **ORPHAN}, "tests/foo/test_orphan.py | manual |\n")
check("an entry without a note fails", rc == 1, out)

rc, out = run(BASE, "tests/foo/test_gone.py | manual | deleted long ago\n")
check("an entry that names nothing fails", rc == 1 and "not a check or recipe" in out, out)

ORPH = {**BASE, **ORPHAN}
rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | run by hand before a release\n")
check("a manual entry with neither a stated duration nor an outside need fails", rc == 1 and "states no duration" in out, out)

rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | run when the area changes\n")
check("a manual entry whose note says only when to run it fails", rc == 1 and "states no duration" in out, out)

rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | the local check (about 2 minutes)\n")
check("a manual entry of under five minutes with no outside need fails", rc == 1 and "five minutes" in out, out)

rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | about 20 s on the simulator\n")
check("a manual entry of seconds with no outside need fails", rc == 1 and "five minutes" in out, out)

rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | needs the vLLM environment (about 2 minutes)\n")
check("a manual entry that needs an outside environment passes whatever its duration", rc == 0, out)

rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | on a quiet host (about 3 minutes)\n")
check("a manual entry that needs a quiet host passes", rc == 0, out)

rc, out = run(ORPH, "tests/foo/test_orphan.py | manual | the local check (about ten minutes)\n")
check("a manual entry of five minutes or more passes", rc == 0, out)

commented = {**BASE, **ORPHAN, "justfile": BASE["justfile"] + "    # python3 tests/foo/test_orphan.py\n"}
rc, out = run(commented)
check("a mention in a comment wires nothing", rc == 1 and "tests/foo/test_orphan.py" in out, out)

rc, out = run({**BASE, "tools/just/x.just": "orphan:\n    echo hi\n"})
check("a recipe of tools/just that no gate calls fails", rc == 1 and "orphan" in out, out)

rc, out = run({"justfile": "test: orphan\n", "tests/foo/test_a.py": "", "tools/just/x.just": "orphan:\n    python3 tests/foo/test_a.py\n"})
check("a recipe that a root depends on is wired, and so is the test it runs", rc == 0, out)

# `iss-replay-test` was declared an area root, which counted it as wired whether or not any gate ran it; `test` runs it, so it is no root
rc, out = run({"justfile": "test:\n    echo nothing\n", "tools/just/x.just": "iss-replay-test:\n    cargo test --features replay\n"})
check("iss-replay-test is not a root: when no gate runs it, it is reported", rc == 1 and "iss-replay-test" in out, out)
rc, out = run({"justfile": "test:\n    just iss-replay-test\n", "tools/just/x.just": "iss-replay-test:\n    cargo test --features replay\n"})
check("iss-replay-test run by `test` is wired", rc == 0, out)

rc, out = run({
    "justfile": "test:\n    bash tools/run.sh\n",
    "tools/run.sh": "set -e\njust orphan\n",
    "tools/just/x.just": "orphan:\n    echo hi\n",
})
check("a recipe that a wired script runs with `just` is wired", rc == 0, out)

rc, out = run({
    "justfile": "test:\n    for s in a-step b-step; do\n        just \"$s\"\n    done\n",
    "tools/just/x.just": "a-step:\n    echo a\nb-step:\n    echo b\n",
})
check("a recipe named in a `for` list is wired", rc == 0, out)

rc, out = run({"justfile": "test:\n    $PYTHON -m pytest -q tests/showcase\n", "tests/showcase/test_x.py": "", "tests/showcase/sub/test_y.py": ""})
check("a directory handed to a runner wires the tests below it", rc == 0, out)

rc, out = run({**BASE, "opentsp-arch-spec/tools/gen_x.py": "import sys\nif '--check' in sys.argv:\n    pass\n"})
check("a script of the specification tools with a --check mode that nobody calls fails", rc == 1 and "gen_x.py" in out, out)

rc, out = run({**BASE, "opentsp-arch-spec/tools/gen_y.py": "print('a generator without a check mode')\n"})
check("a generator without a check mode is not a gate", rc == 0, out)

WF = ".github/" + "workflows/ci.yml"
CI = {WF: "jobs:\n  a:\n    steps:\n      - run: python3 tests/foo/test_orphan.py\n"}
rc, out = run({**BASE, **CI, **ORPHAN})
check("a test that only a CI workflow runs is not wired: CI does not run here", rc == 1 and "CI workflow" in out and "tests/foo/test_orphan.py" in out, out)

rc, out = run({**BASE, **CI, **ORPHAN}, "tests/foo/test_orphan.py | ci | run by the CI job only\n")
check("a CI-only test listed with the reason ci passes", rc == 0, out)

rc, out = run({**BASE, **ORPHAN}, "tests/foo/test_orphan.py | ci | run by the CI job only\n")
check("an entry with the reason ci that no workflow runs fails", rc == 1 and "no CI workflow" in out, out)

rc, out = run({**wired, **CI}, "tests/foo/test_orphan.py | ci | run by the CI job only\n")
check("an entry with the reason ci that a local gate runs fails: remove it", rc == 1 and "wired now" in out, out)

rc, out = run({**BASE, **ORPHAN}, "tests/foo/test_orphan.py | nightly | the schedule\n")
check("the reason nightly is refused: no scheduler runs anything", rc == 1 and "reason" in out, out)

# A script is wired by a line that runs it: never by a name that is only printed, nor by a directory that is only named.
ORPHAN_SH = {"opentsp-arch-spec/test/formal/orphan_check.sh": "exit 1\n"}
rc, out = run({**BASE, **ORPHAN_SH})
check("a check under the specification test directory that nobody calls fails", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + '    echo "note: the check is run by opentsp-arch-spec/test/formal/orphan_check.sh"\n',
})
check("a path printed by echo in a recipe wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    bash tools/run.sh\n",
    "tools/run.sh": "echo \"see test/formal/orphan_check.sh\" >&2\nprintf 'and %s\\n' opentsp-arch-spec/test/formal/orphan_check.sh\n",
})
check("a path printed by echo or printf in a script wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/note.py\n",
    "tools/note.py": "print('run opentsp-arch-spec/test/formal/orphan_check.sh by hand')\n",
})
check("a path printed by print in a Python script wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/note.py\n",
    "tools/note.py": "print('subprocess.run([\"bash\", \"opentsp-arch-spec/test/formal/orphan_check.sh\"])')\n"
    "sys.exit('subprocess.run([\"bash\", \"opentsp-arch-spec/test/formal/orphan_check.sh\"])')\n",
})
check("a bare path that print or sys.exit prints wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/note.py\n",
    "tools/note.py": "msg = 'see opentsp-arch-spec/test/formal/orphan_check.sh for the check'\n",
})
check("a path inside a longer message of a Python script wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    bash tools/run.sh\n",
    "tools/run.sh": 'echo "first line\n  opentsp-arch-spec/test/formal/orphan_check.sh\nlast line"\n',
})
check("a path inside a multi-line echo wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/note.py\n",
    "tools/note.py": '"""Example: subprocess.run(["bash", "opentsp-arch-spec/test/formal/orphan_check.sh"])"""\n'
    '# subprocess.run(["bash", "opentsp-arch-spec/test/formal/orphan_check.sh"])\n',
})
check("a path in a docstring or comment of a Python script wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/note.py\n",
    "tools/note.py": "import sys\nsys.path.insert(0, 'opentsp-arch-spec/test')\n",
})
check("a directory put on sys.path wires nothing below it", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/note.py\n",
    "tools/note.py": 'CHECK_DIRS = ("opentsp-arch-spec/tools/", "opentsp-arch-spec/test/")\n',
})
check("a directory named in a Python tuple wires nothing below it", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    bash tools/run.sh\n",
    "tools/run.sh": "ls opentsp-arch-spec/test\nls opentsp-arch-spec/test/formal/\n",
})
check("a directory named by a command that is not a test runner wires nothing below it", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 -m pytest -q opentsp-arch-spec/test/formal\n",
})
check("a directory handed to pytest wires the checks below it", rc == 0, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nsubprocess.run(['bash', 'opentsp-arch-spec/test/formal/orphan_check.sh'], check=True)\n",
})
check("a script that a Python gate runs with subprocess is wired", rc == 0, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nsubprocess.run('python3 -m pytest opentsp-arch-spec/test/formal', shell=True)\n",
})
check("a directory that a Python gate hands to pytest wires the checks below it", rc == 0, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "def run(cmd):\n    return cmd\n\nrun(['bash', 'opentsp-arch-spec/test/formal/orphan_check.sh'])\nTABLE = {'a': 'bash opentsp-arch-spec/test/formal/orphan_check.sh'}\n",
})
check("a path in a table or in a call that spawns nothing wires nothing", rc == 1 and "orphan_check.sh" in out, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nCHECK = 'opentsp-arch-spec/test/formal/orphan_check.sh'\nsubprocess.run(['bash', CHECK], check=True)\n",
})
check("a script named by a variable that a subprocess call uses is wired", rc == 0, out)

rc, out = run({
    "justfile": "test:\n    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nsubprocess.run(['just', 'orphan'], check=True)\n",
    "tools/just/x.just": "orphan:\n    echo hi\n",
})
check("a recipe that a Python gate runs with `just` is wired", rc == 0, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nsubprocess.run(['bash', ROOT / 'opentsp-arch-spec' / 'test' / 'formal' / 'orphan_check.sh'])\n",
})
check("a script a Python gate names by its file name in a path expression is wired", rc == 0, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    bash tools/run.sh\n",
    "tools/run.sh": "bash opentsp-arch-spec/test/formal/orphan_check.sh\n",
})
check("a script that a shell gate runs is wired", rc == 0, out)

rc, out = run({
    **BASE_FILES,
    **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/check_gates_wired.py\n",
    "tools/check_gates_wired.py": "import subprocess\nsubprocess.run(['bash', 'opentsp-arch-spec/test/formal/orphan_check.sh'])\n",
})
check("the text of the checker itself is no edge", rc == 1 and "orphan_check.sh" in out, out)

# A spawned shell prefix can add a check conditionally; an unrelated string must not wire it.
prefix_script = "opentsp-arch-spec/test/formal/orphan_check.sh"
rc, out = run({
    **BASE_FILES, **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nhelper = 'bash " + prefix_script + " && ' if True else ''\ngate = helper + 'true'\nsubprocess.run(['bash', '-c', gate])\n",
})
check("a spawned conditional assignment wires its check", rc == 0, out)
rc, out = run({
    **BASE_FILES, **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nprefix = ''\nprefix += 'bash " + prefix_script + "'\nsubprocess.run(['true'])\n",
})
check("an unused shell prefix wires nothing", rc == 1 and "orphan_check.sh" in out, out)
rc, out = run({
    **BASE_FILES, **ORPHAN_SH,
    "justfile": BASE["justfile"] + "    python3 tools/run.py\n",
    "tools/run.py": "import subprocess\nprefix = ''\nprefix = 'bash " + prefix_script + "'\nprefix = 'true'\nsubprocess.run(['bash', '-c', prefix])\n",
})
check("an overwritten shell command wires nothing", rc == 1 and "orphan_check.sh" in out, out)

# Rust integration tests and the cargo features they need.
CARGO = {
    "Cargo.toml": '[workspace]\nmembers = ["crates/a"]\n',
    "crates/a/Cargo.toml": '[package]\nname = "a"\nversion = "0.1.0"\n\n[features]\nreplay = []\n',
    "crates/a/tests/plain.rs": "#[test]\nfn t() {}\n",
    "crates/a/tests/gated.rs": '#![cfg(feature = "replay")]\n#[test]\nfn t() {}\n',
}
JUST_WS = {"justfile": "test:\n    cargo test --workspace\n"}
rc, out = run({**JUST_WS, **CARGO})
check(
    "a feature-gated Rust test that no gate builds with its feature is reported",
    rc == 1 and "crates/a/tests/gated.rs" in out and "replay" in out and "plain.rs" not in out,
    out,
)

rc, out = run({"justfile": "test:\n    cargo test --workspace\n    cargo test -p a --features replay --test gated\n", **CARGO})
check("a feature-gated Rust test built with its feature by a gate passes", rc == 0, out)

rc, out = run({"justfile": "test:\n    cargo test --workspace --all-features\n", **CARGO})
check("--all-features builds every gated Rust test", rc == 0, out)

rc, out = run({"justfile": "test:\n    cargo test --workspace\n    cargo test -p a --test gated\n", **CARGO})
check("naming the gated test without its feature does not build it", rc == 1 and "gated.rs" in out, out)

rc, out = run({"justfile": "test:\n    cargo test --workspace\n    cargo test -p a --features replay --test plain\n", **CARGO})
check("the feature on another test does not build the gated one", rc == 1 and "gated.rs" in out, out)

rc, out = run({"justfile": "test:\n    cargo test --workspace\n    # cargo test -p a --features replay --test gated\n", **CARGO})
check("a cargo line in a comment builds nothing", rc == 1 and "gated.rs" in out, out)

rc, out = run({"justfile": "test:\n    cargo test --workspace\n    echo cargo test -p a --features replay --test gated\n", **CARGO})
check("a cargo line that is only printed builds nothing", rc == 1 and "gated.rs" in out, out)

rc, out = run({"justfile": "test:\n    cargo test -p a --test plain\n", **CARGO})
check("--test names the one Rust test a gate runs: an unnamed test is not wired", rc == 1 and "gated.rs" in out, out)

rc, out = run({"justfile": "test:\n    cargo test -p a --features replay\n", **CARGO})
check("a package run with the feature and no --test runs every test of the package", rc == 0, out)

rc, out = run({"justfile": "test:\n    cargo test -p b\n", **CARGO})
check("a run of another package wires no test of this one", rc == 1 and "plain.rs" in out, out)

rc, out = run({"justfile": "test:\n    cargo test --release -p a --features replay --test gated -- --ignored\n    cargo test --workspace\n", **CARGO})
check("arguments after -- are not cargo arguments", rc == 0, out)

rc, out = run({"justfile": "test:\n    cargo test --workspace --lib\n", **CARGO})
check("a library-only run builds no integration test", rc == 1 and "plain.rs" in out, out)

REQ = {
    **CARGO,
    "crates/a/Cargo.toml": CARGO["crates/a/Cargo.toml"] + '\n[[test]]\nname = "plain"\nrequired-features = ["replay"]\n',
}
rc, out = run({**JUST_WS, **REQ, "crates/a/tests/gated.rs": "#[test]\nfn t() {}\n"})
check("a required-features entry of Cargo.toml gates a Rust test as well", rc == 1 and "plain.rs" in out and "replay" in out, out)

rc, out = run({**JUST_WS, **CARGO}, "crates/a/tests/gated.rs | manual | needs the replay build\n")
check("a Rust test that no gate runs can be listed with a reason", rc == 0, out)

rc, out = run({"justfile": "test: rs\nrs:\n    cargo test --workspace --features replay\n", **CARGO})
check("a Rust test run by a recipe that a root depends on is wired", rc == 0, out)

CI_RUST = {WF: "jobs:\n  a:\n    steps:\n      - run: cargo test --workspace --all-features\n"}
rc, out = run({**JUST_WS, **CARGO, **CI_RUST})
check("a Rust test that only a CI workflow builds is not wired", rc == 1 and "CI workflow" in out and "gated.rs" in out, out)

r = subprocess.run([sys.executable, str(TOOL)], capture_output=True, text=True)
check("the real tree passes", r.returncode == 0, r.stdout + r.stderr)

r = subprocess.run([sys.executable, str(TOOL), "--root", "/nonexistent-root-xyz"], capture_output=True, text=True)
check("a root that does not exist is refused with exit 2", r.returncode == 2 and "check_gates_wired" in r.stdout + r.stderr, r.stdout + r.stderr)

print(f"test_check_gates_wired: {'FAIL' if fails else 'all passed'}")
sys.exit(1 if fails else 0)
