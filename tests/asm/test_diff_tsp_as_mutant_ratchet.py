"""Check whether assembler differential tests catch wrong profiles and topology.

A forwarding control preserves parity. Wrappers that remap each test profile to ref or discard bad topology
must produce differences. A blind differential that always reports agreement must fail these controls.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "opentsp-sw-toolchain" / "tests"
CANONICAL_DIFF = TESTS / "diff_tsp_as.py"
# RATCHET_DIFF_TSP_AS points the test at a mutated copy of the differential (it must then fail).
DIFF = Path(os.environ.get("RATCHET_DIFF_TSP_AS", str(CANONICAL_DIFF)))
ASM = ROOT / "target" / "debug" / "ots-asm"


def load_diff():
    src = DIFF.read_text()
    src = re.sub(r'if __name__ == ["\']__main__["\']:\n.*', "if False:\n    pass\n", src, flags=re.S)
    # Pin __file__ to the canonical differential so SPEC/HERE resolve even when DIFF is a mutant copy.
    ns = {"__file__": str(CANONICAL_DIFF), "__name__": "diff_tsp_as_ratchet"}
    exec(compile(src, str(DIFF), "exec"), ns)
    return ns


def ensure_asm():
    """Build the debug `ots-asm` of this tree every time (an incremental no-op when current): a stale binary would make the ratchet
    test the wrong code. `CARGO_TARGET_DIR` is cleared so the binary is the one at `ASM`."""
    env = {k: v for k, v in os.environ.items() if k != "CARGO_TARGET_DIR"}
    subprocess.run(["cargo", "build", "-q", "-p", "ots-asm"], cwd=ROOT, check=True, env=env)
    assert ASM.is_file(), f"missing {ASM}"


def write_remap(path: Path, needle: str, replacement: str, exe: Path):
    path.write_text(
        "#!/usr/bin/env bash\n"
        "args=()\n"
        'for a in "$@"; do\n'
        f'  case "$a" in *{needle}) a="${{a%{needle}}}{replacement}";; esac\n'
        '  args+=("$a")\n'
        "done\n"
        f'exec "{exe}" "${{args[@]}}"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def write_topo_mutant(path: Path, good: Path, exe: Path):
    # After --topology, force the good file so bad topologies are ignored.
    path.write_text(
        "#!/usr/bin/env bash\n"
        "args=(); skip=0\n"
        'for a in "$@"; do\n'
        "  if [[ $skip -eq 1 ]]; then\n"
        f'    args+=("{good}")\n'
        "    skip=0; continue\n"
        "  fi\n"
        '  if [[ "$a" == "--topology" ]]; then args+=("$a"); skip=1; continue; fi\n'
        '  args+=("$a")\n'
        "done\n"
        f'exec "{exe}" "${{args[@]}}"\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def materialize_mut_same(path: Path) -> None:
    """Write a blind differential: new_key/topology phases always report 0 differences.

    A same()-only mutant is not enough: object rows also compare raw bytes (`ra != rb`), so
    remaps can still be caught. Blind phases reproduce the false "0 diffs" claim.
    """
    text = CANONICAL_DIFF.read_text()
    if "MUT_SAME blind" not in text:
        text2, n1 = re.subn(
            r"def new_key_profile_phase\(exe, mod, tmp\):[\s\S]*?\n(?=def topology_phase)",
            (
                "def new_key_profile_phase(exe, mod, tmp):\n"
                "    print('  new-key profiles: MUT_SAME blind')\n"
                "    return 22, 0\n\n\n"
            ),
            text,
            count=1,
        )
        text3, n2 = re.subn(
            r"def topology_phase\(exe, mod, tmp\):[\s\S]*?\n(?=def main\()",
            (
                "def topology_phase(exe, mod, tmp):\n"
                "    print('  mpack --topology: MUT_SAME blind')\n"
                "    return 6, 0\n\n\n"
            ),
            text2,
            count=1,
        )
        if n1 != 1 or n2 != 1:
            raise SystemExit(f"failed to gut phases (new_key={n1} topology={n2})")
        text = text3
    path.write_text(text)


def meta_red_same_mutant() -> int:
    """Require the blind differential to fail the intended controls."""
    if os.environ.get("RATCHET_META") == "1":
        return 0
    with tempfile.TemporaryDirectory() as directory:
        mutant = Path(directory) / "blind_differential.py"
        materialize_mut_same(mutant)
        env = dict(os.environ, RATCHET_DIFF_TSP_AS=str(mutant), RATCHET_META="1", RATCHET_SKIP_ASM_BUILD="1")
        result = subprocess.run(
            [os.environ.get("PYTHON", "python3"), str(Path(__file__).resolve())],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    print(result.stdout, end="")
    print(result.stderr, end="")
    observed = {line.split(" (compared ", 1)[0] for line in result.stdout.splitlines() if line.startswith("FAIL ")}
    expected = {f"FAIL mutant {profile} new_key bad>=1" for profile in ("w16", "acc8", "one_plane", "c2c_wide")}
    expected.add("FAIL mutant topology bad>=1")
    ok = result.returncode == 1 and observed == expected
    verdict = "PASS" if ok else "FAIL"
    print(f"{verdict} blind differential rejected by all five controls (child exit={result.returncode})")
    return 0 if ok else 1


def main():
    if os.environ.get("RATCHET_SKIP_ASM_BUILD") != "1":
        ensure_asm()
    elif not ASM.is_file():
        ensure_asm()
    d = load_diff()
    mod = d["references"]()[0][1]
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
        fails += 0 if ok else 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # good binary
        n, bad = d["new_key_profile_phase"](str(ASM), mod, str(tmp))
        check(f"good new_key bad==0 (compared {n})", n > 0 and bad == 0, f"bad={bad}")
        n2, bad2 = d["topology_phase"](str(ASM), mod, str(tmp))
        check(f"good topology bad==0 (compared {n2})", n2 > 0 and bad2 == 0, f"bad={bad2}")

        # control: a wrapper that changes nothing must stay at bad == 0
        ctl = tmp / "ctl.sh"
        write_remap(ctl, "/no/such/needle", "/ref.json", ASM)
        ctl_tmp = tmp / "ctl"
        ctl_tmp.mkdir()
        n, bad = d["new_key_profile_phase"](str(ctl), mod, str(ctl_tmp))
        check(f"control wrapper new_key bad==0 (compared {n})", n > 0 and bad == 0, f"bad={bad}")
        n, bad = d["topology_phase"](str(ctl), mod, str(ctl_tmp))
        check(f"control wrapper topology bad==0 (compared {n})", n > 0 and bad == 0, f"bad={bad}")

        for name in ("w16", "acc8", "one_plane", "c2c_wide"):
            mut = tmp / f"mut-{name}.sh"
            write_remap(mut, f"/test/{name}.json", "/ref.json", ASM)
            phase_tmp = tmp / name
            phase_tmp.mkdir()
            n, bad = d["new_key_profile_phase"](str(mut), mod, str(phase_tmp))
            check(f"mutant {name} new_key bad>=1 (compared {n})", bad >= 1, f"bad={bad}")

        topo_tmp = tmp / "topo"
        topo_tmp.mkdir()
        run_tmp = topo_tmp / "run"
        run_tmp.mkdir()
        good = topo_tmp / "good.json"
        good.write_text('{"chips": 2, "links": []}')
        mut = tmp / "mut-topo.sh"
        write_topo_mutant(mut, good, ASM)
        n, bad = d["topology_phase"](str(mut), mod, str(run_tmp))
        check(f"mutant topology bad>=1 (compared {n})", bad >= 1, f"bad={bad}")

    fails += meta_red_same_mutant()
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
