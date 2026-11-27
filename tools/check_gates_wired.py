#!/usr/bin/env python3
"""Fail when a check or recipe is called by no local gate (the pattern of a gate that exists, passes by hand and is run by nobody).

Candidates:
  * every `test_*.py` and `test_*.sh` under root, toolchain and serving test directories;
  * every `*.sh` and `*.py` under `opentsp-arch-spec/test/` (except the Python oracle library `test/oracle/`), whatever its name, and every
    one under `opentsp-arch-spec/tools/` that is a check: a name containing `check`, `test_` or `prove`, or a `--check` mode in its text;
  * every recipe of the `justfile` and of `tools/just/*.just`;
  * every Rust integration test `<crate>/tests/*.rs`, with the cargo features it needs (a `#![cfg(feature = "x")]` line at the top of the
    file, or `required-features` of a `[[test]]` entry in Cargo.toml).

A candidate is wired when it is reachable from one of the root recipes (`test`, `check-docs`, `lint`, `verify-all`, `iss-diff`,
`iss-vectors`, `model-test`, `programs`, `e2e`) or from an area recipe of AGENTS.md section 1 (`asm-diff`, `showcase-reference`, `vllm-unit`,
`vllm-e2e`, `vllm-serve-test`, `model-prove`, `ras-prove`, `ras-qualify`). A CI workflow is not a root: CI does not run
in this repository, so a check that only a workflow runs is reported, and is listed with the reason `ci` until a local gate runs it.

Reachable means run, not mentioned. A recipe is run by a dependency, by a `just <recipe>` call or by a `for ... in` list of a reachable
text. A script is run when a reachable text names it (by its repository path, by its last two path components, or by its file name when no
other file has it) outside a comment, a docstring and a printed message (`echo`, `printf`, `print`, an error text, a `sys.path` entry),
and in a Python text only as a string literal. A directory wires the checks below it only on the command line of a test runner
(`pytest DIR`, `unittest discover -s DIR`). The text of this checker is never an edge. A Rust test is run by a `cargo test` line (also
`nextest run`, `llvm-cov`) that selects its package (`-p`, `--workspace`, the workspace default or `--manifest-path`), does not restrict
the targets to others (`--test`, `--lib`), and builds every feature the test needs (`--features`, `--all-features`).

A candidate that nothing reaches is listed in `tools/gates_allow.txt` (what a listed recipe runs needs no line of its own), one line
`<path or recipe> | <reason> | <note>` with the reason one of manual, long, owner, ci. An entry that is wired already, or names nothing, or
has the reason ci while no workflow runs it, fails too, so the list cannot go stale.

usage: check_gates_wired.py [--root DIR] [--allow FILE]  (exit 0: every gate is wired or listed, 1: a problem, 2: bad usage)"""
import argparse
import re
from source_files import inventory
import sys
from pathlib import Path

# The gates every push runs, then the area recipes of AGENTS.md section 1 (each is the recipe of its area that is run before a push).
ROOTS = ["test", "check-docs", "lint", "verify-all", "iss-diff", "iss-vectors", "model-test", "programs", "e2e"]
AREA_ROOTS = [
    "asm-diff",
    "showcase-reference",
    "vllm-unit",
    "vllm-e2e",
    "vllm-serve-test",
    "model-prove",
    "ras-prove",
    "ras-qualify",
]
REASONS = {"manual", "long", "owner", "ci"}
# `manual` is for a check that needs an outside environment, a quiet host, or about five minutes or more; a local check that is faster belongs in
# a gate. The note states which: it carries one of the words of MANUAL_NEED, or a duration of five minutes or more; a note with neither,
# or with a shorter duration and no such word, fails.
MANUAL_MINUTES = 5
MANUAL_NEED = re.compile(r"(?i)\b(needs?|quiet|network|rewrites|not a check|builds? (?:a|the)|environment)\b")
DURATION = re.compile(
    r"(?i)\babout (?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty)(?: cpu)? (?P<u>seconds?|s|minutes?|min)\b"
)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20}


def manual_problem(note):
    """Why a `manual` note is not allowed (no outside need and no duration of MANUAL_MINUTES or more), or None."""
    if MANUAL_NEED.search(note):
        return None
    durations = []
    for m in DURATION.finditer(note):
        n = m.group("n").lower()
        n = int(n) if n.isdigit() else NUMBER_WORDS[n]
        minutes = n / 60 if m.group("u").lower().startswith("s") else n
        if minutes >= MANUAL_MINUTES:
            return None
        durations.append(m.group(0)[6:])
    if durations:
        return (
            f"takes about {durations[0]} and needs no outside environment or quiet host: a manual entry is for a check that does, "
            f"or that takes five minutes or more; wire it into a gate"
        )
    return (
        "states no duration of five minutes or more (`about N minutes`) and names no outside need (needs, quiet, network, rewrites, "
        "not a check, builds a/the, environment): say which, or wire it into a gate"
    )
JUSTFILES = ["justfile"]
JUST_GLOB = "tools/just/*.just"
CHECK_DIRS = ("opentsp-arch-spec/tools/", "opentsp-arch-spec/test/")
# Every script under these is a candidate whatever its name (`ilog2_equiv.sh` has no `check`, `test_` or `prove` in it).
ALL_SCRIPT_DIRS = ("opentsp-arch-spec/test/",)
TEST_DIRS = ("tests/", "opentsp-sw-toolchain/tests/", "opentsp-vllm/tests/")
# The Python oracle is a library the checks import; its modules are run by their importers, not by a gate.
LIBRARY_DIRS = ("opentsp-arch-spec/test/oracle/",)
SCRIPT_SUFFIXES = (".sh", ".py")
SELF = "tools/check_gates_wired.py"
HEADER = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)([^:=\n]*?):(?!=)(.*)$")
# A printed message: `echo` or `printf` with its arguments (quoted strings may span lines) and redirections.
PRINTED = re.compile(
    r"""(?<![\w./$-])(?:echo|printf)\b(?:[ \t]+(?:"(?:[^"\\]|\\.)*"|'[^']*'|\\\n|>&\d|\d?>>?[ \t]*[\w./&-]+|[^\s;&|"'()<>]+))*"""
)
STRING = r"""[rbfRBF]{0,2}(?:"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')"""
TRIPLE = re.compile(r"""[rbfRBF]{0,2}(\"\"\"|''')(?:.|\n)*?\1""")
PY_QUIET_CALL = re.compile(r"\b(?:print|sys\.exit|[\w.]*\.(?:error|write|fail|exit))\s*\(|\braise\s+[\w.]+\s*\(")
PY_QUIET_KWARG = re.compile(r"\b(?:help|description|epilog|usage)\s*=\s*(?:\(\s*)?" + STRING + r"(?:\s*" + STRING + r")*")
# A Python text runs something only inside a call of the subprocess family (or an exec or spawn); a string that sits in a table, a message
# or a fixture runs nothing.
PY_SPAWN = re.compile(
    r"\b(?:subprocess\.\w+|os\.(?:system|popen|exec\w*|spawn\w*)|runpy\.run_\w+|Popen|check_call|check_output)\s*\("
)
PY_RUNNER = re.compile(r"^\s*(?:bash|sh|python3?|\$PYTHON|\$py|\$\{PYTHON[^}]*\}|just|cargo|pytest|exec)\b")
TEST_RUNNER_LINE = re.compile(r"\b(?:pytest|unittest|py\.test)\b")


def tracked(root):
    """Repository files as git lists them (a scratch tree without git: every file under the root)."""
    return inventory(root)



def live(text, path):
    """The lines of `text` that are not whole-line comments."""
    keep = []
    for line in text.splitlines():
        s = line.lstrip()
        if s.startswith("#") and not s.startswith("#!{{"):
            continue
        if path.endswith(".md") or not s:
            continue
        keep.append(line)
    return "\n".join(keep)


def drop_calls(text, opener):
    """`text` without the calls that `opener` (a regex ending at the opening parenthesis) starts, up to the matching parenthesis."""
    out, pos = [], 0
    while True:
        m = opener.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos : m.start()])
        depth, i, quote = 1, m.end(), None
        while i < len(text) and depth:
            c = text[i]
            if quote:
                if c == "\\":
                    i += 1
                elif c == quote:
                    quote = None
            elif c in "\"'":
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        pos = i


def quiet_python(text):
    """Python text without what only says something: docstrings, comments, printed or raised messages, help texts, `sys.path` lines."""
    text = TRIPLE.sub(" ", text)
    text = "\n".join(line for line in text.splitlines() if "sys.path" not in line)
    text = re.sub(r"""(?m)[ \t]#[^\n"']*$""", "", text)
    text = drop_calls(text, PY_QUIET_CALL)
    return PY_QUIET_KWARG.sub(" ", text)


def call_arguments(text, opener):
    """The argument texts (one line each) of the calls that `opener` (a regex ending at the opening parenthesis) starts."""
    found, pos = [], 0
    while True:
        m = opener.search(text, pos)
        if not m:
            return found
        depth, i, quote = 1, m.end(), None
        while i < len(text) and depth:
            c = text[i]
            if quote:
                if c == "\\":
                    i += 1
                elif c == quote:
                    quote = None
            elif c in "\"'":
                quote = c
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        found.append(" ".join(text[m.end() : i - 1].split()))
        pos = i


def spawned(text):
    """What the subprocess-family calls of a cleaned Python text receive: their arguments, and the right-hand sides of the one-line
    assignments (two levels) of the names that the arguments use."""
    args = call_arguments(text, PY_SPAWN)
    assigned = {}
    for m in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*(?::[^=\n]+)?=\s*(.+)$", text):
        assigned.setdefault(m.group(1), m.group(2))
    out = list(args)
    for _ in range(2):
        for a in list(out):
            for name in set(re.findall(r"[A-Za-z_]\w*", re.sub(STRING, " ", a))):
                if name in assigned and assigned[name] not in out:
                    out.append(assigned[name])
    return "\n".join(out)


def python_literals(text):
    """The string literals of what a Python text spawns that can name a script: without whitespace, or a command line of a known runner."""
    text = spawned(text)
    lits = []
    for m in re.finditer(STRING, text):
        s = m.group(0).lstrip("rbfRBF")[1:-1]
        if not s.strip():
            continue
        if not re.search(r"\s", s) or PY_RUNNER.match(s):
            lits.append(s)
    return "\n".join(lits)


def shell_text(text, path):
    """A shell, recipe or workflow text without printed messages (and, in a workflow, without step names)."""
    if path.endswith((".yml", ".yaml")):
        text = re.sub(r"(?m)^\s*-?\s*name:.*$", "", text)
    return PRINTED.sub(" ", text)


class Recipe:
    def __init__(self, name, deps, file):
        self.name, self.deps, self.file, self.body = name, deps, file, []


def parse_just(root, files):
    recipes = {}
    for f in files:
        cur = None
        for line in (root / f).read_text().splitlines():
            if line.startswith((" ", "\t")):
                if cur:
                    cur.body.append(line)
                continue
            if line.startswith("#") or not line.strip():
                if not line.strip() and cur is None:
                    continue
                continue
            m = HEADER.match(line)
            if not m or line.startswith(("import", "set ", "alias ", "export ")):
                cur = None
                continue
            name, rest = m.group(1), m.group(3)
            deps = [d for d in re.findall(r"[A-Za-z_][A-Za-z0-9_-]*", re.sub(r"\([^)]*\)", " ", rest))]
            cur = recipes[name] = Recipe(name, deps, f)
    return recipes


def names_in(text, names):
    """Recipe names that `text` calls: after `just`, after `just_executable()`, or in a `for ... in` list."""
    found = set()
    for line in text.splitlines():
        words = re.findall(r"[A-Za-z0-9_.-]+", line)
        calls = re.search(r"\bjust\b|just_executable\(\)", line)
        looped = re.match(r"\s*for\s+\w+\s+in\s", line)
        if calls or looped:
            found.update(w for w in words if w in names)
    return found


def script_refs(text, scripts, by_name):
    """Scripts that the command lines of `text` run: by path, by their last two path components or by a unique file name; a directory
    counts only on the command line of a test runner (`pytest tests/showcase` runs every test below it)."""
    found = set()
    words = set(re.findall(r"[\w.-]+", text))
    runner_dirs = set()
    for line in text.splitlines():
        if TEST_RUNNER_LINE.search(line):
            runner_dirs.update(re.findall(r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+)/?(?![\w/.-])", line))
    for name in words & by_name.keys():
        if len(by_name[name]) == 1:
            found.add(by_name[name][0])
    for s in scripts:
        p = Path(s)
        if s in text or (len(p.parts) > 1 and "/".join(p.parts[-2:]) in text):
            found.add(s)
        elif any("/".join(p.parts[:k]) in runner_dirs for k in range(2, len(p.parts))):
            found.add(s)
    return found


def is_check(path, text):
    name = Path(path).name
    return any(k in name for k in ("check", "test_", "prove")) or "--check" in text


# ---- Rust integration tests and cargo features -------------------------------------------------------------------------------------


def parse_crates(root, files):
    """({crate directory: (package name, {test name: [required features]})}, the member directories of the root workspace)."""
    crates, members = {}, set()
    for f in files:
        if Path(f).name != "Cargo.toml":
            continue
        text = (root / f).read_text(errors="replace")
        d = str(Path(f).parent)
        d = "" if d == "." else d
        w = re.search(r"(?ms)^\[workspace\].*?^members\s*=\s*\[(.*?)\]", text)
        if w and d == "":
            members = {m.strip(' "\n\t') for m in w.group(1).replace("\n", " ").split(",") if m.strip(' "\n\t')}
        name = re.search(r'(?ms)^\[package\].*?^name\s*=\s*"([^"]+)"', text)
        if not name:
            continue
        req = {}
        for block in re.split(r"(?m)^\[\[test\]\]", text)[1:]:
            block = re.split(r"(?m)^\[", block)[0]
            n = re.search(r'name\s*=\s*"([^"]+)"', block)
            r = re.search(r"required-features\s*=\s*\[(.*?)\]", block)
            if n and r:
                req[n.group(1)] = re.findall(r'"([^"]+)"', r.group(1))
        crates[d] = (name.group(1), req)
    return crates, members


def rust_tests(root, files, crates):
    """{path of a Rust integration test: (crate directory, test name, features it needs)}."""
    tests = {}
    for f in files:
        m = re.fullmatch(r"(.*)/tests/([^/]+)\.rs", f) or re.fullmatch(r"(.*)/tests/([^/]+)/main\.rs", f)
        if not m or m.group(1) not in crates:
            continue
        crate, name = m.group(1), m.group(2)
        head = (root / f).read_text(errors="replace")
        need = list(crates[crate][1].get(name, []))
        for attr in re.findall(r"(?m)^#!\[cfg\((.*)\)\]\s*$", head):
            if "not(" not in attr and "any(" not in attr:
                need += re.findall(r'feature\s*=\s*"([^"]+)"', attr)
        tests[f] = (crate, name, sorted(set(need)))
    return tests


def cargo_runs(text):
    """The `cargo test` lines of `text` as dicts of what they select."""
    runs = []
    for m in re.finditer(r"\bcargo\b(?:[ \t]+\+\S+)?[ \t]+(?:test|nextest[ \t]+run|llvm-cov)\b([^;&|\n]*)", text.replace("\\\n", " ")):
        args = m.group(1).split()
        if "--" in args:
            args = args[: args.index("--")]
        run = {"pkgs": set(), "workspace": False, "tests": set(), "features": set(), "all_features": False, "manifest": None,
               "exclude": set(), "no_integration": False, "all_tests": False}
        i = 0
        while i < len(args):
            a = args[i]
            val = args[i + 1] if i + 1 < len(args) else ""
            key, _, inline = a.partition("=")
            if a in ("-p", "--package"):
                run["pkgs"].add(val)
                i += 1
            elif key == "--package" and inline:
                run["pkgs"].add(inline)
            elif a in ("--workspace", "--all"):
                run["workspace"] = True
            elif a == "--test":
                run["tests"].add(val)
                i += 1
            elif key == "--test" and inline:
                run["tests"].add(inline)
            elif a in ("--features", "-F"):
                run["features"].update(re.split(r"[,\s]+", val))
                i += 1
            elif key in ("--features", "-F") and inline:
                run["features"].update(re.split(r"[,\s]+", inline))
            elif a == "--all-features":
                run["all_features"] = True
            elif a == "--manifest-path":
                run["manifest"] = str(Path(val).parent)
                i += 1
            elif key == "--manifest-path" and inline:
                run["manifest"] = str(Path(inline).parent)
            elif a == "--exclude":
                run["exclude"].add(val)
                i += 1
            elif a in ("--lib", "--bins", "--doc", "--examples") or a == "--bin":
                run["no_integration"] = True
                if a == "--bin":
                    i += 1
            elif a == "--tests":
                run["all_tests"] = True
            i += 1
        runs.append(run)
    return runs


def covered(run, crate, name, crates, members, need):
    """Whether the cargo run `run` builds and runs the test `name` of the crate directory `crate`."""
    pkg = crates[crate][0]
    if run["manifest"] is not None:
        selected = run["manifest"] in (crate, "." if crate == "" else crate)
    elif run["pkgs"]:
        selected = pkg in run["pkgs"]
    else:
        selected = crate in members and pkg not in run["exclude"]
    if run["workspace"] and run["manifest"] is None:
        selected = selected or (crate in members and pkg not in run["exclude"])
    if not selected:
        return False
    if run["tests"]:
        if name not in run["tests"]:
            return False
    elif run["no_integration"] and not run["all_tests"]:
        return False
    given = set()
    for f in run["features"]:
        head, _, tail = f.rpartition("/")
        if not head or head == pkg:
            given.add(tail)
    return run["all_features"] or all(f in given for f in need)


def analyse(root):
    """(candidates, a function giving what is reachable from some starting nodes, the local starting nodes, the workflow nodes, all
    nodes, the features each Rust test needs)."""
    files = tracked(root)
    just_files = [f for f in files if f in JUSTFILES or re.fullmatch(r"tools/just/[^/]+\.just", f)]
    recipes = parse_just(root, just_files)
    scripts = [f for f in files if f.endswith(SCRIPT_SUFFIXES)]
    workflows = [f for f in files if f.startswith(".github/workflows/") and f.endswith((".yml", ".yaml"))]
    raw = {f: live((root / f).read_text(errors="replace"), f) for f in scripts + workflows}
    texts = {}
    for f, t in raw.items():
        if f == SELF:
            texts[f] = ""
        elif f.endswith(".py"):
            texts[f] = quiet_python(t)
        else:
            texts[f] = shell_text(t, f)
    bodies = {n: shell_text(live("\n".join(r.body), "x"), "x") for n, r in recipes.items()}
    names = set(recipes)
    by_name = {}
    for f in scripts:
        by_name.setdefault(Path(f).name, []).append(f)
    crates, members = parse_crates(root, files)
    rtests = rust_tests(root, files, crates)
    cands = {}
    for f in scripts:
        if f.startswith(TEST_DIRS) and re.fullmatch(r"test_[^/]*\.(py|sh)", Path(f).name):
            cands[f] = f
        elif (f.startswith(ALL_SCRIPT_DIRS) and not f.startswith(LIBRARY_DIRS)) or (f.startswith(CHECK_DIRS) and is_check(f, raw[f])):
            cands[f] = f
    for n in recipes:
        cands[n] = n
    for f in rtests:
        cands[f] = f
    edges = {}

    def cargo_refs(text):
        runs = cargo_runs(text)
        return {f for f, (crate, name, need) in rtests.items() if any(covered(r, crate, name, crates, members, need) for r in runs)}

    def out(n):
        if n not in edges:
            if n in rtests:
                edges[n] = set()
            elif n in recipes:
                edges[n] = (
                    set(recipes[n].deps) & names
                    | names_in(bodies[n], names)
                    | script_refs(bodies[n], scripts, by_name)
                    | cargo_refs(bodies[n])
                )
            else:
                refs = script_refs(python_literals(texts[n]) if n.endswith(".py") else texts[n], scripts, by_name)
                run_text = spawned(texts[n]) if n.endswith(".py") else texts[n]
                edges[n] = (names_in(run_text, names) | refs | cargo_refs(run_text)) - {n}
        return edges[n]

    def reach(start):
        seen, todo = set(), list(start)
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo.extend(out(n) - seen)
        return seen

    starts = [r for r in ROOTS + AREA_ROOTS if r in recipes]
    need = {f: v[2] for f, v in rtests.items()}
    return cands, reach, starts, workflows, set(recipes) | set(texts) | set(rtests), need


def read_allow(path):
    allow, problems = {}, []
    if not path.exists():
        return allow, [f"{path}: the allow list does not exist"]
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or not parts[0] or parts[1] not in REASONS or not parts[2]:
            problems.append(f"{path}:{i}: need `<path or recipe> | <reason: {'/'.join(sorted(REASONS))}> | <note>`")
            continue
        if parts[1] == "manual" and manual_problem(parts[2]):
            problems.append(f"{path}:{i}: {parts[0]} {manual_problem(parts[2])}")
        allow[parts[0]] = (parts[1], parts[2], i)
    return allow, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root_default = Path(__file__).resolve().parents[1]
    ap.add_argument("--root", type=Path, default=root_default)
    ap.add_argument("--allow", type=Path, default=None)
    a = ap.parse_args()
    root = a.root.resolve()
    if not root.is_dir():
        print(f"check_gates_wired: {root} is not a directory", file=sys.stderr)
        return 2
    allow_path = a.allow or root / "tools/gates_allow.txt"
    cands, reach, starts, workflows, nodes, need = analyse(root)
    allow, problems = read_allow(allow_path)
    wired = reach(starts)
    in_ci = reach(starts + workflows)
    kept = reach(starts + [k for k in allow if k in nodes])
    for c in sorted(cands):
        if c in kept:
            continue
        if c in in_ci:
            problems.append(
                f"{c}: only a CI workflow runs it, and CI does not run in this repository; wire it into a local gate or list it in "
                "tools/gates_allow.txt with the reason ci"
            )
        elif c in need and need[c]:
            problems.append(
                f"{c}: no gate builds it with the feature `{'`, `'.join(need[c])}` it needs; run it with `--features` in a recipe of the gates "
                "or list it in tools/gates_allow.txt with a reason"
            )
        else:
            problems.append(f"{c}: no gate calls it; wire it into a recipe of the gates or list it in tools/gates_allow.txt with a reason")
    for k, (reason, _, line) in sorted(allow.items(), key=lambda kv: kv[1][2]):
        if k not in cands:
            problems.append(f"{allow_path.name}:{line}: {k} is not a check or recipe of this tree: remove the entry")
        elif k in wired:
            problems.append(f"{allow_path.name}:{line}: {k} is wired now: remove the entry")
        elif reason == "ci" and k not in in_ci:
            problems.append(f"{allow_path.name}:{line}: {k} has the reason ci but no CI workflow runs it: use another reason or remove the entry")
    for p in problems:
        print("check_gates_wired:", p)
    n_wired = sum(1 for c in cands if c in wired)
    n_listed = sum(1 for c in cands if c not in wired and c in kept)
    print(f"check_gates_wired: {len(cands)} gates: {n_wired} wired, {n_listed} listed or run by a listed gate, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
