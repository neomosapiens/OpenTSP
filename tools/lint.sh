#!/usr/bin/env bash
# The code-quality gate, run by `just lint` (part of `just test`). Every check must be clean:
#  1. rustfmt (rustfmt.toml) over the workspace and the binding crate;
#  2. clippy with warnings denied, on every target and feature of the workspace and of the binding crate (opentsp-vllm/native, its own
#     workspace; it needs the interpreter on PATH for PyO3's configuration, not Python headers);
#  3. pyflakes over the tracked Python (unused names, undefined names, placeholder-free f-strings);
#  4. `bash -n` over the tracked shell scripts;
#  5. tests live in the crate's tests/ directory; white-box tests may be linked from their owning module.
# A lint is fixed, not silenced: a lint is allowed only with a targeted #[allow(clippy::...)] and a
# one-line reason where it is wrong for the code. The frozen specification files are not exempt.
# Needs cargo with rustfmt and clippy, and python3 with pyflakes (the specification toolchain's
# environment has it: source ~/.local/opt/opentsp-toolchain/env.sh).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
py="${PYTHON:-python3}"

echo "== rustfmt"
cargo fmt --all --check

echo "== clippy"
cargo clippy --workspace --all-targets --all-features -- -D warnings

echo "== binding crate (opentsp-vllm/native): rustfmt, clippy"
cargo fmt --manifest-path opentsp-vllm/native/Cargo.toml --all --check
# no Python is needed to check the binding crate (PyO3 abi3, extension-module mode)
PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1 cargo clippy --manifest-path opentsp-vllm/native/Cargo.toml --all-targets --all-features -- -D warnings

echo "== pyflakes"
"$py" tools/source_files.py --suffix .py --null | xargs -0 "$py" -m pyflakes

echo "== shell syntax"
"$py" tools/source_files.py --suffix .sh | while IFS= read -r f; do bash -n "$f"; done

echo "== Bash recipes"
just --dump --dump-format json | "$py" -c '
import json, shlex, subprocess, sys

recipes = json.load(sys.stdin)["recipes"]
failed = []
for name, recipe in recipes.items():
    if not recipe["shebang"]:
        continue
    lines = ["".join(part if isinstance(part, str) else "__just_expression__" for part in line) for line in recipe["body"]]
    if not lines or "bash" not in shlex.split(lines[0][2:]):
        continue
    result = subprocess.run(["bash", "-n"], input="\n".join(lines), text=True, capture_output=True)
    if result.returncode:
        failed.append(name)
        print(f"{name}: {result.stderr.strip()}", file=sys.stderr)
sys.exit(bool(failed))
'

echo "== tests outside src"
if "$py" - <<'CHECK_TESTS'
from pathlib import Path
import re
import sys

conditional_test = re.compile(r"^\s*#\[cfg\s*\(\s*test\s*\)\]", re.MULTILINE)
external_module = re.compile(r'\s*#\[path\s*=\s*"([^"]+)"\]\s*mod\s+\w+\s*;')
bad = []
for project in ("opentsp-iss", "opentsp-sw-toolchain", "opentsp-vllm/native"):
    for manifest in Path(project).rglob("Cargo.toml"):
        if "target" in manifest.parts:
            continue
        tests = (manifest.parent / "tests").resolve()
        for path in (manifest.parent / "src").rglob("*.rs"):
            source = path.read_text()
            for declaration in conditional_test.finditer(source):
                module = external_module.match(source, declaration.end())
                if module:
                    target = (path.parent / module.group(1)).resolve()
                    if target.is_file() and target.is_relative_to(tests):
                        continue
                bad.append(f"{path}:{source.count(chr(10), 0, declaration.start()) + 1}")
print("\n".join(bad))
sys.exit(0 if bad else 1)
CHECK_TESTS
then
    echo "lint: test code under src; integration tests belong in tests/" >&2
    exit 1
fi

echo "lint: clean"
