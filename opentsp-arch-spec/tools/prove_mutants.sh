#!/usr/bin/env bash
# Mutation ratchet for the proof gate: apply one-line mutants (test/formal/mutants.py) to a scratch copy of the
# model, run model_prove.sh on each and compare the outcomes with test/formal/mutants_expected.txt.
#   prove_mutants.sh --expect-current   all mutants; fail on a new survivor or on a survivor that is now killed
#   prove_mutants.sh --only m_k1        one mutant (repeatable);   --jobs N   N mutants at a time
# About 3 min per mutant and profile (measured; see the fix report); the tree itself is never modified.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
exec "${PYTHON:-python3}" "$here/test/formal/mutants.py" "$@"
