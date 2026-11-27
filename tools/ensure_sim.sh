#!/usr/bin/env bash
# Reuse a simulator only when its source/toolchain receipt and executable hash match.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$root/tools/ensure_sim.py" "$@"
