#!/usr/bin/env bash
# Explicit source-baseline A/B. See --help; no repository history or private baseline pins.
set -euo pipefail
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$root/tools/iss_perf_gate.py" "$@"
