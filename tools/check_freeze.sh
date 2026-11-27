#!/usr/bin/env bash
# Check the public specification 0.1 baseline without repository metadata.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$root/tools/check_freeze.py" "$@"
