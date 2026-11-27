#!/usr/bin/env bash
# Documentation and source naming checks for an unpacked source directory.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$root/tools/check_docs_location.py" "$@"
