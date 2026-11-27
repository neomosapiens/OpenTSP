#!/usr/bin/env bash
# Measure line and branch coverage for the ISS crates.
set -euo pipefail
exec "${PYTHON:-python3}" "$(dirname "$0")/coverage_rust.py" coverage-iss tsp-numerics tsp-iss
