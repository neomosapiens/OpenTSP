#!/usr/bin/env bash
# Measure line and branch coverage for the assembler crates.
set -euo pipefail
exec "${PYTHON:-python3}" "$(dirname "$0")/coverage_rust.py" coverage-asm ots-profile ots-isa ots-format ots-asm
