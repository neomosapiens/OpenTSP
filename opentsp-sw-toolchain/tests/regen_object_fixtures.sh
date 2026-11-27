#!/usr/bin/env bash
# Regenerates the stored objects and manifests of crates/ots-asm/tests/data with the reference assembler
# tools/tsp_as.py. They embed the digest of encoding.json and of the profile, so every specification change
# that touches either needs this run (the stale files fail tests/object.rs and tests/manifest.rs with
# OTS-E-ASM-003). The programs, the weight packages and the listings are inputs and stay as they are.
# parcel_word.elf keeps its ascending word order: only its header digests are taken from the new object.
set -euo pipefail
cd "$(dirname "$0")/../.."
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
D=opentsp-sw-toolchain/crates/ots-asm/tests/data
A=opentsp-arch-spec/tools/tsp_as.py
T="$(mktemp -d)"
trap 'rm -f "$T"/*.elf; rmdir "$T"' EXIT

for n in example example_mini bind bind_mini parcel host hazard; do
  python3 "$A" asm "$D/$n.ots" -o "$T/$n.elf"
done
for n in example example_mini bind bind_mini; do cp "$T/$n.elf" "$D/$n.elf"; done
cp "$T/parcel.elf" "$D/parcel_src.elf"
cp "$T/host.elf" "$D/host.elf"
cp "$T/hazard.elf" "$D/hazard.elf"
python3 - "$D" "$T" <<'EOF'
import sys
d, t = sys.argv[1:3]
new = open(t + "/parcel.elf", "rb").read()
old = bytearray(open(d + "/parcel_word.elf", "rb").read())
assert old[:7] == new[:7] == b"\x7fELF\x01\x01\x01"
assert old[64:72] == new[64:72] == b"OpenTSP\0"
old[76:156] = new[76:156]
open(d + "/parcel_word.elf", "wb").write(old)
EOF

python3 "$A" mpack --profile ref -o "$D/ref1.otsm" "$D/example.elf"
python3 "$A" mpack --profile ref -o "$D/ref2.otsm" "$D/example.elf" "$D/bind.elf"
python3 "$A" mpack --profile mini -o "$D/mini3.otsm" "$D/example_mini.elf" "$D/bind_mini.elf" "$D/example_mini.elf"
objs=()
for _ in 1 2 3 4 5 6 7 8; do objs+=("$D/example_mini.elf" "$D/bind_mini.elf"); done
python3 "$A" mpack --profile mini -o "$D/mini16.otsm" "${objs[@]}"

# The fixtures of the `tsp-iss verify` tests (opentsp-iss/crates/tsp-iss/tests/data/verify): every .ots is the source of the .elf next to it,
# the manifests pack the objects named here, chip 0 first. They embed the same digests, so they need this run too.
V=opentsp-iss/crates/tsp-iss/tests/data/verify
for src in "$V"/*.ots; do python3 "$A" asm "$src" -o "${src%.ots}.elf"; done
python3 "$A" mpack --profile ref -o "$V/dr.otsm" "$V/dr.elf" "$V/e.elf"
python3 "$A" mpack --profile ref -o "$V/lat.otsm" "$V/r0_216.elf" "$V/s1.elf"
python3 "$A" mpack --profile ref -o "$V/pair.otsm" "$V/c0.elf" "$V/c1.elf"
python3 "$A" mpack --profile ref -o "$V/pair_bad.otsm" "$V/c0.elf" "$V/c1bad.elf"
python3 "$A" mpack --profile ref -o "$V/tri.otsm" "$V/tri_c0.elf" "$V/tri_c1.elf" "$V/tri_c2.elf"
python3 "$A" mpack --profile ref -o "$V/strad.otsm" "$V/tri_c0.elf" "$V/strad_c1.elf"
python3 "$A" mpack --profile ref -o "$V/strad_haz.otsm" "$V/strad_haz_c0.elf" "$V/strad_haz_c1.elf"
objs=()
for ((i=0; i<265; i++)); do objs+=("$V/e.elf"); done
python3 "$A" mpack --profile ref --topology "$V/topo265.json" -o "$V/m265.otsm" "${objs[@]}"
