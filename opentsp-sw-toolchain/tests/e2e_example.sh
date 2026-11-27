#!/usr/bin/env bash
# The §3 example of assembly.md, end to end:
#   ots-asm asm -o | ots-asm wpack | ots-asm objinput --weights | tsp_sim
# with x = 3 and y = 4 in every lane, SRAM word 0x9 of slice 44 (0x2c) becomes 7 in every lane.
# The program is read from the document, so the example that is documented is the one that runs. Both
# shipped profiles; `ref` has 320 lanes, `mini` 16.
# usage: e2e_example.sh <ots-asm> <tsp_sim>      (absolute paths)
set -euo pipefail
asm="$1"; sim="$2"
here="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 - "$here/opentsp-arch-spec/docs/assembly.md" "$work/ex.ots" <<'PY'
import re, sys
doc = open(sys.argv[1]).read()
open(sys.argv[2], "w").write([b for b in re.findall(r"```\n(.*?)```", doc, re.S) if b.startswith(".arch")][0])
PY
for p in ref mini; do
  lanes=$([ "$p" = ref ] && echo 320 || echo 16)
  sed "s/^\.profile ref.*/.profile $p/" "$work/ex.ots" > "$work/$p.ots"
  python3 -c "import sys; open(sys.argv[1], 'wb').write(bytes([int(sys.argv[3])]) * int(sys.argv[2]))" "$work/x.bin" "$lanes" 3
  python3 -c "import sys; open(sys.argv[1], 'wb').write(bytes([int(sys.argv[3])]) * int(sys.argv[2]))" "$work/y.bin" "$lanes" 4
  "$asm" asm "$work/$p.ots" -o "$work/$p.elf"
  "$asm" wpack --profile "$p" -o "$work/$p.safetensors" "x=i8:$work/x.bin" "y=i8:$work/y.bin"
  { "$asm" objinput "$work/$p.elf" --weights "$work/$p.safetensors"; echo "D 2c 9"; } > "$work/$p.in"
  "$sim" "$here/opentsp-arch-spec/config/$p.json" < "$work/$p.in" > "$work/$p.out"
  python3 - "$work/$p.out" "$lanes" "$p" <<'PY'
import sys
lines = open(sys.argv[1]).read().split("\n")
lanes, p = int(sys.argv[2]), sys.argv[3]
assert lines[0].startswith("E -"), (p, "the model reported a diagnostic", lines[0])
w = [l for l in lines if l.startswith("W 2c 9 ")]
assert len(w) == 1, (p, lines)
b = w[0].split()[3:]
assert b[:lanes] == ["07"] * lanes, (p, b[:lanes])
print(f"e2e_example [{p}]: SRAM word 0x9 of slice 44 = 7 in all {lanes} lanes")
PY
done
