#!/usr/bin/env bash
# Install the exact solver required by the proof gate; do not use Sail's bundled older solver.
set -euo pipefail
dest="${1:?toolchain directory}"
case "$(uname -m)" in
  x86_64)
    url=https://files.pythonhosted.org/packages/21/c9/bb51a96af0091324c81b803f16c49f719f9f6ea0b0bb52200f5c97ec4892/z3_solver-4.15.4.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    digest=7e103a6f203f505b8b8b8e5c931cc407c95b61556512d4921c1ddc0b3f41b08e ;;
  aarch64)
    url=https://files.pythonhosted.org/packages/bf/2e/0b49f7e4e53817cfb09a0f6585012b782dfe0b666e8abefcb4fac0570606/z3_solver-4.15.4.0-py3-none-manylinux_2_34_aarch64.whl
    digest=62c7e9cbdd711932301f29919ad9158de9b2f58b4d281dd259bbcd0a2f408ba1 ;;
  *) echo "No pinned Linux Z3 binary for $(uname -m)" >&2; exit 1 ;;
esac
archive="$(mktemp)"
trap 'rm -f "$archive"' EXIT
if [ -n "${OPENTSP_Z3_WHEEL:-}" ]; then
  cp "$OPENTSP_Z3_WHEEL" "$archive"
else
  curl -fsSL --connect-timeout 10 --max-time 180 --retry 1 --retry-max-time 360 "$url" -o "$archive"
fi
echo "$digest  $archive" | sha256sum --check --status
python3 - "$archive" "$dest/sail/bin/z3" <<'PY'
import pathlib
import sys
import zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    names = [n for n in archive.namelist() if n.endswith(".data/data/bin/z3")]
    if len(names) != 1:
        sys.exit("Pinned Z3 wheel has no unique solver executable")
    target = pathlib.Path(sys.argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".new")
    temporary.write_bytes(archive.read(names[0]))
    temporary.chmod(0o755)
    temporary.replace(target)
PY
test "$("$dest/sail/bin/z3" --version)" = "Z3 version 4.15.4 - 64 bit"
