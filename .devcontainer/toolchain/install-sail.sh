#!/usr/bin/env bash
set -euo pipefail
dest="${1:?toolchain directory}"
base="${2:-https://github.com/rems-project/sail/releases/download/0.20.3-binary}"
arch="$(uname -m)"
case "$arch" in
    x86_64) digest=8d7cd75624ece407d7001ad32905a13ae111d5cc1a055404f21510e113207e3f ;;
    aarch64) digest=bdb1d68ca6b3a9eb74c409e42e54f71e77418debe3376dabdaabf3b6cd2bcf2d ;;
    *) echo "No pinned Linux Sail binary for $arch" >&2; exit 1 ;;
esac
archive="$(mktemp)"
trap 'rm -f "$archive"' EXIT
curl -fsSL --retry 3 "$base/sail-Linux-$arch.tar.gz" -o "$archive"
echo "$digest  $archive" | sha256sum --check --status
mkdir -p "$dest"
tar xzf "$archive" -C "$dest"
"$dest/sail/bin/sail" --version
bash "$(dirname "$0")/install-z3.sh" "$dest"
