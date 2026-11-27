#!/usr/bin/env bash
# Install the pinned CPU vLLM and the editable plugin. See docs/installation.md.
# OPENTSP_VLLM_VENV selects the environment, OPENTSP_VLLM_PYTHON selects its interpreter.
# OPENTSP_VLLM_WHEEL accepts a local wheel or its URL. OPENTSP_TORCH_LINKS can point at a CPU wheelhouse.
# OPENTSP_VLLM_FROM_SOURCE=1 selects the original checked source build.
set -euo pipefail

# The pin. A re-pin updates these three lines and opentsp-vllm/tests/test_vllm_surface.py.
VLLM_VERSION="0.33.0"
VLLM_SDIST_SHA256="f4c33c2503d52bb411146afa9618a4f48bed68912f617e6803ac8d76e4c74112"
VLLM_SDIST_URL="https://files.pythonhosted.org/packages/f7/25/aa5684e9c59d6fc75f78926b7f15e8d9adc7de2d9e6e7575b80776ad8169/vllm-${VLLM_VERSION}.tar.gz"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${OPENTSP_VLLM_VENV:-$HOME/.local/opt/opentsp-vllm-venv}"
py_req="${OPENTSP_VLLM_PYTHON:-3.12}"
cache="${OPENTSP_VLLM_CACHE:-$HOME/.cache/opentsp/vllm}"
step="start"
# The wheels are large and the link can be slow; uv's default 30 s read timeout is too short for it.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
# the same for the pip fallback (no uv)
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

fail() { trap - EXIT; echo "setup_vllm_env: FAILED at step '$step': $*" >&2; exit 1; }
trap 'rc=$?; [ "$rc" -eq 0 ] || echo "setup_vllm_env: FAILED at step '"'"'$step'"'"' (exit $rc)" >&2' EXIT
say() { echo "== $*"; }

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

pip_install() {
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$venv/bin/python" "$@"
  else "$venv/bin/python" -m pip install "$@"; fi
}

step="venv"
if [ ! -x "$venv/bin/python" ]; then
  if [ -e "$venv" ] && [ -n "$(ls -A "$venv" 2>/dev/null)" ]; then
    fail "$venv exists and has no bin/python (an interrupted setup or a foreign directory); remove it, or point OPENTSP_VLLM_VENV elsewhere, and rerun"
  fi
  say "creating $venv"
  mkdir -p "$(dirname "$venv")"
  if command -v uv >/dev/null 2>&1; then uv venv --python "$py_req" "$venv"
  elif command -v "python$py_req" >/dev/null 2>&1; then "python$py_req" -m venv "$venv"
  else fail "neither uv nor python$py_req is installed"; fi
fi
"$venv/bin/python" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 15), sys.version' ||
  fail "$venv has an unsupported Python (vLLM $VLLM_VERSION needs >= 3.11, < 3.15); remove it and rerun"

installed="$("$venv/bin/python" -c 'import importlib.metadata as m
try: print(m.version("vllm"))
except m.PackageNotFoundError: print("")' 2>/dev/null || true)"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64|Linux/aarch64) platform="manylinux_2_39_$(uname -m)" ;;
  Darwin/arm64) platform=macosx_11_0_arm64 ;;
  *) fail "unsupported CPU wheel platform $(uname -s)/$(uname -m)" ;;
esac
if [[ "$platform" == manylinux* ]]; then
  step="install CPU torch"
  links="${OPENTSP_TORCH_LINKS:-https://download.pytorch.org/whl/cpu}"
  if [ -d "$links" ]; then index=(--no-index --find-links "$links"); else index=(--index-url "$links"); fi
  pip_install --no-deps "${index[@]}" 'torch==2.14.0+cpu' 'torchvision==0.29.0+cpu' 'torchaudio==2.11.0+cpu'
fi
if [ "$installed" != "$VLLM_VERSION+cpu" ] && [ "${OPENTSP_VLLM_FROM_SOURCE:-0}" != 1 ]; then
  step="install CPU wheels"
  base=https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5
  wheel="${OPENTSP_VLLM_WHEEL:-$base/vllm-$VLLM_VERSION%2Bcpu-cp38-abi3-$platform.whl}"
  if [ -n "${OPENTSP_TRITON_WHEEL:-}" ]; then
    # Upstream declares a direct GitHub dependency. Allow a verified offline copy for Python 3.12+ on x86.
    command -v uv >/dev/null || fail "OPENTSP_TRITON_WHEEL requires uv"
    [[ "$platform" == manylinux_2_39_x86_64 ]] || fail "the cached Triton wheel is for Linux x86-64"
    [ "$(sha256 "$OPENTSP_TRITON_WHEEL")" = 693f657501e6d791e889d5fd880d3b84a9bf4d8f3087dbb685534bb8533361fa ] ||
      fail "Triton wheel differs from the pinned upstream CPU release"
    triton_uri="$("$venv/bin/python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve().as_uri())' "$OPENTSP_TRITON_WHEEL")"
    pip_install --overrides <(echo "triton @ $triton_uri") "$wheel"
  else
    pip_install "$wheel"
  fi
elif [ "$installed" != "$VLLM_VERSION+cpu" ]; then
  step="download vllm-$VLLM_VERSION.tar.gz"
  mkdir -p "$cache"
  sdist="$cache/vllm-$VLLM_VERSION.tar.gz"
  if [ ! -f "$sdist" ] || [ "$(sha256 "$sdist")" != "$VLLM_SDIST_SHA256" ]; then
    say "downloading $VLLM_SDIST_URL"
    curl -fL --retry 5 --retry-delay 5 -o "$sdist.part" "$VLLM_SDIST_URL" || fail "download failed"
    mv "$sdist.part" "$sdist"
  fi
  [ "$(sha256 "$sdist")" = "$VLLM_SDIST_SHA256" ] ||
    fail "SHA-256 of $sdist differs from the pin $VLLM_SDIST_SHA256; delete the file and rerun"

  step="unpack"
  src="$cache/vllm-$VLLM_VERSION"
  [ -f "$src/pyproject.toml" ] || { say "unpacking"; tar -xzf "$sdist" -C "$cache"; }

  # The build runs in the venv itself (--no-build-isolation) so torch is downloaded once, not twice.
  step="build requirements"
  say "installing the CPU requirements and the build requirements"
  pip_install -r "$src/requirements/cpu.txt" -r "$src/requirements/build/cpu.txt"

  step="build vllm"
  say "building vllm $VLLM_VERSION from the sdist (CPU backend; tens of minutes)"
  ( cd "$src" && VLLM_TARGET_DEVICE=cpu MAX_JOBS="${MAX_JOBS:-8}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-8}" \
      pip_install --no-build-isolation "$src" ) || fail "the vLLM build failed (see the log above)"
else
  say "vllm $VLLM_VERSION already installed in $venv"
fi

step="verify vllm"
# The source build carries the local version tag "+cpu".
v="$("$venv/bin/python" -c 'import importlib.metadata as m; print(m.version("vllm").split("+")[0])')"
[ "$v" = "$VLLM_VERSION" ] || fail "installed vllm is $v, not $VLLM_VERSION"

"$venv/bin/python" -c 'import torch; assert torch.version.cuda is None, "OpenTSP requires CPU torch in this environment"'
step="install opentsp-vllm"
[ -f "$root/opentsp-vllm/pyproject.toml" ] ||
  fail "$root/opentsp-vllm/pyproject.toml is missing (the package directory is part of the repository)"
say "installing opentsp-vllm (editable; the venv points at this worktree until the next run from another)"
pip_install "setuptools>=77" pytest
# The pinned CPU runtime is installed and checked above. Solving the optional
# extra again would revisit the upstream platform-specific Triton wheel URL.
pip_install -e "$root/opentsp-arch-spec" -e "$root/opentsp-sw-toolchain" -e "$root/opentsp-vllm" ||
  fail "cannot install $root/opentsp-vllm"

step="verify opentsp-vllm"
"$venv/bin/python" -c "from vllm import LLM" || fail "the installed CPU vLLM runtime cannot load"
"$venv/bin/python" "$root/tools/check_python_checkout.py" "$root" ||
  fail "the OpenTSP packages do not load from this checkout"

step="done"
say "vLLM environment ready: $venv (vllm $VLLM_VERSION, opentsp-vllm editable)"
