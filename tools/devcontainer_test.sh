#!/usr/bin/env bash
# Runs `just` recipes inside the development image (docs/installation.md), by default `just test`:
#
#   tools/devcontainer_test.sh [recipe ...]        e.g. tools/devcontainer_test.sh test iss-vectors programs
#   tools/devcontainer_test.sh -- recipe arg ...   e.g. tools/devcontainer_test.sh -- vllm-ship-e2e
#
# The container runs detached (`docker run -d`), so a terminal that closes does not kill it; the script follows the log
# and exits with the exit status of the recipes. Run it again with the same arguments to follow a container that is
# still running.
#
#   OPENTSP_DEV_PLATFORM  linux/arm64 or linux/amd64 (default: the architecture of the Docker engine)
#   OPENTSP_DEV_IMAGE     image name (default opentsp-dev; built by tools/devcontainer_build.sh)
#   OPENTSP_DEV_KEEP=1    keep the container after the run (default: removed; the volumes stay)
#   OPENTSP_DEV_TAG       name suffix of the container and its volumes (default: the tree's directory name)
#   JOBS, ...             environment variables listed in OPENTSP_DEV_ENV (default JOBS) are passed to the recipes
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=devcontainer_common.sh
source "$here/devcontainer_common.sh"
image_id="$(docker image inspect --format '{{.Id}}' "$OPENTSP_DEV_IMAGE_REF" 2>/dev/null)" ||
  { echo "devcontainer_test: image $OPENTSP_DEV_IMAGE_REF is missing; run tools/devcontainer_build.sh" >&2; exit 1; }

tag="${OPENTSP_DEV_TAG:-$(basename "$root")}"
tag="$(printf '%s' "$tag" | tr -c 'A-Za-z0-9_.-' '-')"
name="opentsp-dev-$tag-${OPENTSP_DEV_PLATFORM#linux/}"
[ "$#" -gt 0 ] || set -- test
recipes="$(printf '%q ' "$@")"
if [ "$1" = -- ] && [ "$#" -eq 1 ]; then echo "missing recipe after --" >&2; exit 2; fi

if [ -z "$(docker ps -aq --filter "name=^$name\$")" ]; then
  args=(run -d --init --name "$name" --label "opentsp.recipes=$recipes"
    --label "opentsp.source=$root" --label "opentsp.image=$image_id" --platform "$OPENTSP_DEV_PLATFORM" --runtime runc --user dev
    --cap-add PERFMON
    -e CUDA_VISIBLE_DEVICES=-1 -e NVIDIA_VISIBLE_DEVICES=void
    -v "$root:/workspaces/opentsp"
    -v "opentsp-$tag-target:/workspaces/opentsp/target"
    -v "opentsp-$tag-native-target:/workspaces/opentsp/opentsp-vllm/native/target"
    -v "opentsp-$tag-venv:/home/dev/.local/opt/opentsp-vllm-venv"
    -v "opentsp-$tag-cargo-registry:/home/dev/.cargo/registry"
    -v "opentsp-$tag-cargo-git:/home/dev/.cargo/git"
    -v "opentsp-$tag-uv-cache:/home/dev/.cache/uv"
    -v "opentsp-$tag-pip-cache:/home/dev/.cache/pip"
    -v "${OPENTSP_DEV_CACHE:-opentsp-$tag-vllm-cache}:/home/dev/.cache/opentsp")
  # a worktree's .git file names the main repository's git directory by its host path
  if [ -f "$root/.git" ]; then
    gitdir="$(git -C "$root" rev-parse --git-common-dir)"
    gitdir="$(cd "$root" && cd "$gitdir" && pwd -P)"
    args+=(-v "$gitdir:$gitdir")
  fi
  if [ -n "${OPENTSP_DEV_HF_CACHE:-}" ]; then
    args+=(-v "$OPENTSP_DEV_HF_CACHE:/home/dev/.cache/huggingface:ro")
  fi
  for v in ${OPENTSP_DEV_ENV:-JOBS}; do [ -z "${!v:-}" ] || args+=(-e "$v=${!v}"); done
  # the recipes run in one login-less bash; any failing recipe stops the run with its status
  docker "${args[@]}" "$image_id" bash -c '
    set -euo pipefail
    bash .devcontainer/post_create.sh
    source "$OPENTSP_TOOLCHAIN/env.sh"
    if [ "$1" = -- ]; then
      shift
      just "$@"
    else
      for r in "$@"; do just "$r"; done
    fi
    echo "=== devcontainer_test: all recipes passed"' bash "$@" >/dev/null
  echo "devcontainer_test: started $name ($recipes)" >&2
elif [ "$(docker inspect --format '{{index .Config.Labels "opentsp.recipes"}}' "$name")" != "$recipes" ] ||
     [ "$(docker inspect --format '{{index .Config.Labels "opentsp.source"}}' "$name")" != "$root" ] ||
     [ "$(docker inspect --format '{{index .Config.Labels "opentsp.image"}}' "$name")" != "$image_id" ]; then
  echo "devcontainer_test: $name belongs to another source, image or command. Use its original inputs or another OPENTSP_DEV_TAG." >&2
  exit 1
fi

docker logs -f "$name" 2>&1 || true
status="$(docker wait "$name")"
[ "${OPENTSP_DEV_KEEP:-0}" = 1 ] || docker rm "$name" >/dev/null
echo "devcontainer_test: exit status $status" >&2
exit "$status"
