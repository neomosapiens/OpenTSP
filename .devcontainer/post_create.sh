#!/usr/bin/env bash
# Runs once after the container is created (devcontainer.json postCreateCommand; tools/devcontainer_test.sh runs it too):
# gives the development user the volumes mounted over the repository and the caches, installs the editable
# opentsp-vllm into the vLLM environment of the image, and builds tsp_sim into the target volume when it is absent.
#
# A git worktree's `.git` file names the main repository's git directory by its host path; tools/devcontainer_test.sh
# mounts that directory at the same path, so git works in a worktree too.
set -euo pipefail
for volume in /workspaces/opentsp/{target,opentsp-vllm/native/target} \
    "$HOME"/.cargo/{registry,git} "$HOME"/.cache/{uv,pip,opentsp} "$HOME/.local/opt/opentsp-vllm-venv"; do
  if [ ! -w "$volume" ]; then
    sudo install -d -o "$(id -u)" -g "$(id -g)" "$volume"
  fi
done
cd /workspaces/opentsp
source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
just python-setup
bash tools/ensure_sim.sh target/spec/tsp_sim
cargo build --release --workspace
just vllm-env
just vllm-binding
