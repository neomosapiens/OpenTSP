#!/usr/bin/env bash
# Builds the development image of .devcontainer/Dockerfile (docs/installation.md).
#
#   OPENTSP_DEV_PLATFORM  linux/arm64 or linux/amd64 (default: the architecture of the Docker engine)
#   OPENTSP_DEV_IMAGE     the image name (default opentsp-dev); the tag is the architecture, e.g. opentsp-dev:arm64
#
# Extra arguments go to docker buildx build.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=devcontainer_common.sh
source "$here/devcontainer_common.sh"
cd "$root"
echo "devcontainer_build: $OPENTSP_DEV_PLATFORM -> $OPENTSP_DEV_IMAGE_REF"
exec docker buildx build --platform "$OPENTSP_DEV_PLATFORM" \
  -f .devcontainer/Dockerfile \
  -t "$OPENTSP_DEV_IMAGE_REF" --load "$@" .
