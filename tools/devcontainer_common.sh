# Sourced by tools/devcontainer_*.sh: resolves OPENTSP_DEV_PLATFORM and OPENTSP_DEV_IMAGE_REF.
command -v docker >/dev/null 2>&1 || { echo "devcontainer: docker is not installed" >&2; exit 1; }
if [ -z "${OPENTSP_DEV_PLATFORM:-}" ]; then
  case "$(docker version --format '{{.Server.Arch}}')" in
    arm64|aarch64) OPENTSP_DEV_PLATFORM=linux/arm64 ;;
    amd64|x86_64) OPENTSP_DEV_PLATFORM=linux/amd64 ;;
    *) echo "devcontainer: unsupported engine architecture" >&2; exit 1 ;;
  esac
fi
OPENTSP_DEV_IMAGE_REF="${OPENTSP_DEV_IMAGE:-opentsp-dev}:${OPENTSP_DEV_PLATFORM#linux/}"
