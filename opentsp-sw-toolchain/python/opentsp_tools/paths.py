"""Validated source-checkout discovery shared by Python tools and model adapters."""
import os
from pathlib import Path

def repository_root():
    """Resolve an explicit OPENTSP_ROOT or this package's checkout, never the current directory."""
    chosen = os.environ.get("OPENTSP_ROOT")
    candidates = [Path(chosen).expanduser()] if chosen else Path(__file__).resolve().parents
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "Cargo.toml").is_file() and (root / "opentsp-arch-spec/config/ref.json").is_file() and (root / "opentsp-arch-spec/isa-db/generated/encoding.json").is_file():
            return str(root)
    if chosen:
        raise RuntimeError(f"OPENTSP_ROOT {chosen!r} is not an OpenTSP source checkout (missing workspace/profile/encoding)")
    raise RuntimeError("OpenTSP source checkout not found; install editable packages from a checkout or set OPENTSP_ROOT")

ROOT = repository_root()
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
