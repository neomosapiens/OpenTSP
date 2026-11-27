import os
from pathlib import Path
from platform import machine, system

# A mounted source tree can serve macOS and Linux at the same time. Load the binding
# from this host's build target before considering extensions in the source package.
__path__.insert(0, str(Path(__file__).resolve().parents[2] / "native" / "target" / f"{system()}-{machine()}"))


def register():
    """The `vllm.platform_plugins` entry point: the platform's qualified name when OPENTSP_VLLM=1, else None.

    Called by vLLM in every process while it picks a platform, before anything of vLLM is configured, so it must not import vLLM.
    """
    if os.environ.get("OPENTSP_VLLM") == "1":
        from . import watchdog      # noqa: PLC0415 (no vLLM import): the server ends when its launcher is gone
        watchdog.start()
        return "opentsp_vllm.platform.TspPlatform"
    return None
