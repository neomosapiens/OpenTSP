"""Launch vLLM's OpenAI-compatible GPT-2 small endpoint on four simulated chips.

The launcher verifies the sealed program bundle and its shared tensors, then starts
a loopback service. Ctrl-C forwards shutdown to its vLLM process group.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

from opentsp_vllm import options, ship_plan
from opentsp_vllm.ship_chip import DEFAULT_CHECKPOINT

GRACE = 3.0  # seconds from the first Ctrl-C to the kill of the server's processes
CONTEXT, BLOCKS = ship_plan.kv_limits(2)
POSITIONS = f"the longest context of the pool ({BLOCKS - 1} pages of 16 slots)"
# the vLLM options the pipeline limits: (flag, limit, what the value exceeds)
VLLM_LIMITS = (
    ("--max-num-batched-tokens", options.SHIP_MAX_NUM_BATCHED_TOKENS, f"the {options.SHIP_MAX_NUM_BATCHED_TOKENS} token lanes of one step"),
    ("--max-model-len", CONTEXT, POSITIONS),
    ("--max-num-seqs", options.SHIP_MAX_NUM_SEQS, f"the {options.SHIP_MAX_NUM_SEQS} logit rows of one step"),
)


class LauncherError(ValueError):
    """A bad option: `str(e)` is the message, `e.hint` the second line (or None)."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint


def _number(text):
    """The integer of a vLLM count (`2048`, `2k`, `1M`), None when it is not one (vLLM then reports it)."""
    m = re.fullmatch(r"(-?\d+)([kKmM]?)", text)
    return int(m.group(1)) * {"": 1, "k": 1000, "m": 1000000}[m.group(2).lower()] if m else None


def check_vllm_options(rest):
    """Refuse out-of-range sizes before starting vLLM."""
    for flag, limit, what in VLLM_LIMITS:
        for i, arg in enumerate(rest):
            if arg == flag and i + 1 < len(rest):
                value = rest[i + 1]
            elif arg.startswith(flag + "="):
                value = arg.split("=", 1)[1]
            else:
                continue
            n = _number(value)
            if n is not None and n > limit:
                raise LauncherError(f"{flag} {value} exceeds {what}", f"the largest value is {limit}: give {flag} {limit} or less, or leave it out")


def build(argv):
    """(the `vllm serve` argument list, the environment) of the launcher's command line; ValueError for a bad option value."""
    ap = argparse.ArgumentParser(prog="python -m opentsp_vllm.serve", description=__doc__.split("\n")[0], add_help=True, allow_abbrev=False)
    ap.add_argument("--tsp-model", default="gpt2", help="the model of the pipeline (gpt2)")
    ap.add_argument("--tsp-chips", type=int, default=4, help="the resident four-chip route")
    ap.add_argument("--tsp-check", action="store_true", help="compare every step with the numpy twin bit for bit")
    ap.add_argument("--tsp-shutdown-timeout", type=float, default=GRACE, help="seconds from the first Ctrl-C to the kill of the server's processes (default 3)")
    ap.add_argument("--tsp-kv-chunks", type=int, default=2, help="two 128-slot chunks in the shared pool")
    ap.add_argument("--tsp-programs", default=os.environ.get("OPENTSP_GPT2_4_PROGRAMS", os.path.expanduser("~/.cache/opentsp/gpt2-four-chip")),
                    help="directory of the sealed four-chip program bundle")
    ap.add_argument("--tsp-ras-domain", help="persistent recovery domain for this deployment")
    a, rest = ap.parse_known_args(argv)
    if not math.isfinite(a.tsp_shutdown_timeout) or a.tsp_shutdown_timeout < 0:
        raise LauncherError(f"--tsp-shutdown-timeout {a.tsp_shutdown_timeout:g} must be a number of seconds, zero or more",
                            "give --tsp-shutdown-timeout 3, or leave it out")
    if os.path.exists(a.tsp_programs) and not os.path.isdir(a.tsp_programs):
        raise LauncherError(f"--tsp-programs {a.tsp_programs} is not a directory")
    check_vllm_options(rest)
    if not any(arg.split("=", 1)[0] == "--max-model-len" for arg in rest):
        rest += ["--max-model-len", str(CONTEXT)]
    a.model = os.path.join(a.tsp_programs, "model")
    shared = os.path.join(a.tsp_programs, "weights.safetensors")
    domain = a.tsp_ras_domain or os.environ.get("OPENTSP_RAS_DOMAIN")
    if domain is None:
        domain = f"ship:{os.path.realpath(DEFAULT_CHECKPOINT)}"
    opts = dict(chip="ship", model=a.tsp_model, chips=a.tsp_chips, kv_chunks=a.tsp_kv_chunks,
                programs=a.tsp_programs, weights=shared, ras_domain=domain)
    if a.tsp_check:
        opts["check"] = True
    options.parse({"opentsp": opts})
    if not os.path.isfile(shared) or not os.path.isfile(os.path.join(a.model, "config.json")):
        raise LauncherError(f"{a.tsp_programs}: build the four-chip program bundle with gpt2-programs")
    from opentsp_vllm.models.gpt2.four_chip_packages import manifests

    try:
        manifests(a.tsp_programs)
    except (OSError, ValueError) as error:
        raise LauncherError(f"four-chip program bundle: {error}") from error
    cmd = ["serve", a.model, "--dtype", "float32"]
    if not any(arg.split("=", 1)[0] in ("--host", "--uds") for arg in rest):
        cmd += ["--host", "127.0.0.1"]
    for flag, value in (("--max-num-seqs", "4"), ("--max-num-batched-tokens", "8")):
        if not any(r == flag or r.startswith(flag + "=") for r in rest):
            cmd += [flag, value]
    if not any(r == "--served-model-name" or r.startswith("--served-model-name=") for r in rest):
        cmd += ["--served-model-name", a.tsp_model, a.model]    # a request names "gpt2"; the directory stays a valid name
    cmd += rest + ["--additional-config", json.dumps({"opentsp": opts})]
    build.grace = a.tsp_shutdown_timeout
    return cmd


def _killpg(pgid, sig):
    """Signal a process group; a group that is gone, or whose leader has ended and left only what macOS answers EPERM for, is not an error."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _group_alive(pgid):
    return _killpg(pgid, 0)


def supervise(cmd, env, grace=GRACE, err=None):
    """Run `cmd` (the server) as a child in a process group of its own and return the launcher's exit status.

    The first SIGINT, SIGTERM or SIGHUP (the terminal was closed: the server ends as a plain `vllm serve` does) is passed to the API server (vLLM aborts its requests and stops its engine); if the child has not ended `grace` seconds later, or at a second
    signal, the group is killed and the status is 130. A child that ends by itself gives its own status (a signal death is 128 + the signal). Nothing of the group is left."""
    err = err or sys.stderr
    seen = []
    child = None
    forwarded = False

    def forward_first():
        nonlocal forwarded
        if child is not None and seen and not forwarded:
            # Mark first, so an intervening signal cannot forward it twice.
            forwarded = True
            # vLLM must close its worker before the launcher signals the group.
            try:
                os.kill(child.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def on_signal(signum, frame):
        seen.append(time.monotonic())
        if len(seen) == 1:
            forward_first()

    old = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        # the server's processes end by themselves when this process is killed (SIGKILL cannot be caught): see opentsp_vllm.watchdog
        child = subprocess.Popen(cmd, env=dict(env, OPENTSP_VLLM_PARENT=str(os.getpid())), start_new_session=True)
        pgid = child.pid
        # A signal can arrive inside Popen, before its process handle is returned.
        # Keep its original timestamp/grace and deliver it as soon as the group exists.
        forward_first()
        forced = None
        while child.poll() is None:
            if len(seen) >= 2:
                forced = "second Ctrl-C: the server's processes were killed"
            elif seen and time.monotonic() - seen[0] > grace:
                forced = f"the server did not end within {grace:g} s of Ctrl-C (a client may keep its connection open): its processes were killed"
            if forced:
                _killpg(pgid, signal.SIGKILL)
                child.wait()
                break
            time.sleep(0.05)
        rc = child.returncode
        deadline = time.monotonic() + 2  # the engine core and helpers of a server that ended by itself
        while _group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        _killpg(pgid, signal.SIGKILL)
    except BaseException:
        if child is not None:
            _killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if forced:
        print(f"opentsp_vllm.serve: {forced}", file=err)
        return 130
    return 128 - rc if rc < 0 else rc


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not {"-h", "--help"}.intersection(args) and importlib.util.find_spec("vllm") is None:
        print("opentsp_vllm.serve: error: the serving dependency is not installed; run just vllm-env and use its Python environment", file=sys.stderr)
        return 2
    try:
        cmd = build(args)
    except LauncherError as e:
        print(f"opentsp_vllm.serve: error: {e}", file=sys.stderr)
        if e.hint:
            print(f"opentsp_vllm.serve: {e.hint}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"opentsp_vllm.serve: error: {e}", file=sys.stderr)
        return 2
    env = dict(os.environ, OPENTSP_VLLM="1")
    # Give vLLM time to close the device journal after aborting requests.
    env.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(int(build.grace / 2)))
    return supervise([sys.executable, "-m", "vllm.entrypoints.cli.main", *cmd], env, grace=build.grace)


if __name__ == "__main__":
    sys.exit(main())
