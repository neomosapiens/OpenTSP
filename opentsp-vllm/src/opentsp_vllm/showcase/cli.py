"""The TinyStories showcase: `chat` (interactive or one shot), `batch` (several prompts at once), `serve` (`vllm serve` with the plugin).

    python opentsp-vllm/src/opentsp_vllm/showcase/cli.py chat [--prompt TEXT] [--max-tokens N] [--temperature X] [--top-k K] [--seed N] [--verbose]
    python opentsp-vllm/src/opentsp_vllm/showcase/cli.py batch [--prompt TEXT]... [-n N]
    python opentsp-vllm/src/opentsp_vllm/showcase/cli.py serve [--port N]

Run it with the Python of the vLLM environment (`just showcase`, `just showcase-batch`, `just showcase-serve` do); the weights and the programs
come from `just showcase-prepare`. See docs/demos.md#single-chip-tinystories.
"""

import argparse
import contextlib
import importlib
import json
import os
import pathlib
import signal
import socket
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent

from . import backend as B  # noqa: E402
from . import session as S  # noqa: E402

SERVED_NAME = "tinystories"

# the first 36 tokens are shared by all of them: two 16-token KV blocks the prefix cache can reuse
STORY_START = (
    "Once upon a time, there was a little girl named Lily. She lived in a small house near the big green forest, "
    "and every morning she went outside to play. One sunny day, she"
)
BATCH_PROMPTS = [
    STORY_START + " found a shiny red ball.",
    STORY_START + " met a friendly brown dog.",
    STORY_START + " saw a big bird in the tree.",
    STORY_START + " wanted to bake a cake.",
]


def default_root():
    return pathlib.Path(os.environ.get("OPENTSP_SHOWCASE_DIR") or os.path.expanduser("~/.cache/opentsp/showcase"))


def settings_from(a):
    return S.Settings(max_tokens=a.max_tokens, temperature=a.temperature, top_k=a.top_k, seed=a.seed, verbose=a.verbose)


def check_settings(a):
    """The command-line sampling arguments checked by the session's own parsers: ShowcaseError on a bad one."""
    flags = {"max": ("/max", "--max-tokens"), "temp": ("/temp", "--temperature"), "topk": ("/topk", "--top-k")}
    for name, value in (("max", a.max_tokens), ("temp", a.temperature), ("topk", a.top_k)):
        try:
            S._value(name, str(value))
        except S.ShowcaseError as e:
            raise S.ShowcaseError(str(e).replace(*flags[name])) from None


SEED_MAX = 2**63 - 1  # the sampler's generator takes a signed 64-bit seed


def check_options(a):
    """Every numeric option and prompt checked before the model loads: ShowcaseError on a bad one."""
    if a.mode in ("chat", "batch"):
        check_settings(a)
        if a.seed is not None and a.seed > SEED_MAX:
            raise S.ShowcaseError(f"--seed must be at most {SEED_MAX}, not {a.seed}")
        prompts = [a.prompt] if a.mode == "chat" else (a.prompt or [])
        for prompt in prompts:
            if prompt is not None and any(ord(ch) < 32 and ch != "\t" or ord(ch) == 127 for ch in prompt):
                raise S.ShowcaseError(f"--prompt contains a control character: {prompt!r}")
    if a.mode == "batch" and a.n < 0:
        raise S.ShowcaseError(f"-n must be at least 0, not {a.n}")
    if a.mode == "serve" and not 1 <= a.port <= 65535:
        raise S.ShowcaseError(f"--port must be between 1 and 65535, not {a.port}")


@contextlib.contextmanager
def quiet_stderr():
    """File descriptor 2 goes to a scratch file while the libraries load (torch, the OS's duplicate-class notices about ffmpeg, vLLM's start-up
    logging); what they wrote is shown only when the load fails."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile("w+") as log:
        os.dup2(log.fileno(), 2)
        try:
            yield
        except BaseException:
            os.dup2(saved, 2)
            log.seek(0)
            sys.stderr.write(log.read())
            raise
        finally:
            os.dup2(saved, 2)
            os.close(saved)


@contextlib.contextmanager
def own_terminal():
    """Ctrl-C reaches this process alone, not the `just` recipe or the shell script above it (which would report an interrupted run after
    the session ends): this process becomes the terminal's foreground process group while the session runs, and hands it back."""
    fd = sys.stdin.fileno() if sys.stdin and sys.stdin.isatty() else None
    if fd is None or not hasattr(os, "tcsetpgrp"):
        yield
        return
    previous = os.tcgetpgrp(fd)
    ignored = signal.signal(signal.SIGTTOU, signal.SIG_IGN)  # taking over the terminal from the background group would stop us
    try:
        os.setpgid(0, 0)
        os.tcsetpgrp(fd, os.getpgrp())
    except OSError:  # a session leader, or no controlling terminal: keep the group we have
        pass
    try:
        yield
    finally:
        try:
            os.tcsetpgrp(fd, previous)
        except OSError:
            pass
        signal.signal(signal.SIGTTOU, ignored)


def load(root):
    p = B.check_prepared(root)
    print(f"loading TinyStories (calum/tinystories-gpt2-3M, 8 layers) on the ISS from {root} ...", file=sys.stderr, flush=True)
    t = time.perf_counter()
    with quiet_stderr():
        llm = B.build_llm(p["model"], B.opentsp_options(p))
    print(f"loaded in {time.perf_counter() - t:.0f} s: host embeddings ready, KV cache in chip SRAM", file=sys.stderr, flush=True)
    return B.VllmBackend(llm)


def chat(a):
    settings = settings_from(a)
    backend = load(a.root)
    session = S.Session(backend, sys.stdout, sys.stderr, settings, color=sys.stdout.isatty())
    try:
        with own_terminal():
            return repl(session, backend, a)
    finally:
        backend.close()


def repl(session, backend, a):
    if a.prompt is not None:
        session.generate(a.prompt)
        return 0
    try:
        importlib.import_module("readline")  # line editing and history for input
    except ImportError:
        pass
    print(
        f"TinyStories on the ISS: type a prompt, /help for the commands, /quit to leave.\n{session.describe()}; context {backend.max_len} positions",
        flush=True,
    )
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n(Ctrl-C at the prompt: /quit or Ctrl-D leaves)")
            continue
        if not session.handle(line):
            return 0


def batch(a):
    settings = settings_from(a)
    prompts = a.prompt or BATCH_PROMPTS
    if a.n:
        prompts = (prompts * a.n)[: a.n]
    backend = load(a.root)
    try:
        print(f"{len(prompts)} prompts submitted at once; the engine batches them (4 sequences, up to 8 tokens per scheduler step)", flush=True)
        S.run_batch(backend, prompts, settings, sys.stdout, sys.stderr)
    finally:
        backend.close()
    return 0


def serve_command(root, port=8000, host="127.0.0.1", python=None):
    """The `vllm serve` command line of the showcase (the weights and programs of `root`, the plugin switched on by the environment)."""
    p = B.check_prepared(root)
    options = B.opentsp_options(p)
    return [
        python or sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(p["model"]), "--served-model-name", SERVED_NAME,
        "--dtype", "float32", "--max-model-len", "128", "--max-num-seqs", "4", "--max-num-batched-tokens", "8",
        "--enable-chunked-prefill", "--host", host, "--port", str(port), "--additional-config", json.dumps({"opentsp": options}),
    ]  # fmt: skip


def check_port_free(host, port):
    """ShowcaseError when something already listens on `host:port` (the server would otherwise fail much later, in its own process)."""
    with socket.socket() as s:
        if s.connect_ex((host, port)) == 0:
            raise S.ShowcaseError(f"{host}:{port} is already in use; choose another port (PORT=8001 just showcase-serve)")


def serve(a):
    check_port_free(a.host, a.port)
    command = serve_command(a.root, a.port, a.host)
    env = dict(os.environ, OPENTSP_VLLM="1")
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)  # the API server and the engine core are two processes here
    print(f"serving on http://{a.host}:{a.port} as model '{SERVED_NAME}' (about 20 s to start); try:", flush=True)
    print(
        f"  curl -s http://{a.host}:{a.port}/v1/completions -H 'content-type: application/json' "
        f"-d '{{\"model\": \"{SERVED_NAME}\", \"prompt\": \"Once upon a time\", \"max_tokens\": 40, \"temperature\": 0.7, \"seed\": 1}}'",
        flush=True,
    )
    os.execvpe(command[0], command, env)


def seed_arg(text):
    if text.lower() in ("none", "random"):
        return None
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a whole number or none, not {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"a seed is not negative: {text}")
    return value


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(default_root()), help="the prepared showcase directory (default $OPENTSP_SHOWCASE_DIR or ~/.cache/opentsp/showcase)")
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("chat", "batch"):
        c = sub.add_parser(name)
        c.add_argument("--max-tokens", type=int, default=60)
        c.add_argument("--temperature", type=float, default=0.7)
        c.add_argument("--top-k", type=int, default=40)
        c.add_argument("--seed", type=seed_arg, default=1, help="the sampler's seed, or none for a random one")
        c.add_argument("--verbose", action="store_true")
    c = sub.choices["chat"]
    c.add_argument("--prompt", default=None, help="one shot: continue this text and exit (default: an interactive session)")
    c = sub.choices["batch"]
    c.add_argument("--prompt", action="append", help="a prompt (repeatable; default: four story openings with a common start)")
    c.add_argument("-n", type=int, default=0, help="the number of requests (prompts are repeated or cut to it); more than 4 queue")
    c = sub.add_parser("serve")
    c.add_argument("--port", type=int, default=8000)
    c.add_argument("--host", default="127.0.0.1")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        check_options(a)
        return {"chat": chat, "batch": batch, "serve": serve}[a.mode](a)
    except S.ShowcaseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
