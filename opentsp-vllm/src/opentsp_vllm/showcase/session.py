"""The logic of the TinyStories showcase session: command parsing, settings, one generation streamed token by token, the batch run.

Nothing here imports vLLM or the ISS: a `Backend` (the vLLM engine of `backend.py`, or a scripted one in the tests) provides
`encode`, `submit`, `step`, `abort`, `busy` and the context limits. The session talks to the user through two text streams:
`out` (the story, the summary) and `err` (notes and errors); a bad input is a message on `err`, never a traceback.
"""

import signal
import threading
import time
from dataclasses import dataclass, field, replace

VOCAB = 50257


class ShowcaseError(ValueError):
    """An input the session refuses: the message is shown to the user as it is."""


@dataclass(frozen=True)
class Settings:
    max_tokens: int = 60
    temperature: float = 0.7
    top_k: int = 40  # 0: no top-k filter
    seed: "int | None" = 1  # None: a random seed for every request
    verbose: bool = False

    def sampling(self):
        """The vLLM sampler's arguments of these settings (temperature 0 is greedy)."""
        return dict(temperature=self.temperature, top_k=self.top_k, seed=self.seed)


@dataclass(frozen=True)
class Command:
    name: str  # canonical: max, temp, topk, seed, verbose, set, start, help, quit
    arg: object = None


ALIASES = {
    "max": "max", "max_tokens": "max", "tokens": "max", "temp": "temp", "temperature": "temp", "topk": "topk", "top_k": "topk",
    "top-k": "topk", "seed": "seed", "verbose": "verbose", "v": "verbose", "set": "set", "settings": "set", "start": "start",
    "help": "help", "h": "help", "?": "help", "quit": "quit", "exit": "quit", "q": "quit",
}  # fmt: skip

HELP = """\
Type a prompt and press Enter: the story continues from it, token by token, on the ISS.
Commands (a line starting with "/"; write "//" to start a prompt with a slash):
  /max N          generate at most N new tokens (1..{room}); the prompt and the new tokens share {positions} positions
  /temp X         sampling temperature, 0 (greedy) to 2
  /topk K         sample among the K most likely tokens; 0 turns the filter off
  /seed N|none    the sampler's seed (none: a new random one for every prompt)
  /verbose [on|off]  show every chip step (bucket, lanes, cycles) instead of streaming the text
  /start          start a story from nothing (an empty prompt: just <|endoftext|>)
  /set            show the settings
  /help           this text
  /quit           leave (also Ctrl-D); Ctrl-C stops a generation in progress"""


def parse_line(line):
    """('empty'), ('prompt', text) or ('command', Command) for a line typed at the prompt; ShowcaseError for a bad command."""
    line = line.rstrip("\r\n")
    if not line.strip():
        return ("empty",)
    if line.startswith("//"):
        return ("prompt", line[1:])
    if not line.startswith("/"):
        return ("prompt", line)
    word, _, rest = line[1:].strip().partition(" ")
    rest = rest.strip()
    name = ALIASES.get(word.lower())
    if name is None:
        raise ShowcaseError(f"unknown command /{word}; /help lists the commands")
    if name in ("set", "start", "help", "quit"):
        if rest:
            raise ShowcaseError(f"/{word} takes no argument")
        return ("command", Command(name))
    if name == "verbose":
        if rest.lower() not in ("", "on", "off"):
            raise ShowcaseError("/verbose takes on or off")
        return ("command", Command(name, None if not rest else rest.lower() == "on"))
    if not rest:
        raise ShowcaseError(f"/{word} needs a value (/help shows the syntax)")
    return ("command", Command(name, _value(name, rest)))


def _value(name, text):
    if name == "seed":
        if text.lower() in ("none", "random"):
            return None
        return _int("/seed", text, 0, 2**63 - 1)
    if name == "max":
        return _int("/max", text, 1, None)
    if name == "topk":
        return _int("/topk", text, 0, VOCAB)
    try:
        value = float(text)
    except ValueError:
        raise ShowcaseError(f"/temp needs a number, not {text!r}") from None
    if not 0.0 <= value <= 2.0:
        raise ShowcaseError(f"/temp must be between 0 and 2, not {text}")
    return value


def _int(label, text, low, high):
    try:
        value = int(text)
    except ValueError:
        raise ShowcaseError(f"{label} needs a whole number, not {text!r}") from None
    if value < low or (high is not None and value > high):
        raise ShowcaseError(f"{label} must be {'at least ' + str(low) if high is None else f'between {low} and {high}'}, not {text}")
    return value


def apply(settings, command):
    """The settings after a `Command` that sets one."""
    field_of = {"max": "max_tokens", "temp": "temperature", "topk": "top_k", "seed": "seed"}
    if command.name == "verbose":
        return replace(settings, verbose=(not settings.verbose) if command.arg is None else command.arg)
    return replace(settings, **{field_of[command.name]: command.arg})


@dataclass
class Event:
    """What one engine step returned for one request: the new text and ids, and, when it finished, why."""

    rid: str
    text: str = ""
    token_ids: tuple = ()
    finished: bool = False
    finish_reason: "str | None" = None
    num_cached_tokens: "int | None" = None


@dataclass
class StepInfo:
    """The chip step behind an engine step (None cycles on a chip that has none)."""

    bucket: tuple
    lanes: int
    padding: int
    cycles: "int | None"
    seconds: float


@dataclass
class Summary:
    rid: str
    prompt_tokens: int
    token_ids: list = field(default_factory=list)
    text: str = ""
    finish_reason: "str | None" = None
    interrupted: bool = False
    num_cached_tokens: int = 0
    cycles: int = 0
    steps: int = 0
    seconds: float = 0.0
    first_token_seconds: "float | None" = None

    @property
    def tokens(self):
        return len(self.token_ids)

    @property
    def tokens_per_second(self):
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


def format_summary(s):
    parts = [f"{s.tokens} tokens"]
    cached = f" ({s.num_cached_tokens} from the prefix cache)" if s.num_cached_tokens else ""
    parts.append(f"prompt {s.prompt_tokens}{cached}")
    parts.append(f"{s.cycles:,} ISS cycles in {s.steps} steps" if s.cycles else f"{s.steps} steps")
    first = f" (first token after {s.first_token_seconds:.1f} s)" if s.first_token_seconds is not None else ""
    parts.append(f"{s.seconds:.1f} s wall{first}")
    parts.append(f"{s.tokens_per_second:.2f} tokens/s")
    parts.append("interrupted" if s.interrupted else f"stopped: {s.finish_reason}")
    return "[" + " | ".join(parts) + "]"


class Session:
    def __init__(self, backend, out, err, settings=None, clock=time.perf_counter, color=False):
        self.backend = backend
        self.out, self.err = out, err
        self.settings = settings or Settings()
        self.clock = clock
        self.color = color
        self.count = 0
        self._interrupt = threading.Event()

    # -- the interactive loop

    def handle(self, line):
        """Act on one typed line; False when the session is over."""
        try:
            kind, *rest = parse_line(line)
            if kind == "empty":
                return True
            if kind == "prompt":
                self.generate(rest[0])
                return True
            return self._command(rest[0])
        except ShowcaseError as e:
            print(f"error: {e}", file=self.err, flush=True)
            return True

    def _command(self, command):
        if command.name == "quit":
            return False
        if command.name == "help":
            print(HELP.format(room=self.backend.max_len - 1, positions=self.backend.max_len), file=self.out)
        elif command.name == "start":
            self.generate("")
        else:
            if command.name != "set":
                self.settings = apply(self.settings, command)
            print(self.describe(), file=self.out)
        return True

    def describe(self):
        s = self.settings
        seed = "random" if s.seed is None else s.seed
        return f"max {s.max_tokens} tokens, temperature {s.temperature:g}, top-k {s.top_k or 'off'}, seed {seed}, verbose {'on' if s.verbose else 'off'}"

    def interrupt(self):
        """Stop the generation in progress after the current step (the Ctrl-C handler; also called by tests)."""
        self._interrupt.set()

    # -- one prompt

    def prepare(self, text):
        """(prompt token ids, max new tokens) of a prompt under the settings; ShowcaseError when it cannot be served."""
        backend, settings = self.backend, self.settings
        ids = list(backend.encode(text)) if text else []
        if not ids:
            ids = [backend.bos]
            print("note: empty prompt, the story starts from <|endoftext|>", file=self.err)
        if len(ids) >= backend.max_len:
            raise ShowcaseError(
                f"the prompt is {len(ids)} tokens, but the chip's context is {backend.max_len} positions (the prompt and the new tokens share them): "
                f"shorten it to at most {backend.max_len - 1} tokens, and to fewer to leave room for the story"
            )
        room = backend.max_len - len(ids)
        max_new = min(settings.max_tokens, room)
        if max_new < settings.max_tokens:
            print(f"note: the prompt is {len(ids)} tokens, leaving room for {room} new ones: generating at most {max_new}", file=self.err)
        return ids, max_new

    def generate(self, text, rid=None):
        """Stream the continuation of `text` to `out` token by token and print the summary line; returns the `Summary`."""
        ids, max_new = self.prepare(text)
        self.count += 1
        rid = rid or f"req-{self.count}"
        settings = self.settings
        self._interrupt.clear()
        summary = Summary(rid, len(ids))
        self.backend.submit(rid, ids, dict(settings.sampling(), max_tokens=max_new))
        if not settings.verbose:
            self._write(text if text else "", dim=True)
        start = self.clock()
        previous = _install_sigint(self.interrupt)
        try:
            self._drive(rid, summary, start)
        finally:
            _restore_sigint(previous)
        summary.seconds = self.clock() - start
        if settings.verbose:
            self._write(summary.text)
        print(file=self.out)
        print(format_summary(summary), file=self.out, flush=True)
        return summary

    def _drive(self, rid, summary, start):
        verbose = self.settings.verbose
        status = _Status(self.out) if self.color and not verbose else None
        processed = 0
        try:
            while self.backend.busy():
                if self._interrupt.is_set():
                    self.backend.abort(rid)
                    summary.interrupted = True
                    return
                events, info = self.backend.step()
                if info is not None:
                    summary.steps += 1
                    summary.cycles += info.cycles or 0
                    processed += info.lanes
                for e in events:
                    if e.rid != rid:
                        continue
                    if e.token_ids and summary.first_token_seconds is None:
                        summary.first_token_seconds = self.clock() - start
                        if status:
                            status.clear()
                    summary.token_ids.extend(e.token_ids)
                    summary.text += e.text
                    if e.num_cached_tokens is not None:
                        summary.num_cached_tokens = e.num_cached_tokens
                    if verbose:
                        print(_step_line(summary.steps, info, e), file=self.out, flush=True)
                    else:
                        self._write(e.text)
                    if e.finished:
                        summary.finish_reason = e.finish_reason
                if status and summary.first_token_seconds is None and info is not None:
                    status.show(f" prefilling the prompt on the chip: {min(processed + summary.num_cached_tokens, summary.prompt_tokens)}/{summary.prompt_tokens} tokens")
        finally:
            if status:
                status.clear()

    def _write(self, text, dim=False):
        if not text:
            return
        if dim and self.color:
            text = f"\033[2m{text}\033[0m"
        self.out.write(text)
        self.out.flush()


class _Status:
    """A dim progress note after the echoed prompt, rewritten in place (save and restore cursor) and erased before the first token."""

    def __init__(self, out):
        self.out = out
        self.shown = False

    def show(self, text):
        self.out.write(("\0338\033[K" if self.shown else "") + f"\0337\033[2m{text}\033[0m")
        self.out.flush()
        self.shown = True

    def clear(self):
        if self.shown:
            self.out.write("\0338\033[K")
            self.out.flush()
            self.shown = False


def _step_line(n, info, event):
    where = f"scheduled tokens {info.lanes}  logit capacity {info.bucket[1]}  context capacity {info.bucket[2]}" if info else "no chip step"
    cycles = f"  {info.cycles:,} cycles  {info.seconds:.2f} s" if info and info.cycles else ""
    return f"  step {n:3d}  {where}{cycles}  -> {event.text!r}" if event.token_ids else f"  step {n:3d}  {where}{cycles}"


def _install_sigint(handler):
    if threading.current_thread() is not threading.main_thread():
        return None
    return signal.signal(signal.SIGINT, lambda *_: handler())


def _restore_sigint(previous):
    if previous is not None:
        signal.signal(signal.SIGINT, previous)


# -- several requests at once


def run_batch(backend, prompts, settings, out, err):
    """Submit every prompt at once, let the engine batch them, print each stream's text when it completes. Returns the summaries in
    prompt order. A prompt that cannot be served is reported and skipped."""
    session = Session(backend, out, err, settings)
    submitted = {}
    start = time.perf_counter()
    for i, text in enumerate(prompts):
        try:
            ids, max_new = session.prepare(text)
        except ShowcaseError as e:
            print(f"error: prompt {i + 1}: {e}", file=err, flush=True)
            continue
        rid = f"req-{i + 1}"
        backend.submit(rid, ids, dict(settings.sampling(), max_tokens=max_new))
        submitted[rid] = (text, Summary(rid, len(ids)))
    steps = cycles = 0
    while backend.busy():
        events, info = backend.step()
        if info is not None:
            steps += 1
            cycles += info.cycles or 0
        for e in events:
            text, summary = submitted[e.rid]
            summary.token_ids.extend(e.token_ids)
            summary.text += e.text
            if e.num_cached_tokens is not None:
                summary.num_cached_tokens = e.num_cached_tokens
            if e.finished:
                summary.finish_reason = e.finish_reason
                summary.seconds = time.perf_counter() - start
                summary.steps = steps
                print(f"{e.rid} finished at step {steps}: {text}{summary.text}", file=out, flush=True)
                print(f"        {format_batch_line(summary)}", file=out, flush=True)
    wall = time.perf_counter() - start
    done = [s for _, s in submitted.values()]
    total = sum(s.tokens for s in done)
    print(
        f"{len(done)} requests, {total} tokens in {steps} scheduler steps"
        + (f", {cycles:,} ISS cycles" if cycles else "")
        + f", {wall:.1f} s wall, {total / wall if wall > 0 else 0:.2f} tokens/s",
        file=out,
        flush=True,
    )
    return done


def format_batch_line(s):
    cached = f"{s.num_cached_tokens} of {s.prompt_tokens} prompt tokens from the prefix cache" if s.prompt_tokens else ""
    return f"[{s.tokens} tokens, {cached or 'prompt ' + str(s.prompt_tokens)}, stopped: {s.finish_reason}]"
