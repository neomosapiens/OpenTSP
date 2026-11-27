"""A scripted `Backend` for the session tests: no vLLM, no chip. Each request generates the pieces of `script[rid]` (or `DEFAULT`) one per
step, in the order the requests were submitted; `on_step(n)` runs before step n returns, so a test can see what the session has written
so far or interrupt it."""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SHOWCASE = HERE.parents[1] / "showcase"
if str(SHOWCASE) not in sys.path:
    sys.path.insert(0, str(SHOWCASE))

from opentsp_vllm.showcase.session import Event, StepInfo

DEFAULT = [" there", " was", " a", " cat", "."]


class ScriptedBackend:
    max_len = 128
    bos = 50256

    def __init__(self, script=None, cycles=1000, on_step=None, cached=None, prefill_steps=0):
        self.script = script or {}
        self.cycles = cycles
        self.on_step = on_step
        self.cached = cached or {}
        self.submitted = []  # (rid, token ids, sampling)
        self.aborted = []
        self.live = {}  # rid -> pieces left
        self.steps = 0
        self.prefill_steps = prefill_steps  # the first steps of the run return no token: the prompt is still being prefilled

    def encode(self, text):
        return [ord(c) for c in text] if len(text) < 1000 else list(range(len(text)))  # a token per character

    def submit(self, rid, token_ids, sampling):
        self.submitted.append((rid, list(token_ids), dict(sampling)))
        self.live[rid] = list(self.script.get(rid, DEFAULT))[: sampling["max_tokens"]]

    def busy(self):
        return bool(self.live)

    def abort(self, rid):
        self.aborted.append(rid)
        self.live.pop(rid, None)

    def step(self):
        if self.on_step:
            self.on_step(self.steps)
        self.steps += 1
        if self.steps <= self.prefill_steps:
            return [], StepInfo((8, 1, 64), 8, 0, self.cycles, 0.01)
        events = []
        for rid in list(self.live):
            piece = self.live[rid].pop(0)
            done = not self.live[rid]
            events.append(Event(rid, piece, (len(piece),), done, "length" if done else None, self.cached.get(rid, 0)))
            if done:
                del self.live[rid]
        return events, StepInfo((2, 1, 64), len(events), max(0, 2 - len(events)), self.cycles, 0.01)
