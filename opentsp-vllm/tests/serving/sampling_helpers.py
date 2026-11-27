"""Helpers of `test_sampling_and_scenarios.py` : the bit-exact reference reference of a sampling scenario.

The engine (`make_llm`, `Engine`) is 's `conftest.py` and `engine.py`. The reference is the oracle's paged step
(`paged_reference.decode_step_paged`, one lane per call): `teacher_logits` gives the logits a sequence has at every position of a forced
token string, and `generate` generates greedily after a transform of the logits (penalties, bias, bad words). Both are exact for the
chip: bit-exact reference is its bit-exact twin.
"""

import numpy as np

import reference_loader as oracle

reference_paged, W, config = oracle.load()

C = 64  # the test model's context bucket


class Oracle:
    """bit-exact reference over one set of weights, with a cache of the logits of every token string it was asked about."""

    def __init__(self, cfg, seed=1):
        self.cfg = cfg
        self.w = W.synthetic(cfg, seed=seed)
        self._rows = {}

    def _walk(self, tokens, first_logit):
        """Run `tokens` through the paged step one position at a time; the logits of positions >= `first_logit`."""
        pool = reference_paged.BlockPool(self.cfg, 6)
        table = [pool.allocate() for _ in range(C // 16)]
        rows = []
        for pos, tok in enumerate(tokens):
            want = pos >= first_logit
            r = reference_paged.decode_step_paged(self.w, pool, int(tok), pos, table, C=C, logits=want)
            if want:
                rows.append(np.asarray(r.logits, dtype=np.float32))
        return rows

    def teacher_logits(self, prompt, forced):
        """(len(forced) + 1, vocab): the row k is the logits after prompt + forced[:k] (the row that picks forced[k])."""
        key = tuple(prompt) + tuple(forced)
        if key not in self._rows:
            self._rows[key] = np.stack(self._walk(list(prompt) + list(forced), len(prompt) - 1))
        return self._rows[key]

    def generate(self, prompt, n, transform=None):
        """Greedy tokens after `transform(logits, prompt, output) -> logits` (identity by default)."""
        tokens = list(prompt)
        out = []
        pool = reference_paged.BlockPool(self.cfg, 6)
        table = [pool.allocate() for _ in range(C // 16)]
        pos = 0
        while len(out) < n:
            last = pos >= len(prompt) - 1
            r = reference_paged.decode_step_paged(self.w, pool, tokens[pos], pos, table, C=C, logits=last)
            if last:
                logits = np.asarray(r.logits, dtype=np.float32)
                if transform is not None:
                    logits = transform(logits.copy(), list(prompt), list(out))
                t = int(np.argmax(logits))
                out.append(t)
                tokens.append(t)
            pos += 1
        return out


def prompt_ids(seed, n, vocab):
    return [int(t) for t in np.random.default_rng(seed).integers(1, vocab, n)]
