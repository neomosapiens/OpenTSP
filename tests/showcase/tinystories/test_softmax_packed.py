"""Verify that the packed softmax (`opentsp_tools.programs.kernels.softmax_packed.py`) gives, word for word, bit-exact reference's per-head softmax
(`paged_reference.softmax_full`, the `ts_softmax` order over one segment of C/16 superlanes), and it cuts the layer's cycles."""
import numpy as np
import pytest


from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_tools.programs import ref
from opentsp_tools.programs.kernels import softmax_packed as spk

HEADS = 16


def shift(v, direction, n):
    out = np.zeros_like(v)
    if direction == "up":
        out[n:] = v[:len(v) - n]
    else:
        out[:len(v) - n] = v[n:]
    return out


def chip_scores(rng, C, pos):
    """16 score vectors as the chip holds them: lanes 0..C-1 the masked scores, lanes C.. finite junk (rows beyond the
    install), with exact zeros of both signs among the scores."""
    s = np.zeros((HEADS, spk.VLANES), np.float32)
    s[:, :C] = (rng.standard_normal((HEADS, C)) * 4).astype(np.float32)
    s[:, :C][rng.random((HEADS, C)) < 0.1] = np.float32(-0.0)
    s[:, :C] = ref._add(s[:, :C], G.position_mask(pos, C))
    s[:, C:] = (rng.standard_normal((HEADS, spk.VLANES - C)) * 100).astype(np.float32)
    return s


def packed_twin(s, C):
    """The packed kernel's arithmetic: place (two shifts), sum per pack, softmax with segments of C/16 superlanes,
    retrieve (two shifts)."""
    n = spk.per_vector(C)
    placed = []
    for h in range(HEADS):
        v = s[h]
        for d, a in spk.placement(C, h % n):
            v = shift(v, d, a)
        placed.append(v)
    probs = []
    for j in range(0, HEADS, n):
        acc = placed[j]
        for v in placed[j + 1:j + n]:
            acc = ref._add(acc, v)
        probs.append(ref.softmax(acc, C // 16))
    out = []
    for h in range(HEADS):
        v = probs[h // n]
        if h % n:
            for d, a in spk.retrieval(C, h % n):
                v = shift(v, d, a)
        out.append(v)
    return np.array(out)


@pytest.mark.parametrize("C", [64, 128])
def test_the_packed_softmax_equals_reference_word_for_word(C):
    rng = np.random.default_rng(C)
    for pos in (0, 1, 15, 16, C // 2, C - 1):
        s = chip_scores(rng, C, pos)
        want = G.softmax_full(s[:, :C])
        got = packed_twin(s, C)
        assert got[:, :C].view(np.uint32).tolist() == want.view(np.uint32).tolist()
        assert np.isfinite(got).all()


def softmax_span(layers, C):
    b = E.emit_step_batch(1, 1, C, layers=layers, measure=True)
    m = {(l, st): c for l, st, c in b.marks}
    return m[(0, "softmax")] - m[(0, "scores")]


def test_the_packed_softmax_stage_is_under_a_third_of_the_serial_one():
    """The serial stage (16 calls) takes about 8 400 cycles at C = 64."""
    assert softmax_span(2, 64) < 2800
