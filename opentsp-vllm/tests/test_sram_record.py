"""The binding's two read-only accessors the independent replay shadow needs : `Chip.sram_changes`,
the SRAM words that changed since the previous call (the host's record of a program's effect), and `Chip.state_digest`, the SHA-256 of
everything but the SRAM. Not part of specification 0.1: the binding's. Run after `just vllm-binding` (`just vllm-binding-test`); a missing
binding fails, never skips."""

import os
import pathlib
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))

from opentsp_vllm import _tsp_iss  # noqa: E402


def word(chip, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(1, 256) for _ in range(chip.vlanes))


def test_the_first_call_reports_every_nonzero_word_and_a_second_call_none():
    c = _tsp_iss.Chip("ref")
    assert c.sram_changes() == []
    c.write_words(5, 0x40, word(c, 1) + word(c, 2))
    c.write_words(3, 7, word(c, 3))
    c.write_words(3, 9, bytes(c.vlanes))  # a zero word is not a change
    assert [(s, a, bytes(b)) for s, a, b in c.sram_changes()] == [(3, 7, word(c, 3)), (5, 0x40, word(c, 1)), (5, 0x41, word(c, 2))]
    assert c.sram_changes() == []


def test_a_rewritten_word_is_reported_with_its_new_bytes_and_a_zeroed_one_with_zeros():
    c = _tsp_iss.Chip("ref")
    c.write_words(1, 1, word(c, 1) + word(c, 2) + word(c, 3))
    c.sram_changes()
    c.write_words(1, 1, word(c, 1))  # the same bytes: no change
    c.write_words(1, 2, word(c, 9))
    c.write_words(1, 3, bytes(c.vlanes))
    assert [(s, a, bytes(b)) for s, a, b in c.sram_changes()] == [(1, 2, word(c, 9)), (1, 3, bytes(c.vlanes))]
    assert c.sram_changes() == []
    c.write_words(1, 3, word(c, 4))  # a zeroed word that is written again
    assert [(s, a, bytes(b)) for s, a, b in c.sram_changes()] == [(1, 3, word(c, 4))]


def test_the_changes_of_every_call_in_order_rebuild_the_sram_of_a_new_chip():
    c, new = _tsp_iss.Chip("ref"), _tsp_iss.Chip("ref")
    rng = random.Random(7)
    for _ in range(6):
        for _ in range(40):
            s, a = rng.randrange(0, 60), rng.randrange(0, 50)
            c.write_words(s, a, word(c, rng.randrange(10)) if rng.random() < 0.7 else bytes(c.vlanes))
        for s, a, b in c.sram_changes():
            new.write_words(s, a, bytes(b))
        assert new.sram_digest() == c.sram_digest()


def test_the_state_digest_is_the_hash_of_the_registers_and_ignores_the_sram_contents():
    a, b = _tsp_iss.Chip("ref"), _tsp_iss.Chip("ref")
    d = a.state_digest()
    assert isinstance(d, bytes) and len(d) == 32 and d == b.state_digest()
    a.write_words(4, 4, word(a, 1))  # the aperture's address register moves with a write; the contents are not state
    b.write_words(4, 4, word(b, 2))
    assert a.state_digest() == b.state_digest() != d
    assert a.sram_digest() != b.sram_digest()


def test_a_warm_restart_gives_the_state_of_a_new_chip_and_keeps_the_sram():
    """`_restart_for_test` is the binding's restart in the current mode without a run: the state digest after a warm restart is a new
    chip's, which no step can show (a program installs the state it uses). A replay leaves the aperture's address register at the last word it
    wrote back (the sequence is reset and aperture writes), so its digest is not a new chip's."""
    new = _tsp_iss.Chip("ref").state_digest()
    for mode in ("warm", "replay"):
        c = _tsp_iss.Chip("ref")
        c.restart_mode(mode)
        c.write_words(5, 0x40, word(c, 1))
        c.write_words(3, 7, word(c, 3))
        before = c.sram_digest()
        assert c.state_digest() != new, "the aperture's address register moved"
        d = c._restart_for_test()
        assert (d == new) == (mode == "warm")
        assert c.sram_digest() == before
        assert c.stats()["restarts_warm" if mode == "warm" else "restarts_replay"] == 1
