"""The `replay` option of the `iss` chip : unset by default (the environment decides, else the normal driver), true or false only,
and only for the chip it belongs to: what a person who mistypes it sees."""

import pytest

from opentsp_vllm import options


def parse(**kw):
    return options.parse({"opentsp": kw})


def test_replay_is_unset_by_default_and_takes_true_or_false():
    assert parse()["replay"] is None
    assert parse(replay=True)["replay"] is True
    assert parse(replay=False)["replay"] is False


@pytest.mark.parametrize("bad", ["yes", 1, 0, "on"])
def test_replay_must_be_a_boolean(bad):
    with pytest.raises(ValueError, match="opentsp.replay must be true or false"):
        parse(replay=bad)


def test_replay_is_an_option_of_chip_iss_only():
    with pytest.raises(ValueError, match="opentsp.replay is an option of chip 'iss' only"):
        parse(chip="ship", replay=True)
    parse(chip="iss", replay=True)
