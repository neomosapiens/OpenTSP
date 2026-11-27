"""The `profile` option of the `iss` chip: the profile the chip and its programs are for. The serving binding's host side
(five vocabulary rows of 64 lanes per embedding word, 63 head installs of 160 rows, 320-byte output words, contexts 64 and 128) is written
for the geometry of `ref`: a profile of that geometry with other timing is accepted, any other is refused naming the first number
that differs, instead of reading the wrong words of a program built for it."""

import pytest

from opentsp_vllm import options


def parse(**kw):
    return options.parse({"opentsp": kw})


def test_the_profile_defaults_to_ref():
    assert parse()["profile"] == "ref"


@pytest.mark.parametrize("name", ["ref", "test/ref_slow", "test/ref_hw", "test/ref_patrol"])
def test_a_profile_of_the_geometry_of_ref_is_accepted(name):
    assert parse(profile=name)["profile"] == name


@pytest.mark.parametrize("name, what", [
    ("test/sl8", "8 superlanes"), ("test/sl32", "32 superlanes"), ("test/ref_small", "160 MXM rows"), ("mini", "1 superlanes"),
])
def test_another_geometry_is_refused_with_the_number_that_differs(name, what):
    with pytest.raises(ValueError, match=rf"opentsp.profile '{name}': the serving binding is written for the geometry of ref .*{what}"):
        parse(profile=name)


def test_the_profile_must_be_a_string_and_an_option_of_chip_iss():
    with pytest.raises(ValueError, match="opentsp.profile must be a string"):
        parse(profile=7)
    with pytest.raises(ValueError, match="opentsp.profile is an option of chip 'iss' only"):
        parse(chip="ship", profile="test/ref_slow")
