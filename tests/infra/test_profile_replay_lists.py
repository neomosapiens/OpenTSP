#!/usr/bin/env python3
"""Every test profile is replayed by the model's vector check and by the ISS, and the recipes that do it clean up after themselves.

The profile lists are kept by hand in three places: `VARIANTS` of opentsp-arch-spec/tools/model_test.sh (golden vectors generated and
replayed on the model), the `for p in ...` list of `iss-vectors-variants` and the `for c in ...` list of `iss-vectors-superlanes` (the
same vectors replayed by the ISS). A profile added under opentsp-arch-spec/config/test/ that is in none of them is never compared with the
ISS, so this test names every profile that one of the lists lacks.

Run: python3 tests/infra/test_profile_replay_lists.py"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_PROFILES = os.path.join(ROOT, "opentsp-arch-spec", "config", "test")


def read(*parts):
    with open(os.path.join(ROOT, *parts)) as fh:
        return fh.read()


def recipe(text, name):
    """The body of a just recipe: the lines after `name:` up to the next line that starts in column 0 and is not blank."""
    m = re.search(rf"^{re.escape(name)}[^\n]*:\n((?:[ \t]+[^\n]*\n|\n)+)", text, re.M)
    assert m, f"recipe {name} not found"
    return m.group(1)


def model_profiles():
    m = re.search(r'^VARIANTS="([^"]*)"', read("opentsp-arch-spec", "tools", "model_test.sh"), re.M)
    assert m, "VARIANTS not found in model_test.sh"
    return {p.split("/")[-1] for p in m.group(1).split()}


def iss_profiles():
    body = recipe(read("tools", "just", "conformance.just"), "iss-vectors-variants")
    names = set(re.search(r"for p in ([^;]+); do", body).group(1).split())
    body = recipe(read("tools", "just", "spec-verification.just"), "iss-vectors-superlanes")
    names |= {c.split("/")[-1] for c in re.search(r"for c in ([^;]+); do", body).group(1).split()}
    return names


class ReplayLists(unittest.TestCase):
    def setUp(self):
        self.profiles = {f[:-5] for f in os.listdir(TEST_PROFILES) if f.endswith(".json")}

    def test_there_are_test_profiles(self):
        self.assertGreaterEqual(len(self.profiles), 11)

    def test_every_test_profile_is_replayed_on_the_model(self):
        self.assertEqual(sorted(self.profiles - model_profiles()), [], "profiles missing from VARIANTS of model_test.sh")
        self.assertEqual(sorted(model_profiles() - self.profiles), [], "VARIANTS names a profile that does not exist")

    def test_every_test_profile_is_replayed_by_the_iss(self):
        self.assertEqual(sorted(self.profiles - iss_profiles()), [], "profiles missing from the ISS replay recipes")
        self.assertEqual(sorted(iss_profiles() - self.profiles), [], "an ISS replay recipe names a profile that does not exist")

    def test_the_replay_recipes_remove_their_scratch_directory(self):
        for f, name in (("conformance.just", "iss-vectors-variants"), ("spec-verification.just", "iss-vectors-superlanes")):
            body = recipe(read("tools", "just", f), name)
            if "mktemp -d" in body:
                self.assertRegex(body, r"trap '[^']*' EXIT|trap \"[^\"]*\" EXIT", f"{name} leaves its vectors in /tmp")

    def test_a_duration_in_the_vector_recipes_comments_is_measured_or_labelled(self):
        text = read("justfile") + read("tools", "just", "spec-verification.just")
        for m in re.finditer(r"about (\d+) minutes more|About (\d+) minutes\.", text):
            line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]
            self.assertIn("estimate", line, line)


if __name__ == "__main__":
    unittest.main()
