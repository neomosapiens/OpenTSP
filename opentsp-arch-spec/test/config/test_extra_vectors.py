"""The side vector sets of conformance/model-extra/ (, , ; outside the freeze set): they pin what the frozen vectors of
conformance/model/ do not. Checks the committed files, not the model: every (VXM entry, allowed element type) pair of encoding.json occurs in the
text of some vector, OTS-E-IQ-003 is expected by some vector, the patrol scrubber corrects in the patrol profile's vectors, and every `F` kind
(sram, srf, iq, link, mxm) is injected by some vector. `gen_model_vectors.py --extra --check` replays them on the model.
usage: test_extra_vectors.py"""
import json, sys, unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC / "tools"))
sys.path.insert(0, str(SPEC / "test"))
import gen_model_vectors as g
EXTRA = SPEC / "conformance/model-extra"


def load(name):
    return json.load(open(EXTRA / f"{name}.json"))


def pairs_of(doc, cfg):
    """The (VXM entry, element type) pairs the text of the vectors holds."""
    from oracle import numerics as nx
    cat = g.Catalogue(cfg)
    out = set()
    for v in doc["vectors"]:
        for rec in g.recs_of_text(g.images_of(v["text"])):
            d = cat.isa.decode64(rec[3]) if rec[3] >> 60 in cat.wide else None
            if d and cat.isa.by_name[d[0]].slice == "VXM" and "type" in d[1]:
                t = d[1]["type"]
                out.add((d[0], nx.TYPE_NAME[t] if isinstance(t, int) else t))
    return out


class ExtraSets(unittest.TestCase):
    def test_every_vxm_type_pair_occurs(self):
        for prof, cfg in (("ref", SPEC / "config/ref.json"), ("mini", SPEC / "config/mini.json"), ("ref_hw", SPEC / "config/test/ref_hw.json")):
            from oracle import numerics as nx
            cat = g.Catalogue(str(cfg))
            declared = json.loads(cfg.read_text()).get("extensions", [])
            want = {(n, nx.TYPE_NAME[c]) for n, c in g.vxm_type_pairs(cat, declared)}
            have = pairs_of(load(prof), str(cfg))
            self.assertEqual(sorted(want - have), [], f"{prof}: pairs no side vector uses")

    def test_iq003_is_expected(self):
        for prof in ("ref", "ref_hw", "ref_patrol"):      # not mini: its 32-byte payload leaves the staging FIFO before the next IfetchM can issue
            self.assertTrue(any(d["code"] == "OTS-E-IQ-003" for v in load(prof)["vectors"] for d in v["expect"]["diagnostics"]), prof)

    def test_patrol_profile_corrects(self):
        doc = load("ref_patrol")
        scrubs = [e for v in doc["vectors"] for e in v["expect"].get("ext_events", []) if " scrub " in e]
        self.assertTrue(any(" unc" not in e for e in scrubs), "no corrected scrub")
        self.assertTrue(any(e.endswith(" unc") for e in scrubs), "no uncorrectable scrub")
        self.assertEqual(doc["config_sha256"], g.config_digest(str(SPEC / "config/test/ref_patrol.json")))

    def test_every_fault_kind_is_injected(self):
        for prof in ("ref", "mini", "ref_hw"):
            kinds = {l.split()[1] for v in load(prof)["vectors"] for l in v.get("host", []) if l.startswith("F ")}
            self.assertEqual(kinds, {"sram", "srf", "iq", "link", "mxm"}, prof)


if __name__ == "__main__":
    unittest.main()
