"""Model-reaching hold obligations and their exact mutation controls remain wired."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = ROOT / "opentsp-arch-spec/test/formal/mutants_expected.txt"
UNIT = ROOT / "opentsp-arch-spec/model/unit_tests/test_vxm_mxm.sail"
MINI = ROOT / "opentsp-arch-spec/config/mini.json"
HOLD = ROOT / "opentsp-arch-spec/test/formal/props/hold.sail"
PE = ROOT / "opentsp-arch-spec/test/formal/prove_extra.sh"
SCRIPTS = ROOT / "opentsp-arch-spec/test/formal/test_scripts.py"

KILLED = ("m_t1", "m_ws1", "m_be1", "m_bf1")


def test_model_reaching_obligations():
    expected = EXPECTED.read_text(encoding="utf-8")
    for n in KILLED:
        assert re.search(rf"(?m)^{n}\s+kill\b", expected), n
    unit = UNIT.read_text(encoding="utf-8")
    assert 'entry_latency_typed("Permute", ""' in unit
    assert 'entry_row("Permute").d_func != latency_of(T_SXM).d_func' in unit
    sail = HOLD.read_text(encoding="utf-8")
    assert "function hold_decide" not in sail
    assert "option(mem_path)" in sail
    pe = PE.read_text(encoding="utf-8")
    hold_arm = pe.split("hold)", 1)[1].split("*)", 1)[0]
    assert "postlude" in hold_arm
    scripts = SCRIPTS.read_text(encoding="utf-8")
    assert "class ModelReachingObligations" in scripts
    assert 'KILLED = ("m_t1", "m_ws1", "m_be1", "m_bf1")' in scripts
    mini = json.loads(MINI.read_text(encoding="utf-8"))
    perm = next(o for o in mini["timing"]["overrides"] if o["entry"] == "Permute")
    assert perm["d_func"] != mini["timing"]["sxm"]["d_func"], "m_t1 kill needs Permute != SXM class"


if __name__ == "__main__":
    test_model_reaching_obligations()
    print("ok")
