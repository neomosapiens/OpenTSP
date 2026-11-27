"""Bring-up IQ coverage agrees with the actual model-extra vectors."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
bring = (ROOT / "opentsp-arch-spec/docs/bring-up.md").read_text()
errors = []
# must not claim mini lacks the side vectors
if "IQ-003" in bring and "in none of `mini`" in bring:
    errors.append("bring-up still says IQ-003 is in none of mini")
if "lacks both" in bring or "mini side file lacks" in bring.lower():
    errors.append("bring-up still claims the mini side file lacks IQ-002/003")
# must name the side-file vectors
for name in ("iq002-000", "iq003-000"):
    if name not in bring:
        errors.append(f"bring-up does not name {name}")
# files must actually contain them
for prof in ("mini", "ref"):
    text = (ROOT / f"opentsp-arch-spec/conformance/model-extra/{prof}.json").read_text()
    if "iq002" not in text or "iq003" not in text:
        errors.append(f"model-extra/{prof}.json missing iq002/iq003")
# coverage plan and tap-to-trace sections exist
if "## 3. Coverage plan" not in bring or "## 4. Tap-to-trace map" not in bring:
    errors.append("bring-up missing coverage plan or tap-to-trace sections")
if errors:
    print("FAIL:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
print("ok: bring-up matches IQ side vectors")
