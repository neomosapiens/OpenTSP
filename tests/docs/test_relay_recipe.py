#!/usr/bin/env python3
"""topology.md gives the relay recipe: the cost a relay adds at the intermediate chip and the end-to-end latency, both computed here
from the `ref` profile (they are the figures `test/c2c/relay_directed.py` checks on the simulator), and the DeskewRecv form of the relay's receive.
No sentence of the specification points to a section that does not hold it."""
import json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cfg = json.load(open(os.path.join(ROOT, "opentsp-arch-spec", "config", "ref.json")))
L = cfg["c2c"]["link_latency"]
add = cfg["timing"]["c2c"]["d_func"] + cfg["layout"]["c2c_tx_gap"] - cfg["layout"]["c2c_rx_gap"]
total = 2 * L + add
text = open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "topology.md"), encoding="utf-8").read()
m = re.search(r"^### Relay through an intermediate chip\n(.*?)(?=^## |^### )", text, re.S | re.M)
fails = []
if not m:
    fails.append("topology.md has no section 'Relay through an intermediate chip'")
else:
    body = m.group(1)
    for what, needle in (("the cost added at the relay", f"{add} cycles"), ("the end-to-end latency", f"2 L + {add}"),
                         ("the end-to-end latency on ref", str(total)), ("the DeskewRecv form", "deskew_recv"),
                         ("the test that measures it", "relay_directed.py")):
        if needle not in body:
            fails.append(f"the relay section does not give {what} ({needle!r})")
if "microarchitecture.md §10" in text and "relay" in text.split("microarchitecture.md §10")[0][-200:]:
    fails.append("topology.md still sends the reader to microarchitecture.md §10 for the relay")
if fails:
    print("FAIL:\n  " + "\n  ".join(fails)); sys.exit(1)
print(f"relay recipe: all pass (relay adds {add}, end to end {total})")
