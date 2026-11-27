"""Every model harness checks its configuration like tsp_sim.

Runs <harness> [args...] on configurations derived from <config.json> that break one schema
rule (c2c.links 0, a negative d_func, link_latency_max not below hac_period) or a rule of
the timing table the model checks (an override of an unknown entry). Each must be
refused with exit 2 and one stderr line "<prog>: config <path>: <key>: <reason>" naming the
key, before the model runs. (The profile itself is loaded by the
harness's normal run in the calling script.)

usage: harness_reject.py <harness> <config.json> [harness args after the config...]
"""
import copy, json, os, subprocess, sys, tempfile

harness, base_path, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
base = json.load(open(base_path))


def mutate(key, value):
    c = copy.deepcopy(base)
    node = c
    *path, last = key.split(".")
    for k in path:
        node = node[k]
    node[last] = value
    return c


CASES = [
    ("links 0", "c2c.links", mutate("c2c.links", 0)),
    ("mem d_func -3", "timing.mem.d_func", mutate("timing.mem.d_func", -3)),
    ("link_latency_max 300 >= hac_period", "c2c.link_latency_max", mutate("c2c.link_latency_max", 300)),
    ("override of an unknown entry", "timing.overrides", mutate("timing.overrides", [{"entry": "Frobnicate", "d_func": 5, "d_skew": 0}])),
]
fails = 0
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "cfg.json")
    for name, key, cfg in CASES:
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        p = subprocess.run([harness, path, *extra], input="", capture_output=True, text=True, timeout=600)
        ok = p.returncode == 2 and f"config {path}: {key}" in p.stderr and p.stderr.count("\n") == 1
        print(f"{'ok  ' if ok else 'FAIL'} {os.path.basename(harness)} {name}: rc {p.returncode}: {p.stderr.strip()[:160]}")
        fails += not ok
print(f"harness config check ({os.path.basename(harness)}): " + ("FAILURES" if fails else "all pass"))
sys.exit(1 if fails else 0)
