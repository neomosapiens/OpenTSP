#!/usr/bin/env python3
"""Write conformance/model-extra/all_ext.json from the ACC x128 / bfp8 programs only.

Full gen_model_vectors.py on all_ext also captures the directed/ref templates; this helper keeps the
model-extra dump (outside the freeze set) to the extension programs named by the extension-vector work, so the RTL sidecar can bind them.
"""
import os
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC / "tools"))
import gen_model_vectors as g


def main(argv):
    if len(argv) < 3:
        print("usage: gen_all_ext_vectors.py <tsp_sim> <out.json>  (use conformance/model-extra/all_ext.json)", file=sys.stderr)
        return 2
    sim, out = argv[1], argv[2]
    cfg = str(SPEC / "config/test/all_ext.json")
    cat = g.Catalogue(cfg)
    progs = [("directed", p) for p in g.wide_programs(sim, cfg)]
    if not progs:
        print("gen_all_ext_vectors: wide_programs returned nothing", file=sys.stderr)
        return 1
    vectors = []
    for i, (kind, prog) in enumerate(progs):
        recs, sram, dumps, units = g.fit(prog, cfg, cat, False)
        host = list(prog[4]) if len(prog) > 4 else []
        images = g.lowered(recs, units)
        d, w, ev, ext = g.run_ext(sim, cfg, images, sram, dumps, host)
        ins, uni = cat.tags(g.recs_of_text(images))
        vectors.append({
            "name": f"{kind}-{i:03d}",
            "instructions": ins,
            "units": uni,
            "text": g.text_json(images),
            "sram": {f"{sl}:{a}": bytes(v).hex() for (sl, a), v in sram.items()},
            "dump": [f"{sl}:{a}" for sl, a in dumps],
            **({"host": host} if host else {}),
            "expect": {"diagnostics": d, "words": w, **g.expect_trace(ev), **g.ext_part(ext)},
        })
    doc = {
        "schema": "opentsp.conformance.model",
        "spec_version": "0.1",
        "profile": "all_ext",
        "config_sha256": g.config_digest(cfg),
        "text": "queue (0..143, 65535 = control channel) -> the 32-bit units of its initial instruction queue, hex, space separated",
        "vectors": vectors,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    g.write_vectors(doc, out)
    print(f"golden vectors: {len(vectors)} written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
