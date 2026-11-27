#!/usr/bin/env python3
"""An analytic mapping of a decoder-only transformer onto N chips of a profile .

usage: model_map.py <model.json> <profile.json> --chips N [--clock-mhz 900] [--batch B] [--phase decode|prefill --prompt-len P]
                    [--links-per-hop K] [--sram-fraction F] [--layer-overhead-cycles C]
                    [--topology flat|FILE | --wiring-params FILE] [--json]

For each of three splits (a layer pipeline, tensor parallelism inside every layer, and a hybrid of pipeline stages that are tensor
groups inside one node) the tool derives: the SRAM footprint per chip (weights plus the KV cache of `batch` sequences), the MXM
cycles of a layer from the derived rates, the chip-to-chip bytes and cycles per token, and tokens per second with the binding bound
named. EVERY figure is an estimate: the clock is an assumption
(derive_peaks.py), the MXM is a throughput bound, and the only measured evidence is one toy model (below). A report always ends
with the model checked against that evidence, and states the error.

Model file: `layers`, `hidden`, `heads`, `kv_heads`, `ffn`, `ffn_matrices` (2 or 3), `vocab`, `context`, `weight_bytes` (1 or 2),
`kv_bytes` (1 or 2), the optional `weight_type` (`int8` or `fp8` for 1 byte, `f16`, `bf16` or `bfp8` for 2; bf16 and bfp8 need the profile's extension and have the rate of f16; default `int8` and `f16`: it picks the MAC rate the
weights use) and the optional `act_bytes` (1, 2 or 4; default 2): the bytes per element of what crosses a link, that is the
width of a partial sum in an all-reduce, of an activation in a pipeline hop and of a logit in the all-gather. The default is an f16
partial sum: the matrix unit's INT8 products accumulate in 32 bits, and converting them to f16 before the reduction is a vector-unit
operation this tool does not price. `act_bytes` 4 sends the accumulators unconverted; 1 needs a requantisation that is not modelled and
is for sensitivity runs only. The payload sets the all-reduce cycles directly (one vector of 16 x superlanes bytes per
c2c.cycles_per_vector cycles per link), so the tokens per second of a split that communicates every layer move with it.

The model (all in cycles of the profile; `rate` = 2 x streams readers x vlanes bytes per cycle, the weight-install rate that
derive_peaks.py derives, and `mac` = the MACCs per cycle of the arithmetic of the weights):
  weights    layer = 2 H^2 + 2 H kv_dim + ffn_matrices H F parameters; head = vocab H; bytes = parameters x weight_bytes
             (H = hidden, kv_dim = kv_heads x H / heads; the embedding table stays on the host and only the head is on chip)
  KV cache   per sequence and layer, context x 2 x kv_dim x kv_bytes; its K and V are installed into the MXM like weights
  install    (weight bytes + batch x KV bytes of a layer on the chip) / rate: a weight is used once per installation, so the
             install is the bound at small batch
  MAC        tokens x (layer parameters + 2 x context x H attention MACCs) / mac: INT8 mac = 2 P x rows x vlanes (P =
             mxm.planes_per_hemisphere); FP8 mac = mxm.fp8_rate x the INT8 mac; FP16 mac = 2 x mxm.f16_rows x vlanes /
             timing.mxm_abc.occupancy_f16, and 0 with one plane per hemisphere (no plane can hold f16: an f16 model is refused)
  layer      ceil(max(install, MAC)) + the layer overhead (a measured latency floor the throughput bound cannot see)
  pipeline   stages = min(chips, layers), stage i on chip i; a hop moves the activations of the step over one link
             (links-per-hop links), or, across a node boundary, over three link hops and two relays (below)
  tensor     weights and KV sharded over the chips (KV and attention over min(chips, heads)); two all-reduces per layer and one
             all-gather of the logits, priced on the wiring (below); the reduction arithmetic itself is not counted
  hybrid     stages that are tensor groups of t chips inside one node (t = 2, 4, 8 on the modelled wiring); the best of the
             groupings; one copy of the activations crosses between stages and the next group's first chip serves the others
  best       of the splits that fit in SRAM, the one with more tokens per second; none when none fits

The wiring (WIRING). By default the wiring `tsp_sim --chips 8N` builds (topology.md section 7): nodes of 8 chips, fully connected by
7 local links; one global link between every pair of nodes, on global ports 4 per chip, so that with 8 nodes two chips of a node own
its seven global links. A chip has no link to a partner in another node: data crosses through the chip that owns the global link,
and every receive-then-send turnaround costs a relay, d_func(c2c) + c2c_tx_gap - c2c_rx_gap cycles (34 on ref; a relay of two hops
measures 2 L + 34 in test/c2c/relay_directed.py). An all-reduce of more than one node is hierarchical: an all-reduce inside each
node, the node sum over the global links, an exchange between the owner chips, and the owners serving the other chips of the node.
`--topology flat` prices a direct link between every pair of chips (HYPOTHETICAL: no harness builds it). `--topology FILE` reads the file
`tsp_sim --topology` reads (the files tools/gen_topology.py writes: `chips` and `links` of `{"a": [chip, link], "b": [chip, link]}`) and
derives the nodes (groups of fully connected chips), the global links per node pair and the global ports per chip from the link list;
a wiring that is not nodes of fully connected chips joined by global links (a torus, a tray ring) is refused with the reason. Such a wiring
is the simulator's, so the report does not call it hypothetical. `--chips` defaults to the file's chip count. `--wiring-params FILE` prices
a parametric wiring no harness builds (`name`, `chips_per_node`, `global_links_per_node_pair` (default 1) and `global_ports_per_chip`
(default 4)), HYPOTHETICAL in every report line.
Limits (topology.md section 7): a run takes 1 to 264 chips (`tsp_sim --chips`, the last node may be partial), a topology file up to 512 (the
9-bit chip_id addresses 512); beyond 264 the default wiring needs a topology file, beyond 512 nothing can be addressed.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import derive_peaks as dp  # noqa: E402

LABEL = "estimate"
RUN_CHIPS = 264              # tsp_sim --chips takes 1..264 (topology.md section 7)
ADDRESSABLE_CHIPS = 512      # chip_id has 9 bits: --topology and the addressing stop here
MAX_NODES = 33               # a node has 8 chips x 4 global ports = 32 global links: one to each of 32 other nodes (33 nodes in all)
WEIGHT_TYPES = {1: ("int8", "fp8"), 2: ("f16", "bf16", "bfp8")}  # bf16 and bfp8 need their extension and take the capacity and rate of f16
NODE_CHIPS = 8               # tsp_sim --chips 8N: nodes of 8 chips
LOCAL_LINKS = 7              # local links 0..6 of a node (tsp_sim.c: fully connected over 7 links)
GLOBAL_PORTS_PER_CHIP = 4    # global links 7..10 of a chip
MODEL_KEYS = ["layers", "hidden", "heads", "kv_heads", "ffn", "ffn_matrices", "vocab", "context", "weight_bytes", "kv_bytes"]
DEFAULT_ACT_BYTES = 2        # an f16 partial sum (see the docstring)
ACT_BYTES = (1, 2, 4)

# The one measured reference (the showcase model, calum/tinystories-gpt2-3M, 64 KV slots, f16 weights): the serial layer of the
# TinyStories block program (last write at cycle 11,362) and the
# eight-layer decode step (26,398 cycles).
TINYSTORIES = {"name": "tinystories-3m", "layers": 8, "hidden": 64, "heads": 16, "kv_heads": 16, "ffn": 256, "ffn_matrices": 2,
               "vocab": 50257, "context": 64, "weight_bytes": 2, "kv_bytes": 2}
MEASURED = [("serial layer", 11362, "one layer, one token, stages run one after another"),
            ("8-layer decode step", 26398, "decode step of 8 layers, final LayerNorm and head, after the program optimisation")]


class ModelError(Exception):
    """An input the tool refuses; main prints it and exits 2."""


def weight_type(m):
    """The MAC rate the weights use: the model's `weight_type`, or int8 for 1 byte and f16 for 2."""
    return m.get("weight_type", WEIGHT_TYPES[m["weight_bytes"]][0])


def check_model(m):
    for key in MODEL_KEYS:
        if key not in m:
            raise ModelError(f"{key}: missing from the model file")
        if isinstance(m[key], bool) or not isinstance(m[key], int) or m[key] <= 0:
            raise ModelError(f"{key}: must be a positive integer (got {m[key]!r})")
    if m["hidden"] % m["heads"]:
        raise ModelError(f"heads: {m['heads']} must divide hidden ({m['hidden']})")
    if m["heads"] % m["kv_heads"]:
        raise ModelError(f"kv_heads: {m['kv_heads']} must divide heads ({m['heads']})")
    if m["ffn_matrices"] not in (2, 3):
        raise ModelError("ffn_matrices: must be 2 (up, down) or 3 (gated: gate, up, down)")
    if m["weight_bytes"] not in (1, 2):
        raise ModelError("weight_bytes: must be 1 (INT8) or 2 (FP16); no other type has a MAC rate in the model")
    if m["kv_bytes"] not in (1, 2):
        raise ModelError("kv_bytes: must be 1 or 2")
    if "weight_type" in m and m["weight_type"] not in WEIGHT_TYPES[m["weight_bytes"]]:
        raise ModelError(f"weight_type: {m['weight_type']!r} does not go with weight_bytes {m['weight_bytes']}: it must be "
                         + " or ".join(repr(x) for x in WEIGHT_TYPES[m["weight_bytes"]]))
    if "act_bytes" in m:
        a = m["act_bytes"]
        if isinstance(a, bool) or not isinstance(a, int) or a not in ACT_BYTES:
            raise ModelError(f"act_bytes: must be 1, 2 or 4, the bytes per element that cross a link (got {a!r}; "
                             f"default {DEFAULT_ACT_BYTES}, an f16 partial sum)")
    return m


def links_of(cfg):
    c2c = cfg["c2c"]
    return {"links": dp.lookup(cfg, "c2c.links"), "cpv": dp.lookup(cfg, "c2c.cycles_per_vector"),
            "latency": dp.lookup(cfg, "c2c.link_latency"), "vlanes": 16 * dp.lookup(cfg, "geometry.superlanes"),
            "min_latency": c2c.get("link_latency_min")}


def rates(cfg):
    """Per-cycle rates of the profile: weight-install bytes, INT8, FP8 and FP16 MACCs (the formulas of derive_peaks.py, over the same keys)."""
    q = dp.derive(cfg, 1.0)  # at 1 MHz: the cycle-based quantities do not depend on the clock
    vlanes = int(q["vlanes"].value)
    rows, per_hemisphere = dp.lookup(cfg, "mxm.rows"), dp.lookup(cfg, "mxm.planes_per_hemisphere")
    int8 = dp.HEMISPHERES * per_hemisphere * rows * vlanes
    f16 = dp.f16_matrices(per_hemisphere) * dp.lookup(cfg, "mxm.f16_rows") * vlanes / dp.lookup(cfg, "timing.mxm_abc.occupancy_f16")
    return {"install_bytes": 2 * int(q["install_readers"].value) * vlanes,
            "int8_macs": int8, "fp8_macs": dp.lookup(cfg, "mxm.fp8_rate") * int8,
            "f16_macs": f16,
            "bf16_macs": 0 if q["bf16_tflops"].value == 0 else f16, "bfp8_macs": 0 if q["bfp8_tflops"].value == 0 else f16,
            "sram_bytes": int(q["sram_capacity_bytes"].value)}


def vectors(nbytes, vlanes):
    return -(-int(math.ceil(nbytes)) // vlanes)


def allreduce_cycles(cfg, chips, nbytes, links_per_hop=1, direct_peers=None):
    """Cycles of an all-reduce of `nbytes` per chip, by algorithm, with a direct link to every partner (the flat wiring);
    `direct_peers` is how many peers a chip has a link to (default c2c.links). The reduction arithmetic is not counted."""
    if chips == 1:
        return {"none": 0}
    k = links_of(cfg)
    lat, cpv, v = k["latency"], k["cpv"], k["vlanes"]
    whole = vectors(nbytes, v)
    chunk = vectors(-(-nbytes // chips), v)
    out = {"ring": 2 * (chips - 1) * (lat + -(-chunk // links_per_hop) * cpv),
           "recursive_doubling": math.ceil(math.log2(chips)) * (lat + -(-whole // links_per_hop) * cpv)}
    if chips - 1 <= (k["links"] if direct_peers is None else direct_peers):
        out["direct"] = lat + max(whole * cpv, -(-whole * (chips - 1) // dp.C2C_TX_QUEUES))  # the injection: 4 vectors per cycle
    return out


def allgather_cycles(cfg, chips, shard_bytes, links_per_hop=1, direct_peers=None):
    """Cycles of an all-gather where every chip contributes `shard_bytes`, by algorithm (flat wiring, as allreduce_cycles)."""
    if chips == 1:
        return {"none": 0}
    k = links_of(cfg)
    lat, cpv, v = k["latency"], k["cpv"], k["vlanes"]
    shard = vectors(shard_bytes, v)
    out = {"ring": (chips - 1) * (lat + -(-shard // links_per_hop) * cpv),
           "recursive_doubling": sum(lat + -(-vectors(shard_bytes * 2 ** i, v) // links_per_hop) * cpv
                                     for i in range(math.ceil(math.log2(chips))))}
    if chips - 1 <= (k["links"] if direct_peers is None else direct_peers):
        out["direct"] = lat + max(shard * cpv, -(-shard * (chips - 1) // dp.C2C_TX_QUEUES))  # the injection: 4 vectors per cycle
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# The wiring: which chips have a link between them, and what a receive-then-send turnaround costs.

class WiringError(Exception):
    """A topology file the tool refuses."""


def modelled_wiring():
    """The wiring of `tsp_sim --chips 8N` (topology.md section 7)."""
    return {"name": "modelled", "kind": "nodes", "chips_per_node": NODE_CHIPS, "local_links": LOCAL_LINKS,
            "global_links_per_node_pair": 1, "global_ports_per_chip": GLOBAL_PORTS_PER_CHIP, "hypothetical": False,
            "describe": "modelled: tsp_sim --chips 8N, nodes of 8 fully connected chips, one global link per node pair"}


def flat_wiring():
    """A direct link between every pair of chips: no real harness wiring (pricing kept for comparison)."""
    return {"name": "flat", "kind": "flat", "hypothetical": True,
            "describe": "HYPOTHETICAL flat wiring: a direct link between every pair of chips (not what the harness builds)"}


def read_json(path, what):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError as e:
        raise WiringError(f"{path}: cannot read the {what} ({e.strerror})")
    except ValueError as e:
        raise WiringError(f"{path}: not valid JSON ({e})")


def load_wiring_params(path):
    """A parametric wiring no harness builds (`--wiring-params FILE`): HYPOTHETICAL."""
    t = read_json(path, "wiring file")
    if not isinstance(t, dict):
        raise WiringError(f"{path}: the wiring must be a JSON object")
    if not isinstance(t.get("name"), str) or not t["name"]:
        raise WiringError(f"{path}: name: missing (a hypothetical wiring is always named)")
    out = {"name": t["name"], "kind": "nodes", "hypothetical": True}
    for key, default, floor in (("chips_per_node", None, 2), ("global_links_per_node_pair", 1, 1), ("global_ports_per_chip", GLOBAL_PORTS_PER_CHIP, 1)):
        v = t.get(key, default)
        if v is None:
            raise WiringError(f"{path}: {key}: missing")
        if isinstance(v, bool) or not isinstance(v, int) or v < floor:
            raise WiringError(f"{path}: {key}: must be an integer of at least {floor} (got {v!r})")
        out[key] = v
    out["local_links"] = out["chips_per_node"] - 1
    out["describe"] = (f"HYPOTHETICAL wiring '{t['name']}': nodes of {out['chips_per_node']} fully connected chips, "
                       f"{out['global_links_per_node_pair']} global link(s) per node pair, {out['global_ports_per_chip']} global ports per chip")
    return out


def _end(path, i, key, end, chips):
    """One end [chip, link] of a link entry of a topology file."""
    if (not isinstance(end, list) or len(end) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in end)):
        raise WiringError(f"{path}: links[{i}].{key}: must be [chip, link] (two non-negative integers, got {end!r})")
    if end[0] >= chips:
        raise WiringError(f"{path}: links[{i}].{key}: chip {end[0]} does not exist in a topology of {chips} chips")
    return end[0], end[1]


def load_topology(path):
    """`--topology FILE`: the topology file `tsp_sim --topology` reads (topology.md section 7), as `tools/gen_topology.py` writes it. The
    nodes (groups of fully connected chips), the global links per node pair and the global ports per chip are derived from its link list."""
    if path == "flat":
        return flat_wiring()
    t = read_json(path, "topology file")
    if not isinstance(t, dict):
        raise WiringError(f"{path}: the topology must be a JSON object")
    chips = t.get("chips")
    if isinstance(chips, bool) or not isinstance(chips, int) or chips < 1:
        raise WiringError(f"{path}: chips: missing or not a positive integer (got {chips!r})")
    if chips > ADDRESSABLE_CHIPS:
        raise WiringError(f"{path}: chips: {chips} is above the {ADDRESSABLE_CHIPS} that the 9-bit chip_id addresses")
    links = t.get("links")
    if not isinstance(links, list):
        raise WiringError(f"{path}: links: missing or not a list of {{\"a\": [chip, link], \"b\": [chip, link]}} entries")
    peers = {c: {} for c in range(chips)}          # chip -> {peer chip: number of links}
    for i, entry in enumerate(links):
        if not isinstance(entry, dict) or "a" not in entry or "b" not in entry:
            raise WiringError(f"{path}: links[{i}]: must be an object with the ends a and b")
        a, b = _end(path, i, "a", entry["a"], chips), _end(path, i, "b", entry["b"], chips)
        if a[0] != b[0]:
            peers[a[0]][b[0]] = peers[a[0]].get(b[0], 0) + 1
            peers[b[0]][a[0]] = peers[b[0]].get(a[0], 0) + 1
    name = os.path.basename(path)
    return nodes_of(name, chips, peers)


def nodes_of(name, chips, peers):
    """The nodes of a wiring: a chip's node is the largest group of chips fully connected to it that the greedy walk over its peers finds, in
    chip order. A wiring is priced when the nodes cover every chip, all nodes but the last have the same size and every pair of full nodes
    is joined; otherwise it is refused with the reason."""
    node_of, nodes = {}, []
    for c in range(chips):
        if c in node_of:
            continue
        group = [c]
        for p in sorted(peers[c]):
            if p not in node_of and all(p in peers[g] for g in group):
                group.append(p)
        for g in group:
            node_of[g] = len(nodes)
        nodes.append(group)
    sizes = [len(g) for g in nodes]
    cpn = max(sizes)
    refuse = (f"{name}: this wiring is not nodes of fully connected chips joined by global links, which is what the collectives are priced on "
              "(%s); `--topology flat` prices a direct link between every pair of chips, `--wiring-params FILE` a parametric wiring")
    if cpn < 2 and chips > 1:
        raise WiringError(refuse % "no two chips of it are fully connected with a third")
    if any(s != cpn for s in sizes[:-1]) or sizes[-1] > cpn:
        raise WiringError(refuse % f"the node sizes are {sizes}, not equal but for the last")
    for g in nodes:
        for a in g:
            for b in g:
                if a != b and b not in peers[a]:
                    raise WiringError(refuse % f"chips {a} and {b} of one node have no link")
    between, ports = {}, {c: 0 for c in range(chips)}
    for a in range(chips):
        for b, n in peers[a].items():
            if node_of[a] != node_of[b]:
                ports[a] += n
                if a < b:
                    key = (node_of[a], node_of[b])
                    between[key] = between.get(key, 0) + n
    full = len(nodes) - (1 if sizes[-1] < cpn else 0)
    pairs = [(i, j) for i in range(full) for j in range(i + 1, full)]
    missing = [pr for pr in pairs if pr not in between]
    if missing:
        raise WiringError(refuse % f"nodes {missing[0][0]} and {missing[0][1]} have no link between them")
    per_pair = min((between[pr] for pr in pairs), default=1)
    global_ports = max(ports.values()) or GLOBAL_PORTS_PER_CHIP
    out = {"name": name, "kind": "nodes", "hypothetical": False, "chips": chips, "nodes": len(nodes), "chips_per_node": cpn,
           "local_links": cpn - 1, "global_links_per_node_pair": max(1, per_pair), "global_ports_per_chip": global_ports}
    out["describe"] = (f"simulator wiring {name}: {chips} chips in {len(nodes)} node(s) of {cpn} fully connected chips, "
                       f"{out['global_links_per_node_pair']} global link(s) per node pair, up to {global_ports} global ports per chip")
    return out


def relay_cycles(cfg):
    """A receive-then-send turnaround on a chip: d_func(c2c) + c2c_tx_gap - c2c_rx_gap (34 on ref, test/c2c/relay_directed.py)."""
    return (dp.lookup(cfg, "timing.c2c.d_func") + dp.lookup(cfg, "layout.c2c_tx_gap") - dp.lookup(cfg, "layout.c2c_rx_gap"))


def _vec(cfg, nbytes):
    return vectors(nbytes, links_of(cfg)["vlanes"])


def _local_best(kind, cfg, chips, nbytes, lph, peers):
    f = allreduce_cycles if kind == "allreduce" else allgather_cycles
    return best(f(cfg, chips, nbytes, lph, direct_peers=peers))


def _owners(w, nodes):
    """Chips of a node that own its global links: (nodes - 1) x global_links_per_node_pair links over `global_ports_per_chip` ports."""
    return max(1, -(-((nodes - 1) * w["global_links_per_node_pair"]) // w["global_ports_per_chip"]))


def allreduce_on(cfg, w, group, nbytes, links_per_hop=1):
    """(algorithm, cycles, phases) of an all-reduce over `group` chips (consecutive chips, from chip 0) of the wiring `w`."""
    if group == 1:
        return "none", 0, []
    if w["kind"] == "flat":
        name, c = best(allreduce_cycles(cfg, group, nbytes, links_per_hop))
        return name, c, [{"phase": f"{name} over direct links", "cycles": c}]
    k, relay = links_of(cfg), relay_cycles(cfg)
    cpn, peers = w["chips_per_node"], min(w["local_links"], k["links"])
    lph, whole = links_per_hop, _vec(cfg, nbytes)
    local = min(group, cpn)
    n1, c1 = _local_best("allreduce", cfg, local, nbytes, lph, peers)
    phases = [{"phase": "node-local all-reduce", "cycles": c1, "detail": f"{n1} over {local} chips"}]
    nodes = -(-group // cpn)
    if nodes == 1:
        return n1, c1, phases
    owners = min(_owners(w, nodes), cpn)
    g = w["global_links_per_node_pair"]
    phases.append({"phase": "global links", "cycles": relay + k["latency"] + -(-whole // (g * lph)) * k["cpv"],
                   "detail": f"node sum over {g} global link(s) per node pair, {nodes - 1} peers, one relay"})
    if owners > 1:
        n3, c3 = _local_best("allreduce", cfg, owners, nbytes, lph, peers)
        phases.append({"phase": "owner chips exchange", "cycles": relay + c3, "detail": f"{n3} over {owners} owner chips, one relay"})
    if cpn > owners:
        serve = -(-(cpn - owners) // owners)
        phases.append({"phase": "owner chips serve the node", "cycles": relay + k["latency"] + -(-whole // lph) * k["cpv"],
                       "detail": f"each owner sends the sum to {serve} chip(s) on direct local links, one relay"})
    return "hierarchical", sum(p["cycles"] for p in phases), phases


def allgather_on(cfg, w, group, shard_bytes, links_per_hop=1):
    """(algorithm, cycles, phases) of an all-gather of `shard_bytes` per chip over `group` chips of the wiring `w`."""
    if group == 1:
        return "none", 0, []
    if w["kind"] == "flat":
        name, c = best(allgather_cycles(cfg, group, shard_bytes, links_per_hop))
        return name, c, [{"phase": f"{name} over direct links", "cycles": c}]
    k, relay = links_of(cfg), relay_cycles(cfg)
    cpn, peers = w["chips_per_node"], min(w["local_links"], k["links"])
    lph = links_per_hop
    local = min(group, cpn)
    n1, c1 = _local_best("allgather", cfg, local, shard_bytes, lph, peers)
    phases = [{"phase": "node-local all-gather", "cycles": c1, "detail": f"{n1} over {local} chips"}]
    nodes = -(-group // cpn)
    if nodes == 1:
        return n1, c1, phases
    owners = min(_owners(w, nodes), cpn)
    g = w["global_links_per_node_pair"]
    node_bytes = local * shard_bytes
    phases.append({"phase": "global links", "cycles": relay + k["latency"] + -(-_vec(cfg, node_bytes) // (g * lph)) * k["cpv"],
                   "detail": f"the node's data ({node_bytes} bytes) over {g} global link(s) per node pair, one relay"})
    piece = -(-(nodes - 1) * node_bytes // owners)
    phases.append({"phase": "owner chips serve the node", "cycles": relay + k["latency"] + -(-_vec(cfg, piece) // lph) * k["cpv"],
                   "detail": f"each owner sends its {piece} bytes to every other chip of the node, one relay"})
    return "hierarchical", sum(p["cycles"] for p in phases), phases


def hop_cycles_on(cfg, w, nbytes, crossing, links_per_hop=1):
    """One pipeline hop of `nbytes`: a link, or across a node boundary the local hop to the owner chip, the global hop and the local hop
    from the owner chip on the far side, with two relays (the worst case: neither end owns the global link)."""
    k = links_of(cfg)
    v = -(-_vec(cfg, nbytes) // links_per_hop)
    if not crossing or w["kind"] == "flat":
        return k["latency"] + v * k["cpv"]
    return 3 * k["latency"] + 2 * relay_cycles(cfg) + v * k["cpv"]


def best(options):
    name = min(options, key=options.get)
    return name, options[name]


class Shapes:
    """The sizes of a model, derived once."""

    def __init__(self, m, ctx_attn):
        self.m = m
        head_dim = m["hidden"] // m["heads"]
        self.kv_dim = m["kv_heads"] * head_dim
        h = m["hidden"]
        self.layer_params = 2 * h * h + 2 * h * self.kv_dim + m["ffn_matrices"] * h * m["ffn"]
        self.head_params = m["vocab"] * h
        self.layer_w = self.layer_params * m["weight_bytes"]
        self.head_w = self.head_params * m["weight_bytes"]
        self.kv_layer_full = m["context"] * 2 * self.kv_dim * m["kv_bytes"]
        self.kv_layer_attn = ctx_attn * 2 * self.kv_dim * m["kv_bytes"]
        self.attn_macs_token = 2 * ctx_attn * h
        self.act_bytes = m.get("act_bytes", DEFAULT_ACT_BYTES)


def pipelined(entry, stages, batch, bytes_per_chip, cap):
    """The rate with the pipeline full needs one batch per stage in flight: record how many sequences that is and the cache they need on a
    chip, and give no rate when that does not fit in SRAM."""
    entry["pipelined_sequences"] = stages * batch
    entry["pipelined_bytes_per_chip"] = math.ceil(bytes_per_chip)
    entry["pipelined_fits"] = bytes_per_chip <= cap
    if not entry["pipelined_fits"]:
        entry["tokens_per_s_pipelined"] = None


def compute_cycles(shapes, rt, weight_bytes, kv_bytes, macs, kind):
    """(install cycles, MAC cycles, cycles, bound) of one chip's share of a layer or of the head; `kind` is int8, fp8 or f16."""
    install = (weight_bytes + kv_bytes) / rt["install_bytes"]
    mac = macs / rt[kind + "_macs"]
    return math.ceil(install), math.ceil(mac), math.ceil(max(install, mac)), "install" if install >= mac else "MAC"


def split_layers(layers, stages):
    base, extra = divmod(layers, stages)
    return [base + 1 if i < extra else base for i in range(stages)]


def crossings(w, t, stages):
    """Hops between consecutive stages (stage j on chips [j t, (j + 1) t)) that cross a node boundary."""
    if w["kind"] == "flat":
        return 0
    return sum(1 for j in range(stages - 1) if ((j + 1) * t) % w["chips_per_node"] == 0)


def warnings_for(w, chips, cfg=None, clock_mhz=dp.DEFAULT_CLOCK_MHZ):
    out = []
    if chips > ADDRESSABLE_CHIPS:
        out.append(f"{chips} chips is beyond the {ADDRESSABLE_CHIPS} that the 9-bit chip_id of Send and Transmit addresses (topology.md "
                   f"section 7): the architecture cannot address this system, so every figure prices a system that does not exist")
    elif chips > RUN_CHIPS and w["name"] == "modelled":
        out.append(f"{chips} chips is beyond the {RUN_CHIPS} of `tsp_sim --chips` (addressable up to {ADDRESSABLE_CHIPS}): a run of this size "
                   f"needs a topology file (`--topology`, tools/gen_topology.py), and the figures price the node wiring extended to "
                   f"{-(-chips // NODE_CHIPS)} nodes")
    nodes = -(-chips // NODE_CHIPS)
    if w["name"] == "modelled" and nodes > MAX_NODES:
        out.append(f"{nodes} nodes of {NODE_CHIPS} chips need more global links per node than its {NODE_CHIPS * GLOBAL_PORTS_PER_CHIP} global "
                   f"ports give ({MAX_NODES} nodes at most are joined all-to-all, one link per pair): the global wiring of the figures does not exist")
    if w["name"] == "modelled" and chips > NODE_CHIPS and chips % NODE_CHIPS:
        out.append(f"{chips} chips: the last node holds {chips % NODE_CHIPS} chips; the collectives are priced as if every node held {NODE_CHIPS}")
    if cfg is not None:
        links = dp.lookup(cfg, "c2c.links")
        need = (NODE_CHIPS - 1 + GLOBAL_PORTS_PER_CHIP) if chips > NODE_CHIPS else max(chips - 1, 1)
        if w["name"] == "modelled" and links < need:
            out.append(f"c2c.links is {links}: the wiring of {chips} chips uses links up to {need - 1}, so it needs at least {need} links; "
                       "the links beyond the profile's do not exist")
        out += [x.replace("note: ", "", 1) + "; every collective figure inherits it" for x in dp.rate_notes(dp.derive(cfg, clock_mhz))]
    return out


def map_model(m, cfg, chips, clock_mhz=dp.DEFAULT_CLOCK_MHZ, batch=1, phase="decode", prompt_len=None, links_per_hop=1,
              sram_fraction=1.0, layer_overhead_cycles=0, topology=None):
    check_model(m)
    extra = []
    cap = dp.C2C_TX_QUEUES * dp.lookup(cfg, "c2c.cycles_per_vector")  # links a chip can keep busy: 4 vectors injected per cycle
    if links_per_hop > cap:
        extra.append(f"--links-per-hop {links_per_hop} is above the {cap} links that 4 transmit queues keep busy (4 vectors per cycle, "
                     f"{dp.lookup(cfg, 'c2c.cycles_per_vector')} cycles per vector per link): priced as {cap}")
        links_per_hop = cap
    w = topology or modelled_wiring()
    if phase == "prefill" and not prompt_len:
        raise ModelError("--prompt-len: required for --phase prefill")
    rt = rates(cfg)
    k = links_of(cfg)
    f16 = weight_type(m)  # the kind of MAC rate the weights use
    if rt[f16 + "_macs"] <= 0:
        raise ModelError(f"weight_type {f16}: this profile has no {f16} rate (mxm.planes_per_hemisphere {cfg['mxm']['planes_per_hemisphere']}: "
                         "no plane can hold f16, an f16 pair is the two planes of a hemisphere, or the extension of the type is not declared); use INT8 or FP8 weights or a profile with two planes and the extension")
    toks_seq = prompt_len if phase == "prefill" else 1
    ctx_attn = prompt_len if phase == "prefill" else m["context"]
    s = Shapes(m, ctx_attn)
    tokens = batch * toks_seq
    cap = int(rt["sram_bytes"] * sram_fraction)
    layers = m["layers"]
    overhead = layer_overhead_cycles
    freq = clock_mhz * 1e6
    kv_total_seq = layers * s.kv_layer_full
    weights_total = layers * s.layer_w + s.head_w
    footprint = {"weights_bytes": weights_total, "kv_bytes_per_sequence": kv_total_seq, "sequences": batch,
                 "sram_bytes_per_chip": cap,
                 "min_chips_total": -(-(weights_total + batch * kv_total_seq) // cap), "label": LABEL}
    act = tokens * m["hidden"] * s.act_bytes

    def hop_transfer(nbytes):
        return -(-vectors(nbytes, k["vlanes"]) // links_per_hop) * k["cpv"]

    # pipeline
    stages = min(chips, layers)
    sizes = split_layers(layers, stages)
    ins, mac, lc, bound = compute_cycles(
        s, rt, s.layer_w, batch * s.kv_layer_attn, tokens * (s.layer_params + s.attn_macs_token), f16)
    layer_total = lc + overhead
    _, _, head_c, _ = compute_cycles(s, rt, s.head_w, 0, batch * s.head_params, f16)  # logits of the last token of each sequence
    def pipe_bytes(seqs):
        return max(n * (s.layer_w + seqs * s.kv_layer_full) + (s.head_w if i == stages - 1 else 0) for i, n in enumerate(sizes))

    stage_bytes = [n * (s.layer_w + batch * s.kv_layer_full) + (s.head_w if i == stages - 1 else 0) for i, n in enumerate(sizes)]
    stage_cycles = [n * layer_total + (head_c if i == stages - 1 else 0) for i, n in enumerate(sizes)]
    hop_in = hop_cycles_on(cfg, w, act, False, links_per_hop) if stages > 1 else 0
    hop_x = hop_cycles_on(cfg, w, act, True, links_per_hop) if stages > 1 else 0
    cross = crossings(w, 1, stages)
    hops_total = (stages - 1 - cross) * hop_in + cross * hop_x
    step = sum(stage_cycles) + hops_total
    pipe = {
        "chips_used": stages, "layers_per_stage": sizes, "bytes_per_chip": max(stage_bytes), "fits": max(stage_bytes) <= cap,
        "install_cycles_layer": ins, "mac_cycles_layer": mac, "compute_bound": bound, "layer_cycles": layer_total,
        "head_cycles": head_c, "hop_cycles": hop_in, "hop_cycles_across_node": hop_x, "hops_across_node": cross,
        "c2c_bytes_per_token": (stages - 1) * m["hidden"] * s.act_bytes,
        "c2c_cycles_per_step": hops_total, "step_cycles": step, "tokens_per_step": tokens,
        "tokens_per_s": tokens * freq / step,
        "tokens_per_s_pipelined": tokens * freq / max(max(stage_cycles), hop_transfer(act) if stages > 1 else 0),
        "label": LABEL,
    }
    pipelined(pipe, stages, batch, pipe_bytes(stages * batch), cap)
    parts = {("weight and KV install" if bound == "install" else "MAC throughput"): layers * lc,
             "C2C hop latency and transfer": pipe["c2c_cycles_per_step"], "latency floor (layer overhead)": layers * overhead}
    pipe["binding"] = max(parts, key=parts.get)

    def tensor_group(t):
        """The share of one layer and of the head on a tensor group of t chips, and its collectives on the wiring."""
        shard_kv, shard_heads = min(t, m["kv_heads"]), min(t, m["heads"])
        ins_t, mac_t, lc_t, bound_t = compute_cycles(
            s, rt, s.layer_w / t, batch * s.kv_layer_attn / shard_kv,
            tokens * (s.layer_params / t + s.attn_macs_token / shard_heads), f16)
        ar_name, ar_c, ar_phases = allreduce_on(cfg, w, t, act, links_per_hop)
        _, _, head_t, _ = compute_cycles(s, rt, s.head_w / t, 0, batch * s.head_params / t, f16)
        ag_name, ag_c, ag_phases = allgather_on(cfg, w, t, batch * m["vocab"] * s.act_bytes // t, links_per_hop)
        return {"shard_kv": shard_kv, "install": ins_t, "mac": mac_t, "layer_compute": lc_t, "bound": bound_t,
                "ar_name": ar_name, "ar_cycles": ar_c, "ar_phases": ar_phases, "ag_name": ag_name, "ag_cycles": ag_c,
                "ag_phases": ag_phases, "head": head_t, "layer": lc_t + 2 * ar_c + overhead}

    # tensor parallel
    n = chips
    g = tensor_group(n)
    step_t = layers * g["layer"] + g["head"] + g["ag_cycles"]
    bytes_t = layers * (s.layer_w / n + batch * s.kv_layer_full / g["shard_kv"]) + s.head_w / n
    tens = {
        "chips_used": n, "bytes_per_chip": math.ceil(bytes_t), "fits": bytes_t <= cap,
        "install_cycles_layer": g["install"], "mac_cycles_layer": g["mac"], "compute_bound": g["bound"], "layer_cycles": g["layer"],
        "collective_algorithm": g["ar_name"], "collective_cycles_per_allreduce": g["ar_cycles"], "collective_phases": g["ar_phases"],
        "allreduces_per_layer": 2,
        "allgather_algorithm": g["ag_name"], "allgather_cycles": g["ag_cycles"], "allgather_phases": g["ag_phases"],
        "head_cycles": g["head"],
        "c2c_bytes_per_token": 0 if n == 1 else layers * 2 * m["hidden"] * s.act_bytes + m["vocab"] * s.act_bytes,
        "c2c_cycles_per_step": layers * 2 * g["ar_cycles"] + g["ag_cycles"], "step_cycles": step_t, "tokens_per_step": tokens,
        "tokens_per_s": tokens * freq / step_t, "tokens_per_s_pipelined": tokens * freq / step_t, "label": LABEL,
    }
    parts_t = {("weight and KV install" if g["bound"] == "install" else "MAC throughput"): layers * g["layer_compute"],
               "C2C collective latency": tens["c2c_cycles_per_step"], "latency floor (layer overhead)": layers * overhead}
    tens["binding"] = max(parts_t, key=parts_t.get)

    def hybrid(t):
        """Pipeline stages that are tensor groups of t chips: stage j on chips [j t, (j + 1) t)."""
        p = min(layers, chips // t)
        g = tensor_group(t)
        sizes = split_layers(layers, p)
        stage_cycles = [nl * g["layer"] + (g["head"] + g["ag_cycles"] if i == p - 1 else 0) for i, nl in enumerate(sizes)]
        stage_bytes = [nl * (s.layer_w / t + batch * s.kv_layer_full / g["shard_kv"]) + (s.head_w / t if i == p - 1 else 0)
                       for i, nl in enumerate(sizes)]
        # one copy of the activations crosses between two stages; the first chip of the next group serves the others of its group
        serve = 0 if t == 1 else relay_cycles(cfg) + k["latency"] + -(-vectors(act, k["vlanes"]) // links_per_hop) * k["cpv"]
        h_in = hop_cycles_on(cfg, w, act, False, links_per_hop) + serve
        h_x = hop_cycles_on(cfg, w, act, True, links_per_hop) + serve
        cross = crossings(w, t, p)
        hops_total = (p - 1 - cross) * h_in + cross * h_x
        coll = layers * 2 * g["ar_cycles"] + g["ag_cycles"]
        step = sum(stage_cycles) + hops_total
        bytes_chip = max(stage_bytes)
        parts = {("weight and KV install" if g["bound"] == "install" else "MAC throughput"): layers * g["layer_compute"],
                 "C2C collective latency": coll, "C2C hop latency and transfer": hops_total,
                 "latency floor (layer overhead)": layers * overhead}
        full_bytes = max(nl * (s.layer_w / t + p * batch * s.kv_layer_full / g["shard_kv"]) + (s.head_w / t if i == p - 1 else 0)
                         for i, nl in enumerate(sizes))
        entry = {
            "tensor_group": t, "stages": p, "chips_used": p * t, "layers_per_stage": sizes, "bytes_per_chip": math.ceil(bytes_chip),
            "fits": bytes_chip <= cap, "install_cycles_layer": g["install"], "mac_cycles_layer": g["mac"],
            "compute_bound": g["bound"], "layer_cycles": g["layer"], "collective_algorithm": g["ar_name"],
            "collective_cycles_per_allreduce": g["ar_cycles"], "collective_phases": g["ar_phases"], "allreduces_per_layer": 2,
            "allgather_algorithm": g["ag_name"], "allgather_cycles": g["ag_cycles"], "head_cycles": g["head"],
            "hop_cycles": h_in, "hop_cycles_across_node": h_x, "hops_across_node": cross,
            "c2c_bytes_per_token": (0 if t == 1 else layers * 2 * m["hidden"] * s.act_bytes) + (p - 1) * m["hidden"] * s.act_bytes,
            "c2c_cycles_per_step": coll + hops_total, "step_cycles": step, "tokens_per_step": tokens,
            "tokens_per_s": tokens * freq / step,
            "tokens_per_s_pipelined": tokens * freq / max(max(stage_cycles), hop_transfer(act)),
            "binding": max(parts, key=parts.get), "label": LABEL,
        }
        pipelined(entry, p, batch, full_bytes, cap)
        return entry

    if w["kind"] == "flat":
        groups = [t for t in (2 ** i for i in range(1, 32)) if t < chips and chips // t >= 2]
    else:
        groups = [t for t in range(2, w["chips_per_node"] + 1) if w["chips_per_node"] % t == 0 and chips // t >= 2]
    candidates = [hybrid(t) for t in groups if min(layers, chips // t) >= 2]
    strategies = {"pipeline": pipe, "tensor": tens}
    if candidates:
        fitting = [c for c in candidates if c["fits"]]
        strategies["hybrid"] = (max(fitting, key=lambda c: (c["tokens_per_s"], -c["tensor_group"])) if fitting
                                else min(candidates, key=lambda c: (c["bytes_per_chip"], c["tensor_group"])))

    fit = {name: st for name, st in strategies.items() if st["fits"]}
    order = {"pipeline": 0, "tensor": 1, "hybrid": 2}
    choice = max(fit, key=lambda name: (fit[name]["tokens_per_s"], -order[name])) if fit else None
    return {"label": LABEL, "model": m.get("name", "(unnamed)"), "chips": chips, "clock_mhz": clock_mhz, "phase": phase,
            "batch": batch, "prompt_len": prompt_len, "links_per_hop": links_per_hop, "layer_overhead_cycles": overhead,
            "wiring": {"name": w["name"], "hypothetical": w["hypothetical"], "description": w["describe"],
                       "relay_cycles": relay_cycles(cfg)},
            "warnings": extra + warnings_for(w, chips, cfg, clock_mhz),
            "footprint": footprint, "strategies": strategies, "best": choice,
            "notes": ["every figure is an estimate; the clock is an assumption (derive_peaks.py)",
                      "the MXM terms are throughput bounds: a program that is latency bound (the measured reference is) takes longer"]}


def validate_measured(cfg, clock_mhz=dp.DEFAULT_CLOCK_MHZ):
    """The model against the measured serial layer and decode step of the showcase model (one chip, decode, batch 1). The showcase model has
    f16 weights, so a profile with no f16 rate has no such check: an empty list."""
    if rates(cfg)["f16_macs"] <= 0:
        return []
    r = map_model(TINYSTORIES, cfg, chips=1, clock_mhz=clock_mhz)["strategies"]["pipeline"]
    model = {"serial layer": r["layer_cycles"], "8-layer decode step": r["step_cycles"]}
    rows = []
    for name, measured, what in MEASURED:
        cycles = model[name]
        rows.append({"name": name, "measured_cycles": measured, "model_cycles": cycles, "what": what,
                     "error_percent": (cycles - measured) / measured * 100, "overhead_cycles": measured - cycles,
                     "label": "model: derived; measured: measured (this project)"})
    return rows


def pipeline_full_text(s):
    need = f"{s['pipelined_sequences']:,} sequences in flight, {s['pipelined_bytes_per_chip'] / 2**20:,.1f} MiB per chip"
    if s["pipelined_fits"]:
        return f" ({s['tokens_per_s_pipelined']:,.1f} tokens/s with the pipeline full: {need})"
    return f" (the pipeline full would need {need}, DOES NOT FIT: no rate given for it)"


def render(r, validation):
    f = r["footprint"]
    wiring = r["wiring"]
    tag = f" [HYPOTHETICAL wiring '{wiring['name']}']" if wiring["hypothetical"] else ""
    lines = [f"model_map: {r['model']} on {r['chips']} chip(s), {r['phase']}"
             + (f" of {r['prompt_len']} tokens" if r["phase"] == "prefill" else "") + f", batch {r['batch']}, clock {r['clock_mhz']:g} MHz",
             f"every figure is an {LABEL}; the clock is an assumption; the MXM terms are throughput bounds",
             f"wiring: {wiring['description']}; a relay costs {wiring['relay_cycles']} cycles", ""]
    lines += [f"WARNING: {x}" for x in r["warnings"]] + ([""] if r["warnings"] else [])
    lines += [f"footprint: weights {f['weights_bytes'] / 2**20:,.1f} MiB, KV {f['kv_bytes_per_sequence'] / 2**20:,.1f} MiB per sequence "
              f"x {f['sequences']}, SRAM {f['sram_bytes_per_chip'] / 2**20:,.1f} MiB per chip; at least {f['min_chips_total']} chip(s) "
              "hold it with a perfect split"]
    for name, s in r["strategies"].items():
        shape = (f", {s['stages']} stages of tensor groups of {s['tensor_group']}" if name == "hybrid" else "")
        lines += ["", f"{name} ({s['label']}): {s['chips_used']} chip(s) used{shape}, {s['bytes_per_chip'] / 2**20:,.1f} MiB per chip, "
                  + ("fits" if s["fits"] else "DOES NOT FIT in SRAM (the rates below are hypothetical)"),
                  f"  layer: install {s['install_cycles_layer']:,} cycles, MAC {s['mac_cycles_layer']:,} cycles -> {s['layer_cycles']:,} "
                  f"({s['compute_bound']}-bound, with the layer overhead)",
                  f"  C2C: {s['c2c_bytes_per_token']:,} bytes per token, {s['c2c_cycles_per_step']:,} cycles per step"
                  + (f" ({s['collective_algorithm']} all-reduce {s['collective_cycles_per_allreduce']:,} cycles)"
                     if s.get("collective_algorithm", "none") != "none" else "")
                  + (f" (hop {s['hop_cycles']:,} cycles, {s['hop_cycles_across_node']:,} across a node boundary, "
                     f"{s['hops_across_node']} such hop(s))" if "hop_cycles" in s else ""),
                  f"  step: {s['step_cycles']:,} cycles for {s['tokens_per_step']:,} token(s): {s['tokens_per_s']:,.1f} tokens/s"
                  + (pipeline_full_text(s) if "hop_cycles" in s else ""),
                  f"  binding bound: {s['binding']}"]
        for ph in s.get("collective_phases", []):
            if len(s["collective_phases"]) > 1:
                lines.append(f"  all-reduce phase: {ph['phase']}: {ph['cycles']:,} cycles ({ph['detail']})")
    lines += ["", f"best split that fits: {r['best'] or 'none (no split fits in SRAM; more chips or a smaller model needed)'}"]
    if tag:
        lines = [x + tag if x.strip() else x for x in lines]
    lines += ["", "check against the measured showcase model (one chip, decode, batch 1; the model is a throughput bound):"]
    if not validation:
        lines.append("  not run: the showcase model has f16 weights and this profile has no f16 rate (one plane per hemisphere)")
    for v in validation:
        lines.append(f"  {v['name']}: measured {v['measured_cycles']:,} cycles, model {v['model_cycles']:,} cycles, "
                     f"error {v['error_percent']:+.2f} percent (latency overhead {v['overhead_cycles']:,} cycles)")
    lines.append("  the measured program keeps the matrix unit busy about 2 percent of the cycles: it is latency bound, so the "
                 "bound is far below it. --layer-overhead-cycles adds a measured floor (estimate for any other shape).")
    return "\n".join(lines)


def positive_int(flag):
    def parse(text):
        try:
            v = int(text)
        except ValueError:
            v = 0
        if v <= 0:
            raise argparse.ArgumentTypeError(f"{flag} must be a positive integer (got {text!r})")
        return v
    return parse


def fraction(text):
    try:
        v = float(text)
    except ValueError:
        v = float("nan")
    if not 0 < v <= 1:
        raise argparse.ArgumentTypeError(f"--sram-fraction must be in (0, 1] (got {text!r})")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("profile")
    ap.add_argument("--chips", type=positive_int("--chips"),
                    help="the number of chips (a run takes 1 to 264, a topology file up to 512); defaults to the chips of --topology FILE")
    ap.add_argument("--clock-mhz", type=dp.positive_clock, default=dp.DEFAULT_CLOCK_MHZ)
    ap.add_argument("--batch", type=positive_int("--batch"), default=1, help="sequences decoded or prefilled together")
    ap.add_argument("--phase", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--prompt-len", type=positive_int("--prompt-len"))
    ap.add_argument("--links-per-hop", type=positive_int("--links-per-hop"), default=1)
    ap.add_argument("--sram-fraction", type=fraction, default=1.0, help="share of the SRAM usable for weights and KV (default 1)")
    ap.add_argument("--layer-overhead-cycles", type=int, default=0,
                    help="latency floor per layer, e.g. the measured TinyStories overhead (estimate for another shape)")
    ap.add_argument("--topology", default=None, metavar="flat|FILE",
                    help="the wiring: `flat` (a direct link between every pair of chips, HYPOTHETICAL) or a topology file as `tsp_sim --topology` "
                         "reads it (tools/gen_topology.py writes them); the nodes and global links are derived from its link list")
    ap.add_argument("--wiring-params", default=None, metavar="FILE",
                    help="a parametric wiring no harness builds, HYPOTHETICAL in every report line: a JSON file with name, chips_per_node, "
                         "global_links_per_node_pair, global_ports_per_chip")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        try:
            with open(args.model) as fh:
                m = json.load(fh)
        except OSError as e:
            raise ModelError(f"{args.model}: cannot read the model file ({e.strerror})")
        except ValueError as e:
            raise ModelError(f"{args.model}: not valid JSON ({e})")
        try:
            cfg = dp.load_profile(args.profile)
        except dp.ProfileError as e:
            raise ModelError(str(e))
        if args.layer_overhead_cycles < 0:
            raise ModelError("--layer-overhead-cycles: must not be negative")
        if args.links_per_hop > dp.lookup(cfg, "c2c.links"):
            raise ModelError(f"--links-per-hop: the profile has {cfg['c2c']['links']} links")
        if args.topology and args.wiring_params:
            raise ModelError("--topology and --wiring-params are alternatives: give one wiring")
        topology = (load_topology(args.topology) if args.topology
                    else load_wiring_params(args.wiring_params) if args.wiring_params else None)
        chips = args.chips
        if chips is None:
            if topology is None or "chips" not in topology:
                raise ModelError("--chips: required (only a topology file gives its own chip count)")
            chips = topology["chips"]
        elif topology is not None and "chips" in topology and topology["chips"] != chips:
            raise ModelError(f"--chips {chips}: the topology file {topology['name']} has {topology['chips']} chips (leave --chips out to use them)")
        r = map_model(m, cfg, chips, args.clock_mhz, args.batch, args.phase, args.prompt_len, args.links_per_hop,
                      args.sram_fraction, args.layer_overhead_cycles, topology)
        for warning in r["warnings"]:
            print(f"model_map: warning: {warning}", file=sys.stderr)
        validation = validate_measured(cfg, args.clock_mhz)
    except (ModelError, WiringError, dp.ProfileError) as e:
        print(f"model_map: {e}", file=sys.stderr)
        return 2
    if args.json:
        r["validation"] = validation
        print(json.dumps(r, indent=2))
    else:
        print(render(r, validation))
    return 0


if __name__ == "__main__":
    dp.run(main, "model_map")
