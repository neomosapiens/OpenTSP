#!/usr/bin/env python3
"""The gate driver: TinyStories generation on the ISS, the host loop around the decode step.

This is a gate (bit-exact reference/rounded PyTorch reference/FP32 reference checks of every step), not the showcase: the host keeps the K/V rows and rewrites them into the chip every step. The
showcase is `just showcase` (real vLLM, the KV cache in the chip's SRAM; docs/demos.md#single-chip-tinystories).

The step program (`emit_step.py`: eight layers, the final LayerNorm) is built and assembled once. Every token, prompt
(prefill) or generated, is one run of the same object on `tsp-iss` with a new weight package: the layer weights, the
constants and the instruction text blocks are the package's static tensors; the embedded token `x0 = wte[id] +
wpe[pos]`, the mask and every layer's K and V slots are the tensors the host rewrites per step from the K and V rows
the chip wrote at the earlier positions . The host reads the final hidden state and the new K and V rows of
every layer from the dump, computes the output head `logits = h . wte^T` and the greedy token as bit-exact reference does (the f64 sum
rounded to f32; the chip's head window is not used), and loops until N tokens or the end token.

    driver.py [--prompt TEXT | --ids 1,2,3] --tokens N [--weights synthetic|real] [--golden exact,rounded,fp32] [--check]
              [--interface native|compat] [--json FILE] [--sim-step K] [--keep DIR]

`--check` is `--golden exact`: every step's dumped words (each layer's output, K and V rows, the hidden state, the head
window's logits) and the host logits equal bit-exact reference's own step (`reference_model.decode_step`, its cache and its embedding). `rounded`:
per step, on the K/V rows the chip wrote, the final hidden state and the new K/V rows against PyTorch with f16-rounded
weights and matmul inputs (cosine >= 0.99999, max difference <= `driver_gates.ERR_ROUNDED` max |h|), and the free-running sequence against a
calibrated bound; `fp32`: against the original fp32 model (real weights only): per-step hidden cosine >= 0.999, the
greedy token equal on the shared prefix unless the top-1/top-2 margin is below 1e-3 .
`--sim-step K` also runs `tsp_sim` ($TSP_SIM) on step K's input and requires its dumped words to equal the ISS's (a minute).

Machine-readable result lines: `IDS` (prompt and generated ids), `NEW` (generated ids), `TEXT`, `PROGRAM`, `GATE`.
Dependencies: numpy; `tokenizers` (text prompt and decoded text) and PyTorch (rounded PyTorch reference, FP32 reference), `transformers` (FP32 reference), the
checkpoint in the Hugging Face cache (real weights) are optional: without one the driver says so, and takes ids.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths
from opentsp_tools.programs import data
from opentsp_tools.programs import needs
from opentsp_tools.programs import numerics as nx
from opentsp_tools import profile as tprofile
from opentsp_tools.programs import ref as tref
from opentsp_tools.programs.kernels import install as k_install
from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import emit_step as es
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import comparison as comparison_ref
from opentsp_vllm.models.tinystories import layout
from opentsp_vllm.models.tinystories import runner_util
from opentsp_vllm.models.tinystories.config import PROMPT, PROMPT_IDS, TINYSTORIES

ROOT = _paths.ROOT
CFG = TINYSTORIES
EOS = 50256                        # GPT-2 end of text
MAX_STEPS = CFG.slots              # positions 0 .. slots - 1: slots - 1 cached rows and the current slot
pad = eb.pad


class DriverError(Exception):
    pass


def digest(b):
    return hashlib.sha256(b).hexdigest()


# -- tokenizer (optional) ---------------------------------------------------------------------------------------
def load_tokenizer(path=None):
    """(tokenizer, None) from the GPT-2 `tokenizer.json` (`path`, or the Hugging Face cache), or (None, the reason)."""
    try:
        from tokenizers import Tokenizer                          # noqa: PLC0415 (optional dependency)
    except ImportError:
        return None, "the `tokenizers` package is not installed (pip install tokenizers into the toolchain venv)"
    cands = [path] if path else sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json")))
    if not cands or not os.path.exists(cands[0]):
        return None, "no GPT-2 tokenizer.json (--tokenizer PATH, or the cached `gpt2` snapshot)"
    return Tokenizer.from_file(cands[0]), None


def prompt_ids(a):
    """(ids, tokenizer or None, note): from --ids, else the GPT-2 tokenizer; the design's prompt has known ids."""
    tok, why = load_tokenizer(a.tokenizer)
    if a.ids:
        return [int(x) for x in a.ids.split(",")], tok, None
    if tok is not None:
        return list(tok.encode(a.prompt).ids), tok, None
    if a.prompt == PROMPT:
        return list(PROMPT_IDS), None, f"tokenizer unavailable ({why}); the design prompt's known ids are used"
    raise DriverError(f"cannot tokenize {a.prompt!r}: {why}; pass the ids with --ids 1,2,3")


# -- the step program and its per-step inputs --------------------------------------------------------------------
class StepProgram:
    """The step assembled once: `ots-asm asm` of the emitter's text, the static tensors written once, and `run` for one
    position."""

    def __init__(self, w, work, interface="native", say=print, profile="ref"):
        runner_util.ensure_binaries()
        self.w, self.work, self.interface = w, work, interface
        t0 = time.perf_counter()
        p = tprofile.load(profile)
        needs.SHOWCASE.check(p)
        self.profile, self.cfg_path = p.asm_name, p.path
        text, case = es.build(p, 0, w=w)
        self.t_emit = time.perf_counter() - t0
        self.text = text
        self.case = case
        self.vlanes = p.vlanes
        self.budget = case["budget"]["cycles"]
        t0 = time.perf_counter()
        self.src = os.path.join(work, "step.ots")
        self.obj = os.path.join(work, "step.elf")
        open(self.src, "w").write(text)
        self._sh([runner_util.ASM, "asm", "--config", self.cfg_path, "-o", self.obj, self.src])
        self.t_asm = time.perf_counter() - t0
        self.text_digest = digest("\n".join(ln for ln in text.split("\n") if not ln.startswith(";")).encode())
        self.obj_digest = digest(open(self.obj, "rb").read())
        if self.text_digest != es.PINNED_TEXT:
            raise DriverError(f"the program text (sha256 {self.text_digest}) is not TS-3's pinned text "
                              f"({es.PINNED_TEXT}): the step is not the one TS-3's gate covers")
        base = es.step_inputs(0, w=w)
        self.head_rows = es.head_window(base)
        self._map_dumps(base)
        dyn = self.dynamic(base.x0, *exact_ref.HostKV(CFG).slots(0))
        for n, b in dyn.items():
            if b != bytes.fromhex(case["weights"][n]["hex"]):
                raise DriverError(f"the host's tensor {n} differs from the emitter's at position 0")
        self.dynamic_names = list(dyn)
        os.makedirs(os.path.join(work, "t"), exist_ok=True)
        self.static = []
        for n, wt in case["weights"].items():
            path = os.path.join(work, "t", n + ".bin")
            if n not in dyn:
                open(path, "wb").write(bytes.fromhex(wt["hex"]))
            self.static.append((n, wt["dtype"], path))
        self.dtypes = {n: wt["dtype"] for n, wt in case["weights"].items()}
        self.pkg = os.path.join(work, "step.safetensors")
        self.last_cycle = None
        say(f"program: {len(text):,} B of text, {len(case['weights'])} tensors ({len(dyn)} rewritten per step), "
            f"emitted in {self.t_emit:.1f} s, assembled in {self.t_asm:.2f} s, budget {self.budget} cycles")

    @staticmethod
    def _sh(cmd, stdin=None, ok=(0,), timeout=900):
        r = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
        if r.returncode not in ok:
            err = r.stderr.decode(errors="replace")[:400]
            raise DriverError(f"{os.path.basename(cmd[0])} exited {r.returncode}: {err}")
        return r

    def _map_dumps(self, base):
        """{name: [(slice, address)] per plane} of every dumped word. The case dumps, per layer, the V row (4 planes),
        the K row (2) and the output x (4), then the final hidden state (4) and the head window's logits; the mapping
        is checked against the emitter's expectation at position 0, plane by plane."""
        keys = [tuple(int(x) for x in k.split(":")) for k in self.case["dump"]]
        want = es.expected_words(base)
        order = []
        for l in range(CFG.layers):
            order += [(f"v{l}", 4), (f"k{l}", 2), (f"x{l + 1}", 4)]
        order.append(("hidden", 4))
        order += [(f"logits{c}", 4) for c in range(base.head_chunks)]
        self.dumps, i = {}, 0
        for name, n in order:
            self.dumps[name] = keys[i:i + n]
            for key, plane in zip(self.dumps[name], want[name]):
                if self.case["expect"]["words"][f"{key[0]}:{key[1]}"] != bytes(plane).hex():
                    raise DriverError(f"dump {name}: word {key} is not the emitter's expectation (mapping broken)")
            i += n
        if i != len(keys):
            raise DriverError(f"the case dumps {len(keys)} words, the mapping covers {i}")

    def dynamic(self, x0, k_slots, v_slots, mask):
        """The tensors the host rewrites per step, by extern name: x0 and the mask (4 planes each), and for every layer
        the K cache (the install layout, one tensor per slice of the east plane) and the V cache (4 planes), as
        `Block.bind` and `Block.run` build them."""
        out = {}
        for pre, vec in (("x0", x0), ("mask", mask)):
            for j, pl in enumerate(data.to_planes(pad(vec))):
                out[f"{pre}p{j}"] = bytes(pl)
        for l in range(CFG.layers):
            for j, words in enumerate(k_install.weight_planes(k_slots[l], self.vlanes)):
                out[f"kc{l}s{layout.EAST_W[j]}"] = b"".join(bytes(x) for x in words)
            planes = [data.to_planes(pad(row)) for row in v_slots[l]]
            for j, s in enumerate(layout.VCACHE_QUAD):
                out[f"vc{l}s{s}"] = b"".join(pl[j] for pl in planes)
        return out

    def check_unchanged(self):
        if digest(open(self.obj, "rb").read()) != self.obj_digest or digest(open(self.src, "rb").read()) != digest(
                self.text.encode()):
            raise DriverError("the program changed between steps")

    def _dump_args(self):
        return [f"{s:x}:{a:x}" for ks in self.dumps.values() for s, a in ks]

    def _package(self, dyn):
        args = []
        for n, dt, path in self.static:
            if n in dyn:
                path = os.path.join(self.work, "t", n + ".bin")
                open(path, "wb").write(dyn[n])
            args.append(f"{n}={dt}:{path}")
        self._sh([runner_util.ASM, "wpack", "--profile", self.profile, "--config", self.cfg_path, "-o", self.pkg, *args])

    def compat_input(self):
        text = self._sh([runner_util.ASM, "objinput", "--config", self.cfg_path, "--weights", self.pkg,
                         self.obj]).stdout.decode()
        return text + "".join(f"D {s:x} {a:x}\n" for s, a in (tuple(k) for ks in self.dumps.values() for k in ks))

    @staticmethod
    def _parse(out):
        e = [ln for ln in out.split("\n") if ln.startswith("E ")]
        if len(e) != 1:
            raise DriverError(f"the ISS printed {len(e)} E lines")
        words = {}
        for ln in out.split("\n"):
            if ln.startswith("W "):
                f = ln.split()
                words[(int(f[1], 16), int(f[2], 16))] = bytes(int(x, 16) for x in f[3:])
        return e[0], words

    def step(self, x0, k, v, mask):
        """Run one position: ({name: [plane bytes]}, cycles, {load, run, read} seconds)."""
        t0 = time.perf_counter()
        self._package(self.dynamic(x0, k, v, mask))
        self.check_unchanged()
        if self.interface == "compat":
            stdin = self.compat_input().encode()
            cmd = [runner_util.ISS, self.cfg_path, "--max-cycles", str(self.budget)]
        else:
            stats = os.path.join(self.work, "stats.json")
            cmd = [runner_util.ISS, "run", "--config", self.cfg_path, "--weights", self.pkg, f"--stats={stats}",
                   "--max-cycles", str(self.budget)]
            for d in self._dump_args():
                cmd += ["--dump", d]
            cmd.append(self.obj)
            stdin = None
        t1 = time.perf_counter()
        r = self._sh(cmd, stdin=stdin, ok=(0, 3))
        t2 = time.perf_counter()
        if r.returncode == 3:
            raise DriverError(f"--max-cycles {self.budget} exceeded")
        e, words = self._parse(r.stdout.decode())
        if e != "E - -1 0":
            raise DriverError(f"the ISS reported {e!r} (a diagnostic): the step is not valid")
        cycles = json.load(open(stats))["last_cycle"] if self.interface == "native" else None
        if cycles is not None:
            self.last_cycle = cycles
        out = {n: [words[k][:self.vlanes] for k in ks] for n, ks in self.dumps.items()}
        self.last_words = words
        return out, cycles, {"load": t1 - t0, "run": t2 - t1, "read": time.perf_counter() - t2}

    def sim_words(self, timeout=900):
        """The dumped words of tsp_sim on the input of the last step (`--sim-step`)."""
        sim = os.environ.get("TSP_SIM", os.path.join(ROOT, "target", "spec", "tsp_sim"))
        if not os.access(sim, os.X_OK):
            raise DriverError(f"tsp_sim not found at {sim} (TSP_SIM=PATH)")
        r = self._sh([sim, self.cfg_path, "--max-cycles", str(self.budget)], stdin=self.compat_input().encode(),
                     ok=(0, 3), timeout=timeout)
        if r.returncode == 3:
            raise DriverError("tsp_sim: --max-cycles exceeded")
        e, words = self._parse(r.stdout.decode())
        return e, words


# -- decoding the dump -------------------------------------------------------------------------------------------
def plane_array(planes, t):
    return data.from_planes([bytes(p) for p in planes], t)[:CFG.hidden]


def planes_of(arr):
    return [bytes(p) for p in data.to_planes(pad(arr))]


def first_difference(a, b):
    """(plane, lane) of the first differing byte of two lists of planes, or None."""
    for j, (x, y) in enumerate(zip(a, b)):
        if bytes(x) != bytes(y):
            return j, next(i for i in range(len(x)) if x[i:i + 1] != y[i:i + 1])
    return None


def check_reference(name, got, want):
    d = first_difference(got, planes_of(want))
    if d:
        raise DriverError(f"bit-exact reference: {name}: plane {d[0]} lane {d[1]} differs from the numpy twin")


def margin_of(logits):
    top = np.partition(logits, -2)[-2:]
    return float(top[1] - top[0])


# -- the loop ----------------------------------------------------------------------------------------------------
def generate(w, ids, n_new, prog, golden=(), eos=None, say=print, sim_step=None, on_token=None):
    """Greedy generation of `n_new` tokens after `ids`. Returns the dict of the run (rows, tokens, hiddens)."""
    if len(ids) + n_new - 1 > MAX_STEPS:
        raise DriverError(f"the prompt ({len(ids)} tokens) and {n_new} generated tokens need {len(ids) + n_new - 1} "
                          f"positions, the cache holds {MAX_STEPS - 1} rows and the current slot ({MAX_STEPS} "
                          f"positions): at most {MAX_STEPS - len(ids) + 1} tokens for this prompt")
    kv = exact_ref.HostKV(CFG)
    ref_kv = exact_ref.HostKV(CFG) if "exact" in golden else None
    tokens = [int(t) for t in ids]
    rows, hiddens, reference_words = [], [], 0
    pos = 0
    say(f"{'pos':>3} {'token':>6} {'cycles':>7} {'load s':>7} {'run s':>7} {'read s':>7} {'head s':>7} {'step s':>7}"
        "  next")
    while True:
        t_start = time.perf_counter()
        tok = tokens[pos]
        last = pos >= len(tokens) - 1
        x0 = exact_ref.embed(w, tok, pos)
        k, v, mask = kv.slots(pos)
        words, cycles, tm = prog.step(x0, k, v, mask)
        t0 = time.perf_counter()
        hidden = plane_array(words["hidden"], "f32")
        k_cur = np.stack([plane_array(words[f"k{l}"], "f16") for l in range(CFG.layers)])
        v_cur = np.stack([plane_array(words[f"v{l}"], "f32") for l in range(CFG.layers)])
        tm["read"] += time.perf_counter() - t0
        kv.append(k_cur, v_cur)
        hiddens.append(hidden)
        t0 = time.perf_counter()
        logits = exact_ref.head_logits(w, hidden) if last else None
        nxt = int(np.argmax(logits)) if last else None
        tm["head"] = time.perf_counter() - t0
        if ref_kv is not None:
            trace = {}
            res = exact_ref.decode_step(w, ref_kv, tok, pos, logits=last, trace=trace)
            check_reference("hidden", words["hidden"], res.hidden)
            for l in range(CFG.layers):
                check_reference(f"x{l + 1}", words[f"x{l + 1}"], trace[f"x{l + 1}"])
                check_reference(f"k{l}", words[f"k{l}"], res.k_cur[l])
                check_reference(f"v{l}", words[f"v{l}"], res.v_cur[l])
            if last and not np.array_equal(res.logits, logits):
                raise DriverError("bit-exact reference: the logits differ from bit-exact reference's host head")
            h16 = tref.to_f16(res.hidden)                  # the chip's own head window: f16 hidden, exact dot per chunk
            for c in range(len(prog.head_rows) // es.CHUNK):
                rows_c = prog.head_rows[c * es.CHUNK:(c + 1) * es.CHUNK]
                check_reference(f"logits{c}", words[f"logits{c}"], nx.dot_f16(rows_c, h16, tref.LANES))
            reference_words += 4 + 10 * CFG.layers + 4 * (len(prog.head_rows) // es.CHUNK)
        t_sim = 0.0
        if sim_step is not None and len(rows) == sim_step:
            t_sim = time.perf_counter()
            e, sw = prog.sim_words()
            t_sim = time.perf_counter() - t_sim
            same = e == "E - -1 0" and all(sw[k][:prog.vlanes] == prog.last_words[k][:prog.vlanes]
                                           for ks in prog.dumps.values() for k in ks)
            say(f"{'PASS' if same else 'FAIL'} tsp_sim on the input of step {sim_step} (position {pos}): "
                f"{len(prog.last_words)} dumped words {'equal to' if same else 'DIFFER from'} the ISS's "
                f"({t_sim:.0f} s on tsp_sim, not in the step time)")
            if not same:
                raise DriverError("tsp_sim and the ISS disagree on a dumped word")
        total = time.perf_counter() - t_start - t_sim
        row = dict(pos=pos, token=tok, cycles=cycles, load=tm["load"], run=tm["run"], read=tm["read"],
                   head=tm["head"], step=total, next=nxt, margin=margin_of(logits) if last else None)
        rows.append(row)
        say(f"{pos:>3} {tok:>6} {cycles if cycles is not None else '-':>7} {tm['load']:>7.2f} {tm['run']:>7.2f} "
            f"{tm['read']:>7.2f} {tm['head']:>7.3f} {total:>7.2f}  {'' if nxt is None else nxt}")
        if last:
            tokens.append(nxt)
            if on_token is not None:
                on_token(nxt)
            if len(tokens) - len(ids) >= n_new or (eos is not None and nxt == eos):
                break
        pos += 1
    return dict(rows=rows, tokens=tokens, hiddens=np.stack(hiddens), reference_words=reference_words, kv=kv)


# -- reporting ---------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--ids", help="the prompt as comma-separated GPT-2 token ids (no tokenizer needed)")
    ap.add_argument("--tokens", type=int, default=20, help="tokens to generate")
    ap.add_argument("--weights", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--seed", type=int, default=es.SEED, help="synthetic weights seed")
    ap.add_argument("--checkpoint", default=comparison_ref.DEFAULT_CHECKPOINT)
    ap.add_argument("--sampler", choices=("greedy",), default="greedy")
    ap.add_argument("--golden", default="", help="comma-separated: exact, rounded, fp32")
    ap.add_argument("--check", action="store_true", help="= --golden exact")
    ap.add_argument("--interface", choices=("native", "compat"), default="native",
                    help="tsp-iss run --weights (native) or the tsp_sim-compatible input on stdin (compat)")
    ap.add_argument("--eos", type=int, default=None, help="stop after this id (default: none; real weights: 50256)")
    ap.add_argument("--tokenizer", help="path of the GPT-2 tokenizer.json")
    ap.add_argument("--sim-step", type=int, help="also run tsp_sim on this step's input and compare the words")
    ap.add_argument("--stream", action="store_true",
                    help="print the prompt, then each generated token's text as it is produced (no per-step table)")
    ap.add_argument("--json", help="write the per-step rows and the gate results here")
    ap.add_argument("--keep", help="build into this directory and keep it (default: a temporary directory)")
    a = ap.parse_args(argv)
    golden = set(x for x in a.golden.split(",") if x) | ({"exact"} if a.check else set())
    if golden - {"exact", "rounded", "fp32"}:
        ap.error(f"--golden: unknown tier(s) {sorted(golden - {'exact', 'rounded', 'fp32'})}")
    if a.tokens < 1:
        ap.error("--tokens must be at least 1")
    try:
        ids, tok, note = prompt_ids(a)
        if note:
            print(f"note: {note}")
        if a.weights == "real":
            w, why = comparison_ref.load_weights(a.checkpoint)
            if w is None:
                raise DriverError(f"checkpoint {a.checkpoint!r} is not available ({why})")
            print(f"weights: real, {why}")
            eos = EOS if a.eos is None else a.eos
        else:
            w = eb.weights(a.seed)
            print(f"weights: synthetic seed {a.seed}")
            eos = a.eos
        if len(ids) + a.tokens - 1 > MAX_STEPS:
            raise DriverError(f"the prompt ({len(ids)} tokens) and {a.tokens} generated tokens need "
                              f"{len(ids) + a.tokens - 1} positions, the cache holds {MAX_STEPS - 1} rows and the "
                              f"current slot: at most {MAX_STEPS - len(ids) + 1} tokens for this prompt")
        work = a.keep or tempfile.mkdtemp(prefix="opentsp-tinystories-")
        os.makedirs(work, exist_ok=True)
        print(f"prompt ids {ids}, {a.tokens} tokens, {a.interface} interface, build directory {work}")
        t_all = time.perf_counter()
        prog = StepProgram(w, work, a.interface)
        say = print
        on_token = None
        if a.stream:
            say = lambda *_: None                                        # noqa: E731 (the stream replaces the step table)
            shown = [0]
            new_ids = []

            def stream_token(t):
                new_ids.append(t)
                if tok is None:
                    print(f"[{t}]", end=" ", flush=True)
                    return
                text = tok.decode(new_ids)
                if text.endswith("\ufffd"):                              # an incomplete UTF-8 sequence: wait for the next token
                    return
                print(text[shown[0]:], end="", flush=True)
                shown[0] = len(text)

            on_token = stream_token
            print("--- generation ---")
            print(tok.decode(ids) if tok is not None else f"prompt ids {ids}", end="", flush=True)
        res = generate(w, ids, a.tokens, prog, golden, eos, say=say, sim_step=a.sim_step, on_token=on_token)
        if a.stream:
            print("\n--- end ---")
        wall = time.perf_counter() - t_all
        rows, toks = res["rows"], res["tokens"]
        new = toks[len(ids):]
        prog.check_unchanged()
        cycles = [r["cycles"] for r in rows if r["cycles"] is not None]
        summary = dict(prompt_ids=ids, new_ids=new, steps=len(rows), emit_s=prog.t_emit, asm_s=prog.t_asm,
                       wall_s=wall, text_digest=prog.text_digest, object_digest=prog.obj_digest, gates={},
                       cycles_total=sum(cycles) if cycles else None, rows=rows,
                       sum_s={k: sum(r[k] for r in rows) for k in ("load", "run", "read", "head", "step")})
        print(f"IDS {' '.join(map(str, toks))}")
        print(f"NEW {' '.join(map(str, new))}")
        if tok is not None:
            text = tok.decode(toks)
            print(f"TEXT {text!r}")
            summary["text"] = text
        pin = prog.text_digest == es.PINNED_TEXT
        print(f"PROGRAM text sha256 {prog.text_digest} (matches expected text: {pin}), object sha256 {prog.obj_digest[:16]}...: "
              f"identical at all {len(rows)} steps")
        s = summary["sum_s"]
        print(f"steps {len(rows)}: {sum(cycles):,} cycles ({cycles[0]:,} per step), wall {wall:.1f} s "
              f"(emit {prog.t_emit:.1f}, assemble {prog.t_asm:.2f}, load {s['load']:.1f}, run {s['run']:.1f}, "
              f"read {s['read']:.1f}, host head {s['head']:.2f}); {len(new)} tokens at "
              f"{len(new) / s['step']:.2f} tokens/s with the prefill of {len(ids)} tokens counted"
              if cycles else f"steps {len(rows)}, wall {wall:.1f} s")
        bad = 0
        if "exact" in golden:
            print(f"GATE bit-exact reference PASS: {len(rows)} steps, {res['reference_words']} dumped words (the head window's logits "
                  "included) and the host logits equal the numpy twin's, token sequence equal")
            summary["gates"]["exact"] = dict(ok=True, steps=len(rows), words=res["reference_words"])
        if golden & {"rounded", "fp32"}:
            from opentsp_vllm.models.tinystories import driver_gates
            bad += driver_gates.run(w, a, toks, ids, res, golden, summary, tok, eos)
        if a.json:
            json.dump(summary, open(a.json, "w"), indent=1, default=str)
        if not a.keep:
            import shutil                                                # noqa: PLC0415
            shutil.rmtree(work, ignore_errors=True)
        return 1 if bad else 0
    except DriverError as e:
        print(f"FAIL driver: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
