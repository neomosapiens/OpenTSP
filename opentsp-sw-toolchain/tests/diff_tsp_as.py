#!/usr/bin/env python3
"""The one differential entry point of `ots-asm` against `opentsp-arch-spec/tools/tsp_as.py`: every command is compared
(`insn`, `word`, `asm --input` and `-o`, `dis` and `objinput` of text and of objects, `wpack`, `objinput --weights`,
`mpack`, `mobjinput`).

Instructions: per profile, the accept and reject corpora of `opentsp-arch-spec/test/isa_db/corpus/` and seeded mutations of
every text (delete, duplicate or replace a character, swap two tokens, insert a tab, CR, NUL or non-ASCII byte, change a number
or a stream, change case, add or remove spaces, change a separator); `word` on the words of the corpora, on seeded random words
of both widths, on words with a flipped bit, with unusual widths and numerals (signs, 0x, underscores, huge values).

Test corpus:
  (a) the program blocks of `assembly.md`;
  (b) `opentsp-sw-toolchain/tests/programs/*.ots` (the programs of `ots-asm/tests/program.rs`: every error
      class of assembly.md section 5, the externs, the AGT warnings, the iq_depth warning);
  (c) `tsp_as.py dis` of the `text` of every golden vector (`opentsp-arch-spec/conformance/model/{ref,mini}.json`),
      on both profiles, and the same text again after `asm --input` of the listing (annotated ones fail
      there, in both tools);
  (d) seeded mutations of (a) and (b): delete, duplicate or swap a line, change a character, insert a
      tab, CR, NUL or non-ASCII byte, change a number or a cycle or a queue, duplicate or drop a
      directive, perturb an extern, splice an instruction of the instruction corpus into a bundle.
Objects: every accepted program is also run with `asm -o` (the object files must be byte for byte
equal); each distinct object is read by `dis` and `objinput` (plain; with `--profile` equal to its header and
with another one; read against the other configuration), and seeded mutations of the objects (byte flips,
truncation, extension, header and table fields, extern entries, names and digests) are read the same way.
Each program runs under the shipped profiles and under a variant of `ref` with 22 MEM slices a hemisphere
(`--config`, the absent-slice rules). Weight packages: `wpack` runs in both tools over seeded tensor sets (names, element types, word counts, file lengths,
both profiles and the 22-slice variant; the package files must be byte for byte equal), and `objinput --weights`
over every distinct object that has externs: with the package that fits it (tensors of the extern names and sizes, in
either order, plus an unused one), with packages that miss, rename or resize a tensor or are for the other profile,
and with seeded mutations of those packages (byte flips, truncation, extension, header and table fields, names, offsets,
digests; and structural edits whose offsets and digests are rebuilt, so the deeper checks are reached).
Manifests: `mpack` runs in both tools over seeded chip sets drawn from the corpus objects (K from the counts of the wiring, 1..8 and
8 N, and the counts it refuses; both profiles and the 22-slice variant; chips that are another profile's, mutated, empty, text or a manifest; `--profile`
default, equal, other and unknown; `--config` right and wrong; the manifest files must be byte for byte equal), and `mobjinput` over every manifest
`tsp_as.py` made: without weights, with one package per chip (fitting, none unused), with the wrong number of packages, swapped, of the other profile,
mutated, with a tensor missing or resized, and over seeded mutations of the manifests (byte flips, truncation, extension, header fields, entries,
digests; and structural edits whose entries are rebuilt: a mutated or foreign object, a chip count the wiring cannot take, a trailer, a size of 0,
version, revision, topology and reserved fields).
Known differences, not generated: a `wpack` element type that is not in `encoding.json`, a malformed `name=dtype:file`
argument and a missing `-o` end in a Python traceback in `tsp_as.py` (this script maps them to 2); `ots-asm` reports
them as ASM-004 (exit 1) and usage (exit 2). Malformed `word` arguments and missing input files are coded diagnostics in both
(OTS-E-ASM-001 and OTS-E-ASM-003, exit 1, no traceback), and a refused `--config` file is exit 2 in both: `bad_inputs` compares them.
Both tools run `asm <file> --input`, and `dis` of the text that
`asm --input` printed (when it accepted the program) and of seeded mutations of the text of (c) (digits,
units, queues, bundle and NOP bits, lines); the compared
compared observables are exit status, stdout byte for byte, and the diagnostics on
stderr as `<where>: <code>` (plus the ` (warning)` marker), in order; for an instruction no template matches
(OTS-E-DEC-003) also the message text (`no instruction matches "frob 1"`, RV2-B-10).

Parcel order: both tools keep a parcel in source order (`tsp_as.py` sorts the cycle view stably by
(cycle, queue); `ots-asm` the same), so the default of `ots-asm` is compared with the unmodified `tsp_as.py`.

usage: diff_tsp_as.py <ots-asm> [--fuzz N] [--seed S] [--insn-only | --weights-only | --manifest-only] [--objects]       exit 0 when there is no difference
(`--insn-only`, `--weights-only` and `--manifest-only` skip the mutations of the programs, `dis` and the object rows: the corpus objects are
assembled once and only the rows of that phase run, with all N mutations)
(`--objects` is accepted: the object rows always run unless one of the `-only` flags is given)
"""
import concurrent.futures, contextlib, glob, io, json, os, random, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "..", "opentsp-arch-spec")
sys.path.insert(0, os.path.join(SPEC, "tools"))
import tsp_as as A

# The comparisons of the shared behaviour run without the profile validation (`run_ref` passes `--no-validate`; the profiles of the
# differential include one the schema refuses): the API calls of this file skip it as well. `bad_inputs` compares the validation itself.
A.CHECK_CAPS = False

MODEL = os.path.join(SPEC, "conformance", "model")
DOC = os.path.join(HERE, "..", "..", "opentsp-arch-spec", "docs", "assembly.md")
PROGRAMS = os.path.join(HERE, "programs")
INSN_CORPUS = os.path.join(SPEC, "test", "isa_db", "corpus")


def corpus(profile, name):
    rows = []
    for line in open(os.path.join(INSN_CORPUS, profile, name), encoding="utf-8"):
        line = line.rstrip("\n")
        if line and not line.startswith("#"):
            rows.append(line.rsplit("\t", 1) if name == "reject.txt" else line.split("\t"))
    return rows


def memoize_isa(mod):
    """One ISA per (profile, config): constructing it compiles every template."""
    cache, make = {}, mod.ISA

    def isa(profile="ref", config=None, *a, **k):
        key = (profile, config)
        if key not in cache:
            cache[key] = make(profile, config, *a, **k)
        return cache[key]
    for n in dir(make):
        if not n.startswith("__"):
            setattr(isa, n, getattr(make, n))
    mod.ISA = isa
    return mod


def references():
    """[(name, tsp_as module, flags for ots-asm)]: the unmodified `tsp_as.py` and the default of `ots-asm`."""
    return [("source order", memoize_isa(A), [])]


def run_ref(mod, argv, validate=False):
    """`tsp_as.main` on argv. The comparisons of the shared behaviour run without the profile validation (`--no-validate`); `validate`
    leaves it on, as `bad_inputs` does."""
    if "--config" in argv and not validate and "--no-validate" not in argv:
        argv = argv + ["--no-validate"]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = mod.main(["tsp_as.py"] + argv)
        except (IndexError, ValueError, KeyError, OSError):
            rc = 2
            err.write("usage")
    return rc, out.getvalue(), err.getvalue()


def run_new(exe, argv, validate=False):
    # `tsp_as.py` does not warn about a multi-byte `wpack` dtype, so that behaviour of `ots-asm` is switched off here and the
    # comparison stays on the shared behaviour (`ots-asm/tests/weights.rs` pins the addition). Both tools validate a `--config`
    # file; the comparisons of the shared behaviour skip it in both (`--no-validate`) because the reference contract of
    # the causal rows accepts signed timing values that the full schema refuses, and `bad_inputs` compares the validation itself.
    extra = (["--no-validate"] if "--config" in argv and not validate else []) + (["--raw-bytes"] if argv[:1] == ["wpack"] else [])
    r = subprocess.run([exe] + argv + extra, capture_output=True)
    return r.returncode, r.stdout.decode("latin-1"), r.stderr.decode("utf-8", "replace")


def prefix(err):
    """`<where>: <code>` (and the warning marker) of each diagnostic line; the message is free."""
    out = []
    for ln in err.splitlines():
        parts = ln.split(": ", 2)
        out.append(": ".join(parts[:2]) if len(parts) > 1 else ln)
    return out


def dec003(err):
    """The full text of the OTS-E-DEC-003 lines (the rejected instruction is quoted in double quotes by both tools)."""
    return [ln.split(": ", 1)[1] if ": " in ln else ln for ln in err.splitlines() if "OTS-E-DEC-003" in ln]


def same(a, b):
    return a[0] == b[0] and a[1] == b[1] and (prefix(a[2]) == prefix(b[2]) or a[0] == 2) and (dec003(a[2]) == dec003(b[2]) or a[0] == 2)


def compare(label, mod, exe, argv, flags=(), show=""):
    """Run both tools on argv; return (equal, ref result, new result), printing a DIFF when not."""
    a, b = run_ref(mod, argv), run_new(exe, list(argv) + list(flags))
    ok = same(a, b)
    if not ok:
        print(f"DIFF [{label}]{show}\n  tsp_as : {a[0]} {a[1][:400]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1][:400]!r} {b[2].strip()!r}")
    return ok, a, b


def blocks():
    doc = open(DOC).read()
    return [b for b in re.findall(r"```\n(.*?)```", doc, re.S) if b.startswith(".arch")]


def programs():
    out = [("assembly.md", b) for b in blocks()]
    for f in sorted(glob.glob(os.path.join(PROGRAMS, "*.ots"))):
        out.append((os.path.basename(f), open(f, "rb").read().decode("latin-1")))
    return out


QUEUES = ["MEM[4]", "MEM[5]", "MEM[44]", "MEM.E[0]", "MEM[007]", "MEM[88]", "VXM[0]", "VXM[15]", "VXM[16]", "CU", "cu", "MXM[0].LW", "MXM[3].ACC",
          "SXM.W.DST[3]", "SXM.E.PRM", "C2C.W.TX[1]", "MEM [0]", "", "MEM[22]", "MEM[43]"]
SPLICE_RE = re.compile(r"^(\s*)([^:]+):(.*)$")
# decimal cycles past Python's 4,300-digit integer-string limit
LONG_CYCLES = ["0" * 5000 + "1", "0" * 5000 + "4294967296", "9" * 5000, "0" * 5001]
LONG_PROGRAM = ".arch opentsp-tsa\n.profile mini\n.section text\nbundle @cycle={c} {{\nMEM[0]: read 0x0, E:0, 0x0\n}}\n"


def mutate_program(rnd, text, insns):
    lines = text.split("\n")
    for _ in range(rnd.choice((1, 1, 1, 2, 3))):
        if not lines:
            lines = [""]
        k = rnd.randrange(14)
        i = rnd.randrange(len(lines))
        if k == 0:
            lines.pop(i)
        elif k == 1:
            lines.insert(i, lines[i])
        elif k == 2:
            j = rnd.randrange(len(lines))
            lines[i], lines[j] = lines[j], lines[i]
        elif k == 3:
            lines[i] = mutate(rnd, lines[i])
        elif k == 4:
            ln = lines[i]
            p = rnd.randrange(len(ln) + 1)
            lines[i] = ln[:p] + rnd.choice(["\t", "\r", "\x00", "\u00e9", "\x1f", "\x0b"]) + ln[p:]
        elif k == 5:
            ms = list(re.finditer(r"0x[0-9a-fA-F]+|[0-9]+", lines[i]))
            if ms:
                m = rnd.choice(ms)
                new = rnd.choice(["0x0", "0x1", "0x1fff", "0x2000", "0xffff", "0xffffffff", "0x" + "f" * 30, "0", "4", "7", "21", "22", "44", "87", "88", "4294967295",
                                  "4294967296", "99999999999999999999"])
                lines[i] = lines[i][:m.start()] + new + lines[i][m.end():]
        elif k == 6:
            m = SPLICE_RE.match(lines[i])
            if m and not lines[i].lstrip().startswith((".", "bundle")):
                lines[i] = f"{m.group(1)}{rnd.choice(QUEUES)}:{m.group(3)}"
        elif k == 7:
            lines[i] = re.sub(r"@cycle=[0-9]+", lambda m: "@cycle=" + rnd.choice(["0", "1", "2", "5", "6", "7", "40", "70000", "4294967295", "4294967296"] + LONG_CYCLES), lines[i])
        elif k == 8:
            d = [j for j, ln in enumerate(lines) if ln.lstrip().startswith(".")]
            if d:
                j = rnd.choice(d)
                if rnd.randrange(2):
                    lines.insert(rnd.randrange(len(lines)), lines[j])
                else:
                    lines.pop(j)
        elif k == 9:
            lines[i] = re.sub(r"(slice|addr|size)=(0x[0-9a-fA-F]+|[0-9]+)", lambda m: f"{m.group(1)}=" + rnd.choice(
                ["0", "1", "4", "5", "21", "22", "44", "66", "87", "88", "0x0", "0x1", "0x2", "0x10", "0x11", "0x1fff", "0x2000", "0x100", "0xffffffffffffffff"]), lines[i])
        elif k == 10:
            q = rnd.choice(QUEUES[:12])
            lines.insert(i, f"  {q}: {rnd.choice(insns)}")
        elif k == 11:
            lines = lines[:i]
        elif k == 12:      # a whole bundle of one splice, at a free-ish cycle
            lines[i:i] = [f"bundle @cycle={rnd.choice([0, 1, 2, 3, 9, 33, 41, 100])} {{", f"  {rnd.choice(QUEUES[:12])}: {rnd.choice(insns)}", "}"]
        else:
            lines[i] = lines[i].replace(" ", rnd.choice(["  ", "", " ; c ", "\t"]), 1)
    return "\n".join(lines)


def vector_texts():
    """[(profile, name, Q text)] of every golden vector: `text` maps a queue (hex) to its units."""
    out = []
    for profile in ("ref", "mini"):
        for v in json.load(open(os.path.join(MODEL, f"{profile}.json")))["vectors"]:
            out.append((profile, v["name"], "".join(f"Q {q} {us}\n" for q, us in v["text"].items())))
    return out


def unit_value(u):
    """A unit as a number; a unit an earlier mutation of the same line made unreadable counts as 0."""
    try:
        return int(u, 16)
    except ValueError:
        return 0


def mutate_text(rnd, text):
    """A seeded mutation of harness Q text (the unit words are what `dis` decodes)."""
    lines = text.split("\n")
    for _ in range(rnd.choice((1, 1, 2, 3))):
        i = rnd.randrange(len(lines))
        t = lines[i].split()
        k = rnd.randrange(12)
        if k == 0:
            lines.pop(i)
            if not lines:
                lines = [""]
            continue
        elif k == 1:
            lines.insert(i, lines[i])
            continue
        elif k == 2:
            j = rnd.randrange(len(lines))
            lines[i], lines[j] = lines[j], lines[i]
            continue
        elif len(t) < 3:
            lines[i] = rnd.choice(["Q 4 00000000", "Q ffff 80000000", "Q 5 80000021 10000010 00000000", "Q", "", "q 4 0", "Q 4 g", "Q 90 0"])
        elif k == 3:      # a hex digit of a unit
            j = rnd.randrange(2, len(t))
            u = list(t[j])
            u[rnd.randrange(len(u))] = rnd.choice("0123456789abcdef")
            t[j] = "".join(u)
        elif k == 4:      # the bundle bit of a unit
            j = rnd.randrange(2, len(t))
            t[j] = f"{unit_value(t[j]) ^ 0x80000000:08x}"
        elif k == 5:      # a unit becomes a NOP prefix or padding
            t[rnd.randrange(2, len(t))] = rnd.choice(["00000001", "0000ffff", "80000003", "00010002", "ffffffff", "00000000", "70000000"])
        elif k == 6:      # the tag of a unit (decode layout)
            j = rnd.randrange(2, len(t))
            t[j] = f"{(unit_value(t[j]) & 0x8FFFFFFF) | rnd.randrange(8) << 28:08x}"
        elif k == 7:      # drop or duplicate a unit
            j = rnd.randrange(2, len(t))
            t.pop(j) if rnd.randrange(2) else t.insert(j, t[j])
        elif k == 8:      # another queue
            t[1] = rnd.choice(["0", "4", "5", "2c", "2d", "58", "59", "ffff", "16", "2b", "55", "8f", "90"])
        elif k == 9:      # swap two units
            a, b = rnd.randrange(2, len(t)), rnd.randrange(2, len(t))
            t[a], t[b] = t[b], t[a]
        elif k == 10:     # truncate
            t = t[:rnd.randrange(2, len(t) + 1)]
        else:
            t[rnd.randrange(2, len(t))] = rnd.choice(["8", "zz", "123456789", "0x1", "-1", "+1", "00_1"])
        if len(t) >= 2 or k in (3, 4, 5, 6, 7, 8, 9, 10, 11):
            lines[i] = " ".join(t)
    return "\n".join(lines)


def make_config(tmp):
    """`ref` with 22 MEM slices a hemisphere."""
    text = open(os.path.join(SPEC, "config", "ref.json")).read()
    assert '"mem_slices_per_hemisphere": 44' in text
    path = os.path.join(tmp, "mem22.json")
    open(path, "w").write(text.replace('"mem_slices_per_hemisphere": 44', '"mem_slices_per_hemisphere": 22'))
    return path


def dis_phase(exe, refs, cfg, tmp, rnd, n, cases, bad, compared, seen):
    """`dis` rows. tsp_as `dis` does not depend on parcel order, so the first reference stands for all."""
    mod = refs[0][1]
    path = os.path.join(tmp, "d.txt")
    vecs = vector_texts()
    texts = [(f"vector {p} {name}", p, t) for p, name, t in vecs]
    # the Q text each accepted program lowers to, decoded under both profiles
    for name, text, what in cases:
        open(os.path.join(tmp, "p.ots"), "wb").write(text.encode("latin-1", "replace"))
        rc, out, _ = run_ref(mod, ["asm", os.path.join(tmp, "p.ots"), "--input"])
        if rc == 0 and out:
            texts += [(f"lowering of {what}", p, out) for p in ("ref", "mini")]
    base = [t for t in texts]
    for _ in range(n):
        p, name, t = rnd.choice(vecs)
        texts.append((f"mutation of vector {p} {name}", p, mutate_text(rnd, t)))
    nvec = len(vecs)
    dis_seen = {"exit 0": 0, "exit 1": 0, "other": 0}
    relisted = 0
    for what, p, t in texts:
        open(path, "wb").write(t.encode("latin-1", "replace"))
        for opts in ([], ["--config", cfg]):
            argv_ = ["dis", path, "--profile", p] + opts
            a, b = run_ref(mod, argv_), run_new(exe, argv_)
            compared += 1
            dis_seen["exit 0" if a[0] == 0 else "exit 1" if a[0] == 1 else "other"] += 1
            if not same(a, b):
                bad += 1
                print(f"DIFF [dis; {what}; profile {p}; {' '.join(opts) or 'shipped profile'}]\n--- text\n{t}\n---\n"
                      f"  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r}")
                if bad >= 10:
                    print(f"diff_tsp_as: stopped after {bad} differences")
                    return bad, compared
            # `objinput` of the same text: the Q lines, no profile needed
            oi = ["objinput", path, "--profile", p] + opts
            ok, _, _ = compare(f"objinput; {what}; profile {p}", mod, exe, oi, show=f"\n--- text\n{t}")
            compared += 1
            if not ok:
                bad += 1
            # corpus (c): the listing reassembled (only the vectors, both profiles' listings)
            if what.startswith("vector ") and not opts and a[1]:
                lp = os.path.join(tmp, "l.ots")
                open(lp, "wb").write(a[1].encode("latin-1", "replace"))
                for rname, rmod, flags in refs:
                    argv2 = ["asm", lp, "--input"]
                    a2, b2 = run_ref(rmod, argv2), run_new(exe, argv2 + flags)
                    compared += 1
                    relisted += 1
                    if not same(a2, b2):
                        bad += 1
                        print(f"DIFF [asm of the listing of {what}; {rname}]\n  tsp_as : {a2[0]} {a2[1][:200]!r} {a2[2].strip()!r}\n  ots-asm: {b2[0]} {b2[1][:200]!r} {b2[2].strip()!r}")
    print(f"  dis: {len(texts)} texts ({nvec} golden-vector texts of corpus (c), {len(base) - nvec} lowerings, {len(texts) - len(base)} mutations), "
          f"{', '.join(f'{k} {v}' for k, v in dis_seen.items())} (tsp_as), {relisted} listings of (c) reassembled")
    return bad, compared


def le(v, n):
    return v.to_bytes(n, "little")


def mutate_bytes(rnd, blob):
    data = bytearray(blob)
    mode = rnd.randrange(4)
    if mode == 0 and data:
        data[rnd.randrange(len(data))] ^= 1 << rnd.randrange(8)
    elif mode == 1 and data:
        data[rnd.randrange(len(data))] = rnd.randrange(256)
    elif mode == 2:
        del data[rnd.randrange(len(data) + 1):]
    else:
        data += rnd.randbytes(rnd.choice((1, 4, 8, 32)))
    return bytes(data)


def mutate_object(rnd, data):
    """Mutate the ELF carrier or a decoded ABI field, placement or instruction."""
    mode = rnd.randrange(13)
    if mode < 4:
        return mutate_bytes(rnd, data)
    sections = A._read_elf(data)
    note = bytearray(sections[b".note.opentsp"])
    iq = bytearray(sections[b".tsp.iq"])
    loads = bytearray(sections[b".tsp.load"])
    count = int.from_bytes(iq[:4], "little")
    nloads = int.from_bytes(loads[:4], "little")
    if mode == 4:
        pos = rnd.choice([4, 5, 6, 16, 18, 20, 32, 40, 46, 48, 50])
        corrupted = bytearray(data)
        corrupted[pos] ^= 1 << rnd.randrange(8)
        return bytes(corrupted)
    if mode == 5:
        note[20 + rnd.randrange(len(note) - 20)] ^= 1 << rnd.randrange(8)
    elif mode == 6 and count:
        pos = 4 + rnd.randrange(count) * 8 + rnd.choice((0, 2, 4))
        iq[pos:pos+2] = le(rnd.choice((0, 1, 0xffff, 144, 145)), 2)
    elif mode == 7 and count >= 2:
        u, v = [4 + 8*i for i in rnd.sample(range(count), 2)]
        iq[u:u+8], iq[v:v+8] = iq[v:v+8], iq[u:u+8]
    elif mode == 8 and len(iq) >= 8 + 8*count:
        pos = rnd.randrange(4+8*count, len(iq), 4)
        iq[pos:pos+4] = le(rnd.choice((0, 1, 0xffffffff, 0x80000000, 0xfcb3, 0x10000010)), 4)
    elif mode in (9, 10, 11) and nloads:
        pos = 4 + 32*rnd.randrange(nloads)
        if mode == 9:
            off = rnd.choice((16, 18, 20, 24, 28))
            loads[pos+off:pos+off+2] = le(rnd.choice((0, 1, 0xffff, 88, 8192)), 2)
        elif mode == 10:
            name = rnd.choice((b"x", b"x\n", b"1x", b"a b", b"a\0b", b"", b"abcdefghijklmnop"))
            loads[pos:pos+16] = name.ljust(16, b"\0")
        else:
            other = 4 + 32*rnd.randrange(nloads)
            loads[pos:pos+32] = loads[other:other+32]
    elif mode == 12:
        section = rnd.choice((iq, loads))
        section[:4] = le(rnd.choice((0, 1, 144, 0xffffffff)), 4)
    sections.update({b".note.opentsp": note, b".tsp.iq": iq, b".tsp.load": loads})
    return A._write_elf([(key.decode("ascii"), 7 if key == b".note.opentsp" else 1,
                          0, value, 4) for key, value in sections.items() if key != b".shstrtab"])


def object_phase(exe, mod, cfg, tmp, rnd, n, objects, bad, compared):
    """`dis` and `objinput` of the assembled objects and of mutations of them. tsp_as `dis` and `objinput`
    do not depend on parcel order, so the first reference stands for all."""
    path = os.path.join(tmp, "o.elf")
    pool = sorted(objects.items(), key=lambda kv: kv[0])
    rows = []                 # (what, bytes, config of the object)
    for data, conf in pool:
        rows.append(("assembled object", data, conf))
    for _ in range(n):
        data, conf = rnd.choice(pool)
        rows.append(("mutated object", mutate_object(rnd, data), conf))
    seen = {}
    for what, data, conf in rows:
        open(path, "wb").write(data)
        try:
            prof = A._read_elf(data)[b".note.opentsp"][24:40].split(b"\0")[0].decode("latin-1")
        except A.AsmError:
            prof = "ref"
        other = "mini" if prof != "mini" else "ref"
        variants = [[], ["--profile", prof], ["--profile", other]]
        variants += [["--config", conf]] if conf else [["--config", cfg]]
        for cmd in ("dis", "objinput"):
            for opts in variants:
                ok, a, _ = compare(f"{cmd} of {what}; {' '.join(opts) or 'plain'}", mod, exe, [cmd, path] + opts,
                                   show=f"\n--- object ({len(data)} bytes)\n{data.hex()}")
                compared += 1
                msg = re.sub(r"0x[0-9a-f]+|[0-9]+|'[^']*'", "#", a[2].strip().split(": ", 2)[-1])[:46] if a[2] and a[0] == 1 else ""
                key = f"{cmd} exit {a[0]}" + (f" {msg}" if msg else "")
                seen[key] = seen.get(key, 0) + 1
                if not ok:
                    bad += 1
                    if bad >= 10:
                        print(f"diff_tsp_as: stopped after {bad} differences")
                        return bad, compared
    kinds = {}
    for what, *_ in rows:
        kinds[what] = kinds.get(what, 0) + 1
    print(f"  objects: {', '.join(f'{v} {k}s' for k, v in kinds.items())}; tsp_as outcomes:\n    " + "\n    ".join(f"{v:5d} {k}" for k, v in sorted(seen.items())))
    return bad, compared


WDT = ["i8", "f16", "i32", "f32", "u8", "i16", "u16", "u32", "b8", "b16", "b32", "e4m3", "e5m2", "raw"]
WNAMES = ["w", "x", "y", "a_1", "_t", "W0", "abcdefghijklmno", "z9", "bias", "k"]
BAD_WNAMES = ["", "1a", "a b", "a-b", "abcdefghijklmnop", "a.b", "x\n", "\u00e9", "a$"]
def vlanes(profile, config=None):
    return A.ISA(profile, config).pvars["vlanes"]


def random_words(rnd, vl, n):
    base = bytes(rnd.randrange(256) for _ in range(vl))
    return [bytes((b + k) & 0xff for b in base) for k in range(n)]


def fit_package(rnd, profile, exts, conf=None, extra=True):
    """The package that fits `exts`: tensors of the extern names and sizes (in a seeded order), plus an unused one."""
    vl = vlanes(profile, conf)
    ts = [(x["name"], rnd.choice(WDT), random_words(rnd, vl, x["size"])) for x in exts]
    if extra:
        ts.append(("unused", rnd.choice(WDT), random_words(rnd, vl, rnd.choice((1, 2)))))
    rnd.shuffle(ts)
    return [(n, None if d == "raw" else d, w) for n, d, w in ts]


def package_bytes(mod, profile, tensors):
    try:
        return mod.write_weights(profile, tensors)
    except A.AsmError:
        return None


def mutate_package(rnd, data, vl):
    """Exercise binary framing and independent shape, offset and digest validation."""
    case = rnd.randrange(14)
    if case < 4:
        return mutate_bytes(rnd, data)
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8+n])
    image = data[8+n:]
    metadata = header["__metadata__"]
    keys = [key for key in header if key != "__metadata__"]
    if case == 4:
        return le(rnd.choice((0, 1, n - 1, n + 1, 0xffffffffffffffff)), 8) + data[8:]
    if case == 5:
        metadata["opentsp.profile"] = rnd.choice(("mini", "ref", "REF", "", "re\0f", "nosuch"))
    elif keys:
        key = rnd.choice(keys)
        item = header[key]
        if case == 6:
            item["shape"] = rnd.choice(([0, vl], [1, vl + 1], [0xffffffff, vl], [-1, vl], [3]))
        elif case == 7:
            item["data_offsets"] = rnd.choice(([1, 0], [0, 0], [0, len(image)+1], [-1, 0]))
        elif case == 8:
            new = rnd.choice(BAD_WNAMES)
            header[new] = header.pop(key)
            for attr in ("dtype.", "sha256."):
                metadata[attr+new] = metadata.pop(attr+key)
        elif case == 9:
            metadata["sha256."+key] = "0" * 64
        elif case == 10:
            item["dtype"] = rnd.choice(("F32", "I8", "", 42))
        elif case == 11:
            metadata["dtype."+key] = rnd.choice(("254", "100", "-1", "", "nan"))
        elif case == 12:
            del header[key]
        elif case == 13:
            header["another"] = item
            metadata["dtype.another"] = metadata["dtype."+key]
            metadata["sha256.another"] = metadata["sha256."+key]
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw)%8)
    return le(len(raw), 8) + raw + image


def assigned_weights_roundtrip(exe, mod, tmp):
    """Require success (not just agreement on failure) for each writer/reader pairing."""
    from pathlib import Path
    root = Path(tmp)
    obj = str(root / "types.elf")
    example = os.path.join(HERE, "../crates/ots-asm/tests/data/example.ots")
    assert run_new(exe, ["asm", "-o", obj, example])[0] == 0
    data = bytes(i % 256 for i in range(320))
    raw = root / "types.bin"
    raw.write_bytes(data)
    readers = (lambda argv: run_ref(mod, argv), lambda argv: run_new(exe, argv))
    for dtype, code in [*mod.ISA("ref").t.eltype_codes.items(), ("raw", 255)]:
        packages = []
        for idx, writer in enumerate(readers):
            package = root / f"types-{idx}.safetensors"
            args = ["wpack", "--profile", "ref", "-o", str(package), f"x={dtype}:{raw}", f"y={dtype}:{raw}"]
            assert writer(args)[0] == 0, (dtype, idx)
            blob = package.read_bytes()
            packages.append(blob)
            assert mod.read_weights(blob, "ref") == {"x": (code, [data]), "y": (code, [data])}
            for reader in readers:
                rc, out, err = reader(["objinput", obj, "--weights", str(package)])
                assert rc == 0 and out.startswith("S "), (dtype, idx, rc, err)
        assert packages[0] == packages[1], dtype


def weights_phase(exe, mod, cfg, tmp, rnd, n, objects, bad, compared):
    """`wpack` and `objinput --weights`. tsp_as does not depend on parcel order here, so the first reference stands for all."""
    seen = {}

    def note(key):
        seen[key] = seen.get(key, 0) + 1

    assigned_weights_roundtrip(exe, mod, tmp)
    # ---- wpack ----
    wp = 0
    out_r, out_n = os.path.join(tmp, "wr.safetensors"), os.path.join(tmp, "wn.safetensors")
    for _ in range(max(40, n // 4)):
        profile = rnd.choice(["ref", "mini"])
        conf = rnd.choice([None, None, cfg]) if profile == "ref" else None
        vl = vlanes(profile, conf)
        args = []
        for t in range(rnd.choice((0, 1, 1, 2, 3, 6))):
            name = rnd.choice(WNAMES if rnd.randrange(8) else WNAMES + BAD_WNAMES)
            nwords = rnd.choice((0, 1, 1, 2, 3, 5))
            ln = nwords * vl + rnd.choice((0, 0, 0, 0, 0, 1, -1)) if nwords else rnd.choice((0, vl))
            path = os.path.join(tmp, f"wp{t}.bin")
            open(path, "wb").write(bytes(rnd.randrange(256) for _ in range(max(ln, 0))))
            args.append(f"{name}={rnd.choice(WDT)}:{path}")
        for f in (out_r, out_n):
            if os.path.exists(f):
                os.remove(f)
        argv_r = ["wpack", "--profile", profile] + (["--config", conf] if conf else []) + ["-o", out_r] + args
        argv_n = ["wpack", "--profile", profile] + (["--config", conf] if conf else []) + ["-o", out_n] + args
        ra, rb = run_ref(mod, argv_r), run_new(exe, argv_n)
        compared += 1
        wp += 1
        fa = open(out_r, "rb").read() if os.path.exists(out_r) else None
        fb = open(out_n, "rb").read() if os.path.exists(out_n) else None
        # tsp_as opens the output before it checks the tensors, so a refused pack leaves an empty file there; ours leaves none
        if ra[0] != 0 and fa == b"":
            fa = None
        key = "wpack exit %d %s" % (ra[0], re.sub(r"[0-9]+|'[^']*'", "#", ra[2].strip().split(": ", 2)[-1])[:40] if ra[0] else "")
        note(key)
        if not same(ra, rb) or fa != fb:
            bad += 1
            print(f"DIFF [wpack {' '.join(args)}; {profile}; {conf or 'shipped'}]\n  tsp_as : {ra[0]} {ra[1][:200]!r} {ra[2].strip()!r} {None if fa is None else fa.hex()[:300]}\n"
                  f"  ots-asm: {rb[0]} {rb[1][:200]!r} {rb[2].strip()!r} {None if fb is None else fb.hex()[:300]}")
            if bad >= 10:
                return bad, compared
    # ---- objinput --weights ----
    path_o, path_w = os.path.join(tmp, "wo.elf"), os.path.join(tmp, "w.safetensors")
    pool = []
    for data, conf in sorted(objects.items(), key=lambda kv: kv[0]):
        try:
            prof, _, exts = A.read_object(data, None, conf)
        except A.AsmError:
            continue
        if sum(x["size"] for x in exts) <= 300:
            pool.append((data, conf, prof, exts))
    rows = []                       # (what, object bytes, object config, package bytes or None, profile)
    for data, conf, prof, exts in pool:
        other = "mini" if prof == "ref" else "ref"
        pk = package_bytes(mod, prof, fit_package(rnd, prof, exts, conf))
        if pk is None:
            continue
        rows.append(("fitting package", data, conf, pk, prof))
        rows.append(("package without unused", data, conf, package_bytes(mod, prof, fit_package(rnd, prof, exts, conf, extra=False)), prof))
        rows.append(("package of the other profile", data, conf, package_bytes(mod, other, fit_package(rnd, other, [], None)), prof))
        if exts:
            x = rnd.choice(exts)
            vl = vlanes(prof, conf)
            rest = [e for e in exts if e is not x]
            miss = fit_package(rnd, prof, rest, conf)
            rows.append(("package with a tensor missing", data, conf, package_bytes(mod, prof, miss), prof))
            for delta in (-1, 1):
                if x["size"] + delta >= 1:
                    resized = fit_package(rnd, prof, rest, conf) + [(x["name"], "i8", random_words(rnd, vl, x["size"] + delta))]
                    rows.append((f"tensor of {x['size'] + delta} words", data, conf, package_bytes(mod, prof, resized), prof))
        for _ in range(max(1, n // max(1, len(pool)))):
            rows.append(("mutated package", data, conf, mutate_package(rnd, pk, vlanes(prof, conf)), prof))
    rows = [r for r in rows if r[3] is not None]
    for what, data, conf, pk, prof in rows:
        open(path_o, "wb").write(data)
        open(path_w, "wb").write(pk)
        other = "mini" if prof == "ref" else "ref"
        variants = [[], ["--profile", prof], ["--profile", other]] if what == "fitting package" else [[], ["--profile", prof]]
        if conf:
            variants = [v + ["--config", conf] for v in variants]
        for opts in variants:
            ok, a, _ = compare(f"objinput --weights; {what}; {' '.join(opts) or 'plain'}", mod, exe, ["objinput", path_o] + opts + ["--weights", path_w],
                               show=f"\n--- object {data.hex()}\n--- package {pk.hex()[:600]}")
            compared += 1
            msg = re.sub(r"0x[0-9a-f]+|[0-9]+|'[^']*'|\"[^\"]*\"", "#", a[2].strip().split(": ", 2)[-1])[:46] if a[2] and a[0] == 1 else ""
            note(f"objinput --weights exit {a[0]}" + (f" {msg}" if msg else ""))
            if not ok:
                bad += 1
                if bad >= 10:
                    return bad, compared
    # text input: no externs, the package must be for --profile (default ref)
    for what, text_in, prof in (("text", "Q 4 00000021 10000010 00000000\n", "ref"), ("empty text", "", "ref")):
        tp = os.path.join(tmp, "t.txt")
        open(tp, "w").write(text_in)
        for pk_prof in ("ref", "mini"):
            open(path_w, "wb").write(package_bytes(mod, pk_prof, fit_package(rnd, pk_prof, [], None)))
            for opts in ([], ["--profile", "mini"]):
                ok, a, _ = compare(f"objinput --weights; {what}; package of {pk_prof}", mod, exe, ["objinput", tp] + opts + ["--weights", path_w])
                compared += 1
                note(f"objinput --weights of text exit {a[0]}")
                bad += 0 if ok else 1
    kinds = {}
    for r in rows:
        kinds[r[0]] = kinds.get(r[0], 0) + 1
    print(f"  weights: {wp} wpack runs; {len(pool)} objects with externs; {', '.join(f'{v} {k}s' for k, v in sorted(kinds.items()))}; tsp_as outcomes:\n    "
          + "\n    ".join(f"{v:5d} {k}" for k, v in sorted(seen.items())))
    return bad, compared
MUT_CHARS = ["\t", "\r", "\u00e9", " ", ",", ":", "x", "0", "9", "E", "W", "+", "-", ".", "_", "a", "Z", "\x0b", "\n"]


def mutate(rnd, t):
    """A seeded mutation of an instruction text."""
    k = rnd.randrange(12)
    if not t:
        return rnd.choice(MUT_CHARS)
    i = rnd.randrange(len(t))
    if k == 0:
        return t[:i] + t[i + 1:]
    if k == 1:
        return t[:i] + t[i] + t[i:]
    if k == 2:
        return t[:i] + rnd.choice(MUT_CHARS) + t[i + 1:]
    if k == 3:
        return t[:i] + rnd.choice(MUT_CHARS) + t[i:]
    if k == 4:
        toks = t.split(" ")
        if len(toks) > 1:
            a, b = rnd.sample(range(len(toks)), 2)
            toks[a], toks[b] = toks[b], toks[a]
        return " ".join(toks)
    if k == 5:      # change a number
        ms = list(re.finditer(r"0x[0-9a-fA-F]+|[0-9]+", t))
        if ms:
            m = rnd.choice(ms)
            new = rnd.choice(["0x0", "0x1", "0xffff", "0x10000", "0xffffffff", "0x100000000", "0x" + "f" * 30, "0", "7", "31", "32", "99999999999999999999",
                              "0x00", "0X1", "0x", "-1", "1_0"])
            return t[:m.start()] + new + t[m.end():]
        return t + " 0x1"
    if k == 6:      # change a stream
        ms = list(re.finditer(r"[EW]:[0-9]+", t))
        if ms:
            m = rnd.choice(ms)
            return t[:m.start()] + rnd.choice(["E:0", "W:31", "E:32", "W:16", "E:15", "W:017", "E:", "X:1", "e:1", "E:99999999999"]) + t[m.end():]
        return t
    if k == 7:
        return rnd.choice([t.upper(), t.lower(), t.capitalize()])
    if k == 8:
        return rnd.choice([t.replace(", ", ","), t.replace(", ", " , "), t.replace(", ", "  ,  "), t.replace(" ", "  "), "  " + t, t + "  ", t.replace(" ", "\t", 1)])
    if k == 9:
        return t + rnd.choice([" ", ",", " ,", "\n", " 0x1", ", E:0", "."])
    if k == 10:
        return t[:i]
    return t[i:]


NUMERALS = ["0", "1", "32", "032", "64", "+32", " 32 ", "3_2", "6_4", "16", "8", "-32", "33", "1" + "0" * 30, "0x20", "32.0", "", "x"]
HEXES = ["0", "1", "400000", "0x400000", "0X400000", "+400000", "-1", "-0x1", "ffffffff", "FFFFFFFF", "100000000", "ffffffffffffffff", "10000000000000000",
         "f" * 40, "dead_beef", "_1", "1_", "1__0", "0x_1", "0x", "", "g", " 1 ", "fcb3"]


def insn_phase(exe, mod, rnd, n, bad, compared):
    """`insn` and `word`, per profile: the accept and reject corpora, `n` seeded mutations of their texts, `word` on the words of
    the corpora, on `n` seeded random words of both widths, on words with a flipped bit and with unusual widths and numerals."""
    for profile in ("ref", "mini"):
        texts = [r[0] for r in corpus(profile, "accept.txt")] + [r[0] for r in corpus(profile, "reject.txt")]
        cases = [(["insn", "--profile", profile, t], f"{profile} corpus") for t in texts]
        for _ in range(n):
            t = rnd.choice(texts)
            for _ in range(rnd.choice((1, 1, 2, 3))):
                t = mutate(rnd, t)
            cases.append((["insn", "--profile", profile, t], f"{profile} mutation"))
        words = [(int(r[1]), int(r[2], 16)) for r in corpus(profile, "accept.txt")]
        for width, w in words[::3]:
            cases.append((["word", "--profile", profile, str(width), f"{w:x}"], f"{profile} word"))
        for _ in range(n):
            width, w = rnd.choice(words)
            k = rnd.randrange(6)
            if k == 0:
                w ^= 1 << rnd.randrange(width)
            elif k == 1:
                w = rnd.getrandbits(width)
            elif k == 2:
                w |= 1 << rnd.randrange(width, 70)
            elif k == 3:
                width = rnd.choice((32, 64))
            ws, hs = (str(width), f"{w:x}") if k < 4 else (rnd.choice(NUMERALS), rnd.choice(HEXES))
            cases.append((["word", "--profile", profile, ws, hs], f"{profile} word fuzz"))
        with concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 4) as pool:    # the runs of `ots-asm` are independent processes
            news = list(pool.map(lambda c: run_new(exe, c[0]), cases))
        for (argv_, what), b in zip(cases, news):
            a = run_ref(mod, argv_)
            compared += 1
            if not same(a, b):
                bad += 1
                print(f"DIFF [{what}] {argv_!r}\n  tsp_as : {a[0]} {a[1][:400]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1][:400]!r} {b[2].strip()!r}")
                if bad >= 10:
                    return bad, compared
    return bad, compared


MHDR = 64
MENT = 8
MPROFILES = [b"mini", b"ref", b"REF", b"", b"re\0f", b"ref\0\0\0x", b"r\xe9f", b"abcdefghijklmnop", b"nosuch", b"ref\n", b"a/b"]
CHIP_COUNTS = [1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 24, 32, 40, 48, 56, 64, 65, 264]
BAD_COUNTS = [0, 265, 300, 512, 1000]


def build_manifest(profile, objs, conf=None, k=None, topo=0, rev=2, resv=0, major=0, minor=1, hsize=64, prof16=None, digests=None, entries=None, trailer=b""):
    """A manifest from object bytes with the layout of assembly.md section 4, also for the counts, fields and entries the tools refuse
    (`entries`: {chip: (offset, size)} replaces an entry)."""
    enc, prof = digests or A.interface_digests(profile, conf)
    head = b"OTSM" + bytes([major, minor]) + le(hsize, 2) + (prof16 if prof16 is not None else profile.encode().ljust(16, b"\0")[:16]) \
        + le(len(objs) if k is None else k, 4) + le(topo, 4) + le(rev, 4) + le(resv, 4) + enc + prof
    off, table = MHDR + MENT * len(objs), b""
    for i, o in enumerate(objs):
        o_off, o_size = (entries or {}).get(i, (off, len(o)))
        table += le(o_off, 4) + le(o_size, 4)
        off += len(o)
    return head + table + b"".join(objs) + trailer


def mutate_manifest(rnd, data, objs, prof, conf, pool):
    """A seeded mutation of a manifest: bytes and length, header fields, entries, digests, and structural edits whose
    entries are rebuilt (a mutated, foreign or dropped object, a chip count the wiring cannot take, a trailer)."""
    b = bytearray(data)
    k_edit = rnd.randrange(16)
    k = int.from_bytes(b[24:28], "little") if len(b) >= 28 else 0
    nent = k if len(b) >= MHDR + MENT * k else 0
    if k_edit == 0 and b:
        i = rnd.randrange(len(b))
        b[i] ^= 1 << rnd.randrange(8)
    elif k_edit == 1 and b:
        b[rnd.randrange(len(b))] = rnd.randrange(256)
    elif k_edit == 2:
        del b[rnd.choice([rnd.randrange(len(b) + 1), rnd.randrange(min(len(b), MHDR + MENT * 4) + 1)]):]
    elif k_edit == 3:
        b += bytes(rnd.randrange(256) for _ in range(rnd.choice((1, 4, 8, 32))))
    elif k_edit == 4 and len(b) >= MHDR:
        f, w = rnd.choice([(4, 1), (5, 1), (6, 2), (24, 4), (28, 4), (32, 4), (36, 4), (8, 1), (11, 1), (23, 1), (40, 1), (51, 1), (52, 1), (63, 1)])
        b[f:f + w] = le(rnd.choice([0, 1, 2, 3, 8, 9, 16, 63, 64, 65, 72, 255, 256, 0xffff, 0xffffffff, 0x7fffffff]) & ((1 << 8 * w) - 1), w)
    elif k_edit == 5 and len(b) >= MHDR:
        b[8:24] = rnd.choice(MPROFILES).ljust(16, b"\0")[:16]
    elif k_edit == 6 and nent:
        e = MHDR + MENT * rnd.randrange(nent)
        f = rnd.choice((0, 4))
        old = int.from_bytes(b[e + f:e + f + 4], "little")
        b[e + f:e + f + 4] = le(rnd.choice([0, 1, old + 1, max(old - 1, 0), old + 8, 64, 72, len(b), len(b) + 1, 0xffffffff, 0x80000000]) & 0xffffffff, 4)
    elif k_edit == 7 and nent >= 2:
        i, j = rnd.sample(range(nent), 2)
        ei, ej = MHDR + MENT * i, MHDR + MENT * j
        b[ei:ei + MENT], b[ej:ej + MENT] = bytes(b[ej:ej + MENT]), bytes(b[ei:ei + MENT])
    elif k_edit == 8 and len(b) >= MHDR:
        b[40 + rnd.randrange(24)] ^= 1 << rnd.randrange(8)
    elif k_edit == 9:                                   # one embedded object mutated, entries rebuilt
        o = list(objs)
        o[rnd.randrange(len(o))] = mutate_object(rnd, rnd.choice(o))
        return build_manifest(prof, o, conf)
    elif k_edit == 10:                                  # one embedded object of another profile or configuration
        o = list(objs)
        o[rnd.randrange(len(o))] = rnd.choice(pool)[0]
        return build_manifest(prof, o, conf)
    elif k_edit == 11:                                  # a chip count the wiring cannot take (objects repeated)
        n_ = rnd.choice(BAD_COUNTS[1:])
        return build_manifest(prof, [objs[i % len(objs)] for i in range(n_)], conf)
    elif k_edit == 12:                                  # no object at all
        return build_manifest(prof, [], conf)
    elif k_edit == 13:                                  # the header's count differs from the entries
        return build_manifest(prof, objs, conf, k=rnd.choice(CHIP_COUNTS + BAD_COUNTS))
    elif k_edit == 14:                                  # a trailer, or an entry of size 0
        if rnd.randrange(2):
            return build_manifest(prof, objs, conf, trailer=bytes(rnd.randrange(256) for _ in range(rnd.choice((1, 8)))))
        i = rnd.randrange(len(objs))
        return build_manifest(prof, objs, conf, entries={i: (MHDR + MENT * len(objs) + sum(len(x) for x in objs[:i]), 0)})
    else:                                               # a field the layout fixes, rebuilt: version, revision, topology, reserved
        f = rnd.choice([dict(minor=2), dict(major=1), dict(hsize=60), dict(rev=0), dict(rev=1), dict(topo=1), dict(resv=1), dict(topo=0xffffffff)])
        return build_manifest(prof, objs, conf, **f)
    return bytes(b)


def manifest_phase(exe, mod, cfg, tmp, rnd, n, objects, bad, compared):
    """`mpack` and `mobjinput`. tsp_as does not depend on parcel order here, so the first reference stands for all."""
    seen = {}

    def note(key):
        seen[key] = seen.get(key, 0) + 1

    pool = []                                           # (object bytes, config, profile, externs)
    for data, conf in sorted(objects.items(), key=lambda kv: kv[0]):
        try:
            prof, _, exts = A.read_object(data, None, conf)
        except A.AsmError:
            continue
        pool.append((data, conf, prof, exts))
    groups = {}
    for o in pool:
        groups.setdefault((o[2], o[1]), []).append(o)
    gkeys = sorted(groups, key=lambda g: (g[0], g[1] or ""))
    foreign = [o for o in pool]
    path_m, out_r, out_n = os.path.join(tmp, "m.otsm"), os.path.join(tmp, "mr.otsm"), os.path.join(tmp, "mn.otsm")
    # ---- mpack ----
    made = []                                           # (manifest bytes, profile, config, objects of the chips)
    mp = 0
    for t in range(max(60, n // 4)):
        gk = rnd.choice(gkeys)
        profile, conf = gk
        k = rnd.choice(CHIP_COUNTS if t % 7 == 0 else CHIP_COUNTS[:7] * 3 + BAD_COUNTS[:3])
        chips = [rnd.choice(groups[gk]) for _ in range(k)]
        blobs = [c[0] for c in chips]
        bad_chip = rnd.randrange(6)
        if blobs and bad_chip == 0:
            i = rnd.randrange(len(blobs))
            blobs[i] = rnd.choice([rnd.choice(foreign)[0], mutate_object(rnd, blobs[i]), b"", b"Q 4 00000021\n", b"OTSM" + bytes(60),
                                   build_manifest(profile, [blobs[0]], conf)])
        paths = []
        for i, blob in enumerate(blobs):
            fp = os.path.join(tmp, f"mc{i}.elf")
            open(fp, "wb").write(blob)
            paths.append(fp)
        opts = rnd.choice([["--profile", profile], ["--profile", profile], ["--profile", profile], [], ["--profile", "mini" if profile == "ref" else "ref"], ["--profile", "nosuch"]])
        copts = ["--config", conf] if conf and rnd.randrange(8) else (["--config", cfg] if not conf and rnd.randrange(10) == 0 else [])
        for f in (out_r, out_n):
            if os.path.exists(f):
                os.remove(f)
        ra = run_ref(mod, ["mpack"] + opts + copts + ["-o", out_r] + paths)
        rb = run_new(exe, ["mpack"] + opts + copts + ["-o", out_n] + paths)
        compared += 1
        mp += 1
        fa = open(out_r, "rb").read() if os.path.exists(out_r) else None
        fb = open(out_n, "rb").read() if os.path.exists(out_n) else None
        if ra[0] != 0 and fa == b"":                    # tsp_as opens the output first, so a refused pack leaves an empty file there
            fa = None
        note("mpack exit %d %s" % (ra[0], re.sub(r"[0-9]+|'[^']*'", "#", ra[2].strip().split(": ", 2)[-1])[:40] if ra[0] else ""))
        if not same(ra, rb) or fa != fb:
            bad += 1
            print(f"DIFF [mpack {' '.join(opts + copts)}; {len(paths)} chips of {profile}]\n  tsp_as : {ra[0]} {ra[1][:200]!r} {ra[2].strip()!r} {None if fa is None else fa.hex()[:300]}\n"
                  f"  ots-asm: {rb[0]} {rb[1][:200]!r} {rb[2].strip()!r} {None if fb is None else fb.hex()[:300]}")
            if bad >= 10:
                return bad, compared
        elif fa is not None:
            made.append((fa, profile, copts[1] if copts else None, blobs))     # the configuration the manifest was accepted under
    # ---- mobjinput of the manifests tsp_as made, with weights ----
    rows = []                                           # (what, manifest bytes, config, [package bytes], profile)
    for data, profile, conf, blobs in made:
        rows.append(("no weights", data, conf, [], profile))
        chips = [A.read_object(b, None, conf)[2] for b in blobs]       # the externs of every chip
        if len(blobs) <= 8 and sum(x["size"] for c in chips for x in c) <= 200:
            fit = [package_bytes(mod, profile, fit_package(rnd, profile, c, conf)) for c in chips]
            if None in fit:
                continue
            rows.append(("one package per chip", data, conf, fit, profile))
            rows.append(("one package per chip, none unused", data, conf, [package_bytes(mod, profile, fit_package(rnd, profile, c, conf, extra=False)) for c in chips], profile))
            other = "mini" if profile == "ref" else "ref"
            for cnt in sorted({1, len(fit) - 1, len(fit) + 1, 2 * len(fit)} - {0, len(fit)}):
                rows.append((f"{cnt} packages for {len(fit)} chips", data, conf, [fit[i % len(fit)] for i in range(cnt)], profile))
            if len(fit) >= 2:
                rows.append(("packages swapped", data, conf, fit[1:] + fit[:1], profile))
            i = rnd.randrange(len(fit))
            exts_i = chips[i]
            with_ = list(fit)
            with_[i] = package_bytes(mod, other, fit_package(rnd, other, [], None))
            rows.append(("a package of the other profile", data, conf, with_, profile))
            with_ = list(fit)
            with_[i] = mutate_package(rnd, fit[i], vlanes(profile, conf))
            rows.append(("a mutated package", data, conf, with_, profile))
            if exts_i:
                x = rnd.choice(exts_i)
                with_ = list(fit)
                with_[i] = package_bytes(mod, profile, fit_package(rnd, profile, [e for e in exts_i if e is not x], conf))
                rows.append(("a tensor missing", data, conf, with_, profile))
                with_ = list(fit)
                with_[i] = package_bytes(mod, profile, fit_package(rnd, profile, [e for e in exts_i if e is not x], conf) + [(x["name"], "i8", random_words(rnd, vlanes(profile, conf), x["size"] + 1))])
                rows.append(("a tensor of the wrong size", data, conf, with_, profile))
    # ---- mutations of the manifests ----
    for _ in range(max(60, n // 2)):
        if not made:
            break
        data, profile, conf, blobs = rnd.choice(made)
        rows.append(("mutated manifest", mutate_manifest(rnd, data, blobs, profile, conf, pool), conf, [], profile))
    mi = 0
    for what, data, conf, pks, profile in rows:
        if any(p is None for p in pks):
            continue
        open(path_m, "wb").write(data)
        wargs = []
        for i, pk in enumerate(pks):
            wp = os.path.join(tmp, f"mw{i}.safetensors")
            open(wp, "wb").write(pk)
            wargs += ["--weights", wp]
        other = "mini" if profile != "mini" else "ref"
        variants = [[]] if pks else [[], ["--profile", other], ["--config", conf or cfg]]
        if conf and pks:
            variants = [["--config", conf]]
        for opts in variants:
            ok, a, _ = compare(f"mobjinput; {what}; {' '.join(opts) or 'plain'}", mod, exe, ["mobjinput", path_m] + opts + wargs,
                               show=f"\n--- manifest ({len(data)} bytes) {data.hex()[:1500]}")
            compared += 1
            mi += 1
            msg = re.sub(r"0x[0-9a-f]+|[0-9]+|'[^']*'|\"[^\"]*\"", "#", a[2].strip().split(": ", 2)[-1])[:46] if a[2] and a[0] == 1 else ""
            note(f"mobjinput exit {a[0]}" + (f" {msg}" if msg else ""))
            if not ok:
                bad += 1
                if bad >= 10:
                    return bad, compared
    kinds = {}
    for r in rows:
        kinds[r[0]] = kinds.get(r[0], 0) + 1
    print(f"  manifests: {mp} mpack runs ({len(made)} manifests made), {mi} mobjinput runs over {len(pool)} objects: {'; '.join(f'{k}: {v}' for k, v in sorted(kinds.items()))}; tsp_as outcomes:\n    "
          + "\n    ".join(f"{v:5d} {k}" for k, v in sorted(seen.items())))
    return bad, compared


def causal_profiles(exe, mod, tmp):
    """Check both edges and the ordered diagnostic when several constraints fail."""
    source = os.path.join(tmp, "causal.ots")
    with open(source, "w") as out:
        out.write(".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n VXM[0]: setvl 0x1\n}\n")
    cases = [(field, valid) for field in ("Shift", "Permute", "ABC", "ABC8", "map", "fetch")
             for valid in (False, True)] + [("all", False), ("signed", True)]
    for field, valid in cases:
        with open(os.path.join(SPEC, "config", "ref.json")) as inp:
            cfg = json.load(inp)
        for name in ("Shift", "Permute", "ABC", "ABC8"):
            if field in (name, "all"):
                cfg["timing"]["overrides"] = [r for r in cfg["timing"]["overrides"] if r["entry"] != name]
                cfg["timing"]["overrides"].append({"entry": name, "d_skew": 15, "d_func": 34 + valid})
        if field == "signed":
            # Causal at the boundary. The full simulator schema separately rejects negative skew.
            cfg["timing"]["overrides"].append({"entry": "Shift", "d_skew": -1, "d_func": 19})
        if field in ("map", "all"):
            cfg["timing"]["sxm_map_ready"] = 19 + valid
        if field in ("fetch", "all"):
            cfg["icu"].update(fetch_bytes_per_cycle=64, fetch_latency=14 + valid)
        path = os.path.join(tmp, f"{field}-{valid}.json")
        with open(path, "w") as out:
            json.dump(cfg, out)
        args = ["asm", source, "--config", path]
        outputs = []
        for tool in ("reference", "rust"):
            obj = os.path.join(tmp, f"{field}-{valid}-{tool}.elf")
            cmd = args + ["-o", obj]
            result = run_ref(mod, cmd) if tool == "reference" else run_new(exe, cmd)
            data = None
            if os.path.exists(obj):
                with open(obj, "rb") as inp:
                    data = inp.read()
            outputs.append((result, data))
        (ref, data), native = outputs
        assert ref[0] == (0 if valid else 1) and (data is not None) == valid, (field, ref)
        assert valid or "OTS-E-ASM-003" in ref[2], (field, ref)
        assert native == (ref, data), (field, valid, ref, native)
    return len(cases)


def bad_inputs(exe, mod, tmp):
    """, : malformed arguments, missing files and refused profile files end in a coded diagnostic and no traceback, in both
    tools with the same status and the same `<where>: <code>`; a refused `--config` file (the schema, a relation, a timing rule, bad JSON)
    is status 2 in both. Returns (cases, differences)."""
    with open(os.path.join(SPEC, "config", "ref.json")) as inp:
        ref = json.load(inp)
    good = os.path.join(tmp, "bad-good.ots")
    with open(good, "w") as out:
        out.write(".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n VXM[0]: setvl 0x1\n}\n")
    cfgs = {}

    def cfg(name, edit=None, raw=None):
        c = json.loads(json.dumps(ref))
        if edit:
            edit(c)
        path = os.path.join(tmp, f"badcfg-{name}.json")
        with open(path, "wb") as out:
            out.write(raw if raw is not None else json.dumps(c).encode())
        cfgs[name] = path
        return path
    cfg("bank_bit", lambda c: c["sram"].update(bank_bit=3))
    cfg("unknown_key", lambda c: c.update(frobnicate=1))
    cfg("type", lambda c: c["icu"].update(iq_depth="big"))
    cfg("bound", lambda c: c["c2c"].update(links=129))
    cfg("timing_row", lambda c: c["timing"]["vxm"].update(d_skew=9))
    cfg("typed_not_taken", lambda c: c["timing"]["overrides"].append({"entry": "Read.i8", "d_func": 5, "d_skew": 0}))
    cfg("typed_unknown_type", lambda c: c["timing"]["overrides"].append({"entry": "Max.q9", "d_func": 5, "d_skew": 2}))
    cfg("typed_skew", lambda c: c["timing"]["overrides"].append({"entry": "Max.i8", "d_func": 2, "d_skew": 2}))
    cfg("typed_twice", lambda c: c["timing"]["overrides"].extend([{"entry": "Max.i8", "d_func": 3, "d_skew": 2}] * 2))
    # /-05: the five profiles the assemblers once accepted and the other tools refuse
    cfg("ext_unknown", lambda c: c.update(extensions=["nonesuch"]))
    cfg("ext_twice", lambda c: c.update(extensions=["bf16", "bf16"]))
    cfg("wide_acc_without_128", lambda c: c.update(extensions=c["extensions"] + ["wide_acc"]))
    cfg("128_without_wide_acc", lambda c: c["mxm"].update(acc_bits=128))
    cfg("acc_bits_64", lambda c: c["mxm"].update(acc_bits=64))
    cfg("json", raw=b"{not json")
    cfg("nul", raw=b"{}\0")
    cfg("ok")
    cfg("typed_ok", lambda c: c["timing"]["overrides"].append({"entry": "Max.i8", "d_func": 3, "d_skew": 2}))
    missing = os.path.join(tmp, "no-such-file")
    one_word = os.path.join(tmp, "one-word.bin")
    with open(one_word, "wb") as out:
        out.write(bytes(ref["geometry"]["superlanes"] * 16))
    pack = ["wpack", "--profile", "ref", "-o", os.path.join(tmp, "dup.safetensors")]
    cases = [(["word", "--profile", "ref", "64", "zz"], 1), (["word", "--profile", "ref", "x", "40"], 1), (["word", "--profile", "ref", "64"], 1),
             # word: a width other than 32 or 64, a value wider than its width and a negative value are OTS-E-ASM-001
             (["word", "--profile", "ref", "48", "40"], 1), (["word", "--profile", "ref", "0", "0"], 1), (["word", "--profile", "ref", "32", "100000000"], 1),
             (["word", "--profile", "ref", "64", "10000000000000000"], 1), (["word", "--profile", "ref", "64", "-1"], 1), (["word", "--profile", "ref", "32", "-40"], 1),
             # wpack: an unknown element type and a duplicate tensor name are OTS-E-ASM-004; a malformed argument is usage
             (pack + [f"t=zz:{one_word}"], 1), (pack + [f"t=i8:{one_word}", f"t=raw:{one_word}"], 1), (pack + [f"t=raw:{one_word}", f"u=raw:{one_word}"], 0),
             (pack + [f"t{one_word}"], 2), (pack + [f"t=raw{one_word}"], 2), (pack + ["t=raw:"], 2), (pack + [f"=raw:{one_word}"], 1),
             (["wpack", "--profile", "ref", f"t=raw:{one_word}"], 2),
             (["asm", missing], 1), (["dis", missing], 1), (["objinput", missing], 1), (["wpack", "--profile", "ref", "-o", os.path.join(tmp, "w.safetensors"), f"a=raw:{missing}"], 1),
             (["mpack", "--profile", "ref", "-o", os.path.join(tmp, "m.otsm"), missing], 1), (["asm", good, "--input", "--config", cfgs["ok"]], 0),
             (["asm", good, "--input", "--config", cfgs["typed_ok"]], 0)]
    cases += [(["asm", good, "--input", "--config", p], 2) for k, p in cfgs.items() if k not in ("ok", "typed_ok")]
    # , : the extension instructions on the three extension profiles, accepted or refused (the message names the
    # configuration in force), in both tools; the same on the shipped profiles, which declare no extension
    for name in ("ext_bf16", "mini_ext", "ref_ext"):
        path = os.path.join(SPEC, "config", "test", f"{name}.json")
        for text in ("abc.bf16 W:24", "abc.bfp8 W:24", "lw.bf16 W:0", "iw.bfp8 W:0", "acc.x128 W:0, replace, 0", "acc.f32 W:0, replace, 0",
                     "add.bf16 W:0, W:1, W:2", "convert.bf16 W:0, W:1", "acc.x128 W:0, replace, 7"):
            cases.append((["insn", "--config", path, text], None))
    for text in ("abc.bf16 W:24", "acc.x128 W:0, replace, 0"):
        cases.append((["insn", "--profile", "ref", text], None))
    # Profile-variable bounds under --config.
    for name in ("w16", "acc8", "one_plane", "c2c_wide"):
        path = os.path.join(SPEC, "config", "test", f"{name}.json")
        cases.append((["insn", "--config", path, "setvl 0x1"], None))
    bad = 0
    for argv, want in cases:
        a, b = run_ref(mod, argv, validate=True), run_new(exe, argv, validate=True)
        if want is None:
            if a[0] != b[0] or a[1] != b[1] or a[2] != b[2] or "Traceback" in a[2] + b[2]:
                bad += 1
                print(f"DIFF [extension insn {argv[-1]!r} {argv[2] if '--config' in argv else ''}]\n  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r}")
            elif "--config" in argv and "needs the extension" in a[2] and "configuration in force" not in a[2]:
                bad += 1
                print(f"DIFF [extension insn {argv[-1]!r}] the message does not name the configuration in force: {a[2].strip()!r}")
            continue
        tb = "Traceback" in a[2] or "Traceback" in b[2]
        ok = a[0] == want and b[0] == want and not tb and a[1] == b[1]
        if want == 1:
            ok = ok and prefix(a[2]) == prefix(b[2]) and re.search(r"OTS-E-ASM-00[134]", a[2]) is not None
        if want == 2 and "--config" in argv:
            ok = ok and "config" in a[2] and "config" in b[2]
        if want == 2 and "--config" not in argv:
            ok = ok and "usage" in a[2].lower() and "usage" in b[2].lower()
        if not ok:
            bad += 1
            print(f"DIFF [bad input {argv[0]} {argv[-1]}] want exit {want}\n  tsp_as : {a[0]} {a[1][:200]!r} {a[2].strip()[:300]!r}\n  ots-asm: {b[0]} {b[1][:200]!r} {b[2].strip()[:300]!r}")
    # : mpack --topology parity on a 2-chip ring (both tools, same bytes).
    topo = os.path.join(tmp, "iaf04-topo.json")
    with open(topo, "w") as out:
        out.write('{"chips":2,"default_latency":216,"links":['
                  '{"chip":0,"link":0,"peer":1,"peer_link":0},'
                  '{"chip":1,"link":0,"peer":0,"peer_link":0}]}\n')
    objs = []
    for chip in (0, 1):
        src = os.path.join(tmp, f"iaf04-c{chip}.ots")
        dest = os.path.join(tmp, f"iaf04-c{chip}.elf")
        with open(src, "w") as out:
            out.write(".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n VXM[0]: setvl 0x1\n}\n")
        ra = run_ref(mod, ["asm", src, "-o", dest], validate=True)
        if ra[0] != 0:
            bad += 1
            print(f"DIFF [iaf04 asm chip {chip}] tsp_as exit {ra[0]} {ra[2][:200]!r}")
        objs.append(dest)
    out_r = os.path.join(tmp, "iaf04-r.otsm")
    out_n = os.path.join(tmp, "iaf04-n.otsm")
    ra = run_ref(mod, ["mpack", "--profile", "ref", "--topology", topo, "-o", out_r] + objs, validate=True)
    rb = run_new(exe, ["mpack", "--profile", "ref", "--topology", topo, "-o", out_n] + objs, validate=True)
    if ra[0] != rb[0] or ra[0] != 0:
        bad += 1
        print(f"DIFF [iaf04 mpack --topology] tsp_as {ra[0]} {ra[2][:200]!r} ots-asm {rb[0]} {rb[2][:200]!r}")
    elif open(out_r, "rb").read() != open(out_n, "rb").read():
        bad += 1
        print("DIFF [iaf04 mpack --topology] package bytes differ")
    return len(cases) + 1, bad


VARIANT_CONFIGS = ("w16", "acc8", "one_plane", "c2c_wide")
NEW_KEY_PROFILES = VARIANT_CONFIGS


def variant_phase(exe, refs, tmp, rnd, cases, objects, bad, compared, sample=40):
    """`asm --input` and `asm -o` under each of VARIANT_CONFIGS over every corpus program and a seeded sample of the other cases; the
    objects that assemble are handed to the object, weight and manifest phases under their configuration (`objects`)."""
    path = os.path.join(tmp, "v.ots")
    sampled = [c for c in cases if c[2].startswith("corpus ")] + rnd.sample(cases, min(sample, len(cases)))
    rows = {}
    for vname in VARIANT_CONFIGS:
        conf = os.path.join(SPEC, "config", "test", f"{vname}.json")
        rows[vname] = [0, 0]
        for name, text, what in sampled:
            open(path, "wb").write(text.encode("latin-1", "replace"))
            for rname, mod, flags in refs:
                opts = ["--config", conf]
                a, b = run_ref(mod, ["asm", path, "--input"] + opts), run_new(exe, ["asm", path, "--input"] + opts + flags)
                compared += 1
                ro, no = os.path.join(tmp, "vref.elf"), os.path.join(tmp, "vnew.elf")
                for f in (ro, no):
                    if os.path.exists(f):
                        os.remove(f)
                a_o, b_o = run_ref(mod, ["asm", path, "-o", ro] + opts), run_new(exe, ["asm", path, "-o", no] + opts + flags)
                compared += 1
                ra = open(ro, "rb").read() if os.path.exists(ro) else None
                rb = open(no, "rb").read() if os.path.exists(no) else None
                rows[vname][0 if a[0] == 0 else 1] += 1
                if not same(a, b) or not same(a_o, b_o) or ra != rb:
                    bad += 1
                    print(f"DIFF [{what}; {rname}; --config {vname}; asm]\n--- program\n{text}\n---\n"
                          f"  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r} -o {a_o[0]} {None if ra is None else ra.hex()[:300]}\n"
                          f"  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r} -o {b_o[0]} {None if rb is None else rb.hex()[:300]}")
                    if bad >= 10:
                        return bad, compared
                elif ra is not None:
                    objects.setdefault(ra, conf)
    print("  variant configurations: " + "; ".join(f"{v}: {c[0]} accepted, {c[1]} refused" for v, c in rows.items()))
    return bad, compared

def new_key_profile_phase(exe, mod, tmp):
    """`insn` / `asm --input` / `asm -o` under `--config` for each widened-key profile, so a drift on the new keys is caught."""
    bad = 0
    compared = 0
    # probes that exercise the new keys: 16-bit address, high accumulator index, one-plane geometry, wide chip/link
    probes = {
        "w16": ["ifetchm 0xffff", "read 0xffff, E:0, 0x0", "setsegment 0x10000", "step 0x7fff", "ifetchm 0x10000"],
        "acc8": ["acc.i32 E:0, replace, 7", "acc.i32 E:0, replace, 8", "nop 1"],
        "one_plane": ["nop 1", "setvl 0x1"],
        "c2c_wide": ["send 0x1ff, 0x7f, W:0", "receive 0x1ff, 0xf, W:31", "send 0x200, 0x0, E:0", "transmit 0x41, 0x10"],
    }
    for name in NEW_KEY_PROFILES:
        path = os.path.join(SPEC, "config", "test", f"{name}.json")
        for text in probes[name]:
            argv = ["insn", "--config", path, text]
            a, b = run_ref(mod, argv), run_new(exe, argv)
            compared += 1
            if not same(a, b):
                bad += 1
                print(f"DIFF [new-key insn {name} {text!r}]\n  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r}")
        # a tiny program under --config of the profile
        src = os.path.join(tmp, f"nk-{name}.ots")
        open(src, "w").write(".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n VXM[0]: setvl 0x1\n}\n")
        for cmd in (["asm", src, "--input", "--config", path],):
            a, b = run_ref(mod, cmd), run_new(exe, cmd)
            compared += 1
            if not same(a, b):
                bad += 1
                print(f"DIFF [new-key asm {name}]\n  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r}")
        ro, no = os.path.join(tmp, f"nk-{name}-r.elf"), os.path.join(tmp, f"nk-{name}-n.elf")
        for f in (ro, no):
            if os.path.exists(f):
                os.remove(f)
        a_o = run_ref(mod, ["asm", src, "-o", ro, "--config", path])
        b_o = run_new(exe, ["asm", src, "-o", no, "--config", path])
        compared += 1
        ra = open(ro, "rb").read() if os.path.exists(ro) else None
        rb = open(no, "rb").read() if os.path.exists(no) else None
        if not same(a_o, b_o) or ra != rb:
            bad += 1
            print(f"DIFF [new-key asm -o {name}]\n  tsp_as : {a_o[0]} object {None if ra is None else ra.hex()[:80]}\n  ots-asm: {b_o[0]} object {None if rb is None else rb.hex()[:80]}")
    print(f"  new-key profiles: {len(NEW_KEY_PROFILES)} configs ({', '.join(NEW_KEY_PROFILES)}), {compared} comparisons, {bad} differences")
    return compared, bad

def topology_phase(exe, mod, tmp):
    """`mpack --topology`: a good file, a count mismatch, non-JSON, chips 0 / 513 / true."""
    bad = 0
    compared = 0
    # two identical chip objects so a 2-chip topology is the good case
    src = os.path.join(tmp, "topo-chip.ots")
    open(src, "w").write(".arch opentsp-tsa\n.profile ref\n.section text\nbundle @cycle=0 {\n VXM[0]: setvl 0x1\n}\n")
    objs = []
    for i in range(2):
        p = os.path.join(tmp, f"topo-c{i}.elf")
        r = run_ref(mod, ["asm", src, "-o", p])
        if r[0] != 0:
            print(f"DIFF [topology setup] could not assemble a chip: {r[2].strip()!r}")
            return 0, 1
        objs.append(p)
    cases = []
    def topo(name, body):
        path = os.path.join(tmp, f"topo-{name}.json")
        open(path, "wb").write(body if isinstance(body, (bytes, bytearray)) else body.encode())
        return path
    good = topo("good", json.dumps({"chips": 2, "links": []}))
    cases.append((["mpack", "--topology", good, "-o", os.path.join(tmp, "tg.otsm")] + objs, 0))
    # count mismatch: topology names 3 chips, 2 objects given
    mismatch = topo("mismatch", json.dumps({"chips": 3, "links": []}))
    cases.append((["mpack", "--topology", mismatch, "-o", os.path.join(tmp, "tm.otsm")] + objs, 1))
    # non-JSON
    non_json = topo("nonjson", "{not json")
    cases.append((["mpack", "--topology", non_json, "-o", os.path.join(tmp, "tn.otsm")] + objs, 1))
    # chips 0 / 513 / true
    for name, body, want in (
        ("chips0", '{"chips": 0, "links": []}', 1),
        ("chips513", '{"chips": 513, "links": []}', 1),
        ("chipstrue", '{"chips": true, "links": []}', 1),
    ):
        cases.append((["mpack", "--topology", topo(name, body), "-o", os.path.join(tmp, f"t-{name}.otsm")] + objs, want))
    for argv, want in cases:
        a, b = run_ref(mod, argv), run_new(exe, argv)
        compared += 1
        ok = a[0] == b[0] and a[1] == b[1] and prefix(a[2]) == prefix(b[2]) and "Traceback" not in a[2] + b[2]
        if want is not None:
            ok = ok and a[0] == want
        if not ok:
            bad += 1
            print(f"DIFF [mpack --topology {argv[2]}]\n  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r}")
    print(f"  mpack --topology: {compared} comparisons, {bad} differences")
    return compared, bad


def main(argv):
    exe = argv[1]
    n = int(argv[argv.index("--fuzz") + 1]) if "--fuzz" in argv else 2000
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else 1
    rnd = random.Random(seed)
    refs = references()
    base = programs()
    insns = sorted({r[0] for p in ("ref", "mini") for r in corpus(p, "accept.txt")})
    only = {k for k in ("insn", "weights", "manifest") if f"--{k}-only" in argv}
    cases = [(name, text, f"corpus {name}") for name, text in base]
    for c in LONG_CYCLES:
        cases.append((f"@cycle={c[:12]}... ({len(c)} digits)", LONG_PROGRAM.format(c=c), "long decimal cycle"))
    for _ in range(0 if only else n):
        name, text = rnd.choice(base)
        cases.append((name, mutate_program(rnd, text, insns), f"mutation of {name}"))
    bad = 0
    compared = 0
    seen = {}
    objects = {}          # object bytes -> the --config it was assembled under
    with tempfile.TemporaryDirectory() as tmp:
        compared += causal_profiles(exe, refs[0][1], tmp)
        n_bad, d_bad = bad_inputs(exe, refs[0][1], tmp)
        compared += n_bad
        bad += d_bad
        n_nk, d_nk = new_key_profile_phase(exe, refs[0][1], tmp)
        compared += n_nk
        bad += d_nk
        n_tp, d_tp = topology_phase(exe, refs[0][1], tmp)
        compared += n_tp
        bad += d_tp
        cfg = make_config(tmp)
        path = os.path.join(tmp, "p.ots")
        for name, text, what in cases:
            open(path, "wb").write(text.encode("latin-1", "replace"))
            for opts in ([], ["--config", cfg]):
                for rname, mod, flags in refs:
                    argv_ = ["asm", path, "--input"] + opts
                    a, b = run_ref(mod, argv_), run_new(exe, argv_ + flags)
                    compared += 1
                    # the same with `-o`: the object files, byte for byte
                    ro, no = os.path.join(tmp, "ref.elf"), os.path.join(tmp, "new.elf")
                    for f in (ro, no):
                        if os.path.exists(f):
                            os.remove(f)
                    a_o, b_o = run_ref(mod, ["asm", path, "-o", ro] + opts), run_new(exe, ["asm", path, "-o", no] + opts + flags)
                    compared += 1
                    ra = open(ro, "rb").read() if os.path.exists(ro) else None
                    rb = open(no, "rb").read() if os.path.exists(no) else None
                    if not same(a_o, b_o) or ra != rb:
                        bad += 1
                        print(f"DIFF [{what}; {rname}; {' '.join(opts) or 'shipped profiles'}; asm -o]\n--- program\n{text}\n---\n"
                              f"  tsp_as : {a_o[0]} {a_o[1]!r} {a_o[2].strip()!r} object {None if ra is None else ra.hex()}\n"
                              f"  ots-asm: {b_o[0]} {b_o[1]!r} {b_o[2].strip()!r} object {None if rb is None else rb.hex()}")
                    elif ra is not None:
                        objects.setdefault(ra, opts[1] if opts else None)
                    for p_ in prefix(a[2]) or ["accepted"]:
                        k_ = re.sub(r"^.*?: (OTS-E-\S+(?: \(warning\))?)$", r"\1", p_)
                        seen[k_] = seen.get(k_, 0) + 1
                    if not same(a, b):
                        bad += 1
                        print(f"DIFF [{what}; {rname}; {' '.join(opts) or 'shipped profiles'}]\n--- program\n{text}\n---\n"
                              f"  tsp_as : {a[0]} {a[1]!r} {a[2].strip()!r}\n  ots-asm: {b[0]} {b[1]!r} {b[2].strip()!r}")
                        if bad >= 10:
                            print(f"diff_tsp_as: stopped after {bad} differences")
                            return 1
        if not only:
            bad, compared = variant_phase(exe, refs, tmp, rnd, cases, objects, bad, compared)
            if bad >= 10:
                return 1
            bad, compared = dis_phase(exe, refs, cfg, tmp, rnd, n, cases, bad, compared, seen)
            if bad >= 10:
                return 1
            bad, compared = object_phase(exe, refs[0][1], cfg, tmp, rnd, n, objects, bad, compared)
            if bad >= 10:
                return 1
        for name, phase in (("insn", lambda: insn_phase(exe, refs[0][1], rnd, n, bad, compared)),
                            ("weights", lambda: weights_phase(exe, refs[0][1], cfg, tmp, rnd, n, objects, bad, compared)),
                            ("manifest", lambda: manifest_phase(exe, refs[0][1], cfg, tmp, rnd, n, objects, bad, compared))):
            if only and name not in only:
                continue
            bad, compared = phase()
            if bad >= 10:
                print(f"diff_tsp_as: stopped after {bad} differences")
                return 1
    print("  outcomes of tsp_as: " + ", ".join(f"{k} {v}" for k, v in sorted(seen.items())))
    print(f"diff_tsp_as: {len(cases)} programs, {compared} comparisons (asm --input and -o, dis and objinput of text and objects, insn and word, wpack and objinput --weights, mpack and mobjinput; {', '.join(r[0] for r in refs)}), seed {seed}, {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
