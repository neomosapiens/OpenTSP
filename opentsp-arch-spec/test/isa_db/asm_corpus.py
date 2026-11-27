"""Assembler acceptance corpora as checked-in files (readiness R5), per profile:

  test/isa_db/corpus/<profile>/accept.txt   <text> TAB <width> TAB <hex>
        the text assembles to this word (32: bits 31..0; 64: the whole word)
  test/isa_db/corpus/<profile>/reject.txt   <text> TAB <code>
        the text is rejected with this diagnostic code
  test/isa_db/corpus/<profile>/text.txt    <queue> TAB <unit0> TAB <unit1|-> TAB <code|-> TAB <text>
        the start of one queue's text (hex units; unit1 only for a two-unit instruction, whose
        first unit has a wide tag) through the text decode rule (encoding.md, encoding.json
        decode_rule) and placement: the model's code (OTS-E-DEC-003, OTS-E-QUE-002, or - for
        none) and the canonical text (illegal32/illegal64 0x... when it does not decode)
  test/isa_db/corpus/<profile>/groups.txt   <group> TAB <queue> TAB <word0> TAB <code|-> TAB <text>
        instructions that share one parcel of one queue (the rows of one <group> number; word0
        is the 64-bit word the lowering turns into units), through the slot rule of
        microarchitecture.md sec. 3 "Instruction front end" (encoding.json mem_slots): the model's
        group code per instruction (OTS-E-QUE-001, OTS-E-MEM-001, or -)

Lines starting with "#" are comments. The expected results are those of the reference
assembler tools/tsp_as.py, which is held to the Sail model by asm_roundtrip.py and
check_asm_examples.py; --verify also checks every corpus line against the model itself.

usage:
  asm_corpus.py --write                          regenerate the corpora (ref and mini)
  asm_corpus.py --check                          fail if the checked-in corpora are out of date
  asm_corpus.py --verify <isa_query> <tsp_sim> [profile]  check every line against the Sail model
        (isa_query 32/64 for accept, isa_query text and tsp_sim --trace for text and groups)
  asm_corpus.py --against <assembler> [profile]  run another assembler over the corpora; it
        is called as `<assembler> insn --profile <p> <text>` (stdout "<width> <hex>", exit 0;
        or exit 1 with "<code>:" in stderr) and `<assembler> dis --profile <p> <input.txt>`
        (the tsp_as program text with "; <code>" annotations; the input is Q lines), as tools/tsp_as.py is
"""
import os, random, re, shlex, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(SPEC, "tools"))
import check_isa_db as C
import tsp_as as A
from asm_roundtrip import bad_tokens

CORPUS = os.path.join(HERE, "corpus")
PROFILES = ("ref", "mini")
SEED = 20261108


def code_of(fn):
    try:
        fn()
        return None
    except A.AsmError as e:
        return e.code


def accept_reject(isa, rnd):
    acc, rej = [], []
    for e in isa.entries:
        trials = []
        for corner in (min, max):
            fv = {}
            for v in e["variables"]:
                vals = C.legal_values(v, isa.pvars)
                if vals is None:
                    vals = C.corner_values(v, isa.pvars)
                fv[v["name"]] = corner(vals)
            trials.append(fv)
        trials += [{v["name"]: C.pick(v, rnd, isa.pvars) for v in e["variables"]} for _ in range(4)]
        base = None
        for fv in trials:
            w = isa.encode(e, fv)
            if isa.decode(e["width"], w) is None:
                continue                 # a joint constraint (boolean cmp modes)
            text = isa.render(e, fv)
            base = base or fv
            acc.append((text, e["width"], w))
        # Lexical variants of one legal text (sec. 1): spacing, leading zeros, upper-case hex.
        if base is not None:
            t = isa.render(e, base)
            w = isa.encode(e, base)
            for var in (t.replace(", ", ","), t.replace(", ", "  ,  "), "  " + t + " ", re.sub(r"0x([0-9a-f]+)", lambda m: "0x00" + m.group(1).upper(), t),
                        re.sub(r"([EW]):([0-9]+)", r"\1:0\2", t)):
                if var != t:
                    acc.append((var, e["width"], w))
            for v in e["variables"]:
                for tok, why in bad_tokens(isa, e, v):
                    text = e["assembly"]
                    for u in e["variables"]:
                        text = text.replace("{" + u["name"] + "}", tok if u is v else C.render(u["kind"], base[u["name"]], e["width"]))
                    rej.append(text)
            for var in (t.replace(" ", "\t", 1), t.upper(), t.replace(e["mnemonic"], e["mnemonic"] + "x", 1)):
                if code_of(lambda: isa.parse(var)):      # e.g. "syn" + "c" may name another entry
                    rej.append(var)
    rej += ["", "nop", "nop 0x", "nop 0X1", "nop -0x1", "nop 1", "read 0x0 E:0, 0x0", "read 0x0, E:0, 0x0,", "add.i8 E:8, E:0",
            "read sym, E:0, 0x0", "café"]
    out_rej = []
    seen = set()
    for t in rej:
        if t in seen or "\n" in t:
            continue
        seen.add(t)
        c = code_of(lambda: isa.parse(t))
        if c is None:
            raise SystemExit(f"asm_corpus: reject candidate {t!r} is accepted")
        out_rej.append((t, c))
    dedup, seen = [], set()
    for a in acc:
        if a[0] not in seen:
            seen.add(a[0])
            got = isa.parse(a[0])
            if got[1] != a[2]:
                raise SystemExit(f"asm_corpus: {a[0]!r} assembles to {got[1]:#x}, not {a[2]:#x}")
            dedup.append(a)
    return dedup, out_rej


def text_rows(isa, rnd, n=1500):
    """Rows (queue, unit0, unit1 or None, code, text): the first instruction of a queue's text."""
    ctl = A.CONTROL
    queues = list(range(144)) + [ctl]
    out = []
    for i in range(n):
        e = rnd.choice(isa.entries)
        free = int(e["match"].replace("1", "0").replace("-", "1"), 2)
        w = int(e["match"].replace("-", "0"), 2) | (rnd.getrandbits(e["width"]) & free)
        k = i % 6
        if k == 1:
            w ^= 1 << rnd.choice([j for j, c in enumerate(reversed(e["match"])) if c != "-"])
        elif k == 2:
            w = rnd.getrandbits(64) if i % 12 == 2 else rnd.getrandbits(32)
        elif k == 3 and e["width"] == 32:
            w |= rnd.getrandbits(32) << 32
        elif k == 4:
            fv = {v["name"]: C.pick(v, rnd, isa.pvars) for v in e["variables"]}
            w = isa.encode(e, fv)
        q = ctl if i % 7 == 0 else rnd.choice(queues)
        if k == 5:                       # a legal word on one of its own queues
            fv = {v["name"]: C.pick(v, rnd, isa.pvars) for v in e["variables"]}
            w = isa.encode(e, fv)
            own = sorted(set().union(*(isa.t.units[u] for u in e["units"])))
            q = rnd.choice(own)
        if i % 5 == 3:                   # random units: any tag, 6 and 7 included
            units = [rnd.getrandbits(31), rnd.getrandbits(32)]
        else:
            units = A.word_units(w)
        units = units[:2 if (units[0] >> 28) & 7 in (1, 2, 4) else 1]
        units[0] &= 0x7FFFFFFF           # the bundle bit is the front end's (frontend.py), not the decoder's
        if units[-1] == A.PADDING:
            units[-1] = 0
        out.append((q, units[0], units[1] if len(units) > 1 else None) + text_result(isa, q, units))
    return out


def text_result(isa, q, units):
    code, e, f = isa.decode_text(q, units)
    if code:
        w = units[0] if len(units) == 1 else (units[0] << 32) | units[1]
        return code, f"illegal{32 if len(units) == 1 else 64} {w:#x}"
    return isa.placement(e, q) or "-", isa.render(e, f)


def groups(isa, rnd, n=400):
    """Multi-instruction parcels on one queue: every pair of slot classes (and some
    triples and quads) with random fields, on MEM and non-MEM queues."""
    by = lambda *names: [e for e in isa.entries if e["name"] in names]
    pool = by("Ifetch", "IfetchM", "Read", "Write", "Gather", "Scatter", "Countdown", "Step", "StreamStep", "Iterations",
              "NOP", "SetVL", "Repeat", "Sync")
    out = []
    for i in range(n):
        k = 2 + (i % 5 == 0) + (i % 15 == 0)
        q = rnd.randrange(88) if i % 9 else rnd.randrange(88, 144)
        g = []
        for _ in range(k):
            e = rnd.choice(pool)
            fv = {v["name"]: C.pick(v, rnd, isa.pvars) for v in e["variables"]}
            if "addr" in fv and i % 3 == 0:
                fv["addr"] &= 0x3                     # same/near addresses: bank clashes
            if "thread" in fv and i % 4 == 0:
                fv["thread"] = 0
            g.append(isa.encode(e, fv))
        if len(set(g)) != len(g):
            continue
        g.sort()
        codes = isa.group_codes(q, [isa.decode_text(q, A.word_units(w))[1:] for w in g])
        for w, c in zip(g, codes):
            _, e, f = isa.decode_text(q, A.word_units(w))
            if e is None:
                continue
            out.append((i, q, w, c or "-", isa.render(e, f)))
    return out


def render_files(profile):
    isa = A.ISA(profile)
    rnd = random.Random(SEED)
    acc, rej = accept_reject(isa, rnd)
    recs = text_rows(isa, rnd)
    grps = groups(isa, random.Random(SEED + 1))
    head = f"# generated by test/isa_db/asm_corpus.py --write (profile {profile}, spec version 0.1); do not edit\n"
    return {
        "accept.txt": head + "# text\twidth\thex\n" + "".join(f"{t}\t{wd}\t{w:x}\n" for t, wd, w in acc),
        "reject.txt": head + "# text\tcode\n" + "".join(f"{t}\t{c}\n" for t, c in rej),
        "text.txt": head + "# queue\tunit0\tunit1\tcode\ttext\n"
        + "".join(f"{q:x}\t{u0:08x}\t{'-' if u1 is None else format(u1, '08x')}\t{c}\t{t}\n" for q, u0, u1, c, t in recs),
        "groups.txt": head + "# group\tqueue\tword0\tcode\ttext\n"
        + "".join(f"{g}\t{q:x}\t{w0:x}\t{c}\t{t}\n" for g, q, w0, c, t in grps),
    }


def read(profile, name):
    rows = []
    for ln in open(os.path.join(CORPUS, profile, name), encoding="utf-8").read().split("\n"):
        if ln and not ln.startswith("#"):
            rows.append(ln.split("\t") if name != "reject.txt" else ln.rsplit("\t", 1))
    return rows


def verify(iq, sim, profiles=PROFILES):
    """Every corpus line against the Sail model (isa_query, tsp_sim)."""
    errors = []
    for profile in profiles:
        cfg = os.path.join(SPEC, "config", f"{profile}.json")
        m = C.Model(iq, cfg)
        isa = A.ISA(profile)
        for t, wd, w in read(profile, "accept.txt"):
            ok, text = m.query(int(wd), int(w, 16))
            if not ok or text.startswith("illegal") or A.ISA.parse(isa, text)[1] != int(w, 16):
                errors.append(f"{profile} accept {t!r}: the model prints {w} as {text!r}")
        rows = read(profile, "text.txt")
        for q, u0, u1, code, text in rows:
            # isa_query `text`: the text-level decode rule (porting-guide.md sec. 3).
            mc, _, mt = m.text(int(u0, 16), 0 if u1 == "-" else int(u1, 16))
            want = None if code in ("-", "OTS-E-QUE-002") else code
            if mc != want or (mc is None and mt != text):
                errors.append(f"{profile} text {q} {u0} {u1}: corpus {code} {text!r}, isa_query text {mc} {mt!r}")
        m.p.stdin.close()
        m.p.wait()
        # Codes. Round k runs the k-th row of every queue at once (one text of one or two units
        # per queue, dispatched at cycle 0); the model's DEC-003 and QUE-002 at cycle 0 are the
        # codes of the rows. A Sync on the control channel parks every queue at cycle 0, so those
        # rows run alone.
        def units_of(r):
            return [int(r[1], 16)] + ([] if r[2] == "-" else [int(r[2], 16)])
        def diags(imgs):
            out = subprocess.run([sim, cfg, "--trace"], input="\n".join(A.q_lines(imgs)) + "\n", capture_output=True, text=True, timeout=1200).stdout
            got = {}
            for ln in out.splitlines():
                t = ln.split()
                if ln.startswith("T diag") and int(t[2]) == 0 and t[5] in ("OTS-E-DEC-003", "OTS-E-QUE-002"):
                    got[int(t[4])] = t[5]
            return got
        solo = [i for i, r in enumerate(rows) if int(r[0], 16) == A.CONTROL and r[4].split(" ")[0] == "sync"]
        perq = {}
        for i, r in enumerate(rows):
            if i not in solo:
                perq.setdefault(int(r[0], 16), []).append(i)
        rounds = [[v[k] for v in perq.values() if k < len(v)] for k in range(max((len(v) for v in perq.values()), default=0))]
        rounds += [[i] for i in solo]
        def one_round(idx):
            got = diags({int(rows[i][0], 16): units_of(rows[i]) for i in idx})
            return [f"{profile} text {rows[i][:3]}: corpus {rows[i][3]}, model {got.get(int(rows[i][0], 16))}"
                    for i in idx if got.get(int(rows[i][0], 16)) != (None if rows[i][3] == "-" else rows[i][3])]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(4) as ex:
            for e in ex.map(one_round, rounds):
                errors += e
        # Groups: round k runs the k-th group of every queue as one parcel at cycle 0; the model's
        # QUE-001 / MEM-001 of the queue at cycle 0 are the group's codes.
        gs = {}
        for g, q, w0, code, text in read(profile, "groups.txt"):
            gs.setdefault(int(g), []).append((int(q, 16), int(w0, 16), code))
        perq = {}
        for g, rs in sorted(gs.items()):
            perq.setdefault(rs[0][0], []).append(g)
        grounds = [[v[k] for v in perq.values() if k < len(v)] for k in range(max((len(v) for v in perq.values()), default=0))]
        def one_group_round(gl):
            imgs = {}
            for g in gl:
                units = []
                for j, (_, w0, _) in enumerate(gs[g]):
                    u = A.word_units(w0)
                    if j < len(gs[g]) - 1:
                        u[0] |= A.BUNDLE_BIT
                    units += u
                imgs[gs[g][0][0]] = units
            out = subprocess.run([sim, cfg, "--trace"], input="\n".join(A.q_lines(imgs)) + "\n", capture_output=True, text=True, timeout=1200).stdout
            seq = {}
            for ln in out.splitlines():
                t = ln.split()
                if ln.startswith("T diag") and int(t[2]) == 0 and t[5] in ("OTS-E-QUE-001", "OTS-E-MEM-001"):
                    seq.setdefault(int(t[4]), []).append(t[5])
            errs = []
            for g in gl:
                q = gs[g][0][0]
                want = sorted(c for _, _, c in gs[g] if c != "-")
                if sorted(seq.get(q, [])) != want:
                    errs.append(f"{profile} group {g}: corpus {want}, model {sorted(seq.get(q, []))}")
            return errs
        with ThreadPoolExecutor(4) as ex:
            for e in ex.map(one_group_round, grounds):
                errors += e
    return errors


def run_dis(exe, profile, imgs, what, expect):
    """Run `<assembler> dis` over Q lines; expect: [(cycle, queue, code, text, row)] in listing order."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(A.input_text(imgs))
    r = subprocess.run(exe + ["dis", "--profile", profile, f.name], capture_output=True, text=True)
    os.unlink(f.name)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("  ")]
    if len(lines) != len(expect):
        return [f"{what}: {len(lines)} disassembled lines for {len(expect)} instructions"]
    errors = []
    for (k, q, code, text, row), ln in zip(expect, lines):
        body = ln.split(": ", 1)[1]
        got_text, _, note = body.partition("   ; ")
        got = note.split(":")[0] if note else "-"
        if got != code or (text and got_text != text):
            errors.append(f"{what} {row}: want {code} {text!r}, got {got} {got_text!r}")
    return errors


def against(cmd, profiles):
    errors = []
    exe = shlex.split(cmd)
    for profile in profiles:
        for t, wd, w in read(profile, "accept.txt"):
            r = subprocess.run(exe + ["insn", "--profile", profile, t], capture_output=True, text=True)
            if r.returncode != 0 or r.stdout.split() != [wd, w]:
                errors.append(f"{profile} accept {t!r}: want {wd} {w}, got rc {r.returncode} {r.stdout.strip()!r} {r.stderr.strip()!r}")
        for t, code in read(profile, "reject.txt"):
            r = subprocess.run(exe + ["insn", "--profile", profile, t], capture_output=True, text=True)
            if r.returncode == 0 or code + ":" not in r.stderr:
                errors.append(f"{profile} reject {t!r}: want {code}, got rc {r.returncode} {r.stdout.strip()!r} {r.stderr.strip()!r}")
        rows = read(profile, "text.txt")
        # Each queue's rows are one text image, one parcel each; a NOP is a timing prefix and has no listing line.
        imgs, expect = {}, []
        for row in rows:
            q = int(row[0], 16)
            if row[4].startswith("nop "):
                continue
            k = sum(1 for e in expect if e[1] == q)
            expect.append((k, q, row))
            imgs.setdefault(q, []).extend([int(row[1], 16)] + ([] if row[2] == "-" else [int(row[2], 16)]))
        imgs = {q: [u for u in us if u != A.PADDING] for q, us in imgs.items()}
        errors += run_dis(exe, profile, imgs, f"{profile} text",
                          [(k, q, row[3], row[4] or row[4], row) for k, q, row in sorted(expect, key=lambda e: (e[0], e[1] == A.CONTROL, e[1]))])
        # Groups: one parcel per group; the listing gives each instruction's group code.
        rows = read(profile, "groups.txt")
        imgs, expect, gl, nq = {}, [], {}, {}
        for g, q, w0, code, text in rows:
            gl.setdefault(int(g), []).append((int(q, 16), int(w0, 16), code, text))
        for g, rs in sorted(gl.items()):
            q = rs[0][0]
            k = nq.get(q, 0)
            nq[q] = k + 1
            units = []
            for j, (_, w0, code, text) in enumerate(rs):
                u = A.word_units(w0)
                if j < len(rs) - 1:
                    u[0] |= A.BUNDLE_BIT
                units += u
                expect.append((k, q, code, text, (g, q, w0)))
            imgs.setdefault(q, []).extend(units)
        errors += run_dis(exe, profile, imgs, f"{profile} group",
                          sorted(((k, q, code, text, row) for k, q, code, text, row in expect), key=lambda e: (e[0], e[1] == A.CONTROL, e[1])))
    return errors


def main(argv):
    if "--write" in argv or "--check" in argv:
        stale = []
        for profile in PROFILES:
            for name, text in render_files(profile).items():
                p = os.path.join(CORPUS, profile, name)
                if "--write" in argv:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    open(p, "w", encoding="utf-8").write(text)
                elif not os.path.exists(p) or open(p, encoding="utf-8").read() != text:
                    stale.append(p)
        if stale:
            print("assembler corpora out of date (run test/isa_db/asm_corpus.py --write):", *stale)
            return 1
        n = {p: [len(read(p, f)) for f in ("accept.txt", "reject.txt", "text.txt", "groups.txt")] for p in PROFILES}
        print(f"assembler corpora {'written' if '--write' in argv else 'up to date'}: "
              + ", ".join(f"{p} accept {a} reject {r} text {d} group instructions {g}" for p, (a, r, d, g) in n.items()))
        return 0
    if "--verify" in argv:
        i = argv.index("--verify")
        errors = verify(argv[i + 1], argv[i + 2], argv[i + 3:i + 4] or PROFILES)
        print(f"assembler corpora against the model: {len(errors)} differences")
    elif "--against" in argv:
        i = argv.index("--against")
        errors = against(argv[i + 1], argv[i + 2:i + 3] or PROFILES)
        print(f"assembler corpora against {argv[i + 1]}: {len(errors)} differences")
    else:
        print(__doc__, file=sys.stderr)
        return 2
    for e in errors[:30]:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
