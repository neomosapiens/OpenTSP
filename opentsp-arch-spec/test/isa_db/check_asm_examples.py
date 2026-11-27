"""The assembly.md examples and program-file rules, through the reference assembler.

1. Every example of assembly.md sec. 2 assembles with tools/tsp_as.py, and the model
   prints the word back identically.
2. The example program of sec. 3 assembles; its text images run on the emulator (tsp_sim)
   with the result the document states (SRAM[44][9] = x + y in every lane).
3. Objects: Q-line text and binary .elf round-trip, a disassembled program reassembles
   to the same images. Unrelated ELF files, unsupported ABIs and profile
   mismatches are rejected.
4. The lexical rules of sec. 1, the queue-name table of sec. 3 and the error classes of
   sec. 5, each on positive and negative texts.
5. With tsp_sim: placement and disassembler codes against the model, and the decoder
   built from encoding.json alone (json_decoder.py) and the checked-in corpora
   (asm_corpus.py --verify) against the model. Always: asm_corpus.py --check.

usage: check_asm_examples.py <isa_query> <config.json> [tsp_sim]
"""
import json
import os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import check_isa_db as C
import tsp_as as A

DOC = os.path.join(HERE, "..", "..", "docs", "assembly.md")
bad = []


def check(name, cond, info=""):
    if not cond:
        bad.append(f"{name}: {info}")


def code_of(fn):
    try:
        fn()
        return None
    except A.AsmError as e:
        return e.code


def prog(body, profile="ref", head=""):
    return f".arch opentsp-tsa\n.profile {profile}\n{head}.section text\n{body}"


def obj_bytes(images, profile="ref", exts=(), *, name=None, config=None):
    """Assemble ELF sections around independent raw image and extern records."""
    enc, cfg = A.full_interface_digests(profile, config)
    note_name = A.ELF_NOTE_NAME
    note = A.ELF_NOTE.pack(len(note_name), A.ELF_ABI.size, 1) + note_name
    note += A.ELF_ABI.pack(1, 0, name if name is not None else profile.encode(), enc, cfg)
    iq = len(images).to_bytes(4, "little") + b"".join(A.IMAGE.pack(q, 0, len(us)) for q, us in images)
    iq += b"".join(u.to_bytes(4, "little") for q, us in images for u in us)
    loads = len(exts).to_bytes(4, "little") + b"".join(A.EXTERN.pack(*e) for e in exts)
    return A._write_elf([(".note.opentsp", 7, 0, note, 4), (".tsp.iq", 1, 0, iq, 4),
                         (".tsp.load", 1, 0, loads, 4)])


def main(argv):
    exe, cfg = argv[1], argv[2]
    sim = argv[3] if len(argv) > 3 else None
    profile = os.path.splitext(os.path.basename(cfg))[0]
    # A test profile (config/test/<name>.json) is found by its file name, not under config/: the assembler's lookup by profile name
    # then reads the file given on the command line.
    load, digests, full_digests = A.load_profile, A.interface_digests, A.full_interface_digests
    A.load_profile = lambda n, config=None: load(n, config or (cfg if n == profile else None))
    A.interface_digests = lambda n, config=None: digests(n, config or (cfg if n == profile else None))
    A.full_interface_digests = lambda n, config=None: full_digests(n, config or (cfg if n == profile else None))
    isa = A.ISA(profile, cfg)
    m = C.Model(exe, cfg)
    doc = open(DOC).read()
    blocks = re.findall(r"```\n(.*?)```", doc, re.S)

    # 1. Section 2 examples.
    sec2 = blocks[0]
    for line in sec2.strip().splitlines():
        t = A.strip_comment(line).strip(" ")
        try:
            e, w = isa.parse(t)
        except A.AsmError as err:
            check("sec. 2 example", False, f"{t!r}: {err}")
            continue
        _, text = m.query(e["width"], w)
        check("sec. 2 example prints back", text == t, f"{t!r} -> {text!r}")

    # isa_query `text`: a legal two-unit and one-unit instruction decode, and tags 6 and 7 are DEC-003 (one unit).
    check("isa_query text: Read decodes", m.text(0x10000000, 0)[0] is None)
    check("isa_query text: Sync decodes", m.text(0x00400000, 0)[0] is None)
    for tag in (6, 7):
        r = m.text((tag << 28) | 5, 0)
        check(f"isa_query text: tag {tag} is OTS-E-DEC-003 and consumes one unit", r[0] == "OTS-E-DEC-003" and r[1] == 1, r)

    # 2. Section 3 example program.
    example = next(b for b in blocks if b.startswith(".arch"))
    if profile == "ref":
        _, p = A.assemble(example, "assembly.md")
        check("sec. 3 example: instructions", len(p.records) == 4, p.records)
        for c, q, wc, w0, w1 in p.records:
            width = isa.width_of(w0)
            _, text = m.query(width, w0)
            check("sec. 3 example: model decodes", not text.startswith("illegal"), text)
        if sim:
            # The externs are filled from a weight package, through the object loader.
            x, y = 3, 4
            vl = isa.pvars["vlanes"]
            wp = A.write_weights(profile, [("x", "i8", [bytes([x]) * vl]), ("y", "i8", [bytes([y]) * vl])])
            _, imgs3, exts3 = A.read_object(A.write_object(p), profile)
            pre = A.preload_text(A.bind_weights(exts3, A.read_weights(wp, profile)))
            out = subprocess.run([sim, cfg], input=pre + A.input_text(imgs3) + "D 2c 9\n",
                                 capture_output=True, text=True, timeout=600).stdout.split("\n")
            check("sec. 3 example: no diagnostics", out[0].startswith("E - "), out[0])
            word = out[1].split()[3:]
            check("sec. 3 example: SRAM[44][9] = x + y", word == ["%02x" % (x + y)] * 320, out[1][:80])

    # (assembly.md sec. 4): the instructions of one bundle are one parcel in SOURCE order, not in word
    # order. The Write has the larger word and comes first in the source; the Read follows.
    par = (".arch opentsp-tsa\n.profile " + profile + "\n.section text\nbundle @cycle=2 {\n"
           "  MEM[4]: write 0x20, E:0, 0x0\n  MEM[4]: read 0x1010, E:4, 0x0\n}\n")
    _, pp = A.assemble(par, "parcel.ots")
    ww, wr = isa.parse("write 0x20, E:0, 0x0")[1], isa.parse("read 0x1010, E:4, 0x0")[1]
    check("parcel: records keep source order (the Write has the larger word)",
          ww > wr and [r[3] for r in pp.records] == [ww, wr], pp.records)
    check("parcel: the text image has the instructions in source order",
          pp.images[4] == [2] + [A.BUNDLE_BIT | A.word_units(ww)[0]] + A.word_units(ww)[1:] + A.word_units(wr), pp.images[4])
    ptext, pbad = A.disassemble(isa, pp.images)
    check("parcel: disassembly lists source order and reassembles to the same records",
          pbad == 0 and A.assemble(ptext)[1].records == pp.records and ptext.index("write") < ptext.index("read"), ptext)

    if sim:
        # Placement: every entry on every data queue and on the control
        # channel; tsp_as's placement code (QUE-002) must be the model's, and the disassembler
        # must annotate the same code. One word per queue, all at cycle 0; the control channel
        # runs apart from the data queues (a Sync there would park them before they dispatch).
        want, images = {}, {}
        declared = isa.cfg.get("extensions", [])

        def first_legal(v):
            """The first value of a variable that the profile decodes: not a code of an extension it does not declare."""
            vals = v.get("values") or v.get("codes") or [1 if v.get("not_values") else 0]
            held = {c for ext, cs in v.get("extension_codes", {}).items() if ext not in declared for c in cs}
            return next((x for x in vals if x not in held), vals[0])

        for e in isa.entries:
            fv = {v["name"]: first_legal(v) for v in e["variables"]}
            w = isa.encode(e, fv)
            for qs in (list(range(144)), [A.CONTROL]):
                out = subprocess.run([sim, cfg, "--trace"], input=A.input_text({q: A.word_units(w) for q in qs}), capture_output=True, text=True, timeout=1200).stdout
                got = {int(ln.split()[4]): ln.split()[5] for ln in out.splitlines()
                       if ln.startswith("T diag") and int(ln.split()[2]) == 0 and ln.split()[5] in ("OTS-E-QUE-002", "OTS-E-DEC-003")}
                for q in qs:
                    want[(e["name"], q)] = (isa.placement(e, q), got.get(q))
                    images.setdefault(q, []).extend(A.word_units(w))
        nbad = sum(1 for k, (a, g) in want.items() if a != g)
        check("placement = model (every entry x every queue)", nbad == 0, [(k, v) for k, v in want.items() if v[0] != v[1]][:5])
        dis, nd = A.disassemble(isa, images)
        check("disassembler annotates every placement error", nd == sum(1 for a, g in want.values() if a), nd)

        # The disassembler's decode order and codes are the model's: DEC-003 and QUE-002
        # for a single instruction, QUE-001 / MEM-001 for the instructions of one parcel.
        add = isa.parse("add.i8 E:8, E:0, E:4")[1]
        other_bank0 = 1 << isa.cfg["sram"]["bank_bit"]
        rd, wr = isa.parse("read 0x0, E:0, 0x0")[1], isa.parse("write 0x0, E:1, 0x0")[1]
        rd1 = isa.parse(f"read {other_bank0:#x}, E:0, 0x0")[1]
        cases = [(0xFFFF, [0x1000000000000000]), (0xFFFF, [add]), (88, [0x100400000]), (0, [0x400000]),
                 (88, [0x2fc0000000000000]), (88, [rd, add]), (0, [rd, wr]), (0, [wr, rd1]),
                 (88, [0xffffffffffffffff, add]), (0xFFFF, [isa.parse("repeat 0x1, 0x1")[1]]), (0xFFFF, [0x0fc00000])]
        # MEM slot rule (microarchitecture.md sec. 3 "Instruction front end"): one
        # Ifetch/IfetchM, per liquid thread one R and one W reference, one configuration record;
        # any other record alone. Every accepted and rejected combination against the model.
        P = lambda t: isa.parse(t)[1]
        ob = other_bank0
        slot_groups = [
            ["read 0x0, E:0, 0x0", "read 0x1, E:1, 0x1"],                        # R t0 + R t1
            ["ifetch E:6", "read 0x0, E:0, 0x0"],                                 # Ifetch + R
            ["ifetchm 0x10", "read 0x0, E:0, 0x0"],                               # IfetchM + R
            ["ifetchm 0x10", "read 0x0, E:0, 0x0", f"write {ob:#x}, E:1, 0x0"],    # IfetchM + R + W
            ["ifetchm 0x10", "read 0x0, E:0, 0x1"],                               # IfetchM + R t1: QUE-001
            ["ifetchm 0x10", f"write {ob:#x}, E:1, 0x0"],                          # IfetchM + W other bank
            ["ifetchm 0x10", "write 0x1, E:1, 0x1"],                              # IfetchM + W t1 same bank: MEM-001
            [f"ifetchm {ob - 1:#x}", f"write {ob:#x}, E:1, 0x0"],                  # second word's bank is used next cycle: legal
            ["ifetch E:6", "read 0x0, E:0, 0x0", f"write {ob:#x}, E:1, 0x0"],      # Ifetch + R + W
            ["write 0x0, E:1, 0x0", "write 0x1, E:2, 0x1"],                       # W t0 + W t1
            ["read 0x0, E:0, 0x0", "read 0x1, E:1, 0x1", f"write {ob:#x}, E:2, 0x0", f"write {ob + 1:#x}, E:3, 0x1"],
            ["read 0x1, E:0, 0x0", "write 0x1, E:1, 0x1"],                        # same address, other threads
            ["gather W:3, E:1, 0x2, 0x0", "scatter E:2, E:1, 0x2, 0x1"],           # Gather t0 + Scatter t1
            ["gather W:3, E:1, 0x2, 0x0", f"write {ob:#x}, E:2, 0x0"],             # indirect, same thread: MEM-001
            ["read 0x0, E:0, 0x1", "scatter E:2, E:1, 0x2, 0x1"],                 # indirect, same thread: MEM-001
            ["read 0x0, E:0, 0x0", "write 0x1, E:1, 0x0"],                        # same bank: MEM-001
            ["read 0x0, E:0, 0x0", "step 0x2"],                                   # reference + configuration
            ["ifetch E:6", "ifetchm 0x10"],                                       # two fetches: QUE-001
            ["read 0x0, E:0, 0x0", "gather W:3, E:1, 0x2, 0x0"],                  # two R t0: QUE-001
            ["step 0x2", "iterations 0x2"],                                       # two configurations: QUE-001
            ["read 0x0, E:0, 0x0", "nop 0x1"],                                    # whole-queue record + other
            ["setvl 0x1", "ifetch E:6"],
        ]
        for g in slot_groups:
            cases.append((4, [P(t) for t in g]))
        cases.append((4, [P("read 0x0, E:0, 0x0"), 0x1fc0000000000000]))   # undecodable among several
        cases.append((88, [add, P("ifetch E:6")]))                          # non-MEM queue: one instruction
        # IfetchM / Ifetch fetch text; the model reads padding (0xFF) from the slices so that it adds no diagnostics.
        vl = isa.pvars["vlanes"]
        pad = "".join(f"S 4 {a:x} " + " ".join(["ff"] * vl) + "\n" for a in (0x10, 0x11, other_bank0 - 1, other_bank0))
        for q, words in cases:
            rs = [(0, q, 1, w, 0) for w in words]
            imgs = A.lower_records(rs)
            out = subprocess.run([sim, cfg, "--trace"], input=pad + A.input_text(imgs), capture_output=True, text=True, timeout=600).stdout
            static = ("OTS-E-DEC-", "OTS-E-QUE-001", "OTS-E-QUE-002", "OTS-E-MEM-001")     # run-time codes (TIM-001) aside
            model = sorted(ln.split()[5] for ln in out.splitlines() if ln.startswith("T diag") and ln.split()[5].startswith(static))
            text, nb = A.disassemble(isa, imgs)
            mine = sorted(re.findall(r"; (OTS-E-[A-Z]+-[0-9]+)", text))
            check("disassembler codes = model", mine == model and nb == len(model), (rs, model, mine))
            if not model:
                _, p2 = A.assemble(text)
                check("clean disassembly reassembles", p2.records == rs, text)       # one queue, one cycle: source order is the order of rs
            if len(rs) > 1:
                # The assembler and the object loader apply the same rule: a group the model
                # dispatches without QUE-001 assembles and loads; one that is QUE-001 does neither.
                body = "bundle @cycle=0 {\n" + "".join(f"  {A.queue_name(q)}: {isa.render(*isa.decode(isa.width_of(w0, q), w0))}\n"
                                                      for _, q, _, w0, _ in rs if isa.decode(isa.width_of(w0, q), w0)) + "}\n"
                if len({r[3] for r in rs}) == len(rs) and all(isa.decode(isa.width_of(r[3], r[1]), r[3]) for r in rs):
                    want = "OTS-E-QUE-001" if "OTS-E-QUE-001" in model else "OTS-E-MEM-001" if "OTS-E-MEM-001" in model else None
                    check("assembler group code = model", code_of(lambda: A.assemble(prog(body, profile))) == want, (body, model))
                ld = code_of(lambda: A.read_object(obj_bytes(sorted(imgs.items(), key=lambda kv: (kv[0] == A.CONTROL, kv[0])), profile)))
                check("loader group rule = model", ld == ("OTS-E-ASM-003" if "OTS-E-QUE-001" in model else None), (rs, model, ld))

    # 3. Objects.
    src = prog("bundle @cycle=3 {\n  CU: notify\n  MEM[1]: read w+0x1, W:2, 0x0\n  MXM[2].ACC: acc.f32 E:8, replace, 0\n}\n"
               "bundle @cycle=7 {\n  SXM.E.SEL: sxselect W:0, W:1, W:2, W:4, E:3\n}\n", profile,
               ".extern w, slice=1, addr=0x20, size=0x4\n")
    _, p = A.assemble(src)
    obj = A.write_object(p)
    check("ELF32 and ABI note", obj[:7] == A.ELF_IDENT[:7] and obj[76:92].rstrip(b"\0") == profile.encode(), obj[:12])
    check("full interface identity", obj[92:156] == b"".join(A.full_interface_digests(profile)), obj[92:156].hex())
    for why, off, val in [("another format revision", 72, b"\x02"), ("a reserved note field", 74, b"\x01"),
                          ("another encoding.json", 92, bytes([obj[92] ^ 1])), ("another profile configuration", 124, bytes([obj[124] ^ 1]))]:
        badobj = obj[:off] + val + obj[off + len(val):]
        check(f"object built against {why} rejected", code_of(lambda: A.read_object(badobj, profile)) == "OTS-E-ASM-003")
    prof, imgs, exts = A.read_object(obj, profile)
    check("object round trip", prof == profile and imgs == p.images and exts == list(p.externs.values()), (imgs, exts))
    check("text images round trip", A.parse_images(A.input_text(p.images)) == p.images)
    text, nbad = A.disassemble(isa, imgs, exts)
    _, p2 = A.assemble(text)
    check("disassemble then assemble", nbad == 0 and p2.records == p.records, text)
    check("disassembly uses symbols' resolved address", "read 0x21, W:2, 0x0" in text, text)
    elf = b"\x7fELF" + bytes(60)
    check("unrelated ELF rejected", code_of(lambda: A.read_object(elf)) == "OTS-E-ASM-003")
    check("other version rejected", code_of(lambda: A.read_object(obj[:5] + b"\x02" + obj[6:])) == "OTS-E-ASM-003")
    check("other profile rejected", code_of(lambda: A.read_object(obj, "other")) == "OTS-E-ASM-003")
    check("truncated object rejected", code_of(lambda: A.read_object(obj[:-1])) == "OTS-E-ASM-003")
    badq = obj_bytes([(0x90,[1])], profile)
    check("image with queue 0x90 rejected", code_of(lambda: A.read_object(badq)) == "OTS-E-ASM-003")
    _, illeg = A.disassemble(isa, A.lower([(0, 88, 0x2fc0000000000000)]))
    check("illegal word is reported by the disassembler", illeg == 1)
    # Object robustness: every malformed field is OTS-E-ASM-003, never a crash.
    def obj_with(recs=None, exts=None, prof=None):
        r = sorted(p.images.items(), key=lambda kv: (kv[0] == A.CONTROL, kv[0])) if recs is None else recs
        x = [(e["name"].encode(), e["slice"], 0, e["addr"], e["size"], 0) for e in p.externs.values()] if exts is None else exts
        return obj_bytes(r, profile, x, name=prof)
    def rd_code(data):
        try:
            A.read_object(data)
            return None
        except A.AsmError as e:
            return e.code
        except Exception as e:
            return f"crash {type(e).__name__}"
    check("object rebuilt by the test reads back", rd_code(obj_with()) is None)
    r0 = sorted(p.images.items(), key=lambda kv: (kv[0] == A.CONTROL, kv[0]))
    two = A.lower([(0, 88, 0x2002001000000000), (0, 88, 0x2002001000000001)])[88]
    cap = isa.cfg["icu"]["iq_depth"] // 4
    for why, data in [("non-ASCII profile", obj_with(prof=b"r\xe9f")), ("bytes after the profile NUL", obj_with(prof=profile.encode() + b"\0x")),
                      ("unknown profile", obj_with(prof=b"nosuch")),
                      ("non-ASCII extern name", obj_with(exts=[(b"w\xe9", 1, 0, 0x20, 4, 0)])),
                      ("extern pad 1", obj_with(exts=[(b"w", 1, 1, 0x20, 4, 0)])), ("extern pad 2", obj_with(exts=[(b"w", 1, 0, 0x20, 4, 7)])),
                      ("extern slice 999", obj_with(exts=[(b"w", 999, 0, 0x20, 4, 0)])), ("extern size 0", obj_with(exts=[(b"w", 1, 0, 0x20, 0, 0)])),
                      ("extern beyond the slice", obj_with(exts=[(b"w", 1, 0, isa.words - 1, 0xffffffff, 0)])),
                      ("extern bad name", obj_with(exts=[(b"1w", 1, 0, 0x20, 4, 0)])),
                      ("extern name with a final line feed", obj_with(exts=[(b"w\n", 1, 0, 0x20, 4, 0)])),
                      ("duplicate extern", obj_with(exts=[(b"w", 1, 0, 0x20, 4, 0), (b"w", 2, 0, 0x20, 4, 0)])),
                      ("overlapping externs", obj_with(exts=[(b"a", 1, 0, 0x0, 0x10, 0), (b"b", 1, 0, 0x8, 0x10, 0)])),
                      ("images out of order", obj_with(recs=list(reversed(r0)))),
                      ("duplicate image", obj_with(recs=[r0[0], r0[0]] + r0[1:])),
                      ("an empty image", obj_with(recs=[(5, [])])),
                      ("an image longer than the IQ", obj_with(recs=[(5, [0] * 0 + [1] * (cap + 1))])),
                      ("two instructions of one queue in one parcel", obj_with(recs=[(88, two)]))]:
        check(f"object: {why} rejected with OTS-E-ASM-003", rd_code(data) == "OTS-E-ASM-003", rd_code(data))
    rdw = [(0, A.lower([(0, 0, isa.parse("read 0x0, E:0, 0x0")[1]), (0, 0, isa.parse(f"write {1 << isa.cfg['sram']['bank_bit']:#x}, E:1, 0x0")[1])])[0])]
    for why, g in [("read t0 + read t1", ["read 0x0, E:0, 0x0", "read 0x1, E:1, 0x1"]), ("ifetch + read", ["ifetch E:6", "read 0x0, E:0, 0x0"]),
                   ("read + configuration", ["read 0x0, E:0, 0x0", "step 0x2"])]:
        rs = [(0, A.lower([(0, 0, isa.parse(t)[1]) for t in g])[0])]
        check(f"object: {why} is accepted", rd_code(obj_with(recs=rs, exts=[])) is None, rd_code(obj_with(recs=rs, exts=[])))
    rs = [(0, A.lower([(0, 0, isa.parse(t)[1]) for t in ["read 0x0, E:0, 0x0", "nop 0x1"]])[0])]
    check("object: read + nop is rejected", rd_code(obj_with(recs=rs, exts=[])) == "OTS-E-ASM-003")
    check("object: a MEM read + write pair is accepted", rd_code(obj_with(recs=rdw, exts=[])) is None, rd_code(obj_with(recs=rdw, exts=[])))

    # Weight packages (assembly.md sec. 4).
    vl = isa.pvars["vlanes"]
    words = [bytes((k + i) & 255 for i in range(vl)) for k in range(4)]
    wp = A.write_weights(profile, [("w", "f16", words), ("spare", None, words[:1])])
    t = A.read_weights(wp, profile)
    check("weights round trip", t["w"] == (isa.t.eltype_codes["f16"], words) and t["spare"] == (255, words[:1]))
    pre = A.bind_weights(exts, t)
    check("weights bind to the externs", pre == [(1, 0x20 + k, words[k]) for k in range(4)], pre[:1])
    check("weights preload lines", A.preload_text(pre).splitlines()[0] == "S 1 20 " + " ".join(f"{b:x}" for b in words[0]))
    def w_code(f):
        try:
            f()
            return None
        except A.AsmError as e:
            return e.code
    size = int.from_bytes(wp[:8], "little")

    def change_header(update):
        header = json.loads(wp[8:8+size])
        update(header)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        return len(encoded).to_bytes(8, "little") + encoded + wp[8+size:]

    for why, data, code in [
            ("bad length", b"OTSX" + wp[4:], "OTS-E-ASM-003"),
            ("legacy carrier", b"OTSW" + wp[4:], "OTS-E-ASM-003"),
            ("another profile", A.write_weights("mini" if profile == "ref" else "ref", []), "OTS-E-ASM-003"),
            ("truncated", wp[:-1], "OTS-E-ASM-003"),
            ("trailing bytes", wp + b"\0", "OTS-E-ASM-003"),
            ("corrupt image", wp[:-1] + bytes([wp[-1] ^ 1]), "OTS-E-ASM-004"),
            ("wrong dtype", change_header(lambda h: h["spare"].update(dtype="F32")), "OTS-E-ASM-003")]:
        check(f"weight package: {why} rejected with {code}", w_code(lambda: A.read_weights(data, profile)) == code, w_code(lambda: A.read_weights(data, profile)))

    def with_name(raw, k=0):
        def rename(header):
            old = ("spare", "w")[k]
            new = raw.decode("ascii")
            header[new] = header.pop(old)
            meta = header["__metadata__"]
            for prefix in ("dtype.", "sha256."):
                meta[prefix + new] = meta.pop(prefix + old)
        return change_header(rename)
    check("weight package: a valid name read back is accepted", w_code(lambda: A.read_weights(with_name(b"_a1"), profile)) is None)
    check("weight package: the longest name (15 characters) is accepted", w_code(lambda: A.read_weights(with_name(b"A" * 15), profile)) is None)
    for why, raw, k in [("starts with a digit", b"1x", 0), ("has a hyphen", b"a-b", 0), ("has a space", b"a b", 0),
                        ("has a dot", b"a.b", 0), ("ends in a line feed", b"x\n", 0), ("fills the field (16 characters)", b"B" * 16, 0),
                        ("has a control character", b"a\x01b", 0), ("is bad in the second tensor", b"9", 1)]:
        check(f"weight package: a tensor name that {why} is rejected with OTS-E-ASM-003",
              w_code(lambda: A.read_weights(with_name(raw, k), profile)) == "OTS-E-ASM-003", w_code(lambda: A.read_weights(with_name(raw, k), profile)))
    check("weights: a missing tensor is OTS-E-ASM-004", w_code(lambda: A.bind_weights(exts, {"spare": t["spare"]})) == "OTS-E-ASM-004")
    check("weights: a tensor of another size is OTS-E-ASM-004", w_code(lambda: A.bind_weights(exts, {"w": t["spare"]})) == "OTS-E-ASM-004")

    # 4a. Lexical rules (sec. 1).
    ok_texts = {
        "add.i8 E:8,E:0,E:4": "add.i8 E:8, E:0, E:4",          # zero spaces after ','
        "add.i8   E:8 ,  E:0 , E:4": "add.i8 E:8, E:0, E:4",    # spaces before and after ','
        "read 0x0010, E:0, 0x0": "read 0x10, E:0, 0x0",         # leading zeros
        "read 0X10, E:0, 0x0": None,                                  # 0X is not hex
        "read 0x1F, E:0, 0x0": "read 0x1f, E:0, 0x0",           # upper-case digits accepted
        "add.i8 E:007, E:0, E:4": "add.i8 E:7, E:0, E:4",       # decimal leading zeros
        "acc.i32 E:0, replace, 01": "acc.i32 E:0, replace, 1",
        "add.i8\tE:8, E:0, E:4": None,                           # tab
        "ADD.i8 E:8, E:0, E:4": None,                            # upper-case mnemonic
        "add.I8 E:8, E:0, E:4": None,
        "add.i8E:8, E:0, E:4": None,                             # no space after the mnemonic
        "read -0x1, E:0, 0x0": None,
        "read 0x, E:0, 0x0": None,
    }
    for t, want in ok_texts.items():
        try:
            e, w = isa.parse(t)
            got = m.query(e["width"], w)[1]
        except A.AsmError:
            got = None
        check("lexical rule", got == want, f"{t!r} -> {got!r}, want {want!r}")
    for t in ("# x", "// x"):
        check("only ';' starts a comment", code_of(lambda: A.assemble(prog(t + "\n", profile))) == "OTS-E-ASM-001", t)
    check("';' comment", code_of(lambda: A.assemble(prog("; only a comment\nbundle @cycle=0 { ; c\n  CU: sync ; c\n} ; c\n", profile))) is None)
    check("tab in program", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n\tCU: sync\n}\n", profile))) == "OTS-E-ASM-001")
    # Sec. 1 holds on the whole line, comments included.
    check("tab in a comment", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n  CU: sync ;\tc\n}\n", profile))) == "OTS-E-ASM-001")
    check("non-ASCII in a comment", code_of(lambda: A.assemble(prog("; caf\u00e9\n", profile))) == "OTS-E-ASM-001")
    check("CR is not a space", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\r\n  CU: sync\r\n}\r\n", profile))) == "OTS-E-ASM-001")
    for t in ("sync ", " sync", "  nop 0x1  "):
        check("leading/trailing spaces of an instruction are ignored", code_of(lambda: isa.parse(t)) is None, t)
    # A malformed queue name in a slot is OTS-E-ASM-005, as in sec. 3.
    for q in ("cu", "MEM [0]", "mem[0]", "VXM[16]", "Q"):
        check("slot queue name", code_of(lambda: A.assemble(prog(f"bundle @cycle=0 {{\n  {q}: nop 0x1\n}}\n", profile))) == "OTS-E-ASM-005", q)
    check("leading zeros in a queue index", A.queue_id("MEM[007]") == 7 and A.queue_id("SXM.W.DST[01]") == 117)
    check("slot without a queue", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n  : nop 0x1\n}\n", profile))) == "OTS-E-ASM-001")
    # A symbol is `name` or `name+0x<off>` with no spaces.
    sym = ".extern a, slice=1, addr=0x0, size=0x10\n"
    for t, want in (("read a+0x3, E:0, 0x0", None), ("read a  +  0x3, E:0, 0x0", "OTS-E-DEC-003"), ("read a +0x3, E:0, 0x0", "OTS-E-DEC-003")):
        check("symbol spacing", code_of(lambda: A.assemble(prog(f"bundle @cycle=0 {{\n  MEM[1]: {t}\n}}\n", profile, sym))) == want, t)
    # The text of OTS-E-DEC-003 quotes the instruction in double quotes, as the Rust assembler does (RV2-B-10).
    for t in ("host_notify 5", "frob 1"):
        try:
            A.ISA(profile).parse(t)
            msg = ""
        except A.AsmError as e:
            msg = str(e)
        check("OTS-E-DEC-003 text", msg.endswith('no instruction matches "%s"' % t), (t, msg))
    # A 64-bit word on the control channel is a placement error, like any other word the channel does not run.
    for t in ("read 0x0, E:0, 0x0", "add.i8 E:8, E:0, E:4", "shift E:0, up, 0x1, E:1"):
        check("64-bit instruction on CU", code_of(lambda: A.assemble(prog(f"bundle @cycle=0 {{\n  CU: {t}\n}}\n", profile))) == "OTS-E-QUE-002", t)
    check("32-bit non-control instruction on CU", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n  CU: lw.i8 E:0\n}\n", profile))) == "OTS-E-QUE-002")
    # AGT placement and extern overlap.
    def warns(body, head=""):
        ws = []
        A.assemble(prog(body, profile, head), warn=ws.append)
        return [w for w in ws if "OTS-E-OPR-001" in w and "configures no loop" in w or "level beyond" in w]
    check("stray AGT configuration warns", len(warns("bundle @cycle=0 {\n  MEM[0]: step 0x2\n}\n")) == 1)
    check("AGT configuration one cycle late warns", len(warns("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n}\nbundle @cycle=2 {\n  MEM[0]: step 0x2\n}\n")) == 1)
    check("AGT configuration on another queue warns", len(warns("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n}\nbundle @cycle=1 {\n  MEM[1]: step 0x2\n}\n")) == 1)
    check("AGT configuration chain is silent", warns("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n}\nbundle @cycle=1 {\n  MEM[0]: step 0x2\n}\n"
                                                     "bundle @cycle=2 {\n  MEM[0]: iterations 0x2\n}\nbundle @cycle=3 {\n  MEM[0]: countdown 0x2\n}\n") == [])
    # Per-path AGT (microarchitecture.md sec. 4): a Read + Write pair, its
    # configuration records alternating R, W, is silent; two references on one path open none.
    us = ("bundle @cycle=10 {\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: write %#x, E:1, 0x0\n}\n" % (1 << isa.cfg["sram"]["bank_bit"])
          + "bundle @cycle=11 {\n  MEM[0]: step 0x18\n}\nbundle @cycle=12 {\n  MEM[0]: step 0x1\n}\n"
          + "bundle @cycle=13 {\n  MEM[0]: iterations 0x70\n}\nbundle @cycle=14 {\n  MEM[0]: iterations 0x70\n}\n")
    check("AGT: Read + Write pair configuration is silent", warns(us) == [], warns(us))
    check("AGT: a pair's configuration is silent", warns(us.split("bundle @cycle=12")[0]) == [])
    two = "bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: read 0x1, E:1, 0x1\n}\nbundle @cycle=1 {\n  MEM[0]: step 0x2\n}\n"
    check("AGT: two references on one path open no configuration", len(warns(two)) == 1, warns(two))
    rw = "bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: step 0x2\n}\n"
    check("AGT: a configuration with its reference in one cycle is stray", len(warns(rw)) == 1, warns(rw))
    ifr = "bundle @cycle=0 {\n  MEM[0]: ifetch E:6\n  MEM[0]: read 0x0, E:0, 0x0\n}\nbundle @cycle=1 {\n  MEM[0]: step 0x2\n}\n"
    check("AGT: a reference beside an Ifetch opens its path", warns(ifr) == [], warns(ifr))
    gap = "bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n}\nbundle @cycle=1 {\n  MEM[0]: step 0x2\n}\nbundle @cycle=2 {\n  MEM[1]: nop 0x1\n}\nbundle @cycle=3 {\n  MEM[0]: step 0x1\n}\n"
    check("AGT: a cycle without a configuration record ends it", len(warns(gap)) == 1, warns(gap))
    # IfetchM reads words a and a+1 (microarchitecture.md sec. 3).
    def ow(body, head=""):
        ws = []
        A.assemble(prog(body, profile, head), warn=ws.append)
        return [w for w in ws if "OTS-E-OPR-001" in w and "beyond" in w]
    check("IfetchM: a+1 beyond words_per_slice warns", len(ow(f"bundle @cycle=0 {{\n  MEM[0]: ifetchm {isa.words - 1:#x}\n}}\n")) == 1)
    check("IfetchM: a+1 inside is silent", ow(f"bundle @cycle=0 {{\n  MEM[0]: ifetchm {isa.words - 2:#x}\n}}\n") == [])
    check("Read: last word is silent", ow(f"bundle @cycle=0 {{\n  MEM[0]: read {isa.words - 1:#x}, E:0, 0x0\n}}\n") == [])
    e1 = ".extern w, slice=1, addr=0x20, size=0x1\n"
    check("IfetchM: a symbol needs two words of its extern", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n  MEM[1]: ifetchm w\n}\n", profile, e1))) == "OTS-E-ASM-002")
    e2 = ".extern w, slice=1, addr=0x20, size=0x2\n"
    check("IfetchM: a symbol on two words", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n  MEM[1]: ifetchm w\n}\n", profile, e2))) is None)
    check("IfetchM: a symbol's last word", code_of(lambda: A.assemble(prog("bundle @cycle=0 {\n  MEM[1]: ifetchm w+0x1\n}\n", profile, e2))) == "OTS-E-ASM-002")
    # Text input (F8): every malformed Q line is OTS-E-ASM-003, never a crash; text is
    # recognised by its first non-blank byte.
    import tempfile
    def dis_rc(data):
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
            f.write(data)
        r = subprocess.run([sys.executable, os.path.join(HERE, "..", "..", "tools", "tsp_as.py"), "dis", "--profile", profile, "--config", cfg, f.name],
                           capture_output=True, text=True)
        os.unlink(f.name)
        return r.returncode, r.stdout, r.stderr
    for why, data in [("queue without units", b"Q\n"), ("non-hex unit", b"Q 0 4000g000\n"),
                      ("negative unit", b"Q 0 -1\n"), ("unit above 32 bits", b"Q 0 100000000\n"),
                      ("queue above 16 bits", b"Q 10000 00400000\n"), ("queue 0x90", b"Q 90 00400000\n"), ("queue fffe", b"Q fffe 00400000\n"),
                      ("old P line", b"P 0 0 1 400000 0\n"), ("unknown line", b"X 0 0 1 400000 0\n")]:
        rc, out, err = dis_rc(data)
        check(f"text input: {why} is OTS-E-ASM-003", rc == 1 and "OTS-E-ASM-003" in err and "Traceback" not in err, (rc, err[-300:]))
    rc, out, err = dis_rc(b"\n  \nQ 0 00400000\n")
    check("text input after blank lines", rc == 0 and "MEM[0]: " in out, (rc, out, err))
    five = "".join(f"bundle @cycle={k} {{\n  MEM[0]: step 0x1\n}}\n" for k in range(1, 6))
    check("a fifth AGT level warns", len(warns("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n}\n" + five)) == 1)
    check("overlapping externs", code_of(lambda: A.assemble(prog("", profile, ".extern a, slice=1, addr=0x0, size=0x10\n.extern b, slice=1, addr=0x8, size=0x10\n"))) == "OTS-E-ASM-002")
    check("adjacent externs", code_of(lambda: A.assemble(prog("", profile, ".extern a, slice=1, addr=0x0, size=0x8\n.extern b, slice=1, addr=0x8, size=0x10\n"))) is None)

    # 4b. Queue names (sec. 3 table) against the model's queue map comment in core/queues.sail.
    table = {"MEM[0]": 0, "MEM[87]": 87, "MEM.W[43]": 43, "MEM.E[0]": 44, "MEM.E[43]": 87, "VXM[0]": 88, "VXM[15]": 103,
             "MXM[0].LW": 104, "MXM[0].ABC": 105, "MXM[0].ACC": 106, "MXM[1].LW": 107, "MXM[3].ACC": 115,
             "SXM.W.DST[0]": 116, "SXM.W.DST[3]": 119, "SXM.W.TSP[1]": 121, "SXM.W.SHF[0]": 122, "SXM.W.SHF[1]": 123,
             "SXM.W.PRM": 124, "SXM.W.SEL": 125, "C2C.W.TX[0]": 126, "C2C.W.TX[1]": 127, "C2C.W.RX[0]": 128,
             "C2C.W.RX[1]": 129, "SXM.E.DST[0]": 130, "SXM.E.TSP[0]": 134, "SXM.E.SHF[1]": 137, "SXM.E.PRM": 138,
             "SXM.E.SEL": 139, "C2C.E.TX[0]": 140, "C2C.E.RX[1]": 143, "CU": 0xFFFF}
    for n, q in table.items():
        check("queue name", A.queue_id(n) == q, f"{n} -> {A.queue_id(n)}, want {q}")
    for q in list(range(144)) + [0xFFFF]:
        check("queue name inverse", A.queue_id(A.queue_name(q)) == q, q)
    for n in ("MEM[88]", "MEM.W[44]", "VXM[16]", "MXM[4].LW", "MXM[0].XX", "MXM[0].W", "C2C.W.TX", "C2C.E.RX[2]", "SXM.W.DST[4]", "SXM.W.TSP[2]", "SXM.W.PRM[0]",
              "SXM.N.SEL", "C2C.W.TR", "cu", "MEM[01x]", "MEM[]", "SXM.W.SEL[0]"):
        check("bad queue name", code_of(lambda: A.queue_id(n)) == "OTS-E-ASM-005", n)

    # 4c. Error classes (sec. 5).
    def asm(body, head=""):
        return code_of(lambda: A.assemble(prog(body, profile, head)))
    ext = ".extern w, slice=1, addr=0x20, size=0x4\n"
    other_bank = 1 << isa.cfg["sram"]["bank_bit"]     # bank 1; address 0x0 is in bank 0
    cases = [
        ("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: read 0x1, E:1, 0x0\n}\n", "", "OTS-E-QUE-001"),
        (f"bundle @cycle=0 {{\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: write {other_bank:#x}, E:1, 0x0\n}}\n", "", None),
        ("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: write 0x1, E:1, 0x0\n}\n", "", "OTS-E-MEM-001"),
        (f"bundle @cycle=0 {{\n  MEM[0]: write {other_bank:#x}, E:1, 0x0\n  MEM[0]: write 0x0, E:0, 0x0\n}}\n", "", "OTS-E-QUE-001"),
        ("bundle @cycle=0 {\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: read 0x1, E:1, 0x1\n}\n", "", None),
        ("bundle @cycle=0 {\n  MEM[0]: ifetchm 0x10\n  MEM[0]: read 0x0, E:1, 0x0\n}\n", "", "OTS-E-QUE-001"),
        ("bundle @cycle=0 {\n  MEM[0]: ifetchm 0x10\n  MEM[0]: read 0x0, E:1, 0x1\n}\n", "", "OTS-E-QUE-001"),
        ("bundle @cycle=0 {\n  MEM[0]: read 0x1, E:0, 0x0\n  MEM[0]: write 0x1, E:1, 0x1\n}\n", "", None),
        ("bundle @cycle=0 {\n  MEM[0]: gather W:3, E:1, 0x2, 0x0\n  MEM[0]: write 0x1, E:1, 0x0\n}\n", "", "OTS-E-MEM-001"),
        ("bundle @cycle=0 {\n  MEM[0]: ifetch E:6\n  MEM[0]: ifetchm 0x10\n}\n", "", "OTS-E-QUE-001"),
        (f"bundle @cycle=0 {{\n  MEM[0]: read 0x0, E:0, 0x0\n  MEM[0]: write {other_bank:#x}, E:1, 0x0\n  MEM[0]: nop 0x1\n}}\n", "", "OTS-E-QUE-001"),
        ("bundle @cycle=0 {\n  VXM[0]: nop 0x1\n  VXM[0]: nop 0x1\n}\n", "", "OTS-E-QUE-001"),
        ("bundle @cycle=0 {\n  VXM[0]: read 0x0, E:0, 0x0\n}\n", "", "OTS-E-QUE-002"),
        ("bundle @cycle=0 {\n  SXM.W.SEL: shift E:0, up, 0x1, E:1\n}\n", "", "OTS-E-QUE-002"),
        ("bundle @cycle=0 {\n  CU: repeat 0x1, 0x1\n}\n", "", "OTS-E-QUE-002"),
        ("bundle @cycle=0 {\n  CU: setvl 0x1\n}\n", "", "OTS-E-QUE-002"),
        ("bundle @cycle=0 {\n  CU: config 0x1\n}\n", "", None),
        ("bundle @cycle=0 {\n  MEM[0]: setvl 0x1\n}\n", "", None),
        ("bundle @cycle=0 {\n  XYZ[0]: nop 0x1\n}\n", "", "OTS-E-ASM-005"),
        ("bundle @cycle=0 {\n  MEM[1]: read v, E:0, 0x0\n}\n", ext, "OTS-E-ASM-002"),
        ("bundle @cycle=0 {\n  MEM[1]: read w+0x4, E:0, 0x0\n}\n", ext, "OTS-E-ASM-002"),
        ("bundle @cycle=0 {\n  MEM[2]: read w, E:0, 0x0\n}\n", ext, "OTS-E-ASM-002"),
        ("bundle @cycle=0 {\n  MEM[1]: write w+0x3, E:0, 0x0\n}\n", ext, None),
        ("", ext + ext, "OTS-E-ASM-002"),
        ("", ".extern w, slice=88, addr=0x0, size=0x1\n", "OTS-E-ASM-002"),
        ("", f".extern w, slice=0, addr={isa.words - 1:#x}, size=0x2\n", "OTS-E-ASM-002"),
        ("bundle @cycle=0 {\n  VXM[0]: add.i8 E:8, E:0, E:4\n", "", "OTS-E-ASM-001"),
        ("bundle @cycle=5 {\n}\nbundle @cycle=5 {\n}\n", "", "OTS-E-ASM-001"),
        ("bundle @cycle=5 {\n}\nbundle @cycle=4 {\n}\n", "", "OTS-E-ASM-001"),
        (".data\n", "", "OTS-E-ASM-001"),
        ("bundle @cycle=0 {\n  VXM[0]: nosuch E:0\n}\n", "", "OTS-E-DEC-003"),
        ("bundle @cycle=0 {\n  VXM[0]: cmp.lt.b8 E:0, E:1, E:2\n}\n", "", "OTS-E-DEC-003"),
        ("bundle @cycle=0 {\n  VXM[0]: add.i8 E:40, E:1, E:2\n}\n", "", "OTS-E-OPR-001"),
    ]
    for body, head, want in cases:
        check("error class", asm(body, head) == want, f"{head + body!r}: {asm(body, head)}, want {want}")
    check("unknown profile", code_of(lambda: A.assemble(".arch opentsp-tsa\n.profile ref1\n.section text\n")) == "OTS-E-ASM-003")
    check("unknown arch", code_of(lambda: A.assemble(".arch tsp-v1\n.profile ref\n.section text\n")) == "OTS-E-ASM-003")
    check("unrecognised instruction syntax", asm("bundle @cycle=0 {\n  MEM[0]: Read addr=0x100 -> E:0\n}\n") == "OTS-E-DEC-003")
    check("missing .section", code_of(lambda: A.assemble(".arch opentsp-tsa\n.profile ref\n")) == "OTS-E-ASM-001")
    # The decoder built from encoding.json alone agrees with the model.
    if sim:
        r = subprocess.run([sys.executable, os.path.join(HERE, "json_decoder.py"), exe, sim, cfg, "2000", "1"],
                           capture_output=True, text=True, timeout=1200)
        print(r.stdout.strip())
        check("json-only decoder = model", r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:])
        # The checked-in assembler corpora (readiness R5) against the model; they exist for the shipped profiles only.
        if profile in ("ref", "mini"):
            r = subprocess.run([sys.executable, os.path.join(HERE, "asm_corpus.py"), "--verify", exe, sim, profile],
                               capture_output=True, text=True, timeout=1200)
            print(r.stdout.strip())
            check("assembler corpora = model", r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:])
    # porting-guide.md sec. 2.1 documents every key of encoding.json (readiness R3).
    guide = open(os.path.join(HERE, "..", "..", "docs", "porting-guide.md")).read()
    db = A.load_encoding()
    keys = set(db) | {k for e in db["instructions"] for k in e} | {k for e in db["instructions"] for v in e["variables"] for k in v} \
        | {k for o in db["operand_kinds"].values() for k in o} | {k for k in db["decode_rule"] if k != "text"}
    check("porting guide documents every encoding.json key", all(f"`{k}`" in guide for k in keys),
          sorted(k for k in keys if f"`{k}`" not in guide))
    # The conformance vectors satisfy conformance/model/schema.json (readiness R13c).
    r = subprocess.run([sys.executable, os.path.join(HERE, "check_conformance_schema.py")], capture_output=True, text=True, timeout=600)
    check("conformance vectors = schema.json", r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:])
    # The checked-in assembler corpora are those tsp_as.py produces now.
    r = subprocess.run([sys.executable, os.path.join(HERE, "asm_corpus.py"), "--check"], capture_output=True, text=True, timeout=1200)
    check("assembler corpora up to date", r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:])
    m.p.stdin.close()
    m.p.wait()
    print(f"assembly.md examples and program files ({profile}): {len(bad)} errors")
    for b in bad[:30]:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
