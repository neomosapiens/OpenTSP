"""Tests of the multi-chip manifest (.otsm; assembly.md sec. 4): packing,
reading, every rejection, and the `tsp_as.py mpack` / `mobjinput` commands, which turn a manifest and one
weight package per chip into the `C <chip>` input of the harness (c_emulator/tsp_sim.c --chips K).
usage: manifest_test.py <tsp_sim> <config.json>"""
import json, os, struct, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import tsp_as
from tsp_as import AsmError
SIM, CFG = sys.argv[1], sys.argv[2]
cfg = json.load(open(CFG))
PROFILE = os.path.splitext(os.path.basename(CFG))[0]
VL = 16 * cfg["geometry"]["superlanes"]
L = cfg["c2c"]["link_latency"]
TOOL = os.path.join(HERE, "..", "..", "tools", "tsp_as.py")
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def asm(text):
    isa, prog = tsp_as.assemble(text, "<test>", CFG)
    return tsp_as.write_object(prog, CFG)


def rejects(name, fn, code="OTS-E-ASM-003"):
    try:
        fn()
    except AsmError as e:
        check(name, e.code == code, (e.code, e.msg))
    except Exception as e:                                  # a crash is a failure
        check(name, False, repr(e))
    else:
        check(name, False, "accepted")


HEAD = f".arch opentsp-tsa\n.profile {PROFILE}\n"
# chip 0 transmits on link 0 to chip 1; chip 1 receives from chip 0 on link 0. Both have an extern.
P0 = HEAD + ".extern x, slice=0, addr=0x10, size=0x1\n.section text\nbundle @cycle=10 {\n  C2C.E.TX[0]: transmit 0x1, 0x0\n}\n"
P1 = HEAD + ".extern x, slice=0, addr=0x10, size=0x1\n.section text\nbundle @cycle=%d {\n  C2C.W.RX[0]: receive 0x0, 0x0, E:3\n}\n" % (10 + L)
P2 = HEAD + ".section text\nbundle @cycle=5 {\n  C2C.W.TX[0]: transmit 0x0, 0x6\n}\n"
o0, o1, o2 = asm(P0), asm(P1), asm(P2)

# --- the layout of the manifest (assembly.md sec. 4)
m = tsp_as.write_manifest(PROFILE, [o0, o1], CFG)
magic, major, minor, hsize, prof, k, topo, rev, z0, denc, dprof = tsp_as.MHEADER.unpack_from(m)
check("header: magic OTSM, version 0.1, header size 64, profile, K, topology 0, format revision 2, reserved 0",
      (magic, major, minor, hsize, prof.rstrip(b"\0").decode(), k, topo, rev, z0) == (b"OTSM", 0, 1, 64, PROFILE, 2, 0, 2, 0), (magic, major, minor, hsize, prof, k, topo, rev, z0))
check("header: the two interface digests are those of an object",
      (denc, dprof) == tsp_as.interface_digests(PROFILE, CFG) and (o0[92:104], o0[124:136]) == (denc, dprof))
ents = [tsp_as.MENTRY.unpack_from(m, 64 + 8 * i) for i in range(2)]
check("K (offset, size) entries follow the header; the objects follow the entries back to back, in chip order",
      ents == [(80, len(o0)), (80 + len(o0), len(o1))] and m[80:80 + len(o0)] == o0 and m[ents[1][0]:] == o1 and len(m) == 80 + len(o0) + len(o1), ents)
check("read_manifest returns the profile and the embedded objects", tsp_as.read_manifest(m, None, CFG) == (PROFILE, [o0, o1]))
check("the format is deterministic", tsp_as.write_manifest(PROFILE, [o0, o1], CFG) == m)
check("a manifest of one chip is valid", tsp_as.read_manifest(tsp_as.write_manifest(PROFILE, [o0], CFG), PROFILE, CFG)[1] == [o0])
for n in (1, 2, 8, 12, 16, 64, 264):
    check(f"a manifest of {n} chips is valid", len(tsp_as.read_manifest(tsp_as.write_manifest(PROFILE, [o2] * n, CFG), None, CFG)[1]) == n)

# --- rejections
for n in (0, 265, 512):
    rejects(f"mpack of {n} objects is rejected (K is 1..264 without a topology file)", lambda n=n: tsp_as.write_manifest(PROFILE, [o2] * n, CFG))
rejects("mpack of 513 objects is rejected even with a topology file (K is at most 512)", lambda: tsp_as.write_manifest(PROFILE, [o2] * 513, CFG, 512))


def patch(off, fmt, *v):
    b = bytearray(m); struct.pack_into(fmt, b, off, *v); return bytes(b)


BAD = [
    ("bad magic", b"OTSX" + m[4:]),
    ("an OpenTSP ELF object", o0),
    ("an ELF file", b"\x7fELF" + bytes(80)),
    ("empty", b""),
    ("truncated header", m[:40]),
    ("version 0.2", patch(5, "<B", 2)),
    ("version 1.1", patch(4, "<B", 1)),
    ("header size 80", patch(6, "<H", 80)),
    ("format revision 1", patch(32, "<I", 1)),
    ("format revision 0", patch(32, "<I", 0)),
    ("reserved word non-zero", patch(36, "<I", 1)),
    ("topology 1", patch(28, "<I", 1)),
    ("chip count 0", patch(24, "<I", 0)),
    ("chip count 3 with two entries (entry 3 is object text: bad offset)", patch(24, "<I", 3)),
    ("chip count 513", patch(24, "<I", 513)),
    ("chip count 2**32-1", patch(24, "<I", 0xFFFFFFFF)),
    ("encoding digest", patch(40, "<B", m[40] ^ 1)),
    ("profile digest", patch(52, "<B", m[52] ^ 1)),
    ("profile not ASCII", patch(8, "<B", 0xFF)),
    ("unknown profile", patch(8, "<4s", b"zzzz")),
    ("entry 0 offset 81", patch(64, "<I", 81)),
    ("entry 0 offset 0", patch(64, "<I", 0)),
    ("entry 0 size 0", patch(68, "<I", 0)),
    ("entry 0 size too large", patch(68, "<I", len(m))),
    ("entry 0 size one short (entry 1 does not follow)", patch(68, "<I", len(o0) - 1)),
    ("entry 1 offset one more", patch(72, "<I", 80 + len(o0) + 1)),
    ("entry 1 size one short (trailing byte)", patch(76, "<I", len(o1) - 1)),
    ("trailing byte", m + b"\0"),
    ("truncated object", m[:-1]),
    ("embedded object with a bad magic", m[:80] + b"OTSX" + m[84:]),
    ("embedded object with a bad digest", patch(80 + 40, "<B", m[80 + 40] ^ 1)),
]
for name, data in BAD:
    rejects(f"read_manifest rejects: {name}", lambda d=data: tsp_as.read_manifest(d, None, CFG))
rejects("read_manifest rejects a profile other than the requested one", lambda: tsp_as.read_manifest(m, "mini" if PROFILE != "mini" else "ref"))
# an object of another profile cannot be packed (profile field and digests)
other = "mini" if PROFILE != "mini" else "ref"
oo = tsp_as.write_object(tsp_as.assemble(f".arch opentsp-tsa\n.profile {other}\n.section text\nbundle @cycle=1 {{\n  C2C.W.TX[0]: transmit 0x0, 0x0\n}}\n", "<test>")[1])
rejects("mpack rejects an object of another profile", lambda: tsp_as.write_manifest(PROFILE, [o0, oo], CFG))
rejects("mpack rejects a non-object", lambda: tsp_as.write_manifest(PROFILE, [o0, b"Q 0 0\n"], CFG))
# a manifest of another profile: a valid manifest of that profile is refused under this profile's config
mo = tsp_as.write_manifest(other, [oo])
rejects("a manifest of another profile is refused", lambda: tsp_as.read_manifest(mo, PROFILE, CFG))

# --- weights per chip: manifest_input
w = lambda v: tsp_as.write_weights(PROFILE, [("x", None, [bytes([v]) * VL])], CFG)
text = tsp_as.manifest_input(m, [w(0x11), w(0x22)], CFG)
want = "".join(f"C {i}\n" + tsp_as.preload_text(tsp_as.bind_weights(ex, tsp_as.read_weights(wp, PROFILE, CFG))) + tsp_as.input_text(im)
               for i, (o, wp) in enumerate(zip((o0, o1), (w(0x11), w(0x22)))) for (_, im, ex) in [tsp_as.read_object(o, PROFILE, CFG)])
check("manifest_input: per chip a C line, the S lines of its weight package and its Q lines (as objinput prints)", text == want and text.startswith("C 0\nS 0 10 11 "), text[:60])
check("manifest_input without weights: only C and Q lines", tsp_as.manifest_input(m, [], CFG).count("S ") == 0 and tsp_as.manifest_input(m, [], CFG).count("C ") == 2)
rejects("manifest_input: one weight package for two chips", lambda: tsp_as.manifest_input(m, [w(1)], CFG), "OTS-E-ASM-004")
rejects("manifest_input: three weight packages for two chips", lambda: tsp_as.manifest_input(m, [w(1)] * 3, CFG), "OTS-E-ASM-004")
rejects("manifest_input: a weight package without the extern's tensor", lambda: tsp_as.manifest_input(m, [w(1), tsp_as.write_weights(PROFILE, [("y", None, [bytes(VL)])], CFG)], CFG), "OTS-E-ASM-004")
c16 = "".join(l for l in tsp_as.manifest_input(tsp_as.write_manifest(PROFILE, [o2] * 16, CFG), [], CFG).splitlines(True) if l.startswith("C "))
check("manifest_input: chip numbers are hex", c16.split() == [t for i in range(16) for t in ("C", f"{i:x}")], c16)

# --- the commands, and the run: chip 0 transmits to chip 1, which receives and (via the harness) is dumped.
with tempfile.TemporaryDirectory() as td:
    def path(n): return os.path.join(td, n)
    def tool(*a, inp=None):
        return subprocess.run([sys.executable, TOOL, *a, "--config", CFG], capture_output=True, text=True, input=inp)
    for n, src in (("c0", P0), ("c1", P1)):
        open(path(n + ".ots"), "w").write(src)
        p = tool("asm", path(n + ".ots"), "-o", path(n + ".elf"), "--profile", PROFILE)
        assert p.returncode == 0, p.stderr
    open(path("w0.bin"), "wb").write(bytes([0x11]) * VL)
    open(path("w1.bin"), "wb").write(bytes([0x22]) * VL)
    for n in ("w0", "w1"):
        p = tool("wpack", "--profile", PROFILE, "-o", path(n + ".safetensors"), "x=raw:" + path(n + ".bin"))
        assert p.returncode == 0, p.stderr
    p = tool("mpack", "--profile", PROFILE, "-o", path("m.otsm"), path("c0.elf"), path("c1.elf"))
    check("tsp_as.py mpack writes the manifest of the library", p.returncode == 0 and open(path("m.otsm"), "rb").read() == m, p.stderr)
    p = tool("mobjinput", path("m.otsm"), "--weights", path("w0.safetensors"), "--weights", path("w1.safetensors"))
    check("tsp_as.py mobjinput prints the C-line input, one weight package per chip, in chip order", p.returncode == 0 and p.stdout == text, p.stderr)
    p = tool("mobjinput", path("m.otsm"), "--weights", path("w0.safetensors"))
    check("tsp_as.py mobjinput with too few weight packages: OTS-E-ASM-004, exit 1", p.returncode == 1 and "OTS-E-ASM-004" in p.stderr, (p.returncode, p.stderr))
    p = tool("mobjinput", path("c0.elf"))
    check("tsp_as.py mobjinput of an object (not a manifest): OTS-E-ASM-003, exit 1", p.returncode == 1 and "OTS-E-ASM-003" in p.stderr, (p.returncode, p.stderr))
    p = tool("mpack", "--profile", PROFILE, "-o", path("bad.otsm"), path("c0.elf"), path("m.otsm"))
    check("tsp_as.py mpack of a manifest: OTS-E-ASM-003, exit 1", p.returncode == 1 and "OTS-E-ASM-003" in p.stderr, (p.returncode, p.stderr))
    # the harness run of the input: chip 0 sends (dump the preload of both), chip 1 receives the notification
    inp = tool("mobjinput", path("m.otsm"), "--weights", path("w0.safetensors"), "--weights", path("w1.safetensors")).stdout
    inp = inp.replace("C 1\n", "D 0 10\nC 1\n") + "D 0 10\n"
    r = subprocess.run([SIM, CFG, "--trace", "--chips", "2"], input=inp, capture_output=True, text=True)
    parts = r.stdout.split("C 1\n")
    def word(s):
        ws = [l.split()[3:] for l in s.splitlines() if l.startswith("W 0 10 ")]
        return ws[0][:VL] if len(ws) == 1 else None
    # the preloads reach their chips (the programs do not write them): chip 0's word is 0x11 in every lane, chip 1's 0x22
    check("the harness runs the manifest: the preload of each chip is that chip's weight package",
          r.returncode == 0 and len(parts) == 2 and word(parts[0]) == ["11"] * VL and word(parts[1]) == ["22"] * VL,
          (r.returncode, r.stderr, r.stdout[:200]))
    check("the run has no diagnostics (the Receive at link_latency after the Transmit finds the vector)", not any(l.startswith("T diag") for l in r.stdout.splitlines()) and "E - -1 0" in r.stdout, r.stdout[:300])

print("manifest:", "all pass" if ok else "FAILURES")
sys.exit(0 if ok else 1)
