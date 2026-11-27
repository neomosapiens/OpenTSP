"""Tests of tools/check_docs_location.sh (run by `just check-docs`): each case runs the checker on a scratch source directory."""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
V = "gr" + "oq"
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    fails += 0 if ok else 1


def run(files, allow=None):
    """Checker on a scratch tree holding `files` (path -> bytes); returns (exit code, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        (t / "tools").mkdir()
        for name in ("check_docs_location.sh", "check_docs_location.py", "source_files.py", "check_docs_allow.txt"):
            if (ROOT / "tools" / name).exists():
                shutil.copy(ROOT / "tools" / name, t / "tools" / name)
        if allow is not None:
            (t / "tools/check_docs_allow.txt").write_text(allow)
        for path, data in files.items():
            p = t / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        r = subprocess.run(["bash", str(t / "tools/check_docs_location.sh")], capture_output=True, text=True)
        return r.returncode, r.stderr


def fw(s):
    return "".join(chr(ord(c) + 0xFEE0) for c in s)


def refused(name, files, allow=None):
    rc, err = run(files, allow)
    check(name, rc != 0 and "HARD RULE" in err, (rc, err))


def accepted(name, files, allow=None):
    rc, err = run(files, allow)
    check(name, rc == 0, (rc, err))


accepted("a clean tree", {"docs/a.md": b"text\n", "README.md": b"x\n", "AGENTS.md": b"x\n", "src/a.txt": b"hello\n"})
refused("Markdown outside docs", {"NOTES.md": b"x"})
refused("uppercase Markdown suffix outside docs", {"NOTES.MD": b"x"})
refused("mixed-case Markdown suffix outside docs", {"sub/Notes.Md": b"x"})
accepted("README.md and AGENTS.md at a project directory", {"proj/README.md": b"x", "proj/AGENTS.md": b"x"})
refused("the name in text", {"src/a.txt": V.encode()})
refused("the name in mixed case", {"src/a.txt": V.swapcase().encode()})
refused("the name in full-width letters", {"src/a.txt": fw(V).encode()})
refused("the name with separators", {"src/a.txt": "-".join(V).encode()})
refused("the name with spaces and dots", {"src/a.txt": ". ".join(V).encode()})
refused("the name with a zero-width character", {"src/a.txt": (V[:2] + "\u200b" + V[2:]).encode()})
refused("the name in a path", {f"docs/{V}-x.txt": b"x"})
refused("the name in a full-width path", {f"docs/{fw(V)}.txt": b"x"})
refused("the name in a binary file", {"src/a.bin": b"\0" + V.encode() + b"\0"})
refused("the name in a UTF-16 file", {"src/a.bin": V.encode("utf-16")})
refused("the name in documentation", {"docs/reference/a.md": V.encode()})
accepted("ordinary words with those letters", {"src/a.txt": b"program of q, great quote, grow quickly, gr\n"})
allow = "src/a.bin | a binary fixture that quotes the name on purpose\n"
accepted("an allow-listed binary", {"src/a.bin": b"\0" + V.encode()}, allow)
refused("an allow-listed path does not exempt a text file", {"src/a.txt": V.encode()}, "src/a.txt | text\n")
refused("an allow-list line without a reason", {"src/a.bin": b"\0" + V.encode()}, "src/a.bin\n")
refused("an allow-list line without a reason (empty)", {"src/a.bin": b"\0" + V.encode()}, "src/a.bin |  \n")


attribution = f"OpenTSP is inspired by {V.title()}'s original Tensor Streaming Processor papers.".encode()
accepted("publication credit in both READMEs", {"README.md": attribution, "opentsp-arch-spec/README.md": attribution})
refused("credit exception is limited to the two READMEs", {"docs/reference/a.md": attribution})
refused("the second mention in a credited README is still checked", {"README.md": attribution + b"\n" + V.encode()})
refused("the exception is exact and cannot remove text within another line", {"README.md": attribution + b" Other text"})
refused("duplicating the attribution is refused", {"README.md": attribution + b"\n" + attribution})
accepted("architecture prose", {"opentsp-arch-spec/docs/architecture.md": b"text"})
print("check_docs_location tests: " + ("all passed" if not fails else f"{fails} failed"))
sys.exit(1 if fails else 0)
