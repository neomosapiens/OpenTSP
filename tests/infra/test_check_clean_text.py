"""Tests of tools/check_clean_text.sh (run by `just check-docs`): a seeded violation of each family fails with file:line,
the clean tree and an exempt path pass."""
import os
import pathlib
import subprocess
import sys
import tempfile

TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools/check_clean_text.sh"
PATTERNS = pathlib.Path(__file__).resolve().parents[2] / "tools/clean_text_patterns.txt"
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    fails += 0 if ok else 1


def run(files, allow="", *args):
    """Run the check on a scratch tree holding `files` ({path: text}); returns (exit status, standard output)."""
    with tempfile.TemporaryDirectory() as d:
        for path, text in files.items():
            p = pathlib.Path(d, path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        allow_file = pathlib.Path(d, "allow.txt")
        allow_file.write_text(allow)
        env = dict(os.environ, CLEAN_TEXT_ROOT=d, CLEAN_TEXT_PATTERNS=str(PATTERNS), CLEAN_TEXT_ALLOW=str(allow_file))
        r = subprocess.run(["bash", str(TOOL), *args], capture_output=True, text=True, env=env)
        return r.returncode, r.stdout


# The seeded lines are assembled from two parts so that this file does not match the patterns itself.
PAT = "pat" + "ent"
SEEDS = {
    "patent": "// see the vendor's " + PAT + " for the queue",
    "patent number": "// US" + "11243880 describes it",
    "vendor name": "// as on the " + "LP" + "U",
    "finding": "// review " + "finding M8",
    "review": "// " + "audit " + "batch 2",
    "decision id": "// decision " + "D-VXM-CHAIN",
    "change request": "// the erratum " + "CR" + "-2",
    "owner": "// owner " + "decision O-13",
    "history": "// " + "WP" + "-998 changed it",
    "retired suite": "// strict " + "V" + "9 checks",
    "retired package": "// uses " + "tsp" + "lib",
    "retired task label": "// handled by " + "W7" + "-4",
    "retired helper label": "fn " + "g2_" + "issue_on() {}",
    "retired screening name": "fn f32_matches_" + "v12() {}",
    "private-process phrase": "// version is for the " + "owner to " + "declare",

}

rc, out = run({"src/a.rs": "fn f() {}\n// a plain comment about the lane width\n"})
check("a clean tree passes", rc == 0 and "clean" in out, out)

for name, line in SEEDS.items():
    rc, out = run({"src/a.rs": "fn f() {}\n" + line + "\n"})
    check(f"seeded {name} fails with file:line", rc == 1 and "src/a.rs:2:" in out, out)

rc, out = run({"docs/a.md": "A claim from the " + PAT + " of the vendor.\n"})
check("a document is checked for the patent family", rc == 1 and "docs/a.md:1:" in out, out)
rc, out = run({"docs/a.md": "The review finding M8 changed it.\n"})
check("a document is checked for the history family", rc == 1, out)
rc, out = run({"src/a.rs": "// ASM-003 and OTS-E-ASM-001 are diagnostics\n"})
check("a diagnostic code is not an id", rc == 0, out)
rc, out = run({"src/a.rs": "// vLLM V1 scheduler, V2 runner and profiler phase 14\n"})
check("external versions and runtime phases remain valid", rc == 0, out)

rc, out = run({"old/a.rs": SEEDS["patent"] + "\n"}, "old/ | exempt\n")
check("an exempt directory passes", rc == 0, out)
rc, out = run({"a.rs": SEEDS["patent"] + "\n", "b.rs": "fn g() {}\n"}, "a.rs | exempt\n")
check("an exempt file passes", rc == 0, out)
rc, out = run({"old/a.rs": SEEDS["patent"] + "\n"}, "old/ | exempt\n", "--no-allow")
check("--no-allow scans the exempt path", rc == 1 and "old/a.rs:1:" in out, out)
rc, out = run({"a.rs": 'let m = "' + SEEDS["owner"] + '";\nlet n = 1; // ' + SEEDS["history"] + "\n"}, 'a.rs::"// owner | an assertion message\n')
check("a line allowance exempts the lines with its text only", rc == 1 and "a.rs:2:" in out and "a.rs:1:" not in out, out)
rc, out = run({"a.rs": SEEDS["patent"] + "\n"}, "a.rs\n")
check("an allowlist line without a reason is an error", rc != 0, out)
rc, out = run({"a.rs": SEEDS["patent"] + "\n" + SEEDS["finding"] + "\n"}, "", "--count")
check("--count lists the matches by pattern name", rc == 0 and "TOTAL" in out and "2" in out.split("TOTAL")[1], out)

sys.exit(1 if fails else 0)
