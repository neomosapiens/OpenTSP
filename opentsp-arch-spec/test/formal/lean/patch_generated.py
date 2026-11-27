#!/usr/bin/env python3
"""Work around the layout defects (four groups of rules, below) of the Lean generator of Sail 0.20.3 in the generated model (Lean is the proof assistant of the repository).

The model is frozen and the defects are in the generator, so the generated Lean is patched, by exact rules with expected counts
(a rule that applies a different number of times fails: the generator or the model changed and the rule needs a look):

  1. a configuration accessor whose body is `let n = <literal>; assert(cond, msg); n` is written without `do`, so Lean reads the
     `(pure n)` on the following line as a third argument of `assert`: `unwrapValue (let n : Int := 16 / assert ... / (pure n))`
     becomes `unwrapValue (do let n : Int := 16 / assert ... / (pure n))` (c2c_links, c2c_cycles_per_vector, c2c_hac_period);
  2. a record update whose field is a list cons with a register read inside is written `abc := ← (e :: (abc_prune ...))`, lifting a
     pure list: the `←` is dropped, the register read inside stays lifted (mxm_insts.sail, the ABC list of the matrix unit);
  3. in Tick.lean, a list of `(pure x)` elements with register reads inside is written element by element, so Lean cannot type it as a
     list: it becomes `(pure [x, ...])` (one place), and the `| .some x => (x :: cands)` arms get their `pure` (three places);
  4. a variable named `open` or `matches`, which are Lean keywords, is renamed (InstsEnd.lean, five lines; `open` statements and
     doc comments are left alone).

usage: patch_generated.py OUT_DIR    (the `Out` directory of the generated Lean project)
"""
from pathlib import Path
import re
import sys

RULES = [
    # (file, regex, replacement, expected count)
    ("C2cInsts.lean",
     r"unwrapValue \(let (\w+) : Int := (\d+)\n\s+assert (.*?)\n\s+\(pure \1\)\)",
     lambda m: "unwrapValue (do\n  let %s : Int := %s\n  assert %s\n  (pure %s))" % (m.group(1), m.group(2), m.group(3), m.group(1)),
     3),
    ("MxmInsts.lean",
     r"abc := ← \(e :: \(abc_prune p\.abc \(← readReg issue_cycle\)\)\)",
     lambda m: "abc := (e :: (abc_prune p.abc (← readReg issue_cycle)))",
     1),
]


def split_top(text):
    """Split a comma separated list at bracket depth 0."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += ch in "([" and 1 or (ch in ")]" and -1 or 0)
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def lift_pure_list(text):
    """`[(pure a), (pure b)]` in the one place of tick.lean that has it becomes `(pure [a, b])`; returns (text, count)."""
    pattern = re.compile(r"\(\( do\n\s+\[(\(pure base\),.*?)\] \) : SailM \(List Nat\) \)", re.S)
    def rewrite(match):
        parts = split_top(match.group(1))
        if not all(p.startswith("(pure ") and p.endswith(")") for p in parts):
            raise ValueError("an element of the list is not a pure value")
        inner = ", ".join(" ".join(p[len("(pure "):-1].split()) for p in parts)
        return "(( do\n    (pure [%s]) ) : SailM (List Nat) )" % inner
    return pattern.subn(rewrite, text)


KEYWORD_FILES = {"InstsEnd.lean": 5}
KEYWORDS = re.compile(r"\b(open|matches)\b")


def rename_keywords(text):
    """Rename the identifiers `open` and `matches` on the lines that use them as variables; returns (text, number of lines)."""
    lines, changed = text.split("\n"), 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("open ") or stripped.startswith("/--") or not KEYWORDS.search(line):
            continue
        lines[i] = KEYWORDS.sub(lambda m: m.group(1) + "_", line)
        changed += 1
    return "\n".join(lines), changed


def patch(out_dir):
    done = []
    path = Path(out_dir) / "Tick.lean"
    text, count = lift_pure_list(path.read_text())
    text, count2 = re.subn(r"\| \.some x => \(x :: cands\)", "| .some x => (pure (x :: cands))", text)
    if (count, count2) != (1, 3):
        raise ValueError("Tick.lean: list rules applied %d and %d times, expected 1 and 3" % (count, count2))
    path.write_text(text)
    done.append(("Tick.lean", count + count2))
    for name, expected in KEYWORD_FILES.items():
        path = Path(out_dir) / name
        text, count = rename_keywords(path.read_text())
        if count != expected:
            raise ValueError("%s: %d lines use a Lean keyword as a variable, expected %d" % (name, count, expected))
        path.write_text(text)
        done.append((name, count))
    for name, pattern, replacement, expected in RULES:
        path = Path(out_dir) / name
        text = path.read_text()
        text, count = re.subn(pattern, replacement, text)
        if count != expected:
            raise ValueError("%s: the rule %r applied %d times, expected %d" % (name, pattern[:40], count, expected))
        path.write_text(text)
        done.append((name, count))
    return done


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    try:
        for name, count in patch(sys.argv[1]):
            print("patch_generated: %s: %d replacement(s)" % (name, count))
    except (ValueError, OSError) as error:
        print("patch_generated: " + str(error), file=sys.stderr)
        sys.exit(1)
