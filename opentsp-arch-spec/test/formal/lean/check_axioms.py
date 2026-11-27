#!/usr/bin/env python3
"""Check the `#print axioms` output of the Lean proofs against the allowed set.

usage: check_axioms.py PROOFS_LOG AXIOMS_LEAN [PROOF_DIRECTORY]

PROOFS_LOG is the output of `lake build Proofs`; AXIOMS_LEAN is proofs/Axioms.lean, whose `#print axioms NAME` lines name the theorems.
Every named theorem must appear in the log with an axiom list that is a subset of ALLOWED (an axiom the proof declares itself, `sorryAx`
and `ofReduceBool` are all outside it) or with "does not depend on any axioms". Prints the number of theorems checked, in the form
`theorems=N`, and exits 1 with one line per problem.
"""
from pathlib import Path
import re
import sys

ALLOWED = {"propext", "Classical.choice", "Quot.sound"}


def declaration_text(source):
    """Remove nested Lean comments and strings without changing command line boundaries."""
    out, i, depth = [], 0, 0
    while i < len(source):
        if source.startswith("/-", i):
            depth += 1
            out.append("  ")
            i += 2
        elif depth and source.startswith("-/", i):
            depth -= 1
            out.append("  ")
            i += 2
        elif depth:
            out.append("\n" if source[i] == "\n" else " ")
            i += 1
        elif source.startswith("--", i):
            end = source.find("\n", i)
            if end < 0:
                break
            out.append(" " * (end - i))
            i = end
        elif source[i] == '"':
            out.append(" ")
            i += 1
            while i < len(source):
                char = source[i]
                out.append("\n" if char == "\n" else " ")
                i += 1
                if char == "\\" and i < len(source):
                    out.append(" ")
                    i += 1
                elif char == '"':
                    break
        else:
            out.append(source[i])
            i += 1
    if depth:
        raise ValueError("unterminated Lean block comment")
    return "".join(out)


def declared_theorems(source):
    """Inventory direct project theorem/lemma commands, including inline attributes.

    Search the declaration keyword rather than the start of its line: attributes and
    modifiers may precede it. Compiler-generated helper declarations are covered by
    the audited axiom dependencies; this inventory concerns written declarations.
    """
    pattern = re.compile(
        r"^[ \t]*(?P<scope>namespace|section|end)\b(?P<label>[^\n]*)"
        r"|(?<![\w.])(?P<mods>(?:(?:private|protected|noncomputable)\s+)*)"
        r"(?:theorem|lemma)\s+(?P<name>[^\s(:{]+)", re.M)
    scopes, found = [], []
    for match in pattern.finditer(declaration_text(source)):
        command = match["scope"]
        if command in {"namespace", "section"}:
            label = match["label"].strip()
            scopes.append((command, label))
        elif command == "end":
            if not scopes:
                raise ValueError("end without namespace or section")
            _, label = scopes.pop()
            closing = match["label"].strip()
            if closing and closing != label:
                raise ValueError("mismatched namespace/section end: " + closing)
        elif "private" not in match["mods"].split():
            name = match["name"]
            prefix = ".".join(label for kind, label in scopes if kind == "namespace")
            found.append(name[7:] if name.startswith("_root_.") else
                         prefix + "." + name if prefix else name)
    if scopes:
        raise ValueError("unclosed namespace or section")
    return found


def inventory_problems(directory, axioms_lean):
    declared = []
    for path in sorted(Path(directory).rglob("*.lean")):
        try:
            declared.extend(declared_theorems(path.read_text()))
        except ValueError as error:
            return [str(path) + ": " + str(error)]
    wanted = theorems(axioms_lean)
    out = [name + ": public declaration missing explicit axiom registration"
           for name in sorted(set(declared) - set(wanted))]
    out += [name + ": registered name has no written public theorem/lemma declaration"
            for name in sorted(set(wanted) - set(declared))]
    out += [name + ": duplicate explicit axiom registration"
            for name in sorted(set(wanted)) if wanted.count(name) > 1]
    return out


def theorems(axioms_lean):
    return re.findall(r"^#print axioms\s+(\S+)", axioms_lean, re.M)


def report(log):
    """Exact declaration name -> union of every reported axiom list, including repeats.

    Namespace suffixes are not declaration identities; repeated empty/core lists cannot
    erase an earlier forbidden axiom. A long printed list may wrap across lines.
    """
    found = {}
    text = re.sub(r"\s+", " ", log)
    for name, listing in re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text):
        found.setdefault(name, set()).update(a.strip() for a in listing.split(",") if a.strip())
    for name in re.findall(r"'([^']+)' does not depend on any axioms", text):
        found.setdefault(name, set())
    return found


def problems(log, axioms_lean):
    wanted = theorems(axioms_lean)
    found = report(log)
    out = []
    if not wanted:
        out.append("Axioms.lean names no theorem")
    for name in wanted:
        if name not in found:
            out.append("%s: no `#print axioms` line in the log" % name)
        elif found[name] - ALLOWED:
            out.append("%s depends on %s, outside the allowed %s" % (name, sorted(found[name] - ALLOWED), sorted(ALLOWED)))
    for name in sorted(set(found) - set(wanted)):
        out.append("%s: the log reports a theorem that Axioms.lean does not name" % name)
    return out, len(wanted)


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    axioms = Path(argv[1]).read_text()
    found, count = problems(Path(argv[0]).read_text(), axioms)
    if len(argv) == 3:
        found.extend(inventory_problems(argv[2], axioms))
    for line in found:
        print("check_axioms: " + line, file=sys.stderr)
    if found:
        return 1
    print("check_axioms: theorems=%d, every axiom list within %s" % (count, sorted(ALLOWED)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
