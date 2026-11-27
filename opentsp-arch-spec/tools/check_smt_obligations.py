#!/usr/bin/env python3
"""Check every named Sail SMT obligation, failing closed on missing/unknown results."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

TIMEOUT = 60          # seconds per query, requested from the solver
KILL_AFTER = 305      # seconds: the gate's own limit on one solver process; above the 300 s the toolchain's z3 wrapper
                      # grants, so the gate never kills a query the solver is still allowed to run (p_fp8_decode at 256-bit
                      # integers takes about 95 s)


def obligations(source):
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
    annotations = re.findall(r"\$(property|counterexample)\b", source)
    declarations = re.findall(
        r"\$(property|counterexample)\s+function\s+(\w+)\s*\(", source)
    names = {name: "unsat" if kind == "property" else "sat"
             for kind, name in declarations}
    if not names or len(names) != len(declarations) or len(declarations) != len(annotations):
        raise ValueError("missing, duplicate or unsupported obligation declaration")
    return names


# A property body that can raise is proved vacuously: the SMT path that raises is unsat, so the
# obligation holds whatever the property says . These words are rejected.
RAISING = ("throw", "internal_error", "not_implemented", "exit", "assert")
DECLARATION = re.compile(
    r"^(?:\$\w+|private\s+function|function|val|register|let|type|struct|enum|union|newtype|overload|default|scattered)\b",
    re.M)


def strip_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)


def check_vacuity(source):
    """Reject a `$property` function whose text contains throw, internal_error, not_implemented, exit or assert.

    The text of a property runs from its `function` keyword to the next declaration at column 0. A `$property` must be
    followed directly by its `function` (comments and blank lines in between are skipped, as Sail skips them): any other
    declaration after it is rejected, because the obligation lister and the checker would then disagree on what is a property.
    Counterexamples and helper functions are not checked: a call to a throwing helper is reported
    by the solver (sat), and a counterexample is expected to be satisfiable."""
    source = strip_comments(source)
    found = []
    for match in re.finditer(r"\$property\b\s*", source):
        head = re.match(r"function\s+(\w+)\b", source[match.end():])
        if not head:
            line = source[:match.start()].count("\n") + 1
            raise ValueError("a $property (line %d, comments removed) is not followed directly by a function declaration" % line)
        start = match.end() + head.end()
        stop = re.compile(r"^" + DECLARATION.pattern[1:], re.M).search(source, start)
        text = source[match.end():stop.start() if stop else len(source)]
        for word in RAISING:
            if re.search(r"\b" + word + r"\b", text):
                found.append("%s (%s)" % (head.group(1), word))
    if found:
        raise ValueError("vacuous property: a body that can raise is proved by the unsat raising path: " +
                         ", ".join(found))


def split_top(text):
    """Split a parameter list at the commas outside brackets."""
    parts, depth, current = [], 0, ""
    for ch in text:
        depth += (ch in "([{") - (ch in ")]}")
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def twin_source(source):
    """Sail text of the twin reachability query of every `$property`: `twin_<name>`, a `$counterexample` whose body is the
    negated body of the property, so that the solver is asked for an input on which the property completes without
    raising and returns true.

    A property is proved when its negation is unsatisfiable; that is also the answer when every path of its body raises (the
    raising paths are not part of the query). The twin is satisfiable exactly when some input reaches a normal return with the
    claim true, so `unsat` for a twin means the obligation proves nothing. The body is copied, not called: Sail's SMT backend
    leaves the result of a call to a function that always raises unconstrained, which would make the twin satisfiable for the
    very case it is meant to catch. A property that returns true on a guard before it compares (`if guard then true else ...`)
    is not caught by this generic twin when the guard is satisfiable. The two guarded properties of the suite
    `test/formal/props/frontend.sail` carry a hand-written twin (`t_iq_pop_accounting`, `t_iq_push_accounting`); the guarded properties
    of the model's own `props/props.sail` (`p_iq_pop_conservation`, `p_iq_append`, `p_fetch_word_poison`) have none: the
    model directory is frozen, and no twin has been written for them in a suite."""
    source = strip_comments(source)
    names = set(re.findall(r"function\s+(\w+)\b", source))
    ending = re.compile(r"^" + DECLARATION.pattern[1:], re.M)
    twins = []
    for match in re.finditer(r"\$property\s+function\s+(\w+)\s*\(", source):
        name, depth, index = match.group(1), 1, match.end()
        while depth and index < len(source):
            depth += (source[index] == "(") - (source[index] == ")")
            index += 1
        params = source[match.end():index - 1]
        signature = re.match(r"\s*->\s*bool\s*=", source[index:])
        if not signature:
            raise ValueError("property %s does not return bool: its twin cannot be built" % name)
        start = index + signature.end()
        stop = ending.search(source, start)
        body = source[start:stop.start() if stop else len(source)].strip()
        if "twin_" + name in names:
            raise ValueError("twin_%s already exists" % name)
        twins.append("$counterexample\nfunction twin_%s(%s) -> bool = {\n  let claim : bool = {\n%s\n  };\n  not(claim)\n}\n" % (name, " ".join(params.split()), body))
    return "\n".join(twins)


def solver_info(pin, solver="z3", allow=False, timeout=None, kill_after=None):
    """One line naming the solver in use, its version against the pin, and the effective timeouts.

    A wrapper script around z3 may rewrite `-T:` (the toolchain's does), so the line says what the gate
    asks for, what a recognised wrapper turns it into, and the gate's own kill limit. `timeout` and `kill_after`
    are the limits of a caller that overrides the defaults (the extra suites ask for 240 and 305 s). A version
    other than the pinned one is an error unless `allow`."""
    timeout = TIMEOUT if timeout is None else timeout
    kill_after = KILL_AFTER if kill_after is None else kill_after
    path = shutil.which(solver)
    if path is None:
        raise ValueError("solver %s not found on PATH" % solver)
    banner = subprocess.run([solver, "--version"], capture_output=True, text=True).stdout.strip()
    found = re.search(r"(\d+\.\d+\.\d+)", banner)
    version = found.group(1) if found else "unknown"
    pinned = Path(pin).read_text().split()[0]
    head = Path(path).read_bytes()[:2]
    note = "binary"
    effective = timeout
    if head == b"#!":
        rewrite = re.search(r"-T:\*\)\s*args\+=\(\"-T:(\d+)\"\)", Path(path).read_text())
        note = "wrapper script%s" % (" rewriting -T to %s s" % rewrite.group(1) if rewrite else " (timeout rewrite not recognised)")
        if rewrite:
            effective = int(rewrite.group(1))
    line = ("z3: %s (%s), %s; pinned %s; per-query timeout requested %d s, effective %d s, the gate kills a query "
            "after %d s" % (path, banner, note, pinned, timeout, effective, kill_after))
    if version != pinned:
        if not allow:
            raise ValueError(line + "\nz3 %s is not the pinned %s (PROVE_ALLOW_Z3=1 accepts it)" % (version, pinned))
        line += "\nwarning: z3 %s is not the pinned %s" % (version, pinned)
    return line


def check_inventory(source, inventory):
    names = set(obligations(source))
    required = set(inventory.split())
    if names != required:
        raise ValueError("proof contract mismatch: missing=%s unexpected=%s" %
                         (sorted(required - names), sorted(names - required)))


class UnprovedObligations(ValueError):
    """Clean mathematical mismatches or undecided solver verdicts."""


class SolverProtocolError(ValueError):
    """Infrastructure/protocol failure, never a mathematical counterexample."""


def report_solver_result(name, wanted, result):
    """Only clean exit-zero SAT/UNSAT are mathematical verdicts.

    Keep protocol diagnostics explicit so a mutation consumer cannot mistake
    solver stderr, malformed output or nonzero exit for a counterexample.
    """
    actual = result.stdout.strip()
    if result.returncode != 0 or result.stderr.strip() or actual not in {"sat", "unsat", "unknown", "timeout"}:
        print("ERROR SMT %s: expected %s, stdout=%r stderr=%r (exit %d)" %
              (name, wanted, result.stdout, result.stderr, result.returncode))
        return "error"
    status = "pass" if actual == wanted else "fail"
    print("%s %s: expected %s, got %s (exit 0)" %
          (status.upper(), name, wanted, actual))
    return status


def check(source, directory, solver="z3"):
    expected = obligations(source)
    files = {p.stem: p for p in Path(directory).glob("*.smt2")}
    if files.keys() != expected.keys():
        raise ValueError("SMT inventory mismatch: missing=%s unexpected=%s" %
                         (sorted(expected.keys() - files.keys()), sorted(files.keys() - expected.keys())))
    failed, errors = [], []
    for name, wanted in expected.items():
        query = files[name].read_text()
        # Sail requests models for sanity counterexamples. The satisfiability
        # result is sufficient here; get-model after unsat is itself a Z3 error.
        query = re.sub(r"(?m)^\(get-model\)\s*$", "", query)
        result = subprocess.run([solver, "-in", "-T:%d" % TIMEOUT], input=query, text=True,
                                capture_output=True, timeout=KILL_AFTER)
        status = report_solver_result(name, wanted, result)
        if status == "error":
            errors.append(name)
        elif status != "pass":
            failed.append(name)
    if errors:
        raise SolverProtocolError("solver protocol failure; unproved obligations: " + ", ".join(errors))
    if failed:
        raise UnprovedObligations("unproved obligations: " + ", ".join(failed))


if __name__ == "__main__":
    try:
        if sys.argv[1] == "--solver-info":
            limits = [int(v) for v in sys.argv[3:5]]
            print(solver_info(sys.argv[2], allow=os.environ.get("PROVE_ALLOW_Z3") == "1",
                              timeout=limits[0] if limits else None, kill_after=limits[1] if limits else None))
            sys.exit(0)
        source = Path(sys.argv[1]).read_text()
        inventory = Path(__file__).resolve().parents[1] / "test/formal/obligations.txt"
        check_vacuity(source)
        check_inventory(source, inventory.read_text())
        # With --twins the project was generated with the twin of every property (props_project.sh --twins).
        check(source + ("\n" + twin_source(source) if "--twins" in sys.argv[3:] else ""), sys.argv[2])
    except UnprovedObligations as error:
        print("model-prove: " + str(error), file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("ERROR SMT model-prove: " + str(error), file=sys.stderr)
        sys.exit(1)
