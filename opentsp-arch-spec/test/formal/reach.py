#!/usr/bin/env python3
"""Which model functions do the proof obligations call? 

The call graph comes from Sail itself: `sail --ddump-tc-ast` prints the type-checked AST of the props project
for one profile, with overloaded operators already resolved to their functions and no comments or strings, so an
edge is a name of a defined function in the body of another (the earlier script read the source text with a regex
and counted every identifier). Obligations are the names in test/formal/obligations.txt; the model functions are
those defined under model/. Reachable is still not the same as covered: a function that an obligation calls is
exercised by it only on the paths its inputs reach. The report is an upper bound, and the dynamic measure is
`just model-coverage`.

A model function can be reached by three kinds of obligation source: the SMT obligations of the model's own props project (the AST_FILE
and --obligations), the extra SMT suites of test/formal/props/ (each has its own project, so its own AST: `--source NAME=AST:INVENTORY`) and the Lean
theorems (`--lean DIR`: the model functions named in the proof files, followed through the call graph of the model's AST). With more than one
source the report prints one figure per source and the union.

Usage: reach.py AST_FILE [--model DIR] [--obligations FILE] [--functions FILE] [--per-obligation]
                [--source NAME=AST:INVENTORY ...] [--lean DIR]
"""
from pathlib import Path
import argparse
import re
import sys

SPEC = Path(__file__).resolve().parents[2]
TOP = r"(?:\$\[|\$\w+|val |function |mapping |private |struct |enum |union |register |let |type |overload |newtype |mapping |bitfield |default |scattered |outcome |instantiation |end |///)"
TOP_LINE = re.compile(r"^" + TOP)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def base(name):
    """A mapping `m` is compiled to functions m_forwards, m_backwards and their `_matches` forms: all are `m`."""
    return re.sub(r"_(?:forwards|backwards)(?:_matches)?$", "", name)


def ast_functions(text):
    """name -> set of identifiers in the body of each `function` of the dump (clauses are merged)."""
    bodies = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"(?:function|mapping)\s+(?:clause\s+)?(\w+)\b", lines[index])
        if not match:
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not TOP_LINE.match(lines[end]):
            end += 1
        body = "\n".join(lines[index:end])
        bodies.setdefault(base(match.group(1)), set()).update(base(t) for t in IDENT.findall(body[match.end():]))
        index = end
    return bodies


def strip_comments(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def source_functions(model):
    """name -> number of code lines (blank and comment lines excluded) of the clauses defined under `model`."""
    sizes = {}
    for path in sorted(Path(model).rglob("*.sail")):
        if path.relative_to(model).parts[0] in ("props", "unit_tests"):
            continue
        lines = strip_comments(path.read_text()).splitlines()
        index = 0
        while index < len(lines):
            match = re.match(r"(?:private\s+)?(?:function|mapping)\s+(?:clause\s+)?(\w+)\b", lines[index])
            if not match:
                index += 1
                continue
            end = index + 1
            while end < len(lines) and not re.match(r"^(?:\$|private |function |mapping |val |struct |enum |union |register |let |type |overload |newtype |mapping |bitfield |default |scattered |end |\w+ \{)", lines[end]):
                end += 1
            sizes[match.group(1)] = sizes.get(match.group(1), 0) + sum(1 for l in lines[index:end] if l.strip())
            index = end
    return sizes


def reach(graph, root):
    seen, todo = {root}, [root]
    while todo:
        for callee in graph.get(todo.pop(), ()):
            if callee in graph and callee not in seen:
                seen.add(callee)
                todo.append(callee)
    return seen


def analyse(ast_text, model, obligations, copies=()):
    """The report data: per model function the obligations that call it, and per obligation the functions it calls."""
    graph = ast_functions(ast_text)
    sizes = source_functions(model)
    missing = [name for name in obligations if name not in graph]
    if missing:
        raise ValueError("obligations missing from the AST dump: " + ", ".join(sorted(missing)))
    called_by = {name: [] for name in sizes}
    per_obligation = {}
    for name in obligations:
        calls = (reach(graph, name) & set(sizes)) - {name}
        per_obligation[name] = sorted(calls)
        for fn in calls:
            called_by[fn].append(name)
    copied = sorted(fn for fn in copies if fn in sizes and not called_by[fn])
    return sizes, called_by, per_obligation, copied


def report(sizes, called_by, per_obligation, copied, per=False):
    total_fn, total_lines = len(sizes), sum(sizes.values())
    hit = [fn for fn, who in called_by.items() if who]
    hit_lines = sum(sizes[fn] for fn in hit)
    out = ["reach: %d of %d model functions and %d of %d code lines are called by at least one obligation "
           "(%.0f%% and %.0f%%; an upper bound on coverage)" % (len(hit), total_fn, hit_lines, total_lines,
                                                              100.0 * len(hit) / total_fn, 100.0 * hit_lines / total_lines)]
    if copied:
        out.append("reach: %d production functions are proved only through a textual copy in the port proofs "
                   "(tools/fetch_port_proof.py): %s" % (len(copied), ", ".join(copied)))
    if per:
        for name, calls in per_obligation.items():
            out.append("reach: %-34s calls %3d model functions" % (name, len(calls)))
    return "\n".join(out)


def lean_roots(directory, sizes):
    """The model functions that the Lean files under `directory` name (comments and strings are not stripped: a name in a comment counts,
    which makes the figure an upper bound, like the rest)."""
    names = set()
    for path in sorted(Path(directory).rglob("*.lean")):
        names.update(IDENT.findall(path.read_text()))
    return sorted(name for name in names if name in sizes)


def lean_reached(ast_text, directory, sizes):
    """Model functions reachable from the functions the Lean files name."""
    graph = ast_functions(ast_text)
    reached = set()
    for root in lean_roots(directory, sizes):
        reached |= (reach(graph, root) if root in graph else {root}) & set(sizes)
    return reached


def figure(label, hit, sizes):
    total_fn, total_lines = len(sizes), sum(sizes.values())
    hit_lines = sum(sizes[fn] for fn in hit)
    return ("reach: %-24s %3d of %d model functions and %d of %d code lines (%.0f%% and %.0f%%)" %
            (label, len(hit), total_fn, hit_lines, total_lines, 100.0 * len(hit) / total_fn, 100.0 * hit_lines / total_lines))


def source_report(sources, sizes):
    """One figure per source and the union: `sources` maps a label to the set of model functions it reaches."""
    out = [figure(label, hit, sizes) for label, hit in sources.items()]
    union = set().union(*sources.values()) if sources else set()
    out.append(figure("union of all sources", union, sizes))
    return "\n".join(out)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ast")
    parser.add_argument("--model", default=str(SPEC / "model"))
    parser.add_argument("--obligations", default=str(SPEC / "test/formal/obligations.txt"))
    parser.add_argument("--functions", help="write every model function and the obligations that call it to this file")
    parser.add_argument("--per-obligation", action="store_true")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=AST:INVENTORY",
                        help="an extra SMT obligation source with its own AST dump and inventory of obligation names")
    parser.add_argument("--lean", metavar="DIR", help="the Lean proof files: the model functions they name count as reached")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(SPEC / "tools"))
    import fetch_port_proof
    copies = [name for _path, functions, _table in fetch_port_proof.BINDINGS for name in functions]
    try:
        sizes, called_by, per, copied = analyse(Path(args.ast).read_text(), args.model,
                                                Path(args.obligations).read_text().split(), copies)
    except ValueError as error:
        print("reach: " + str(error), file=sys.stderr)
        return 1
    print(report(sizes, called_by, per, copied, args.per_obligation))
    if args.source or args.lean:
        sources = {"model-prove": {fn for fn, who in called_by.items() if who}}
        try:
            for spec in args.source:
                label, _, files = spec.partition("=")
                ast, _, inventory = files.partition(":")
                _s, extra_by, _p, _c = analyse(Path(ast).read_text(), args.model, Path(inventory).read_text().split())
                sources["extra:" + label] = {fn for fn, who in extra_by.items() if who}
        except (ValueError, OSError) as error:
            print("reach: " + str(error), file=sys.stderr)
            return 1
        if args.lean:
            sources["lean"] = lean_reached(Path(args.ast).read_text(), args.lean, sizes)
        print(source_report(sources, sizes))
    if args.functions:
        Path(args.functions).write_text("".join("%s\t%d\t%s\n" % (fn, sizes[fn], " ".join(who))
                                                for fn, who in sorted(called_by.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
