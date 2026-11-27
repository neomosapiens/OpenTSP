"""Instantiate production transitions with symbolic port implementations for Sail SMT.

Each production function is copied textually and a table of substitutions rebinds its ports to the
symbolic probes of test/formal/fetch_ports.sail. Every substitution carries the exact number of
occurrences it must make : any other count fails, so a renamed or added use
of a port cannot pass unnoticed. The substitution is also checked to be reversible, so the proof body
equals the production body modulo the table; `--diff` prints that difference. The long-term form is a
Sail `val` ports module selected in tsp.sail_project: that needs the production functions to call their ports through declarations
the proof build can rebind, which is a change of model/ (handed to the package that owns it). Until then the textual copy stands;
`barrier_retire` is proved for one queue, which stands for all 145 only because `check_retire_uniform` shows the loop body treats
every queue alike (the SMT backend does not translate a loop over a symbolic queue index).
"""
from pathlib import Path
import difflib
import re
import sys

SPEC = Path(__file__).resolve().parents[1]
DECLARATION = r"^(?:private )?(?:function |struct |register |val |let )"

# (old, new, exact count): applied in this order, so later entries see earlier results.
FETCH = [("stream_fetch_start", "probe_start", 1), ("stream_fetch_service", "probe_service", 1),
         ("read_planes", "probe_read", 2), ("iq_fetch_ov", "probe_stage", 1),
         ("diag_report", "probe_report", 3),
         ("fill_len(fill_fifo[issue_queue])", "slots_used", 1)]
DECODE = [("iq_text_unc", "probe_text_unc", 2), ("iq_decode_at", "probe_decode_at", 1),
          ("iq_word_unc", "probe_word_unc", 2), ("iq_unit_word", "probe_unit_word", 2),
          ("decode_text", "probe_decode", 1), ("iq_view", "probe_view", 2)]
CSR = [("csr_write", "probe_csr_write", 1), ("v", "write_value", 7), ("host_deliver", "probe_irq_delivery", 1),
       ("host_ring", "unexpected_doorbell", 1)]
# Each iteration reads and writes only its own queue's four registers. Keep the real release
# predicates and assignments for one observed queue.
RETIRE = [("barrier_retire", "probe_retire", 1), ("foreach (k from 0 to 144)", "foreach (k from 3 to 3)", 1)]

BINDINGS = [("slices/cu/cu_insts.sail", ["stream_fetch_start", "stream_fetch_service"], FETCH),
            ("postlude/iq.sail", ["iq_text_unc", "iq_decode_at"], DECODE),
            ("core/csr.sail", ["csr_write"], CSR),
            ("postlude/tick.sail", ["barrier_retire"], RETIRE)]


def extract(source, functions):
    """The text of each named top-level function, from its `function` keyword to the next declaration."""
    bodies = []
    for name in functions:
        # Sail typechecks the extracted body alongside the original on each profile.
        match = re.search(r"^(?:private )?function " + name + r"\(.*?(?=" + DECLARATION + r"|\Z)", source, re.M | re.S)
        if match is None:
            raise ValueError("missing Sail definition " + name)
        bodies.append(match.group())
    return "\n".join(bodies)


def pattern(text):
    return r"\b" + re.escape(text) + (r"\b" if text[-1].isalnum() else "")


def substitute(body, table):
    for old, new, expected in table:
        body, count = re.subn(pattern(old), new, body)
        if count != expected:
            raise ValueError("proof port %r made %d substitutions, expected exactly %d" % (old, count, expected))
    return body


def reverse(proof, table):
    """Undo `substitute`; the result must be the production text, or a new name clashed with an old one."""
    for old, new, expected in reversed(table):
        proof, count = re.subn(pattern(new), lambda _m, old=old: old, proof)
        if count != expected:
            raise ValueError("proof name %r occurs %d times, expected %d: it clashes with a production name" %
                             (new, count, expected))
    return proof


def unified(production, proof, name):
    return "".join(difflib.unified_diff(production.splitlines(True), proof.splitlines(True),
                                        "production/" + name, "proof/" + name))


def bind(path, functions, table, show_diff=False):
    production = extract((SPEC / "model" / path).read_text(), functions)
    proof = substitute(production, table)
    if "barrier_retire" in functions:
        check_retire_uniform(production)
    if reverse(proof, table) != production:
        raise ValueError("%s: the proof body is not the production body modulo the substitution table:\n%s" %
                         (path, unified(production, proof, path)))
    if show_diff:
        print(unified(production, proof, path), end="")
    return proof


RETIRE_REGISTERS = ("release_at", "hw_release", "parked_since", "hw_since")


def check_retire_uniform(production):
    """The proof observes queue 3 of the 145 that `barrier_retire` walks. That stands for every queue only if the loop body treats
    them alike: the loop variable may appear only as the index of the four registers, never in a condition or an expression."""
    body = production.partition("foreach (k from 0 to 144)")[2]
    if not body:
        raise ValueError("barrier_retire: the loop over every queue was not found")
    for register in RETIRE_REGISTERS:
        body = re.sub(register + r"\[k\]", register, body)
    if re.search(r"\bk\b", body):
        raise ValueError("barrier_retire uses its loop variable outside the index of %s: the proof of one queue no longer stands "
                         "for all of them" % ", ".join(RETIRE_REGISTERS))


def build(show_diff=False):
    harness = (SPEC / "test/formal/fetch_ports.sail").read_text()
    fetch, decode, csr, retire = (bind(path, functions, table, show_diff) for path, functions, table in BINDINGS)
    return (harness.replace("// INSERT QUEUE TRANSITIONS", fetch).replace("// INSERT DECODE TRANSITIONS", decode)
            .replace("// INSERT CSR TRANSITION", csr).replace("// INSERT RETIRE TRANSITION", retire))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--diff"]
    try:
        text = build("--diff" in sys.argv[1:])
    except ValueError as error:
        print("fetch_port_proof: " + str(error), file=sys.stderr)
        sys.exit(1)
    if args:
        Path(args[0]).write_text(text)
