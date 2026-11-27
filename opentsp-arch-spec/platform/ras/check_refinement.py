"""Bind Sail's unconstrained gate witness to the Yosys netlist and check for disagreement."""
from pathlib import Path
import re
import subprocess
import sys


def check(sail_path, rtl_path):
    query = Path(sail_path).read_text()
    # Sail emits one existential input per structure. Bind by type and field,
    # never by generated variable number or the backend's structure layout.
    declarations = dict((kind, name) for name, kind in re.findall(r"\(declare-const (\S+) (Z\w+)\)", query))
    ports = {
        "Zattempt": dict(running="active", fenced="fenced", epoch="tag", members="participants", acks="acknowledged", remaining="remaining"),
        "Zevidence": dict(reset_seen="power_reset", fatal="fault", bad_state="integrity_bad", needs_service="service_required", report_epoch="report_tag", settled="settled",
                          write_valid="dma_valid", write_epoch="dma_tag", visible="output_visible", visible_epoch="visible_tag"),
        "Zgate_outputs": dict(write_allowed="allow_write", published="publish"),
    }
    if declarations.keys() != ports.keys():
        raise ValueError("unmapped Sail inputs")
    preamble, mismatch = query.split("(assert ", 1)
    mismatch = "(assert " + mismatch.split("(check-sat)", 1)[0]
    lines = [preamble, Path(rtl_path).read_text(), "(declare-const hw |ras_commit_s|)"]
    for kind, mapping in ports.items():
        fields = set(re.findall(rf"\({kind}_z(\w+) ", preamble))
        if fields != mapping.keys():
            raise ValueError(f"unmapped {kind} fields: {fields ^ mapping.keys()}")
        for field, signal in mapping.items():
            lines.append(f"(assert (= ({kind}_z{field} {declarations[kind]}) (|ras_commit_n {signal}| hw)))")
    # First require a positive publication with the binding active. Then ask
    # if either observed output can disagree for any values of regs or inputs.
    lines += ["(push)", "(assert (|ras_commit_n publish| hw))", "(check-sat)", "(pop)", mismatch, "(check-sat)"]
    run = subprocess.run(["z3", "-T:60", "-in"], text=True, input="\n".join(lines),
                         capture_output=True, timeout=65, check=True)
    if run.stdout.split() != ["sat", "unsat"] or run.stderr:
        raise ValueError(f"RTL/Sail refinement failed: {run.stdout} {run.stderr}")
    print("PASS RTL publication reachable; RTL write and publication equivalent to Sail for all inputs and states")


if __name__ == "__main__":
    from check_smt_obligations import check as solve_obligations, check_inventory

    outputs = Path(sys.argv[1])
    sources = Path(__file__).resolve().parent
    model = (sources / "commit.sail").read_text()
    check_inventory(model, (sources / "obligations.txt").read_text())
    solve_obligations(model, outputs / "sail")
    check(outputs / "sail/c_gate_always_agrees.smt2", outputs / "gate.smt2")
