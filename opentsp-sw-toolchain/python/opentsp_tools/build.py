"""Build object and weight packages, with independent comparisons for conformance tests."""
import os
from pathlib import Path
import subprocess
import sys

from .paths import SPEC

TSP_AS = os.path.join(SPEC, "tools", "tsp_as.py")


class Fail(RuntimeError):
    pass


def config_of(profile):
    return os.path.join(SPEC, "config", profile + ".json")


def sh(cmd, stdin=None, timeout=600, ok=(0,)):
    result = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
    if result.returncode not in ok:
        name = os.path.basename(cmd[0]) if cmd[0] != sys.executable else cmd[1]
        raise Fail(f"{name} exited {result.returncode}: {result.stderr.decode(errors='replace').strip()[:400]}")
    return result


def warnings_of(stderr):
    return [ln.split(": ")[1] for ln in stderr.decode().splitlines() if ": OTS-W-" in ln]


class CaseBuilder:
    def __init__(self, options):
        self.a = options

    def build_artifacts(self, case, cdir, work):
        """Build the object and optional preload package consumed by the native runtime."""
        native, _ = self._assemble(case, Path(cdir), Path(work), compare=False)
        return native

    def _assemble(self, case, cdir, work, *, compare):
        profile = case["profile"]
        cfg = ["--config", config_of(profile)]
        program = str(cdir / case["program"])
        native_object = str(work / "prog.elf")
        result = sh([self.a.asm, "asm", *cfg, "-o", native_object, program])
        expected = case["expect"].get("asm_warnings", [])
        if warnings_of(result.stderr) != expected:
            raise Fail(f"ots-asm warnings {warnings_of(result.stderr)} != {expected}")
        weights = []
        for name, weight in case["weights"].items():
            path = work / f"w_{name}.bin"
            data = bytes.fromhex(weight["hex"]) if "hex" in weight else (cdir / weight["data"]).read_bytes()
            path.write_bytes(data)
            weights.append(f"{name}={weight['dtype']}:{path}")
        pack = ["wpack", "--profile", os.path.basename(profile), *cfg]
        native_package = str(work / "in.safetensors") if weights else None
        if native_package:
            sh([self.a.asm, *pack, "-o", native_package, *weights])
        reference_object = reference_package = None
        if compare:
            reference_object = str(work / "prog.tsp_as.elf")
            result = sh([sys.executable, TSP_AS, "asm", program, "-o", reference_object, *cfg])
            if Path(native_object).read_bytes() != Path(reference_object).read_bytes():
                raise Fail("ots-asm and tsp_as.py objects differ")
            if warnings_of(result.stderr) != expected:
                raise Fail(f"tsp_as.py warnings {warnings_of(result.stderr)} != {expected}")
            if native_package:
                reference_package = str(work / "in.tsp_as.safetensors")
                sh([sys.executable, TSP_AS, *pack, "-o", reference_package, *weights])
                if Path(native_package).read_bytes() != Path(reference_package).read_bytes():
                    raise Fail("ots-asm and tsp_as.py weight packages differ")
        return (native_object, native_package), (reference_object, reference_package)

    def build_input(self, case, cdir, work):
        """Validate both assemblers and build a text input for reference-harness comparisons."""
        (obj, pkg), (ref_obj, ref_pkg) = self._assemble(case, Path(cdir), Path(work), compare=True)
        cfg = ["--config", config_of(case["profile"])]
        weight_args = ["--weights", pkg] if pkg else []
        reference_weight_args = ["--weights", ref_pkg] if ref_pkg else []
        native = sh([self.a.asm, "objinput", *cfg, *weight_args, obj]).stdout
        reference = sh([sys.executable, TSP_AS, "objinput", ref_obj, *cfg, *reference_weight_args]).stdout
        if native != reference:
            raise Fail("ots-asm and tsp_as.py objinput outputs differ")
        lines = [native.decode()]
        # Case keys use decimal slice and address. The text harness takes hexadecimal.
        for location, hex_bytes in case["preload"].items():
            slice_id, addr = (int(x) for x in location.split(":"))
            payload = " ".join(f"{byte:x}" for byte in bytes.fromhex(hex_bytes))
            lines.append(f"S {slice_id:x} {addr:x} {payload}\n")
        for location in case["dump"]:
            slice_id, addr = (int(x) for x in location.split(":"))
            lines.append(f"D {slice_id:x} {addr:x}\n")
        for cycle, hex_bytes in case.get("host", {}).get("dma", []):
            payload = " ".join(f"{byte:x}" for byte in bytes.fromhex(hex_bytes))
            lines.append(f"H dma {cycle:x} {payload}\n")
        text = "".join(lines)
        (Path(work) / "input.txt").write_text(text)
        return text, (obj, pkg)
