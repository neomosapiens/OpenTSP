"""The cycle-annotated kernel cookbook points to its maintained implementations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    cook = (ROOT / "docs/kernel-cookbook.md").read_text()
    asm = (ROOT / "opentsp-arch-spec/docs/assembly.md").read_text()
    errors = []
    if "<!-- cookbook:" not in cook:
        errors.append("kernel-cookbook.md lacks cycle-annotated cookbook markers")
    if "kernel-cookbook" not in asm:
        errors.append("assembly.md does not point at the kernel cookbook")
    if "opentsp-sw-toolchain/python/opentsp_tools/programs/kernels/" not in cook:
        errors.append("cookbook does not name the toolchain kernel path")
    if errors:
        print("FAIL:")
        for e in errors:
            print(" ", e)
        raise SystemExit(1)
    print("ok: kernel cookbook paths")


if __name__ == "__main__":
    main()
