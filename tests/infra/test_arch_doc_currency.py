"""Architecture documentation follows current configurable geometry and tool interfaces."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARCH = ROOT / "opentsp-arch-spec/docs"
fails = 0


def text(name):
    return (ARCH / name).read_text()


def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
    fails += 0 if ok else 1


design, valid, topo, perf = (text(n) for n in ("design-space.md", "validation.md", "topology.md", "performance.md"))

check("design-space: the type code space is 5 bits", "three free codes" not in design and "13..15 and 17..31 are free" in design)
check("design-space: poison is an opentsp choice, not a published property",
      "with poison propagation (ISCA20" not in design and "`uncorrectable_poison`" in design)
for key in ("mxm.acc_bits", "extensions"):
    check(f"validation: the conformance key table lists {key}", f"| `{key}` |" in valid)
    check(f"topology: the profile parameters list {key}", f"- `{key}`:" in topo)
check("topology: MXM plane count is a profile value, not 4", "- **MXM:** 4 planes of" not in topo and "planes_per_hemisphere" in topo.split("- **MXM:**")[1][:200])
check("topology: the link count is a profile value", "| 16 configurable links;" not in topo)
check("performance: serving is generalised to the listed geometries", "planned and not done" not in perf and "test_serve_geometry" in perf)
check("performance: the group-form row names bf16", "i16, u16, f16, and bf16 with the extension" in perf)
isa = text("isa.md")
check("isa.md: the f16 capacity is mxm.f16_rows, not mxm.rows/2",
      "mxm.rows/2 f16" not in isa and "mxm.rows/2 for f16" not in isa and "mxm.rows/2 rows" not in isa and "mxm.f16_rows f16, bf16 or bfp8 rows" in isa)
geometry = (ROOT / "opentsp-arch-spec/model/core/geometry.sail").read_text()
check("geometry.sail: the wide accumulators are mxm.acc_depth per plane, not two", "two per plane" not in geometry and "`mxm.acc_depth` per plane" in geometry)
activity = (ROOT / "opentsp-iss/crates/tsp-iss/src/slices/activity.rs").read_text()
check("activity.rs: Activity::new does not say 16 links", "4 matrix planes, 16 links" not in activity)
new_at = activity.index("pub fn new(vlanes: u64")
doc_block = activity[:new_at].rsplit("\n\n", 1)[-1]  # the doc comment directly above Activity::new
check("activity.rs: the doc of Activity::new names the 16 C2C links it builds, not MAX_LINKS", "16 C2C links" in doc_block and "MAX_LINKS" not in doc_block)

geom = (ROOT / "opentsp-iss/crates/tsp-iss/src/geom.rs").read_text()
readme = (ROOT / "opentsp-iss/README.md").read_text()
for const, what in (("MAX_CHIPS", "chips"), ("MAX_LINKS", "chip-to-chip links")):
    value = int(re.search(r"pub const %s: usize = (\d+);" % const, geom).group(1))
    check(f"opentsp-iss/README.md states {const} = {value} ({what}), as geom.rs does",
          re.search(r"\b%d %s\b[^\n]*\(`%s`\)" % (value, what, const), readme) is not None)

validation = (ROOT / "opentsp-arch-spec/docs/validation.md").read_text()
check("validation.md: the 'two accumulators' ISCA20 row names this specification's count",
      "two accumulators (the ISCA20 design; this specification has `mxm.acc_depth`" in validation)

topology = (ROOT / "opentsp-arch-spec/docs/topology.md").read_text()

extending = (ROOT / "opentsp-arch-spec/docs/extending.md").read_text()
check("extending.md: the VXM element type is a 5-bit code with 13..15 and 17..31 free, not a 4-bit field",
      "4-bit VXM type field" not in extending and "13 to 15 and 17 to 31" in extending)
check("extending.md: the `extensions` profile key exists",
      "A planned `extensions`" not in extending and "(OTS-E-DEC-003" in extending)

porting = (ROOT / "opentsp-arch-spec/docs/porting-guide.md").read_text()
check("porting-guide.md: the wide MXM accumulator count is profile-driven",
      "`mxm.acc_depth` wide accumulators per MXM plane" in porting)
check("porting-guide.md: the element-type code is 5 bits and includes bf16",
      "5-bit code space" in porting and "including bf16 when its extension is enabled" in porting)
cli = (ROOT / "opentsp-sw-toolchain/crates/ots-asm/src/cli.rs").read_text()
check("ots-asm prints `mpack ... [--topology T.json]` (the source the README follows)", "ots-asm mpack [--profile P] [--config F] [--topology T.json]" in cli)
toolchain_readme = (ROOT / "opentsp-sw-toolchain/README.md").read_text()
check("opentsp-sw-toolchain/README.md: the mpack usage has [--topology T.json]",
      "ots-asm mpack [--profile P] [--config F] [--topology T.json] -o M.otsm" in toolchain_readme)
iss_sources = "".join(p.read_text() for p in (ROOT / "opentsp-iss/crates/tsp-iss/src").rglob("*.rs"))
check("tsp-iss prints `--chips K | --topology FILE` (the source the README follows)", "--chips K | --topology FILE" in iss_sources)
iss_readme = (ROOT / "opentsp-iss/README.md").read_text()
check("opentsp-iss/README.md: the harness and `run` usage lines have [--chips K | --topology FILE]",
      iss_readme.count("[--chips K | --topology FILE]") >= 2)

check("opentsp-iss/README.md: the matrix planes are sized by mxm.planes_per_hemisphere, not a constant 4",
      "4 matrix planes," not in readme and "`mxm.planes_per_hemisphere` sets its matrix plane count" in readme)
check("extending.md: the extensions of 0.1 are bf16, wide_acc and block_fp8, not one",
      "the one extension of version 0.1" not in extending and "`wide_acc`" in extending and "`block_fp8`" in extending)

verify_src = (ROOT / "opentsp-iss/crates/tsp-iss/src/verify.rs").read_text()
verify_usage = re.search(r'pub const USAGE: &str = "usage: (tsp-iss verify [^"]*)";', verify_src).group(1)
check("opentsp-sw-toolchain/README.md: the verify usage line is the one verify.rs prints, with [--topology FILE]",
      f"`{verify_usage}`" in toolchain_readme)

sys.exit(1 if fails else 0)
