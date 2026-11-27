# TSP reference architecture: extending the specification

This chapter describes how an instruction, unit class or queue extension must remain coherent across specification and implementations. Extensions outside existing profile keys change the contract; update every affected artifact and state compatibility/version implications explicitly. [Design space](design-space.md) · [Cap table](topology.md)

## 1. What is free

| Space | Free today | Where it is defined |
|---|---|---|
| Slice tags | 6 and 7 (decode to OTS-E-DEC-003 and consume one unit) | `encoding.md`, "Word formats" and the decode rule |
| Opcodes per unit class (6-bit field) | VXM 29, MEM 53, C2C 54, CU 56, SXM 56, MXM 60 | `encoding.md`, "Opcode allocation" (the counts are the free opcodes of 64 as of this writing; `isa-db/generated/encoding.json` is the source) |
| Element-type codes | 13 to 15 and 17 to 31 (the 5-bit VXM type code; 16 is bf16) | `encoding.md`, the VXM element type legend |
| Queue ids | none below 144; new ids are appended **above 143** (the growth rule below) | `topology.md` §5, `isa-db/generated/encoding.json` `queues` |
| Reserved zero bits of every 64-bit entry | at least 21 per entry | `encoding.md` |

Base instructions have no per-instruction disable key. The `extensions` profile key  names the extensions a profile declares, and a word that uses an extension the profile does not name does not decode (OTS-E-DEC-003; `encoding.md`, the VXM element type; `extension_codes` in `isa-db/generated/encoding.json`); every entry that no extension owns is part of every profile. Tag 7 remains reserved; it is not an experimental exemption from conformance.

## 2. The growth rule for the queue map

- **Ids 0..143 never change.** They are written into every object and manifest (the image entries of `assembly.md` §4), into `encoding.json` and into `provenance.yaml` (`queues`), and the object digest covers `encoding.json`.
- **New queues take the next ids, 144 and up.** A model that names ids above 143 widens `queue_id` (`range(0, 143)` in `model/core/types.sail`), the issuer table of `model/core/queues.sail`, the queue ranges of `provenance.yaml` (`check_provenance.py` requires the ranges to be contiguous and to end at 143, and its `OPENTSP_SPLIT` table names the blocks: both change with the map), the queue-name table of `assembly.md` §3 and `tsp_as.py`, and the image-entry check of the loaders (`queue_id` of 0..143 or 0xFFFF).
- **The control channel stays 0xFFFF.** Its internal index (144 in the harness `NQ` and in the ISS `geom.rs`) is an implementation detail and moves when the data queues grow; no document or object records it.
- **Fixed sizes in the ISS and the harness** (`geom.rs DATA_QUEUES`, `QUEUES`, the harness `NQ`) are the cap table's row for queues (`topology.md` §1).

## 3. Worked example: add a unit class with tag 6

The example adds a hypothetical unit class `XYZ` with one instruction `Foo` on a 32-bit layout with slice tag 6 and two new queues. No such unit exists; the example lists the touch points in the order a contributor meets them. Every step names the command that shows it is done.

1. **Contract.** State the new semantics, encoding, configuration/profile domain and compatibility consequences before implementation.
2. **Sail model** (normative). Create `opentsp-arch-spec/model/slices/xyz/xyz_insts.sail` with the `union clause instruction`, the `encdec32` mapping, the `assembly` mapping (`"foo" ^ spc ^...`) and the `execute` clause; add the file to `model/tsp.sail_project`. Extend: the slice-tag decode of the text scanner (`model/postlude/iq.sail` and `postlude/isa_query.sail`: which tags are one unit and which two), the issuer of each queue (`model/core/queues.sail`), the timing class (`model/core/timing.sail` and the schema keys), the diagnostics it can raise (`model/core/diagnostics.sail`, only if it needs a new code), and the unit-state staging if it keeps state (`unit_state_flush` in `postlude/tick.sail`). Check: `tools/model_check.sh` (typecheck, every profile).
3. **Catalogue.** Add the entry in `opentsp-arch-spec/tools/catalogue/gen_catalogue.py` (fields, legal values, slots, `timing` class, `streams` and `align`) and regenerate: `python3 opentsp-arch-spec/tools/catalogue/gen_catalogue.py opentsp-arch-spec/isa-db/instructions`. Add the slice tag to `SLICE_TAG` and the unit to `ORDER` of `tools/gen_isa_doc.py`. Check: `tools/isa_db_check.sh` (catalogue checks and reference assembler checks: the catalogue against the model's `encdec` and `assembly` mappings, the pairwise distinctness of the patterns).
4. **Generated documents.** `python3 opentsp-arch-spec/tools/gen_isa_doc.py` rewrites `isa.md`, `encoding.md` and `isa-db/generated/encoding.json`. Check: `python3 opentsp-arch-spec/tools/gen_isa_doc.py --check` (generated documentation).
5. **Provenance.** Add the instruction's record to `isa-db/provenance.yaml` (`instructions`, a `timing_rows` row, `queues` for the new ids, a `config` record for every new profile key) with its basis (`paper`, `figure`, `slides`, `decision` or `implementation-choice`) and a citation or the stated absence of one. Check: `python3 opentsp-arch-spec/test/isa_db/check_provenance.py` (provenance checks).
6. **Hand-written documents.** The unit's semantics in `microarchitecture.md`, its queues in `topology.md` §5, its operand kinds and lexical rules in `assembly.md` §1 and the queue-name table, its diagnostics in the §8 table, and a row in `validation.md`. Check: `just check-docs` and `python3 opentsp-arch-spec/tools/gen_hazard_table.py --check` (a new diagnostic needs a row of the hazard catalogue in `assembly.md`).
7. **Tests that fail without the change** (write them first: RED, then GREEN): a `$[test]` in `model/unit_tests/` (Sail unit tests: `tools/model_unit_tests.sh`), a directed program in `test/directed/` (directed tests), a template in the independent oracle `test/oracle/` (Python: `diff_oracle.py`), and the golden vectors: regenerate with `tools/gen_model_vectors.py` (for `ref` and `mini`), then run it with `--check`. Show that each new test fails against a small mutation of the code and keep the mutant result in the report.
8. **ISS and assemblers.** The ISS decodes and executes the instruction (`opentsp-iss/crates/tsp-iss/src/slices/` and the decode and front-end files `frontend.rs`, `machine.rs`, `geom.rs`); the Rust assembler learns the grammar from the regenerated `encoding.json` (`opentsp-sw-toolchain/crates/ots-isa`, `ots-asm`, `ots-format`) and the reference assembler `tools/tsp_as.py` derives it. Check: `just iss-vectors`, `just iss-diff`, `just asm-diff`, `just programs`, `just e2e` (`AGENTS.md` section 1). The traces and cycle counts of every existing program must be unchanged.
9. **Integrity manifest.** Refresh the tracked specification digests only after all affected artifacts and checks agree. Run just check-freeze to verify the result.

All nine steps form one coherent change. A change that stops at the Sail model, or that adds an instruction without a test that fails first, is not done.

## 4. Smaller extensions

- **A new instruction in an existing unit class** is steps 2 to 9 with an existing slice tag and a free opcode: no decode-rule change, no new queue. The opcode, fields and bits go into the catalogue entry; reserved bits must stay zero (OTS-E-DEC-003 otherwise).
- **A new element type** takes one of the codes 13 to 15, or, as an extension, a code with the type-extension bit set (17 to 31; bf16 is 16): an extension type is declared by a name in the profile key `extensions` (the extensions of version 0.1 are `bf16`, `wide_acc` and `block_fp8`), its codes are listed in `extension_codes` of the catalogue variables, and the model's `ty_valid` and `wtype_ok` accept them only for a profile that declares the name. It widens the type tables of the catalogue (`T4` in `gen_catalogue.py`), the numerics of `model/core/numerics/`, the oracle and the ISS numerics (`opentsp-iss/crates/tsp-numerics`), and the legal-type lists of every VXM entry that accepts it.
- **A new profile key** touches the schema (`config/schema.json` is generated by `sail --output-schema`, with `tools/schema_invariants.py` for the relations Sail cannot state), `provenance.yaml` `config`, both configuration checkers (the C harness `tsp_config_check.c` and the Rust `ots-profile`) and every shipped profile. A key that changes behaviour changes the profile digest of every object built against it.
- **A new queue** without a new unit (a second queue of an existing unit) is §2 plus the issuer table; the unit's instruction semantics do not change.

## 5. What an extension cannot assume

A hardware implementation that follows 0.1 has no obligation to provide an extension. State explicitly which of the extension's behaviour is architectural (a conforming implementation must reproduce it) and which is a model convenience, and where the rule is written. The `extensions` key declares the optional format gates already defined by this version. A new base instruction or unit class changes the interface every conforming implementation of that version must meet; for that reason a new unit class is normally proposed as a new version rather than an erratum.
