"""Independent Python reference model (oracle) of the TSP ISA, spec version 0.1.

Written from the architecture documents in opentsp-arch-spec/docs/
(microarchitecture.md, topology.md, encoding.md, isa.md) and the catalogue
isa-db/generated/encoding.json, not translated from the Sail model. It is used by
diff_oracle.py to run random programs on tsp_sim and on this
oracle and compare every diagnostic and every SRAM write.

Modules:
  numerics  element kernels: integer, boolean, IEEE-754 binary16/binary32 with
            round-to-nearest-even, correctly rounded transcendentals (mpmath)
  isa       table-driven decoder and encoder over encoding.json
  machine   the cycle machine: queues, stream register file, AGT, Repeat,
            barriers, MEM / VXM / MXM / SXM / C2C semantics
"""
