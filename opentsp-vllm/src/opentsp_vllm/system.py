"""`System`: N chips in lock step with the host interface of every chip, over the binding.

The class is the binding's `_tsp_iss.System` (see its docstrings: construction, `load_object`, `dma_write`, `run`, `egress`, the CSR
and doorbell calls) with a few helpers of the host's side. A run, a diagnostic and a cycle limit are as described there; a diagnostic
raises `opentsp_vllm.chip.TspChipError` with `chip`, `cycle`, `queue` and `code`.
"""

from . import _tsp_iss


class System(_tsp_iss.System):
    @classmethod
    def from_objects(cls, profile, objects, **options):
        """A system of `len(objects)` chips of `profile` with `objects[i]` (bytes or a path of an `.elf`) loaded on chip i."""
        system = cls(profile, len(objects), **options)
        for chip, elf in enumerate(objects):
            system.load_object(chip, elf)
        return system

    def vectors(self, chip, start=0):
        """The vectors PciSend gave the host of `chip`, from the `start`-th on, as bytes (the cycles are in `egress`)."""
        return [vector for _, vector in self.egress(chip, start)]

    def run_to_done(self, max_cycles=None):
        """`run` again and again through the interrupts (HostNotify) until every chip is done or a limit is hit; returns the list of
        `(chip, cycle, vector)` interrupts met on the way and the last result."""
        notifies = []
        while True:
            result = self.run(max_cycles=max_cycles)
            notifies += result["notifies"]
            if result["reason"] != "notify":
                return notifies, result
