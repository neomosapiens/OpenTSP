# ISS performance data

- `energy.yaml`: the post-hoc energy table (every value an estimate). Use it with the activity counters:

  ```
  tsp-iss run --profile ref --stats=stats.json --energy opentsp-iss/perf/energy.yaml --clock-mhz 900 prog.elf
  ```

  `--stats` writes the counters (`activity`), and with `--energy` also the priced run (`energy`: joules per part, watts at the
  clock, a canonical busy mix, the calibration targets); standard error gets a short summary. The counters are defined in
  `crates/tsp-iss/src/slices/activity.rs`, the pricing in `crates/tsp-iss/src/slices/energy.rs`. The table's header says what
  is estimated and what is not implemented (PWR-001 semantics). The selected mix is neither a universal power upper bound nor a physical power prediction; the existing output key remains `full_activity_watts`.
- **What the energy table scales with.** `static_w` (split over the unit classes by `static_share`) and the per-cycle energies of
  `cycles` are calibrated for one structure, the `ref` profile named under `reference` (320 lanes, 88 MEM slices, 4 matrix planes of
  320 rows, 16 links). A chip is priced at the ratio of its own size to it, class by class (MEM: slices times lanes; VXM and SXM:
  lanes; MXM: planes times rows times lanes; C2C: links; the control unit does not scale), read from the profile through the
  counters' `structure`, and the canonical busy mix counts the profile's own MEM slices and matrix planes. The `energy` object
  of `--stats` lists the `structure` and the `class_scale` it used. The same program on `mini` (16 lanes) and on `ref` therefore
  gets different watts. What is calibrated: only the `ref` structure, against the two published ranges (the canonical busy mix of
  `ref` is 138.1 W, inside 35-350 W and below the 300 W maximum: `tests/counters.rs`, `tests/perf/test_counters_xcheck.py`;
  `vxm_valid` on `ref` averages 38.2 W and on `mini` 8.1 W; power figures for other profiles depend on the model's size scaling). What is not: every
  per-event and per-cycle value (all estimates), the split of the static power, the choice of the size measure per class, and any
  structure other than `ref` (the scaling is a proportionality, not a measurement of a smaller chip).
- **Numeric range.** Pricing retains magnitudes through products, unit conversions and chip aggregation, so an overflowing
  intermediate does not discard finite final joules or watts. Outputs remain floating-point approximations: a truly
  unrepresentable final quantity is JSON `null` and `n/a` in the summary, with an explanatory note; unavailable values are
  never added as zero. Tiny positive final quantities may round to zero. Reference values enter the scale ratios as parsed,
  without integer truncation or saturation. A displayed scale is also rounded: it can round to zero while its retained
  positive magnitude still contributes to a representable final estimate.
- **ACC x128 pricing.** `acc_lanes` and `mxm_acc_pj_per_lane` price one 32-bit lane unit per matrix row; an ACC with `wide_acc` / `acc_bits` 128 is not charged 4× and has no separate wide-lane counter or energy row.
- The counters are off unless `--stats` or `--energy` asks for them; see [simulator activity and inspection](../README.md#activity-and-schedule-inspection).
- Set `ISS_PERF_BASE_DIR` to an independently prepared source checkout for comparative benchmarks. The benchmark builds that checkout’s tools and materializes its fixture in its own Python environment. See the [simulator README](../README.md) for entry points.
