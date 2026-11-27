// Sampled binary32 add, sub and mul: the model (Sail, C backend) against the host's `float` arithmetic (IEEE 754 round to nearest even;
// built with -ffp-contract=off, no fast math). 2^64 pairs cannot be run, so the pairs are drawn from strata that exercise the cases of
// the arithmetic: uniform bit patterns, operands of nearly equal exponent (cancellation), subnormal and near-zero operands, and
// operands near the overflow threshold; each stratum has its own share of the count. NaN results are compared by class (the model's
// must be the canonical 0x7FC00000), everything else bit for bit.
// usage: f32_sample <add|sub|mul> <seed> <pairs per stratum>
// prints "ok <pairs> <nan results>" or the first mismatches and "errors <pairs> <errors>".
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "sail.h"
#include "sail_config.h"

extern void model_init(void);
extern uint64_t zf32_add(uint64_t, uint64_t);
extern uint64_t zf32_sub(uint64_t, uint64_t);
extern uint64_t zf32_mul(uint64_t, uint64_t);

static uint64_t rng_state;
static uint64_t next64(void) {   // xorshift64*
  rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
  return rng_state * 0x2545F4914F6CDD1DULL;
}
static float of_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t to_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
static int is_nan32(uint32_t b) { return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0; }
static uint32_t pattern(uint32_t sign, uint32_t ex, uint32_t frac) { return (sign << 31) | ((ex & 0xff) << 23) | (frac & 0x7fffff); }

static void draw(int stratum, uint32_t *x, uint32_t *y) {
  uint64_t r = next64(), s = next64();
  uint32_t ex, fx = (uint32_t)r, fy = (uint32_t)s;
  switch (stratum) {
    case 0: *x = (uint32_t)(r >> 32); *y = (uint32_t)s; break;                                        // uniform
    case 1: ex = (uint32_t)(r >> 40) % 254 + 1;                                                          // equal exponents, +-2
            *x = pattern((r >> 33) & 1, ex, fx); *y = pattern((s >> 33) & 1, ex + ((uint32_t)(s >> 40) % 5) - 2, fy); break;
    case 2: *x = pattern((r >> 33) & 1, (uint32_t)(r >> 40) % 4, fx);                                   // subnormal and tiny
            *y = pattern((s >> 33) & 1, (uint32_t)(s >> 40) % 8, fy); break;
    default: *x = pattern((r >> 33) & 1, 246 + (uint32_t)(r >> 40) % 9, fx);                            // near overflow
             *y = pattern((s >> 33) & 1, 118 + (uint32_t)(s >> 40) % 16, fy); break;                     // (and the factors that bring a product there)
  }
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: %s add|sub|mul <seed> <pairs per stratum>\n", argv[0]); return 2; }
  int op = !strcmp(argv[1], "add") ? 0 : !strcmp(argv[1], "sub") ? 1 : !strcmp(argv[1], "mul") ? 2 : -1;
  if (op < 0) { fprintf(stderr, "unknown operation %s\n", argv[1]); return 2; }
  rng_state = strtoull(argv[2], 0, 0) * 0x9E3779B97F4A7C15ULL + 1;
  uint64_t count = strtoull(argv[3], 0, 0), pairs = 0, errors = 0, nans = 0;
  model_init();
  for (int stratum = 0; stratum < 4; stratum++) {
    for (uint64_t k = 0; k < count; k++) {
      uint32_t x, y;
      draw(stratum, &x, &y);
      volatile float hx = of_bits(x), hy = of_bits(y);
      uint32_t want = to_bits(op == 0 ? hx + hy : op == 1 ? hx - hy : hx * hy);
      uint32_t got = (uint32_t)(op == 0 ? zf32_add(x, y) : op == 1 ? zf32_sub(x, y) : zf32_mul(x, y));
      pairs++;
      int bad;
      if (is_nan32(want)) { nans++; bad = got != 0x7FC00000u; } else bad = got != want;
      if (bad && errors++ < 10) printf("MISMATCH %s stratum %d %08x %08x: model %08x host %08x\n", argv[1], stratum, x, y, got, want);
    }
  }
  printf("%s %llu %llu\n", errors ? "errors" : "ok", (unsigned long long)pairs, errors ? (unsigned long long)errors : (unsigned long long)nans);
  return errors != 0;
}
