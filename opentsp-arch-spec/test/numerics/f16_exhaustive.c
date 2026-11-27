// Exhaustive binary16 add, sub and mul: the model (Sail, C backend) against the host's _Float16 arithmetic, an independent
// implementation (IEEE 754 round to nearest even in the host's floating-point unit or its compiler runtime).
// For every pair (x, y) of the 2^32 pairs in a range of x values:
//   - a NaN result of the host must be a NaN result of the model and the model's NaN must be the canonical 0x7E00 (the class is
//     compared, not the payload: the model's NaN is canonical, the host's depends on the operands);
//   - every other result must match bit for bit (signed zeros and infinities included).
// usage: f16_exhaustive <op: add|sub|mul> <first x> <count of x values>
// prints "ok <pairs> <nan results>" or the first mismatches and "errors <pairs> <errors>".
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "sail.h"
#include "sail_config.h"

extern void model_init(void);
extern uint64_t zf16_add(uint64_t, uint64_t);
extern uint64_t zf16_sub(uint64_t, uint64_t);
extern uint64_t zf16_mul(uint64_t, uint64_t);

static _Float16 of_bits(uint16_t b) { _Float16 h; memcpy(&h, &b, 2); return h; }
static uint16_t to_bits(_Float16 h) { uint16_t b; memcpy(&b, &h, 2); return b; }
static int is_nan16(uint16_t b) { return (b & 0x7c00) == 0x7c00 && (b & 0x03ff) != 0; }

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: %s add|sub|mul <first x> <count>\n", argv[0]); return 2; }
  int op = !strcmp(argv[1], "add") ? 0 : !strcmp(argv[1], "sub") ? 1 : !strcmp(argv[1], "mul") ? 2 : -1;
  if (op < 0) { fprintf(stderr, "unknown operation %s\n", argv[1]); return 2; }
  uint64_t first = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0), pairs = 0, errors = 0, nans = 0;
  model_init();
  for (uint64_t k = 0; k < count; k++) {
    uint16_t x = (uint16_t)(first + k);
    volatile _Float16 hx = of_bits(x);
    for (uint32_t y = 0; y < 65536; y++) {
      volatile _Float16 hy = of_bits((uint16_t)y);
      uint16_t want = to_bits(op == 0 ? (_Float16)(hx + hy) : op == 1 ? (_Float16)(hx - hy) : (_Float16)(hx * hy));
      uint16_t got = (uint16_t)(op == 0 ? zf16_add(x, y) : op == 1 ? zf16_sub(x, y) : zf16_mul(x, y));
      pairs++;
      int bad;
      if (is_nan16(want)) { nans++; bad = got != 0x7E00; } else bad = got != want;
      if (bad && errors++ < 10) printf("MISMATCH %s %04x %04x: model %04x host %04x\n", argv[1], x, y, got, want);
    }
  }
  printf("%s %llu %llu\n", errors ? "errors" : "ok", (unsigned long long)pairs, errors ? (unsigned long long)errors : (unsigned long long)nans);
  return errors != 0;
}
