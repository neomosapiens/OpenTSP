// SPDX-License-Identifier: Apache-2.0
// Check the complete actual bounded integer-root argument family.
// Precision index selects one of the seven actual Ziv precisions; u64 half-open
// ranges cover every m2 in [0,2^25), including the extra zero input.
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sail.h"
#include "sail_config.h"

extern void model_init(void);
extern void model_fini(void);
extern void zisqrt(sail_int *result, sail_int n);
static const unsigned precisions[7] = {64,128,256,512,1024,2048,4096};

static uint64_t number(const char *s) {
  char *end;
  errno = 0;
  uint64_t n = strtoull(s, &end, 10);
  if (errno || s[0] == '-' || !s[0] || *end) {
    fprintf(stderr, "invalid nonnegative decimal integer: %s\n", s);
    exit(2);
  }
  return n;
}

static char *read_config(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  if (fseek(f, 0, SEEK_END)) { perror(path); exit(2); }
  long n = ftell(f);
  if (n < 0 || fseek(f, 0, SEEK_SET)) { perror(path); exit(2); }
  char *p = malloc((size_t)n + 1);
  if (!p || fread(p, 1, (size_t)n, f) != (size_t)n || fclose(f)) {
    fprintf(stderr, "cannot read configuration\n"); exit(2);
  }
  p[n] = 0;
  return p;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: isqrt_driver CONFIG PRECISION_INDEX 23 START END\n");
    return 2;
  }
  uint64_t format = number(argv[2]), op = number(argv[3]);
  uint64_t start = number(argv[4]), end = number(argv[5]);
  if (format > 6 || op != 23) return 2;
  uint64_t limit = UINT64_C(1) << 25;
  if (start >= end || end > limit) return 2;
  char *config = read_config(argv[1]);
  sail_config_set_string(config);
  model_init();
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  uint64_t count = 0, digest = UINT64_C(14695981039346656037);
  sail_int input, root;
  mpz_t lower, upper;
  mpz_inits(input, root, lower, upper, NULL);
  for (uint64_t x = start; x < end; x++) {
    // Input construction and postcondition are independent GMP arithmetic.
    // zisqrt is the actual unaltered Sail-generated production function.
    mpz_set_ui(input, (unsigned long)x);
    mpz_mul_2exp(input, input, 2 * precisions[format]);
    zisqrt(&root, input);
    mpz_mul(lower, root, root);
    mpz_add_ui(upper, root, 1);
    mpz_mul(upper, upper, upper);
    if (mpz_sgn(root) < 0 || mpz_cmp(lower, input) > 0 || mpz_cmp(input, upper) >= 0) {
      printf("{\"status\":\"POSTCONDITION_FAILURE\",\"format\":%" PRIu64
             ",\"op\":%" PRIu64 ",\"input\":%" PRIu64
             ",\"precision\":%u,\"start\":%" PRIu64 ",\"end\":%" PRIu64
             ",\"checked_before_failure\":%" PRIu64 "}\n",
             format, op, x, precisions[format], start, end, count);
      mpz_clears(input, root, lower, upper, NULL);
      model_fini(); free(config); return 1;
    }
    digest = (digest ^ x) * UINT64_C(1099511628211);
    digest = (digest ^ mpz_get_ui(root)) * UINT64_C(1099511628211);
    count++;
  }
  mpz_clears(input, root, lower, upper, NULL);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double seconds = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
  printf("{\"status\":\"COMPLETE\",\"format\":%" PRIu64
         ",\"op\":%" PRIu64 ",\"start\":%" PRIu64 ",\"end\":%" PRIu64
         ",\"checked\":%" PRIu64 ",\"digest\":\"%016" PRIx64
         "\",\"seconds\":%.9f}\n", format, op, start, end, count, digest, seconds);
  model_fini(); free(config);
  return count == end - start ? 0 : 3;
}
