// SPDX-License-Identifier: Apache-2.0
// Enumerate a half-open, contiguous interval through the actual Sail-generated
// entry point. u64 induction avoids wrapping at the last binary32 pattern.
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
extern uint64_t znumeric_cap_eval(int64_t format, int64_t op, uint64_t x);

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
    fprintf(stderr, "usage: numeric_cap_driver CONFIG FORMAT OP START END\n");
    return 2;
  }
  uint64_t format = number(argv[2]), op = number(argv[3]);
  uint64_t start = number(argv[4]), end = number(argv[5]);
  if (format > 5 || op < 21 || op > 24) return 2;
  unsigned width = format == 0 ? 32 : format <= 2 ? 16 : 8;
  uint64_t limit = UINT64_C(1) << width;
  if (start >= end || end > limit) return 2;
  char *config = read_config(argv[1]);
  sail_config_set_string(config);
  model_init();
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  uint64_t count = 0, digest = UINT64_C(14695981039346656037);
  for (uint64_t x = start; x < end; x++) {
    uint64_t value = znumeric_cap_eval((int64_t)format, (int64_t)op, x);
    if (value >> 32) {
      printf("{\"status\":\"NONCONVERGENCE\",\"format\":%" PRIu64
             ",\"op\":%" PRIu64 ",\"input\":%" PRIu64
             ",\"start\":%" PRIu64 ",\"end\":%" PRIu64
             ",\"checked_before_failure\":%" PRIu64 "}\n",
             format, op, x, start, end, count);
      model_fini(); free(config); return 1;
    }
    digest = (digest ^ x) * UINT64_C(1099511628211);
    digest = (digest ^ value) * UINT64_C(1099511628211);
    count++;
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double seconds = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
  printf("{\"status\":\"COMPLETE\",\"format\":%" PRIu64
         ",\"op\":%" PRIu64 ",\"start\":%" PRIu64 ",\"end\":%" PRIu64
         ",\"checked\":%" PRIu64 ",\"digest\":\"%016" PRIx64
         "\",\"seconds\":%.9f}\n", format, op, start, end, count, digest, seconds);
  model_fini(); free(config);
  return count == end - start ? 0 : 3;
}
