// Sail-evaluator driver for the exhaustive f32 transcendental check
// (tools/check_transcendental_f32.sh). Links the C output of the model's numerics (built
// by the script together with transcendental_f32_driver.sail) and evaluates the model's
// own fp_tr_value(op, FMT32, x) on a job list.
//
// usage: transcendental_f32_driver <config.json> <jobs.txt>
//   job line "E <op> <x hex> <expected hex>": evaluate, compare with the expected f32 bits
//   job line "A <op> <x hex> <p>"           : print the enclosure accuracy at precision p
//   op is 21 TanH, 22 Exp, 23 RSqrt, 24 Log; x is the f32 bit pattern.
// Output: "MISMATCH ..." / "NONCONV ..." lines, "ACC ..." lines and, per op, one
// "SUMMARY op=<op> evals=<n> mismatches=<m> nonconv=<k> levels=<count at p=64,128,..,4096> seconds=<t>" line.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sail.h"
#include "sail_config.h"

extern void model_init(void);
extern void model_fini(void);
extern uint64_t ztf32_eval(int64_t op, uint64_t x);
extern uint64_t ztf32_level(int64_t op, uint64_t x);
extern uint64_t ztf32_acc(int64_t op, uint64_t x, int64_t p);

static char *slurp(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(2); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc(n + 1);
  if (fread(buf, 1, n, f) != (size_t)n) { perror(path); exit(2); }
  buf[n] = 0;
  fclose(f);
  return buf;
}

// Does the model's fp_tr_value reach tr_eval (the Ziv loop) for this input?
static int reaches_ziv(int op, uint32_t x) {
  uint32_t ex = (x >> 23) & 0xff, mag = x & 0x7fffffff;
  int neg = x >> 31;
  if (ex == 255 || mag == 0) return 0;
  if ((op == 23 || op == 24) && neg) return 0;
  if (op == 24 && x == 0x3f800000u) return 0;
  if (op == 22 && ex >= 127 + 9) return 0;                       // |x| >= 512
  if (op == 21 && ex >= 127 + 6) return 0;                       // |x| >= 64
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <config.json> <jobs.txt>\n", argv[0]); return 2; }
  char *cfg = slurp(argv[1]);
  sail_config_set_string(cfg);
  model_init();
  FILE *jobs = fopen(argv[2], "r");
  if (!jobs) { perror(argv[2]); return 2; }
  long evals[25] = {0}, mism[25] = {0}, nonconv[25] = {0}, levels[25][7];
  double secs[25] = {0};
  memset(levels, 0, sizeof levels);
  char line[256];
  while (fgets(line, sizeof line, jobs)) {
    char kind; int op; unsigned x, y;
    if (sscanf(line, "%c %d %x %x", &kind, &op, &x, &y) != 4 || op < 21 || op > 24) {
      if (line[0] == '#' || line[0] == '\n') continue;
      fprintf(stderr, "bad job line: %s", line);
      return 2;
    }
    if (kind == 'A') {
      printf("ACC op=%d x=%08x p=%u acc=%u\n", op, x, y, (unsigned)ztf32_acc(op, x, y));
      continue;
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t r = ztf32_eval(op, x);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    secs[op] += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    evals[op]++;
    if (r >> 32) {
      nonconv[op]++;
      printf("NONCONV op=%d x=%08x\n", op, x);
    } else if ((uint32_t)r != y) {
      mism[op]++;
      printf("MISMATCH op=%d x=%08x sail=%08x expected=%08x\n", op, x, (unsigned)(uint32_t)r, y);
    }
    if (reaches_ziv(op, x)) {
      uint32_t lv = (uint32_t)ztf32_level(op, x);
      int k = 0;
      while (k < 7 && (64u << k) != lv) k++;
      if (k < 7) levels[op][k]++;
    }
  }
  for (int op = 21; op <= 24; op++) {
    if (!evals[op]) continue;
    printf("SUMMARY op=%d evals=%ld mismatches=%ld nonconv=%ld levels=", op, evals[op], mism[op], nonconv[op]);
    for (int k = 0; k < 7; k++) printf("%s%ld", k ? "," : "", levels[op][k]);
    printf(" seconds=%.3f\n", secs[op]);
  }
  model_fini();
  return (mism[21] + mism[22] + mism[23] + mism[24] + nonconv[21] + nonconv[22] + nonconv[23] + nonconv[24]) ? 1 : 0;
}
