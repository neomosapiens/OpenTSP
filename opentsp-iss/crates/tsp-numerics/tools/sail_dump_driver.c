// Sail-evaluator driver of the ISS numerics differential check (tools/sail_dump.sh).
// Links the C output of the model's numerics (built together with sail_dump.sail) and evaluates
// the model's own functions on a job list.
//
// usage: sail_dump_driver <config.json> <jobs.txt>
// job lines (op, type codes and modes decimal; values hex):
//   tr <op 21..24> <width 16|32> <x>        fp_tr_value
//   i8 <op 21..24> <x>                      i8_transcendental (saturated i8)
//   bin <op> <type> <x> <y>                 k_binary
//   un <op> <type> <x>                      k_unary / transcendental dispatch of `apply`
//   cmp <mode> <type> <x> <y>               k_cmp
//   cvt <src type> <dst type> <x>           k_convert_any
//   conv <op 25..32> <x>                    k_convert
//   nz <type> <x>                           k_nonzero
// output: every job line, then " = " and the result as 8 hex digits ("NC" if the Ziv loop did
// not converge), one line per job.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sail.h"
#include "sail_config.h"

extern void model_init(void);
extern void model_fini(void);
extern uint64_t zdmp_tr(int64_t op, int64_t w, uint64_t x);
extern uint64_t zdmp_i8(int64_t op, uint64_t x);
extern uint64_t zdmp_bin(int64_t op, int64_t t, uint64_t x, uint64_t y);
extern uint64_t zdmp_un(int64_t op, int64_t t, uint64_t x);
extern uint64_t zdmp_cmp(int64_t mode, int64_t t, uint64_t x, uint64_t y);
extern uint64_t zdmp_cvt(int64_t st, int64_t dt, uint64_t x);
extern uint64_t zdmp_conv(int64_t op, uint64_t x);
extern uint64_t zdmp_anybit(int64_t t, uint64_t x);

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

static void emit(const char *line, uint64_t r, int wide) {
  if (wide && (r >> 32)) printf("%s = NC\n", line);
  else printf("%s = %08x\n", line, (unsigned)(uint32_t)r);
}

int main(int argc, char **argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s <config.json> <jobs.txt>\n", argv[0]); return 2; }
  char *cfg = slurp(argv[1]);
  sail_config_set_string(cfg);
  model_init();
  FILE *jobs = fopen(argv[2], "r");
  if (!jobs) { perror(argv[2]); return 2; }
  static char buf[1 << 16];
  setvbuf(stdout, buf, _IOFBF, sizeof buf);
  char line[256];
  long n = 0;
  while (fgets(line, sizeof line, jobs)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    if (len == 0 || line[0] == '#') continue;
    char kind[8];
    long a, b;
    unsigned x, y;
    n++;
    if (sscanf(line, "tr %ld %ld %x", &a, &b, &x) == 3) emit(line, zdmp_tr(a, b, x), 1);
    else if (sscanf(line, "i8 %ld %x", &a, &x) == 2) emit(line, zdmp_i8(a, x), 1);
    else if (sscanf(line, "bin %ld %ld %x %x", &a, &b, &x, &y) == 4) emit(line, zdmp_bin(a, b, x, y), 0);
    else if (sscanf(line, "un %ld %ld %x", &a, &b, &x) == 3) emit(line, zdmp_un(a, b, x), 0);
    else if (sscanf(line, "cmp %ld %ld %x %x", &a, &b, &x, &y) == 4) emit(line, zdmp_cmp(a, b, x, y), 0);
    else if (sscanf(line, "cvt %ld %ld %x", &a, &b, &x) == 3) emit(line, zdmp_cvt(a, b, x), 0);
    else if (sscanf(line, "conv %ld %x", &a, &x) == 2) emit(line, zdmp_conv(a, x), 0);
    else if (sscanf(line, "nz %ld %x", &a, &x) == 2) emit(line, zdmp_anybit(a, x), 0);
    else { fprintf(stderr, "bad job line: %s\n", line); return 2; }
    (void)kind;
  }
  fflush(stdout);
  model_fini();
  fprintf(stderr, "sail_dump_driver: %ld jobs\n", n);
  return 0;
}
