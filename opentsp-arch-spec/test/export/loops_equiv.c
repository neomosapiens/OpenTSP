// The C main of loops_equiv.sh: loads the profile (the model's types read it), then runs loops_equiv_run of loops_equiv.sail.
// usage: loops_equiv <config.json>
#include <stdio.h>
#include <stdlib.h>
#include "sail.h"
#include "sail_config.h"

extern void model_init(void);
extern void model_fini(void);
extern unit zloops_equiv_run(unit);

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s <config.json>\n", argv[0]); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 2; }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc(n + 1);
  if (fread(buf, 1, n, f) != (size_t)n) { perror(argv[1]); return 2; }
  buf[n] = 0;
  fclose(f);
  sail_config_set_string(buf);
  free(buf);
  model_init();
  zloops_equiv_run(UNIT);
  model_fini();
  return 0;
}
