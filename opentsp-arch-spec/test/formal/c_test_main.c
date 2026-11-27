// Runner of the C tests of test/formal (c_test.sh): loads the profile into the Sail configuration like the harnesses do and runs the
// $[test] functions of the build.
// usage: c_test <config.json>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sail.h"
#include "sail_config.h"

extern void model_test(void);

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
  model_test();                   // runs each $[test] function between model_init and model_fini
  printf("c_test: the $[test] functions passed (%s)\n", argv[1]);
  return 0;
}
