// Exhaustive 32-bit encoding check over the Sail model (the C backend of the normative
// mappings). For every one of the 2^32 words w (split across processes by range):
//   - decode(w) then encode must give w back whenever w decodes (isa_roundtrip32);
//   - w decodes iff exactly one 32-bit catalogue entry's match pattern accepts it with
//     legal field values (catalogue tables compiled in by gen_exhaustive_tables.py).
// usage: exhaustive32 <config.json> <first> <count>   prints status, decoded, errors and examined words.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "sail.h"
#include "sail_config.h"
#include "tsp_config_check.h"
#include "tsp_config_schema.h"
#include "exhaustive_tables.h"

extern void model_init(void);
extern bool zisa_roundtrip32(uint64_t w);
extern bool zisa_decodes32(uint64_t w);

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: %s <config> <first> <count>\n", argv[0]); return 2; }
  tsp_config_load("exhaustive32", argv[1], TSP_CONFIG_SCHEMA);   // 
  model_init();
  tsp_config_model_check("exhaustive32", argv[1]);
  uint64_t first = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0), decoded = 0, errors = 0;
  uint64_t k = 0;
  for (; k < count; k++) {
    uint32_t w = (uint32_t)(first + k);
    bool dec = zisa_decodes32(w);
    int cat = catalogue_matches32(w);
    if (dec) {
      decoded++;
      if (!zisa_roundtrip32(w)) { if (errors++ < 10) printf("ROUNDTRIP %08x\n", w); }
    }
    if (dec != (cat == 1) || cat > 1) { if (errors++ < 10) printf("CATALOGUE %08x model=%d catalogue_matches=%d\n", w, dec, cat); }
  }
  printf("%s %llu %llu %llu\n", errors ? "errors" : "ok", (unsigned long long)decoded,
         (unsigned long long)errors, (unsigned long long)k);
  return errors != 0;
}
