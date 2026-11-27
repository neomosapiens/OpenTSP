// isa_query: answers encoding queries from the Sail model, for the ISA catalogue check
// (test/isa_db/check_isa_db.py) and for a second implementation's decoder (porting-guide.md
// sec. 3). One query per stdin line, one answer per stdout line (all numbers hex):
//   32 <word>                  ->  <roundtrip 0|1> TAB <assembly text>   (bits 31..0 as a 32-bit layout)
//   64 <word>                  ->  same                                  (the word as a 64-bit layout)
//   text <u0> <u1>             ->  <code|-> TAB <units 1|2> TAB <assembly text>
//   e32 <word> / e64 <word>    ->  <catalogue entry> TAB <timing class>  (entry_of, timing_of)
//   class <entry name>         ->  <timing class> in the model's table of catalogue entries, or -
//   gap <queue> <row 0=E,1=W> <0 read|1 write>  ->  the station gap, or -1 (isa_station)
// `text` applies the text-level decode rule of encoding.md (encoding.json decode_rule;
// model/postlude/iq.sail decode_text) to the unit u0 and, for a two-unit instruction, u1: the
// bundle bit (bit 31) is cleared, bits 30..28 pick the layout (0, 3, 5: one unit; 1, 2, 4: two
// units; 6, 7: OTS-E-DEC-003, one unit); a word that does not decode is OTS-E-DEC-003 (text
// "illegal"). The code is the decode-level one: placement (OTS-E-QUE-002) is checked after it.
// There are no 128-bit instructions; the former `128` mode is gone.
// usage: isa_query <config.json>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "sail.h"
#include "sail_config.h"
#include "tsp_config_check.h"
#include "tsp_config_schema.h"

extern void model_init(void);
extern void model_fini(void);
extern void zisa_disasm32(sail_string *out, uint64_t w);
extern void zisa_disasm64(sail_string *out, uint64_t w);
extern bool zisa_roundtrip32(uint64_t w);
extern bool zisa_roundtrip64(uint64_t w);
extern void zisa_decode_text(sail_string *out, uint64_t u0, uint64_t u1);
extern void zisa_entry32(sail_string *out, uint64_t w);
extern void zisa_entry64(sail_string *out, uint64_t w);
extern void zisa_timing32(sail_string *out, uint64_t w);
extern void zisa_timing64(sail_string *out, uint64_t w);
extern void zisa_entry_class(sail_string *out, sail_string name);
extern void zisa_timing_type32(sail_string *out, uint64_t w);
extern void zisa_timing_type64(sail_string *out, uint64_t w);
extern bool zisa_entry_takes_type(sail_string name, sail_string type);
extern void zisa_station(sail_int *out, int64_t q, int64_t r, bool w);

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s <config.json>\n", argv[0]); return 2; }
  // The same configuration check as tsp_sim: exit 2 naming the key.
  tsp_config_load("isa_query", argv[1], TSP_CONFIG_SCHEMA);
  model_init();
  tsp_config_model_check("isa_query", argv[1]);
  sail_string s;
  CREATE(sail_string)(&s);
  char mode[8];
  while (scanf("%7s", mode) == 1) {
    uint64_t a = 0;
    if (strcmp(mode, "text") == 0) {
      uint64_t u1;
      if (scanf("%" SCNx64 " %" SCNx64, &a, &u1) != 2) return 2;
      int w = 1 + (((a >> 28) & 7) == 1 || ((a >> 28) & 7) == 2 || ((a >> 28) & 7) == 4);
      zisa_decode_text(&s, a & 0xffffffffu, u1 & 0xffffffffu);
      printf("%s\t%d\t%s\n", strcmp(s, "illegal") == 0 ? "OTS-E-DEC-003" : "-", w, s);
    } else if (strcmp(mode, "32") == 0 || strcmp(mode, "64") == 0) {
      if (scanf("%" SCNx64, &a) != 1) return 2;
      bool ok;
      if (mode[0] == '3') { ok = zisa_roundtrip32(a); zisa_disasm32(&s, a); }
      else { ok = zisa_roundtrip64(a); zisa_disasm64(&s, a); }
      printf("%d\t%s\n", ok ? 1 : 0, s);
    } else if (strcmp(mode, "gap") == 0) {
      int q, r, w;
      if (scanf("%d %d %d", &q, &r, &w) != 3) return 2;
      if (q < 0 || q > 143 || r < 0 || r > 1) return 2;
      sail_int out;
      CREATE(sail_int)(&out);
      zisa_station(&out, q, r, w != 0);
      printf("%ld\n", (long)CONVERT_OF(mach_int, sail_int)(out));
      KILL(sail_int)(&out);
    } else if (strcmp(mode, "class") == 0) {
      char name[64];
      if (scanf("%63s", name) != 1) return 2;
      zisa_entry_class(&s, name);
      printf("%s\n", s);
    } else if (strcmp(mode, "e32") == 0 || strcmp(mode, "e64") == 0) {
      if (scanf("%" SCNx64, &a) != 1) return 2;
      sail_string t, ty;
      CREATE(sail_string)(&t);
      CREATE(sail_string)(&ty);
      if (mode[1] == '3') { zisa_entry32(&s, a); zisa_timing32(&t, a); zisa_timing_type32(&ty, a); }
      else { zisa_entry64(&s, a); zisa_timing64(&t, a); zisa_timing_type64(&ty, a); }
      printf("%s\t%s\t%s\n", s, t, ty);
      KILL(sail_string)(&ty);
      KILL(sail_string)(&t);
    } else if (strcmp(mode, "takes") == 0) {
      char name[64], type[64];
      if (scanf("%63s %63s", name, type) != 2) return 2;
      printf("%d\n", zisa_entry_takes_type(name, type) ? 1 : 0);
    } else {
      fprintf(stderr, "isa_query: unknown mode %s (32, 64, text, e32, e64, class, takes, gap)\n", mode);
      return 2;
    }
    fflush(stdout);
  }
  KILL(sail_string)(&s);
  model_fini();
  return 0;
}
