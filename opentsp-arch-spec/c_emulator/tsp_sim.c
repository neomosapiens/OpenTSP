// tsp_sim: harness for the Sail-generated opentsp TSP model (sail-riscv: c_emulator/).
//
// Reads a program on stdin, one line each:
//   S <slice> <addr> <b0> .. <bn>             preload one SRAM word (hex bytes, lane 0 first)
//   Q <queue> <u0> <u1> ...                  the initial IQ image of a queue: 32-bit text units
//                                            (queue 0..8f, or ffff = the control channel); several
//                                            Q lines for one queue append
//   D <slice> <addr>                         dump this SRAM word after the run
//   C <chip>                                 the following lines are for chip <chip> (--chips)
//   R <csr>                                  print this CSR after the run (X line)
//   H start <cycle>                          the queues start dispatching at <cycle> (default 0)
//   H dma <cycle> <b0> .. <bn>               the host's DMA engine pushes a vector (the ingress FIFO)
//   H db <cycle> <i>                         the host rings doorbell i (CSR 0x300 + i)
//   H csr <cycle> r|w <addr> [<value>]       the host reads or writes a CSR (a read prints T csr)
//   P ctrl <cycle> <link>                    the control unit of the chip sends a ctrl packet on <link> (see "The control unit")
//   P csr <cycle> <link> r|w <addr> [<v>]    the control unit sends a csr packet on <link>: the chip at its far end reads or
//                                            writes its CSR and answers a read (host.remote_csr; see "The control unit")
//   F sram <cycle> <slice> <addr> <superlane> <bit> [t|p]
//                                            ECC fault injection: a bit of a stored SRAM word flips at the start of the cycle
//   F srf <cycle> e|w <stream> <gap> <superlane> <bit> [t|p]
//                                            the same for the value in a stream register (the row is the word e or w);
//                                            bit 0..7f data, 80..88 check bits c[0..8], 89 the spare bit (model/core/ecc.sail);
//                                            two lines for the same bit in one cycle are an input error
//   F link <cycle> <link> <flit> corr|uncorr [t|p]
//                                            link FEC fault injection: flit <flit> (0 = header, 1 .. vlanes / 8 = payload, the last = tail) of the vector
//                                            this chip sends on <link> in <cycle> arrives with a simple error the FEC corrects (corr) or a burst
//                                            it only detects (uncorr); the chip must send a vector on <link> in <cycle> (microarchitecture.md sec. 17)
//   F mxm <cycle> <plane> <lane> <bit> [t|p]
//                                            matrix-unit fault injection: bit <bit> (0..5f) of the exact sum of lane <lane> of the ABC that executes
//                                            on <plane> in <cycle> flips; an ABC must execute on <plane> in <cycle> (microarchitecture.md sec. 18)
//   H inject <cycle> <queue> <u0> <u1> ...   the host pushes instruction text into a queue's IQ
//                                            (every chip has its own host interface: the H lines after a `C <chip>` line act on
//                                            that chip, a line without one on chip 0; the words r, w, start, dma, db, csr, inject
//                                            are not hex)
// All numbers are hex. slice < 88 and an existing MEM slice (slice mod 44 below
// geometry.mem_slices_per_hemisphere; the queue of an absent slice takes a Q line, and every
// instruction on it is OTS-E-QUE-002), addr < words_per_slice, queue 0..8f or ffff, every unit
// at most ffffffff; anything else is an input error (stderr, exit 2), as is an image of more
// than icu.iq_depth bytes after the padding units (ffffffff) are dropped. The program is
// the images and the SRAM preload: there are no per-cycle records (the text of a queue
// reaches its IQ at cycle 0 or through Ifetch/IfetchM, assembly.md
// sec. 7).
//
// Writes "E <first-code|-> <cycle> <count>", then "W <slice> <addr> <bytes>" per D line, then
// "Y <cycle> <bytes>" per vector PciSend gave the host (vlanes bytes, lane 0 first), then
// "X <csr> <value>" per R line (hex).
// --trace adds "T srf <cycle> <E|W>:<stream> gap <gap> <value>", "T sram <cycle> slice <s> addr <a> <value>"
// and "T diag <cycle> queue <q> <code>" lines as they happen; a value is the vector as vlanes hex byte pairs
// (lane 0 first) after one space.
// --trace also prints the host's lines: "T dma <cycle> in|wait|out", "T irq <cycle> msix <n>" and
// "T csr <cycle> <addr> <value>" (a remote read: "T csr <reply cycle> <chip> <addr> <value>", when the reply reaches the chip that
// asked) and "T pkt <cycle> csr|ctrl <link>" (the control unit took a packet in). They are not part of the conformance trace digest.
// --trace also prints the ECC events: "T ecc <cycle> queue <q> corr|unc <row>:<stream>|sram <slice>:<addr> sl <superlane> pos <p>".
// --trace also prints the link FEC events, at the arrival cycle of the vector, in the receiving chip's section:
// "T ecc <cycle> link <l> corr|unc flit <f>". It also prints the corrections of the matrix-unit sum check, at the ABC's execute cycle:
// "T ecc <cycle> queue <q> mxm corr plane <p> lane <r>".
// --lint adds "T lint <cycle> queue <q> <code>" lines (simulator lints, OTS-W-..., non-normative; not counted in E).
// With --chips K > 1 each chip's output is preceded by a "C <chip>" line.
//
// --on-irq stop|continue|replay is the host runtime's policy for the error interrupt (MSI-X vector 0; microarchitecture.md sec. 16):
// continue (the default) runs on, as the chip does not stop. stop ends the run at the end of the cycle of the first error
// interrupt, prints the results as they stand and exits with status 5. replay runs the program to its end and, if it raised
// an error interrupt, runs it once more from reset with the `F` lines marked `p` (persistent) applied again and those marked `t`
// (transient, the default) dropped: the second run's output follows the first, after the line "T replay start" (always printed,
// not only with --trace), and "T replay resolved" (no error interrupt in the second run: exit 0) or "T replay persistent"
// (exit 5) ends the output. A run with no error interrupt is unchanged by the option. stop and replay need --chips 1.
//
// The run starts with tick(0) and ends when the model has nothing left to do
// (model_quiescent, no next event). Idle cycles are skipped: after tick(t) the next tick
// is at model_next_event(t). --max-cycles N (default 10^8): if the run would need a cycle
// beyond N, it stops with a message on stderr and exit code 3. An exception of the model (an
// internal_error: a fault of the model, never an outcome of a program) stops it at once with a
// message on stderr and exit code 4 (as do the other consistency failures of the harness).
//
// C2C links (model/core/callbacks.sail; topology and latency: the C2C section below): a
// vector sent at cycle c on a link arrives at the far end at c + the link's latency. With
// one chip every link loops back; K = 2..8 chips form one fully connected node; K > 8
// chips form ceil(K / 8) nodes joined by global links (K at most 264); --topology FILE names
// any other wiring of up to 512 chips. With K > 1 every chip runs in its own process
// (the model's state is global); they run in lock step windows of the smallest link
// latency, which no packet can cross (--window N shortens them; the output does not depend on it).
//
// The configuration is checked against config/schema.json (embedded at build time,
// c_emulator/tsp_config_check.c) before the model sees it; an invalid one is an input
// error: "tsp_sim: config <path>: <key>: <reason>", exit 2.
//
// usage: tsp_sim <config.json> [--trace] [--observe] [--lint] [--on-irq stop|continue|replay] [--max-cycles N] [--chips K | --topology FILE] [--window N]
//                [--link-latency N | --link-latency-seed S | --link-jitter-seed S]
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "sail.h"
#include "sail_config.h"
#include "cJSON.h"
#include "tsp_model_if.h"
#include "tsp_config_check.h"
#include "tsp_config_schema.h"   // TSP_CONFIG_SCHEMA, generated by tools/build_sim.sh
#include "model.h"               // the generated model's interface: have_exception, current_exception

extern void model_init(void);
extern void model_fini(void);
extern unit ztick(sail_int);
extern unit zreset(unit);
extern bool zmodel_quiescent(unit);
extern unit zmodel_finish(unit);
extern void zmodel_next_event(sail_int *out, sail_int c);
extern void zmodel_poll_event(sail_int *out, unit u);
extern unit zmodel_scrub_before(sail_int c);
extern sail_int zscrub_next;                   // next existing model patrol slot (natural, may exceed u64)
extern int64_t zscrub_period;                  // config range0..65535; zero disables patrol
extern int64_t ziq_depth;
extern int64_t zvlanes;
extern int64_t zwords_per_slice;
extern int64_t zmem_slices_per_hemisphere;
extern sail_int zc2c_link_latency;
extern sail_int zc2c_rx_depth;                   // the receive FIFO of a link holds this many vectors
extern int64_t zhost_cycles_per_vector, zhost_ingress_depth, zhost_doorbells;
extern bool zhost_remote_csr;
extern unit zmodel_csr_write(uint64_t addr, uint64_t v, sail_int c);
extern uint64_t zmodel_csr_read(uint64_t addr);
extern unit zmodel_inject_unit(int64_t q, sail_int c, uint64_t u);
extern unit zmodel_set_start(sail_int c);
extern unit zmodel_flip_sram(int64_t slice, sail_int addr, int64_t sl, int64_t bit);
extern unit zmodel_link_fec(sail_int c, int64_t l, int64_t flit, bool unc);
extern unit zmodel_inject_mxm(int64_t plane, sail_int lane, int64_t bit);
extern void zmodel_mxm_pending(sail_int *out, unit u);
extern int64_t zmxm_rows;
extern int64_t zmxm_planes;
extern void zmodel_link_flits(sail_int *out, unit u);
extern int64_t zmodel_link_flit_mask(int64_t f);
extern bool zmodel_flip_iq(int64_t q, sail_int word, int64_t bit);
extern unit zmodel_flip_srf(sail_int cycle, bool w, int64_t stream, int64_t gap, int64_t sl, int64_t bit);

#define NQ 145           // 144 data queues + control channel (index 144)
#define NSLICES 88
#define MAXCHIPS 512            // chip_id is 9 bits (model/slices/c2c/c2c_insts.sail)
#define MAXLANES 512            // 32 superlanes of 16 lanes: the most a profile can have (core/geometry.sail)
// One bit per flit of a link transfer: a header flit and one flit per 8 lanes, 65 for 32 superlanes.
typedef unsigned __int128 flitmask_t;
#define DUMP_MIN_LANES 320       // the least number of bytes of a `W` line (the width of the format since the first profile)
typedef unsigned __int128 cycle_t;
#define NONE ((cycle_t)-1)

static uint64_t max_cycles = 100000000ULL;
static void over_budget(cycle_t t);
static void cycle_decimal(char b[40], cycle_t t) {
  size_t n = 0;
  do { b[n++] = (char)('0' + t % 10); t /= 10; } while (t);
  b[n] = 0;
  for (size_t i = 0; i < n / 2; i++) { char c = b[i]; b[i] = b[n - 1 - i]; b[n - 1 - i] = c; }
}
static void set_cycle(sail_int n, cycle_t c) {
  uint64_t w[2] = {(uint64_t)c, (uint64_t)(c >> 64)};
  mpz_import(n, 2, -1, sizeof *w, 0, 0, w);
}
static cycle_t model_cycle(sail_int n) {
  if (mpz_sgn(n) < 0) return NONE;
  if (mpz_sizeinbase(n, 2) > 128) {
    fprintf(stderr, "tsp_sim: future model event exceeds internal 128-bit time\n"); exit(4);
  }
  uint64_t w[2] = {0, 0}; size_t count = 0;
  mpz_export(w, &count, -1, sizeof *w, 0, 0, n);
  return (cycle_t)w[0] | ((cycle_t)w[1] << 64);
}

static int vl(void);
static void (*process_cleanup)(void);
static void *xrealloc(void *p, size_t n) {
  p = realloc(p, n ? n : 1);
  if (!p) { fprintf(stderr, "tsp_sim: out of memory\n"); if (process_cleanup) process_cleanup(); exit(2); }
  return p;
}
static void *xcalloc(size_t count, size_t size) {
  void *p = calloc(count, size);
  if (!p) { fprintf(stderr, "tsp_sim: out of memory\n"); if (process_cleanup) process_cleanup(); exit(2); }
  return p;
}
#define GROW(arr, n, cap) do { if ((n) == (cap)) { (cap) = (cap) ? 2 * (cap) : 64; (arr) = xrealloc((arr), (cap) * sizeof *(arr)); } } while (0)

// ---------------------------------------------------------------------------------------
// Input (all chips).

typedef struct { int chip; int q; uint32_t *u; size_t n, cap; } img_t;   // initial IQ image (q: 0..143, 144 = control)
typedef struct { int chip; long slice, addr; unsigned char b[MAXLANES]; int n; } pre_t;
typedef struct { int chip; long slice, addr; } dump_t;

// The host's script (H lines) and the CSR reads (R lines), per chip.
enum { H_DMA, H_CSR_W, H_CSR_R, H_INJECT, H_FLIP_SRAM, H_FLIP_SRF, H_FLIP_IQ, H_FLIP_MXM };   // H_FLIP_*: the F lines (ECC fault injection), applied like the host's actions
typedef struct { int chip; int kind; uint64_t cycle, addr, val; int q; int fw, fa, fsl, fbit, persist; uint32_t *u; size_t n, cap; unsigned char b[MAXLANES]; } hev_t;
typedef struct { int chip; uint64_t addr; } rcsr_t;
// `P ctrl <cycle> <link>` and `P csr <cycle> <link> r|w <addr> [<value>]`: the control unit of a chip sends a ctrl or a csr packet
// on a link (a remote CSR access is one hop: the far end of the link serves it). id numbers the packets of the input.
enum { PK_CTRL, PK_CSR_R, PK_CSR_W };
typedef struct { int chip; uint64_t cycle, id; int link; int kind; uint64_t addr, val; } pk_t;
static pk_t *in_pk;      static size_t n_in_pk, cap_in_pk;
static uint64_t next_pkt_id;
// `F link <cycle> <link> <flit> corr|uncorr`: one flit of the vector chip `chip` sends on `link` in `cycle` (link FEC, microarchitecture.md sec. 17).
typedef struct { int chip; uint64_t cycle; int link, flit, unc, used, persist; } lf_t;
static lf_t *in_lf; static size_t n_in_lf, cap_in_lf;
static hev_t *in_host;   static size_t n_in_host, cap_in_host;
static rcsr_t *in_rcsr;  static size_t n_in_rcsr, cap_in_rcsr;
static uint64_t in_start[MAXCHIPS];                    // H start per chip (0: the default)

static img_t *in_img;   static size_t n_in_img, cap_in_img;
static pre_t *in_pre;   static size_t n_in_pre, cap_in_pre;
static dump_t *in_dump; static size_t n_in_dump, cap_in_dump;

static size_t lineno;
static int my_chip, n_chips = 1;
static int chips_given;                           // --chips was given
static const char *topo_path;                     // --topology
static void input_error(const char *msg) {
  fprintf(stderr, "tsp_sim: input line %zu: %s\n", lineno, msg);
  exit(2);
}

// The next hex field; NULL at end of line.
static int hexfield(char **p, uint64_t *v) {
  char *s = *p;
  while (*s == ' ' || *s == '\t' || *s == '\r') s++;
  if (*s == '\0' || *s == '\n') return 0;
  if (*s == '-' || *s == '+') input_error("expected a hex number");
  char *e;
  errno = 0;
  unsigned long long x = strtoull(s, &e, 16);
  if (e == s || (*e && *e != ' ' && *e != '\t' && *e != '\r' && *e != '\n')) input_error("expected a hex number");
  if (errno == ERANGE) input_error("number out of range");
  *v = x; *p = e;
  return 1;
}
static uint64_t need(char **p, const char *what) {
  uint64_t v;
  if (!hexfield(p, &v)) { char m[96]; snprintf(m, sizeof m, "missing %s", what); input_error(m); }
  return v;
}
static void at_end(char **p) { uint64_t v; if (hexfield(p, &v)) input_error("too many fields"); }

// The next word (a run of letters); "" at end of line.
static void wordfield(char **p, char *w, size_t cap) {
  char *s = *p;
  while (*s == ' ' || *s == '\t' || *s == '\r') s++;
  size_t n = 0;
  while ((*s >= 'a' && *s <= 'z') && n + 1 < cap) w[n++] = *s++;
  w[n] = 0;
  *p = s;
}

static int link_peer(int c, int l, int *pl);
extern int64_t zc2c_links;
static void read_pk_line(char **p, int chip) {
  char w[16], rw[16];
  wordfield(p, w, sizeof w);
  int kind = PK_CTRL;
  if (strcmp(w, "csr") == 0) kind = PK_CSR_R;
  else if (strcmp(w, "ctrl") != 0) input_error("unknown packet type (expected ctrl or csr)");
  uint64_t c = need(p, "cycle"), l = need(p, "link"), a = 0, v = 0;
  if (kind == PK_CSR_R) {
    if (!zhost_remote_csr) input_error("P csr: remote CSR access is off (config host.remote_csr)");
    wordfield(p, rw, sizeof rw);
    if (strcmp(rw, "w") == 0) kind = PK_CSR_W;
    else if (strcmp(rw, "r") != 0) input_error("P csr: expected r or w");
    a = need(p, "csr address");
    if (a > 0xfff) input_error("csr address out of range (0..fff)");
    if (kind == PK_CSR_W) v = need(p, "value");
  }
  at_end(p);
  int pl;
  if (l >= (uint64_t)zc2c_links || link_peer(chip, (int)l, &pl) < 0) input_error("P: that link is not connected");
  GROW(in_pk, n_in_pk, cap_in_pk);
  in_pk[n_in_pk++] = (pk_t){chip, c, next_pkt_id++, (int)l, kind, a, v};
}

// `F sram <cycle> <slice> <addr> <superlane> <bit> [t|p]` and `F srf <cycle> e|w <stream> <gap> <superlane> <bit> [t|p]`:
// a bit of a stored SRAM word or of a stream register's value flips at the start of the cycle (before the tick). Fields
// are hex like every other line; bit 0..7f is a data bit, 80..88 a check bit c[0..8], 89 the spare bit (model/core/ecc.sail).
// t (transient, the default) or p (persistent) matters only to --on-irq replay: the replay run applies the p lines again and drops the t lines.
static uint64_t link_flit_count(void);
static void read_link_fault(char **p, int chip) {
  char o[16], tp[16];
  uint64_t c = need(p, "cycle"), l = need(p, "link"), f = need(p, "flit");
  if (l >= (uint64_t)zc2c_links) input_error("link out of range (>= c2c.links)");
  int pl;
  if (link_peer(chip, (int)l, &pl) < 0) input_error("F link: that link is not connected");
  if (f >= link_flit_count()) input_error("flit out of range (>= the vector's flits: header, payload flits, tail)");
  wordfield(p, o, sizeof o);
  int unc = strcmp(o, "uncorr") == 0;
  if (!unc && strcmp(o, "corr") != 0) input_error("F link: expected corr or uncorr");
  wordfield(p, tp, sizeof tp);
  if (*tp && strcmp(tp, "t") != 0 && strcmp(tp, "p") != 0) input_error("F: expected t or p at the end of the line");
  at_end(p);
  for (size_t i = 0; i < n_in_lf; i++) {
    const lf_t *x = &in_lf[i];
    if (x->chip == chip && x->cycle == c && x->link == (int)l && x->flit == (int)f) input_error("F link: two lines mark the same flit of one vector");
  }
  GROW(in_lf, n_in_lf, cap_in_lf);
  in_lf[n_in_lf++] = (lf_t){chip, c, (int)l, (int)f, unc, 0, *tp == 'p'};
}
// F mxm <cycle> <plane> <lane> <bit> [t|p]: a bit of the exact sum of a lane of an ABC flips (microarchitecture.md sec. 18).
static void read_mxm_fault(char **p, int chip) {
  char tp[16];
  uint64_t c = need(p, "cycle"), pl = need(p, "plane"), ln = need(p, "lane"), bit = need(p, "bit");
  if (pl > 3) input_error("plane out of range (0..3)");
  if (pl >= 2 * (uint64_t)zmxm_planes) input_error("plane out of range (>= 2 * mxm.planes_per_hemisphere)");
  if (ln >= (uint64_t)zmxm_rows) input_error("lane out of range (>= mxm.rows)");
  if (bit > 95) input_error("bit out of range (0..5f)");
  wordfield(p, tp, sizeof tp);
  if (*tp && strcmp(tp, "t") != 0 && strcmp(tp, "p") != 0) input_error("F: expected t or p at the end of the line");
  at_end(p);
  for (size_t i = 0; i < n_in_host; i++) {
    const hev_t *o = &in_host[i];
    if (o->chip == chip && o->kind == H_FLIP_MXM && o->cycle == c && o->fa == (int)pl && o->addr == ln && o->fbit == (int)bit)
      input_error("F: two lines flip the same bit in one cycle");
  }
  GROW(in_host, n_in_host, cap_in_host);
  hev_t *h = &in_host[n_in_host];
  memset(h, 0, sizeof *h);
  h->chip = chip; h->cycle = c; h->kind = H_FLIP_MXM; h->fa = (int)pl; h->addr = ln; h->fbit = (int)bit; h->persist = *tp == 'p';
  n_in_host++;
}
static void read_fault_line(char **p, int chip) {
  char w[16], tp[16], rw[16];
  wordfield(p, w, sizeof w);
  if (strcmp(w, "link") == 0) { read_link_fault(p, chip); return; }
  if (strcmp(w, "mxm") == 0) { read_mxm_fault(p, chip); return; }
  int sram_kind = strcmp(w, "sram") == 0, iq_kind = strcmp(w, "iq") == 0;
  if (!sram_kind && !iq_kind && strcmp(w, "srf") != 0) input_error("unknown F line (expected sram, srf, iq, link or mxm)");
  uint64_t c = need(p, "cycle");
  GROW(in_host, n_in_host, cap_in_host);
  hev_t *h = &in_host[n_in_host];
  memset(h, 0, sizeof *h);
  h->chip = chip; h->cycle = c;
  if (iq_kind) {                                           // F iq <cycle> <queue> <word> <bit> [t|p]: a bit of an instruction text word
    uint64_t qq = need(p, "queue"), wd = need(p, "word"), bit = need(p, "bit");
    if (qq > 144) input_error("queue out of range (0..90 hex)");
    if (wd > 0xFFFFFF) input_error("text word out of range (0..ffffff hex)");
    if (bit > 137) input_error("bit out of range (0..89 hex: data 0..7f, check bits 80..88, spare 89)");
    h->kind = H_FLIP_IQ; h->fa = (int)qq; h->addr = wd; h->fbit = (int)bit;
    wordfield(p, tp, sizeof tp);
    if (*tp && strcmp(tp, "t") != 0 && strcmp(tp, "p") != 0) input_error("F: expected t or p at the end of the line");
    h->persist = *tp == 'p';
    at_end(p);
    for (size_t i = 0; i < n_in_host; i++) {
      const hev_t *o = &in_host[i];
      if (o->chip == chip && o->kind == h->kind && o->cycle == c && o->fa == h->fa && o->addr == h->addr && o->fbit == h->fbit)
        input_error("F: two lines flip the same bit in one cycle");
    }
    n_in_host++;
    return;
  }
  if (sram_kind) {
    uint64_t s = need(p, "slice"), a = need(p, "address");
    if (s >= NSLICES) input_error("slice out of range (0..57 hex)");
    if (s % 44 >= (uint64_t)zmem_slices_per_hemisphere) input_error("slice does not exist (slice mod 44 >= geometry.mem_slices_per_hemisphere)");
    if (a >= (uint64_t)zwords_per_slice) input_error("address out of range (>= words_per_slice)");
    h->kind = H_FLIP_SRAM; h->fa = (int)s; h->addr = a;
  } else {
    wordfield(p, rw, sizeof rw);
    if (strcmp(rw, "e") != 0 && strcmp(rw, "w") != 0) input_error("F srf: expected the row e or w");
    uint64_t st = need(p, "stream"), g = need(p, "gap");
    if (st > 31) input_error("stream out of range (0..1f)");
    if (g > 43) input_error("gap out of range (0..2b)");
    h->kind = H_FLIP_SRF; h->fw = rw[0] == 'w'; h->fa = (int)st; h->q = (int)g;
  }
  uint64_t sl = need(p, "superlane"), bit = need(p, "bit");
  if (sl >= (uint64_t)(vl() / 16)) input_error("superlane out of range (>= geometry.superlanes)");
  if (bit > 137) input_error("bit out of range (0..89 hex: data 0..7f, check bits 80..88, spare 89)");
  h->fsl = (int)sl; h->fbit = (int)bit;
  wordfield(p, tp, sizeof tp);
  if (*tp && strcmp(tp, "t") != 0 && strcmp(tp, "p") != 0) input_error("F: expected t or p at the end of the line");
  h->persist = *tp == 'p';
  at_end(p);
  for (size_t i = 0; i < n_in_host; i++) {                // the same bit twice in one cycle is an input error
    const hev_t *o = &in_host[i];
    if (o->chip == chip && o->kind == h->kind && o->cycle == c && o->fa == h->fa && o->addr == h->addr && o->q == h->q && o->fw == h->fw && o->fsl == h->fsl && o->fbit == h->fbit)
      input_error("F: two lines flip the same bit in one cycle");
  }
  n_in_host++;
}

static void read_host_line(char **p, int chip) {
  char w[16];
  wordfield(p, w, sizeof w);
  if (!*w) input_error("unknown H line (expected start, dma, db, csr or inject)");
  uint64_t c = need(p, "cycle");
  if (strcmp(w, "start") == 0) {
    at_end(p);
    in_start[chip] = c;
    return;
  }
  GROW(in_host, n_in_host, cap_in_host);
  hev_t *h = &in_host[n_in_host];
  memset(h, 0, sizeof *h);
  h->chip = chip; h->cycle = c;
  if (strcmp(w, "dma") == 0) {
    h->kind = H_DMA;
    uint64_t x; int n = 0;
    while (hexfield(p, &x)) {
      if (x > 0xff) input_error("byte out of range");
      if (n == vl()) { char m[96]; snprintf(m, sizeof m, "an H dma line has more bytes than the %d lanes of this profile", vl()); input_error(m); }
      h->b[n++] = (unsigned char)x;
    }
  } else if (strcmp(w, "db") == 0) {
    uint64_t i = need(p, "doorbell");
    at_end(p);
    if (i >= (uint64_t)zhost_doorbells) input_error("doorbell out of range (host.doorbells)");
    h->kind = H_CSR_W; h->addr = 0x300 + i; h->val = 1;
  } else if (strcmp(w, "csr") == 0) {
    char rw[16];
    wordfield(p, rw, sizeof rw);
    uint64_t a = need(p, "csr address");
    if (a > 0xfff) input_error("csr address out of range (0..fff)");
    h->addr = a;
    if (strcmp(rw, "w") == 0) { h->kind = H_CSR_W; h->val = need(p, "value"); at_end(p); }
    else if (strcmp(rw, "r") == 0) { h->kind = H_CSR_R; at_end(p); }
    else input_error("H csr: expected r or w");
  } else if (strcmp(w, "inject") == 0) {
    uint64_t q = need(p, "queue");
    if (!(q <= 143 || q == 0xffff)) input_error("queue out of range (0..8f or ffff)");
    h->kind = H_INJECT; h->q = q == 0xffff ? 144 : (int)q;
    uint64_t x;
    while (hexfield(p, &x)) {
      if (x > 0xffffffffULL) input_error("text unit out of range (more than 32 bits)");
      GROW(h->u, h->n, h->cap);
      h->u[h->n++] = (uint32_t)x;
    }
  } else input_error("unknown H line (expected start, dma, db, csr or inject)");
  n_in_host++;
}

static void read_input(int chips) {
  char *ln = NULL; size_t lcap = 0; int chip = 0;
  uint64_t wps = (uint64_t)zwords_per_slice;
  while (getline(&ln, &lcap, stdin) != -1) {
    lineno++;
    char *p = ln;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
    char tag = *p++;
    if (*p != ' ' && *p != '\t') input_error("unknown line (expected S, Q, D, C, R, H, F or P)");
    if (tag == 'S' || tag == 'D') {
      uint64_t s = need(&p, "slice"), a = need(&p, "address");
      if (s >= NSLICES) input_error("slice out of range (0..57 hex)");
      if (s % 44 >= (uint64_t)zmem_slices_per_hemisphere) input_error("slice does not exist (slice mod 44 >= geometry.mem_slices_per_hemisphere)");
      if (a >= wps) input_error("address out of range (>= words_per_slice)");
      if (tag == 'S') {
        GROW(in_pre, n_in_pre, cap_in_pre);
        pre_t *r = &in_pre[n_in_pre++];
        r->chip = chip; r->slice = (long)s; r->addr = (long)a; r->n = 0;
        memset(r->b, 0, sizeof r->b);
        uint64_t x;
        while (hexfield(&p, &x)) {
          if (x > 0xff) input_error("byte out of range");
          if (r->n == (size_t)zvlanes) {
            char m[96];
            snprintf(m, sizeof m, "an S line has more bytes than the %" PRId64 " lanes of this profile", (int64_t)zvlanes);
            input_error(m);
          }
          r->b[r->n++] = (unsigned char)x;
        }
      } else {
        at_end(&p);
        GROW(in_dump, n_in_dump, cap_in_dump);
        in_dump[n_in_dump++] = (dump_t){chip, (long)s, (long)a};
      }
    } else if (tag == 'Q') {
      uint64_t q = need(&p, "queue");
      if (!(q <= 143 || q == 0xffff)) input_error("queue out of range (0..8f or ffff)");
      int qi = q == 0xffff ? 144 : (int)q;
      img_t *im = NULL;
      for (size_t i = 0; i < n_in_img; i++) if (in_img[i].chip == chip && in_img[i].q == qi) im = &in_img[i];
      if (!im) {
        GROW(in_img, n_in_img, cap_in_img);
        im = &in_img[n_in_img++];
        *im = (img_t){chip, qi, NULL, 0, 0};
      }
      uint64_t x;
      while (hexfield(&p, &x)) {
        if (x > 0xffffffffULL) input_error("text unit out of range (more than 32 bits)");
        if (x == 0xffffffffULL) continue;                      // padding is never enqueued
        if (im->n >= (size_t)ziq_depth / 4) input_error("initial IQ image longer than icu.iq_depth bytes");
        GROW(im->u, im->n, im->cap);
        im->u[im->n++] = (uint32_t)x;
      }
    } else if (tag == 'C') {
      uint64_t k = need(&p, "chip");
      at_end(&p);
      if (k >= (uint64_t)chips) input_error("chip out of range (--chips)");
      chip = (int)k;
    } else if (tag == 'R') {
      uint64_t a = need(&p, "csr address");
      at_end(&p);
      if (a > 0xfff) input_error("csr address out of range (0..fff)");
      GROW(in_rcsr, n_in_rcsr, cap_in_rcsr);
      in_rcsr[n_in_rcsr++] = (rcsr_t){chip, a};
    } else if (tag == 'H') {
      read_host_line(&p, chip);
    } else if (tag == 'F') {
      read_fault_line(&p, chip);
    } else if (tag == 'P') {
      read_pk_line(&p, chip);
    } else input_error("unknown line (expected S, Q, D, C, R, H, F or P)");
  }
  if (ferror(stdin)) { fprintf(stderr, "tsp_sim: cannot read standard input: %s\n", strerror(errno)); exit(2); }
  free(ln);
}

// ---------------------------------------------------------------------------------------
// This chip's program.

static FILE *out;
static int trace, observe;
enum { IRQ_CONTINUE, IRQ_STOP, IRQ_REPLAY };
static int on_irq = IRQ_CONTINUE;                 // --on-irq
static int err_irq_seen;                          // an error interrupt (MSI-X vector 0) reached the host
static int stopped;                               // --on-irq stop: the run ended at an error interrupt
static int lints;                                 // --lint: print the simulator lints (core/lints.sail), not counted in E

static const img_t *my_img[NQ];                  // this chip's initial IQ images by queue (NULL: empty)
static uint32_t *iq_ring[NQ];                    // the IQ units of each queue (iq_depth / 4 slots)
#include "iq_tags.h"
static size_t iq_tag_capacity(void) { return (size_t)ziq_depth / 4; }

static void build_program(void) {
  for (size_t i = 0; i < n_in_img; i++) if (in_img[i].chip == my_chip) my_img[in_img[i].q] = &in_img[i];
}

// The model's front end (model/slices/cu/cu_iq.sail) keeps head and length; the units are here.
unit iq_unit_write(int64_t q, sail_int slot, uint64_t v) {
  if (!iq_ring[q]) iq_ring[q] = xrealloc(NULL, ((size_t)ziq_depth / 4) * sizeof(uint32_t));
  iq_ring[q][mpz_get_ui(slot)] = (uint32_t)v;
  return UNIT;
}
uint64_t iq_unit_read(int64_t q, sail_int slot) { return iq_ring[q] ? iq_ring[q][mpz_get_ui(slot)] : 0; }
unit observe_site(const_sail_string event, sail_int c, int64_t q, sail_int head, sail_int count, sail_int detail) {
  if (!observe) return UNIT;
  uint64_t t = mpz_get_ui(c), n = mpz_get_ui(count);
  gmp_fprintf(out, "O %d %" PRIu64 " %" PRId64 " %s %Zd", my_chip, t, q, event, detail);
  if (strcmp(event, "fetch") == 0) fprintf(out, " %" PRIu64, n);
  else if (n == 0 && strcmp(event, "repeat") != 0) gmp_fprintf(out, " %Zd", head); // execute element index
  else for (uint64_t i = 0; i < n; i++) {
    size_t slot = (mpz_get_ui(head) + i) % ((size_t)ziq_depth / 4);
    mpz_srcptr tag = iq_tags[q][slot];
    uint64_t base = my_img[q] ? 4 * ((my_img[q]->n + 3) / 4) : 0;
    uint64_t size = (uint64_t)vl() / 2; // two vectors in 4-byte units
    if (mpz_sgn(tag) < 0) fprintf(out, " host");
    else if (mpz_cmp_ui(tag, base) < 0) gmp_fprintf(out, " image:%Zd", tag);
    else {
      mpz_t offset;
      mpz_init(offset);
      mpz_sub_ui(offset, tag, base);
      unsigned long unit_offset = mpz_fdiv_q_ui(offset, offset, size);
      gmp_fprintf(out, " fetch:%Zd:%lu", offset, unit_offset);
      mpz_clear(offset);
    }
  }
  fputc('\n', out);
  return UNIT;
}
int64_t iq_image_len(int64_t q) { return my_img[q] ? (int64_t)my_img[q]->n : 0; }
uint64_t iq_image_unit(int64_t q, sail_int i) { return my_img[q]->u[mpz_get_ui(i)]; }

// ---------------------------------------------------------------------------------------
// SRAM: lazily allocated words of vlanes bytes, keyed by slice * words_per_slice + addr.

static unsigned char **sram;
static uint16_t **sram_ov;                       // the ECC overlay of a word (10 bits per superlane), NULL while the word's check bits are valid
static int vl(void) { return (int)zvlanes; }
static size_t sram_key(int64_t slice, uint64_t addr) {
  if (slice < 0 || slice >= NSLICES || addr >= (uint64_t)zwords_per_slice) {
    fprintf(stderr, "tsp_sim: model accessed SRAM slice %" PRId64 " address %" PRIu64 " out of range\n", slice, addr);
    exit(4);
  }
  return (size_t)slice * (size_t)zwords_per_slice + addr;
}

static void to_bytes(unsigned char *dst, lbits v) {
  memset(dst, 0, vl());
  if (mpz_sizeinbase(*v.bits, 256) > (size_t)vl() && mpz_sgn(*v.bits) != 0) {
    fprintf(stderr, "tsp_sim: vector wider than vlanes\n"); exit(4);
  }
  size_t n = 0;
  mpz_export(dst, &n, -1, 1, 0, 0, *v.bits);          // byte l = lane l (LSB first)
}
static void from_bytes(lbits *out_, const unsigned char *src) {
  out_->len = 8 * (mp_bitcnt_t)vl();
  mpz_import(*out_->bits, vl(), -1, 1, 0, 0, src);
}

void read_sram_word(lbits *o, int64_t slice, sail_int index) {
  size_t k = sram_key(slice, mpz_get_ui(index));
  if (!sram[k]) { o->len = 8 * (mp_bitcnt_t)vl(); mpz_set_ui(*o->bits, 0); return; }
  from_bytes(o, sram[k]);
}
void read_sram_ecc(lbits *o, int64_t slice, sail_int index) {
  size_t k = sram_key(slice, mpz_get_ui(index));
  int nsl = vl() / 16;
  o->len = 10 * (mp_bitcnt_t)nsl;
  mpz_set_ui(*o->bits, 0);
  if (!sram_ov || !sram_ov[k]) return;
  for (int g = 0; g < nsl; g++)
    for (int b = 0; b < 10; b++)
      if (sram_ov[k][g] >> b & 1) mpz_setbit(*o->bits, (mp_bitcnt_t)(10 * g + b));
}
unit write_sram_ecc(int64_t slice, sail_int index, lbits ov) {
  size_t k = sram_key(slice, mpz_get_ui(index));
  int nsl = vl() / 16;
  if (!sram_ov) {
    if (mpz_sgn(*ov.bits) == 0) return UNIT;           // valid check bits, nothing ever faulted: nothing to keep
    sram_ov = calloc((size_t)NSLICES * (size_t)zwords_per_slice, sizeof *sram_ov);
    if (!sram_ov) { fprintf(stderr, "tsp_sim: out of memory\n"); exit(2); }
  }
  if (!sram_ov[k]) {
    if (mpz_sgn(*ov.bits) == 0) return UNIT;
    sram_ov[k] = xrealloc(NULL, (size_t)nsl * sizeof(uint16_t));
  }
  for (int g = 0; g < nsl; g++) {
    uint16_t x = 0;
    for (int b = 0; b < 10; b++) if (mpz_tstbit(*ov.bits, (mp_bitcnt_t)(10 * g + b))) x |= (uint16_t)(1u << b);
    sram_ov[k][g] = x;
  }
  return UNIT;
}
unit write_sram_word(int64_t slice, sail_int index, lbits value) {
  size_t k = sram_key(slice, mpz_get_ui(index));
  if (!sram[k]) sram[k] = xrealloc(NULL, vl());
  to_bytes(sram[k], value);
  return UNIT;
}

// ---------------------------------------------------------------------------------------
// Trace and diagnostics.

static mpz_t ndiag;
static bool ndiag_initialized;
static char first_code[32] = "-";
static uint64_t first_cycle;

// A trace value is a space and the vector's vlanes bytes as lower-case hex pairs, lane 0 first.
static void put_value(lbits v) {
  unsigned char b[MAXLANES];
  to_bytes(b, v);
  fputc(' ', out);
  for (int i = 0; i < vl(); i++) fprintf(out, "%02x", b[i]);
}
unit srf_write_callback(sail_int c, bool w, int64_t s, int64_t g, lbits v) {
  if (trace) {
    fprintf(out, "T srf %lu %c:%" PRId64 " gap %" PRId64, mpz_get_ui(c), w ? 'W' : 'E', s, g);
    put_value(v);
    fputc('\n', out);
  }
  return UNIT;
}
unit sram_write_callback(sail_int c, int64_t s, uint64_t a, lbits v) {
  if (trace) {
    fprintf(out, "T sram %lu slice %" PRId64 " addr %" PRIu64, mpz_get_ui(c), s, a);
    put_value(v);
    fputc('\n', out);
  }
  return UNIT;
}
// An ECC event of the model (model/core/ecc.sail): `T ecc <cycle> queue <q> corr|unc <row>:<stream>|sram <slice>:<addr> sl <g> pos <s>`, or for a word of instruction text
// `T ecc <cycle> queue <q> corr|unc text word <w> pos <s>`.
// Printed with --trace, in the order the model produces it; not part of trace_sha256 (it is an extended event).
unit ecc_event_callback(sail_int c, int64_t q, int64_t kind, bool unc, bool w, int64_t id, int64_t addr, int64_t sl, int64_t pos) {
  if (trace) {
    fprintf(out, "T ecc %lu queue %" PRId64 " %s ", mpz_get_ui(c), q == 144 ? (int64_t)65535 : q, unc ? "unc" : "corr");
    if (kind == 2) { fprintf(out, "text word %" PRId64 " pos %" PRId64 "\n", addr, pos); return UNIT; }
    if (kind == 0) fprintf(out, "%c:%" PRId64, w ? 'W' : 'E', id);
    else fprintf(out, "sram %" PRId64 ":%" PRId64, id, addr);
    fprintf(out, " sl %" PRId64 " pos %" PRId64 "\n", sl, pos);
  }
  return UNIT;
}
// The patrol scrubber corrected or found uncorrectable a word: `T ecc <cycle> scrub slice <s> addr <a> sl <g>` with ` unc` appended when it is left as it is.
unit scrub_event_callback(sail_int c, int64_t s, sail_int a, int64_t sl, bool unc) {
  if (trace) fprintf(out, "T ecc %lu scrub slice %" PRId64 " addr %lu sl %" PRId64 "%s\n", mpz_get_ui(c), s, mpz_get_ui(a), sl, unc ? " unc" : "");
  return UNIT;
}
// A stream operand with its spare bit set (a MEM error report, seen by its consumer): `T ecc <cycle> queue <q> spare E:<s> sl <x> pos <S[0..3]>`.
unit spare_event_callback(sail_int c, int64_t q, bool w, int64_t id, int64_t gap, int64_t sl, int64_t s03) {
  (void)gap;
  if (trace) fprintf(out, "T ecc %lu queue %" PRId64 " spare %c:%" PRId64 " sl %" PRId64 " pos %" PRId64 "\n", mpz_get_ui(c), q == 144 ? (int64_t)65535 : q, w ? 'W' : 'E', id, sl, s03);
  return UNIT;
}
// The sum check of an ABC corrected one lane of its product: `T ecc <cycle> queue <q> mxm corr plane <p> lane <r>`.
unit mxm_corr_callback(sail_int c, int64_t q, int64_t plane, sail_int lane) {
  if (trace) fprintf(out, "T ecc %lu queue %" PRId64 " mxm corr plane %" PRId64 " lane %lu\n", mpz_get_ui(c), q, plane, mpz_get_ui(lane));
  return UNIT;
}
static void diag_note(uint64_t c, int64_t q, const char *code) {
  if (!ndiag_initialized) { mpz_init(ndiag); ndiag_initialized = true; }
  if (mpz_sgn(ndiag) == 0) { snprintf(first_code, sizeof first_code, "%s", code); first_cycle = c; }
  mpz_add_ui(ndiag, ndiag, 1);
  if (trace) fprintf(out, "T diag %" PRIu64 " queue %" PRId64 " %s\n", c, q, code);
}
unit diag_callback(sail_int c, int64_t q, const_sail_string code) {
  diag_note(mpz_get_ui(c), q, code);
  return UNIT;
}

// A simulator lint (core/lints.sail): not a diagnostic, so it is
// not counted in the E line and is not in the golden vectors. Printed only with --lint, as it happens.
unit lint_callback(sail_int c, int64_t q, const_sail_string code) {
  if (lints) fprintf(out, "T lint %lu queue %" PRId64 " %s\n", mpz_get_ui(c), q, code);
  return UNIT;
}

// ---------------------------------------------------------------------------------------
// C2C links (microarchitecture.md sec. 10). Every chip has c2c.links links (at most 128).
// Topology, after the node of ISCA 2022 sec. 2.2 ("7 local links ... provide full
// connectivity between the group of 8 TSPs"; "4 global links"):
//   One chip: every link loops back to itself (link l TX -> link l RX of the same chip).
//   K = 2..8 chips: one node. Local link l (0..6) of chip i connects to chip
//     (i + l + 1) mod K, arriving on its local link K - 2 - l; links l >= K - 1 are not
//     connected. Global links 7..10 and links 11.. are not connected.
//   K > 8 chips: N = ceil(K / 8) nodes of 8 chips (chip i is in node i / 8; the last node holds
//     the rest), local links as above inside each node. Global port p = 4 (i mod 8) + g of
//     node n (global link 7 + g) connects to node (n + p + 1) mod N, port N - 2 - p (ports
//     p >= N - 1, and ports whose far chip does not exist, are not connected): every pair of
//     nodes has one global link (the all-to-all group level of the Dragonfly, ISCA 2022 sec.
//     2.1, 2.2). K is at most 264 = 33 nodes: more nodes than 33 would leave pairs of nodes
//     without a global link.
//   --topology FILE: the adjacency of every (chip, link), see topology_load.
// Latency: every directed link has its own latency in [c2c.link_latency_min,
// c2c.link_latency_max] (ISCA 2022 Table 2). Default: c2c.link_latency on every link;
// --link-latency N: N on every link; --link-latency-seed S: each directed link's latency
// drawn uniformly from the range by a fixed generator seeded with S (the same S gives the
// same latencies). Each link's inbound FIFO is in arrival order; a Receive takes its head.
// --link-jitter-seed S: the latency varies per transfer (ISCA 2022 Table 2 gives a per-link
// spread). Each vector sent on chip c's link l draws its own
// latency uniformly from [c2c.link_latency_min, c2c.link_latency_max] with a fixed
// generator per (S, c, l), and arrives at max(sent + drawn, previous arrival on that link
// + 1): the link still delivers in order, one packet per cycle. The clamp is one per link over
// every packet type (data, sync, csr, ctrl: HC34 p.45, a link is one serial channel of flits, so
// no packet passes another on it); only the draws come from two generators (see jit_state_c).
// The link's bandwidth is the model's (OTS-E-LNK-001); the receive FIFO is here.
// A link's FIFO holds the vectors that have arrived and not been taken. A vector that
// arrives (cycle a) when the FIFO holds c2c.rx_depth vectors is dropped and the chip reports
// OTS-E-LNK-002 at cycle a, queue 65535, before tick(a) (a Receive that executes in cycle a
// takes after the arrival; one that executes before a has freed its slot). The decision
// is taken in arrival order, when the harness reaches cycle a or a Receive looks at the link.
// Arrivals that can overrun are events of the run (link_next_event).

#define MAXLINKS 128           // c2c.links is at most 128 (the link number is 7 bits)
enum { V_PENDING, V_QUEUED, V_TAKEN, V_DROPPED };   // not yet arrived/decided, in the FIFO, taken by a Receive, dropped (LNK-002)
// Every transfer on a link has a type (model/core/callbacks.sail c2c_ptype; HC34 p.45: "Type (data, CSR, Sync, Ctrl)"). Data and
// sync packets carry a vector and enter the receive FIFO (the model's c2c_ptype_queued); csr and ctrl packets are
// consumed by the receiving chip's control unit (see "The control unit" below) and carry a ctlpkt_t instead.
enum { PT_DATA, PT_CSR, PT_SYNC, PT_CTRL };
enum { OP_RD, OP_WR, OP_REPLY, OP_CTRL };         // csr: read request, write request, reply to a read; ctrl
typedef struct { int op, origin, target; uint64_t addr, val, id; } ctlpkt_t;
// poison: bit g = superlane g was uncorrectable when the vector was sent (inherited by the Receive)
// fec_c, fec_u: the flits of the transfer (bit f) that the link FEC corrected in flight, and that it detected and could not correct.
typedef struct { cycle_t arrival; int chip, link; int state; cycle_t taken_at; int type; ctlpkt_t ctl; uint32_t poison; flitmask_t fec_c, fec_u; unsigned char b[MAXLANES]; } vecmsg_t;
typedef struct { vecmsg_t *v; size_t head, n, cap; } fifo_t;
static fifo_t inbound[MAXLINKS];
static vecmsg_t *outbound; static size_t n_out, cap_out;
extern sail_int zc2c_link_latency_min, zc2c_link_latency_max;
extern int64_t zc2c_links;
static long lat_fixed = -1;                       // --link-latency
static long long lat_seed = -1;                   // --link-latency-seed
static uint64_t link_lat[MAXCHIPS][MAXLINKS];           // latency of chip c's outbound link l
static long long jit_seed = -1;                   // --link-jitter-seed
static uint64_t jit_state[MAXCHIPS][MAXLINKS];          // per-link generator state
static cycle_t jit_last[MAXCHIPS][MAXLINKS];           // previous arrival + 1 over every packet type of the link (0: none yet)
static uint64_t jit_state_c[MAXCHIPS][MAXLINKS];        // the generator of the packets of the control unit: its own, so that the draws of the
                                                  // vectors do not depend on when a control packet leaves (the in-order clamp is shared)

// The topology file (--topology): the adjacency of every (chip, link). NULL: the wiring of --chips K (dragonfly_peer).
static int *topo_peer;                            // [chip * MAXLINKS + link]: peer chip * MAXLINKS + peer link, or -1
static uint64_t *topo_lat;                        // [chip * MAXLINKS + link]: the latency of the directed link, 0 when the file gives none
static uint64_t topo_lat_default;                 // the file's "latency" (0: none)
static long long topo_jit_seed = -1;              // the file's "jitter_seed"

// The wiring of --chips K (the Dragonfly of ISCA22 sec. 2.1, 2.2): the far end of chip c's link l of a K chip system, or -1.
static int dragonfly_peer(int K, int c, int l, int *pl) {
  if (K == 1) { *pl = l; return c; }
  int nn = K <= 8 ? 1 : (K + 7) / 8;                  // nodes of 8 chips, the last may hold fewer
  int node = K <= 8 ? 0 : c / 8, k = K <= 8 ? c : c % 8;
  int nk = K <= 8 ? K : (K - 8 * node < 8 ? K - 8 * node : 8);   // chips in this node
  if (l <= 6) {
    if (l >= nk - 1) return -1;
    *pl = nk - 2 - l;
    return node * 8 + (k + l + 1) % nk;
  }
  if (l <= 10 && nn > 1) {
    int p = 4 * k + (l - 7);
    if (p >= nn - 1) return -1;
    int m = (node + p + 1) % nn, q = nn - 2 - p;
    int peer = m * 8 + q / 4;
    if (peer >= K) return -1;                           // the far chip of a node that is not full does not exist
    *pl = 7 + q % 4;
    return peer;
  }
  return -1;
}

// The far end of chip c's link l: chip (or -1) and its link there.
static int link_peer(int c, int l, int *pl) {
  if (l >= (int)zc2c_links) return -1;
  if (topo_peer) {
    int v = topo_peer[c * MAXLINKS + l];
    if (v < 0) return -1;
    *pl = v % MAXLINKS;
    return v / MAXLINKS;
  }
  return dragonfly_peer(n_chips, c, l, pl);
}


// --topology FILE: {"chips": K, "links": [{"a": [chip, link], "b": [chip, link], "latency": N}, ...], "latency": N, "jitter_seed": S,
// "cycles_per_vector": N}. Each entry joins two link ends (a and b, the same end twice loops a link back); every end is named once.
// "latency" of an entry is the latency of both directions of that link, "latency" at the top is the default of the others (both within
// [c2c.link_latency_min, c2c.link_latency_max]); "cycles_per_vector", when present, must be the profile's c2c.cycles_per_vector (the
// link rate is a profile key, not wiring).
extern sail_int zc2c_cycles_per_vector;
static void topo_fail(const char *path, const char *fmt, ...) {
  va_list ap;
  fprintf(stderr, "tsp_sim: topology %s: ", path);
  va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
  fputc('\n', stderr);
  exit(2);
}
static int topo_int(const char *path, const cJSON *j, const char *what, long long lo, long long hi) {
  if (!cJSON_IsNumber(j) || j->valuedouble != (double)(long long)j->valuedouble) topo_fail(path, "%s must be an integer", what);
  if (j->valuedouble < (double)lo || j->valuedouble > (double)hi) topo_fail(path, "%s: %.0f is outside %lld..%lld", what, j->valuedouble, lo, hi);
  return (int)j->valuedouble;
}
static void topo_end(const char *path, const cJSON *e, const char *what, int K, int *chip, int *link) {
  if (!cJSON_IsArray(e) || cJSON_GetArraySize(e) != 2) topo_fail(path, "%s must be [chip, link]", what);
  *chip = topo_int(path, cJSON_GetArrayItem(e, 0), "chip", 0, K - 1);
  *link = topo_int(path, cJSON_GetArrayItem(e, 1), "link", 0, (int)zc2c_links - 1);
}
static int topology_load(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "tsp_sim: --topology %s: %s\n", path, strerror(errno)); exit(2); }
  size_t cap = 4096, n = 0; char *buf = xrealloc(NULL, cap);
  size_t got;
  while ((got = fread(buf + n, 1, cap - n - 1, f)) > 0) { n += got; if (n + 1 >= cap) { cap *= 2; buf = xrealloc(buf, cap); } }
  fclose(f);
  buf[n] = 0;
  if (memchr(buf, 0, n) || tsp_json_syntax_error(buf)) topo_fail(path, "not a JSON object");
  const char *parse_end = NULL;
  cJSON *root = cJSON_ParseWithOpts(buf, &parse_end, 1);      // text after the object is refused, as tsp-iss does
  if (!root || !cJSON_IsObject(root)) topo_fail(path, "not a JSON object");
  for (cJSON *k = root->child; k; k = k->next)
    for (cJSON *j = root->child; j != k; j = j->next)
      if (!strcmp(j->string, k->string)) topo_fail(path, "duplicate key \"%s\"", k->string);
  uint64_t lo = mpz_get_ui(zc2c_link_latency_min), hi = mpz_get_ui(zc2c_link_latency_max);
  for (cJSON *k = root->child; k; k = k->next)
    if (strcmp(k->string, "chips") && strcmp(k->string, "links") && strcmp(k->string, "latency") && strcmp(k->string, "jitter_seed") &&
        strcmp(k->string, "cycles_per_vector"))
      topo_fail(path, "unknown key \"%s\"", k->string);
  cJSON *jc = cJSON_GetObjectItem(root, "chips"), *jl = cJSON_GetObjectItem(root, "links");
  if (!jc) topo_fail(path, "\"chips\" is missing");
  int K = topo_int(path, jc, "chips", 1, MAXCHIPS);
  if (!cJSON_IsArray(jl)) topo_fail(path, "\"links\" must be an array");
  cJSON *jd = cJSON_GetObjectItem(root, "latency");
  if (jd) topo_lat_default = (uint64_t)topo_int(path, jd, "latency", (long long)lo, (long long)hi);
  cJSON *js = cJSON_GetObjectItem(root, "jitter_seed");
  if (js) topo_jit_seed = topo_int(path, js, "jitter_seed", 0, 2147483647);
  cJSON *jv = cJSON_GetObjectItem(root, "cycles_per_vector");
  if (jv && topo_int(path, jv, "cycles_per_vector", 1, 1 << 20) != (int)mpz_get_ui(zc2c_cycles_per_vector))
    topo_fail(path, "cycles_per_vector %d is not the profile's c2c.cycles_per_vector %" PRIu64, (int)jv->valuedouble, (uint64_t)mpz_get_ui(zc2c_cycles_per_vector));
  topo_peer = xrealloc(NULL, (size_t)MAXCHIPS * MAXLINKS * sizeof *topo_peer);
  topo_lat = xrealloc(NULL, (size_t)MAXCHIPS * MAXLINKS * sizeof *topo_lat);
  for (size_t i = 0; i < (size_t)MAXCHIPS * MAXLINKS; i++) { topo_peer[i] = -1; topo_lat[i] = 0; }
  int idx = 0;
  for (cJSON *e = jl->child; e; e = e->next, idx++) {
    if (!cJSON_IsObject(e)) topo_fail(path, "links[%d] must be an object", idx);
    for (cJSON *k = e->child; k; k = k->next)
      if (strcmp(k->string, "a") && strcmp(k->string, "b") && strcmp(k->string, "latency")) topo_fail(path, "links[%d]: unknown key \"%s\"", idx, k->string);
    for (cJSON *k = e->child; k; k = k->next)
      for (cJSON *j = e->child; j != k; j = j->next)
        if (!strcmp(j->string, k->string)) topo_fail(path, "duplicate key \"%s\"", k->string);
    int ac, al, bc, bl;
    char what[64];
    snprintf(what, sizeof what, "links[%d].a", idx); topo_end(path, cJSON_GetObjectItem(e, "a"), what, K, &ac, &al);
    snprintf(what, sizeof what, "links[%d].b", idx); topo_end(path, cJSON_GetObjectItem(e, "b"), what, K, &bc, &bl);
    int a = ac * MAXLINKS + al, b = bc * MAXLINKS + bl;
    if (topo_peer[a] >= 0) topo_fail(path, "links[%d]: chip %d link %d is joined twice", idx, ac, al);
    if (topo_peer[b] >= 0) topo_fail(path, "links[%d]: chip %d link %d is joined twice", idx, bc, bl);
    uint64_t lat = 0;
    cJSON *jlat = cJSON_GetObjectItem(e, "latency");
    if (jlat) { snprintf(what, sizeof what, "links[%d].latency", idx); lat = (uint64_t)topo_int(path, jlat, what, (long long)lo, (long long)hi); }
    topo_peer[a] = b; topo_peer[b] = a;
    topo_lat[a] = lat; topo_lat[b] = lat;
  }
  cJSON_Delete(root);
  free(buf);
  return K;
}

// splitmix64: a fixed, portable generator for --link-latency-seed.
static uint64_t splitmix(uint64_t *x) {
  uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static cycle_t min_link_latency = NONE;          // the lock-step window (run_chips)
static uint64_t win_cap;                          // --window N: at most N cycles
static void links_setup(void) {
  uint64_t lo = mpz_get_ui(zc2c_link_latency_min), hi = mpz_get_ui(zc2c_link_latency_max);
  uint64_t nom = mpz_get_ui(zc2c_link_latency);
  if (lo > nom || nom > hi) { fprintf(stderr, "tsp_sim: need c2c.link_latency_min <= c2c.link_latency <= c2c.link_latency_max\n"); exit(2); }
  if (lat_fixed >= 0 && ((uint64_t)lat_fixed < lo || (uint64_t)lat_fixed > hi)) {
    fprintf(stderr, "tsp_sim: --link-latency must be in [%" PRIu64 ", %" PRIu64 "]\n", lo, hi); exit(2);
  }
  uint64_t x = (uint64_t)lat_seed;
  // The seeded latencies are drawn chip by chip over links 0..15, then over the links above 15, so the draws of a profile with at most
  // 16 links are those of the earlier harness; the state of each link's jitter generator likewise keeps its earlier value for links 0..15.
  for (int pass = 0; pass < 2; pass++)
    for (int c = 0; c < n_chips; c++)
      for (int l = pass ? 16 : 0; l < (pass ? MAXLINKS : 16); l++) {
        uint64_t tl = topo_lat ? topo_lat[c * MAXLINKS + l] : 0;
        if (tl == 0) tl = topo_lat_default;
        uint64_t L = lat_fixed >= 0 ? (uint64_t)lat_fixed : lat_seed >= 0 ? lo + splitmix(&x) % (hi - lo + 1) : jit_seed >= 0 ? lo : tl ? tl : nom;
        if (jit_seed >= 0) {                        // the window is the minimum
          uint64_t idx = l < 16 ? (uint64_t)(c * 16 + l + 1) : (uint64_t)(MAXCHIPS * 16 + c * (MAXLINKS - 16) + (l - 16) + 1);
          jit_state[c][l] = (uint64_t)jit_seed ^ (idx * 0xD1B54A32D192ED03ULL);
          jit_state_c[c][l] = ~jit_state[c][l];
          jit_last[c][l] = 0;
        }
        link_lat[c][l] = L;
        int pl;
        if (link_peer(c, l, &pl) >= 0 && L < min_link_latency) min_link_latency = L;
      }
}
static void fifo_push(fifo_t *f, const vecmsg_t *m) {
  // Keep arrival order: a later-sent vector on a link never overtakes (equal latency per
  // link), so this is an append; the insertion keeps it correct for any order of delivery.
  GROW(f->v, f->n, f->cap);
  size_t i = f->n++;
  while (i > f->head && f->v[i - 1].arrival > m->arrival) { f->v[i] = f->v[i - 1]; i--; }
  f->v[i] = *m;
}

// The arrival cycle of a packet that leaves chip my_chip's link at cycle sent (control: a packet of the control unit).
static cycle_t link_arrival(int link, cycle_t sent, int control) {
  if (jit_seed < 0) return sent + link_lat[my_chip][link];
  uint64_t lo = mpz_get_ui(zc2c_link_latency_min), hi = mpz_get_ui(zc2c_link_latency_max);
  uint64_t *st = control ? &jit_state_c[my_chip][link] : &jit_state[my_chip][link];
  cycle_t *last = &jit_last[my_chip][link];              // one clamp per link: a packet never arrives before an earlier one of any type
  cycle_t a = sent + lo + splitmix(st) % (hi - lo + 1);   // per-transfer latency, in order
  if (a < *last) a = *last;
  *last = a + 1;
  return a;
}

// The cycles at which each of this chip's links accepted a vector of the program (data or sync), ascending: the
// schedule that a control packet has to fit in (see "The control unit").
static uint64_t *accs[MAXLINKS]; static size_t n_acc[MAXLINKS], cap_acc[MAXLINKS];

int64_t c2c_peer(int64_t link) { int pl; return link_peer(my_chip, (int)link, &pl); }
static void fifo_or_out(const vecmsg_t *m);
unit c2c_send(sail_int sent, int64_t link, int64_t ptype, lbits v, int64_t poison) {
  vecmsg_t m;
  memset(&m, 0, sizeof m);
  int pl = 0;
  m.chip = link_peer(my_chip, (int)link, &pl);
  if (m.chip < 0) return UNIT;                     // the model checks chip_id first
  uint64_t t = mpz_get_ui(sent);
  m.arrival = link_arrival((int)link, t, 0); m.link = pl; m.state = V_PENDING; m.type = (int)ptype;
  to_bytes(m.b, v);
  m.poison = (uint32_t)poison;
  for (size_t i = 0; i < n_in_lf; i++) {                      // the F link lines of this vector
    lf_t *x = &in_lf[i];
    if (x->chip != my_chip || x->cycle != t || x->link != (int)link) continue;
    x->used = 1;
    if (x->unc) m.fec_u |= (flitmask_t)1 << x->flit; else m.fec_c |= (flitmask_t)1 << x->flit;
  }
  GROW(accs[link], n_acc[link], cap_acc[link]);
  accs[link][n_acc[link]++] = t;
  fifo_or_out(&m);
  return UNIT;
}
// The FIFO rule. Occupancy seen by a vector arriving at a: the vectors before it that
// are in the FIFO, or were taken by a Receive that executes at a or later.
static uint64_t rx_depth_v(void) { return mpz_get_ui(zc2c_rx_depth); }
typedef struct { cycle_t arrival; int link; int kind; flitmask_t fec_c, fec_u; } report_t;   // kind 0: link FEC events of a vector, 1: LNK-002 (an overrun)
static report_t *reports; static size_t n_rep, cap_rep;

static void link_resolve(int l, uint64_t upto) {
  fifo_t *f = &inbound[l];
  for (size_t i = f->head; i < f->n; i++) {
    vecmsg_t *m = &f->v[i];
    if (m->arrival > upto) break;
    if (m->state != V_PENDING) continue;
    uint64_t occ = 0;
    for (size_t j = f->head; j < i; j++) {
      vecmsg_t *o = &f->v[j];
      if (o->state == V_QUEUED || (o->state == V_TAKEN && o->taken_at >= m->arrival)) occ++;
    }
    if (occ >= rx_depth_v()) {
      m->state = V_DROPPED;
      GROW(reports, n_rep, cap_rep);
      reports[n_rep++] = (report_t){m->arrival, l, 1, 0, 0};
    } else m->state = V_QUEUED;
  }
}
// Drop the prefix that can no longer matter: dropped vectors, and vectors taken more than
// the largest execution skew (63) before the latest cycle looked at.
static void fifo_trim(fifo_t *f, uint64_t latest) {
  while (f->head < f->n) {
    vecmsg_t *m = &f->v[f->head];
    if (m->state == V_DROPPED || (m->state == V_TAKEN && m->taken_at + 64 < latest)) f->head++;
    else break;
  }
}
// Decide every arrival up to cycle upto and report the overruns due, ascending (cycle, link).
static void process_arrivals(uint64_t upto) {
  for (int l = 0; l < MAXLINKS; l++) { link_resolve(l, upto); fifo_trim(&inbound[l], upto); }
  for (;;) {
    size_t b = n_rep;
    for (size_t i = 0; i < n_rep; i++)
      if (reports[i].arrival <= upto && (b == n_rep || reports[i].arrival < reports[b].arrival ||
          (reports[i].arrival == reports[b].arrival && (reports[i].link < reports[b].link ||
          (reports[i].link == reports[b].link && reports[i].kind < reports[b].kind))))) b = i;
    if (b == n_rep) break;
    if (reports[b].kind == 0) {                                // link FEC: the FEC acts before the receive FIFO sees the vector
      flitmask_t fc = reports[b].fec_c, fu = reports[b].fec_u;
      for (int f = 0; f < 128; f++) {
        if (!(((fc | fu) >> f) & 1)) continue;
        int unc = (int)((fu >> f) & 1);
        if (trace) fprintf(out, "T ecc %" PRIu64 " link %d %s flit %d\n", (uint64_t)reports[b].arrival, reports[b].link, unc ? "unc" : "corr", f);
        sail_int ca; mpz_init(ca); set_cycle(ca, reports[b].arrival);
        zmodel_link_fec(ca, reports[b].link, f, unc);   // the counter and the error registers (microarchitecture.md sec. 16)
        mpz_clear(ca);
      }
    } else
    diag_note(reports[b].arrival, 65535, "OTS-E-LNK-002");
    reports[b] = reports[--n_rep];
  }
}
// The earliest cycle at which the harness must act on a link: a report waiting, or an arrival that
// would find the FIFO full if nothing more were taken (NONE: none). Cheaper than making every
// arrival an event, and a run that only sends a few vectors does not need the cycles after it.
static cycle_t link_next_event(void) {
  cycle_t best = NONE;
  for (size_t i = 0; i < n_rep; i++) if (reports[i].arrival < best) best = reports[i].arrival;
  for (int l = 0; l < MAXLINKS; l++) {
    fifo_t *f = &inbound[l];
    uint64_t active = 0;
    size_t taken[64], nt = 0;
    for (size_t i = f->head; i < f->n; i++) {
      vecmsg_t *m = &f->v[i];
      if (m->state == V_PENDING) {
        uint64_t occ = active;
        for (size_t k = 0; k < nt; k++) if (f->v[taken[k]].taken_at >= m->arrival) occ++;
        if (occ >= rx_depth_v()) { if (m->arrival < best) best = m->arrival; break; }
        active++;
      } else if (m->state == V_QUEUED) active++;
      else if (m->state == V_TAKEN && nt < 64) taken[nt++] = i;
    }
  }
  return best;
}
static vecmsg_t *fifo_front(fifo_t *f) {
  for (size_t i = f->head; i < f->n; i++) {
    if (f->v[i].state == V_QUEUED) return &f->v[i];
    if (f->v[i].state == V_PENDING) return NULL;
  }
  return NULL;
}
bool c2c_recv_ready(sail_int c, int64_t link) {
  fifo_t *f = &inbound[link];
  link_resolve((int)link, mpz_get_ui(c));
  vecmsg_t *m = fifo_front(f);
  return m && m->arrival <= mpz_get_ui(c);
}
static int64_t last_taken_bad;                    // the superlanes of the vector the last c2c_recv took that the FEC found uncorrectable
// The arrival cycle of the oldest vector of the link's FIFO that is neither taken nor dropped (-1: none known): the model
// (model_poll_event) wakes the chip at the epoch boundary at which a DeskewRecv looks for it.
void c2c_next_arrival(sail_int *o, int64_t link) {
  fifo_t *f = &inbound[link];
  for (size_t i = f->head; i < f->n; i++)
    if (f->v[i].state == V_PENDING || f->v[i].state == V_QUEUED) { set_cycle(*o, f->v[i].arrival); return; }
  mpz_set_si(*o, -1);
}
static uint32_t recv_poison;                     // the poison of the vector the last c2c_recv took
void c2c_recv(lbits *o, sail_int c, int64_t link) {
  fifo_t *f = &inbound[link];
  static const unsigned char zero[MAXLANES];
  vecmsg_t *m = fifo_front(f);
  if (m) { m->state = V_TAKEN; m->taken_at = mpz_get_ui(c); }
  recv_poison = m ? m->poison : 0;
  last_taken_bad = 0;
  if (m && m->fec_u)
    for (int fl = 0; fl < 128; fl++) if ((m->fec_u >> fl) & 1) last_taken_bad |= zmodel_link_flit_mask(fl);
  from_bytes(o, m ? m->b : zero);
}
int64_t c2c_recv_poison(sail_int c, int64_t link) { (void)c; (void)link; return (int64_t)recv_poison; }
int64_t c2c_recv_bad(int64_t link) { (void)link; return last_taken_bad; }



// ---------------------------------------------------------------------------------------
// The control unit (microarchitecture.md sec. 10, "Packet types"). Every transfer on a link has a type (data, csr, sync,
// ctrl). Send and Transmit give the model's data and sync packets (above); the control unit sends the two others:
//   csr:  a remote CSR access (`P csr <cycle> <link> r|w <addr> [<value>]`): one hop, no routing. The packet goes out on the named
//         link; the control unit of the chip at its far end reads or writes the CSR in the arrival cycle (before that cycle's
//         tick) and answers a read with a reply csr packet on the link the request came in; the chip that asked prints
//         `T csr <cycle> <chip> <addr> <value>` when the reply arrives. (Every chip has its own host interface, so the host of a
//         chip reaches the registers of that chip with `H csr`; the csr packet is the control unit's access to the chip next to it.)
//   ctrl: `P ctrl <cycle> <link>`: a control packet of the chip's control unit, consumed by the control unit of the peer
//         (`T pkt`), with no other effect (its content is not published; the sender of the clock alignment messages
//         would use the same path).
// A csr or ctrl packet never enters a receive FIFO (the model's c2c_ptype_queued), so no Receive sees it and OTS-E-LNK-002
// counts data and sync packets only. It occupies a link for model_packet_cycles cycles. The packets of a link leave in the order
// in which they are made (ready cycle, packet number), each at the first cycle c >= its ready cycle at which the link is free: no
// packet before it still occupies the link (the typed packets share it) and no vector the link accepted before c is still on it
// (a + cycles_per_vector <= c). The control unit knows the cycles before c and nothing of the cycles after it: the rule is causal.
// A Send or Transmit that the program releases into the cycles of a packet in flight (c <= s < c + k) is OTS-E-LNK-003
// (model: c2c_packet_busy, below).
//
// The departure cycle depends on the packet's ready cycle and the chip's own schedule (its accepted vectors, which depend
// on its program, the host script and the control packets that reached it) only, so it does not depend on how the chips
// are scheduled. The harness decides a departure when it is certain: ctl_depart(frontier, H) knows every accepted vector
// before H (the chip's next tick or event) and every arrival before `frontier` (the end of the lock-step window: a packet sent
// at or after the window's start arrives at or after its end). A packet whose departure is not yet known is a wake-up of the
// chip at its earliest possible departure, which keeps the windows from passing it.

typedef struct { ctlpkt_t p; int type, link; cycle_t ready; uint64_t seq; } cpend_t;
typedef struct { ctlpkt_t p; int type, link; cycle_t arrival; uint64_t seq; } cin_t;
static cpend_t *cpend; static size_t n_cpend, cap_cpend;     // packets waiting to leave this chip
static cin_t *cin; static size_t n_cin, cap_cin, cin_head;    // packets that arrived and are not yet consumed
static sail_int sc;
static cycle_t ctl_start[MAXLINKS], ctl_end[MAXLINKS];       // the last control packet that left a link occupies [start, end) (end 0: none)
static pk_t **my_pk; static size_t n_my_pk, pk_pos;           // this chip's P lines, by cycle

extern bool zmodel_ptype_queued(int64_t code);
extern void zmodel_packet_cycles(sail_int *out, int64_t code);
extern sail_int zc2c_cycles_per_vector;
static uint64_t pkt_cycles(int type) {
  static uint64_t cache[4];
  if (!cache[type]) {
    sail_int o; CREATE(sail_int)(&o);
    zmodel_packet_cycles(&o, type);
    cache[type] = mpz_get_ui(o);
    KILL(sail_int)(&o);
  }
  return cache[type];
}

// True when a csr or ctrl packet that left link l at or before cycle c still occupies it in c.
bool c2c_packet_busy(sail_int c, int64_t l) {
  uint64_t t = mpz_get_ui(c);
  return ctl_end[l] != 0 && ctl_start[l] <= t && t < ctl_end[l];
}

static uint64_t seq_of(const ctlpkt_t *p) { return p->id * 2 + (p->op == OP_REPLY); }

static void cpend_add(const ctlpkt_t *p, int type, int link, uint64_t ready) {
  GROW(cpend, n_cpend, cap_cpend);
  cpend[n_cpend++] = (cpend_t){*p, type, link, ready, seq_of(p)};
}

static void cin_add(const vecmsg_t *m) {
  GROW(cin, n_cin, cap_cin);
  cin_t x = {m->ctl, m->type, m->link, m->arrival, seq_of(&m->ctl)};
  size_t i = n_cin++;
  while (i > cin_head && (cin[i - 1].arrival > x.arrival || (cin[i - 1].arrival == x.arrival && cin[i - 1].seq > x.seq))) { cin[i] = cin[i - 1]; i--; }
  cin[i] = x;
}

static uint64_t link_flit_count(void) {
  sail_int o; CREATE(sail_int)(&o);
  zmodel_link_flits(&o, UNIT);
  uint64_t n = mpz_get_ui(o);
  KILL(sail_int)(&o);
  return n;
}

// A packet that reaches this chip's end of a link (a data or sync packet enters the receive FIFO).
static void deliver(const vecmsg_t *m) {
  if (m->fec_c | m->fec_u) {                                    // link FEC events: reported at the arrival cycle (process_arrivals)
    GROW(reports, n_rep, cap_rep);
    reports[n_rep++] = (report_t){m->arrival, m->link, 0, m->fec_c, m->fec_u};
  }
  if (zmodel_ptype_queued(m->type)) fifo_push(&inbound[m->link], m); else cin_add(m);
}
static void fifo_or_out(const vecmsg_t *m) {
  if (m->chip == my_chip) deliver(m);
  else { GROW(outbound, n_out, cap_out); outbound[n_out++] = *m; }
}

// The first cycle >= ready at which link l is free: after the control packet before it (they share the link) and after the
// vector the link accepted last before that cycle (accs[l] holds every accepted vector of the program that matters).
static cycle_t ctl_candidate(int l, cycle_t ready) {
  uint64_t cpv = mpz_get_ui(zc2c_cycles_per_vector);
  cycle_t c = ready > ctl_end[l] ? ready : ctl_end[l];
  for (;;) {
    size_t lo = 0, hi = n_acc[l];                               // the vectors accepted before c: accs[l][0 .. lo)
    while (lo < hi) { size_t mid = (lo + hi) / 2; if (accs[l][mid] < c) lo = mid + 1; else hi = mid; }
    if (lo > 0 && (cycle_t)accs[l][lo - 1] + cpv > c) c = (cycle_t)accs[l][lo - 1] + cpv; else return c;
  }
}

// The pending packet of link l that leaves first (NULL: none).
static cpend_t *ctl_head(int l) {
  cpend_t *h = NULL;
  for (size_t i = 0; i < n_cpend; i++) {
    cpend_t *x = &cpend[i];
    if (x->link != l) continue;
    if (!h || x->ready < h->ready || (x->ready == h->ready && x->seq < h->seq)) h = x;
  }
  return h;
}

static cycle_t real_next(void);
static cycle_t ctl_wake(void) {
  cycle_t best = NONE;
  for (int l = 0; l < MAXLINKS; l++) {
    cpend_t *h = ctl_head(l);
    if (!h) continue;
    cycle_t c = ctl_candidate(l, h->ready);
    if (c < best) best = c;
  }
  return best;
}

static void ctl_send(cpend_t *h, cycle_t c) {
  if (c > max_cycles) over_budget(c);
  int l = h->link, pl = 0;
  vecmsg_t m;
  memset(&m, 0, sizeof m);
  m.chip = link_peer(my_chip, l, &pl);
  m.link = pl; m.type = h->type; m.ctl = h->p; m.state = V_PENDING;
  m.arrival = link_arrival(l, c, 1);
  ctl_start[l] = c; ctl_end[l] = c + pkt_cycles(h->type);
  *h = cpend[--n_cpend];                                       // order does not matter: ctl_head picks by key
  fifo_or_out(&m);
}

// Send every packet whose departure is certain: its cycle c is at most H (every vector accepted before c is known, since the
// chip has ticked every cycle below H) and its ready cycle is below frontier (no packet that has not reached this chip arrives
// before it, so none is made ahead of it).
static void ctl_depart(cycle_t frontier, cycle_t H) {
  if (frontier < H) H = frontier;
  for (int l = 0; l < MAXLINKS; l++) {
    cpend_t *h;
    while ((h = ctl_head(l))) {
      cycle_t c = ctl_candidate(l, h->ready);
      if (h->ready >= frontier || c > H) break;
      ctl_send(h, c);
    }
  }
}

static cycle_t ctl_next_event(void) { return cin_head < n_cin ? cin[cin_head].arrival : NONE; }
static cycle_t pk_next_event(void) { return pk_pos < n_my_pk ? my_pk[pk_pos]->cycle : NONE; }

// The control unit's packets of cycle t: the packets that arrived (consumed here) and the chip's own P lines. Returns
// true when a CSR was written (the cycle is then ticked).
static bool ctl_process(uint64_t t) {
  bool mutated = false;
  while (cin_head < n_cin && cin[cin_head].arrival == t) {
    cin_t x = cin[cin_head++];
    if (cin_head == n_cin) cin_head = n_cin = 0;
    if (trace) fprintf(out, "T pkt %" PRIu64 " %s %d\n", t, x.type == PT_CTRL ? "ctrl" : "csr", x.link);
    if (x.type == PT_CTRL) continue;
    ctlpkt_t p = x.p;
    if (p.op == OP_REPLY) {
      if (trace) fprintf(out, "T csr %" PRIu64 " %d %03" PRIx64 " %" PRIx64 "\n", t, p.target, p.addr, p.val);
    } else if (p.op == OP_WR) {
      mpz_set_ui(sc, t);
      zmodel_csr_write(p.addr, p.val, sc);
      mutated = true;
    } else {
      p.op = OP_REPLY; p.val = zmodel_csr_read(p.addr);
      cpend_add(&p, PT_CSR, x.link, t);                        // the answer goes back on the link the request came in
    }
  }
  while (pk_pos < n_my_pk && my_pk[pk_pos]->cycle == t) {
    pk_t *k = my_pk[pk_pos++];
    if (k->kind == PK_CTRL) {
      ctlpkt_t p = {OP_CTRL, my_chip, my_chip, 0, 0, k->id};
      cpend_add(&p, PT_CTRL, k->link, t);
    } else {
      int pl; int peer = link_peer(my_chip, k->link, &pl);
      ctlpkt_t p = {k->kind == PK_CSR_W ? OP_WR : OP_RD, my_chip, peer, k->addr, k->val, k->id};
      cpend_add(&p, PT_CSR, k->link, t);
    }
  }
  return mutated;
}

static void pk_setup(int chip) {
  n_my_pk = pk_pos = 0;
  free(my_pk); my_pk = NULL;
  for (size_t i = 0; i < n_in_pk; i++) if (in_pk[i].chip == chip) {   // the input is in line order; sort by cycle (stable)
    my_pk = xrealloc(my_pk, (n_my_pk + 1) * sizeof *my_pk);
    size_t j = n_my_pk++;
    while (j > 0 && my_pk[j - 1]->cycle > in_pk[i].cycle) { my_pk[j] = my_pk[j - 1]; j--; }
    my_pk[j] = &in_pk[i];
  }
}

// ---------------------------------------------------------------------------------------
// The host (model/core/csr.sail; microarchitecture.md sec. 14). Every chip has its own host interface; the host of a chip is a
// script of timed actions (the H lines after the chip's C line), applied before the tick of their cycle. A DMA vector requested at
// cycle h is eligible at e = max(h, previous arrival + host.cycles_per_vector); it enters the ingress FIFO at e when the FIFO holds
// fewer than host.ingress_depth vectors. Otherwise the DMA engine waits (nothing is dropped, "T dma <e> wait") and the vector enters
// in the cycle after the PciReceive that frees a slot.

static sail_int sc, snext;
typedef struct { unsigned char b[MAXLANES]; } hvec_t;
typedef struct { uint64_t cycle; unsigned char b[MAXLANES]; } yvec_t;
static hev_t **my_host; static size_t n_my_host, host_pos;        // this chip's CSR and inject actions, by cycle
static hev_t **my_dma; static size_t n_my_dma, dma_pos;           // its DMA requests, by request cycle
static cycle_t dma_last, dma_floor;                              // the previous arrival (NONE: none), the earliest entry after a freed slot
static int dma_blocked;                                           // the head request found the FIFO full and waits for a PciReceive
static hvec_t *ing; static size_t ing_head, ing_n, ing_cap;       // the ingress FIFO
static yvec_t *egress; static size_t n_egr, cap_egr;              // the vectors PciSend gave the host
static uint64_t my_start;

static void hev_sort(hev_t **a, size_t n) {                           // stable insertion sort by cycle
  for (size_t i = 1; i < n; i++) {
    hev_t *x = a[i];
    size_t j = i;
    while (j > 0 && a[j - 1]->cycle > x->cycle) { a[j] = a[j - 1]; j--; }
    a[j] = x;
  }
}
static void host_setup(int chip) {
  my_start = in_start[chip];
  n_my_host = host_pos = n_my_dma = dma_pos = 0; ing_head = ing_n = 0; n_egr = 0;
  dma_last = NONE; dma_floor = 0; dma_blocked = 0;
  free(my_host); my_host = NULL; free(my_dma); my_dma = NULL;
  for (size_t i = 0; i < n_in_host; i++) if (in_host[i].chip == chip) {
    if (in_host[i].kind == H_DMA) {
      my_dma = xrealloc(my_dma, (n_my_dma + 1) * sizeof *my_dma);
      my_dma[n_my_dma++] = &in_host[i];
    } else {
      my_host = xrealloc(my_host, (n_my_host + 1) * sizeof *my_host);
      my_host[n_my_host++] = &in_host[i];
    }
  }
  hev_sort(my_host, n_my_host);
  hev_sort(my_dma, n_my_dma);
}
// The cycle at which the head DMA request enters the FIFO if it finds room (NONE: none, or it waits for a PciReceive).
static cycle_t dma_eligible(void) {
  if (dma_pos >= n_my_dma || dma_blocked) return NONE;
  cycle_t e = my_dma[dma_pos]->cycle;
  if (dma_last != NONE && dma_last + (uint64_t)zhost_cycles_per_vector > e) e = dma_last + (uint64_t)zhost_cycles_per_vector;
  if (dma_floor > e) e = dma_floor;
  return e;
}
static cycle_t host_next_event(void) {
  cycle_t a = host_pos < n_my_host ? my_host[host_pos]->cycle : NONE, d = dma_eligible();
  return d < a ? d : a;
}

// Apply the host's actions of cycle t. Returns true when the model was changed (the cycle is
// then ticked, so that the model's own events are recomputed from it).
static bool host_process(uint64_t t) {
  bool mutated = false;
  while (dma_eligible() == t) {
    if (ing_n - ing_head >= (size_t)zhost_ingress_depth) {
      dma_blocked = 1;
      if (trace) fprintf(out, "T dma %" PRIu64 " wait\n", t);
    } else {
      GROW(ing, ing_n, ing_cap);
      memcpy(ing[ing_n++].b, my_dma[dma_pos++]->b, MAXLANES);
      dma_last = t;
      if (trace) fprintf(out, "T dma %" PRIu64 " in\n", t);
    }
  }
  while (host_pos < n_my_host && my_host[host_pos]->cycle == t) {
    hev_t *h = my_host[host_pos++];
    switch (h->kind) {
    case H_CSR_W:
      mpz_set_ui(sc, t);
      zmodel_csr_write(h->addr, h->val, sc);
      mutated = true;
      break;
    case H_CSR_R:
      if (trace) fprintf(out, "T csr %" PRIu64 " %03" PRIx64 " %" PRIx64 "\n", t, h->addr, zmodel_csr_read(h->addr));
      break;
    case H_INJECT:
      mpz_set_ui(sc, t);
      for (size_t i = 0; i < h->n; i++) zmodel_inject_unit(h->q, sc, h->u[i]);
      mutated = true;
      break;
    case H_FLIP_SRAM:
      mpz_set_ui(sc, h->addr);
      zmodel_flip_sram(h->fa, sc, h->fsl, h->fbit);
      mutated = true;
      break;
    case H_FLIP_IQ:
      mpz_set_ui(sc, h->addr);
      if (!zmodel_flip_iq(h->fa, sc, h->fbit)) {
        fprintf(stderr, "tsp_sim: F iq at cycle %" PRIu64 ": text word %" PRIx64 " of queue %x is not in the instruction queue\n", t, h->addr, h->fa);
        exit(2);
      }
      mutated = true;
      break;
    case H_FLIP_SRF:
      mpz_set_ui(sc, t);
      zmodel_flip_srf(sc, h->fw != 0, h->fa, h->q, h->fsl, h->fbit);
      mutated = true;
      break;
    case H_FLIP_MXM:
      mpz_set_ui(sc, h->addr);
      zmodel_inject_mxm(h->fa, sc, h->fbit);
      mutated = true;
      break;
    default: break;
    }
  }
  return mutated;
}

int64_t host_chip_id(unit u) { return my_chip; }
bool host_recv_ready(sail_int c) { return ing_head < ing_n; }
void host_recv(lbits *o, sail_int c) {
  static const unsigned char zero[MAXLANES];
  if (ing_head < ing_n) {
    from_bytes(o, ing[ing_head++].b);
    if (dma_blocked) { dma_blocked = 0; dma_floor = (cycle_t)mpz_get_ui(c) + 1; }   // a slot is free: the waiting vector enters in the next cycle
  } else from_bytes(o, zero);
}
unit host_send(sail_int c, lbits v) {
  GROW(egress, n_egr, cap_egr);
  yvec_t *y = &egress[n_egr++];
  y->cycle = mpz_get_ui(c);
  to_bytes(y->b, v);
  if (trace) fprintf(out, "T dma %" PRIu64 " out\n", y->cycle);
  return UNIT;
}
unit host_irq_callback(sail_int c, int64_t n) {
  if (n == 0) err_irq_seen = 1;                   // vector 0 is the error interrupt (HostNotify has n = 1..15)
  if (trace) fprintf(out, "T irq %lu msix %" PRId64 "\n", mpz_get_ui(c), n);
  return UNIT;
}

// ---------------------------------------------------------------------------------------
// The cycle loop.

static cycle_t t_next = NONE;                   // the next cycle to tick (NONE: nothing)
static sail_int sc, snext;

static void over_budget(cycle_t t) {
  char b[40]; cycle_decimal(b, t);
  fprintf(stderr, "tsp_sim: the run needs cycle %s, beyond --max-cycles %" PRIu64 "; stopped (output incomplete)\n", b, max_cycles);
  exit(3);
}

// A Sail exception (the model's internal_error or not_implemented) leaves the model's state half
// updated, and the generated code only reports it in model_fini(): a run that went on would
// loop or print results of a broken state. Stop at once: a message on stderr, exit code 4.
static void check_model(uint64_t t) {
  if (!have_exception) return;
  const char *m = current_exception->kind == Kind_zError_internal_error ? current_exception->variants.zError_internal_error
                                                                          : current_exception->variants.zError_not_implemented;
  fprintf(stderr, "tsp_sim: model fault at cycle %" PRIu64 ": %s\n", t, m);
  fflush(out);
  exit(4);
}

// Tick every cycle that has something to do, below wend.
// The earliest cycle at which something happens on this chip (an event that can change what the program does).
// The earliest epoch boundary at which a queue that has a DeskewRecv in its polling loop can find its vector (NONE: none).
static sail_int spoll;
static cycle_t poll_next_event(void) {
  zmodel_poll_event(&spoll, UNIT);
  return model_cycle(spoll);
}
static cycle_t real_next(void) {
  cycle_t a = link_next_event(), h = host_next_event(), c = ctl_next_event(), k = pk_next_event(), pe = poll_next_event();
  if (pe < a) a = pe;
  if (h < a) a = h;
  if (c < a) a = c;
  if (k < a) a = k;
  if (t_next < a) a = t_next;
  // Stop mode must visit a patrol slot before later external actions: patrol can
  // raise the first error IRQ. Patrol alone does not extend an otherwise finished run.
  if (on_irq == IRQ_STOP && zscrub_period > 0 && a != NONE) {
    cycle_t p = model_cycle(zscrub_next);
    if (p < a) a = p;
  }
  return a;
}
static cycle_t next_time(void) { return real_next(); }
// What the chip reports to the lock-step coordinator: its next event, or the earliest cycle at which a control packet
// that is waiting for a gap in the schedule could leave, whichever is earlier.
static cycle_t report_time(void) {
  cycle_t a = real_next(), w = ctl_wake();
  return w < a ? w : a;
}

static void run_until(cycle_t wend) {
  cycle_t t;
  for (;;) {
    if (n_cpend) ctl_depart(wend, real_next());     // the control packets whose departure is now certain
    t = next_time();
    if (t == NONE || t >= wend) break;
    if (t > max_cycles) over_budget(t);
    mpz_set_ui(sc, t);
    zmodel_scrub_before(sc);                        // the patrol scrub cycles before t, before anything acts at the start of t
    process_arrivals(t);                            // the arrivals of cycle t, before its tick
    bool mut = host_process(t);                     // the host's actions of cycle t
    if (ctl_process(t)) mut = true;                 // the control unit's: arrived csr and ctrl packets
    if (on_irq == IRQ_STOP && zscrub_period > 0 && model_cycle(zscrub_next) == t) mut = true;
    if (on_irq == IRQ_STOP && err_irq_seen) mut = true; // finish the IRQ cycle even without queued model work
    if (poll_next_event() == t) mut = true;         // an epoch boundary at which a DeskewRecv looks for its vector
    if (n_cpend) ctl_depart(wend, t);               // the packets made in cycle t that leave in it, before the tick
    if (mut && t_next > t) t_next = t;
    if (t_next != t) continue;                      // an arrival with nothing to tick
    mpz_set_ui(sc, t);
    ztick(sc);
    check_model(t);
    {                                               // F mxm: a fault that no ABC consumed in its cycle is an input error
      sail_int pend; CREATE(sail_int)(&pend);
      zmodel_mxm_pending(&pend, UNIT);
      long pl = mpz_get_si(pend);
      KILL(sail_int)(&pend);
      if (pl >= 0) {
        fflush(out);
        fprintf(stderr, "tsp_sim: F mxm: no ABC executes on plane %ld in cycle %" PRIx64 " (chip %d)\n", pl, (uint64_t)t, my_chip);
        exit(2);
      }
    }
    if (on_irq == IRQ_STOP && err_irq_seen) { stopped = 1; return; }   // the end of the cycle of the first error interrupt
    zmodel_next_event(&snext, sc);
    check_model(t);
    cycle_t ev = NONE;
    if (mpz_sgn(snext) >= 0) {
      if (model_cycle(snext) <= t) { fprintf(stderr, "tsp_sim: bad next event from the model\n"); exit(4); }
      ev = model_cycle(snext);
    }
    t_next = ev;
  }
}

static void finish_check(void) {
  bool q = zmodel_quiescent(UNIT);
  check_model(0);
  if (!q) { fprintf(stderr, "tsp_sim: model not quiescent with no event left\n"); exit(4); }
  zmodel_finish(UNIT);                              // OTS-E-IQ-004: text left in an IQ
  check_model(0);
  for (size_t i = 0; i < n_in_lf; i++)              // an F link line that marked no vector is an input error
    if (in_lf[i].chip == my_chip && !in_lf[i].used) {
      fflush(out);
      fprintf(stderr, "tsp_sim: F link: no vector is sent on link %d in cycle %" PRIx64 " (chip %d)\n", in_lf[i].link, in_lf[i].cycle, my_chip);
      exit(2);
    }
}

static void print_results(void) {
  if (ndiag_initialized && mpz_sgn(ndiag) != 0) gmp_fprintf(out, "E %s %" PRIu64 " %Zd\n", first_code, first_cycle, ndiag);
  else fprintf(out, "E - -1 0\n");
  for (size_t i = 0; i < n_in_dump; i++) {
    dump_t *d = &in_dump[i];
    if (d->chip != my_chip) continue;
    size_t k = sram_key(d->slice, (uint64_t)d->addr);
    fprintf(out, "W %lx %lx", d->slice, d->addr);
    for (int l = 0; l < (vl() > DUMP_MIN_LANES ? vl() : DUMP_MIN_LANES); l++) fprintf(out, " %02x", sram[k] && l < vl() ? sram[k][l] : 0);
    fprintf(out, "\n");
  }
  for (size_t i = 0; i < n_egr; i++) {
    fprintf(out, "Y %" PRIu64, egress[i].cycle);
    for (int l = 0; l < vl(); l++) fprintf(out, " %02x", egress[i].b[l]);
    fprintf(out, "\n");
  }
  for (size_t i = 0; i < n_in_rcsr; i++)
    if (in_rcsr[i].chip == my_chip) fprintf(out, "X %03" PRIx64 " %" PRIx64 "\n", in_rcsr[i].addr, zmodel_csr_read(in_rcsr[i].addr));
}

static void chip_setup(int chip) {
  my_chip = chip;
  build_program();                                  // the initial IQ images: zreset loads them
  zreset(UNIT);
  sram = calloc((size_t)NSLICES * (size_t)zwords_per_slice, sizeof *sram);
  if (!sram) { fprintf(stderr, "tsp_sim: out of memory\n"); exit(2); }
  sram_ov = NULL;
  for (size_t i = 0; i < n_in_pre; i++) {
    pre_t *p = &in_pre[i];
    if (p->chip != chip) continue;
    size_t k = sram_key(p->slice, (uint64_t)p->addr);
    if (!sram[k]) sram[k] = xrealloc(NULL, vl());
    memset(sram[k], 0, vl());
    memcpy(sram[k], p->b, p->n);
  }
  CREATE(sail_int)(&sc);
  CREATE(sail_int)(&snext);
  CREATE(sail_int)(&spoll);
  host_setup(chip);
  pk_setup(chip);
  if (my_start) { mpz_set_ui(sc, my_start); zmodel_set_start(sc); }
  t_next = my_start;                                // tick(start) dispatches the images; the model drives the rest
}

// Several chips: one process per chip, lock step windows coordinated by the parent.

static void wr(int fd, const void *p, size_t n) {
  const char *s = p;
  while (n) { ssize_t k = write(fd, s, n); if (k <= 0) { if (errno == EINTR) continue; if (process_cleanup) process_cleanup(); _exit(5); } s += k; n -= (size_t)k; }
}
static int rd(int fd, void *p, size_t n) {
  char *s = p;
  while (n) { ssize_t k = read(fd, s, n); if (k == 0) return 0; if (k < 0) { if (errno == EINTR) continue; return 0; } s += k; n -= (size_t)k; }
  return 1;
}

// Child: report (next time, outbound vectors); receive the inbound vectors and report the next
// time again (an arrival that could overrun is an event, known only after delivery);
// receive (command, wend). command 0 = run to wend, 1 = finish, 2 = stop (over budget).
static void child_main(int up, int down) {
  cycle_t wend = 0;
  for (;;) {
    run_until(wend);
    cycle_t tn = report_time();
    wr(up, &tn, sizeof tn);
    uint64_t n = n_out;
    wr(up, &n, sizeof n);
    if (n_out) wr(up, outbound, n_out * sizeof *outbound);
    n_out = 0;
    int32_t cmd;
    if (!rd(down, &n, sizeof n)) _exit(5);
    for (uint64_t i = 0; i < n; i++) {
      vecmsg_t m;
      if (!rd(down, &m, sizeof m)) _exit(5);
      deliver(&m);
    }
    tn = report_time();
    wr(up, &tn, sizeof tn);
    if (!rd(down, &cmd, sizeof cmd) || !rd(down, &wend, sizeof wend)) _exit(5);
    if (cmd == 2) { fflush(out); _exit(0); }
    if (cmd == 1) break;
  }
  finish_check();
  print_results();
  fflush(out);
  _exit(0);
}

// The parent owns every coordinator descriptor and child. This context is active only
// while run_chips is on the stack; inherited callbacks in a child are deliberately inert.
typedef struct {
  pid_t owner;
  int count;
  int *up, *down;
  pid_t *pid;
  FILE **trace;
  vecmsg_t **route;
  size_t *nroute, *caproute;
  cycle_t *time;
  struct sigaction old_chld;
  int restore_chld;
  struct sigaction old_pipe;
  int restore_pipe;
} chips_context;
static chips_context *active_chips;

static int reap_child(pid_t pid, int *status) {
  pid_t result;
  do { result = waitpid(pid, status, 0); } while (result < 0 && errno == EINTR);
  return result == pid;
}

static void release_chips(void) {
  chips_context *ctx = active_chips;
  if (!ctx || ctx->owner != getpid()) return;
  // Disconnect peers first; kill only children still owned by this coordinator.
  for (int i = 0; i < ctx->count; i++) {
    if (ctx->up[i] >= 0) close(ctx->up[i]);
    if (ctx->down[i] >= 0) close(ctx->down[i]);
    if (ctx->pid[i] > 0) kill(ctx->pid[i], SIGKILL);
  }
  for (int i = 0; i < ctx->count; i++) {
    int status;
    if (ctx->pid[i] > 0) reap_child(ctx->pid[i], &status);
    if (ctx->trace[i]) fclose(ctx->trace[i]);
    free(ctx->route[i]);
  }
  if (ctx->restore_chld) sigaction(SIGCHLD, &ctx->old_chld, NULL);
  if (ctx->restore_pipe) sigaction(SIGPIPE, &ctx->old_pipe, NULL);
  free(ctx->up); free(ctx->down); free(ctx->pid); free(ctx->trace);
  free(ctx->route); free(ctx->nroute); free(ctx->caproute); free(ctx->time);
  active_chips = NULL;
  process_cleanup = NULL;
}

static int failed_chip(int chip) {
  chips_context *ctx = active_chips;
  int status;
  // A failed read need not mean the child has exited (e.g. an I/O failure).
  // Stop it before waiting so cleanup never waits for its next command.
  kill(ctx->pid[chip], SIGKILL);
  int reaped = reap_child(ctx->pid[chip], &status);
  ctx->pid[chip] = 0;
  return reaped && WIFEXITED(status) && WEXITSTATUS(status) ? WEXITSTATUS(status) : 4;
}

// Each chip writes its trace and results to its own temporary file; the parent copies
// them to stdout in chip order, each after a "C <chip>" line.
static int run_chips(void) {
  cycle_t L = min_link_latency == NONE ? 1 : min_link_latency;   // no connected link: any window
  if (L == 0) { fprintf(stderr, "tsp_sim: --chips needs link latencies >= 1\n"); return 2; }
  if (win_cap) {                                  // --window N: a shorter window, for testing; the results do not depend on it
    if (win_cap < L) L = win_cap;
  }
  int K = n_chips;
  int *up = xcalloc(K, sizeof *up), *down = xcalloc(K, sizeof *down);
  pid_t *pid = xcalloc(K, sizeof *pid);
  FILE **tf = xcalloc(K, sizeof *tf);
  vecmsg_t **route = xcalloc(K, sizeof *route);
  size_t *nroute = xcalloc(K, sizeof *nroute), *caproute = xcalloc(K, sizeof *caproute);
  cycle_t *tn = xcalloc(K, sizeof *tn);
  for (int i = 0; i < K; i++) { up[i] = -1; down[i] = -1; }
  chips_context context = {getpid(), K, up, down, pid, tf, route, nroute, caproute, tn};
  active_chips = &context;
  process_cleanup = release_chips;
  int rc = 0;
  if (sigaction(SIGCHLD, NULL, &context.old_chld)) { perror("tsp_sim: sigaction"); rc = 2; goto done; }
  if (context.old_chld.sa_handler == SIG_IGN || (context.old_chld.sa_flags & SA_NOCLDWAIT)) {
    struct sigaction action = {0};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGCHLD, &action, NULL)) { perror("tsp_sim: sigaction"); rc = 2; goto done; }
    context.restore_chld = 1;
  }
  fflush(stdout);
  for (int i = 0; i < K; i++) {
    int u[2], d[2];
    tf[i] = tmpfile();
    if (!tf[i]) { perror("tsp_sim: tmpfile"); rc = 2; goto done; }
    if (pipe(u)) { perror("tsp_sim: pipe"); rc = 2; goto done; }
    if (pipe(d)) { perror("tsp_sim: pipe"); close(u[0]); close(u[1]); rc = 2; goto done; }
    pid[i] = fork();
    if (pid[i] < 0) {
      perror("tsp_sim: fork"); close(u[0]); close(u[1]); close(d[0]); close(d[1]);
      pid[i] = 0; rc = 2; goto done;
    }
    if (pid[i] == 0) {
      close(u[0]); close(d[1]);
      for (int j = 0; j < i; j++) { close(up[j]); close(down[j]); }
      out = tf[i];
      chip_setup(i);
      child_main(u[1], d[0]);
    }
    close(u[1]); close(d[0]);
    up[i] = u[0]; down[i] = d[1];
  }
  // Parent IPC errors must reach wr's cleanup path instead of terminating at SIGPIPE.
  // Install after the forks so each chip retains its original signal disposition.
  if (sigaction(SIGPIPE, NULL, &context.old_pipe)) { perror("tsp_sim: sigaction"); rc = 2; goto done; }
  struct sigaction pipe_action = {0};
  pipe_action.sa_handler = SIG_IGN;
  sigemptyset(&pipe_action.sa_mask);
  if (sigaction(SIGPIPE, &pipe_action, NULL)) { perror("tsp_sim: sigaction"); rc = 2; goto done; }
  context.restore_pipe = 1;
  cycle_t wend = 0, g = NONE;
  int32_t cmd = 0;
  while (cmd == 0) {
    for (int i = 0; i < K; i++) {
      uint64_t n;
      if (!rd(up[i], &tn[i], sizeof tn[i]) || !rd(up[i], &n, sizeof n)) {
        rc = failed_chip(i); goto done;
      }
      for (uint64_t j = 0; j < n; j++) {
        vecmsg_t m;
        if (!rd(up[i], &m, sizeof m)) { rc = 4; goto done; }
        GROW(route[m.chip], nroute[m.chip], caproute[m.chip]);
        route[m.chip][nroute[m.chip]++] = m;
      }
    }
    // Deliver the vectors; each chip then reports its next time again, which includes the
    // arrivals that could overrun.
    for (int i = 0; i < K; i++) {
      uint64_t n = nroute[i];
      wr(down[i], &n, sizeof n);
      if (n) wr(down[i], route[i], n * sizeof *route[i]);
      nroute[i] = 0;
    }
    for (int i = 0; i < K; i++)
      if (!rd(up[i], &tn[i], sizeof tn[i])) {
        rc = failed_chip(i); goto done;
      }
    // Every chip has ticked every cycle below wend and next acts at tn[i] >= wend. A
    // vector sent at t >= g arrives at t + L >= g + L (L: the smallest connected link
    // latency), so all chips may run [g, g + L).
    g = NONE;
    for (int i = 0; i < K; i++) if (tn[i] < g) g = tn[i];
    if (g == NONE) cmd = 1;
    else if (g > max_cycles) cmd = 2;
    else wend = g + L;
    for (int i = 0; i < K; i++) { wr(down[i], &cmd, sizeof cmd); wr(down[i], &wend, sizeof wend); }
  }
  for (int i = 0; i < K; i++) {
    int st;
    if (!reap_child(pid[i], &st)) { pid[i] = 0; rc = 4; goto done; }
    pid[i] = 0;
    if (!WIFEXITED(st)) rc = 4; else if (WEXITSTATUS(st) && !rc) rc = WEXITSTATUS(st);
  }
  if (rc) goto done;
  // IPC writes need a recoverable EPIPE; stdout keeps the caller's signal policy.
  if (context.restore_pipe) {
    sigaction(SIGPIPE, &context.old_pipe, NULL);
    context.restore_pipe = 0;
  }
  for (int i = 0; i < K; i++) {
    printf("C %d\n", i);
    rewind(tf[i]);
    char b[65536]; size_t k;
    while ((k = fread(b, 1, sizeof b, tf[i])) > 0) fwrite(b, 1, k, stdout);
  }
  fflush(stdout);
  release_chips();
  if (cmd == 2) over_budget(g);
  return 0;
done:
  release_chips();
  return rc;
}

// ---------------------------------------------------------------------------------------

// --on-irq replay: the first run in a child process (the model's state is global), then, if it raised an error interrupt,
// the second run in this process from reset with the persistent faults only.
static int replay_run(void) {
  fflush(stdout);
  pid_t pid = fork();
  if (pid < 0) { perror("tsp_sim: fork"); return 2; }
  if (pid == 0) {
    chip_setup(0);
    run_until(NONE);
    finish_check();
    print_results();
    fflush(out);
    _exit(err_irq_seen ? 5 : 0);
  }
  int st = 0;
  if (waitpid(pid, &st, 0) < 0) { perror("tsp_sim: waitpid"); return 2; }
  int rc = WIFEXITED(st) ? WEXITSTATUS(st) : 4;
  if (rc != 5) { model_fini(); return rc; }        // no error interrupt (0), or a failure of the run (2, 3, 4)
  size_t k = 0;                                   // the replay: the transient faults are gone, the persistent ones come again
  for (size_t i = 0; i < n_in_host; i++) {
    int fault = in_host[i].kind == H_FLIP_SRAM || in_host[i].kind == H_FLIP_SRF || in_host[i].kind == H_FLIP_IQ || in_host[i].kind == H_FLIP_MXM;
    if (!fault || in_host[i].persist) in_host[k++] = in_host[i];
  }
  n_in_host = k;
  k = 0;                                          // the F link lines likewise
  for (size_t i = 0; i < n_in_lf; i++) if (in_lf[i].persist) in_lf[k++] = in_lf[i];
  n_in_lf = k;
  fprintf(out, "T replay start\n");
  fflush(out);
  chip_setup(0);
  run_until(NONE);
  finish_check();
  print_results();
  fprintf(out, err_irq_seen ? "T replay persistent\n" : "T replay resolved\n");
  model_fini();
  return err_irq_seen ? 5 : 0;
}

static void usage(const char *p) {
  fprintf(stderr, "usage: %s <config.json> [--trace] [--lint] [--on-irq stop|continue|replay] [--max-cycles N] [--chips K | --topology FILE] [--window N] [--link-latency N | --link-latency-seed S | --link-jitter-seed S]\n", p);
  exit(2);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    printf("usage: %s <config.json> [--trace] [--observe] [--lint] [--on-irq stop|continue|replay] [--max-cycles N] [--chips K | --topology FILE] [--window N] [--link-latency N | --link-latency-seed S | --link-jitter-seed S]\n", argv[0]);
    return 0;
  }
  if (argc < 2) usage(argv[0]);
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--trace") == 0) trace = 1;
    else if (strcmp(argv[i], "--observe") == 0) observe = 1;
    else if (strcmp(argv[i], "--lint") == 0) lints = 1;
    else if (strcmp(argv[i], "--on-irq") == 0 && i + 1 < argc) {
      const char *a = argv[++i];
      if (strcmp(a, "stop") == 0) on_irq = IRQ_STOP;
      else if (strcmp(a, "continue") == 0) on_irq = IRQ_CONTINUE;
      else if (strcmp(a, "replay") == 0) on_irq = IRQ_REPLAY;
      else { fprintf(stderr, "tsp_sim: --on-irq must be stop, continue or replay\n"); return 2; }
    }
    else if (strcmp(argv[i], "--max-cycles") == 0 && i + 1 < argc) {
      // strtoull accepts leading space and a sign ("-1" is 2^64 - 1): digits only.
      const char *a = argv[++i];
      if (!(*a >= '0' && *a <= '9')) usage(argv[0]);
      char *e; errno = 0; max_cycles = strtoull(a, &e, 0);
      if (*e || errno) usage(argv[0]);
    } else if (strcmp(argv[i], "--chips") == 0 && i + 1 < argc) {
      char *e; long k = strtol(argv[++i], &e, 0);
      if (*e || k < 1 || k > 264) { fprintf(stderr, "tsp_sim: --chips must be 1..264 (33 nodes of 8; a larger system needs --topology)\n"); return 2; }
      n_chips = (int)k;
      chips_given = 1;
    } else if (strcmp(argv[i], "--topology") == 0 && i + 1 < argc) {
      topo_path = argv[++i];
    } else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
      char *e; errno = 0; long long w = strtoll(argv[++i], &e, 0);
      if (*e || errno || w < 1) usage(argv[0]);
      win_cap = (uint64_t)w;
    } else if (strcmp(argv[i], "--link-latency") == 0 && i + 1 < argc) {
      char *e; errno = 0; lat_fixed = strtol(argv[++i], &e, 0);
      if (*e || errno || lat_fixed < 0) usage(argv[0]);
    } else if (strcmp(argv[i], "--link-latency-seed") == 0 && i + 1 < argc) {
      char *e; errno = 0; lat_seed = strtoll(argv[++i], &e, 0);
      if (*e || errno || lat_seed < 0) usage(argv[0]);
    } else if (strcmp(argv[i], "--link-jitter-seed") == 0 && i + 1 < argc) {
      char *e; errno = 0; jit_seed = strtoll(argv[++i], &e, 0);
      if (*e || errno || jit_seed < 0) usage(argv[0]);
    } else usage(argv[0]);
  }
  tsp_config_load("tsp_sim", argv[1], TSP_CONFIG_SCHEMA);
  model_init();
  tsp_config_model_check("tsp_sim", argv[1]);
  // The schema relations already give link_latency_min <= link_latency <= link_latency_max
  // (tools/schema_invariants.py C2C_RELATIONS); this keeps the harness safe if the schema
  // is ever weakened, with the same message form as every other config error.
  if (mpz_cmp(zc2c_link_latency_min, zc2c_link_latency) > 0 || mpz_cmp(zc2c_link_latency, zc2c_link_latency_max) > 0) {
    fprintf(stderr, "tsp_sim: config %s: c2c.link_latency: must be in [c2c.link_latency_min, c2c.link_latency_max]\n", argv[1]);
    model_fini();
    return 2;
  }
  // Every buffer here holds MAXLANES bytes; the schema bounds superlanes, this keeps the
  // two in step if a profile with more lanes is ever allowed.
  if (zvlanes < 1 || zvlanes > MAXLANES) {
    fprintf(stderr, "tsp_sim: config %s: geometry.superlanes: %" PRId64 " lanes, this harness holds at most %d\n",
            argv[1], zvlanes, MAXLANES);
    model_fini();
    return 2;
  }
  out = stdout;
  if (topo_path) {
    if (chips_given) { fprintf(stderr, "tsp_sim: --topology gives the number of chips: do not combine it with --chips\n"); model_fini(); return 2; }
    n_chips = topology_load(topo_path);
    if (jit_seed < 0 && topo_jit_seed >= 0 && lat_fixed < 0 && lat_seed < 0) jit_seed = topo_jit_seed;
  }
  if (lat_fixed >= 0 && lat_seed >= 0) usage(argv[0]);
  if (jit_seed >= 0 && (lat_fixed >= 0 || lat_seed >= 0)) usage(argv[0]);
  if (on_irq != IRQ_CONTINUE && n_chips > 1) { fprintf(stderr, "tsp_sim: --on-irq stop and replay need --chips 1\n"); model_fini(); return 2; }
  read_input(n_chips);
  links_setup();
  if (n_chips > 1) { int rc = run_chips(); model_fini(); return rc; }

  // One chip: the TSP with every link looped back. The trace is printed as it happens.
  if (on_irq == IRQ_REPLAY) return replay_run();
  chip_setup(0);
  run_until(NONE);
  if (stopped) { print_results(); model_fini(); return 5; }
  finish_check();
  print_results();
  model_fini();
  return 0;
}
