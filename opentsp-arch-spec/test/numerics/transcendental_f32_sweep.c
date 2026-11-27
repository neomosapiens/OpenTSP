// Double-precision sweep of every f32 bit pattern for TanH, Exp, RSqrt and Log
// (tools/check_transcendental_f32.sh).
//
// For every input whose result is a finite non-exact value the function is evaluated in
// double precision (libm; error far below 2^-45 relative) and the input is flagged "hard"
// when the double value lies within 2^-40 (relative) of a rounding boundary of f32 (a
// midpoint between consecutive f32 values; subnormal output range and the overflow
// threshold 2^128 - 2^103 included). Everything else is farther than 2^-41 from every
// boundary only if the assumed libm error bound holds globally. Sampling below
// cannot establish that bound or correctness of every untested Sail evaluation.
//
// Inputs that the model's fp_tr_value treats as specials (NaN, +-Inf, +-0, a negative
// argument of RSqrt/Log, Log(1)) are counted but not filtered: the check script feeds
// them to the Sail evaluator separately.
//
// usage: transcendental_f32_sweep <threads> <stride> <hard.txt> <stats.txt>
//   stride 1 sweeps every pattern; stride s > 1 sweeps x = 0, s, 2s, ... (quick mode).
//   hard.txt : one line "<op> <x hex>" per hard input (op 21 TanH, 22 Exp, 23 RSqrt, 24 Log)
//   stats.txt: one line per op "op=.. swept=.. special=.. hard=.. min_rel_dist=.. min_at=..
//              audit_n=.. audit_max_rel_err=.."
// The audit compares the double result with the long double result (expl, logl, tanhl,
// sqrtl) on every 256th input a thread sweeps and records the largest relative difference: it
// samples the assumption only when long double has more mantissa bits than double.
// A separate mpmath audit is mandatory on every platform; neither sample proves
// a global error bound for all inputs.
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUNK (1u << 20)
#define HARD_LOG2 (-40)

static const int OPS[4] = {21, 22, 23, 24};

typedef struct { uint32_t x; } hard_rec;

typedef struct {
  int op;
  uint64_t swept, special, hard, audit_n;
  double min_rel;
  uint32_t min_at;
  double audit_max;
  hard_rec *recs;
  size_t nrecs, cap;
} acc_t;

static uint32_t stride;
static atomic_uint_fast64_t next_chunk;

static inline float bits_to_f(uint32_t x) { float f; memcpy(&f, &x, 4); return f; }

// Is the input handled by the model's fp_tr_value without the Ziv evaluator (or not a
// finite non-exact problem at all)? NaN, Inf, 0, negative RSqrt/Log, Log(1).
static inline int special(int op, uint32_t x) {
  uint32_t ex = (x >> 23) & 0xff, mag = x & 0x7fffffffu;
  if (ex == 255 || mag == 0) return 1;
  if ((op == 23 || op == 24) && (x >> 31)) return 1;
  if (op == 24 && x == 0x3f800000u) return 1;
  return 0;
}

static inline double f32_ulp(float g) {   // spacing of f32 at g (g >= 0, finite)
  if (g < 0x1p-126f) return 0x1p-149;
  int e;
  frexpf(g, &e);                          // g = m 2^e, m in [0.5, 1): ilogb = e - 1
  return ldexp(1.0, e - 1 - 23);
}

// Relative distance of a (> 0, finite double) to the nearest f32 rounding boundary.
static inline double rel_dist(double a) {
  float g = (float)a;
  if (isinf(g)) g = 0x1.fffffep127f;                      // beyond FLT_MAX: boundary 2^128 - 2^103
  double ulp = f32_ulp(g);
  double up = (double)g + ulp / 2;
  double ulp_dn = ulp;
  if (g > 0x1p-126f && frexpf(g, &(int){0}) == 0.5f) ulp_dn = ulp / 2;   // g is a power of two
  double dn = (double)g - ulp_dn / 2;
  double d1 = fabs(a - up), d2 = fabs(a - dn);
  double d = d1 < d2 ? d1 : d2;
  return d / a;
}

static inline double eval_d(int op, double x) {
  switch (op) {
    case 21: return tanh(x);
    case 22: return exp(x);
    case 23: return 1.0 / sqrt(x);
    default: return log(x);
  }
}

static inline long double eval_l(int op, long double x) {
  switch (op) {
    case 21: return tanhl(x);
    case 22: return expl(x);
    case 23: return 1.0L / sqrtl(x);
    default: return logl(x);
  }
}

static void push(acc_t *a, uint32_t x) {
  if (a->nrecs == a->cap) {
    a->cap = a->cap ? a->cap * 2 : 1024;
    a->recs = realloc(a->recs, a->cap * sizeof(hard_rec));
    if (!a->recs) { perror("realloc"); exit(2); }
  }
  a->recs[a->nrecs++].x = x;
}

typedef struct { int op; acc_t *acc; } work_t;

static void *worker(void *arg) {
  work_t *w = arg;
  acc_t *a = w->acc;
  const int op = w->op;
  const uint64_t total = (uint64_t)1 << 32;
  const uint64_t nchunks = total / CHUNK;
  const double thr = ldexp(1.0, HARD_LOG2);
  for (;;) {
    uint64_t c = atomic_fetch_add(&next_chunk, 1);
    if (c >= nchunks) break;
    uint64_t lo = c * (uint64_t)CHUNK, hi = lo + CHUNK;
    uint64_t start = lo;
    if (stride > 1) start = (lo + stride - 1) / stride * stride;
    for (uint64_t xx = start; xx < hi; xx += stride) {
      uint32_t x = (uint32_t)xx;
      a->swept++;
      if (special(op, x)) { a->special++; continue; }
      double y = eval_d(op, (double)bits_to_f(x));
      double ay = fabs(y);
      if (!(ay > 0) || isinf(ay) || isnan(ay)) {
        // exact 0 / overflow in double: the result is far below / above every f32 boundary
        if (ay == 0 || isinf(ay)) continue;
        fprintf(stderr, "sweep: unexpected NaN for op %d x %08x\n", op, x);
        exit(3);
      }
      double rd = rel_dist(ay);
      if (rd < a->min_rel) { a->min_rel = rd; a->min_at = x; }
      if (rd <= thr) { a->hard++; push(a, x); }
      if (LDBL_MANT_DIG > DBL_MANT_DIG && (a->swept & 0xffu) == 0 && ay > 0x1p-1000) {
        long double yl = eval_l(op, (long double)bits_to_f(x));
        double err = (double)fabsl((y - yl) / yl);
        a->audit_n++;
        if (err > a->audit_max) a->audit_max = err;
      }
    }
  }
  return NULL;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--audit") == 0) {
    int op;
    unsigned x;
    while (scanf("%d %x", &op, &x) == 2) {
      if (op < 21 || op > 24) return 2;
      printf("%d %08x %a\n", op, x, eval_d(op, (double)bits_to_f(x)));
    }
    return ferror(stdin) ? 2 : 0;
  }
  if (argc != 5) { fprintf(stderr, "usage: %s <threads> <stride> <hard.txt> <stats.txt>\n", argv[0]); return 2; }
  int nthreads = atoi(argv[1]);
  stride = (uint32_t)strtoul(argv[2], NULL, 10);
  if (nthreads < 1 || nthreads > 64 || stride < 1) { fprintf(stderr, "bad arguments\n"); return 2; }
  FILE *fh = fopen(argv[3], "w"), *fs = fopen(argv[4], "w");
  if (!fh || !fs) { perror("open"); return 2; }
  for (int oi = 0; oi < 4; oi++) {
    int op = OPS[oi];
    atomic_store(&next_chunk, 0);
    pthread_t th[64];
    acc_t accs[64];
    work_t ws[64];
    for (int t = 0; t < nthreads; t++) {
      memset(&accs[t], 0, sizeof accs[t]);
      accs[t].op = op;
      accs[t].min_rel = INFINITY;
      ws[t].op = op; ws[t].acc = &accs[t];
      pthread_create(&th[t], NULL, worker, &ws[t]);
    }
    acc_t tot; memset(&tot, 0, sizeof tot);
    tot.min_rel = INFINITY;
    for (int t = 0; t < nthreads; t++) {
      pthread_join(th[t], NULL);
      acc_t *a = &accs[t];
      tot.swept += a->swept; tot.special += a->special; tot.hard += a->hard; tot.audit_n += a->audit_n;
      if (a->min_rel < tot.min_rel || (a->min_rel == tot.min_rel && a->min_at < tot.min_at)) {
        tot.min_rel = a->min_rel; tot.min_at = a->min_at;
      }
      if (a->audit_max > tot.audit_max) tot.audit_max = a->audit_max;
      for (size_t i = 0; i < a->nrecs; i++) fprintf(fh, "%d %08x\n", op, a->recs[i].x);
      free(a->recs);
    }
    fprintf(fs, "op=%d swept=%llu special=%llu hard=%llu min_rel_dist=%.6e min_at=%08x audit_n=%llu audit_max_rel_err=%.6e double_bits=%d long_double_bits=%d\n",
            op, (unsigned long long)tot.swept, (unsigned long long)tot.special, (unsigned long long)tot.hard,
            tot.min_rel, tot.min_at, (unsigned long long)tot.audit_n, tot.audit_max, DBL_MANT_DIG, LDBL_MANT_DIG);
    fflush(fs); fflush(fh);
  }
  fclose(fh); fclose(fs);
  return 0;
}
