/* Coverage recorder for a Sail C model built with `sail -c --c-coverage FILE`.
 *
 * The generated code calls sail_function_entry, sail_branch_reached and sail_branch_target_taken. This file
 * implements them without the Rust runtime of the Sail distribution (not installed here) and writes, at process
 * exit, one file `cov.<pid>` in the directory named by SAIL_COVERAGE_DIR, in the line format of the static file that
 * `--c-coverage` writes: `F id, "name", "file", l1, c1, l2, c2`, `B id, "file", ...` and `T branch, target, "file", ...`.
 * test/formal/coverage_report.py merges the files of many runs against the static file.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sail_coverage.h"

typedef struct {
  uint64_t key;     /* kind in bits 62..63 (1 F, 2 B, 3 T), ids below; 0 marks an empty slot */
  int a, b;
  const char *name, *file;
  int l1, c1, l2, c2;
} entry;

static entry *set;
static size_t capacity, used;
static int registered;

static entry *insert(uint64_t key) {
  if (used * 2 >= capacity) {
    size_t old_capacity = capacity;
    entry *old = set;
    capacity = capacity ? capacity * 2 : 1 << 14;
    set = calloc(capacity, sizeof(entry));
    if (set == NULL) abort();
    for (size_t i = 0; i < old_capacity; i++) {
      if (old[i].key == 0) continue;
      size_t j = (old[i].key * 0x9E3779B97F4A7C15ULL) >> 20;
      for (j %= capacity; set[j].key != 0; j = (j + 1) % capacity) {}
      set[j] = old[i];
    }
    free(old);
  }
  size_t j = ((key * 0x9E3779B97F4A7C15ULL) >> 20) % capacity;
  for (; set[j].key != 0; j = (j + 1) % capacity) {
    if (set[j].key == key) return NULL;     /* already recorded */
  }
  set[j].key = key;
  used++;
  return &set[j];
}

int sail_coverage_exit(void) {
  const char *dir = getenv("SAIL_COVERAGE_DIR");
  if (dir == NULL) return 0;
  char path[4096];
  snprintf(path, sizeof path, "%s/cov.%d", dir, (int)getpid());
  FILE *f = fopen(path, "w");
  if (f == NULL) return 1;
  for (size_t i = 0; i < capacity; i++) {
    const entry *e = &set[i];
    switch (e->key >> 62) {
      case 1: fprintf(f, "F %d, \"%s\", \"%s\", %d, %d, %d, %d\n", e->a, e->name, e->file, e->l1, e->c1, e->l2, e->c2); break;
      case 2: fprintf(f, "B %d, \"%s\", %d, %d, %d, %d\n", e->a, e->file, e->l1, e->c1, e->l2, e->c2); break;
      case 3: fprintf(f, "T %d, %d, \"%s\", %d, %d, %d, %d\n", e->a, e->b, e->file, e->l1, e->c1, e->l2, e->c2); break;
      default: break;
    }
  }
  return fclose(f) != 0;
}

static void at_exit(void) { (void)sail_coverage_exit(); }

static void record(uint64_t kind, int a, int b, const char *name, const char *file, int l1, int c1, int l2, int c2) {
  if (!registered) {
    registered = 1;
    atexit(at_exit);
  }
  entry *e = insert((kind << 62) | ((uint64_t)(uint32_t)a << 24) | ((uint64_t)(uint32_t)b & 0xFFFFFF) | (1ULL << 61));
  if (e == NULL) return;
  e->a = a; e->b = b; e->name = name; e->file = file; e->l1 = l1; e->c1 = c1; e->l2 = l2; e->c2 = c2;
}

/* The generated runtime's `-c FILE` option calls this; the recorder takes a directory (SAIL_COVERAGE_DIR) instead. */
void sail_set_coverage_file(const char *output_file) { (void)output_file; }

void sail_function_entry(int function_id, const char *function_name, const char *sail_file, int l1, int c1, int l2, int c2) {
  record(1, function_id, 0, function_name, sail_file, l1, c1, l2, c2);
}

void sail_branch_reached(int branch_id, const char *sail_file, int l1, int c1, int l2, int c2) {
  record(2, branch_id, 0, NULL, sail_file, l1, c1, l2, c2);
}

void sail_branch_target_taken(int branch_id, int branch_target_id, const char *sail_file, int l1, int c1, int l2, int c2) {
  record(3, branch_id, branch_target_id, NULL, sail_file, l1, c1, l2, c2);
}
