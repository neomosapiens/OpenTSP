// Configuration validation (see tsp_config_check.h). A JSON Schema subset, enough for the
// schema tools/model_check.sh writes: type (object, array, integer, string), items, properties,
// required, additionalProperties: false, const, minimum, maximum, multipleOf, anyOf, allOf, plus two
// opentsp keywords: "x-opentsp-relations" (linear relations between two integer fields,
// which JSON Schema cannot express) and "x-opentsp-why" (the message for a failed anyOf).
// An unknown keyword in the schema is an error (exit 2 with "schema: ..."), so the checker
// can never silently ignore a constraint.
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sail.h"
#include "sail_config.h"
#include "cJSON.h"
#include "tsp_config_check.h"

extern void ztiming_config_error(sail_string *out, unit u);

static char msg[512];
static int schema_error;

#define FAIL(...) do { snprintf(msg, sizeof msg, __VA_ARGS__); return 0; } while (0)

// An integer value as Sail reads it (mpz_set_str on the literal text, base 10): an
// optional '-' and decimal digits only. 18 digits keep it inside int64_t.
static int int_value(const cJSON *v, long long *out) {
  if (!cJSON_IsNumber(v) || !v->valuestring) return 0;
  const char *s = v->valuestring;
  const char *d = *s == '-' ? s + 1 : s;
  size_t n = strlen(d);
  if (n == 0 || n > 18) return 0;
  for (size_t i = 0; i < n; i++) if (d[i] < '0' || d[i] > '9') return 0;
  errno = 0;
  *out = strtoll(s, NULL, 10);
  return errno == 0;
}

static const char *const KNOWN[] = {"$schema", "type", "properties", "items", "required", "additionalProperties",
  "const", "minimum", "maximum", "multipleOf", "anyOf", "allOf", "description", "default", "x-opentsp-relations", "x-opentsp-why", NULL};

static int check(const cJSON *sch, const cJSON *v, const char *key);

static int check_keywords(const cJSON *sch, const char *key) {
  for (const cJSON *k = sch->child; k; k = k->next) {
    int ok = 0;
    for (int i = 0; KNOWN[i]; i++) if (strcmp(k->string, KNOWN[i]) == 0) ok = 1;
    if (!ok) { schema_error = 1; FAIL("schema: unsupported keyword \"%s\" at %s", k->string, key); }
  }
  return 1;
}

static int check_object(const cJSON *sch, const cJSON *v, const char *key) {
  if (!cJSON_IsObject(v)) FAIL("%s: expected an object", *key ? key : "(top level)");
  const cJSON *props = cJSON_GetObjectItemCaseSensitive(sch, "properties");
  const cJSON *req = cJSON_GetObjectItemCaseSensitive(sch, "required");
  const cJSON *addl = cJSON_GetObjectItemCaseSensitive(sch, "additionalProperties");
  char sub[256];
  for (const cJSON *f = v->child; f; f = f->next) {
    for (const cJSON *g = v->child; g != f; g = g->next)
      if (strcmp(g->string, f->string) == 0) FAIL("%s%s%s: duplicate key", key, *key ? "." : "", f->string);
    const cJSON *ps = props ? cJSON_GetObjectItemCaseSensitive(props, f->string) : NULL;
    snprintf(sub, sizeof sub, "%s%s%s", key, *key ? "." : "", f->string);
    if (!ps) {
      if (addl && cJSON_IsFalse(addl)) FAIL("%s: unknown key", sub);
      continue;
    }
    if (!check(ps, f, sub)) return 0;
  }
  if (req) for (const cJSON *r = req->child; r; r = r->next) {
    if (!cJSON_IsString(r)) { schema_error = 1; FAIL("schema: bad required list at %s", key); }
    if (!cJSON_GetObjectItemCaseSensitive(v, r->valuestring)) FAIL("%s%s%s: missing", key, *key ? "." : "", r->valuestring);
  }
  return 1;
}

static int check(const cJSON *sch, const cJSON *v, const char *key) {
  if (!cJSON_IsObject(sch)) { schema_error = 1; FAIL("schema: not an object at %s", key); }
  if (!check_keywords(sch, key)) return 0;
  const char *k = *key ? key : "(top level)";
  const cJSON *type = cJSON_GetObjectItemCaseSensitive(sch, "type");
  if (type) {
    const char *t = cJSON_GetStringValue(type);
    long long x;
    if (!t) { schema_error = 1; FAIL("schema: bad type at %s", k); }
    if (strcmp(t, "object") == 0) { if (!check_object(sch, v, key)) return 0; }
    else if (strcmp(t, "array") == 0) {
      if (!cJSON_IsArray(v)) FAIL("%s: expected an array", k);
      const cJSON *items = cJSON_GetObjectItemCaseSensitive(sch, "items");
      char sub[256];
      int n = 0;
      if (items) for (const cJSON *e = v->child; e; e = e->next, n++) {
        snprintf(sub, sizeof sub, "%s[%d]", key, n);
        if (!check(items, e, sub)) return 0;
      }
    }
    else if (strcmp(t, "integer") == 0) {
      if (!int_value(v, &x)) FAIL("%s: expected an integer (decimal digits, at most 18)", k);
      const cJSON *mn = cJSON_GetObjectItemCaseSensitive(sch, "minimum");
      const cJSON *mx = cJSON_GetObjectItemCaseSensitive(sch, "maximum");
      const cJSON *cs = cJSON_GetObjectItemCaseSensitive(sch, "const");
      const cJSON *mo = cJSON_GetObjectItemCaseSensitive(sch, "multipleOf");
      long long b, lo = 0, hi = 0;
      if (mn && (!int_value(mn, &b) ? (schema_error = 1) : x < b)) {
        if (schema_error) FAIL("schema: bad minimum at %s", k);
        if (mx && int_value(mx, &hi)) FAIL("%s: %lld is below the minimum %lld (must be between %lld and %lld)", k, x, b, b, hi);
        FAIL("%s: %lld is below the minimum %lld", k, x, b);
      }
      if (mx && (!int_value(mx, &b) ? (schema_error = 1) : x > b)) {
        if (schema_error) FAIL("schema: bad maximum at %s", k);
        if (mn && int_value(mn, &lo)) FAIL("%s: %lld is above the maximum %lld (must be between %lld and %lld)", k, x, b, lo, b);
        FAIL("%s: %lld is above the maximum %lld", k, x, b);
      }
      if (cs && (!int_value(cs, &b) ? (schema_error = 1) : x != b)) {
        if (schema_error) FAIL("schema: bad const at %s", k);
        FAIL("%s: must be %lld", k, b);
      }
      if (mo && (!int_value(mo, &b) || b <= 0 ? (schema_error = 1) : x % b != 0)) {
        if (schema_error) FAIL("schema: bad multipleOf at %s", k);
        FAIL("%s: %lld is not a multiple of %lld", k, x, b);
      }
    } else if (strcmp(t, "string") == 0) {
      if (!cJSON_IsString(v)) FAIL("%s: expected a string", k);
      const cJSON *cs = cJSON_GetObjectItemCaseSensitive(sch, "const");
      if (cs && (!cJSON_IsString(cs) || strcmp(cs->valuestring, v->valuestring) != 0))
        FAIL("%s: must be \"%s\"", k, cJSON_IsString(cs) ? cs->valuestring : "?");
    } else if (strcmp(t, "boolean") == 0) {
      if (!cJSON_IsBool(v)) FAIL("%s: expected true or false", k);
    } else { schema_error = 1; FAIL("schema: unsupported type %s at %s", t, k); }
  }
  // An untyped schema with properties (an anyOf alternative) constrains an object's fields.
  if (!type && cJSON_GetObjectItemCaseSensitive(sch, "properties")) {
    if (!check_object(sch, v, key)) return 0;
  }
  const cJSON *all = cJSON_GetObjectItemCaseSensitive(sch, "allOf");
  if (all) for (const cJSON *s = all->child; s; s = s->next) if (!check(s, v, key)) return 0;
  const cJSON *any = cJSON_GetObjectItemCaseSensitive(sch, "anyOf");
  if (any) {
    char first[512] = "";
    int ok = 0;
    for (const cJSON *s = any->child; s && !ok; s = s->next) {
      if (check(s, v, key)) ok = 1;
      else if (schema_error) return 0;
      else if (!*first) snprintf(first, sizeof first, "%s", msg);
    }
    if (!ok) {
      const cJSON *why = cJSON_GetObjectItemCaseSensitive(sch, "x-opentsp-why");
      if (why && cJSON_IsString(why)) FAIL("%s: %s", k, why->valuestring);
      // Alternatives that are all integer constants: name the set.
      char list[512] = "";
      size_t used = 0;
      int consts = 1;
      for (const cJSON *s = any->child; s && consts; s = s->next) {
        const cJSON *c = cJSON_GetObjectItemCaseSensitive(s, "const");
        long long b;
        if (!c || !int_value(c, &b)) { consts = 0; break; }
        int w = snprintf(list + used, sizeof list - used, "%s%lld", used ? ", " : "", b);
        if (w < 0 || used + (size_t)w >= sizeof list) consts = 0; else used += (size_t)w;
      }
      if (consts && used) FAIL("%s: must be one of %s", k, list);
      FAIL("%s: none of the allowed alternatives (%s)", k, first);
    }
  }
  return 1;
}

static const cJSON *lookup(const cJSON *root, const char *dotted) {
  char buf[256];
  snprintf(buf, sizeof buf, "%s", dotted);
  const cJSON *v = root;
  for (char *save = NULL, *p = strtok_r(buf, ".", &save); p; p = strtok_r(NULL, ".", &save)) {
    if (!cJSON_IsObject(v)) return NULL;
    v = cJSON_GetObjectItemCaseSensitive(v, p);
  }
  return v;
}

// "x-opentsp-relations": [{"a": key, "op": "<"|"<="|"=="|"!=", "b": key, "offset": n,
// "quads": k (optional), "why": text}, ...]: value(a) op value(b) + offset + k * Q, with Q =
// geometry.mem_slices_per_hemisphere / 4, the number of MEM quads per hemisphere.
static int check_relations(const cJSON *sch, const cJSON *cfg) {
  const cJSON *rels = cJSON_GetObjectItemCaseSensitive(sch, "x-opentsp-relations");
  if (!rels) return 1;
  for (const cJSON *r = rels->child; r; r = r->next) {
    const char *a = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(r, "a"));
    const char *b = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(r, "b"));
    const char *op = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(r, "op"));
    const char *why = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(r, "why"));
    const cJSON *off = cJSON_GetObjectItemCaseSensitive(r, "offset");
    const cJSON *qd = cJSON_GetObjectItemCaseSensitive(r, "quads");
    long long x, y, o = 0, k = 0, ms;
    if (!a || !b || !op || !why || (off && !int_value(off, &o)) || (qd && !int_value(qd, &k))) { schema_error = 1; FAIL("schema: bad x-opentsp-relations entry"); }
    if (k) {
      if (!int_value(lookup(cfg, "geometry.mem_slices_per_hemisphere"), &ms)) { schema_error = 1; FAIL("schema: a quads relation needs geometry.mem_slices_per_hemisphere"); }
      o += k * (ms / 4);
    }
    if (!int_value(lookup(cfg, a), &x) || !int_value(lookup(cfg, b), &y)) { schema_error = 1; FAIL("schema: relation names a non-integer key %s / %s", a, b); }
    int ok = strcmp(op, "<") == 0 ? x < y + o : strcmp(op, "<=") == 0 ? x <= y + o
           : strcmp(op, "==") == 0 ? x == y + o : strcmp(op, "!=") == 0 ? x != y + o : -1;
    if (ok < 0) { schema_error = 1; FAIL("schema: bad relation operator %s", op); }
    if (!ok) {
      if (o) FAIL("%s: must be %s %s %+lld (%s = %lld, %s = %lld): %s", a, op, b, o, a, x, b, y, why);
      FAIL("%s: must be %s %s (%s = %lld, %s = %lld): %s", a, op, b, a, x, b, y, why);
    }
  }
  return 1;
}

// The strict JSON grammar of RFC 8259, checked on the raw text before cJSON (which also accepts
// leading zeros and any isspace() byte as whitespace). A \u0000 escape is rejected: cJSON hands
// keys and values to the checker as NUL-terminated strings, so text after it would be ignored.
static int ws(const char **p) { while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') (*p)++; return 1; }
static int hex4(const char *p, unsigned *u) {
  *u = 0;
  for (int i = 0; i < 4; i++) {
    char c = p[i];
    unsigned d = c >= '0' && c <= '9' ? (unsigned)(c - '0') : c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10)
               : c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 16u;
    if (d == 16) return 0;
    *u = *u << 4 | d;
  }
  return 1;
}
static int jstring(const char **p) {
  const unsigned char *s = (const unsigned char *)*p + 1;
  for (;;) {
    unsigned c = *s;
    if (c == '"') { *p = (const char *)s + 1; return 1; }
    if (c < 0x20) { *p = (const char *)s; return 0; }
    if (c == '\\') {
      unsigned u, l;
      c = s[1];
      if (c == 'u') {
        if (!hex4((const char *)s + 2, &u) || u == 0 || (u >= 0xDC00 && u <= 0xDFFF)) { *p = (const char *)s; return 0; }
        s += 6;
        if (u >= 0xD800 && u <= 0xDBFF) {
          if (s[0] != '\\' || s[1] != 'u' || !hex4((const char *)s + 2, &l) || l < 0xDC00 || l > 0xDFFF) { *p = (const char *)s; return 0; }
          s += 6;
        }
      } else if (c && strchr("\"\\/bfnrt", (int)c)) s += 2;
      else { *p = (const char *)s; return 0; }
    } else if (c < 0x80) s++;
    else {
      int n = c >= 0xC2 && c <= 0xDF ? 1 : c >= 0xE0 && c <= 0xEF ? 2 : c >= 0xF0 && c <= 0xF4 ? 3 : -1;
      if (n < 0) { *p = (const char *)s; return 0; }
      unsigned lo = 0x80, hi = 0xBF;
      if (c == 0xE0) lo = 0xA0; else if (c == 0xED) hi = 0x9F; else if (c == 0xF0) lo = 0x90; else if (c == 0xF4) hi = 0x8F;
      if (s[1] < lo || s[1] > hi) { *p = (const char *)s; return 0; }
      for (int i = 2; i <= n; i++) if (s[i] < 0x80 || s[i] > 0xBF) { *p = (const char *)s; return 0; }
      s += n + 1;
    }
  }
}
static int digits(const char **p) { if (**p < '0' || **p > '9') return 0; while (**p >= '0' && **p <= '9') (*p)++; return 1; }
static int jvalue(const char **p, int depth) {
  if (depth > 128) return 0;
  ws(p);
  const char *s = *p;
  if (*s == '{' || *s == '[') {
    char close = *s == '{' ? '}' : ']';
    (*p)++; ws(p);
    if (**p == close) { (*p)++; return 1; }
    for (;;) {
      if (close == '}') {
        if (**p != '"' || !jstring(p)) return 0;
        ws(p);
        if (**p != ':') return 0;
        (*p)++;
      }
      if (!jvalue(p, depth + 1)) return 0;
      ws(p);
      if (**p == close) { (*p)++; return 1; }
      if (**p != ',') return 0;
      (*p)++; ws(p);
    }
  }
  if (*s == '"') return jstring(p);
  if (strncmp(s, "true", 4) == 0) { *p += 4; return 1; }
  if (strncmp(s, "false", 5) == 0) { *p += 5; return 1; }
  if (strncmp(s, "null", 4) == 0) { *p += 4; return 1; }
  if (**p == '-') (*p)++;
  if (**p == '0') (*p)++;
  else if (!digits(p)) return 0;
  if (**p == '.') { (*p)++; if (!digits(p)) return 0; }
  if (**p == 'e' || **p == 'E') { (*p)++; if (**p == '+' || **p == '-') (*p)++; if (!digits(p)) return 0; }
  return 1;
}
const char *tsp_json_syntax_error(const char *text) {
  const char *p = text;
  if (!jvalue(&p, 0)) return p;
  ws(&p);
  return *p ? p : NULL;
}

// Default values: a schema property with a "default" may be left out of its section. The loader inserts the missing members into the
// text it hands to Sail, which reads every key. The text is checked first, so the scan below can rely on valid JSON.
// A section is a top-level member that is an object; a default is looked for in the section's properties only (a key at the top level
// or deeper than one section has none today). Every other byte of the file is passed on unchanged.
static const char *skip_ws(const char *p) { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++; return p; }

// The value of member `name` of the object whose '{' is at obj, or NULL. Returns the position of the first byte of the value.
static const char *member_value(const char *obj, const char *name) {
  const char *p = skip_ws(obj + 1);
  while (*p == '"') {
    const char *q = p;
    if (!jstring(&q)) return NULL;
    // Match the decoded JSON key, as schema validation and Sail's config reader
    // do. Keep raw value bytes/positions intact; reserializing the whole profile
    // through binary64 would lose precision in large integer fields.
    cJSON *key = cJSON_ParseWithLength(p, (size_t)(q - p));
    if (!key) { fprintf(stderr, "out of memory\n"); exit(2); }
    int hit = cJSON_IsString(key) && strcmp(key->valuestring, name) == 0;
    cJSON_Delete(key);
    p = skip_ws(q);
    if (*p != ':') return NULL;
    p = skip_ws(p + 1);
    if (hit) return p;
    if (!jvalue(&p, 1)) return NULL;
    p = skip_ws(p);
    if (*p != ',') return NULL;
    p = skip_ws(p + 1);
  }
  return NULL;
}

// The configuration text with every missing defaulted member inserted; the result is malloc'd (the same pointer when nothing is missing).
static char *with_defaults(char *json, const char *schema_json) {
  cJSON *sch = cJSON_Parse(schema_json);
  if (!sch) return json;
  const cJSON *props = cJSON_GetObjectItemCaseSensitive(sch, "properties");
  for (const cJSON *sec = props ? props->child : NULL; sec; sec = sec->next) {
    const cJSON *sp = cJSON_GetObjectItemCaseSensitive(sec, "properties");
    for (const cJSON *k = sp ? sp->child : NULL; k; k = k->next) {
      const cJSON *def = cJSON_GetObjectItemCaseSensitive(k, "default");
      long long dv;
      if (!def || !int_value(def, &dv)) continue;
      const char *top = skip_ws(json);
      const char *obj = *top == '{' ? member_value(top, sec->string) : NULL;
      if (!obj || *obj != '{' || member_value(obj, k->string)) continue;
      const char *after = skip_ws(obj + 1);
      char ins[160];
      int w = snprintf(ins, sizeof ins, "\"%s\": %lld%s", k->string, dv, *after == '}' ? "" : ", ");
      size_t head = (size_t)(obj + 1 - json), len = strlen(json);
      char *out = malloc(len + (size_t)w + 1);
      if (!out) { fprintf(stderr, "out of memory\n"); exit(2); }
      memcpy(out, json, head);
      memcpy(out + head, ins, (size_t)w);
      memcpy(out + head + w, json + head, len - head + 1);
      free(json);
      json = out;
    }
  }
  cJSON_Delete(sch);
  return json;
}

const char *tsp_config_check(const char *json, const char *schema_json) {
  static char out[600];
  schema_error = 0;
  const char *bad = tsp_json_syntax_error(json);
  if (bad) {
    snprintf(out, sizeof out, "not valid JSON (at byte %td)", bad - json);
    return out;
  }
  cJSON *sch = cJSON_Parse(schema_json);
  if (!sch) return "schema: not valid JSON";
  const char *end = NULL;
  cJSON *cfg = cJSON_ParseWithOpts(json, &end, 1);
  if (!cfg) {
    snprintf(out, sizeof out, "not valid JSON (at byte %td)", end ? end - json : (ptrdiff_t)0);
    cJSON_Delete(sch);
    return out;
  }
  int ok = check(sch, cfg, "") && check_relations(sch, cfg);
  cJSON_Delete(cfg);
  cJSON_Delete(sch);
  if (ok) return NULL;
  snprintf(out, sizeof out, "%s", msg);
  return out;
}

void tsp_config_load(const char *prog, const char *path, const char *schema_json) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "%s: config %s: cannot open: %s\n", prog, path, strerror(errno)); exit(2); }
  size_t cap = 4096, n = 0, k;
  char *buf = malloc(cap + 1);
  while (buf && (k = fread(buf + n, 1, cap - n, f)) > 0) {
    n += k;
    if (n == cap) { if (cap > (1u << 24)) { fprintf(stderr, "%s: config %s: file too large\n", prog, path); exit(2); } cap *= 2; buf = realloc(buf, cap + 1); }
  }
  if (!buf) { fprintf(stderr, "%s: out of memory\n", prog); exit(2); }
  if (ferror(f)) { fprintf(stderr, "%s: config %s: read error\n", prog, path); exit(2); }
  fclose(f);
  buf[n] = 0;
  if (memchr(buf, 0, n)) { fprintf(stderr, "%s: config %s: NUL byte in file\n", prog, path); exit(2); }
  const char *err = tsp_config_check(buf, schema_json);
  if (err) { fprintf(stderr, "%s: config %s: %s\n", prog, path, err); exit(2); }
  buf = with_defaults(buf, schema_json);
  sail_config_set_string(buf);
  free(buf);
}

void tsp_config_model_check(const char *prog, const char *path) {
  sail_string err;
  CREATE(sail_string)(&err);
  ztiming_config_error(&err, UNIT);
  if (*err) { fprintf(stderr, "%s: config %s: %s\n", prog, path, err); exit(2); }
  KILL(sail_string)(&err);
}
