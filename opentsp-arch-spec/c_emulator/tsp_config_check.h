// Configuration validation for the harnesses around the Sail-generated model.
//
// The model's types (geometry.sail, layout.sail, timing.sail, c2c_insts.sail) assume the
// configuration satisfies config/schema.json, which tools/model_check.sh generates from
// those types plus the cross-field invariants Sail cannot emit (tools/schema_invariants.py).
// Sail's own config loader trusts the file: a value outside a type's range becomes C
// undefined behaviour or silently wrong semantics. tsp_config_load() therefore checks the
// file against the same schema, embedded at build time, before the model sees it, and
// then hands the checked text to Sail.
#pragma once

// Read, parse and validate the configuration file at path; on success load it into Sail's
// configuration (sail_config_set_string) and return. On any error print
// "<prog>: config <path>: <key>: <reason>" on stderr and exit with status 2.
// schema_json is the text of config/schema.json.
void tsp_config_load(const char *prog, const char *path, const char *schema_json);

// Strict JSON syntax shared by configuration and topology readers. The input is NUL-terminated;
// callers reading files reject embedded NUL bytes first. Decoded NUL strings cannot be represented
// by cJSON and are refused. Returns NULL on success, otherwise the first invalid byte.
const char *tsp_json_syntax_error(const char *json);

// The same check on a JSON text already in memory; returns NULL when it is valid, otherwise
// a static message "<key>: <reason>". Exposed for tests.
const char *tsp_config_check(const char *json, const char *schema_json);

// After model_init: the rules the model checks on the effective timing table
// (model/postlude/timing_check.sail timing_config_error: per-entry rows, d_skew < d_func, ...).
// On a violation print "<prog>: config <path>: <key>: <reason>" on stderr and exit with status 2.
void tsp_config_model_check(const char *prog, const char *path);
