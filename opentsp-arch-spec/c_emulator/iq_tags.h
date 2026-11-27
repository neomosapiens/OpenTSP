// SPDX-License-Identifier: Apache-2.0
// Each occupied IQ slot records its source unit. Host text has tag -1.
// The harness and Sail unit tests share this backing store.
#pragma once
#include "tsp_model_if.h"
#include <stdlib.h>

static mpz_t *iq_tags[145];
static size_t iq_tag_capacity(void);

unit iq_tag_write(int64_t q, sail_int slot, sail_int value) {
  if (!iq_tags[q]) {
    size_t cap = iq_tag_capacity();
    iq_tags[q] = calloc(cap, sizeof(mpz_t));
    if (!iq_tags[q]) abort();
    for (size_t i = 0; i < cap; i++) mpz_init_set_si(iq_tags[q][i], -1);
  }
  mpz_set(iq_tags[q][mpz_get_ui(slot)], value);
  return UNIT;
}

void iq_tag_read(sail_int *out, int64_t q, sail_int slot) {
  if (iq_tags[q]) mpz_set(*out, iq_tags[q][mpz_get_ui(slot)]);
  else mpz_set_si(*out, -1);
}
