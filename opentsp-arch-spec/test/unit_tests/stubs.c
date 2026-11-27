// No-op platform for the Sail unit tests: empty program, zero SRAM, silent callbacks.
#include "tsp_model_if.h"

// Lanes per stream value, read from the model's config at first use.
extern int64_t zvlanes;

// The instruction queues: units kept in plain arrays (4096 bytes at most per queue), no
// initial images.
#include <stdint.h>
static uint32_t iq_ring[145][1024];
unit iq_unit_write(int64_t q, sail_int slot, uint64_t v) { iq_ring[q][mpz_get_ui(slot) % 1024] = (uint32_t)v; return UNIT; }
uint64_t iq_unit_read(int64_t q, sail_int slot) { return iq_ring[q][mpz_get_ui(slot) % 1024]; }
#include "iq_tags.h"
static size_t iq_tag_capacity(void) { return sizeof iq_ring[0] / sizeof iq_ring[0][0]; }
int64_t iq_image_len(int64_t q) { (void)q; return 0; }
uint64_t iq_image_unit(int64_t q, sail_int i) { (void)q; (void)i; return 0; }
void read_sram_word(lbits *out, int64_t s, sail_int i) {
  (void)s; (void)i;
  out->len = 8 * (mp_bitcnt_t)zvlanes;   // a zero word of the profile's width
  mpz_set_ui(*out->bits, 0);
}
unit write_sram_word(int64_t s, sail_int i, lbits v) { (void)s; (void)i; (void)v; return UNIT; }
void read_sram_ecc(lbits *out, int64_t s, sail_int i) {
  (void)s; (void)i;
  out->len = 10 * (mp_bitcnt_t)(zvlanes / 16);   // an all-valid overlay
  mpz_set_ui(*out->bits, 0);
}
unit write_sram_ecc(int64_t s, sail_int i, lbits v) { (void)s; (void)i; (void)v; return UNIT; }
unit scrub_event_callback(sail_int c, int64_t s, sail_int a, int64_t sl, bool unc) { (void)c; (void)s; (void)a; (void)sl; (void)unc; return UNIT; }
unit spare_event_callback(sail_int c, int64_t q, bool w, int64_t id, int64_t gap, int64_t g, int64_t s03) { (void)c; (void)q; (void)w; (void)id; (void)gap; (void)g; (void)s03; return UNIT; }
unit mxm_corr_callback(sail_int c, int64_t q, int64_t plane, sail_int lane) { (void)c; (void)q; (void)plane; (void)lane; return UNIT; }
unit ecc_event_callback(sail_int c, int64_t q, int64_t kind, bool unc, bool w, int64_t id, int64_t a, int64_t g, int64_t pos) { (void)c; (void)q; (void)kind; (void)unc; (void)w; (void)id; (void)a; (void)g; (void)pos; return UNIT; }
unit srf_write_callback(sail_int c, bool w, int64_t s, int64_t g, lbits v) { (void)c; (void)w; (void)s; (void)g; (void)v; return UNIT; }
unit sram_write_callback(sail_int c, int64_t s, uint64_t a, lbits v) { (void)c; (void)s; (void)a; (void)v; return UNIT; }
unit diag_callback(sail_int c, int64_t q, const_sail_string d) { (void)c; (void)q; (void)d; return UNIT; }
unit observe_site(const_sail_string event, sail_int c, int64_t q, sail_int head, sail_int count, sail_int detail) { return UNIT; }
unit lint_callback(sail_int c, int64_t q, const_sail_string d) { (void)c; (void)q; (void)d; return UNIT; }
// One chip (id 0), both links looped back, nothing ever arrives.
int64_t c2c_peer(int64_t side) { (void)side; return 0; }
bool c2c_packet_busy(sail_int c, int64_t l) { (void)c; (void)l; return false; }
unit c2c_send(sail_int arrival, int64_t side, int64_t ptype, lbits v, int64_t poison) { (void)arrival; (void)side; (void)ptype; (void)v; (void)poison; return UNIT; }
int64_t c2c_recv_poison(sail_int c, int64_t side) { (void)c; (void)side; return 0; }
bool c2c_recv_ready(sail_int c, int64_t side) { (void)c; (void)side; return false; }
void c2c_next_arrival(sail_int *out, int64_t side) { (void)side; mpz_set_si(*out, -1); }   // no vector is known
void c2c_recv(lbits *out, sail_int c, int64_t side) {
  (void)c; (void)side;
  out->len = 8 * (mp_bitcnt_t)zvlanes;
  mpz_set_ui(*out->bits, 0);
}
int64_t c2c_recv_bad(int64_t link) { (void)link; return 0; }
// The host interface: chip 0, nothing arrives, nothing is kept.
int64_t host_chip_id(unit u) { (void)u; return 0; }
bool host_recv_ready(sail_int c) { (void)c; return false; }
void host_recv(lbits *out, sail_int c) {
  (void)c;
  out->len = 8 * (mp_bitcnt_t)zvlanes;
  mpz_set_ui(*out->bits, 0);
}
unit host_send(sail_int c, lbits v) { (void)c; (void)v; return UNIT; }
unit host_irq_callback(sail_int c, int64_t n) { (void)c; (void)n; return UNIT; }
