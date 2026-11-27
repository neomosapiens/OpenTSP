// Interface between the Sail-generated model and its harness (sail-riscv:
// c_emulator/riscv_platform_if.h). Included into the generated C via --c-include.
#pragma once
#include <stdbool.h>
#include <stdint.h>
#include "sail.h"

// Program: the instruction queues. The harness keeps the units of each queue's IQ (a ring
// of icu.iq_depth / 4 units per queue; queue 144 = control channel) while the model keeps the
// head and the length (model/slices/cu/cu_iq.sail), and the initial image of each queue (the
// host's program load): its length in units, padding already dropped, and unit i.
unit iq_unit_write(int64_t queue, sail_int slot, uint64_t unit_value);
uint64_t iq_unit_read(int64_t queue, sail_int slot);
unit iq_tag_write(int64_t queue, sail_int slot, sail_int tag);   // the text-word tag of a slot (ECC on instruction text)
void iq_tag_read(sail_int *out, int64_t queue, sail_int slot);
int64_t iq_image_len(int64_t queue);
uint64_t iq_image_unit(int64_t queue, sail_int index);

// SRAM, one stream value (8 * vlanes bits) per word.
void read_sram_word(lbits *out, int64_t slice, sail_int index);
unit write_sram_word(int64_t slice, sail_int index, lbits value);

// The ECC overlay of a stored word (model/core/ecc.sail): 10 bits per superlane (check-bit
// deviation 9, spare bit 1), all zero for a word whose check bits are valid.
void read_sram_ecc(lbits *out, int64_t slice, sail_int index);
unit write_sram_ecc(int64_t slice, sail_int index, lbits ov);

// Trace callbacks.
unit srf_write_callback(sail_int cycle, bool row_w, int64_t stream, int64_t gap, lbits value);
unit sram_write_callback(sail_int cycle, int64_t slice, uint64_t addr, lbits value);
// A spare-bit error report seen by its consumer (model/core/ecc.sail ecc_check_value).
unit scrub_event_callback(sail_int cycle, int64_t slice, sail_int addr, int64_t superlane, bool unc);
unit spare_event_callback(sail_int cycle, int64_t queue, bool row_w, int64_t id, int64_t gap, int64_t superlane, int64_t s03);
// An ECC event (model/core/ecc.sail): kind 0 a stream operand (id = stream), 1 an SRAM word (id = slice, addr).
unit mxm_corr_callback(sail_int cycle, int64_t queue, int64_t plane, sail_int lane);
unit ecc_event_callback(sail_int cycle, int64_t queue, int64_t kind, bool uncorrectable, bool row_w, int64_t id, int64_t addr, int64_t superlane, int64_t pos);
unit observe_site(const_sail_string event, sail_int c, int64_t q, sail_int head, sail_int count, sail_int detail);
unit diag_callback(sail_int cycle, int64_t queue, const_sail_string code);
unit lint_callback(sail_int cycle, int64_t queue, const_sail_string code);   // simulator lints, non-normative (core/lints.sail)

// Chip-to-chip links (model/core/callbacks.sail). side 0 = west, 1 = east.
int64_t c2c_peer(int64_t link);
bool c2c_packet_busy(sail_int cycle, int64_t link);
unit c2c_send(sail_int sent, int64_t link, int64_t ptype, lbits v, int64_t poison);
bool c2c_recv_ready(sail_int cycle, int64_t link);
void c2c_recv(lbits *out, sail_int cycle, int64_t link);
int64_t c2c_recv_poison(sail_int cycle, int64_t link);
int64_t c2c_recv_bad(int64_t link);              // link FEC: the superlanes of the vector just taken that were uncorrectable (bit g)
void c2c_next_arrival(sail_int *out, int64_t link);   // -1: no vector known (DeskewRecv wake-up, model_poll_event)

// The host interface (model/core/callbacks.sail, model/core/csr.sail).
int64_t host_chip_id(unit u);
bool host_recv_ready(sail_int cycle);
void host_recv(lbits *out, sail_int cycle);
unit host_send(sail_int cycle, lbits v);
unit host_irq_callback(sail_int cycle, int64_t vector);
