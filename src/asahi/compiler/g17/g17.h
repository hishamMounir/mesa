/*
 * SPDX-License-Identifier: MIT
 *
 * G17 (Apple M5) machine-code layer: table-driven encode/decode and instruction
 * length. All encodings come from g17_isa.py via the generated g17_isa.{c,h}.
 */
#pragma once

#include "g17_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One instruction: op + RAW field values (the bits scattered into the encoding,
 * LSB first). Signed fields (the back-edge offset) hold the two's-complement
 * raw bits; use g17_get/g17_set for logical values. */
struct g17_instr {
   enum g17_op op;
   uint8_t length;
   uint32_t raw[G17_MAX_FIELDS];
};

/* Every field set to its default (0, or the template's bits for hints / fixed
 * immediates). */
void g17_instr_init(struct g17_instr *I, enum g17_op op);

/* Logical field access: sign-extends signed fields; g17_set asserts the value
 * fits. */
int64_t g17_get(const struct g17_instr *I, unsigned field);
void g17_set(struct g17_instr *I, unsigned field, int64_t value);

/* Encode into out[0..length); returns the length in bytes. */
unsigned g17_pack(const struct g17_instr *I, uint8_t *out);

/* Length of the instruction at code[0..avail) in bytes, 0 if unknown. */
unsigned g17_length(const uint8_t *code, size_t avail);

/* Decode one instruction. Returns false if the length is unknown. If the length
 * is known but no op matches, I->op = G17_OP_UNKNOWN and I->length is still set
 * (so a stream can be walked). */
bool g17_unpack(const uint8_t *code, size_t avail, struct g17_instr *I);

/* Loop back-edge offset: the back-edge at byte `fe_offset` jumps to
 * `target_offset` (the first instruction of the loop body); target = fe + 4 + 2
 * * offset, offset in halfwords. */
int32_t g17_back_edge_offset(size_t fe_offset, size_t target_offset);

#ifdef __cplusplus
}
#endif
