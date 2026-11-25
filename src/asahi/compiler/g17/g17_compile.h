/*
 * SPDX-License-Identifier: MIT
 *
 * G17 (Apple M5) machine-code emission from AGX IR.
 *
 * The whole AGX front-end (NIR -> AGX IR, optimization, scheduling, register
 * allocation) is shared; g17_pack_binary is the G17 analogue of
 * agx_pack_binary: it walks the final, post-RA AGX IR and emits G17 bytes using
 * the table-driven g17 machine-code layer (g17.h).
 *
 * Compute only, and a subset of it (see g17_compile.c); anything else stops
 * compilation with the opcode name.
 */
#pragma once

#include "agx_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

void g17_pack_binary(agx_context *ctx, struct util_dynarray *emission);

#ifdef __cplusplus
}
#endif
