/*
 * SPDX-License-Identifier: MIT
 *
 * Tests g17_pack_binary: build a bit of post-RA AGX IR, emit G17 bytes, and
 * compare against the expected encoding, packed from g17_isa.py.
 */
#include "g17/g17_compile.h"
#include "util/u_dynarray.h"
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

static agx_index
reg32(unsigned n)
{
   return agx_register(n * 2, AGX_SIZE_32);
}

/* A register whose value dies here, which is what register allocation marks on
 * a last use. It matters for the compare family: cmp_reg / cmp_uni / cmp_eq
 * KILL their register sources, so the backend has to be told when an operand is
 * still needed afterwards (it then clears byte2 bit 4 for op2, and swaps or
 * copies for op1, which has no keep encoding). */
static agx_index
dead32(unsigned n)
{
   agx_index idx = reg32(n);
   idx.discard = true;
   return idx;
}

/* Build IR via `emit`, pack it for G17, and compare to the expected hex byte
 * string. */
#define G17_CASE(name, emit, expected_hex)                                     \
   TEST(G17Pack, name)                                                         \
   {                                                                           \
      void *ctx = ralloc_context(NULL);                                        \
      agx_builder *b = agx_test_builder(ctx);                                  \
      emit;                                                                    \
      struct util_dynarray emission;                                           \
      util_dynarray_init(&emission, ctx);                                      \
      g17_pack_binary(b->shader, &emission);                                   \
      const char *exp = expected_hex;                                          \
      ASSERT_EQ(emission.size * 2, strlen(exp));                               \
      char got[256] = {0};                                                     \
      for (unsigned i = 0; i < emission.size; ++i)                             \
         snprintf(got + 2 * i, sizeof(got) - 2 * i, "%02x",                    \
                  ((uint8_t *)emission.data)[i]);                              \
      EXPECT_STREQ(got, exp);                                                  \
      ralloc_free(ctx);                                                        \
   }

G17_CASE(mov_imm, agx_mov_imm_to(b, reg32(2), 0x1234), "2cb4020008080000")
G17_CASE(tid,
         agx_get_sr_to(b, reg32(1),
                       (enum agx_sr)80 /* thread_position_in_grid.x */),
         "1ca01006")
G17_CASE(imul,
         agx_imad_to(b, reg32(2), reg32(1), reg32(1), agx_immediate(0), 0),
         "2706041a2100a10a888000000102")
G17_CASE(iadd_reg, agx_iadd_to(b, reg32(3), reg32(2), reg32(1), 0),
         "3704043a2900a30a88800100")
G17_CASE(iadd_shift, agx_iadd_to(b, reg32(3), reg32(2), reg32(1), 2),
         "3704043a2900a30a88800500")
G17_CASE(iadd_shift3, agx_iadd_to(b, reg32(3), reg32(2), reg32(1), 3),
         "3704043a2900a30a88800001")
G17_CASE(iadd_imm, agx_iadd_to(b, reg32(2), reg32(1), agx_immediate(7), 0),
         "2703043a2500a30aa0800100")

/* imm + (reg << s): shadd_imm directly for s in 0..4. */
G17_CASE(iadd_imm_shift2,
         agx_iadd_to(b, reg32(2), agx_immediate(5), reg32(1), 2),
         "2705043a2d00a30aa0800500")
G17_CASE(iadd_imm_shift3,
         agx_iadd_to(b, reg32(2), agx_immediate(5), reg32(1), 3),
         "2705043a2d00a30aa0800001")
G17_CASE(iadd_imm_shift3_dead,
         agx_iadd_to(b, reg32(2), agx_immediate(5), dead32(1), 3),
         "2705043a2d00a30aa0802001")
G17_CASE(iadd_imm_shift3_alias,
         agx_iadd_to(b, reg32(1), agx_immediate(5), dead32(1), 3),
         "3705043a2d00a302a0802001")

/* Shifts and register-register bitwise ops. The value's last-use hint is b4.2
 * in the shift family. ushr = bfeil(0, x, s, mask 0); register ishl = bfi(0, x,
 * s, mask 0). */
G17_CASE(ushr_imm,
         agx_bfeil_to(b, reg32(2), agx_zero(), reg32(1), agx_immediate(3), 0),
         "2702001a2200a10af88000340000")
G17_CASE(ushr_imm_dead,
         agx_bfeil_to(b, reg32(2), agx_zero(), dead32(1), agx_immediate(16), 0),
         "2702001a2600a10a388400340000")
G17_CASE(ushr_imm_mod32,
         agx_bfeil_to(b, reg32(2), agx_zero(), reg32(1), agx_immediate(35), 0),
         "2702001a2200a10af88000340000")
G17_CASE(ushr_zero_is_move,
         agx_bfeil_to(b, reg32(2), agx_zero(), reg32(1), agx_immediate(0), 0),
         "2704041a2d00a30aa0800100")
G17_CASE(ushr_reg,
         agx_bfeil_to(b, reg32(2), agx_zero(), reg32(1), dead32(3), 0),
         "2706001a2200a10a888120340000")
G17_CASE(asr_imm, agx_asr_to(b, reg32(4), dead32(5), agx_immediate(17)),
         "2702005a2600a31278844000")
G17_CASE(asr_reg, agx_asr_to(b, reg32(4), reg32(5), reg32(6)),
         "2706005a2200a31208034000")
G17_CASE(ishl_reg, agx_bfi_to(b, reg32(2), agx_zero(), reg32(1), reg32(7), 0),
         "2706001a2200a00a888300340000")
G17_CASE(and_reg, agx_and_to(b, reg32(3), dead32(1), reg32(2)),
         "330207803030a00a0481")
G17_CASE(or_reg, agx_or_to(b, reg32(3), reg32(1), reg32(9)),
         "330207803238a00a8084")
G17_CASE(xor_reg, agx_xor_to(b, reg32(0), reg32(2), reg32(0)),
         "230006803278a0020080")

/* Bitfield extract and the general truth table. ubitfield_extract is bfeil(0,
 * x, s, width): lsr with a width, even at offset 0. bw_reg takes AGX's truth
 * table as is, with last-use hints for both operands (b's is b6.2); not(x) is
 * bitop(x, 0, 0x5), emitted as nor(x, x). */
G17_CASE(bfe_imm,
         agx_bfeil_to(b, reg32(2), agx_zero(), reg32(1), agx_immediate(5), 8),
         "2702001a2200a10a788100340080")
G17_CASE(bfe_offset0,
         agx_bfeil_to(b, reg32(2), agx_zero(), reg32(1), agx_immediate(0), 13),
         "2702001a2200a10a3880003400d0")
G17_CASE(bfe_reg,
         agx_bfeil_to(b, reg32(2), agx_zero(), dead32(1), reg32(3), 16),
         "2706001a2600a10a888100340100")
G17_CASE(not_reg, agx_bitop_to(b, reg32(4), reg32(5), agx_zero(), 0x5),
         "2302068031b0a0128082")
G17_CASE(andn_reg, agx_andn2_to(b, reg32(3), dead32(1), reg32(2)),
         "330206803230a00a0481")
G17_CASE(nand_reg, agx_bitop_to(b, reg32(3), reg32(6), reg32(7), 0x7),
         "3300068033f8a00a8083")
G17_CASE(xnor_reg, agx_bitop_to(b, reg32(0), reg32(1), reg32(9), 0x9),
         "230207803130a0028084")
G17_CASE(xor_reg_b_dead, agx_xor_to(b, reg32(3), reg32(1), dead32(2)),
         "330206803238a40a0081")
G17_CASE(not_reg_dead, agx_bitop_to(b, reg32(4), dead32(5), agx_zero(), 0x5),
         "2302068031b0a4128482")

/* Shift left by an immediate: shl_imm, one instruction, no scratch register. */
G17_CASE(shl_imm,
         agx_bfi_to(b, reg32(2), agx_zero(), reg32(1), agx_immediate(24), 0),
         "2702001a2200a00a388600340000")

/* Through a scratch register above the allocation (max_reg = 40 -> r21, r22):
 * a constant shifted by a register (bitfield insert's mask << offset), an
 * immediate multiplicand (x << k as x * 2^k), and a 16-bit bitop operand. */
G17_CASE(const_shl_reg,
         (b->shader->max_reg = 40,
          agx_bfi_to(b, reg32(2), agx_zero(), agx_immediate(255), dead32(1),
                     0)),
         "5cff4200020000002706005a2280a00a888020340000")
G17_CASE(imad_imm_multiplicand,
         (b->shader->max_reg = 40,
          agx_imad_to(b, reg32(3), dead32(3), agx_immediate(32),
                      agx_immediate(160), 0)),
         "5ca04200000000002706045a2180a13a8881200001023700041a2580a30a228b0100")
G17_CASE(or_16bit_operand,
         (b->shader->max_reg = 40,
          agx_or_to(b, reg32(1), reg32(1), agx_register(4, AGX_SIZE_16))),
         "3704041a2d00a33220010100330207803238a002808a")
/* A 16-bit source of a 32-bit add sign-extends in AGX: the high half becomes an
 * arithmetic shift by 16. */
G17_CASE(sext_high_half,
         agx_iadd_to(b, reg32(2), agx_register(7, AGX_SIZE_16),
                     agx_immediate(0), 0),
         "2702003a2200a30a38844000")
G17_CASE(fadd, agx_fadd_to(b, reg32(4), reg32(5), reg32(6)), "490b000d")
G17_CASE(fmul, agx_fmul_to(b, reg32(4), reg32(5), reg32(6)), "490b010d")
G17_CASE(bitop_and,
         agx_bitop_to(b, reg32(2), reg32(1), agx_immediate(0x3f), 0x8),
         "230607803030a30ac08f")
G17_CASE(bitop_xor,
         agx_bitop_to(b, reg32(2), reg32(1), agx_immediate(0x55), 0x6),
         "230606803238a30a4095")
G17_CASE(bitop_copy, agx_bitop_to(b, reg32(2), reg32(1), agx_zero(), 0xa),
         "2704041a2d00a30aa0800100")
G17_CASE(mov_reg, agx_mov_to(b, reg32(2), reg32(1)), "2704041a2d00a30aa0800100")
G17_CASE(stop, agx_stop(b), "0e000000")

/* Device memory, uniform-base form (what agx_nir_lower_address produces): store
 * r5 -> u0:u1 + 4*r1. */
static agx_index
u64(unsigned n)
{
   return agx_uniform(n, AGX_SIZE_64);
}
G17_CASE(device_store,
         agx_device_store(b, reg32(5), u64(0), reg32(1), AGX_FORMAT_I32, 0x1, 0,
                          false),
         "5f00030201061040")
G17_CASE(device_load,
         agx_device_load_to(b, reg32(5), u64(0), reg32(1), AGX_FORMAT_I32, 0x1,
                            0, false),
         "5f00030218021040")

/* Load, then an AGX wait, then consume: the wait lands on the consumer (shadd
 * b0.3) and the store's extended form. r5 = load(u0:u1 + 4*r1); wait; r6 = r5 +
 * r5; store r6 -> u0:u1 + 4*r0. */
G17_CASE(
   load_wait_consume,
   {
      agx_device_load_to(b, reg32(5), u64(0), reg32(1), AGX_FORMAT_I32, 0x1, 0,
                         false);
      agx_wait(b, 0);
      agx_iadd_to(b, reg32(6), reg32(5), reg32(5), 0);
      agx_device_store(b, reg32(6), u64(0), reg32(0), AGX_FORMAT_I32, 0x1, 0,
                       false);
   },
   "5f000302180210402f06045a2900a31a888201006f00030001061040")
G17_CASE(
   load_wait_store,
   {
      agx_device_load_to(b, reg32(5), u64(0), reg32(1), AGX_FORMAT_I32, 0x1, 0,
                         false);
      agx_wait(b, 0);
      agx_device_store(b, reg32(5), u64(0), reg32(0), AGX_FORMAT_I32, 0x1, 0,
                       false);
   },
   "5f000302180210405f000300010610c00020")
/* A float consumer has no wait bit, so a 27-family identity shim (on the
 * consumer's dest, carrying the wait) is inserted before it: load r5; wait;
 * [shim r6=r6+0 wait] fmul r6=r5*r5; store r6. */
G17_CASE(
   load_wait_float,
   {
      agx_device_load_to(b, reg32(5), u64(0), reg32(1), AGX_FORMAT_I32, 0x1, 0,
                         false);
      agx_wait(b, 0);
      agx_fmul_to(b, reg32(6), reg32(5), reg32(5));
      agx_device_store(b, reg32(6), u64(0), reg32(0), AGX_FORMAT_I32, 0x1, 0,
                       false);
   },
   "5f000302180210402f04041a2d00a31a20830100690b010b6f00030001061040")

/* A small straight-line kernel: r1 = tid; r2 = r1 * r1; r2 = r2 + 7; stop. */
G17_CASE(
   program,
   {
      agx_get_sr_to(b, reg32(1), (enum agx_sr)80);
      agx_imad_to(b, reg32(2), reg32(1), reg32(1), agx_immediate(0), 0);
      agx_iadd_to(b, reg32(2), reg32(2), agx_immediate(7), 0);
      agx_stop(b);
   },
   "1ca010062706041a2100a10a8880000001022703043a2500a30a208101000e000000")

/* icmpsel / fcmpsel: dest = (a cond b) ? c : d, lowered to an if/else --
 * compare on p1, branch-if-false, then-arm move, jump, else-arm move, endif.
 * The predicate is p1, not p0, because loops run on p0 and an inner compare
 * that reuses the loop's predicate hangs the GPU.
 *
 * Forward branches carry no offset -- the hardware scans for the matching
 * marker -- so there is no fixup pass and the byte sequence here is complete
 * and position-independent. The compare's op1_32 / op2_32 bits are SET: they
 * are operand WIDTH bits, and clearing one makes the hardware compare only that
 * register's low 16 bits. */
G17_CASE(
   icmpsel_imm,
   agx_icmpsel_to(b, reg32(2), dead32(4), agx_immediate(5), reg32(6), reg32(7),
                  AGX_ICOND_ULT),
   "2a09228415221e10000e2704041a2d00a30a208301005e63400e2704041a2d00a30aa08301003e03400e")
G17_CASE(
   icmpsel_reg,
   agx_icmpsel_to(b, reg32(2), dead32(4), dead32(5), reg32(6), reg32(7),
                  AGX_ICOND_UGT),
   "2a093a0b04001e10000e2704041a2d00a30a208301005e63400e2704041a2d00a30aa08301003e03400e")
G17_CASE(
   icmpsel_eq,
   agx_icmpsel_to(b, reg32(2), dead32(4), dead32(5), reg32(6), reg32(7),
                  AGX_ICOND_UEQ),
   "2a093b0b0600070000001e10000e2704041a2d00a30a208301005e63400e2704041a2d00a30aa08301003e03400e")
/* Immediate arms go through mov32 rather than the register copy shim. */
G17_CASE(icmpsel_imm_arms,
         agx_icmpsel_to(b, reg32(2), dead32(4), agx_immediate(5),
                        agx_immediate(1), agx_immediate(0), AGX_ICOND_ULT),
         "2a09228415221e10000e2c810200000000005e63400e2c800200000000003e03400e")
/* op2 still live after the compare (it is also the then-arm), so byte2 bit 4
 * must be CLEARED to stop the compare killing it -- 3a becomes 2a. Leaving it
 * set is silently wrong: the arm would move a dead register. op1 is dead here,
 * so it needs no swap or copy. */
G17_CASE(
   icmpsel_live_op2,
   agx_icmpsel_to(b, reg32(2), dead32(4), reg32(5), reg32(5), reg32(7),
                  AGX_ICOND_UGT),
   "2a092a0b04001e10000e2704041a2d00a30aa08201005e63400e2704041a2d00a30aa08301003e03400e")

/* Float >= has no strict G17 compare, so it is emitted as the strict < with the
 * ARMS SWAPPED (the then-arm now moves r7 and the else-arm r6) rather than by
 * flipping the branch polarity. */
G17_CASE(
   fcmpsel_ge,
   agx_fcmpsel_to(b, reg32(2), dead32(4), dead32(5), reg32(6), reg32(7),
                  AGX_FCOND_GE),
   "2a093a0b03001e10000e2704041a2d00a30aa08301005e63400e2704041a2d00a30a208301003e03400e")
