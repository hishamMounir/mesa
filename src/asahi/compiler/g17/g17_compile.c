/*
 * SPDX-License-Identifier: MIT
 *
 * G17 (Apple M5) machine-code emission from AGX IR (g17_pack_binary).
 *
 * This is the G17 analogue of agx_pack_binary. The front-end up to and
 * including register allocation is shared with G13; here we walk the final AGX
 * IR and emit G17 bytes with the table-driven g17 layer.
 *
 * Compute only, and a subset of it: moves, thread_position_in_grid.x, the
 * integer and float ALU forms in g17_isa.py, shifts and bitfield extract,
 * device load/store, selects (lowered to an if/else) and single-level loops.
 * Anything else stops compilation in g17_unsupported() with the opcode name.
 */
#include "g17_compile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "g17/g17.h"
#include "util/half_float.h"
#include "util/u_dynarray.h"

/* Per-shader emission state. */
struct g17_state {
   agx_context *ctx;
   struct util_dynarray *emission;

   /* G17 loads are asynchronous and the wait is on the CONSUMER, not a
    * standalone instruction (AGX uses a `wait` op + scoreboard slots).
    * agx_insert_waits places each AGX wait immediately before its consumer, so
    * a wait becomes "the next instruction must wait": one G17 wait bit drains
    * all outstanding loads. Ops that carry a wait bit consume it directly --
    * device load/store (their 10-byte extended form) and the 27-family ALU
    * (b0.3); any other consumer gets a 27-family identity shim carrying the
    * wait. Reset per block. */
   bool pending_wait;

   /* A loop's condition and its back-edge are separate AGX instructions
    * (while_icmp, then jmp_exec_any), but G17 puts the inversion on the
    * BACK-EDGE's polarity bit, because it has no >= or <= compare to invert
    * into. So while_icmp stashes it here for the jmp_exec_any that closes the
    * loop. -1 means "no loop condition is pending", which makes a stray
    * back-edge a hard error. */
   int loop_invert;

   /* The predicate that pending loop condition wrote. jmp_exec_any carries only
    * a target -- no nesting level -- so it cannot re-derive it. */
   unsigned loop_pred;
};

/* A coverage gap: an op (or op shape) we don't translate yet. */
NORETURN static void
g17_unsupported(const agx_instr *I, const char *why)
{
   fprintf(stderr, "g17_pack_binary: unsupported %s (%s): ",
           agx_opcodes_info[I->op].name, why);
   agx_print_instr(I, stderr);
   abort();
}

/* A 32-bit GPR number (r0..r63) from a post-RA operand. AGX register indices
 * are in 16-bit units (r1 == 2); G17's register fields take the 32-bit register
 * number. */
static unsigned
g17_gpr(const agx_instr *I, agx_index idx)
{
   if (idx.size != AGX_SIZE_32)
      g17_unsupported(I, "non-32-bit operand");

   unsigned halfword = (idx.type == AGX_INDEX_REGISTER)
                          ? idx.value
                          : (idx.has_reg ? idx.reg : ~0u);
   if (halfword == ~0u || (idx.type != AGX_INDEX_REGISTER && !idx.has_reg))
      g17_unsupported(I, "operand is not a register");
   if (idx.abs || idx.neg) {
      /* The 09-family float forms have no modifier bits, so abs and neg have to
       * be expressed as instructions. `neg` on a float add becomes G17's own
       * fsub (handled in AGX_OPCODE_FADD); `abs` has no cheap equivalent --
       * clearing the sign bit needs a 32-bit mask and G17's bitwise immediate
       * is 8 bits, so it would take max(x, -x) or a materialised mask, neither
       * wired up. */
      g17_unsupported(
         I, idx.abs ? "abs source modifier (no G17 encoding; needs lowering)"
                    : "neg source modifier (no G17 encoding; needs lowering)");
   }
   if ((halfword & 1) || (halfword / 2) >= 64)
      g17_unsupported(I, "register out of range");

   return halfword / 2;
}

/* The 32-bit uniform WORD index (u0..u63) of a uniform operand. AGX counts
 * uniforms in 16-bit halves, like registers, so the word index is value / 2 --
 * the same conversion the memory base uses, and what agx_print shows (it prints
 * value >> 1). Only some fields can hold a uniform: the 27-family's A operand
 * (via its a_uniform kind bit) and the compare family's op2 (the cmp_uni form).
 */
static unsigned
g17_uniform_word(const agx_instr *I, agx_index idx)
{
   if (idx.type != AGX_INDEX_UNIFORM || idx.size != AGX_SIZE_32)
      g17_unsupported(I, "expected a 32-bit uniform operand");
   if (idx.value & 1)
      g17_unsupported(I, "misaligned uniform operand");
   if ((idx.value / 2) >= 64)
      g17_unsupported(I, "uniform operand out of range");
   return idx.value / 2;
}

/* A 16-bit operand: returns the 32-bit register it lives in and sets *half to
 * which one. AGX numbers registers in 16-bit units, so a 16-bit operand at
 * halfword h is the low (h even) or high (h odd) half of register h/2. */
static unsigned
g17_gpr16(const agx_instr *I, agx_index idx, unsigned *half)
{
   if (idx.size != AGX_SIZE_16)
      g17_unsupported(I, "expected a 16-bit operand");

   unsigned halfword = (idx.type == AGX_INDEX_REGISTER)
                          ? idx.value
                          : (idx.has_reg ? idx.reg : ~0u);
   if (halfword == ~0u || (idx.type != AGX_INDEX_REGISTER && !idx.has_reg))
      g17_unsupported(I, "16-bit operand is not a register");
   if (idx.abs || idx.neg)
      g17_unsupported(I, "source modifier on a 16-bit operand");
   if ((halfword / 2) >= 64)
      g17_unsupported(I, "16-bit register out of range");

   *half = halfword & 1;
   return halfword / 2;
}

/* Configure a 27-family op's destination and A-operand width/half from the AGX
 * operand sizes. The four bits are GPU-verified on this layout (see
 * g17_isa.py): clear selects 16-bit, and the half bit then picks which one. A
 * 32-bit destination with a 16-bit source zero-extends. */
static void
g17_set_w16(struct g17_instr *g, const agx_instr *I, agx_index dest,
            agx_index src, unsigned f_d, unsigned f_d32, unsigned f_dhalf,
            unsigned f_a, unsigned f_a32, unsigned f_ahalf)
{
   unsigned half;

   if (dest.size == AGX_SIZE_16) {
      g17_set(g, f_d, g17_gpr16(I, dest, &half));
      g17_set(g, f_d32, 0);
      g17_set(g, f_dhalf, half);
   } else {
      g17_set(g, f_d, g17_gpr(I, dest));
      g17_set(g, f_d32, 1);
   }

   if (src.size == AGX_SIZE_16) {
      g17_set(g, f_a, g17_gpr16(I, src, &half));
      g17_set(g, f_a32, 0);
      g17_set(g, f_ahalf, half);
   } else {
      g17_set(g, f_a, g17_gpr(I, src));
      g17_set(g, f_a32, 1);
   }
}

/* Configure a 09-family float op for fp16 or f32 from the AGX operand sizes.
 * The width and half bits are GPU-verified (see g17_isa.py): byte0 bit 0 clear
 * selects fp16, and each of the destination and the two operands then carries
 * its own half select, with the 6-bit fields still naming the 32-bit register.
 */
static void
g17_set_f16(struct g17_instr *g, const agx_instr *I, agx_index dest,
            agx_index a, agx_index b, unsigned f_d, unsigned f_a, unsigned f_b,
            unsigned f_32, unsigned f_dhalf, unsigned f_ahalf, unsigned f_bhalf)
{
   bool is16 = dest.size == AGX_SIZE_16;
   unsigned half;

   if (is16 != (a.size == AGX_SIZE_16) || is16 != (b.size == AGX_SIZE_16))
      g17_unsupported(I, "mixed 16- and 32-bit float operands");

   if (!is16) {
      g17_set(g, f_32, 1);
      g17_set(g, f_d, g17_gpr(I, dest));
      g17_set(g, f_a, g17_gpr(I, a));
      g17_set(g, f_b, g17_gpr(I, b));
      return;
   }

   unsigned d = g17_gpr16(I, dest, &half);
   /* byte0 would come out 0x00, which the length model reads as the 1-byte nop,
    * so the instruction could not be parsed back. See the caveat in g17_isa.py.
    */
   if (half == 0 && (d & 0xF) == 0)
      g17_unsupported(
         I, "fp16 destination is the low half of r0/r16/r32/r48 (byte0 = 0)");

   g17_set(g, f_32, 0);
   g17_set(g, f_d, d);
   g17_set(g, f_dhalf, half);
   g17_set(g, f_a, g17_gpr16(I, a, &half));
   g17_set(g, f_ahalf, half);
   g17_set(g, f_b, g17_gpr16(I, b, &half));
   g17_set(g, f_bhalf, half);
}

static bool
g17_is_uniform(agx_index idx)
{
   return idx.type == AGX_INDEX_UNIFORM;
}

static bool
g17_is_imm(agx_index idx)
{
   return idx.type == AGX_INDEX_IMMEDIATE;
}

/* g17_set writes RAW field bits (it does not apply the enum encoding), so the
 * shadd shift amount must be given as the G17_SHIFT_* raw constant, not the
 * logical shift. */
static unsigned
g17_shift_raw(const agx_instr *I, unsigned shift)
{
   switch (shift) {
   case 0:
      return G17_SHIFT_0;
   case 1:
      return G17_SHIFT_1;
   case 2:
      return G17_SHIFT_2;
   case 3:
      return G17_SHIFT_3;
   case 4:
      return G17_SHIFT_4;
   default:
      g17_unsupported(I, "shift not in 0..4");
      return 0;
   }
}

/* src.discard (last use) -> the G17 discard hint, when the op exposes one for
 * that operand. */
static void
g17_set_discard(struct g17_instr *g, unsigned field, agx_index src)
{
   g17_set(g, field, src.discard ? 1 : 0);
}

static void
g17_emit(struct g17_state *st, const struct g17_instr *g)
{
   uint8_t buf[G17_MAX_LENGTH];
   unsigned n = g17_pack(g, buf);
   memcpy(util_dynarray_grow_bytes(st->emission, 1, n), buf, n);
}

/* A 32-bit GPR no live value occupies, claimed by raising the shader's register
 * demand.
 *
 * Needed because, among other things, G17 writes zeros when a store's value
 * register and index register are the same (both the 8- and 10-byte forms, any
 * register pair). `o[i] = i` allocates one register for both. agx_compile.c
 * computes `nr_gprs = ctx->max_reg + 1` AFTER g17_pack_binary runs, so raising
 * max_reg here is seen by the driver and the thread-occupancy calculation.
 *
 * AGX register indices count 16-bit halves, so a 32-bit register n owns halves
 * 2n and 2n+1. */
static unsigned
g17_scratch_gpr(struct g17_state *st, const agx_instr *I)
{
   unsigned scratch = (st->ctx->max_reg + 2) / 2;
   if (scratch >= 64)
      g17_unsupported(I, "no free register for a scratch value");
   st->ctx->max_reg = MAX2(st->ctx->max_reg, (scratch * 2) + 1);
   return scratch;
}

/* Raw bits of the compare family's `type` field (g17_set does not apply the
 * enum encoding). */
#define G17_CMP_TYPE_U 2
#define G17_CMP_TYPE_S 3
#define G17_CMP_TYPE_F 1

/* The predicate a lowered select uses. Not p0: loops run on p0, and an inner
 * compare that reuses the loop's predicate hangs the GPU. The in-loop if/else
 * on p1 is the shape the golden vector "loop_add K=3 + inner if/else on p1"
 * covers. Selects do not nest -- a cmpsel's arms are operands, not blocks -- so
 * a single predicate covers all of them. */
#define G17_SEL_PRED 1

/* The register-operand compares (cmp_reg, cmp_uni, cmp_eq) KILL their register
 * sources: a later read of a compared register returns garbage. cmp_imm does
 * not. op2's kill is controllable (byte2 bit 4, clear = keep) but op1's is not,
 * so a live op1 has to be routed into op2 or copied. See g17_isa.py. */
static bool
g17_same_gpr(const agx_instr *I, agx_index x, agx_index y)
{
   return !g17_is_imm(x) && !g17_is_imm(y) && x.type == y.type &&
          g17_gpr(I, x) == g17_gpr(I, y);
}

/* True if `src` must still be readable after this instruction: AGX says it is
 * not a last use, or one of the select's arms reads it (the arms belong to the
 * same AGX instruction, so liveness does not cover them). */
static bool
g17_live_after(const agx_instr *I, agx_index src, agx_index arm0,
               agx_index arm1)
{
   return !src.discard || g17_same_gpr(I, src, arm0) ||
          g17_same_gpr(I, src, arm1);
}

/* dest = src, for a register or an immediate source. The register form is the
 * same `d = a + 0` shim AGX_OPCODE_MOV uses. The 27-family discard hint IS a
 * real last-use hint (unlike the compare family's b1.0 / b3.0, see below) and
 * it is left clear here: the arm's source may also be an operand of the
 * enclosing compare, and discarding a still-live register is silently wrong. */
static void
g17_emit_move_to(struct g17_state *st, const agx_instr *I, unsigned dest,
                 agx_index src)
{
   struct g17_instr g;

   if (g17_is_imm(src)) {
      /* AGX immediates are sign-extended 16-bit; mov32 takes the full 32-bit
       * pattern. */
      int32_t k = (int32_t)(int16_t)src.value;
      g17_instr_init(&g, G17_OP_MOV32);
      g17_set(&g, G17_MOV32_D, dest);
      g17_set(&g, G17_MOV32_IMM, (uint32_t)k);
   } else {
      g17_instr_init(&g, G17_OP_SHADD_IMM);
      g17_set(&g, G17_SHADD_IMM_D, dest);
      g17_set(&g, G17_SHADD_IMM_A, g17_gpr(I, src));
      g17_set(&g, G17_SHADD_IMM_IMM, 0);
      g17_set(&g, G17_SHADD_IMM_S, g17_shift_raw(I, 0));
      g17_set(&g, G17_SHADD_IMM_DISCARD_A, 0);
   }
   g17_emit(st, &g);
}

/* An integer operand as a GPR: an immediate is materialised into a scratch
 * register with mov32 (AGX immediates are sign-extended 16-bit), like
 * g17_float_operand does for floats. */
static unsigned
g17_int_operand(struct g17_state *st, const agx_instr *I, agx_index src)
{
   if (!g17_is_imm(src))
      return g17_gpr(I, src);
   unsigned scratch = g17_scratch_gpr(st, I);
   g17_emit_move_to(st, I, scratch, src);
   return scratch;
}

/* A 16-bit operand zero-extended into a scratch register, with the verified
 * 16->32 shadd_imm copy (A width 16, d width 32). It reads the register, so it
 * takes any pending load wait. */
static unsigned
g17_zext16_to_scratch(struct g17_state *st, const agx_instr *I, agx_index src)
{
   unsigned half, r = g17_gpr16(I, src, &half);
   unsigned scratch = g17_scratch_gpr(st, I);
   struct g17_instr g;
   g17_instr_init(&g, G17_OP_SHADD_IMM);
   g17_set(&g, G17_SHADD_IMM_D, scratch);
   g17_set(&g, G17_SHADD_IMM_D32, 1);
   g17_set(&g, G17_SHADD_IMM_A, r);
   g17_set(&g, G17_SHADD_IMM_A32, 0);
   g17_set(&g, G17_SHADD_IMM_AHALF, half);
   g17_set(&g, G17_SHADD_IMM_IMM, 0);
   g17_set(&g, G17_SHADD_IMM_S, g17_shift_raw(I, 0));
   g17_set(&g, G17_SHADD_IMM_DISCARD_A, 0);
   if (st->pending_wait) {
      g17_set(&g, G17_SHADD_IMM_WAIT, 1);
      st->pending_wait = false;
   }
   g17_emit(st, &g);
   return scratch;
}

enum g17_shift_kind { G17_LSR, G17_ASR, G17_SHL };

/* Shifts: the value is in C and the amount in the A slot, either a register or
 * a 7-bit immediate. NIR takes shift amounts modulo 32. A logical shift right
 * can also keep only `width` bits (1..31, 0 = all): that is bitfield extract. A
 * zero immediate shift of the whole word is a plain copy, done with the
 * verified shadd_imm move instead. A constant value (AGX's mask << offset) goes
 * through a scratch register: mov32, then the shift. Builds into *g, after
 * emitting any such mov32. */
static void
g17_build_shift(struct g17_state *st, struct g17_instr *g, const agx_instr *I,
                enum g17_shift_kind kind, agx_index value, agx_index amount,
                unsigned width)
{
   if (g17_is_uniform(value) || (g17_is_imm(value) && g17_is_imm(amount)))
      g17_unsupported(
         I, "shift of a uniform value, or of a constant by a constant");
   unsigned d = g17_gpr(I, I->dest[0]), c = g17_int_operand(st, I, value);
   if (width && kind != G17_LSR)
      g17_unsupported(I, "bitfield width on a shift other than lsr");

   if (g17_is_imm(amount)) {
      unsigned sh = amount.value & 31;
      if (sh == 0 && width == 0) {
         g17_instr_init(g, G17_OP_SHADD_IMM);
         g17_set(g, G17_SHADD_IMM_D, d);
         g17_set(g, G17_SHADD_IMM_A, c);
         g17_set(g, G17_SHADD_IMM_IMM, 0);
         g17_set(g, G17_SHADD_IMM_S, G17_SHIFT_0);
         g17_set_discard(g, G17_SHADD_IMM_DISCARD_A, value);
         if (st->pending_wait) {
            g17_set(g, G17_SHADD_IMM_WAIT, 1);
            st->pending_wait = false;
         }
         return;
      }
      /* The three immediate forms list d, c, imm, discard_c, cache and wait in
       * the same order; only lsr has width. */
      STATIC_ASSERT((int)G17_ASR_IMM_WAIT == (int)G17_SHR_IMM_WAIT &&
                    (int)G17_SHL_IMM_WAIT == (int)G17_SHR_IMM_WAIT);
      static const unsigned imm_op[] = {G17_OP_SHR_IMM, G17_OP_ASR_IMM,
                                        G17_OP_SHL_IMM};
      g17_instr_init(g, imm_op[kind]);
      g17_set(g, G17_SHR_IMM_D, d);
      g17_set(g, G17_SHR_IMM_C, c);
      g17_set(g, G17_SHR_IMM_IMM, sh);
      if (kind == G17_LSR)
         g17_set(g, G17_SHR_IMM_WIDTH, width);
      g17_set_discard(g, G17_SHR_IMM_DISCARD_C, value);
      if (st->pending_wait) {
         g17_set(g, G17_SHR_IMM_WAIT, 1);
         st->pending_wait = false;
      }
      return;
   }

   if (g17_is_uniform(amount))
      g17_unsupported(I, "shift by a uniform amount");
   unsigned a = g17_gpr(I, amount);
   static const unsigned op[] = {G17_OP_SHR_REG, G17_OP_ASR_REG,
                                 G17_OP_SHL_REG};
   g17_instr_init(g, op[kind]);
   /* The three register forms list their fields in the same order (d, c, a,
    * discard_a, discard_c, cache, wait). */
   g17_set(g, G17_SHR_REG_D, d);
   g17_set(g, G17_SHR_REG_C, c);
   g17_set(g, G17_SHR_REG_A, a);
   g17_set_discard(g, G17_SHR_REG_DISCARD_A, amount);
   g17_set_discard(g, G17_SHR_REG_DISCARD_C, value);
   if (kind == G17_LSR)
      g17_set(g, G17_SHR_REG_WIDTH, width);
   if (st->pending_wait) {
      g17_set(g, G17_SHR_REG_WAIT, 1);
      st->pending_wait = false;
   }
}

/* A float operand as a GPR, materialising an immediate into a scratch register
 * if needed.
 *
 * AGX float immediates are 8-bit minifloats (agx_immediate_f ->
 * agx_minifloat_encode), and G17 does have immediate float forms -- fma_imm,
 * fminmax, faddu -- but their immediate encoding is a DIFFERENT minifloat
 * (G17's exponent bias is 15 against AGX's 7) and fma_imm's addend field is 13
 * bits whose layout is not fully characterised. Decoding the AGX minifloat to a
 * float and materialising its exact 32-bit pattern with mov32 sidesteps all of
 * that, is exact for every constant rather than only the representable ones,
 * and uses only verified instructions. The cost is one mov32 and one register.
 * Worth revisiting once the G17 immediate forms are decoded, for the smaller
 * code. */
static unsigned
g17_float_operand(struct g17_state *st, const agx_instr *I, agx_index src)
{
   if (!g17_is_imm(src))
      return g17_gpr(I, src);

   float f = agx_minifloat_decode(src.value);
   uint32_t bits;
   memcpy(&bits, &f, sizeof(bits));

   unsigned scratch = g17_scratch_gpr(st, I);
   struct g17_instr g;
   g17_instr_init(&g, G17_OP_MOV32);
   g17_set(&g, G17_MOV32_D, scratch);
   g17_set(&g, G17_MOV32_IMM, bits);
   g17_emit(st, &g);
   return scratch;
}

/* icmpsel / fcmpsel: dest = (src0 <cond> src1) ? src2 : src3.
 *
 * There is no select instruction in g17_isa.py, so this lowers to an if/else --
 * compare, branch-if-false, then-arm, jump, else-arm, endif -- with a move in
 * each arm. Forward branches carry NO offset (the hardware scans for the
 * matching marker), so nothing here needs a fixup pass. Register writes inside
 * the arms are per-lane masked and readable after the endif.
 *
 * Conditions that would need a negated compare are emitted by SWAPPING THE ARMS
 * rather than by flipping the branch polarity, so every select uses the
 * `branch.true = 0` shape the golden vectors cover.
 *
 * This is the path every small conditional takes: the shared front-end flattens
 * `if` and `if/else` into a cmpsel before the backend sees control flow. */
static void
g17_emit_cmpsel(struct g17_state *st, const agx_instr *I, bool is_float)
{
   struct g17_instr g;
   unsigned dest = g17_gpr(I, I->dest[0]);
   agx_index a = I->src[0], b = I->src[1], then_src = I->src[2],
             else_src = I->src[3];
   bool eq = false, lt = false, swap = false;
   unsigned type = is_float ? G17_CMP_TYPE_F : G17_CMP_TYPE_U;

   if (is_float) {
      switch (I->fcond) {
      case AGX_FCOND_LT:
         lt = true;
         break;
      case AGX_FCOND_GT:
         lt = false;
         break;
      /* >= is not(<) and <= is not(>); both compares are strict, so negate by
       * swapping the arms. */
      case AGX_FCOND_GE:
         lt = true;
         swap = true;
         break;
      case AGX_FCOND_LE:
         lt = false;
         swap = true;
         break;
      /* LTN / GTN are AGX's NaN-aware less/greater, and they are what fmin and
       * fmax lower to: fmin(a, b) is fcmpsel(a, b, a, b, LTN). G17's float
       * compare is NaN-false in both directions, so mapping them to the plain
       * strict compares is exact for every non-NaN input and differs only when
       * the SECOND operand is NaN, where IEEE min/max would return the first
       * and this returns the NaN. Exactness would need a G17 compare mode with
       * NaN semantics, which this backend does not have. AGX's own comment
       * notes its gtn/ltn are already wrong for signed zeros and that denorms
       * are flushed, so bit-exact IEEE is not on the table at this stage
       * either; min and max are far too common to refuse over it. */
      case AGX_FCOND_LTN:
         lt = true;
         break;
      case AGX_FCOND_GTN:
         lt = false;
         break;
      /* Float == is NOT the bitwise cmp_eq (-0.0 == 0.0, and NaN compares false
       * both ways). */
      default:
         g17_unsupported(I, "fcmpsel condition");
         return;
      }
   } else {
      switch (I->icond) {
      case AGX_ICOND_ULT:
         lt = true;
         type = G17_CMP_TYPE_U;
         break;
      case AGX_ICOND_UGT:
         lt = false;
         type = G17_CMP_TYPE_U;
         break;
      case AGX_ICOND_SLT:
         lt = true;
         type = G17_CMP_TYPE_S;
         break;
      case AGX_ICOND_SGT:
         lt = false;
         type = G17_CMP_TYPE_S;
         break;
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         eq = true;
         break; /* bitwise equality, so the type does not apply */
      default:
         g17_unsupported(I, "icmpsel condition");
         return;
      }
   }

   if (swap) {
      agx_index t = then_src;
      then_src = else_src;
      else_src = t;
   }

   /* Route a live compare operand away from op1, which the register-operand
    * compares always kill. cmp_imm kills nothing, so it needs none of this.
    * Swapping the operands is free for equality and costs a direction flip for
    * an ordered compare; if both are live, or op2 cannot take a register, copy
    * op1 into a register the allocator did not use. */
   bool a_live = g17_live_after(I, a, then_src, else_src);
   bool b_live = g17_live_after(I, b, then_src, else_src);
   bool kills_operands = eq || !g17_is_imm(b);

   if (a_live && kills_operands) {
      if (!b_live && !g17_is_imm(b) && !g17_is_uniform(b)) {
         agx_index t = a;
         a = b;
         b = t;
         lt = !lt; /* a < b  ==  b > a; equality ignores `lt` entirely */
         b_live = true;
      } else {
         unsigned scratch = g17_scratch_gpr(st, I);
         g17_emit_move_to(st, I, scratch, a);
         a = agx_register(scratch * 2, AGX_SIZE_32);
      }
   }

   /* The compare. An immediate second operand uses cmp_imm (0..127); otherwise
    * both operands must be registers, because there is no scratch register to
    * materialise one into after RA.
    *
    * op1_32 / op2_32 are set on every form. They are NOT liveness hints --
    * clearing one makes the hardware compare only that register's low 16 bits,
    * which silently changes the answer (see the width note in g17_isa.py). AGX
    * operands here are 32-bit, so both are 1. */
   if (eq) {
      if (g17_is_imm(a) || g17_is_imm(b))
         g17_unsupported(I, "cmpsel equality against an immediate");
      g17_instr_init(&g, G17_OP_CMP_EQ);
      g17_set(&g, G17_CMP_EQ_OP1, g17_gpr(I, a));
      g17_set(&g, G17_CMP_EQ_OP2, g17_gpr(I, b));
      g17_set(&g, G17_CMP_EQ_P, G17_SEL_PRED);
      g17_set(&g, G17_CMP_EQ_OP1_32, 1);
      g17_set(&g, G17_CMP_EQ_OP2_32, 1);
      g17_set(&g, G17_CMP_EQ_DISCARD_OP2, b_live ? 0 : 1);
   } else if (g17_is_uniform(b)) {
      /* op2 in the uniform file: the cmp_uni form. nir_lower_int64's 64-bit add
       * compares the sum against the uniform addend to get the carry, so this
       * is on that path too. */
      if (g17_is_imm(a) || g17_is_uniform(a))
         g17_unsupported(I, "cmpsel first compare operand is not a register");
      g17_instr_init(&g, G17_OP_CMP_UNI);
      g17_set(&g, G17_CMP_UNI_OP1, g17_gpr(I, a));
      g17_set(&g, G17_CMP_UNI_U, g17_uniform_word(I, b));
      g17_set(&g, G17_CMP_UNI_LT, lt ? 1 : 0);
      g17_set(&g, G17_CMP_UNI_P, G17_SEL_PRED);
      g17_set(&g, G17_CMP_UNI_TYPE, type);
      g17_set(&g, G17_CMP_UNI_OP1_32, 1);
   } else if (g17_is_imm(b)) {
      int32_t k = (int32_t)(int16_t)b.value;
      if (g17_is_imm(a))
         g17_unsupported(I, "cmpsel with two immediate compare operands");
      if (k < 0 || k > 127)
         g17_unsupported(I, "cmpsel immediate out of 0..127");
      g17_instr_init(&g, G17_OP_CMP_IMM);
      g17_set(&g, G17_CMP_IMM_OP1, g17_gpr(I, a));
      g17_set(&g, G17_CMP_IMM_IMM, (uint32_t)k);
      g17_set(&g, G17_CMP_IMM_LT, lt ? 1 : 0);
      g17_set(&g, G17_CMP_IMM_P, G17_SEL_PRED);
      g17_set(&g, G17_CMP_IMM_TYPE, type);
      g17_set(&g, G17_CMP_IMM_OP1_32, 1);
   } else {
      if (g17_is_imm(a))
         g17_unsupported(I, "cmpsel immediate first compare operand");
      g17_instr_init(&g, G17_OP_CMP_REG);
      g17_set(&g, G17_CMP_REG_OP1, g17_gpr(I, a));
      g17_set(&g, G17_CMP_REG_OP2, g17_gpr(I, b));
      g17_set(&g, G17_CMP_REG_LT, lt ? 1 : 0);
      g17_set(&g, G17_CMP_REG_P, G17_SEL_PRED);
      g17_set(&g, G17_CMP_REG_TYPE, type);
      g17_set(&g, G17_CMP_REG_OP1_32, 1);
      g17_set(&g, G17_CMP_REG_OP2_32, 1);
      g17_set(&g, G17_CMP_REG_DISCARD_OP2, b_live ? 0 : 1);
   }
   g17_emit(st, &g);

   g17_instr_init(&g, G17_OP_BRANCH);
   g17_set(&g, G17_BRANCH_TRUE, 0);
   g17_set(&g, G17_BRANCH_P, G17_SEL_PRED);
   g17_emit(st, &g);

   g17_emit_move_to(st, I, dest, then_src);

   g17_instr_init(&g, G17_OP_JUMP);
   g17_emit(st, &g);

   g17_emit_move_to(st, I, dest, else_src);

   /* The endif marker, NOT the loop-exit marker: a `be` here makes the else
    * arm's register writes land on every lane. */
   g17_instr_init(&g, G17_OP_ENDIF_MARKER);
   g17_emit(st, &g);
}

/* The predicate a loop at AGX nesting level `nest` uses. Level 1 (the
 * outermost) gets p0, which is what the golden vectors' loops use; a select
 * inside the body runs on p1 (G17_SEL_PRED), since an inner compare reusing the
 * loop's predicate hangs the GPU. Nested loops would need p2 upwards and have
 * not been tested on hardware, so they are refused. */
static unsigned
g17_loop_pred(const agx_instr *I)
{
   if (I->nest != 1)
      g17_unsupported(I, "nested loop");
   return 0;
}

/* The branch that skips a loop body when the condition fails; `invert` is the
 * loop's invert_cond, and a branch with true = 0 skips when the predicate is
 * false. */
static void
g17_emit_loop_guard(struct g17_state *st, unsigned p, unsigned invert)
{
   struct g17_instr g;
   g17_instr_init(&g, G17_OP_BRANCH);
   g17_set(&g, G17_BRANCH_TRUE, invert);
   g17_set(&g, G17_BRANCH_P, p);
   g17_emit(st, &g);
}

/* while_icmp / while_fcmp: the loop continues while (src0 <cond> src1) XOR
 * invert_cond -- that is how agx_lower_pseudo builds it from a break_if,
 * passing !invert_cond. We emit the compare for (src0 <cond> src1) and hand the
 * inversion to the back-edge.
 *
 * The register-operand compares KILL op1 (see g17_isa.py), and a loop
 * condition's operands are loop-carried by definition, so op1 has to be
 * expendable: copy it into a register the allocator did not use, inside the
 * loop, where it is remade every iteration. op2 is kept. Without this the
 * counter is destroyed on the first iteration.
 *
 * The compare is followed by a BRANCH that skips the body. AGX's while_icmp
 * masks off the lanes that fail the condition, so the body does not run for
 * them; a G17 compare only writes a predicate, and the body would run
 * regardless -- costing exactly one extra iteration. Wrapping the body in the
 * if/else shape restores AGX's
 * semantics, with jmp_exec_any emitting the matching endif marker before the
 * back-edge. `invert_cond` drives both the branch polarity and the back-edge's:
 * the branch must skip the body when the loop-continue condition is false, and
 * the back-edge must jump while it is true. */
static void
g17_emit_while(struct g17_state *st, const agx_instr *I, bool is_float)
{
   struct g17_instr g;
   unsigned p = g17_loop_pred(I);
   agx_index a = I->src[0], b = I->src[1];
   bool eq = false, lt = false;
   unsigned type = is_float ? G17_CMP_TYPE_F : G17_CMP_TYPE_U;

   if (is_float) {
      switch (I->fcond) {
      case AGX_FCOND_LT:
         lt = true;
         break;
      case AGX_FCOND_GT:
         lt = false;
         break;
      default:
         g17_unsupported(I, "while_fcmp condition");
         return;
      }
   } else {
      switch (I->icond) {
      case AGX_ICOND_ULT:
         lt = true;
         type = G17_CMP_TYPE_U;
         break;
      case AGX_ICOND_UGT:
         lt = false;
         type = G17_CMP_TYPE_U;
         break;
      case AGX_ICOND_SLT:
         lt = true;
         type = G17_CMP_TYPE_S;
         break;
      case AGX_ICOND_SGT:
         lt = false;
         type = G17_CMP_TYPE_S;
         break;
      case AGX_ICOND_UEQ:
      case AGX_ICOND_SEQ:
         eq = true;
         break;
      default:
         g17_unsupported(I, "while_icmp condition");
         return;
      }
   }

   if (g17_is_imm(a))
      g17_unsupported(I, "while with an immediate first operand");

   /* An immediate bound uses cmp_imm, which kills nothing. */
   if (!eq && g17_is_imm(b)) {
      int32_t k = (int32_t)(int16_t)b.value;
      if (k < 0 || k > 127)
         g17_unsupported(I, "while immediate bound out of 0..127");
      g17_instr_init(&g, G17_OP_CMP_IMM);
      g17_set(&g, G17_CMP_IMM_OP1, g17_gpr(I, a));
      g17_set(&g, G17_CMP_IMM_IMM, (uint32_t)k);
      g17_set(&g, G17_CMP_IMM_LT, lt ? 1 : 0);
      g17_set(&g, G17_CMP_IMM_P, p);
      g17_set(&g, G17_CMP_IMM_TYPE, type);
      g17_set(&g, G17_CMP_IMM_OP1_32, 1);
      g17_emit(st, &g);
      g17_emit_loop_guard(st, p, I->invert_cond ? 1 : 0);
      st->loop_invert = I->invert_cond ? 1 : 0;
      st->loop_pred = p;
      return;
   }

   /* op1 must be expendable: copy it. (Swapping instead would need the inverse
    * comparison, which G17 cannot express, and both operands are live across
    * the back-edge anyway.) */
   unsigned scratch = g17_scratch_gpr(st, I);
   g17_emit_move_to(st, I, scratch, a);

   if (eq) {
      if (g17_is_imm(b) || g17_is_uniform(b))
         g17_unsupported(I, "while equality against a non-register");
      g17_instr_init(&g, G17_OP_CMP_EQ);
      g17_set(&g, G17_CMP_EQ_OP1, scratch);
      g17_set(&g, G17_CMP_EQ_OP2, g17_gpr(I, b));
      g17_set(&g, G17_CMP_EQ_P, p);
      g17_set(&g, G17_CMP_EQ_OP1_32, 1);
      g17_set(&g, G17_CMP_EQ_OP2_32, 1);
      g17_set(&g, G17_CMP_EQ_DISCARD_OP2, 0);
   } else if (g17_is_uniform(b)) {
      g17_instr_init(&g, G17_OP_CMP_UNI);
      g17_set(&g, G17_CMP_UNI_OP1, scratch);
      g17_set(&g, G17_CMP_UNI_U, g17_uniform_word(I, b));
      g17_set(&g, G17_CMP_UNI_LT, lt ? 1 : 0);
      g17_set(&g, G17_CMP_UNI_P, p);
      g17_set(&g, G17_CMP_UNI_TYPE, type);
      g17_set(&g, G17_CMP_UNI_OP1_32, 1);
   } else {
      g17_instr_init(&g, G17_OP_CMP_REG);
      g17_set(&g, G17_CMP_REG_OP1, scratch);
      g17_set(&g, G17_CMP_REG_OP2, g17_gpr(I, b));
      g17_set(&g, G17_CMP_REG_LT, lt ? 1 : 0);
      g17_set(&g, G17_CMP_REG_P, p);
      g17_set(&g, G17_CMP_REG_TYPE, type);
      g17_set(&g, G17_CMP_REG_OP1_32, 1);
      g17_set(&g, G17_CMP_REG_OP2_32, 1);
      g17_set(&g, G17_CMP_REG_DISCARD_OP2, 0);
   }
   g17_emit(st, &g);
   g17_emit_loop_guard(st, p, I->invert_cond ? 1 : 0);
   st->loop_invert = I->invert_cond ? 1 : 0;
   st->loop_pred = p;
}

static void
g17_emit_one(struct g17_state *st, const agx_instr *I)
{
   struct g17_instr g;

   if (I->op == AGX_OPCODE_WAIT) {
      st->pending_wait = true;
      return;
   }

   switch (I->op) {
   case AGX_OPCODE_STOP:
      st->pending_wait = false; /* nothing after the shader to protect */
      g17_instr_init(&g, G17_OP_STOP);
      break;

   case AGX_OPCODE_GET_SR:
      /* Only thread_position_in_grid.x maps to the one G17 tid producer. */
      if (I->sr != 80 /* thread_position_in_grid.x */)
         g17_unsupported(
            I, "special register other than thread_position_in_grid.x");
      g17_instr_init(&g, G17_OP_MOV_SR);
      g17_set(&g, G17_MOV_SR_D, g17_gpr(I, I->dest[0]));
      break;

   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
      if (st->pending_wait)
         g17_unsupported(I, "loop condition consuming an outstanding load");
      g17_emit_while(st, I, I->op == AGX_OPCODE_WHILE_FCMP);
      return;

   case AGX_OPCODE_JMP_EXEC_ANY: {
      /* The loop back-edge. G17 computes its target as `fe + 4 + 2 * offset` in
       * signed halfwords, and a back-edge's target always precedes it, so the
       * offset is known at emit time -- no fixup pass, unlike agx_pack_binary.
       * Forward jumps have no offset field at all on G17 (the hardware scans
       * for the matching marker), so a forward jmp_exec_any is a different
       * shape entirely. */
      if (I->target == NULL)
         g17_unsupported(I, "indirect branch");
      if (st->loop_invert < 0)
         g17_unsupported(I,
                         "back-edge with no loop condition emitted before it");

      int64_t here = (int64_t)st->emission->size;
      int64_t delta = (int64_t)I->target->offset - here - 4;
      if (I->target->offset > (off_t)here)
         g17_unsupported(
            I, "forward jmp_exec_any (G17 forward branches carry no offset)");
      if (delta % 2)
         g17_unsupported(I, "odd back-edge displacement");
      if (delta / 2 < -4096 || delta / 2 > 4095)
         g17_unsupported(I, "back-edge out of the 13-bit offset field");

      /* Close the guard that while_icmp opened: the body is the then-block. */
      struct g17_instr endif;
      g17_instr_init(&endif, G17_OP_ENDIF_MARKER);
      g17_emit(st, &endif);

      /* The endif moved the back-edge, so re-read where we are before computing
       * the displacement. */
      here = (int64_t)st->emission->size;
      delta = (int64_t)I->target->offset - here - 4;
      if (delta % 2)
         g17_unsupported(I, "odd back-edge displacement");
      if (delta / 2 < -4096 || delta / 2 > 4095)
         g17_unsupported(I, "back-edge out of the 13-bit offset field");

      g17_instr_init(&g, G17_OP_BACK_EDGE);
      g17_set(&g, G17_BACK_EDGE_OFFSET, delta / 2);
      g17_set(&g, G17_BACK_EDGE_P, st->loop_pred);
      g17_set(&g, G17_BACK_EDGE_TRUE, st->loop_invert);
      st->loop_invert = -1;
      break;
   }

   case AGX_OPCODE_POP_EXEC:
      /* Closes the loop with the loop-exit marker. (Inside an if/else the same
       * marker would make the else arm's register writes land on every lane,
       * which is why selects use the endif marker.) */
      if (I->nest != 1)
         g17_unsupported(I, "pop_exec of more than one nesting level");
      g17_instr_init(&g, G17_OP_LOOP_EXIT);
      break;

   case AGX_OPCODE_ICMPSEL:
   case AGX_OPCODE_FCMPSEL:
      /* Emits a whole if/else chain, so it bypasses the single-instruction tail
       * below. A pending load wait must be drained before the compare reads its
       * sources, and no op in the chain carries a wait bit, so fall back to the
       * shim by leaving st->pending_wait for the next consumer only if there is
       * nothing to drain. */
      if (st->pending_wait)
         g17_unsupported(
            I, "cmpsel consuming an outstanding load (needs a wait shim)");
      g17_emit_cmpsel(st, I, I->op == AGX_OPCODE_FCMPSEL);
      return;

   case AGX_OPCODE_MOV_IMM:
      /* r0l is AGX's exec-mask bookkeeping register: `break` sets it to the
       * nesting level to unwind to, and jmp_exec_* / pop_exec consume it
       * implicitly. G17's loop structure is explicit -- predicates plus the
       * back-edge and loop-exit markers -- so it has no counter to maintain and
       * this initialisation is dead. Skip exactly that register; any other
       * 16-bit mov_imm is real data and still refused. */
      if (I->dest[0].size == AGX_SIZE_16 &&
          I->dest[0].type == AGX_INDEX_REGISTER && I->dest[0].value == 0)
         return;
      if (I->dest[0].size != AGX_SIZE_32)
         g17_unsupported(I, "non-32-bit mov_imm");
      g17_instr_init(&g, G17_OP_MOV32);
      g17_set(&g, G17_MOV32_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_MOV32_IMM, (uint32_t)I->imm);
      break;

   case AGX_OPCODE_MOV:
      /* reg -> reg copy: d = a + 0 (shadd_imm, imm 0). */
      g17_instr_init(&g, G17_OP_SHADD_IMM);
      g17_set(&g, G17_SHADD_IMM_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_SHADD_IMM_A, g17_gpr(I, I->src[0]));
      g17_set(&g, G17_SHADD_IMM_IMM, 0);
      g17_set(&g, G17_SHADD_IMM_S, g17_shift_raw(I, 0));
      g17_set_discard(&g, G17_SHADD_IMM_DISCARD_A, I->src[0]);
      break;

   case AGX_OPCODE_IADD:
      /* dst = src0 + (src1 << shift). */
      if (g17_is_imm(I->src[1]) && I->shift == 0) {
         int64_t k = (int64_t)(int16_t)I->src[1]
                        .value; /* AGX immediates are sign-extended 16-bit */
         if (k < 0 || k > 255)
            g17_unsupported(I, "iadd immediate out of 0..255");
         if (I->dest[0].size == AGX_SIZE_32 && I->src[0].size == AGX_SIZE_16) {
            /* A 16-bit source of a 32-bit AGX add SIGN-extends (AGX lowers
             * signext to `x + 0`; .abs would zero-extend, and g17_gpr16 refuses
             * it). G17's 16-bit A form zero-extends, which is wrong for
             * negative halves. The high half sign-extends as an arithmetic
             * shift right by 16; the low half is not handled yet. */
            unsigned half, r = g17_gpr16(I, I->src[0], &half);
            if (!half || k != 0)
               g17_unsupported(
                  I, "16-bit sign extension of a low half, or with an addend");
            g17_instr_init(&g, G17_OP_ASR_IMM);
            g17_set(&g, G17_ASR_IMM_D, g17_gpr(I, I->dest[0]));
            g17_set(&g, G17_ASR_IMM_C, r);
            g17_set(&g, G17_ASR_IMM_IMM, 16);
            if (st->pending_wait) {
               g17_set(&g, G17_ASR_IMM_WAIT, 1);
               st->pending_wait = false;
            }
            break;
         }
         g17_instr_init(&g, G17_OP_ADD_IMM);
         g17_set_w16(&g, I, I->dest[0], I->src[0], G17_ADD_IMM_D,
                     G17_ADD_IMM_D32, G17_ADD_IMM_DHALF, G17_ADD_IMM_A,
                     G17_ADD_IMM_A32, G17_ADD_IMM_AHALF);
         g17_set(&g, G17_ADD_IMM_IMM, (uint32_t)k);
         g17_set_discard(&g, G17_ADD_IMM_DISCARD_A, I->src[0]);
         if (st->pending_wait) {
            g17_set(&g, G17_ADD_IMM_WAIT, 1);
            st->pending_wait = false;
         }
      } else if (!g17_is_imm(I->src[0]) && !g17_is_imm(I->src[1]) &&
                 !g17_is_uniform(I->src[0]) && !g17_is_uniform(I->src[1])) {
         /* shadd: d = (A << s) + C, so the shifted operand src1 -> A, src0 ->
          * C. Both operands must be registers here; a uniform one is handled
          * below, since only the A field can hold one. */
         g17_instr_init(&g, G17_OP_SHADD);
         g17_set(&g, G17_SHADD_D, g17_gpr(I, I->dest[0]));
         g17_set(&g, G17_SHADD_A, g17_gpr(I, I->src[1]));
         g17_set(&g, G17_SHADD_C, g17_gpr(I, I->src[0]));
         g17_set(&g, G17_SHADD_S, g17_shift_raw(I, I->shift));
         g17_set_discard(&g, G17_SHADD_DISCARD_A, I->src[1]);
         g17_set_discard(&g, G17_SHADD_DISCARD_C, I->src[0]);
         if (st->pending_wait) {
            g17_set(&g, G17_SHADD_WAIT, 1);
            st->pending_wait = false;
         }
      } else if (g17_is_uniform(I->src[0]) != g17_is_uniform(I->src[1]) &&
                 !g17_is_imm(I->src[0]) && !g17_is_imm(I->src[1])) {
         /* One operand is a uniform. Only the 27-family's A field can hold one
          * (selected by its a_uniform kind bit), and A is also the SHIFTED
          * operand, so:
          *   - the uniform is src1, the shifted operand: A = uniform, C = src0,
          * any shift;
          *   - the uniform is src0, the unshifted one: addition commutes, so A
          * = uniform and C = src1 is the same value -- but only at shift 0,
          * since otherwise the register would need the shift and A is taken.
          * This is what nir_lower_int64's 64-bit add expansion produces (lo =
          * u0 + index, then hi = carry + u1), so without it no 64-bit address
          * arithmetic can be emitted. */
         bool uni_is_shifted = g17_is_uniform(I->src[1]);
         agx_index uni = uni_is_shifted ? I->src[1] : I->src[0];
         agx_index reg = uni_is_shifted ? I->src[0] : I->src[1];
         unsigned d = g17_gpr(I, I->dest[0]);

         if (!uni_is_shifted && I->shift != 0) {
            /* (reg << s) + uniform: the shift needs the A field and so does the
             * uniform, so do it in two steps through the destination -- no
             * scratch register required, and safe even when dest aliases the
             * source, since the first step only reads it. d = reg << s
             * (shadd_imm with a zero addend) d = uniform + d   (shadd with the
             * uniform in A, s = 0) The first step reads the register, so it is
             * the one that takes any pending load wait. */
            struct g17_instr sh;
            g17_instr_init(&sh, G17_OP_SHADD_IMM);
            g17_set(&sh, G17_SHADD_IMM_D, d);
            g17_set(&sh, G17_SHADD_IMM_A, g17_gpr(I, reg));
            g17_set(&sh, G17_SHADD_IMM_IMM, 0);
            g17_set(&sh, G17_SHADD_IMM_S, g17_shift_raw(I, I->shift));
            g17_set_discard(&sh, G17_SHADD_IMM_DISCARD_A, reg);
            if (st->pending_wait) {
               g17_set(&sh, G17_SHADD_IMM_WAIT, 1);
               st->pending_wait = false;
            }
            g17_emit(st, &sh);

            g17_instr_init(&g, G17_OP_SHADD);
            g17_set(&g, G17_SHADD_D, d);
            g17_set(&g, G17_SHADD_A, g17_uniform_word(I, uni));
            g17_set(&g, G17_SHADD_A_UNIFORM, 1);
            g17_set(&g, G17_SHADD_C, d);
            g17_set(&g, G17_SHADD_S, g17_shift_raw(I, 0));
            g17_set(&g, G17_SHADD_DISCARD_C, 0);
            g17_emit(st, &g);
            return;
         }

         g17_instr_init(&g, G17_OP_SHADD);
         g17_set(&g, G17_SHADD_D, d);
         g17_set(&g, G17_SHADD_A, g17_uniform_word(I, uni));
         g17_set(&g, G17_SHADD_A_UNIFORM, 1);
         g17_set(&g, G17_SHADD_C, g17_gpr(I, reg));
         g17_set(&g, G17_SHADD_S,
                 g17_shift_raw(I, uni_is_shifted ? I->shift : 0));
         g17_set_discard(&g, G17_SHADD_DISCARD_C, reg);
         if (st->pending_wait) {
            g17_set(&g, G17_SHADD_WAIT, 1);
            st->pending_wait = false;
         }
      } else if (g17_is_imm(I->src[0]) && !g17_is_imm(I->src[1])) {
         /* imm + (reg << shift) -> shadd_imm, which computes exactly d = (A <<
          * s) + imm. AGX puts the constant in src[0] whenever it folds a
          * constant into an add, so this is the ordinary shape for `x * 2` and
          * `x + 7` after lowering -- without it every conditional shader
          * stopped here. */
         int64_t k = (int64_t)(int16_t)I->src[0].value;
         if (k < 0 || k > 255)
            g17_unsupported(I, "iadd immediate out of 0..255");
         g17_instr_init(&g, G17_OP_SHADD_IMM);
         g17_set(&g, G17_SHADD_IMM_D, g17_gpr(I, I->dest[0]));
         g17_set(&g, G17_SHADD_IMM_A, g17_gpr(I, I->src[1]));
         g17_set(&g, G17_SHADD_IMM_IMM, (uint32_t)k);
         g17_set(&g, G17_SHADD_IMM_S, g17_shift_raw(I, I->shift));
         g17_set_discard(&g, G17_SHADD_IMM_DISCARD_A, I->src[1]);
         if (st->pending_wait) {
            g17_set(&g, G17_SHADD_IMM_WAIT, 1);
            st->pending_wait = false;
         }
      } else {
         g17_unsupported(I, "iadd operand form");
      }
      break;

   case AGX_OPCODE_IMAD: {
      /* dst = (src0 * src1) + (src2 << shift). G17 has no multiply-add, so a
       * nonzero addend becomes imul into the dest followed by an add -- which
       * needs the dest not to be one of the multiplicands, since the imul would
       * clobber it before the add reads it. */
      bool zero_addend =
         g17_is_imm(I->src[2]) && I->src[2].value == 0 && I->shift == 0;
      /* An immediate multiplicand (NIR turns x << k into x * 2^k) goes through
       * a scratch register. */
      if (g17_is_imm(I->src[0]) && g17_is_imm(I->src[1]))
         g17_unsupported(I, "imad of two immediates");
      unsigned ma = g17_int_operand(st, I, I->src[0]);
      unsigned mb = g17_int_operand(st, I, I->src[1]);

      if (!zero_addend) {
         unsigned d = g17_gpr(I, I->dest[0]);
         /* The imul writes before the add reads, so the product cannot land in
          * the dest when the dest is also a multiplicand (or the addend). Put
          * it in a register the allocator did not use. */
         unsigned prod = d;
         if (d == ma || d == mb ||
             (!g17_is_imm(I->src[2]) && d == g17_gpr(I, I->src[2])))
            prod = g17_scratch_gpr(st, I);

         struct g17_instr mul;
         g17_instr_init(&mul, G17_OP_IMUL);
         g17_set(&mul, G17_IMUL_D, prod);
         g17_set(&mul, G17_IMUL_A, ma);
         g17_set(&mul, G17_IMUL_B, mb);
         g17_set_discard(&mul, G17_IMUL_DISCARD_A, I->src[0]);
         g17_set_discard(&mul, G17_IMUL_DISCARD_B, I->src[1]);
         if (st->pending_wait) {
            g17_set(&mul, G17_IMUL_WAIT, 1);
            st->pending_wait = false;
         }
         g17_emit(st, &mul);

         if (g17_is_imm(I->src[2])) {
            int64_t k = (int64_t)(int16_t)I->src[2].value;
            if (k < 0 || k > 255 || I->shift != 0)
               g17_unsupported(I, "imad addend out of 0..255 or shifted");
            g17_instr_init(&g, G17_OP_ADD_IMM);
            g17_set(&g, G17_ADD_IMM_D, d);
            g17_set(&g, G17_ADD_IMM_A, prod);
            g17_set(&g, G17_ADD_IMM_IMM, (uint32_t)k);
            g17_set(&g, G17_ADD_IMM_DISCARD_A, 0);
         } else {
            /* shadd: d = (A << s) + C, so the shifted addend goes in A and the
             * product in C. */
            g17_instr_init(&g, G17_OP_SHADD);
            g17_set(&g, G17_SHADD_D, d);
            g17_set(&g, G17_SHADD_A, g17_gpr(I, I->src[2]));
            g17_set(&g, G17_SHADD_C, prod);
            g17_set(&g, G17_SHADD_S, g17_shift_raw(I, I->shift));
            g17_set_discard(&g, G17_SHADD_DISCARD_A, I->src[2]);
            g17_set(&g, G17_SHADD_DISCARD_C, 0);
         }
         break;
      }

      g17_instr_init(&g, G17_OP_IMUL);
      g17_set(&g, G17_IMUL_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_IMUL_A, ma);
      g17_set(&g, G17_IMUL_B, mb);
      g17_set_discard(&g, G17_IMUL_DISCARD_A, I->src[0]);
      g17_set_discard(&g, G17_IMUL_DISCARD_B, I->src[1]);
      if (st->pending_wait) {
         g17_set(&g, G17_IMUL_WAIT, 1);
         st->pending_wait = false;
      }
      break;
   }

   case AGX_OPCODE_CONVERT: {
      /* agx_convert_to(dst, immediate(MODE), value, round): src[0] names the
       * conversion. Only the unsigned 32-bit integer<->float directions are
       * handled, so name the mode in the refusal. */
      if (!g17_is_imm(I->src[0]))
         g17_unsupported(I, "convert with a non-constant mode");
      unsigned mode = I->src[0].value;
      if (mode == AGX_CONVERT_U32_TO_F)
         g17_instr_init(&g, G17_OP_U2F);
      else if (mode == AGX_CONVERT_F_TO_U32)
         g17_instr_init(&g, G17_OP_F2U);
      else {
         char why[48];
         snprintf(why, sizeof(why), "convert mode %u (only u32<->f32)", mode);
         g17_unsupported(I, why);
      }
      bool to_float = mode == AGX_CONVERT_U32_TO_F;
      unsigned half;

      /* u32 -> f16 is one instruction: the destination carries the same width
       * and half selects as the 12-byte 27-family forms. */
      if (to_float) {
         if (I->dest[0].size == AGX_SIZE_16) {
            g17_set(&g, G17_U2F_D, g17_gpr16(I, I->dest[0], &half));
            g17_set(&g, G17_U2F_D32, 0);
            g17_set(&g, G17_U2F_DHALF, half);
         } else {
            g17_set(&g, G17_U2F_D, g17_gpr(I, I->dest[0]));
         }
         if (I->src[1].size != AGX_SIZE_32)
            g17_unsupported(I, "convert to float from a non-32-bit integer");
         g17_set(&g, G17_U2F_S, g17_gpr(I, I->src[1]));
         g17_set_discard(&g, G17_U2F_DISCARD_S, I->src[1]);
         break;
      }

      /* f16 -> u32 sets s16, whose polarity is the opposite of the A operand's
       * width bit, with the half in b1.0. The CEIL_SRC layout keeps them in
       * different places from add_imm's. */
      if (I->dest[0].size != AGX_SIZE_32)
         g17_unsupported(I, "convert to a non-32-bit integer");
      g17_set(&g, G17_F2U_D, g17_gpr(I, I->dest[0]));
      if (I->src[1].size == AGX_SIZE_16) {
         g17_set(&g, G17_F2U_S, g17_gpr16(I, I->src[1], &half));
         g17_set(&g, G17_F2U_S16, 1);
         g17_set(&g, G17_F2U_SHALF, half);
      } else {
         g17_set(&g, G17_F2U_S, g17_gpr(I, I->src[1]));
      }
      g17_set_discard(&g, G17_F2U_DISCARD_S, I->src[1]);
      break;
   }

   case AGX_OPCODE_POPCOUNT:
      g17_instr_init(&g, G17_OP_POPCOUNT);
      g17_set(&g, G17_POPCOUNT_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_POPCOUNT_S, g17_gpr(I, I->src[0]));
      g17_set_discard(&g, G17_POPCOUNT_DISCARD_S, I->src[0]);
      break;

   case AGX_OPCODE_SRSQRT:
      /* AGX computes sqrt(x) as x * srsqrt(x), so srsqrt is the reciprocal
       * square root. */
      g17_instr_init(&g, G17_OP_FRSQRT);
      g17_set(&g, G17_FRSQRT_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_FRSQRT_S, g17_gpr(I, I->src[0]));
      g17_set_discard(&g, G17_FRSQRT_DISCARD_S, I->src[0]);
      break;

   case AGX_OPCODE_EXP2:
   case AGX_OPCODE_LOG2:
   case AGX_OPCODE_CEIL: {
      unsigned op = I->op == AGX_OPCODE_EXP2   ? G17_OP_FEXP2
                    : I->op == AGX_OPCODE_LOG2 ? G17_OP_FLOG2
                                               : G17_OP_FCEIL;
      /* All three share the 10-byte unary layout, so the field indices
       * coincide. */
      g17_instr_init(&g, op);
      g17_set(&g, G17_FEXP2_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_FEXP2_S, g17_gpr(I, I->src[0]));
      g17_set_discard(&g, G17_FEXP2_DISCARD_S, I->src[0]);
      break;
   }

   case AGX_OPCODE_FFMA: {
      /* AGX: dst = src0 * src1 + src2. G17's ffma is `d = d * A + C`, so the
       * DEST doubles as one multiplicand -- it has to already hold src0 or
       * src1. When it holds neither, copy one in first (an integer add of 0 is
       * a bit-exact register copy, so it is safe on a float). */
      agx_index mul_a = I->src[0], mul_b = I->src[1], addend = I->src[2];
      unsigned d = g17_gpr(I, I->dest[0]);

      /* Multiplication commutes, so prefer whichever multiplicand the dest
       * already holds; that avoids the copy entirely. An immediate can never be
       * in the dest, so check registers first. */
      if (!g17_is_imm(mul_b) && d == g17_gpr(I, mul_b)) {
         agx_index t = mul_a;
         mul_a = mul_b;
         mul_b = t;
      }

      unsigned a_gpr = g17_float_operand(st, I, mul_b);
      unsigned c_gpr = g17_float_operand(st, I, addend);
      bool dest_holds_mul = !g17_is_imm(mul_a) && d == g17_gpr(I, mul_a);

      if (!dest_holds_mul) {
         if (d == a_gpr || d == c_gpr)
            g17_unsupported(
               I, "ffma whose destination is one of its other operands");
         if (g17_is_imm(mul_a)) {
            unsigned m = g17_float_operand(st, I, mul_a);
            g17_emit_move_to(st, I, d, agx_register(m * 2, AGX_SIZE_32));
         } else {
            g17_emit_move_to(st, I, d, mul_a);
         }
      }

      g17_instr_init(&g, G17_OP_FFMA);
      g17_set(&g, G17_FFMA_D, d);
      g17_set(&g, G17_FFMA_A, a_gpr);
      g17_set(&g, G17_FFMA_C, c_gpr);
      g17_set(&g, G17_FFMA_DISCARD_A, 0);
      g17_set(&g, G17_FFMA_DISCARD_C, 0);
      break;
   }

   case AGX_OPCODE_HADD:
   case AGX_OPCODE_HMUL: {
      /* AGX has dedicated fp16 opcodes rather than sizing fadd/fmul, but they
       * map onto the same 09-family forms with byte0's fp16 bit clear. An fp16
       * immediate is materialised into the LOW half of a scratch register:
       * mov32 of the zero-extended fp16 pattern puts the value exactly there,
       * so the operand is then an ordinary half register. */
      agx_index a = I->src[0], b = I->src[1];
      agx_index arg[2] = {a, b};
      for (unsigned k = 0; k < 2; ++k) {
         if (!g17_is_imm(arg[k]))
            continue;
         float f = agx_minifloat_decode(arg[k].value);
         uint16_t h = _mesa_float_to_half(f);
         unsigned scratch = g17_scratch_gpr(st, I);
         struct g17_instr mv;
         g17_instr_init(&mv, G17_OP_MOV32);
         g17_set(&mv, G17_MOV32_D, scratch);
         g17_set(&mv, G17_MOV32_IMM, h);
         g17_emit(st, &mv);
         arg[k] =
            agx_register(scratch * 2, AGX_SIZE_16); /* the low half holds it */
      }

      g17_instr_init(&g, I->op == AGX_OPCODE_HADD ? G17_OP_FADD : G17_OP_FMUL);
      g17_set_f16(&g, I, I->dest[0], arg[0], arg[1], G17_FADD_D, G17_FADD_A,
                  G17_FADD_B, G17_FADD_F32, G17_FADD_DHALF, G17_FADD_AHALF,
                  G17_FADD_BHALF);
      break;
   }

   case AGX_OPCODE_FADD:
      /* a + (-b) is a subtraction, which G17 has as its own op; the float
       * source modifiers have no encoding in the 09-family forms, so this is
       * the only way to express it. (-a) + b is the same as b - a, so the
       * operands just swap. */
      if (!I->src[0].neg && I->src[1].neg && !I->src[0].abs && !I->src[1].abs) {
         g17_instr_init(&g, G17_OP_FSUB);
         g17_set(&g, G17_FSUB_D, g17_gpr(I, I->dest[0]));
         g17_set(&g, G17_FSUB_A, g17_float_operand(st, I, agx_abs(I->src[0])));
         g17_set(&g, G17_FSUB_B, g17_float_operand(st, I, agx_abs(I->src[1])));
         g17_set(&g, G17_FSUB_DISCARD_A, 0);
         g17_set(&g, G17_FSUB_DISCARD_B, 0);
         break;
      }
      if (I->src[0].neg && !I->src[1].neg && !I->src[0].abs && !I->src[1].abs) {
         g17_instr_init(&g, G17_OP_FSUB);
         g17_set(&g, G17_FSUB_D, g17_gpr(I, I->dest[0]));
         g17_set(&g, G17_FSUB_A, g17_float_operand(st, I, agx_abs(I->src[1])));
         g17_set(&g, G17_FSUB_B, g17_float_operand(st, I, agx_abs(I->src[0])));
         g17_set(&g, G17_FSUB_DISCARD_A, 0);
         g17_set(&g, G17_FSUB_DISCARD_B, 0);
         break;
      }
      FALLTHROUGH;
   case AGX_OPCODE_FMUL:
      if (g17_is_imm(I->src[0]) || g17_is_imm(I->src[1])) {
         if (I->dest[0].size == AGX_SIZE_16)
            g17_unsupported(
               I, "fp16 float immediate (mov32 materialises a 32-bit pattern)");
         /* Materialise the constant rather than refusing; see
          * g17_float_operand. */
         unsigned a = g17_float_operand(st, I, I->src[0]);
         unsigned b = g17_float_operand(st, I, I->src[1]);
         g17_instr_init(&g,
                        I->op == AGX_OPCODE_FADD ? G17_OP_FADD : G17_OP_FMUL);
         g17_set(&g, G17_FADD_D, g17_gpr(I, I->dest[0]));
         g17_set(&g, G17_FADD_A, a);
         g17_set(&g, G17_FADD_B, b);
         g17_set(&g, G17_FADD_DISCARD_A, 0);
         g17_set(&g, G17_FADD_DISCARD_B, 0);
         break;
      }
      if (g17_is_imm(I->src[0]) || g17_is_imm(I->src[1]))
         g17_unsupported(I, "float ALU immediate operand");
      g17_instr_init(&g, I->op == AGX_OPCODE_FADD ? G17_OP_FADD : G17_OP_FMUL);
      g17_set_f16(&g, I, I->dest[0], I->src[0], I->src[1], G17_FADD_D,
                  G17_FADD_A, G17_FADD_B, G17_FADD_F32, G17_FADD_DHALF,
                  G17_FADD_AHALF, G17_FADD_BHALF);
      g17_set_discard(&g, G17_FADD_DISCARD_A, I->src[0]);
      g17_set_discard(&g, G17_FADD_DISCARD_B, I->src[1]);
      break;

   /* Shifts (g17_build_shift). AGX spells ushr as bfeil(0, x, s, mask 0),
    * ubitfield_extract as bfeil(0, x, s, mask = width), and a register ishl as
    * bfi(0, x, s, mask 0). A nonzero insert base, or a bitfield insert with a
    * mask, is not handled yet. */
   case AGX_OPCODE_BFEIL:
   case AGX_OPCODE_BFI: {
      bool extract = I->op == AGX_OPCODE_BFEIL;
      if (!g17_is_imm(I->src[0]) || I->src[0].value != 0 ||
          (!extract && I->bfi_mask != 0))
         g17_unsupported(I, "bitfield insert, or extract into a nonzero base");
      if (I->bfi_mask > 31)
         g17_unsupported(I, "bitfield width out of 1..31");
      g17_build_shift(st, &g, I, extract ? G17_LSR : G17_SHL, I->src[1],
                      I->src[2], extract ? I->bfi_mask : 0);
      break;
   }

   case AGX_OPCODE_ASR:
      g17_build_shift(st, &g, I, G17_ASR, I->src[0], I->src[1], 0);
      break;

   case AGX_OPCODE_BITOP: {
      /* `truth_table` is the FOUR-bit two-source LUT, the index into
       * agx_print's lut2_to_function: 0x6 xor, 0x8 and, 0xa mov1 (= src0), 0xc
       * mov2 (= src1), 0xe or, and so on. It is NOT the 8-bit value: UTIL_LUT2
       * evaluates the expression over a = 0xAA / b = 0xCC and then masks with
       * ~c = 0x0F and narrows to util_lut2, so AGX_OPCODE_MOV lowers to table
       * 0xa, iand to 0x8, ior to 0xe, ixor to 0x6. */
      if (I->truth_table == 0xa || I->truth_table == 0xc) {
         /* mov1 / mov2: out = that source, which may be a register OR an
          * immediate. */
         agx_index src = I->src[I->truth_table == 0xa ? 0 : 1];
         if (g17_is_imm(src)) {
            /* mov32 carries no wait bit, so a pending load wait could not be
             * honoured here. */
            if (st->pending_wait)
               g17_unsupported(I,
                               "constant move consuming an outstanding load");
            g17_emit_move_to(st, I, g17_gpr(I, I->dest[0]), src);
            return;
         }
         /* A 16-bit copy, or a 16->32 zero extension, is the same shim with the
          * width bits set from the operand sizes -- which is how AGX spells
          * both (`mov1 r0l` and `mov1 ^r1l` into a 32-bit dest). */
         g17_instr_init(&g, G17_OP_SHADD_IMM);
         g17_set_w16(&g, I, I->dest[0], src, G17_SHADD_IMM_D, G17_SHADD_IMM_D32,
                     G17_SHADD_IMM_DHALF, G17_SHADD_IMM_A, G17_SHADD_IMM_A32,
                     G17_SHADD_IMM_AHALF);
         g17_set(&g, G17_SHADD_IMM_IMM, 0);
         g17_set(&g, G17_SHADD_IMM_S, g17_shift_raw(I, 0));
         g17_set_discard(&g, G17_SHADD_IMM_DISCARD_A, src);
         break;
      }
      /* Two registers: any truth table (bw_reg takes AGX's table bit for bit).
       * With an immediate: and/or/xor (bw_imm), or a NOT -- AGX lowers not(x)
       * to bitop(x, 0, ~a), and any table whose result for b = 0 is ~s is a
       * NOT. G17 has no not form of its own, so it is nor(x, x): with both
       * inputs equal only table bits 0 and 3 matter, and nor's are 1 and 0. */
      if (I->dest[0].size != AGX_SIZE_32)
         g17_unsupported(I, "non-32-bit bitwise operation");
      bool not_s = g17_is_imm(I->src[1]) && I->src[1].value == 0 &&
                   (I->truth_table & 3) == 1;
      if (!g17_is_imm(I->src[1]) || not_s) {
         if (g17_is_imm(I->src[0]) || g17_is_uniform(I->src[0]) ||
             g17_is_uniform(I->src[1]))
            g17_unsupported(I, "bitop with a constant or uniform operand");
         /* A 16-bit operand of a 32-bit bitop is zero-extended (AGX packs
          * halves this way, e.g. bitfield insert at bit 0 or 16): copy it into
          * a scratch register first. */
         unsigned rs = I->src[0].size == AGX_SIZE_16
                          ? g17_zext16_to_scratch(st, I, I->src[0])
                          : g17_gpr(I, I->src[0]);
         unsigned rb = not_s ? rs
                       : I->src[1].size == AGX_SIZE_16
                          ? g17_zext16_to_scratch(st, I, I->src[1])
                          : g17_gpr(I, I->src[1]);
         g17_instr_init(&g, G17_OP_BW_REG);
         g17_set(&g, G17_BW_REG_D, g17_gpr(I, I->dest[0]));
         g17_set(&g, G17_BW_REG_S, rs);
         g17_set(&g, G17_BW_REG_B, rb);
         g17_set(&g, G17_BW_REG_LUT, not_s ? 0x1 : I->truth_table);
         g17_set_discard(&g, G17_BW_REG_DISCARD_S, I->src[0]);
         g17_set_discard(&g, G17_BW_REG_DISCARD_B,
                         not_s ? I->src[0] : I->src[1]);
         if (st->pending_wait) {
            g17_set(&g, G17_BW_REG_WAIT, 1);
            st->pending_wait = false;
         }
         break;
      }
      unsigned op;
      switch (I->truth_table) {
      case 0x8:
         op = G17_BITOP_AND;
         break;
      case 0xe:
         op = G17_BITOP_OR;
         break;
      case 0x6:
         op = G17_BITOP_XOR;
         break;
      default: {
         /* Name the table (agx_print's lut2_to_function lists all 16). */
         char why[64];
         snprintf(why, sizeof(why), "bitop truth table %#x with an immediate",
                  I->truth_table);
         g17_unsupported(I, why);
      }
      }
      unsigned imm = I->src[1].value;
      if (imm > 255)
         g17_unsupported(I, "bitop immediate out of 0..255");
      g17_instr_init(&g, G17_OP_BW_IMM);
      g17_set(&g, G17_BW_IMM_D, g17_gpr(I, I->dest[0]));
      g17_set(&g, G17_BW_IMM_S, g17_gpr(I, I->src[0]));
      g17_set(&g, G17_BW_IMM_OP, op);
      g17_set(&g, G17_BW_IMM_IMM, imm);
      g17_set_discard(&g, G17_BW_IMM_DISCARD_S, I->src[0]);
      break;
   }

   case AGX_OPCODE_DEVICE_LOAD:
   case AGX_OPCODE_DEVICE_STORE: {
      bool is_store = I->op == AGX_OPCODE_DEVICE_STORE;
      agx_index data = is_store ? I->src[0] : I->dest[0];
      agx_index mbase = I->src[is_store ? 1 : 0];
      agx_index off = I->src[is_store ? 2 : 1];

      /* The base is a 64-bit pair in either the uniform file or the general
       * register file, selected by byte4 bit 7 (`base_reg`). The uniform form
       * is what agx_nir_lower_address produces for ordinary buffer access; the
       * register form is for a computed pointer -- one loaded from memory, or
       * chosen per lane -- and AGX encodes the same split with its own flag, so
       * the two files share one numbering (the index of the pair's first 32-bit
       * half, hence /2 below). */
      bool base_is_reg = mbase.type == AGX_INDEX_REGISTER;
      if (!base_is_reg && mbase.type != AGX_INDEX_UNIFORM)
         g17_unsupported(I, "memory base is neither a uniform nor a register");
      if (mbase.size != AGX_SIZE_64)
         g17_unsupported(I, "memory base is not 64-bit");
      /* A 64-bit register occupies four 16-bit halves, so the pair starts on an
       * even 32-bit register; uniforms only need to start on an even half. */
      if (mbase.value & (base_is_reg ? 3 : 1))
         g17_unsupported(I, "misaligned memory base");
      if (I->format != AGX_FORMAT_I32)
         g17_unsupported(I, "non-32-bit memory format");
      if (!g17_is_imm(off) && off.size != AGX_SIZE_32)
         g17_unsupported(I, "non-32-bit memory index");

      /* G17's `scale` is the index's BYTE stride. AGX's I->shift is NOT that:
       * it is the shift BEYOND the format size. agx_nir_lower_address computes
       * `.base = shift - format_shift` and agx_emit_store passes it straight
       * through as I->shift, so the hardware stride is format_size << shift --
       * for I32 (the only format accepted above) that is 4 << shift: shift 0 ->
       * 4 B    shift 1 -> 8 B    shift 2 -> 16 B    shift 3 -> 32 B (G17 cannot
       * express) */
      unsigned stride = 4u << I->shift;
      unsigned scale_raw;
      switch (stride) {
      case 1:
         scale_raw = G17_SCALE_1;
         break;
      case 4:
         scale_raw = G17_SCALE_4;
         break;
      case 8:
         scale_raw = G17_SCALE_8;
         break;
      case 16:
         scale_raw = G17_SCALE_16;
         break;
      default:
         g17_unsupported(I, "index stride not 1/4/8/16 bytes");
         return;
      }
      unsigned mask_raw;
      switch (I->mask) {
      case 0x1:
         mask_raw = G17_WRMASK_X;
         break;
      case 0x3:
         mask_raw = G17_WRMASK_XY;
         break;
      case 0x7:
         mask_raw = G17_WRMASK_XYZ;
         break;
      case 0xf:
         mask_raw = G17_WRMASK_XYZW;
         break;
      default:
         g17_unsupported(I, "memory component mask");
         return;
      }

      /* To wait on prior loads, use the 10-byte extended form with its wait
       * bit. */
      bool wait = st->pending_wait;
      unsigned op = is_store ? (wait ? G17_OP_STORE_W : G17_OP_STORE)
                             : (wait ? G17_OP_LOAD_W : G17_OP_LOAD);
      g17_instr_init(&g, op);
      if (wait) {
         g17_set(&g, is_store ? G17_STORE_W_WAIT : G17_LOAD_W_WAIT, 1);
         st->pending_wait = false;
      }
      /* Normalise the index register once. `.abs` on a memory index is AGX's
       * zero-extend flag, not an absolute value, so it is stripped before any
       * register check -- see the index handling below. */
      agx_index idx = off;
      idx.abs = false;

      /* G17 writes zeros when a store's value register is ALSO one of its
       * address registers -- the index, or either half of a register base pair.
       * So copy the value somewhere else first (see g17_scratch_gpr). The copy,
       * not the store, carries any pending load wait: it is the instruction
       * that reads the loaded register. */
      unsigned data_gpr = g17_gpr(I, data);
      unsigned base_gpr = mbase.value / 2;
      bool conflict =
         is_store &&
         ((!g17_is_imm(off) && data_gpr == g17_gpr(I, idx)) ||
          (base_is_reg && (data_gpr == base_gpr || data_gpr == base_gpr + 1)));
      if (conflict) {
         struct g17_instr cp;
         unsigned scratch = g17_scratch_gpr(st, I);
         g17_instr_init(&cp, G17_OP_SHADD_IMM);
         g17_set(&cp, G17_SHADD_IMM_D, scratch);
         g17_set(&cp, G17_SHADD_IMM_A, data_gpr);
         g17_set(&cp, G17_SHADD_IMM_IMM, 0);
         g17_set(&cp, G17_SHADD_IMM_S, g17_shift_raw(I, 0));
         g17_set(&cp, G17_SHADD_IMM_DISCARD_A, 0);
         if (st->pending_wait) {
            g17_set(&cp, G17_SHADD_IMM_WAIT, 1);
            st->pending_wait = false;
            wait = false;
            op = G17_OP_STORE;
            g17_instr_init(&g, op);
         }
         g17_emit(st, &cp);
         data_gpr = scratch;
      }
      g17_set(&g, is_store ? G17_STORE_SRC : G17_LOAD_D, data_gpr);
      g17_set(&g, is_store ? G17_STORE_BASE : G17_LOAD_BASE, mbase.value / 2);
      g17_set(&g, is_store ? G17_STORE_BASE_REG : G17_LOAD_BASE_REG,
              base_is_reg ? 1 : 0);
      g17_set(&g, is_store ? G17_STORE_SCALE : G17_LOAD_SCALE, scale_raw);
      g17_set(&g, is_store ? G17_STORE_MASK : G17_LOAD_MASK, mask_raw);
      if (g17_is_imm(off)) {
         /* A constant index folds into G17's `offset` field instead of burning
          * a register. That field counts WORDS (the address is base +
          * scale*r[idx] + 4*offset) and holds 0..63, so the byte displacement
          * has to be a multiple of 4 and fit. */
         uint64_t bytes = (uint64_t)off.value * stride;
         if ((bytes % 4) != 0 || (bytes / 4) > 63)
            g17_unsupported(
               I, "constant index outside the 0..63-word offset field");
         g17_set(&g, is_store ? G17_STORE_INDEXED : G17_LOAD_INDEXED, 0);
         g17_set(&g, is_store ? G17_STORE_OFFSET : G17_LOAD_OFFSET,
                 (unsigned)(bytes / 4));
      } else {
         /* `.abs` on a memory index is not an absolute value: it is AGX's
          * ZERO-EXTEND flag (agx_pack puts it in bit 25, and
          * agx_nir_lower_address sets it for the unsigned u2u64 lea, clear for
          * the signed i2i64 one). G17's index register is zero-extended and
          * this backend has no extension control for it, so .abs is emitted the
          * same either way. That is exact for the unsigned form and an
          * approximation for the signed one: it differs only for a NEGATIVE
          * index, which buffer addressing does not generate -- GL and VK bound
          * SSBO/UBO offsets to INT32_MAX and agx_nir_lower_address relies on
          * that -- but OpenCL pointer arithmetic could.
          * `.neg` is a genuine negation and has no equivalent at all, so it is
          * refused. */
         if (off.neg)
            g17_unsupported(I, "negated memory index");
         g17_set(&g, is_store ? G17_STORE_INDEXED : G17_LOAD_INDEXED, 1);
         g17_set(&g, is_store ? G17_STORE_IDX : G17_LOAD_IDX, g17_gpr(I, idx));
      }
      break;
   }

   default:
      g17_unsupported(I, "no G17 translation");
      return;
   }

   /* The op we just built didn't carry the pending load wait -> it has no wait
    * bit (e.g. a float op). Emit a 27-family wait-shim first: an identity add
    * (d = d + 0) with b0.3 set drains the loads (one wait covers all
    * outstanding), writing the consumer's own dest (which it overwrites next),
    * so the consumer then reads valid data. Verified on the M5. */
   if (st->pending_wait) {
      struct g17_instr shim;
      unsigned d = g17_gpr(I, I->dest[0]);
      g17_instr_init(&shim, G17_OP_SHADD_IMM);
      g17_set(&shim, G17_SHADD_IMM_D, d);
      g17_set(&shim, G17_SHADD_IMM_A, d);
      g17_set(&shim, G17_SHADD_IMM_IMM, 0);
      g17_set(&shim, G17_SHADD_IMM_S, g17_shift_raw(I, 0));
      g17_set(&shim, G17_SHADD_IMM_WAIT, 1);
      g17_emit(st, &shim);
      st->pending_wait = false;
   }

   g17_emit(st, &g);
}

void
g17_pack_binary(agx_context *ctx, struct util_dynarray *emission)
{
   struct g17_state st = {.ctx = ctx, .emission = emission, .loop_invert = -1};

   agx_foreach_block(ctx, block) {
      /* Where this block starts, so a back-edge can resolve its target without
       * a fixup pass. */
      block->offset = emission->size;

      st.pending_wait = false; /* AGX waits are inserted per block */
      agx_foreach_instr_in_block(block, I)
         g17_emit_one(&st, I);
   }
}
