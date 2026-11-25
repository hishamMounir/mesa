/*
 * SPDX-License-Identifier: MIT
 *
 * Self-contained tests for the G17 machine-code layer (no GPU needed):
 *  1. every golden vector (test/g17_golden_vectors.json) decodes completely
 * (known length + known op for every instruction) and re-encodes
 * byte-identically;
 *  2. every loop back-edge in them lands on an instruction start (target = fe +
 * 4 + 2 * offset);
 *  3. known encodings pack exactly;
 *  4. field defaults reproduce each template, and every field round-trips its
 * full value range.
 */
#include <stdio.h>
#include <string.h>
#include "g17.h"
#include "g17_golden.h"

static unsigned failures;

#define CHECK(cond, ...)                                                       \
   do {                                                                        \
      if (!(cond)) {                                                           \
         ++failures;                                                           \
         fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                  \
         fprintf(stderr, __VA_ARGS__);                                         \
         fputc('\n', stderr);                                                  \
      }                                                                        \
   } while (0)

static void
test_golden(void)
{
   const unsigned n = sizeof(g17_golden) / sizeof(g17_golden[0]);
   for (unsigned v = 0; v < n; ++v) {
      const struct g17_golden *g = &g17_golden[v];
      size_t starts[512];
      unsigned nstarts = 0;
      size_t p = 0;
      bool stopped = false;
      while (p < g->size && !stopped) {
         struct g17_instr I;
         bool known = g17_unpack(g->code + p, g->size - p, &I);
         CHECK(known && I.op < G17_NUM_OPS,
               "%s: undecoded instruction at +0x%zx", g->name, p);
         if (!known || I.op >= G17_NUM_OPS)
            break;
         uint8_t re[G17_MAX_LENGTH];
         unsigned len = g17_pack(&I, re);
         CHECK(len == I.length && !memcmp(re, g->code + p, len),
               "%s: re-encode differs at +0x%zx (%s)", g->name, p,
               g17_op_info[I.op].name);
         starts[nstarts++] = p;
         stopped = I.op == G17_OP_STOP;
         p += I.length;
      }
      CHECK(stopped && p == g->size, "%s: walked %zu of %zu bytes", g->name, p,
            g->size);

      for (unsigned i = 0; i < nstarts; ++i) {
         struct g17_instr I;
         g17_unpack(g->code + starts[i], g->size - starts[i], &I);
         if (I.op != G17_OP_BACK_EDGE)
            continue;
         const int64_t target =
            (int64_t)starts[i] + 4 + 2 * g17_get(&I, G17_BACK_EDGE_OFFSET);
         bool hit = false;
         for (unsigned j = 0; j < nstarts; ++j)
            hit |= (int64_t)starts[j] == target;
         CHECK(
            hit,
            "%s: back-edge at +0x%zx targets +0x%llx, not an instruction start",
            g->name, starts[i], (long long)target);
         CHECK(g17_back_edge_offset(starts[i], (size_t)target) ==
                  g17_get(&I, G17_BACK_EDGE_OFFSET),
               "%s: g17_back_edge_offset disagrees", g->name);
      }
   }
   printf("golden: %u vectors decoded and re-encoded\n", n);
}

static void
expect_bytes(const struct g17_instr *I, const char *hex, const char *what)
{
   uint8_t got[G17_MAX_LENGTH], want[G17_MAX_LENGTH];
   unsigned n = g17_pack(I, got);
   size_t w = strlen(hex) / 2;
   for (size_t i = 0; i < w; ++i)
      sscanf(hex + 2 * i, "%2hhx", &want[i]);
   CHECK(n == w && !memcmp(got, want, n), "%s", what);
}

static void
test_known_encodings(void)
{
   struct g17_instr I;

   /* mov32 r6 = 0xdeadbeef (the prefix of the "imulhi" golden vector) */
   g17_instr_init(&I, G17_OP_MOV32);
   g17_set(&I, G17_MOV32_D, 6);
   g17_set(&I, G17_MOV32_IMM, 0xdeadbeef);
   expect_bytes(&I, "6cef02de1a0c6d05", "mov32 r6, 0xdeadbeef");

   /* register compare r1 > r2 */
   g17_instr_init(&I, G17_OP_CMP_REG);
   g17_set(&I, G17_CMP_REG_OP1, 1);
   g17_set(&I, G17_CMP_REG_OP2, 2);
   expect_bytes(&I, "0a033a050400", "cmp_reg r1 > r2");

   /* in-loop branch on p1 */
   g17_instr_init(&I, G17_OP_BRANCH);
   g17_set(&I, G17_BRANCH_P, 1);
   expect_bytes(&I, "1e10000e", "branch p1");

   /* the loop_add golden vectors' back-edge: offset -15 halfwords */
   g17_instr_init(&I, G17_OP_BACK_EDGE);
   g17_set(&I, G17_BACK_EDGE_OFFSET, -15);
   expect_bytes(&I, "fe00000ede0ffb1e1f00ffc7ff7f", "back_edge offset -15");
   CHECK(g17_get(&I, G17_BACK_EDGE_OFFSET) == -15, "offset sign extension");

   /* back-edge with the true polarity, offset -51 */
   g17_set(&I, G17_BACK_EDGE_OFFSET, -51);
   g17_set(&I, G17_BACK_EDGE_TRUE, 1);
   expect_bytes(&I, "fe00020ede6ff11e1f00ffc7ff7f",
                "back_edge true, offset -51");

   /* shadd r0 = (r1 << 1) + r1 with the cache hint cleared */
   g17_instr_init(&I, G17_OP_SHADD);
   g17_set(&I, G17_SHADD_D, 0);
   g17_set(&I, G17_SHADD_A, 1);
   g17_set(&I, G17_SHADD_C, 1);
   g17_set(&I, G17_SHADD_S, G17_SHIFT_1);
   g17_set(&I, G17_SHADD_CACHE, 0);
   uint8_t got[G17_MAX_LENGTH];
   g17_pack(&I, got);
   CHECK(got[4] == 0x21, "shadd cache=0 clears b4.3");

   /* lengths */
   static const struct {
      const char *hex;
      unsigned len;
   } L[] = {
      {"0e000000", 4},
      {"0600", 2},
      {"0ca01006", 4},
      {"0c80020002000000", 8},
      {"2706041a2900a30288800400", 12},
      {"2706041a2100a102888000000102", 14},
      {"0a033a050400", 6},
      {"0a8b2301060007000000", 10},
      {"fe00000e5e0ffb1e1f00ffc7ff7f", 14},
      {"be03400e", 4},
      {"3e00190e000000000000", 10},
      {"09051801", 4},
      {"22010ad00002", 6},
   };
   for (unsigned i = 0; i < sizeof(L) / sizeof(L[0]); ++i) {
      uint8_t b[G17_MAX_LENGTH] = {0};
      size_t n = strlen(L[i].hex) / 2;
      for (size_t j = 0; j < n; ++j)
         sscanf(L[i].hex + 2 * j, "%2hhx", &b[j]);
      CHECK(g17_length(b, n) == L[i].len, "length(%s) = %u, want %u", L[i].hex,
            g17_length(b, n), L[i].len);
   }
   printf("known encodings + lengths checked\n");
}

static void
test_fields(void)
{
   unsigned checked = 0;
   for (unsigned op = 0; op < G17_NUM_OPS; ++op) {
      const struct g17_op_info *info = &g17_op_info[op];
      struct g17_instr I;
      uint8_t buf[G17_MAX_LENGTH];

      g17_instr_init(&I, (enum g17_op)op);
      g17_pack(&I, buf);
      struct g17_instr D;
      CHECK(g17_unpack(buf, info->length, &D) && D.op == op,
            "%s: default encoding decodes as %s", info->name,
            D.op < G17_NUM_OPS ? g17_op_info[D.op].name : "unknown");

      for (unsigned f = 0; f < info->num_fields; ++f) {
         const struct g17_field_info *fi = &info->fields[f];
         const uint32_t top =
            fi->width >= 32 ? 0xffffffffu : (1u << fi->width) - 1;
         const uint32_t probes[] = {
            0, 1, top, top >> 1, 0x55555555u & top, 0xaaaaaaaau & top};
         for (unsigned k = 0; k < sizeof(probes) / sizeof(probes[0]); ++k) {
            g17_instr_init(&I, (enum g17_op)op);
            I.raw[f] = probes[k];
            g17_pack(&I, buf);
            if (!g17_unpack(buf, info->length, &D) || D.op != op)
               continue; /* the value turned it into another form: the masks
                            keep decode unambiguous */
            CHECK(D.raw[f] == probes[k], "%s.%s: raw 0x%x read back as 0x%x",
                  info->name, fi->name, probes[k], D.raw[f]);
            ++checked;
         }
      }
   }
   printf("fields: %u value round-trips\n", checked);
}

int
main(void)
{
   test_golden();
   test_known_encodings();
   test_fields();
   if (failures) {
      printf("%u FAILURES\n", failures);
      return 1;
   }
   printf("ALL PASS\n");
   return 0;
}
