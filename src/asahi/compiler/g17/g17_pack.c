/*
 * SPDX-License-Identifier: MIT
 *
 * Table-driven G17 encode / decode / length. The tables (templates, opcode
 * masks, field bit positions, defaults, length model) are generated from
 * g17_isa.py by g17_gen.py.
 */
#include <assert.h>
#include <string.h>
#include "g17.h"

static void
scatter(uint8_t *buf, const struct g17_field_info *f, uint32_t raw)
{
   for (unsigned i = 0; i < f->width; ++i) {
      const struct g17_bitpos p = f->pos[i];
      buf[p.byte] =
         (buf[p.byte] & ~(1u << p.bit)) | (((raw >> i) & 1u) << p.bit);
   }
}

static uint32_t
gather(const uint8_t *buf, const struct g17_field_info *f)
{
   uint32_t v = 0;
   for (unsigned i = 0; i < f->width; ++i)
      v |= ((uint32_t)(buf[f->pos[i].byte] >> f->pos[i].bit) & 1u) << i;
   return v;
}

void
g17_instr_init(struct g17_instr *I, enum g17_op op)
{
   assert(op < G17_NUM_OPS);
   const struct g17_op_info *info = &g17_op_info[op];
   memset(I, 0, sizeof(*I));
   I->op = op;
   I->length = info->length;
   for (unsigned i = 0; i < info->num_fields; ++i)
      I->raw[i] = info->fields[i].default_raw;
}

int64_t
g17_get(const struct g17_instr *I, unsigned field)
{
   const struct g17_field_info *f = &g17_op_info[I->op].fields[field];
   const uint32_t raw = I->raw[field];
   if (f->is_signed && (raw >> (f->width - 1)) & 1)
      return (int64_t)raw - ((int64_t)1 << f->width);
   return raw;
}

void
g17_set(struct g17_instr *I, unsigned field, int64_t value)
{
   assert(field < g17_op_info[I->op].num_fields);
   const struct g17_field_info *f = &g17_op_info[I->op].fields[field];
   const int64_t span = (int64_t)1 << f->width;
   if (f->is_signed) {
      assert(value >= -span / 2 && value < span / 2);
      I->raw[field] = (uint32_t)(value & (span - 1));
   } else {
      assert(value >= 0 && value < span);
      I->raw[field] = (uint32_t)value;
   }
}

unsigned
g17_pack(const struct g17_instr *I, uint8_t *out)
{
   const struct g17_op_info *info = &g17_op_info[I->op];
   memcpy(out, info->tmpl, info->length);
   for (unsigned i = 0; i < info->num_fields; ++i) {
      const struct g17_field_info *f = &info->fields[i];
      assert(f->width >= 32 || (I->raw[i] >> f->width) == 0);
      scatter(out, f, I->raw[i]);
   }
   return info->length;
}

unsigned
g17_length(const uint8_t *code, size_t avail)
{
   uint8_t x[14] = {0};
   memcpy(x, code, avail < 14 ? avail : 14);

   for (unsigned r = 0; r < g17_num_length_rules; ++r) {
      const struct g17_length_rule *rule = &g17_length_rules[r];
      bool hit = true;
      for (unsigned i = 0; i < 14 && hit; ++i)
         hit = (x[i] & rule->mask[i]) == rule->match[i];
      if (hit)
         return rule->length;
   }

   const unsigned row = (x[0] & 7) << 4 |
                        ((x[0] & 7) == 4 && (x[1] & 0x80) ? 1 : 0) << 3 |
                        (x[2] & 7);
   const uint8_t r = g17_length_rows[row];
   if (!(r & 0x80))
      return r;
   const unsigned col = (x[4] & 7) << 7 | (x[6] & 0xF) << 3 | (x[8] & 7);
   const uint8_t packed = g17_length_sub[r & 0x7f][col >> 1];
   return 2 * ((col & 1) ? (packed >> 4) : (packed & 0xF));
}

bool
g17_unpack(const uint8_t *code, size_t avail, struct g17_instr *I)
{
   memset(I, 0, sizeof(*I));
   I->op = G17_OP_UNKNOWN;
   const unsigned n = g17_length(code, avail);
   if (n == 0 || n > avail)
      return false;
   I->length = n;

   /* Most specific matching form of this length (largest opcode mask), as in
    * g17_isa.match(). */
   int best = -1;
   for (unsigned op = 0; op < G17_NUM_OPS; ++op) {
      const struct g17_op_info *info = &g17_op_info[op];
      if (info->length != n)
         continue;
      bool hit = true;
      for (unsigned i = 0; i < n && hit; ++i)
         hit = (code[i] & info->mask[i]) == info->match[i];
      if (hit &&
          (best < 0 || info->specificity > g17_op_info[best].specificity))
         best = op;
   }
   if (best < 0)
      return true;

   I->op = (enum g17_op)best;
   const struct g17_op_info *info = &g17_op_info[best];
   for (unsigned i = 0; i < info->num_fields; ++i)
      I->raw[i] = gather(code, &info->fields[i]);
   return true;
}

int32_t
g17_back_edge_offset(size_t fe_offset, size_t target_offset)
{
   const int64_t delta = (int64_t)target_offset - (int64_t)fe_offset - 4;
   assert((delta & 1) == 0);
   return (int32_t)(delta / 2);
}
