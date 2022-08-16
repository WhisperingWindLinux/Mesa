/*
 * Copyright (C) 2022 Collabora, Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "etnaviv_nir.h"

static bool
lower_global(nir_builder *b, nir_instr *instr, UNUSED void *unused)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_store_global:
      break;
   default:
      return false;
   }

   /* We're purposefully ignoring the second commponent of the address,
    * since we only care about 32-bit addresses. */

   b->cursor = nir_before_instr(instr);

   if (intr->intrinsic == nir_intrinsic_store_global) {
      assert(nir_intrinsic_src_components(intr, 1) == 1);
      nir_ssa_def *addr = nir_vec2(b, nir_ssa_for_src(b, intr->src[1], 1),
                                   nir_imm_zero(b, 1, 32));

      unsigned num_comp = nir_intrinsic_src_components(intr, 0);

      nir_ssa_def *value = nir_ssa_for_src(b, intr->src[0], num_comp);
      nir_ssa_def *v = nir_channels(b, value, BITFIELD_MASK(num_comp));
      nir_build_store_global_2x32_offset(b, v, addr, nir_imm_zero(b, 1, 32));
   } else {
      assert(nir_intrinsic_src_components(intr, 0) == 1);
      nir_ssa_def *addr = nir_vec2(b, nir_ssa_for_src(b, intr->src[0], 1),
                                   nir_imm_zero(b, 1, 32));

      unsigned num_comp = nir_dest_num_components(intr->dest);
      unsigned bitsize = nir_dest_bit_size(intr->dest);

      nir_ssa_def *v =
         nir_build_load_global_2x32_offset(b, num_comp, bitsize, addr,
                                           nir_imm_zero(b, 1, 32));
      nir_ssa_def_rewrite_uses(&intr->dest.ssa, v);
   }

   nir_instr_remove(instr);

   return true;
}

void
etna_nir_lower_global(nir_shader *shader)
{
   nir_shader_instructions_pass(shader, lower_global,
                                nir_metadata_block_index |
                                nir_metadata_dominance, NULL);
}
