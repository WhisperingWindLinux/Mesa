/*
 * Copyright (c) 2022 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sub license,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#include "nir.h"
#include "nir_builder.h"

static void
lower_uniform_width(nir_builder *b, nir_intrinsic_instr *intr, unsigned width)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_ssa_def *loads[NIR_MAX_VEC_COMPONENTS];

   for (unsigned i = 0; i < DIV_ROUND_UP(intr->num_components, width); i++) {
      nir_ssa_def *offset = nir_iadd_imm(b, nir_ssa_for_src(b, intr->src[0], 1),
                                         i * (width * nir_dest_bit_size(intr->dest) / 8));
      nir_ssa_def *dest =
         nir_load_uniform(b, width, nir_dest_bit_size(intr->dest), offset,
                          .base = nir_intrinsic_base(intr),
                          .range = nir_intrinsic_range(intr),
                          .dest_type = nir_intrinsic_dest_type(intr));

      for (unsigned j = 0; j < MIN2(width, intr->num_components - i * width); j++) {
         loads[i * width + j] = nir_swizzle(b, dest, &j, 1);
      }
   }

   nir_ssa_def_rewrite_uses(&intr->dest.ssa,
                            nir_vec(b, loads, intr->num_components));
   nir_instr_remove(&intr->instr);
}

void
nir_lower_uniform_width(nir_shader *shader, unsigned width)
{
   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_builder b;
         nir_builder_init(&b, function->impl);

         nir_foreach_block(block, function->impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;

               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

               if (intr->num_components <= width)
                  continue;

               if (intr->intrinsic != nir_intrinsic_load_uniform)
                  continue;

               lower_uniform_width(&b, intr, width);
            }
         }
      }
   }
}
