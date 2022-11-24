/*
 * Copyright (C) 2022 Collabora, Ltd.
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
 * Authors:
 *    Tomeu Vizoso <tomeu.vizoso@collabora.com>
 */

#include "util/macros.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "nir_serialize.h"

#include "etnaviv_compiler.h"
#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_disk_cache.h"
#include "etnaviv_screen.h"
#include "etnaviv_shader.h"
#include "etnaviv_util.h"

static void
etna_set_compute_resources(struct pipe_context *pctx,
                         unsigned start, unsigned count,
                         struct pipe_surface **resources)
{
   /* Not using constant buffers, we have the constants in the command stream.
    * Since we advertise 256 vec4s and a 4096B kernel input size, we don't need
    * constant buffers for now. */
}

static void
etna_set_global_binding(struct pipe_context *pctx,
                      unsigned first, unsigned count,
                      struct pipe_resource **resources,
                      uint32_t **handles)
{
   struct etna_context *ctx = etna_context(pctx);
   struct etna_global_bindings_state *so = &ctx->global_bindings;
   unsigned mask = 0;

   /* We're relying on softpin for global bindings */
   assert(etnaviv_device_softpin_capable(ctx->screen->dev));

   if (resources) {
      for (unsigned i = 0; i < count; i++) {
         unsigned n = i + first;

         mask |= BITFIELD_BIT(n);

         pipe_resource_reference(&so->buf[n], resources[i]);

         if (so->buf[n]) {
            struct etna_resource *rsc = etna_resource(so->buf[n]);
            uint32_t offset = *handles[i];
            uint64_t iova = etna_bo_gpu_va(rsc->bo) + offset;

            /* TODO: There must be a way to know if this buffer was created with eg. CL_MEM_READ_ONLY */
            resource_written(ctx, resources[i]);
            resource_read(ctx, resources[i]);

            /* Yes, really, despite what the type implies: */
            memcpy(handles[i], &iova, sizeof(iova));

            so->enabled_mask |= BITFIELD_BIT(n);
         } else {
            so->enabled_mask &= ~BITFIELD_BIT(n);
         }
      }
   } else {
      mask = (BITFIELD_BIT(count) - 1) << first;

      for (unsigned i = 0; i < count; i++) {
         unsigned n = i + first;
         pipe_resource_reference(&so->buf[n], NULL);
      }

      so->enabled_mask &= ~mask;
   }
}

static void
etna_memory_barrier(struct pipe_context *pctx, unsigned flags)
{
   /* TODO */
}

void
etna_compute_context_init(struct pipe_context *pctx)
{
   pctx->set_compute_resources = etna_set_compute_resources;
   pctx->set_global_binding = etna_set_global_binding;

   pctx->memory_barrier = etna_memory_barrier;
}
