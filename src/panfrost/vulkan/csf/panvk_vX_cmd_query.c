/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "util/os_time.h"

#include "genxml/gen_macros.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_macros.h"
#include "panvk_query_pool.h"

void
panvk_per_arch(cmd_write_timestamp)(struct panvk_cmd_buffer *cmd,
                                    struct panvk_query_pool *pool,
                                    uint32_t query,
                                    UNUSED VkPipelineStageFlags2 stage)
{
   struct cs_builder *b = panvk_get_cs_builder(cmd, PANVK_SUBQUEUE_COMPUTE);

   struct cs_index report_address = cs_scratch_reg64(b, 0);
   cs_move64_to(b, report_address, panvk_query_report_dev_addr(pool, query));
   cs_store_state(b, report_address, 0, MALI_CS_STATE_TIMESTAMP, cs_now());

   struct cs_index available_address = cs_scratch_reg64(b, 2);
   struct cs_index value_scratch = cs_scratch_reg32(b, 4);
   cs_move64_to(b, available_address,
                panvk_query_available_dev_addr(pool, query));
   cs_move32_to(b, value_scratch, 1);
   cs_store32(b, value_scratch, available_address, 0);
   cs_wait_slot(b, SB_ID(LS), false);
}

void
panvk_per_arch(cmd_begin_end_query)(struct panvk_cmd_buffer *cmd,
                                    struct panvk_query_pool *pool,
                                    uint32_t query, VkQueryControlFlags flags,
                                    ASSERTED uint32_t index, bool end)
{
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);

   struct cs_builder *b = panvk_get_cs_builder(cmd, PANVK_SUBQUEUE_COMPUTE);

   /* TODO: transform feedback */
   assert(index == 0);

   uint64_t report_addr = panvk_query_report_dev_addr(pool, query);

   const VkMemoryBarrier pre_barrier = {
      /* XXX: Narrow this */
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };

   dev->vk.dispatch_table.CmdPipelineBarrier(
      panvk_cmd_buffer_to_handle(cmd),
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre_barrier, 0, NULL, 0,
      NULL);

   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_OCCLUSION: {
      if (end) {
         cmd->state.gfx.occlusion_query.ptr = 0;
         cmd->state.gfx.occlusion_query.mode = MALI_OCCLUSION_MODE_DISABLED;
      } else {
         cmd->state.gfx.occlusion_query.ptr = report_addr;
         cmd->state.gfx.occlusion_query.mode =
            flags & VK_QUERY_CONTROL_PRECISE_BIT
               ? MALI_OCCLUSION_MODE_COUNTER
               : MALI_OCCLUSION_MODE_PREDICATE;

         /* From the Vulkan spec:
          *
          *   "When an occlusion query begins, the count of passing samples
          *    always starts at zero."
          *
          */
         struct cs_index value_scratch = cs_scratch_reg64(b, 2);
         cs_move64_to(b, value_scratch, 0);

         for (unsigned i = 0; i < pool->reports_per_query; i++) {
            /* XXX: slow, pack it with store multiple */
            struct cs_index address = cs_scratch_reg64(b, 0);
            cs_move64_to(b, address,
                         report_addr + i * sizeof(struct panvk_query_report));
            cs_store64(b, value_scratch, address, 0);
            cs_wait_slot(b, SB_ID(LS), false);
         }
      }
      break;
   }
   default:
      unreachable("Unsupported query type");
   }

   if (end) {
      struct cs_index address = cs_scratch_reg64(b, 0);
      struct cs_index value_scratch = cs_scratch_reg32(b, 2);
      cs_move64_to(b, address, panvk_query_available_dev_addr(pool, query));
      /* XXX: Only for debug, use 0x1 */
      cs_move32_to(b, value_scratch, 0xDEADBEEF);
      cs_store32(b, value_scratch, address, 0);
      cs_wait_slot(b, SB_ID(LS), false);
   }

   const VkMemoryBarrier post_barrier = {
      /* XXX: Narrow this */
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };

   dev->vk.dispatch_table.CmdPipelineBarrier(
      panvk_cmd_buffer_to_handle(cmd), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      0, 1, &post_barrier, 0, NULL, 0, NULL);
}
