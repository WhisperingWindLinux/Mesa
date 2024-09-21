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

static void
panvk_emit_write_job(struct panvk_cmd_buffer *cmd, struct panvk_batch *batch,
                     enum mali_write_value_type type, uint64_t addr,
                     uint64_t value)
{
   struct panfrost_ptr job =
      pan_pool_alloc_desc(&cmd->desc_pool.base, WRITE_VALUE_JOB);

   pan_section_pack(job.cpu, WRITE_VALUE_JOB, PAYLOAD, payload) {
      payload.type = type;
      payload.address = addr;
      payload.immediate_value = value;
   };

   pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_WRITE_VALUE, true, false, 0, 0,
                  &job, false);
}

static struct panvk_batch *
open_batch(struct panvk_cmd_buffer *cmd, bool *had_batch)
{
   bool res = cmd->cur_batch != NULL;

   if (!res)
      panvk_per_arch(cmd_open_batch)(cmd);

   *had_batch = res;

   return cmd->cur_batch;
}

static void
close_batch(struct panvk_cmd_buffer *cmd, bool had_batch)
{
   if (!had_batch)
      panvk_per_arch(cmd_close_batch)(cmd);
}

void
panvk_per_arch(cmd_write_timestamp)(struct panvk_cmd_buffer *cmd,
                                    struct panvk_query_pool *pool,
                                    uint32_t query,
                                    UNUSED VkPipelineStageFlags2 stage)
{
   bool had_batch;
   struct panvk_batch *batch = open_batch(cmd, &had_batch);
   batch->needs_job_req_cycle_count = true;

   uint64_t report_addr = panvk_query_report_dev_addr(pool, query);
   panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_SYSTEM_TIMESTAMP,
                        report_addr, 0);

   uint64_t available_addr = panvk_query_available_dev_addr(pool, query);
   panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_32,
                        available_addr, 1);
   close_batch(cmd, had_batch);
}

void
panvk_per_arch(cmd_begin_end_query)(struct panvk_cmd_buffer *cmd,
                                    struct panvk_query_pool *pool,
                                    uint32_t query, VkQueryControlFlags flags,
                                    ASSERTED uint32_t index, bool end)
{
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);

   /* TODO: transform feedback */
   assert(index == 0);

   uint64_t report_addr = panvk_query_report_dev_addr(pool, query);

   /* Close to ensure we are sync and flush caches */
   if (end) {
      dev->vk.dispatch_table.CmdPipelineBarrier(
         panvk_cmd_buffer_to_handle(cmd), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 0,
         NULL);
   }

   bool had_batch;
   struct panvk_batch *batch = open_batch(cmd, &had_batch);

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
         for (unsigned i = 0; i < pool->reports_per_query; i++) {
            panvk_emit_write_job(
               cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64,
               report_addr + i * sizeof(struct panvk_query_report), 0);
         }
      }
      break;
   }
   default:
      unreachable("Unsupported query type");
   }

   if (end) {
      uint64_t available_addr = panvk_query_available_dev_addr(pool, query);
      panvk_emit_write_job(cmd, batch, MALI_WRITE_VALUE_TYPE_IMMEDIATE_32,
                           available_addr, 1);
   }

   close_batch(cmd, had_batch);
}
