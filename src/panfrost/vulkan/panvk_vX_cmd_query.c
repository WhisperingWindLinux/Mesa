/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "util/os_time.h"

#include "nir_builder.h"

#include "vk_log.h"
#include "vk_meta.h"
#include "vk_pipeline.h"

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_meta.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_macros.h"
#include "panvk_query_pool.h"

static nir_def *
panvk_nir_available_dev_addr(nir_builder *b, nir_def *pool_addr, nir_def *query)
{
   nir_def *offset = nir_imul_imm(b, query, sizeof(uint32_t));
   return nir_iadd(b, pool_addr, nir_u2u64(b, offset));
}

static nir_def *
panvk_nir_query_report_dev_addr(nir_builder *b, nir_def *pool_addr,
                                nir_def *query_start, nir_def *query_stride,
                                nir_def *query)
{
   nir_def *offset =
      nir_iadd(b, query_start, nir_umul_2x32_64(b, query, query_stride));
   return nir_iadd(b, pool_addr, offset);
}

#define load_info(__b, __type, __field_name)                                   \
   nir_load_push_constant((__b), 1,                                            \
                          sizeof(((__type *)NULL)->__field_name) * 8,          \
                          nir_imm_int(b, offsetof(__type, __field_name)))

static void
nir_write_query_result(nir_builder *b, nir_def *dst_addr, nir_def *idx,
                       nir_def *flags, nir_def *result)
{
   assert(result->num_components == 1);
   assert(result->bit_size == 64);

   nir_push_if(b, nir_test_mask(b, flags, VK_QUERY_RESULT_64_BIT));
   {
      nir_def *offset = nir_i2i64(b, nir_imul_imm(b, idx, 8));
      nir_store_global(b, nir_iadd(b, dst_addr, offset), 8, result, 0x1);
   }
   nir_push_else(b, NULL);
   {
      nir_def *result32 = nir_u2u32(b, result);
      nir_def *offset = nir_i2i64(b, nir_imul_imm(b, idx, 4));
      nir_store_global(b, nir_iadd(b, dst_addr, offset), 4, result32, 0x1);
   }
   nir_pop_if(b, NULL);
}

static void
nir_write_occlusion_query_result(nir_builder *b, nir_def *dst_addr,
                                 nir_def *idx, nir_def *flags,
                                 nir_def *report_addr, unsigned core_count)
{
   nir_def *value = nir_imm_int64(b, 0);

   for (unsigned core_idx = 0; core_idx < core_count; core_idx++) {
      /* Start values start at the second entry */
      unsigned report_offset = core_idx * sizeof(struct panvk_query_report);

      value = nir_iadd(
         b, value,
         nir_load_global(
            b, nir_iadd(b, report_addr, nir_imm_int64(b, report_offset)), 8, 1,
            64));
   }

   nir_write_query_result(b, dst_addr, idx, flags, value);
}

struct panvk_copy_query_push {
   uint64_t pool_addr;
   uint32_t query_start;
   uint32_t query_stride;
   uint32_t first_query;
   uint32_t query_count;
   uint64_t dst_addr;
   uint64_t dst_stride;
   uint32_t flags;
};

static void
panvk_nir_copy_query(nir_builder *b, VkQueryType query_type,
                     unsigned core_count, nir_def *i)
{
   nir_def *pool_addr = load_info(b, struct panvk_copy_query_push, pool_addr);
   nir_def *query_start =
      nir_u2u64(b, load_info(b, struct panvk_copy_query_push, query_start));
   nir_def *query_stride =
      load_info(b, struct panvk_copy_query_push, query_stride);
   nir_def *first_query =
      load_info(b, struct panvk_copy_query_push, first_query);
   nir_def *dst_addr = load_info(b, struct panvk_copy_query_push, dst_addr);
   nir_def *dst_stride = load_info(b, struct panvk_copy_query_push, dst_stride);
   nir_def *flags = load_info(b, struct panvk_copy_query_push, flags);

   nir_def *query = nir_iadd(b, first_query, i);

   nir_def *avail_addr = panvk_nir_available_dev_addr(b, pool_addr, query);
   nir_def *available = nir_i2b(b, nir_load_global(b, avail_addr, 4, 1, 32));

   nir_def *partial = nir_test_mask(b, flags, VK_QUERY_RESULT_PARTIAL_BIT);
   nir_def *write_results = nir_ior(b, available, partial);

   nir_def *report_addr = panvk_nir_query_report_dev_addr(
      b, pool_addr, query_start, query_stride, query);
   nir_def *dst_offset = nir_imul(b, nir_u2u64(b, i), dst_stride);

   nir_push_if(b, write_results);
   {
      switch (query_type) {
      case VK_QUERY_TYPE_OCCLUSION: {
         nir_write_occlusion_query_result(b, nir_iadd(b, dst_addr, dst_offset),
                                          nir_imm_int(b, 0), flags, report_addr,
                                          core_count);
         break;
      }
      case VK_QUERY_TYPE_TIMESTAMP: {
         nir_def *value = nir_load_global(b, report_addr, 8, 1, 64);
         nir_write_query_result(b, nir_iadd(b, dst_addr, dst_offset),
                                nir_imm_int(b, 0), flags, value);

         break;
      }
      default:
         unreachable("Unsupported query type");
      }
   }
   nir_pop_if(b, NULL);

   nir_push_if(b,
               nir_test_mask(b, flags, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   {
      nir_write_query_result(b, nir_iadd(b, dst_addr, dst_offset),
                             nir_imm_int(b, 1), flags, nir_b2i64(b, available));
   }
   nir_pop_if(b, NULL);
}

static nir_shader *
build_copy_queries_shader(VkQueryType query_type, uint32_t max_threads_per_wg,
                          unsigned core_count)
{
   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL,
      "panvk-meta-copy-queries(query_type=%d,core_count=%u)", query_type,
      core_count);
   nir_builder *b = &build;

   b->shader->info.workgroup_size[0] = max_threads_per_wg;
   nir_def *wg_id = nir_load_workgroup_id(b);
   nir_def *i =
      nir_iadd(b, nir_load_subgroup_invocation(b),
               nir_imul_imm(b, nir_channel(b, wg_id, 0), max_threads_per_wg));

   nir_def *query_count =
      load_info(b, struct panvk_copy_query_push, query_count);
   nir_push_if(b, nir_ilt(b, i, query_count));
   {
      panvk_nir_copy_query(b, query_type, core_count, i);
   }
   nir_pop_if(b, NULL);

   return build.shader;
}

static VkResult
get_copy_queries_pipeline(struct panvk_device *dev, VkQueryType query_type,
                          const char *key, size_t key_size,
                          VkPipelineLayout layout, VkPipeline *pipeline_out)
{
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);

   unsigned core_count;
   panfrost_query_core_count(&phys_dev->kmod.props, &core_count);
   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = build_copy_queries_shader(
         query_type, phys_dev->kmod.props.max_threads_per_wg, core_count),
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &nir_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, key,
                                          key_size, pipeline_out);
}

static void
panvk_meta_copy_query_pool_results(struct panvk_cmd_buffer *cmd,
                                   struct panvk_query_pool *pool,
                                   uint32_t first_query, uint32_t query_count,
                                   uint64_t dst_addr, uint64_t dst_stride,
                                   VkQueryResultFlags flags)
{
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   VkResult result;

   const struct panvk_copy_query_push push = {
      .pool_addr = panvk_priv_mem_dev_addr(pool->mem),
      .query_start = pool->query_start,
      .query_stride = pool->query_stride,
      .first_query = first_query,
      .query_count = query_count,
      .dst_addr = dst_addr,
      .dst_stride = dst_stride,
      .flags = flags,
   };

   char key[256];
   snprintf(key, sizeof(key),
            "panvk-meta-copy-query-pool-results(query_type=%d)",
            pool->vk.query_type);

   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(push),
   };
   VkPipelineLayout layout;
   result = vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, NULL, &push_range,
                                        key, sizeof(key), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkPipeline pipeline = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(key));

   if (pipeline == VK_NULL_HANDLE) {
      result = get_copy_queries_pipeline(dev, pool->vk.query_type, key,
                                         sizeof(key), layout, &pipeline);

      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
   }

   /* Save previous cmd state */
   struct panvk_cmd_meta_compute_save_ctx save = {0};
   panvk_per_arch(cmd_meta_compute_start)(cmd, &save);

   dev->vk.dispatch_table.CmdBindPipeline(panvk_cmd_buffer_to_handle(cmd),
                                          VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pipeline);

   dev->vk.dispatch_table.CmdPushConstants(panvk_cmd_buffer_to_handle(cmd),
                                           layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                           0, sizeof(push), &push);

   dev->vk.dispatch_table.CmdDispatchBase(
      panvk_cmd_buffer_to_handle(cmd), 0, 0, 0,
      DIV_ROUND_UP(query_count, phys_dev->kmod.props.max_threads_per_wg), 1, 1);

   /* Restore previous cmd state */
   panvk_per_arch(cmd_meta_compute_end)(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyQueryPoolResults)(
   VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery,
   uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset,
   VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);
   VK_FROM_HANDLE(panvk_buffer, dst_buffer, dstBuffer);
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);

   /* The Vulkan 1.3.293 spec says:
    *
    *    "The first synchronization scope includes all commands which reference
    * the queries in queryPool indicated by query that occur earlier in
    * submission order."
    *
    *    "The second synchronization scope includes all commands which reference
    * the queries in queryPool indicated by query that occur later in submission
    * order."
    *
    *    "vkCmdCopyQueryPoolResults is considered to be a transfer operation,
    *    and its writes to buffer memory must be synchronized using
    *    VK_PIPELINE_STAGE_TRANSFER_BIT and VK_ACCESS_TRANSFER_WRITE_BIT before
    *    using the results."
    *
    */
   const VkBufferMemoryBarrier pre_buf_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = NULL,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = dstBuffer,
      .offset = dstOffset,
      .size = panvk_buffer_range(dst_buffer, dstOffset, VK_WHOLE_SIZE),
   };

   unsigned src_mask = flags & VK_QUERY_RESULT_WAIT_BIT
                          ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                          : VK_PIPELINE_STAGE_TRANSFER_BIT;

   /* XXX: Revisit this, we might need more here */
   dev->vk.dispatch_table.CmdPipelineBarrier(
      commandBuffer, src_mask, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
      1, &pre_buf_barrier, 0, NULL);

   uint64_t dst_addr = panvk_buffer_gpu_ptr(dst_buffer, dstOffset);
   panvk_meta_copy_query_pool_results(cmd, pool, firstQuery, queryCount,
                                      dst_addr, stride, flags);

   /* XXX: Revisit this, we might need more here */
   const VkBufferMemoryBarrier post_buf_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .pNext = NULL,
      .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = dstBuffer,
      .offset = dstOffset,
      .size = panvk_buffer_range(dst_buffer, dstOffset, VK_WHOLE_SIZE),
   };
   dev->vk.dispatch_table.CmdPipelineBarrier(
      commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &post_buf_barrier, 0,
      NULL);
}

#define load_info(__b, __type, __field_name)                                   \
   nir_load_push_constant((__b), 1,                                            \
                          sizeof(((__type *)NULL)->__field_name) * 8,          \
                          nir_imm_int(b, offsetof(__type, __field_name)))

struct panvk_clear_query_push {
   uint64_t pool_addr;
   uint32_t query_start;
   uint32_t query_stride;
   uint32_t first_query;
   uint32_t query_count;
   uint32_t reports_per_query;
   uint32_t availaible_value;
};

static void
panvk_nir_clear_query(nir_builder *b, nir_def *i)
{
   nir_def *pool_addr = load_info(b, struct panvk_clear_query_push, pool_addr);
   nir_def *query_start =
      nir_u2u64(b, load_info(b, struct panvk_clear_query_push, query_start));
   nir_def *query_stride =
      load_info(b, struct panvk_clear_query_push, query_stride);
   nir_def *first_query =
      load_info(b, struct panvk_clear_query_push, first_query);
   nir_def *reports_per_query =
      load_info(b, struct panvk_clear_query_push, reports_per_query);
   nir_def *avail_value =
      load_info(b, struct panvk_clear_query_push, availaible_value);

   nir_def *query = nir_iadd(b, first_query, i);

   nir_def *avail_addr = panvk_nir_available_dev_addr(b, pool_addr, query);
   nir_def *report_addr = panvk_nir_query_report_dev_addr(
      b, pool_addr, query_start, query_stride, query);

   nir_store_global(b, avail_addr, 4, avail_value, 0x1);

   nir_def *zero = nir_imm_int64(b, 0);
   nir_variable *r = nir_local_variable_create(b->impl, glsl_uint_type(), "r");
   nir_store_var(b, r, nir_imm_int(b, 0), 0x1);

   uint32_t qwords_per_report =
      DIV_ROUND_UP(sizeof(struct panvk_query_report), sizeof(uint64_t));

   nir_push_loop(b);
   {
      nir_def *report_idx = nir_load_var(b, r);
      nir_break_if(b, nir_ige(b, report_idx, reports_per_query));

      nir_def *base_addr = nir_iadd(
         b, report_addr,
         nir_i2i64(
            b, nir_imul_imm(b, report_idx, sizeof(struct panvk_query_report))));

      for (uint32_t y = 0; y < qwords_per_report; y++) {
         nir_def *addr = nir_iadd_imm(b, base_addr, y * sizeof(uint64_t));
         nir_store_global(b, addr, 8, zero, 0x1);
      }

      nir_store_var(b, r, nir_iadd_imm(b, report_idx, 1), 0x1);
   }
   nir_pop_loop(b, NULL);
}

static nir_shader *
build_clear_queries_shader(uint32_t max_threads_per_wg)
{
   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "panvk-meta-clear-queries");
   nir_builder *b = &build;

   b->shader->info.workgroup_size[0] = max_threads_per_wg;
   nir_def *wg_id = nir_load_workgroup_id(b);
   nir_def *i =
      nir_iadd(b, nir_load_subgroup_invocation(b),
               nir_imul_imm(b, nir_channel(b, wg_id, 0), max_threads_per_wg));

   nir_def *query_count =
      load_info(b, struct panvk_clear_query_push, query_count);
   nir_push_if(b, nir_ilt(b, i, query_count));
   {
      panvk_nir_clear_query(b, i);
   }
   nir_pop_if(b, NULL);

   return build.shader;
}

static VkResult
get_clear_queries_pipeline(struct panvk_device *dev, const char *key,
                           size_t key_size, VkPipelineLayout layout,
                           VkPipeline *pipeline_out)
{
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir =
         build_clear_queries_shader(phys_dev->kmod.props.max_threads_per_wg),
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &nir_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, key,
                                          key_size, pipeline_out);
}

static void
panvk_emit_clear_queries(struct panvk_cmd_buffer *cmd,
                         struct panvk_query_pool *pool, bool availaible,
                         uint32_t first_query, uint32_t query_count)
{
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   VkResult result;

   const struct panvk_clear_query_push push = {
      .pool_addr = panvk_priv_mem_dev_addr(pool->mem),
      .query_start = pool->query_start,
      .query_stride = pool->query_stride,
      .first_query = first_query,
      .query_count = query_count,
      .reports_per_query = pool->reports_per_query,
      .availaible_value = availaible};

   const char key[] = "panvk-meta-clear-query-pool";
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(push),
   };
   VkPipelineLayout layout;
   result = vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, NULL, &push_range,
                                        key, sizeof(key), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkPipeline pipeline = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(key));

   if (pipeline == VK_NULL_HANDLE) {
      result =
         get_clear_queries_pipeline(dev, key, sizeof(key), layout, &pipeline);

      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
   }

   /* Save previous cmd state */
   struct panvk_cmd_meta_compute_save_ctx save = {0};
   panvk_per_arch(cmd_meta_compute_start)(cmd, &save);

   /* XXX: Narrow this */
   const VkMemoryBarrier pre_barrier = {
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };

   dev->vk.dispatch_table.CmdPipelineBarrier(
      panvk_cmd_buffer_to_handle(cmd),
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &pre_barrier, 0, NULL, 0,
      NULL);

   dev->vk.dispatch_table.CmdBindPipeline(panvk_cmd_buffer_to_handle(cmd),
                                          VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pipeline);

   dev->vk.dispatch_table.CmdPushConstants(panvk_cmd_buffer_to_handle(cmd),
                                           layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                           0, sizeof(push), &push);

   dev->vk.dispatch_table.CmdDispatchBase(
      panvk_cmd_buffer_to_handle(cmd), 0, 0, 0,
      DIV_ROUND_UP(query_count, phys_dev->kmod.props.max_threads_per_wg), 1, 1);

   /* XXX: Narrow this */
   const VkMemoryBarrier post_barrier = {
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };

   dev->vk.dispatch_table.CmdPipelineBarrier(
      panvk_cmd_buffer_to_handle(cmd), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      0, 1, &post_barrier, 0, NULL, 0, NULL);

   /* Restore previous cmd state */
   panvk_per_arch(cmd_meta_compute_end)(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdResetQueryPool)(VkCommandBuffer commandBuffer,
                                  VkQueryPool queryPool, uint32_t firstQuery,
                                  uint32_t queryCount)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   if (queryCount == 0)
      return;

   panvk_emit_clear_queries(cmd, pool, false, firstQuery, queryCount);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdWriteTimestamp2)(VkCommandBuffer commandBuffer,
                                   VkPipelineStageFlags2 stage,
                                   VkQueryPool queryPool, uint32_t query)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   panvk_per_arch(cmd_write_timestamp)(cmd, pool, query, stage);

   /* From the Vulkan spec:
    *
    *   "If vkCmdWriteTimestamp2 is called while executing a render pass
    *    instance that has multiview enabled, the timestamp uses N consecutive
    *    query indices in the query pool (starting at query) where N is the
    *    number of bits set in the view mask of the subpass the command is
    *    executed in. The resulting query values are determined by an
    *    implementation-dependent choice of one of the following behaviors:"
    *
    */
   uint32_t view_mask = 1; /* TODO: multiview */
   if (view_mask != 0) {
      const uint32_t num_queries = util_bitcount(view_mask);
      if (num_queries > 1)
         panvk_emit_clear_queries(cmd, pool, true, query + 1, num_queries - 1);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginQueryIndexedEXT)(VkCommandBuffer commandBuffer,
                                        VkQueryPool queryPool, uint32_t query,
                                        VkQueryControlFlags flags,
                                        uint32_t index)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   panvk_per_arch(cmd_begin_end_query)(cmd, pool, query, flags, index, false);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndQueryIndexedEXT)(VkCommandBuffer commandBuffer,
                                      VkQueryPool queryPool, uint32_t query,
                                      uint32_t index)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   panvk_per_arch(cmd_begin_end_query)(cmd, pool, query, 0, index, true);

   /* From the Vulkan spec:
    *
    *   "If queries are used while executing a render pass instance that has
    *    multiview enabled, the query uses N consecutive query indices in
    *    the query pool (starting at query) where N is the number of bits set
    *    in the view mask in the subpass the query is used in. How the
    *    numerical results of the query are distributed among the queries is
    *    implementation-dependent."
    *
    */
   uint32_t view_mask = 1; /* TODO: multiview */
   if (view_mask != 0) {
      const uint32_t num_queries = util_bitcount(view_mask);
      if (num_queries > 1)
         panvk_emit_clear_queries(cmd, pool, true, query + 1, num_queries - 1);
   }
}
