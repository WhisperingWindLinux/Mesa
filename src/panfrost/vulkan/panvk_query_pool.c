/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "vk_log.h"

#include "pan_props.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_query_pool.h"

VKAPI_ATTR VkResult VKAPI_CALL
panvk_CreateQueryPool(VkDevice _device,
                      const VkQueryPoolCreateInfo *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(device->vk.physical);
   struct panvk_query_pool *pool;

   pool =
      vk_query_pool_create(&device->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* We place the availability first and then data */
   pool->query_start = align(pool->vk.query_count * sizeof(uint32_t),
                             sizeof(struct panvk_query_report));

   uint32_t reports_per_query;
   switch (pCreateInfo->queryType) {
   case VK_QUERY_TYPE_OCCLUSION: {
      /* The counter is per core */
      panfrost_query_core_count(&phys_dev->kmod.props, &reports_per_query);
      break;
   }
   case VK_QUERY_TYPE_TIMESTAMP:
      reports_per_query = 1;
      break;
   default:
      unreachable("Unsupported query type");
   }
   pool->reports_per_query = reports_per_query;
   pool->query_stride = reports_per_query * sizeof(struct panvk_query_report);

   if (pool->vk.query_count > 0) {
      struct panvk_pool_alloc_info alloc_info = {
         .size = pool->query_start + pool->query_stride * pool->vk.query_count,
         .alignment = sizeof(struct panvk_query_report),
      };
      pool->mem = panvk_pool_alloc_mem(&device->mempools.rw_nc, alloc_info);
      if (!pool->mem.bo) {
         vk_query_pool_destroy(&device->vk, pAllocator, &pool->vk);
         return vk_error(device, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   *pQueryPool = panvk_query_pool_to_handle(pool);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_DestroyQueryPool(VkDevice _device, VkQueryPool queryPool,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_query_pool, pool, queryPool);

   if (!pool)
      return;

   panvk_pool_free_mem(&device->mempools.rw_nc, pool->mem);
   vk_query_pool_destroy(&device->vk, pAllocator, &pool->vk);
}
