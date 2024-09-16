/*
 * Copyright © 2024 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_QUERY_POOL_H
#define PANVK_QUERY_POOL_H

#include <stdint.h>

#include "panvk_mempool.h"
#include "vk_query_pool.h"

struct panvk_query_report {
   uint64_t value;
};

static_assert(sizeof(struct panvk_query_report) % 8 == 0,
              "panvk_query_report size should be aligned to 8");

struct panvk_query_pool {
   struct vk_query_pool vk;

   uint32_t query_start;
   uint32_t query_stride;
   uint32_t reports_per_query;

   struct panvk_priv_mem mem;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

static uint64_t
panvk_query_available_dev_addr(struct panvk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return panvk_priv_mem_dev_addr(pool->mem) + query * sizeof(uint32_t);
}

static uint32_t *
panvk_query_available_host_addr(struct panvk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return (uint32_t *)panvk_priv_mem_host_addr(pool->mem) + query;
}

static uint64_t
panvk_query_offset(struct panvk_query_pool *pool, uint32_t query)
{
   assert(query < pool->vk.query_count);
   return pool->query_start + query * pool->query_stride;
}

static uint64_t
panvk_query_report_dev_addr(struct panvk_query_pool *pool, uint32_t query)
{
   return panvk_priv_mem_dev_addr(pool->mem) + panvk_query_offset(pool, query);
}

static struct panvk_query_report *
panvk_query_report_host_addr(struct panvk_query_pool *pool, uint32_t query)
{
   return (void *)((char *)panvk_priv_mem_host_addr(pool->mem) +
                   panvk_query_offset(pool, query));
}

#endif