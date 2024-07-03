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

#endif