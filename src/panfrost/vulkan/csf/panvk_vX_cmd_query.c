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

void panvk_per_arch(cmd_write_timestamp)(struct panvk_cmd_buffer *cmd,
                                         struct panvk_query_pool *pool,
                                         uint32_t query,
                                         UNUSED VkPipelineStageFlags2 stage)
{
   panvk_stub();
}

void
panvk_per_arch(cmd_begin_end_query)(struct panvk_cmd_buffer *cmd,
                                    struct panvk_query_pool *pool,
                                    uint32_t query, VkQueryControlFlags flags,
                                    ASSERTED uint32_t index, bool end)
{
   panvk_stub();
}
