/*
 * Copyright 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "intel/perf/intel_perf.h"

struct eustall_config {
   struct intel_perf_query_eustall_result result;
   struct intel_device_info *devinfo;
   int drm_fd;
   uint8_t* buf;
   size_t buf_len;
   int record_size;

   int fd;
   size_t gpu_buf_size;
   uint64_t poll_period_ns;
   uint32_t sample_rate;
   uint32_t min_event_count;
};

struct eustall_config* eustall_setup(int drm_fd,
                                     struct intel_device_info* devinfo,
                                     uint64_t poll_period_ns);
bool eustall_sample(void *cfg);
void eustall_dump_results(void *cfg, FILE* file);
void eustall_close(void *cfg);
