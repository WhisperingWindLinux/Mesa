/*
 * Copyright 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct intel_perf_config;
struct intel_perf_registers;
struct intel_perf_query_eustall_result;

uint64_t xe_perf_get_oa_format(struct intel_perf_config *perf);

bool xe_oa_metrics_available(struct intel_perf_config *perf, int fd, bool use_register_snapshots);

uint64_t xe_add_config(struct intel_perf_config *perf, int fd, const struct intel_perf_registers *config, const char *guid);
void xe_remove_config(struct intel_perf_config *perf, int fd, uint64_t config_id);

int xe_perf_stream_open(struct intel_perf_config *perf_config, int drm_fd,
                        uint32_t exec_id, uint64_t metrics_set_id,
                        uint64_t report_format, uint64_t period_exponent,
                        bool hold_preemption, bool enable);
int xe_perf_stream_set_state(int perf_stream_fd, bool enable);
int xe_perf_stream_set_metrics_id(int perf_stream_fd, uint64_t metrics_set_id);
int xe_perf_stream_read_samples(struct intel_perf_config *perf_config, int perf_stream_fd,
                                uint8_t *buffer, size_t buffer_len);
int xe_perf_eustall_stream_open(int drm_fd, size_t gpu_buf_size, uint64_t poll_period_ns,
                                uint32_t sample_rate, uint32_t min_event_count, bool enable);
int xe_perf_eustall_stream_record_size(int perf_stream_fd);
int xe_perf_eustall_stream_read_samples(int perf_stream_fd, uint8_t *buffer,
                                        size_t buffer_len, bool *overflow);
void xe_perf_eustall_accumulate_results(struct intel_perf_query_eustall_result *result,
                                        const uint8_t *start, const uint8_t *end,
                                        size_t record_size);
