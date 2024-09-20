/*
 * Generated using libdrm_amdgpu_gen_loader.py script
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef AC_LIBDRM_AMDGPU_LOADER_H
#define AC_LIBDRM_AMDGPU_LOADER_H
#include <amdgpu.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
        
typedef int  (*amdgpu_device_initialize_type)(int fd, uint32_t *major_version, uint32_t *minor_version, amdgpu_device_handle *device_handle);
typedef int  (*amdgpu_device_initialize2_type)(int fd, _Bool deduplicate_device, uint32_t *major_version, uint32_t *minor_version, amdgpu_device_handle *device_handle);
typedef int  (*amdgpu_device_deinitialize_type)(amdgpu_device_handle device_handle);
typedef int  (*amdgpu_device_get_fd_type)(amdgpu_device_handle device_handle);
typedef int  (*amdgpu_bo_alloc_type)(amdgpu_device_handle dev, struct amdgpu_bo_alloc_request *alloc_buffer, amdgpu_bo_handle *buf_handle);
typedef int  (*amdgpu_bo_set_metadata_type)(amdgpu_bo_handle buf_handle, struct amdgpu_bo_metadata *info);
typedef int  (*amdgpu_bo_query_info_type)(amdgpu_bo_handle buf_handle, struct amdgpu_bo_info *info);
typedef int  (*amdgpu_bo_export_type)(amdgpu_bo_handle buf_handle, enum amdgpu_bo_handle_type type, uint32_t *shared_handle);
typedef int  (*amdgpu_bo_import_type)(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type, uint32_t shared_handle, struct amdgpu_bo_import_result *output);
typedef int  (*amdgpu_create_bo_from_user_mem_type)(amdgpu_device_handle dev, void *cpu, uint64_t size, amdgpu_bo_handle *buf_handle);
typedef int  (*amdgpu_find_bo_by_cpu_mapping_type)(amdgpu_device_handle dev, void *cpu, uint64_t size, amdgpu_bo_handle *buf_handle, uint64_t *offset_in_bo);
typedef int  (*amdgpu_bo_free_type)(amdgpu_bo_handle buf_handle);
typedef void  (*amdgpu_bo_inc_ref_type)(amdgpu_bo_handle bo);
typedef int  (*amdgpu_bo_cpu_map_type)(amdgpu_bo_handle buf_handle, void **cpu);
typedef int  (*amdgpu_bo_cpu_unmap_type)(amdgpu_bo_handle buf_handle);
typedef int  (*amdgpu_bo_wait_for_idle_type)(amdgpu_bo_handle buf_handle, uint64_t timeout_ns, _Bool *buffer_busy);
typedef int  (*amdgpu_bo_list_create_raw_type)(amdgpu_device_handle dev, uint32_t number_of_buffers, struct drm_amdgpu_bo_list_entry *buffers, uint32_t *result);
typedef int  (*amdgpu_bo_list_destroy_raw_type)(amdgpu_device_handle dev, uint32_t bo_list);
typedef int  (*amdgpu_bo_list_create_type)(amdgpu_device_handle dev, uint32_t number_of_resources, amdgpu_bo_handle *resources, uint8_t *resource_prios, amdgpu_bo_list_handle *result);
typedef int  (*amdgpu_bo_list_destroy_type)(amdgpu_bo_list_handle handle);
typedef int  (*amdgpu_bo_list_update_type)(amdgpu_bo_list_handle handle, uint32_t number_of_resources, amdgpu_bo_handle *resources, uint8_t *resource_prios);
typedef int  (*amdgpu_cs_ctx_create2_type)(amdgpu_device_handle dev, uint32_t priority, amdgpu_context_handle *context);
typedef int  (*amdgpu_cs_ctx_create_type)(amdgpu_device_handle dev, amdgpu_context_handle *context);
typedef int  (*amdgpu_cs_ctx_free_type)(amdgpu_context_handle context);
typedef int  (*amdgpu_cs_ctx_override_priority_type)(amdgpu_device_handle dev, amdgpu_context_handle context, int master_fd, unsigned priority);
typedef int  (*amdgpu_cs_ctx_stable_pstate_type)(amdgpu_context_handle context, uint32_t op, uint32_t flags, uint32_t *out_flags);
typedef int  (*amdgpu_cs_query_reset_state_type)(amdgpu_context_handle context, uint32_t *state, uint32_t *hangs);
typedef int  (*amdgpu_cs_query_reset_state2_type)(amdgpu_context_handle context, uint64_t *flags);
typedef int  (*amdgpu_cs_submit_type)(amdgpu_context_handle context, uint64_t flags, struct amdgpu_cs_request *ibs_request, uint32_t number_of_requests);
typedef int  (*amdgpu_cs_query_fence_status_type)(struct amdgpu_cs_fence *fence, uint64_t timeout_ns, uint64_t flags, uint32_t *expired);
typedef int  (*amdgpu_cs_wait_fences_type)(struct amdgpu_cs_fence *fences, uint32_t fence_count, _Bool wait_all, uint64_t timeout_ns, uint32_t *status, uint32_t *first);
typedef int  (*amdgpu_query_buffer_size_alignment_type)(amdgpu_device_handle dev, struct amdgpu_buffer_size_alignments *info);
typedef int  (*amdgpu_query_firmware_version_type)(amdgpu_device_handle dev, unsigned fw_type, unsigned ip_instance, unsigned index, uint32_t *version, uint32_t *feature);
typedef int  (*amdgpu_query_hw_ip_count_type)(amdgpu_device_handle dev, unsigned type, uint32_t *count);
typedef int  (*amdgpu_query_hw_ip_info_type)(amdgpu_device_handle dev, unsigned type, unsigned ip_instance, struct drm_amdgpu_info_hw_ip *info);
typedef int  (*amdgpu_query_heap_info_type)(amdgpu_device_handle dev, uint32_t heap, uint32_t flags, struct amdgpu_heap_info *info);
typedef int  (*amdgpu_query_crtc_from_id_type)(amdgpu_device_handle dev, unsigned id, int32_t *result);
typedef int  (*amdgpu_query_gpu_info_type)(amdgpu_device_handle dev, struct amdgpu_gpu_info *info);
typedef int  (*amdgpu_query_info_type)(amdgpu_device_handle dev, unsigned info_id, unsigned size, void *value);
typedef int  (*amdgpu_query_sw_info_type)(amdgpu_device_handle dev, enum amdgpu_sw_info info, void *value);
typedef int  (*amdgpu_query_gds_info_type)(amdgpu_device_handle dev, struct amdgpu_gds_resource_info *gds_info);
typedef int  (*amdgpu_query_sensor_info_type)(amdgpu_device_handle dev, unsigned sensor_type, unsigned size, void *value);
typedef int  (*amdgpu_query_video_caps_info_type)(amdgpu_device_handle dev, unsigned cap_type, unsigned size, void *value);
typedef int  (*amdgpu_query_gpuvm_fault_info_type)(amdgpu_device_handle dev, unsigned size, void *value);
typedef int  (*amdgpu_read_mm_registers_type)(amdgpu_device_handle dev, unsigned dword_offset, unsigned count, uint32_t instance, uint32_t flags, uint32_t *values);
typedef int  (*amdgpu_va_range_alloc_type)(amdgpu_device_handle dev, enum amdgpu_gpu_va_range va_range_type, uint64_t size, uint64_t va_base_alignment, uint64_t va_base_required, uint64_t *va_base_allocated, amdgpu_va_handle *va_range_handle, uint64_t flags);
typedef int  (*amdgpu_va_range_free_type)(amdgpu_va_handle va_range_handle);
typedef uint64_t  (*amdgpu_va_get_start_addr_type)(amdgpu_va_handle va_handle);
typedef int  (*amdgpu_va_range_query_type)(amdgpu_device_handle dev, enum amdgpu_gpu_va_range type, uint64_t *start, uint64_t *end);
typedef amdgpu_va_manager_handle  (*amdgpu_va_manager_alloc_type)();
typedef void  (*amdgpu_va_manager_init_type)(amdgpu_va_manager_handle va_mgr, uint64_t low_va_offset, uint64_t low_va_max, uint64_t high_va_offset, uint64_t high_va_max, uint32_t virtual_address_alignment);
typedef void  (*amdgpu_va_manager_deinit_type)(amdgpu_va_manager_handle va_mgr);
typedef int  (*amdgpu_va_range_alloc2_type)(amdgpu_va_manager_handle va_mgr, enum amdgpu_gpu_va_range va_range_type, uint64_t size, uint64_t va_base_alignment, uint64_t va_base_required, uint64_t *va_base_allocated, amdgpu_va_handle *va_range_handle, uint64_t flags);
typedef int  (*amdgpu_bo_va_op_type)(amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr, uint64_t flags, uint32_t ops);
typedef int  (*amdgpu_bo_va_op_raw_type)(amdgpu_device_handle dev, amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr, uint64_t flags, uint32_t ops);
typedef int  (*amdgpu_cs_create_semaphore_type)(amdgpu_semaphore_handle *sem);
typedef int  (*amdgpu_cs_signal_semaphore_type)(amdgpu_context_handle ctx, uint32_t ip_type, uint32_t ip_instance, uint32_t ring, amdgpu_semaphore_handle sem);
typedef int  (*amdgpu_cs_wait_semaphore_type)(amdgpu_context_handle ctx, uint32_t ip_type, uint32_t ip_instance, uint32_t ring, amdgpu_semaphore_handle sem);
typedef int  (*amdgpu_cs_destroy_semaphore_type)(amdgpu_semaphore_handle sem);
typedef const char* (*amdgpu_get_marketing_name_type)(amdgpu_device_handle dev);
typedef int  (*amdgpu_cs_create_syncobj2_type)(amdgpu_device_handle dev, uint32_t flags, uint32_t *syncobj);
typedef int  (*amdgpu_cs_create_syncobj_type)(amdgpu_device_handle dev, uint32_t *syncobj);
typedef int  (*amdgpu_cs_destroy_syncobj_type)(amdgpu_device_handle dev, uint32_t syncobj);
typedef int  (*amdgpu_cs_syncobj_reset_type)(amdgpu_device_handle dev, const uint32_t *syncobjs, uint32_t syncobj_count);
typedef int  (*amdgpu_cs_syncobj_signal_type)(amdgpu_device_handle dev, const uint32_t *syncobjs, uint32_t syncobj_count);
typedef int  (*amdgpu_cs_syncobj_timeline_signal_type)(amdgpu_device_handle dev, const uint32_t *syncobjs, uint64_t *points, uint32_t syncobj_count);
typedef int  (*amdgpu_cs_syncobj_wait_type)(amdgpu_device_handle dev, uint32_t *handles, unsigned num_handles, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled);
typedef int  (*amdgpu_cs_syncobj_timeline_wait_type)(amdgpu_device_handle dev, uint32_t *handles, uint64_t *points, unsigned num_handles, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled);
typedef int  (*amdgpu_cs_syncobj_query_type)(amdgpu_device_handle dev, uint32_t *handles, uint64_t *points, unsigned num_handles);
typedef int  (*amdgpu_cs_syncobj_query2_type)(amdgpu_device_handle dev, uint32_t *handles, uint64_t *points, unsigned num_handles, uint32_t flags);
typedef int  (*amdgpu_cs_export_syncobj_type)(amdgpu_device_handle dev, uint32_t syncobj, int *shared_fd);
typedef int  (*amdgpu_cs_import_syncobj_type)(amdgpu_device_handle dev, int shared_fd, uint32_t *syncobj);
typedef int  (*amdgpu_cs_syncobj_export_sync_file_type)(amdgpu_device_handle dev, uint32_t syncobj, int *sync_file_fd);
typedef int  (*amdgpu_cs_syncobj_import_sync_file_type)(amdgpu_device_handle dev, uint32_t syncobj, int sync_file_fd);
typedef int  (*amdgpu_cs_syncobj_export_sync_file2_type)(amdgpu_device_handle dev, uint32_t syncobj, uint64_t point, uint32_t flags, int *sync_file_fd);
typedef int  (*amdgpu_cs_syncobj_import_sync_file2_type)(amdgpu_device_handle dev, uint32_t syncobj, uint64_t point, int sync_file_fd);
typedef int  (*amdgpu_cs_syncobj_transfer_type)(amdgpu_device_handle dev, uint32_t dst_handle, uint64_t dst_point, uint32_t src_handle, uint64_t src_point, uint32_t flags);
typedef int  (*amdgpu_cs_fence_to_handle_type)(amdgpu_device_handle dev, struct amdgpu_cs_fence *fence, uint32_t what, uint32_t *out_handle);
typedef int  (*amdgpu_cs_submit_raw_type)(amdgpu_device_handle dev, amdgpu_context_handle context, amdgpu_bo_list_handle bo_list_handle, int num_chunks, struct drm_amdgpu_cs_chunk *chunks, uint64_t *seq_no);
typedef int  (*amdgpu_cs_submit_raw2_type)(amdgpu_device_handle dev, amdgpu_context_handle context, uint32_t bo_list_handle, int num_chunks, struct drm_amdgpu_cs_chunk *chunks, uint64_t *seq_no);
typedef void  (*amdgpu_cs_chunk_fence_to_dep_type)(struct amdgpu_cs_fence *fence, struct drm_amdgpu_cs_chunk_dep *dep);
typedef void  (*amdgpu_cs_chunk_fence_info_to_data_type)(struct amdgpu_cs_fence_info *fence_info, struct drm_amdgpu_cs_chunk_data *data);
typedef int  (*amdgpu_vm_reserve_vmid_type)(amdgpu_device_handle dev, uint32_t flags);
typedef int  (*amdgpu_vm_unreserve_vmid_type)(amdgpu_device_handle dev, uint32_t flags);
struct libdrm_amdgpu {
   void *handle;
   int32_t refcount;
amdgpu_device_initialize_type device_initialize;
amdgpu_device_initialize2_type device_initialize2;
amdgpu_device_deinitialize_type device_deinitialize;
amdgpu_device_get_fd_type device_get_fd;
amdgpu_bo_alloc_type bo_alloc;
amdgpu_bo_set_metadata_type bo_set_metadata;
amdgpu_bo_query_info_type bo_query_info;
amdgpu_bo_export_type bo_export;
amdgpu_bo_import_type bo_import;
amdgpu_create_bo_from_user_mem_type create_bo_from_user_mem;
amdgpu_find_bo_by_cpu_mapping_type find_bo_by_cpu_mapping;
amdgpu_bo_free_type bo_free;
amdgpu_bo_inc_ref_type bo_inc_ref;
amdgpu_bo_cpu_map_type bo_cpu_map;
amdgpu_bo_cpu_unmap_type bo_cpu_unmap;
amdgpu_bo_wait_for_idle_type bo_wait_for_idle;
amdgpu_bo_list_create_raw_type bo_list_create_raw;
amdgpu_bo_list_destroy_raw_type bo_list_destroy_raw;
amdgpu_bo_list_create_type bo_list_create;
amdgpu_bo_list_destroy_type bo_list_destroy;
amdgpu_bo_list_update_type bo_list_update;
amdgpu_cs_ctx_create2_type cs_ctx_create2;
amdgpu_cs_ctx_create_type cs_ctx_create;
amdgpu_cs_ctx_free_type cs_ctx_free;
amdgpu_cs_ctx_override_priority_type cs_ctx_override_priority;
amdgpu_cs_ctx_stable_pstate_type cs_ctx_stable_pstate;
amdgpu_cs_query_reset_state_type cs_query_reset_state;
amdgpu_cs_query_reset_state2_type cs_query_reset_state2;
amdgpu_cs_submit_type cs_submit;
amdgpu_cs_query_fence_status_type cs_query_fence_status;
amdgpu_cs_wait_fences_type cs_wait_fences;
amdgpu_query_buffer_size_alignment_type query_buffer_size_alignment;
amdgpu_query_firmware_version_type query_firmware_version;
amdgpu_query_hw_ip_count_type query_hw_ip_count;
amdgpu_query_hw_ip_info_type query_hw_ip_info;
amdgpu_query_heap_info_type query_heap_info;
amdgpu_query_crtc_from_id_type query_crtc_from_id;
amdgpu_query_gpu_info_type query_gpu_info;
amdgpu_query_info_type query_info;
amdgpu_query_sw_info_type query_sw_info;
amdgpu_query_gds_info_type query_gds_info;
amdgpu_query_sensor_info_type query_sensor_info;
amdgpu_query_video_caps_info_type query_video_caps_info;
amdgpu_query_gpuvm_fault_info_type query_gpuvm_fault_info;
amdgpu_read_mm_registers_type read_mm_registers;
amdgpu_va_range_alloc_type va_range_alloc;
amdgpu_va_range_free_type va_range_free;
amdgpu_va_get_start_addr_type va_get_start_addr;
amdgpu_va_range_query_type va_range_query;
amdgpu_va_manager_alloc_type va_manager_alloc;
amdgpu_va_manager_init_type va_manager_init;
amdgpu_va_manager_deinit_type va_manager_deinit;
amdgpu_va_range_alloc2_type va_range_alloc2;
amdgpu_bo_va_op_type bo_va_op;
amdgpu_bo_va_op_raw_type bo_va_op_raw;
amdgpu_cs_create_semaphore_type cs_create_semaphore;
amdgpu_cs_signal_semaphore_type cs_signal_semaphore;
amdgpu_cs_wait_semaphore_type cs_wait_semaphore;
amdgpu_cs_destroy_semaphore_type cs_destroy_semaphore;
amdgpu_get_marketing_name_type get_marketing_name;
amdgpu_cs_create_syncobj2_type cs_create_syncobj2;
amdgpu_cs_create_syncobj_type cs_create_syncobj;
amdgpu_cs_destroy_syncobj_type cs_destroy_syncobj;
amdgpu_cs_syncobj_reset_type cs_syncobj_reset;
amdgpu_cs_syncobj_signal_type cs_syncobj_signal;
amdgpu_cs_syncobj_timeline_signal_type cs_syncobj_timeline_signal;
amdgpu_cs_syncobj_wait_type cs_syncobj_wait;
amdgpu_cs_syncobj_timeline_wait_type cs_syncobj_timeline_wait;
amdgpu_cs_syncobj_query_type cs_syncobj_query;
amdgpu_cs_syncobj_query2_type cs_syncobj_query2;
amdgpu_cs_export_syncobj_type cs_export_syncobj;
amdgpu_cs_import_syncobj_type cs_import_syncobj;
amdgpu_cs_syncobj_export_sync_file_type cs_syncobj_export_sync_file;
amdgpu_cs_syncobj_import_sync_file_type cs_syncobj_import_sync_file;
amdgpu_cs_syncobj_export_sync_file2_type cs_syncobj_export_sync_file2;
amdgpu_cs_syncobj_import_sync_file2_type cs_syncobj_import_sync_file2;
amdgpu_cs_syncobj_transfer_type cs_syncobj_transfer;
amdgpu_cs_fence_to_handle_type cs_fence_to_handle;
amdgpu_cs_submit_raw_type cs_submit_raw;
amdgpu_cs_submit_raw2_type cs_submit_raw2;
amdgpu_cs_chunk_fence_to_dep_type cs_chunk_fence_to_dep;
amdgpu_cs_chunk_fence_info_to_data_type cs_chunk_fence_info_to_data;
amdgpu_vm_reserve_vmid_type vm_reserve_vmid;
amdgpu_vm_unreserve_vmid_type vm_unreserve_vmid;
};

struct libdrm_amdgpu * ac_init_libdrm_amdgpu(void);

void ac_deinit_libdrm_amdgpu(struct libdrm_amdgpu *h);

struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio(void);

struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio_stubs(void);

#ifdef __cplusplus
}
#endif

#endif /* AC_LIBDRM_AMDGPU_LOADER_H */

