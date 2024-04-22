
#include "libdrm_amdgpu_loader.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void ac_deinit_libdrm_amdgpu(struct libdrm_amdgpu *h) {
    if (h->handle)
        dlclose(h->handle);
    free(h);
}
struct libdrm_amdgpu * ac_init_libdrm_amdgpu(void) {
   struct libdrm_amdgpu *libdrm_amdgpu = calloc(1, sizeof(struct libdrm_amdgpu));
   void *libdrm = dlopen("libdrm_amdgpu.so.1", RTLD_NOW | RTLD_LOCAL);
   assert(libdrm);
        
libdrm_amdgpu->device_initialize = dlsym(libdrm, "amdgpu_device_initialize");
assert(libdrm_amdgpu->device_initialize != NULL);
libdrm_amdgpu->device_initialize2 = dlsym(libdrm, "amdgpu_device_initialize2");
assert(libdrm_amdgpu->device_initialize2 != NULL);
libdrm_amdgpu->device_deinitialize = dlsym(libdrm, "amdgpu_device_deinitialize");
assert(libdrm_amdgpu->device_deinitialize != NULL);
libdrm_amdgpu->device_get_fd = dlsym(libdrm, "amdgpu_device_get_fd");
assert(libdrm_amdgpu->device_get_fd != NULL);
libdrm_amdgpu->bo_alloc = dlsym(libdrm, "amdgpu_bo_alloc");
assert(libdrm_amdgpu->bo_alloc != NULL);
libdrm_amdgpu->bo_set_metadata = dlsym(libdrm, "amdgpu_bo_set_metadata");
assert(libdrm_amdgpu->bo_set_metadata != NULL);
libdrm_amdgpu->bo_query_info = dlsym(libdrm, "amdgpu_bo_query_info");
assert(libdrm_amdgpu->bo_query_info != NULL);
libdrm_amdgpu->bo_export = dlsym(libdrm, "amdgpu_bo_export");
assert(libdrm_amdgpu->bo_export != NULL);
libdrm_amdgpu->bo_import = dlsym(libdrm, "amdgpu_bo_import");
assert(libdrm_amdgpu->bo_import != NULL);
libdrm_amdgpu->create_bo_from_user_mem = dlsym(libdrm, "amdgpu_create_bo_from_user_mem");
assert(libdrm_amdgpu->create_bo_from_user_mem != NULL);
libdrm_amdgpu->find_bo_by_cpu_mapping = dlsym(libdrm, "amdgpu_find_bo_by_cpu_mapping");
assert(libdrm_amdgpu->find_bo_by_cpu_mapping != NULL);
libdrm_amdgpu->bo_free = dlsym(libdrm, "amdgpu_bo_free");
assert(libdrm_amdgpu->bo_free != NULL);
libdrm_amdgpu->bo_inc_ref = dlsym(libdrm, "amdgpu_bo_inc_ref");
assert(libdrm_amdgpu->bo_inc_ref != NULL);
libdrm_amdgpu->bo_cpu_map = dlsym(libdrm, "amdgpu_bo_cpu_map");
assert(libdrm_amdgpu->bo_cpu_map != NULL);
libdrm_amdgpu->bo_cpu_unmap = dlsym(libdrm, "amdgpu_bo_cpu_unmap");
assert(libdrm_amdgpu->bo_cpu_unmap != NULL);
libdrm_amdgpu->bo_wait_for_idle = dlsym(libdrm, "amdgpu_bo_wait_for_idle");
assert(libdrm_amdgpu->bo_wait_for_idle != NULL);
libdrm_amdgpu->bo_list_create_raw = dlsym(libdrm, "amdgpu_bo_list_create_raw");
assert(libdrm_amdgpu->bo_list_create_raw != NULL);
libdrm_amdgpu->bo_list_destroy_raw = dlsym(libdrm, "amdgpu_bo_list_destroy_raw");
assert(libdrm_amdgpu->bo_list_destroy_raw != NULL);
libdrm_amdgpu->bo_list_create = dlsym(libdrm, "amdgpu_bo_list_create");
assert(libdrm_amdgpu->bo_list_create != NULL);
libdrm_amdgpu->bo_list_destroy = dlsym(libdrm, "amdgpu_bo_list_destroy");
assert(libdrm_amdgpu->bo_list_destroy != NULL);
libdrm_amdgpu->bo_list_update = dlsym(libdrm, "amdgpu_bo_list_update");
assert(libdrm_amdgpu->bo_list_update != NULL);
libdrm_amdgpu->cs_ctx_create2 = dlsym(libdrm, "amdgpu_cs_ctx_create2");
assert(libdrm_amdgpu->cs_ctx_create2 != NULL);
libdrm_amdgpu->cs_ctx_create = dlsym(libdrm, "amdgpu_cs_ctx_create");
assert(libdrm_amdgpu->cs_ctx_create != NULL);
libdrm_amdgpu->cs_ctx_free = dlsym(libdrm, "amdgpu_cs_ctx_free");
assert(libdrm_amdgpu->cs_ctx_free != NULL);
libdrm_amdgpu->cs_ctx_override_priority = dlsym(libdrm, "amdgpu_cs_ctx_override_priority");
assert(libdrm_amdgpu->cs_ctx_override_priority != NULL);
libdrm_amdgpu->cs_ctx_stable_pstate = dlsym(libdrm, "amdgpu_cs_ctx_stable_pstate");
assert(libdrm_amdgpu->cs_ctx_stable_pstate != NULL);
libdrm_amdgpu->cs_query_reset_state = dlsym(libdrm, "amdgpu_cs_query_reset_state");
assert(libdrm_amdgpu->cs_query_reset_state != NULL);
libdrm_amdgpu->cs_query_reset_state2 = dlsym(libdrm, "amdgpu_cs_query_reset_state2");
assert(libdrm_amdgpu->cs_query_reset_state2 != NULL);
libdrm_amdgpu->cs_submit = dlsym(libdrm, "amdgpu_cs_submit");
assert(libdrm_amdgpu->cs_submit != NULL);
libdrm_amdgpu->cs_query_fence_status = dlsym(libdrm, "amdgpu_cs_query_fence_status");
assert(libdrm_amdgpu->cs_query_fence_status != NULL);
libdrm_amdgpu->cs_wait_fences = dlsym(libdrm, "amdgpu_cs_wait_fences");
assert(libdrm_amdgpu->cs_wait_fences != NULL);
libdrm_amdgpu->query_buffer_size_alignment = dlsym(libdrm, "amdgpu_query_buffer_size_alignment");
assert(libdrm_amdgpu->query_buffer_size_alignment != NULL);
libdrm_amdgpu->query_firmware_version = dlsym(libdrm, "amdgpu_query_firmware_version");
assert(libdrm_amdgpu->query_firmware_version != NULL);
libdrm_amdgpu->query_hw_ip_count = dlsym(libdrm, "amdgpu_query_hw_ip_count");
assert(libdrm_amdgpu->query_hw_ip_count != NULL);
libdrm_amdgpu->query_hw_ip_info = dlsym(libdrm, "amdgpu_query_hw_ip_info");
assert(libdrm_amdgpu->query_hw_ip_info != NULL);
libdrm_amdgpu->query_heap_info = dlsym(libdrm, "amdgpu_query_heap_info");
assert(libdrm_amdgpu->query_heap_info != NULL);
libdrm_amdgpu->query_crtc_from_id = dlsym(libdrm, "amdgpu_query_crtc_from_id");
assert(libdrm_amdgpu->query_crtc_from_id != NULL);
libdrm_amdgpu->query_gpu_info = dlsym(libdrm, "amdgpu_query_gpu_info");
assert(libdrm_amdgpu->query_gpu_info != NULL);
libdrm_amdgpu->query_info = dlsym(libdrm, "amdgpu_query_info");
assert(libdrm_amdgpu->query_info != NULL);
libdrm_amdgpu->query_sw_info = dlsym(libdrm, "amdgpu_query_sw_info");
assert(libdrm_amdgpu->query_sw_info != NULL);
libdrm_amdgpu->query_gds_info = dlsym(libdrm, "amdgpu_query_gds_info");
assert(libdrm_amdgpu->query_gds_info != NULL);
libdrm_amdgpu->query_sensor_info = dlsym(libdrm, "amdgpu_query_sensor_info");
assert(libdrm_amdgpu->query_sensor_info != NULL);
libdrm_amdgpu->query_video_caps_info = dlsym(libdrm, "amdgpu_query_video_caps_info");
assert(libdrm_amdgpu->query_video_caps_info != NULL);
libdrm_amdgpu->query_gpuvm_fault_info = dlsym(libdrm, "amdgpu_query_gpuvm_fault_info");
assert(libdrm_amdgpu->query_gpuvm_fault_info != NULL);
libdrm_amdgpu->read_mm_registers = dlsym(libdrm, "amdgpu_read_mm_registers");
assert(libdrm_amdgpu->read_mm_registers != NULL);
libdrm_amdgpu->va_range_alloc = dlsym(libdrm, "amdgpu_va_range_alloc");
assert(libdrm_amdgpu->va_range_alloc != NULL);
libdrm_amdgpu->va_range_free = dlsym(libdrm, "amdgpu_va_range_free");
assert(libdrm_amdgpu->va_range_free != NULL);
libdrm_amdgpu->va_get_start_addr = dlsym(libdrm, "amdgpu_va_get_start_addr");
assert(libdrm_amdgpu->va_get_start_addr != NULL);
libdrm_amdgpu->va_range_query = dlsym(libdrm, "amdgpu_va_range_query");
assert(libdrm_amdgpu->va_range_query != NULL);
libdrm_amdgpu->va_manager_alloc = dlsym(libdrm, "amdgpu_va_manager_alloc");
assert(libdrm_amdgpu->va_manager_alloc != NULL);
libdrm_amdgpu->va_manager_init = dlsym(libdrm, "amdgpu_va_manager_init");
assert(libdrm_amdgpu->va_manager_init != NULL);
libdrm_amdgpu->va_manager_deinit = dlsym(libdrm, "amdgpu_va_manager_deinit");
assert(libdrm_amdgpu->va_manager_deinit != NULL);
libdrm_amdgpu->va_range_alloc2 = dlsym(libdrm, "amdgpu_va_range_alloc2");
assert(libdrm_amdgpu->va_range_alloc2 != NULL);
libdrm_amdgpu->bo_va_op = dlsym(libdrm, "amdgpu_bo_va_op");
assert(libdrm_amdgpu->bo_va_op != NULL);
libdrm_amdgpu->bo_va_op_raw = dlsym(libdrm, "amdgpu_bo_va_op_raw");
assert(libdrm_amdgpu->bo_va_op_raw != NULL);
libdrm_amdgpu->cs_create_semaphore = dlsym(libdrm, "amdgpu_cs_create_semaphore");
assert(libdrm_amdgpu->cs_create_semaphore != NULL);
libdrm_amdgpu->cs_signal_semaphore = dlsym(libdrm, "amdgpu_cs_signal_semaphore");
assert(libdrm_amdgpu->cs_signal_semaphore != NULL);
libdrm_amdgpu->cs_wait_semaphore = dlsym(libdrm, "amdgpu_cs_wait_semaphore");
assert(libdrm_amdgpu->cs_wait_semaphore != NULL);
libdrm_amdgpu->cs_destroy_semaphore = dlsym(libdrm, "amdgpu_cs_destroy_semaphore");
assert(libdrm_amdgpu->cs_destroy_semaphore != NULL);
libdrm_amdgpu->get_marketing_name = dlsym(libdrm, "amdgpu_get_marketing_name");
assert(libdrm_amdgpu->get_marketing_name != NULL);
libdrm_amdgpu->cs_create_syncobj2 = dlsym(libdrm, "amdgpu_cs_create_syncobj2");
assert(libdrm_amdgpu->cs_create_syncobj2 != NULL);
libdrm_amdgpu->cs_create_syncobj = dlsym(libdrm, "amdgpu_cs_create_syncobj");
assert(libdrm_amdgpu->cs_create_syncobj != NULL);
libdrm_amdgpu->cs_destroy_syncobj = dlsym(libdrm, "amdgpu_cs_destroy_syncobj");
assert(libdrm_amdgpu->cs_destroy_syncobj != NULL);
libdrm_amdgpu->cs_syncobj_reset = dlsym(libdrm, "amdgpu_cs_syncobj_reset");
assert(libdrm_amdgpu->cs_syncobj_reset != NULL);
libdrm_amdgpu->cs_syncobj_signal = dlsym(libdrm, "amdgpu_cs_syncobj_signal");
assert(libdrm_amdgpu->cs_syncobj_signal != NULL);
libdrm_amdgpu->cs_syncobj_timeline_signal = dlsym(libdrm, "amdgpu_cs_syncobj_timeline_signal");
assert(libdrm_amdgpu->cs_syncobj_timeline_signal != NULL);
libdrm_amdgpu->cs_syncobj_wait = dlsym(libdrm, "amdgpu_cs_syncobj_wait");
assert(libdrm_amdgpu->cs_syncobj_wait != NULL);
libdrm_amdgpu->cs_syncobj_timeline_wait = dlsym(libdrm, "amdgpu_cs_syncobj_timeline_wait");
assert(libdrm_amdgpu->cs_syncobj_timeline_wait != NULL);
libdrm_amdgpu->cs_syncobj_query = dlsym(libdrm, "amdgpu_cs_syncobj_query");
assert(libdrm_amdgpu->cs_syncobj_query != NULL);
libdrm_amdgpu->cs_syncobj_query2 = dlsym(libdrm, "amdgpu_cs_syncobj_query2");
assert(libdrm_amdgpu->cs_syncobj_query2 != NULL);
libdrm_amdgpu->cs_export_syncobj = dlsym(libdrm, "amdgpu_cs_export_syncobj");
assert(libdrm_amdgpu->cs_export_syncobj != NULL);
libdrm_amdgpu->cs_import_syncobj = dlsym(libdrm, "amdgpu_cs_import_syncobj");
assert(libdrm_amdgpu->cs_import_syncobj != NULL);
libdrm_amdgpu->cs_syncobj_export_sync_file = dlsym(libdrm, "amdgpu_cs_syncobj_export_sync_file");
assert(libdrm_amdgpu->cs_syncobj_export_sync_file != NULL);
libdrm_amdgpu->cs_syncobj_import_sync_file = dlsym(libdrm, "amdgpu_cs_syncobj_import_sync_file");
assert(libdrm_amdgpu->cs_syncobj_import_sync_file != NULL);
libdrm_amdgpu->cs_syncobj_export_sync_file2 = dlsym(libdrm, "amdgpu_cs_syncobj_export_sync_file2");
assert(libdrm_amdgpu->cs_syncobj_export_sync_file2 != NULL);
libdrm_amdgpu->cs_syncobj_import_sync_file2 = dlsym(libdrm, "amdgpu_cs_syncobj_import_sync_file2");
assert(libdrm_amdgpu->cs_syncobj_import_sync_file2 != NULL);
libdrm_amdgpu->cs_syncobj_transfer = dlsym(libdrm, "amdgpu_cs_syncobj_transfer");
assert(libdrm_amdgpu->cs_syncobj_transfer != NULL);
libdrm_amdgpu->cs_fence_to_handle = dlsym(libdrm, "amdgpu_cs_fence_to_handle");
assert(libdrm_amdgpu->cs_fence_to_handle != NULL);
libdrm_amdgpu->cs_submit_raw = dlsym(libdrm, "amdgpu_cs_submit_raw");
assert(libdrm_amdgpu->cs_submit_raw != NULL);
libdrm_amdgpu->cs_submit_raw2 = dlsym(libdrm, "amdgpu_cs_submit_raw2");
assert(libdrm_amdgpu->cs_submit_raw2 != NULL);
libdrm_amdgpu->cs_chunk_fence_to_dep = dlsym(libdrm, "amdgpu_cs_chunk_fence_to_dep");
assert(libdrm_amdgpu->cs_chunk_fence_to_dep != NULL);
libdrm_amdgpu->cs_chunk_fence_info_to_data = dlsym(libdrm, "amdgpu_cs_chunk_fence_info_to_data");
assert(libdrm_amdgpu->cs_chunk_fence_info_to_data != NULL);
libdrm_amdgpu->vm_reserve_vmid = dlsym(libdrm, "amdgpu_vm_reserve_vmid");
assert(libdrm_amdgpu->vm_reserve_vmid != NULL);
libdrm_amdgpu->vm_unreserve_vmid = dlsym(libdrm, "amdgpu_vm_unreserve_vmid");
assert(libdrm_amdgpu->vm_unreserve_vmid != NULL);
libdrm_amdgpu->handle = libdrm;

   return libdrm_amdgpu;
}

static int  amdgpu_device_initialize_stub(int fd, uint32_t *major_version, uint32_t *minor_version, amdgpu_device_handle *device_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_device_initialize2_stub(int fd, _Bool deduplicate_device, uint32_t *major_version, uint32_t *minor_version, amdgpu_device_handle *device_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_device_deinitialize_stub(amdgpu_device_handle device_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_device_get_fd_stub(amdgpu_device_handle device_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_alloc_stub(amdgpu_device_handle dev, struct amdgpu_bo_alloc_request *alloc_buffer, amdgpu_bo_handle *buf_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_set_metadata_stub(amdgpu_bo_handle buf_handle, struct amdgpu_bo_metadata *info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_query_info_stub(amdgpu_bo_handle buf_handle, struct amdgpu_bo_info *info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_export_stub(amdgpu_bo_handle buf_handle, enum amdgpu_bo_handle_type type, uint32_t *shared_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_import_stub(amdgpu_device_handle dev, enum amdgpu_bo_handle_type type, uint32_t shared_handle, struct amdgpu_bo_import_result *output)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_create_bo_from_user_mem_stub(amdgpu_device_handle dev, void *cpu, uint64_t size, amdgpu_bo_handle *buf_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_find_bo_by_cpu_mapping_stub(amdgpu_device_handle dev, void *cpu, uint64_t size, amdgpu_bo_handle *buf_handle, uint64_t *offset_in_bo)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_free_stub(amdgpu_bo_handle buf_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static void  amdgpu_bo_inc_ref_stub(amdgpu_bo_handle bo)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        }
static int  amdgpu_bo_cpu_map_stub(amdgpu_bo_handle buf_handle, void **cpu)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_cpu_unmap_stub(amdgpu_bo_handle buf_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_wait_for_idle_stub(amdgpu_bo_handle buf_handle, uint64_t timeout_ns, _Bool *buffer_busy)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_list_create_raw_stub(amdgpu_device_handle dev, uint32_t number_of_buffers, struct drm_amdgpu_bo_list_entry *buffers, uint32_t *result)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_list_destroy_raw_stub(amdgpu_device_handle dev, uint32_t bo_list)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_list_create_stub(amdgpu_device_handle dev, uint32_t number_of_resources, amdgpu_bo_handle *resources, uint8_t *resource_prios, amdgpu_bo_list_handle *result)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_list_destroy_stub(amdgpu_bo_list_handle handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_list_update_stub(amdgpu_bo_list_handle handle, uint32_t number_of_resources, amdgpu_bo_handle *resources, uint8_t *resource_prios)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_ctx_create2_stub(amdgpu_device_handle dev, uint32_t priority, amdgpu_context_handle *context)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_ctx_create_stub(amdgpu_device_handle dev, amdgpu_context_handle *context)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_ctx_free_stub(amdgpu_context_handle context)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_ctx_override_priority_stub(amdgpu_device_handle dev, amdgpu_context_handle context, int master_fd, unsigned priority)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_ctx_stable_pstate_stub(amdgpu_context_handle context, uint32_t op, uint32_t flags, uint32_t *out_flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_query_reset_state_stub(amdgpu_context_handle context, uint32_t *state, uint32_t *hangs)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_query_reset_state2_stub(amdgpu_context_handle context, uint64_t *flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_submit_stub(amdgpu_context_handle context, uint64_t flags, struct amdgpu_cs_request *ibs_request, uint32_t number_of_requests)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_query_fence_status_stub(struct amdgpu_cs_fence *fence, uint64_t timeout_ns, uint64_t flags, uint32_t *expired)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_wait_fences_stub(struct amdgpu_cs_fence *fences, uint32_t fence_count, _Bool wait_all, uint64_t timeout_ns, uint32_t *status, uint32_t *first)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_buffer_size_alignment_stub(amdgpu_device_handle dev, struct amdgpu_buffer_size_alignments *info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_firmware_version_stub(amdgpu_device_handle dev, unsigned fw_type, unsigned ip_instance, unsigned index, uint32_t *version, uint32_t *feature)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_hw_ip_count_stub(amdgpu_device_handle dev, unsigned type, uint32_t *count)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_hw_ip_info_stub(amdgpu_device_handle dev, unsigned type, unsigned ip_instance, struct drm_amdgpu_info_hw_ip *info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_heap_info_stub(amdgpu_device_handle dev, uint32_t heap, uint32_t flags, struct amdgpu_heap_info *info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_crtc_from_id_stub(amdgpu_device_handle dev, unsigned id, int32_t *result)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_gpu_info_stub(amdgpu_device_handle dev, struct amdgpu_gpu_info *info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_info_stub(amdgpu_device_handle dev, unsigned info_id, unsigned size, void *value)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_sw_info_stub(amdgpu_device_handle dev, enum amdgpu_sw_info info, void *value)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_gds_info_stub(amdgpu_device_handle dev, struct amdgpu_gds_resource_info *gds_info)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_sensor_info_stub(amdgpu_device_handle dev, unsigned sensor_type, unsigned size, void *value)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_video_caps_info_stub(amdgpu_device_handle dev, unsigned cap_type, unsigned size, void *value)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_query_gpuvm_fault_info_stub(amdgpu_device_handle dev, unsigned size, void *value)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_read_mm_registers_stub(amdgpu_device_handle dev, unsigned dword_offset, unsigned count, uint32_t instance, uint32_t flags, uint32_t *values)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_va_range_alloc_stub(amdgpu_device_handle dev, enum amdgpu_gpu_va_range va_range_type, uint64_t size, uint64_t va_base_alignment, uint64_t va_base_required, uint64_t *va_base_allocated, amdgpu_va_handle *va_range_handle, uint64_t flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_va_range_free_stub(amdgpu_va_handle va_range_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static uint64_t  amdgpu_va_get_start_addr_stub(amdgpu_va_handle va_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_va_range_query_stub(amdgpu_device_handle dev, enum amdgpu_gpu_va_range type, uint64_t *start, uint64_t *end)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static amdgpu_va_manager_handle  amdgpu_va_manager_alloc_stub()
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        return NULL; }
static void  amdgpu_va_manager_init_stub(amdgpu_va_manager_handle va_mgr, uint64_t low_va_offset, uint64_t low_va_max, uint64_t high_va_offset, uint64_t high_va_max, uint32_t virtual_address_alignment)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        }
static void  amdgpu_va_manager_deinit_stub(amdgpu_va_manager_handle va_mgr)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        }
static int  amdgpu_va_range_alloc2_stub(amdgpu_va_manager_handle va_mgr, enum amdgpu_gpu_va_range va_range_type, uint64_t size, uint64_t va_base_alignment, uint64_t va_base_required, uint64_t *va_base_allocated, amdgpu_va_handle *va_range_handle, uint64_t flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_va_op_stub(amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr, uint64_t flags, uint32_t ops)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_bo_va_op_raw_stub(amdgpu_device_handle dev, amdgpu_bo_handle bo, uint64_t offset, uint64_t size, uint64_t addr, uint64_t flags, uint32_t ops)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_create_semaphore_stub(amdgpu_semaphore_handle *sem)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_signal_semaphore_stub(amdgpu_context_handle ctx, uint32_t ip_type, uint32_t ip_instance, uint32_t ring, amdgpu_semaphore_handle sem)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_wait_semaphore_stub(amdgpu_context_handle ctx, uint32_t ip_type, uint32_t ip_instance, uint32_t ring, amdgpu_semaphore_handle sem)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_destroy_semaphore_stub(amdgpu_semaphore_handle sem)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static const char* amdgpu_get_marketing_name_stub(amdgpu_device_handle dev)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        return NULL; }
static int  amdgpu_cs_create_syncobj2_stub(amdgpu_device_handle dev, uint32_t flags, uint32_t *syncobj)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_create_syncobj_stub(amdgpu_device_handle dev, uint32_t *syncobj)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_destroy_syncobj_stub(amdgpu_device_handle dev, uint32_t syncobj)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_reset_stub(amdgpu_device_handle dev, const uint32_t *syncobjs, uint32_t syncobj_count)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_signal_stub(amdgpu_device_handle dev, const uint32_t *syncobjs, uint32_t syncobj_count)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_timeline_signal_stub(amdgpu_device_handle dev, const uint32_t *syncobjs, uint64_t *points, uint32_t syncobj_count)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_wait_stub(amdgpu_device_handle dev, uint32_t *handles, unsigned num_handles, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_timeline_wait_stub(amdgpu_device_handle dev, uint32_t *handles, uint64_t *points, unsigned num_handles, int64_t timeout_nsec, unsigned flags, uint32_t *first_signaled)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_query_stub(amdgpu_device_handle dev, uint32_t *handles, uint64_t *points, unsigned num_handles)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_query2_stub(amdgpu_device_handle dev, uint32_t *handles, uint64_t *points, unsigned num_handles, uint32_t flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_export_syncobj_stub(amdgpu_device_handle dev, uint32_t syncobj, int *shared_fd)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_import_syncobj_stub(amdgpu_device_handle dev, int shared_fd, uint32_t *syncobj)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_export_sync_file_stub(amdgpu_device_handle dev, uint32_t syncobj, int *sync_file_fd)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_import_sync_file_stub(amdgpu_device_handle dev, uint32_t syncobj, int sync_file_fd)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_export_sync_file2_stub(amdgpu_device_handle dev, uint32_t syncobj, uint64_t point, uint32_t flags, int *sync_file_fd)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_import_sync_file2_stub(amdgpu_device_handle dev, uint32_t syncobj, uint64_t point, int sync_file_fd)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_syncobj_transfer_stub(amdgpu_device_handle dev, uint32_t dst_handle, uint64_t dst_point, uint32_t src_handle, uint64_t src_point, uint32_t flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_fence_to_handle_stub(amdgpu_device_handle dev, struct amdgpu_cs_fence *fence, uint32_t what, uint32_t *out_handle)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_submit_raw_stub(amdgpu_device_handle dev, amdgpu_context_handle context, amdgpu_bo_list_handle bo_list_handle, int num_chunks, struct drm_amdgpu_cs_chunk *chunks, uint64_t *seq_no)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_cs_submit_raw2_stub(amdgpu_device_handle dev, amdgpu_context_handle context, uint32_t bo_list_handle, int num_chunks, struct drm_amdgpu_cs_chunk *chunks, uint64_t *seq_no)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static void  amdgpu_cs_chunk_fence_to_dep_stub(struct amdgpu_cs_fence *fence, struct drm_amdgpu_cs_chunk_dep *dep)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        }
static void  amdgpu_cs_chunk_fence_info_to_data_stub(struct amdgpu_cs_fence_info *fence_info, struct drm_amdgpu_cs_chunk_data *data)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
        }
static int  amdgpu_vm_reserve_vmid_stub(amdgpu_device_handle dev, uint32_t flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }
static int  amdgpu_vm_unreserve_vmid_stub(amdgpu_device_handle dev, uint32_t flags)
            {
            printf("IMPLEMENT ME %s\n", __FUNCTION__);
            assert(!getenv("VIRTIO_MISSING"));
         return -1; }

    struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio_stubs(void) {
       struct libdrm_amdgpu *libdrm_amdgpu = calloc(1, sizeof(struct libdrm_amdgpu));

libdrm_amdgpu->device_initialize = amdgpu_device_initialize_stub;
libdrm_amdgpu->device_initialize2 = amdgpu_device_initialize2_stub;
libdrm_amdgpu->device_deinitialize = amdgpu_device_deinitialize_stub;
libdrm_amdgpu->device_get_fd = amdgpu_device_get_fd_stub;
libdrm_amdgpu->bo_alloc = amdgpu_bo_alloc_stub;
libdrm_amdgpu->bo_set_metadata = amdgpu_bo_set_metadata_stub;
libdrm_amdgpu->bo_query_info = amdgpu_bo_query_info_stub;
libdrm_amdgpu->bo_export = amdgpu_bo_export_stub;
libdrm_amdgpu->bo_import = amdgpu_bo_import_stub;
libdrm_amdgpu->create_bo_from_user_mem = amdgpu_create_bo_from_user_mem_stub;
libdrm_amdgpu->find_bo_by_cpu_mapping = amdgpu_find_bo_by_cpu_mapping_stub;
libdrm_amdgpu->bo_free = amdgpu_bo_free_stub;
libdrm_amdgpu->bo_inc_ref = amdgpu_bo_inc_ref_stub;
libdrm_amdgpu->bo_cpu_map = amdgpu_bo_cpu_map_stub;
libdrm_amdgpu->bo_cpu_unmap = amdgpu_bo_cpu_unmap_stub;
libdrm_amdgpu->bo_wait_for_idle = amdgpu_bo_wait_for_idle_stub;
libdrm_amdgpu->bo_list_create_raw = amdgpu_bo_list_create_raw_stub;
libdrm_amdgpu->bo_list_destroy_raw = amdgpu_bo_list_destroy_raw_stub;
libdrm_amdgpu->bo_list_create = amdgpu_bo_list_create_stub;
libdrm_amdgpu->bo_list_destroy = amdgpu_bo_list_destroy_stub;
libdrm_amdgpu->bo_list_update = amdgpu_bo_list_update_stub;
libdrm_amdgpu->cs_ctx_create2 = amdgpu_cs_ctx_create2_stub;
libdrm_amdgpu->cs_ctx_create = amdgpu_cs_ctx_create_stub;
libdrm_amdgpu->cs_ctx_free = amdgpu_cs_ctx_free_stub;
libdrm_amdgpu->cs_ctx_override_priority = amdgpu_cs_ctx_override_priority_stub;
libdrm_amdgpu->cs_ctx_stable_pstate = amdgpu_cs_ctx_stable_pstate_stub;
libdrm_amdgpu->cs_query_reset_state = amdgpu_cs_query_reset_state_stub;
libdrm_amdgpu->cs_query_reset_state2 = amdgpu_cs_query_reset_state2_stub;
libdrm_amdgpu->cs_submit = amdgpu_cs_submit_stub;
libdrm_amdgpu->cs_query_fence_status = amdgpu_cs_query_fence_status_stub;
libdrm_amdgpu->cs_wait_fences = amdgpu_cs_wait_fences_stub;
libdrm_amdgpu->query_buffer_size_alignment = amdgpu_query_buffer_size_alignment_stub;
libdrm_amdgpu->query_firmware_version = amdgpu_query_firmware_version_stub;
libdrm_amdgpu->query_hw_ip_count = amdgpu_query_hw_ip_count_stub;
libdrm_amdgpu->query_hw_ip_info = amdgpu_query_hw_ip_info_stub;
libdrm_amdgpu->query_heap_info = amdgpu_query_heap_info_stub;
libdrm_amdgpu->query_crtc_from_id = amdgpu_query_crtc_from_id_stub;
libdrm_amdgpu->query_gpu_info = amdgpu_query_gpu_info_stub;
libdrm_amdgpu->query_info = amdgpu_query_info_stub;
libdrm_amdgpu->query_sw_info = amdgpu_query_sw_info_stub;
libdrm_amdgpu->query_gds_info = amdgpu_query_gds_info_stub;
libdrm_amdgpu->query_sensor_info = amdgpu_query_sensor_info_stub;
libdrm_amdgpu->query_video_caps_info = amdgpu_query_video_caps_info_stub;
libdrm_amdgpu->query_gpuvm_fault_info = amdgpu_query_gpuvm_fault_info_stub;
libdrm_amdgpu->read_mm_registers = amdgpu_read_mm_registers_stub;
libdrm_amdgpu->va_range_alloc = amdgpu_va_range_alloc_stub;
libdrm_amdgpu->va_range_free = amdgpu_va_range_free_stub;
libdrm_amdgpu->va_get_start_addr = amdgpu_va_get_start_addr_stub;
libdrm_amdgpu->va_range_query = amdgpu_va_range_query_stub;
libdrm_amdgpu->va_manager_alloc = amdgpu_va_manager_alloc_stub;
libdrm_amdgpu->va_manager_init = amdgpu_va_manager_init_stub;
libdrm_amdgpu->va_manager_deinit = amdgpu_va_manager_deinit_stub;
libdrm_amdgpu->va_range_alloc2 = amdgpu_va_range_alloc2_stub;
libdrm_amdgpu->bo_va_op = amdgpu_bo_va_op_stub;
libdrm_amdgpu->bo_va_op_raw = amdgpu_bo_va_op_raw_stub;
libdrm_amdgpu->cs_create_semaphore = amdgpu_cs_create_semaphore_stub;
libdrm_amdgpu->cs_signal_semaphore = amdgpu_cs_signal_semaphore_stub;
libdrm_amdgpu->cs_wait_semaphore = amdgpu_cs_wait_semaphore_stub;
libdrm_amdgpu->cs_destroy_semaphore = amdgpu_cs_destroy_semaphore_stub;
libdrm_amdgpu->get_marketing_name = amdgpu_get_marketing_name_stub;
libdrm_amdgpu->cs_create_syncobj2 = amdgpu_cs_create_syncobj2_stub;
libdrm_amdgpu->cs_create_syncobj = amdgpu_cs_create_syncobj_stub;
libdrm_amdgpu->cs_destroy_syncobj = amdgpu_cs_destroy_syncobj_stub;
libdrm_amdgpu->cs_syncobj_reset = amdgpu_cs_syncobj_reset_stub;
libdrm_amdgpu->cs_syncobj_signal = amdgpu_cs_syncobj_signal_stub;
libdrm_amdgpu->cs_syncobj_timeline_signal = amdgpu_cs_syncobj_timeline_signal_stub;
libdrm_amdgpu->cs_syncobj_wait = amdgpu_cs_syncobj_wait_stub;
libdrm_amdgpu->cs_syncobj_timeline_wait = amdgpu_cs_syncobj_timeline_wait_stub;
libdrm_amdgpu->cs_syncobj_query = amdgpu_cs_syncobj_query_stub;
libdrm_amdgpu->cs_syncobj_query2 = amdgpu_cs_syncobj_query2_stub;
libdrm_amdgpu->cs_export_syncobj = amdgpu_cs_export_syncobj_stub;
libdrm_amdgpu->cs_import_syncobj = amdgpu_cs_import_syncobj_stub;
libdrm_amdgpu->cs_syncobj_export_sync_file = amdgpu_cs_syncobj_export_sync_file_stub;
libdrm_amdgpu->cs_syncobj_import_sync_file = amdgpu_cs_syncobj_import_sync_file_stub;
libdrm_amdgpu->cs_syncobj_export_sync_file2 = amdgpu_cs_syncobj_export_sync_file2_stub;
libdrm_amdgpu->cs_syncobj_import_sync_file2 = amdgpu_cs_syncobj_import_sync_file2_stub;
libdrm_amdgpu->cs_syncobj_transfer = amdgpu_cs_syncobj_transfer_stub;
libdrm_amdgpu->cs_fence_to_handle = amdgpu_cs_fence_to_handle_stub;
libdrm_amdgpu->cs_submit_raw = amdgpu_cs_submit_raw_stub;
libdrm_amdgpu->cs_submit_raw2 = amdgpu_cs_submit_raw2_stub;
libdrm_amdgpu->cs_chunk_fence_to_dep = amdgpu_cs_chunk_fence_to_dep_stub;
libdrm_amdgpu->cs_chunk_fence_info_to_data = amdgpu_cs_chunk_fence_info_to_data_stub;
libdrm_amdgpu->vm_reserve_vmid = amdgpu_vm_reserve_vmid_stub;
libdrm_amdgpu->vm_unreserve_vmid = amdgpu_vm_unreserve_vmid_stub;

   return libdrm_amdgpu;
}

