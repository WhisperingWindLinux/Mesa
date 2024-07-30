/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <xf86drm.h>
#include <libsync.h>

#include <dlfcn.h>
#include <libdrm/amdgpu.h>

#include "amdgpu_virtio_private.h"

#include "util/log.h"

/* These functions belong to libdrm_amdgpu but we don't link with this
 * library anymore to avoid accidental use of it from guest.
 * Instead, the function pointers will be lookup up on first use.
 */
amdgpu_va_range_free_type libdrm_amdgpu_va_range_free = NULL;
amdgpu_va_range_alloc2_type libdrm_amdgpu_va_range_alloc2 = NULL;
amdgpu_va_manager_init_type libdrm_amdgpu_va_manager_init = NULL;
amdgpu_va_manager_deinit_type libdrm_amdgpu_va_manager_deinit = NULL;
amdgpu_va_manager_alloc_type libdrm_amdgpu_va_manager_alloc = NULL;
amdgpu_va_get_start_addr_type libdrm_amdgpu_va_get_start_addr = NULL;

int init_libdrm_amdgpu_va_manager_fn(void) {
   void *libdrm = dlopen("libdrm_amdgpu.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!libdrm) {
      mesa_loge("Error: Failed to open libdrm_amdgpu");
      return -1;
   }
   libdrm_amdgpu_va_range_free = dlsym(libdrm, "amdgpu_va_range_free");
   if (!libdrm_amdgpu_va_range_free) {
      mesa_loge("Error: Failed to dlsym amdgpu_va_range_free\n");
      return -1;
   }
   libdrm_amdgpu_va_range_alloc2 = dlsym(libdrm, "amdgpu_va_range_alloc2");
   if (!libdrm_amdgpu_va_range_alloc2) {
      mesa_loge("Error: Failed to dlsym amdgpu_va_range_alloc2\n");
      return -1;
   }
   libdrm_amdgpu_va_manager_init = dlsym(libdrm, "amdgpu_va_manager_init");
   if (!libdrm_amdgpu_va_manager_init) {
      mesa_loge("Error: Failed to dlsym amdgpu_va_manager_init\n");
      return -1;
   }
   libdrm_amdgpu_va_manager_deinit = dlsym(libdrm, "amdgpu_va_manager_deinit");
   if (!libdrm_amdgpu_va_manager_deinit) {
      mesa_loge("Error: Failed to dlsym amdgpu_va_manager_deinit\n");
      return -1;
   }
   libdrm_amdgpu_va_manager_alloc = dlsym(libdrm, "amdgpu_va_manager_alloc");
   if (!libdrm_amdgpu_va_manager_alloc) {
      mesa_loge("Error: Failed to dlsym amdgpu_va_manager_alloc\n");
      return -1;
   }
   libdrm_amdgpu_va_get_start_addr = dlsym(libdrm, "amdgpu_va_get_start_addr");
   if (!libdrm_amdgpu_va_get_start_addr) {
      mesa_loge("Error: Failed to dlsym amdgpu_va_get_start_addr\n");
      return -1;
   }
   return 0;
}

int
amdvgpu_query_info(amdvgpu_device_handle dev, struct drm_amdgpu_info *info)
{
   unsigned req_len = sizeof(struct amdgpu_ccmd_query_info_req);
   unsigned rsp_len = sizeof(struct amdgpu_ccmd_query_info_rsp) + info->return_size;

   uint8_t buf[req_len];
   struct amdgpu_ccmd_query_info_req *req = (void *)buf;
   struct amdgpu_ccmd_query_info_rsp *rsp;
   assert(0 == (offsetof(struct amdgpu_ccmd_query_info_rsp, payload) % 8));

   req->hdr = AMDGPU_CCMD(QUERY_INFO, req_len);
   memcpy(&req->info, info, sizeof(struct drm_amdgpu_info));

   rsp = vdrm_alloc_rsp(dev->vdev, &req->hdr, rsp_len);

   int r = vdrm_send_req_wrapper(dev, &req->hdr, &rsp->hdr, true);
   if (r)
      return r;

   memcpy((void*)(uintptr_t)info->return_pointer, rsp->payload, info->return_size);

   return 0;
}

static int
amdvgpu_query_info_simple(amdvgpu_device_handle dev, unsigned info_id, unsigned size, void *out)
{
   if (info_id == AMDGPU_INFO_DEV_INFO) {
      assert(size == sizeof(dev->dev_info));
      memcpy(out, &dev->dev_info, size);
      return 0;
   }
   struct drm_amdgpu_info info;
   info.return_pointer = (uintptr_t)out;
   info.query = info_id;
   info.return_size = size;
   return amdvgpu_query_info(dev, &info);
}

static int
amdvgpu_query_heap_info(amdvgpu_device_handle dev, unsigned heap, unsigned flags, struct amdgpu_heap_info *info)
{
   struct amdvgpu_shmem *shmem = to_amdvgpu_shmem(dev->vdev->shmem);
   /* Get heap information from shared memory */
   switch (heap) {
   case AMDGPU_GEM_DOMAIN_VRAM:
      if (flags & AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED)
         memcpy(info, &shmem->vis_vram, sizeof(*info));
      else
         memcpy(info, &shmem->vram, sizeof(*info));
      break;
   case AMDGPU_GEM_DOMAIN_GTT:
      memcpy(info, &shmem->gtt, sizeof(*info));
      break;
   default:
      return -EINVAL;
   }

   return 0;
}

int
amdvgpu_query_hw_ip_info(amdvgpu_device_handle dev, unsigned type, unsigned ip_instance,
                         struct drm_amdgpu_info_hw_ip *info)
{
   struct drm_amdgpu_info request;
   request.return_pointer = (uintptr_t) info;
   request.return_size = sizeof(struct drm_amdgpu_info_hw_ip);
   request.query = AMDGPU_INFO_HW_IP_INFO;
   request.query_hw_ip.type = type;
   request.query_hw_ip.ip_instance = ip_instance;
   return amdvgpu_query_info(dev, &request);
}

static int
amdvgpu_query_hw_ip_count(amdvgpu_device_handle dev, unsigned type, uint32_t *count)
{
   struct drm_amdgpu_info request;
   request.return_pointer = (uintptr_t) count;
   request.return_size = sizeof(*count);
   request.query = AMDGPU_INFO_HW_IP_COUNT;
   request.query_hw_ip.type = type;
   return amdvgpu_query_info(dev, &request);
}

static int
amdvgpu_query_video_caps_info(amdvgpu_device_handle dev, unsigned cap_type,
                              unsigned size, void *value)
{
   struct drm_amdgpu_info request;
   request.return_pointer = (uintptr_t)value;
   request.return_size = size;
   request.query = AMDGPU_INFO_VIDEO_CAPS;
   request.sensor_info.type = cap_type;

   return amdvgpu_query_info(dev, &request);
}

static int
amdvgpu_query_sw_info(amdvgpu_device_handle dev, enum amdgpu_sw_info info, void *value)
{
   if (info != amdgpu_sw_info_address32_hi)
      return -EINVAL;
   memcpy(value, &dev->vdev->caps.u.amdgpu.address32_hi, 4);
   return 0;
}

static int
amdvgpu_query_firmware_version(amdvgpu_device_handle dev, unsigned fw_type, unsigned ip_instance, unsigned index,
                               uint32_t *version, uint32_t *feature)
{
   struct drm_amdgpu_info request;
   struct drm_amdgpu_info_firmware firmware = {};
   int r;

   memset(&request, 0, sizeof(request));
   request.return_pointer = (uintptr_t)&firmware;
   request.return_size = sizeof(firmware);
   request.query = AMDGPU_INFO_FW_VERSION;
   request.query_fw.fw_type = fw_type;
   request.query_fw.ip_instance = ip_instance;
   request.query_fw.index = index;

   r = amdvgpu_query_info(dev, &request);

   *version = firmware.ver;
   *feature = firmware.feature;
   return r;
}

static int
amdvgpu_query_buffer_size_alignment(amdvgpu_device_handle dev,
                                    struct amdgpu_buffer_size_alignments *info)
{
   memcpy(info, &dev->vdev->caps.u.amdgpu.alignments, sizeof(*info));
   return 0;
}

static int
amdvgpu_query_gpu_info(amdvgpu_device_handle dev, struct amdgpu_gpu_info *info)
{
   memcpy(info, &dev->vdev->caps.u.amdgpu.gpu_info, sizeof(*info));
   return 0;
}

static int
amdvgpu_bo_set_metadata(amdvgpu_bo_handle bo, struct amdgpu_bo_metadata *info)
{
   unsigned req_len = sizeof(struct amdgpu_ccmd_set_metadata_req) + info->size_metadata;
   unsigned rsp_len = sizeof(struct amdgpu_ccmd_rsp);

   uint8_t buf[req_len];
   struct amdgpu_ccmd_set_metadata_req *req = (void *)buf;
   struct amdgpu_ccmd_rsp *rsp;

   req->hdr = AMDGPU_CCMD(SET_METADATA, req_len);
   req->res_id = amdvgpu_get_resource_id(bo);
   req->flags = info->flags;
   req->tiling_info = info->tiling_info;
   req->size_metadata = info->size_metadata;
   memcpy(req->umd_metadata, info->umd_metadata, info->size_metadata);

   rsp = vdrm_alloc_rsp(bo->dev->vdev, &req->hdr, rsp_len);
   return vdrm_send_req_wrapper(bo->dev, &req->hdr, rsp, true);
}

static
int amdvgpu_bo_query_info(amdvgpu_bo_handle bo, struct amdgpu_bo_info *info) {
   unsigned req_len = sizeof(struct amdgpu_ccmd_bo_query_info_req);
   unsigned rsp_len = sizeof(struct amdgpu_ccmd_bo_query_info_rsp);

   uint8_t buf[req_len];
   struct amdgpu_ccmd_bo_query_info_req *req = (void *)buf;
   struct amdgpu_ccmd_bo_query_info_rsp *rsp;

   req->hdr = AMDGPU_CCMD(BO_QUERY_INFO, req_len);
   req->res_id = amdvgpu_get_resource_id(bo);
   req->pad = 0;

   rsp = vdrm_alloc_rsp(bo->dev->vdev, &req->hdr, rsp_len);

   int r = vdrm_send_req_wrapper(bo->dev, &req->hdr, &rsp->hdr, true);
   if (r)
      return r;

   info->alloc_size = rsp->info.alloc_size;
   info->phys_alignment = rsp->info.phys_alignment;
   info->preferred_heap = rsp->info.preferred_heap;
   info->alloc_flags = rsp->info.alloc_flags;

   info->metadata.flags = rsp->info.metadata.flags;
   info->metadata.tiling_info = rsp->info.metadata.tiling_info;
   info->metadata.size_metadata = rsp->info.metadata.size_metadata;
   memcpy(info->metadata.umd_metadata, rsp->info.metadata.umd_metadata,
          MIN2(sizeof(info->metadata.umd_metadata), rsp->info.metadata.size_metadata));

   return 0;
}

static
int amdvgpu_cs_ctx_create2(amdvgpu_device_handle dev, int32_t priority, void **ctx_virtio) {
   if (dev->amdgpu_ctx) {
      p_atomic_inc(&dev->amdgpu_ctx->refcount);
      *ctx_virtio = dev->amdgpu_ctx;
      return 0;
   }

   struct amdgpu_ccmd_create_ctx_req req = {
      .priority = priority,
      .flags = 0,
   };
   struct amdgpu_ccmd_create_ctx_rsp *rsp;

   req.hdr = AMDGPU_CCMD(CREATE_CTX, sizeof(req));

   rsp = vdrm_alloc_rsp(dev->vdev, &req.hdr, sizeof(struct amdgpu_ccmd_create_ctx_rsp));
   int r = vdrm_send_req_wrapper(dev, &req.hdr, &rsp->hdr, true);

   if (r)
      return r;

   if (rsp->ctx_id == 0)
      return -ENOTSUP;

   struct amdvgpu_context *ctx = calloc(1, sizeof(struct amdvgpu_context) + dev->num_virtio_rings * sizeof(uint64_t));
   if (ctx == NULL)
      return -ENOMEM;

   p_atomic_inc(&ctx->refcount);
   ctx->dev = dev;
   ctx->host_context_id = rsp->ctx_id;
   for (int i = 0; i < dev->num_virtio_rings; i++)
      ctx->ring_next_seqno[i] = 1;
   *ctx_virtio = ctx;

   if (!dev->allow_multiple_amdgpu_ctx)
      dev->amdgpu_ctx = ctx;

   return 0;
}

static
int amdvgpu_cs_ctx_free(void *ctx)
{
   struct amdvgpu_context *context = ctx;
   amdvgpu_device_handle dev = context->dev;

   if (!dev->allow_multiple_amdgpu_ctx) {
      assert((struct amdvgpu_context*)context == dev->amdgpu_ctx);
      if (p_atomic_dec_return(&dev->amdgpu_ctx->refcount))
         return 0;
   }

   struct amdgpu_ccmd_create_ctx_req req = {
      .id = ((struct amdvgpu_context*)context)->host_context_id,
      .flags = AMDGPU_CCMD_CREATE_CTX_DESTROY,
   };
   req.hdr = AMDGPU_CCMD(CREATE_CTX, sizeof(req));

   free(context);

   dev->amdgpu_ctx = NULL;

   struct amdgpu_ccmd_create_ctx_rsp *rsp;
   rsp = vdrm_alloc_rsp(dev->vdev, &req.hdr, sizeof(struct amdgpu_ccmd_create_ctx_rsp));

   return vdrm_send_req_wrapper(dev, &req.hdr, &rsp->hdr, false);
}

static int
amdvgpu_device_get_fd(amdvgpu_device_handle dev) {
   return dev->fd;
}

static const char *
amdvgpu_get_marketing_name(amdvgpu_device_handle dev) {
   return dev->vdev->caps.u.amdgpu.marketing_name;
}

static
int amdvgpu_cs_create_syncobj2(amdvgpu_device_handle dev, uint32_t flags, uint32_t *handle)
{
   return drmSyncobjCreate(dev->fd, flags, handle);
}

static
int amdvgpu_cs_create_syncobj(amdvgpu_device_handle dev, uint32_t *handle)
{
   return drmSyncobjCreate(dev->fd, 0, handle);
}

static
int amdvgpu_cs_destroy_syncobj(amdvgpu_device_handle dev, uint32_t handle)
{
   return drmSyncobjDestroy(dev->fd, handle);
}

static void
amdvgpu_cs_chunk_fence_info_to_data(struct amdgpu_cs_fence_info *fence_info,
                                    struct drm_amdgpu_cs_chunk_data *data)
{
   data->fence_data.handle = amdvgpu_get_resource_id((void*)fence_info->handle);
   data->fence_data.offset = fence_info->offset * sizeof(uint64_t);
}

static int
amdvgpu_cs_syncobj_export_sync_file(amdvgpu_device_handle dev,
                                    uint32_t syncobj,
                                    int *sync_file_fd)
{
   return drmSyncobjExportSyncFile(dev->fd, syncobj, sync_file_fd);
}

static int
amdvgpu_cs_syncobj_import_sync_file(amdvgpu_device_handle dev,
                                    uint32_t syncobj,
                                    int sync_file_fd)
{
   return drmSyncobjImportSyncFile(dev->fd, syncobj, sync_file_fd);
}

static uint32_t cs_chunk_ib_to_virtio_ring_idx(amdvgpu_device_handle dev,
                                               struct drm_amdgpu_cs_chunk_ib *ib) {
   assert(dev->virtio_ring_mapping[ib->ip_type] != 0);
   return dev->virtio_ring_mapping[ib->ip_type] + ib->ring;
}

static int
amdvgpu_cs_submit_raw2(amdvgpu_device_handle dev, amdgpu_context_handle ctx,
                       uint32_t bo_list_handle,
                       int num_chunks, struct drm_amdgpu_cs_chunk *chunks,
                       uint64_t *seqno)
{
   unsigned rsp_len = sizeof(struct amdgpu_ccmd_rsp);
   struct amdvgpu_context *vctx = (struct amdvgpu_context *)ctx;

   struct extra_data_info {
      const void *ptr;
      uint32_t size;
   } extra[1 + num_chunks];

   int chunk_count = 0;
   unsigned offset = 0;

   struct desc {
      uint16_t chunk_id;
      uint16_t length_dw;
      uint32_t offset;
   };
   struct desc descriptors[num_chunks];

   unsigned virtio_ring_idx = 0xffffffff;

   uint32_t syncobj_in_count = 0, syncobj_out_count = 0;
   struct drm_virtgpu_execbuffer_syncobj *syncobj_in = NULL;
   struct drm_virtgpu_execbuffer_syncobj *syncobj_out = NULL;
   uint8_t *buf = NULL;
   int ret;

   const bool sync_submit = dev->sync_cmd & (1u << AMDGPU_CCMD_CS_SUBMIT);

   /* Extract pointers from each chunk and copy them to the payload. */
   for (int i = 0; i < num_chunks; i++) {
      int extra_idx = 1 + chunk_count;
      if (chunks[i].chunk_id == AMDGPU_CHUNK_ID_BO_HANDLES) {
         struct drm_amdgpu_bo_list_in *list_in = (void*) (uintptr_t)chunks[i].chunk_data;
         extra[extra_idx].ptr = (void*) (uintptr_t)list_in->bo_info_ptr;
         extra[extra_idx].size = list_in->bo_info_size * list_in->bo_number;
      } else if (chunks[i].chunk_id == AMDGPU_CHUNK_ID_DEPENDENCIES ||
                 chunks[i].chunk_id == AMDGPU_CHUNK_ID_FENCE ||
                 chunks[i].chunk_id == AMDGPU_CHUNK_ID_IB) {
         extra[extra_idx].ptr = (void*)(uintptr_t)chunks[i].chunk_data;
         extra[extra_idx].size = chunks[i].length_dw * 4;

         if (chunks[i].chunk_id == AMDGPU_CHUNK_ID_IB) {
            struct drm_amdgpu_cs_chunk_ib *ib = (void*)(uintptr_t)chunks[i].chunk_data;
            virtio_ring_idx = cs_chunk_ib_to_virtio_ring_idx(dev, ib);
         }
      } else if (chunks[i].chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_OUT ||
                 chunks[i].chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_IN) {
         /* Translate from amdgpu CHUNK_ID_SYNCOBJ_* to drm_virtgpu_execbuffer_syncobj */
         struct drm_amdgpu_cs_chunk_sem *amd_syncobj = (void*) (uintptr_t)chunks[i].chunk_data;
         unsigned syncobj_count = (chunks[i].length_dw * 4) / sizeof(struct drm_amdgpu_cs_chunk_sem);
         struct drm_virtgpu_execbuffer_syncobj *syncobjs =
            calloc(syncobj_count, sizeof(struct drm_virtgpu_execbuffer_syncobj));

         if (syncobjs == NULL) {
            ret = -ENOMEM;
            goto error;
         }

         for (int j = 0; j < syncobj_count; j++)
            syncobjs[j].handle = amd_syncobj[j].handle;

         if (chunks[i].chunk_id == AMDGPU_CHUNK_ID_SYNCOBJ_IN) {
            syncobj_in_count = syncobj_count;
            syncobj_in = syncobjs;
         } else {
            syncobj_out_count = syncobj_count;
            syncobj_out = syncobjs;
         }

         /* This chunk was converted to virtgpu UAPI so we don't need to forward it
          * to the host.
          */
         continue;
      } else {
         mesa_loge("Unhandled chunk_id: %d\n", chunks[i].chunk_id);
         continue;
      }
      descriptors[chunk_count].chunk_id = chunks[i].chunk_id;
      descriptors[chunk_count].offset = offset;
      descriptors[chunk_count].length_dw = extra[extra_idx].size / 4;
      offset += extra[extra_idx].size;
      chunk_count++;
   }
   assert(virtio_ring_idx != 0xffffffff);

   /* Copy the descriptors at the beginning. */
   extra[0].ptr = descriptors;
   extra[0].size = chunk_count * sizeof(struct desc);

   /* Determine how much extra space we need. */
   uint32_t req_len = sizeof(struct amdgpu_ccmd_cs_submit_req);
   uint32_t e_offset = req_len;
   for (unsigned i = 0; i < 1 + chunk_count; i++)
      req_len += extra[i].size;

   /* Allocate the command buffer. */
   buf = malloc(req_len);
   if (buf == NULL) {
      ret = -ENOMEM;
      goto error;
   }
   struct amdgpu_ccmd_cs_submit_req *req = (void*)buf;
   req->hdr = AMDGPU_CCMD(CS_SUBMIT, req_len);
   req->ctx_id = vctx->host_context_id;
   req->num_chunks = chunk_count;
   req->ring_idx = virtio_ring_idx;
   req->pad = 0;

   UNUSED struct amdgpu_ccmd_rsp *rsp = vdrm_alloc_rsp(dev->vdev, &req->hdr, rsp_len);

   /* Copy varying data after the fixed part of cs_submit_req. */
   for (unsigned i = 0; i < 1 + chunk_count; i++) {
      if (extra[i].size) {
         memcpy(&buf[e_offset], extra[i].ptr, extra[i].size);
         e_offset += extra[i].size;
      }
   }

   /* Optional fence out (if we want synchronous submits). */
   int *fence_fd_ptr = NULL;

   struct vdrm_execbuf_params vdrm_execbuf_p = {
      .ring_idx = virtio_ring_idx,
      .req = &req->hdr,
      .handles = NULL,
      .num_handles = 0,
      .in_syncobjs = syncobj_in,
      .out_syncobjs = syncobj_out,
      .has_in_fence_fd = 0,
      .needs_out_fence_fd = sync_submit,
      .fence_fd = 0,
      .num_in_syncobjs = syncobj_in_count,
      .num_out_syncobjs = syncobj_out_count,
   };

   if (sync_submit)
      fence_fd_ptr = &vdrm_execbuf_p.fence_fd;

   /* Push job to the host. */
   ret = vdrm_execbuf(dev->vdev, &vdrm_execbuf_p);

   /* Determine the host seqno for this job. */
   *seqno = vctx->ring_next_seqno[virtio_ring_idx - 1]++;

   if (ret == 0 && fence_fd_ptr) {
      /* Sync execution */
      sync_wait(*fence_fd_ptr, -1);
      close(*fence_fd_ptr);
      vdrm_host_sync(dev->vdev, &req->hdr);
   }

error:
   free(buf);
   free(syncobj_in);
   free(syncobj_out);

   return ret;
}

static
int amdvgpu_cs_query_reset_state2(void *_dev, uint64_t *flags)
{
   amdvgpu_device_handle dev = _dev;
   *flags = 0;

   if (to_amdvgpu_shmem(dev->vdev->shmem)->async_error > 0)
      *flags = AMDGPU_CTX_QUERY2_FLAGS_RESET | AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST;

   return 0;
}

static
int amdvgpu_cs_query_fence_status(struct amdgpu_cs_fence *fence,
                                  uint64_t timeout_ns,
                                  uint64_t flags,
                                  uint32_t *expired)
{
   struct amdvgpu_context *ctx = (struct amdvgpu_context *)fence->context;
   unsigned req_len = sizeof(struct amdgpu_ccmd_cs_query_fence_status_req);
   unsigned rsp_len = sizeof(struct amdgpu_ccmd_cs_query_fence_status_rsp);

   uint8_t buf[req_len];
   struct amdgpu_ccmd_cs_query_fence_status_req *req = (void *)buf;
   struct amdgpu_ccmd_cs_query_fence_status_rsp *rsp;

   amdvgpu_device_handle dev = ctx->dev;

   req->hdr = AMDGPU_CCMD(CS_QUERY_FENCE_STATUS, req_len);
   req->ctx_id = ctx->host_context_id;
   req->ip_type = fence->ip_type;
   req->ip_instance = fence->ip_instance;
   req->ring = fence->ring;
   req->fence = fence->fence;
   req->timeout_ns = timeout_ns;
   req->flags = flags;

   rsp = vdrm_alloc_rsp(dev->vdev, &req->hdr, rsp_len);

   int r = vdrm_send_req_wrapper(dev, &req->hdr, &rsp->hdr, true);

   if (r == 0)
      *expired = rsp->expired;

   return r;
}

static int amdvgpu_cs_syncobj_wait(amdvgpu_device_handle dev, uint32_t *handles,
                                  unsigned num_handles, int64_t timeout_nsec,
                                  unsigned flags, uint32_t *first_signaled) {
   return drmSyncobjWait(dev->fd, handles, num_handles, timeout_nsec,
                         flags, first_signaled);
}

static int amdvgpu_cs_syncobj_reset(amdvgpu_device_handle dev, const uint32_t *syncobjs, uint32_t syncobj_count) {
   return drmSyncobjReset(dev->fd, syncobjs, syncobj_count);
}

static int amdvgpu_vm_reserve_vmid_helper(amdvgpu_device_handle dev, int reserve) {
   unsigned req_len = sizeof(struct amdgpu_ccmd_reserve_vmid_req);

   uint8_t buf[req_len];
   struct amdgpu_ccmd_reserve_vmid_req *req = (void *)buf;
   struct amdgpu_ccmd_rsp *rsp = vdrm_alloc_rsp(dev->vdev, &req->hdr, sizeof(struct amdgpu_ccmd_rsp));

   req->hdr = AMDGPU_CCMD(RESERVE_VMID, req_len);
   req->flags = reserve ? 0 : AMDGPU_CCMD_RESERVE_VMID_UNRESERVE;

   return vdrm_send_req_wrapper(dev, &req->hdr, rsp, true);
}

static int amdvgpu_vm_reserve_vmid(amdvgpu_device_handle dev, uint32_t flags) {
   assert(flags == 0);
   return amdvgpu_vm_reserve_vmid_helper(dev, 1);
}

static int amdvgpu_vm_unreserve_vmid(amdvgpu_device_handle dev, uint32_t flags) {
   assert(flags == 0);
   return amdvgpu_vm_reserve_vmid_helper(dev, 0);
}

static int amdvgpu_cs_ctx_stable_pstate(struct amdvgpu_context *ctx,
                                        uint32_t op,
                                        uint32_t flags,
                                        uint32_t *out_flags) {
   unsigned req_len = sizeof(struct amdgpu_ccmd_set_pstate_req);
   unsigned rsp_len = sizeof(struct amdgpu_ccmd_set_pstate_rsp);

   uint8_t buf[req_len];
   struct amdgpu_ccmd_set_pstate_req *req = (void *)buf;
   struct amdgpu_ccmd_set_pstate_rsp *rsp;

   amdvgpu_device_handle dev = ctx->dev;

   req->hdr = AMDGPU_CCMD(SET_PSTATE, req_len);
   req->ctx_id = ctx->host_context_id;
   req->op = op;
   req->flags = flags;
   req->pad = 0;

   rsp = vdrm_alloc_rsp(dev->vdev, &req->hdr, rsp_len);

   int r = vdrm_send_req_wrapper(dev, &req->hdr, &rsp->hdr, out_flags);

   if (r == 0 && out_flags)
      *out_flags = rsp->out_flags;

   return r;
}

static int
amdvgpu_va_range_alloc(amdvgpu_device_handle dev,
                       enum amdgpu_gpu_va_range va_range_type,
                       uint64_t size,
                       uint64_t va_base_alignment,
                       uint64_t va_base_required,
                       uint64_t *va_base_allocated,
                       amdgpu_va_handle *va_range_handle,
                       uint64_t flags)
{
   return libdrm_amdgpu_va_range_alloc2(dev->va_mgr, va_range_type, size,
                                        va_base_alignment, va_base_required,
                                        va_base_allocated, va_range_handle,
                                        flags);
}

static int
amdvgpu_va_range_free(amdgpu_va_handle va_range_handle)
{
   return libdrm_amdgpu_va_range_free(va_range_handle);
}


struct libdrm_amdgpu * ac_init_libdrm_amdgpu_for_virtio(void) {
   struct libdrm_amdgpu *libdrm_amdgpu = ac_init_libdrm_amdgpu_for_virtio_stubs();

   if (libdrm_amdgpu_va_manager_alloc == NULL) {
      if (init_libdrm_amdgpu_va_manager_fn() != 0) {
         return NULL;
      }
   }

   libdrm_amdgpu->device_initialize = (amdgpu_device_initialize_type) amdvgpu_device_initialize;
   libdrm_amdgpu->device_deinitialize = (amdgpu_device_deinitialize_type) amdvgpu_device_deinitialize;
   libdrm_amdgpu->cs_ctx_create2 = (amdgpu_cs_ctx_create2_type) amdvgpu_cs_ctx_create2;
   libdrm_amdgpu->cs_ctx_free = (amdgpu_cs_ctx_free_type) amdvgpu_cs_ctx_free;
   libdrm_amdgpu->bo_query_info = (amdgpu_bo_query_info_type) amdvgpu_bo_query_info;
   libdrm_amdgpu->bo_free = (amdgpu_bo_free_type) amdvgpu_bo_free;
   libdrm_amdgpu->bo_cpu_map = (amdgpu_bo_cpu_map_type) amdvgpu_bo_cpu_map;
   libdrm_amdgpu->bo_cpu_unmap = (amdgpu_bo_cpu_unmap_type) amdvgpu_bo_cpu_unmap;
   libdrm_amdgpu->bo_alloc = (amdgpu_bo_alloc_type) amdvgpu_bo_alloc;
   libdrm_amdgpu->va_range_alloc = (amdgpu_va_range_alloc_type) amdvgpu_va_range_alloc;
   libdrm_amdgpu->bo_va_op_raw = (amdgpu_bo_va_op_raw_type) amdvgpu_bo_va_op_raw;
   libdrm_amdgpu->bo_set_metadata = (amdgpu_bo_set_metadata_type) amdvgpu_bo_set_metadata;
   libdrm_amdgpu->bo_import = (amdgpu_bo_import_type) amdvgpu_bo_import;
   libdrm_amdgpu->bo_export = (amdgpu_bo_export_type) amdvgpu_bo_export;
   libdrm_amdgpu->device_get_fd = (amdgpu_device_get_fd_type) amdvgpu_device_get_fd;
   libdrm_amdgpu->get_marketing_name = (amdgpu_get_marketing_name_type) amdvgpu_get_marketing_name;
   libdrm_amdgpu->va_range_free = (amdgpu_va_range_free_type) amdvgpu_va_range_free;
   libdrm_amdgpu->cs_create_syncobj = (amdgpu_cs_create_syncobj_type) amdvgpu_cs_create_syncobj;
   libdrm_amdgpu->cs_create_syncobj2 = (amdgpu_cs_create_syncobj2_type) amdvgpu_cs_create_syncobj2;
   libdrm_amdgpu->cs_destroy_syncobj = (amdgpu_cs_destroy_syncobj_type) amdvgpu_cs_destroy_syncobj;
   libdrm_amdgpu->cs_chunk_fence_info_to_data = (amdgpu_cs_chunk_fence_info_to_data_type) amdvgpu_cs_chunk_fence_info_to_data;
   libdrm_amdgpu->cs_submit_raw2 = (amdgpu_cs_submit_raw2_type) amdvgpu_cs_submit_raw2;
   libdrm_amdgpu->cs_syncobj_export_sync_file = (amdgpu_cs_syncobj_export_sync_file_type) amdvgpu_cs_syncobj_export_sync_file;
   libdrm_amdgpu->cs_syncobj_import_sync_file = (amdgpu_cs_syncobj_import_sync_file_type) amdvgpu_cs_syncobj_import_sync_file;
   libdrm_amdgpu->query_info = (amdgpu_query_info_type) amdvgpu_query_info_simple;
   libdrm_amdgpu->query_sw_info = (amdgpu_query_sw_info_type) amdvgpu_query_sw_info;
   libdrm_amdgpu->query_firmware_version = (amdgpu_query_firmware_version_type) amdvgpu_query_firmware_version;
   libdrm_amdgpu->query_buffer_size_alignment = (amdgpu_query_buffer_size_alignment_type) amdvgpu_query_buffer_size_alignment;
   libdrm_amdgpu->query_gpu_info = (amdgpu_query_gpu_info_type) amdvgpu_query_gpu_info;
   libdrm_amdgpu->query_hw_ip_info = (amdgpu_query_hw_ip_info_type) amdvgpu_query_hw_ip_info;
   libdrm_amdgpu->query_hw_ip_count = (amdgpu_query_hw_ip_count_type) amdvgpu_query_hw_ip_count;
   libdrm_amdgpu->query_video_caps_info = (amdgpu_query_video_caps_info_type) amdvgpu_query_video_caps_info;
   libdrm_amdgpu->cs_query_reset_state2 = (amdgpu_cs_query_reset_state2_type) amdvgpu_cs_query_reset_state2;
   libdrm_amdgpu->query_heap_info = (amdgpu_query_heap_info_type) amdvgpu_query_heap_info;
   libdrm_amdgpu->cs_syncobj_wait = (amdgpu_cs_syncobj_wait_type) amdvgpu_cs_syncobj_wait;
   libdrm_amdgpu->cs_syncobj_reset = (amdgpu_cs_syncobj_reset_type) amdvgpu_cs_syncobj_reset;
   libdrm_amdgpu->vm_reserve_vmid = (amdgpu_vm_reserve_vmid_type) amdvgpu_vm_reserve_vmid;
   libdrm_amdgpu->vm_unreserve_vmid = (amdgpu_vm_unreserve_vmid_type) amdvgpu_vm_unreserve_vmid;
   libdrm_amdgpu->cs_ctx_stable_pstate = (amdgpu_cs_ctx_stable_pstate_type) amdvgpu_cs_ctx_stable_pstate;
   libdrm_amdgpu->cs_query_fence_status = (amdgpu_cs_query_fence_status_type) amdvgpu_cs_query_fence_status;

   /* For this function we can use the stock libdrm_amdgpu version. */
   libdrm_amdgpu->va_get_start_addr = (amdgpu_va_get_start_addr_type) libdrm_amdgpu_va_get_start_addr;

   return libdrm_amdgpu;
}
