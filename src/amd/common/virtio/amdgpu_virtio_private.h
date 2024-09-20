/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef AMDGPU_VIRTIO__PRIVATE_H
#define AMDGPU_VIRTIO__PRIVATE_H

#include "drm-uapi/amdgpu_drm.h"
#include "drm-uapi/virtgpu_drm.h"

#include "util/hash_table.h"
#include "util/simple_mtx.h"

#include "libdrm_amdgpu_loader.h"
#include "amd_family.h"

#include "virtio/vdrm/vdrm.h"
#include "virtio/virtio-gpu/drm_hw.h"
#include "amdgpu_virtio_proto.h"

struct amdvgpu_bo;
struct amdvgpu_device;
typedef struct amdvgpu_device* amdvgpu_device_handle;
typedef struct amdvgpu_bo* amdvgpu_bo_handle;
struct amdvgpu_host_blob;
struct amdvgpu_host_blob_allocator;

/* Host context seqno handling.
 * seqno are monotonically increasing integer, so we don't need
 * to actually submit to know the value. This allows to not
 * wait for the submission to go to the host (= no need to wait
 * in the guest) and to know the seqno (= so we can take advantage
 * of user fence).
 */
struct amdvgpu_context {
   amdvgpu_device_handle dev;
   uint32_t refcount;
   uint32_t host_context_id;
   uint64_t ring_next_seqno[];
};

struct amdvgpu_device {
   struct vdrm_device * vdev;

   /* List of existing devices */
   int refcount;
   struct amdvgpu_device *next;

   int fd;

   /* Table mapping kms handles to amdvgpu_bo instances.
    * Used to maintain a 1-to-1 mapping between the 2.
    */
   simple_mtx_t handle_to_vbo_mutex;
   struct hash_table *handle_to_vbo;

   /* Submission through virtio-gpu are ring based.
    * Ring 0 is used for CPU jobs, then N rings are allocated: 1
    * per IP type per instance (so if the GPU has 1 gfx queue and 2
    * queues -> ring0 + 3 hw rings = 4 rings total).
    */
   uint32_t num_virtio_rings;
   uint32_t virtio_ring_mapping[AMD_NUM_IP_TYPES];

   struct drm_amdgpu_info_device dev_info;

   /* Blob id are per drm_file identifiers of host blobs.
    * Use a monotically increased integer to assign the blob id.
    */
   uint32_t next_blob_id;

   /* GPU VA management (allocation / release). */
   amdgpu_va_manager_handle va_mgr;

   /* Debug option to make some protocol commands synchronous.
    * If bit N is set, then the specific command will be sync.
    */
   int64_t sync_cmd;

   /* virtio-gpu uses a single context per drm_file and expects that
    * any 2 jobs submitted to the same {context, ring} will execute in
    * order.
    * amdgpu on the other hand allows for multiple context per drm_file,
    * so we either have to open multiple virtio-gpu drm_file to be able to
    * have 1 virtio-gpu context per amdgpu-context or use a single amdgpu
    * context.
    * Using multiple drm_file might cause BO sharing issues so for now limit
    * ourselves to a single amdgpu context. Each amdgpu_ctx object can schedule
    * parallel work on 1 gfx, 2 sdma, 4 compute, 1 of each vcn queue.
    */
   struct amdvgpu_context *amdgpu_ctx;
   bool allow_multiple_amdgpu_ctx;
};

int amdvgpu_bo_free(struct amdvgpu_bo *bo);
int amdvgpu_query_info(amdvgpu_device_handle dev, struct drm_amdgpu_info *info);
int amdvgpu_query_hw_ip_info(amdvgpu_device_handle dev, unsigned type, unsigned ip_instance,
                             struct drm_amdgpu_info_hw_ip *info);
/* Helpers to get useful helpers out of libdrm_amdgpu without
 * linking to the library.
 */
extern amdgpu_va_manager_init_type libdrm_amdgpu_va_manager_init;
extern amdgpu_va_manager_deinit_type libdrm_amdgpu_va_manager_deinit;
extern amdgpu_va_manager_alloc_type libdrm_amdgpu_va_manager_alloc;
extern amdgpu_va_range_free_type libdrm_amdgpu_va_range_free;
extern amdgpu_va_range_alloc2_type libdrm_amdgpu_va_range_alloc2;
int init_libdrm_amdgpu_va_manager_fn(void);


/* Refcounting helpers. Returns true when dst reaches 0. */
static inline bool update_references(int *dst, int *src)
{
   if (dst != src) {
      /* bump src first */
      if (src) {
         assert(p_atomic_read(src) > 0);
         p_atomic_inc(src);
      }
      if (dst) {
         return p_atomic_dec_zero(dst);
      }
   }
   return false;
}


static void release_vbo_cb(struct hash_entry *entry) {
   amdvgpu_bo_free(entry->data);
}

#define virtio_ioctl(fd, name, args) ({                              \
      int ret = drmIoctl((fd), DRM_IOCTL_ ## name, (args));          \
      ret;                                                           \
      })

/* amdgpu_virtio_device.c */
int amdvgpu_device_initialize(int fd, uint32_t *drm_major, uint32_t *drm_minor,
                              amdvgpu_device_handle* dev_out);
int amdvgpu_device_deinitialize(amdvgpu_device_handle dev);

struct amdvgpu_host_blob_creation_params {
   struct drm_virtgpu_resource_create_blob args;
   struct amdgpu_ccmd_gem_new_req req;
};

struct amdvgpu_bo {
   struct amdvgpu_device *dev;

   /* Importing the same kms handle must return the same
    * amdvgpu_pointer, so we need a refcount.
    */
   int refcount;

   /* The size of the BO (might be smaller that the host
    * bo' size).
    */
   unsigned size;

   /* The host blob backing this bo. */
   struct amdvgpu_host_blob *host_blob;
};

int amdvgpu_bo_va_op_raw(amdvgpu_device_handle dev,
                         amdvgpu_bo_handle bo,
                         uint64_t offset,
                         uint64_t size,
                         uint64_t addr,
                         uint64_t flags,
                         uint32_t ops);
int amdvgpu_bo_import(amdvgpu_device_handle dev,
                      enum amdgpu_bo_handle_type type,
                      uint32_t handle,
                      struct amdgpu_bo_import_result *result);
int amdvgpu_bo_export(amdvgpu_bo_handle bo,
                      enum amdgpu_bo_handle_type type,
                      uint32_t *shared_handle);
int amdvgpu_bo_cpu_map(amdvgpu_bo_handle bo_handle, void **cpu);
int amdvgpu_bo_cpu_unmap(amdvgpu_bo_handle bo);
int amdvgpu_bo_alloc(amdvgpu_device_handle dev,
                     struct amdgpu_bo_alloc_request *request,
                     amdvgpu_bo_handle *bo);

uint32_t amdvgpu_get_resource_id(amdvgpu_bo_handle bo);

/* There are 2 return-code:
 *    - the virtio one, returned by vdrm_send_req
 *    - the host one, which only makes sense for sync
 *      requests.
 */
static inline
int vdrm_send_req_wrapper(amdvgpu_device_handle dev,
                          struct vdrm_ccmd_req *req,
                          struct amdgpu_ccmd_rsp *rsp,
                          bool sync) {
   if (dev->sync_cmd & (1u << req->cmd))
      sync = true;

   int r = vdrm_send_req(dev->vdev, req, sync);

   if (r)
      return r;

   if (sync)
      return rsp->ret;

   return 0;
}
#endif
