/*
 * Copyright © 2023 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "amdgpu_bo.h"
#include <amdgpu_drm.h>

static bool
amdgpu_userq_ring_init(struct amdgpu_winsys *aws, struct amdgpu_userq *userq)
{
   /* allocate memory for ring */
   userq->ring_bo = amdgpu_bo_create(aws, AMDGPU_USERQ_RING_SIZE, 256, RADEON_DOMAIN_GTT,
                                     RADEON_FLAG_GL2_BYPASS | RADEON_FLAG_NO_SUBALLOC |
                                     RADEON_FLAG_NO_INTERPROCESS_SHARING);
   if (!userq->ring_bo)
      return false;

   userq->ring_base_ptr = (uint32_t*)amdgpu_bo_map(&aws->dummy_sws.base, userq->ring_bo, NULL,
                                                   PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED);
   if (!userq->ring_base_ptr)
      return false;

   /* allocate memory for rptr */
   userq->rptr_bo = amdgpu_bo_create(aws, aws->info.gart_page_size, 256, RADEON_DOMAIN_GTT,
                                     RADEON_FLAG_GL2_BYPASS | RADEON_FLAG_NO_SUBALLOC |
                                     RADEON_FLAG_NO_INTERPROCESS_SHARING);
   if (!userq->rptr_bo)
      return false;

   userq->mono_rptr = amdgpu_bo_map(&aws->dummy_sws.base, userq->rptr_bo, NULL,
                                    PIPE_MAP_READ | PIPE_MAP_UNSYNCHRONIZED);
   if (!userq->mono_rptr)
      return false;

   /* allocate memory for wptr */
   userq->wptr_bo = amdgpu_bo_create(aws, aws->info.gart_page_size, 256, RADEON_DOMAIN_GTT,
                                     RADEON_FLAG_GL2_BYPASS | RADEON_FLAG_NO_SUBALLOC |
                                     RADEON_FLAG_NO_INTERPROCESS_SHARING);
   if (!userq->wptr_bo)
      return false;

   userq->mono_wptr = amdgpu_bo_map(&aws->dummy_sws.base, userq->wptr_bo, NULL,
                                    PIPE_MAP_READ | PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED);
   if (!userq->mono_wptr)
      return false;

   *userq->mono_rptr = 0;
   *userq->mono_wptr = 0;
   return true;
}

static bool
amdgpu_userq_user_fence_init(struct amdgpu_winsys *aws, struct amdgpu_userq *userq)
{
   /* allocate memory for ring */
   userq->user_fence_bo = amdgpu_bo_create(aws, aws->info.gart_page_size, aws->info.gart_page_size,
                                           RADEON_DOMAIN_GTT, RADEON_FLAG_NO_SUBALLOC |
                                           RADEON_FLAG_NO_INTERPROCESS_SHARING);
   if (!userq->user_fence_bo)
      return false;

   userq->user_fence_ptr = amdgpu_bo_map(&aws->dummy_sws.base, userq->user_fence_bo, NULL,
                                         PIPE_MAP_READ | PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED);
   if (!userq->user_fence_ptr)
      return false;

   *userq->user_fence_ptr = 0;
   return true;
}

bool
amdgpu_userq_init(struct amdgpu_winsys *aws, struct amdgpu_userq *userq, enum amd_ip_type ip_type)
{
   int r = -1;
   uint32_t hw_ip_type;
   struct drm_amdgpu_userq_mqd_gfx_v11 gfx_mqd;
   struct drm_amdgpu_userq_mqd_compute_gfx_v11 compute_mqd;
   void *mqd;

   simple_mtx_lock(&userq->lock);

   if (userq->init_once) {
      simple_mtx_unlock(&userq->lock);
      return true;
   }

   userq->ip_type = ip_type;

   if (!amdgpu_userq_ring_init(aws, userq))
      goto fail;

   if (!amdgpu_userq_user_fence_init(aws, userq))
      goto fail;

   switch (userq->ip_type) {
   case AMD_IP_GFX:
      hw_ip_type = AMDGPU_HW_IP_GFX;
      userq->gfx_data.gds_bo = amdgpu_bo_create(aws, aws->info.gart_page_size, 256,
                                                RADEON_DOMAIN_VRAM, RADEON_FLAG_NO_SUBALLOC |
                                                RADEON_FLAG_NO_INTERPROCESS_SHARING);
      if (!userq->gfx_data.gds_bo)
         goto fail;

      userq->gfx_data.csa_bo = amdgpu_bo_create(aws, aws->info.fw_based_mcbp.csa_size,
                                                aws->info.fw_based_mcbp.csa_alignment,
                                                RADEON_DOMAIN_VRAM,
                                                RADEON_FLAG_NO_SUBALLOC |
                                                RADEON_FLAG_NO_INTERPROCESS_SHARING);
      if (!userq->gfx_data.csa_bo)
         goto fail;

      userq->gfx_data.shadow_bo = amdgpu_bo_create(aws, aws->info.fw_based_mcbp.shadow_size,
                                                   aws->info.fw_based_mcbp.shadow_alignment,
                                                   RADEON_DOMAIN_VRAM,
                                                   RADEON_FLAG_NO_SUBALLOC |
                                                   RADEON_FLAG_NO_INTERPROCESS_SHARING);
      if (!userq->gfx_data.shadow_bo)
         goto fail;

      gfx_mqd.shadow_va = amdgpu_bo_get_va(userq->gfx_data.shadow_bo);
      gfx_mqd.gds_va = amdgpu_bo_get_va(userq->gfx_data.gds_bo);
      gfx_mqd.csa_va = amdgpu_bo_get_va(userq->gfx_data.csa_bo);
      mqd = &gfx_mqd;
      break;
   case AMD_IP_COMPUTE:
      hw_ip_type = AMDGPU_HW_IP_COMPUTE;
      userq->compute_data.eop_bo = amdgpu_bo_create(aws, aws->info.gart_page_size, 256,
                                                    RADEON_DOMAIN_VRAM, RADEON_FLAG_NO_SUBALLOC |
                                                    RADEON_FLAG_NO_INTERPROCESS_SHARING);
      if (!userq->compute_data.eop_bo)
         goto fail;

      compute_mqd.eop_va = amdgpu_bo_get_va(userq->compute_data.eop_bo);
      mqd = &compute_mqd;
      break;
   case AMD_IP_SDMA:
      hw_ip_type = AMDGPU_HW_IP_DMA;
      mqd = NULL;
      break;
   default:
      fprintf(stderr, "amdgpu: userq unsupported for ip = %d\n", userq->ip_type);
   };

   userq->doorbell_bo = amdgpu_bo_create(aws, aws->info.gart_page_size, 256,
                                         RADEON_DOMAIN_DOORBELL, RADEON_FLAG_NO_SUBALLOC |
                                         RADEON_FLAG_NO_INTERPROCESS_SHARING);
   if (!userq->doorbell_bo)
      goto fail;
   userq->doorbell_ptr = amdgpu_bo_map(&aws->dummy_sws.base, userq->doorbell_bo, NULL,
                                       PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED);
   if (!userq->doorbell_ptr)
      goto fail;

   /* Create the Usermode Queue */
   r = amdgpu_create_userqueue(aws->dev, hw_ip_type, 0,
                               get_real_bo(amdgpu_winsys_bo(userq->doorbell_bo))->kms_handle,
                               AMDGPU_USERQ_DOORBELL_INDEX, amdgpu_bo_get_va(userq->ring_bo),
                               AMDGPU_USERQ_RING_SIZE, amdgpu_bo_get_va(userq->wptr_bo),
                               amdgpu_bo_get_va(userq->rptr_bo), mqd, &userq->q_id);
   if (r)
      fprintf(stderr, "amdgpu: failed to create userq\n");

   userq->init_once = true;
fail:
   simple_mtx_unlock(&userq->lock);
   if (r)
      return false;
   else
      return true;
}

void
amdgpu_userq_free(struct amdgpu_winsys *aws, struct amdgpu_userq *userq)
{
   if (userq->q_id)
      amdgpu_free_userqueue(aws->dev, userq->q_id);

   radeon_bo_reference(&aws->dummy_sws.base, &userq->ring_bo, NULL);
   radeon_bo_reference(&aws->dummy_sws.base, &userq->rptr_bo, NULL);
   radeon_bo_reference(&aws->dummy_sws.base, &userq->wptr_bo, NULL);
   radeon_bo_reference(&aws->dummy_sws.base, &userq->user_fence_bo, NULL);
   radeon_bo_reference(&aws->dummy_sws.base, &userq->doorbell_bo, NULL);

   switch (userq->ip_type) {
   case AMD_IP_GFX:
      radeon_bo_reference(&aws->dummy_sws.base, &userq->gfx_data.gds_bo, NULL);
      radeon_bo_reference(&aws->dummy_sws.base, &userq->gfx_data.csa_bo, NULL);
      radeon_bo_reference(&aws->dummy_sws.base, &userq->gfx_data.shadow_bo, NULL);
      break;
   case AMD_IP_COMPUTE:
      radeon_bo_reference(&aws->dummy_sws.base, &userq->compute_data.eop_bo, NULL);
      break;
   case AMD_IP_SDMA:
      break;
   default:
      fprintf(stderr, "amdgpu: userq unsupported for ip = %d\n", userq->ip_type);
   };
}
