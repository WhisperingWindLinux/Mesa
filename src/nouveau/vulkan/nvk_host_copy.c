/*
 * Copyright © 2024 Valve Corp.
 * SPDX-License-Identifier: MIT
 */

#include "nvk_device.h"
#include "nvk_device_memory.h"
#include "nvk_entrypoints.h"
#include "nvk_format.h"
#include "nvk_image.h"

#include "vk_format.h"

static struct nil_Offset4D_Pixels
vk_to_nil_offset(VkOffset3D offset, uint32_t base_array_layer)
{
   return (struct nil_Offset4D_Pixels) {
      .x = offset.x,
      .y = offset.y,
      .z = offset.z,
      .a = base_array_layer
   };
}

static struct nil_Extent4D_Pixels
vk_to_nil_extent(VkExtent3D extent, uint32_t array_layers)
{
   return (struct nil_Extent4D_Pixels) {
      .width      = extent.width,
      .height     = extent.height,
      .depth      = extent.depth,
      .array_len  = array_layers,
   };
}

static void
memcpy2d(void *dst, size_t dst_stride_B,
         const void *src, size_t src_stride_B,
         size_t width_B, size_t height)
{
   if (dst_stride_B == width_B && src_stride_B == width_B) {
      memcpy(dst, src, width_B * height);
   } else {
      for (uint32_t y = 0; y < height; y++) {
         memcpy(dst, src, width_B);
         dst += dst_stride_B;
         src += src_stride_B;
      }
   }
}

/* TODO: remove helpers and merge them with Vk entrypoints. They are split just
 * for ease of editing. 
 */
/* TODO2: don't forget remapping for depth/stencil! */
static VkResult
nvk_copy_memory_to_image(struct nvk_image *dst,
                         const VkMemoryToImageCopyEXT *info,
                         bool no_swizzle)
{
   VkResult result;

   struct vk_image_buffer_layout buffer_layout =
      vk_memory_to_image_copy_layout(&dst->vk, info);

   const VkExtent3D extent_px =
      vk_image_sanitize_extent(&dst->vk, info->imageExtent);
   const uint32_t layer_count =
      vk_image_subresource_layer_count(&dst->vk, &info->imageSubresource);
   const struct nil_Extent4D_Pixels extent4d_px =
      vk_to_nil_extent(extent_px, layer_count);
   const VkOffset3D offset_px =
         vk_image_sanitize_offset(&dst->vk, info->imageOffset);
   const struct nil_Offset4D_Pixels offset4d_px =
         vk_to_nil_offset(offset_px, info->imageSubresource.baseArrayLayer);

   const VkImageAspectFlagBits aspects = info->imageSubresource.aspectMask;
   const uint8_t plane = nvk_image_aspects_to_plane(dst, aspects);
   const struct nvk_image_plane *dst_plane = &dst->planes[plane];

   const uint32_t dst_miplevel = info->imageSubresource.mipLevel;
   const unsigned bpp = util_format_get_blocksize(dst_plane->nil.format.p_format);

   struct nvk_device_memory *host_mem = dst_plane->host_mem;
   uint64_t host_offset = dst_plane->host_offset;
   void *mem_map_dst;
   result = nvkmd_mem_map(host_mem->mem, &host_mem->vk.base,
                          NVKMD_MEM_MAP_RDWR, NULL, &mem_map_dst);
   if (result != VK_SUCCESS)
      return result;

   struct nil_Extent4D_Elements extent_el =
      nil_extent4d_px_to_el(extent4d_px, dst_plane->nil.format,
                            dst_plane->nil.sample_layout);
   assert(extent_el.depth == 1 || extent_el.array_len == 1);

   const void *src_addr_B = info->pHostPointer;
   void *dst_addr_B = mem_map_dst + host_offset + 
                      dst_plane->nil.levels[dst_miplevel].offset_B;

   struct nil_Offset4D_Elements offset_el =
         nil_offset4d_px_to_el(offset4d_px, dst_plane->nil.format,
                               dst_plane->nil.sample_layout);
   dst_addr_B += offset_el.a * dst_plane->nil.array_stride_B;

   for (unsigned a = 0; a < layer_count; a++) {
      uint64_t layer_size_B = nil_image_level_size_B(&dst_plane->nil,
                                                     dst_miplevel);

      uint64_t src_layer_stride_B = no_swizzle ? layer_size_B :
                                    buffer_layout.image_stride_B;

      if (no_swizzle) {
         memcpy((void *) dst_addr_B, (void *) src_addr_B, src_layer_stride_B);
      } else if (!dst_plane->nil.levels[dst_miplevel].tiling.is_tiled) {
         assert(layer_count == 1);
         const unsigned dst_pitch_B =
            dst_plane->nil.levels[dst_miplevel].row_stride_B;
         memcpy2d((void *) dst_addr_B +
                  info->imageOffset.y * dst_pitch_B +
                  info->imageOffset.x * bpp,
                  dst_pitch_B,
                  (void *) src_addr_B,
                  buffer_layout.row_stride_B,
                  extent_px.width * bpp,
                  extent_px.height);
      } else {
         const struct nil_Extent4D_Pixels level_extent_px =
            nil_image_level_extent_px(&dst_plane->nil, dst_miplevel);
         const struct nil_Extent4D_Bytes level_extent_B =
            nil_extent4d_px_to_B(level_extent_px, dst_plane->nil.format,
                                 dst_plane->nil.sample_layout);
         const struct nil_Extent4D_Bytes extent4d_B =
            nil_extent4d_px_to_B(extent4d_px, dst_plane->nil.format,
                                 dst_plane->nil.sample_layout);
         const struct nil_Offset4D_Bytes offset4d_B =
            nil_offset4d_px_to_B(offset4d_px, dst_plane->nil.format,
                                 dst_plane->nil.sample_layout);
         
         nil_copy_linear_to_tiled(dst_addr_B,
                                  level_extent_B,
                                  src_addr_B,
                                  buffer_layout.row_stride_B,
                                  buffer_layout.image_stride_B,
                                  offset4d_B,
                                  extent4d_B,
                                  0,
                                  &dst_plane->nil.levels[dst_miplevel].tiling);
      }

      src_addr_B += src_layer_stride_B;
      dst_addr_B += dst_plane->nil.array_stride_B;
   }

   nvkmd_mem_unmap(host_mem->mem, NVKMD_MEM_MAP_RDWR);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CopyMemoryToImageEXT(VkDevice _device,
                         const VkCopyMemoryToImageInfoEXT *info)
{
   VK_FROM_HANDLE(nvk_image, dst_image, info->dstImage);

   VkResult result;

   /* From the EXT spec:
    * VK_HOST_IMAGE_COPY_MEMCPY_EXT specifies that no memory layout swizzling is
    * to be applied during data copy. For copies between memory and images, this
    * flag indicates that image data in host memory is swizzled in exactly the
    * same way as the image data on the device. Using this flag indicates that
    * the implementations may use a simple memory copy to transfer the data
    * between the host memory and the device memory. The format of the swizzled
    * data in host memory is platform dependent and is not defined in this
    * specification.
    */
   const bool no_swizzle = info->flags &
      VK_HOST_IMAGE_COPY_MEMCPY_EXT;

   for (unsigned r = 0; r < info->regionCount; r++) {
      result = nvk_copy_memory_to_image(dst_image, &info->pRegions[r],
                                        no_swizzle);
      if (result != VK_SUCCESS)
         return result;
   }

   return result;
}

static VkResult
nvk_copy_image_to_memory(struct nvk_image *src,
                         const VkImageToMemoryCopyEXT *info,
                         bool no_swizzle)
{
   VkResult result;

   struct vk_image_buffer_layout buffer_layout =
      vk_image_to_memory_copy_layout(&src->vk, info);

   const VkExtent3D extent_px =
      vk_image_sanitize_extent(&src->vk, info->imageExtent);
   const uint32_t layer_count =
      vk_image_subresource_layer_count(&src->vk, &info->imageSubresource);
   const struct nil_Extent4D_Pixels extent4d_px =
      vk_to_nil_extent(extent_px, layer_count);
   const VkOffset3D offset_px =
         vk_image_sanitize_offset(&src->vk, info->imageOffset);
   const struct nil_Offset4D_Pixels offset4d_px =
         vk_to_nil_offset(offset_px, info->imageSubresource.baseArrayLayer);

   const VkImageAspectFlagBits aspects = info->imageSubresource.aspectMask;
   const uint8_t plane = nvk_image_aspects_to_plane(src, aspects);
   struct nvk_image_plane *src_plane = &src->planes[plane];

   const uint32_t src_miplevel = info->imageSubresource.mipLevel;
   const unsigned bpp = util_format_get_blocksize(src_plane->nil.format.p_format);

   struct nvk_device_memory *host_mem = src->planes[plane].host_mem;
   uint64_t host_offset = src->planes[plane].host_offset;
   void *mem_map_src;
   result = nvkmd_mem_map(host_mem->mem, &host_mem->vk.base,
                          NVKMD_MEM_MAP_RDWR, NULL, &mem_map_src);
   if (result != VK_SUCCESS)
      return result;

   struct nil_Extent4D_Elements extent_el =
      nil_extent4d_px_to_el(extent4d_px, src_plane->nil.format,
                            src_plane->nil.sample_layout);
   assert(extent_el.depth == 1 || extent_el.array_len == 1);

   const void *src_addr_B = mem_map_src + host_offset +
                            src_plane->nil.levels[src_miplevel].offset_B;
   void *dst_addr_B = info->pHostPointer;

   struct nil_Offset4D_Elements offset_el =
         nil_offset4d_px_to_el(offset4d_px, src_plane->nil.format,
                               src_plane->nil.sample_layout);
   src_addr_B += offset_el.a * src_plane->nil.array_stride_B;

   for (unsigned a = 0; a < layer_count; a++) {
      uint64_t layer_size_B = nil_image_level_size_B(&src_plane->nil,
                                                     src_miplevel);

      uint64_t dst_layer_stride_B = no_swizzle ? layer_size_B :
                                    buffer_layout.image_stride_B;

      if (no_swizzle) {
         memcpy((void *) dst_addr_B, (void *) src_addr_B, dst_layer_stride_B);
      } else if (!src_plane->nil.levels[src_miplevel].tiling.is_tiled) {
         assert(layer_count == 1);
         const unsigned src_pitch_B =
            src_plane->nil.levels[src_miplevel].row_stride_B;
         memcpy2d((void *) dst_addr_B,
                  buffer_layout.row_stride_B,
                  (void *) src_addr_B +
                  info->imageOffset.y * src_pitch_B +
                  info->imageOffset.x * bpp,
                  src_pitch_B,
                  extent_px.width * bpp,
                  extent_px.height);
      } else {
         const struct nil_Extent4D_Pixels level_extent_px =
            nil_image_level_extent_px(&src_plane->nil, src_miplevel);
         const struct nil_Extent4D_Bytes level_extent_B =
            nil_extent4d_px_to_B(level_extent_px, src_plane->nil.format,
                                 src_plane->nil.sample_layout);
         const struct nil_Extent4D_Bytes extent4d_B =
            nil_extent4d_px_to_B(extent4d_px, src_plane->nil.format,
                                 src_plane->nil.sample_layout);
         const struct nil_Offset4D_Bytes offset4d_B =
            nil_offset4d_px_to_B(offset4d_px, src_plane->nil.format,
                                 src_plane->nil.sample_layout);

         nil_copy_tiled_to_linear(dst_addr_B,
                                  buffer_layout.row_stride_B,
                                  buffer_layout.image_stride_B,
                                  src_addr_B,
                                  level_extent_B,
                                  offset4d_B,
                                  extent4d_B,
                                  0,
                                  &src_plane->nil.levels[src_miplevel].tiling);
      }

      src_addr_B += src_plane->nil.array_stride_B;
      dst_addr_B += dst_layer_stride_B;
   }

   nvkmd_mem_unmap(host_mem->mem, NVKMD_MEM_MAP_RDWR);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CopyImageToMemoryEXT(VkDevice _device,
                         const VkCopyImageToMemoryInfoEXT *info)
{
   VK_FROM_HANDLE(nvk_image, image, info->srcImage);

   VkResult result;

   const bool no_swizzle = info->flags &
      VK_HOST_IMAGE_COPY_MEMCPY_EXT;

   for (unsigned r = 0; r < info->regionCount; r++) {
      result = nvk_copy_image_to_memory(image, &info->pRegions[r],
                                        no_swizzle);
      if (result != VK_SUCCESS)
         return result;
   }

   return result;
}

static VkResult
nvk_copy_image_to_image(struct nvk_device *device,
                        struct nvk_image *src,
                        struct nvk_image *dst,
                        const VkImageCopy2 *info)
{
   VkResult result;

   /* From the Vulkan 1.3.217 spec:
    *
    *    "When copying between compressed and uncompressed formats the
    *    extent members represent the texel dimensions of the source image
    *    and not the destination."
    */
   const VkExtent3D extent_px =
      vk_image_sanitize_extent(&src->vk, info->extent);
   const uint32_t layer_count =
      vk_image_subresource_layer_count(&src->vk, &info->srcSubresource);
   const struct nil_Extent4D_Pixels extent4d_px =
      vk_to_nil_extent(extent_px, layer_count);

   const VkOffset3D src_offset_px =
      vk_image_sanitize_offset(&src->vk, info->srcOffset);
   const struct nil_Offset4D_Pixels src_offset4d_px =
      vk_to_nil_offset(src_offset_px, info->srcSubresource.baseArrayLayer);

   const VkOffset3D dst_offset_px =
      vk_image_sanitize_offset(&dst->vk, info->dstOffset);
   const struct nil_Offset4D_Pixels dst_offset4d_px =
      vk_to_nil_offset(dst_offset_px, info->dstSubresource.baseArrayLayer);

   const VkImageAspectFlagBits src_aspects =
      info->srcSubresource.aspectMask;
   const uint8_t src_plane = nvk_image_aspects_to_plane(src, src_aspects);
   struct nvk_image_plane *src_img_plane = &dst->planes[src_plane];

   const VkImageAspectFlagBits dst_aspects =
      info->dstSubresource.aspectMask;
   const uint8_t dst_plane = nvk_image_aspects_to_plane(dst, dst_aspects);
   struct nvk_image_plane *dst_img_plane = &dst->planes[dst_plane];

   const uint32_t src_miplevel = info->srcSubresource.mipLevel;
   const unsigned src_bpp = util_format_get_blocksize(src_img_plane->nil.format.p_format);

   const uint32_t dst_miplevel = info->dstSubresource.mipLevel;
   const unsigned dst_bpp = util_format_get_blocksize(dst_img_plane->nil.format.p_format);

   struct nvk_device_memory *src_host_mem = src->planes[src_plane].host_mem;
   uint64_t src_host_offset = src->planes[src_plane].host_offset;
   void *mem_map_src;
   result = nvkmd_mem_map(src_host_mem->mem, &src_host_mem->vk.base, NVKMD_MEM_MAP_RDWR, NULL, &mem_map_src);
   if (result != VK_SUCCESS)
      return result;

   struct nvk_device_memory *dst_host_mem = src->planes[dst_plane].host_mem;
   uint64_t dst_host_offset = src->planes[dst_plane].host_offset;
   void *mem_map_dst;
   result = nvkmd_mem_map(dst_host_mem->mem, &dst_host_mem->vk.base, NVKMD_MEM_MAP_RDWR, NULL, &mem_map_dst);
   if (result != VK_SUCCESS)
      return result;

   uint32_t src_layer_stride_B =
      src_img_plane->nil.levels[src_miplevel].row_stride_B;

   struct nil_Offset4D_Elements src_offset_el =
      nil_offset4d_px_to_el(src_offset4d_px, src_img_plane->nil.format,
                            src_img_plane->nil.sample_layout);

   uint32_t dst_layer_stride_B =
      dst_img_plane->nil.levels[dst_miplevel].row_stride_B;

   struct nil_Offset4D_Elements dst_offset_el =
      nil_offset4d_px_to_el(dst_offset4d_px, dst_img_plane->nil.format,
                            dst_img_plane->nil.sample_layout);

   void *src_addr_B = mem_map_src + src_host_offset +
                      src_img_plane->nil.levels[src_miplevel].offset_B;
   src_addr_B += src_offset_el.a * src_img_plane->nil.array_stride_B;
   void *dst_addr_B = mem_map_dst + dst_host_offset +
                      dst_img_plane->nil.levels[dst_miplevel].offset_B;
   dst_addr_B += dst_offset_el.a * dst_img_plane->nil.array_stride_B;

   if (!src_img_plane->nil.levels[src_miplevel].tiling.is_tiled) {
      assert(src_img_plane->nil.dim == NIL_IMAGE_DIM_2D);
      assert(src_img_plane->nil.extent_px.array_len == 1);
      assert(extent4d_px.depth == 1 && extent4d_px.array_len == 1);
   }

   if (!dst_img_plane->nil.levels[src_miplevel].tiling.is_tiled) {
      assert(dst_img_plane->nil.dim == NIL_IMAGE_DIM_2D);
      assert(dst_img_plane->nil.extent_px.array_len == 1);
      assert(extent4d_px.depth == 1 && extent4d_px.array_len == 1);
   }

   if (!src_img_plane->nil.levels[src_miplevel].tiling.is_tiled &&
       !dst_img_plane->nil.levels[dst_miplevel].tiling.is_tiled) {
      memcpy2d((void *) dst_addr_B +
               info->dstOffset.y * dst_layer_stride_B +
               info->dstOffset.x * dst_bpp,
               dst_layer_stride_B,
               (void *) src_addr_B +
               info->srcOffset.y * src_layer_stride_B +
               info->srcOffset.x * src_bpp,
               src_layer_stride_B,
               extent_px.width * src_bpp,
               extent_px.height);
      } else if (!src_img_plane->nil.levels[src_miplevel].tiling.is_tiled) {
         const struct nil_Extent4D_Pixels level_extent_px =
            nil_image_level_extent_px(&dst_img_plane->nil, dst_miplevel);
         const struct nil_Extent4D_Bytes level_extent_B =
            nil_extent4d_px_to_B(level_extent_px, dst_img_plane->nil.format,
                                 dst_img_plane->nil.sample_layout);
         const struct nil_Extent4D_Bytes extent4d_B =
            nil_extent4d_px_to_B(extent4d_px, dst_img_plane->nil.format,
                                 dst_img_plane->nil.sample_layout);
         const struct nil_Offset4D_Bytes offset4d_B =
            nil_offset4d_px_to_B(dst_offset4d_px, dst_img_plane->nil.format,
                                 dst_img_plane->nil.sample_layout);

         nil_copy_linear_to_tiled(dst_addr_B,
                                  level_extent_B,
                                  src_addr_B + src_layer_stride_B * info->srcOffset.y + info->srcOffset.x * src_bpp,
                                  extent_px.width * dst_bpp,
                                  extent_px.width * extent_px.height * dst_bpp,
                                  offset4d_B,
                                  extent4d_B,
                                  0,
                                  &dst_img_plane->nil.levels[dst_miplevel].tiling);
      } else if (!dst_img_plane->nil.levels[dst_miplevel].tiling.is_tiled) {
         const struct nil_Extent4D_Pixels level_extent_px =
            nil_image_level_extent_px(&src_img_plane->nil, src_miplevel);
         const struct nil_Extent4D_Bytes level_extent_B =
            nil_extent4d_px_to_B(level_extent_px, src_img_plane->nil.format,
                                 src_img_plane->nil.sample_layout);
         const struct nil_Extent4D_Bytes extent4d_B =
            nil_extent4d_px_to_B(extent4d_px, src_img_plane->nil.format,
                                 src_img_plane->nil.sample_layout);
         const struct nil_Offset4D_Bytes offset4d_B =
            nil_offset4d_px_to_B(src_offset4d_px, src_img_plane->nil.format,
                                 src_img_plane->nil.sample_layout);

         nil_copy_tiled_to_linear(dst_addr_B + dst_layer_stride_B * info->dstOffset.y + info->dstOffset.x * dst_bpp,
                                  extent_px.width * dst_bpp,
                                  extent_px.width * extent_px.height * src_bpp,
                                  src_addr_B,
                                  level_extent_B,
                                  offset4d_B,
                                  extent4d_B,
                                  0,
                                  &src_img_plane->nil.levels[src_miplevel].tiling);
      } else {
         size_t temp_tile_size_B = dst_img_plane->nil.size_B;
         void *tmp_mem = vk_alloc(&device->vk.alloc, temp_tile_size_B, 8,
                                  VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
         for (unsigned a = 0; a < layer_count; a++) {
            const struct nil_Extent4D_Pixels src_level_extent_px =
               nil_image_level_extent_px(&src_img_plane->nil, src_miplevel);
            const struct nil_Extent4D_Bytes src_level_extent_B =
               nil_extent4d_px_to_B(src_level_extent_px, src_img_plane->nil.format,
                                    src_img_plane->nil.sample_layout);
            const struct nil_Extent4D_Bytes extent4d_B =
               nil_extent4d_px_to_B(extent4d_px, src_img_plane->nil.format,
                                    src_img_plane->nil.sample_layout);
            const struct nil_Offset4D_Bytes src_offset4d_B =
               nil_offset4d_px_to_B(src_offset4d_px, src_img_plane->nil.format,
                                    src_img_plane->nil.sample_layout);
            
            const struct nil_Extent4D_Pixels dst_level_extent_px =
               nil_image_level_extent_px(&dst_img_plane->nil, dst_miplevel);
            const struct nil_Extent4D_Bytes dst_level_extent_B =
               nil_extent4d_px_to_B(dst_level_extent_px, dst_img_plane->nil.format,
                                    dst_img_plane->nil.sample_layout);
            const struct nil_Offset4D_Bytes dst_offset4d_B =
               nil_offset4d_px_to_B(dst_offset4d_px, dst_img_plane->nil.format,
                                 dst_img_plane->nil.sample_layout);

            nil_copy_tiled_to_linear(tmp_mem,
                                     extent_px.width * dst_bpp,
                                     extent_px.width * extent_px.height * src_bpp,
                                     src_addr_B,
                                     src_level_extent_B,
                                     src_offset4d_B,
                                     extent4d_B,
                                     0,
                                     &src_img_plane->nil.levels[src_miplevel].tiling);
            
            nil_copy_linear_to_tiled(dst_addr_B,
                                     dst_level_extent_B,
                                     tmp_mem,
                                     extent_px.width * dst_bpp,
                                     extent_px.width * extent_px.height * dst_bpp,
                                     dst_offset4d_B,
                                     extent4d_B,
                                     0,
                                     &dst_img_plane->nil.levels[dst_miplevel].tiling);
            src_addr_B += src_img_plane->nil.array_stride_B;
            dst_addr_B += dst_img_plane->nil.array_stride_B;
         }
         vk_free(&device->vk.alloc, tmp_mem);
      }

   nvkmd_mem_unmap(src_host_mem->mem, NVKMD_MEM_MAP_RDWR);
   nvkmd_mem_unmap(dst_host_mem->mem, NVKMD_MEM_MAP_RDWR);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CopyImageToImageEXT(VkDevice _device,
                        const VkCopyImageToImageInfoEXT *pCopyImageToImageInfo)
{
   VK_FROM_HANDLE(nvk_device, device, _device);
   VK_FROM_HANDLE(nvk_image, src, pCopyImageToImageInfo->srcImage);
   VK_FROM_HANDLE(nvk_image, dst, pCopyImageToImageInfo->dstImage);

   VkResult result;

   for (unsigned r = 0; r < pCopyImageToImageInfo->regionCount; r++) {
      result = nvk_copy_image_to_image(device, src, dst,
                                       pCopyImageToImageInfo->pRegions + r);
      if (result != VK_SUCCESS)
         return result;
   }

   return result;
}


VKAPI_ATTR VkResult VKAPI_CALL
nvk_TransitionImageLayoutEXT(VkDevice device,
                             uint32_t transitionCount,
                             const VkHostImageLayoutTransitionInfoEXT *transitions)
{
   /* Nothing to do here */
   return VK_SUCCESS;
}
