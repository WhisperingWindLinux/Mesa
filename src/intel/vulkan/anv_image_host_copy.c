/* Copyright © 2024 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdbool.h>

#include "anv_private.h"
#include "util/u_debug.h"
#include "vk_util.h"

static void
get_image_offset_el(const struct isl_surf *surf, unsigned level, unsigned z,
                    uint32_t *out_x0_el, uint32_t *out_y0_el)
{
   ASSERTED uint32_t z0_el, a0_el;
   if (surf->dim == ISL_SURF_DIM_3D) {
      isl_surf_get_image_offset_el(surf, level, 0, z,
                                   out_x0_el, out_y0_el, &z0_el, &a0_el);
   } else {
      isl_surf_get_image_offset_el(surf, level, z, 0,
                                   out_x0_el, out_y0_el, &z0_el, &a0_el);
   }
   assert(z0_el == 0 && a0_el == 0);
}

/* Compute extent parameters for use with tiled_memcpy functions.
 * xs are in units of bytes and ys are in units of strides.
 */
static inline void
tile_extents(const struct isl_surf *surf,
             const VkOffset3D *offset,
             const VkExtent3D *extent,
             unsigned level, int z,
             uint32_t *x1_B, uint32_t *x2_B,
             uint32_t *y1_el, uint32_t *y2_el)
{
   const struct isl_format_layout *fmtl = isl_format_get_layout(surf->format);
   const unsigned cpp = fmtl->bpb / 8;

   assert(offset->x % fmtl->bw == 0);
   assert(offset->y % fmtl->bh == 0);

   unsigned x0_el, y0_el;
   get_image_offset_el(surf, level, z, &x0_el, &y0_el);

   *x1_B = (offset->x / fmtl->bw + x0_el) * cpp;
   *y1_el = offset->y / fmtl->bh + y0_el;
   *x2_B = (DIV_ROUND_UP(offset->x + extent->width, fmtl->bw) + x0_el) * cpp;
   *y2_el = DIV_ROUND_UP(offset->y + extent->height, fmtl->bh) + y0_el;
}

static void
anv_copy_image_memory(struct anv_device *device,
                      const struct isl_surf *surf,
                      const struct anv_image_binding *binding,
                      uint64_t binding_offset,
                      void *mem_ptr,
                      uint32_t mem_row_length,
                      uint32_t mem_img_height,
                      const VkOffset3D *offset,
                      const VkExtent3D *extent,
                      uint32_t base_array_layer,
                      uint32_t layer_count,
                      uint32_t level,
                      bool mem_to_img)
{
   const struct isl_format_layout *fmt_layout =
      isl_format_get_layout(surf->format);
   const uint32_t bpp = fmt_layout->bpb / 8;
   void *img_ptr = binding->host_map + binding->map_delta + binding_offset;

   /* Memory distance between each row */
   uint32_t mem_row_pitch = mem_row_length ?
                            (bpp * mem_row_length) :
                            bpp * DIV_ROUND_UP(extent->width, fmt_layout->bw);
   /* Memory distance between each slice (1 3D level or 1 array layer) */
   uint32_t mem_height_pitch = mem_img_height ?
                               (mem_img_height * mem_row_pitch) :
                               (extent->height * mem_row_pitch);

   for (uint32_t a = 0; a < layer_count; a++) {
      for (uint32_t z = 0; z < extent->depth; z++) {
         struct isl_surf sub_surf;
         uint32_t x_offset_sa, y_offset_sa;
         uint64_t offset_B;
         isl_surf_get_image_surf(&device->physical->isl_dev, surf,
                                 level,
                                 base_array_layer + a,
                                 offset->z + z,
                                 &sub_surf,
                                 &offset_B,
                                 &x_offset_sa,
                                 &y_offset_sa);

         if (device->physical->memory.need_flush && !mem_to_img)
            intel_invalidate_range(img_ptr + offset_B, sub_surf.size_B);

         if (surf->tiling == ISL_TILING_LINEAR) {
            uint64_t mem_row_offset = mem_height_pitch * MAX2(a, z);
            uint64_t img_row_offset = ((x_offset_sa + offset->x) / fmt_layout->bw) * bpp;
            uint64_t row_copy_size = MIN2(surf->row_pitch_B - img_row_offset,
                                          (extent->width / fmt_layout->bw) * bpp);
            for (uint32_t h = 0; h < extent->height; h += fmt_layout->bh) {
               uint64_t img_offset =
                  offset_B +
                  ((y_offset_sa + h + offset->y) / fmt_layout->bh) * surf->row_pitch_B +
                  img_row_offset;
               assert((img_offset + row_copy_size) <= binding->memory_range.size);

               if (mem_to_img)
                  memcpy(img_ptr + img_offset, mem_ptr + mem_row_offset, row_copy_size);
               else
                  memcpy(mem_ptr + mem_row_offset, img_ptr + img_offset, row_copy_size);

               mem_row_offset += mem_row_pitch;
            }
         } else {
            uint32_t x1, x2, y1, y2;
            tile_extents(surf, offset, extent, level,
                         MAX2(offset->z + z, base_array_layer + a),
                         &x1, &x2, &y1, &y2);

            if (mem_to_img) {
               isl_memcpy_linear_to_tiled(x1, x2, y1, y2,
                                          img_ptr, mem_ptr + mem_height_pitch * MAX2(a, z),
                                          surf->row_pitch_B,
                                          mem_row_pitch,
                                          false,
                                          surf->tiling,
                                          ISL_MEMCPY);
            } else {
               isl_memcpy_tiled_to_linear(x1, x2, y1, y2,
                                          mem_ptr + mem_height_pitch * MAX2(a, z), img_ptr,
                                          mem_row_pitch,
                                          surf->row_pitch_B,
                                          false,
                                          surf->tiling,
                                          ISL_MEMCPY);
            }
         }

         if (device->physical->memory.need_flush && mem_to_img)
            intel_flush_range(img_ptr + offset_B, sub_surf.size_B);
      }
   }
}

VkResult
anv_CopyMemoryToImageEXT(
    VkDevice                                    _device,
    const VkCopyMemoryToImageInfoEXT*           pCopyMemoryToImageInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_image, image, pCopyMemoryToImageInfo->dstImage);

   for (uint32_t r = 0; r < pCopyMemoryToImageInfo->regionCount; r++) {
      const VkMemoryToImageCopyEXT *region = &pCopyMemoryToImageInfo->pRegions[r];

      anv_foreach_image_aspect_bit(aspect_bit, image, region->imageSubresource.aspectMask) {
         const VkImageAspectFlags aspect = 1U << aspect_bit;
         const uint32_t plane =
            anv_image_aspect_to_plane(image, aspect);
         const struct anv_surface *anv_surf =
            &image->planes[plane].primary_surface;
         const struct isl_surf *surf = &anv_surf->isl;
         const struct anv_image_binding *binding =
            &image->bindings[anv_surf->memory_range.binding];

         assert(binding->host_map != NULL);

         anv_copy_image_memory(device, surf,
                               binding, anv_surf->memory_range.offset,
                               (void *)region->pHostPointer,
                               region->memoryRowLength,
                               region->memoryImageHeight,
                               &region->imageOffset,
                               &region->imageExtent,
                               region->imageSubresource.baseArrayLayer,
                               region->imageSubresource.layerCount,
                               region->imageSubresource.mipLevel,
                               true /* mem_to_img */);
      }
   }

   return VK_SUCCESS;
}

VkResult
anv_CopyImageToMemoryEXT(
    VkDevice                                    _device,
    const VkCopyImageToMemoryInfoEXT*           pCopyImageToMemoryInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_image, image, pCopyImageToMemoryInfo->srcImage);

   for (uint32_t r = 0; r < pCopyImageToMemoryInfo->regionCount; r++) {
      const VkImageToMemoryCopyEXT *region = &pCopyImageToMemoryInfo->pRegions[r];

      anv_foreach_image_aspect_bit(aspect_bit, image, region->imageSubresource.aspectMask) {
         const VkImageAspectFlags aspect = 1U << aspect_bit;
         const uint32_t plane =
            anv_image_aspect_to_plane(image, aspect);
         const struct anv_surface *anv_surf =
            &image->planes[plane].primary_surface;
         const struct isl_surf *surf = &anv_surf->isl;
         const struct anv_image_binding *binding =
            &image->bindings[anv_surf->memory_range.binding];

         assert(binding->host_map != NULL);

         anv_copy_image_memory(device, surf,
                               binding, anv_surf->memory_range.offset,
                               region->pHostPointer,
                               region->memoryRowLength,
                               region->memoryImageHeight,
                               &region->imageOffset,
                               &region->imageExtent,
                               region->imageSubresource.baseArrayLayer,
                               region->imageSubresource.layerCount,
                               region->imageSubresource.mipLevel,
                               false /* mem_to_img */);
      }
   }

   return VK_SUCCESS;
}

VkResult
anv_CopyImageToImageEXT(
    VkDevice                                    _device,
    const VkCopyImageToImageInfoEXT*            pCopyImageToImageInfo)
{
   ANV_FROM_HANDLE(anv_device, device, _device);
   ANV_FROM_HANDLE(anv_image, src_image, pCopyImageToImageInfo->srcImage);
   ANV_FROM_HANDLE(anv_image, dst_image, pCopyImageToImageInfo->dstImage);

   /* Work with a tile's worth of data */
   void *tmp_map = vk_alloc(&device->vk.alloc, 4096, 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
   if (tmp_map == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t r = 0; r < pCopyImageToImageInfo->regionCount; r++) {
      const VkImageCopy2 *region = &pCopyImageToImageInfo->pRegions[r];

      anv_foreach_image_aspect_bit(aspect_bit, src_image, region->srcSubresource.aspectMask) {
         const VkImageAspectFlags aspect = 1U << aspect_bit;
         const uint32_t src_plane = anv_image_aspect_to_plane(src_image, aspect);
         const uint32_t dst_plane = anv_image_aspect_to_plane(dst_image, aspect);
         const struct anv_surface *src_anv_surf =
            &src_image->planes[src_plane].primary_surface;
         const struct anv_surface *dst_anv_surf =
            &dst_image->planes[dst_plane].primary_surface;
         const struct isl_surf *src_surf = &src_anv_surf->isl;
         const struct isl_surf *dst_surf = &dst_anv_surf->isl;
         const struct anv_image_binding *src_binding =
            &src_image->bindings[src_anv_surf->memory_range.binding];
         const struct anv_image_binding *dst_binding =
            &dst_image->bindings[dst_anv_surf->memory_range.binding];

         struct isl_tile_info src_tile;
         struct isl_tile_info dst_tile;

         isl_surf_get_tile_info(src_surf, &src_tile);
         isl_surf_get_tile_info(dst_surf, &dst_tile);

         uint32_t tile_width, tile_height;
         if (src_tile.phys_extent_B.w > dst_tile.phys_extent_B.w) {
            tile_width = src_tile.logical_extent_el.w;
            tile_height = src_tile.logical_extent_el.h;
         } else {
            tile_width = dst_tile.logical_extent_el.w;
            tile_height = dst_tile.logical_extent_el.h;
         }

         if (tile_width == 1 && tile_height == 1) {
            tile_width = MIN2(4096 / (src_tile.format_bpb / 8), region->extent.width);
            tile_height = 4096 / (tile_width * (src_tile.format_bpb / 8));
         }

         for (uint32_t a = 0; a < region->srcSubresource.layerCount; a++) {
            for (uint32_t z = 0; z < region->extent.depth; z++) {
               for (uint32_t y = 0; y < region->extent.height; y += tile_height) {
                  for (uint32_t x = 0; x < region->extent.width; x += tile_width) {
                     VkOffset3D src_offset = {
                        .x = region->srcOffset.x + x,
                        .y = region->srcOffset.y + y,
                        .z = region->srcOffset.z + z,
                     };
                     VkOffset3D dst_offset = {
                        .x = region->dstOffset.x + x,
                        .y = region->dstOffset.y + y,
                        .z = region->dstOffset.z + z,
                     };
                     VkExtent3D extent = {
                        .width  = MIN2(region->extent.width - src_offset.x,
                                       tile_width),
                        .height = MIN2(region->extent.height - src_offset.y,
                                       tile_height),
                        .depth  = 1,
                     };

                     anv_copy_image_memory(device, src_surf,
                                           src_binding,
                                           src_anv_surf->memory_range.offset,
                                           tmp_map,
                                           tile_width, tile_height,
                                           &src_offset, &extent,
                                           region->srcSubresource.baseArrayLayer + a, 1,
                                           region->srcSubresource.mipLevel,
                                           false /* mem_to_img */);
                     anv_copy_image_memory(device, dst_surf,
                                           dst_binding,
                                           dst_anv_surf->memory_range.offset,
                                           tmp_map,
                                           tile_width, tile_height,
                                           &dst_offset, &extent,
                                           region->dstSubresource.baseArrayLayer + a, 1,
                                           region->dstSubresource.mipLevel,
                                           true /* mem_to_img */);
                  }
               }
            }
         }
      }
   }

   vk_free(&device->vk.alloc, tmp_map);

   return VK_SUCCESS;
}

VkResult
anv_TransitionImageLayoutEXT(
    VkDevice                                    device,
    uint32_t                                    transitionCount,
    const VkHostImageLayoutTransitionInfoEXT*   pTransitions)
{
   /* Our layout transitions are mostly about resolving the auxiliary surface
    * into the main surface. Since we disable the auxiliary surface, there is
    * nothing here for us to do.
    */
   return VK_SUCCESS;
}
