/*
 * Copyright 2022 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "vn_feedback.h"

#include "vn_device.h"
#include "vn_physical_device.h"
#include "vn_queue.h"

/* coherent buffer with bound and mapped memory */
struct vn_feedback_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *data;

   struct list_head head;
};

static uint32_t
vn_get_memory_type_index(const VkPhysicalDeviceMemoryProperties *mem_props,
                         uint32_t type_bits,
                         VkFlags mask)
{
   uint32_t index = 0;
   while (type_bits && index < mem_props->memoryTypeCount) {
      if ((type_bits & 0x1) &&
          (mem_props->memoryTypes[index].propertyFlags & mask) == mask)
         return index;

      type_bits >>= 1;
      index++;
   }

   return UINT32_MAX;
}

static VkResult
vn_feedback_buffer_create(struct vn_device *dev,
                          uint32_t size,
                          const VkAllocationCallbacks *alloc,
                          struct vn_feedback_buffer **out_feedback_buf)
{
   const bool exclusive = dev->queue_family_count == 1;
   const VkPhysicalDeviceMemoryProperties *mem_props =
      &dev->physical_device->memory_properties.memoryProperties;
   VkDevice dev_handle = vn_device_to_handle(dev);
   VkBufferCreateInfo buf_create_info;
   VkMemoryRequirements *mem_req;
   VkMemoryAllocateInfo mem_alloc_info;
   VkBindBufferMemoryInfo bind_info;
   struct vn_buffer *buf;
   uint32_t mem_type_index;
   struct vn_feedback_buffer *feedback_buf;
   VkResult result;

   feedback_buf = vk_zalloc(alloc, sizeof(*feedback_buf), VN_DEFAULT_ALIGN,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!feedback_buf)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* use concurrent to avoid explicit queue family ownership transfer for
    * device created with queues from multiple queue families
    */
   buf_create_info = (VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode =
         exclusive ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      /* below favors the current venus protocol */
      .queueFamilyIndexCount = exclusive ? 0 : dev->queue_family_count,
      .pQueueFamilyIndices = exclusive ? NULL : dev->queue_families,
   };
   result = vn_CreateBuffer(dev_handle, &buf_create_info, alloc,
                            &feedback_buf->buffer);
   if (result != VK_SUCCESS)
      goto out_free_feedback_buf;

   buf = vn_buffer_from_handle(feedback_buf->buffer);
   mem_req = &buf->requirements.memory.memoryRequirements;
   mem_type_index =
      vn_get_memory_type_index(mem_props, mem_req->memoryTypeBits,
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (mem_type_index >= mem_props->memoryTypeCount) {
      result = VK_ERROR_INITIALIZATION_FAILED;
      goto out_destroy_buffer;
   }

   mem_alloc_info = (VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mem_req->size,
      .memoryTypeIndex = mem_type_index,
   };
   result = vn_AllocateMemory(dev_handle, &mem_alloc_info, alloc,
                              &feedback_buf->memory);
   if (result != VK_SUCCESS)
      goto out_destroy_buffer;

   bind_info = (VkBindBufferMemoryInfo){
      .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
      .buffer = feedback_buf->buffer,
      .memory = feedback_buf->memory,
      .memoryOffset = 0,
   };
   result = vn_BindBufferMemory2(dev_handle, 1, &bind_info);
   if (result != VK_SUCCESS)
      goto out_free_memory;

   result = vn_MapMemory(dev_handle, feedback_buf->memory, 0, VK_WHOLE_SIZE,
                         0, &feedback_buf->data);
   if (result != VK_SUCCESS)
      goto out_free_memory;

   *out_feedback_buf = feedback_buf;

   return VK_SUCCESS;

out_free_memory:
   vn_FreeMemory(dev_handle, feedback_buf->memory, alloc);

out_destroy_buffer:
   vn_DestroyBuffer(dev_handle, feedback_buf->buffer, alloc);

out_free_feedback_buf:
   vk_free(alloc, feedback_buf);

   return result;
}

static void
vn_feedback_buffer_destroy(struct vn_device *dev,
                           struct vn_feedback_buffer *feedback_buf,
                           const VkAllocationCallbacks *alloc)
{
   VkDevice dev_handle = vn_device_to_handle(dev);

   vn_UnmapMemory(dev_handle, feedback_buf->memory);
   vn_FreeMemory(dev_handle, feedback_buf->memory, alloc);
   vn_DestroyBuffer(dev_handle, feedback_buf->buffer, alloc);
   vk_free(alloc, feedback_buf);
}

static VkResult
vn_feedback_pool_grow(struct vn_feedback_pool *pool)
{
   VN_TRACE_FUNC();
   struct vn_feedback_buffer *feedback_buf = NULL;
   VkResult result;

   result = vn_feedback_buffer_create(pool->device, pool->size, pool->alloc,
                                      &feedback_buf);
   if (result != VK_SUCCESS)
      return result;

   pool->used = 0;

   list_add(&feedback_buf->head, &pool->feedback_buffers);

   return VK_SUCCESS;
}

VkResult
vn_feedback_pool_init(struct vn_device *dev,
                      struct vn_feedback_pool *pool,
                      uint32_t size,
                      const VkAllocationCallbacks *alloc)
{
   simple_mtx_init(&pool->mutex, mtx_plain);

   pool->device = dev;
   pool->alloc = alloc;
   pool->size = size;
   list_inithead(&pool->feedback_buffers);
   list_inithead(&pool->free_slots);

   return vn_feedback_pool_grow(pool);
}

void
vn_feedback_pool_fini(struct vn_feedback_pool *pool)
{
   list_for_each_entry_safe(struct vn_feedback_slot, slot, &pool->free_slots,
                            head)
      vk_free(pool->alloc, slot);

   list_for_each_entry_safe(struct vn_feedback_buffer, feedback_buf,
                            &pool->feedback_buffers, head)
      vn_feedback_buffer_destroy(pool->device, feedback_buf, pool->alloc);

   simple_mtx_destroy(&pool->mutex);
}

static struct vn_feedback_buffer *
vn_feedback_pool_alloc_internal(struct vn_feedback_pool *pool,
                                uint32_t size,
                                uint32_t *out_offset)
{
   VN_TRACE_FUNC();
   const uint32_t aligned_size = align(size, 4);

   if (unlikely(aligned_size > pool->size - pool->used)) {
      VkResult result = vn_feedback_pool_grow(pool);
      if (result != VK_SUCCESS)
         return NULL;

      assert(aligned_size <= pool->size - pool->used);
   }

   *out_offset = pool->used;
   pool->used += aligned_size;

   return list_first_entry(&pool->feedback_buffers, struct vn_feedback_buffer,
                           head);
}

struct vn_feedback_slot *
vn_feedback_pool_alloc(struct vn_feedback_pool *pool,
                       enum vn_feedback_type type)
{
   static const uint32_t slot_size = MAX2(sizeof(VkResult), sizeof(uint64_t));
   const VkAllocationCallbacks *alloc = pool->alloc;
   struct vn_feedback_buffer *feedback_buf;
   uint32_t offset;
   struct vn_feedback_slot *slot;

   simple_mtx_lock(&pool->mutex);
   if (!list_is_empty(&pool->free_slots)) {
      slot =
         list_first_entry(&pool->free_slots, struct vn_feedback_slot, head);
      list_del(&slot->head);
      simple_mtx_unlock(&pool->mutex);

      slot->type = type;
      return slot;
   }

   slot = vk_alloc(alloc, sizeof(*slot), VN_DEFAULT_ALIGN,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!slot) {
      simple_mtx_unlock(&pool->mutex);
      return NULL;
   }

   feedback_buf = vn_feedback_pool_alloc_internal(pool, slot_size, &offset);
   simple_mtx_unlock(&pool->mutex);

   if (!feedback_buf) {
      vk_free(alloc, slot);
      return NULL;
   }

   slot->type = type;
   slot->offset = offset;
   slot->buffer = feedback_buf->buffer;
   slot->data = feedback_buf->data + offset;

   return slot;
}

void
vn_feedback_pool_free(struct vn_feedback_pool *pool,
                      struct vn_feedback_slot *slot)
{
   simple_mtx_lock(&pool->mutex);
   list_add(&slot->head, &pool->free_slots);
   simple_mtx_unlock(&pool->mutex);
}

static void
vn_feedback_cmd_record_internal(VkCommandBuffer cmd_handle,
                                struct vn_feedback_slot *slot,
                                VkPipelineStageFlags stage_mask,
                                VkResult status)
{
   static const VkMemoryBarrier barrier_before = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = NULL,
      .srcAccessMask =
         VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
   };
   static const VkMemoryBarrier barrier_after = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = NULL,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT,
   };

   static_assert(sizeof(*slot->status) == 4);

   vn_CmdPipelineBarrier(cmd_handle, stage_mask,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &barrier_before, 0, NULL, 0, NULL);
   vn_CmdFillBuffer(cmd_handle, slot->buffer, slot->offset, 4, status);
   vn_CmdPipelineBarrier(cmd_handle, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier_after, 0,
                         NULL, 0, NULL);
}

void
vn_feedback_event_cmd_record(VkCommandBuffer cmd_handle,
                             VkEvent ev_handle,
                             VkPipelineStageFlags stage_mask,
                             VkResult status)
{
   /* for vkCmdSetEvent and vkCmdResetEvent interception */
   struct vn_event *ev = vn_event_from_handle(ev_handle);

   if (ev->feedback_slot) {
      /* TODO use VkBufferMemoryBarrier instead */
      vn_feedback_cmd_record_internal(cmd_handle, ev->feedback_slot,
                                      stage_mask |
                                         VK_PIPELINE_STAGE_HOST_BIT |
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      status);
   }
}
