/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include "vk_staging_buffer.hh"
#include "vk_context.hh"

namespace blender::gpu {

VKStagingBuffer::VKStagingBuffer(const VKBuffer &device_buffer,
                                 Direction direction,
                                 VkDeviceSize device_buffer_offset,
                                 VkDeviceSize region_size)
    : device_buffer_(device_buffer),
      device_buffer_offset_(device_buffer_offset),
      region_size_(region_size == UINT64_MAX ? device_buffer.size_in_bytes() : region_size)
{
  VkBufferUsageFlags usage;
  switch (direction) {
    case Direction::HostToDevice:
      usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      break;
    case Direction::DeviceToHost:
      usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }

  /* The host access hint has to match the direction. `HOST_ACCESS_SEQUENTIAL_WRITE` lets the
   * allocator hand out write combined (uncached) host memory, which is fast to fill from the CPU
   * but slow to read: a readback of a 12 MB storage buffer measured ~150 ms that way, versus
   * ~10 ms with `HOST_ACCESS_RANDOM` (which lets VMA pick cached memory). Uploads keep the write
   * hint, readbacks get the random access one. */
  const VmaAllocationCreateFlags host_access_flags =
      direction == Direction::HostToDevice ?
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT :
          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

  host_buffer_.create(region_size_,
                      usage,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                      VMA_ALLOCATION_CREATE_MAPPED_BIT | host_access_flags,
                      0.4f,
                      false,
                      "StagingBuffer");
}

void VKStagingBuffer::copy_to_device(VKContext &context)
{
  BLI_assert(host_buffer_.is_allocated() && host_buffer_.is_mapped());
  render_graph::VKCopyBufferNode::CreateInfo copy_buffer = {};
  copy_buffer.src_buffer = host_buffer_.vk_handle();
  copy_buffer.dst_buffer = device_buffer_.vk_handle();
  copy_buffer.region.dstOffset = device_buffer_offset_;
  copy_buffer.region.size = region_size_;

  context.render_graph().add_node(copy_buffer);
}

void VKStagingBuffer::copy_from_device(VKContext &context)
{
  BLI_assert(host_buffer_.is_allocated() && host_buffer_.is_mapped());
  render_graph::VKCopyBufferNode::CreateInfo copy_buffer = {};
  copy_buffer.src_buffer = device_buffer_.vk_handle();
  copy_buffer.dst_buffer = host_buffer_.vk_handle();
  copy_buffer.region.srcOffset = device_buffer_offset_;
  copy_buffer.region.size = region_size_;

  context.render_graph().add_node(copy_buffer);
}

void VKStagingBuffer::free()
{
  host_buffer_.free();
}

}  // namespace blender::gpu
