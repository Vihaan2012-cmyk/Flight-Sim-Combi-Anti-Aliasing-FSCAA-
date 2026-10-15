// cmaa2_file.cpp - run the shipped CMAA2 (src/vklayer/cmaa2.h, in place, as the layer does after the
// resolve) on one RGBA16F image read from disk, and write the result. Used by tools/nn/fscaa_eval.py to
// score CMAA2 against the burst-accumulated truth.
// Usage: cmaa2_file <in.rgba16f> <width> <height> <out.rgba16f> [preset 0..3]
// SPDX-License-Identifier: GPL-3.0-or-later
#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cmaa2.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d\n", #x, (int)r_); exit(1); } } while (0)
static uint32_t W = 0, H = 0;

struct Ctx {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; VkCommandPool pool;
    VkPhysicalDeviceMemoryProperties mp;
};

static uint32_t memType(const Ctx &c, uint32_t bits, VkMemoryPropertyFlags f)
{
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (c.mp.memoryTypes[i].propertyFlags & f) == f) return i;
    return 0;
}

// One CMAA2 run; returns the RGBA16F result and Intel's control counters.
static std::vector<uint16_t> runOnce(Ctx &c, const std::vector<uint16_t> &pix, bool inplace, int preset, uint32_t ctr[16])
{
    cmaa2::Fns f;
    if (!cmaa2::loadFns(f, vkGetDeviceProcAddr, c.dev, c.mp)) { printf("FAIL loadFns\n"); exit(1); }
    cmaa2::State s;
    if (!cmaa2::create(f, s, c.dev, W, H, !inplace)) { printf("FAIL create (%s)\n", inplace ? "in place" : "copy"); exit(1); }

    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D; ic.format = VK_FORMAT_R16G16B16A16_SFLOAT; ic.extent = {W, H, 1};
    ic.mipLevels = 1; ic.arrayLayers = 1; ic.samples = VK_SAMPLE_COUNT_1_BIT; ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (inplace ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    VkImage src; CK(vkCreateImage(c.dev, &ic, nullptr, &src));
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(c.dev, src, &mr);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = mr.size; ma.memoryTypeIndex = memType(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory srcMem; CK(vkAllocateMemory(c.dev, &ma, nullptr, &srcMem)); CK(vkBindImageMemory(c.dev, src, srcMem, 0));
    const VkDeviceSize bytes = (VkDeviceSize)W * H * 8;
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bc.size = bytes; bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer stg; CK(vkCreateBuffer(c.dev, &bc, nullptr, &stg));
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(c.dev, stg, &br);
    VkMemoryAllocateInfo bm{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; bm.allocationSize = br.size;
    bm.memoryTypeIndex = memType(c, br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory stgMem; CK(vkAllocateMemory(c.dev, &bm, nullptr, &stgMem)); CK(vkBindBufferMemory(c.dev, stg, stgMem, 0));
    void *map; CK(vkMapMemory(c.dev, stgMem, 0, bytes, 0, &map)); memcpy(map, pix.data(), (size_t)bytes);

    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = c.pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(c.dev, &ca, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bi));
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = src; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {W, H, 1};
    vkCmdCopyBufferToImage(cb, stg, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    // In-place runs once (it modifies its input, as the layer does once per frame); copy mode runs
    // twice so the second pass sees the heads/counters state the first left behind - its output
    // is the same single-pass result because the copy restarts from the untouched source.
    const bool ok = inplace
        ? cmaa2::recordInPlace(f, s, cb, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, preset)
        : (cmaa2::record(f, s, cb, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, preset) &&
           cmaa2::record(f, s, cb, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, preset));
    if (!ok) { printf("FAIL record (%s)\n", inplace ? "in place" : "copy"); exit(1); }
    cmaa2::recordCounterCopy(f, s, cb);
    const VkImage result = inplace ? src : s.color;
    b.image = result; b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    vkCmdCopyImageToBuffer(cb, result, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stg, 1, &cp);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(c.q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(c.q));

    std::vector<uint16_t> out((size_t)W * H * 4);
    memcpy(out.data(), map, (size_t)bytes);
    cmaa2::readCounters(s, ctr);
    cmaa2::destroy(f, s);
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    vkDestroyBuffer(c.dev, stg, nullptr); vkFreeMemory(c.dev, stgMem, nullptr);
    vkDestroyImage(c.dev, src, nullptr); vkFreeMemory(c.dev, srcMem, nullptr);
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 5) { printf("usage: cmaa2_file <in.rgba16f> <width> <height> <out.rgba16f> [preset]\n"); return 2; }
    W = (uint32_t)atoi(argv[2]); H = (uint32_t)atoi(argv[3]);
    const int preset = argc > 5 ? atoi(argv[5]) : 2;
    std::vector<uint16_t> pix((size_t)W * H * 4);
    FILE *fi = fopen(argv[1], "rb");
    if (!fi || fread(pix.data(), 8, (size_t)W * H, fi) != (size_t)W * H) { printf("FAIL reading %s\n", argv[1]); return 1; }
    fclose(fi);
    Ctx c{};
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    CK(vkCreateInstance(&ici, nullptr, &c.inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(c.inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(c.inst, &n, pds.data());
    c.pd = pds[0];
    for (VkPhysicalDevice p : pds) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) c.pd = p;
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures feat{};
    feat.shaderStorageImageReadWithoutFormat = VK_TRUE; feat.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &feat;
    CK(vkCreateDevice(c.pd, &dci, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, 0, 0, &c.q);
    vkGetPhysicalDeviceMemoryProperties(c.pd, &c.mp);
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pc.queueFamilyIndex = 0;
    CK(vkCreateCommandPool(c.dev, &pc, nullptr, &c.pool));
    uint32_t ctr[16] = {};
    const std::vector<uint16_t> out = runOnce(c, pix, true, preset, ctr);
    FILE *fo = fopen(argv[4], "wb");
    if (!fo || fwrite(out.data(), 8, (size_t)W * H, fo) != (size_t)W * H) { printf("FAIL writing %s\n", argv[4]); return 1; }
    fclose(fo);
    printf("cmaa2_file: %ux%u preset %d, blend locations %u\n", W, H, preset, ctr[3]);
    vkDestroyCommandPool(c.dev, c.pool, nullptr);
    vkDestroyDevice(c.dev, nullptr); vkDestroyInstance(c.inst, nullptr);
    return 0;
}
