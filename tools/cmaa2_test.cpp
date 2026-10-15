// cmaa2_test.cpp - drive src/vklayer/cmaa2.h on the real GPU against a synthetic edge (FSCAA layer 1).
// 256x256 RGBA16F: 4.0 (HDR white) above the line y = 0.6x + 40, 0.05 below. Runs CMAA2 twice:
// COPY mode (our own colour image) and IN-PLACE mode (on a storage-capable "scene" image, the
// layer's perf path). PASS when, for copy mode:
//   (1) >= 100 pixels within 4 px of the line end up strictly between the two values,
//   (2) every pixel >= 4 px from the line is bit-identical to the source,
//   (3) Intel's control word 3 (blend locations of the last pass) is non-zero,
// and (4) the in-place result is bit-identical to the copy-mode result,
// and (5) zero Khronos validation errors (run with VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation).
// Usage: cmaa2_test [preset 0..3]
// SPDX-License-Identifier: GPL-3.0-or-later
#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "cmaa2.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d\n", #x, (int)r_); exit(1); } } while (0)
static const uint32_t W = 256, H = 256;

static uint16_t toHalf(float f)
{
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u, e8 = (x >> 23) & 0xffu, m = (x >> 13) & 0x3ffu;
    if (e8 < 113) return (uint16_t)s;
    return (uint16_t)(s | ((e8 - 112) << 10) | m);
}
static float fromHalf(uint16_t h)
{
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    uint32_t x = e ? (s | ((e + 112) << 23) | (m << 13)) : s;
    float v; memcpy(&v, &x, 4); return v;
}
static float lineDist(uint32_t x, uint32_t y) { return std::fabs(0.6f * (x + 0.5f) + 40.0f - (y + 0.5f)) / std::sqrt(1.36f); }

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
    printf("%s mode: module memory %.2f MB\n", inplace ? "in-place" : "copy", cmaa2::bytes(s) / 1048576.0);
    cmaa2::destroy(f, s);
    vkFreeCommandBuffers(c.dev, c.pool, 1, &cb);
    vkDestroyBuffer(c.dev, stg, nullptr); vkFreeMemory(c.dev, stgMem, nullptr);
    vkDestroyImage(c.dev, src, nullptr); vkFreeMemory(c.dev, srcMem, nullptr);
    return out;
}

int main(int argc, char **argv)
{
    const int preset = argc > 1 ? atoi(argv[1]) : 2;
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
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.pd, &props);
    printf("device: %s, preset %d\n", props.deviceName, preset);
    VkPhysicalDeviceFeatures have; vkGetPhysicalDeviceFeatures(c.pd, &have);
    if (!have.shaderStorageImageReadWithoutFormat || !have.shaderStorageImageWriteWithoutFormat) {
        printf("FAIL: device lacks storage-image read/write without format\n"); return 1;
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

    std::vector<uint16_t> pix((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const float v = (y + 0.5f) < 0.6f * (x + 0.5f) + 40.0f ? 4.0f : 0.05f;
            for (int k = 0; k < 3; ++k) pix[((size_t)y * W + x) * 4 + k] = toHalf(v);
            pix[((size_t)y * W + x) * 4 + 3] = toHalf(1.0f);
        }

    uint32_t ctrCopy[16] = {}, ctrIn[16] = {};
    const std::vector<uint16_t> outCopy = runOnce(c, pix, false, preset, ctrCopy);
    const std::vector<uint16_t> outIn = runOnce(c, pix, true, preset, ctrIn);

    uint32_t blended = 0, flatBad = 0, modeDiff = 0;
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const size_t i = ((size_t)y * W + x) * 4;
            for (int k = 0; k < 4; ++k) if (outCopy[i + k] != outIn[i + k]) { ++modeDiff; break; }
            if (lineDist(x, y) >= 4.0f) {
                for (int k = 0; k < 4; ++k) if (outCopy[i + k] != pix[i + k]) { ++flatBad; break; }
            } else {
                const float v = fromHalf(outCopy[i]);
                if (v > 0.06f && v < 3.99f) ++blended;
            }
        }
    printf("blended edge pixels %u (need >= 100), changed flat pixels %u (need 0)\n", blended, flatBad);
    printf("blend locations last pass: copy %u, in-place %u (need > 0)\n", ctrCopy[3], ctrIn[3]);
    printf("in-place vs copy: %u pixels differ (need 0)\n", modeDiff);
    vkDestroyCommandPool(c.dev, c.pool, nullptr);
    vkDestroyDevice(c.dev, nullptr); vkDestroyInstance(c.inst, nullptr);
    if (blended < 100 || flatBad != 0 || ctrCopy[3] == 0 || ctrIn[3] == 0 || modeDiff != 0) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
