// cmaa2.h - Intel CMAA2 (third_party/cmaa2, Apache-2.0) as a Vulkan compute pass. FSCAA layer 1.
// X-Plane's scene has no STORAGE usage, so CMAA2 runs in place on its own RGBA16F copy; the TAA
// resolve then samples that copy. Self-contained behind a function table so tools/cmaa2_test.cpp
// can drive it on the real GPU. Bindings follow build.ps1's DXC shifts: u0..u7 -> 0..7, t0 -> 10
// (reflection, 2026-10-15: storage images 0/1/5, storage buffers 2/3/4/6/7, sampled image 10).
// Sizes and dispatch order are Intel's (third_party/cmaa2/reference/vaCMAA2DX12.cpp).
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <map>
#include "cmaa2_spv.h"

namespace cmaa2 {

#define CMAA2_FNS(X) X(CreateImage) X(DestroyImage) X(CreateImageView) X(DestroyImageView) \
    X(CreateBuffer) X(DestroyBuffer) X(GetImageMemoryRequirements) X(GetBufferMemoryRequirements) \
    X(AllocateMemory) X(FreeMemory) X(BindImageMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
    X(CreateShaderModule) X(DestroyShaderModule) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
    X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
    X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdDispatch) X(CmdDispatchIndirect) \
    X(CmdPipelineBarrier) X(CmdFillBuffer) X(CmdClearColorImage) X(CmdCopyImage) X(CmdCopyBuffer)

struct Fns {
#define CMAA2_DECL(n) PFN_vk##n n = nullptr;
    CMAA2_FNS(CMAA2_DECL)
#undef CMAA2_DECL
    VkPhysicalDeviceMemoryProperties memProps{};
};

// Inside the layer, gdpa must be the NEXT layer's vkGetDeviceProcAddr (DeviceData::gdpa), so these
// calls never re-enter our own hooks.
inline bool loadFns(Fns &f, PFN_vkGetDeviceProcAddr gdpa, VkDevice dev, const VkPhysicalDeviceMemoryProperties &mp)
{
    if (!gdpa || !dev) return false;
#define CMAA2_LOAD(n) f.n = (PFN_vk##n)gdpa(dev, "vk" #n); if (!f.n) return false;
    CMAA2_FNS(CMAA2_LOAD)
#undef CMAA2_LOAD
    f.memProps = mp;
    return true;
}

enum { kImgColor, kImgEdges, kImgHeads, kBufCand, kBufLoc, kBufItems, kBufControl, kBufIndirect, kBufReadback, kRes };

struct State {
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    VkImage color = VK_NULL_HANDLE, edges = VK_NULL_HANDLE, heads = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE, colorArrayView = VK_NULL_HANDLE, edgesView = VK_NULL_HANDLE, headsView = VK_NULL_HANDLE;
    VkBuffer cand = VK_NULL_HANDLE, loc = VK_NULL_HANDLE, items = VK_NULL_HANDLE, control = VK_NULL_HANDLE,
             indirect = VK_NULL_HANDLE, readback = VK_NULL_HANDLE;
    void *readbackMap = nullptr;     // 64 bytes: the control buffer as of the last recordCounterCopy
    VkDeviceMemory mem[kRes] = {};
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipeline edgesPipe[4] = {};
    VkPipeline argsPipe = VK_NULL_HANDLE, candPipe = VK_NULL_HANDLE, applyPipe = VK_NULL_HANDLE, debugPipe = VK_NULL_HANDLE;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // In-place mode (perf): CMAA2 runs directly on X-Plane's scene image (the layer adds STORAGE
    // usage at creation) - no copy, no colour image of our own. One descriptor set + 2-D view per
    // scene image, because X-Plane may alternate scene targets and a set in flight must not change.
    bool ownColor = true;
    struct Ext { VkImageView view = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE; };
    std::map<VkImage, Ext> ext;
    bool initialised = false;   // working images moved to GENERAL, control/heads cleared
    uint64_t frames = 0;        // record() calls since create()
};

inline uint32_t memType(const Fns &f, uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < f.memProps.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (f.memProps.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

inline bool makeImage(const Fns &f, State &s, int slot, VkFormat fmt, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImage &img)
{
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D; ic.format = fmt; ic.extent = {w, h, 1}; ic.mipLevels = 1; ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT; ic.tiling = VK_IMAGE_TILING_OPTIMAL; ic.usage = usage;
    if (f.CreateImage(s.dev, &ic, nullptr, &img) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; f.GetImageMemoryRequirements(s.dev, img, &mr);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.allocationSize = mr.size;
    ma.memoryTypeIndex = memType(f, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ma.memoryTypeIndex == UINT32_MAX || f.AllocateMemory(s.dev, &ma, nullptr, &s.mem[slot]) != VK_SUCCESS) return false;
    return f.BindImageMemory(s.dev, img, s.mem[slot], 0) == VK_SUCCESS;
}

inline bool makeBuffer(const Fns &f, State &s, int slot, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer &buf,
                       VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.size = size; bc.usage = usage;
    if (f.CreateBuffer(s.dev, &bc, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; f.GetBufferMemoryRequirements(s.dev, buf, &mr);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.allocationSize = mr.size;
    ma.memoryTypeIndex = memType(f, mr.memoryTypeBits, props);
    if (ma.memoryTypeIndex == UINT32_MAX || f.AllocateMemory(s.dev, &ma, nullptr, &s.mem[slot]) != VK_SUCCESS) return false;
    return f.BindBufferMemory(s.dev, buf, s.mem[slot], 0) == VK_SUCCESS;
}

inline VkImageView makeView(const Fns &f, State &s, VkImage img, VkFormat fmt, VkImageViewType type)
{
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = img; vc.viewType = type; vc.format = fmt; vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v = VK_NULL_HANDLE;
    return f.CreateImageView(s.dev, &vc, nullptr, &v) == VK_SUCCESS ? v : VK_NULL_HANDLE;
}

inline VkPipeline makePipe(const Fns &f, State &s, const uint32_t *code, size_t words, const char *entry)
{
    VkShaderModuleCreateInfo mc{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; mc.codeSize = words * 4; mc.pCode = code;
    VkShaderModule m; if (f.CreateShaderModule(s.dev, &mc, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo pc{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pc.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; pc.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.stage.module = m; pc.stage.pName = entry; pc.layout = s.pl;
    VkPipeline p = VK_NULL_HANDLE;
    if (f.CreateComputePipelines(s.dev, VK_NULL_HANDLE, 1, &pc, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
    f.DestroyShaderModule(s.dev, m, nullptr);
    return p;
}

inline void destroy(const Fns &f, State &s)
{
    if (!s.dev) return;
    for (int i = 0; i < 4; ++i) if (s.edgesPipe[i]) f.DestroyPipeline(s.dev, s.edgesPipe[i], nullptr);
    VkPipeline pipes[4] = {s.argsPipe, s.candPipe, s.applyPipe, s.debugPipe};
    for (int i = 0; i < 4; ++i) if (pipes[i]) f.DestroyPipeline(s.dev, pipes[i], nullptr);
    for (std::map<VkImage, State::Ext>::iterator it = s.ext.begin(); it != s.ext.end(); ++it)
        if (it->second.view) f.DestroyImageView(s.dev, it->second.view, nullptr);
    if (s.pool) f.DestroyDescriptorPool(s.dev, s.pool, nullptr);
    if (s.pl) f.DestroyPipelineLayout(s.dev, s.pl, nullptr);
    if (s.dsl) f.DestroyDescriptorSetLayout(s.dev, s.dsl, nullptr);
    VkImageView views[4] = {s.colorView, s.colorArrayView, s.edgesView, s.headsView};
    for (int i = 0; i < 4; ++i) if (views[i]) f.DestroyImageView(s.dev, views[i], nullptr);
    VkImage imgs[3] = {s.color, s.edges, s.heads};
    for (int i = 0; i < 3; ++i) if (imgs[i]) f.DestroyImage(s.dev, imgs[i], nullptr);
    if (s.readbackMap && s.mem[kBufReadback]) f.UnmapMemory(s.dev, s.mem[kBufReadback]);
    VkBuffer bufs[6] = {s.cand, s.loc, s.items, s.control, s.indirect, s.readback};
    for (int i = 0; i < 6; ++i) if (bufs[i]) f.DestroyBuffer(s.dev, bufs[i], nullptr);
    for (int i = 0; i < kRes; ++i) if (s.mem[i]) f.FreeMemory(s.dev, s.mem[i], nullptr);
    s = State();
}

static const uint32_t kMaxExt = 4;   // distinct X-Plane scene images served in place

// Bind every working resource plus `colour` (a 2-D view, GENERAL) into `set`.
inline void writeSet(const Fns &f, State &s, VkDescriptorSet set, VkImageView colour)
{
    const VkDescriptorType types[9] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
    const uint32_t bindings[9] = {0, 1, 2, 3, 4, 5, 6, 7, 10};
    VkDescriptorImageInfo ii[4] = {{VK_NULL_HANDLE, colour, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, s.edgesView, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, s.headsView, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, colour, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi[5] = {{s.cand, 0, VK_WHOLE_SIZE}, {s.loc, 0, VK_WHOLE_SIZE}, {s.items, 0, VK_WHOLE_SIZE},
                                    {s.control, 0, VK_WHOLE_SIZE}, {s.indirect, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[9];
    for (int i = 0; i < 9; ++i) { wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet = set;
                                  wr[i].dstBinding = bindings[i]; wr[i].descriptorCount = 1; wr[i].descriptorType = types[i]; }
    wr[0].pImageInfo = &ii[0]; wr[1].pImageInfo = &ii[1]; wr[5].pImageInfo = &ii[2]; wr[8].pImageInfo = &ii[3];
    wr[2].pBufferInfo = &bi[0]; wr[3].pBufferInfo = &bi[1]; wr[4].pBufferInfo = &bi[2]; wr[6].pBufferInfo = &bi[3]; wr[7].pBufferInfo = &bi[4];
    f.UpdateDescriptorSets(s.dev, 9, wr, 0, nullptr);
}

// Total device memory this module holds, for the FSCAA STATS line.
inline uint64_t bytes(const State &s)
{
    const uint64_t px = (uint64_t)s.w * s.h;
    return (s.ownColor ? px * 8 : 0) + ((uint64_t)(s.w + 1) / 2) * s.h + ((uint64_t)(s.w + 1) / 2) * ((s.h + 1) / 2) * 4 +
           (px / 4) * 4 + ((px + 3) / 6) * 4 + (px / 2) * 8 + 64 * 3;
}

inline bool create(const Fns &f, State &s, VkDevice dev, uint32_t w, uint32_t h, bool ownColor = true)
{
    destroy(f, s);
    if (!w || !h) return false;
    s.dev = dev; s.w = w; s.h = h; s.ownColor = ownColor;
    const VkDeviceSize px = (VkDeviceSize)w * h;
    const VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bool ok = (!ownColor || makeImage(f, s, kImgColor, VK_FORMAT_R16G16B16A16_SFLOAT, w, h,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT, s.color)) &&
              makeImage(f, s, kImgEdges, VK_FORMAT_R8_UINT, (w + 1) / 2, h, VK_IMAGE_USAGE_STORAGE_BIT, s.edges) &&
              makeImage(f, s, kImgHeads, VK_FORMAT_R32_UINT, (w + 1) / 2, (h + 1) / 2,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, s.heads) &&
              makeBuffer(f, s, kBufCand, (px / 4) * 4, SB, s.cand) &&
              makeBuffer(f, s, kBufLoc, ((px + 3) / 6) * 4, SB, s.loc) &&
              makeBuffer(f, s, kBufItems, (px / 2) * 8, SB, s.items) &&
              makeBuffer(f, s, kBufControl, 64, SB | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, s.control) &&
              makeBuffer(f, s, kBufIndirect, 64, SB | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, s.indirect) &&
              makeBuffer(f, s, kBufReadback, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, s.readback,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!ok) { destroy(f, s); return false; }
    if (f.MapMemory(dev, s.mem[kBufReadback], 0, 64, 0, &s.readbackMap) != VK_SUCCESS) s.readbackMap = nullptr;
    else memset(s.readbackMap, 0, 64);
    if (ownColor) {
        s.colorView = makeView(f, s, s.color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
        s.colorArrayView = makeView(f, s, s.color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
        if (!s.colorView || !s.colorArrayView) { destroy(f, s); return false; }
    }
    s.edgesView = makeView(f, s, s.edges, VK_FORMAT_R8_UINT, VK_IMAGE_VIEW_TYPE_2D);
    s.headsView = makeView(f, s, s.heads, VK_FORMAT_R32_UINT, VK_IMAGE_VIEW_TYPE_2D);
    if (!s.edgesView || !s.headsView) { destroy(f, s); return false; }

    VkDescriptorSetLayoutBinding b[9] = {};
    const VkDescriptorType types[9] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
    const uint32_t bindings[9] = {0, 1, 2, 3, 4, 5, 6, 7, 10};
    for (int i = 0; i < 9; ++i) { b[i].binding = bindings[i]; b[i].descriptorType = types[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; lc.bindingCount = 9; lc.pBindings = b;
    if (f.CreateDescriptorSetLayout(dev, &lc, nullptr, &s.dsl) != VK_SUCCESS) { destroy(f, s); return false; }
    VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plc.setLayoutCount = 1; plc.pSetLayouts = &s.dsl;
    if (f.CreatePipelineLayout(dev, &plc, nullptr, &s.pl) != VK_SUCCESS) { destroy(f, s); return false; }
    const uint32_t sets = 1 + kMaxExt;
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * sets}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * sets},
                                  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sets}};
    VkDescriptorPoolCreateInfo pc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pc.maxSets = sets; pc.poolSizeCount = 3; pc.pPoolSizes = ps;
    if (f.CreateDescriptorPool(dev, &pc, nullptr, &s.pool) != VK_SUCCESS) { destroy(f, s); return false; }
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; da.descriptorPool = s.pool; da.descriptorSetCount = 1; da.pSetLayouts = &s.dsl;
    if (f.AllocateDescriptorSets(dev, &da, &s.set) != VK_SUCCESS) { destroy(f, s); return false; }

    if (ownColor) writeSet(f, s, s.set, s.colorView);

    const uint32_t *edgeCode[4] = {kCmaa2Edges0, kCmaa2Edges1, kCmaa2Edges2, kCmaa2Edges3};
    const size_t edgeWords[4] = {kCmaa2Edges0Words, kCmaa2Edges1Words, kCmaa2Edges2Words, kCmaa2Edges3Words};
    for (int i = 0; i < 4; ++i) s.edgesPipe[i] = makePipe(f, s, edgeCode[i], edgeWords[i], "EdgesColor2x2CS");
    s.argsPipe  = makePipe(f, s, kCmaa2Args,  kCmaa2ArgsWords,  "ComputeDispatchArgsCS");
    s.candPipe  = makePipe(f, s, kCmaa2Cand,  kCmaa2CandWords,  "ProcessCandidatesCS");
    s.applyPipe = makePipe(f, s, kCmaa2Apply, kCmaa2ApplyWords, "DeferredColorApply2x2CS");
    s.debugPipe = makePipe(f, s, kCmaa2DebugEdges, kCmaa2DebugEdgesWords, "DebugDrawEdgesCS");
    for (int i = 0; i < 4; ++i) if (!s.edgesPipe[i]) { destroy(f, s); return false; }
    if (!s.argsPipe || !s.candPipe || !s.applyPipe) { destroy(f, s); return false; }
    return true;   // debugPipe is optional: recordDebugEdges() skips without it
}

inline void computeBarrier(const Fns &f, VkCommandBuffer cb, bool toIndirect)
{
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | (toIndirect ? VK_ACCESS_INDIRECT_COMMAND_READ_BIT : 0);
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | (toIndirect ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT : 0),
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

inline void imageBarrier(const Fns &f, VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                         VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.oldLayout = from; b.newLayout = to; b.srcAccessMask = sa; b.dstAccessMask = da;
    f.CmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// First use after create(): working images to GENERAL, list heads to "empty", counters to zero.
inline void initialise(const Fns &f, State &s, VkCommandBuffer cb)
{
    if (s.initialised) return;
    const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, TR = VK_PIPELINE_STAGE_TRANSFER_BIT,
                               CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    imageBarrier(f, cb, s.edges, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT, ALL, CS);
    imageBarrier(f, cb, s.heads, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, ALL, TR);
    VkClearColorValue ones; ones.uint32[0] = ones.uint32[1] = ones.uint32[2] = ones.uint32[3] = 0xFFFFFFFFu;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    f.CmdClearColorImage(cb, s.heads, VK_IMAGE_LAYOUT_GENERAL, &ones, 1, &rng);
    f.CmdFillBuffer(cb, s.control, 0, VK_WHOLE_SIZE, 0);
    f.CmdFillBuffer(cb, s.indirect, 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    f.CmdPipelineBarrier(cb, TR, CS | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    s.initialised = true;
}

// The four-kernel chain on whatever colour `set` binds (colour already in GENERAL).
inline void dispatchKernels(const Fns &f, State &s, VkCommandBuffer cb, VkDescriptorSet set, int preset)
{
    f.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &set, 0, nullptr);
    // Edges: Intel dispatches (res + 2*(16-2) - 1) / (2*(16-2)) groups per axis.
    const uint32_t out = 2 * (16 - 2);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.edgesPipe[preset]);
    f.CmdDispatch(cb, (s.w + out - 1) / out, (s.h + out - 1) / out, 1);
    computeBarrier(f, cb, false);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.argsPipe);
    f.CmdDispatch(cb, 2, 1, 1);
    computeBarrier(f, cb, true);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.candPipe);
    f.CmdDispatchIndirect(cb, s.indirect, 0);
    computeBarrier(f, cb, false);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.argsPipe);
    f.CmdDispatch(cb, 1, 2, 1);
    computeBarrier(f, cb, true);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.applyPipe);
    f.CmdDispatchIndirect(cb, s.indirect, 0);
}

// Debug: Intel's DebugDrawEdgesCS overlays the detected edges on the colour `set` binds (GENERAL).
inline void debugEdgesOn(const Fns &f, State &s, VkCommandBuffer cb, VkDescriptorSet set)
{
    computeBarrier(f, cb, false);
    f.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &set, 0, nullptr);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.debugPipe);
    f.CmdDispatch(cb, (s.w + 15) / 16, (s.h + 15) / 16, 1);
}

// COPY MODE. Copy src (in srcLayout, left as it was) into our own colour, run CMAA2 on it, leave
// the result in SHADER_READ_ONLY_OPTIMAL (sample s.colorArrayView). preset 0..3 = LOW..ULTRA.
inline bool record(const Fns &f, State &s, VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, int preset,
                   bool debugEdges = false)
{
    if (!s.ownColor || !s.set || !src || preset < 0 || preset > 3) return false;
    const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, TR = VK_PIPELINE_STAGE_TRANSFER_BIT,
                               CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    initialise(f, s, cb);
    imageBarrier(f, cb, src, srcLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, ALL, TR);
    imageBarrier(f, cb, s.color, s.colorLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, ALL, TR);
    VkImageCopy cp{}; cp.srcSubresource = cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.extent = {s.w, s.h, 1};
    f.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    imageBarrier(f, cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, srcLayout, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT, TR, ALL);
    imageBarrier(f, cb, s.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, TR, CS);
    dispatchKernels(f, s, cb, s.set, preset);
    if (debugEdges && s.debugPipe) debugEdgesOn(f, s, cb, s.set);
    imageBarrier(f, cb, s.color, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, CS, CS | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    s.colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ++s.frames;
    return true;
}

// IN-PLACE MODE (perf: no copy, no extra image). img must have STORAGE usage, RGBA16F, s.w x s.h,
// and be in `layout`; it is left in `layout`. One descriptor set + 2-D view per image, cached.
inline bool recordInPlace(const Fns &f, State &s, VkCommandBuffer cb, VkImage img, VkImageLayout layout, int preset,
                          bool debugEdges = false)
{
    if (s.ownColor || !s.pl || !img || preset < 0 || preset > 3) return false;
    std::map<VkImage, State::Ext>::iterator it = s.ext.find(img);
    if (it == s.ext.end()) {
        if (s.ext.size() >= kMaxExt) return false;       // more scene targets than expected: caller falls back
        State::Ext e;
        e.view = makeView(f, s, img, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
        if (!e.view) return false;
        VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        da.descriptorPool = s.pool; da.descriptorSetCount = 1; da.pSetLayouts = &s.dsl;
        if (f.AllocateDescriptorSets(s.dev, &da, &e.set) != VK_SUCCESS) { f.DestroyImageView(s.dev, e.view, nullptr); return false; }
        writeSet(f, s, e.set, e.view);
        it = s.ext.insert(std::make_pair(img, e)).first;
    }
    const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    initialise(f, s, cb);
    imageBarrier(f, cb, img, layout, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_MEMORY_WRITE_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, ALL, CS);
    dispatchKernels(f, s, cb, it->second.set, preset);
    if (debugEdges && s.debugPipe) debugEdgesOn(f, s, cb, it->second.set);
    imageBarrier(f, cb, img, VK_IMAGE_LAYOUT_GENERAL, layout, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT, CS, ALL);
    ++s.frames;
    return true;
}

// X-Plane destroyed `img`: drop our view of it (its descriptor set stays allocated, unused).
inline void forget(const Fns &f, State &s, VkImage img)
{
    std::map<VkImage, State::Ext>::iterator it = s.ext.find(img);
    if (it == s.ext.end()) return;
    if (it->second.view) f.DestroyImageView(s.dev, it->second.view, nullptr);
    s.ext.erase(it);
}

// Debug: copy Intel's control buffer (its counters) to the mapped readback buffer. Read it a frame
// later with readCounters(); the copy lands when this command buffer completes.
inline void recordCounterCopy(const Fns &f, State &s, VkCommandBuffer cb)
{
    if (!s.readback || !s.control) return;
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy bc{0, 0, 64};
    f.CmdCopyBuffer(cb, s.control, s.readback, 1, &bc);
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    f.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
}

inline bool readCounters(const State &s, uint32_t out[16])
{
    if (!s.readbackMap) return false;
    memcpy(out, s.readbackMap, 64);
    return true;
}

} // namespace cmaa2
