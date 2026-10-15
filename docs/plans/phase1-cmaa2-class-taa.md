# FSCAA Phase 1 (CMAA2 + class-weighted TAA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add `taa.backend=fscaa`. Intel's CMAA2 cleans each frame's edges before the TAA resolve,
and the resolve keeps history only where it helps: full on scenery, capped on the own aircraft and
cockpit, none on cockpit screens.

**Architecture:**
- Intel's unmodified `CMAA2.hlsl` is vendored and compiled by DXC into 7 SPIR-V blobs
  (`cmaa2_spv.h`).
- A self-contained host module, `cmaa2.h`, owns its images, buffers and pipelines behind a
  function table, so an offline harness can drive it on the real GPU.
- `taa.h` copies the scene into CMAA2's working image, runs the four kernels, and points the
  resolve's `uScene` at the result.
- `taa.comp` applies per-class history caps from two new UBO vec4s.

**Tech Stack:**
- C++17 (MinGW g++)
- Vulkan 1.2
- GLSL (glslangValidator) and HLSL (DXC 1.9 in `C:\VulkanSDK\1.4.357.0\Bin`)
- PowerShell `build.ps1`

**Spec:** `docs/superpowers/specs/2026-10-15-fscaa-design.md`

## Global Constraints

- Repo is `D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors`, branch `gbuffer-harvest`.
  Build with `powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1`, run from Git Bash so
  native stderr warnings don't abort the script.
- If X-Plane is running, build with `MV_LAYER_OUT=<staging dll>`. The loaded layer DLL can't be
  overwritten.
- `third_party/cmaa2/CMAA2.hlsl` is Intel's file, unmodified. Licence is Apache-2.0: keep
  `third_party/cmaa2/license.txt`. Any local change is recorded in `third_party/cmaa2/LOCAL_PATCHES.md`.
- FSCAA is only active when `taa.backend=fscaa` AND one view layer (`g_taa.layers == 1`); VR is
  untouched.
- Live keys and their defaults:
  - `fscaa.cmaa=1`
  - `fscaa.cmaa_preset=2`
  - `fscaa.near_m=80`
  - `fscaa.keep_near=0.5`
  - `fscaa.keep_screen=0.0`
  - `fscaa.keep_light=1.0`
- Trace lines start `FSCAA:`.
- Never touch X-Plane's own images beyond reading the scene. Its scene has no STORAGE usage, so
  CMAA2 runs on our copy.
- Before any test launch:
  - set `taa.enable=1` and `taa.backend=fscaa` in `%TEMP%\taa_live.ini`. This is the user-approved
    pause of the VRAM-only rule.
  - After testing, restore `taa.enable=0` and `taa.backend=taa` unless the user says otherwise.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `third_party/cmaa2/CMAA2.hlsl`, `license.txt`, `LOCAL_PATCHES.md` (create) | vendored Intel source |
| `build.ps1` (modify) | DXC: CMAA2 → 7 blobs → `src/vklayer/cmaa2_spv.h`, spirv-val each |
| `src/vklayer/cmaa2_spv.h` (generated) | `kCmaa2Edges0..3`, `kCmaa2Args`, `kCmaa2Cand`, `kCmaa2Apply` + `...Words` |
| `src/vklayer/cmaa2.h` (create) | function table, resources, pipelines, `record()` |
| `tools/cmaa2_test.cpp` (create) | offline GPU test of `cmaa2.h` on a synthetic edge |
| `src/vklayer/taa.h` (modify) | backend switch, run CMAA2, bind its output as `uScene`, UBO slots 32–39 |
| `src/shaders/taa.comp` (modify) | `uFscaa`/`uFscaa2`, class caps, `taa.viz=13` |

---

### Task 1: Vendor CMAA2 and compile it to SPIR-V in the build

**Files:**
- Create: `third_party/cmaa2/CMAA2.hlsl`, `third_party/cmaa2/license.txt`, `third_party/cmaa2/LOCAL_PATCHES.md`
- Modify: `build.ps1` (insert after the `taau_spv.h` block, near line 290)
- Generated: `src/vklayer/cmaa2_spv.h`

**Interfaces:**
- Produces: in `cmaa2_spv.h`, `static const uint32_t kCmaa2Edges0[]..kCmaa2Edges3[]`, `kCmaa2Args[]`,
  `kCmaa2Cand[]`, `kCmaa2Apply[]`, each with `static const size_t <name>Words`.

- [ ] **Step 1: Vendor the source**

```bash
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"
mkdir -p third_party/cmaa2
curl -fsSL https://raw.githubusercontent.com/GameTechDev/CMAA2/master/Projects/CMAA2/CMAA2/CMAA2.hlsl -o third_party/cmaa2/CMAA2.hlsl
curl -fsSL https://raw.githubusercontent.com/GameTechDev/CMAA2/master/license.txt -o third_party/cmaa2/license.txt
printf '# Local patches to CMAA2.hlsl\n\nNone. The file is byte-identical to GameTechDev/CMAA2 master as of 2026-10-15.\n' > third_party/cmaa2/LOCAL_PATCHES.md
grep -c "EdgesColor2x2CS\|ProcessCandidatesCS\|DeferredColorApply2x2CS\|ComputeDispatchArgsCS" third_party/cmaa2/CMAA2.hlsl
```

Expected: a count of 4 or more, and `license.txt` mentions Apache. If the curl path 404s, find
`CMAA2.hlsl` in the repo tree at https://github.com/GameTechDev/CMAA2 and use that raw URL.

- [ ] **Step 2: Check DXC supports unknown image formats**

```bash
/c/VulkanSDK/1.4.357.0/Bin/dxc.exe --help | grep -i "unknown-image-format"
```

Expected: `-fspv-use-unknown-image-format` is listed. If it isn't, add
`[[vk::image_format("rgba16f")]]` on `g_inoutColorWriteonly`, `[[vk::image_format("r8ui")]]` on
`g_workingEdges` and `[[vk::image_format("r32ui")]]` on `g_workingDeferredBlendItemListHeads` in the
vendored file. Record that in `LOCAL_PATCHES.md`, and drop the flag from Step 3.

- [ ] **Step 3: Add the DXC step to `build.ps1`**

Insert after the block that writes `$src/vklayer/taau_spv.h`:

```powershell
# ---- FSCAA layer 1: Intel CMAA2 (third_party/cmaa2, Apache-2.0), HLSL -> SPIR-V with DXC.
# u0..u7 -> bindings 0..7, t0 -> binding 10. One edges kernel per quality preset (0..3).
$dxc = Join-Path $vksdk "Bin\dxc.exe"
if (-not (Test-Path $dxc)) { throw "dxc not found at $dxc" }
$cmaaSrc = "$root\third_party\cmaa2\CMAA2.hlsl"
$cmaaDefs = @('-D','CMAA2_SUPPORT_HDR_COLOR_RANGE=1','-D','CMAA2_EDGE_DETECTION_LUMA_PATH=1',
              '-D','CMAA2_UAV_STORE_TYPED=1','-D','CMAA2_UAV_STORE_TYPED_UNORM_FLOAT=0',
              '-D','CMAA2_UAV_STORE_CONVERT_TO_SRGB=0')
$cmaaFlags = @('-spirv','-T','cs_6_0','-fspv-target-env=vulkan1.2','-fvk-u-shift','0','0',
               '-fvk-t-shift','10','0','-fvk-s-shift','20','0','-fspv-use-unknown-image-format','-O3')
$cmaaJobs = @(
    @{ Name='kCmaa2Edges0'; Entry='EdgesColor2x2CS';         Preset=0 },
    @{ Name='kCmaa2Edges1'; Entry='EdgesColor2x2CS';         Preset=1 },
    @{ Name='kCmaa2Edges2'; Entry='EdgesColor2x2CS';         Preset=2 },
    @{ Name='kCmaa2Edges3'; Entry='EdgesColor2x2CS';         Preset=3 },
    @{ Name='kCmaa2Args';   Entry='ComputeDispatchArgsCS';   Preset=2 },
    @{ Name='kCmaa2Cand';   Entry='ProcessCandidatesCS';     Preset=2 },
    @{ Name='kCmaa2Apply';  Entry='DeferredColorApply2x2CS'; Preset=2 })
$csb = New-Object System.Text.StringBuilder
[void]$csb.AppendLine("// Generated from third_party/cmaa2/CMAA2.hlsl by build.ps1 (DXC) - do not edit.")
[void]$csb.AppendLine("#pragma once")
[void]$csb.AppendLine("#include <cstdint>")
[void]$csb.AppendLine("#include <cstddef>")
foreach ($j in $cmaaJobs) {
    $tmp = Join-Path $env:TEMP "$($j.Name).spv"
    & $dxc @cmaaFlags @cmaaDefs '-D' "CMAA2_STATIC_QUALITY_PRESET=$($j.Preset)" '-E' $j.Entry $cmaaSrc '-Fo' $tmp | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "dxc failed on $($j.Entry) preset $($j.Preset)" }
    & (Join-Path $vksdk "Bin\spirv-val.exe") --target-env vulkan1.2 $tmp
    if ($LASTEXITCODE -ne 0) { throw "spirv-val failed on $($j.Name)" }
    $bytes = [IO.File]::ReadAllBytes($tmp)
    $words = for ($i = 0; $i -lt $bytes.Length; $i += 4) { '0x{0:x8}u' -f [BitConverter]::ToUInt32($bytes, $i) }
    [void]$csb.AppendLine("static const uint32_t $($j.Name)[] = {")
    for ($i = 0; $i -lt $words.Count; $i += 8) {
        [void]$csb.AppendLine("    " + (($words[$i..([Math]::Min($i + 7, $words.Count - 1))]) -join ', ') + ",")
    }
    [void]$csb.AppendLine("};")
    [void]$csb.AppendLine("static const size_t $($j.Name)Words = sizeof($($j.Name)) / sizeof(uint32_t);")
}
[IO.File]::WriteAllText("$src/vklayer/cmaa2_spv.h", $csb.ToString())
Write-Host "  CMAA2: $($cmaaJobs.Count) kernels -> cmaa2_spv.h"
```

- [ ] **Step 4: Build and check the reflection**

```bash
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"
MV_LAYER_OUT='D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors\build\vklayer\VkLayer_mv.staged.dll' \
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 > "$TEMP/fscaa_t1.log" 2>&1; echo "exit $?"
grep -E "CMAA2:|dxc failed|spirv-val failed" "$TEMP/fscaa_t1.log"
/c/VulkanSDK/1.4.357.0/Bin/spirv-reflect.exe "$TEMP/kCmaa2Cand.spv" | grep -E "binding|descriptor_type" | head -30
```

Expected:
- The build log shows `CMAA2: 7 kernels -> cmaa2_spv.h`.
- The reflection of the candidates kernel lists bindings among 0–7 and 10.
- The types are STORAGE_IMAGE at 0, 1, 5; STORAGE_BUFFER at 2, 3, 4, 6, 7; SAMPLED_IMAGE at 10.
- Write down any binding the reflection shows that this list doesn't. Task 2's layout must match
  the reflection exactly.

- [ ] **Step 5: Commit**

```bash
git add third_party/cmaa2 build.ps1 src/vklayer/cmaa2_spv.h
git commit -m "FSCAA: vendor Intel CMAA2 (Apache-2.0) and compile it to SPIR-V with DXC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `cmaa2.h` host module, tested offline on the GPU

**Files:**
- Create: `src/vklayer/cmaa2.h`
- Create: `tools/cmaa2_test.cpp`

**Interfaces:**
- Consumes: the `cmaa2_spv.h` names from Task 1.
- Produces:
  - `namespace cmaa2 { struct Fns; struct State; }`
  - `bool cmaa2::loadFns(Fns&, PFN_vkGetDeviceProcAddr, VkDevice, const VkPhysicalDeviceMemoryProperties&)`
  - `bool cmaa2::create(const Fns&, State&, VkDevice, uint32_t w, uint32_t h)`
  - `void cmaa2::destroy(const Fns&, State&)`
  - `bool cmaa2::record(const Fns&, State&, VkCommandBuffer, VkImage src, VkImageLayout srcLayout, int preset)`:
    on return, `State::color` holds the antialiased frame in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.
  - `State::colorArrayView` is a 1-layer `VK_IMAGE_VIEW_TYPE_2D_ARRAY` view of it, which is what
    `taa.comp`'s `sampler2DArray uScene` needs.

- [ ] **Step 1: Write the failing test, `tools/cmaa2_test.cpp`**

```cpp
// cmaa2_test.cpp - drive src/vklayer/cmaa2.h on the real GPU against a synthetic edge.
// 256x256 RGBA16F: 4.0 (HDR white) above the diagonal y = 0.6x + 40, 0.05 below.
// PASS: (1) >= 100 pixels near the diagonal end up strictly between the two values,
// (2) every pixel >= 4 px from the diagonal is bit-identical to the source,
// (3) run under VK_LAYER_KHRONOS_validation with zero errors (checked by the caller).
#include <vulkan/vulkan.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../src/vklayer/cmaa2.h"

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d\n", #x, (int)r_); return 1; } } while (0)
static const uint32_t W = 256, H = 256;

static uint16_t toHalf(float f) { uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000, e = ((x >> 23) & 0xff) - 112, m = (x >> 13) & 0x3ff;
    if (((x >> 23) & 0xff) < 113) return (uint16_t)s; return (uint16_t)(s | (e << 10) | m); }
static float fromHalf(uint16_t h) { uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (!e) { float v = 0.0f; uint32_t x = s; memcpy(&v, &x, 4); return v; }
    uint32_t x = s | ((e + 112) << 23) | (m << 13); float v; memcpy(&v, &x, 4); return v; }
static float lineDist(uint32_t x, uint32_t y) { return std::fabs(0.6f * (x + 0.5f) + 40.0f - (y + 0.5f)) / std::sqrt(1.36f); }
static uint32_t memType(const VkPhysicalDeviceMemoryProperties &mp, uint32_t bits, VkMemoryPropertyFlags f) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & f) == f) return i;
    return 0; }

int main()
{
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = pds[0];
    for (VkPhysicalDevice p : pds) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) pd = p; }
    VkPhysicalDeviceFeatures have; vkGetPhysicalDeviceFeatures(pd, &have);
    if (!have.shaderStorageImageReadWithoutFormat || !have.shaderStorageImageWriteWithoutFormat) {
        printf("FAIL: device lacks storage-image read/write without format\n"); return 1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures feat{}; feat.shaderStorageImageReadWithoutFormat = VK_TRUE; feat.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &feat;
    VkDevice dev; CK(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    cmaa2::Fns f;
    if (!cmaa2::loadFns(f, vkGetDeviceProcAddr, dev, mp)) { printf("FAIL loadFns\n"); return 1; }
    cmaa2::State s;
    if (!cmaa2::create(f, s, dev, W, H)) { printf("FAIL create\n"); return 1; }

    // Source image + staging buffer (upload, then readback of the result).
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ic.imageType = VK_IMAGE_TYPE_2D; ic.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ic.extent = {W, H, 1}; ic.mipLevels = 1; ic.arrayLayers = 1; ic.samples = VK_SAMPLE_COUNT_1_BIT; ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage src; CK(vkCreateImage(dev, &ic, nullptr, &src));
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, src, &mr);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.allocationSize = mr.size; ma.memoryTypeIndex = memType(mp, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory srcMem; CK(vkAllocateMemory(dev, &ma, nullptr, &srcMem)); CK(vkBindImageMemory(dev, src, srcMem, 0));
    const VkDeviceSize bytes = (VkDeviceSize)W * H * 8;
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.size = bytes; bc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer stg; CK(vkCreateBuffer(dev, &bc, nullptr, &stg));
    VkMemoryRequirements br; vkGetBufferMemoryRequirements(dev, stg, &br);
    VkMemoryAllocateInfo bm{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; bm.allocationSize = br.size;
    bm.memoryTypeIndex = memType(mp, br.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory stgMem; CK(vkAllocateMemory(dev, &bm, nullptr, &stgMem)); CK(vkBindBufferMemory(dev, stg, stgMem, 0));
    std::vector<uint16_t> pix((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; ++y) for (uint32_t x = 0; x < W; ++x) {
        const float v = (y + 0.5f) < 0.6f * (x + 0.5f) + 40.0f ? 4.0f : 0.05f;
        for (int c = 0; c < 3; ++c) pix[((size_t)y * W + x) * 4 + c] = toHalf(v);
        pix[((size_t)y * W + x) * 4 + 3] = toHalf(1.0f); }
    void *map; CK(vkMapMemory(dev, stgMem, 0, bytes, 0, &map)); memcpy(map, pix.data(), (size_t)bytes);

    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pc.queueFamilyIndex = 0;
    VkCommandPool pool; CK(vkCreateCommandPool(dev, &pc, nullptr, &pool));
    VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ca.commandPool = pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &ca, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CK(vkBeginCommandBuffer(cb, &bi));
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = src; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {W, H, 1};
    vkCmdCopyBufferToImage(cb, stg, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    if (!cmaa2::record(f, s, cb, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 2)) { printf("FAIL record\n"); return 1; }
    // Result (SHADER_READ_ONLY) -> TRANSFER_SRC -> staging.
    b.image = s.color; b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    vkCmdCopyImageToBuffer(cb, s.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stg, 1, &cp);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));

    const uint16_t *out = (const uint16_t *)map;
    uint32_t blended = 0, flatBad = 0;
    for (uint32_t y = 0; y < H; ++y) for (uint32_t x = 0; x < W; ++x) {
        const size_t i = ((size_t)y * W + x) * 4;
        const float v = fromHalf(out[i]);
        if (lineDist(x, y) >= 4.0f) { for (int c = 0; c < 4; ++c) if (out[i + c] != pix[i + c]) { ++flatBad; break; } }
        else if (v > 0.06f && v < 3.99f) ++blended; }
    printf("blended edge pixels %u (need >= 100), changed flat pixels %u (need 0)\n", blended, flatBad);
    cmaa2::destroy(f, s);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    if (blended < 100 || flatBad != 0) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
```

- [ ] **Step 2: Run it to verify it fails (no `cmaa2.h` yet)**

```bash
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"
g++ -O1 -std=c++17 tools/cmaa2_test.cpp -o "$TEMP/cmaa2_test.exe" -I"/c/VulkanSDK/1.4.357.0/Include" -I src/vklayer -L"/c/VulkanSDK/1.4.357.0/Lib" -lvulkan-1 2>&1 | head -3
```

Expected: a compile error, `cmaa2.h: No such file or directory`.

- [ ] **Step 3: Write `src/vklayer/cmaa2.h`**

Before writing it, check the buffer sizes against Intel's host code, `Projects/CMAA2/CMAA2/CMAA2.cpp`
in the same repo: look for `requiredCandidatePixels`, `requiredDeferredColorApplyBuffer`,
`requiredListHeadsPixels`. Where they differ from the constants below, use Intel's values.

```cpp
// cmaa2.h - Intel CMAA2 (third_party/cmaa2, Apache-2.0) as a Vulkan compute pass. FSCAA layer 1.
// X-Plane's scene has no STORAGE usage, so CMAA2 runs in place on its own RGBA16F copy; the TAA
// resolve then samples that copy. Self-contained behind a function table so tools/cmaa2_test.cpp
// can drive it on the real GPU. Bindings follow build.ps1's DXC shifts: u0..u7 -> 0..7, t0 -> 10.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include "cmaa2_spv.h"

namespace cmaa2 {

#define CMAA2_FNS(X) X(CreateImage) X(DestroyImage) X(CreateImageView) X(DestroyImageView) \
    X(CreateBuffer) X(DestroyBuffer) X(GetImageMemoryRequirements) X(GetBufferMemoryRequirements) \
    X(AllocateMemory) X(FreeMemory) X(BindImageMemory) X(BindBufferMemory) X(CreateShaderModule) \
    X(DestroyShaderModule) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
    X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
    X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdDispatch) X(CmdDispatchIndirect) \
    X(CmdPipelineBarrier) X(CmdFillBuffer) X(CmdClearColorImage) X(CmdCopyImage)

struct Fns {
#define CMAA2_DECL(n) PFN_vk##n n = nullptr;
    CMAA2_FNS(CMAA2_DECL)
#undef CMAA2_DECL
    VkPhysicalDeviceMemoryProperties memProps{};
};

// gdpa must be the NEXT layer's vkGetDeviceProcAddr inside the layer (DeviceData::gdpa), so these
// calls never re-enter our own hooks.
inline bool loadFns(Fns &f, PFN_vkGetDeviceProcAddr gdpa, VkDevice dev, const VkPhysicalDeviceMemoryProperties &mp)
{
#define CMAA2_LOAD(n) f.n = (PFN_vk##n)gdpa(dev, "vk" #n); if (!f.n) return false;
    CMAA2_FNS(CMAA2_LOAD)
#undef CMAA2_LOAD
    f.memProps = mp;
    return true;
}

enum { kImgColor, kImgEdges, kImgHeads, kBufCand, kBufLoc, kBufItems, kBufControl, kBufIndirect, kRes };

struct State {
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    VkImage color = VK_NULL_HANDLE, edges = VK_NULL_HANDLE, heads = VK_NULL_HANDLE;
    VkImageView colorView = VK_NULL_HANDLE, colorArrayView = VK_NULL_HANDLE, edgesView = VK_NULL_HANDLE, headsView = VK_NULL_HANDLE;
    VkBuffer cand = VK_NULL_HANDLE, loc = VK_NULL_HANDLE, items = VK_NULL_HANDLE, control = VK_NULL_HANDLE, indirect = VK_NULL_HANDLE;
    VkDeviceMemory mem[kRes] = {};
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipeline edgesPipe[4] = {}; VkPipeline argsPipe = VK_NULL_HANDLE, candPipe = VK_NULL_HANDLE, applyPipe = VK_NULL_HANDLE;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool initialised = false;   // working images moved to GENERAL, control/heads cleared
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

inline bool makeBuffer(const Fns &f, State &s, int slot, VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer &buf)
{
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bc.size = size; bc.usage = usage;
    if (f.CreateBuffer(s.dev, &bc, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; f.GetBufferMemoryRequirements(s.dev, buf, &mr);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ma.allocationSize = mr.size;
    ma.memoryTypeIndex = memType(f, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
    if (s.argsPipe) f.DestroyPipeline(s.dev, s.argsPipe, nullptr);
    if (s.candPipe) f.DestroyPipeline(s.dev, s.candPipe, nullptr);
    if (s.applyPipe) f.DestroyPipeline(s.dev, s.applyPipe, nullptr);
    if (s.pool) f.DestroyDescriptorPool(s.dev, s.pool, nullptr);
    if (s.pl) f.DestroyPipelineLayout(s.dev, s.pl, nullptr);
    if (s.dsl) f.DestroyDescriptorSetLayout(s.dev, s.dsl, nullptr);
    VkImageView views[4] = {s.colorView, s.colorArrayView, s.edgesView, s.headsView};
    for (int i = 0; i < 4; ++i) if (views[i]) f.DestroyImageView(s.dev, views[i], nullptr);
    VkImage imgs[3] = {s.color, s.edges, s.heads};
    for (int i = 0; i < 3; ++i) if (imgs[i]) f.DestroyImage(s.dev, imgs[i], nullptr);
    VkBuffer bufs[5] = {s.cand, s.loc, s.items, s.control, s.indirect};
    for (int i = 0; i < 5; ++i) if (bufs[i]) f.DestroyBuffer(s.dev, bufs[i], nullptr);
    for (int i = 0; i < kRes; ++i) if (s.mem[i]) f.FreeMemory(s.dev, s.mem[i], nullptr);
    s = State();
}

inline bool create(const Fns &f, State &s, VkDevice dev, uint32_t w, uint32_t h)
{
    destroy(f, s);
    s.dev = dev; s.w = w; s.h = h;
    const VkDeviceSize px = (VkDeviceSize)w * h;
    const VkBufferUsageFlags SB = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bool ok = makeImage(f, s, kImgColor, VK_FORMAT_R16G16B16A16_SFLOAT, w, h,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT, s.color) &&
              makeImage(f, s, kImgEdges, VK_FORMAT_R8_UINT, (w + 1) / 2, h, VK_IMAGE_USAGE_STORAGE_BIT, s.edges) &&
              makeImage(f, s, kImgHeads, VK_FORMAT_R32_UINT, (w + 1) / 2, (h + 1) / 2,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, s.heads) &&
              makeBuffer(f, s, kBufCand, (px / 4) * 4, SB, s.cand) &&
              makeBuffer(f, s, kBufLoc, ((px + 3) / 6) * 4, SB, s.loc) &&
              makeBuffer(f, s, kBufItems, (px / 2) * 8, SB, s.items) &&
              makeBuffer(f, s, kBufControl, 64, SB, s.control) &&
              makeBuffer(f, s, kBufIndirect, 64, SB | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, s.indirect);
    if (!ok) { destroy(f, s); return false; }
    s.colorView = makeView(f, s, s.color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D);
    s.colorArrayView = makeView(f, s, s.color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    s.edgesView = makeView(f, s, s.edges, VK_FORMAT_R8_UINT, VK_IMAGE_VIEW_TYPE_2D);
    s.headsView = makeView(f, s, s.heads, VK_FORMAT_R32_UINT, VK_IMAGE_VIEW_TYPE_2D);
    if (!s.colorView || !s.colorArrayView || !s.edgesView || !s.headsView) { destroy(f, s); return false; }

    // Layout: exactly what DXC's reflection reported in Task 1 Step 4.
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
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}};
    VkDescriptorPoolCreateInfo pc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; pc.maxSets = 1; pc.poolSizeCount = 3; pc.pPoolSizes = ps;
    if (f.CreateDescriptorPool(dev, &pc, nullptr, &s.pool) != VK_SUCCESS) { destroy(f, s); return false; }
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; da.descriptorPool = s.pool; da.descriptorSetCount = 1; da.pSetLayouts = &s.dsl;
    if (f.AllocateDescriptorSets(dev, &da, &s.set) != VK_SUCCESS) { destroy(f, s); return false; }

    VkDescriptorImageInfo ii[4] = {{VK_NULL_HANDLE, s.colorView, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, s.edgesView, VK_IMAGE_LAYOUT_GENERAL},
                                   {VK_NULL_HANDLE, s.headsView, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, s.colorView, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi[5] = {{s.cand, 0, VK_WHOLE_SIZE}, {s.loc, 0, VK_WHOLE_SIZE}, {s.items, 0, VK_WHOLE_SIZE},
                                    {s.control, 0, VK_WHOLE_SIZE}, {s.indirect, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[9];
    for (int i = 0; i < 9; ++i) { wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; wr[i].dstSet = s.set;
                                  wr[i].dstBinding = bindings[i]; wr[i].descriptorCount = 1; wr[i].descriptorType = types[i]; }
    wr[0].pImageInfo = &ii[0]; wr[1].pImageInfo = &ii[1]; wr[5].pImageInfo = &ii[2]; wr[8].pImageInfo = &ii[3];
    wr[2].pBufferInfo = &bi[0]; wr[3].pBufferInfo = &bi[1]; wr[4].pBufferInfo = &bi[2]; wr[6].pBufferInfo = &bi[3]; wr[7].pBufferInfo = &bi[4];
    f.UpdateDescriptorSets(dev, 9, wr, 0, nullptr);

    const uint32_t *edgeCode[4] = {kCmaa2Edges0, kCmaa2Edges1, kCmaa2Edges2, kCmaa2Edges3};
    const size_t edgeWords[4] = {kCmaa2Edges0Words, kCmaa2Edges1Words, kCmaa2Edges2Words, kCmaa2Edges3Words};
    for (int i = 0; i < 4; ++i) s.edgesPipe[i] = makePipe(f, s, edgeCode[i], edgeWords[i], "EdgesColor2x2CS");
    s.argsPipe  = makePipe(f, s, kCmaa2Args,  kCmaa2ArgsWords,  "ComputeDispatchArgsCS");
    s.candPipe  = makePipe(f, s, kCmaa2Cand,  kCmaa2CandWords,  "ProcessCandidatesCS");
    s.applyPipe = makePipe(f, s, kCmaa2Apply, kCmaa2ApplyWords, "DeferredColorApply2x2CS");
    for (int i = 0; i < 4; ++i) if (!s.edgesPipe[i]) { destroy(f, s); return false; }
    if (!s.argsPipe || !s.candPipe || !s.applyPipe) { destroy(f, s); return false; }
    return true;
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

// Copy src (in srcLayout, left as it was) into the working colour, run CMAA2 in place, leave the
// result in SHADER_READ_ONLY_OPTIMAL. preset 0..3 = LOW/MEDIUM/HIGH/ULTRA edge threshold.
inline bool record(const Fns &f, State &s, VkCommandBuffer cb, VkImage src, VkImageLayout srcLayout, int preset)
{
    if (!s.set || preset < 0 || preset > 3) return false;
    const VkPipelineStageFlags ALL = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, TR = VK_PIPELINE_STAGE_TRANSFER_BIT,
                               CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (!s.initialised) {
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
    // Scene -> working colour.
    imageBarrier(f, cb, src, srcLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, ALL, TR);
    imageBarrier(f, cb, s.color, s.colorLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, ALL, TR);
    VkImageCopy cp{}; cp.srcSubresource = cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.extent = {s.w, s.h, 1};
    f.CmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    imageBarrier(f, cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, srcLayout, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_READ_BIT, TR, ALL);
    imageBarrier(f, cb, s.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, TR, CS);

    f.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &s.set, 0, nullptr);
    // Edges: Intel dispatches (w + 2*(16-2) - 1) / (2*(16-2)) groups per axis.
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
    imageBarrier(f, cb, s.color, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, CS, CS | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    s.colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return true;
}

} // namespace cmaa2
```

- [ ] **Step 4: Run the test under validation and verify it passes**

```bash
g++ -O1 -std=c++17 tools/cmaa2_test.cpp -o "$TEMP/cmaa2_test.exe" -I"/c/VulkanSDK/1.4.357.0/Include" -I src/vklayer -L"/c/VulkanSDK/1.4.357.0/Lib" -lvulkan-1
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation "$TEMP/cmaa2_test.exe" 2>&1 | tail -6
```

Expected:
- `blended edge pixels N (need >= 100), changed flat pixels 0 (need 0)`
- then `PASS`
- no lines containing `Validation Error`

If the blended count is low, try preset 3 in the `record` call, and record the result in the spec's
HDR-threshold risk.

- [ ] **Step 5: Commit**

```bash
git add src/vklayer/cmaa2.h tools/cmaa2_test.cpp
git commit -m "FSCAA: cmaa2.h host module (Intel CMAA2 on our scene copy), GPU-tested offline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Run CMAA2 in the TAA resolve under `taa.backend=fscaa`

**Files:**
- Modify: `src/vklayer/taa.h`
  - the include block at the top
  - `taaRecordResolve` (starts near line 1669)
  - the scene descriptor `ii[0]` near line 2077

**Interfaces:**
- Consumes: `cmaa2::loadFns`, `cmaa2::create`, `cmaa2::record`, `cmaa2::State::colorArrayView` (Task 2).
- Produces:
  - `static bool fscaaOn()`: true when `taa.backend` is `fscaa`.
  - `static VkImageView g_fscaaSceneView`: set per frame when CMAA2 ran, otherwise `VK_NULL_HANDLE`.
    Task 4 reads it through `fscaaOn()`.

- [ ] **Step 1: Add the module state and switch**

At the top of `taa.h`, next to the other includes, add `#include "cmaa2.h"`. Then, before
`taaRecordResolve`, add:

```cpp
// ---- FSCAA (taa.backend=fscaa, docs/superpowers/specs/2026-10-15-fscaa-design.md). Layer 1 is
// Intel's CMAA2 on a copy of the scene, run before the resolve; the resolve then samples the
// cleaned copy as uScene, so the history accumulates cleaned frames. Layer 2 (class caps) is in
// taa.comp, driven by uFscaa/uFscaa2.
static cmaa2::Fns   g_cmaaFns;
static cmaa2::State g_cmaa;
static bool         g_cmaaFnsOk = false, g_cmaaFailed = false;
static VkImageView  g_fscaaSceneView = VK_NULL_HANDLE;

static bool fscaaOn()
{
    std::string be;
    return live::lookup("taa.backend", be) && be == "fscaa";
}
```

- [ ] **Step 2: Record CMAA2 before the resolve dispatch**

In `taaRecordResolve`, just before the line `VkDescriptorImageInfo ii[8];` (near 2075), add:

```cpp
    // FSCAA layer 1: CMAA2 on a copy of the scene. The scene is sampled by the resolve in
    // SHADER_READ_ONLY_OPTIMAL; record() leaves it in exactly that layout.
    g_fscaaSceneView = VK_NULL_HANDLE;
    if (fscaaOn() && g_taa.layers == 1 && !g_cmaaFailed && live::onoff("fscaa.cmaa", nullptr, true)) {
        if (!g_cmaaFnsOk) {
            VkPhysicalDeviceMemoryProperties mp; memset(&mp, 0, sizeof(mp));
            if (g_getPhysMemProps && dd.phys) g_getPhysMemProps(dd.phys, &mp);
            g_cmaaFnsOk = dd.gdpa && cmaa2::loadFns(g_cmaaFns, dd.gdpa, g_taa.device, mp);
            if (!g_cmaaFnsOk) { g_cmaaFailed = true; trace("FSCAA: CMAA2 unavailable - entry points missing"); }
        }
        if (g_cmaaFnsOk && (g_cmaa.w != g_taa.w || g_cmaa.h != g_taa.h || !g_cmaa.set)) {
            if (!cmaa2::create(g_cmaaFns, g_cmaa, g_taa.device, g_taa.w, g_taa.h)) {
                g_cmaaFailed = true;
                trace("FSCAA: CMAA2 resources for %ux%u FAILED - running class-weighted TAA without it", g_taa.w, g_taa.h);
            } else {
                trace("FSCAA: CMAA2 ready at %ux%u", g_taa.w, g_taa.h);
            }
        }
        const int preset = (int)live::f("fscaa.cmaa_preset", nullptr, 2.0f);
        if (g_cmaaFnsOk && !g_cmaaFailed &&
            cmaa2::record(g_cmaaFns, g_cmaa, cb, g_taa.sceneImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          preset < 0 ? 0 : preset > 3 ? 3 : preset))
            g_fscaaSceneView = g_cmaa.colorArrayView;
    }
```

Check before moving on: by this point in `taaRecordResolve` the scene must already be in
`SHADER_READ_ONLY_OPTIMAL`. Find the barrier that moves `g_taa.sceneImage` into that layout. If it is
recorded after this point, move the block to just after that barrier.

- [ ] **Step 3: Point `uScene` at the cleaned copy**

Replace

```cpp
    ii[0].imageView = g_taa.sceneView;
```

with

```cpp
    ii[0].imageView = g_fscaaSceneView ? g_fscaaSceneView : g_taa.sceneView;
```

- [ ] **Step 4: Free the resources with the TAA state**

In the function that tears down `g_taa` on device destroy, find it with
`grep -n "g_taa = TaaState()" src/vklayer/taa.h`, and add before that line:

```cpp
    if (g_cmaaFnsOk) cmaa2::destroy(g_cmaaFns, g_cmaa);
    g_cmaaFnsOk = false; g_cmaaFailed = false; g_fscaaSceneView = VK_NULL_HANDLE;
```

- [ ] **Step 5: Build to the staging DLL**

```bash
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"
MV_LAYER_OUT='D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors\build\vklayer\VkLayer_mv.staged.dll' \
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 > "$TEMP/fscaa_t3.log" 2>&1; echo "exit $?"
grep -nE "taa\.h:[0-9]+:[0-9]+: (error|warning)" "$TEMP/fscaa_t3.log" | head
```

Expected: `exit 0` (or only the plugin-install failure if X-Plane is running), and no errors or
warnings from `taa.h`.

- [ ] **Step 6: Commit**

```bash
git add src/vklayer/taa.h
git commit -m "FSCAA: run CMAA2 before the TAA resolve under taa.backend=fscaa

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Class-weighted history in the resolve, plus `taa.viz=13`

**Files:**
- Modify: `src/shaders/taa.comp`
  - the `ReprojData` block at lines 124–138
  - the `VIZ_` constants near line 564
  - after the `SCREEN CLASS` block near line 2211
- Modify: `src/vklayer/taa.h` (the slot writes near line 2066)

**Interfaces:**
- Consumes: `fscaaOn()` (Task 3).
- Produces:
  - UBO slots 32–35 = `uFscaa` (x on, y near_m, z/w unused)
  - UBO slots 36–39 = `uFscaa2` (x keep_screen, y keep_near, z keep_light, w unused)

- [ ] **Step 1: Extend the UBO in `taa.comp`**

In `layout(binding = 6, std140) uniform ReprojData`, after `vec4 uReprojParams4;`, add:

```glsl
    // FSCAA (taa.backend=fscaa): x on, y fscaa.near_m. Slots 32-35.
    vec4 uFscaa;
    // FSCAA history caps: x fscaa.keep_screen, y fscaa.keep_near, z fscaa.keep_light. Slots 36-39.
    vec4 uFscaa2;
```

- [ ] **Step 2: Apply the class caps**

Directly after the closing `}` of the `SCREEN CLASS (taa.alpha_screen ...)` block, add:

```glsl
    // ---- FSCAA CLASS CAPS (taa.backend=fscaa, uFscaa.x). History is the right tool for the
    // scenery and the wrong one for the own aircraft and the cockpit: there it only smears. Each
    // class caps how much history it may keep (1 = full TAA, 0 = the CMAA2-cleaned current frame),
    // applied as a floor on the current weight, the same mechanism as alpha_near/alpha_screen.
    if (uFscaa.x > 0.5 && !unwritten) {
        float mwF = texelFetch(uVelocity, ivec3(px, layer), 0).w;
        float keep = 1.0;
        if (abs(mwF - 0.5) < 0.02)  keep = uFscaa2.x;                         // cockpit screens
        else if (mwF >= 1.5)        keep = uFscaa2.z;                         // light class
        else if (velS.z > 0.0 && mvViewDepth(uv + S, float(layer), velS.z) < uFscaa.y)
                                    keep = uFscaa2.y;                         // own aircraft, cockpit
        a = max(a, 1.0 - clamp(keep, 0.0, 1.0));
    }
```

- [ ] **Step 3: Add the class-map debug view**

After `const int VIZ_GI        = 11;` (if 12 is already used by the light marker, keep 13), add:

```glsl
const int VIZ_FSCAA     = 13;  // FSCAA classes: blue screens, yellow lights, red near, green scenery
```

and directly after the `if (pc.viz == VIZ_WRITTEN) { ... return; }` block, add:

```glsl
    if (pc.viz == VIZ_FSCAA) {
        float mwV = texelFetch(uVelocity, ivec3(px, layer), 0).w;
        vec3 c = vec3(0.0, 0.6, 0.0);                                          // scenery: full TAA
        if (unwritten)                   c = vec3(0.2);                       // no vector
        else if (abs(mwV - 0.5) < 0.02)  c = vec3(0.1, 0.3, 1.0);             // screens
        else if (mwV >= 1.5)             c = vec3(1.0, 0.9, 0.1);             // lights
        else if (velS.z > 0.0 && mvViewDepth(uv + S, float(layer), velS.z) < uFscaa.y)
                                         c = vec3(1.0, 0.15, 0.1);            // near
        imageStore(uHistory, ivec3(px, layer), vec4(c, 1.0));
        return;
    }
```

`velS`, `S`, `uv` and `mvViewDepth` must be defined before this point. If the `VIZ_WRITTEN` block
comes before `velS` is computed, move this block to just after `velS`'s definition, and keep its
`return`.

- [ ] **Step 4: Write the slots in `taa.h`**

After the `slot[31] = ...` line near 2072, add:

```cpp
        // uFscaa / uFscaa2 (taa.comp): FSCAA class caps. Off unless taa.backend=fscaa.
        const bool fscaa = fscaaOn() && g_taa.layers == 1;
        slot[32] = fscaa ? 1.0f : 0.0f;
        slot[33] = live::f("fscaa.near_m", nullptr, 80.0f);
        slot[34] = 0.0f; slot[35] = 0.0f;
        slot[36] = live::f("fscaa.keep_screen", nullptr, 0.0f);
        slot[37] = live::f("fscaa.keep_near", nullptr, 0.5f);
        slot[38] = live::f("fscaa.keep_light", nullptr, 1.0f);
        slot[39] = 0.0f;
```

Then check that the UBO buffer is at least 160 bytes per set; it is 2048-byte strided.
`grep -n "2048u" src/vklayer/taa.h` should show the stride.

- [ ] **Step 5: Build and confirm the shader validates**

```bash
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"
MV_LAYER_OUT='D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors\build\vklayer\VkLayer_mv.staged.dll' \
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 > "$TEMP/fscaa_t4.log" 2>&1; echo "exit $?"
grep -nE "taa\.comp|spirv-val|ERROR" "$TEMP/fscaa_t4.log" | head
```

Expected: no `taa.comp` compile errors, and spirv-val passes. The build stops with a message if it
fails.

- [ ] **Step 6: Commit**

```bash
git add src/shaders/taa.comp src/vklayer/taa.h src/vklayer/taa_spv.h
git commit -m "FSCAA: per-class history caps in the resolve (screens/near/lights) and taa.viz=13

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: In-flight verification, A/B, tuning, then restore the launch rule

**Files:**
- Modify: `%TEMP%\taa_live.ini` (live controls)
- Modify: `C:\Users\bansa\.claude\projects\d--Steam-Games-steamapps-common-X-Plane-12\memory\` (one
  new memory file plus a `MEMORY.md` line)

- [ ] **Step 1: Install and configure**

With X-Plane closed, copy `build\vklayer\VkLayer_mv.staged.dll` over `build\vklayer\VkLayer_mv.dll`,
keeping the old one as `VkLayer_mv.pre-fscaa.dll`. Then make sure `%TEMP%\taa_live.ini` has:

```
taa.enable=1
taa.backend=fscaa
fscaa.cmaa=1
fscaa.cmaa_preset=2
fscaa.near_m=80
fscaa.keep_near=0.5
fscaa.keep_screen=0.0
fscaa.keep_light=1.0
taa.fg=0
```

The user launches X-Plane, or asks for it to be launched.

- [ ] **Step 2: Confirm it runs (trace)**

```bash
t=$(ls -t "$LOCALAPPDATA/Temp"/taa_layer_*.txt | head -1); grep -E "FSCAA:|LIVE: (taa\.backend|fscaa\.)" "$t" | head; grep -E "^PERF:" "$t" | tail -3
```

Expected: `FSCAA: CMAA2 ready at 2560x1440`, the `fscaa.*` keys read, and PERF lines with
`backend=fscaa`.

- [ ] **Step 3: Check the class map**

Set `taa.viz=13` live. Expected: the cockpit screens show blue, the own aircraft and cockpit red,
scenery green, and lights yellow at night. Set `taa.viz=0` after.

- [ ] **Step 4: A/B in one flight**

Same spot, about 30 s each, switching live:
1. `taa.backend=taa`
2. `taa.backend=fscaa` with `fscaa.cmaa=0` (caps only)
3. `fscaa.cmaa=1`

The user judges smear on the livery and screens, softness, and shimmer. PERF gives the cost of each
segment.

- [ ] **Step 5: Tune**

Adjust these one at a time, live:
- `fscaa.keep_near`: 0.5 → 0.3 → 0 (watch for edge wobble from jitter)
- `fscaa.near_m`: wide enough to cover the airframe in chase view
- `fscaa.cmaa_preset`: lower it if the sky shows false edges, raise it if edges are missed

Record the final values in the spec's live-keys table, and make them the defaults in Task 4's slot
writes.

- [ ] **Step 6: Restore the launch rule and record it**

1. Set `taa.enable=0` and `taa.backend=taa` in `%TEMP%\taa_live.ini`, unless the user asks to keep
   FSCAA on.
2. Write a memory file `fscaa.md` with the measured cost, the chosen defaults, and what the user
   saw.
3. Add a one-line pointer in `MEMORY.md`.
4. Commit any default changes.
