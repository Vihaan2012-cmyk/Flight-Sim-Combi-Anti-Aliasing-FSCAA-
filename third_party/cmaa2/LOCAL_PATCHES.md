# Local patches to CMAA2.hlsl

Source: GameTechDev/CMAA2 master, `Projects/CMAA2/CMAA2/CMAA2.hlsl`, fetched 2026-10-15.
`reference/` holds Intel's host code (`vaCMAA2*.cpp`) for buffer sizes and dispatch order only; it is not compiled.

## 1. `LoadSourceColor`: constant-offset rule (2026-10-15)

DXC's SPIR-V backend rejects `Texture2D.Load(coord, offset)` with a non-constant offset ("Offsets to
texture access operations must be immediate values"). Vulkan's `OpImageFetch` only takes
`ConstOffset`, and Intel's callers pass loop-variable offsets: `int2(i % 3, i / 3)`.

The offset is now added to the coordinate and clamped to the image. D3D returns 0 for out-of-range
loads, while Vulkan's result is undefined without robustness, so clamping keeps edge pixels defined.
It only differs from D3D at the outermost pixel row and column.

## 2. `g_workingDeferredBlendItemListHeads`: explicit `r32ui` (2026-10-15)

The build uses `-fspv-use-unknown-image-format`, so the RGBA16F colour and R8_UINT edges images
need no format annotations. But `ProcessCandidatesCS` does `InterlockedExchange` on this image, and
Vulkan requires a 32-bit format on images used with `OpImageTexelPointer`
(VUID-StandaloneSpirv-OpImageTexelPointer-04658). Fix: `[[vk::image_format("r32ui")]]` on the
declaration. The image is `R32_UINT`, so nothing else changes.
