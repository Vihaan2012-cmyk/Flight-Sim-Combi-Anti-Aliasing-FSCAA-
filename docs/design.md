# FSCAA: Flight Sim Combined Anti-Aliasing (design)

Date: 2026-10-15. Status: approved direction (user, 2026-10-15). Phase 1 is planned in
`docs/superpowers/plans/2026-10-15-fscaa-phase1.md`.

## Goal

A stack of non-neural anti-aliasing for X-Plane 12, selected with `taa.backend=fscaa`. Each layer
attacks a different source of aliasing, so they compose instead of blurring twice. The stack should
beat our TAA alone and MSAA in motion, and fit an 8 GB card (RTX 4060 at 2560x1440).

## Why a stack

Aliasing in a flight sim has six independent sources. A single method fixes some of them and
damages others: TAA smears the airframe and cockpit screens, and MSAA ignores shimmer.

| # | Layer | Source it fixes | Where it runs | Phase |
|---|---|---|---|---|
| 1 | CMAA2 (Intel, Apache-2.0) | geometric edges in the current frame | compute pass before the TAA resolve | 1 |
| 2 | Class-weighted TAA | temporal shimmer, on scenery only | the existing resolve (`taa.comp`) | 1 |
| 3 | Wire AA (Persson, GPU Pro 5) | power lines and cables breaking into dots | X-Plane's wire vertex shaders (SPIR-V patch) | 2 |
| 4 | Light-point AA (min size + fade) | flickering runway and city lights | X-Plane's light sprite shaders | 3 |
| 5 | Geometric specular AA (Tokuyoshi & Kaplanyan) | sparkle on water, metal, wet runways | X-Plane's `deferred_gbuf` lighting | 4 |
| 6 | Anti-aliased alpha test (Golus) | crawling foliage and fences | X-Plane's alpha-tested material shaders | 5 |

Out of scope:
- MSAA: the layer stands down under MSAA because the 1x velocity target can't sit beside N-sample
  attachments. A multisampled velocity target is a possible later layer 7.
- SRAA: needs MSAA buffers.
- AGAA: requires a renderer rewrite.
- SMAA and FXAA: redundant with CMAA2.
- GBAA: marginal gain over CMAA2 + TAA.
- DLAA and DLDSR: these are separate backends or driver features.

## Phase 1 architecture (CMAA2 + class-weighted TAA)

Per frame, inside `taaRecordResolve` (`src/vklayer/taa.h`), when `taa.backend=fscaa` and there is
one view layer:

1. **Copy.** Copy X-Plane's HDR scene (RGBA16F, no STORAGE usage) into the CMAA2 working colour
   image (RGBA16F, STORAGE|SAMPLED|TRANSFER).
2. **CMAA2 in place on the copy.** These are Intel's four kernels from the vendored, unmodified
   `third_party/cmaa2/CMAA2.hlsl`, compiled by DXC to SPIR-V at build time:
   - `EdgesColor2x2CS`
   - `ComputeDispatchArgsCS`
   - `ProcessCandidatesCS` (indirect)
   - `ComputeDispatchArgsCS`
   - `DeferredColorApply2x2CS` (indirect)

   Defines:
   - `CMAA2_SUPPORT_HDR_COLOR_RANGE=1`
   - `CMAA2_EDGE_DETECTION_LUMA_PATH=1`
   - `CMAA2_UAV_STORE_TYPED=1`
   - `CMAA2_UAV_STORE_TYPED_UNORM_FLOAT=0`
   - `CMAA2_UAV_STORE_CONVERT_TO_SRGB=0`
   - preset 0..3 (one edges kernel per preset, chosen live by `fscaa.cmaa_preset`)
3. **Resolve.** The TAA resolve samples the CMAA2 image as `uScene` (binding 0) instead of X-Plane's
   scene. History therefore accumulates already-cleaned frames.
4. **Class weights.** For each pixel the resolve applies a per-class cap on kept history, as a floor
   on the current-frame weight `a` (`a = max(a, 1 - keep)`). This is the same mechanism as
   `taa.alpha_near` and `taa.alpha_screen`. Classes come from data the layer already writes:

| Class | How it's identified | Key | Default keep |
|---|---|---|---|
| Cockpit screens | velocity `.w` ≈ 0.5 (screen class) | `fscaa.keep_screen` | 0.0 |
| Lights | velocity `.w` ≥ 1.5 (light class) | `fscaa.keep_light` | 1.0 |
| Near (own aircraft, cockpit) | view depth < `fscaa.near_m` (default 80 m) | `fscaa.keep_near` | 0.5 |
| Everything else (scenery) | none of the above | none | 1.0 (full TAA) |

`keep_near` defaults to 0.5, not 0: near geometry is still jittered, and with no history at all the
jitter would show as edge wobble (memory: airframe jitter fringe). Tuning it is part of Phase 1.

New UBO data: two vec4s appended to `ReprojData` (`uFscaa`, `uFscaa2`), host slots 32–39. The UBO
stride is 2048 B per set, so there is room. New debug view: `taa.viz=13` draws the class map.

## Live keys (Phase 1)

| Key | Default | Meaning |
|---|---|---|
| `taa.backend` | (unchanged) | `fscaa` selects this stack |
| `fscaa.cmaa` | 1 | run CMAA2 before the resolve |
| `fscaa.cmaa_preset` | 2 | 0 LOW (0.15), 1 MEDIUM (0.10), 2 HIGH (0.07), 3 ULTRA (0.05) edge thresholds |
| `fscaa.near_m` | 80 | near-class distance |
| `fscaa.keep_near` | 0.5 | history cap, near |
| `fscaa.keep_screen` | 0.0 | history cap, cockpit screens |
| `fscaa.keep_light` | 1.0 | history cap, lights |

## Cost budget

- **VRAM:** about 45 MB at 2560x1440: the colour copy (29.5 MB) plus edges, heads, candidate, list,
  control and indirect buffers.
- **GPU:** about 1 ms, which is Intel's figure for CMAA2 at this resolution.
- **TAA:** cost unchanged.

## Risks

- **HDR edge threshold.** Intel notes that HDR input may need higher thresholds; the preset is live
  for this reason. If even LOW over-detects in bright sky, run edge detection on tonemapped luma
  (`CMAA2_EDGE_DETECTION_LUMA_PATH=2` with a precomputed luma pass). That is a follow-up, not in
  Phase 1.
- **Storage image format.** Our working images are RGBA16F and R8_UINT, while DXC defaults typed UAV
  formats to 32-bit. The build uses `-fspv-use-unknown-image-format`, which requires
  `shaderStorageImageRead/WriteWithoutFormat`. The RTX 4060 has both, and the host checks for them.
  The fallback is adding `[[vk::image_format]]` in the vendored file as a recorded local patch.
- **Standing rule.** Testing needs TAA on, which pauses the VRAM-only launch rule. Restore it after
  testing unless the user says otherwise.

## Testing

- **Build:**
  - DXC compiles all 7 SPIR-V blobs.
  - `spirv-val` passes on each.
  - Reflection shows bindings 0–7 and 10.
- **Offline (`tools/cmaa2_test.cpp`):**
  - Runs `cmaa2.h` on the RTX 4060 against a synthetic 256x256 RGBA16F image with a hard diagonal
    edge.
  - Pass if edge pixels become intermediate values, flat regions stay bit-identical, and Khronos
    validation reports zero errors.
- **In flight:**
  - The trace shows `FSCAA:` lines.
  - `taa.viz=13` shows the class map.
  - A/B in one flight: `taa.backend=taa` against `fscaa`, and `fscaa.cmaa=0/1`.
  - Use the existing metrics suite (`metrics.comp`, aggregate-only) for shimmer and flicker numbers.

## Later phases (each gets its own plan, starting with an investigation)

- **Phase 2, wire AA.** Find the shader family that draws power lines and cables, and how their
  vertices encode width. Patch the vertex shader to clamp the projected width to at least 1 px and
  carry the lost coverage as alpha.
- **Phase 3, light-point AA.** Same treatment for the light sprites: these are the pulled geometry
  from the night-lights fix. Add a minimum pixel size and fade by coverage.
- **Phase 4, specular AA.** In `deferred_gbuf`, widen roughness by screen-space normal variance:
  `r² += min(2·σ², κ²)`, with σ² from `dFdx/dFdy` of the normal.
- **Phase 5, alpha AA.** In alpha-tested materials, sharpen the alpha test with `fwidth` (Golus).
  Only applies where the material path allows it.
