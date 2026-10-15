# FSCAA: Flight Sim Combined Anti-Aliasing

Anti-aliasing for **X-Plane 12**, built specifically for what aliases in a flight sim: power lines,
runway lights, sparkling water, crawling trees, and a cockpit full of screens that temporal AA
smears.

FSCAA doesn't use a neural network. It's a stack of six techniques. Each one targets a different
source of aliasing, so they work together without blurring the same edge twice.

| # | Layer | Fixes | Status |
|---|---|---|---|
| 1 | **CMAA2** (Intel) | geometric edges, every frame, no history | Done |
| 2 | **Class-weighted TAA** | temporal shimmer on scenery, but no history on your aircraft or cockpit screens | Done |
| 3 | **Wire AA** (Persson) | power lines and cables breaking into dots | Done |
| 4 | **Light-point AA** | flickering runway and city lights | Done |
| 5 | **Geometric specular AA** (Tokuyoshi & Kaplanyan) | sparkle on water, metal, wet runways | Done |
| 6 | **Anti-aliased alpha test** (Golus) | crawling foliage and fences | Done |

## Why not just TAA, MSAA or DLAA?

- **TAA** is great at shimmer but smears and softens: the livery, cockpit displays, blinking lights.
- **MSAA** is sharp, but in X-Plane 12 it costs a lot of VRAM: 4x is about +388 MB at 1440p,
  measured. It also does nothing for shimmer, thin lines or specular sparkle.
- **DLAA** is strong, but it can ghost on cockpit displays and is too expensive on 8 GB cards
  (measured at about 38 ms per frame when VRAM was over budget).

FSCAA keeps TAA only where TAA helps (the scenery). Everywhere else it uses CMAA2's sharp,
history-free result, and it fixes thin lines, lights and specular at the source, before they
ever alias.

## How it runs

FSCAA is a backend of the
[MotionVectors](https://github.com/Vihaan2012-cmyk/Motion-Vectors-for-Xplane-12-V2) Vulkan layer for
X-Plane 12. MotionVectors provides the motion vectors, the per-pixel classes (cockpit screens, lights,
near geometry) and the TAA resolve that FSCAA builds on. Select it in the live ini:

```
taa.enable=1
taa.backend=fscaa
```

Settings live in `%TEMP%\taa_live.ini`.

**Live** (take effect in flight):

| Key | Default | Meaning |
|---|---|---|
| `fscaa.layers` | 63 | bitmask: 1 CMAA2, 2 class caps, 4 wire, 8 light, 16 specular, 32 alpha (bits 4–32 latch at load) |
| `fscaa.cmaa_preset` | 3 | CMAA2 edge threshold: 0 LOW … 3 ULTRA |
| `fscaa.near_m` | 80 | distance treated as "your aircraft / cockpit" |
| `fscaa.keep_near` | 1.0 | how much history near geometry may keep (below 1, near edges show the jitter) |
| `fscaa.keep_screen` | 0.0 | history on cockpit screens (0 = none, no ghosting) |
| `fscaa.keep_light` | 1.0 | history on lights |

**Latched at load** (shader rewrites; restart X-Plane after changing):

| Key | Default | Meaning |
|---|---|---|
| `fscaa.wire_min_px` | 1.0 | thinnest a wire may draw; lost coverage becomes alpha |
| `fscaa.light_min_px` | 1.5 | smallest a light sprite may draw; brightness is energy-conserved |
| `fscaa.spec_sigma` / `fscaa.spec_kappa` | 0.25 / 0.18 | specular AA filter width / clamp |
| `fscaa.alpha_edge` | 1.0 | width of the dithered alpha-test edge, in `fwidth` units |

**Debugging:**

| Key | Meaning |
|---|---|
| `taa.viz=13` | class map: blue screens, yellow lights, red near, green scenery |
| `fscaa.debug_edges=1` | overlay CMAA2's detected edges (live) |
| `fscaa.stats_every=600` | `FSCAA STATS` trace line every N frames (CMAA2 counters, caps, memory) |
| `fscaa.debug_tint=<family>` | draw one X-Plane shader family magenta, e.g. `line3d` (latched) |
| `taa.gpu_timing=1` | per-pass GPU time, including the `cmaa2` span |

The trace also logs `FSCAA PATCH:` lines with how many shaders each layer rewrote.

## Verification (offline, 2026-10-15, RTX 4060)

| Check | Result |
|---|---|
| CMAA2 on the GPU (`tools/cmaa2_test.cpp`), copy and in-place, presets 0–3, Khronos validation | PASS: 306 edge pixels blended, 0 flat pixels changed, in-place bit-identical to copy, 0 validation errors |
| Shader rewrites over all 6,775 X-Plane 12.4.4 variants (`tools/fscaa_validate.sh`) | wire 2, light 4, specular 960, alpha 3,724 patched; every output `spirv-val` valid |
| Runtime hash vs generated tables (`tools/fscaa_table_test.cpp`) | 5,617 / 5,617 modules found |
| MotionVectors' own shader injection on top of FSCAA (`tools/fscaa_inject_test.cpp`) | 0 regressions; 1,436 injected modules `spirv-val` valid |

## Image quality against a multi-sample truth (2026-10-16)

`tools/bench/sharpen_eval.py` scores the post-resolve pipeline against a truth accumulated from bursts of
jittered frames (20 bursts, native 2560x1440, captured 2026-09-11). Error is mean |x - truth| in
tonemapped space, x1000, lower is better. Detail is the texture-region Laplacian energy relative to the
truth's: below 1 is softer than the truth.

| Delivered image | Edge error | All pixels | Detail |
|---|---|---|---|
| TAA resolve alone | 36.19 | 8.04 | 0.88 |
| Laplacian sharpen 0.5 + CMAA2 HIGH (first FSCAA build) | 38.46 | 8.70 | 1.24 |
| **CAS sharpen + CMAA2 ULTRA (current)** | **36.61** | **8.38** | **1.11** |

The TAA resolve alone is soft. A fixed Laplacian sharpen restores detail but re-stairs the edges: on its
own it adds 13% edge error. CAS adapts its strength to local contrast. CMAA2 then removes what the sharpen
re-stairs; on the resolve's own output it changes almost nothing. `tools/cmaa2_file.cpp` runs the shipped
CMAA2 kernels on a captured frame for this bench.

GPU cost in flight (2560x1440, RTX 4060): resolve 1.46 ms, sharpen 0.21 ms, CMAA2 0.19 ms, copy 0.09 ms.

## Documents

- [Design](docs/design.md): the architecture, why each layer is there, risks, and testing
- [Full build plan](docs/plans/full-build.md): all six layers + the debugging toolkit, task by task
- [Phase 1 detail](docs/plans/phase1-cmaa2-class-taa.md): CMAA2 + class-weighted TAA, step by step
- [Integration](integration/motionvectors.patch): the changes to the MotionVectors layer (`taa.h`, `taa.comp`, `layer.cpp`, `build.ps1`)

## Target

X-Plane 12.4.x on Vulkan, Windows. Developed and measured on an RTX 4060 (8 GB) at 2560x1440. The
budget is about 1 ms on top of TAA. CMAA2 runs after the resolve, in place on the layer's own sharpen
target: 25 MB at 1440p and no extra frame copy. Measured in flight at 2560x1440 (2026-10-15): resolve
1.46 ms, sharpen 0.21 ms, CMAA2 0.19 ms, copy 0.09 ms - 1.94 ms in all.

## Licence

GPL-3.0 (see [LICENSE](LICENSE)), matching MotionVectors. Third-party code and credits are in
[THIRD_PARTY.md](THIRD_PARTY.md).
