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

Phase 1 settings (all live, no restart):

| Key | Default | Meaning |
|---|---|---|
| `fscaa.cmaa` | 1 | run CMAA2 before the resolve |
| `fscaa.cmaa_preset` | 2 | edge threshold: 0 LOW … 3 ULTRA |
| `fscaa.near_m` | 80 | distance treated as "your aircraft / cockpit" |
| `fscaa.keep_near` | 0.5 | how much history near geometry may keep |
| `fscaa.keep_screen` | 0.0 | history on cockpit screens (0 = none, no ghosting) |
| `fscaa.keep_light` | 1.0 | history on lights |
| `taa.viz=13` | n/a | debug view of the class map |

## Documents

- [Design](docs/design.md): the architecture, why each layer is there, risks, and testing
- [Full build plan](docs/plans/full-build.md): all six layers + the debugging toolkit, task by task
- [Phase 1 detail](docs/plans/phase1-cmaa2-class-taa.md): CMAA2 + class-weighted TAA, step by step

## Target

X-Plane 12.4.x on Vulkan, Windows. Developed and measured on an RTX 4060 (8 GB) at 2560x1440. The
budget for Phase 1 is about 1 ms and about 45 MB of VRAM on top of TAA.

## Licence

GPL-3.0 (see [LICENSE](LICENSE)), matching MotionVectors. Third-party code and credits are in
[THIRD_PARTY.md](THIRD_PARTY.md).
