# Third-party code and credits

## Code

| Component | Author | Licence | Use |
|---|---|---|---|
| CMAA2 (`CMAA2.hlsl`) | Intel Corporation, [GameTechDev/CMAA2](https://github.com/GameTechDev/CMAA2) | Apache-2.0 (compatible with GPL-3.0) | Layer 1. Vendored unmodified and compiled to SPIR-V with DXC. Local changes, if any, are listed in `third_party/cmaa2/LOCAL_PATCHES.md`. |

## Techniques (ideas, not code)

- **Wire anti-aliasing:** Emil Persson (Humus), "Wire Antialiasing", *GPU Pro 5*, CRC Press, 2014, pp. 211–218.
- **Geometric specular anti-aliasing:** Yusuke Tokuyoshi and Anton S. Kaplanyan, "Improved Geometric
  Specular Antialiasing", I3D 2019; "Stable Geometric Specular Antialiasing with Projected-Space NDF
  Filtering", JCGT 10(2), 2021.
- **Anti-aliased alpha test:** Ben Golus, "Anti-aliased Alpha Test: The Esoteric Alpha To Coverage", 2017.
- **Light-point rendering:** the minimum-size and coverage-fade approach used by professional
  flight-simulator image generators. See US patents 6,163,320 and 8,466,874 for background.
- **Temporal AA, motion vectors and pixel classes:** the MotionVectors project.
