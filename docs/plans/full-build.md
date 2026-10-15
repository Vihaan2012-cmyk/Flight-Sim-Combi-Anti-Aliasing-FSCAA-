# FSCAA, Full Build (all six layers + debugging) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build all of FSCAA in one run: CMAA2, class-weighted TAA, wire AA, light-point AA,
geometric specular AA and anti-aliased alpha test, plus a debugging toolkit. Everything that can
be proven offline is proven offline, and one consolidated test flight closes it out.

**Architecture:** three kinds of work.
- **Compute passes and resolve changes** in the MotionVectors layer (`taa.h`, `taa.comp`,
  `cmaa2.h`): layers 1–2.
- **Load-time SPIR-V rewrites of X-Plane's own shaders** (`fscaa_patch.h`), the same mechanism as
  the validated cloud patch:
  - `line3d` for layer 3
  - `light` / `ground_lights` for layer 4
  - `deferred_gbuf` for layer 5
  - alpha-tested `terrain` variants for layer 6
- **Debug tooling:** live viz modes, stats traces, GPU timings, a per-family tint, and an offline
  validator that runs every patch over all 6,775 X-Plane shader variants under `spirv-val`.

**Tech Stack:**
- C++17 (MinGW g++)
- Vulkan 1.2
- GLSL (glslangValidator) and HLSL (DXC 1.9)
- Python 3.12 for the offline tools
- PowerShell `build.ps1`

**Spec:** `docs/superpowers/specs/2026-10-15-fscaa-design.md`

**Detailed sub-plan:** `docs/superpowers/plans/2026-10-15-fscaa-phase1.md`. Its Tasks 1–4 are this
plan's Tasks 1–4, executed exactly as written there; executors read both documents.

## Global Constraints

- **Repo and build:**
  - MotionVectors repo: `D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors`, branch
    `gbuffer-harvest`.
  - Public mirror: `D:\FSCAA` → `github.com/Vihaan2012-cmyk/Flight-Sim-Combi-Anti-Aliasing-FSCAA-`.
  - Build: `powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1`, run from Git Bash.
  - If X-Plane is running, set `MV_LAYER_OUT=...\build\vklayer\VkLayer_mv.staged.dll`.
- **Gating:**
  - Everything is gated on `taa.backend=fscaa`.
  - The shader rewrites are also gated per layer by `fscaa.layers`, a bitmask: 1 CMAA2, 2 class
    caps, 4 wire, 8 light, 16 specular, 32 alpha. Default is 63 (all).
  - Rewrites happen at shader-module creation, so bits 4–32 are **latched at load**: they need an X-Plane relaunch.
  - Bits 1–2 are live.
- **Patch safety:** every shader rewrite:
  - (a) matches an exact FNV-1a-64 hash listed in a generated table
  - (b) refuses anything it doesn't fully understand, leaving the module untouched and logging why
  - (c) must pass `spirv-val --target-env vulkan1.3` on every listed variant offline before it may
    ship

  The model is `msaa_patch.h` from 2026-10-12: 293/293 valid.
- **No X-Plane resources modified** except by the shader rewrites. CMAA2 works on a copy.
- **Trace prefix:** `FSCAA:`. **Viz modes:** 13 (classes), 14 (CMAA2 edges), 15 (family tint).
- **Testing pauses the VRAM-only launch rule** (`taa.enable=1`, `taa.backend=fscaa`). Restore
  `taa.enable=0` and `taa.backend=taa` after the final test unless the user says otherwise.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `third_party/cmaa2/*` (create) | vendored Intel CMAA2 + Apache-2.0 licence + reference host code (already downloaded) |
| `src/vklayer/cmaa2_spv.h` (generated) | CMAA2 kernels, including `DebugDrawEdgesCS` |
| `src/vklayer/cmaa2.h` (create) | CMAA2 resources, pipelines, `record()`, `recordDebugEdges()`, stats readback |
| `src/vklayer/fscaa.h` (create) | FSCAA state, settings, stats/trace, glue between `taa.h` and the patches |
| `src/vklayer/fscaa_patch.h` (create) | SPIR-V editor + the four layer transforms (wire, light, specular, alpha) + tint |
| `src/vklayer/fscaa_table.h` (generated) | hash lists per layer, from `tools/gen_fscaa_table.py` |
| `src/vklayer/taa.h`, `src/shaders/taa.comp` (modify) | Phase 1 integration (sub-plan Tasks 3–4) + viz 14 |
| `src/vklayer/layer.cpp` (modify) | call `fscaa::patchModule()` in `TAA_CreateShaderModule`, blend-state fix for `line3d` |
| `tools/cmaa2_test.cpp` (create) | offline GPU test for CMAA2 |
| `tools/gen_fscaa_table.py` (create) | finds and hashes the target variants of each layer |
| `tools/fscaa_patch_test.cpp` (create) | runs every transform over all variants, writes outputs for `spirv-val` |
| `tools/fscaa_validate.sh` (create) | one command: patch all + `spirv-val` all + summary |

## Execution: three stages (user, 2026-10-15)

1. **Build all.** The code of Tasks 1–12 is written and must compile. Within each task, only the
   build-only checks run in this stage.
2. **Static debug all.** Every offline proof runs and every failure is fixed:
   - DXC reflection
   - the `cmaa2_test` GPU test under Khronos validation
   - `fscaa_validate.sh` across all 6,775 variants with `spirv-val`
   - the patch coverage counts
3. **Live debug with the user flying X-Plane.** This is Task 13.

## Task order

1. Tasks 1–4: Phase 1 (CMAA2 + class caps). Proven offline: build, GPU test under validation.
2. Task 5: the debugging toolkit. It's built before the shader layers because they're debugged with it.
3. Tasks 6–7: SPIR-V editor + validator tooling. Proven offline on all 6,775 variants.
4. Tasks 8–11: layers 3–6. Each is proven offline: transform succeeds on 100% of targeted variants, and `spirv-val` passes.
5. Task 12: integration + public-mirror sync.
6. Task 13: **one consolidated test flight.** It covers the class map, family-tint identification
   of power lines, A/B of every layer bit, tuning, and the cost readout.

---

### Tasks 1–4: Phase 1, CMAA2 + class-weighted TAA

Execute `docs/superpowers/plans/2026-10-15-fscaa-phase1.md` Tasks 1–4 exactly as written there, with
two additions:

- [ ] **Addition A (Task 1, Step 3):** compile `DebugDrawEdgesCS` too. Add this to `$cmaaJobs`:

```powershell
    @{ Name='kCmaa2DebugEdges'; Entry='DebugDrawEdgesCS'; Preset=2 },
```

and expect `CMAA2: 8 kernels -> cmaa2_spv.h`.

- [ ] **Addition B (Task 2, Step 3):** in `cmaa2.h`:
  - add `VkPipeline debugPipe` to `State`
  - create it in `create()` from `kCmaa2DebugEdges`, entry `"DebugDrawEdgesCS"`
  - destroy it in `destroy()`
  - add:

```cpp
// Overlay CMAA2's detected edges onto the working colour (Intel's own debug kernel). Call after
// record(); it moves the colour to GENERAL and back to SHADER_READ_ONLY_OPTIMAL.
inline void recordDebugEdges(const Fns &f, State &s, VkCommandBuffer cb)
{
    if (!s.debugPipe) return;
    imageBarrier(f, cb, s.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    f.CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &s.set, 0, nullptr);
    f.CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.debugPipe);
    f.CmdDispatch(cb, (s.w + 15) / 16, (s.h + 15) / 16, 1);
    imageBarrier(f, cb, s.color, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
```

Facts confirmed from Intel's host code (`third_party/cmaa2/reference/vaCMAA2DX12.cpp`):
- Buffer sizes: candidates `w*h/4` uint, items `w*h/2` uint2, location list `(w*h+3)/6` uint,
  control 16 uint, indirect 4 uint.
- Edges texture: `R8_UINT` at `(w+1)/2 × h`. Heads: `R32_UINT` at `(w+1)/2 × (h+1)/2`.
- Dispatch order: edges `((w+27)/28, (h+27)/28)` → args `(2,1,1)` → candidates indirect → args
  `(1,2,1)` → apply indirect.
- The shader resets the heads to `0xFFFFFFFF` itself after use, so they only need clearing once.
- Intel clears the control buffer on first run by dispatching args `(2,1,1)`; our
  `vkCmdFillBuffer` does the same.

Sub-plan Task 2 matches all of this.

Commit after each sub-plan task, as written there.

---

### Task 5: Debugging toolkit (`fscaa.h`)

**Files:**
- Create: `src/vklayer/fscaa.h`
- Modify: `src/vklayer/cmaa2.h` (stats readback), `src/vklayer/taa.h` (calls),
  `src/shaders/taa.comp` (viz 14 is CMAA2's own overlay, no shader change needed)

**Interfaces:**
- Produces:
  - `namespace fscaa { struct Settings; Settings read(); void frameStats(...); void notePatch(const char *layer, bool ok, const char *why); void reportPatches(); }`
  - `cmaa2::readCounters(...)`

- [ ] **Step 1: Settings in one place**

```cpp
// fscaa.h - FSCAA glue: settings, stats, patch accounting. Spec: docs/superpowers/specs/2026-10-15-fscaa-design.md
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>

namespace fscaa {

enum : uint32_t { kCmaa = 1, kCaps = 2, kWire = 4, kLight = 8, kSpecular = 16, kAlpha = 32, kAll = 63 };

struct Settings {
    bool on = false;            // taa.backend == fscaa
    uint32_t layers = kAll;     // fscaa.layers
    int cmaaPreset = 2;         // fscaa.cmaa_preset 0..3
    float nearM = 80.0f, keepNear = 0.5f, keepScreen = 0.0f, keepLight = 1.0f;
    bool debugEdges = false;    // fscaa.debug_edges: CMAA2 edge overlay (viz 14 equivalent, live)
    int statsEvery = 600;       // fscaa.stats_every (frames, 0 = off)
    float wireMinPx = 1.0f;     // fscaa.wire_min_px   (latched at load)
    float lightMinPx = 1.5f;    // fscaa.light_min_px  (latched at load)
    float specKappa = 0.18f;    // fscaa.spec_kappa    (latched at load)
    float specSigma = 0.25f;    // fscaa.spec_sigma    (latched at load)
    float alphaMipScale = 0.25f;// fscaa.alpha_mip_scale (latched at load)
    std::string tintFamily;     // fscaa.debug_tint: magenta-tint one shader family (latched at load)
};

} // namespace fscaa
```

`read()` fills these with `live::lookup` / `live::f` / `live::i`, using the key names in the
comments. Latched values are read once, at the first shader module, with `live::loadNow()` first,
which is the pattern from `msaaPerPixelOn()` on 2026-10-12.

- [ ] **Step 2: CMAA2 counters.** In `cmaa2.h`:
  - add a host-visible readback buffer (64 bytes, `TRANSFER_DST`, mapped) to `State`
  - add `recordCounterCopy(f, s, cb)`: `CmdCopyBuffer(control → readback, 64 bytes)` after the apply
    dispatch
  - add `readCounters(s, out[16])`, which memcpys the mapped bytes from the previous frame

  Then `fscaa::frameStats` traces, every `statsEvery` frames:

```
FSCAA STATS: frame N | layers 0x3f | CMAA2 preset 2 control[0..3]=a,b,c,d | caps near 0.50 screen 0.00 light 1.00 near_m 80 | cmaa ms X.XX (taa.gpu_timing=1)
```

  The first four control words are Intel's counters: shape candidates, blend items, and so on.
  These answer "is CMAA2 doing anything", which is the HDR-threshold risk.

- [ ] **Step 3: GPU timing.** In `taa.h`, wrap the CMAA2 record in the existing
  `gpuTimeMark(dd, cb, "cmaa2")` before it and `gpuTimeMark(dd, cb, "cmaa2_end")` after it. The
  `PERF GPU` line then shows the span when `taa.gpu_timing=1`.

- [ ] **Step 4: Live CMAA2 edge overlay.** In `taa.h`, after `cmaa2::record(...)` succeeds:

```cpp
        if (live::onoff("fscaa.debug_edges", nullptr, false)) cmaa2::recordDebugEdges(g_cmaaFns, g_cmaa);
```

  Edges then draw straight into the frame. This is viz mode 14, implemented as a live key.

- [ ] **Step 5: Skip and failure reasons, once each.** Every early-out in the FSCAA path traces its
  reason once per distinct reason, using a `static std::set<std::string>`. The reasons:
  - `not fscaa backend`
  - `VR layers`
  - `CMAA2 entry points missing`
  - `resources failed WxH`
  - `record refused`
  - `resolution changed WxH -> WxH`

- [ ] **Step 6: Patch accounting.** `notePatch(layer, ok, why)` keeps counts per layer and
  `reportPatches()` traces them at the first present and every 2,000 modules:

```
FSCAA PATCH: wire 5/5 ok | light 7/7 ok | specular 247/247 ok | alpha 4180/4236 ok, 56 refused (reasons: no-cutoff-const 40, multi-kill 16)
```

  Then a lookup of the patch report says exactly what is and isn't covered.

- [ ] **Step 7: Build, then commit**

```bash
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"
MV_LAYER_OUT='D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors\build\vklayer\VkLayer_mv.staged.dll' powershell.exe -NoProfile -ExecutionPolicy Bypass -File build.ps1 > "$TEMP/fscaa_t5.log" 2>&1; echo "exit $?"
grep -nE "(fscaa|cmaa2)\.h:[0-9]+:[0-9]+: (error|warning)" "$TEMP/fscaa_t5.log" | head
git add src/vklayer/fscaa.h src/vklayer/cmaa2.h src/vklayer/taa.h && git commit -m "FSCAA: debugging toolkit - stats, counters, GPU timing, edge overlay, skip reasons, patch accounting

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: exit 0, and no errors or warnings in `fscaa.h` or `cmaa2.h`.

---

### Task 6: SPIR-V editor (`fscaa_patch.h`, core) + family tint

**Files:**
- Create: `src/vklayer/fscaa_patch.h`

**Interfaces:**
- Produces:
  - `namespace fscaapatch { struct Module; bool parse(const uint32_t*, size_t, Module&); std::vector<uint32_t> emit(const Module&); }`
  - Helpers:
    - `uint32_t newId(Module&)`
    - `uint32_t constF32(Module&, float)`
    - `uint32_t typeF32(Module&)`
    - `uint32_t glslExt(Module&)` (imports `GLSL.std.450` if absent)
    - `int findMemberByName(const Module&, const char *name, uint32_t &structId, uint32_t &memberIndex)`
    - `void insertAfter(Module&, size_t instIndex, std::vector<uint32_t> words)`
    - `void replaceUses(Module&, uint32_t oldId, uint32_t newId, size_t fromInst)`
  - `enum Result { OK, NOT_SPIRV, NOT_FOUND, AMBIGUOUS, UNSUPPORTED };` with `const char *name(Result)`
  - `Result tintMagenta(const Module&, std::vector<uint32_t>&)`

- [ ] **Step 1: The module model**

```cpp
// fscaa_patch.h - load-time SPIR-V rewrites of X-Plane's shaders for FSCAA layers 3-6, and a debug
// tint. Same rules as msaa_patch.h (2026-10-12): exact-hash targets only, refuse what is not fully
// understood, every output spirv-val'd offline (tools/fscaa_validate.sh) before it ships.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace fscaapatch {

struct Inst { uint32_t op; std::vector<uint32_t> w; };   // w[0] = header word, w[1..] = operands
struct Module { uint32_t header[5]; std::vector<Inst> ins; };

enum Result { OK = 0, NOT_SPIRV, NOT_FOUND, AMBIGUOUS, UNSUPPORTED };
inline const char *name(Result r)
{
    switch (r) { case OK: return "ok"; case NOT_SPIRV: return "not SPIR-V"; case NOT_FOUND: return "pattern not found";
                 case AMBIGUOUS: return "pattern ambiguous"; case UNSUPPORTED: return "unsupported shape"; }
    return "?";
}

inline bool parse(const uint32_t *c, size_t n, Module &m)
{
    if (!c || n < 5 || c[0] != 0x07230203u) return false;
    memcpy(m.header, c, sizeof(m.header));
    m.ins.clear();
    for (size_t i = 5; i < n;) {
        uint32_t op = c[i] & 0xffffu, wc = c[i] >> 16;
        if (!wc || i + wc > n) return false;
        Inst in; in.op = op; in.w.assign(c + i, c + i + wc);
        m.ins.push_back(in); i += wc;
    }
    return true;
}

inline std::vector<uint32_t> emit(const Module &m)
{
    std::vector<uint32_t> out(m.header, m.header + 5);
    for (size_t i = 0; i < m.ins.size(); ++i) {
        std::vector<uint32_t> w = m.ins[i].w;
        w[0] = ((uint32_t)w.size() << 16) | m.ins[i].op;
        out.insert(out.end(), w.begin(), w.end());
    }
    return out;
}

inline uint32_t newId(Module &m) { return m.header[3]++; }

inline std::string str(const std::vector<uint32_t> &w, size_t from)
{
    std::string s;
    for (size_t k = from; k < w.size(); ++k)
        for (int b = 0; b < 4; ++b) { char ch = (char)((w[k] >> (8 * b)) & 0xff); if (!ch) return s; s += ch; }
    return s;
}

// Index of the first instruction of the "types/constants/globals" section's end, i.e. the first
// OpFunction (54). New types/constants are inserted just before it.
inline size_t firstFunction(const Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == 54) return i;
    return m.ins.size();
}

inline uint32_t typeF32(Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == 22 && m.ins[i].w.size() >= 3 && m.ins[i].w[2] == 32) return m.ins[i].w[1];   // OpTypeFloat 32
    uint32_t id = newId(m);
    Inst t; t.op = 22; t.w = {0, id, 32};
    m.ins.insert(m.ins.begin() + firstFunction(m), t);
    return id;
}

inline uint32_t constF32(Module &m, float v)
{
    const uint32_t ty = typeF32(m);
    uint32_t bits; memcpy(&bits, &v, 4);
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == 43 && m.ins[i].w.size() == 4 && m.ins[i].w[1] == ty && m.ins[i].w[3] == bits) return m.ins[i].w[2];
    uint32_t id = newId(m);
    Inst c; c.op = 43; c.w = {0, ty, id, bits};
    m.ins.insert(m.ins.begin() + firstFunction(m), c);
    return id;
}

inline uint32_t glslExt(Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == 11 && str(m.ins[i].w, 2) == "GLSL.std.450") return m.ins[i].w[1];   // OpExtInstImport
    uint32_t id = newId(m);
    Inst e; e.op = 11; e.w = {0, id};
    const char *s = "GLSL.std.450"; uint32_t word = 0; int b = 0;
    for (const char *p = s;; ++p) { word |= (uint32_t)(unsigned char)*p << (8 * b); if (++b == 4 || !*p) { e.w.push_back(word); word = 0; b = 0; if (!*p) break; } }
    if (b) e.w.push_back(word);
    // after the last OpExtension (10) / OpCapability (17), before OpMemoryModel (14)
    size_t at = 0;
    for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == 14) { at = i; break; }
    m.ins.insert(m.ins.begin() + at, e);
    return id;
}

// OpMemberName (6): struct id, member index, name.
inline int findMemberByName(const Module &m, const char *nm, uint32_t &structId, uint32_t &member)
{
    int found = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == 6 && str(m.ins[i].w, 3) == nm) { structId = m.ins[i].w[1]; member = m.ins[i].w[2]; ++found; }
    return found;
}

inline void insertAfter(Module &m, size_t at, const Inst &in) { m.ins.insert(m.ins.begin() + at + 1, in); }

// Replace operand uses of oldId with newId in instructions from index `from` on (not result ids).
// Result ids sit at w[2] for instructions that have a result type, w[1] for type declarations;
// replacing only from `from` onwards and skipping the defining instruction keeps it correct for
// the straight-line function-body uses these transforms create.
inline void replaceUses(Module &m, uint32_t oldId, uint32_t newId_, size_t from, size_t skip)
{
    for (size_t i = from; i < m.ins.size(); ++i) {
        if (i == skip) continue;
        std::vector<uint32_t> &w = m.ins[i].w;
        for (size_t k = 1; k < w.size(); ++k) if (w[k] == oldId) w[k] = newId_;
    }
}

} // namespace fscaapatch
```

- [ ] **Step 2: Family tint (debug).** In the same file, add `tintMagenta`. For a fragment shader,
  every `OpStore` (62) whose pointer is an `Output` variable at `Location 0` (from `OpVariable`
  storage class 3 and `OpDecorate ... Location 0`) gets its value replaced by a new
  `OpCompositeConstruct` of `(1, 0, 1, a)`. `a` is extracted from the original value via
  `OpCompositeExtract index 3` when the type is a vec4, and is 1.0 otherwise. Any `Location 0`
  output that isn't a 4-float vector returns `UNSUPPORTED`. With `fscaa.debug_tint=<family>`,
  modules whose hash belongs to that family get this rewrite. That turns one family magenta on
  screen, which is how Task 13 confirms that `line3d` is what draws power lines.

- [ ] **Step 3: Commit** after Task 7's validator passes on the tint transform.

---

### Task 7: Target tables + offline validator

**Files:**
- Create: `tools/gen_fscaa_table.py`, `tools/fscaa_patch_test.cpp`, `tools/fscaa_validate.sh`
- Generated: `src/vklayer/fscaa_table.h`

**Interfaces:**
- Produces:
  - `fscaa_table.h` with sorted `kFscaaWire[]`, `kFscaaLight[]`, `kFscaaSpecular[]`, `kFscaaAlpha[]`
    (FNV-1a-64 over 32-bit words, same as `mv_family_table.h`), each with a `...Count`
  - `kFscaaFamily[]` pairs `{hash, familyIndex}` + `kFscaaFamilyNames[]` for the tint
  - `fscaa_patch_test <layer> <outdir> <spv...>`, which prints `OK` / `REFUSED <reason>` per file
    and a summary

- [ ] **Step 1: `gen_fscaa_table.py`.** Packs are the `spv/` directory and `spv.zip`, as in
  `tools/gen_family_table.py`. Targets:
  - **wire:** family `line3d`, vertex stage.
  - **light:** families `light` and `ground_lights`, vertex stage.
  - **specular:** family `deferred_gbuf`, fragment stage, variants that fetch `gbuffer_nrm`. Find
    the variable by `OpName` and a load through `OpImageFetch`.
  - **alpha:** family `terrain`, fragment stage, variants containing `OpKill` (252) or
    `OpTerminateInvocation` (4416). Measured 2026-10-15: 4,236 of `terrain`'s variants.
  - **family map:** every variant of every family, for the tint.

  Output goes to `src/vklayer/fscaa_table.h`, with a header naming the packs and the date.

- [ ] **Step 2: `fscaa_patch_test.cpp`.**
  - Reads each `.spv` and runs the transform for `<layer>`: `tint`, `wire`, `light`, `specular` or
    `alpha`, from `fscaa_patch.h`.
  - Writes `OK` outputs to `<outdir>`.
  - Counts results by reason.
  - Exits non-zero if any output isn't parseable SPIR-V.

  The model is `tools/msaa_patch_test.cpp` from 2026-10-12; it's in the session scratchpad copy.

- [ ] **Step 3: `fscaa_validate.sh`**

```bash
#!/usr/bin/env bash
# fscaa_validate.sh <extracted spv dir> - run every FSCAA transform over its target variants, spirv-val every output.
set -euo pipefail
SRC="$1"; OUT="${TEMP:-/tmp}/fscaa_val"; V=/c/VulkanSDK/1.4.357.0/Bin/spirv-val.exe
g++ -O2 -std=c++17 tools/fscaa_patch_test.cpp -o "$OUT.exe" -I src/vklayer
for layer in tint:line3d wire:line3d light:light light:ground_lights specular:deferred_gbuf alpha:terrain; do
  L=${layer%%:*}; F=${layer##*:}; rm -rf "$OUT/$L-$F"; mkdir -p "$OUT/$L-$F"
  find "$SRC/$F" -name '*.spv' -print0 | xargs -0 "$OUT.exe" "$L" "$OUT/$L-$F" > "$OUT/$L-$F.log" || true
  ok=0; bad=0; for f in "$OUT/$L-$F"/*.spv; do [ -e "$f" ] || continue; if "$V" --target-env vulkan1.3 "$f" >/dev/null 2>&1; then ok=$((ok+1)); else bad=$((bad+1)); fi; done
  echo "$L on $F: $(grep -c '^OK' "$OUT/$L-$F.log") patched, $(grep -c '^REFUSED' "$OUT/$L-$F.log") refused, spirv-val $ok valid / $bad INVALID"
done
```

- [ ] **Step 4: Run it** (the tint transform only exists so far). Extract the packs first:

```bash
S="$TEMP/fscaa_xpspv"; mkdir -p "$S"; cd "/d/Steam Games/steamapps/common/X-Plane 12/Resources/shaders/bin/spv"; for f in *.xsa; do mkdir -p "$S/${f%.xsa}"; unzip -q -o "$f" -d "$S/${f%.xsa}"; done
cd "/d/Steam Games/steamapps/common/X-Plane 12/MotionVectors"; python -I tools/gen_fscaa_table.py ../Resources/shaders/bin/spv ../Resources/shaders/bin/spv.zip; bash tools/fscaa_validate.sh "$S" 2>&1 | grep '^tint'
```

  Expected: `tint on line3d: 5 patched, 0 refused, spirv-val 5 valid / 0 INVALID`. Some vertex
  variants may be counted as "not fragment" refusals; that's fine.

- [ ] **Step 5: Commit** `fscaa_patch.h`, the tools, and `fscaa_table.h`.

---

### Task 8: Layer 3, wire AA on `line3d`

**Files:**
- Modify: `src/vklayer/fscaa_patch.h` (add `wireVertex`)
- Modify: `src/vklayer/layer.cpp` (blend state for `line3d` pipelines)

**What X-Plane does (measured 2026-10-15, `line3d_0.spv` vertex):** the line is expanded in screen
space:

```
gl_Position = p0 + vec4((perp(dir) * (vec2(u_line_width) / (u_screen_size.xy / w))) / u_screen_size.zw * ...)
```

`u_line_width` is a uniform block member, in pixels. Sub-pixel widths are what break wires into dots.

**Transform (`wireVertex`), in Persson's form:**
1. Find the uniform member named `u_line_width` with `findMemberByName`. Exactly one must exist,
   or return `AMBIGUOUS` / `NOT_FOUND`.
2. Find every `OpAccessChain` (65) whose base is the uniform block variable and whose last index
   constant equals that member index, and the `OpLoad` (61) of each. Call the loaded value `w`.
3. After each such load, insert:
   - `wc = OpExtInst GLSL.std.450 FMax(w, constF32(wireMinPx))`, where FMax is 40
   - `cov = OpFDiv(w, wc)`
4. Replace later uses of `w` with `wc`.
5. Carry `cov` to the fragment shader: add a new `Output` float variable at a free `Location`.
   Use the highest used location + 1 and check it's free in the paired fragment module. Store
   `cov` into it right after it's computed, and add the variable to the `OpEntryPoint` interface.
6. Fragment counterpart `wireFragment`: add the matching `Input` variable at the same location.
   Before each `OpStore` to the `Location 0` output, multiply the stored vector's `.w` by the
   loaded coverage: `OpCompositeExtract 3` → `OpFMul` → `OpCompositeInsert 3`.
7. Refusal cases:
   - more than one `u_line_width` member
   - no `Location 0` vec4 output
   - a store to `Location 0` through an `OpAccessChain`

**Blend:** coverage only shows if the pipeline blends. In `TAA_CreateGraphicsPipelines`, for
pipelines whose fragment module is a patched wire module:
- if attachment 0's `blendEnable` is false, set it to true
- colour: `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`, `ADD`
- alpha: `ONE`, `ONE_MINUS_SRC_ALPHA`
- trace it once: `FSCAA: wire pipeline blend enabled (was off)`

If `pColorBlendState` is null, skip it and trace that.

- [ ] **Step 1: Implement the vertex and fragment transforms** as specified above. Each builds new
  instructions with `newId` / `constF32` / `glslExt` and the `Inst` struct, inserts them with
  `insertAfter`, and rewires uses with `replaceUses`.
- [ ] **Step 2: Validate**

```bash
bash tools/fscaa_validate.sh "$TEMP/fscaa_xpspv" | grep '^wire'
```

  Expected: `wire on line3d: N patched, 0 refused, spirv-val N valid / 0 INVALID`, where N is the
  number of `line3d` vertex and fragment variants.
- [ ] **Step 3: Hook it** in `layer.cpp`'s `TAA_CreateShaderModule`, next to the existing FSR
  substitution. If `fscaa` is on, `wire` is in `layers`, and the module's hash is in
  `kFscaaWire`, substitute the patched code and call `fscaa::notePatch("wire", ok, why)`. Then add
  the blend change in `TAA_CreateGraphicsPipelines`.
- [ ] **Step 4: Build, then commit.**

---

### Task 9: Layer 4, light-point AA on `light` / `ground_lights`

**Files:**
- Modify: `src/vklayer/fscaa_patch.h` (add `lightVertex`, `lightFragment`)
- Modify: `src/vklayer/layer.cpp` (hook)

**What X-Plane does (measured 2026-10-15):**
- `light` (5 variants) sizes sprites from `u_light_scale` and `u_pixel_size` (uniform members).
- `ground_lights` (2 variants) uses `u_imm_a_point_size`.
- A light smaller than a pixel lands on a pixel or misses it frame to frame, which is the flicker.

**Transform:**

1. **Find the size.** Locate the loaded member value:
   - `u_pixel_size` (vec2) in `light`
   - `u_imm_a_point_size` (vec4: use `.x`) in `ground_lights`
2. **Clamp it.** Clamp the effective on-screen radius to at least `lightMinPx`:
   - `light`: multiply the pixel-size vector by `s = max(1, lightMinPx / r)`, where `r` is the
     radius in pixels the shader already computes from it.

     Before coding, find the instruction where the size is turned into a clip-space corner offset.
     Use `spirv-cross` on `light_0..4` and write the matched operation (opcode + operands) into the
     plan's notes, in this task.

     If the offset isn't a single `OpFMul` by a value derived from `u_pixel_size`, return
     `UNSUPPORTED` and record the shape.
   - `ground_lights`: `point_size' = max(point_size, lightMinPx)`.
3. **Fade.** Carry `fade = (r / r')²` to the fragment (energy-conserving: total light stays the
   same), using the same varying technique as Task 8 step 5. Multiply the output colour, all four
   channels, since lights are additive, by `fade`.
4. **Validate:**

```bash
bash tools/fscaa_validate.sh "$TEMP/fscaa_xpspv" | grep '^light'
```

   Expected: 0 INVALID. Any refusal must name the shape found.
5. **Hook** as in Task 8 step 3, with `kFscaaLight`. No blend change: lights already blend
   additively. Commit.

---

### Task 10: Layer 5, geometric specular AA in `deferred_gbuf`

**Files:**
- Modify: `src/vklayer/fscaa_patch.h` (add `specular`)
- Modify: `src/vklayer/layer.cpp` (hook)

**What X-Plane does (measured 2026-10-15, `deferred_gbuf_1.spv`):** it fetches
`gbuffer_nrm` (`.xy`, spheremap) and decodes it to a vec3 normal:

```
N = vec3(n * sqrt(max(0, 1 - dot(n,n)*0.25)), 1 - dot(n,n)*0.5)
```

`gbuffer_mat` is decoded as packed fields: `.y*2`, `.z*2-1`, and `.x` (coverage).

**Transform (Tokuyoshi & Kaplanyan, as implemented in Unity HDRP):**

1. **Decode first.** Find the roughness that feeds the specular lobe. Trace from the specular
   term back to its `gbuffer_mat` / `gbuffer_tex` source with `spirv-cross` on 3 variants,
   including one per-sample. The term is recognisable as the `OpExtInst Pow`, or as
   `a² / (π (…)²)`. Write the found value's derivation into this task's notes.
   - **Done when:** one named SPIR-V value (call it `R`, roughness, or `A`, `alpha = R²`) is
     identified, consistently across the sampled variants.
   - **Otherwise:** if it isn't consistent, the transform returns `UNSUPPORTED` for variants where
     the pattern doesn't match, and the coverage count says how many are covered.
2. **Normal variance.** After the decoded normal `N` exists:
   - `dx = OpDPdx(N)`, `dy = OpDPdy(N)` (opcodes 207 and 208)
   - `var = specSigma * (dot(dx,dx) + dot(dy,dy))`
   - `A' = A + min(2*var, specKappa²)`
3. **Replace uses.** Replace later uses of `A` with `A'`. If the identified value is `R` rather than
   `A`, apply `R' = sqrt(R² + min(2*var, κ²))`.
4. **Per-sample variants** (`SampleRateShading`) run per sample, where derivatives are still per
   2×2 quad, so the same formula holds.
5. **Validate:**

```bash
bash tools/fscaa_validate.sh "$TEMP/fscaa_xpspv" | grep '^specular'
```

   Expected: 0 INVALID. Record the covered count.
6. **Hook** with `kFscaaSpecular`. Commit.

---

### Task 11: Layer 6, anti-aliased alpha test in alpha-tested `terrain`

**Files:**
- Modify: `src/vklayer/fscaa_patch.h` (add `alpha`)
- Modify: `src/vklayer/layer.cpp` (hook)

**What X-Plane does (measured 2026-10-15):** 4,236 `terrain` fragment variants contain `OpKill`.
This is the material shader for objects, foliage and fences: `if (a < cutoff) discard`.

**Transform (Golus):**

1. **Find the test.** Find each `OpKill` whose block is reached by an `OpBranchConditional` (250)
   on `c = OpFOrdLessThan/OpFUnordLessThan(a, k)` (opcodes 184/185), where `k` is a constant or a
   uniform member load.
   - Exactly one such test per variant is required. Otherwise return `AMBIGUOUS` or `NOT_FOUND`.
2. **Mip-coverage preservation** (stops foliage thinning and vanishing at distance). Before the
   compare, rewrite `a`:

```
a' = a * (1 + max(0, lod) * alphaMipScale)
```

   `lod` comes from `OpImageQueryLod` (105) on the same sampled image and coordinate that produced
   `a`. Find the sample instruction whose result `.w` (via `OpCompositeExtract 3`) is `a`.
   - If `a` doesn't come from a texture sample this way, skip this step (record
     `no-texture-alpha`) and do only step 3.
3. **Sharpening** (stable edge across the TAA jitter):

```
a'' = (a' - k) / max(fwidth(a'), 1e-4) + 0.5
c' = a'' < 0.5
```

   `fwidth` is `OpFwidth` (209).
4. **Rewire** the branch condition to `c'`.
5. **Requirement:** `ImageQuery` capability for `OpImageQueryLod`. Add `OpCapability ImageQuery`
   (capability 50) if absent.
6. **Validate:**

```bash
bash tools/fscaa_validate.sh "$TEMP/fscaa_xpspv" | grep '^alpha'
```

   Expected: 0 INVALID, with the patched and refused counts and reasons.
7. **Hook** with `kFscaaAlpha`. Commit.

---

### Task 12: Integration, packaging, public mirror

**Files:**
- Modify: `config/taa_live.ini` (FSCAA keys, commented, `taa.backend` unchanged)
- Modify: `D:\FSCAA` (mirror)

- [ ] **Step 1:** Add the FSCAA block to the packaged `config/taa_live.ini`. Every `fscaa.*` key
  gets its default and a one-line comment; `taa.backend` stays `taa`. Run the whole validator,
  then a full build.

```bash
bash tools/fscaa_validate.sh "$TEMP/fscaa_xpspv"   # every line: 0 INVALID
```

- [ ] **Step 2: Offline GPU test.** Run `tools/cmaa2_test.cpp` under validation again. Expect `PASS`.
- [ ] **Step 3: Mirror** to `D:\FSCAA`. Copy:
  - `src/vklayer/{cmaa2.h,fscaa.h,fscaa_patch.h,fscaa_table.h}` → `src/`
  - `tools/{cmaa2_test.cpp,gen_fscaa_table.py,fscaa_patch_test.cpp,fscaa_validate.sh}` → `tools/`
  - `third_party/cmaa2` → `third_party/cmaa2`
  - this plan → `docs/plans/`
  - the `taa.h` / `taa.comp` / `layer.cpp` integration → `integration/motionvectors.patch`, from
    `git diff <base>..HEAD -- src/vklayer/taa.h src/shaders/taa.comp src/vklayer/layer.cpp`

  Update the README's key table with the final keys. Then commit and push.

---

### Task 13: The one test flight (user flies; X-Plane closed for install)

- [ ] **Step 1: Install.** Install the staged DLL, keeping `VkLayer_mv.pre-fscaa.dll` as a backup.
  In `%TEMP%\taa_live.ini` set:
  - `taa.enable=1`
  - `taa.backend=fscaa`
  - `fscaa.layers=63`
  - `fscaa.stats_every=600`
  - `taa.gpu_timing=1`
  - `taa.fg=0`
- [ ] **Step 2: Load trace.**

```bash
t=$(ls -t "$LOCALAPPDATA/Temp"/taa_layer_*.txt | head -1); grep -E "^FSCAA" "$t" | head -20
```

  Expected:
  - `FSCAA: CMAA2 ready`
  - an `FSCAA PATCH:` line with every layer's counts (zero refusals beyond those recorded offline)
  - `FSCAA STATS` lines with non-zero CMAA2 counters
- [ ] **Step 3: Class map.** `taa.viz=13`: screens blue, near red, scenery green, lights yellow.
  `fscaa.debug_edges=1` shows CMAA2's edges. Turn both off afterwards.
- [ ] **Step 4: Power-line check, using the tint.** This is a separate short launch with
  `fscaa.debug_tint=line3d`, looking at a power line:
  - **Magenta:** layer 3 targets the right shader.
  - **Not magenta:** set `fscaa.debug_tint=terrain`. If the wires turn magenta, power lines are
    ordinary 3-D geometry, so layer 3 doesn't apply to them. Record that in the spec; wires are
    then covered by layers 1 and 2 only.
- [ ] **Step 5: A/B, about 30 s each,** at a spot with power lines, lights (dusk), water and trees.
  The live bits switch in flight; the latched bits need a relaunch.
  - `fscaa.layers=3` (CMAA2 + caps): live
  - then 63 (all six): relaunch
  - then `taa.backend=taa`: live

  The user judges. PERF and PERF GPU give the cost of each segment.
- [ ] **Step 6: Tune live:**
  - `fscaa.keep_near`
  - `fscaa.near_m`
  - `fscaa.cmaa_preset`

  Then in one more relaunch if needed:
  - `fscaa.wire_min_px`
  - `fscaa.light_min_px`
  - `fscaa.spec_kappa`
  - `fscaa.alpha_mip_scale`

  Bake the final values in as the defaults (in `fscaa::read()` and the packaged ini), then commit
  and mirror.
- [ ] **Step 7: Close out.**
  1. Restore `taa.enable=0` and `taa.backend=taa` unless the user keeps FSCAA.
  2. Write memory `fscaa.md` (cost, defaults, coverage counts, what the user saw), plus its
     `MEMORY.md` line.
  3. Update the spec's status.

---

## Self-review notes

- **Spec coverage:**
  - layer 1: Tasks 1–3
  - layer 2: Task 4
  - layer 3: Task 8
  - layer 4: Task 9
  - layer 5: Task 10
  - layer 6: Task 11
  - debugging: Tasks 5–7
  - cost and testing: Tasks 12–13
  - risks: HDR threshold (stats + live preset), image formats (sub-plan Task 1 Step 2), VRAM rule
    (Global Constraints)
- **Measured facts used** (all 2026-10-15, X-Plane 12.4.4 shader pack):
  - `line3d` uses `u_line_width`
  - `light` uses `u_pixel_size` / `u_light_scale`, and `ground_lights` uses `u_imm_a_point_size`
  - `deferred_gbuf` normal decode and `gbuffer_mat` packing
  - 4,236 `terrain` variants contain `OpKill`
  - Intel's CMAA2 buffer sizes and dispatch order
- **Decode-first gates:**
  - **Tasks 9 and 10** each start by reading the decompiled shader to locate one value: the
    clip-offset multiply, and the roughness. Each has a defined outcome when the pattern doesn't
    hold: refuse and count.
  - **Task 13 Step 4** decides wire coverage from what the tint shows on screen.

  In each case the code for both outcomes is specified, so no step is open-ended.
