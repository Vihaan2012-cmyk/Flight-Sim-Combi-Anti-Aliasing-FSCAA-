// fscaa.h - FSCAA (Flight Sim Combined Anti-Aliasing) glue: settings, shader-rewrite dispatch,
// patch accounting, stats. Spec: docs/superpowers/specs/2026-10-15-fscaa-design.md;
// plan: docs/superpowers/plans/2026-10-15-fscaa-full.md.
//
//   layer 1  CMAA2 before the resolve              cmaa2.h, taa.h            live
//   layer 2  per-class history caps in the resolve taa.comp (uFscaa)          live
//   layer 3  wire AA        line3d vertex          fscaa_patch.h::wire       latched at load
//   layer 4  light-point AA light vertex           fscaa_patch.h::light      latched at load
//   layer 5  specular AA    deferred_gbuf          fscaa_patch.h::specular   latched at load
//   layer 6  alpha AA       terrain alpha test     fscaa_patch.h::alpha      latched at load
//
// Included by layer.cpp after mv_live.h and trace(); uses both.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "fscaa_patch.h"
#include "fscaa_table.h"

namespace fscaa {

enum : uint32_t { kCmaa = 1, kCaps = 2, kWire = 4, kLight = 8, kSpecular = 16, kAlpha = 32, kAll = 63 };

struct Settings {
    bool on = false;             // taa.backend == fscaa
    uint32_t layers = kAll;      // fscaa.layers bitmask
    int cmaaPreset = 3;          // fscaa.cmaa_preset 0..3 (3 ULTRA: closest to the truth after the sharpen, tools/nn/sharpen_eval.py --cmaa)
    float nearM = 80.0f, keepNear = 1.0f, keepScreen = 0.0f, keepLight = 1.0f;
    bool debugEdges = false;     // fscaa.debug_edges: CMAA2's edge overlay, live
    int statsEvery = 600;        // fscaa.stats_every, frames (0 = off)
};

// Live settings, cheap enough for once a frame.
inline Settings read()
{
    Settings s;
    std::string be;
    s.on = live::lookup("taa.backend", be) && be == "fscaa";
    s.layers = (uint32_t)live::i("fscaa.layers", nullptr, (int)kAll) & kAll;
    s.cmaaPreset = live::i("fscaa.cmaa_preset", nullptr, 3);
    if (s.cmaaPreset < 0) s.cmaaPreset = 0;
    if (s.cmaaPreset > 3) s.cmaaPreset = 3;
    s.nearM = live::f("fscaa.near_m", nullptr, 80.0f);
    s.keepNear = live::f("fscaa.keep_near", nullptr, 1.0f);
    s.keepScreen = live::f("fscaa.keep_screen", nullptr, 0.0f);
    s.keepLight = live::f("fscaa.keep_light", nullptr, 1.0f);
    s.debugEdges = live::onoff("fscaa.debug_edges", nullptr, false);
    s.statsEvery = live::i("fscaa.stats_every", nullptr, 600);
    return s;
}

// Settings that shape shader rewrites, read once at the first shader module (modules are created
// while the scenery loads, before any present - see msaaPerPixelOn's history, 2026-10-12).
struct Latched {
    bool on = false;
    uint32_t layers = kAll;
    float wireMinPx = 1.0f, lightMinPx = 1.5f, specSigma = 0.25f, specKappa = 0.18f, alphaEdge = 1.0f;
    std::string tint;            // fscaa.debug_tint=<family>: draw that family magenta
};

inline const Latched &latched()
{
    static Latched L;
    static std::once_flag once;
    std::call_once(once, [] {
        live::loadNow();
        Settings s = read();
        L.on = s.on;
        L.layers = s.layers;
        L.wireMinPx = live::f("fscaa.wire_min_px", nullptr, 1.0f);
        L.lightMinPx = live::f("fscaa.light_min_px", nullptr, 1.5f);
        L.specSigma = live::f("fscaa.spec_sigma", nullptr, 0.25f);
        L.specKappa = live::f("fscaa.spec_kappa", nullptr, 0.18f);
        L.alphaEdge = live::f("fscaa.alpha_edge", nullptr, 1.0f);
        live::lookup("fscaa.debug_tint", L.tint);
        trace("FSCAA: shader rewrites %s - layers 0x%02x (wire %d, light %d, specular %d, alpha %d), wire_min_px %.2f, "
              "light_min_px %.2f, spec sigma %.2f kappa %.2f, alpha_edge %.2f, CMAA2 after the resolve%s%s",
              L.on ? "ARMED" : "off (taa.backend is not fscaa)", L.layers, !!(L.layers & kWire), !!(L.layers & kLight),
              !!(L.layers & kSpecular), !!(L.layers & kAlpha), L.wireMinPx, L.lightMinPx, L.specSigma, L.specKappa,
              L.alphaEdge, L.tint.empty() ? "" : ", debug tint on family ", L.tint.c_str());
    });
    return L;
}

// ------------------------------------------------------------------ patch accounting
struct Tally { std::atomic<uint32_t> ok{0}, refused{0}; };
inline Tally &tally(int layer) { static Tally t[5]; return t[layer]; }   // 0 wire 1 light 2 specular 3 alpha 4 tint
inline const char *layerName(int layer)
{
    static const char *const n[5] = {"wire", "light", "specular", "alpha", "tint"};
    return n[layer];
}
inline std::mutex &reasonMtx() { static std::mutex m; return m; }
inline std::set<std::string> &reasons() { static std::set<std::string> r; return r; }

inline void notePatch(int layer, fscaapatch::Result r)
{
    if (r == fscaapatch::OK) { ++tally(layer).ok; return; }
    ++tally(layer).refused;
    std::lock_guard<std::mutex> g(reasonMtx());
    const std::string key = std::string(layerName(layer)) + ": " + fscaapatch::name(r);
    if (reasons().insert(key).second) trace("FSCAA PATCH: first refusal of its kind - %s (module left untouched)", key.c_str());
}

inline void reportPatches()
{
    trace("FSCAA PATCH: wire %u ok / %u refused | light %u / %u | specular %u / %u | alpha %u / %u | tint %u / %u",
          tally(0).ok.load(), tally(0).refused.load(), tally(1).ok.load(), tally(1).refused.load(),
          tally(2).ok.load(), tally(2).refused.load(), tally(3).ok.load(), tally(3).refused.load(),
          tally(4).ok.load(), tally(4).refused.load());
}

inline bool inList(const uint64_t *list, size_t n, uint64_t h)
{
    size_t lo = 0, hi = n;
    while (lo < hi) { const size_t mid = (lo + hi) / 2; if (list[mid] < h) lo = mid + 1; else hi = mid; }
    return lo < n && list[lo] == h;
}

inline const char *familyOf(uint64_t h)
{
    size_t lo = 0, hi = kFscaaFamilyCount;
    while (lo < hi) { const size_t mid = (lo + hi) / 2; if (kFscaaFamily[mid].hash < h) lo = mid + 1; else hi = mid; }
    if (lo < kFscaaFamilyCount && kFscaaFamily[lo].hash == h && kFscaaFamily[lo].family < kFscaaFamilyNameCount)
        return kFscaaFamilyNames[kFscaaFamily[lo].family];
    return nullptr;
}

// Called from TAA_CreateShaderModule with X-Plane's original module. Returns the FSCAA layer bit
// whose rewrite is now in `out` (0 = use the module as it is).
inline uint32_t patchModule(const uint32_t *code, size_t words, uint64_t hash, std::vector<uint32_t> &out)
{
    const Latched &L = latched();
    if (!L.on) return 0;
    static std::atomic<uint64_t> seen(0);
    const uint64_t k = ++seen;
    uint32_t bit = 0;
    fscaapatch::Result r = fscaapatch::NOT_FOUND;
    int layer = -1;
    if ((L.layers & kWire) && inList(kFscaaWire, kFscaaWireCount, hash)) {
        layer = 0; bit = kWire; r = fscaapatch::wire(code, words, out, L.wireMinPx);
    } else if ((L.layers & kLight) && inList(kFscaaLight, kFscaaLightCount, hash)) {
        layer = 1; bit = kLight; r = fscaapatch::light(code, words, out, L.lightMinPx);
    } else if ((L.layers & kSpecular) && inList(kFscaaSpecular, kFscaaSpecularCount, hash)) {
        layer = 2; bit = kSpecular; r = fscaapatch::specular(code, words, out, L.specSigma, L.specKappa);
    } else if ((L.layers & kAlpha) && inList(kFscaaAlpha, kFscaaAlphaCount, hash)) {
        layer = 3; bit = kAlpha; r = fscaapatch::alpha(code, words, out, L.alphaEdge);
    }
    if (layer >= 0) {
        notePatch(layer, r);
        if (r != fscaapatch::OK) bit = 0;
    }
    // Debug tint applies on top of (or instead of) a layer rewrite, to the named family only.
    if (!L.tint.empty()) {
        const char *fam = familyOf(hash);
        if (fam && L.tint == fam) {
            std::vector<uint32_t> src = bit ? out : std::vector<uint32_t>(code, code + words), tinted;
            const fscaapatch::Result tr = fscaapatch::tint(src.data(), src.size(), tinted);
            if (tr == fscaapatch::OK) { out.swap(tinted); bit |= 0x80; ++tally(4).ok; }
            else if (tr != fscaapatch::WRONG_STAGE) notePatch(4, tr);
        }
    }
    if (k % 2000 == 0) reportPatches();
    return bit;
}

// ------------------------------------------------------------------ per-frame stats
inline void frameStats(const Settings &s, uint64_t frame, bool cmaaRan, const uint32_t ctr[16], uint64_t cmaaBytes,
                       uint32_t w, uint32_t h)
{
    if (s.statsEvery <= 0 || frame % (uint64_t)s.statsEvery) return;
    trace("FSCAA STATS: frame %llu | layers 0x%02x | CMAA2 %s preset %d at %ux%u, %.1f MB, blend locations last pass %u "
          "(ctl %u %u %u %u) | caps near %.2f screen %.2f light %.2f near_m %.0f | debug_edges %d",
          (unsigned long long)frame, s.layers, cmaaRan ? "ran" : "SKIPPED", s.cmaaPreset, w, h, cmaaBytes / 1048576.0,
          ctr ? ctr[3] : 0u, ctr ? ctr[0] : 0u, ctr ? ctr[1] : 0u, ctr ? ctr[2] : 0u, ctr ? ctr[4] : 0u,
          s.keepNear, s.keepScreen, s.keepLight, s.nearM, s.debugEdges ? 1 : 0);
    static uint64_t lastReport = 0;
    if (frame - lastReport >= (uint64_t)s.statsEvery * 5) { lastReport = frame; reportPatches(); }
}

// Each distinct reason the FSCAA frame path skipped CMAA2, once.
inline void skipOnce(const char *why)
{
    static std::mutex m;
    static std::set<std::string> said;
    std::lock_guard<std::mutex> g(m);
    if (said.insert(why).second) trace("FSCAA: CMAA2 skipped - %s", why);
}

} // namespace fscaa
