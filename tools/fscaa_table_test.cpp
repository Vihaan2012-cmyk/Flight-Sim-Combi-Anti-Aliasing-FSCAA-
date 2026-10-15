// fscaa_table_test.cpp - prove the layer's runtime hash (fscaapatch::hashWords, C++) finds X-Plane's
// modules in the tables tools/gen_fscaa_table.py wrote (Python). A mismatch would mean no module is
// ever rewritten in X-Plane, which no transform test can catch.
//
//   fscaa_table_test <file.spv> [...]   (pass every variant of line3d, light, ground_lights,
//                                         deferred_gbuf and terrain)
// Prints, per list, how many of the given files hash into it, and how many listed files the
// transform then accepts. Exits 1 if any list is never hit.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "fscaa_patch.h"
#include "fscaa_table.h"
#include <cstdio>
#include <vector>

static bool inList(const uint64_t *list, size_t n, uint64_t h)
{
    size_t lo = 0, hi = n;
    while (lo < hi) { const size_t mid = (lo + hi) / 2; if (list[mid] < h) lo = mid + 1; else hi = mid; }
    return lo < n && list[lo] == h;
}

int main(int argc, char **argv)
{
    unsigned hit[4] = {}, ok[4] = {}, fam = 0, total = 0;
    const char *names[4] = {"wire", "light", "specular", "alpha"};
    for (int a = 1; a < argc; ++a) {
        FILE *f = fopen(argv[a], "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint32_t> w((size_t)n / 4);
        const size_t got = fread(w.data(), 4, w.size(), f);
        fclose(f);
        if (got != w.size()) continue;
        ++total;
        const uint64_t h = fscaapatch::hashWords(w.data(), w.size());
        std::vector<uint32_t> out;
        size_t lo = 0, hi = kFscaaFamilyCount;
        while (lo < hi) { const size_t mid = (lo + hi) / 2; if (kFscaaFamily[mid].hash < h) lo = mid + 1; else hi = mid; }
        if (lo < kFscaaFamilyCount && kFscaaFamily[lo].hash == h) ++fam;
        if (inList(kFscaaWire, kFscaaWireCount, h)) { ++hit[0]; ok[0] += fscaapatch::wire(w.data(), w.size(), out, 1.0f) == fscaapatch::OK; }
        if (inList(kFscaaLight, kFscaaLightCount, h)) { ++hit[1]; ok[1] += fscaapatch::light(w.data(), w.size(), out, 1.5f) == fscaapatch::OK; }
        if (inList(kFscaaSpecular, kFscaaSpecularCount, h)) { ++hit[2]; ok[2] += fscaapatch::specular(w.data(), w.size(), out, 0.25f, 0.18f) == fscaapatch::OK; }
        if (inList(kFscaaAlpha, kFscaaAlphaCount, h)) { ++hit[3]; ok[3] += fscaapatch::alpha(w.data(), w.size(), out, 1.0f) == fscaapatch::OK; }
    }
    printf("%u files, %u found in the family map\n", total, fam);
    int bad = 0;
    for (int i = 0; i < 4; ++i) {
        printf("%-9s %5u files hash into the list, transform accepts %5u\n", names[i], hit[i], ok[i]);
        if (!hit[i]) bad = 1;
    }
    if (fam != total) { printf("FAIL: %u files missing from the family map\n", total - fam); bad = 1; }
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
