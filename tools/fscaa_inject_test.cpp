// fscaa_inject_test.cpp - FSCAA's rewritten shaders must still take MotionVectors' own injection.
// With TAA on, the layer re-patches X-Plane's shaders at pipeline creation (spirv_inject.h: vertex
// prevClip varying, fragment velocity write) starting from the stored module code - which for an
// FSCAA-rewritten module is the rewrite. This runs that injection over FSCAA's outputs and over the
// matching X-Plane originals, and writes the injected FSCAA modules for spirv-val.
//
//   fscaa_inject_test <outdir> <original dir> <fscaa output .spv> [...]
// For each FSCAA output named X.spv, <original dir>/X.spv is the unpatched module. Prints per file
// whether injection accepts original / FSCAA version; a module the injector takes in its original
// form but refuses after FSCAA is a regression (exit 1).
// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdarg>
static void trace(const char *, ...) {}   // the layer's logger; silent here
#include "spirv_inject.h"
#include "fscaa_patch.h"
#include <cstdio>
#include <string>
#include <vector>

static bool readAll(const std::string &p, std::vector<uint32_t> &w)
{
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    w.resize((size_t)n / 4);
    const bool ok = fread(w.data(), 4, w.size(), f) == w.size();
    fclose(f);
    return ok;
}

static spvinj::Result injectAny(const std::vector<uint32_t> &code, std::vector<uint32_t> &out, bool &frag)
{
    uint32_t loc = 0;
    spvinj::Result r = spvinj::inject(code.data(), code.size() * 4, out, &loc);
    frag = false;
    if (r == spvinj::INJ_NOT_VERTEX) {
        frag = true;
        // Attachment 5: the G-buffer pass has 5 colour targets, so the velocity slot is the 6th.
        r = spvinj::injectFragment(code.data(), code.size() * 4, out, 5, false);
    }
    return r;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: fscaa_inject_test <outdir> <original dir> <fscaa spv>...\n"); return 2; }
    unsigned both = 0, onlyOrig = 0, onlyFscaa = 0, neither = 0;
    for (int a = 3; a < argc; ++a) {
        std::string p = argv[a], base = p.substr(p.find_last_of("/\\") + 1);
        std::vector<uint32_t> fs, orig, o1, o2;
        if (!readAll(p, fs) || !readAll(std::string(argv[2]) + "/" + base, orig)) { printf("SKIP %s (unreadable)\n", base.c_str()); continue; }
        bool f1 = false, f2 = false;
        const spvinj::Result r1 = injectAny(orig, o1, f1), r2 = injectAny(fs, o2, f2);
        const bool ok1 = r1 == spvinj::INJ_OK, ok2 = r2 == spvinj::INJ_OK;
        if (ok1 && ok2) ++both; else if (ok1) ++onlyOrig; else if (ok2) ++onlyFscaa; else ++neither;
        if (ok1 && !ok2) printf("REGRESSION %s: original injects, FSCAA version refused (reason %d)\n", base.c_str(), (int)r2);
        if (ok2) {
            FILE *f = fopen((std::string(argv[1]) + "/" + base).c_str(), "wb");
            if (f) { fwrite(o2.data(), 4, o2.size(), f); fclose(f); }
        }
    }
    printf("injection: both %u | original only %u (REGRESSIONS) | FSCAA only %u | neither %u\n", both, onlyOrig, onlyFscaa, neither);
    return onlyOrig ? 1 : 0;
}
