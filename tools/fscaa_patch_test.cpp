// fscaa_patch_test.cpp - run one fscaa_patch.h transform over X-Plane SPIR-V files offline.
//
//   fscaa_patch_test <tint|wire|light|specular|alpha> <outdir> <file.spv> [...]
//
// Prints OK / REFUSED <reason> per file and a tally; OK outputs go to <outdir>/<name>.spv for
// spirv-val (tools/fscaa_validate.sh). Exits 1 if an OK output does not re-parse as SPIR-V.
// Parameters are the FSCAA defaults (fscaa.h).
// SPDX-License-Identifier: GPL-3.0-or-later
#include "fscaa_patch.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static bool readAll(const char *p, std::vector<uint32_t> &w)
{
    FILE *f = fopen(p, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    w.resize((size_t)n / 4);
    const bool ok = fread(w.data(), 4, w.size(), f) == w.size();
    fclose(f);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: fscaa_patch_test <layer> <outdir> <spv>...\n"); return 2; }
    const std::string layer = argv[1];
    std::map<std::string, int> tally;
    int bad = 0;
    for (int a = 3; a < argc; ++a) {
        std::vector<uint32_t> in, out;
        if (!readAll(argv[a], in)) { fprintf(stderr, "cannot read %s\n", argv[a]); ++bad; continue; }
        fscaapatch::Result r;
        if (layer == "tint") r = fscaapatch::tint(in.data(), in.size(), out);
        else if (layer == "wire") r = fscaapatch::wire(in.data(), in.size(), out, 1.0f);
        else if (layer == "light") r = fscaapatch::light(in.data(), in.size(), out, 1.5f);
        else if (layer == "specular") r = fscaapatch::specular(in.data(), in.size(), out, 0.25f, 0.18f);
        else if (layer == "alpha") r = fscaapatch::alpha(in.data(), in.size(), out, 1.0f);
        else { fprintf(stderr, "unknown layer %s\n", layer.c_str()); return 2; }
        std::string base = argv[a];
        const size_t s = base.find_last_of("/\\");
        if (s != std::string::npos) base = base.substr(s + 1);
        ++tally[fscaapatch::name(r)];
        if (r != fscaapatch::OK) { printf("REFUSED %s: %s\n", base.c_str(), fscaapatch::name(r)); continue; }
        fscaapatch::Module chk;
        if (!fscaapatch::parse(out.data(), out.size(), chk)) { printf("BROKEN %s: output does not parse\n", base.c_str()); ++bad; continue; }
        const std::string op = std::string(argv[2]) + "/" + base;
        FILE *f = fopen(op.c_str(), "wb");
        if (!f) { ++bad; continue; }
        fwrite(out.data(), 4, out.size(), f);
        fclose(f);
        printf("OK %s: %zu -> %zu words\n", base.c_str(), in.size(), out.size());
    }
    for (std::map<std::string, int>::iterator it = tally.begin(); it != tally.end(); ++it)
        printf("result %-24s %d\n", it->first.c_str(), it->second);
    return bad ? 1 : 0;
}
