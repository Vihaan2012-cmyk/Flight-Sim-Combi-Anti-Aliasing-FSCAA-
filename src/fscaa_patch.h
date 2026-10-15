// fscaa_patch.h - load-time SPIR-V rewrites of X-Plane's own shaders for FSCAA layers 3-6, and a
// debug tint. Same rules as msaa_patch.h (2026-10-12): exact-hash targets only (fscaa_table.h),
// refuse anything not fully understood (the module is then used untouched), and every output is
// spirv-val'd offline over all of X-Plane's variants (tools/fscaa_validate.sh) before it ships.
//
// Shapes these transforms match were read from X-Plane 12.4.4's shader pack on 2026-10-15:
//   wire (line3d vertex)        gl_Position = p + perp * (vec2(u_line_width) / ...) ; v_color = a_color
//   light (light vertex)        gl_Position = P * (eye + (corner - 0.5) * S)        ; v_color per sprite
//   specular (deferred_gbuf)    r = 1 - 0.96875*max(0, 2*mat.y - 1); a = r*r; a2 = a*a (GGX D);
//                               N = second operand of reflect()
//   alpha (terrain fragment)    if (a <= u_alpha_cut.x) discard   (a from a Phi of texture layers)
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace fscaapatch {

// ------------------------------------------------------------------ opcodes used
enum : uint32_t {
    OpName = 5, OpMemberName = 6, OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14,
    OpEntryPoint = 15, OpCapability = 17, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypeImage = 25, OpTypeSampledImage = 27, OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43,
    OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62, OpAccessChain = 65,
    OpDecorate = 71, OpMemberDecorate = 72, OpVectorShuffle = 79, OpCompositeConstruct = 80,
    OpCompositeExtract = 81, OpCompositeInsert = 82, OpSampledImage = 86, OpImageSampleImplicitLod = 87,
    OpImageQueryLod = 105, OpFNegate = 127, OpFAdd = 129, OpFSub = 131, OpFMul = 133, OpFDiv = 136,
    OpVectorTimesScalar = 142, OpMatrixTimesVector = 145, OpDot = 148,
    OpFOrdLessThan = 184, OpFUnordLessThan = 185, OpFOrdLessThanEqual = 188, OpFUnordLessThanEqual = 189,
    OpDPdx = 207, OpDPdy = 208, OpLabel = 248,
    // GLSL.std.450
    GlslFMin = 37, GlslFMax = 40, GlslReflect = 71,
    // enums
    StorageUniform = 2, StorageOutput = 3, DecLocation = 30, DecBuiltIn = 11, BuiltInPosition = 0,
    CapImageQuery = 50, ModelVertex = 0, ModelFragment = 4,
};

enum Result { OK = 0, NOT_SPIRV, WRONG_STAGE, NOT_FOUND, AMBIGUOUS, UNSUPPORTED };
inline const char *name(Result r)
{
    switch (r) {
    case OK: return "ok"; case NOT_SPIRV: return "not SPIR-V"; case WRONG_STAGE: return "wrong stage";
    case NOT_FOUND: return "pattern not found"; case AMBIGUOUS: return "pattern ambiguous"; case UNSUPPORTED: return "unsupported shape";
    }
    return "?";
}

// ------------------------------------------------------------------ module model
struct Inst { uint32_t op; std::vector<uint32_t> w; };   // w[0] = header word (rewritten on emit)
struct Module { uint32_t header[5]; std::vector<Inst> ins; };

inline bool parse(const uint32_t *c, size_t n, Module &m)
{
    if (!c || n < 5 || c[0] != 0x07230203u) return false;
    memcpy(m.header, c, sizeof(m.header));
    m.ins.clear();
    for (size_t i = 5; i < n;) {
        const uint32_t op = c[i] & 0xffffu, wc = c[i] >> 16;
        if (!wc || i + wc > n) return false;
        Inst in; in.op = op; in.w.assign(c + i, c + i + wc);
        m.ins.push_back(in);
        i += wc;
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

inline uint64_t hashWords(const uint32_t *code, size_t words)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < words; ++i) { h ^= (uint64_t)code[i]; h *= 1099511628211ull; }
    return h;
}

inline uint32_t newId(Module &m) { return m.header[3]++; }

inline Inst mk(uint32_t op, std::initializer_list<uint32_t> operands)
{
    Inst in; in.op = op; in.w.push_back(0); in.w.insert(in.w.end(), operands.begin(), operands.end()); return in;
}

inline std::string str(const std::vector<uint32_t> &w, size_t from)
{
    std::string s;
    for (size_t k = from; k < w.size(); ++k)
        for (int b = 0; b < 4; ++b) { const char ch = (char)((w[k] >> (8 * b)) & 0xff); if (!ch) return s; s += ch; }
    return s;
}

// Result id of an instruction that has one (w[2] when it has a result type, w[1] for types).
inline bool hasTypeAndResult(uint32_t op)
{
    switch (op) {
    case OpExtInst: case OpConstant: case OpFunction: case OpVariable: case OpLoad: case OpAccessChain:
    case OpVectorShuffle: case OpCompositeConstruct: case OpCompositeExtract: case OpCompositeInsert:
    case OpSampledImage: case OpImageSampleImplicitLod: case 88: case OpImageQueryLod: case OpFNegate:
    case OpFAdd: case OpFSub: case OpFMul: case OpFDiv: case OpVectorTimesScalar: case OpMatrixTimesVector:
    case OpDot: case OpFOrdLessThan: case OpFUnordLessThan: case OpFOrdLessThanEqual: case OpFUnordLessThanEqual:
    case OpDPdx: case OpDPdy: case 209: case 83 /*CopyObject*/: case 245 /*Phi*/: case 169 /*Select*/:
        return true;
    default: return false;
    }
}

// Index: result id -> instruction index (only for instructions with type+result, plus types).
struct Index {
    std::map<uint32_t, size_t> def;
    void build(const Module &m)
    {
        def.clear();
        for (size_t i = 0; i < m.ins.size(); ++i) {
            const Inst &in = m.ins[i];
            if (hasTypeAndResult(in.op) && in.w.size() > 2) def[in.w[2]] = i;
            else if ((in.op >= OpTypeInt && in.op <= 39) || in.op == OpTypePointer) { if (in.w.size() > 1) def[in.w[1]] = i; }
        }
    }
    const Inst *of(const Module &m, uint32_t id) const
    {
        std::map<uint32_t, size_t>::const_iterator it = def.find(id);
        return it == def.end() ? nullptr : &m.ins[it->second];
    }
};

inline size_t firstFunction(const Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpFunction) return i;
    return m.ins.size();
}

inline int executionModel(const Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpEntryPoint) return (int)m.ins[i].w[1];
    return -1;
}

// ------------------------------------------------------------------ types and constants
inline uint32_t typeF32(Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpTypeFloat && m.ins[i].w.size() >= 3 && m.ins[i].w[2] == 32) return m.ins[i].w[1];
    const uint32_t id = newId(m);
    m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpTypeFloat, {id, 32}));
    return id;
}

inline uint32_t typeI32(Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpTypeInt && m.ins[i].w.size() >= 4 && m.ins[i].w[2] == 32 && m.ins[i].w[3] == 1) return m.ins[i].w[1];
    const uint32_t id = newId(m);
    m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpTypeInt, {id, 32, 1}));
    return id;
}

inline uint32_t typeVecF(Module &m, uint32_t n)
{
    const uint32_t f = typeF32(m);
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpTypeVector && m.ins[i].w.size() >= 4 && m.ins[i].w[2] == f && m.ins[i].w[3] == n) return m.ins[i].w[1];
    const uint32_t id = newId(m);
    m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpTypeVector, {id, f, n}));
    return id;
}

inline uint32_t constF32(Module &m, float v)
{
    const uint32_t ty = typeF32(m);
    uint32_t bits; memcpy(&bits, &v, 4);
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpConstant && m.ins[i].w.size() == 4 && m.ins[i].w[1] == ty && m.ins[i].w[3] == bits) return m.ins[i].w[2];
    const uint32_t id = newId(m);
    m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpConstant, {ty, id, bits}));
    return id;
}

inline uint32_t constI32(Module &m, int32_t v)
{
    const uint32_t ty = typeI32(m);
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpConstant && m.ins[i].w.size() == 4 && m.ins[i].w[1] == ty && m.ins[i].w[3] == (uint32_t)v) return m.ins[i].w[2];
    const uint32_t id = newId(m);
    m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpConstant, {ty, id, (uint32_t)v}));
    return id;
}

inline bool constValueF(const Module &m, const Index &ix, uint32_t id, float &v)
{
    const Inst *d = ix.of(m, id);
    if (!d || d->op != OpConstant || d->w.size() != 4) return false;
    memcpy(&v, &d->w[3], 4);
    return true;
}

inline bool constValueU(const Module &m, const Index &ix, uint32_t id, uint32_t &v)
{
    const Inst *d = ix.of(m, id);
    if (!d || d->op != OpConstant || d->w.size() != 4) return false;
    v = d->w[3];
    return true;
}

inline uint32_t glslExt(Module &m)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpExtInstImport && str(m.ins[i].w, 2) == "GLSL.std.450") return m.ins[i].w[1];
    const uint32_t id = newId(m);
    Inst e; e.op = OpExtInstImport; e.w = {0, id};
    const char s[] = "GLSL.std.450";   // 12 chars + NUL = 4 words
    for (size_t k = 0; k < sizeof(s); k += 4) {
        uint32_t word = 0;
        for (size_t b = 0; b < 4 && k + b < sizeof(s); ++b) word |= (uint32_t)(unsigned char)s[k + b] << (8 * b);
        e.w.push_back(word);
    }
    size_t at = 0;
    for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpMemoryModel) { at = i; break; }
    m.ins.insert(m.ins.begin() + at, e);
    return id;
}

inline void requireCapability(Module &m, uint32_t cap)
{
    for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpCapability && m.ins[i].w[1] == cap) return;
    m.ins.insert(m.ins.begin(), mk(OpCapability, {cap}));
}

// ------------------------------------------------------------------ lookups
// OpMemberName: (struct id, member index). Returns how many members carry this name.
inline int findMemberByName(const Module &m, const char *nm, uint32_t &structId, uint32_t &member)
{
    int found = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpMemberName && m.ins[i].w.size() > 3 && str(m.ins[i].w, 3) == nm) {
            structId = m.ins[i].w[1]; member = m.ins[i].w[2]; ++found;
        }
    return found;
}

// The OpVariable (in storage class sc) whose pointee type is structId.
inline uint32_t varOfStruct(const Module &m, uint32_t structId, uint32_t sc, uint32_t *ptrTypeOut = nullptr)
{
    for (size_t i = 0; i < m.ins.size(); ++i) {
        if (m.ins[i].op != OpTypePointer || m.ins[i].w.size() < 4 || m.ins[i].w[2] != sc || m.ins[i].w[3] != structId) continue;
        const uint32_t pt = m.ins[i].w[1];
        for (size_t j = 0; j < m.ins.size(); ++j)
            if (m.ins[j].op == OpVariable && m.ins[j].w[1] == pt && m.ins[j].w[3] == sc) { if (ptrTypeOut) *ptrTypeOut = pt; return m.ins[j].w[2]; }
    }
    return 0;
}

inline uint32_t pointerType(const Module &m, uint32_t sc, uint32_t pointee)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpTypePointer && m.ins[i].w.size() >= 4 && m.ins[i].w[2] == sc && m.ins[i].w[3] == pointee) return m.ins[i].w[1];
    return 0;
}

// The struct member's type id.
inline uint32_t memberType(const Module &m, uint32_t structId, uint32_t member)
{
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpTypeStruct && m.ins[i].w[1] == structId && m.ins[i].w.size() > 2 + member) return m.ins[i].w[2 + member];
    return 0;
}

// Does `id` read struct member (var, member)? Follows CompositeExtract and Load to an AccessChain
// whose base is var and whose first index is a constant equal to member.
inline bool readsMember(const Module &m, const Index &ix, uint32_t id, uint32_t var, uint32_t member, int depth = 0)
{
    if (depth > 4) return false;
    const Inst *d = ix.of(m, id);
    if (!d) return false;
    if (d->op == OpCompositeExtract) return readsMember(m, ix, d->w[3], var, member, depth + 1);
    if (d->op == OpLoad) {
        const Inst *ac = ix.of(m, d->w[3]);
        if (!ac || ac->op != OpAccessChain || ac->w.size() < 5 || ac->w[3] != var) return false;
        uint32_t v; return constValueU(m, ix, ac->w[4], v) && v == member;
    }
    return false;
}

// Output variable decorated Location loc (0 if none).
inline uint32_t outputAtLocation(const Module &m, uint32_t loc)
{
    for (size_t i = 0; i < m.ins.size(); ++i) {
        if (m.ins[i].op != OpDecorate || m.ins[i].w.size() < 4 || m.ins[i].w[2] != DecLocation || m.ins[i].w[3] != loc) continue;
        const uint32_t v = m.ins[i].w[1];
        for (size_t j = 0; j < m.ins.size(); ++j)
            if (m.ins[j].op == OpVariable && m.ins[j].w[2] == v && m.ins[j].w[3] == StorageOutput) return v;
    }
    return 0;
}

inline uint32_t pointeeOfVar(const Module &m, const Index &ix, uint32_t var)
{
    for (size_t j = 0; j < m.ins.size(); ++j)
        if (m.ins[j].op == OpVariable && m.ins[j].w[2] == var) {
            const Inst *pt = ix.of(m, m.ins[j].w[1]);
            return (pt && pt->op == OpTypePointer) ? pt->w[3] : 0;
        }
    return 0;
}

inline bool isVec(const Module &m, const Index &ix, uint32_t type, uint32_t n)
{
    const Inst *t = ix.of(m, type);
    return t && t->op == OpTypeVector && t->w[3] == n;
}

// True if no OpLabel lies strictly between instruction indices a < b (same basic block).
inline bool sameBlock(const Module &m, size_t a, size_t b)
{
    if (a > b) return false;
    for (size_t i = a + 1; i < b; ++i) if (m.ins[i].op == OpLabel) return false;
    return true;
}

// Replace operand uses of `from` with `to` in instructions [begin, end), skipping result-id slots.
inline void replaceUses(Module &m, uint32_t from, uint32_t to, size_t begin, size_t end)
{
    for (size_t i = begin; i < end && i < m.ins.size(); ++i) {
        Inst &in = m.ins[i];
        const size_t first = hasTypeAndResult(in.op) ? 3 : 1;
        for (size_t k = first; k < in.w.size(); ++k) if (in.w[k] == from) in.w[k] = to;
    }
}

// ------------------------------------------------------------------ debug: tint
// Every direct OpStore to the Location 0 output becomes (1, 0, 1, a): one shader family drawn
// magenta on screen, to see which family draws what (fscaa.debug_tint=<family>).
inline Result tint(const uint32_t *code, size_t n, std::vector<uint32_t> &out)
{
    Module m; if (!parse(code, n, m)) return NOT_SPIRV;
    if (executionModel(m) != ModelFragment) return WRONG_STAGE;
    Index ix; ix.build(m);
    const uint32_t var = outputAtLocation(m, 0);
    if (!var) return NOT_FOUND;
    const uint32_t vt = pointeeOfVar(m, ix, var);
    if (!isVec(m, ix, vt, 4)) return UNSUPPORTED;
    const uint32_t f = typeF32(m), c1 = constF32(m, 1.0f), c0 = constF32(m, 0.0f);
    int done = 0;
    for (size_t i = 0; i < m.ins.size(); ++i) {
        if (m.ins[i].op != OpStore || m.ins[i].w[1] != var) continue;
        const uint32_t val = m.ins[i].w[2], a = newId(m), mag = newId(m);
        m.ins.insert(m.ins.begin() + i, mk(OpCompositeExtract, {f, a, val, 3}));
        m.ins.insert(m.ins.begin() + i + 1, mk(OpCompositeConstruct, {vt, mag, c1, c0, c1, a}));
        m.ins[i + 2].w[2] = mag;
        i += 2; ++done;
    }
    if (!done) return NOT_FOUND;
    out = emit(m);
    return OK;
}

// ------------------------------------------------------------------ layer 3: wire AA (line3d vertex)
// Persson's wire AA: the screen-space width never drops below minPx; the coverage lost by widening
// (w / w') goes into the line's alpha, so a half-pixel wire draws one pixel wide at half alpha
// instead of breaking into dots. Needs a blending pipeline (layer.cpp enables it if off).
inline Result wire(const uint32_t *code, size_t n, std::vector<uint32_t> &out, float minPx)
{
    Module m; if (!parse(code, n, m)) return NOT_SPIRV;
    if (executionModel(m) != ModelVertex) return WRONG_STAGE;
    uint32_t st = 0, mem = 0;
    const int nw = findMemberByName(m, "u_line_width", st, mem);
    if (nw == 0) return NOT_FOUND;
    if (nw > 1) return AMBIGUOUS;
    const uint32_t ubo = varOfStruct(m, st, StorageUniform);
    if (!ubo) return NOT_FOUND;
    Index ix; ix.build(m);
    // The single load of u_line_width.
    size_t loadAt = 0; int loads = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpLoad && readsMember(m, ix, m.ins[i].w[2], ubo, mem)) { loadAt = i; ++loads; }
    if (loads == 0) return NOT_FOUND;
    if (loads > 1) return AMBIGUOUS;
    const uint32_t colorVar = outputAtLocation(m, 0);
    if (!colorVar || !isVec(m, ix, pointeeOfVar(m, ix, colorVar), 4)) return UNSUPPORTED;
    const uint32_t vec4t = pointeeOfVar(m, ix, colorVar);

    const uint32_t f = typeF32(m), ext = glslExt(m), cMin = constF32(m, minPx);
    // Indices shifted by the type/constant insertions above: re-find the load.
    ix.build(m);
    loadAt = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpLoad && readsMember(m, ix, m.ins[i].w[2], ubo, mem)) { loadAt = i; break; }
    const uint32_t w = m.ins[loadAt].w[2], wc = newId(m), cov = newId(m);
    m.ins.insert(m.ins.begin() + loadAt + 1, mk(OpExtInst, {f, wc, ext, GlslFMax, w, cMin}));
    m.ins.insert(m.ins.begin() + loadAt + 2, mk(OpFDiv, {f, cov, w, wc}));
    replaceUses(m, w, wc, loadAt + 3, m.ins.size());
    // Every store to v_color after the coverage, in the same block: alpha *= cov.
    int stores = 0;
    for (size_t i = loadAt + 3; i < m.ins.size(); ++i) {
        if (m.ins[i].op != OpStore || m.ins[i].w[1] != colorVar) continue;
        if (!sameBlock(m, loadAt + 2, i)) return UNSUPPORTED;
        const uint32_t val = m.ins[i].w[2], a = newId(m), a2 = newId(m), nv = newId(m);
        m.ins.insert(m.ins.begin() + i, mk(OpCompositeExtract, {f, a, val, 3}));
        m.ins.insert(m.ins.begin() + i + 1, mk(OpFMul, {f, a2, a, cov}));
        m.ins.insert(m.ins.begin() + i + 2, mk(OpCompositeInsert, {vec4t, nv, a2, val, 3}));
        m.ins[i + 3].w[2] = nv;
        i += 3; ++stores;
    }
    if (!stores) return UNSUPPORTED;   // colour written before the width is known
    out = emit(m);
    return OK;
}

// ------------------------------------------------------------------ layer 4: light-point AA (light vertex)
// gl_Position = P * (X + V * S), S the sprite's eye-space size. A sprite is never smaller than
// minPx pixels: S' = max(S, minPx * u_pixel_size.x * (-X.z) / P[0][0]) (u_pixel_size is one pixel
// in NDC), and v_color is scaled by (S/S')^2 so the light's total energy is unchanged - a
// sub-pixel light becomes a stable, dimmer pixel instead of flickering on and off.
inline Result light(const uint32_t *code, size_t n, std::vector<uint32_t> &out, float minPx)
{
    Module m; if (!parse(code, n, m)) return NOT_SPIRV;
    if (executionModel(m) != ModelVertex) return WRONG_STAGE;
    uint32_t pst = 0, pmem = 0;
    if (findMemberByName(m, "u_pixel_size", pst, pmem) != 1) return NOT_FOUND;
    uint32_t uboPtr = 0;
    const uint32_t ubo = varOfStruct(m, pst, StorageUniform, &uboPtr);
    if (!ubo) return NOT_FOUND;
    const uint32_t colorVar = outputAtLocation(m, 0);
    Index ix; ix.build(m);
    if (!colorVar || !isVec(m, ix, pointeeOfVar(m, ix, colorVar), 4)) return UNSUPPORTED;
    const uint32_t pixType = memberType(m, pst, pmem);
    if (!isVec(m, ix, pixType, 2)) return UNSUPPORTED;
    const uint32_t pixPtr = pointerType(m, StorageUniform, pixType);
    if (!pixPtr) return UNSUPPORTED;

    // Find MatrixTimesVector(P, FAdd(X, VectorTimesScalar(V, S))) feeding a store to gl_Position.
    struct Hit { size_t mtv; uint32_t P, X, V, S, add; };
    std::vector<Hit> hits;
    for (size_t i = 0; i < m.ins.size(); ++i) {
        if (m.ins[i].op != OpMatrixTimesVector) continue;
        const Inst *add = ix.of(m, m.ins[i].w[4]);
        if (!add || add->op != OpFAdd) continue;
        for (int side = 0; side < 2; ++side) {
            const uint32_t X = add->w[3 + side], off = add->w[4 - side];
            const Inst *vts = ix.of(m, off);
            if (!vts || vts->op != OpVectorTimesScalar) continue;
            // the product must be stored (directly) to a Position builtin through an access chain
            bool toPos = false;
            for (size_t j = i + 1; j < m.ins.size() && !toPos; ++j) {
                if (m.ins[j].op != OpStore || m.ins[j].w[2] != m.ins[i].w[2]) continue;
                const Inst *ac = ix.of(m, m.ins[j].w[1]);
                if (ac && ac->op == OpAccessChain && ac->w.size() == 5) toPos = true;
            }
            if (!toPos) continue;
            Hit h; h.mtv = i; h.P = m.ins[i].w[3]; h.X = X; h.V = vts->w[3]; h.S = vts->w[4]; h.add = add->w[2];
            hits.push_back(h);
            break;
        }
    }
    if (hits.empty()) return NOT_FOUND;
    if (hits.size() > 1) return AMBIGUOUS;
    const Hit h = hits[0];
    const uint32_t vec4t = pointeeOfVar(m, ix, colorVar);

    const uint32_t f = typeF32(m), ext = glslExt(m), cMin = constF32(m, minPx), cIdx = constI32(m, (int32_t)pmem);
    // Re-locate the multiply after the insertions above.
    size_t at = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpMatrixTimesVector && m.ins[i].w[4] == h.add) { at = i; break; }
    const uint32_t mtvType = m.ins[at].w[1];
    std::vector<Inst> add;
    const uint32_t z = newId(m), nz = newId(m), p00 = newId(m), pc = newId(m), pv = newId(m), px = newId(m),
                   t1 = newId(m), t2 = newId(m), sMin = newId(m), s2 = newId(m), r = newId(m), fade = newId(m),
                   off2 = newId(m), add2 = newId(m), col = newId(m), col2 = newId(m);
    add.push_back(mk(OpCompositeExtract, {f, z, h.X, 2}));
    add.push_back(mk(OpFNegate, {f, nz, z}));
    add.push_back(mk(OpCompositeExtract, {f, p00, h.P, 0, 0}));
    add.push_back(mk(OpAccessChain, {pixPtr, pc, ubo, cIdx}));
    add.push_back(mk(OpLoad, {pixType, pv, pc}));
    add.push_back(mk(OpCompositeExtract, {f, px, pv, 0}));
    add.push_back(mk(OpFMul, {f, t1, cMin, px}));
    add.push_back(mk(OpFMul, {f, t2, t1, nz}));
    add.push_back(mk(OpFDiv, {f, sMin, t2, p00}));
    add.push_back(mk(OpExtInst, {f, s2, ext, GlslFMax, h.S, sMin}));
    add.push_back(mk(OpFDiv, {f, r, h.S, s2}));
    add.push_back(mk(OpFMul, {f, fade, r, r}));
    add.push_back(mk(OpVectorTimesScalar, {vec4t, off2, h.V, s2}));
    add.push_back(mk(OpFAdd, {vec4t, add2, h.X, off2}));
    add.push_back(mk(OpLoad, {vec4t, col, colorVar}));
    add.push_back(mk(OpVectorTimesScalar, {vec4t, col2, col, fade}));
    Inst st; st.op = OpStore; st.w = {0, colorVar, col2};
    add.push_back(st);
    (void)mtvType;
    m.ins.insert(m.ins.begin() + at, add.begin(), add.end());
    m.ins[at + add.size()].w[4] = add2;
    out = emit(m);
    return OK;
}

// ------------------------------------------------------------------ layer 5: specular AA (deferred_gbuf)
// Tokuyoshi & Kaplanyan: widen GGX alpha^2 by the screen-space variance of the normal,
// a2' = a2 + min(2 * sigma * (|dN/dx|^2 + |dN/dy|^2), kappa^2). N is the decoded G-buffer normal
// (second operand of reflect()); a2 = (r*r)*(r*r) with r = 1 - 0.96875 * (...).
inline Result specular(const uint32_t *code, size_t n, std::vector<uint32_t> &out, float sigma, float kappa)
{
    Module m; if (!parse(code, n, m)) return NOT_SPIRV;
    if (executionModel(m) != ModelFragment) return WRONG_STAGE;
    Index ix; ix.build(m);
    uint32_t ext0 = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpExtInstImport && str(m.ins[i].w, 2) == "GLSL.std.450") ext0 = m.ins[i].w[1];
    if (!ext0) return NOT_FOUND;
    uint32_t N = 0;
    for (size_t i = 0; i < m.ins.size(); ++i)
        if (m.ins[i].op == OpExtInst && m.ins[i].w[3] == ext0 && m.ins[i].w[4] == GlslReflect && m.ins[i].w.size() >= 7) {
            if (N && N != m.ins[i].w[6]) return AMBIGUOUS;
            N = m.ins[i].w[6];
        }
    if (!N) return NOT_FOUND;
    // a2 = a*a, a = r*r, r = 1 - (0.96875 * x)
    std::vector<uint32_t> a2s;
    for (size_t i = 0; i < m.ins.size(); ++i) {
        const Inst &mul = m.ins[i];
        if (mul.op != OpFMul || mul.w[3] != mul.w[4]) continue;
        const Inst *a = ix.of(m, mul.w[3]);
        if (!a || a->op != OpFMul || a->w[3] != a->w[4]) continue;
        const Inst *r = ix.of(m, a->w[3]);
        if (!r || r->op != OpFSub) continue;
        float one; if (!constValueF(m, ix, r->w[3], one) || one != 1.0f) continue;
        const Inst *k = ix.of(m, r->w[4]);
        if (!k || k->op != OpFMul) continue;
        float c; if (!(constValueF(m, ix, k->w[3], c) || constValueF(m, ix, k->w[4], c)) || c != 0.96875f) continue;
        a2s.push_back(mul.w[2]);
    }
    if (a2s.empty()) return NOT_FOUND;
    const uint32_t f = typeF32(m), v3 = typeVecF(m, 3), ext = glslExt(m),
                   cSig = constF32(m, sigma), c2 = constF32(m, 2.0f), cK2 = constF32(m, kappa * kappa);
    for (size_t k = 0; k < a2s.size(); ++k) {
        size_t at = 0;
        for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpFMul && m.ins[i].w[2] == a2s[k]) { at = i; break; }
        const uint32_t dx = newId(m), dy = newId(m), d1 = newId(m), d2 = newId(m), s = newId(m), var = newId(m),
                       v2 = newId(m), kk = newId(m), a2n = newId(m);
        std::vector<Inst> add;
        add.push_back(mk(OpDPdx, {v3, dx, N}));
        add.push_back(mk(OpDPdy, {v3, dy, N}));
        add.push_back(mk(OpDot, {f, d1, dx, dx}));
        add.push_back(mk(OpDot, {f, d2, dy, dy}));
        add.push_back(mk(OpFAdd, {f, s, d1, d2}));
        add.push_back(mk(OpFMul, {f, var, s, cSig}));
        add.push_back(mk(OpFMul, {f, v2, var, c2}));
        add.push_back(mk(OpExtInst, {f, kk, ext, GlslFMin, v2, cK2}));
        add.push_back(mk(OpFAdd, {f, a2n, a2s[k], kk}));
        m.ins.insert(m.ins.begin() + at + 1, add.begin(), add.end());
        replaceUses(m, a2s[k], a2n, at + 1 + add.size(), m.ins.size());
    }
    out = emit(m);
    return OK;
}

// ------------------------------------------------------------------ layer 6: alpha AA (terrain fragment)
// X-Plane's alpha test is `if (a <= u_alpha_cut.x) discard` (a = .w of a Phi that blends several
// texture layers, so there is no single sample to take a mip level from - measured 2026-10-15).
// Fix at the edge only: the cutoff gets a per-pixel dither scaled by fwidth(a),
//     cut' = cut + (ign(gl_FragCoord.xy) - 0.5) * min(fwidth(a) * kWidth, 0.5)
// Inside a leaf or a gap fwidth(a) ~ 0 and the test is exactly X-Plane's; across the 1-px alpha
// transition the edge is dithered, and TAA's per-frame jitter resolves the dither into a smooth,
// anti-aliased edge (hashed-alpha in the spirit of Wyman & McGuire, fwidth-scaled as Golus does).
// ign = interleaved gradient noise (Jimenez 2014). kWidth is fscaa.alpha_edge (default 1.0).
inline Result alpha(const uint32_t *code, size_t n, std::vector<uint32_t> &out, float kWidth)
{
    Module m; if (!parse(code, n, m)) return NOT_SPIRV;
    if (executionModel(m) != ModelFragment) return WRONG_STAGE;
    uint32_t st = 0, mem = 0;
    if (findMemberByName(m, "u_alpha_cut", st, mem) != 1) return NOT_FOUND;
    const uint32_t ubo = varOfStruct(m, st, StorageUniform);
    if (!ubo) return NOT_FOUND;
    Index ix; ix.build(m);
    std::vector<uint32_t> cmps;
    for (size_t i = 0; i < m.ins.size(); ++i) {
        const uint32_t op = m.ins[i].op;
        if (op != OpFOrdLessThan && op != OpFUnordLessThan && op != OpFOrdLessThanEqual && op != OpFUnordLessThanEqual) continue;
        if (readsMember(m, ix, m.ins[i].w[4], ubo, mem)) cmps.push_back(m.ins[i].w[2]);
    }
    if (cmps.empty()) return NOT_FOUND;
    if (cmps.size() > 1) return AMBIGUOUS;
    // gl_FragCoord: reuse X-Plane's input if it has one, otherwise declare it.
    const uint32_t f = typeF32(m), v4 = typeVecF(m, 4), ext = glslExt(m);
    uint32_t fc = 0;
    for (size_t i = 0; i < m.ins.size() && !fc; ++i)
        if (m.ins[i].op == OpDecorate && m.ins[i].w.size() >= 4 && m.ins[i].w[2] == DecBuiltIn && m.ins[i].w[3] == 15 /*FragCoord*/)
            fc = m.ins[i].w[1];
    if (!fc) {
        uint32_t pIn = pointerType(m, 1 /*Input*/, v4);
        if (!pIn) { pIn = newId(m); m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpTypePointer, {pIn, 1, v4})); }
        fc = newId(m);
        m.ins.insert(m.ins.begin() + firstFunction(m), mk(OpVariable, {pIn, fc, 1}));
        size_t dAt = 0;
        for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpDecorate) { dAt = i; break; }
        m.ins.insert(m.ins.begin() + dAt, mk(OpDecorate, {fc, DecBuiltIn, 15}));
        for (size_t i = 0; i < m.ins.size(); ++i) if (m.ins[i].op == OpEntryPoint) { m.ins[i].w.push_back(fc); break; }
    }
    const uint32_t cK = constF32(m, kWidth), cHalf = constF32(m, 0.5f),
                   cA = constF32(m, 0.06711056f), cB = constF32(m, 0.00583715f), cC = constF32(m, 52.9829189f);
    size_t at = 0;
    for (size_t i = 0; i < m.ins.size(); ++i) if (hasTypeAndResult(m.ins[i].op) && m.ins[i].w[2] == cmps[0]) { at = i; break; }
    const uint32_t a = m.ins[at].w[3], cut = m.ins[at].w[4];
    const uint32_t fw = newId(m), w1 = newId(m), wd = newId(m), p = newId(m), x = newId(m), y = newId(m),
                   t1 = newId(m), t2 = newId(m), t3 = newId(m), t4 = newId(m), t5 = newId(m), nz = newId(m),
                   d = newId(m), off = newId(m), cut2 = newId(m);
    std::vector<Inst> add;
    add.push_back(mk(209 /*OpFwidth*/, {f, fw, a}));
    add.push_back(mk(OpFMul, {f, w1, fw, cK}));
    add.push_back(mk(OpExtInst, {f, wd, ext, GlslFMin, w1, cHalf}));
    add.push_back(mk(OpLoad, {v4, p, fc}));
    add.push_back(mk(OpCompositeExtract, {f, x, p, 0}));
    add.push_back(mk(OpCompositeExtract, {f, y, p, 1}));
    add.push_back(mk(OpFMul, {f, t1, x, cA}));
    add.push_back(mk(OpFMul, {f, t2, y, cB}));
    add.push_back(mk(OpFAdd, {f, t3, t1, t2}));
    add.push_back(mk(OpExtInst, {f, t4, ext, 10 /*Fract*/, t3}));
    add.push_back(mk(OpFMul, {f, t5, t4, cC}));
    add.push_back(mk(OpExtInst, {f, nz, ext, 10 /*Fract*/, t5}));
    add.push_back(mk(OpFSub, {f, d, nz, cHalf}));
    add.push_back(mk(OpFMul, {f, off, d, wd}));
    add.push_back(mk(OpFAdd, {f, cut2, cut, off}));
    m.ins.insert(m.ins.begin() + at, add.begin(), add.end());
    m.ins[at + add.size()].w[4] = cut2;
    out = emit(m);
    return OK;
}

} // namespace fscaapatch
