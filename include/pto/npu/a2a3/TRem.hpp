/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>

namespace pto {
// ---- High precision remainder (issue #2), A2/A3 ----
// TREM owns the shared HP math here (digit-peeling core, scratch/select
// machinery, staged loop inversion); TRemS.hpp adds the scalar-divisor
// variants. All operands are UB buffers with pipe_barrier between dependent
// ops. Rigorous algorithm notes: docs/en/isa/TREM.md appendix.
constexpr uint32_t FMOD_ITERATION_NUM_MAX_A2A3 = 11;

constexpr float FMOD_HP_SUBNORMAL = 1.1754944e-38f; // min normal, same threshold as A5
constexpr float FMOD_HP_MAXVALUE = 3.4028235e38f;

// Scaling schedule identical to the A5 implementation (a5/TRem.hpp).
constexpr float FMOD_HP_SCALE1[FMOD_ITERATION_NUM_MAX_A2A3] = {16777216.0f, 16777216.0f, 2.8147e14f, 4.7224e21f,
                                                               7.9228e28f,  1.3292e36f,  1.3292e36f, 1.3292e36f,
                                                               1.3292e36f,  1.3292e36f,  1.3292e36f};
constexpr float FMOD_HP_SCALE2[FMOD_ITERATION_NUM_MAX_A2A3] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 16777216.0f, 2.8147e14f, 4.7224e21f, 7.9228e28f, 1.3292e36f};
constexpr float FMOD_HP_POST_SCALE = 5.9604645e-8f; // 2^-24
// Compensation batching of the final peel iteration (HpFusedSub): the staged
// final pass advances this many vectors per batched call; 1 reproduces the
// per-vector behavior. The comp scratch scales with it (512 floats per
// vector pair - see HpScratchRequiredColsV).
#ifndef TREM_HP_FUSED_CHUNK_A2A3_OVERRIDE
constexpr uint32_t TREM_HP_FUSED_CHUNK_A2A3 = 8;
#else
constexpr uint32_t TREM_HP_FUSED_CHUNK_A2A3 = TREM_HP_FUSED_CHUNK_A2A3_OVERRIDE;
#endif

// Scratch contract, in floats from the tmp tile base, for a capacity of C
// 64-element vectors (C = full vectors per row for the staged passes, 1 for
// the per-vector path); disjoint lifetimes share storage:
//   [0, 64C)        bTmp    scaled divisor (dup'd from |scalar|*scale per iter)
//   [64C, 128C)     k/t     quotient (f32/s32 views), then general temp
//                           (k*bTmp product, sign, select/probe operands)
//   [128C, +maskF)  cmpMask packed compare bitmask (32B aligned, padded)
//   [+maskF, +8)    addrBuf 64-bit cmpmask address buffer
//   cap == 1 only: [+8, +72) state across the 11 iterations (staged passes
//                  keep state in dst; sign handling is compare+select on t)
struct HpScratchA2A3 {
    __ubuf__ float* bTmp;
    __ubuf__ float* k; // quotient and post-quotient temp share this region
    __ubuf__ uint8_t* cmpMask;
    __ubuf__ uint32_t* addrBuf;
    __ubuf__ float* state; // per-vector path (cap == 1) only
    __ubuf__ float* div;   // TREM only: |src1| row, live through one pass
    __ubuf__ float* comp;  // 5 vectors: final-iteration compensated subtract
};

PTO_INTERNAL unsigned HpScratchMaskFloats(unsigned capVecs)
{
    unsigned maskBytes = capVecs * BIT_TO_BYTE; // 64 bits per vector = 8 bytes
    return ((maskBytes + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE) * (BLOCK_BYTE_SIZE / 4);
}

// TREM (vector divisor) scratch: the |src1| row stays live through a whole
// pass, adding one more 64C region ahead of the TREMS layout:
//   [0, 64C) div, [64C, 128C) bTmp, [128C, 192C) k, then cmpMask/addrBuf,
//   and state (cap == 1 only) after them.
PTO_INTERNAL unsigned HpScratchRequiredColsV(unsigned capVecs)
{
    // + the compensation window of the final-iteration HpFusedSub: eight
    // vectors per batched chunk (clamped to the row's vector count)
    unsigned chunk = capVecs < TREM_HP_FUSED_CHUNK_A2A3 ? capVecs : TREM_HP_FUSED_CHUNK_A2A3;
    return 192 * capVecs + HpScratchMaskFloats(capVecs) + 8 + (capVecs == 1 ? 64 : 0) + 10 * 64 * chunk;
}

PTO_INTERNAL void HpScratchInitV(HpScratchA2A3& s, __ubuf__ float* tmp, unsigned capVecs)
{
    unsigned v = 64 * capVecs;
    s.div = tmp;
    s.bTmp = tmp + v;
    s.k = tmp + 2 * v;
    s.cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + 3 * v);
    s.addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp + 3 * v + HpScratchMaskFloats(capVecs));
    s.state = reinterpret_cast<__ubuf__ float*>(s.addrBuf + BIT_TO_BYTE);
    s.comp = s.state + (capVecs == 1 ? 64 : 0);
}

// set_cmpmask dereferences a full 64-bit address (see TRemS.hpp): scalar UB
// store with V<->S flags; the caller already produced s.cmpMask.
PTO_INTERNAL void HpSetCmpMask(const HpScratchA2A3& s)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    *(reinterpret_cast<__ubuf__ uint64_t*>(s.addrBuf)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(s.cmpMask));
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    set_cmpmask(s.addrBuf);
    pipe_barrier(PIPE_V);
}

// dst = mask ? a : b over `rep` repeats (`lanes` active lanes) using the
// packed mask currently in s.cmpMask.
PTO_INTERNAL void HpSelect(
    __ubuf__ float* dst, __ubuf__ float* a, __ubuf__ float* b, const HpScratchA2A3& s, unsigned rep, unsigned lanes)
{
    HpSetCmpMask(s);
    set_mask_count();
    set_vector_mask(0, lanes);
    vsel(dst, a, b, rep, 1, 1, 1, 8, 8, 8, 2);
    pipe_barrier(PIPE_V);
}

// Scalar helpers without libm (device side): bit-level inf/nan tests.
PTO_INTERNAL bool HpScalarIsInf(float v)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
    return (bits & 0x7FFFFFFFu) == 0x7F800000u;
}

PTO_INTERNAL float HpInf()
{
    uint32_t bits = 0x7F800000u;
    return *reinterpret_cast<float*>(&bits);
}

PTO_INTERNAL float HpNan()
{
    uint32_t bits = 0x7FC00000u;
    return *reinterpret_cast<float*>(&bits);
}

// k = roundToInt(k) as float (A5's vtrc: a2a3 converts through int32, which
// saturates far outside the algorithm's quotient range).
template <bool isFmod>
PTO_INTERNAL void HpRoundToInt(__ubuf__ float* k, unsigned rep)
{
    __ubuf__ int32_t* ki = reinterpret_cast<__ubuf__ int32_t*>(k);
    if constexpr (isFmod) {
        vconv_f322s32z(ki, k, rep, 1, 1, 8, 8);
    } else {
        vconv_f322s32f(ki, k, rep, 1, 1, 8, 8);
    }
    pipe_barrier(PIPE_V);
    vconv_s322f32(k, ki, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
}

// Compensated `state -= k * bTmp` for the FINAL peel iteration (single
// rounding): exact k*bTmp via a 257-split of bTmp and 8-bit digits of k,
// Sterbenz subtract, finiteness-guarded with a plain fallback. Details in
// docs/en/isa/TREM.md appendix. Needs s.comp (10 vectors) live.
PTO_INTERNAL void HpFusedSub(
    __ubuf__ float* state, __ubuf__ float* k, __ubuf__ float* bTmp, HpScratchA2A3& s, unsigned rep)
{
    // plain locals: the bisheng variadic intrinsic checker rejects
    // constexpr-qualified scalar operands
    float splitC = 257.0f;              // Veltkamp constant 2^8 + 1
    float pow2m16 = 1.52587890625e-05f; // 2^-16
    float pow2p16 = 65536.0f;
    float pow2m8 = 0.00390625f; // 2^-8
    float pow2p8 = 256.0f;
    float maxV = FMOD_HP_MAXVALUE;
    float zeroV = 0.0f;
    unsigned lanes = rep * REPEAT_BYTE / sizeof(float);
    unsigned v = lanes;

    // c0 bH | c1 bL | c2 k1 | c3 k2 | c4 k3 | c5 hi | c6 lo | c7 p | c8 s | c9 z/fb
    __ubuf__ float* bH = s.comp;
    __ubuf__ float* bL = s.comp + v;
    __ubuf__ float* k1 = s.comp + 2 * v;
    __ubuf__ float* k2 = s.comp + 3 * v;
    __ubuf__ float* k3 = s.comp + 4 * v;
    __ubuf__ float* hi = s.comp + 5 * v;
    __ubuf__ float* lo = s.comp + 6 * v;
    __ubuf__ float* p = s.comp + 7 * v;
    __ubuf__ float* sc = s.comp + 8 * v;
    __ubuf__ float* z = s.comp + 9 * v;

    // Veltkamp split of bTmp: bH + bL == bTmp exactly, bH <= 8 bits
    vmuls(sc, bTmp, splitC, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, sc, bTmp, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(bH, sc, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(bL, bTmp, bH, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

    // k (< 2^24) into three exact 8-bit digits k1*2^16 + k2*2^8 + k3
    vmuls(k1, k, pow2m16, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpRoundToInt<false>(k1, rep);
    vmuls(sc, k1, pow2p16, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, k, sc, rep, 1, 1, 1, 8, 8, 8); // z = krem
    pipe_barrier(PIPE_V);
    vmuls(k2, z, pow2m8, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpRoundToInt<false>(k2, rep);
    vmuls(sc, k2, pow2p8, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(k3, z, sc, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

    // six exact partial products, streamed into (hi, lo) via TwoSums
    vmul(hi, k1, bH, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(hi, hi, pow2p16, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vector_dup(lo, zeroV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // p = k1*bL*2^16
    vmul(p, k1, bL, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(p, p, pow2p16, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(sc, hi, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, sc, hi, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(p, p, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(lo, lo, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(hi, sc, zeroV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // p = k2*bH*2^8
    vmul(p, k2, bH, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(p, p, pow2p8, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(sc, hi, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, sc, hi, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(p, p, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(lo, lo, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(hi, sc, zeroV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // p = k2*bL*2^8
    vmul(p, k2, bL, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(p, p, pow2p8, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(sc, hi, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, sc, hi, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(p, p, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(lo, lo, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(hi, sc, zeroV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // p = k3*bH
    vmul(p, k3, bH, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(sc, hi, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, sc, hi, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(p, p, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(lo, lo, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(hi, sc, zeroV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // p = k3*bL
    vmul(p, k3, bL, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(sc, hi, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, sc, hi, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(p, p, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(lo, lo, p, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(hi, sc, zeroV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    // r1 = state - hi (Sterbenz-exact); r = r1 - lo rounds once
    vsub(sc, state, hi, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(sc, sc, lo, rep, 1, 1, 1, 8, 8, 8); // compensated r in sc
    pipe_barrier(PIPE_V);

    // fallback fb = state - RN(k*bTmp)
    vmul(z, k, bTmp, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(z, state, z, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

    // finiteness guard: |r| <= FLT_MAX (NaN compares false) picks r; the
    // rest takes the fallback
    vabs(p, sc, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(-1, -1);
    vcmpvs_le(s.cmpMask, p, maxV, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, sc, z, s, rep, lanes); // finite ? compensated : fallback
}

// ---- TREM (vector divisor) variants ----
// Same digit-peeling iteration, but the divisor comes from a |src1| row
// buffer and the subnormal regime is decided per element (A5 semantics:
// masked 2^24 lift of divisor and state where |div| <= min normal). The k
// region is free before the quotient, so it backs the lift temporaries.

// One SolveScale iteration over `rep` batched vectors with a vector divisor.
// `div` holds |src1| for the same lanes; state is updated in place.
template <bool isFmod, int32_t iterationNum>
PTO_INTERNAL void HpSolveScaleV(__ubuf__ float* state, __ubuf__ float* div, HpScratchA2A3& s, unsigned rep)
{
    // plain locals: the bisheng variadic intrinsic checker rejects
    // constexpr-qualified scalar operands
    float maxValue = FMOD_HP_MAXVALUE;
    float subNormal = FMOD_HP_SUBNORMAL;
    float zeroV = 0.0f;
    float scale1 = FMOD_HP_SCALE1[iterationNum - 1];
    float scale2 = FMOD_HP_SCALE2[iterationNum - 1];
    unsigned lanes = rep * REPEAT_BYTE / sizeof(float);

    if constexpr (iterationNum == 1) {
        // per-element subnormal-divisor regime: lift divisor and state by
        // 2^24 only where |div| <= min normal (the post pass rescales those
        // lanes by 2^-24)
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_le(s.cmpMask, div, subNormal, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(s.k, div, scale1, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(s.bTmp, s.k, div, s, rep, lanes); // bTmp = sub ? div*2^24 : div
        vmuls(s.k, state, scale1, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(state, s.k, state, s, rep, lanes); // state = sub ? state*2^24 : state
    } else {
        vmuls(s.bTmp, div, scale1, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        if constexpr (iterationNum > 5) {
            vmuls(s.bTmp, s.bTmp, scale2, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    }

    vdiv(s.k, state, s.bTmp, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    HpRoundToInt<isFmod>(s.k, rep);
    if constexpr (iterationNum != 1) {
        // not necessary to check for inf in the final iteration
        vmins(s.bTmp, s.bTmp, maxValue, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    if constexpr (iterationNum == 1) {
        // final iteration: compensated single-rounding subtract (HpFusedSub)
        HpFusedSub(state, s.k, s.bTmp, s, rep);
    } else {
        // state -= k * bTmp (positive k; see HpSolveScaleS)
        vmul(s.k, s.k, s.bTmp, rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vsub(state, state, s.k, rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    if constexpr (iterationNum == 1) {
        // r = r + (r < 0) * bTmp, reusing the k region (quotient dead)
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(s.cmpMask, state, 0.0f, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vector_dup(s.k, zeroV, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(s.k, s.bTmp, s.k, s, rep, lanes);
        vadd(state, state, s.k, rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
}

// Full iteration chain 11 -> 1 on one state buffer (per-vector TREM path).
template <bool isFmod, int32_t iterationNum>
PTO_INTERNAL void HpSolveScaleSeqV(__ubuf__ float* state, __ubuf__ float* div, HpScratchA2A3& s, unsigned rep)
{
    HpSolveScaleV<isFmod, iterationNum>(state, div, s, rep);
    if constexpr (iterationNum > 1) {
        HpSolveScaleSeqV<isFmod, iterationNum - 1>(state, div, s, rep);
    }
}

// Postprocessing with a vector divisor (TREM), mirroring a5
// TFmodRemHPPost<false>: sign restore, subnormal rescale, exception fixups,
// sign correction. Precondition: src1 is nonzero and not NaN.
PTO_INTERNAL void HpPostV(
    __ubuf__ float* state, __ubuf__ float* src0, __ubuf__ float* src1, HpScratchA2A3& s, unsigned rep)
{
    float subNormal = FMOD_HP_SUBNORMAL;
    float postScale = FMOD_HP_POST_SCALE;
    unsigned lanes = rep * REPEAT_BYTE / sizeof(float);

    // sign restore: state = (src0 < 0) ? -state : state
    set_mask_norm();
    set_vector_mask(-1, -1);
    vcmpvs_lt(s.cmpMask, src0, 0.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(s.k, state, -1.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.k, state, s, rep, lanes);

    // per-element subnormal rescale: state *= 2^-24 where |src1| <= min normal
    vabs(s.k, src1, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vcmpvs_le(s.cmpMask, s.k, subNormal, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(s.k, state, postScale, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.k, state, s, rep, lanes);

    // exception: src1 = +-inf -> src0
    vabs(s.k, src1, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vcmpvs_eq(s.cmpMask, s.k, HpInf(), rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, src0, state, s, rep, lanes);

    // exception: both dividend and divisor +-inf -> NaN (must run after the
    // src1-inf select, which would otherwise leave the dividend there)
    vabs(s.k, src0, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vcmpvs_eq(s.cmpMask, s.k, HpInf(), rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vector_dup(s.k, HpNan(), rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.k, state, s, rep, lanes);

    // exception: src0 = +-0 (divisor in-domain) -> src0
    set_mask_norm();
    set_vector_mask(-1, -1);
    vcmpvs_eq(s.cmpMask, src0, 0.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, src0, state, s, rep, lanes);

    // remainder sign correction: if state * src1 < 0, state += src1
    vmul(s.k, state, src1, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(-1, -1);
    vcmpvs_lt(s.cmpMask, s.k, 0.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(s.k, state, src1, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.k, state, s, rep, lanes);
    // inf-divisor lanes -> floored semantics: same-sign x -> x, opposite
    // sign -> x + y; zero lanes keep x (0*inf probe is NaN). The candidate
    // lives in s.div: 64C-wide and dead in the post pass (s.comp only holds
    // the min(C,8)-sized HpFusedSub window). See appendix.
    vmul(s.k, src0, src1, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vadd(s.div, src0, src1, rep, 1, 1, 1, 8, 8, 8); // candidate x + y
    pipe_barrier(PIPE_V);
    vcmpvs_lt(s.cmpMask, s.k, 0.0f, rep, 1, 1, 8, 8); // opposite-sign lanes
    pipe_barrier(PIPE_V);
    HpSelect(s.bTmp, s.div, src0, s, rep, lanes); // selA = opp ? x+y : x
    vabs(s.k, src1, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vcmpvs_eq(s.cmpMask, s.k, HpInf(), rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.bTmp, state, s, rep, lanes); // inf-div ? selA : state
    // re-apply the both-inf NaN clobbered by that select
    vabs(s.k, src0, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vcmpvs_eq(s.cmpMask, s.k, HpInf(), rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vector_dup(s.bTmp, HpNan(), rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.bTmp, state, s, rep, lanes);
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// ---- Staged loop inversion (issue #2) ----
// One SolveScale iteration per pass over all full vectors of a row (state
// staged in dst), each op batched into single repeat instructions; the
// per-vector path pays one instruction + pipe drain per 64 elements. The
// per-op pipe_barrier subsumes the A5 mem_bar(VST_VLD) between passes.
#ifndef TREM_HP_STAGED_MIN_VECS
#define TREM_HP_STAGED_MIN_VECS 48
#endif
constexpr uint32_t TREM_HP_STAGED_MIN_VECS_A2A3 = TREM_HP_STAGED_MIN_VECS;

template <typename T>
PTO_INTERNAL bool TRemHpStagedDisjointA2A3(const __ubuf__ T* a, uint32_t aElems, const __ubuf__ T* b, uint32_t bElems)
{
    return a + aElems <= b || b + bElems <= a;
}

// ---- TREM (vector divisor) drivers ----
// Same eligibility and loop inversion as the TREMS drivers; the divisor row
// is reloaded (vabs) from the original src1 each pass because a scaled
// divisor must not feed the next pass.

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL bool TRemVStagedEligibleA2A3(
    const __ubuf__ float* dst, const __ubuf__ float* src0, const __ubuf__ float* src1, uint32_t validRows,
    uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = REPEAT_BYTE / sizeof(float);
    if (validRows * (validCols / elemsPerRepeat) < TREM_HP_STAGED_MIN_VECS_A2A3) {
        return false;
    }
    return TRemHpStagedDisjointA2A3<float>(
               dst, TileDataDst::Rows * TileDataDst::Cols, src0, TileDataSrc0::Rows * TileDataSrc0::Cols) &&
           TRemHpStagedDisjointA2A3<float>(
               dst, TileDataDst::Rows * TileDataDst::Cols, src1, TileDataSrc1::Rows * TileDataSrc1::Cols);
}

// One 64-lane vector through all iterations + post (tails and fallback).
template <bool isFmod, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
PTO_INTERNAL void TRemVHpVector(
    __ubuf__ float* dstVec, __ubuf__ float* src0Vec, __ubuf__ float* src1Vec, HpScratchA2A3& s, unsigned count)
{
    set_mask_count();
    set_vector_mask(0, count);
    vabs(s.state, src0Vec, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vabs(s.div, src1Vec, 1, 1, 1, 8, 8); // full-lane; masked lanes never reach dst
    pipe_barrier(PIPE_V);

    HpSolveScaleSeqV<isFmod, FMOD_ITERATION_NUM_MAX_A2A3>(s.state, s.div, s, 1);
    HpPostV(s.state, src0Vec, src1Vec, s, 1);

    set_mask_count();
    set_vector_mask(0, count);
    vadds(dstVec, s.state, 0.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
}

// Per-vector TREM path over every full vector and tail of the tile.
template <bool isFmod, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
PTO_INTERNAL void TRemVHpRegister(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* tmp, unsigned validRows,
    unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(float);
    unsigned fullPerRow = validCols / elementsPerRepeat;
    unsigned tailCount = validCols - fullPerRow * elementsPerRepeat;

    HpScratchA2A3 s;
    HpScratchInitV(s, tmp, 1);

    for (unsigned row = 0; row < validRows; ++row) {
        __ubuf__ float* dstRow = dst + row * dstRowStride;
        __ubuf__ float* s0Row = src0 + row * src0RowStride;
        __ubuf__ float* s1Row = src1 + row * src1RowStride;
        for (unsigned j = 0; j < fullPerRow; ++j) {
            TRemVHpVector<isFmod, dstRowStride, src0RowStride, src1RowStride>(
                dstRow + j * elementsPerRepeat, s0Row + j * elementsPerRepeat, s1Row + j * elementsPerRepeat, s,
                elementsPerRepeat);
        }
        if (tailCount > 0) {
            TRemVHpVector<isFmod, dstRowStride, src0RowStride, src1RowStride>(
                dstRow + fullPerRow * elementsPerRepeat, s0Row + fullPerRow * elementsPerRepeat,
                s1Row + fullPerRow * elementsPerRepeat, s, tailCount);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// One iteration over every row's full vectors: reload |src1| per pass, then
// run the iteration with the state staged in dst rows.
template <bool isFmod, int32_t iterationNum, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
PTO_INTERNAL void TRemVStageSeqV(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, HpScratchA2A3& s, unsigned validRows,
    unsigned fullPerRow)
{
    (void)src0;
    constexpr unsigned elemsPerRepeat = REPEAT_BYTE / sizeof(float);
    if constexpr (iterationNum == 1) {
        // final iteration: the compensated subtract runs in chunks of
        // TREM_HP_FUSED_CHUNK_A2A3 vectors (its scratch window scales with
        // the chunk; see HpFusedSub). The chunk selects leave a count mask
        // behind; restore the full mask before each row's batched vabs.
        for (unsigned row = 0; row < validRows; ++row) {
            set_mask_norm();
            set_vector_mask(-1, -1);
            vabs(s.div, src1 + row * src1RowStride, fullPerRow, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            for (unsigned j = 0; j < fullPerRow; j += TREM_HP_FUSED_CHUNK_A2A3) {
                unsigned rep = fullPerRow - j;
                if (rep > TREM_HP_FUSED_CHUNK_A2A3) {
                    rep = TREM_HP_FUSED_CHUNK_A2A3;
                }
                // the previous chunk's selects leave a count mask behind;
                // dup/vmuls would be clipped to it, so restore the full mask
                set_mask_norm();
                set_vector_mask(-1, -1);
                HpSolveScaleV<isFmod, 1>(
                    dst + row * dstRowStride + j * elemsPerRepeat, s.div + j * elemsPerRepeat, s, rep);
            }
        }
    } else {
        for (unsigned row = 0; row < validRows; ++row) {
            vabs(s.div, src1 + row * src1RowStride, fullPerRow, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            HpSolveScaleV<isFmod, iterationNum>(dst + row * dstRowStride, s.div, s, fullPerRow);
        }
        TRemVStageSeqV<isFmod, iterationNum - 1, dstRowStride, src0RowStride, src1RowStride>(
            dst, src0, src1, s, validRows, fullPerRow);
    }
}

template <bool isFmod, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
PTO_INTERNAL void TRemVStagedA2A3(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* tmp, unsigned validRows,
    unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(float);
    unsigned fullPerRow = validCols / elementsPerRepeat;
    unsigned tailCount = validCols - fullPerRow * elementsPerRepeat;

    // repeatTimes is 8-bit: one row-batch instruction covers at most 255
    // full vectors (see docs/en/isa/TREMS.md for the resulting tiling caps)
    PTO_ASSERT(fullPerRow <= 255, "TREM HP staged: validCols exceeds 255 full vectors per row; split the tile.");

    HpScratchA2A3 s;
    HpScratchInitV(s, tmp, fullPerRow);

    // seed: dst = |src0| over every row's full vectors
    for (unsigned row = 0; row < validRows; ++row) {
        vabs(dst + row * dstRowStride, src0 + row * src0RowStride, fullPerRow, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    TRemVStageSeqV<isFmod, FMOD_ITERATION_NUM_MAX_A2A3, dstRowStride, src0RowStride, src1RowStride>(
        dst, src0, src1, s, validRows, fullPerRow);

    // postprocessing pass
    for (unsigned row = 0; row < validRows; ++row) {
        HpPostV(dst + row * dstRowStride, src0 + row * src0RowStride, src1 + row * src1RowStride, s, fullPerRow);
    }

    if (tailCount > 0) {
        // per-vector tails reuse the same tmp tile through the cap-1 layout
        HpScratchA2A3 s1;
        HpScratchInitV(s1, tmp, 1);
        for (unsigned row = 0; row < validRows; ++row) {
            TRemVHpVector<isFmod, dstRowStride, src0RowStride, src1RowStride>(
                dst + row * dstRowStride + fullPerRow * elementsPerRepeat,
                src0 + row * src0RowStride + fullPerRow * elementsPerRepeat,
                src1 + row * src1RowStride + fullPerRow * elementsPerRepeat, s1, tailCount);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// Formula: remainder(a, b) = a - floor(a/b) * b
// Note: For fp32, after computing remainder, we check if result * divider < 0.
//       If signs differ, we add divider to result to ensure the result has the same sign as divider.
template <typename T, unsigned dstRowStride>
struct RemOp {
    PTO_INTERNAL static void RemF32Instr(
        __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* tmp, unsigned repeatTimes,
        unsigned validCols)
    {
        set_mask_count();
        set_vector_mask(0, validCols);
        vdiv(tmp, src0, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322f32f(tmp, tmp, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // keep q*y and r in tmp: dst may alias src0, so dst must not be
        // written before src0's last read below (src1 never aliases dst)
        vmul(tmp, tmp, src1, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vsub(tmp, src0, tmp, 1, 1, 1, 1, 8, 8, 8); // r in tmp
        pipe_barrier(PIPE_V);
        vmul(dst, tmp, src1, 1, 1, 1, 1, 8, 8, 8); // sign probe r*y into dst
        pipe_barrier(PIPE_V);

        // vcmpvs_lt must run in norm mask mode with explicit repeatTimes: unlike
        // arithmetic ops, the compare op does not auto-extend a single repeat to
        // the whole count (CANN CompareScalarCompute passes ceil(count*sizeof(T)/256)
        // repeats), so repeat=1 would only emit the first 64 bits of the packed
        // mask and leave stale UB in the rest.
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, dst, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, validCols);

        vadd(dst, tmp, src1, 1, 1, 1, 1, 8, 8, 8); // candidate r + y into dst
        pipe_barrier(PIPE_V);

        // VSEL reads the packed bitmask via 64-bit address indirection: dup of
        // the 32-bit address puts garbage in the high half, so store the true
        // 64-bit address with a scalar UB store (flags around it). cmpMask
        // occupies dstRowStride..dstRowStride+maskFloats; addrBuf follows, 32B-aligned.
        unsigned maskBytes = CeilDivision(dstRowStride, BIT_TO_BYTE);
        unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint32_t* addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp + dstRowStride + maskFloats);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *(reinterpret_cast<__ubuf__ uint64_t*>(addrBuf)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBuf);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, validCols);
        vsel(dst, dst, tmp, 1, 1, 1, 1, 8, 8, 8, 2); // pick dst (r + y) where the mask bit is set
        pipe_barrier(PIPE_V);
    }

    // INT32 remainder in a strictly read-only-src form: the float-domain
    // dividend (row [0,R)), divisor (row [R,2R), reused by the epilogue probe)
    // and the quotient/rint chain (row [2R,3R)) all live in tmp. Neither src
    // tile is ever written; dst receives rint(a) as src0's last read
    // completes. tmp contract: 3*dstRowStride + maskF + 8 floats.
    PTO_INTERNAL static void RemInt32Instr(
        __ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1, __ubuf__ int32_t* tmp,
        unsigned repeatTimes, unsigned validCols)
    {
        __ubuf__ float* dst_f = reinterpret_cast<__ubuf__ float*>(dst);
        __ubuf__ float* rowA_f = reinterpret_cast<__ubuf__ float*>(tmp);
        __ubuf__ float* rowY_f = rowA_f + dstRowStride;
        __ubuf__ float* rowQ_f = rowY_f + dstRowStride;
        __ubuf__ int32_t* rowA_i = reinterpret_cast<__ubuf__ int32_t*>(rowA_f);
        __ubuf__ int32_t* rowQ_i = reinterpret_cast<__ubuf__ int32_t*>(rowQ_f);

        set_mask_count();
        set_vector_mask(0, validCols);

        vconv_s322f32(rowA_f, src0, 1, 1, 1, 8, 8); // rowA = float(a); src0 untouched
        vconv_s322f32(rowY_f, src1, 1, 1, 1, 8, 8); // rowY = float(y); src1 untouched
        pipe_barrier(PIPE_V);

        vconv_f322s32r(dst, rowA_f, 1, 1, 1, 8, 8); // dst = rint(a) (src0's last read above)
        pipe_barrier(PIPE_V);

        vdiv(rowQ_f, rowA_f, rowY_f, 1, 1, 1, 1, 8, 8, 8); // rowQ = q_f
        pipe_barrier(PIPE_V);
        vconv_f322s32f(rowQ_i, rowQ_f, 1, 1, 1, 8, 8); // floor quotient q (int)
        pipe_barrier(PIPE_V);
        vconv_f322s32r(rowA_i, rowY_f, 1, 1, 1, 8, 8); // y' = rint(y) int into rowA (a_f dead)
        pipe_barrier(PIPE_V);
        vmul(rowA_i, rowQ_i, rowA_i, 1, 1, 1, 1, 8, 8, 8); // rowA = q * y'
        pipe_barrier(PIPE_V);

        vsub(dst, dst, rowA_i, 1, 1, 1, 1, 8, 8, 8); // r = rint(a) - q*y'
        pipe_barrier(PIPE_V);

        vconv_s322f32(dst_f, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(rowQ_f, dst_f, rowY_f, 1, 1, 1, 1, 8, 8, 8); // sign probe: r * float(y)
        pipe_barrier(PIPE_V);

        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(rowQ_f + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, rowQ_f, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_count();
        set_vector_mask(0, validCols);

        vadd(rowQ_f, dst_f, rowY_f, 1, 1, 1, 1, 8, 8, 8); // candidate r + y
        pipe_barrier(PIPE_V);

        unsigned maskByte = CeilDivision(dstRowStride, BIT_TO_BYTE); // shape
        unsigned maskFloat = CeilDivision(maskByte, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint32_t* addrBufs = reinterpret_cast<__ubuf__ uint32_t*>(rowQ_f + dstRowStride + maskFloat);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *(reinterpret_cast<__ubuf__ uint64_t*>(addrBufs)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBufs);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, validCols);
        vsel(dst_f, rowQ_f, dst_f, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);

        vconv_f322s32z(dst, dst_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    PTO_INTERNAL static void RemInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, __ubuf__ T* tmp, unsigned repeatTimes, unsigned validCols)
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemF32Instr(dst, src0, src1, tmp, repeatTimes, validCols);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemInt32Instr(dst, src0, src1, tmp, repeatTimes, validCols);
        }
    }
};

template <
    auto PrecisionType, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
    unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride = dstRowStride,
    unsigned src1RowStride = dstRowStride>
__tf__ PTO_INTERNAL void TRem(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, typename TileDataTmp::TileDType __in__ tmp, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    if constexpr (
        PrecisionType == RemAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        // High precision remainder (issue #2): staged loop inversion when the
        // tile has enough full vectors and dst aliases neither input;
        // per-vector iterations otherwise (see the staged drivers above).
        if (TRemVStagedEligibleA2A3<TileDataDst, TileDataSrc0, TileDataSrc1>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(src0Ptr),
                reinterpret_cast<__ubuf__ float*>(src1Ptr), validRows, validCols)) {
            TRemVStagedA2A3<false, dstRowStride, src0RowStride, src1RowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(src0Ptr),
                reinterpret_cast<__ubuf__ float*>(src1Ptr), reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows,
                validCols);
        } else {
            TRemVHpRegister<false, dstRowStride, src0RowStride, src1RowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(src0Ptr),
                reinterpret_cast<__ubuf__ float*>(src1Ptr), reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows,
                validCols);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    constexpr unsigned tmpRowStride = TileDataTmp::RowStride;
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    uint16_t rowLimit = static_cast<uint16_t>(validRows);
    for (uint16_t i = 0; i < rowLimit; i++) {
        __ubuf__ T* dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T* s0Next = src0Ptr + i * src0RowStride;
        __ubuf__ T* s1Next = src1Ptr + i * src1RowStride;

        RemOp<T, dstRowStride>::RemInstr(dstNext, s0Next, s1Next, tmpPtr, repeatTimes, validCols);
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TRemCheck(
    const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1, const TileDataTmp& tmp)
{
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
        "Fix: TREM supports only float and int32 element types.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc0::DType>,
        "Fix: TREM type of dst must be same with src0 and src1.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TREM support only row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
    // Single-row tmp layout: [0, dstRowStride) candidate, [dstRowStride, dstRowStride+maskFloats)
    // packed bitmask (32-byte aligned), then an 8-element (32-byte) address buffer.
    unsigned maskBytes = CeilDivision(TileDataDst::RowStride, BIT_TO_BYTE);
    unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
    // the int path keeps the float-domain dividend, divisor and quotient
    // chain in tmp (both src tiles are read-only), so it needs two rows more
    // than the float path
    unsigned tmpRequiredCols =
        (std::is_same_v<T, int32_t> ? 3 * TileDataDst::RowStride : TileDataDst::RowStride) + maskFloats + BIT_TO_BYTE;
    PTO_ASSERT(
        tmp.GetValidRow() >= 1 && tmp.GetValidCol() >= tmpRequiredCols,
        "Fix: TREM tmp tile must have at least 1 row and enough columns for candidate, mask and address buffer.");
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    TRemCheck<T, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
    if constexpr (
        PrecisionType == RemAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        // HP scratch contract (vector divisor, HpScratchRequiredColsV above),
        // sized for the staged capacity (full vectors per row, >= 1)
        constexpr unsigned elemsPerRepeatHp = REPEAT_BYTE / sizeof(float);
        unsigned capVecs = dst.GetValidCol() / elemsPerRepeatHp;
        PTO_ASSERT(
            tmp.GetValidRow() >= 1 && tmp.GetValidCol() >= HpScratchRequiredColsV(capVecs > 0 ? capVecs : 1),
            "TREM HP: tmp tile too small for the high precision scratch contract (see docs/en/isa/TREMS.md).");
    }
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    TRem<
        PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp, elementsPerRepeat, blockSizeElem,
        dstRowStride, src0RowStride, src1RowStride>(
        dst.data(), src0.data(), src1.data(), tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
