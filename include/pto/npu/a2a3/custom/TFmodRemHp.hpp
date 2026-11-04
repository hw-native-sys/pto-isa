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
#ifndef TFMODREMHP_A2A3_HPP
#define TFMODREMHP_A2A3_HPP

#include <pto/common/constants.hpp>

namespace pto {

// High precision FP32 remainder (issue #2), ported to the a2a3 (Ascend910B1)
// UB-to-UB vector ISA. a2a3 has no vector register tensors: every operand is a
// UB buffer and consecutive dependent ops are separated by pipe_barrier, so
// the A5 register-path structure maps to per-vector sequences over UB, and the
// A5 "staged" loop inversion maps to per-iteration passes whose ops batch a
// whole row of full vectors into single repeat instructions (see
// custom/TRemHpStaged.hpp).
constexpr uint32_t FMOD_ITERATION_NUM_MAX_A2A3 = 11;

constexpr float FMOD_HP_SUBNORMAL = 1.1754944e-38f; // min normal, same threshold as A5
constexpr float FMOD_HP_MAXVALUE = 3.4028235e38f;

// Scaling schedule identical to the A5 implementation (a5/custom/TFmodRemHp.hpp).
constexpr float FMOD_HP_SCALE1[FMOD_ITERATION_NUM_MAX_A2A3] = {16777216.0f, 16777216.0f, 2.8147e14f, 4.7224e21f,
                                                               7.9228e28f,  1.3292e36f,  1.3292e36f, 1.3292e36f,
                                                               1.3292e36f,  1.3292e36f,  1.3292e36f};
constexpr float FMOD_HP_SCALE2[FMOD_ITERATION_NUM_MAX_A2A3] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 16777216.0f, 2.8147e14f, 4.7224e21f, 7.9228e28f, 1.3292e36f};
constexpr float FMOD_HP_POST_SCALE = 5.9604645e-8f; // 2^-24

// Scratch contract, in floats from the tmp tile base, for a capacity of C
// 64-element vectors (C = full vectors per row for the staged passes, 1 for
// the per-vector path). Buffers with disjoint lifetimes share storage:
//   [0, 64C)        bTmp     scaled divisor (dup'd from a host-computed
//                           |scalar|*scale product each iteration; the A5
//                           divisor broadcast buffer is not needed)
//   [64C, 128C)     k/t      quotient (f32 / s32 views) and, once the
//                           quotient is consumed, the general temp: k*bTmp
//                           product, sign factor, select operands, probe and
//                           correction candidate. k dies at the vmul that
//                           produces its product, and the probe dies at the
//                           vcmp that reads it, so one region serves both
//                           lifetimes via in-place ops (same dst==src0
//                           in-place idiom as TPow's vmul(dst, dst, exp)).
//   [128C, +maskF)  cmpMask  packed compare bitmask (32B aligned, padded)
//   [+maskF, +8)    addrBuf  64-bit cmpmask address buffer
//   cap == 1 only: [+8, +72) state per-vector state across the 11 iterations
//                  (staged passes keep state in dst; the old one/zero/sign
//                  buffers are gone - sign handling is compare+select on t)
struct HpScratchA2A3 {
    __ubuf__ float* bTmp;
    __ubuf__ float* k; // quotient and post-quotient temp share this region
    __ubuf__ uint8_t* cmpMask;
    __ubuf__ uint32_t* addrBuf;
    __ubuf__ float* state; // per-vector path (cap == 1) only
    __ubuf__ float* div;   // TREM only: |src1| row, live through one pass
};

PTO_INTERNAL inline unsigned HpScratchMaskFloats(unsigned capVecs)
{
    unsigned maskBytes = capVecs * BIT_TO_BYTE; // 64 bits per vector = 8 bytes
    return ((maskBytes + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE) * (BLOCK_BYTE_SIZE / 4);
}

PTO_INTERNAL inline unsigned HpScratchRequiredCols(unsigned capVecs)
{
    // the extra 64 floats of per-vector state keep the fallback's footprint
    // negligible instead of scaling with the staged capacity
    return 128 * capVecs + HpScratchMaskFloats(capVecs) + 8 + (capVecs == 1 ? 64 : 0);
}

PTO_INTERNAL inline void HpScratchInit(HpScratchA2A3& s, __ubuf__ float* tmp, unsigned capVecs)
{
    unsigned v = 64 * capVecs;
    s.bTmp = tmp;
    s.k = tmp + v;
    s.cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + 2 * v);
    s.addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp + 2 * v + HpScratchMaskFloats(capVecs));
    s.state = reinterpret_cast<__ubuf__ float*>(s.addrBuf + BIT_TO_BYTE);
    s.div = nullptr; // TREMS: the divisor is dup'd per iteration from scalars
}

// TREM (vector divisor) scratch: the |src1| row stays live through a whole
// pass, adding one more 64C region ahead of the TREMS layout:
//   [0, 64C) div, [64C, 128C) bTmp, [128C, 192C) k, then cmpMask/addrBuf,
//   and state (cap == 1 only) after them.
PTO_INTERNAL inline unsigned HpScratchRequiredColsV(unsigned capVecs)
{
    return 192 * capVecs + HpScratchMaskFloats(capVecs) + 8 + (capVecs == 1 ? 64 : 0);
}

PTO_INTERNAL inline void HpScratchInitV(HpScratchA2A3& s, __ubuf__ float* tmp, unsigned capVecs)
{
    unsigned v = 64 * capVecs;
    s.div = tmp;
    s.bTmp = tmp + v;
    s.k = tmp + 2 * v;
    s.cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + 3 * v);
    s.addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp + 3 * v + HpScratchMaskFloats(capVecs));
    s.state = reinterpret_cast<__ubuf__ float*>(s.addrBuf + BIT_TO_BYTE);
}

// set_cmpmask dereferences a full 64-bit address (see TRemS.hpp: duplicating
// the 32-bit address leaves garbage in the high half and vsel then reads the
// packed mask from a bogus address). Scalar UB store with V<->S flags, same
// idiom as pypto's arch32 select path. The caller has already produced
// s.cmpMask with a norm-mode compare.
PTO_INTERNAL inline void HpSetCmpMask(const HpScratchA2A3& s)
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
PTO_INTERNAL inline void HpSelect(
    __ubuf__ float* dst, __ubuf__ float* a, __ubuf__ float* b, const HpScratchA2A3& s, unsigned rep, unsigned lanes)
{
    HpSetCmpMask(s);
    set_mask_count();
    set_vector_mask(0, lanes);
    vsel(dst, a, b, rep, 1, 1, 1, 8, 8, 8, 2);
    pipe_barrier(PIPE_V);
}

// Scalar helpers without libm (device side): bit-level inf/nan tests.
PTO_INTERNAL inline bool HpScalarIsInf(float v)
{
    uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
    return (bits & 0x7FFFFFFFu) == 0x7F800000u;
}

PTO_INTERNAL inline float HpInf()
{
    uint32_t bits = 0x7F800000u;
    return *reinterpret_cast<float*>(&bits);
}

PTO_INTERNAL inline float HpNan()
{
    uint32_t bits = 0x7FC00000u;
    return *reinterpret_cast<float*>(&bits);
}

// k = roundToInt(k) as float (A5's vtrc: a2a3 converts through int32, which
// saturates far outside the algorithm's quotient range).
template <bool isFmod>
PTO_INTERNAL inline void HpRoundToInt(__ubuf__ float* k, unsigned rep)
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

// One SolveScale iteration over `rep` batched vectors with a scalar divisor.
// state is updated in place. The scaled divisor is dup'd from a host-computed
// |scalar|*scale product (same two-step rounding as A5's vmuls chain), so no
// divisor buffer stays live across iterations.
template <bool isFmod, int32_t iterationNum>
PTO_INTERNAL inline void HpSolveScaleS(
    __ubuf__ float* state, float scalarAbs, bool divIsSub, HpScratchA2A3& s, unsigned rep)
{
    // plain locals: the bisheng variadic intrinsic checker rejects
    // constexpr-qualified scalar operands
    float maxValue = FMOD_HP_MAXVALUE;
    float zeroV = 0.0f;
    unsigned lanes = rep * REPEAT_BYTE / sizeof(float);

    if constexpr (iterationNum == 1) {
        if (divIsSub) {
            // subnormal-divisor regime: lift divisor and state by 2^24 (the
            // post pass rescales by 2^-24)
            float lift = FMOD_HP_SCALE1[0];
            vmuls(state, state, lift, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vector_dup(s.bTmp, scalarAbs * lift, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        } else {
            vector_dup(s.bTmp, scalarAbs, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    } else {
        float bScale = scalarAbs * FMOD_HP_SCALE1[iterationNum - 1];
        if constexpr (iterationNum > 5) {
            bScale = bScale * FMOD_HP_SCALE2[iterationNum - 1];
        }
        vector_dup(s.bTmp, bScale, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    vdiv(s.k, state, s.bTmp, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    HpRoundToInt<isFmod>(s.k, rep);
    if constexpr (iterationNum != 1) {
        // not necessary to check for inf in the final iteration
        vmins(s.bTmp, s.bTmp, maxValue, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    // state -= k * bTmp. A5 spells this vneg(k) + vmula(state += -k*bTmp);
    // the manual decomposition keeps k positive and subtracts directly.
    vmul(s.k, s.k, s.bTmp, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(state, state, s.k, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

    if constexpr (iterationNum == 1) {
        // r = r + (r < 0) * bTmp, without dedicated one/zero/sign buffers:
        // t = mask ? bTmp : 0, then accumulate
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

// Full iteration chain 11 -> 1 on one state buffer (per-vector path).
template <bool isFmod, int32_t iterationNum>
PTO_INTERNAL inline void HpSolveScaleSeqS(
    __ubuf__ float* state, float scalarAbs, bool divIsSub, HpScratchA2A3& s, unsigned rep)
{
    HpSolveScaleS<isFmod, iterationNum>(state, scalarAbs, divIsSub, s, rep);
    if constexpr (iterationNum > 1) {
        HpSolveScaleSeqS<isFmod, iterationNum - 1>(state, scalarAbs, divIsSub, s, rep);
    }
}

// Postprocessing for a scalar divisor (TREMS): dividend sign restore,
// subnormal rescale, exception fixups and the remainder sign correction.
// Mirrors a5 TFmodRemHPPost<false> with the divisor-dependent branches
// resolved on the host (the scalar is known at kernel launch); t is reused
// across disjoint lifetimes (probe -> candidate) instead of a second temp.
PTO_INTERNAL inline void HpPostS(
    __ubuf__ float* state, __ubuf__ float* src0, float scalar, float absScalar, HpScratchA2A3& s, unsigned rep)
{
    unsigned lanes = rep * REPEAT_BYTE / sizeof(float);
    // sign restore: state = (src0 < 0) ? -state : state
    // (A5: multiply by 1 - 2*signbit(src0); same op count without the
    // sign-bit extraction buffers. vcmpvs_lt sees negative subnormals; -0.0
    // reads as 0, and the src0==0 exception fix overwrites that lane anyway.)
    set_mask_norm();
    set_vector_mask(-1, -1);
    vcmpvs_lt(s.cmpMask, src0, 0.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vmuls(s.k, state, -1.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.k, state, s, rep, lanes);

    // subnormal-divisor rescale: state *= 2^-24 when |scalar| <= min normal
    if (absScalar <= FMOD_HP_SUBNORMAL) {
        float postScale = FMOD_HP_POST_SCALE;
        vmuls(state, state, postScale, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    if (HpScalarIsInf(scalar)) {
        // x % +-inf: src0 (NaN when src0 is also +-inf). Order mirrors
        // SolveExceptionScenarios: src1-inf selects src0, then any +-inf
        // dividend selects NaN.
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_eq(s.cmpMask, src0, scalar, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(state, src0, state, s, rep, lanes);
        vabs(s.k, src0, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vcmpvs_eq(s.cmpMask, s.k, HpInf(), rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vector_dup(s.k, HpNan(), rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(state, s.k, state, s, rep, lanes);
    } else if (scalar != 0.0f) {
        // finite nonzero scalar: src0 = +-0 keeps src0 (sign of zero result)
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_eq(s.cmpMask, src0, 0.0f, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(state, src0, state, s, rep, lanes);
    }

    // remainder sign correction: if state * scalar < 0, state += scalar.
    // t holds the probe only until vcmp consumes it, then becomes the
    // candidate - no second temp buffer needed.
    vmuls(s.k, state, scalar, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(-1, -1);
    vcmpvs_lt(s.cmpMask, s.k, 0.0f, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(s.k, state, scalar, rep, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    HpSelect(state, s.k, state, s, rep, lanes);
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// ---- TREM (vector divisor) variants ----
// Same digit-peeling iteration, but the divisor comes from a |src1| row
// buffer and the subnormal regime is decided per element (A5 semantics:
// masked 2^24 lift of divisor and state where |div| <= min normal). The k
// region is free before the quotient, so it backs the lift temporaries.

// One SolveScale iteration over `rep` batched vectors with a vector divisor.
// `div` holds |src1| for the same lanes; state is updated in place.
template <bool isFmod, int32_t iterationNum>
PTO_INTERNAL inline void HpSolveScaleV(__ubuf__ float* state, __ubuf__ float* div, HpScratchA2A3& s, unsigned rep)
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

    // state -= k * bTmp (positive k; see HpSolveScaleS)
    vmul(s.k, s.k, s.bTmp, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vsub(state, state, s.k, rep, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);

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
PTO_INTERNAL inline void HpSolveScaleSeqV(__ubuf__ float* state, __ubuf__ float* div, HpScratchA2A3& s, unsigned rep)
{
    HpSolveScaleV<isFmod, iterationNum>(state, div, s, rep);
    if constexpr (iterationNum > 1) {
        HpSolveScaleSeqV<isFmod, iterationNum - 1>(state, div, s, rep);
    }
}

// Postprocessing with a vector divisor (TREM). Divisor precondition (same
// domain as A5): src1 is nonzero and not NaN - divisors of +-0 are outside
// the op's contract, and NaN divisors already propagate through the
// iteration arithmetic. Everything else mirrors a5 TFmodRemHPPost<false>:
// sign restore, per-element subnormal rescale, exception fixups, remainder
// sign correction.
PTO_INTERNAL inline void HpPostV(
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
    set_mask_norm();
    set_vector_mask(-1, -1);
}

} // namespace pto

#endif // TFMODREMHP_A2A3_HPP
