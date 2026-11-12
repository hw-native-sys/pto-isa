/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP
#define TREMS_HPP

#include <pto/common/constants.hpp>
#include "TRem.hpp"

namespace pto {
// ---- High precision remainder (issue #2), TREMS (scalar divisor) ----
// Scalar-divisor variants of the TRem.hpp machinery: the scaled divisor is
// dup'd per iteration from a host-computed |scalar|*scale product (no
// divisor buffer lives across iterations) and the divisor-dependent special
// cases are resolved on the host.
PTO_INTERNAL unsigned HpScratchRequiredCols(unsigned capVecs)
{
    // the extra 64 floats of per-vector state keep the fallback's footprint
    // negligible instead of scaling with the staged capacity; the final term
    // is the compensation window of the final-iteration HpFusedSub (eight
    // vectors per batched chunk, clamped to the row's vector count)
    unsigned chunk = capVecs < TREM_HP_FUSED_CHUNK_A2A3 ? capVecs : TREM_HP_FUSED_CHUNK_A2A3;
    return 128 * capVecs + HpScratchMaskFloats(capVecs) + 8 + (capVecs == 1 ? 64 : 0) + 8 * 64 * chunk;
}

PTO_INTERNAL void HpScratchInit(HpScratchA2A3& s, __ubuf__ float* tmp, unsigned capVecs)
{
    unsigned v = 64 * capVecs;
    s.bTmp = tmp;
    s.k = tmp + v;
    s.cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + 2 * v);
    s.addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(tmp + 2 * v + HpScratchMaskFloats(capVecs));
    s.state = reinterpret_cast<__ubuf__ float*>(s.addrBuf + BIT_TO_BYTE);
    s.div = nullptr; // TREMS: the divisor is dup'd per iteration from scalars
    s.comp = s.state + (capVecs == 1 ? 64 : 0);
}

// One SolveScale iteration over `rep` batched vectors with a scalar divisor.
// state is updated in place. The scaled divisor is dup'd from a host-computed
// |scalar|*scale product (same two-step rounding as A5's vmuls chain), so no
// divisor buffer stays live across iterations.
template <bool isFmod, int32_t iterationNum>
PTO_INTERNAL void HpSolveScaleS(__ubuf__ float* state, float scalarAbs, bool divIsSub, HpScratchA2A3& s, unsigned rep)
{
    // plain locals: the bisheng variadic intrinsic checker rejects
    // constexpr-qualified scalar operands
    float maxValue = FMOD_HP_MAXVALUE;
    float zeroV = 0.0f;
    unsigned lanes = rep * REPEAT_BYTE / sizeof(float);

    if constexpr (iterationNum == 1) {
        float one = 1.0f;
        if (divIsSub) {
            // subnormal-divisor regime: lift divisor and state by 2^24 (the
            // post pass rescales by 2^-24)
            float lift = FMOD_HP_SCALE1[0];
            vmuls(state, state, lift, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vector_dup(s.bTmp, one, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(s.bTmp, s.bTmp, scalarAbs * lift, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        } else {
            // dup(1) + vmuls: vector_dup of an +-inf scalar poisons to NaN
            vector_dup(s.bTmp, one, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(s.bTmp, s.bTmp, scalarAbs, rep, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
    } else {
        float bScale = scalarAbs * FMOD_HP_SCALE1[iterationNum - 1];
        if constexpr (iterationNum > 5) {
            bScale = bScale * FMOD_HP_SCALE2[iterationNum - 1];
        }
        float one = 1.0f;
        vector_dup(s.bTmp, one, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(s.bTmp, s.bTmp, bScale, rep, 1, 1, 8, 8);
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

    if constexpr (iterationNum == 1) {
        // final iteration: compensated single-rounding subtract. A5 computes
        // state -= k * bTmp as one fused vmula; a2a3 has no fused op, so the
        // plain vmul+vsub pair loses the product's low bits whenever k*bTmp
        // needs more than 24 significand bits (see HpFusedSub in TRem.hpp).
        HpFusedSub(state, s.k, s.bTmp, s, rep);
    } else {
        // state -= k * bTmp. A5 spells this vneg(k) + vmula(state += -k*bTmp);
        // the manual decomposition keeps k positive and subtracts directly.
        vmul(s.k, s.k, s.bTmp, rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vsub(state, state, s.k, rep, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

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
PTO_INTERNAL void HpSolveScaleSeqS(
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
PTO_INTERNAL void HpPostS(
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
    // candidate - no second temp buffer needed. Skipped for an inf scalar:
    // x % +-inf = x (the exception chain above already fixed those lanes),
    // and the probe r*(+-inf) overflows to +-inf and would corrupt the pick.
    if (!HpScalarIsInf(scalar)) {
        vmuls(s.k, state, scalar, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(s.cmpMask, s.k, 0.0f, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(s.k, state, scalar, rep, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSelect(state, s.k, state, s, rep, lanes);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// x % +-inf short-circuit: dst = |src| inf ? NaN : src (fmod semantics).
// The per-iteration divisor dups and the sign-correction probe both
// misbehave when the scalar itself is +-inf, so an inf scalar never enters
// the iteration chain.
template <unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void TRemSInfScalar(
    __ubuf__ float* dst, __ubuf__ float* src, __ubuf__ float* tmp, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(float);
    unsigned fullPerRow = validCols / elementsPerRepeat;
    unsigned tailCount = validCols - fullPerRow * elementsPerRepeat;
    HpScratchA2A3 s;
    HpScratchInit(s, tmp, 1);
    for (unsigned row = 0; row < validRows; ++row) {
        __ubuf__ float* dstRow = dst + row * dstRowStride;
        __ubuf__ float* srcRow = src + row * srcRowStride;
        for (unsigned j = 0; j <= fullPerRow; ++j) {
            unsigned count = (j < fullPerRow) ? elementsPerRepeat : tailCount;
            if (count == 0) {
                break;
            }
            __ubuf__ float* d = dstRow + j * elementsPerRepeat;
            __ubuf__ float* x = srcRow + j * elementsPerRepeat;
            set_mask_norm();
            set_vector_mask(-1, -1);
            vabs(s.k, x, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vcmpvs_eq(s.cmpMask, s.k, HpInf(), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vector_dup(s.k, HpNan(), 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            HpSelect(d, s.k, x, s, 1, count);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// ---- Staged TREMS drivers ----
// Staging reuses dst as scratch, so it needs enough full vectors to amortize
// the extra seed/fixup passes and must not alias the input.
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL bool TRemSStagedEligibleA2A3(
    const __ubuf__ float* dst, const __ubuf__ float* src, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = REPEAT_BYTE / sizeof(float);
    if (validRows * (validCols / elemsPerRepeat) < TREM_HP_STAGED_MIN_VECS_A2A3) {
        return false;
    }
    return TRemHpStagedDisjointA2A3<float>(
        dst, TileDataDst::Rows * TileDataDst::Cols, src, TileDataSrc::Rows * TileDataSrc::Cols);
}

// One full-vector row through all 11 iterations with the state kept in a
// scratch buffer, then postprocessing and one masked write to dst. This is
// the per-vector ("register path") organization: it is the fallback below the
// staging threshold, for aliased tiles, and for each row's tail.
template <bool isFmod, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void TRemSHpVector(
    __ubuf__ float* dstRow, __ubuf__ float* srcRow, float scalar, float absScalar, bool divIsSub, HpScratchA2A3& s,
    unsigned count)
{
    set_mask_count();
    set_vector_mask(0, count);
    vabs(s.state, srcRow, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);

    HpSolveScaleSeqS<isFmod, FMOD_ITERATION_NUM_MAX_A2A3>(s.state, absScalar, divIsSub, s, 1);
    HpPostS(s.state, srcRow, scalar, absScalar, s, 1);

    set_mask_count();
    set_vector_mask(0, count);
    vadds(dstRow, s.state, 0.0f, 1, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
}

// Per-vector path over every full vector and tail of the tile.
template <bool isFmod, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void TRemSHpRegister(
    __ubuf__ float* dst, __ubuf__ float* src, float scalar, __ubuf__ float* tmp, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(float);
    unsigned fullPerRow = validCols / elementsPerRepeat;
    unsigned tailCount = validCols - fullPerRow * elementsPerRepeat;
    float absScalar = scalar < 0.0f ? -scalar : scalar;
    bool divIsSub = absScalar <= FMOD_HP_SUBNORMAL;

    HpScratchA2A3 s;
    HpScratchInit(s, tmp, 1);

    for (unsigned row = 0; row < validRows; ++row) {
        __ubuf__ float* dstRow = dst + row * dstRowStride;
        __ubuf__ float* srcRow = src + row * srcRowStride;
        for (unsigned j = 0; j < fullPerRow; ++j) {
            TRemSHpVector<isFmod, dstRowStride, srcRowStride>(
                dstRow + j * elementsPerRepeat, srcRow + j * elementsPerRepeat, scalar, absScalar, divIsSub, s,
                elementsPerRepeat);
        }
        if (tailCount > 0) {
            TRemSHpVector<isFmod, dstRowStride, srcRowStride>(
                dstRow + fullPerRow * elementsPerRepeat, srcRow + fullPerRow * elementsPerRepeat, scalar, absScalar,
                divIsSub, s, tailCount);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// One iteration over every row's full vectors (state staged in dst rows).
template <bool isFmod, int32_t iterationNum, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void TRemSStageSeqS(
    __ubuf__ float* dst, float scalarAbs, bool divIsSub, HpScratchA2A3& s, unsigned validRows, unsigned fullPerRow)
{
    constexpr unsigned elemsPerRepeat = REPEAT_BYTE / sizeof(float);
    if constexpr (iterationNum == 1) {
        // final iteration: the compensated subtract runs in chunks of
        // TREM_HP_FUSED_CHUNK_A2A3 vectors (its scratch window scales with
        // the chunk; see HpFusedSub)
        for (unsigned row = 0; row < validRows; ++row) {
            for (unsigned j = 0; j < fullPerRow; j += TREM_HP_FUSED_CHUNK_A2A3) {
                unsigned rep = fullPerRow - j;
                if (rep > TREM_HP_FUSED_CHUNK_A2A3) {
                    rep = TREM_HP_FUSED_CHUNK_A2A3;
                }
                // the previous chunk's selects leave a count mask behind;
                // dup/vmuls would be clipped to it, so restore the full mask
                set_mask_norm();
                set_vector_mask(-1, -1);
                HpSolveScaleS<isFmod, 1>(dst + row * dstRowStride + j * elemsPerRepeat, scalarAbs, divIsSub, s, rep);
            }
        }
    } else {
        for (unsigned row = 0; row < validRows; ++row) {
            HpSolveScaleS<isFmod, iterationNum>(dst + row * dstRowStride, scalarAbs, divIsSub, s, fullPerRow);
        }
        TRemSStageSeqS<isFmod, iterationNum - 1, dstRowStride, srcRowStride>(
            dst, scalarAbs, divIsSub, s, validRows, fullPerRow);
    }
}

// Staged path: seed each row's state from |src0| once, then run one iteration
// over all full vectors per pass (ops batched over the row's full vectors),
// then the postprocessing pass, then tails through the per-vector path.
// Scratch is a capacity-sized window the passes reuse: the per-iteration
// divisor is dup'd from scalars and no buffer stays live across passes, so
// the UB footprint no longer carries redundant worst-case-simultaneous
// buffers (see the liveness notes in TFmodRemHp.hpp).
template <bool isFmod, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void TRemSStagedA2A3(
    __ubuf__ float* dst, __ubuf__ float* src, float scalar, __ubuf__ float* tmp, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(float);
    unsigned fullPerRow = validCols / elementsPerRepeat;
    unsigned tailCount = validCols - fullPerRow * elementsPerRepeat;
    float absScalar = scalar < 0.0f ? -scalar : scalar;
    bool divIsSub = absScalar <= FMOD_HP_SUBNORMAL;

    // repeatTimes is 8-bit: one row-batch instruction covers at most 255
    // full vectors (see docs/isa/TREMS.md for the resulting tiling caps)
    PTO_ASSERT(fullPerRow <= 255, "TREMS HP staged: validCols exceeds 255 full vectors per row; split the tile.");

    HpScratchA2A3 s;
    HpScratchInit(s, tmp, fullPerRow);

    // seed: dst = |src0| over every row's full vectors
    for (unsigned row = 0; row < validRows; ++row) {
        vabs(dst + row * dstRowStride, src + row * srcRowStride, fullPerRow, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    TRemSStageSeqS<isFmod, FMOD_ITERATION_NUM_MAX_A2A3, dstRowStride, srcRowStride>(
        dst, absScalar, divIsSub, s, validRows, fullPerRow);

    // postprocessing pass
    for (unsigned row = 0; row < validRows; ++row) {
        __ubuf__ float* dstRow = dst + row * dstRowStride;
        HpPostS(dstRow, src + row * srcRowStride, scalar, absScalar, s, fullPerRow);
    }

    if (tailCount > 0) {
        // per-vector tails reuse the same tmp tile through the cap-1 layout
        HpScratchA2A3 s1;
        HpScratchInit(s1, tmp, 1);
        for (unsigned row = 0; row < validRows; ++row) {
            TRemSHpVector<isFmod, dstRowStride, srcRowStride>(
                dst + row * dstRowStride + fullPerRow * elementsPerRepeat,
                src + row * srcRowStride + fullPerRow * elementsPerRepeat, scalar, absScalar, divIsSub, s1, tailCount);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

// Formula: remainder(a, b) = a - floor(a/b) * b
// Note: For fp32, after computing remainder, we check if result * scalar < 0.
//       If signs differ, we add scalar to result to ensure the result has the same sign as scalar.
//
// tmp layout (single row, columns measured in T elements):
//   [0, dstRowStride)                    sign-correction candidate (dst + x)
//   [dstRowStride, dstRowStride+maskF)   packed compare bitmask, 32-byte aligned
//   [dstRowStride+maskF, +8)             cmpmask address buffer (64-bit)
struct RemSOp {
    // vcmpvs_lt must run in norm mask mode with explicit repeatTimes: a single
    // repeat only emits the first 64 bits of the packed mask and leaves stale
    // UB beyond element 64, so vsel would pick stale bits for validCols > 64.
    // VSEL_TENSOR_TENSOR_MODE (2) reads the packed bitmask through the
    // two-level address indirection (same convention as TRem / TSel).
    PTO_INTERNAL static void RemSF32Instr(
        __ubuf__ float* dst, __ubuf__ float* src, float x, __ubuf__ float* tmp, unsigned dstRowStride,
        unsigned repeatTimes, unsigned validCols)
    {
        // Step 1: tmp = x (broadcast scalar)
        vector_dup(tmp, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 2: tmp = src / x
        vdiv(tmp, src, tmp, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 3: tmp = floor(tmp) - truncate towards zero
        vconv_f322f32f(tmp, tmp, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 4: tmp = tmp * x = floor(src/x) * x
        // (kept in tmp: dst may alias src, so dst must not be written before
        // src's last read below)
        vmuls(tmp, tmp, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 5: tmp = src - tmp = src - floor(src/x) * x (remainder r)
        vsub(tmp, src, tmp, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // Sign correction: if r * scalar < 0, then r += scalar.
        // src is fully consumed, so dst is safe to use as the probe register.
        // Step 6: dst = r * x (check if signs differ)
        vmuls(dst, tmp, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        unsigned maskBytes = CeilDivision(dstRowStride, BIT_TO_BYTE);
        unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(tmp + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, dst, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // set_cmpmask dereferences a full 64-bit address. Duplicating the 32-bit
        // address with vector_dup fills both halves ((addr<<32)|addr), so vsel
        // reads the packed mask from a garbage address (observed as effectively
        // random per-element selects). Store the true 64-bit address with a
        // scalar UB store, flags around it (pypto arch32 select idiom).
        __ubuf__ uint64_t* addrBuf = reinterpret_cast<__ubuf__ uint64_t*>(tmp + dstRowStride + maskFloats);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *addrBuf = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBuf);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0, validCols);
        // Step 7: candidate = r + x into dst[0..validCols)
        vadds(dst, tmp, x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // Step 8: vsel mode 2 — pick dst (r + x) where the packed mask bit is
        // set, tmp (r) otherwise
        vsel(dst, dst, tmp, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);
    }

    // INT32 remainder in a strictly read-only-src form: the float-domain
    // dividend and the divisor/quotient chain live in tmp (rows [0,R) and
    // [R,2R)), src is never written, and dst receives rint(a) as src's last
    // read completes. tmp contract: 2*dstRowStride + maskF + 8 floats.
    PTO_INTERNAL static void RemSInt32Instr(
        __ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t x, __ubuf__ int32_t* tmp, unsigned dstRowStride,
        unsigned repeatTimes, unsigned validCols)
    {
        __ubuf__ float* dst_f = reinterpret_cast<__ubuf__ float*>(dst);
        __ubuf__ float* rowA_f = reinterpret_cast<__ubuf__ float*>(tmp);
        __ubuf__ float* rowQ_f = rowA_f + dstRowStride;
        __ubuf__ int32_t* rowQ_i = reinterpret_cast<__ubuf__ int32_t*>(rowQ_f);

        vconv_s322f32(rowA_f, src, 1, 1, 1, 8, 8); // rowA = float(a); src untouched
        vector_dup(rowQ_f, (float)x, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vconv_f322s32r(dst, rowA_f, 1, 1, 1, 8, 8); // dst = rint(a) (src's last read above)
        pipe_barrier(PIPE_V);

        vdiv(rowQ_f, rowA_f, rowQ_f, 1, 1, 1, 1, 8, 8, 8); // rowQ = q_f (divisor consumed)
        pipe_barrier(PIPE_V);
        vconv_f322s32f(rowQ_i, rowQ_f, 1, 1, 1, 8, 8); // floor quotient q (int)
        pipe_barrier(PIPE_V);
        vmuls(rowQ_i, rowQ_i, x, 1, 1, 1, 8, 8); // q * x (int)
        pipe_barrier(PIPE_V);
        vsub(dst, dst, rowQ_i, 1, 1, 1, 1, 8, 8, 8); // r = rint(a) - q*x
        pipe_barrier(PIPE_V);

        vconv_s322f32(dst_f, dst, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(rowQ_f, dst_f, (float)x, 1, 1, 1, 8, 8); // sign probe: r * x
        pipe_barrier(PIPE_V);

        unsigned maskBytes = CeilDivision(dstRowStride, BIT_TO_BYTE);
        unsigned maskFloats = CeilDivision(maskBytes, BLOCK_BYTE_SIZE) * BIT_TO_BYTE;
        __ubuf__ uint8_t* cmpMask = reinterpret_cast<__ubuf__ uint8_t*>(rowQ_f + dstRowStride);
        set_mask_norm();
        set_vector_mask(-1, -1);
        vcmpvs_lt(cmpMask, rowQ_f, 0.0f, repeatTimes, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        __ubuf__ uint32_t* addrBuf = reinterpret_cast<__ubuf__ uint32_t*>(rowQ_f + dstRowStride + maskFloats);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        *(reinterpret_cast<__ubuf__ uint64_t*>(addrBuf)) = static_cast<uint64_t>(reinterpret_cast<int64_t>(cmpMask));
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        set_cmpmask(addrBuf);
        pipe_barrier(PIPE_V);

        set_mask_count();
        set_vector_mask(0, validCols);
        vadds(rowQ_f, dst_f, (float)x, 1, 1, 1, 8, 8); // candidate r + x
        pipe_barrier(PIPE_V);

        vsel(dst_f, rowQ_f, dst_f, 1, 1, 1, 1, 8, 8, 8, 2);
        pipe_barrier(PIPE_V);

        vconv_f322s32z(dst, dst_f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
};

template <
    auto PrecisionType, typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL void TRemS(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataDst::DType x, typename TileDataTmp::TileDType __in__ tmp, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;

    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    if constexpr (
        PrecisionType == RemSAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        // High precision remainder (issue #2): staged loop inversion when the
        // tile has enough full vectors and dst does not alias src, per-vector
        // iterations otherwise (see the staged drivers above). An inf scalar
        // short-circuits: x % +-inf = x (NaN where |x| = inf) - the iteration
        // chain's divisor dups and sign-correction probe misbehave with it.
        if (HpScalarIsInf((float)x)) {
            TRemSInfScalar<dstRowStride, srcRowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr),
                reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows, validCols);
            set_mask_norm();
            set_vector_mask(-1, -1);
            return;
        }
        unsigned capVecs = validCols / elementsPerRepeat;
        if (capVecs == 0) {
            capVecs = 1; // per-vector path scratch
        }
        if (TRemSStagedEligibleA2A3<TileDataDst, TileDataSrc>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr), validRows,
                validCols)) {
            TRemSStagedA2A3<false, dstRowStride, srcRowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr), (float)x,
                reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows, validCols);
        } else {
            TRemSHpRegister<false, dstRowStride, srcRowStride>(
                reinterpret_cast<__ubuf__ float*>(dstPtr), reinterpret_cast<__ubuf__ float*>(srcPtr), (float)x,
                reinterpret_cast<__ubuf__ float*>(tmpPtr), validRows, validCols);
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
        return;
    }

    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    set_mask_count();
    set_vector_mask(0, validCols);
    for (int i = 0; i < (int)validRows; ++i) {
        __ubuf__ T* dstNext = dstPtr + i * dstRowStride;
        __ubuf__ T* s0Next = srcPtr + i * srcRowStride;
        // Note: tmp buffer is reused for each row iteration to save memory
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, float32_t>) {
            RemSOp::RemSF32Instr(dstNext, s0Next, x, tmpPtr, dstRowStride, repeatTimes, validCols);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            RemSOp::RemSInt32Instr(dstNext, s0Next, x, tmpPtr, dstRowStride, repeatTimes, validCols);
        } else {
            static_assert(sizeof(T) == 0, "TREMS: Unsupported tile DType.");
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;

    // static assertions
    static_assert(
        std::is_same_v<T, typename TileDataSrc::DType>, "TREMS: The data types of dst and src must be the same.");
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float32_t> || std::is_same_v<T, int32_t>,
        "Fix: TREMS supports only float and int32 element types.");
    static_assert(
        TileDataDst::Loc == TileType::Vec && TileDataSrc::Loc == TileType::Vec,
        "TREMS: TileType of dst and src tiles must be TileType::Vec.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "TREMS: Only support row major layout.");

    // dynamic checks
    PTO_ASSERT(
        dst.GetValidRow() == src.GetValidRow() && dst.GetValidRow() > 0,
        "TREMS: Number of valid rows of dst and src must be the same, and both greater than 0.");
    PTO_ASSERT(
        dst.GetValidCol() == src.GetValidCol() && dst.GetValidCol() > 0,
        "TREMS: Number of valid columns of dst and src must be the same, and all greater than 0.");
    // Single-row tmp layout: [0, dstRowStride) sign-correction candidate,
    // [dstRowStride, dstRowStride+maskFloats) packed bitmask (32-byte aligned),
    // then an 8-element (32-byte) address buffer.
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    // BIT_TO_BYTE bits per byte, packed mask rounded up to BLOCK_BYTE_SIZE bytes
    constexpr unsigned maskBytes = (dstRowStride + BIT_TO_BYTE - 1) / BIT_TO_BYTE;
    constexpr unsigned maskFloats = ((maskBytes + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE) * (BLOCK_BYTE_SIZE / 4);
    // the int path keeps the float-domain dividend and the quotient chain in
    // tmp (src is read-only), so it needs one row more than the float path
    constexpr unsigned tmpRequiredCols =
        (std::is_same_v<T, int32_t> ? 2 * dstRowStride : dstRowStride) + maskFloats + BIT_TO_BYTE;
    // The high precision path needs the larger scratch contract from
    // HpScratchRequiredCols above, sized for the staged capacity (full vectors per
    // row, at least 1 for the per-vector fallback).
    unsigned hpRequiredCols = 0;
    if constexpr (
        PrecisionType == RemSAlgorithm::HIGH_PRECISION && (std::is_same_v<T, float> || std::is_same_v<T, float32_t>)) {
        constexpr unsigned elemsPerRepeatHp = REPEAT_BYTE / sizeof(float);
        unsigned capVecs = dst.GetValidCol() / elemsPerRepeatHp;
        hpRequiredCols = HpScratchRequiredCols(capVecs > 0 ? capVecs : 1);
    }
    PTO_ASSERT(
        tmp.GetValidRow() >= 1 &&
            tmp.GetValidCol() >= (tmpRequiredCols > hpRequiredCols ? tmpRequiredCols : hpRequiredCols),
        "TREMS: tmp tile must have 1 row and enough columns for candidate, mask and address buffer.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    TRemS<
        PrecisionType, TileDataDst, TileDataSrc, TileDataTmp, elementsPerRepeat, blockSizeElem, dstRowStride,
        srcRowStride>(dst.data(), src.data(), scalar, tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif
