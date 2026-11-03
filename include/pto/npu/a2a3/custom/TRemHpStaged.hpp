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
#ifndef TREMHPSTAGED_A2A3_HPP
#define TREMHPSTAGED_A2A3_HPP

#include <pto/common/constants.hpp>
#include "TFmodRemHp.hpp"

namespace pto {

// Loop-inverted high precision remainder (issue #2) for a2a3. Every full
// vector advances one SolveScale iteration per pass with the per-vector state
// staged in dst; the a2a3 form of the inversion batches each op of a pass
// over all full vectors of a row into one repeat instruction, where the
// per-vector path pays one instruction + pipe drain per 64 elements.
// a2a3 ops are UB-to-UB inside a single in-order vector pipe, so the A5
// mem_bar(VST_VLD) between passes is subsumed by the per-op pipe_barrier.
#ifndef TREM_HP_STAGED_MIN_VECS
#define TREM_HP_STAGED_MIN_VECS 48
#endif
constexpr uint32_t TREM_HP_STAGED_MIN_VECS_A2A3 = TREM_HP_STAGED_MIN_VECS;

template <typename T>
PTO_INTERNAL bool TRemHpStagedDisjointA2A3(const __ubuf__ T* a, uint32_t aElems, const __ubuf__ T* b, uint32_t bElems)
{
    return a + aElems <= b || b + bElems <= a;
}

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
    for (unsigned row = 0; row < validRows; ++row) {
        HpSolveScaleS<isFmod, iterationNum>(dst + row * dstRowStride, scalarAbs, divIsSub, s, fullPerRow);
    }
    if constexpr (iterationNum > 1) {
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
    // full vectors (see docs/trems_hp_a2a3.md for the resulting tiling caps)
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
    for (unsigned row = 0; row < validRows; ++row) {
        vabs(s.div, src1 + row * src1RowStride, fullPerRow, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        HpSolveScaleV<isFmod, iterationNum>(dst + row * dstRowStride, s.div, s, fullPerRow);
    }
    if constexpr (iterationNum > 1) {
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
    // full vectors (see docs/trems_hp_a2a3.md for the resulting tiling caps)
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

} // namespace pto

#endif // TREMHPSTAGED_A2A3_HPP
