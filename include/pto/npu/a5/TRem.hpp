/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>
#include "TDiv.hpp"

namespace pto {
// ---- High precision remainder/fmod core (issue #2), shared by TFMOD/TFMODS and TREM/TREMS ----
constexpr uint32_t FMOD_ITERATION_NUM_MAX = 11;
constexpr FloatUnion inf(0x7f800000);
constexpr FloatUnion negInf(0xff800000);
constexpr FloatUnion nan(0x7fc00000);

constexpr FloatUnion scaleList1[FMOD_ITERATION_NUM_MAX] = {
    FloatUnion(0x4b800000), FloatUnion(0x4b800000), FloatUnion(0x57800000), FloatUnion(0x63800000),
    FloatUnion(0x6f800000), FloatUnion(0x7b800000), FloatUnion(0x7b800000), FloatUnion(0x7b800000),
    FloatUnion(0x7b800000), FloatUnion(0x7b800000), FloatUnion(0x7b800000)};
constexpr FloatUnion scaleList2[FMOD_ITERATION_NUM_MAX] = {
    FloatUnion(0x3f800000), FloatUnion(0x3f800000), FloatUnion(0x3f800000), FloatUnion(0x3f800000),
    FloatUnion(0x3f800000), FloatUnion(0x3f800000), FloatUnion(0x4b800000), FloatUnion(0x57800000),
    FloatUnion(0x63800000), FloatUnion(0x6f800000), FloatUnion(0x7b800000)};

PTO_INTERNAL void GetSignBit(RegTensor<float>& dstReg, RegTensor<float>& srcReg, MaskReg& mask)
{
    constexpr int16_t signRightNum = BLOCK_BYTE_SIZE - 1;
    RegTensor<uint32_t> oneReg, tmpReg;
    uint32_t len32 = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(float));
    MaskReg preg_b32 = CreatePredicate<float>(len32);
    vdup(oneReg, 1, mask, MODE_ZEROING);
    vshrs(tmpReg, (RegTensor<uint32_t>&)srcReg, signRightNum, mask, MODE_ZEROING);
    vand(tmpReg, tmpReg, oneReg, mask, MODE_ZEROING);
    vcvt(dstReg, (RegTensor<int32_t>&)tmpReg, preg_b32, ROUND_R);
}

template <typename RoundMode, int32_t iterationNum>
PTO_INTERNAL void SolveScale(
    RegTensor<float>& dstReg, RegTensor<float>& srcReg, const float scale1, const float scale2, MaskReg& mask)
{
    constexpr float maxValue = 3.4028235e38;
    constexpr float subNormal = 1.1754944e-38;

    MaskReg subNormalMask;
    RegTensor<float> bTmpReg, tmpReg, kReg, signReg;

    if constexpr (iterationNum == 1) { // iter 1 (last iteration) handles subnormal case
        vcmps_le(subNormalMask, srcReg, subNormal, mask);
        vmuls(tmpReg, srcReg, scale1, subNormalMask, MODE_ZEROING);
        vsel(bTmpReg, tmpReg, srcReg, subNormalMask);
        vmuls(tmpReg, dstReg, scale1, subNormalMask, MODE_ZEROING);
        vsel(dstReg, tmpReg, dstReg, subNormalMask);
        vdiv(kReg, dstReg, bTmpReg, mask, MODE_ZEROING);
        vtrc(kReg, kReg, RoundMode(), mask);
    } else {
        vmuls(bTmpReg, srcReg, scale1, mask, MODE_ZEROING);
        if constexpr (iterationNum > 5) { // last 5 iterations do not need extra scaling
            vmuls(bTmpReg, bTmpReg, scale2, mask, MODE_ZEROING);
        }
        vdiv(kReg, dstReg, bTmpReg, mask, MODE_ZEROING);
        vtrc(kReg, kReg, RoundMode(), mask);
    }

    // not necessary to check for inf in the final iteration
    if constexpr (iterationNum != 1) {
        vmins(bTmpReg, bTmpReg, maxValue, mask, MODE_ZEROING);
    }
    vneg(kReg, kReg, mask, MODE_ZEROING);
    // res = -k * bTmp + y
    vmula(dstReg, kReg, bTmpReg, mask, MODE_ZEROING);

    if constexpr (iterationNum == 1) { // iter 1 handles subnormal case
        // r = r + np.float32(np.signbit(r)) * bTmp
        GetSignBit(signReg, dstReg, mask);
        vmul(signReg, signReg, bTmpReg, mask, MODE_ZEROING);
        vadd(dstReg, dstReg, signReg, mask, MODE_ZEROING);
    }
}

// recurse from itermax to 1
template <typename RoundMode, int32_t iterationNum>
PTO_INTERNAL void SolveScaleIter(RegTensor<float>& dstReg, RegTensor<float>& srcReg, MaskReg& mask)
{
    SolveScale<RoundMode, iterationNum>(
        dstReg, srcReg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, mask);

    if constexpr (iterationNum > 1) {
        SolveScaleIter<RoundMode, iterationNum - 1>(dstReg, srcReg, mask);
    }
}

PTO_INTERNAL void SolveExceptionScenarios(
    RegTensor<float>& dstReg, RegTensor<float>& src0Reg, RegTensor<float>& src1Reg, RegTensor<float>& nanReg,
    MaskReg& mask)
{
    MaskReg src0Is0CmpReg, src0IsNeg0CmpReg, src0InfCmpReg, src0NegInfCmpReg;
    MaskReg src1Not0CmpReg, src1NotNeg0CmpReg, src1NotNanCmpReg, src1InfCmpReg, src1NegInfCmpReg;
    MaskReg srcBothInfCmpReg;

    vcmps_eq(src1InfCmpReg, src1Reg, inf.f, mask);
    vsel(dstReg, src0Reg, dstReg, src1InfCmpReg);
    vcmps_eq(src1NegInfCmpReg, src1Reg, negInf.f, mask);
    vsel(dstReg, src0Reg, dstReg, src1NegInfCmpReg);
    vcmps_eq(src0InfCmpReg, src0Reg, inf.f, mask);
    vcmps_eq(src0NegInfCmpReg, src0Reg, negInf.f, mask);
    por(src0InfCmpReg, src0InfCmpReg, src0NegInfCmpReg, mask);
    por(src1InfCmpReg, src1InfCmpReg, src1NegInfCmpReg, mask);
    pand(srcBothInfCmpReg, src0InfCmpReg, src1InfCmpReg, mask);
    vsel(dstReg, nanReg, dstReg, srcBothInfCmpReg);
    vcmps_eq(src0Is0CmpReg, src0Reg, static_cast<float>(0), mask);
    vcmps_eq(src0IsNeg0CmpReg, src0Reg, static_cast<float>(-0), mask);
    por(src0Is0CmpReg, src0Is0CmpReg, src0IsNeg0CmpReg, mask);

    vcmps_ne(src1Not0CmpReg, src1Reg, static_cast<float>(0), mask);
    vcmps_ne(src1NotNeg0CmpReg, src1Reg, static_cast<float>(-0), mask);
    vcmp_ne(src1NotNanCmpReg, src1Reg, src1Reg, mask);
    pnot(src1NotNanCmpReg, src1NotNanCmpReg, mask);

    por(src1Not0CmpReg, src1Not0CmpReg, src1NotNeg0CmpReg, mask);
    pand(src1Not0CmpReg, src1Not0CmpReg, src1NotNanCmpReg, mask);

    pand(src0Is0CmpReg, src0Is0CmpReg, src1Not0CmpReg, mask);
    vsel(dstReg, src0Reg, dstReg, src0Is0CmpReg);
}

// Post-SolveScale fixups shared by the register and staged kernels:
// dividend sign restore, subnormal rescale, exceptions, rem sign correction.
template <bool isFmod>
PTO_INTERNAL void TFmodRemHPPost(
    RegTensor<float>& dstReg, RegTensor<float>& src0Reg, RegTensor<float>& src1Reg, RegTensor<float>& nanReg,
    MaskReg& mask)
{
    constexpr FloatUnion scale2(0x33800000); // 2**-24
    constexpr float subNormal = 1.1754944e-38;
    RegTensor<float> srcReg;
    RegTensor<float> src0SignBitReg, src0SignBitTmpReg;
    RegTensor<float> tmpReg;
    MaskReg subNormalMask, signDiffMask;

    GetSignBit(src0SignBitReg, src0Reg, mask);
    vmuls(src0SignBitTmpReg, src0SignBitReg, -2.0f, mask, MODE_ZEROING);
    vadds(src0SignBitTmpReg, src0SignBitTmpReg, 1.0f, mask, MODE_ZEROING);
    vmul(dstReg, dstReg, src0SignBitTmpReg, mask, MODE_ZEROING);

    vabs(srcReg, src1Reg, mask, MODE_ZEROING);
    vcmps_le(subNormalMask, srcReg, subNormal, mask);
    vmuls(tmpReg, dstReg, scale2.f, subNormalMask, MODE_ZEROING);
    vsel(dstReg, tmpReg, dstReg, subNormalMask);

    SolveExceptionScenarios(dstReg, src0Reg, src1Reg, nanReg, mask);

    if constexpr (!isFmod) {
        vmul(tmpReg, src1Reg, dstReg, mask, MODE_ZEROING);
        vcmps_lt(signDiffMask, tmpReg, 0.0f, mask);
        vadd(tmpReg, dstReg, src1Reg, signDiffMask, MODE_ZEROING);
        vsel(dstReg, tmpReg, dstReg, signDiffMask);
    }
}

template <bool isFmod>
PTO_INTERNAL void TFmodRemHP(
    RegTensor<float>& dstReg, RegTensor<float>& src0Reg, RegTensor<float>& src1Reg, MaskReg& mask)
{
    using RoundMode = std::conditional_t<isFmod, decltype(ROUND_Z), decltype(ROUND_F)>;
    constexpr FloatUnion nanLocal(0x7fc00000);
    RegTensor<float> srcReg;
    RegTensor<float> nanReg;

    vdup(nanReg, nanLocal.f, mask, MODE_ZEROING);
    vabs(dstReg, src0Reg, mask, MODE_ZEROING);
    vabs(srcReg, src1Reg, mask, MODE_ZEROING);
    SolveScaleIter<RoundMode, FMOD_ITERATION_NUM_MAX>(dstReg, srcReg, mask);

    TFmodRemHPPost<isFmod>(dstReg, src0Reg, src1Reg, nanReg, mask);
}

// ---- Staged loop inversion (issue #2), shared scaffolding + vector-divisor (TREM) drivers ----
// TRemS.hpp includes this header and adds the scalar-divisor (TREMS) staged drivers.
// Loop-inverted high precision remainder: every full vector advances one
// SolveScale iteration per pass, with the per-vector state staged in dst.
// Measured break-even on Ascend950PR sim: 48 full vectors (issue #2).
constexpr uint32_t TREM_HP_STAGED_MIN_VECS = 48;

template <typename T>
PTO_INTERNAL bool TRemHpStagedDisjoint(const __ubuf__ T* a, uint32_t aElems, const __ubuf__ T* b, uint32_t bElems)
{
    return a + aElems <= b || b + bElems <= a;
}

// Staging reuses dst as scratch, so it needs enough full vectors to pay for
// the extra UB round trips and must not alias either input.
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL bool TRemHpStagedEligible(
    const __ubuf__ float* dst, const __ubuf__ float* src0, const __ubuf__ float* src1, uint32_t validRows,
    uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    if (validRows * (validCols / elemsPerRepeat) < TREM_HP_STAGED_MIN_VECS) {
        return false;
    }
    constexpr uint32_t dstElems = TileDataDst::Rows * TileDataDst::Cols;
    constexpr uint32_t src0Elems = TileDataSrc0::Rows * TileDataSrc0::Cols;
    constexpr uint32_t src1Elems = TileDataSrc1::Rows * TileDataSrc1::Cols;
    return TRemHpStagedDisjoint<float>(dst, dstElems, src0, src0Elems) &&
           TRemHpStagedDisjoint<float>(dst, dstElems, src1, src1Elems);
}

template <typename T>
PTO_INTERNAL void TRemHpStagedVecs(uint32_t validRows, uint32_t validCols, uint32_t& fullPerRow, uint32_t& tailCount)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(T);
    fullPerRow = validCols / elemsPerRepeat;
    tailCount = validCols - fullPerRow * elemsPerRepeat;
}

// One SolveScale iteration over every full vector; iteration == FMOD_ITERATION_NUM_MAX
// also seeds the state from the original dividend. The divisor is reloaded from
// the original src1 each pass because a scaled divisor must not feed the next pass.
template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpStagePass(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, divReg, src0Reg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            if constexpr (iterationNum == FMOD_ITERATION_NUM_MAX) {
                vlds(src0Reg, src0, row * Src0Stride + col, NORM);
                vabs(stateReg, src0Reg, preg, MODE_ZEROING);
            } else {
                vlds(stateReg, dst, row * DstStride + col, NORM);
            }
            vlds(divReg, src1, row * Src1Stride + col, NORM);
            vabs(divReg, divReg, preg, MODE_ZEROING);
            SolveScale<RoundMode, iterationNum>(
                stateReg, divReg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
    mem_bar(VST_VLD);
}

// Runs passes [first, first+count) as plain loop bodies (no scopes); the
// caller groups them into a scoped chunk.
template <
    typename RoundMode, int32_t first, int32_t count, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpStageSeq(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t fullPerRow)
{
    TRemHpStagePass<RoundMode, first, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, fullPerRow);
    if constexpr (count > 1) {
        TRemHpStageSeq<RoundMode, first - 1, count - 1, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
}

// Sign restore, subnormal rescale, exceptions and rem sign correction over the
// staged state; reloads the original src0/src1 needed by the fixups.
template <bool isFmod, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpPostPass(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, src0Reg, src1Reg, nanReg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    vdup(nanReg, nan.f, preg, MODE_ZEROING);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            vlds(stateReg, dst, row * DstStride + col, NORM);
            vlds(src0Reg, src0, row * Src0Stride + col, NORM);
            vlds(src1Reg, src1, row * Src1Stride + col, NORM);
            TFmodRemHPPost<isFmod>(stateReg, src0Reg, src1Reg, nanReg, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
}

// Tail vectors keep the original register path: 11 chained iterations in
// registers with predicated loads/stores.
template <bool isFmod, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpTailPass(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> vDst, vSrc0, vSrc1;
    MaskReg preg = CreatePredicate<float>(tailCount);
    uint32_t col = fullPerRow * elemsPerRepeat;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        vlds(vSrc0, src0, row * Src0Stride + col, NORM);
        vlds(vSrc1, src1, row * Src1Stride + col, NORM);
        TFmodRemHP<isFmod>(vDst, vSrc0, vSrc1, preg);
        vsts(vDst, dst, row * DstStride + col, distValue, preg);
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride, unsigned Src1Stride>
PTO_INTERNAL void TRemHpStaged(
    __ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, uint32_t validRows, uint32_t validCols)
{
    using RoundMode = std::conditional_t<isFmod, decltype(ROUND_Z), decltype(ROUND_F)>;
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    // Passes are VF-fused in small chunks: mem_bar(VST_VLD) between passes
    // inside a chunk, a full barrier between chunks. One VF for everything
    // regresses 2D shapes (queue/register pressure); 13 separate VFs pay
    // too much launch overhead (npu_skills vf-fusion stop-conditions). TREMS
    // keeps one VF per pass (see the S variants below).
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, FMOD_ITERATION_NUM_MAX, 3, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, FMOD_ITERATION_NUM_MAX - 3, 3, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, FMOD_ITERATION_NUM_MAX - 6, 3, DstStride, Src0Stride, Src1Stride>(
            dst, src0, src1, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
    __VEC_SCOPE__
    {
        TRemHpStageSeq<RoundMode, 2, 2, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, fullPerRow);
        TRemHpPostPass<isFmod, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, fullPerRow);
    }
    if (tailCount > 0) {
        __VEC_SCOPE__
        {
            TRemHpTailPass<isFmod, DstStride, Src0Stride, Src1Stride>(dst, src0, src1, validRows, validCols);
        }
    }
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <typename T>
PTO_INTERNAL void Int64RemRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    using Reg64 = std::conditional_t<std::is_same_v<T, int64_t>, vector_2xvl_s64, vector_2xvl_u64>;
    using Reg32 = std::conditional_t<std::is_same_v<T, int64_t>, vector_s32, vector_u32>;
    Reg64 lhs, rhs, remainder;
    lhs.val[0] = (Reg32)lhsLow;
    lhs.val[1] = (Reg32)lhsHigh;
    rhs.val[0] = (Reg32)rhsLow;
    rhs.val[1] = (Reg32)rhsHigh;
    vmod<DivisionMode::DIV_FLOOR>(remainder, lhs, rhs, mask, MODE_ZEROING);
    dstLow = (vector_s32)remainder.val[0];
    dstHigh = (vector_s32)remainder.val[1];
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols, bool FullLoad>
PTO_INTERNAL void Int64RemRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_s32 dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh;
    Int64LoadRegs<T, Src0Cols, FullLoad>(lhsLow, lhsHigh, src0, row, colOffset);
    Int64LoadRegs<T, Src1Cols, FullLoad>(rhsLow, rhsHigh, src1, row, colOffset);
    Int64RemRegs<T>(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    Int64StoreRegs<T, DstCols>(dstLow, dstHigh, dst, row, colOffset, mask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Rem(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
    uint16_t fullRepeats = Int64FullLoadRepeats<Int64MinCols<Src0Cols, Src1Cols>, elementsPerRepeat>(colRepeats);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh, half0, half1;
        MaskReg lowMask, highMask;
        uint16_t rows = validRows;

        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = validCols;

            for (uint16_t colRepeat = 0; colRepeat < fullRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<Src0Cols, true>(
                    al, ah, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, colOffset);
                Int64LoadBounded<Src1Cols, true>(
                    bl, bh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
            for (uint16_t colRepeat = fullRepeats; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<Src0Cols, false>(
                    al, ah, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, colOffset);
                Int64LoadBounded<Src1Cols, false>(
                    bl, bh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
    uint16_t fullRepeats = Int64FullLoadRepeats<SrcCols, elementsPerRepeat>(colRepeats);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh, half0, half1;
        MaskReg lowMask, highMask;
        uint64_t bits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegs(bl, bh, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
        uint16_t rows = validRows;

        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = validCols;

            for (uint16_t colRepeat = 0; colRepeat < fullRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<SrcCols, true>(
                    al, ah, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
            for (uint16_t colRepeat = fullRepeats; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<SrcCols, false>(
                    al, ah, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Rem(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols);

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols);
#endif

template <RemAlgorithm PrecisionType, typename T>
struct RemOp {
    PTO_INTERNAL static void BinInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        if constexpr (PrecisionType == RemAlgorithm::HIGH_PRECISION && std::is_same<T, float>::value) {
            TFmodRemHP<false>(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (std::is_same<T, float>::value) {
            MaskReg sign_diff_mask;
            RegTensor<T> reg_tmp;
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(reg_dst, reg_dst, ROUND_F, preg);
            vmul(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);

            vmul(reg_tmp, reg_src1, reg_dst, preg, MODE_ZEROING);
            vcmps_lt(sign_diff_mask, reg_tmp, 0.0f, preg);
            vadd(reg_tmp, reg_dst, reg_src1, sign_diff_mask, MODE_ZEROING);
            vsel(reg_dst, reg_tmp, reg_dst, sign_diff_mask);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<float> reg_tmp_even0, reg_tmp_even1, reg_tmp_even2, reg_tmp_odd0, reg_tmp_odd1, reg_tmp_odd2;
            RegTensor<T> reg_dst_even, reg_dst_odd, reg_tmp;
            MaskReg sign_diff_mask;
            vcvt(reg_tmp_even0, reg_src0, preg, PART_EVEN);
            vcvt(reg_tmp_even1, reg_src1, preg, PART_EVEN);
            vcvt(reg_tmp_odd0, reg_src0, preg, PART_ODD);
            vcvt(reg_tmp_odd1, reg_src1, preg, PART_ODD);

            vdiv(reg_tmp_even2, reg_tmp_even0, reg_tmp_even1, preg, MODE_ZEROING);
            vdiv(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd1, preg, MODE_ZEROING);

            vtrc(reg_tmp_even2, reg_tmp_even2, ROUND_F, preg);
            vtrc(reg_tmp_odd2, reg_tmp_odd2, ROUND_F, preg);

            vmul(reg_tmp_even2, reg_tmp_even2, reg_tmp_even1, preg, MODE_ZEROING);
            vmul(reg_tmp_odd2, reg_tmp_odd2, reg_tmp_odd1, preg, MODE_ZEROING);

            vsub(reg_tmp_even2, reg_tmp_even0, reg_tmp_even2, preg, MODE_ZEROING);
            vsub(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd2, preg, MODE_ZEROING);

            vcvt(reg_dst_even, reg_tmp_even2, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
            vcvt(reg_dst_odd, reg_tmp_odd2, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);

            vmul(reg_tmp, reg_src1, reg_dst, preg, MODE_ZEROING);
            vcmps_lt(sign_diff_mask, reg_tmp, 0.0f, preg);
            vadd(reg_tmp, reg_dst, reg_src1, sign_diff_mask, MODE_ZEROING);
            vsel(reg_dst, reg_tmp, reg_dst, sign_diff_mask);
        } else {
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
    unsigned ElementsPerRepeat, unsigned BlockSizeElem, auto PrecisionType = RemAlgorithm::DEFAULT>
__tf__ PTO_INTERNAL OP_NAME(TREM) OP_TYPE(element_wise) void TRem(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, typename TileDataTmp::TileDType __in__ tmp, unsigned validRows,
    unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    // The A5 implementation does not use the temporary tile.
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64Rem<T, TileDataDst::Cols, TileDataSrc0::Cols, TileDataSrc1::Cols>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols);
    } else {
        if constexpr (PrecisionType == RemAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            if (TRemHpStagedEligible<TileDataDst, TileDataSrc0, TileDataSrc1>(
                    dstPtr, src0Ptr, src1Ptr, validRows, validCols)) {
                TRemHpStaged<false, TileDataDst::RowStride, TileDataSrc0::RowStride, TileDataSrc1::RowStride>(
                    dstPtr, src0Ptr, src1Ptr, validRows, validCols);
                return;
            }
        }
        BinaryInstr<RemOp<PrecisionType, T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TRemCheck(
    const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1, const TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value || std::is_same<T, half>::value ||
            std::is_same<T, float>::value || std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
        "Fix: TREM has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TREM only support row major layout.");
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TREM input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    TRemCheck<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TRem<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp, elementsPerRepeat, blockSizeElem, PrecisionType>(
        dst.data(), src0.data(), src1.data(), tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
