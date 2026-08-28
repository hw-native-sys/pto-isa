/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/
#ifndef __ROW_REDUCE__
#define __ROW_REDUCE__
#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include "TPartBinOps.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <typename T, unsigned SrcCols>
PTO_INTERNAL void Int64RowSumRepeat(
    vector_u32& outLow, vector_u32& outHigh, __ubuf__ T* src, uint16_t row, uint32_t colOffset, vector_u32& mask16,
    MaskReg& mask)
{
    vector_u32 low, high, low16, mid16, tmp;
    vlds((vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
    vand(low16, low, mask16, mask, MODE_ZEROING);
    vcadd(low16, low16, mask, MODE_ZEROING);
    vshrs(mid16, low, 16, mask, MODE_ZEROING);
    vcadd(mid16, mid16, mask, MODE_ZEROING);
    vcadd(outHigh, high, mask, MODE_ZEROING);
    vshrs(tmp, low16, 16, mask, MODE_ZEROING);
    vadd(mid16, mid16, tmp, mask, MODE_ZEROING);
    vshrs(tmp, mid16, 16, mask, MODE_ZEROING);
    vadd(outHigh, outHigh, tmp, mask, MODE_ZEROING);
    vand(low16, low16, mask16, mask, MODE_ZEROING);
    vand(mid16, mid16, mask16, mask, MODE_ZEROING);
    vshls(mid16, mid16, 16, mask, MODE_ZEROING);
    vor(outLow, low16, mid16, mask);
}

template <typename T, unsigned SrcCols>
PTO_INTERNAL void Int64RowSumAccumulate(
    vector_u32& accLow, vector_u32& accHigh, __ubuf__ T* src, uint16_t row, uint32_t validCols, uint16_t repeatTimes,
    vector_u32& mask16, MaskReg& oneMask, MaskReg& rowMask, MaskReg& fullMask)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    vector_u32 outLow, outHigh;
    vbr(accLow, 0);
    vbr(accHigh, 0);
    for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
        uint32_t colOffset = colRepeat * elementsPerRepeat;
        uint32_t remainingCols = validCols - colOffset;
        MaskReg colMask = plt_b32(remainingCols, POST_UPDATE);
        MaskReg repeatMask;
        pand(repeatMask, colMask, rowMask, fullMask);
        Int64RowSumRepeat<T, SrcCols>(outLow, outHigh, src, row, colOffset, mask16, repeatMask);
        MaskReg carry, carryOut;
        vaddc(carry, (vector_s32&)accLow, (vector_s32&)accLow, (vector_s32&)outLow, oneMask);
        vaddcs(carryOut, (vector_s32&)accHigh, (vector_s32&)accHigh, (vector_s32&)outHigh, carry, oneMask);
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowSum(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr unsigned alignedBytes = 32;
    constexpr unsigned dstRowBytes = DstCols * sizeof(T);
    constexpr bool dstRowAligned = dstRowBytes % alignedBytes == 0;
    __VEC_SCOPE__
    {
        vector_u32 mask16, accLow, accHigh;
        vbr(mask16, 0xffffu);
        MaskReg oneMask = pset_b32(PAT_VL1);
        uint16_t rows = validRows;
        uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
        uint32_t fullMaskCols = elementsPerRepeat;
        MaskReg allMask = plt_b32(fullMaskCols, POST_UPDATE);
        if constexpr (dstRowAligned) {
            for (uint16_t row = 0; row < rows; ++row) {
                Int64RowSumAccumulate<T, SrcCols>(
                    accLow, accHigh, src, row, validCols, repeatTimes, mask16, oneMask, allMask, allMask);
                vsts(
                    (vector_s32&)accLow, (vector_s32&)accHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32,
                    oneMask);
            }
        } else {
            static_assert(DstCols == 1, "Unaligned int64 row reduction output must be compact scalar rows.");
            constexpr uint16_t rowsPerStore = alignedBytes / sizeof(T);
            uint16_t groups = CeilDivision(rows, rowsPerStore);
            for (uint16_t g = 0; g < groups; ++g) {
                vector_u32 slotALow, slotAHigh, slotBLow, slotBHigh, slotCLow, slotCHigh, slotDLow, slotDHigh;
                uint16_t rowBase = g * rowsPerStore;
                uint32_t row0Valid = 1;
                uint32_t row1Valid = rowBase + 1 < rows;
                uint32_t row2Valid = rowBase + 2 < rows;
                uint32_t row3Valid = rowBase + 3 < rows;
                uint32_t row0MaskCols = row0Valid * elementsPerRepeat;
                uint32_t row1MaskCols = row1Valid * elementsPerRepeat;
                uint32_t row2MaskCols = row2Valid * elementsPerRepeat;
                uint32_t row3MaskCols = row3Valid * elementsPerRepeat;
                MaskReg row0Mask = plt_b32(row0MaskCols, POST_UPDATE);
                MaskReg row1Mask = plt_b32(row1MaskCols, POST_UPDATE);
                MaskReg row2Mask = plt_b32(row2MaskCols, POST_UPDATE);
                MaskReg row3Mask = plt_b32(row3MaskCols, POST_UPDATE);
                uint16_t row0 = rowBase;
                uint16_t row1 = rowBase + row1Valid;
                uint16_t row2 = rowBase + row2Valid * 2;
                uint16_t row3 = rowBase + row3Valid * 3;
                Int64RowSumAccumulate<T, SrcCols>(
                    accLow, accHigh, src, row0, validCols, repeatTimes, mask16, oneMask, row0Mask, allMask);
                vdup((vector_s32&)slotALow, (vector_s32&)accLow, allMask, POS_LOWEST, MODE_ZEROING);
                vdup((vector_s32&)slotAHigh, (vector_s32&)accHigh, allMask, POS_LOWEST, MODE_ZEROING);
                Int64RowSumAccumulate<T, SrcCols>(
                    accLow, accHigh, src, row1, validCols, repeatTimes, mask16, oneMask, row1Mask, allMask);
                vdup((vector_s32&)slotCLow, (vector_s32&)accLow, allMask, POS_LOWEST, MODE_ZEROING);
                vdup((vector_s32&)slotCHigh, (vector_s32&)accHigh, allMask, POS_LOWEST, MODE_ZEROING);
                Int64RowSumAccumulate<T, SrcCols>(
                    accLow, accHigh, src, row2, validCols, repeatTimes, mask16, oneMask, row2Mask, allMask);
                vdup((vector_s32&)slotBLow, (vector_s32&)accLow, allMask, POS_LOWEST, MODE_ZEROING);
                vdup((vector_s32&)slotBHigh, (vector_s32&)accHigh, allMask, POS_LOWEST, MODE_ZEROING);
                Int64RowSumAccumulate<T, SrcCols>(
                    accLow, accHigh, src, row3, validCols, repeatTimes, mask16, oneMask, row3Mask, allMask);
                vdup((vector_s32&)slotDLow, (vector_s32&)accLow, allMask, POS_LOWEST, MODE_ZEROING);
                vdup((vector_s32&)slotDHigh, (vector_s32&)accHigh, allMask, POS_LOWEST, MODE_ZEROING);
                vector_u32 p0, p1, q0, q1, r0, r1, ph0, ph1, qh0, qh1, rh0, rh1;
                vintlv((vector_s32&)p0, (vector_s32&)p1, (vector_s32&)slotALow, (vector_s32&)slotBLow);
                vintlv((vector_s32&)q0, (vector_s32&)q1, (vector_s32&)slotCLow, (vector_s32&)slotDLow);
                vintlv((vector_s32&)r0, (vector_s32&)r1, (vector_s32&)p0, (vector_s32&)q0);
                vintlv((vector_s32&)ph0, (vector_s32&)ph1, (vector_s32&)slotAHigh, (vector_s32&)slotBHigh);
                vintlv((vector_s32&)qh0, (vector_s32&)qh1, (vector_s32&)slotCHigh, (vector_s32&)slotDHigh);
                vintlv((vector_s32&)rh0, (vector_s32&)rh1, (vector_s32&)ph0, (vector_s32&)qh0);
                uint32_t groupMaskElems = row0Valid + row1Valid + row2Valid + row3Valid;
                MaskReg groupMask = plt_b32(groupMaskElems, POST_UPDATE);
                vsts(
                    (vector_s32&)r0, (vector_s32&)rh0, (__ubuf__ int32_t*)dst + g * rowsPerStore * DstCols * 2, 0,
                    INTLV_B32, groupMask);
            }
        }
    }
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64RowReduceHigh(vector_s32& reducedHigh, vector_s32& high, MaskReg& mask)
{
    if constexpr (Op == Int64Op::Max) {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmax(reducedHigh, high, mask, MODE_ZEROING);
        else
            vcmax((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
    } else {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmin(reducedHigh, high, mask, MODE_ZEROING);
        else
            vcmin((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
    }
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64RowSelectHigh(vector_s32& selectedHigh, vector_s32& high, MaskReg& equalLow)
{
    if constexpr (Op == Int64Op::Max) {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmax(selectedHigh, high, equalLow, MODE_ZEROING);
        else
            vcmax((vector_u32&)selectedHigh, (vector_u32&)high, equalLow, MODE_ZEROING);
    } else {
        if constexpr (std::is_same_v<T, int64_t>)
            vcmin(selectedHigh, high, equalLow, MODE_ZEROING);
        else
            vcmin((vector_u32&)selectedHigh, (vector_u32&)high, equalLow, MODE_ZEROING);
    }
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64RowMinMaxInit(vector_s32& accLow, vector_s32& accHigh)
{
    if constexpr (Op == Int64Op::Max) {
        if constexpr (std::is_same_v<T, int64_t>) {
            vbr((vector_u32&)accLow, 0u);
            vbr((vector_u32&)accHigh, 0x80000000u);
        } else {
            vbr((vector_u32&)accLow, 0u);
            vbr((vector_u32&)accHigh, 0u);
        }
    } else {
        if constexpr (std::is_same_v<T, int64_t>) {
            vbr((vector_u32&)accLow, 0xffffffffu);
            vbr((vector_u32&)accHigh, 0x7fffffffu);
        } else {
            vbr((vector_u32&)accLow, 0xffffffffu);
            vbr((vector_u32&)accHigh, 0xffffffffu);
        }
    }
}

template <Int64Op Op, typename T, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMaxRepeat(
    vector_s32& outLow, vector_s32& outHigh, __ubuf__ T* src, unsigned row, unsigned colOffset, MaskReg& mask,
    MaskReg& allMask)
{
    vector_s32 low, high, reducedHigh, highDup, selectedHigh, lowDup;
    vector_u32 reducedLow;
    vlds(low, high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
    Int64RowReduceHigh<Op, T>(reducedHigh, high, mask);
    vdup(highDup, reducedHigh, allMask, POS_LOWEST, MODE_ZEROING);
    MaskReg equalHigh;
    vcmp_eq(equalHigh, highDup, high, mask);
    if constexpr (Op == Int64Op::Max)
        vcmax(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);
    else
        vcmin(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);

    vdup((vector_s32&)lowDup, (vector_s32&)reducedLow, allMask, POS_LOWEST, MODE_ZEROING);
    MaskReg equalLow;
    vcmp_eq(equalLow, (vector_u32&)lowDup, (vector_u32&)low, mask);
    Int64RowSelectHigh<Op, T>(selectedHigh, high, equalLow);
    outLow = (vector_s32&)reducedLow;
    outHigh = selectedHigh;
}

template <Int64Op Op, typename T, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMaxAccumulate(
    vector_s32& accLow, vector_s32& accHigh, __ubuf__ T* src, uint16_t row, uint32_t validCols, uint16_t repeatTimes,
    MaskReg& dupMask, MaskReg& rowMask, MaskReg& fullMask)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    vector_s32 repeatLow, repeatHigh;
    Int64RowMinMaxInit<Op, T>(accLow, accHigh);
    MaskReg oneMask = pset_b32(PAT_VL1);
    for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
        uint32_t colOffset = colRepeat * elementsPerRepeat;
        uint32_t remainingCols = validCols - colOffset;
        MaskReg colMask = plt_b32(remainingCols, POST_UPDATE);
        MaskReg repeatMask;
        pand(repeatMask, colMask, rowMask, fullMask);
        Int64RowMinMaxRepeat<Op, T, SrcCols>(repeatLow, repeatHigh, src, row, colOffset, repeatMask, dupMask);
        Int64MinMax<Op, T>(accLow, accHigh, accLow, accHigh, repeatLow, repeatHigh, oneMask);
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMax(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr unsigned alignedBytes = 32;
    constexpr unsigned dstRowBytes = DstCols * sizeof(T);
    constexpr bool dstRowAligned = dstRowBytes % alignedBytes == 0;
    __VEC_SCOPE__
    {
        vector_s32 accLow, accHigh;
        MaskReg dupMask = pset_b32(PAT_ALL);
        uint16_t rows = validRows;
        uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
        uint32_t fullMaskCols = elementsPerRepeat;
        MaskReg fullMask = plt_b32(fullMaskCols, POST_UPDATE);
        if constexpr (dstRowAligned) {
            for (uint16_t row = 0; row < rows; ++row) {
                Int64RowMinMaxAccumulate<Op, T, SrcCols>(
                    accLow, accHigh, src, row, validCols, repeatTimes, dupMask, fullMask, fullMask);
                MaskReg oneMask = pset_b32(PAT_VL1);
                vsts(accLow, accHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, oneMask);
            }
        } else {
            static_assert(DstCols == 1, "Unaligned int64 row reduction output must be compact scalar rows.");
            constexpr uint16_t rowsPerStore = alignedBytes / sizeof(T);
            uint16_t groups = CeilDivision(rows, rowsPerStore);
            for (uint16_t g = 0; g < groups; ++g) {
                vector_s32 slotALow, slotAHigh, slotBLow, slotBHigh, slotCLow, slotCHigh, slotDLow, slotDHigh;
                uint16_t rowBase = g * rowsPerStore;
                uint32_t row0Valid = 1;
                uint32_t row1Valid = rowBase + 1 < rows;
                uint32_t row2Valid = rowBase + 2 < rows;
                uint32_t row3Valid = rowBase + 3 < rows;
                uint32_t row0MaskCols = row0Valid * elementsPerRepeat;
                uint32_t row1MaskCols = row1Valid * elementsPerRepeat;
                uint32_t row2MaskCols = row2Valid * elementsPerRepeat;
                uint32_t row3MaskCols = row3Valid * elementsPerRepeat;
                MaskReg row0Mask = plt_b32(row0MaskCols, POST_UPDATE);
                MaskReg row1Mask = plt_b32(row1MaskCols, POST_UPDATE);
                MaskReg row2Mask = plt_b32(row2MaskCols, POST_UPDATE);
                MaskReg row3Mask = plt_b32(row3MaskCols, POST_UPDATE);
                uint16_t row0 = rowBase;
                uint16_t row1 = rowBase + row1Valid;
                uint16_t row2 = rowBase + row2Valid * 2;
                uint16_t row3 = rowBase + row3Valid * 3;
                Int64RowMinMaxAccumulate<Op, T, SrcCols>(
                    accLow, accHigh, src, row0, validCols, repeatTimes, dupMask, row0Mask, fullMask);
                vdup(slotALow, accLow, dupMask, POS_LOWEST, MODE_ZEROING);
                vdup(slotAHigh, accHigh, dupMask, POS_LOWEST, MODE_ZEROING);
                Int64RowMinMaxAccumulate<Op, T, SrcCols>(
                    accLow, accHigh, src, row1, validCols, repeatTimes, dupMask, row1Mask, fullMask);
                vdup(slotCLow, accLow, dupMask, POS_LOWEST, MODE_ZEROING);
                vdup(slotCHigh, accHigh, dupMask, POS_LOWEST, MODE_ZEROING);
                Int64RowMinMaxAccumulate<Op, T, SrcCols>(
                    accLow, accHigh, src, row2, validCols, repeatTimes, dupMask, row2Mask, fullMask);
                vdup(slotBLow, accLow, dupMask, POS_LOWEST, MODE_ZEROING);
                vdup(slotBHigh, accHigh, dupMask, POS_LOWEST, MODE_ZEROING);
                Int64RowMinMaxAccumulate<Op, T, SrcCols>(
                    accLow, accHigh, src, row3, validCols, repeatTimes, dupMask, row3Mask, fullMask);
                vdup(slotDLow, accLow, dupMask, POS_LOWEST, MODE_ZEROING);
                vdup(slotDHigh, accHigh, dupMask, POS_LOWEST, MODE_ZEROING);
                vector_s32 p0, p1, q0, q1, r0, r1, ph0, ph1, qh0, qh1, rh0, rh1;
                vintlv(p0, p1, slotALow, slotBLow);
                vintlv(q0, q1, slotCLow, slotDLow);
                vintlv(r0, r1, p0, q0);
                vintlv(ph0, ph1, slotAHigh, slotBHigh);
                vintlv(qh0, qh1, slotCHigh, slotDHigh);
                vintlv(rh0, rh1, ph0, qh0);
                uint32_t groupMaskElems = row0Valid + row1Valid + row2Valid + row3Valid;
                MaskReg groupMask = plt_b32(groupMaskElems, POST_UPDATE);
                vsts(r0, rh0, (__ubuf__ int32_t*)dst + g * rowsPerStore * DstCols * 2, 0, INTLV_B32, groupMask);
            }
        }
    }
}

#else
template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowSum(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMax(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);
#endif

template <typename T>
struct ROWSUM {
    using TIN = T;
    using TOUT = std::conditional_t<std::is_same_v<T, int16_t>, int32_t, T>;
    static constexpr auto InitVal = Padding<TOUT>::Zero;
    static PTO_INTERNAL void Accumulate(
        RegTensor<TOUT>& dst, RegTensor<TOUT>& src0, RegTensor<TOUT>& src1, MaskReg& pred)
    {
        vadd(dst, src0, src1, pred, MODE_ZEROING);
    }
    static PTO_INTERNAL void Reduce(RegTensor<TOUT>& dst, RegTensor<TIN>& src, MaskReg& pred, MaskReg& pregdst)
    {
        vcadd(dst, src, pred, MODE_ZEROING);
    }
};

template <typename T>
struct ROWMAX {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Min;
    using TIN = T;
    using TOUT = T;

    static PTO_INTERNAL void Accumulate(
        RegTensor<TOUT>& dst, RegTensor<TOUT>& src0, RegTensor<TOUT>& src1, MaskReg& pred)
    {
        vmax(dst, src0, src1, pred, MODE_ZEROING);
    }

    static PTO_INTERNAL void Reduce(RegTensor<TOUT>& dst, RegTensor<TIN>& src, MaskReg& pred, MaskReg& pregdst)
    {
        if constexpr (std::is_same<TIN, uint8_t>::value) {
            RegTensor<uint16_t> srcOdd, srcEven, dstOdd, dstEven, dstTmp;
            vcvt(srcEven, src, pred, PART_EVEN);
            vcvt(srcOdd, src, pred, PART_ODD);
            vcmax(dstEven, srcEven, pred, MODE_ZEROING);
            vcmax(dstOdd, srcOdd, pred, MODE_ZEROING);
            vmax(dstTmp, dstEven, dstOdd, pregdst);
            vcvt(dst, dstTmp, pregdst, RS_DISABLE, PART_EVEN);
        } else if constexpr (std::is_same<TIN, int8_t>::value) {
            RegTensor<half> srcOdd, srcEven, dstOdd, dstEven, dstTmp;
            vcvt(srcEven, src, pred, PART_EVEN);
            vcvt(srcOdd, src, pred, PART_ODD);
            vcmax(dstEven, srcEven, pred, MODE_ZEROING);
            vcmax(dstOdd, srcOdd, pred, MODE_ZEROING);
            vmax(dstTmp, dstEven, dstOdd, pregdst);
            vcvt(dst, dstTmp, pregdst, ROUND_R, RS_DISABLE, PART_EVEN);
        } else {
            vcmax(dst, src, pred, MODE_ZEROING);
        }
    }
};

template <typename T>
struct ROWMIN {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Max;
    using TIN = T;
    using TOUT = T;

    static PTO_INTERNAL void Accumulate(
        RegTensor<TOUT>& dst, RegTensor<TOUT>& src0, RegTensor<TOUT>& src1, MaskReg& pred)
    {
        vmin(dst, src0, src1, pred, MODE_ZEROING);
    }

    static PTO_INTERNAL void Reduce(RegTensor<TOUT>& dst, RegTensor<TIN>& src, MaskReg& pred, MaskReg& pregdst)
    {
        if constexpr (std::is_same<TIN, uint8_t>::value) {
            RegTensor<uint16_t> srcOdd, srcEven, dstOdd, dstEven, dstTmp;
            vcvt(srcEven, src, pred, PART_EVEN);
            vcvt(srcOdd, src, pred, PART_ODD);
            vcmin(dstEven, srcEven, pred, MODE_ZEROING);
            vcmin(dstOdd, srcOdd, pred, MODE_ZEROING);
            vmin(dstTmp, dstEven, dstOdd, pregdst);
            vcvt(dst, dstTmp, pregdst, RS_DISABLE, PART_EVEN);
        } else if constexpr (std::is_same<TIN, int8_t>::value) {
            RegTensor<half> srcOdd, srcEven, dstOdd, dstEven, dstTmp;
            vcvt(srcEven, src, pred, PART_EVEN);
            vcvt(srcOdd, src, pred, PART_ODD);
            vcmin(dstEven, srcEven, pred, MODE_ZEROING);
            vcmin(dstOdd, srcOdd, pred, MODE_ZEROING);
            vmin(dstTmp, dstEven, dstOdd, pregdst);
            vcvt(dst, dstTmp, pregdst, ROUND_R, RS_DISABLE, PART_EVEN);
        } else {
            vcmin(dst, src, pred, MODE_ZEROING);
        }
    }
};

template <typename TileDataOut, typename TileDataIn, bool idx = false>
PTO_INTERNAL void TRowReduceCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValidRow)
{
    using T = typename TileDataIn::DType;
    static_assert(idx || std::is_same_v<T, typename TileDataOut::DType>, "Input/output DType mismatch.");
    static_assert(
        TileDataOut::Loc == pto::TileType::Vec && TileDataIn::Loc == pto::TileType::Vec,
        "Row reduction only works on vector tiles.");
    static_assert(TileDataIn::isRowMajor && !TileDataIn::isBoxedLayout, "Input tile must use ND row-major layout.");
    static_assert(
        (!TileDataOut::isBoxedLayout &&
         (TileDataOut::isRowMajor || (!TileDataOut::isRowMajor && TileDataOut::Cols == 1))),
        "Output tile must use ND layout or DN layout with one column.");
    PTO_ASSERT(srcValidRows != 0 && srcValidCols != 0, "Row reduction input must be non-empty.");
    PTO_ASSERT(srcValidRows == dstValidRow, "Row reduction preserves row count.");
}

template <
    typename ReduceOp, typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat, bool postUpdate = true>
PTO_INTERNAL void TRowReduceProc(
    __ubuf__ typename TileDataOut::DType* dstPtr, __ubuf__ typename TileDataOut::DType* srcPtr, uint32_t rows,
    uint32_t cols, uint16_t repeatTimes, int32_t srcRowAdjust)
{
    using TIN = typename ReduceOp::TIN;
    using TOUT = typename ReduceOp::TOUT;
    using TDST = typename TileDataOut::DType;
    __VEC_SCOPE__
    {
        RegTensor<TIN> vreg0;
        RegTensor<TOUT> vreg1;
        RegTensor<TOUT> vregdst;
        RegTensor<TDST> vreg_result;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDST, DistVST::DIST_ONEPT>())>();
        uint32_t destItems = 1;
        MaskReg pregdst = CreatePredicate<TIN>(destItems);
        for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
            vbr((RegTensor<typename Padding<TOUT>::Type>&)vregdst, ReduceOp::InitVal);
            uint32_t sreg = cols;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
                MaskReg preg = CreatePredicate<TIN>(sreg);
                if constexpr (postUpdate) {
                    vlds(vreg0, srcPtr, elementsPerRepeat, NORM, POST_UPDATE);
                } else {
                    vlds(vreg0, srcPtr, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
                }
                ReduceOp::Reduce(vreg1, vreg0, preg, pregdst);
                ReduceOp::Accumulate(vregdst, vregdst, vreg1, pregdst);
            }
            if constexpr (!std::is_same_v<TOUT, TDST>) {
                vcvt(vreg_result, vregdst, pregdst, RS_DISABLE, PART_EVEN);
                if constexpr (postUpdate) {
                    vsts(vreg_result, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
                } else {
                    vsts(vreg_result, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
                }
            } else {
                if constexpr (postUpdate) {
                    vsts(vregdst, dstPtr, TileDataOut::RowStride, distValue, pregdst, POST_UPDATE);
                } else {
                    vsts(vregdst, dstPtr, i * TileDataOut::RowStride, distValue, pregdst);
                }
            }
            if constexpr (postUpdate) {
                srcPtr += srcRowAdjust;
            }
        }
    }
}

template <typename ReduceOp, typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
PTO_INTERNAL void TRowReduceImpl(
    __ubuf__ typename TileDataOut::DType* dstPtr, __ubuf__ typename TileDataOut::DType* srcPtr, uint32_t rows,
    uint32_t cols, unsigned version)
{
    using TIN = typename ReduceOp::TIN;
    using TOUT = typename ReduceOp::TOUT;
    using TDST = typename TileDataOut::DType;
    constexpr int SAT_MODE_BIT_60 = 60;
    constexpr int SAT_MODE_BIT_59 = 59;
    constexpr bool needsNonSatMode = std::is_same_v<TOUT, int32_t> && std::is_same_v<TDST, int16_t>;
    bool originalCtrl60 = false;
    bool originalCtrl59 = false;
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);

    if constexpr (needsNonSatMode) {
        uint64_t originalCtrl = get_ctrl();
        originalCtrl60 = (originalCtrl & (1ULL << SAT_MODE_BIT_60)) != 0;
        originalCtrl59 = (originalCtrl & (1ULL << SAT_MODE_BIT_59)) != 0;
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
        set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
    }

    if (version == VFIMPL_2D_NO_POST_UPDATE) {
        TRowReduceProc<ReduceOp, TileDataOut, TileDataIn, elementsPerRepeat, false>(
            dstPtr, srcPtr, rows, cols, repeatTimes, 0);
    } else {
        int32_t srcRowAdjust =
            static_cast<int32_t>(TileDataIn::RowStride) - static_cast<int32_t>(repeatTimes) * elementsPerRepeat;
        TRowReduceProc<ReduceOp, TileDataOut, TileDataIn, elementsPerRepeat, true>(
            dstPtr, srcPtr, rows, cols, repeatTimes, srcRowAdjust);
    }
    if constexpr (needsNonSatMode) {
        if (originalCtrl60)
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_60));
        else
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_60));
        if (originalCtrl59)
            set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT_59));
        else
            set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT_59));
    }
}

template <typename ReduceOp, typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
PTO_INTERNAL void TRowReduceEntry(
    typename TileDataOut::TileDType dst, typename TileDataIn::TileDType src, uint32_t dstValidRow,
    uint32_t srcValidRows, uint32_t srcValidCols, unsigned version)
{
    TRowReduceCheck<TileDataOut, TileDataIn>(srcValidRows, srcValidCols, dstValidRow);
    using T = typename TileDataIn::DType;
    TRowReduceImpl<ReduceOp, TileDataOut, TileDataIn, elementsPerRepeat>(
        (__ubuf__ T*)__cce_get_tile_ptr(dst), (__ubuf__ T*)__cce_get_tile_ptr(src), srcValidRows, srcValidCols,
        version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMAX) OP_TYPE(reduce) void TRowMax(
    typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src, uint32_t dstValidRow,
    uint32_t srcValidRows, uint32_t srcValidCols, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>,
        "TROWMAX supports half, float, int32, int16, int8, and uint8.");
    TRowReduceEntry<ROWMAX<T>, TileDataOut, TileDataIn, elementsPerRepeat>(
        dst, src, dstValidRow, srcValidRows, srcValidCols, version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWSUM) OP_TYPE(reduce) void TRowSum(
    typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src, uint32_t dstValidRow,
    uint32_t srcValidRows, uint32_t srcValidCols, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>,
        "TROWSUM supports half, float, int32, and int16.");
    TRowReduceEntry<ROWSUM<T>, TileDataOut, TileDataIn, elementsPerRepeat>(
        dst, src, dstValidRow, srcValidRows, srcValidCols, version);
}

template <typename TileDataOut, typename TileDataIn, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL OP_NAME(TROWMIN) OP_TYPE(reduce) void TRowMin(
    typename TileDataOut::TileDType __out__ dst, typename TileDataIn::TileDType __in__ src, uint32_t dstValidRow,
    uint32_t srcValidRows, uint32_t srcValidCols, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataIn::DType;
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
        "TROWMIN supports half, float, int32, int16, uint8, and int8.");
    TRowReduceEntry<ROWMIN<T>, TileDataOut, TileDataIn, elementsPerRepeat>(
        dst, src, dstValidRow, srcValidRows, srcValidCols, version);
}

template <template <typename> class ReduceOp, Int64Op int64Op, bool isSum, typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TROWREDUCE_IMPL_COMMON(TileDataOut& dst, TileDataIn& src)
{
    using T = typename TileDataIn::DType;
    constexpr bool is64Bit = std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;
    constexpr bool supported = is64Bit || std::is_same_v<T, half> || std::is_same_v<T, float> ||
                               std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> ||
                               (!isSum && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>));
    static_assert(supported, "Unsupported row reduction dtype.");
    TRowReduceCheck<TileDataOut, TileDataIn>(src.GetValidRow(), src.GetValidCol(), dst.GetValidRow());
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        if constexpr (isSum)
            Int64RowSum<T, TileDataOut::Cols, TileDataIn::Cols>(
                (__ubuf__ T*)dst.data(), (__ubuf__ T*)src.data(), src.GetValidRow(), src.GetValidCol());
        else
            Int64RowMinMax<int64Op, T, TileDataOut::Cols, TileDataIn::Cols>(
                (__ubuf__ T*)dst.data(), (__ubuf__ T*)src.data(), src.GetValidRow(), src.GetValidCol());
    } else {
        constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
        TRowReduceImpl<ReduceOp<T>, TileDataOut, TileDataIn, elementsPerRepeat>(
            (__ubuf__ T*)dst.data(), (__ubuf__ T*)src.data(), src.GetValidRow(), src.GetValidCol(), VFIMPL_DEFAULT);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TROWREDUCE_IMPL_COMMON<ROWMAX, Int64Op::Max, false>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TROWREDUCE_IMPL_COMMON<ROWSUM, Int64Op::Add, true>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TROWREDUCE_IMPL_COMMON<ROWMIN, Int64Op::Min, false>(dst, src);
}

} // namespace pto

#endif
