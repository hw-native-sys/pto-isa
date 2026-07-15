/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLREDUCEIDX_HPP
#define TCOLREDUCEIDX_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

// ----------------------------------------------------------------------------
// Helper: comparison + select for one pair of old/new value and index registers
// ----------------------------------------------------------------------------
template <typename VregOldT, typename VregNewT, typename VregIdxT, bool IsArgMax>
PTO_INTERNAL void TColReduceIdxCompare(
    VregOldT& vregOld, VregNewT& vregNew, VregIdxT& vregIdxOld, VregIdxT& vregIdxNew, MaskReg& pregSelect,
    MaskReg& pregMask)
{
    if constexpr (IsArgMax) {
        vcmp_gt(pregSelect, vregNew, vregOld, pregMask);
    } else {
        vcmp_lt(pregSelect, vregNew, vregOld, pregMask);
    }
    vsel(vregIdxOld, vregIdxNew, vregIdxOld, pregSelect);
    vsel(vregOld, vregNew, vregOld, pregSelect);
}

// ----------------------------------------------------------------------------
// Helper: convert 16-bit index vector to 32-bit and store with interleave
// Used by both 8-bit (calls twice for even/odd halves) and 16-bit (calls once)
// ----------------------------------------------------------------------------
template <typename VregSrcT, typename TOUT>
PTO_INTERNAL void TColReduceIdxStoreParts(
    VregSrcT& vregSrc, __ubuf__ TOUT* dstPtr, unsigned offset, MaskReg& pregPart0, MaskReg& pregPart1, MaskReg& pregAll)
{
    RegTensor<TOUT> vregOutEven;
    RegTensor<TOUT> vregOutOdd;
    RegTensor<TOUT> vregOut0;
    RegTensor<TOUT> vregOut1;
    vcvt(vregOutEven, vregSrc, pregAll, PART_EVEN);
    vcvt(vregOutOdd, vregSrc, pregAll, PART_ODD);
    vintlv(vregOut0, vregOut1, vregOutEven, vregOutOdd);
    vsts(vregOut0, dstPtr, offset, NORM_B32, pregPart0);
    vsts(vregOut1, dstPtr, offset + ELE_CNT_B32, NORM_B32, pregPart1);
}

// ----------------------------------------------------------------------------
// Check function (fixed assertion on line dstValidRow == 1)
// ----------------------------------------------------------------------------
template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool WithVal = false>
PTO_INTERNAL void TColReduceIdxCheck(
    unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol, unsigned dstValValidRow = 0,
    unsigned dstValValidCol = 0)
{
    static_assert(
        (sizeof(typename TileDataIn::DType) == 1) || (sizeof(typename TileDataIn::DType) == 2) ||
            (sizeof(typename TileDataIn::DType) == 4),
        "Fix: TCOLREDUCEIDX data type must be b8/b16/b32");

    static_assert(TileDataIn::Loc == pto::TileType::Vec, "Fix: TCOLREDUCEIDX Src TileType must be Vec Tile!");

    static_assert(TileDataOutIdx::Loc == pto::TileType::Vec, "Fix: TCOLREDUCEIDX DstIdx TileType must be Vec Tile!");

    static_assert(TileDataIn::SFractal == SLayout::NoneBox, "Fix: TCOLREDUCEIDX only support Nd or Dn fractal Tile");

    static_assert(
        TileDataOutIdx::isRowMajor && TileDataOutIdx::SFractal == SLayout::NoneBox,
        "Fix: TCOLREDUCEIDX DstIdx only supports Nd fractal Tile");

    PTO_ASSERT(
        srcValidRow != 0 && srcValidCol != 0, "Fix: TCOLREDUCEIDX input shape is invalid, validCol or validRow is 0");
    PTO_ASSERT(dstValidRow == 1, "Fix: TCOLREDUCEIDX output validRow must be 1");
    PTO_ASSERT(srcValidCol == dstValidCol, "Fix: TCOLREDUCEIDX input validCol must equal idx output validCol");

    if constexpr (WithVal) {
        static_assert((sizeof(typename TileDataIn::DType) != 1), "Fix: TCOLREDUCEOPSIDX not support b8");
        static_assert(
            TileDataOutVal::Loc == pto::TileType::Vec, "Fix: TCOLREDUCEOPSIDX DstVal TileType must be Vec Tile");
        static_assert(
            TileDataOutVal::isRowMajor && TileDataOutVal::SFractal == SLayout::NoneBox,
            "Fix: TCOLREDUCEOPSIDX DstVal only supports Nd fractal Tile");
        static_assert(
            std::is_same_v<typename TileDataOutVal::DType, typename TileDataIn::DType>,
            "Fix: TCOLREDUCEOPSIDX DstVal data type must match input type");

        PTO_ASSERT(dstValValidRow == 1, "Fix: TCOLREDUCEOPSIDX output value validRow must be 1");
        PTO_ASSERT(dstValValidCol != 0, "Fix: TCOLREDUCEOPSIDX output value validCol must be non-zero");
        PTO_ASSERT(
            srcValidCol == dstValValidCol, "Fix: TCOLREDUCEOPSIDX input validCol must equal value output validCol");
        PTO_ASSERT(
            dstValValidRow == dstValidRow, "Fix: TCOLREDUCEOPSIDX value and idx output tiles must have same validRow");
        PTO_ASSERT(
            dstValValidCol == dstValidCol, "Fix: TCOLREDUCEOPSIDX value and idx output tiles must have same validCol");

        if constexpr (sizeof(typename TileDataIn::DType) == 2) {
            static_assert(
                std::is_same_v<typename TileDataOutIdx::DType, uint16_t> ||
                    std::is_same_v<typename TileDataOutIdx::DType, int16_t>,
                "Fix: TCOLREDUCEOPSIDX DstIdx data type must be s16 or u16 when input type size <= 2 bytes");
        } else {
            static_assert(
                std::is_same_v<typename TileDataOutIdx::DType, uint32_t> ||
                    std::is_same_v<typename TileDataOutIdx::DType, int32_t>,
                "Fix: TCOLREDUCEOPSIDX DstIdx data type must be s32 or u32 when input type size is 4 bytes");
        }
    } else {
        static_assert(
            std::is_same_v<typename TileDataOutIdx::DType, uint32_t> ||
                std::is_same_v<typename TileDataOutIdx::DType, int32_t>,
            "Fix: TCOLREDUCEOPSIDX DstIdx data type must be s32 or u32");
    }
}

// ----------------------------------------------------------------------------
// Per-chunk processing for 8-bit input
// ----------------------------------------------------------------------------
template <typename TileDataOut, typename TileDataIn, bool IsArgMax>
PTO_INTERNAL void TColReduceIdxChunk8(
    __ubuf__ typename TileDataOut::DType* dstPtr, __ubuf__ typename TileDataIn::DType* srcPtr, unsigned srcValidRow,
    uint16_t repeatTimes, unsigned srcRowStride, unsigned elementsPerRepeat, uint32_t& sregValidCol, MaskReg& pregAll)
{
    using TIN = typename TileDataIn::DType;
    using TOUT = typename TileDataOut::DType;
    using T16 = std::conditional_t<
        std::is_same_v<TIN, int8_t>, vector_s16, std::conditional_t<std::is_same_v<TIN, uint8_t>, vector_u16, void>>;

    vector_s16 vregIndexOldEven;
    vector_s16 vregIndexOldOdd;
    vector_s16 vregIndexNewEven;
    vector_s16 vregIndexNewOdd;
    RegTensor<TIN> vregOld;
    RegTensor<TIN> vregNew;
    T16 vregOldEven;
    T16 vregOldOdd;
    T16 vregNewEven;
    T16 vregNewOdd;
    MaskReg preg0;
    MaskReg preg1;
    MaskReg preg2;
    MaskReg preg3;
    MaskReg pregSelectEven;
    MaskReg pregSelectOdd;

    for (uint16_t j = 0; j < repeatTimes; j++) {
        preg0 = plt_b32(sregValidCol, POST_UPDATE);
        preg1 = plt_b32(sregValidCol, POST_UPDATE);
        preg2 = plt_b32(sregValidCol, POST_UPDATE);
        preg3 = plt_b32(sregValidCol, POST_UPDATE);
        vdup(vregIndexOldEven, 0, pregAll, MODE_ZEROING);
        vdup(vregIndexOldOdd, 0, pregAll, MODE_ZEROING);
        vdup(vregIndexNewEven, 0, pregAll, MODE_ZEROING);
        vdup(vregIndexNewOdd, 0, pregAll, MODE_ZEROING);
        vlds(vregOld, srcPtr, j * elementsPerRepeat, NORM);
        vcvt(vregOldEven, vregOld, pregAll, PART_EVEN);
        vcvt(vregOldOdd, vregOld, pregAll, PART_ODD);

        for (uint16_t i = 1; i < (uint16_t)srcValidRow; i++) {
            vadds(vregIndexNewEven, vregIndexNewEven, 1, pregAll, MODE_ZEROING);
            vadds(vregIndexNewOdd, vregIndexNewOdd, 1, pregAll, MODE_ZEROING);
            vlds(vregNew, srcPtr, i * srcRowStride + j * elementsPerRepeat, NORM);
            vcvt(vregNewEven, vregNew, pregAll, PART_EVEN);
            vcvt(vregNewOdd, vregNew, pregAll, PART_ODD);
            TColReduceIdxCompare<T16, T16, vector_s16, IsArgMax>(
                vregOldEven, vregNewEven, vregIndexOldEven, vregIndexNewEven, pregSelectEven, pregAll);
            TColReduceIdxCompare<T16, T16, vector_s16, IsArgMax>(
                vregOldOdd, vregNewOdd, vregIndexOldOdd, vregIndexNewOdd, pregSelectOdd, pregAll);
        }

        vector_s16 vregTmp0;
        vector_s16 vregTmp1;
        vintlv(vregTmp0, vregTmp1, vregIndexOldEven, vregIndexOldOdd);
        TColReduceIdxStoreParts(vregTmp0, dstPtr, j * elementsPerRepeat, preg0, preg1, pregAll);
        TColReduceIdxStoreParts(vregTmp1, dstPtr, j * elementsPerRepeat + 2 * ELE_CNT_B32, preg2, preg3, pregAll);
    }
}

// ----------------------------------------------------------------------------
// Per-chunk processing for 16-bit and 32-bit input (unified template)
// Differences handled by if constexpr (sizeof(TIN)):
//   16-bit: IdxRegT = vector_s16, compute with pregAll, store via interleave
//   32-bit: IdxRegT = RegTensor<TOUT>, compute with plt_b32, store direct
// ----------------------------------------------------------------------------
template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool IsArgMax, bool WithVal>
PTO_INTERNAL void TColReduceIdxChunk16_32(
    __ubuf__ typename TileDataOutVal::DType* dstValPtr, __ubuf__ typename TileDataOutIdx::DType* dstIdxPtr,
    __ubuf__ typename TileDataIn::DType* srcPtr, unsigned srcValidRow, uint16_t repeatTimes, unsigned srcRowStride,
    unsigned elementsPerRepeat, uint32_t& sregValidCol, MaskReg& pregAll)
{
    using TIN = typename TileDataIn::DType;
    using TIDX = typename TileDataOutIdx::DType;
    using IdxRegT = std::conditional_t<sizeof(TIN) == 2, vector_s16, RegTensor<TIDX>>;
    IdxRegT vregIndexOld;
    IdxRegT vregIndexNew;
    RegTensor<TIN> vregOld;
    RegTensor<TIN> vregNew;
    MaskReg pregSelect;

    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TIDX, DistVST::DIST_NORM>())>();

    for (uint16_t j = 0; j < repeatTimes; j++) {
        MaskReg pregCmp;
        MaskReg pregStore0;
        MaskReg pregStore1;

        if constexpr (sizeof(TIN) == 2 && !WithVal) {
            pregStore0 = plt_b32(sregValidCol, POST_UPDATE);
            pregStore1 = plt_b32(sregValidCol, POST_UPDATE);
            pregCmp = pregAll;
        } else if constexpr (sizeof(TIN) == 4) {
            pregCmp = plt_b32(sregValidCol, POST_UPDATE);
        } else {
            pregCmp = plt_b16(sregValidCol, POST_UPDATE);
        }

        vdup(vregIndexOld, 0, pregCmp, MODE_ZEROING);
        vdup(vregIndexNew, 0, pregCmp, MODE_ZEROING);
        vlds(vregOld, srcPtr, j * elementsPerRepeat, NORM);
        for (uint16_t i = 1; i < (uint16_t)srcValidRow; i++) {
            vadds(vregIndexNew, vregIndexNew, (uint32_t)1, pregCmp, MODE_ZEROING);
            vlds(vregNew, srcPtr, i * srcRowStride + j * elementsPerRepeat, NORM);
            TColReduceIdxCompare<RegTensor<TIN>, RegTensor<TIN>, IdxRegT, IsArgMax>(
                vregOld, vregNew, vregIndexOld, vregIndexNew, pregSelect, pregCmp);
        }

        if constexpr (sizeof(TIN) == 2 && !WithVal) {
            TColReduceIdxStoreParts(vregIndexOld, dstIdxPtr, j * elementsPerRepeat, pregStore0, pregStore1, pregAll);
        } else {
            vsts(vregIndexOld, dstIdxPtr, j * elementsPerRepeat, distValue, pregCmp);
        }

        if constexpr (WithVal) {
            vsts(vregOld, dstValPtr, j * elementsPerRepeat, distValue, pregCmp);
        }
    }
}

// ----------------------------------------------------------------------------
// Unified implementation (thin dispatcher by sizeof(TIN))
// ----------------------------------------------------------------------------
template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool IsArgMax, bool WithVal = false>
__tf__ PTO_INTERNAL void TColReduceIdxImpl(
    typename TileDataOutVal::TileDType __out__ dstValData, typename TileDataOutIdx::TileDType __out__ dstIdxData,
    typename TileDataIn::TileDType __in__ src, unsigned srcValidRow, unsigned srcValidCol)
{
    using TIN = typename TileDataIn::DType;
    using TOUT = typename TileDataOutIdx::DType;

    constexpr unsigned srcRowStride = TileDataIn::Cols;
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(TIN);
    uint16_t repeatTimes = CeilDivision(srcValidCol, elementsPerRepeat);

    __ubuf__ typename TileDataOutVal::DType* dstValPtr =
        (__ubuf__ typename TileDataOutVal::DType*)__cce_get_tile_ptr(dstValData);
    __ubuf__ TOUT* dstIdxPtr = (__ubuf__ TOUT*)__cce_get_tile_ptr(dstIdxData);
    __ubuf__ TIN* srcPtr = (__ubuf__ TIN*)__cce_get_tile_ptr(src);

    __VEC_SCOPE__
    {
        MaskReg pregAll = pset_b8(PAT_ALL);
        uint32_t sregValidCol = srcValidCol;

        if constexpr (sizeof(TIN) == 1) {
            TColReduceIdxChunk8<TileDataOutIdx, TileDataIn, IsArgMax>(
                dstIdxPtr, srcPtr, srcValidRow, repeatTimes, srcRowStride, elementsPerRepeat, sregValidCol, pregAll);
        } else {
            TColReduceIdxChunk16_32<TileDataOutVal, TileDataOutIdx, TileDataIn, IsArgMax, WithVal>(
                dstValPtr, dstIdxPtr, srcPtr, srcValidRow, repeatTimes, srcRowStride, elementsPerRepeat, sregValidCol,
                pregAll);
        }
    }
}

// ----------------------------------------------------------------------------
// Public dispatch (unchanged interface — keeps backward compatibility)
// ----------------------------------------------------------------------------
template <typename TileDataOut, typename TileDataIn, bool IsArgMax>
PTO_INTERNAL void TCOLARG_DISPATCH(TileDataOut& dst, TileDataIn& src)
{
    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();
    TColReduceIdxCheck<TileDataIn, TileDataOut, TileDataIn>(
        srcValidRow, srcValidCol, dst.GetValidRow(), dst.GetValidCol());
    TColReduceIdxImpl<TileDataIn, TileDataOut, TileDataIn, IsArgMax>(
        src.data(), dst.data(), src.data(), srcValidRow, srcValidCol);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool IsArgMax>
PTO_INTERNAL void TCOLARG_DISPATCH(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src)
{
    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();
    TColReduceIdxCheck<TileDataOutVal, TileDataOutIdx, TileDataIn, true>(
        srcValidRow, srcValidCol, dstIdx.GetValidRow(), dstIdx.GetValidCol(), dstVal.GetValidRow(),
        dstVal.GetValidCol());
    TColReduceIdxImpl<TileDataOutVal, TileDataOutIdx, TileDataIn, IsArgMax, true>(
        dstVal.data(), dstIdx.data(), src.data(), srcValidRow, srcValidCol);
}

// ==========================================================================================
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TCOLARG_DISPATCH<TileDataOut, TileDataIn, false>(dst, src);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp)
{
    TCOLARG_DISPATCH<TileDataOut, TileDataIn, true>(dst, src);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp)
{
    TCOLARG_DISPATCH<TileDataOutVal, TileDataOutIdx, TileDataIn, true>(dstVal, dstIdx, src);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp)
{
    TCOLARG_DISPATCH<TileDataOutVal, TileDataOutIdx, TileDataIn, false>(dstVal, dstIdx, src);
}

} // namespace pto
#endif