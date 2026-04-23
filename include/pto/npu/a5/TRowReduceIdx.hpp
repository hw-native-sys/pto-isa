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

#ifndef __ROW_REDUCE_IDX__
#define __ROW_REDUCE_IDX__

#include "common.hpp"
#include "pto/common/pto_tile.hpp"
#include "TPartBinOps.hpp"
#include <math.h>
#include <type_traits>

namespace pto {

template <typename T>
struct ROWIDXMAX {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Min;
    using PaddingType = typename Padding<T>::Type;
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &preg)
    {
        vcmax(dst, src, preg, MODE_ZEROING);
    }
    static PTO_INTERNAL void Compare(MaskReg &dst, RegType &src0, RegType &src1, MaskReg &preg)
    {
        vcmp_lt(dst, src0, src1, preg);
    }
};

template <typename T>
struct ROWIDXMIN {
    static constexpr typename Padding<T>::Type InitVal = Padding<T>::Max;
    using PaddingType = typename Padding<T>::Type;
    using RegType = typename TypeGet<T>::T;
    static PTO_INTERNAL void Reduce(RegType &dst, RegType &src, MaskReg &preg)
    {
        vcmin(dst, src, preg, MODE_ZEROING);
    }
    static PTO_INTERNAL void Compare(MaskReg &dst, RegType &src0, RegType &src1, MaskReg &preg)
    {
        vcmp_gt(dst, src0, src1, preg);
    }
};

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool outputVal>
PTO_INTERNAL void TRowReduceIdxCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValValidRow,
                                     uint32_t dstIdxValidRow)
{
    using TVal = typename TileDataIn::DType;
    using TIdx = typename TileDataOutIdx::DType;
    if constexpr (outputVal) {
        static_assert(
            (sizeof(TVal) == sizeof(half) && (std::is_same_v<int16_t, TIdx> || std::is_same_v<uint16_t, TIdx>)) ||
                (sizeof(TVal) == sizeof(float) && (std::is_same_v<int32_t, TIdx> || std::is_same_v<uint32_t, TIdx>)),
            "Input and output tile data types must match. "
            "Fix: Ensure TileDataOutIdx uses the same DType as TileDataIn.");
        TRowReduceCheck<TileDataOutVal, TileDataIn, false>(srcValidRows, srcValidCols, dstValValidRow);
    } else {
        static_assert(std::is_same_v<int32_t, TIdx> || std::is_same_v<uint32_t, TIdx>,
                      "Input and output tile data types must match. "
                      "Fix: Ensure TileDataOutIdx uses the same DType as TileDataIn.");
    }
    TRowReduceCheck<TileDataOutIdx, TileDataIn, true>(srcValidRows, srcValidCols, dstIdxValidRow);
}

template <typename ReduceIdxOp, typename TDst, typename TSrc>
PTO_INTERNAL void UpdateIdxValue(RegTensor<TDst> &vregIdxOrig, RegTensor<TSrc> &vregValOrig, RegTensor<TSrc> &vregVal,
                                 RegTensor<TSrc> &vregZero, TDst currentOffset, MaskReg &pRegOneElem)
{
    MaskReg pregCmp;
    RegTensor<TDst> vregIdx;
    vdintlv(vregVal, (RegTensor<TSrc> &)vregIdx, vregVal, vregZero);
    vadds(vregIdx, vregIdx, currentOffset, pRegOneElem, MODE_ZEROING);
    ReduceIdxOp::Compare(pregCmp, vregValOrig, vregVal, pRegOneElem);
    vsel(vregValOrig, vregVal, vregValOrig, pregCmp);
    vsel(vregIdxOrig, vregIdx, vregIdxOrig, pregCmp);
}

template <typename ReduceIdxOp, typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool outputVal,
          bool postUpdate>
PTO_INTERNAL void TRowReduceIdxProc(__ubuf__ typename TileDataOutVal::DType *dstValPtr,
                                    __ubuf__ typename TileDataOutIdx::DType *dstIdxPtr,
                                    __ubuf__ typename TileDataIn::DType *srcPtr, uint32_t rows, uint32_t cols)
{
    using TDstVal = typename TileDataOutVal::DType;
    using TDstIdx = typename TileDataOutIdx::DType;
    using TSrc = typename TileDataIn::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TSrc);
    uint16_t repeatTimes = CeilDivision(cols, elementsPerRepeat);
    constexpr auto distValueVal =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDstVal, DistVST::DIST_ONEPT>())>();
    constexpr auto distValueIdx =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<TDstIdx, DistVST::DIST_ONEPT>())>();
    RegTensor<TSrc> vregZero, vregSrc, vregValOrig;
    RegTensor<TDstIdx> vregIdxOrig;
    vbr(vregZero, 0);

    uint32_t dstDup = 1;
    MaskReg pRegOneElem = CreatePredicate<TSrc>(dstDup);
    for (uint16_t i = 0; i < (uint16_t)rows; ++i) {
        vdup((RegTensor<typename ReduceIdxOp::PaddingType> &)vregValOrig, ReduceIdxOp::InitVal, pRegOneElem,
             MODE_ZEROING);
        vbr(vregIdxOrig, 0);
        __ubuf__ TSrc *rowPtr = srcPtr + i * TileDataIn::RowStride;
        uint32_t sregCol = cols;
        for (uint16_t j = 0; j < (uint16_t)repeatTimes; j++) {
            MaskReg pRegSrc = CreatePredicate<TSrc>(sregCol);
            if constexpr (postUpdate) {
                vlds(vregSrc, rowPtr, elementsPerRepeat, NORM, POST_UPDATE);
            } else {
                vlds(vregSrc, srcPtr, i * TileDataIn::RowStride + j * elementsPerRepeat, NORM);
            }
            ReduceIdxOp::Reduce(vregSrc, vregSrc, pRegSrc);
            UpdateIdxValue<ReduceIdxOp, TDstIdx, TSrc>(vregIdxOrig, vregValOrig, vregSrc, vregZero,
                                                       j * elementsPerRepeat, pRegOneElem);
        }
        if constexpr (postUpdate) {
            vsts(vregIdxOrig, dstIdxPtr, TileDataOutIdx::RowStride, distValueIdx, pRegOneElem, POST_UPDATE);
        } else {
            vsts(vregIdxOrig, dstIdxPtr, i * TileDataOutIdx::RowStride, distValueIdx, pRegOneElem);
        }
        if constexpr (outputVal) {
            if constexpr (postUpdate) {
                vsts(vregValOrig, dstValPtr, TileDataOutVal::RowStride, distValueVal, pRegOneElem, POST_UPDATE);
            } else {
                vsts(vregValOrig, dstValPtr, i * TileDataOutVal::RowStride, distValueVal, pRegOneElem);
            }
        }
    }
}

template <typename ReduceIdxOp, typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool outputVal>
PTO_INTERNAL void TRowReduceIdxImpl(__ubuf__ typename TileDataOutVal::DType *dstValPtr,
                                    __ubuf__ typename TileDataOutIdx::DType *dstIdxPtr,
                                    __ubuf__ typename TileDataIn::DType *srcPtr, uint32_t rows, uint32_t cols,
                                    unsigned version)
{
    __VEC_SCOPE__
    {
        if (version == VFIMPL_2D_NO_POST_UPDATE) {
            TRowReduceIdxProc<ReduceIdxOp, TileDataOutVal, TileDataOutIdx, TileDataIn, outputVal, false>(
                dstValPtr, dstIdxPtr, srcPtr, rows, cols);
        } else {
            TRowReduceIdxProc<ReduceIdxOp, TileDataOutVal, TileDataOutIdx, TileDataIn, outputVal, true>(
                dstValPtr, dstIdxPtr, srcPtr, rows, cols);
        }
    }
}

template <typename TVal, typename TIdx, typename TIn, bool outputVal>
__tf__ PTO_INTERNAL OP_NAME(TROWARGMAX)
    OP_TYPE(reduce) void TRowArgMax(typename TVal::TileDType __out__ dstVal, typename TIdx::TileDType __out__ dstIdx,
                                    typename TIn::TileDType __in__ src, uint32_t srcValidRows, uint32_t srcValidCols,
                                    uint32_t dstValValidRow, uint32_t dstIdxValidRow,
                                    unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    TRowReduceIdxCheck<TVal, TIdx, TIn, outputVal>(srcValidRows, srcValidCols, dstValValidRow, dstIdxValidRow);
    __ubuf__ typename TVal::DType *dstValPtr = __cce_get_tile_ptr(dstVal);
    __ubuf__ typename TIdx::DType *dstIdxPtr = __cce_get_tile_ptr(dstIdx);
    __ubuf__ typename TIn::DType *srcPtr = __cce_get_tile_ptr(src);
    TRowReduceIdxImpl<ROWIDXMAX<typename TIn::DType>, TVal, TIdx, TIn, outputVal>(dstValPtr, dstIdxPtr, srcPtr,
                                                                                  srcValidRows, srcValidCols, version);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowArgMax<TileDataOut, TileDataOut, TileDataIn, false>(dst.data(), dst.data(), src.data(), src.GetValidRow(),
                                                            src.GetValidCol(), dst.GetValidRow(), dst.GetValidRow());
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp)
{
    TRowArgMax<TileDataOutVal, TileDataOutIdx, TileDataIn, true>(dstVal.data(), dstIdx.data(), src.data(),
                                                                 src.GetValidRow(), src.GetValidCol(),
                                                                 dstVal.GetValidRow(), dstIdx.GetValidRow());
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool outputVal>
__tf__ PTO_INTERNAL OP_NAME(TROWARGMIN)
    OP_TYPE(reduce) void TRowArgMin(typename TileDataOutVal::TileDType __out__ dstVal,
                                    typename TileDataOutIdx::TileDType __out__ dstIdx,
                                    typename TileDataIn::TileDType __in__ src, uint32_t srcValidRows,
                                    uint32_t srcValidCols, uint32_t dstValValidRow, uint32_t dstIdxValidRow,
                                    unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TDstVal = typename TileDataOutVal::DType;
    using TDstIdx = typename TileDataOutIdx::DType;
    using TSrc = typename TileDataIn::DType;
    TRowReduceIdxCheck<TileDataOutVal, TileDataOutIdx, TileDataIn, outputVal>(srcValidRows, srcValidCols,
                                                                              dstValValidRow, dstIdxValidRow);
    __ubuf__ TDstVal *dstValPtr = __cce_get_tile_ptr(dstVal);
    __ubuf__ TDstIdx *dstIdxPtr = __cce_get_tile_ptr(dstIdx);
    __ubuf__ TSrc *srcPtr = __cce_get_tile_ptr(src);
    TRowReduceIdxImpl<ROWIDXMIN<TSrc>, TileDataOutVal, TileDataOutIdx, TileDataIn, outputVal>(
        dstValPtr, dstIdxPtr, srcPtr, srcValidRows, srcValidCols, version);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowArgMin<TileDataOut, TileDataOut, TileDataIn, false>(dst.data(), dst.data(), src.data(), src.GetValidRow(),
                                                            src.GetValidCol(), dst.GetValidRow(), dst.GetValidRow());
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOutVal &dstVal, TileDataOutIdx &dstIdx, TileDataIn &src, TileDataTmp &tmp)
{
    TRowArgMin<TileDataOutVal, TileDataOutIdx, TileDataIn, true>(dstVal.data(), dstIdx.data(), src.data(),
                                                                 src.GetValidRow(), src.GetValidCol(),
                                                                 dstVal.GetValidRow(), dstIdx.GetValidRow());
}
} // namespace pto

#endif
