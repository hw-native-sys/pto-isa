/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPAIRREDUCESUM_HPP
#define TPAIRREDUCESUM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {

template <typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
PTO_INTERNAL void TPairReduceSum_1D_NoPostUpdate(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned halfElementsPerRepeat = ElementsPerRepeat >> 1;
    uint16_t repeatTimes = CeilDivision(validRows * validCols, ElementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1;
        MaskReg pregLoad;
        MaskReg pregStore;

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned totalElem = validRows * validCols;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            uint32_t remainLoad = totalElem - i * ElementsPerRepeat;
            uint32_t curLoadCount =
                (remainLoad > (uint32_t)ElementsPerRepeat) ? (uint32_t)ElementsPerRepeat : remainLoad;
            uint32_t curStoreCount = curLoadCount >> 1;
            uint32_t sregLoad = curLoadCount;
            uint32_t sregStore = curStoreCount;
            pregLoad = CreatePredicate<T>(sregLoad);
            pregStore = CreatePredicate<T>(sregStore);
            vlds(vreg0, src0Ptr, i * ElementsPerRepeat, NORM);
            vcpadd(vreg1, vreg0, pregLoad);
            vsts(vreg1, dstPtr, i * halfElementsPerRepeat, distValue, pregStore);
        }
    }
}

template <
    typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride = DstRowStride>
PTO_INTERNAL void TPairReduceSum_2D_NoPostUpdate(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned halfElementsPerRepeat = ElementsPerRepeat >> 1;
    uint16_t repeatTimes = CeilDivision(validCols, ElementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1;
        MaskReg pregLoad;
        MaskReg pregStore;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                uint32_t remainLoad = validCols - j * ElementsPerRepeat;
                uint32_t curLoadCount =
                    (remainLoad > (uint32_t)ElementsPerRepeat) ? (uint32_t)ElementsPerRepeat : remainLoad;
                uint32_t curStoreCount = curLoadCount >> 1;
                uint32_t sregLoad = curLoadCount;
                uint32_t sregStore = curStoreCount;
                pregLoad = CreatePredicate<T>(sregLoad);
                pregStore = CreatePredicate<T>(sregStore);
                vlds(vreg0, src0Ptr, i * Src0RowStride + j * ElementsPerRepeat, NORM);
                vcpadd(vreg1, vreg0, pregLoad);
                vsts(vreg1, dstPtr, i * DstRowStride + j * halfElementsPerRepeat, distValue, pregStore);
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc0, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
PTO_INTERNAL void TPairReduceSumInstr(
    __ubuf__ typename TileDataDst::DType* dst, __ubuf__ typename TileDataSrc0::DType* src0, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr bool isContiguous =
        ((TileDataDst::ValidCol == TileDataDst::Cols) && (TileDataSrc0::ValidCol == TileDataSrc0::Cols)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc0::Rows == 1));

    if constexpr (isContiguous) {
        TPairReduceSum_1D_NoPostUpdate<T, ElementsPerRepeat, BlockSizeElem>(dst, src0, validRows, validCols);
    } else {
        TPairReduceSum_2D_NoPostUpdate<T, ElementsPerRepeat, BlockSizeElem, dstRowStride, src0RowStride>(
            dst, src0, validRows, validCols);
    }
}

template <typename TileDataDst, typename TileDataSrc0, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TPAIRREDUCESUM) OP_TYPE(element_wise) void TPairReduceSum(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);

    TPairReduceSumInstr<TileDataDst, TileDataSrc0, ElementsPerRepeat, BlockSizeElem>(
        dstPtr, src0Ptr, validRows, validCols);
    return;
}

template <typename TileDataDst, typename TileDataSrc0>
PTO_INTERNAL void TPairReduceSumCheck(const TileDataDst& dst, const TileDataSrc0& src0)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>, "Fix: TPAIRREDUCESUM has invalid data type.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType>,
        "Fix: TPAIRREDUCESUM input tile src0 and dst tile data type mismatch.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor, "Fix: TPAIRREDUCESUM only support row major layout.");
    unsigned validCols = dst.GetValidCol();
    unsigned validRows = dst.GetValidRow();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TPAIRREDUCESUM input tile src0 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0>
PTO_INTERNAL void TPAIRREDUCESUM_IMPL(TileDataDst& dst, TileDataSrc0& src0)
{
    using T = typename TileDataDst::DType;
    TPairReduceSumCheck<TileDataDst, TileDataSrc0>(dst, src0);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);

    TPairReduceSum<TileDataDst, TileDataSrc0, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
