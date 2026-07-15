/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TAND_HPP
#define TAND_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a2a3/TBinOp.hpp"

namespace pto {

template <typename T>
struct AndOp {
    PTO_INTERNAL static void BinInstr(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeats)
    {
        if constexpr (sizeof(T) == 2) {
            vand(
                (__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src0, (__ubuf__ uint16_t*)src1, repeats, 1, 1, 1, 8, 8, 8);
        } else {
            vand(dst, src0, src1, repeats, 1, 1, 1, 8, 8, 8);
        }
    }
    PTO_INTERNAL static void BinInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeats, uint8_t dstRepeatStride,
        uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        if constexpr (sizeof(T) == 2) {
            vand(
                (__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src0, (__ubuf__ uint16_t*)src1, repeats, 1, 1, 1,
                dstRepeatStride, src0RepeatStride, src1RepeatStride);
        } else {
            vand(dst, src0, src1, repeats, 1, 1, 1, dstRepeatStride, src0RepeatStride, src1RepeatStride);
        }
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ PTO_INTERNAL void TAnd(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols)
{
    using TOrig = typename TileDataDst::DType;
    using TRANS = std::conditional_t<sizeof(TOrig) == sizeof(uint32_t), B322B16Trait<TOrig>, B82B16Trait<TOrig>>;
    using T = typename TRANS::TransType;
    int transValidCol = TRANS::TransSize(validCols);
    constexpr int blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr int elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    constexpr unsigned dstRowStride = TRANS::template TransStride<TileDataDst::RowStride>();
    constexpr unsigned src0RowStride = TRANS::template TransStride<TileDataSrc0::RowStride>();
    constexpr unsigned src1RowStride = TRANS::template TransStride<TileDataSrc1::RowStride>();
    if constexpr (dstRowStride == src0RowStride && dstRowStride == src1RowStride && sizeof(TOrig) == sizeof(T)) {
        BinaryInstr<AndOp<T>, T, TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride>(
            dstPtr, src0Ptr, src1Ptr, validRows, transValidCol);
    } else {
        BinaryInstr<AndOp<T>, T, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dstPtr, src0Ptr, src1Ptr, validRows, transValidCol);
    }
    return;
}

template <typename T, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TAndCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TAND the data type of dst must be consistent with of src0 and src1.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TAND only support row major layout.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TAND input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TAND input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TAND_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    TAndCheck<typename TileDataDst::DType, TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);

    TAnd<TileDataDst, TileDataSrc0, TileDataSrc1>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif