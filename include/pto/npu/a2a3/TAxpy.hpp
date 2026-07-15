/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TAXPY_HPP
#define TAXPY_HPP

#include <pto/common/constants.hpp>

namespace pto {
template <typename T>
struct AxpyOp {
    PTO_INTERNAL static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats)
    {
        vaxpy(dst, src0, src1, repeats, 1, 1, 8, 8);
    }
    PTO_INTERNAL static void BinSInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        vaxpy(dst, src0, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};

template <typename T, typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TAxpy(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc::TileDType __in__ srcData, T scalar,
    unsigned validRow, unsigned validCol)
{
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    TBinSInstr<AxpyOp<T>, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(
        dst, src, scalar, validRow, validCol);
}

template <typename T, typename U, unsigned DstRowStride, unsigned Src0RowStride>
PTO_INTERNAL void AxpyCountMode(__ubuf__ T* dstPtr, __ubuf__ U* src0Ptr, U scalar, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        // src一个repeat是4个block，dst8个block
        vaxpy(dstPtr + i * DstRowStride, src0Ptr + i * Src0RowStride, scalar, 0, 1, 1, 8, 4);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename T, typename U, unsigned DstRowStride, unsigned Src0RowStride>
PTO_INTERNAL void AxpyNormModeTail(
    __ubuf__ T* dstPtr, __ubuf__ U* src0Ptr, U scalar, unsigned validRow, unsigned validCol)
{
    constexpr unsigned dstElementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned srcElementsPerRepeat = pto::REPEAT_BYTE / sizeof(U) / 2; // src一个repeat只有4个block，所以除以2
    constexpr uint8_t DstRepeatStride = (uint8_t)(DstRowStride / (BLOCK_BYTE_SIZE / sizeof(T)));
    constexpr uint8_t SrcRepeatStride = (uint8_t)(Src0RowStride / (BLOCK_BYTE_SIZE / sizeof(U)));
    if constexpr (DstRowStride < dstElementsPerRepeat || Src0RowStride < srcElementsPerRepeat) {
        SetContMaskByDType<U>(validCol);
        vaxpy(dstPtr, src0Ptr, scalar, validRow, 1, 1, DstRepeatStride, SrcRepeatStride);
        SetFullVecMaskByDType<U>();
    } else {
        unsigned numLoop = validCol / dstElementsPerRepeat;
        unsigned numRemainAfterLoop = validCol % dstElementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            vaxpy(dstPtr, src0Ptr, scalar, validRow, 1, 1, DstRepeatStride, SrcRepeatStride);
            dstPtr += dstElementsPerRepeat;
            src0Ptr += srcElementsPerRepeat;
        }
        if (numRemainAfterLoop) {
            SetContMaskByDType<U>(numRemainAfterLoop);
            vaxpy(dstPtr, src0Ptr, scalar, validRow, 1, 1, DstRepeatStride, SrcRepeatStride);
            SetFullVecMaskByDType<U>();
        }
    }
}

template <typename T, typename U, unsigned DstRowStride, unsigned Src0RowStride>
PTO_INTERNAL void AxpyNormMode(__ubuf__ T* dstPtr, __ubuf__ U* src0Ptr, U scalar, unsigned validRow, unsigned validCol)
{
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned numRemainAfterLoop = validRow % REPEAT_MAX;
    constexpr unsigned dstOffset = REPEAT_MAX * DstRowStride;
    constexpr unsigned src0Offset = REPEAT_MAX * Src0RowStride;
    for (unsigned i = 0; i < numLoop; i++) {
        AxpyNormModeTail<T, U, DstRowStride, Src0RowStride>(dstPtr, src0Ptr, scalar, REPEAT_MAX, validCol);
        dstPtr += dstOffset;
        src0Ptr += src0Offset;
    }
    if (numRemainAfterLoop) {
        AxpyNormModeTail<T, U, DstRowStride, Src0RowStride>(dstPtr, src0Ptr, scalar, numRemainAfterLoop, validCol);
    }
}

template <typename T, typename U, typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TAxpy(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc::TileDType __in__ srcData, U scalar,
    unsigned validRow, unsigned validCol)
{
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ U* srcPtr = (__ubuf__ U*)__cce_get_tile_ptr(srcData);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    // 类型为float和half时，vaxpy取4个block的src和8个block的dst
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstBlockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned srcBlockSizeElem = BLOCK_BYTE_SIZE / sizeof(U);
    constexpr bool repeatStrideOverflow = dstStride / dstBlockSizeElem > 255 || srcStride / srcBlockSizeElem > 255;
    bool useCountMode = repeatStrideOverflow || validCol / elementsPerRepeat > validRow;
    if (useCountMode) {
        AxpyCountMode<T, U, dstStride, srcStride>(dstPtr, srcPtr, scalar, validRow, validCol);
    } else {
        AxpyNormMode<T, U, dstStride, srcStride>(dstPtr, srcPtr, scalar, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TAXPY_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;
    using U = typename TileDataSrc::DType;
    static_assert(std::is_same_v<T, half> || std::is_same_v<T, float>, "TAXPY: Invalid data type");
    static_assert(
        std::is_same_v<T, U> || (std::is_same_v<T, float> && std::is_same_v<U, half>),
        "TAXPY: The data type of dst must be consistent with src or dst is float while src is half.");

    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");

    PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of cols of src and dst must be the same.");
    PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    if ((dstValidRow != 0 && dstValidCol != 0) &&
        (dstValidRow == src.GetValidRow() && dstValidCol == src.GetValidCol())) {
        if constexpr (std::is_same_v<T, U>) {
            TAxpy<T, TileDataDst, TileDataSrc>(dst.data(), src.data(), scalar, dstValidRow, dstValidCol);
        } else {
            TAxpy<T, U, TileDataDst, TileDataSrc>(dst.data(), src.data(), scalar, dstValidRow, dstValidCol);
        }
    } else {
        PTO_ASSERT(false, "TAXPY: dstTile validRow/validCol must be consistent with of src.");
    }
}
} // namespace pto

#endif