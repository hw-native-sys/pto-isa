/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLSUM_HPP
#define TCOLSUM_HPP

#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

namespace pto {
template <typename T, int SrcStride, int DstStride>
PTO_INTERNAL void BinarySum(__ubuf__ T* dst, __ubuf__ T* src, int validRow, int validCol)
{
    set_mask_count();
    set_vector_mask(0, validCol);
    for (uint32_t i = 0; i < validRow / 2; i++) {
        vadd(dst + i * DstStride, src + 2 * i * SrcStride, src + (2 * i + 1) * SrcStride, 0, 1, 1, 1, 8, 8, 8);
    }
    pipe_barrier(PIPE_V);

    if (validRow % 2 == 1) {
        vadd(dst, dst, src + (validRow - 1) * SrcStride, 0, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, int SrcStride, int DstStride>
PTO_INTERNAL void SequentialSum(__ubuf__ T* dst, __ubuf__ T* src, int validRow, int validCol)
{
    set_mask_count();
    set_vector_mask(0, validCol);
    for (int i = 1; i < validRow; i++) {
        vadd(dst, dst, src + i * SrcStride, 0, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <
    typename T, typename TileDataDst, typename TileDataSrc, typename TileDataTmp, int srcStride, int dstStride,
    int tmpStride, bool IsBinary>
__tf__ PTO_INTERNAL void TColSum(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataTmp::TileDType __in__ tmp, int validRow, int validCol)
{
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);

    constexpr int DTypeSize = sizeof(T);
    int lenBurst = (validCol * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

    if (validRow == 1) {
        pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, lenBurst, 0, 0);
        pipe_barrier(PIPE_V);
        return;
    }

    if (IsBinary) {
        BinarySum<T, srcStride, tmpStride>(tmpPtr, srcPtr, validRow, validCol);
        int cnt = validRow / 2;
        while (cnt > 1) {
            BinarySum<T, tmpStride, tmpStride>(tmpPtr, tmpPtr, cnt, validCol);
            pipe_barrier(PIPE_V);
            cnt /= 2;
        }
        pto_copy_ubuf_to_ubuf(dstPtr, tmpPtr, 1, lenBurst, 0, 0);
        pipe_barrier(PIPE_V);
    } else {
        pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, lenBurst, 0, 0);
        pipe_barrier(PIPE_V);
        SequentialSum<T, srcStride, dstStride>(dstPtr, srcPtr, validRow, validCol);
    }
}

template <typename T, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TColSumCheck()
{
    static_assert(
        TileDataDst::Loc == pto::TileType::Vec && TileDataSrc::Loc == pto::TileType::Vec &&
            TileDataTmp::Loc == pto::TileType::Vec,
        "Fix: TCOLSUM only support Vec Tile");
    static_assert(
        TileDataSrc::isRowMajor && TileDataSrc::SFractal == SLayout::NoneBox,
        "Fix: TCOLSUM only support Nd fractal Tile");
    static_assert(
        TileDataDst::isRowMajor && TileDataDst::SFractal == SLayout::NoneBox,
        "Fix: TCOLSUM only support Nd fractal Tile");
    static_assert(
        TileDataTmp::isRowMajor && TileDataTmp::SFractal == SLayout::NoneBox,
        "Fix: TCOLSUM only support Nd fractal Tile");
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>,
        "Fix: TCOLSUM input data type is not supported by this instruction.");
    static_assert(
        std::is_same_v<typename TileDataDst::DType, T>,
        "Fix: TCOLSUM input data type must be consistent with the output data type.");
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TCOLSUM_IMPL(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, bool IsBinary)
{
    using T = typename TileDataSrc::DType;
    TColSumCheck<T, TileDataDst, TileDataSrc, TileDataTmp>();

    int validRow = src.GetValidRow();
    int validCol = src.GetValidCol();
    constexpr int srcStride = TileDataSrc::RowStride;
    constexpr int dstStride = TileDataDst::RowStride;

    PTO_ASSERT(
        validCol == dst.GetValidCol(), "Fix: TCOLSUM input valid col must be consistent with the output valid row.");
    if (validRow == 0 || validCol == 0) {
        return;
    }

    if (IsBinary) {
        constexpr int tmpStride = TileDataTmp::RowStride * sizeof(typename TileDataTmp::DType) / sizeof(T);
        PTO_ASSERT(
            validCol <= tmpStride, "Fix: TCOLSUM input valid columns must be less than or equal to the tmp columns.");
        TColSum<T, TileDataDst, TileDataSrc, TileDataTmp, srcStride, dstStride, tmpStride, true>(
            dst.data(), src.data(), tmp.data(), validRow, validCol);
    } else {
        TColSum<T, TileDataDst, TileDataSrc, TileDataTmp, srcStride, dstStride, dstStride, false>(
            dst.data(), src.data(), tmp.data(), validRow, validCol);
    }
}

template <typename T, typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TColSum(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src, int validRow, int validCol)
{
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);

    constexpr int DTypeSize = sizeof(T);
    int lenBurst = (validCol * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

    pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, lenBurst, 0, 0);
    if (validRow > 1) {
        pipe_barrier(PIPE_V);
        SequentialSum<T, TileDataSrc::RowStride, TileDataDst::RowStride>(dstPtr, srcPtr, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TCOLSUM_IMPL(TileDataDst& dst, TileDataSrc& src)
{
    using T = typename TileDataSrc::DType;
    TColSumCheck<T, TileDataDst, TileDataSrc, TileDataDst>();

    int validRow = src.GetValidRow();
    int validCol = src.GetValidCol();

    PTO_ASSERT(
        validCol == dst.GetValidCol(), "Fix: TCOLSUM input valid col must be consistent with the output valid row.");
    if (validRow == 0 || validCol == 0) {
        return;
    }

    TColSum<T, TileDataDst, TileDataSrc>(dst.data(), src.data(), validRow, validCol);
}
} // namespace pto
#endif