/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef T_BITWISE_SCALAR_OP_HPP
#define T_BITWISE_SCALAR_OP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "TBinSOp.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TShiftCheck(const TileDataDst& dst, const TileDataSrc& src)
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same_v<T, typename TileDataDst::DType>, "The data type of dst must be consistent with src.");
    static_assert(
        std::is_same<T, int32_t>::value || std::is_same<T, int>::value || std::is_same<T, int16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, uint16_t>::value ||
            std::is_same<T, unsigned int>::value,
        "Invalid data type");
    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src and dst tiles must be TileType::Vec.");
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    unsigned srcValidRow = src.GetValidRow();
    unsigned srcValidCol = src.GetValidCol();
    PTO_ASSERT((dstValidRow != 0 && dstValidCol != 0), "Number of cols of dst must be non-zero.");
    PTO_ASSERT((srcValidRow != 0 && srcValidCol != 0), "Number of cols of src must be non-zero.");
    PTO_ASSERT(dstValidCol == srcValidCol, "Number of cols of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == srcValidCol, "Number of rows of src and dst must be the same.");
}

template <typename Op, typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TShiftS(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc::TileDType __in__ srcData,
    typename TileDataSrc::DType scalar, unsigned validRow, unsigned validCol)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    TBinSInstr<Op, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(
        dst, src, scalar, validRow, validCol);
}

template <typename T>
struct ShrSOp {
    PTO_INTERNAL static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats)
    {
        vshr(dst, src0, src1, repeats, 1, 1, 8, 8, false);
    }
    PTO_INTERNAL static void BinSInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        vshr(dst, src0, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride, false);
    }
};

template <typename T>
struct ShlSOp {
    PTO_INTERNAL static void BinSInstr(__ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats)
    {
        vshl(dst, src0, src1, repeats, 1, 1, 8, 8);
    }
    PTO_INTERNAL static void BinSInstr(
        __ubuf__ T* dst, __ubuf__ T* src0, T src1, uint8_t repeats, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        vshl(dst, src0, src1, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSHLS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar)
{
    TShiftCheck(dst, src);
    TShiftS<ShlSOp<typename TileDataSrc::DType>, TileDataDst, TileDataSrc>(
        dst.data(), src.data(), scalar, dst.GetValidRow(), dst.GetValidCol());
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSHRS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar)
{
    TShiftCheck(dst, src);
    TShiftS<ShrSOp<typename TileDataSrc::DType>, TileDataDst, TileDataSrc>(
        dst.data(), src.data(), scalar, dst.GetValidRow(), dst.GetValidCol());
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TANDS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar)
{
#ifndef __PTO_AUTO__
    PTO_ASSERT(
        dst.data() != src.data(), "Setting the source Tile and destination Tile to the same memory is unsupported");
#endif
    TEXPANDS_IMPL(dst, scalar);
    pipe_barrier(PIPE_V);
    TAND_IMPL(dst, src, dst);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TORS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar)
{
#ifndef __PTO_AUTO__
    PTO_ASSERT(
        dst.data() != src.data(), "Setting the source Tile and destination Tile to the same memory is unsupported");
#endif
    TEXPANDS_IMPL(dst, scalar);
    pipe_barrier(PIPE_V);
    TOR_IMPL(dst, src, dst);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TXORS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, TileDataTmp& tmp)
{
#ifndef __PTO_AUTO__
    PTO_ASSERT(
        dst.data() != src.data() && dst.data() != tmp.data() && src.data() != tmp.data(),
        "dst, src, tmp must in different memory.");
#endif
    TORS_IMPL(tmp, src, scalar);
    pipe_barrier(PIPE_V);
    TANDS_IMPL(dst, src, scalar);
    pipe_barrier(PIPE_V);
    TNOT_IMPL(dst, dst);
    pipe_barrier(PIPE_V);
    TAND_IMPL(dst, dst, tmp);
}
} // namespace pto

#endif
