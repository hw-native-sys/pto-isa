/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXPANDS_HPP
#define TEXPANDS_HPP

#include <pto/common/constants.hpp>
#include "TBinSOp.hpp"

namespace pto {
inline namespace TExpandsInternel {
constexpr const int EXPANDS_MAX_SUPPORT_REPEAT_TIMES = 32767; // [0:14]
} // namespace TExpandsInternel
template <typename T>
struct ExpandSOp {
    PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *src0, T scalar, uint8_t repeats)
    {
        vector_dup(dst, scalar, repeats, 1, 1, 8, 8);
    }
    PTO_INTERNAL static void BinSInstr(__ubuf__ T *dst, __ubuf__ T *src0, T scalar, uint8_t repeats,
                                       uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        vector_dup(dst, scalar, repeats, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};
template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned stride>
__tf__ PTO_INTERNAL void TExpandS(typename TileData::TileDType __out__ dst, typename TileData::DType scalar,
                                  unsigned validRow, unsigned validCol)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);

    TBinSInstr<ExpandSOp<typename TileData::DType>, TileData, TileData, elementsPerRepeat, blockSizeElem, stride,
               stride>(dstPtr, dstPtr, scalar, validRow, validCol);
}

template <typename TileData>
PTO_INTERNAL void TExpandSInstrMat(__cbuf__ typename TileData::DType *dstPtr, int64_t repeatConfig,
                                   typename TileData::DType value)
{
    if constexpr (sizeof(typename TileData::DType) == 4) {
        if constexpr (std::is_same_v<typename TileData::DType, float>) {
            uint32_t bits = *(uint32_t *)(&value);
            create_cbuf_matrix(dstPtr, repeatConfig, bits);
        } else {
            create_cbuf_matrix(dstPtr, repeatConfig, static_cast<uint32_t>(value));
        }
    } else if constexpr (sizeof(typename TileData::DType) == 2) {
        if constexpr (std::is_same<typename TileData::DType, bfloat16_t>::value) {
            create_cbuf_matrix_bf16(reinterpret_cast<__cbuf__ bfloat16_t *>(dstPtr), repeatConfig, value);
        } else {
            create_cbuf_matrix(dstPtr, repeatConfig, value);
        }
    } else if constexpr (sizeof(typename TileData::DType) == 1) {
        auto dstCast = reinterpret_cast<__cbuf__ uint32_t *>(dstPtr);
        uint16_t expanded16 = static_cast<uint16_t>(value) | (static_cast<uint16_t>(value) << 8);
        uint32_t expanded32 = static_cast<uint32_t>(expanded16) | (static_cast<uint32_t>(expanded16) << 16);
        create_cbuf_matrix(dstCast, repeatConfig, expanded32);
    }
}

template <typename TileData>
__tf__ PTO_INTERNAL void TExpandsMatTile(typename TileData::TileDType __out__ dst, typename TileData::DType value)
{
    using U = typename TileData::DType;
    __cbuf__ U *dstPtr = (__cbuf__ U *)__cce_get_tile_ptr(dst);
    // block uint is 32B
    constexpr uint64_t totalBytes = TileData::Rows * TileData::Cols * sizeof(U);
    constexpr uint64_t repeat = totalBytes / BLOCK_BYTE_SIZE;
    constexpr uint16_t repeatTimes = static_cast<uint16_t>(repeat);
    PTO_ASSERT(repeatTimes >= 1 && repeatTimes <= EXPANDS_MAX_SUPPORT_REPEAT_TIMES,
               "ERROR: The range of dstRow * dstCol * sizeof(U) / 32 is [1, 32767].");
    constexpr int64_t repeatConfig =
        ((static_cast<int64_t>(0) & 0x7FFF) << 32) |  // [46:32] is the repeat gap between two consecutive repeats
        ((static_cast<int64_t>(1) & 0xFFFF) << 16) |  // [30:16] is the block number of each repeat
        (static_cast<int64_t>(repeatTimes) & 0xFFFF); // [14:0] is the repeat times
    TExpandSInstrMat<TileData>(dstPtr, repeatConfig, value);
}

template <typename TileData>
__tf__ PTO_INTERNAL void TExpandsMat5HD(typename TileData::TileDType __out__ dst, typename TileData::DType value,
                                        int shape0, int shape1, int shape2, int shape3, int shape4)
{
    using U = typename TileData::DType;
    __cbuf__ U *dstPtr = (__cbuf__ U *)__cce_get_tile_ptr(dst);

    uint64_t totalBytes = shape0 * shape1 * shape2 * shape3 * shape4 * sizeof(U);
    uint64_t repeat = totalBytes / BLOCK_BYTE_SIZE;
    uint16_t repeatTimes = static_cast<uint16_t>(repeat);
    PTO_ASSERT(repeatTimes >= 1 && repeatTimes <= EXPANDS_MAX_SUPPORT_REPEAT_TIMES,
               "ERROR: The range of convtile's (shape0 * shape1 * shape2 * shape3) is [1, 32767].");
    int64_t repeatConfig = 0;
    // [46:32] is the repeat gap between two consecutive repeats
    repeatConfig |= (static_cast<int64_t>(0) & 0x7FFF) << 32;
    repeatConfig |= (static_cast<int64_t>(1) & 0xFFFF) << 16;     // [30:16] is the block number of each repeat
    repeatConfig |= (static_cast<int64_t>(repeatTimes) & 0xFFFF); // [14:0] is the repeat times
    TExpandSInstrMat<TileData>(dstPtr, repeatConfig, value);
}

template <typename TileData>
PTO_INTERNAL void TEXPANDS_IMPL(TileData &dst, typename TileData::DType scalar)
{
    static_assert(TileData::Loc == TileType::Vec || TileData::Loc == pto::TileType::Mat,
                  "TileType of tiles must be TileType::Vec or Mat.");

    if constexpr (TileData::Loc == TileType::Vec) {
        static_assert(TileData::isRowMajor, "TEXPANDS: not supported Layout type.");
        static_assert(TileData::ValidCol <= TileData::Cols,
                      "Number of valid columns must not be greater than number of tile columns.");
        static_assert(TileData::ValidRow <= TileData::Rows,
                      "Number of valid rows must not be greater than number of tile rows.");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
        unsigned numRepeatPerLine = dst.GetValidCol() / elementsPerRepeat;
        unsigned numRemainPerLine = dst.GetValidCol() % elementsPerRepeat;
        constexpr unsigned stride = TileData::RowStride;
        unsigned validRow = dst.GetValidRow();
        unsigned validCol = dst.GetValidCol();
        TExpandS<TileData, elementsPerRepeat, blockSizeElem, stride>(dst.data(), scalar, validRow, validCol);
    } else if constexpr (TileData::Loc == TileType::Mat) {
        if constexpr (is_conv_tile_v<TileData>) { // layout is NC1HWC0 or C1HWNC0, dst dim4 is c0Size
            TExpandsMat5HD<TileData>(dst.data(), scalar, dst.GetShape(0), dst.GetShape(1), dst.GetShape(2),
                                     dst.GetShape(3), dst.GetShape(4));
        } else {
            TExpandsMatTile<TileData>(dst.data(), scalar);
        }
    }
}
} // namespace pto

#endif