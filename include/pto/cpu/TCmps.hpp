/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMPS_HPP
#define TCMPS_HPP
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace pto {

const int32_t CMP_BITS_PER_INDEX = 32;

template <typename T>
AICORE uint8_t CmpCall(T a, T b, CmpMode cmpMode)
{
    uint8_t res = 0;
    const double diff = static_cast<double>(a) - static_cast<double>(b);
    switch (static_cast<CmpMode>(cmpMode)) {
        case CmpMode::EQ:
            res = (std::fabs(diff) < 1e-9);
            break;
        case CmpMode::NE:
            res = (std::fabs(diff) > 1e-9);
            break;
        case CmpMode::LT:
            res = (a < b);
            break;
        case CmpMode::GT:
            res = (a > b);
            break;
        case CmpMode::GE:
            res = (a >= b);
            break;
        case CmpMode::LE:
            res = (a <= b);
            break;
        default:
            res = (std::fabs(diff) < 1e-9);
            break;
    }
    return res;
}

template <typename TileDataDst, typename TileDataSrc, typename T>
AICORE void TCmps(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src0, T src1, CmpMode mode,
    unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol, unsigned dstStride,
    unsigned srcStride)
{
    size_t H = TileDataSrc::Rows;
    size_t W = TileDataSrc::Cols;
    std::vector<uint8_t> golden(H * W, 0);

    for (size_t i = 0; i < srcValidRow; i++) {
        for (size_t j = 0; j < srcValidCol; j++) {
            T a = src0[GetTileElementOffset<TileDataSrc>(i, j)];
            golden[i * W + j] = CmpCall<T>(a, src1, mode);
        }
    }

    std::vector<uint8_t> out_uint8;
    size_t bits_per_row = W / 8;

    for (size_t h = 0; h < H; ++h) {
        for (size_t i = 0; i < bits_per_row; ++i) {
            uint8_t packed_byte = 0;
            for (size_t bit = 0; bit < 8; ++bit) {
                // Get the bit from the golden array and shift it into position
                uint8_t bit_val = golden[h * W + (i * 8 + bit)];
                packed_byte |= (bit_val << bit);
            }
            out_uint8.push_back(packed_byte);
        }
    }

    std::fill(dst, dst + TileDataDst::Numel, static_cast<typename TileDataDst::DType>(0));
    const size_t dstPackedCols = static_cast<size_t>(dstValidCol);
    if (dstPackedCols == 0) {
        return;
    }
    for (size_t c = 0; c < out_uint8.size() && c < TileDataDst::Numel; ++c) {
        const size_t i = c / dstPackedCols;
        const size_t j = c % dstPackedCols;
        if (i < dstValidRow && j < dstValidCol) {
            dst[GetTileElementOffset<TileDataDst>(i, j)] = static_cast<typename TileDataDst::DType>(out_uint8[c]);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, typename T>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst& dst, TileDataSrc& src0, T src1, CmpMode cmpMode)
{
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();

    unsigned srcValidRow = src0.GetValidRow();
    unsigned srcValidCol = src0.GetValidCol();

    unsigned dstStride = TileDataDst::RowStride;
    unsigned srcStride = TileDataSrc::RowStride;

    TCmps<TileDataDst, TileDataSrc, T>(
        dst.data(), src0.data(), src1, cmpMode, srcValidRow, srcValidCol, dstValidRow, dstValidCol, dstStride,
        srcStride);
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename = std::void_t<typename TileDataSrc1::DType>>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode)
{
    static_assert(
        std::is_same_v<typename TileDataSrc0::DType, typename TileDataSrc1::DType>,
        "Fix: TCMPS src0 and src1 must have the same data type.");
    using T = typename TileDataSrc0::DType;

    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();

    unsigned srcValidRow = src0.GetValidRow();
    unsigned srcValidCol = src0.GetValidCol();

    unsigned dstStride = TileDataDst::RowStride;
    unsigned srcStride = TileDataSrc0::RowStride;
    unsigned src1Stride = TileDataSrc1::RowStride;

    size_t H = TileDataSrc0::Rows;
    size_t W = TileDataSrc0::Cols;
    std::vector<uint8_t> golden(H * W, 0);

    for (size_t i = 0; i < srcValidRow; i++) {
        for (size_t j = 0; j < srcValidCol; j++) {
            T a = src0.data()[GetTileElementOffset<TileDataSrc0>(i, j)];
            T b = src1.data()[GetTileElementOffset<TileDataSrc1>(i, j)];
            golden[i * W + j] = CmpCall<T>(a, b, cmpMode);
        }
    }

    std::vector<uint8_t> out_uint8;
    size_t bits_per_row = W / 8;

    for (size_t h = 0; h < H; ++h) {
        for (size_t i = 0; i < bits_per_row; ++i) {
            uint8_t packed_byte = 0;
            for (size_t bit = 0; bit < 8; ++bit) {
                uint8_t bit_val = golden[h * W + (i * 8 + bit)];
                packed_byte |= (bit_val << bit);
            }
            out_uint8.push_back(packed_byte);
        }
    }

    std::fill(dst.data(), dst.data() + TileDataDst::Numel, static_cast<typename TileDataDst::DType>(0));
    const size_t dstPackedCols = static_cast<size_t>(dstValidCol);
    if (dstPackedCols == 0) {
        return;
    }
    for (size_t c = 0; c < out_uint8.size() && c < TileDataDst::Numel; ++c) {
        const size_t i = c / dstPackedCols;
        const size_t j = c % dstPackedCols;
        if (i < dstValidRow && j < dstValidCol) {
            dst.data()[GetTileElementOffset<TileDataDst>(i, j)] =
                static_cast<typename TileDataDst::DType>(out_uint8[c]);
        }
    }
}
} // namespace pto
#endif
