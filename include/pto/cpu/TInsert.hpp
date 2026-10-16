/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TINSERT_HPP
#define TINSERT_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "common.hpp"

namespace pto {

template <typename DstTileData, typename SrcTileData, QuantMode_t quantMode, bool applyRelu>
PTO_INTERNAL void TInsert_Impl(
    DstTileData& dst, SrcTileData& src, uint32_t idxRow, uint32_t idxCol, const std::vector<uint64_t>& scalars = {})
{
    assert(src.GetValidRow() + idxRow <= DstTileData::Rows && src.GetValidCol() + idxCol <= DstTileData::Cols);

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < src.GetValidCol(); c++) {
        for (size_t r = 0; r < src.GetValidRow(); r++) {
            if constexpr (quantMode != QuantMode_t::NoQuant) {
                dst.SetElement(
                    r + idxRow, c + idxCol,
                    quantize_element<D, S, quantMode, applyRelu>(src.GetElement(r, c), scalars[c]));
            } else {
                S val = src.GetElement(r, c);
                if constexpr (applyRelu) {
                    val = ReLU(val);
                }
                dst.SetElement(r + idxRow, c + idxCol, val);
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData& dst, SrcTileData& src, uint32_t idxRow = 0, uint32_t idxCol = 0)
{
    TInsert_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, false>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TINSERT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    (void)Phase;
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TInsert_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TINSERT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    (void)Phase;
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    constexpr QuantMode_t quantMode = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();

    std::vector<uint64_t> scalars(src.GetValidCol(), preQuantScalar);

    TInsert_Impl<DstTileData, SrcTileData, quantMode, useRelu>(dst, src, indexRow, indexCol, scalars);
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TINSERT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    (void)Phase;
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    constexpr QuantMode_t quantMode = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIndex = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIndex];
    }
    TInsert_Impl<DstTileData, SrcTileData, quantMode, useRelu>(dst, src, indexRow, indexCol, scalars);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    TINSERT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
}

template <TInsertMode Mode, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    using T = typename SrcTileData::DType;
    static_assert(
        std::is_same_v<T, typename DstTileData::DType>, "TINSERT: Source and destination data types must match");
    static_assert(
        DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Vec,
        "TINSERT SPLIT: Source must be Vec and destination must be Mat");
    static_assert(
        !SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor,
        "TINSERT SPLIT: Source must be NZ format");
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> ||
            std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, hifloat8_t> ||
            std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t> || std::is_same_v<T, float8_e8m0_t> ||
            std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>,
        "TINSERT SPLIT: Unsupported data type");
    assert(indexRow + src.GetValidRow() <= DstTileData::Rows && indexCol + src.GetValidCol() <= DstTileData::Cols);
    if constexpr (Mode == TInsertMode::SPLIT2 || Mode == TInsertMode::SPLIT4) {
        constexpr size_t ELEMENTS_PER_STORAGE_UNIT = IsTwinType<T>() ? 2 : 1;
        const size_t alignedRows = (src.GetValidRow() + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        const size_t colBytes = src.GetValidCol() * sizeof(T) / ELEMENTS_PER_STORAGE_UNIT;
        const size_t indexColBytes = indexCol * sizeof(T) / ELEMENTS_PER_STORAGE_UNIT;
        const size_t blockCount = (colBytes + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        size_t srcStrideRows = alignedRows;
        if constexpr (SrcTileData::Compact == CompactMode::Null) {
            srcStrideRows = SrcTileData::Rows;
        } else if constexpr (SrcTileData::Compact == CompactMode::RowPlusOne) {
            ++srcStrideRows;
        }
        const size_t srcStrideBytes = srcStrideRows * BLOCK_BYTE_SIZE;
        constexpr size_t DST_STRIDE_BYTES = DstTileData::Rows * BLOCK_BYTE_SIZE;
        const size_t burstBytes = alignedRows * BLOCK_BYTE_SIZE;
        const size_t dstOffset = (indexColBytes / BLOCK_BYTE_SIZE) * DstTileData::Rows * BLOCK_BYTE_SIZE +
                                 indexRow * BLOCK_BYTE_SIZE + indexColBytes % BLOCK_BYTE_SIZE;
        auto* dstBytes = reinterpret_cast<uint8_t*>(dst.data()) + dstOffset;
        const auto* srcBytes = reinterpret_cast<const uint8_t*>(src.data());
        // Splitting the NPU DMA calls preserves block order and includes the aligned row tail.
        for (size_t block = 0; block < blockCount; ++block) {
            std::memcpy(dstBytes + block * DST_STRIDE_BYTES, srcBytes + block * srcStrideBytes, burstBytes);
        }
    }
}
} // namespace pto
#endif // TINSERT_HPP
