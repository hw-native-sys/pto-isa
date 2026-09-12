/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWSUM_HPP
#define TROWSUM_HPP
#include <algorithm>
#include <array>
#include <bit>

#include "pto/cpu/NPUMemoryModel.hpp"
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
namespace cpu {
template <typename TileSrc>
PTO_INTERNAL typename TileSrc::DType rowSumTree(typename TileSrc::TileDType src, std::size_t row, uint32_t cols)
{
    using T = typename TileSrc::DType;
    constexpr std::size_t vectorBytes = 256;
    constexpr std::size_t elementsPerVector = vectorBytes / sizeof(T);
    T total = 0;
    for (std::size_t base = 0; base < cols; base += elementsPerVector) {
        std::array<T, elementsPerVector> lanes{};
        const std::size_t groupCols = std::min<std::size_t>(elementsPerVector, cols - base);
        for (std::size_t col = 0; col < groupCols; ++col) {
            if constexpr (TileSrc::SFractal == SLayout::NoneBox && TileSrc::isRowMajor) {
                lanes[col] = src[row * TileSrc::Cols + base + col];
            } else {
                lanes[col] = src[GetTileElementOffset<TileSrc>(row, base + col)];
            }
        }
        for (std::size_t width = elementsPerVector; width > 1; width /= 2) {
            for (std::size_t col = 0; col < width / 2; ++col) {
                lanes[col] = static_cast<T>(lanes[col * 2] + lanes[col * 2 + 1]);
            }
        }
        total = static_cast<T>(total + lanes[0]);
    }
    return total;
}

template <typename TileSrc>
PTO_INTERNAL typename TileSrc::DType rowSumModular(typename TileSrc::TileDType src, std::size_t row, uint32_t cols)
{
    using T = typename TileSrc::DType;
    using Acc = std::conditional_t<std::is_same_v<T, int16_t>, uint32_t, std::make_unsigned_t<T>>;
    Acc sum = 0;
    // Modular addition preserves the hardware result without signed-overflow UB.
    for (std::size_t col = 0; col < cols; ++col) {
        sum += static_cast<Acc>(src[GetTileElementOffset<TileSrc>(row, col)]);
    }
    return std::bit_cast<T>(static_cast<std::make_unsigned_t<T>>(sum));
}
} // namespace cpu

template <typename TileDst, typename TileSrc>
void TRowSum(typename TileDst::TileDType dst, typename TileSrc::TileDType src, uint32_t M, uint32_t N)
{
    auto& memory = NPUMemoryModel::Instance();
    memory.EnsureInitialized();
    const bool useA5Reduction = memory.GetArch() == NPUArch::A5;
    // Workers have separate memory models; use the calling thread's architecture.
    cpu::parallel_for_1d(0, M, static_cast<std::size_t>(M) * N, [&, useA5Reduction](std::size_t i) {
        TypeSum<TileDst> sum = 0;
        if (useA5Reduction) {
            using T = typename TileSrc::DType;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
                sum = cpu::rowSumTree<TileSrc>(src, i, N);
            } else if constexpr (std::is_integral_v<T>) {
                sum = cpu::rowSumModular<TileSrc>(src, i, N);
            }
        } else if constexpr (TileSrc::SFractal == SLayout::NoneBox && TileSrc::isRowMajor) {
            const std::size_t base = i * TileSrc::Cols;
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t j = 0; j < N; ++j) {
                sum += src[base + j];
            }
        } else {
            for (std::size_t j = 0; j < N; ++j) {
                sum += src[GetTileElementOffset<TileSrc>(i, j)];
            }
        }
        if constexpr (TileDst::SFractal == SLayout::NoneBox && TileDst::isRowMajor) {
            dst[i * TileDst::Cols] = static_cast<typename TileDst::DType>(sum);
        } else {
            dst[GetTileElementOffset<TileDst>(i, 0)] = static_cast<typename TileDst::DType>(sum);
        }
    });
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void CheckRSValid(NPUArch arch, uint32_t srcRows, uint32_t srcCols, uint32_t dstRows)
{
    using SrcType = typename TileSrc::DType;
    using DstType = typename TileDst::DType;
    constexpr bool legacySupported =
        (std::is_same_v<SrcType, half> && std::is_same_v<DstType, half>) ||             // f162f16
        (std::is_same_v<SrcType, bfloat16_t> && std::is_same_v<DstType, bfloat16_t>) || // bf162bf16
        (std::is_same_v<SrcType, half> && std::is_same_v<DstType, float>) ||            // f162f32
        (std::is_same_v<SrcType, bfloat16_t> && std::is_same_v<DstType, float>) ||      // bf162f32
        (std::is_same_v<SrcType, float> && std::is_same_v<DstType, float>) ||           // f322f32
        (std::is_same_v<SrcType, int16_t> && std::is_same_v<DstType, int16_t>) ||       // i162i16
        (std::is_same_v<SrcType, int16_t> && std::is_same_v<DstType, int32_t>) ||       // i162i32
        (std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, int32_t>);         // i322i32
    constexpr bool a5Supported =
        std::is_same_v<SrcType, DstType> &&
        (std::is_same_v<SrcType, half> || std::is_same_v<SrcType, float> || std::is_same_v<SrcType, int16_t> ||
         std::is_same_v<SrcType, int32_t> || std::is_same_v<SrcType, int64_t> || std::is_same_v<SrcType, uint64_t>);
    static_assert(legacySupported || a5Supported, "Not supported data type");
    if (arch == NPUArch::A5) {
        PTO_CPU_ASSERT(a5Supported, "A5 TROWSUM requires matching half, float, int16, int32, int64 or uint64 types.");
        PTO_CPU_ASSERT(
            (TileSrc::Loc == TileType::Vec && TileDst::Loc == TileType::Vec),
            "Row reduction only works on vector tiles.");
        PTO_CPU_ASSERT((TileSrc::isRowMajor && !TileSrc::isBoxedLayout), "Input tile must use ND row-major layout.");
        PTO_CPU_ASSERT(
            (!TileDst::isBoxedLayout && (TileDst::isRowMajor || TileDst::Cols == 1)),
            "Output tile must use ND layout or DN layout with one column.");
        PTO_CPU_ASSERT(srcRows != 0 && srcCols != 0, "Row reduction input must be non-empty.");
        PTO_CPU_ASSERT(srcRows == dstRows, "Row reduction preserves row count.");
    } else {
        PTO_CPU_ASSERT(legacySupported, "Not supported data type for A2A3 CPU simulation.");
        PTO_CPU_ASSERT((TileSrc::Rows == TileDst::Rows), "Inconsistent number of m, n");
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWSUM_IMPL(TileDataOut& dstTile, TileDataIn& srcTile, TileDataTmp& tmp)
{
    auto& memory = NPUMemoryModel::Instance();
    memory.EnsureInitialized();
    CheckRSValid<TileDataOut, TileDataIn>(
        memory.GetArch(), srcTile.GetValidRow(), srcTile.GetValidCol(), dstTile.GetValidRow());

    uint32_t m = srcTile.GetValidRow();
    uint32_t n = srcTile.GetValidCol();

    TRowSum<TileDataOut, TileDataIn>(dstTile.data(), srcTile.data(), m, n);
}
} // namespace pto
#endif
