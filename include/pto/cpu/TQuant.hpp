/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TQUANT_CPU_HPP
#define TQUANT_CPU_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
#if !defined(PTO_NPU_ARCH_A2A3) && !defined(PTO_NPU_ARCH_A5) && !defined(PTO_NPU_ARCH_KIRIN9030)
enum class QuantType
{
    MXFP8,
    INT8_SYM,
    INT8_ASYM
};

enum class VecStoreMode
{
    ND,
    NZ
};
#endif

namespace cpu_quant {
inline float BitsToFloat(uint32_t bits)
{
    return std::bit_cast<float>(bits);
}

inline uint32_t FloatToBits(float value)
{
    return std::bit_cast<uint32_t>(value);
}

template <typename TileDataPara>
inline typename TileDataPara::DType GetParamValue(const TileDataPara &tile, int row, int col)
{
    const int paramRow = std::min(row, tile.GetValidRow() - 1);
    const int paramCol = std::min(col, tile.GetValidCol() - 1);
    return tile.data()[GetTileElementOffset<TileDataPara>(paramRow, paramCol)];
}

inline int8_t ClampInt8(float value)
{
    const float rounded = std::nearbyint(value);
    return static_cast<int8_t>(std::clamp(rounded, -128.0f, 127.0f));
}

inline uint8_t ClampUint8(float value)
{
    const float rounded = std::nearbyint(value);
    return static_cast<uint8_t>(std::clamp(rounded, 0.0f, 255.0f));
}

inline float DecodeE4M3Fn(uint8_t code)
{
    const int sign = (code & 0x80u) ? -1 : 1;
    const int exp = (code >> 3) & 0x0Fu;
    const int mant = code & 0x07u;
    if (exp == 0) {
        if (mant == 0) {
            return sign < 0 ? -0.0f : 0.0f;
        }
        return static_cast<float>(sign) * std::ldexp(static_cast<float>(mant), -9);
    }
    if (exp == 0x0F && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float significand = 1.0f + static_cast<float>(mant) / 8.0f;
    return static_cast<float>(sign) * std::ldexp(significand, exp - 7);
}

inline uint8_t EncodeE4M3Fn(float value)
{
    if (std::isnan(value)) {
        return 0x7Fu;
    }
    const float clipped = std::clamp(value, -448.0f, 448.0f);
    uint8_t bestCode = 0;
    float bestDistance = std::numeric_limits<float>::infinity();
    bool bestEven = true;
    for (int code = 0; code < 256; ++code) {
        if ((code & 0x7F) == 0x7F) {
            continue;
        }
        const float candidate = DecodeE4M3Fn(static_cast<uint8_t>(code));
        const float distance = std::fabs(candidate - clipped);
        const bool isEven = (code & 1) == 0;
        if (distance < bestDistance || (distance == bestDistance && isEven && !bestEven) ||
            (distance == bestDistance && isEven == bestEven && static_cast<uint8_t>(code) < bestCode)) {
            bestDistance = distance;
            bestCode = static_cast<uint8_t>(code);
            bestEven = isEven;
        }
    }
    return bestCode;
}

inline uint8_t ComputeSharedExponent(float maxAbsValue)
{
    const uint32_t bits = FloatToBits(maxAbsValue);
    const uint32_t exponent = (bits & 0x7F800000u) >> 23;
    if (exponent == 0xFFu) {
        return 0xFFu;
    }
    return static_cast<uint8_t>(exponent - 8u);
}

inline float ComputeScalingFromExponent(uint8_t e8m0)
{
    if (e8m0 == 0xFFu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const uint32_t scaleExp = 254u - static_cast<uint32_t>(e8m0);
    float scaling = BitsToFloat(scaleExp << 23);
    if (scaling == 0.0f) {
        scaling = std::ldexp(1.0f, -127);
    }
    return scaling;
}

inline std::vector<uint8_t> ReorderExponentZZ(const std::vector<uint8_t> &exp, int rows, int groupCols)
{
    PTO_CPU_ASSERT(
        rows % 16 == 0 && groupCols % 2 == 0,
        "Fix: MXFP8 NZ exponent reorder currently requires rows multiple of 16 and group cols multiple of 2.");
    const int rowBlocks = rows / 16;
    const int groupBlocks = groupCols / 2;
    std::vector<uint8_t> reordered;
    reordered.reserve(exp.size());
    for (int rb = 0; rb < rowBlocks; ++rb) {
        for (int gb = 0; gb < groupBlocks; ++gb) {
            for (int innerRow = 0; innerRow < 16; ++innerRow) {
                for (int innerGroup = 0; innerGroup < 2; ++innerGroup) {
                    const int row = rb * 16 + innerRow;
                    const int group = gb * 2 + innerGroup;
                    reordered.push_back(exp[row * groupCols + group]);
                }
            }
        }
    }
    return reordered;
}
} // namespace cpu_quant

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr)
{
    using SrcT = typename TileDataSrc::DType;
    static_assert(std::is_same_v<SrcT, float>, "Fix: Input has to be float 32");

    for (int row = 0; row < src.GetValidRow(); ++row) {
        for (int col = 0; col < src.GetValidCol(); ++col) {
            const float srcValue = src.data()[GetTileElementOffset<TileDataSrc>(row, col)];
            const float invScale = static_cast<float>(cpu_quant::GetParamValue(scale, row, 0));
            if constexpr (quant_type == QuantType::INT8_SYM) {
                static_assert(std::is_same_v<typename TileDataOut::DType, int8_t>,
                              "Fix: Quant INT8 sym: Out data type has to be int8");
                dst.data()[GetTileElementOffset<TileDataOut>(row, col)] = cpu_quant::ClampInt8(srcValue * invScale);
            } else {
                static_assert(std::is_same_v<typename TileDataOut::DType, uint8_t>,
                              "Fix: Quant INT8 asym: Out data type has to be uint8");
                PTO_CPU_ASSERT(offset != nullptr, "Fix: Quant INT8 asym requires offset.");
                const float zeroPoint = static_cast<float>(cpu_quant::GetParamValue(*offset, row, 0));
                dst.data()[GetTileElementOffset<TileDataOut>(row, col)] =
                    cpu_quant::ClampUint8(srcValue * invScale + zeroPoint);
            }
        }
    }
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataScaling *scaling)
{
    static_assert(quant_type == QuantType::MXFP8, "Fix: MX overload is reserved for MXFP8.");
    using SrcT = typename TileDataSrc::DType;
    static_assert(std::is_same_v<SrcT, float>, "Fix: Input has to be float 32");
    static_assert(std::is_same_v<typename TileDataOut::DType, int8_t>, "Fix: MXFP8 output must be int8 bytes.");
    static_assert(std::is_same_v<typename TileDataExp::DType, uint8_t>, "Fix: MXFP8 exponent must be uint8 bytes.");

    PTO_CPU_ASSERT(exp != nullptr && max != nullptr && scaling != nullptr,
                   "Fix: MXFP8 requires exp/max/scaling tiles.");

    const int rows = src.GetValidRow();
    const int cols = src.GetValidCol();
    PTO_CPU_ASSERT(cols % 32 == 0, "Fix: MXFP8 CPU sim currently requires valid cols to be a multiple of 32.");
    const int groupCols = cols / 32;

    // Flatten exp, max, scaling to 1D for internal processing
    constexpr int expNumel = TileDataExp::Rows * TileDataExp::Cols;
    using FlatExpTile = Tile<TileType::Vec, typename TileDataExp::DType, 1, expNumel, BLayout::RowMajor, -1, -1>;
    FlatExpTile flatExp(1, expNumel);
    TRESHAPE_IMPL(flatExp, *exp);

    constexpr int maxNumel = TileDataMax::Rows * TileDataMax::Cols;
    using FlatMaxTile = Tile<TileType::Vec, typename TileDataMax::DType, 1, maxNumel, BLayout::RowMajor, -1, -1>;
    FlatMaxTile flatMax(1, maxNumel);
    TRESHAPE_IMPL(flatMax, *max);

    constexpr int scalingNumel = TileDataScaling::Rows * TileDataScaling::Cols;
    using FlatScalingTile =
        Tile<TileType::Vec, typename TileDataScaling::DType, 1, scalingNumel, BLayout::RowMajor, -1, -1>;
    FlatScalingTile flatScaling(1, scalingNumel);
    TRESHAPE_IMPL(flatScaling, *scaling);

    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groupCols; ++group) {
            float maxAbsValue = 0.0f;
            for (int inner = 0; inner < 32; ++inner) {
                const float value = src.data()[GetTileElementOffset<TileDataSrc>(row, group * 32 + inner)];
                maxAbsValue = std::max(maxAbsValue, std::fabs(value));
            }
            const uint8_t e8m0 = cpu_quant::ComputeSharedExponent(maxAbsValue);
            const float groupScaling = cpu_quant::ComputeScalingFromExponent(e8m0);
            const int flatGroupIdx = row * groupCols + group;
            flatMax.data()[flatGroupIdx] = maxAbsValue;
            flatExp.data()[flatGroupIdx] = e8m0;
            for (int inner = 0; inner < 32; ++inner) {
                const int col = group * 32 + inner;
                flatScaling.data()[row * cols + col] = groupScaling;
                const float value = src.data()[GetTileElementOffset<TileDataSrc>(row, col)];
                const uint8_t encoded = cpu_quant::EncodeE4M3Fn(value * groupScaling);
                dst.data()[GetTileElementOffset<TileDataOut>(row, col)] = static_cast<int8_t>(encoded);
            }
        }
    }

    // Reshape back so caller can inspect results
    TRESHAPE_IMPL(*exp, flatExp);
    TRESHAPE_IMPL(*max, flatMax);
    TRESHAPE_IMPL(*scaling, flatScaling);
}

template <QuantType quant_type, VecStoreMode store_mode, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataScaling *scaling, TileDataExp *exp_zz)
{
    static_assert(quant_type == QuantType::MXFP8, "Fix: MX overload is reserved for MXFP8.");
    static_assert(store_mode == VecStoreMode::NZ, "Fix: This overload is reserved for MXFP8 NZ mode.");

    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(dst, src, exp, max,
                                                                                                 scaling);

    PTO_CPU_ASSERT(exp_zz != nullptr, "Fix: MXFP8 NZ mode requires reordered exponents.");
    const int rows = src.GetValidRow();
    const int groupCols = src.GetValidCol() / 32;
    const int totalGroups = rows * groupCols;
    std::vector<uint8_t> expValues(totalGroups);
    for (int i = 0; i < totalGroups; ++i) {
        expValues[i] = exp->data()[GetTileElementOffset<TileDataExp>(0, i)];
    }
    const auto reordered = cpu_quant::ReorderExponentZZ(expValues, rows, groupCols);
    for (int i = 0; i < totalGroups; ++i) {
        exp_zz->data()[GetTileElementOffset<TileDataExp>(0, i)] = reordered[i];
    }
}
} // namespace pto

#endif