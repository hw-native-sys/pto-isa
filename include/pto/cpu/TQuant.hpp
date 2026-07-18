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
#include <pto/common/type.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
namespace cpu_quant {

constexpr uint16_t bf16Bits = 16;

inline float BitsToFloat(uint32_t bits) { return std::bit_cast<float>(bits); }

inline uint32_t FloatToBits(float value) { return std::bit_cast<uint32_t>(value); }

struct NvMxFp8E4M3Spec {
    static constexpr float descaleMultiplier = 1.0f / 448.0f;
    static constexpr uint32_t b16SpecialScaleBits = 0x7F81u;
    static constexpr uint32_t f32SpecialScaleBits = 0x7FC00000u;
};

struct NvMxFp4E2M1Spec {
    static constexpr float descaleMultiplier = 1.0f / 6.0f;
    static constexpr uint32_t b16SpecialScaleBits = 0x7FC0u;
    static constexpr uint32_t f32SpecialScaleBits = 0x7FC00000u;
    static constexpr uint8_t PS_MAX = 0x07u;
    static constexpr uint8_t NG_MIN = 0x0Fu;
};

inline uint16_t FloatToBf16BitsTrunc(float value) { return static_cast<uint16_t>(FloatToBits(value) >> bf16Bits); }

inline uint16_t FloatToBf16BitsRound(float value)
{
    const uint32_t bits = FloatToBits(value);
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

inline float Bf16BitsToFloat(uint16_t bits)
{
    return BitsToFloat(static_cast<uint32_t>(bits) << 16);
}

inline uint16_t AbsBf16BitsFromFloat(float value)
{
    return static_cast<uint16_t>(FloatToBf16BitsTrunc(value) & 0x7FFFu);
}

template <typename SrcT>
inline float ApplyE2M1ScaleForSource(SrcT value, float scaling)
{
    const float scaled = static_cast<float>(value) * scaling;
    if constexpr (std::is_same_v<SrcT, bfloat16_t> && !std::is_same_v<bfloat16_t, half>) {
        return Bf16BitsToFloat(FloatToBf16BitsRound(scaled));
    }
    return scaled;
}

template <typename TileDataPara>
inline typename TileDataPara::DType GetParamValue(const TileDataPara& tile, int row, int col)
{
    const int paramRow = std::min<int>(row, tile.GetValidRow() - 1);
    const int paramCol = std::min<int>(col, tile.GetValidCol() - 1);
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
        constexpr int scaleExp = 9;
        return static_cast<float>(sign) * std::ldexp(static_cast<float>(mant), -scaleExp);
    }
    if (exp == 0x0F && mant == 0x07) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float significand = 1.0f + static_cast<float>(mant) / 8.0f;
    constexpr int scaleExp = 7;
    return static_cast<float>(sign) * std::ldexp(significand, exp - scaleExp);
}

template <QuantScaleAlg scale_alg>
inline uint8_t EncodeE4M3Fn(float value)
{
    if (std::isnan(value)) {
        return 0x7Fu;
    }
    if (scale_alg == QuantScaleAlg::NV && std::fabs(value) < 0.0009765625f) {
        // FIX: Safe Sign-Retention Check mimicking Python's ml_dtypes underflow tracking
        if (std::signbit(value)) {
            return 0x80u; // Match negative zero output requirement
        } else {
            return 0x00u; // Clean positive zero
        }
    }

    const float clipped = std::clamp(value, -448.0f, 448.0f);
    uint8_t bestCode = 0;
    float bestDistance = std::numeric_limits<float>::infinity();
    bool bestEven = true;
    constexpr int codeMax = 256;
    for (int code = 0; code < codeMax; ++code) {
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

inline uint8_t EncodeE2M1Magic(float value)
{
    if (std::isnan(value)) {
        return 0x7u;
    }
    const uint32_t valueBits = FloatToBits(value);
    const uint8_t sign = static_cast<uint8_t>((valueBits >> 28) & 0x8u);
    const float absValue = std::fabs(value);
    if (std::isinf(absValue)) {
        return static_cast<uint8_t>(sign | 0x7u);
    }

    const uint32_t absBits = FloatToBits(absValue);
    uint32_t biasedExp = (absBits & 0x7F800000u) >> 23;
    biasedExp = std::clamp<uint32_t>(biasedExp, 127u, 129u);

    const uint32_t magicBits = (biasedExp + 22u) << 23;
    const uint32_t q = FloatToBits(absValue + BitsToFloat(magicBits)) - magicBits;
    const uint32_t baseCode = (biasedExp - 127u) << 1;
    const uint32_t magCode = std::min<uint32_t>(q + baseCode, 7u);
    return static_cast<uint8_t>(sign | magCode);
}

inline uint8_t ComputeSharedExponent(float maxAbsValue)
{
    const uint32_t bits = FloatToBits(maxAbsValue);
    const uint32_t exponent = (bits & 0x7F800000u) >> 23;
    const uint32_t mantissa = bits & 0x007FFFFFu;
    if (exponent == 0xFFu && mantissa != 0u) {
        return 0xFFu;
    }
    if (exponent <= 8u) {
        return 0u;
    }
    return static_cast<uint8_t>(exponent - 8u);
}

inline uint8_t ComputeE2M1SharedExponent(float maxAbsValue)
{
    const uint32_t bits = FloatToBits(maxAbsValue);
    uint32_t exponent = (bits & 0x7F800000u) >> 23;
    if (exponent == 0xFFu) {
        return 0xFFu;
    }
    if (exponent <= 2u) {
        return 0u;
    }
    return static_cast<uint8_t>(exponent - 2u);
}

inline float ComputeMxScalingFromExponent(uint8_t e8m0)
{
    if (e8m0 == 0xFFu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const uint32_t scaleExp = 254u - static_cast<uint32_t>(e8m0);
    float scaling = BitsToFloat(scaleExp << 23);
    if (scaling == 0.0f) {
        scaling = std::ldexp(1.0f, SCALE_MIN_EXP);
    }
    return scaling;
}

inline float ComputeNvScalingFromExponent(uint8_t e8m0)
{
    if (e8m0 == 0xFFu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (e8m0 == 0xFEu) {
        return std::ldexp(1.0f, SCALE_MIN_EXP);
    }
    return ComputeMxScalingFromExponent(e8m0);
}

inline float ComputeE2M1ScalingFromExponent(uint8_t e8m0) { return ComputeMxScalingFromExponent(e8m0); }

inline float ComputeScalingFromExponent(uint8_t e8m0) { return ComputeMxScalingFromExponent(e8m0); }

inline std::vector<uint8_t> ReorderExponentZZ(const std::vector<uint8_t>& exp, int rows, int groupCols)
{
    constexpr int groupBlockSize = 2;
    constexpr int rowBlockSize = 16;
    PTO_CPU_ASSERT(
        rows % rowBlockSize == 0 && groupCols % groupBlockSize == 0,
        "Fix: MXFP8 NZ exponent reorder currently requires rows "
        "multiple of 16 and group cols multiple of 2.");
    const int rowBlocks = rows / rowBlockSize;
    const int groupBlocks = groupCols / groupBlockSize;
    std::vector<uint8_t> reordered;
    reordered.reserve(exp.size());
    for (int rb = 0; rb < rowBlocks; ++rb) {
        for (int gb = 0; gb < groupBlocks; ++gb) {
            for (int innerRow = 0; innerRow < rowBlockSize; ++innerRow) {
                for (int innerGroup = 0; innerGroup < groupBlockSize; ++innerGroup) {
                    const int row = rb * rowBlockSize + innerRow;
                    const int group = gb * groupBlockSize + innerGroup;
                    reordered.push_back(exp[row * groupCols + group]);
                }
            }
        }
    }
    return reordered;
}

template <int grp_axis, QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataSrc>
inline float ComputeMxGroupMax(TileDataSrc& src, int axis, int group)
{
    float maxAbsValue = 0.0f;
    uint16_t maxAbsBf16Bits = 0;
    constexpr int groupSize = 32;
    for (int inner = 0; inner < groupSize; ++inner) {
        int row = grp_axis == 1 ? axis : (group * groupSize + inner);
        int col = grp_axis == 1 ? (group * groupSize + inner) : axis;
        const float value = src.data()[GetTileElementOffset<TileDataSrc>(row, col)];
        if constexpr (
            quant_type == QuantType::MXFP8 || (quant_type == QuantType::MXFP4_E2M1 && scale_alg == QuantScaleAlg::NV)) {
            if constexpr (quant_type == QuantType::MXFP4_E2M1) {
                if (std::isnan(value)) {
                    return value;
                }
            }
            maxAbsValue = std::max(maxAbsValue, std::fabs(value));
        } else {
            maxAbsBf16Bits = std::max(maxAbsBf16Bits, AbsBf16BitsFromFloat(value));
        }
    }
    if constexpr (quant_type == QuantType::MXFP4_E2M1) {
        maxAbsValue = Bf16BitsToFloat(maxAbsBf16Bits);
    }
    return maxAbsValue;
}

template <QuantType quant_type>
inline uint8_t ComputeMxSharedExponent(float maxAbsValue)
{
    if constexpr (quant_type == QuantType::MXFP8) {
        return ComputeSharedExponent(maxAbsValue);
    }
    return ComputeE2M1SharedExponent(maxAbsValue);
}

template <QuantType quant_type>
inline float ComputeMxGroupScaling(uint8_t e8m0)
{
    if constexpr (quant_type == QuantType::MXFP8) {
        return ComputeScalingFromExponent(e8m0);
    }
    return ComputeE2M1ScalingFromExponent(e8m0);
}

template <QuantType quant_type, QuantScaleAlg scale_alg>
inline uint8_t ComputeMxSharedExponent(float maxAbsValue)
{
    if constexpr (quant_type == QuantType::MXFP8 && scale_alg == QuantScaleAlg::NV) {
        return ComputeNvSharedExponent<NvMxFp8E4M3Spec>(maxAbsValue);
    } else if constexpr (quant_type == QuantType::MXFP4_E2M1 && scale_alg == QuantScaleAlg::NV) {
        return ComputeNvSharedExponent<NvMxFp4E2M1Spec>(maxAbsValue);
    }
    return ComputeMxSharedExponent<quant_type>(maxAbsValue);
}

template <QuantType quant_type, QuantScaleAlg scale_alg>
inline float ComputeMxGroupScaling(float maxAbsValue, uint8_t e8m0)
{
    if (e8m0 == 0xFFu) {
        return maxAbsValue;
    }
    if constexpr (scale_alg == QuantScaleAlg::NV) {
        return ComputeNvScalingFromExponent(e8m0);
    }
    return ComputeMxGroupScaling<quant_type>(e8m0);
}

template <
    QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc, typename FlatScalingTile>
inline void StoreMxEncodedValue(
    TileDataOut& dst, TileDataSrc& src, FlatScalingTile& flatScaling, int row, int col, int flatGroupIdx,
    float groupScaling)
{
    using SrcT = typename TileDataSrc::DType;
    using DstT = typename TileDataOut::DType;
    flatScaling.data()[flatGroupIdx] = static_cast<typename FlatScalingTile::DType>(groupScaling);
    if constexpr (quant_type == QuantType::MXFP8) {
        const float value = static_cast<float>(src.data()[GetTileElementOffset<TileDataSrc>(row, col)]);
        const uint8_t encoded = EncodeE4M3Fn<scale_alg>(value * groupScaling);
        dst.SetElement(row, col, static_cast<DstT>(encoded));
    } else {
        uint8_t finalEncoded = NvMxFp4E2M1Spec::PS_MAX;
        if (std::isinf(groupScaling)) {
            finalEncoded = (groupScaling > 0) ? NvMxFp4E2M1Spec::PS_MAX : NvMxFp4E2M1Spec::NG_MIN;
        } else if (!std::isnan(groupScaling)) {
            const float value = static_cast<float>(src.data()[GetTileElementOffset<TileDataSrc>(row, col)]);
            finalEncoded = EncodeE2M1Magic(ApplyE2M1ScaleForSource<SrcT>(static_cast<SrcT>(value), groupScaling));
        }
        dst.SetElement(row, col, DstT::FromRaw(finalEncoded));
    }
}

template <
    int grp_axis, QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
    typename FlatScalingTile>
inline void QuantizeMxGroup(
    TileDataOut& dst, TileDataSrc& src, FlatScalingTile& flatScaling, int axis, int group, int flatGroupIdx,
    float groupScaling)
{
    constexpr int groupSize = 32;
    for (int inner = 0; inner < groupSize; ++inner) {
        int row = grp_axis == 1 ? axis : (group * groupSize + inner);
        int col = grp_axis == 1 ? (group * groupSize + inner) : axis;
        StoreMxEncodedValue<quant_type, scale_alg>(dst, src, flatScaling, row, col, flatGroupIdx, groupScaling);
    }
}

template <typename TileData>
using FlatMxTile =
    Tile<TileType::Vec, typename TileData::DType, 1, TileData::Rows * TileData::Cols, BLayout::RowMajor, -1, -1>;

template <
    int grp_axis, QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
    typename TileDataExp>
inline void CheckMxQuantTypes()
{
    static_assert(
        quant_type == QuantType::MXFP8 || quant_type == QuantType::MXFP4_E2M1,
        "Fix: MX overload is reserved for MXFP8/MXFP4_E2M1.");
    static_assert(
        scale_alg == QuantScaleAlg::OCP || scale_alg == QuantScaleAlg::NV,
        "Fix: MX scale algorithm must be OCP or NV.");

    static_assert(
        TileDataSrc::isRowMajor && TileDataOut::isRowMajor && TileDataSrc::Rows == TileDataOut::Rows &&
            TileDataSrc::Cols == TileDataOut::Cols,
        "Src and Out tiles should have the same BFractal layout and static shape!");
    static_assert(
        (grp_axis == 1 && (TileDataSrc::Cols % 32 == 0)) || (grp_axis == 0 && (TileDataSrc::Rows % 32 == 0)),
        "Src Rows/Cols should be multiple of 32 for ND/DN quant mode!");

    using SrcT = typename TileDataSrc::DType;
    if constexpr (quant_type == QuantType::MXFP8) {
        static_assert(
            std::is_same_v<SrcT, float> || std::is_same_v<SrcT, half> || std::is_same_v<SrcT, aclFloat16> ||
                std::is_same_v<SrcT, bfloat16_t>,
            "Fix: MXFP8 CPU sim supports float/float16/bfloat16 source.");
        static_assert(std::is_same_v<typename TileDataOut::DType, int8_t>, "Fix: MXFP8 output must be int8 bytes.");
    } else {
        static_assert(
            std::is_same_v<SrcT, float> || std::is_same_v<SrcT, half> || std::is_same_v<SrcT, aclFloat16> ||
                std::is_same_v<SrcT, bfloat16_t>,
            "Fix: MXFP4_E2M1 CPU sim supports float/float16/bfloat16 source.");
        static_assert(
            std::is_same_v<typename TileDataOut::DType, float4_e2m1x2_t>,
            "Fix: MXFP4_E2M1 output must be float4_e2m1x2_t.");
    }
    static_assert(std::is_same_v<typename TileDataExp::DType, uint8_t>, "Fix: MXFP8 exponent must be uint8 bytes.");
}

template <typename TileDataExp, typename TileDataMax, typename TileDataScaling>
inline void CheckMxQuantInputs(TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    PTO_CPU_ASSERT(exp != nullptr && max != nullptr && scaling != nullptr, "Fix: MX quant requires tiles.");
}

template <typename FlatTileData, typename TileData>
inline void FlattenMxTile(FlatTileData& flatTile, TileData& tile)
{
    TRESHAPE_IMPL(flatTile, tile);
}

template <typename TileData, typename FlatTileData>
inline void RestoreMxTile(TileData& tile, FlatTileData& flatTile)
{
    TRESHAPE_IMPL(tile, flatTile);
}

template <QuantType quant_type, typename TileDataOut>
inline void InitMxOutput(TileDataOut& dst)
{
    if constexpr (quant_type == QuantType::MXFP4_E2M1) {
        std::fill(
            reinterpret_cast<uint8_t*>(dst.data()),
            reinterpret_cast<uint8_t*>(dst.data()) + TileDataOut::Rows * TileDataOut::Cols, 0);
    }
}

template <
    int grp_axis, QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
    typename TileDataExp, typename FlatMaxTile, typename FlatScalingTile>
inline void QuantizeMxTile(
    TileDataOut& dst, TileDataSrc& src, TileDataExp& exp, FlatMaxTile& flatMax, FlatScalingTile& flatScaling)
{
    const int rows = (grp_axis == 1) ? src.GetValidRow() : TileDataSrc::Rows;
    const int cols = (grp_axis == 1) ? TileDataSrc::Cols : src.GetValidCol();

    const int numGroupsAlongAxis = (grp_axis == 1) ? (cols / 32) : (rows / 32);
    const int numElementsAlongAxis = (grp_axis == 1) ? rows : cols;

    for (int i = 0; i < numElementsAlongAxis; ++i) {
        for (int group = 0; group < numGroupsAlongAxis; ++group) {
            const int row = (grp_axis == 1) ? i : group; // This needs careful mapping logic
            const int col = (grp_axis == 1) ? group : i;

            const int flatGroupIdx =
                (grp_axis == 1) ? (i * numGroupsAlongAxis + group) : (group * numElementsAlongAxis + i);

            const float maxAbsValue = ComputeMxGroupMax<grp_axis, quant_type, scale_alg>(src, i, group);
            const uint8_t e8m0 = ComputeMxSharedExponent<quant_type, scale_alg>(maxAbsValue);
            const float groupScaling = ComputeMxGroupScaling<quant_type, scale_alg>(maxAbsValue, e8m0);

            flatMax.data()[flatGroupIdx] = static_cast<typename FlatMaxTile::DType>(maxAbsValue);
            exp.data()[GetTileElementOffset<TileDataExp>(row, col)] = e8m0;

            QuantizeMxGroup<grp_axis, quant_type, scale_alg>(
                dst, src, flatScaling, i, group, flatGroupIdx, groupScaling);
        }
    }
}
} // namespace cpu_quant

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara* offset = nullptr)
{
    using SrcT = typename TileDataSrc::DType;
    static_assert(std::is_same_v<SrcT, float>, "Fix: Input has to be float 32");

    for (int row = 0; row < src.GetValidRow(); ++row) {
        for (int col = 0; col < src.GetValidCol(); ++col) {
            const float srcValue = src.data()[GetTileElementOffset<TileDataSrc>(row, col)];
            const float invScale = static_cast<float>(cpu_quant::GetParamValue(scale, row, 0));
            if constexpr (quant_type == QuantType::INT8_SYM) {
                static_assert(
                    std::is_same_v<typename TileDataOut::DType, int8_t>,
                    "Fix: Quant INT8 sym: Out data type has to be int8");
                dst.data()[GetTileElementOffset<TileDataOut>(row, col)] = cpu_quant::ClampInt8(srcValue * invScale);
            } else {
                static_assert(
                    std::is_same_v<typename TileDataOut::DType, uint8_t>,
                    "Fix: Quant INT8 asym: Out data type has to be uint8");
                PTO_CPU_ASSERT(offset != nullptr, "Fix: Quant INT8 asym requires offset.");
                const float zeroPoint = static_cast<float>(cpu_quant::GetParamValue(*offset, row, 0));
                dst.data()[GetTileElementOffset<TileDataOut>(row, col)] =
                    cpu_quant::ClampUint8(srcValue * invScale + zeroPoint);
            }
        }
    }
}

template <
    int grp_axis, QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
    typename TileDataExp, typename TileDataMax, typename TileDataScaling>
inline void TQuantMxCpuImpl(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    cpu_quant::CheckMxQuantTypes<grp_axis, quant_type, scale_alg, TileDataOut, TileDataSrc, TileDataExp>();
    cpu_quant::CheckMxQuantInputs(exp, max, scaling);

    using FlatMaxTile = cpu_quant::FlatMxTile<TileDataMax>;
    using FlatScalingTile = cpu_quant::FlatMxTile<TileDataScaling>;

    FlatMaxTile flatMax(1, TileDataMax::Rows * TileDataMax::Cols);
    cpu_quant::FlattenMxTile(flatMax, *max);

    FlatScalingTile flatScaling(1, TileDataScaling::Rows * TileDataScaling::Cols);
    cpu_quant::FlattenMxTile(flatScaling, *scaling);

    cpu_quant::InitMxOutput<quant_type>(dst);
    cpu_quant::QuantizeMxTile<grp_axis, quant_type, scale_alg>(dst, src, *exp, flatMax, flatScaling);

    cpu_quant::RestoreMxTile(*max, flatMax);
    cpu_quant::RestoreMxTile(*scaling, flatScaling);
}

template <
    QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
    typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    TQuantMxCpuImpl<1, quant_type, QuantScaleAlg::OCP>(dst, src, exp, max, scaling);
}

template <
    int grp_axis, MxQuantAlg mx_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    constexpr QuantScaleAlg scale_alg = (mx_alg == MxQuantAlg::OcpMxFp4E2M1 || mx_alg == MxQuantAlg::OcpMxFp8E4M3) ?
                                            QuantScaleAlg::OCP :
                                            QuantScaleAlg::NV;
    constexpr QuantType quant_type = (mx_alg == MxQuantAlg::OcpMxFp4E2M1 || mx_alg == MxQuantAlg::NvMxFp4E2M1) ?
                                         QuantType::MXFP4_E2M1 :
                                         QuantType::MXFP8;
    TQuantMxCpuImpl<grp_axis, quant_type, scale_alg>(dst, src, exp, max, scaling);
}

template <
    QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    static_assert(
        quant_type == QuantType::MXFP8 || quant_type == QuantType::MXFP4_E2M1,
        "Fix: scale algorithm overload is reserved for MXFP8/MXFP4_E2M1.");
    TQuantMxCpuImpl<1, quant_type, scale_alg>(dst, src, exp, max, scaling);
}

template <
    QuantType quant_type, VecStoreMode store_mode, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling,
    TileDataExp* exp_zz)
{
    static_assert(quant_type == QuantType::MXFP8, "Fix: MX overload is reserved for MXFP8.");
    static_assert(store_mode == VecStoreMode::NZ, "Fix: This overload is reserved for MXFP8 NZ mode.");

    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
        dst, src, exp, max, scaling);

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
