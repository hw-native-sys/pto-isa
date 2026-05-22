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
inline float BitsToFloat(uint32_t bits)
{
    return std::bit_cast<float>(bits);
}

inline uint32_t FloatToBits(float value)
{
    return std::bit_cast<uint32_t>(value);
}

struct NvMxFp8E4M3Spec {
    static constexpr float descaleMultiplier = 1.0f / 448.0f;
    static constexpr uint32_t b16SpecialScaleBits = 0x7F81u;
    static constexpr uint32_t f32SpecialScaleBits = 0x7FC00000u;
};

struct NvMxFp4E2M1Spec {
    static constexpr float descaleMultiplier = 1.0f / 6.0f;
    static constexpr uint32_t b16SpecialScaleBits = 0x7FC0u;
    static constexpr uint32_t f32SpecialScaleBits = 0x7FC00000u;
};

inline uint16_t FloatToBf16BitsTrunc(float value)
{
    return static_cast<uint16_t>(FloatToBits(value) >> 16);
}

inline uint16_t FloatToBf16BitsRound(float value)
{
    constexpr uint32_t bitsShift = 16;
    const uint32_t bits = FloatToBits(value);
    const uint32_t lsb = (bits >> bitsShift) & 1u;
    return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> bitsShift);
}

inline float Bf16BitsToFloat(uint16_t bits)
{
    constexpr uint32_t bitsShift = 16;
    return BitsToFloat(static_cast<uint32_t>(bits) << bitsShift);
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
inline typename TileDataPara::DType GetParamValue(const TileDataPara &tile, int row, int col)
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

inline uint8_t EncodeE4M3Fn(float value)
{
    if (std::isnan(value)) {
        return 0x7Fu;
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

template <typename NvFormatSpec>
inline uint8_t ComputeNvSharedExponent(float maxAbsValue)
{
    if (maxAbsValue == 0.0f) {
        return 0;
    }
    constexpr float descaleMultiplier = NvFormatSpec::descaleMultiplier;
    const float descale = maxAbsValue * descaleMultiplier;
    const uint32_t bits = FloatToBits(descale);
    const uint32_t exponent = (bits & 0x7F800000u) >> 23;
    const uint32_t mantissa = bits & 0x007FFFFFu;
    if (exponent == 0xFFu) {
        return mantissa == 0u ? 0xFEu : 0xFFu;
    }
    const bool roundUp = mantissa > 0u && exponent != 0xFEu && !(exponent == 0u && mantissa <= 0x00400000u);
    return static_cast<uint8_t>(exponent + (roundUp ? 1u : 0u));
}

inline uint8_t ComputeE2M1SharedExponent(float maxAbsValue)
{
    const uint32_t bits = FloatToBits(maxAbsValue);
    const uint32_t exponent = (bits & 0x7F800000u) >> 23;
    const uint32_t mantissa = bits & 0x007FFFFFu;
    if (exponent == 0xFFu && mantissa != 0u) {
        return 0xFFu;
    }
    if (exponent <= 2u) {
        return 0u;
    }
    return static_cast<uint8_t>(exponent - 2u);
}

constexpr int SCALE_MIN_EXP = -127;

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

inline float ComputeE2M1ScalingFromExponent(uint8_t e8m0)
{
    return ComputeMxScalingFromExponent(e8m0);
}

inline float ComputeScalingFromExponent(uint8_t e8m0)
{
    return ComputeMxScalingFromExponent(e8m0);
}

inline std::vector<uint8_t> ReorderExponentZZ(const std::vector<uint8_t> &exp, int rows, int groupCols)
{
    constexpr int groupBlockSize = 2;
    constexpr int rowBlockSize = 16;
    PTO_CPU_ASSERT(rows % rowBlockSize == 0 && groupCols % groupBlockSize == 0,
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

template <QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataSrc>
inline float ComputeMxGroupMax(TileDataSrc &src, int row, int group)
{
    float maxAbsValue = 0.0f;
    uint16_t maxAbsBf16Bits = 0;
    constexpr int colGroupSize = 32;
    for (int inner = 0; inner < colGroupSize; ++inner) {
        const float value = src.data()[GetTileElementOffset<TileDataSrc>(row, group * colGroupSize + inner)];
        if constexpr (quant_type == QuantType::MXFP8 ||
                      (quant_type == QuantType::MXFP4_E2M1 && scale_alg == QuantScaleAlg::NV)) {
            maxAbsValue = std::max(maxAbsValue, std::fabs(value));
        } else {
            maxAbsBf16Bits = std::max(maxAbsBf16Bits, AbsBf16BitsFromFloat(value));
        }
    }
    if constexpr (quant_type == QuantType::MXFP4_E2M1 && scale_alg == QuantScaleAlg::OCP) {
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
    if constexpr (scale_alg == QuantScaleAlg::NV) {
        (void)maxAbsValue;
        return ComputeNvScalingFromExponent(e8m0);
    }
    return ComputeMxGroupScaling<quant_type>(e8m0);
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename FlatScalingTile>
inline void StoreMxEncodedValue(TileDataOut &dst, TileDataSrc &src, FlatScalingTile &flatScaling, int row, int col,
                                int cols, int flatGroupIdx, float groupScaling)
{
    using SrcT = typename TileDataSrc::DType;
    if constexpr (quant_type == QuantType::MXFP8) {
        flatScaling.data()[row * cols + col] = static_cast<typename FlatScalingTile::DType>(groupScaling);
        const float value = static_cast<float>(src.data()[GetTileElementOffset<TileDataSrc>(row, col)]);
        const uint8_t encoded = EncodeE4M3Fn(value * groupScaling);
        dst.data()[GetTileElementOffset<TileDataOut>(row, col)] = static_cast<int8_t>(encoded);
    } else {
        flatScaling.data()[flatGroupIdx] = groupScaling;
        const SrcT srcValue = src.data()[GetTileElementOffset<TileDataSrc>(row, col)];
        const uint8_t encoded = EncodeE2M1Magic(ApplyE2M1ScaleForSource<SrcT>(srcValue, groupScaling));
        auto *dstBytes = reinterpret_cast<uint8_t *>(dst.data());
        const int byteOffset = row * TileDataOut::Cols + col / 2;
        if ((col & 1) == 0) {
            dstBytes[byteOffset] = static_cast<uint8_t>((dstBytes[byteOffset] & 0xF0u) | encoded);
        } else {
            constexpr uint8_t encodedShift = 4;
            dstBytes[byteOffset] = static_cast<uint8_t>((dstBytes[byteOffset] & 0x0Fu) | (encoded << encodedShift));
        }
    }
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename FlatScalingTile>
inline void QuantizeMxGroup(TileDataOut &dst, TileDataSrc &src, FlatScalingTile &flatScaling, int row, int group,
                            int cols, int flatGroupIdx, float groupScaling)
{
    constexpr int colGroupSize = 32;
    for (int inner = 0; inner < colGroupSize; ++inner) {
        const int col = group * colGroupSize + inner;
        StoreMxEncodedValue<quant_type>(dst, src, flatScaling, row, col, cols, flatGroupIdx, groupScaling);
    }
}

template <typename TileData>
using FlatMxTile =
    Tile<TileType::Vec, typename TileData::DType, 1, TileData::Rows * TileData::Cols, BLayout::RowMajor, -1, -1>;

template <QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp>
inline void CheckMxQuantTypes()
{
    static_assert(quant_type == QuantType::MXFP8 || quant_type == QuantType::MXFP4_E2M1,
                  "Fix: MX overload is reserved for MXFP8/MXFP4_E2M1.");
    static_assert(scale_alg == QuantScaleAlg::OCP || scale_alg == QuantScaleAlg::NV,
                  "Fix: MX scale algorithm must be OCP or NV.");
    using SrcT = typename TileDataSrc::DType;
    if constexpr (quant_type == QuantType::MXFP8) {
        static_assert(std::is_same_v<SrcT, float> || std::is_same_v<SrcT, half> || std::is_same_v<SrcT, aclFloat16> ||
                          std::is_same_v<SrcT, bfloat16_t>,
                      "Fix: MXFP8 CPU sim supports float/float16/bfloat16 source.");
        static_assert(std::is_same_v<typename TileDataOut::DType, int8_t>, "Fix: MXFP8 output must be int8 bytes.");
    } else {
        static_assert(std::is_same_v<SrcT, float> || std::is_same_v<SrcT, half> || std::is_same_v<SrcT, aclFloat16> ||
                          std::is_same_v<SrcT, bfloat16_t>,
                      "Fix: MXFP4_E2M1 CPU sim supports float/float16/bfloat16 source.");
        static_assert(std::is_same_v<typename TileDataOut::DType, float4_e2m1x2_t>,
                      "Fix: MXFP4_E2M1 output must be float4_e2m1x2_t.");
    }
    static_assert(std::is_same_v<typename TileDataExp::DType, uint8_t>, "Fix: MXFP8 exponent must be uint8 bytes.");
}

template <typename TileDataSrc, typename TileDataExp, typename TileDataMax, typename TileDataScaling>
inline void CheckMxQuantInputs(TileDataSrc &src, TileDataExp *exp, TileDataMax *max, TileDataScaling *scaling)
{
    constexpr unsigned srcColDiv = 32;
    PTO_CPU_ASSERT(exp != nullptr && max != nullptr && scaling != nullptr, "Fix: MX quant requires tiles.");
    PTO_CPU_ASSERT(src.GetValidCol() % srcColDiv == 0,
                   "Fix: MX CPU sim currently requires valid cols to be a multiple of 32.");
}

template <typename FlatTileData, typename TileData>
inline void FlattenMxTile(FlatTileData &flatTile, TileData &tile)
{
    TRESHAPE_IMPL(flatTile, tile);
}

template <typename TileData, typename FlatTileData>
inline void RestoreMxTile(TileData &tile, FlatTileData &flatTile)
{
    TRESHAPE_IMPL(tile, flatTile);
}

template <QuantType quant_type, typename TileDataOut>
inline void InitMxOutput(TileDataOut &dst)
{
    if constexpr (quant_type == QuantType::MXFP4_E2M1) {
        std::fill(reinterpret_cast<uint8_t *>(dst.data()),
                  reinterpret_cast<uint8_t *>(dst.data()) + TileDataOut::Rows * TileDataOut::Cols, 0);
    }
}

template <QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
          typename FlatExpTile, typename FlatMaxTile, typename FlatScalingTile>
inline void QuantizeMxTile(TileDataOut &dst, TileDataSrc &src, FlatExpTile &flatExp, FlatMaxTile &flatMax,
                           FlatScalingTile &flatScaling)
{
    const int rows = src.GetValidRow();
    const int cols = src.GetValidCol();
    const int groupCols = cols / 32;
    for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groupCols; ++group) {
            const int flatGroupIdx = row * groupCols + group;
            const float maxAbsValue = ComputeMxGroupMax<quant_type, scale_alg>(src, row, group);
            const uint8_t e8m0 = ComputeMxSharedExponent<quant_type, scale_alg>(maxAbsValue);
            const float groupScaling = ComputeMxGroupScaling<quant_type, scale_alg>(maxAbsValue, e8m0);
            flatMax.data()[flatGroupIdx] = maxAbsValue;
            flatExp.data()[flatGroupIdx] = e8m0;
            QuantizeMxGroup<quant_type>(dst, src, flatScaling, row, group, cols, flatGroupIdx, groupScaling);
        }
    }
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

template <QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling>
inline void TQuantMxCpuImpl(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                            TileDataScaling *scaling)
{
    cpu_quant::CheckMxQuantTypes<quant_type, scale_alg, TileDataOut, TileDataSrc, TileDataExp>();
    cpu_quant::CheckMxQuantInputs(src, exp, max, scaling);

    using FlatExpTile = cpu_quant::FlatMxTile<TileDataExp>;
    using FlatMaxTile = cpu_quant::FlatMxTile<TileDataMax>;
    using FlatScalingTile = cpu_quant::FlatMxTile<TileDataScaling>;

    FlatExpTile flatExp(1, TileDataExp::Rows * TileDataExp::Cols);
    cpu_quant::FlattenMxTile(flatExp, *exp);

    FlatMaxTile flatMax(1, TileDataMax::Rows * TileDataMax::Cols);
    cpu_quant::FlattenMxTile(flatMax, *max);

    FlatScalingTile flatScaling(1, TileDataScaling::Rows * TileDataScaling::Cols);
    cpu_quant::FlattenMxTile(flatScaling, *scaling);

    cpu_quant::InitMxOutput<quant_type>(dst);
    cpu_quant::QuantizeMxTile<quant_type, scale_alg>(dst, src, flatExp, flatMax, flatScaling);

    cpu_quant::RestoreMxTile(*exp, flatExp);
    cpu_quant::RestoreMxTile(*max, flatMax);
    cpu_quant::RestoreMxTile(*scaling, flatScaling);
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataScaling *scaling)
{
    TQuantMxCpuImpl<quant_type, QuantScaleAlg::OCP>(dst, src, exp, max, scaling);
}

template <QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataScaling *scaling)
{
    static_assert(quant_type == QuantType::MXFP8 || quant_type == QuantType::MXFP4_E2M1,
                  "Fix: scale algorithm overload is reserved for MXFP8/MXFP4_E2M1.");
    TQuantMxCpuImpl<quant_type, scale_alg>(dst, src, exp, max, scaling);
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
