/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP
#ifdef __CPU_SIM
#include <bit>
#endif
#include <pto/common/type.hpp>
#include <pto/common/memory.hpp>

namespace pto {
constexpr int REPEAT_BYTE = 256;
constexpr int REPEAT_MAX = 255;
constexpr const int BLOCK_BYTE_SIZE = 32;
constexpr const int FIXP_BURST_UNIT_LEN = 64;
constexpr const uint32_t SHIFT_BLOCK_LEN = 4;
constexpr const uint32_t SHIFT_BLOCK_BYTE = 5;
constexpr const uint32_t SHIFT_FRACTAL_BYTE = 9;
constexpr const int REPEAT_STRIDE_MAX = 255;
constexpr const uint64_t BLOCK_MAX_PER_REPEAT = 8;
constexpr const uint32_t TMP_UB_SIZE = 8 * 1024;
constexpr const uint32_t TMP_UB_OFFSET = 184 * 1024;
constexpr const uint64_t MASK_LEN = 64;
constexpr const int BLOCK_LEN = 16;
constexpr const int CUBE_BLOCK_SIZE = 512;
constexpr const int C0_SIZE_BYTE = 32;
constexpr const int FRACTAL_NZ_ROW = 16;
constexpr const int ACC_C0_SIZE = 16;
constexpr const uint32_t B4_C0_SIZE = 64;
constexpr const int MX_COL_LEN = 2;
constexpr const int MX_ROW_LEN = 16;
constexpr const int MX_BLOCK_SIZE = 32;
constexpr const int B8_DATA_TYPE_OFFSET = 8;
constexpr const int MAD_MODE_BIT = 46;
constexpr const int MAD_ROUND_MODE_BIT = 47;
constexpr const int TROW_PROD_LOOP_B16 = 7;
constexpr const int TROW_PROD_LOOP_B32 = 6;
constexpr const int PAD_SHIFT_LENGTH = 32;

// ============================================================================
// Custom pad value helpers for uint64_t-based PadValue enum
// ============================================================================
// PadValue uses uint64_t underlying type
// - Values 0-3 are standard enum cases (Null, Zero, Max, Min)
// - Custom values have bit 32 set, with the float bit pattern in bits [32:63]

// Check if a PadValue is a custom value (bit 32+ set)
AICORE constexpr bool isCustomPadValue(PadValue pv)
{
    return static_cast<uint64_t>(pv) >= static_cast<uint64_t>(PadValue::CustomBase);
}

// Extract the 32-bit value from a custom PadValue (returns the float bits)
AICORE constexpr uint32_t getCustomPadBits(PadValue pv)
{
    return static_cast<uint32_t>(static_cast<uint64_t>(pv) & 0xFFFFFFFFULL);
}

// Helper to create a custom PadValue from a compile-time float/int constant
// Usage: PadCustom<-1.0f>, PadCustom<0.5f>, PadCustom<42>
namespace detail {
// Use union for compile-time float-to-bits (works on NPU compilers)
template <auto V>
constexpr uint32_t floatToBits()
{
    if constexpr (std::is_same_v<decltype(V), float>) {
        union {
            float f;
            uint32_t u;
        } conv = {V};
        return conv.u;
    } else if constexpr (std::is_same_v<decltype(V), double>) {
        union {
            float f;
            uint32_t u;
        } conv = {static_cast<float>(V)};
        return conv.u;
    } else if constexpr (std::is_integral_v<decltype(V)>) {
        return static_cast<uint32_t>(V);
    } else {
        return 0;
    }
}
} // namespace detail

template <auto V>
inline constexpr PadValue PadCustom = static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) |
                                                            static_cast<uint64_t>(detail::floatToBits<V>()));

// Helper constexpr function to create custom PadValue from float
// Works on both CPU_SIM and NPU (host + device) using __builtin_bit_cast
// Usage: constexpr PadValue PadCustomNeg1 = PadValueCustom(-1.0f);
// Note: For fp16/bf16, use PadValueCustomHalf()/PadValueCustomBf16() or pass fp16/bf16 bits directly
AICORE constexpr PadValue PadValueCustom(float value)
{
    return static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) |
                                 static_cast<uint64_t>(__builtin_bit_cast(uint32_t, value)));
}

// For fp16/bf16, pass the raw 16-bit representation directly
AICORE constexpr PadValue PadValueCustom16(uint16_t bits16)
{
    return static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) | static_cast<uint64_t>(bits16));
}

#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
// Usage: constexpr PadValue PadCustomNeg1_Half = PadValueCustom((half)-1.0);
// NPU aicore compiler has half as built-in type
AICORE constexpr PadValue PadValueCustom(half value)
{
    return static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) |
                                 static_cast<uint64_t>(__builtin_bit_cast(uint16_t, value)));
}

// NPU aicore compiler has bfloat16_t as built-in type
AICORE constexpr PadValue PadValueCustom(bfloat16_t value)
{
    return static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) |
                                 static_cast<uint64_t>(__builtin_bit_cast(uint16_t, value)));
}
#endif

#if defined(__CPU_SIM) || defined(__COSTMODEL)
// Usage: constexpr PadValue PadCustomNeg1_Half = PadValueCustom((_Float16)-1.0);
// Or with f16 suffix: PadValueCustom(-1.0f16)
constexpr PadValue PadValueCustom(_Float16 value)
{
    return static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) |
                                 static_cast<uint64_t>(__builtin_bit_cast(uint16_t, value)));
}

#ifdef CPU_SIM_BFLOAT_ENABLED
// Usage: constexpr PadValue PadCustomNeg1_Bf16 = PadValueCustom((bfloat16_t)-1.0);
// Requires C++23 with std::bfloat16_t support
constexpr PadValue PadValueCustom(bfloat16_t value)
{
    return static_cast<PadValue>(static_cast<uint64_t>(PadValue::CustomBase) |
                                 static_cast<uint64_t>(__builtin_bit_cast(uint16_t, value)));
}
#endif
#endif

template <typename DType, PadValue PadVal>
struct PadValueMap {
    PTO_STATIC_ASSERT(sizeof(DType) < 0, "TLOAD: Unsupported DType for PadValue!");
};

template <PadValue PadVal>
struct PadValueMap<int64_t, PadVal> {
    static constexpr auto value = uint32_t(0);
};
template <PadValue PadVal>
struct PadValueMap<uint64_t, PadVal> {
    static constexpr auto value = uint32_t(0);
};

template <>
struct PadValueMap<float, PadValue::Null> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<float, PadValue::Zero> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<float, PadValue::Min> {
    static constexpr auto value = uint32_t(0xff800000UL);
};
template <>
struct PadValueMap<float, PadValue::Max> {
    static constexpr auto value = uint32_t(0x7f800000UL);
};

template <>
struct PadValueMap<int32_t, PadValue::Null> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<int32_t, PadValue::Zero> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<int32_t, PadValue::Min> {
    static constexpr auto value = uint32_t(0x80000000UL);
};
template <>
struct PadValueMap<int32_t, PadValue::Max> {
    static constexpr auto value = uint32_t(0x7fffffffUL);
};

template <>
struct PadValueMap<uint32_t, PadValue::Null> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<uint32_t, PadValue::Zero> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<uint32_t, PadValue::Min> {
    static constexpr auto value = uint32_t(0);
};
template <>
struct PadValueMap<uint32_t, PadValue::Max> {
    static constexpr auto value = uint32_t(0xffffffffUL);
};

#ifndef __CPU_SIM
#ifndef __COSTMODEL
template <>
struct PadValueMap<bfloat16_t, PadValue::Null> {
    static constexpr auto value = uint16_t(0);
};

template <>
struct PadValueMap<bfloat16_t, PadValue::Zero> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<bfloat16_t, PadValue::Min> {
    static constexpr auto value = uint16_t(0xff80);
};
template <>
struct PadValueMap<bfloat16_t, PadValue::Max> {
    static constexpr auto value = uint16_t(0x7f80);
};
#endif
#endif
template <>
struct PadValueMap<half, PadValue::Null> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<half, PadValue::Zero> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<half, PadValue::Min> {
    static constexpr auto value = uint16_t(0xfc00);
};
template <>
struct PadValueMap<half, PadValue::Max> {
    static constexpr auto value = uint16_t(0x7c00);
};

template <>
struct PadValueMap<int16_t, PadValue::Null> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<int16_t, PadValue::Zero> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<int16_t, PadValue::Min> {
    static constexpr auto value = uint16_t(0x8000);
};
template <>
struct PadValueMap<int16_t, PadValue::Max> {
    static constexpr auto value = uint16_t(0x7fff);
};

template <>
struct PadValueMap<uint16_t, PadValue::Null> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<uint16_t, PadValue::Zero> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<uint16_t, PadValue::Min> {
    static constexpr auto value = uint16_t(0);
};
template <>
struct PadValueMap<uint16_t, PadValue::Max> {
    static constexpr auto value = uint16_t(0xffff);
};

template <>
struct PadValueMap<int8_t, PadValue::Null> {
    static constexpr auto value = uint8_t(0);
};
template <>
struct PadValueMap<int8_t, PadValue::Zero> {
    static constexpr auto value = uint8_t(0);
};
template <>
struct PadValueMap<int8_t, PadValue::Min> {
    static constexpr auto value = uint8_t(0xff);
};
template <>
struct PadValueMap<int8_t, PadValue::Max> {
    static constexpr auto value = uint8_t(0x7f);
};

template <>
struct PadValueMap<uint8_t, PadValue::Null> {
    static constexpr auto value = uint8_t(0);
};
template <>
struct PadValueMap<uint8_t, PadValue::Zero> {
    static constexpr auto value = uint8_t(0);
};
template <>
struct PadValueMap<uint8_t, PadValue::Min> {
    static constexpr auto value = uint8_t(0);
};
template <>
struct PadValueMap<uint8_t, PadValue::Max> {
    static constexpr auto value = uint8_t(0xff);
};

#if defined(PTO_NPU_ARCH_A5)
template <PadValue PadVal>
struct PadValueMap<float4_e1m2x2_t, PadVal> {
    static constexpr auto value = uint8_t(0);
};
template <PadValue PadVal>
struct PadValueMap<float4_e2m1x2_t, PadVal> {
    static constexpr auto value = uint8_t(0);
};
template <PadValue PadVal>
struct PadValueMap<float8_e8m0_t, PadVal> {
    static constexpr auto value = uint8_t(0);
};
template <PadValue PadVal>
struct PadValueMap<float8_e4m3_t, PadVal> {
    static constexpr auto value = uint8_t(0);
};
template <PadValue PadVal>
struct PadValueMap<float8_e5m2_t, PadVal> {
    static constexpr auto value = uint8_t(0);
};
template <PadValue PadVal>
struct PadValueMap<hifloat8_t, PadVal> {
    static constexpr auto value = uint8_t(0);
};
#endif

template <typename TileData>
PTO_INTERNAL constexpr auto GetPadValue()
{
    using DType = typename TileData::DType;
    constexpr PadValue PadVal = TileData::PadVal;
    // Handle custom pad values (works on both CPU and NPU)
    if constexpr (isCustomPadValue(PadVal)) {
        constexpr uint32_t bits = getCustomPadBits(PadVal);
        if constexpr (std::is_same_v<DType, float>) {
            return bits; // float uses raw bits directly
        } else if constexpr (sizeof(DType) == 2) {
            // fp16 and bf16 both use lower 16 bits
            // (PadValueCustom(half) and PadValueCustom(bfloat16_t) store native bits)
            return static_cast<uint32_t>(bits & 0xFFFF);
        } else if constexpr (sizeof(DType) == 1) {
            return static_cast<uint32_t>(bits & 0xFF);
        } else {
            return bits;
        }
    } else {
        return PadValueMap<DType, PadVal>::value;
    }
}

template <typename TileData>
PTO_INTERNAL constexpr TileLayoutCustom GetTileLayoutCustom()
{
    if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)) {
        return TileLayoutCustom::ND;
    } else if constexpr (!TileData::isRowMajor && (TileData::SFractal == SLayout::NoneBox)) {
        return TileLayoutCustom::DN;
    } else if constexpr (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor) &&
                         TileData::SFractalSize == 512) {
        return TileLayoutCustom::NZ;
    } else if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor) &&
                         TileData::SFractalSize == 512) {
        return TileLayoutCustom::ZN;
    } else if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor) &&
                         TileData::SFractalSize == 512) {
        return TileLayoutCustom::ZZ;
    } else if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::GpuSwizzle128B) &&
                         TileData::SFractalSize == 128) {
        return TileLayoutCustom::GPU_SWIZZLE_128B;
    } else {
        return TileLayoutCustom::NONE;
    }
}
} // namespace pto
#endif
