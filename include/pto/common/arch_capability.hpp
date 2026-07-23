/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ARCH_CAPABILITY_HPP
#define ARCH_CAPABILITY_HPP

#include <type_traits>
#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp>

namespace pto {

enum class ChipArch : uint8_t {
    A2A3 = 0,
    A5 = 1,
    KIRIN9030 = 2,
    KIRINX90 = 3,
    UNKNOWN = 255,
};

template <ChipArch Arch>
struct ArchTraitsBase {
    static constexpr ChipArch Id = Arch;
    static constexpr bool SupportsBf16 = false;
    static constexpr bool SupportsFp8 = false;
    static constexpr bool SupportsFp4 = false;
    static constexpr bool SupportsSyncAll = false;
    static constexpr bool SupportsComm = false;
    static constexpr bool SupportsTQuant = false;
    static constexpr bool SupportsTHistogram = false;
    static constexpr bool SupportsMxLayout = false;
    static constexpr bool AccSupportsFloat = false;
    static constexpr bool AccSupportsHalf = false;
    static constexpr bool AccSupportsInt32 = false;
    using Bf16Type = void;
    using HiFloat8Type = void;
    using Float8E4M3Type = void;
    using Float8E5M2Type = void;
    using Float8E8M0Type = void;
    using Float4E2M1Type = void;
    using Float4E1M2Type = void;
};

template <ChipArch Arch>
struct ArchTraits;

#if defined(PTO_NPU_ARCH_A2A3)
template <>
struct ArchTraits<ChipArch::A2A3> : ArchTraitsBase<ChipArch::A2A3> {
    static constexpr bool SupportsBf16 = true;
    static constexpr bool SupportsSyncAll = true;
    static constexpr bool SupportsComm = true;
    static constexpr bool AccSupportsFloat = true;
    static constexpr bool AccSupportsInt32 = true;
    using Bf16Type = bfloat16_t;
};
using CurrArch = ArchTraits<ChipArch::A2A3>;

#elif defined(PTO_NPU_ARCH_A5)
template <>
struct ArchTraits<ChipArch::A5> : ArchTraitsBase<ChipArch::A5> {
    static constexpr bool SupportsBf16 = true;
    static constexpr bool SupportsFp8 = true;
    static constexpr bool SupportsFp4 = true;
    static constexpr bool SupportsSyncAll = true;
    static constexpr bool SupportsComm = true;
    static constexpr bool SupportsTQuant = true;
    static constexpr bool SupportsTHistogram = true;
    static constexpr bool SupportsMxLayout = true;
    static constexpr bool AccSupportsFloat = true;
    static constexpr bool AccSupportsInt32 = true;
    using Bf16Type = bfloat16_t;
    using HiFloat8Type = hifloat8_t;
    using Float8E4M3Type = float8_e4m3_t;
    using Float8E5M2Type = float8_e5m2_t;
    using Float8E8M0Type = float8_e8m0_t;
    using Float4E2M1Type = float4_e2m1x2_t;
    using Float4E1M2Type = float4_e1m2x2_t;
};
using CurrArch = ArchTraits<ChipArch::A5>;

#elif defined(PTO_NPU_ARCH_KIRIN9030)
template <>
struct ArchTraits<ChipArch::KIRIN9030> : ArchTraitsBase<ChipArch::KIRIN9030> {
    static constexpr bool AccSupportsHalf = true;
    static constexpr bool AccSupportsInt32 = true;
};
using CurrArch = ArchTraits<ChipArch::KIRIN9030>;

#elif defined(PTO_NPU_ARCH_KIRINX90)
template <>
struct ArchTraits<ChipArch::KIRINX90> : ArchTraitsBase<ChipArch::KIRINX90> {
    static constexpr bool AccSupportsHalf = true;
    static constexpr bool AccSupportsInt32 = true;
};
using CurrArch = ArchTraits<ChipArch::KIRINX90>;

#else
template <>
struct ArchTraits<ChipArch::UNKNOWN> : ArchTraitsBase<ChipArch::UNKNOWN> {
    static constexpr bool SupportsBf16 = true;
    static constexpr bool SupportsFp8 = true;
    static constexpr bool SupportsFp4 = true;
    static constexpr bool SupportsSyncAll = true;
    static constexpr bool SupportsComm = true;
    static constexpr bool SupportsTQuant = true;
    static constexpr bool SupportsTHistogram = true;
    static constexpr bool SupportsMxLayout = true;
    static constexpr bool AccSupportsFloat = true;
    static constexpr bool AccSupportsHalf = true;
    static constexpr bool AccSupportsInt32 = true;
    using Bf16Type = bfloat16_t;
};
using CurrArch = ArchTraits<ChipArch::UNKNOWN>;
#endif

PTO_INTERNAL constexpr ChipArch GetCurrentArch() noexcept { return CurrArch::Id; }

namespace caps {

template <typename T>
PTO_INTERNAL constexpr bool IsFP32()
{
    return std::is_same_v<std::remove_cv_t<T>, float>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsFP16()
{
    return std::is_same_v<std::remove_cv_t<T>, half>;
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsBF16()
{
    if constexpr (Arch::SupportsBf16) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::Bf16Type>;
    } else {
        return false;
    }
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsHF8()
{
    if constexpr (Arch::SupportsFp8) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::HiFloat8Type>;
    } else {
        return false;
    }
}

template <typename T>
PTO_INTERNAL constexpr bool IsInt4()
{
    return std::is_same_v<std::remove_cv_t<T>, int4b_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsSInt8()
{
    return std::is_same_v<std::remove_cv_t<T>, int8_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsUInt8()
{
    return std::is_same_v<std::remove_cv_t<T>, uint8_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsInt8()
{
    return IsSInt8<T>() || IsUInt8<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsSInt16()
{
    return std::is_same_v<std::remove_cv_t<T>, int16_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsUInt16()
{
    return std::is_same_v<std::remove_cv_t<T>, uint16_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsInt16()
{
    return IsSInt16<T>() || IsUInt16<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsSInt32()
{
    return std::is_same_v<std::remove_cv_t<T>, int32_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsUInt32()
{
    return std::is_same_v<std::remove_cv_t<T>, uint32_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsInt32()
{
    return IsSInt32<T>() || IsUInt32<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsSInt64()
{
    return std::is_same_v<std::remove_cv_t<T>, int64_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsUInt64()
{
    return std::is_same_v<std::remove_cv_t<T>, uint64_t>;
}

template <typename T>
PTO_INTERNAL constexpr bool IsInt64()
{
    return IsSInt64<T>() || IsUInt64<T>();
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsFP8E4M3()
{
    if constexpr (Arch::SupportsFp8) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::Float8E4M3Type>;
    } else {
        return false;
    }
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsFP8E5M2()
{
    if constexpr (Arch::SupportsFp8) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::Float8E5M2Type>;
    } else {
        return false;
    }
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsFP8E8M0()
{
    if constexpr (Arch::SupportsFp8) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::Float8E8M0Type>;
    } else {
        return false;
    }
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsFP4E2M1()
{
    if constexpr (Arch::SupportsFp4) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::Float4E2M1Type>;
    } else {
        return false;
    }
}

template <typename T, typename Arch = CurrArch>
PTO_INTERNAL constexpr bool IsFP4E1M2()
{
    if constexpr (Arch::SupportsFp4) {
        return std::is_same_v<std::remove_cv_t<T>, typename Arch::Float4E1M2Type>;
    } else {
        return false;
    }
}

template <typename T>
PTO_INTERNAL constexpr bool IsFP4()
{
    return IsFP4E2M1<T>() || IsFP4E1M2<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsFP8()
{
    return IsHF8<T>() || IsFP8E4M3<T>() || IsFP8E5M2<T>() || IsFP8E8M0<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsFloatingPoint()
{
    return IsFP32<T>() || IsFP16<T>() || IsBF16<T>() || IsFP8<T>() || IsFP4<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsInteger()
{
    return IsInt8<T>() || IsInt16<T>() || IsInt32<T>() || IsInt64<T>() || IsInt4<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsSInteger()
{
    return IsSInt8<T>() || IsSInt16<T>() || IsSInt32<T>() || IsSInt64<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsUInteger()
{
    return IsUInt8<T>() || IsUInt16<T>() || IsUInt32<T>() || IsUInt64<T>();
}

template <typename T>
PTO_INTERNAL constexpr bool IsTypeSupported()
{
    if constexpr (IsFloatingPoint<T>()) {
        return true;
    } else if constexpr (IsInteger<T>()) {
        return true;
    } else {
        return false;
    }
}

} // namespace caps
} // namespace pto

#endif // ARCH_CAPABILITY_HPP
