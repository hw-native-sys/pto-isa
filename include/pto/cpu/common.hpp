/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_HPP
#define COMMON_HPP

#include <pto/common/type.hpp>

namespace pto {

enum QuantModeCPU_t
{
    NoQuant = 0,        // 不使能量化功能
    F322F16 = 1,        // float量化成half, scalar量化
    F322BF16 = 16,      // float量化成bfloat16_t, scalar量化
    DEQF16 = 5,         // int32_t量化成half, scalar量化
    VDEQF16 = 4,        // int32_t量化成half，tensor量化
    QF322B8_PRE = 24,   // float量化成int8_t/uint8_t，scalar量化
    QF322F16_PRE = 32,  // float量化成half，scalar量化
    QF322BF16_PRE = 34, // float量化成bfloat16_t，scalar量化
    VQF322B8_PRE = 23,  // float量化成int8_t/uint8_t，tensor量化
    REQ8 = 3,           // int32_t量化成int8_t/uint8_t，scalar量化
    VREQ8 = 2,          // int32_t量化成int8_t/uint8_t，tensor量化
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
};

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantModeCPU_t GetCastPreQuantMode()
{
    QuantModeCPU_t quantPre = QuantModeCPU_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::F322F16;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantModeCPU_t::F322BF16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantModeCPU_t GetScalarPreQuantMode()
{
    QuantModeCPU_t quantPre = QuantModeCPU_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::QF322B8_PRE;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::QF322F16_PRE;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantModeCPU_t::QF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::REQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::DEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantModeCPU_t::SHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantModeCPU_t GetVectorPreQuantMode()
{
    QuantModeCPU_t quantPre = QuantModeCPU_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::VQF322B8_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantModeCPU_t::VREQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantModeCPU_t::VDEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantModeCPU_t::VSHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename T>
inline T ReLU(T val)
{
    if (val < 0)
        return 0;
    return val;
}

inline float extract_m1_from_quant(uint64_t quant)
{
    uint32_t m1_bits = static_cast<uint32_t>((quant >> 13) & 0x7FFFF);
    uint32_t sign_bit = (m1_bits >> 18) & 0x1;
    uint32_t exponent = (m1_bits >> 10) & 0xFF;
    uint32_t mantissa = m1_bits & 0x3FF;

    if (exponent == 0 && mantissa == 0)
        return 0.0f;

    float sign_val = (sign_bit == 1) ? -1.0f : 1.0f;
    float mantissa_val = 1.0f + (static_cast<float>(mantissa) / 1024.0f);
    float exponent_val = std::pow(2.0f, static_cast<float>(exponent) - 127.0f);

    return sign_val * mantissa_val * exponent_val;
}

template <typename DstType, typename SrcType, QuantModeCPU_t mode, bool use_relu>
DstType quantize_element(SrcType src_val, uint64_t scalar)
{
    uint64_t ctrl_bits = get_task_cookie();
    float f_scale = extract_m1_from_quant(scalar);
    uint32_t offset = static_cast<uint32_t>((scalar >> 37) & 0x1FF);
    uint32_t sign = static_cast<uint32_t>((scalar >> 46) & 0x1);
    uint32_t saturate_inf = static_cast<uint32_t>((ctrl_bits >> 48) & 0x1);

    float result_f = static_cast<float>(src_val) * f_scale;

    if constexpr (mode == QuantModeCPU_t::QF322B8_PRE || mode == QuantModeCPU_t::VQF322B8_PRE ||
                  mode == QuantModeCPU_t::REQ8 || mode == QuantModeCPU_t::VREQ8) {
        float rounded = std::nearbyint(result_f + offset);
        float min = sign == 1 ? -128.0f : 0.0f;
        float max = sign == 1 ? 127.0f : 255.0f;
        result_f = std::clamp(rounded, min, max);
    } else if constexpr (mode == QuantModeCPU_t::DEQF16 || mode == QuantModeCPU_t::VDEQF16) {
        result_f = std::clamp(result_f, -F16_MAX, F16_MAX);
    } else if constexpr (mode == QuantModeCPU_t::QF322F16_PRE) {
        if (std::isnan(result_f) && saturate_inf == 1) {
            result_f = 0.0f;
        } else if (std::isfinite(result_f) || saturate_inf == 1) {
            result_f = std::clamp(result_f, -F16_MAX, F16_MAX);
        }
    } else if constexpr (mode == QuantModeCPU_t::QF322BF16_PRE) {
        if (std::isnan(result_f) && saturate_inf == 1) {
            result_f = 0.0f;
        } else if (std::isfinite(result_f) || saturate_inf == 1) {
            float F32_MAX = std::numeric_limits<float>::max();
            result_f = std::clamp(result_f, -F32_MAX, F32_MAX);
        }
    } else if constexpr (mode == QuantModeCPU_t::SHIFTS322S16 || mode == QuantModeCPU_t::VSHIFTS322S16) {
        int32_t shift_bit = ((scalar >> 32) & 0xF) + 1;
        int32_t shifted_val = src_val >> shift_bit;
        int16_t I16_MAX = std::numeric_limits<int16_t>::max();
        int16_t I16_MIN = std::numeric_limits<int16_t>::min();
        result_f = std::clamp((int16_t)shifted_val, I16_MIN, I16_MAX);
    }

    if constexpr (use_relu)
        result_f = ReLU(result_f);
    return static_cast<DstType>(result_f);
}

} // namespace pto

#endif
