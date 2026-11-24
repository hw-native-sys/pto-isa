/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_LINX_BFLOAT16_HPP
#define PTO_CPU_LINX_BFLOAT16_HPP
#include <bit>
#include <cstdint>
#include <type_traits>

// Match the native compiler's scalar BF16 conversions while retaining the
// two-byte, trivially-copyable carrier used by linx_cvt and Tile storage.
struct __blkc_bf16 {
    uint16_t data = 0;
    constexpr __blkc_bf16() = default;
    constexpr __blkc_bf16(float value)
    {
        const uint32_t raw = std::bit_cast<uint32_t>(value);
        if ((raw & 0x7fffffffU) > 0x7f800000U)
            data = uint16_t((raw >> 16) | 0x40U);
        else
            data = uint16_t((raw + 0x7fffU + ((raw >> 16) & 1U)) >> 16);
    }
    constexpr operator float() const { return std::bit_cast<float>(uint32_t(data) << 16); }
};
static_assert(sizeof(__blkc_bf16) == 2 && std::is_trivially_copyable_v<__blkc_bf16>);
#endif
