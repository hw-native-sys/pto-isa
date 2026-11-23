/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TEXPANDS_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TEXPANDS_HPP_

#include "common.hpp"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16::custom {
namespace detail {
template <uint32_t ScalarBits>
PTO_INTERNAL void expandScalarImpl(__ubuf__ float* dst)
{
    __VEC_SCOPE__
    {
        const auto rows = pset_b32(PAT_VL32);
        vector_f32 value;
        vbr((vector_u32&)value, ScalarBits);
        vsts(value, dst, 0, NORM_B32, rows);
    }
}
} // namespace detail

// Fill the 32 live FP32 lanes with a scalar embedded in the VF instruction.
// ScalarBits is the IEEE-754 representation of the compile-time value.
template <uint32_t ScalarBits, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(RowStateTile& dst, WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::expandScalarImpl<ScalarBits>((__ubuf__ float*)dst.data());
    return {};
}

} // namespace flash_mla_bf16::custom
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TEXPANDS_HPP_
