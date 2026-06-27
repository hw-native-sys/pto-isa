/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CONST_ARGS_HPP
#define CONST_ARGS_HPP

#include <cstdint>

constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
constexpr static int32_t UB_ALIGN = 32;
constexpr uint16_t CROSS_CORE_FLAG_MAX_SET_COUNT = 15;
constexpr uint16_t MEGA_MOE_MAX_BUSINESS_HARD_FLAG_ID = 10;
constexpr uint16_t MEGA_MOE_D2C_HARD_FLAG_BASE = 0;
constexpr uint16_t MEGA_MOE_D2C_HARD_FLAG_COUNT = 9;
constexpr uint16_t MEGA_MOE_D2C_HARD_FLAG_LAST = MEGA_MOE_D2C_HARD_FLAG_BASE + MEGA_MOE_D2C_HARD_FLAG_COUNT - 1;
constexpr uint16_t MEGA_MOE_C2V_HARD_FLAG_BASE = 9;
constexpr uint16_t MEGA_MOE_V2C_HARD_FLAG_BASE = 10;
constexpr uint16_t MEGA_MOE_GMM2_TO_COMBINE_HARD_FLAG_BASE = 0;
constexpr uint32_t MEGA_MOE_D2C_MAX_LOGICAL_GROUP_EVENTS = MEGA_MOE_D2C_HARD_FLAG_COUNT * CROSS_CORE_FLAG_MAX_SET_COUNT;
constexpr uint32_t MEGA_MOE_GMM2_TO_COMBINE_MAX_LOGICAL_GROUP_EVENTS =
    MEGA_MOE_D2C_HARD_FLAG_COUNT * CROSS_CORE_FLAG_MAX_SET_COUNT;
static_assert(MEGA_MOE_D2C_HARD_FLAG_LAST < MEGA_MOE_C2V_HARD_FLAG_BASE);
static_assert(MEGA_MOE_V2C_HARD_FLAG_BASE <= MEGA_MOE_MAX_BUSINESS_HARD_FLAG_ID);

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

#include "moe_swiglu_segment.hpp"

#endif // CONST_ARGS_HPP
