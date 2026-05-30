/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

// ============================================================================
// HcclDeviceContext
//
// Binary layout matches HcclCombinOpParamA5 returned by
// HcclAllocComResourceByTiling() on Ascend 910 A5 (DAV_3510).
// A5 uses HCCL_MTE_MAX_RANK_NUM = 64 for windowsIn/Out arrays,
// unlike A3 which uses AICPU_MAX_RANK_NUM_V1 = 32.
// ============================================================================

static constexpr uint32_t HCCL_MAX_RANK_NUM = 64;

struct HcclDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;

    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];

    // CCU registers (A5-specific, not used by dispatch kernel)
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};
