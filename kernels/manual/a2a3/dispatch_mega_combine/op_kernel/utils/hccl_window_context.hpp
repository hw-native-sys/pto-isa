/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HCCL_WINDOW_CONTEXT_HPP
#define HCCL_WINDOW_CONTEXT_HPP

#include <cstdint>

static constexpr uint32_t PTO_HCCL_MAX_RANKS = 64;

struct PtoRemoteWindowContext {
    uint64_t workspaceBase = 0;
    uint64_t workspaceBytes = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint64_t windowBytes = 0;
    uint64_t windowIn[PTO_HCCL_MAX_RANKS] = {};
    uint64_t windowOut[PTO_HCCL_MAX_RANKS] = {};
};

#endif // HCCL_WINDOW_CONTEXT_HPP
