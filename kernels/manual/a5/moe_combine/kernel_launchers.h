/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_KERNEL_LAUNCHERS_H_
#define MOE_COMBINE_KERNEL_LAUNCHERS_H_

#include "common.h"

#include <cstdint>

namespace moe_combine {

void LaunchMoeCombineKernel(MoeCombineShape shape, uint32_t myRank, uint8_t *expertOutput, uint8_t *probs,
                            uint8_t *outputC, uint8_t *routeMeta, uint8_t *peerWindow, uint8_t *hcclCtx,
                            uint8_t *workspace, void *stream, uint32_t launchBlockCount);

} // namespace moe_combine

#endif // MOE_COMBINE_KERNEL_LAUNCHERS_H_
