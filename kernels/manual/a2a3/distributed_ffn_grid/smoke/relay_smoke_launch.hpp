/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef RELAY_SMOKE_LAUNCH_HPP
#define RELAY_SMOKE_LAUNCH_HPP

#include <cstdint>

// GridPipe reduce <-> unicast CHANNEL RELAY smoke kernel.  Three blocks form a
// 1 x 3 row.  The host first runs reduce, unicast, and reduce in separate launches,
// then runs the same three stages inside one fourth launch.  See
// relay_smoke_config.hpp for the schedule and what each boundary proves.
void launchRelaySmokeKernel(
    uint8_t* ffts, uint8_t* windows, uint8_t* inBuf, uint8_t* outBuf, uint8_t* hcclCtx, int phase, void* stream);

#endif // RELAY_SMOKE_LAUNCH_HPP
