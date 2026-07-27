/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only hard (FFTS) SYNCALL ST, dav-c310-cube single-chevron launch.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int32_t kAicHardBlockCount = 18;

extern "C" __global__ AICORE void RunHardSyncAllAIC(__gm__ int32_t __out__* out)
{
    (void)out;
    SYNCALL<SyncCoreType::AICOnly>();
}

void LaunchHardSyncAllAIC(int32_t* out, void* stream)
{
    RunHardSyncAllAIC<<<kAicHardBlockCount, nullptr, stream>>>(out);
}
