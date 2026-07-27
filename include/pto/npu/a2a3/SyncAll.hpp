/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_SYNCALL_HPP
#define PTO_NPU_A2A3_SYNCALL_HPP

#include <pto/npu/a2a3/TSync.hpp>
// Soft SYNCALL is backend-independent; only the FFTS path below is A2/A3-specific.
#include <pto/common/syncall_soft.hpp>

namespace pto {

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_IMPL()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
#endif
        return;
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
#if defined(__DAV_CUBE__)
        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
#endif
        return;
    }

#if defined(__DAV_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
#endif
}
} // namespace pto
#endif
