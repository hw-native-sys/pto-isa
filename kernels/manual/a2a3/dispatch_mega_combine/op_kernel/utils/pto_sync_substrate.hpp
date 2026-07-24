/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DISPATCH_MEGA_COMBINE_PTO_SYNC_SUBSTRATE_HPP
#define DISPATCH_MEGA_COMBINE_PTO_SYNC_SUBSTRATE_HPP

#include "kernel_operator.h"

constexpr uint16_t kFftsBaseCountMask = 0xf;
constexpr uint16_t kFftsModeMask = 0x3;
constexpr uint16_t kFftsModeShift = 4;
constexpr uint16_t kFftsFlagMask = 0xf;
constexpr uint16_t kFftsFlagShift = 8;

__aicore__ inline uint16_t MakeFftsMsg(uint16_t mode, uint16_t flagId, uint16_t baseCount = 0x1)
{
    return static_cast<uint16_t>(
        (baseCount & kFftsBaseCountMask) + ((mode & kFftsModeMask) << kFftsModeShift) +
        ((flagId & kFftsFlagMask) << kFftsFlagShift));
}

template <pipe_t pipe>
__aicore__ inline constexpr bool IsSplitCubePipe()
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002)
    return pipe == PIPE_S || pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_MTE3 || pipe == PIPE_M;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
    return pipe == PIPE_S || pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_FIX || pipe == PIPE_M;
#else
    return pipe == PIPE_S || pipe == PIPE_MTE1 || pipe == PIPE_MTE2 || pipe == PIPE_MTE3 || pipe == PIPE_FIX ||
           pipe == PIPE_M;
#endif
}

template <uint8_t ModeId, pipe_t Pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
{
    if constexpr (Pipe == PIPE_S || Pipe == PIPE_V || Pipe == PIPE_MTE2 || Pipe == PIPE_MTE3) {
        if ASCEND_IS_AIV {
            ffts_cross_core_sync(Pipe, MakeFftsMsg(ModeId, flagId));
        }
    }
    if constexpr (IsSplitCubePipe<Pipe>()) {
        if ASCEND_IS_AIC {
            ffts_cross_core_sync(Pipe, MakeFftsMsg(ModeId, flagId));
        }
    }
}

template <uint8_t ModeId, pipe_t Pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)
{
    (void)ModeId;
    if constexpr (Pipe == PIPE_S || Pipe == PIPE_V || Pipe == PIPE_MTE2 || Pipe == PIPE_MTE3) {
        if ASCEND_IS_AIV {
            wait_flag_dev(flagId);
        }
    }
    if constexpr (IsSplitCubePipe<Pipe>()) {
        if ASCEND_IS_AIC {
            wait_flag_dev(flagId);
        }
    }
}

#endif // DISPATCH_MEGA_COMBINE_PTO_SYNC_SUBSTRATE_HPP
