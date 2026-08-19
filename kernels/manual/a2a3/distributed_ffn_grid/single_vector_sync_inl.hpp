/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISTRIBUTED_FFN_GRID_SINGLE_VECTOR_SYNC_INL_HPP_H_
#define DISTRIBUTED_FFN_GRID_SINGLE_VECTOR_SYNC_INL_HPP_H_

#include <cstdint>

// dav-c220 chevron kernels are physically 1C2V. A2/A3 mode-2 C/V synchronization
// broadcasts C2V notifications to both AIVs and reduces V2C notifications from
// both AIVs. AIV1 therefore cannot simply return while a phase uses TPipe: it must
// contribute the control handshake that lets the Cube and AIV0 make progress.
// These helpers reproduce only TPOP/TPUSH's cross-core bookkeeping; AIV1 performs
// no tile load/store, vector math, Grid communication, or channel binding.

template <typename Pipe>
AICORE inline void FfnSyncOnlyPop(Pipe& pipe)
{
    if (pipe.cons.getWaitStatus()) {
        pipe.cons.wait();
    }
    if (pipe.cons.getFreeStatus() && Pipe::shouldNotifyFree(pipe.cons.tileIndex)) {
        pipe.cons.free();
    }
    pipe.cons.tileIndex++;
}

template <typename Pipe>
AICORE inline void FfnSyncOnlyPush(Pipe& pipe)
{
    if (pipe.prod.getAllocateStatus() && Pipe::shouldWaitFree(pipe.prod.tileIndex)) {
        pipe.prod.allocate();
    }
    pipe.prod.tileIndex++;
    if (pipe.prod.getRecordStatus()) {
        pipe.prod.record();
    }
}

template <typename GatePipe, typename UpPipe>
AICORE inline void FfnInactiveVectorC2vHandshake(__gm__ uint8_t* gatePartial, __gm__ uint8_t* upPartial)
{
    GatePipe gatePipe(reinterpret_cast<__gm__ void*>(gatePartial), 0x0000, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void*>(upPartial), 0x1000, 0);
    FfnSyncOnlyPop(gatePipe);
    FfnSyncOnlyPop(upPipe);
}

template <typename GatePipe, typename UpPipe, typename HiddenPipe>
AICORE inline void FfnInactiveVectorC2vV2cHandshake(
    __gm__ uint8_t* gatePartial, __gm__ uint8_t* upPartial, __gm__ uint8_t* hidden)
{
    GatePipe gatePipe(reinterpret_cast<__gm__ void*>(gatePartial), 0x0000, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void*>(upPartial), 0x1000, 0);
    HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void*>(hidden), 0, 0x0);
    FfnSyncOnlyPop(gatePipe);
    FfnSyncOnlyPop(upPipe);
    FfnSyncOnlyPush(hiddenPipe);
}

#endif // DISTRIBUTED_FFN_GRID_SINGLE_VECTOR_SYNC_INL_HPP_H_
