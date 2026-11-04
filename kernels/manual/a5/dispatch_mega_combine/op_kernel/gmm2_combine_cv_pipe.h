/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H
#define DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H

#include <pto/pto-inst.hpp>

#include "gmm_task_queue.h"
#include "utils/const_args.hpp"

constexpr uint32_t kGmm2CombineCvTileRows = 256U;
constexpr uint32_t kGmm2CombineCvTileCols = 256U;
constexpr uint32_t kGmm2CombineCvFifoDepth = 1U;
constexpr uint32_t kGmm2CombineCvSlotBytes = kGmm2CombineCvTileRows * kGmm2CombineCvTileCols * sizeof(uint16_t);
constexpr uint32_t kGmm2CombineCvBufferBytes = kGmm2CombineCvFifoDepth * kGmm2CombineCvSlotBytes;
constexpr uint32_t kGmm2CombineMetadataSlotBytes = kGmm2CombineCvTileRows * kMegaMoeRouteMetaFields * sizeof(int32_t);
constexpr uint32_t kGmm2CombineRequiredUbBytes =
    kGmm2CombineCvBufferBytes + kGmm2CombineCvFifoDepth * kGmm2CombineMetadataSlotBytes;

constexpr uint16_t kGmm2CombineCvReadyFlagBase = 3U;
constexpr uint16_t kGmm2CombineCvFreeFlagBase = kGmm2CombineCvReadyFlagBase + kGmm2CombineCvFifoDepth;
constexpr uint16_t kGmm2CombineControlReadyFlagBase = kGmm2CombineCvFreeFlagBase + kGmm2CombineCvFifoDepth;
constexpr uint16_t kGmm2CombineControlFreeFlagBase = kGmm2CombineControlReadyFlagBase + kGmm2CombineCvFifoDepth;
constexpr uint32_t kGmm2CombineControlFifoDepth = kGmm2CombineCvFifoDepth;
constexpr uint32_t kGmm2CombineCvBufferOffset = 0U;
constexpr uint32_t kGmm2CombineMetadataOffset = kGmm2CombineCvBufferOffset + kGmm2CombineCvBufferBytes;
constexpr uint32_t kGmm2CombineMetadataBytes = kGmm2CombineCvFifoDepth * kGmm2CombineMetadataSlotBytes;

struct Gmm2CombineCvPipe {
    struct ProducerEndpoint {
        uint32_t tileIndex = 0U;
        uint32_t controlIndex = 0U;
    };

    ProducerEndpoint prod;
    bool requireInitialFree = false;
};

using Gmm2CombineCvTile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, kGmm2CombineCvTileRows, kGmm2CombineCvTileCols, pto::BLayout::RowMajor,
    pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::NoneBox>;

static_assert(kGmm2CombineCvSlotBytes == 128U * 1024U);
static_assert(kGmm2CombineCvBufferBytes == 128U * 1024U);
static_assert(kGmm2CombineRequiredUbBytes <= A5_MAIN_UB_SIZE);
static_assert(kGmm2CombineCvFreeFlagBase + kGmm2CombineCvFifoDepth <= 16U);
static_assert(kGmm2CombineControlFreeFlagBase + kGmm2CombineControlFifoDepth <= 16U);
static_assert(
    kGmm2CombineControlFreeFlagBase + kGmm2CombineControlFifoDepth - 1U + kMegaMoeFixedSecondAivSubblockFlagOffset <
    32U);
AICORE inline uint16_t Gmm2CombineAiv1Flag(uint16_t logicalFlag)
{
    return static_cast<uint16_t>(logicalFlag + kMegaMoeFixedSecondAivSubblockFlagOffset);
}

AICORE inline void Gmm2CombineProducerAllocate(Gmm2CombineCvPipe& pipe)
{
    if (pipe.requireInitialFree || pipe.prod.tileIndex != 0U) {
        wait_intra_block(PIPE_FIX, Gmm2CombineAiv1Flag(kGmm2CombineCvFreeFlagBase));
        pipe.requireInitialFree = false;
    }
}

AICORE inline void Gmm2CombineConsumerPrime()
{
    // Reused AIV1 UB remains owned by SwiGLU until its final stores drain.
    set_intra_block(PIPE_MTE3, kGmm2CombineCvFreeFlagBase);
}

AICORE inline void Gmm2CombineProducerRecord(Gmm2CombineCvPipe& pipe)
{
    set_intra_block(PIPE_FIX, Gmm2CombineAiv1Flag(kGmm2CombineCvReadyFlagBase));
    ++pipe.prod.tileIndex;
}

AICORE inline void Gmm2CombineControlProducerAllocate(Gmm2CombineCvPipe& pipe)
{
    if (pipe.prod.controlIndex != 0U) {
        wait_intra_block(PIPE_S, Gmm2CombineAiv1Flag(kGmm2CombineControlFreeFlagBase));
    }
}

AICORE inline void Gmm2CombineControlProducerRecord(Gmm2CombineCvPipe& pipe)
{
    set_intra_block(PIPE_S, Gmm2CombineAiv1Flag(kGmm2CombineControlReadyFlagBase));
    ++pipe.prod.controlIndex;
}

AICORE inline void Gmm2CombineControlProducerDrain(Gmm2CombineCvPipe& pipe)
{
    if (pipe.prod.controlIndex != 0U) {
        wait_intra_block(PIPE_S, Gmm2CombineAiv1Flag(kGmm2CombineControlFreeFlagBase));
    }
    pipe.prod.controlIndex = 0U;
}

AICORE inline void Gmm2CombineProducerDrainWave(Gmm2CombineCvPipe& pipe)
{
    if (pipe.prod.tileIndex != 0U) {
        wait_intra_block(PIPE_FIX, Gmm2CombineAiv1Flag(kGmm2CombineCvFreeFlagBase));
    }
    pipe.prod.tileIndex = 0U;
}

AICORE inline void Gmm2CombineControlConsumerWait(Gmm2CombineCvPipe& pipe)
{
    (void)pipe;
    wait_intra_block(PIPE_S, kGmm2CombineControlReadyFlagBase);
}

AICORE inline void Gmm2CombineControlConsumerRelease(Gmm2CombineCvPipe& pipe)
{
    (void)pipe;
    set_intra_block(PIPE_S, kGmm2CombineControlFreeFlagBase);
}

AICORE inline void Gmm2CombineConsumerEnqueueReadyWait(Gmm2CombineCvPipe& pipe)
{
    (void)pipe;
    wait_intra_block(PIPE_MTE3, kGmm2CombineCvReadyFlagBase);
}

AICORE inline void Gmm2CombineConsumerRelease(Gmm2CombineCvPipe& pipe)
{
    (void)pipe;
    set_intra_block(PIPE_MTE3, kGmm2CombineCvFreeFlagBase);
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_COMBINE_CV_PIPE_H
