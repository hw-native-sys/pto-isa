/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM1_SWIGLU_CV_PIPE_H
#define DISPATCH_MEGA_COMBINE_GMM1_SWIGLU_CV_PIPE_H

#include <pto/pto-inst.hpp>

#include "utils/const_args.hpp"

constexpr uint16_t kGmm1SwigluCvReadyFlag = 7U;
constexpr uint16_t kGmm1SwigluCvFreeFlag = 8U;
constexpr uint16_t kGmm1SwigluControlReadyFlagBase = 9U;
constexpr uint16_t kGmm1SwigluControlFreeFlagBase = 11U;
constexpr uint32_t kGmm1SwigluCvTileRows = 256U;
constexpr uint32_t kGmm1SwigluCvOutputCols = 256U;
constexpr uint32_t kGmm1SwigluCvTileCols = 2U * kGmm1SwigluCvOutputCols;
constexpr uint32_t kGmm1SwigluCvFifoDepth = 1U;
constexpr uint32_t kGmm1SwigluControlFifoDepth = kGmm1SwigluCvFifoDepth + 1U;
constexpr uint32_t kGmm1SwigluCvHalfSlotBytes = kGmm1SwigluCvTileRows * kGmm1SwigluCvOutputCols * sizeof(bfloat16_t);
constexpr uint32_t kGmm1SwigluCvXOffset = 0U;
constexpr uint32_t kGmm1SwigluCvGateOffset = kGmm1SwigluCvHalfSlotBytes;
constexpr uint32_t kGmm1SwigluCvSlotBytes = kGmm1SwigluCvTileRows * kGmm1SwigluCvTileCols * sizeof(bfloat16_t);
constexpr uint32_t kGmm1SwigluCvBufferOffset = 0U;
constexpr uint32_t kGmm1SwigluCvBufferBytes = kGmm1SwigluCvFifoDepth * kGmm1SwigluCvSlotBytes;

struct Gmm1SwigluCvPipe {
    struct ProducerEndpoint {
        uint32_t tileIndex = 0U;
        uint32_t controlIndex = 0U;
    };
    struct ConsumerEndpoint {
        uint32_t controlIndex = 0U;
    };

    ProducerEndpoint prod;
    ConsumerEndpoint cons;
    bool producerRequiresInitialFree = false;
    bool producerInitialFreeConsumed = false;
};

using Gmm1SwigluCvHalfTile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, kGmm1SwigluCvTileRows, kGmm1SwigluCvOutputCols, pto::BLayout::RowMajor,
    pto::DYNAMIC, pto::DYNAMIC, pto::SLayout::NoneBox>;

static_assert(2U * kGmm1SwigluCvHalfSlotBytes == kGmm1SwigluCvSlotBytes);
static_assert(kGmm1SwigluCvHalfSlotBytes == 128U * 1024U);
static_assert(kGmm1SwigluCvSlotBytes == AtlasA5::UB_SIZE);
static_assert(
    kGmm1SwigluControlFreeFlagBase + kGmm1SwigluControlFifoDepth - 1U + kMegaMoeFixedSecondAivSubblockFlagOffset < 32U);

AICORE inline uint16_t Gmm1SwigluAiv1Flag(uint16_t logicalFlag)
{
    return static_cast<uint16_t>(logicalFlag + kMegaMoeFixedSecondAivSubblockFlagOffset);
}

AICORE inline uint16_t Gmm1SwigluControlReadyFlag(uint32_t slot)
{
    return static_cast<uint16_t>(kGmm1SwigluControlReadyFlagBase + slot);
}

AICORE inline uint16_t Gmm1SwigluControlFreeFlag(uint32_t slot)
{
    return static_cast<uint16_t>(kGmm1SwigluControlFreeFlagBase + slot);
}

AICORE inline void Gmm1SwigluProducerAllocate(Gmm1SwigluCvPipe& pipe)
{
    const bool waitInitialFree = pipe.producerRequiresInitialFree && !pipe.producerInitialFreeConsumed;
    if (waitInitialFree || pipe.prod.tileIndex >= kGmm1SwigluCvFifoDepth) {
        wait_intra_block(PIPE_FIX, Gmm1SwigluAiv1Flag(kGmm1SwigluCvFreeFlag));
        if (waitInitialFree) {
            pipe.producerInitialFreeConsumed = true;
        }
    }
}

AICORE inline void Gmm1SwigluControlProducerAllocate(Gmm1SwigluCvPipe& pipe)
{
    if (pipe.prod.controlIndex >= kGmm1SwigluControlFifoDepth) {
        const uint32_t slot = pipe.prod.controlIndex % kGmm1SwigluControlFifoDepth;
        wait_intra_block(PIPE_S, Gmm1SwigluAiv1Flag(Gmm1SwigluControlFreeFlag(slot)));
    }
}

AICORE inline void Gmm1SwigluProducerRecord(Gmm1SwigluCvPipe& pipe)
{
    set_intra_block(PIPE_FIX, Gmm1SwigluAiv1Flag(kGmm1SwigluCvReadyFlag));
    ++pipe.prod.tileIndex;
}

AICORE inline void Gmm1SwigluControlProducerRecord(Gmm1SwigluCvPipe& pipe)
{
    const uint32_t slot = pipe.prod.controlIndex % kGmm1SwigluControlFifoDepth;
    set_intra_block(PIPE_S, Gmm1SwigluAiv1Flag(Gmm1SwigluControlReadyFlag(slot)));
    ++pipe.prod.controlIndex;
}

AICORE inline void Gmm1SwigluControlConsumerWait(Gmm1SwigluCvPipe& pipe)
{
    const uint32_t slot = pipe.cons.controlIndex % kGmm1SwigluControlFifoDepth;
    wait_intra_block(PIPE_S, Gmm1SwigluControlReadyFlag(slot));
}

AICORE inline void Gmm1SwigluPayloadConsumerWait(Gmm1SwigluCvPipe& pipe)
{
    (void)pipe;
    wait_intra_block(PIPE_V, kGmm1SwigluCvReadyFlag);
}

AICORE inline void Gmm1SwigluConsumerWait(Gmm1SwigluCvPipe& pipe) { Gmm1SwigluPayloadConsumerWait(pipe); }

AICORE inline void Gmm1SwigluControlConsumerRelease(Gmm1SwigluCvPipe& pipe)
{
    const uint32_t slot = pipe.cons.controlIndex % kGmm1SwigluControlFifoDepth;
    set_intra_block(PIPE_S, Gmm1SwigluControlFreeFlag(slot));
    ++pipe.cons.controlIndex;
}

AICORE inline void Gmm1SwigluConsumerPrime() { set_intra_block(PIPE_V, kGmm1SwigluCvFreeFlag); }

AICORE inline void Gmm1SwigluConsumerRelease(Gmm1SwigluCvPipe& pipe)
{
    (void)pipe;
    set_intra_block(PIPE_V, kGmm1SwigluCvFreeFlag);
}

#endif // DISPATCH_MEGA_COMBINE_GMM1_SWIGLU_CV_PIPE_H
