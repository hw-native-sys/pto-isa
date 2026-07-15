/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_CMO_INTRIN_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_CMO_INTRIN_HPP

#include "pto/comm/async/sdma/sdma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace sdma {

namespace detail {

// ============================================================================
// CMO (Cache Maintenance Operation) Prefetch via SDMA
// ============================================================================

constexpr uint32_t kCmoPrefetchOpcode = 6U;

// Templated to defer codegen until the public __sdma_cmo_prefetch<T>() is
// instantiated. Without this wrapper the function body is parsed and lowered
// in every TU that transitively includes this header (via pto-inst.hpp ->
// pto_instr_impl.hpp -> TPrefetchAsync.hpp), inflating IR for unrelated
// compute kernels and tripping a Bisheng optimizer ICE in
// AnalysisManager::getResultImpl.
template <typename = void>
PTO_INTERNAL void AddOneCmoSqe(
    __gm__ BatchWriteChannelInfo* channelInfo, __gm__ uint8_t* src, uint32_t length, uint32_t sqTail, uint32_t taskId)
{
    __gm__ BatchWriteItem* sqe = (__gm__ BatchWriteItem*)(channelInfo->sq_base);
    sqe += (sqTail % channelInfo->sq_depth);

#ifdef PTO_NPU_ARCH_A5
    sqe->type = RT_STARS_SQE_TYPE_SDMA;
    sqe->wrCqe = 1;
    sqe->numBlocks = 0;
    sqe->rtStreamId = channelInfo->stream_id;
    sqe->taskId = taskId;
    sqe->kernelCredit = K_CREDIT_TIME_DEFAULT;
    sqe->opcode = kCmoPrefetchOpcode;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->lengthMove = length;

    uint64_t srcAddr = reinterpret_cast<uint64_t>(src);
    sqe->srcAddrLow = static_cast<uint32_t>(srcAddr & 0xFFFFFFFF);
    sqe->srcAddrHigh = static_cast<uint32_t>((srcAddr >> 32) & 0xFFFFFFFF);
    sqe->dstAddrLow = 0U;
    sqe->dstAddrHigh = 0U;
#else
    sqe->type = RT_STARS_SQE_TYPE_SDMA;
    sqe->blockDim = 0;
    sqe->rtStreamId = channelInfo->stream_id;
    sqe->taskId = taskId;
    sqe->kernel_credit = K_CREDIT_TIME_DEFAULT;
    sqe->ptr_mode = 0;
    sqe->opcode = kCmoPrefetchOpcode;
    sqe->ie2 = 0;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6;
    sqe->partid = 63U;
    sqe->mpam = 0;
    sqe->length = length;

    uint64_t srcAddr = reinterpret_cast<uint64_t>(src);
    sqe->srcAddrLow = static_cast<uint32_t>(srcAddr & 0xFFFFFFFF);
    sqe->srcAddrHigh = static_cast<uint32_t>((srcAddr >> 32) & 0xFFFFFFFF);
    sqe->dstAddrLow = 0U;
    sqe->dstAddrHigh = 0U;
    sqe->linkType = 0;
#endif

    pipe_barrier(PIPE_ALL);
}

template <typename = void>
PTO_INTERNAL void SubmitCmoPrefetchSqes(
    __gm__ BatchWriteChannelInfo* batchWriteChannelInfo, __gm__ uint8_t* src, const SdmaConfig& config,
    uint32_t* sqTail, uint32_t sqTailLen)
{
    for (uint32_t idx = 0U; idx < config.iter_num; ++idx) {
        uint32_t queueIdx = idx % config.queue_num;
        __gm__ BatchWriteChannelInfo* channelInfo = batchWriteChannelInfo + queueIdx;

        uint32_t transferBytes = config.block_bytes;
        if (idx == config.iter_num - 1) {
            transferBytes = config.per_core_bytes - idx * config.block_bytes;
        }

        __gm__ uint8_t* srcAddr = src + config.comm_block_offset + idx * config.block_bytes;

        AddOneCmoSqe(channelInfo, srcAddr, transferBytes, sqTail[queueIdx], sqTail[queueIdx] - channelInfo->sq_head);

        sqTail[queueIdx] = (sqTail[queueIdx] + 1) % kSqDepth;
        pipe_barrier(PIPE_ALL);
    }
}

template <typename = void>
PTO_INTERNAL uint64_t SdmaCmoPrefetch(__gm__ uint8_t* src, uint64_t messageLen, const SdmaExecContext& execCtx)
{
    __gm__ uint8_t* contextGm = execCtx.contextGm;
    if (contextGm == nullptr || !IsValidTmpBuffer(execCtx.tmpBuf)) {
        return 0;
    }

    const uint32_t syncId = execCtx.syncId;
    const uint32_t channelGroupIndex = execCtx.channelGroupIdx;
    UbTmpBuf tmpBuf = execCtx.tmpBuf;

    SdmaConfig config;
    if (!BuildTransferConfig(execCtx.baseConfig, messageLen, config)) {
        pipe_barrier(PIPE_ALL);
        return 0;
    }
    if (config.iter_num == 0) {
        return 0;
    }
    if (channelGroupIndex >= (kSdmaMaxChannel / config.queue_num)) {
        return 0;
    }
    const uint32_t sqePerQue = (config.iter_num + config.queue_num - 1) / config.queue_num + 1;
    if (sqePerQue > kSqDepth) {
        return 0;
    }

    __gm__ BatchWriteChannelInfo* batchWriteChannelBase =
        (__gm__ BatchWriteChannelInfo*)(contextGm + sizeof(BatchWriteFlagInfo));
    __gm__ BatchWriteChannelInfo* batchWriteChannelInfo = batchWriteChannelBase + channelGroupIndex * config.queue_num;

    uint32_t sqTail[64] = {0};
    InitSqTailArray(batchWriteChannelInfo, config.queue_num, sqTail, 64, tmpBuf);

    SubmitCmoPrefetchSqes(batchWriteChannelInfo, src, config, sqTail, 64);

    FlushCacheAndRingDoorbell(batchWriteChannelInfo, config, sqTail, tmpBuf, syncId);
    UpdateSqTailState(batchWriteChannelInfo, config, sqTail, tmpBuf, syncId);

    pipe_barrier(PIPE_ALL);
    return reinterpret_cast<uint64_t>(contextGm);
}

} // namespace detail

// ============================================================================
// Public CMO prefetch intrinsic
// ============================================================================
template <typename T>
PTO_INTERNAL uint64_t __sdma_cmo_prefetch(__gm__ T* src, uint64_t prefetch_size, const SdmaExecContext& execCtx)
{
    if (prefetch_size == 0) {
        return 0;
    }
    return detail::SdmaCmoPrefetch((__gm__ uint8_t*)src, prefetch_size, execCtx);
}

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_CMO_INTRIN_HPP
