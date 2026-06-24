/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PAGED_ATTENTION_HIGHPERF_ENTRY_HPP
#define PTO_PAGED_ATTENTION_HIGHPERF_ENTRY_HPP

#include "pa_kernel_impl.hpp"

static AICORE __attribute__((always_inline)) void paged_attention_mask_body(
    __gm__ uint8_t *__restrict__ sync,
    uint32_t ptoBlockIdx,
    uint32_t ptoBlockNum,
    uint32_t ptoSubBlockId,
    __gm__ uint8_t *__restrict__ qGm,
    __gm__ uint8_t *__restrict__ kGm,
    __gm__ uint8_t *__restrict__ vGm,
    __gm__ uint8_t *__restrict__ blockTablesGm,
    __gm__ uint8_t *__restrict__ maskGm,
    __gm__ uint8_t *__restrict__ deqScale1Gm,
    __gm__ uint8_t *__restrict__ offset1Gm,
    __gm__ uint8_t *__restrict__ deqScale2Gm,
    __gm__ uint8_t *__restrict__ offset2Gm,
    __gm__ uint8_t *__restrict__ razorOffset,
    __gm__ uint8_t *__restrict__ scaleGm,
    __gm__ uint8_t *__restrict__ logNGm,
    __gm__ uint8_t *__restrict__ eyeGm,
    __gm__ uint8_t *__restrict__ oGm,
    __gm__ uint8_t *__restrict__ sGm,
    __gm__ uint8_t *__restrict__ pGm,
    __gm__ uint8_t *__restrict__ oTmpGm,
    __gm__ uint8_t *__restrict__ goGm,
    __gm__ uint8_t *__restrict__ oCoreTmpGm,
    __gm__ uint8_t *__restrict__ lGm,
    __gm__ uint8_t *__restrict__ gmK16,
    __gm__ uint8_t *__restrict__ gmV16,
    __gm__ uint8_t *__restrict__ tilingParaGm)
{
    (void)maskGm;
    (void)deqScale1Gm;
    (void)offset1Gm;
    (void)deqScale2Gm;
    (void)offset2Gm;
    (void)razorOffset;
    (void)scaleGm;
    (void)logNGm;
    (void)eyeGm;
    (void)sGm;
    (void)pGm;
    (void)oTmpGm;
    (void)goGm;
    (void)gmK16;
    (void)gmV16;

    if (sync != nullptr) {
        set_ffts_base_addr(reinterpret_cast<unsigned long>(sync));
    }
    set_atomic_none();
    set_mask_norm();

#ifdef __DAV_C220_CUBE__
    const int64_t workerIdx = static_cast<int64_t>(ptoBlockIdx);
    const int64_t workerNum = static_cast<int64_t>(ptoBlockNum);
    const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
    if (SupportsPtoPagedAttentionHighPerf(tilingParaGm)) {
        RunPtoPagedAttentionCubePipeline(qGm, kGm, vGm, blockTablesGm, sGm, pGm, oTmpGm, tilingParaGm, workerIdx, workerNum);
    } else if (SupportsPtoPagedAttentionRawSplitKV(tilingParaGm)) {
        RunPtoPagedAttentionCubePipelineSplitKV(qGm, kGm, vGm, blockTablesGm, sGm, pGm, oTmpGm, tilingParaGm,
            workerIdx, workerNum);
    } else if (ctx.kvSplitCoreNum > 1) {
        pipe_barrier(PIPE_ALL);
    } else {
        pipe_barrier(PIPE_ALL);
    }
#elif defined(__DAV_C220_VEC__)
    if (SupportsPtoPagedAttentionHighPerf(tilingParaGm)) {
        const int64_t workerIdx = static_cast<int64_t>(ptoBlockIdx);
        const int64_t workerNum = static_cast<int64_t>(ptoBlockNum);
        RunPtoPagedAttentionVecPipeline(oGm, sGm, pGm, oTmpGm, tilingParaGm, workerIdx, workerNum, ptoSubBlockId);
    } else {
        const int64_t workerIdx = static_cast<int64_t>(ptoBlockIdx) * 2 + static_cast<int64_t>(ptoSubBlockId);
        const int64_t workerNum = static_cast<int64_t>(ptoBlockNum) * 2;
        const PaTilingContext ctx = LoadPaTilingContext(tilingParaGm);
        if (SupportsPtoPagedAttentionRawSplitKV(tilingParaGm)) {
            RunPtoPagedAttentionVecPipelineSplitKV(oGm, sGm, pGm, oTmpGm, oCoreTmpGm, lGm, tilingParaGm,
                static_cast<int64_t>(ptoBlockIdx), static_cast<int64_t>(ptoBlockNum), ptoSubBlockId);
        } else if (ctx.kvSplitCoreNum > 1) {
            RunPtoPagedAttentionDecodeSplitKV(qGm, kGm, vGm, blockTablesGm, oGm, oCoreTmpGm, lGm, tilingParaGm,
                static_cast<int64_t>(ptoBlockIdx), static_cast<int64_t>(ptoBlockNum), ptoSubBlockId);
        } else {
            RunPtoPagedAttentionDecode(qGm, kGm, vGm, blockTablesGm, oGm, tilingParaGm, workerIdx, workerNum);
        }
    }
#else
    pipe_barrier(PIPE_ALL);
#endif
}

#ifndef PTO_PA_NO_GLOBAL_ENTRY
extern "C" __global__ AICORE void paged_attention_mask(
    __gm__ uint8_t *__restrict__ sync,
    __gm__ uint8_t *__restrict__ qGm,
    __gm__ uint8_t *__restrict__ kGm,
    __gm__ uint8_t *__restrict__ vGm,
    __gm__ uint8_t *__restrict__ blockTablesGm,
    __gm__ uint8_t *__restrict__ maskGm,
    __gm__ uint8_t *__restrict__ deqScale1Gm,
    __gm__ uint8_t *__restrict__ offset1Gm,
    __gm__ uint8_t *__restrict__ deqScale2Gm,
    __gm__ uint8_t *__restrict__ offset2Gm,
    __gm__ uint8_t *__restrict__ razorOffset,
    __gm__ uint8_t *__restrict__ scaleGm,
    __gm__ uint8_t *__restrict__ logNGm,
    __gm__ uint8_t *__restrict__ eyeGm,
    __gm__ uint8_t *__restrict__ oGm,
    __gm__ uint8_t *__restrict__ sGm,
    __gm__ uint8_t *__restrict__ pGm,
    __gm__ uint8_t *__restrict__ oTmpGm,
    __gm__ uint8_t *__restrict__ goGm,
    __gm__ uint8_t *__restrict__ oCoreTmpGm,
    __gm__ uint8_t *__restrict__ lGm,
    __gm__ uint8_t *__restrict__ gmK16,
    __gm__ uint8_t *__restrict__ gmV16,
    __gm__ uint8_t *__restrict__ tilingParaGm)
{
    const uint32_t ptoBlockIdx = static_cast<uint32_t>(get_block_idx());
    const uint32_t ptoBlockNum = static_cast<uint32_t>(get_block_num());
#ifdef __DAV_C220_VEC__
    const uint32_t ptoSubBlockId = static_cast<uint32_t>(get_subblockid());
#else
    const uint32_t ptoSubBlockId = 0;
#endif

    paged_attention_mask_body(
        sync, ptoBlockIdx, ptoBlockNum, ptoSubBlockId, qGm, kGm, vGm, blockTablesGm, maskGm, deqScale1Gm, offset1Gm,
        deqScale2Gm, offset2Gm, razorOffset, scaleGm, logNGm, eyeGm, oGm, sGm, pGm, oTmpGm, goGm, oCoreTmpGm,
        lGm, gmK16, gmV16, tilingParaGm);
}
#endif

#endif
