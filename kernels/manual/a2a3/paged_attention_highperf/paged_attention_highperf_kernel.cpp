/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the \"License\").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "pa_entry.hpp"

#ifndef __COSTMODEL
void LaunchPagedAttentionHighPerf(
    uint8_t *sync,
    uint8_t *qGm,
    uint8_t *kGm,
    uint8_t *vGm,
    uint8_t *blockTablesGm,
    uint8_t *maskGm,
    uint8_t *deqScale1Gm,
    uint8_t *offset1Gm,
    uint8_t *deqScale2Gm,
    uint8_t *offset2Gm,
    uint8_t *razorOffset,
    uint8_t *scaleGm,
    uint8_t *logNGm,
    uint8_t *eyeGm,
    uint8_t *oGm,
    uint8_t *sGm,
    uint8_t *pGm,
    uint8_t *oTmpGm,
    uint8_t *goGm,
    uint8_t *oCoreTmpGm,
    uint8_t *lGm,
    uint8_t *gmK16,
    uint8_t *gmV16,
    uint8_t *tilingParaGm,
    uint32_t blockDim,
    void *stream)
{
    paged_attention_mask<<<blockDim, nullptr, stream>>>(
        sync,
        qGm,
        kGm,
        vGm,
        blockTablesGm,
        maskGm,
        deqScale1Gm,
        offset1Gm,
        deqScale2Gm,
        offset2Gm,
        razorOffset,
        scaleGm,
        logNGm,
        eyeGm,
        oGm,
        sGm,
        pGm,
        oTmpGm,
        goGm,
        oCoreTmpGm,
        lGm,
        gmK16,
        gmV16,
        tilingParaGm);
}
#endif
