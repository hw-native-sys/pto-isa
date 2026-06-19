/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <pto/pto-inst.hpp>
#include "runtime/rt.h"

#include "pa_entry.hpp"

extern "C" void call_kernel(
    void *stream,
    uint8_t *qGm,
    uint8_t *kGm,
    uint8_t *vGm,
    uint8_t *blockTablesGm,
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
    uint8_t *nullGm,
    uint32_t blockDim)
{
    uint64_t ffts = 0;
    uint32_t fftsLen = 0;
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    paged_attention_mask<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<__gm__ uint8_t *>(ffts),
        qGm,
        kGm,
        vGm,
        blockTablesGm,
        nullGm,
        nullGm,
        nullGm,
        nullGm,
        nullGm,
        nullGm,
        nullGm,
        nullGm,
        nullGm,
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
