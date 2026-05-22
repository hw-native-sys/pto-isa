/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "../../../../../include/pto/cpu/TPartArgKernelCommon.hpp"
using namespace pto;

struct PartArgMaxRunner {
    template <typename TileDataDstVal, typename TileDataDstIdx, typename TileDataSrc0Val, typename TileDataSrc0Idx,
              typename TileDataSrc1Val, typename TileDataSrc1Idx>
    PTO_INTERNAL static void Run(TileDataDstVal &dstVal, TileDataDstIdx &dstIdx, TileDataSrc0Val &src0Val,
                                 TileDataSrc0Idx &src0Idx, TileDataSrc1Val &src1Val, TileDataSrc1Idx &src1Idx)
    {
        TPARTARGMAX(dstVal, dstIdx, src0Val, src0Idx, src1Val, src1Idx);
    }
};

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
AICORE void runTPARTARGMAX(__gm__ float *__out__ outVal, __gm__ float *__in__ src0Val, __gm__ float *__in__ src1Val,
                           __gm__ uint32_t *__out__ outIdx, __gm__ uint32_t *__in__ src0Idx,
                           __gm__ uint32_t *__in__ src1Idx)
{
    RunPartArgKernel<kRows, kCols, kValidRows1, kValidCols1, PartArgMaxRunner>(outVal, src0Val, src1Val, outIdx,
                                                                               src0Idx, src1Idx);
}

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTARGMAX(float *outVal, float *src0Val, float *src1Val, uint32_t *outIdx, uint32_t *src0Idx,
                       uint32_t *src1Idx, void *stream)
{
    (void)stream;
    runTPARTARGMAX<kRows, kCols, kValidRows1, kValidCols1>(outVal, src0Val, src1Val, outIdx, src0Idx, src1Idx);
}

template void LaunchTPARTARGMAX<64, 64, 32, 32>(float *, float *, float *, uint32_t *, uint32_t *, uint32_t *, void *);
