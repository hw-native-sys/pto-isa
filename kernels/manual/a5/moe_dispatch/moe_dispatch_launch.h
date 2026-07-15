/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

bool LaunchMoeDispatchK128(
    int32_t blockNum, void* stream, void* gmA, void* gmPerTokenScale, void* cumsumMM, void* tokenPerExpert,
    void* preSumBeforeRank, void* shmemBase, void* hcclCtx, void* syncWorkspace, int32_t EP, int32_t expertPerRank,
    int32_t maxOutputSize, int64_t offsetA);

bool LaunchMoeDispatchViaGM_K128(
    int32_t blockNum, void* stream, void* gmA, void* gmPerTokenScale, void* tempGmBuffer, void* cumsumMM,
    void* tokenPerExpert, void* preSumBeforeRank, void* shmemBase, void* hcclCtx, void* syncWorkspace, int32_t EP,
    int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA);

bool LaunchMoeDispatchWithSync_K128(
    int32_t blockNum, void* stream, void* gmA, void* gmPerTokenScale, void* shmemBase, void* hcclCtx, void* workspace,
    void* syncGmWorkspace, int32_t EP, int32_t expertPerRank, int32_t maxOutputSize, int64_t offsetA,
    int64_t offsetTPE);
