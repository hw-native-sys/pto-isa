/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_COMMON_H_
#define MOE_COMBINE_COMMON_H_

#include <cstdint>

namespace moe_combine {

constexpr uint32_t kMaxMoeCombineRanks = 64;
constexpr uint32_t kMoeCombineTileCols = 1024;
constexpr uint32_t kMoeCombineRowChunk = 8;
constexpr uint32_t kMoeCombineMetadataPad = 16;
constexpr uint32_t kMoeCombineSignalValue = 1;

// Shared host/kernel ABI. Keep these struct and field names aligned between
// host layout calculation and device-side views.
struct MoeCombineShape {
    uint32_t ep;
    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t expertPerRank;
    uint32_t expertNum;
    uint32_t maxOutputSize;
    uint32_t aivBlocks;
};

struct WorkspaceLayout {
    uint64_t localSync;
    uint64_t totalBytes;
};

struct CombineRouteMetaLayout {
    uint64_t peerTokenPerExpert;
    uint64_t expandedRowIdx;
    uint64_t cumsumPerExpert;
    uint64_t dispatchOffset;
    uint64_t prevSumBeforeRank;
    uint64_t totalBytes;
};

struct PeerWindowLayout {
    uint64_t ptrD;
    uint64_t countReadySignal;
    uint64_t combineDoneSignal;
    uint64_t totalBytes;
};

struct HcclDeviceContext {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[kMaxMoeCombineRanks];
    uint64_t windowsOut[kMaxMoeCombineRanks];
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};

struct MoeCombineRuntimeConfig {
    uint32_t runMode;
    uint32_t socVersion;
    uint32_t deviceBase;
    uint32_t ndevices;
    uint32_t rankFromMpi;
    uint32_t rank;
    uint32_t nranks;
    uint32_t debug;
    uint32_t iters;
    uint32_t warmup;
    uint32_t seed;
    uint32_t genData;
    uint32_t verify;
    uint32_t skipRun;
    uint32_t skipBuild;
    uint32_t cleanBuild;
    uint32_t skipKernels;
    uint32_t hostGoldenOnly;
    uint32_t combineReturnOnly;
    uint32_t keepHcclShm;
    uint64_t hcclBuffSizeMb;
};

} // namespace moe_combine

#endif // MOE_COMBINE_COMMON_H_
