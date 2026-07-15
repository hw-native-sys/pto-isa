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
#include <cstdlib>
#include <cstring>

#include "pto/common/cpu_stub.hpp"

template <typename T>
inline T* CommRemotePtr(CommDeviceContext* ctx, T* localPtr, int pe)
{
    if (ctx->rankId == pe) {
        return localPtr;
    }
    int memberSize = ctx->winSize / sizeof(T);
    T* buffer = new T[memberSize];
    return buffer;
}

inline void* WindowAlloc(uint64_t windowBase, size_t& offset, size_t bytes)
{
    void* ptr = std::malloc(bytes);
    return ptr;
}

inline void CommMpiBarrier() {}

template <typename T, size_t count>
struct TestContext {
    int32_t deviceId{-1};
    aclrtStream stream{nullptr};
    int aclStatus{0};
    CommDeviceContext* deviceCtx{nullptr};
    CommDeviceContext hostCtx{};

    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo)
    {
        if (nDevices <= 0 || nRanks <= 0) {
            std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
            return false;
        }

        size_t bytesPerRank = count * sizeof(T);
        hostCtx.rankId = rankId;
        hostCtx.rankNum = nRanks;
        hostCtx.winSize = bytesPerRank;
        void* base = std::malloc(nRanks * bytesPerRank);

        for (uint32_t i = 0; i < HCCL_MAX_RANK_NUM; ++i) {
            if (i < static_cast<uint32_t>(nRanks)) {
                uint64_t baseAddress = reinterpret_cast<uintptr_t>(base);
                hostCtx.windowsIn[i] = baseAddress + i * bytesPerRank;
            } else {
                hostCtx.windowsIn[i] = 0;
            }
        }

        deviceCtx = &hostCtx;

        this->deviceId = rankId % nDevices + firstDeviceId;

        return true;
    }

    bool Finalize() { return true; }
};

template <typename Func>
inline bool ForkAndRunWithHcclRootInfo(int nRanks, int firstRankId, int firstDeviceId, Func&& perRankFn)
{
    if (nRanks <= 0) {
        return false;
    }

    HcclRootInfo rootInfo{};
    bool res = false;
    for (size_t i = 0; i < nRanks; i++) {
        res = res || perRankFn(i, &rootInfo);
    }

    return res;
}