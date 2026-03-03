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
#include <iostream>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "hccl_context.h"

#if __has_include("hccl/hcom.h")
#include "hccl/hcom.h"
#endif

#if __has_include("hccl/hccl_comm.h")
#include "hccl/hccl_comm.h"
#endif

#if !__has_include("hccl/hcom.h")
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
#endif

// Mc2 tiling structures passed to HcclAllocComResourceByTiling.
// Binary layout must match the HCCL internal expectation.
#pragma pack(push, 8)
struct Mc2ServerCfg {
    uint32_t version = 0;
    uint8_t debugMode = 0;
    uint8_t sendArgIndex = 0;
    uint8_t recvArgIndex = 0;
    uint8_t commOutArgIndex = 0;
    uint8_t reserved[8] = {};
};
#pragma pack(pop)

#pragma pack(push, 8)
struct Mc2HcommCfg {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    char reserved[13] = {};
    char groupName[128] = {};
    char algConfig[128] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};
#pragma pack(pop)

struct Mc2CommConfig {
    uint32_t version;
    uint32_t hcommCnt;
    Mc2ServerCfg serverCfg;
    Mc2HcommCfg hcommCfg;
};

// ============================================================================
// Device-side helper: convert a local window pointer to the equivalent address
// on a remote rank.
// ============================================================================
template <typename T>
AICORE inline __gm__ T *HcclRemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

// ============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = reinterpret_cast<void *>(windowBase + offset);
    offset += bytes;
    return ptr;
}

// ============================================================================
// TestContext: ACL + HCCL initialization / teardown helper.
// ============================================================================
struct TestContext {
    int32_t deviceId{-1};
    aclrtStream stream{nullptr};
    int aclStatus{0};
    HcclComm comm{nullptr};

    HcclDeviceContext *deviceCtx{nullptr};
    HcclDeviceContext hostCtx{};

    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo *rootInfo)
    {
        if (nDevices <= 0 || nRanks <= 0) {
            std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
            return false;
        }
        deviceId = rankId % nDevices + firstDeviceId;

        aclInit(nullptr);
        aclStatus = aclrtSetDevice(deviceId);
        if (aclStatus != 0) {
            std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed: " << aclStatus << "\n";
            return false;
        }
        aclStatus = aclrtCreateStream(&stream);
        if (aclStatus != 0) {
            std::cerr << "[ERROR] aclrtCreateStream failed: " << aclStatus << "\n";
            return false;
        }

        HcclResult hret =
            HcclCommInitRootInfo(static_cast<uint32_t>(nRanks), rootInfo, static_cast<uint32_t>(rankId), &comm);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclCommInitRootInfo failed: " << hret << std::endl;
            return false;
        }

        char group[128] = {};
        hret = HcclGetCommName(comm, group);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetCommName failed: " << hret << std::endl;
            return false;
        }

        HcclComm commHandle = nullptr;
        hret = HcomGetCommHandleByGroup(group, &commHandle);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcomGetCommHandleByGroup failed: " << hret << std::endl;
            return false;
        }

        Mc2CommConfig tilingCfg{};
        tilingCfg.version = 2;
        tilingCfg.hcommCnt = 1;
        tilingCfg.hcommCfg.opType = 6; // AllToAll
        std::strncpy(tilingCfg.hcommCfg.groupName, group, sizeof(tilingCfg.hcommCfg.groupName) - 1);
        std::strncpy(tilingCfg.hcommCfg.algConfig, "AllGather=level0:ring", sizeof(tilingCfg.hcommCfg.algConfig) - 1);

        void *ctxPtr = nullptr;
        hret = HcclAllocComResourceByTiling(commHandle, stream, &tilingCfg, &ctxPtr);
        if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
            std::cerr << "[ERROR] HcclAllocComResourceByTiling failed: " << hret << std::endl;
            return false;
        }
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);

        aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);

        std::cout << "[INFO] Rank " << rankId << " hccl init OK, winSize=" << hostCtx.winSize
                  << " windowsIn[self]=" << std::hex << hostCtx.windowsIn[rankId] << std::dec << std::endl;
        return true;
    }

    bool Finalize()
    {
        if (comm != nullptr) {
            HcclCommDestroy(comm);
            comm = nullptr;
        }
        aclStatus |= aclrtDestroyStream(stream);
        aclStatus |= aclrtResetDevice(deviceId);
        aclStatus |= aclFinalize();
        return (aclStatus == 0);
    }
};

// ============================================================================
// ForkAndRunWithHcclRootInfo: Fork children, rank-0 child generates
// HcclRootInfo and shares it with siblings via mmap'd shared memory.
// ============================================================================
template <typename Func>
inline bool ForkAndRunWithHcclRootInfo(int nRanks, int firstRankId, int firstDeviceId, Func &&perRankFn)
{
    constexpr int kBootstrapPollUs = 1000;
    constexpr int kBootstrapTimeoutMs = 30000;

    // Query available device count before forking; skip early if insufficient.
    aclInit(nullptr);
    uint32_t deviceCount = 0;
    aclrtGetDeviceCount(&deviceCount);
    int maxDeviceId = firstDeviceId + (nRanks > 0 ? (nRanks - 1) : 0);
    if (static_cast<uint32_t>(maxDeviceId) >= deviceCount) {
        std::cerr << "[SKIP] Need devices [" << firstDeviceId << ".." << maxDeviceId << "] but only " << deviceCount
                  << " available, skipping.\n";
        return true;
    }

    struct SharedBootstrap {
        HcclRootInfo rootInfo;
        // 0: initializing, 1: ready, -1: failed
        volatile int state;
    };

    SharedBootstrap *shared = static_cast<SharedBootstrap *>(
        mmap(nullptr, sizeof(SharedBootstrap), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (shared == MAP_FAILED) {
        std::cerr << "[ERROR] mmap for SharedBootstrap failed" << std::endl;
        return false;
    }
    shared->state = 0;

    std::vector<pid_t> pids;
    for (int r = 0; r < nRanks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            int rankId = firstRankId + r;

            if (r == 0) {
                aclInit(nullptr);
                aclrtSetDevice(firstDeviceId);

                HcclResult hret = HcclGetRootInfo(&shared->rootInfo);
                if (hret != HCCL_SUCCESS) {
                    __sync_synchronize();
                    shared->state = -1;
                    __sync_synchronize();
                    std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
                    _exit(1);
                }

                __sync_synchronize();
                shared->state = 1;
                __sync_synchronize();
            } else {
                int waitUs = 0;
                while (shared->state == 0 && waitUs < (kBootstrapTimeoutMs * 1000)) {
                    usleep(kBootstrapPollUs);
                    waitUs += kBootstrapPollUs;
                }
                __sync_synchronize();

                if (shared->state != 1) {
                    std::cerr << "[ERROR] bootstrap rootInfo wait failed, state=" << shared->state
                              << ", waited_us=" << waitUs << std::endl;
                    _exit(1);
                }
            }

            const bool ok = perRankFn(rankId, &shared->rootInfo);
            _exit(ok ? 0 : 1);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "[ERROR] fork() failed for rank " << r << std::endl;
            munmap(shared, sizeof(SharedBootstrap));
            return false;
        }
    }

    bool success = true;
    for (pid_t p : pids) {
        int status = 0;
        waitpid(p, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            success = false;
        }
    }
    munmap(shared, sizeof(SharedBootstrap));
    return success;
}
