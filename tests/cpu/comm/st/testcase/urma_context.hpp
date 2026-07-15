/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "pto/comm/comm_types.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "pto/common/cpu_stub.hpp"

class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    void Finalize() {}

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

    void* urmaInfoDevice_{nullptr};
};

struct UrmaTestContext {
    int deviceId{-1};
    void* devBuf{nullptr};
    UrmaWorkspaceManager urmaMgr;

    bool AllocHugePageBuffer(size_t commBytesNeeded) { return true; }

    bool Setup(int rank_id, int n_ranks, int n_devices, int first_device_id, int root_rank, size_t commBytesNeeded)
    {
        if (commBytesNeeded > 0) {
            devBuf = malloc(commBytesNeeded);
            urmaMgr.urmaInfoDevice_ = devBuf;
            return true;
        }
        return false;
    }

    void Cleanup()
    {
        urmaMgr.Finalize();
        if (devBuf) {
            aclrtFree(devBuf);
            devBuf = nullptr;
        }
    }
};

AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t* urmaWorkspace, uint32_t peerRank)
{
    return reinterpret_cast<uint64_t>(urmaWorkspace);
}

template <pto::comm::DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, uint32_t destRankId, pto::comm::AsyncSession& session)
{
    static_assert(engine == pto::comm::DmaEngine::URMA, "This overload is for URMA only");
    return true;
}

using UrmaKernelFn = bool (*)(int, int, int, int, int, int);

inline bool RunUrmaTestMpiLaunch(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, UrmaKernelFn kernelFn)
{
    int root = 0;
    bool res = false;
    for (size_t i = 0; i < n_ranks; i++) {
        res = res || kernelFn(i, n_ranks, n_devices, first_device_id, first_rank_id, root);
    }
    return res;
}
