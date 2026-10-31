/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "op_kernel/utils/hccl_window_context.hpp"

class SharedUrmaWorkspace;

using rtError_t = int32_t;
using rtStream_t = void*;

extern "C" rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);
extern "C" HcclResult HcclAllocComResourceByTiling(
    HcclComm comm, void* stream, void* resourceTiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, uint32_t* topoType, uint32_t isSetDevice);

struct StandaloneHcclContext {
    int rank_id = 0;
    int world_size = 0;
    rtStream_t hccl_stream = nullptr;
    HcclComm comm = nullptr;
    PtoRemoteWindowContext* remote_window_ctx = nullptr;
    PtoRemoteWindowContext host_remote_window_ctx{};
    uint64_t raw_window_bytes = 0;
    uint64_t raw_local_window_in = 0;
    bool owns_remote_window_ctx = false;

    PtoRemoteWindowContext* RemoteWindowContextPtr() const { return remote_window_ctx; }

    uint64_t WindowBytes() const { return host_remote_window_ctx.windowBytes; }

    uint64_t WindowClearBytes() const { return raw_window_bytes; }

    void* WindowClearBase() const { return reinterpret_cast<void*>(raw_local_window_in); }

    void AttachExternalRemoteWindowContext(PtoRemoteWindowContext* remoteWindowCtx);
    void ReleaseRemoteWindowContext();
    void ResetHostRemoteWindowContext();
    void SetHostContextWorkspace(uint64_t workspaceBase, uint64_t workspaceBytes);
    void SetHostRankInfo(uint32_t rank, uint32_t rankCount, uint64_t windowBytes);
    void SetHostWindow(uint32_t rank, uint64_t windowIn, uint64_t windowOut);
    bool LoadHostRemoteWindowContextFromDevice();
    bool CopyHostRemoteWindowContextToDevice();
};

struct StandaloneRankRuntime {
    StandaloneHcclContext hccl;
    aclrtStream compute_stream = nullptr;
    SharedUrmaWorkspace* urma_manager = nullptr;
    void* urma_workspace = nullptr;
    uint32_t urma_jetty_count = 0U;
    bool urma_enabled = false;
    uint32_t rank_num_per_server = 0;
};

bool InitStandaloneRankRuntime(
    StandaloneRankRuntime& runtime, int rank_id, int world_size, int device_id, const HcclRootInfo& root_info);
aclError GetStandaloneUrmaWorkspaceSize(uint32_t ranks, uint32_t submittingAivs, uint64_t* bytes);
bool InitStandaloneUrmaRuntime(
    StandaloneRankRuntime& runtime, uint32_t ranksPerServer, uint32_t submittingAivs, void* workspaceAddr,
    uint64_t workspaceSize);
void DestroyStandaloneRankRuntime(StandaloneRankRuntime& runtime);
