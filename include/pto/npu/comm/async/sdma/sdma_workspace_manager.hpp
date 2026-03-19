/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_WORKSPACE_MANAGER_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "sdma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

#include <dlfcn.h>

#include "acl/acl.h"

// aclnn tensor API (from aclnn/acl_meta.h, linked via libnnopbase).
// Forward-declared to avoid pulling in aclnn headers that may conflict.
struct aclTensor;
struct aclOpExecutor;
extern "C" aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                      const int64_t *stride, int64_t offset, aclFormat format,
                                      const int64_t *storageDims, uint64_t storageDimsNum, void *tensorData);
extern "C" int32_t aclDestroyTensor(const aclTensor *tensor);

namespace pto {
namespace comm {
namespace sdma {

// ============================================================================
// SdmaWorkspaceManager: Host-side SDMA workspace initialization.
//
// Mirrors the shmem SdmaTransportManager::OpenDevice() flow:
//   1. Creates MC2 STARS streams via the runtime API
//   2. Allocates device workspace memory
//   3. Launches an AICPU STARS-query kernel to populate the workspace
//      with hardware SQ addresses, register bases, and queue depths
//
// The resulting workspace pointer is passed to the AICORE kernel and
// forwarded to comm::BuildAsyncSession() / BuildSdmaSession().
// ============================================================================

namespace detail {

constexpr uint32_t kSdmaMaxChan = 40;
constexpr size_t kSdmaWorkspaceBytes = 16 * 1024;

enum RtStreamCreateAttrId
{
    RT_STREAM_CREATE_ATTR_FLAGS = 1,
    RT_STREAM_CREATE_ATTR_PRIORITY = 2,
};

union RtStreamCreateAttrValue {
    uint32_t flags;
    uint32_t priority;
    uint32_t rsv[4];
};

struct RtStreamCreateAttr {
    RtStreamCreateAttrId id;
    RtStreamCreateAttrValue value;
};

struct RtStreamCreateConfig {
    RtStreamCreateAttr *attrs;
    size_t numAttrs;
};

struct HostStreamInfo {
    uint64_t stream_;
    uint64_t ctx_;
    int32_t stream_id;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    int32_t dev_id;
    uint8_t reserved[20];
};
static_assert(sizeof(HostStreamInfo) == 64, "HostStreamInfo must be 64 bytes");

struct SdmaOpResInfo {
    HostStreamInfo streams[kSdmaMaxChan];
    uint64_t workspace_addr;
    uint8_t reserved[56];
};

using RtsStreamCreateFn = int32_t (*)(void **, RtStreamCreateConfig *);
using RtStreamGetSqidFn = int32_t (*)(const void *, uint32_t *);
using RtStreamGetCqidFn = int32_t (*)(const void *, uint32_t *, uint32_t *);
using RtGetDeviceInfoFn = int32_t (*)(uint32_t, int32_t, int32_t, int64_t *);

using AclnnStatus = int32_t;
using AclnnGetWsSizeFn = AclnnStatus (*)(const ::aclTensor *, ::aclTensor *, uint64_t *, ::aclOpExecutor **);
using AclnnExecFn = AclnnStatus (*)(void *, uint64_t, ::aclOpExecutor *, aclrtStream);

} // namespace detail

class SdmaWorkspaceManager {
public:
    SdmaWorkspaceManager() = default;
    ~SdmaWorkspaceManager()
    {
        Finalize();
    }

    SdmaWorkspaceManager(const SdmaWorkspaceManager &) = delete;
    SdmaWorkspaceManager &operator=(const SdmaWorkspaceManager &) = delete;

    bool Init()
    {
        if (inited_)
            return true;

        if (!LoadDynamicSymbols())
            return false;
        if (!CreateStarsStreams(detail::kSdmaMaxChan))
            return false;
        if (!MallocWorkspace(detail::kSdmaWorkspaceBytes))
            return false;
        if (!CopyOpResToDevice())
            return false;
        if (!LaunchAicpuKernel(reinterpret_cast<uint64_t>(opResDevicePtr_), opResInfo_.workspace_addr))
            return false;

        inited_ = true;
        return true;
    }

    void Finalize()
    {
        if (!inited_)
            return;
        if (opResDevicePtr_) {
            aclrtFree(opResDevicePtr_);
            opResDevicePtr_ = nullptr;
        }
        if (opResInfo_.workspace_addr) {
            aclrtFree(reinterpret_cast<void *>(opResInfo_.workspace_addr));
            opResInfo_.workspace_addr = 0;
        }
        for (uint32_t i = 0; i < detail::kSdmaMaxChan; ++i) {
            if (opResInfo_.streams[i].stream_) {
                aclrtDestroyStream(reinterpret_cast<aclrtStream>(opResInfo_.streams[i].stream_));
                opResInfo_.streams[i].stream_ = 0;
            }
        }
        opResInfo_ = {};
        CloseDynamicLibs();
        inited_ = false;
    }

    void *GetWorkspaceAddr() const
    {
        return reinterpret_cast<void *>(opResInfo_.workspace_addr);
    }

private:
    bool inited_{false};
    detail::SdmaOpResInfo opResInfo_{};
    void *opResDevicePtr_{nullptr};

    void *rtHandle_{nullptr};
    void *opapiHandle_{nullptr};

    detail::RtsStreamCreateFn pRtsStreamCreate_{nullptr};
    detail::RtStreamGetSqidFn pRtStreamGetSqid_{nullptr};
    detail::RtStreamGetCqidFn pRtStreamGetCqid_{nullptr};
    detail::RtGetDeviceInfoFn pRtGetDeviceInfo_{nullptr};
    detail::AclnnGetWsSizeFn pAclnnGetWsSize_{nullptr};
    detail::AclnnExecFn pAclnnExec_{nullptr};

    bool LoadDynamicSymbols()
    {
        dlerror();

        rtHandle_ = dlopen("libruntime.so", RTLD_NOW);
        if (!rtHandle_) {
            std::cerr << "[SDMA] dlopen libruntime.so failed: " << dlerror() << std::endl;
            return false;
        }

        pRtsStreamCreate_ = reinterpret_cast<detail::RtsStreamCreateFn>(dlsym(rtHandle_, "rtsStreamCreate"));
        pRtStreamGetSqid_ = reinterpret_cast<detail::RtStreamGetSqidFn>(dlsym(rtHandle_, "rtStreamGetSqid"));
        pRtStreamGetCqid_ = reinterpret_cast<detail::RtStreamGetCqidFn>(dlsym(rtHandle_, "rtStreamGetCqid"));
        pRtGetDeviceInfo_ = reinterpret_cast<detail::RtGetDeviceInfoFn>(dlsym(rtHandle_, "rtGetDeviceInfo"));
        if (!pRtsStreamCreate_ || !pRtStreamGetSqid_ || !pRtStreamGetCqid_ || !pRtGetDeviceInfo_) {
            std::cerr << "[SDMA] Failed to resolve runtime symbols: " << dlerror() << std::endl;
            return false;
        }

        opapiHandle_ = dlopen("libopapi.so", RTLD_NOW);
        if (!opapiHandle_) {
            std::cerr << "[SDMA] dlopen libopapi.so failed: " << dlerror() << std::endl;
            return false;
        }

        pAclnnGetWsSize_ =
            reinterpret_cast<detail::AclnnGetWsSizeFn>(dlsym(opapiHandle_, "aclnnShmemSdmaStarsQueryGetWorkspaceSize"));
        pAclnnExec_ = reinterpret_cast<detail::AclnnExecFn>(dlsym(opapiHandle_, "aclnnShmemSdmaStarsQuery"));
        if (!pAclnnGetWsSize_ || !pAclnnExec_) {
            std::cerr << "[SDMA] Failed to resolve opapi symbols: " << dlerror() << std::endl;
            return false;
        }

        return true;
    }

    void CloseDynamicLibs()
    {
        pRtsStreamCreate_ = nullptr;
        pRtStreamGetSqid_ = nullptr;
        pRtStreamGetCqid_ = nullptr;
        pRtGetDeviceInfo_ = nullptr;
        pAclnnGetWsSize_ = nullptr;
        pAclnnExec_ = nullptr;

        if (opapiHandle_) {
            dlclose(opapiHandle_);
            opapiHandle_ = nullptr;
        }
        if (rtHandle_) {
            dlclose(rtHandle_);
            rtHandle_ = nullptr;
        }
    }

    bool CreateStarsStreams(int32_t channelNum)
    {
        int32_t deviceId = -1;
        if (aclrtGetDevice(&deviceId) != 0) {
            std::cerr << "[SDMA] aclrtGetDevice failed" << std::endl;
            return false;
        }

        int64_t dieId = -1;
        constexpr int32_t kInfoTypePhyDieId = 19;
        if (pRtGetDeviceInfo_(static_cast<uint32_t>(deviceId), 0, kInfoTypePhyDieId, &dieId) != 0) {
            std::cerr << "[SDMA] rtGetDeviceInfo (die_id) failed" << std::endl;
            return false;
        }

        for (int32_t i = 0; i < channelNum; ++i) {
            opResInfo_.streams[i].stream_ = 0;

            detail::RtStreamCreateAttr attrs[2];
            attrs[0].id = detail::RT_STREAM_CREATE_ATTR_PRIORITY;
            attrs[0].value.priority = 0;
            attrs[1].id = detail::RT_STREAM_CREATE_ATTR_FLAGS;
            attrs[1].value.flags = 0x800;

            detail::RtStreamCreateConfig config{attrs, 2};
            void *stream = nullptr;
            if (pRtsStreamCreate_(&stream, &config) != 0) {
                std::cerr << "[SDMA] rtsStreamCreate channel " << i << " failed" << std::endl;
                return false;
            }
            opResInfo_.streams[i].stream_ = reinterpret_cast<uint64_t>(stream);

            int32_t streamId = 0;
            if (aclrtStreamGetId(reinterpret_cast<aclrtStream>(stream), &streamId) != 0) {
                std::cerr << "[SDMA] aclrtStreamGetId channel " << i << " failed" << std::endl;
                return false;
            }

            uint32_t sqId = 0;
            if (pRtStreamGetSqid_(stream, &sqId) != 0) {
                std::cerr << "[SDMA] rtStreamGetSqid channel " << i << " failed" << std::endl;
                return false;
            }

            uint32_t cqId = 0, logicCqId = 0;
            if (pRtStreamGetCqid_(stream, &cqId, &logicCqId) != 0) {
                std::cerr << "[SDMA] rtStreamGetCqid channel " << i << " failed" << std::endl;
                return false;
            }

            void *ctx = nullptr;
            if (aclrtGetCurrentContext(&ctx) != 0) {
                std::cerr << "[SDMA] aclrtGetCurrentContext channel " << i << " failed" << std::endl;
                return false;
            }

            opResInfo_.streams[i].ctx_ = reinterpret_cast<uint64_t>(ctx);
            opResInfo_.streams[i].stream_id = streamId;
            opResInfo_.streams[i].sq_id = sqId;
            opResInfo_.streams[i].cq_id = cqId;
            opResInfo_.streams[i].logic_cq_id = logicCqId;
            opResInfo_.streams[i].dev_id = static_cast<int32_t>(dieId);
        }

        std::cerr << "[SDMA] Created " << channelNum << " STARS streams OK" << std::endl;
        return true;
    }

    bool MallocWorkspace(size_t workspaceSize)
    {
        void *workspace = nullptr;
        if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            std::cerr << "[SDMA] aclrtMalloc workspace failed" << std::endl;
            return false;
        }
        if (aclrtMemset(workspace, workspaceSize, 0, workspaceSize) != 0) {
            std::cerr << "[SDMA] aclrtMemset workspace failed" << std::endl;
            aclrtFree(workspace);
            return false;
        }
        opResInfo_.workspace_addr = reinterpret_cast<uint64_t>(workspace);
        return true;
    }

    bool CopyOpResToDevice()
    {
        size_t size = sizeof(opResInfo_);
        if (aclrtMalloc(&opResDevicePtr_, size, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            std::cerr << "[SDMA] aclrtMalloc opResDevice failed" << std::endl;
            return false;
        }
        if (aclrtMemset(opResDevicePtr_, size, 0, size) != 0 ||
            aclrtMemcpy(opResDevicePtr_, size, &opResInfo_, size, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            std::cerr << "[SDMA] CopyOpResToDevice memset/memcpy failed" << std::endl;
            return false;
        }
        return true;
    }

    struct TensorGuard {
        void *deviceAddr{nullptr};
        ::aclTensor *tensor{nullptr};
        void Release()
        {
            if (tensor) {
                ::aclDestroyTensor(tensor);
                tensor = nullptr;
            }
            if (deviceAddr) {
                aclrtFree(deviceAddr);
                deviceAddr = nullptr;
            }
        }
    };

    bool CreateAclTensor(const std::vector<uint64_t> &hostData, const std::vector<int64_t> &shape, TensorGuard &guard)
    {
        if (shape.empty()) {
            std::cerr << "[SDMA] CreateAclTensor empty shape" << std::endl;
            return false;
        }
        uint64_t elemCount = 1;
        for (int64_t dim : shape) {
            if (dim <= 0) {
                std::cerr << "[SDMA] CreateAclTensor invalid dim: " << dim << std::endl;
                return false;
            }
            const uint64_t uDim = static_cast<uint64_t>(dim);
            if (elemCount > std::numeric_limits<uint64_t>::max() / uDim) {
                std::cerr << "[SDMA] CreateAclTensor shape overflow" << std::endl;
                return false;
            }
            elemCount *= uDim;
        }

        if (elemCount != hostData.size()) {
            std::cerr << "[SDMA] CreateAclTensor hostData size mismatch, elemCount=" << elemCount
                      << ", hostData.size=" << hostData.size() << std::endl;
            return false;
        }
        if (elemCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
            std::cerr << "[SDMA] CreateAclTensor totalBytes overflow" << std::endl;
            return false;
        }
        size_t totalBytes = static_cast<size_t>(elemCount * sizeof(uint64_t));

        if (aclrtMalloc(&guard.deviceAddr, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            std::cerr << "[SDMA] CreateAclTensor aclrtMalloc failed" << std::endl;
            return false;
        }
        if (aclrtMemcpy(guard.deviceAddr, totalBytes, hostData.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            std::cerr << "[SDMA] CreateAclTensor aclrtMemcpy failed" << std::endl;
            guard.Release();
            return false;
        }

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        guard.tensor = ::aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_UINT64, strides.data(), 0,
                                         aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), guard.deviceAddr);
        if (!guard.tensor) {
            std::cerr << "[SDMA] aclCreateTensor failed" << std::endl;
            guard.Release();
            return false;
        }
        return true;
    }

    bool LaunchAicpuKernel(uint64_t streamsAddr, uint64_t workspaceAddr)
    {
        aclrtStream aicpuStream = nullptr;
        if (aclrtCreateStreamWithConfig(&aicpuStream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC) != 0) {
            std::cerr << "[SDMA] aclrtCreateStreamWithConfig (aicpu) failed" << std::endl;
            return false;
        }

        aclrtStreamAttrValue value;
        value.failureMode = 1;
        aclrtSetStreamAttribute(aicpuStream, ACL_STREAM_ATTR_FAILURE_MODE, &value);

        TensorGuard inputGuard, outputGuard;
        bool ok = false;
        do {
            std::vector<int64_t> inShape = {2};
            std::vector<int64_t> outShape = {1};
            std::vector<uint64_t> inData = {streamsAddr, workspaceAddr};
            std::vector<uint64_t> outData = {0};

            if (!CreateAclTensor(inData, inShape, inputGuard))
                break;
            if (!CreateAclTensor(outData, outShape, outputGuard))
                break;

            uint64_t aclnnWsSize = 0;
            aclOpExecutor *executor = nullptr;
            if (pAclnnGetWsSize_(inputGuard.tensor, outputGuard.tensor, &aclnnWsSize, &executor) != 0) {
                std::cerr << "[SDMA] aclnnShmemSdmaStarsQueryGetWorkspaceSize failed" << std::endl;
                break;
            }

            void *aclnnWs = nullptr;
            if (aclnnWsSize > 0) {
                if (aclrtMalloc(&aclnnWs, aclnnWsSize, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
                    std::cerr << "[SDMA] aclrtMalloc aclnn workspace failed" << std::endl;
                    break;
                }
            }

            if (pAclnnExec_(aclnnWs, aclnnWsSize, executor, aicpuStream) != 0) {
                std::cerr << "[SDMA] aclnnShmemSdmaStarsQuery exec failed" << std::endl;
                if (aclnnWs)
                    aclrtFree(aclnnWs);
                break;
            }

            if (aclrtSynchronizeStream(aicpuStream) != 0) {
                std::cerr << "[SDMA] aclrtSynchronizeStream (aicpu) failed" << std::endl;
                if (aclnnWs)
                    aclrtFree(aclnnWs);
                break;
            }

            if (aclnnWs)
                aclrtFree(aclnnWs);
            ok = true;
        } while (false);

        inputGuard.Release();
        outputGuard.Release();
        aclrtDestroyStream(aicpuStream);

        if (ok) {
            std::cerr << "[SDMA] STARS query AICPU kernel completed OK" << std::endl;
        }
        return ok;
    }
};

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_WORKSPACE_MANAGER_HPP
