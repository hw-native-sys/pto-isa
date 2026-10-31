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
#include <limits>
#include <vector>

#include <dlfcn.h>

#include "acl/acl.h"
#include "pto/comm/async_common/sdma_constants.hpp"
#include "pto/comm/logger/host_log.hpp"

#ifndef ACL_STREAM_DEVICE_USE_ONLY
#define ACL_STREAM_DEVICE_USE_ONLY 0x00000020U
#endif

// aclnn tensor API (from aclnn/acl_meta.h, linked via libnnopbase).
// Forward-declared to avoid pulling in aclnn headers that may conflict.
struct aclTensor;
struct aclOpExecutor;
extern "C" aclTensor* aclCreateTensor(
    const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride, int64_t offset,
    aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum, void* tensorData);
extern "C" int32_t aclDestroyTensor(const aclTensor* tensor);

namespace pto {
namespace comm {
namespace sdma {

// ============================================================================
// SdmaWorkspaceManager: Host-side SDMA workspace initialization.
//
// Mirrors the shmem SdmaTransportManager::OpenDevice() flow:
//   1. Creates MC2 STARS streams via the runtime API
//   2. Binds caller-provided device workspace memory
//   3. Launches an AICPU STARS-query kernel to populate the workspace
//      with hardware SQ addresses, register bases, and queue depths
//
// The resulting workspace pointer is passed to the AICORE kernel and
// forwarded to comm::BuildAsyncSession() / BuildSdmaSession().
// ============================================================================

namespace detail {

constexpr uint32_t kSdmaMaxChan = kSdmaMaxChannelGroups;

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
    uint64_t size;
    uint64_t streams_addr;
    uint64_t workspace_addr;
    uint8_t reserved[40];
};
static_assert(sizeof(SdmaOpResInfo) == 64, "SdmaOpResInfo must be 64 bytes");

using RtStreamGetSqidFn = int32_t (*)(const void*, uint32_t*);
using RtStreamGetCqidFn = int32_t (*)(const void*, uint32_t*, uint32_t*);
using RtGetDeviceInfoFn = int32_t (*)(uint32_t, int32_t, int32_t, int64_t*);

using AclnnStatus = int32_t;
using AclnnGetWsSizeFn = AclnnStatus (*)(const ::aclTensor*, ::aclTensor*, uint64_t*, ::aclOpExecutor**);
using AclnnExecFn = AclnnStatus (*)(void*, uint64_t, ::aclOpExecutor*, aclrtStream);
using AclExecutorFn = AclnnStatus (*)(::aclOpExecutor*);

// Fixed workspace region; reserved fields preserve the tensor offsets and ACLNN scratch boundary.
struct SdmaWorkspaceLayout {
    uint8_t workspace[kSdmaWorkspaceBytes];
    HostStreamInfo streams[kSdmaMaxChan];
    SdmaOpResInfo opRes;
    uint64_t input[2];
    uint64_t reserved0[6];
    uint64_t output;
    uint64_t reserved1[47];
};

constexpr uint64_t kSdmaStreamsOffset = offsetof(SdmaWorkspaceLayout, streams);
constexpr uint64_t kSdmaOpResOffset = offsetof(SdmaWorkspaceLayout, opRes);
constexpr uint64_t kSdmaInputOffset = offsetof(SdmaWorkspaceLayout, input);
constexpr uint64_t kSdmaOutputOffset = offsetof(SdmaWorkspaceLayout, output);
constexpr uint64_t kSdmaScratchOffset = sizeof(SdmaWorkspaceLayout);

} // namespace detail

class SdmaWorkspaceManager {
public:
    SdmaWorkspaceManager() = default;
    ~SdmaWorkspaceManager() { Finalize(); }

    SdmaWorkspaceManager(const SdmaWorkspaceManager&) = delete;
    SdmaWorkspaceManager& operator=(const SdmaWorkspaceManager&) = delete;

    static aclError GetWorkspaceSize(uint64_t* bytes)
    {
        if (bytes == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        *bytes = 0;
        SdmaWorkspaceManager query;
        if (!query.LoadDynamicSymbols()) {
            return ACL_ERROR_FEATURE_UNSUPPORTED;
        }
        uint64_t scratchBytes = 0;
        if (!query.QueryScratchSize(scratchBytes)) {
            return ACL_ERROR_FAILURE;
        }
        *bytes = detail::kSdmaScratchOffset + scratchBytes;
        return ACL_SUCCESS;
    }

    bool Init(void* workspaceAddr, uint64_t workspaceSize)
    {
        if (inited_) {
            return workspaceAddr == GetWorkspaceAddr() && workspaceSize >= workspaceBytes_;
        }

        if (workspaceAddr == nullptr || workspaceSize < detail::kSdmaScratchOffset) {
            return false;
        }
        if (!LoadDynamicSymbols())
            return false;
        workspaceBytes_ = workspaceSize;
        if (!CreateStarsStreams(detail::kSdmaMaxChan))
            return false;
        if (!BindWorkspace(workspaceAddr))
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
        streamsDevicePtr_ = nullptr;
        opResDevicePtr_ = nullptr;
        for (auto& s : streams_) {
            if (s.stream_) {
                aclrtDestroyStream(reinterpret_cast<aclrtStream>(s.stream_));
                s.stream_ = 0;
            }
        }
        streams_.clear();
        opResInfo_ = {};
        workspaceBytes_ = 0;
        CloseDynamicLibs();
        inited_ = false;
    }

    void* GetWorkspaceAddr() const { return reinterpret_cast<void*>(opResInfo_.workspace_addr); }
    uint64_t GetWorkspaceSize() const { return inited_ ? workspaceBytes_ : 0; }

private:
    uint8_t inited_{0};
    uint64_t workspaceBytes_{0}; // Available capacity during Init; required bytes after the execution query.
    detail::SdmaOpResInfo opResInfo_{};
    void* opResDevicePtr_{nullptr};
    std::vector<detail::HostStreamInfo> streams_;
    void* streamsDevicePtr_{nullptr};

    void* rtHandle_{nullptr};
    void* opapiHandle_{nullptr};

    detail::RtStreamGetSqidFn pRtStreamGetSqid_{nullptr};
    detail::RtStreamGetCqidFn pRtStreamGetCqid_{nullptr};
    detail::RtGetDeviceInfoFn pRtGetDeviceInfo_{nullptr};
    detail::AclnnGetWsSizeFn pAclnnGetWsSize_{nullptr};
    detail::AclnnExecFn pAclnnExec_{nullptr};
    detail::AclExecutorFn pAclSetExecutorRepeatable_{nullptr};
    detail::AclExecutorFn pAclDestroyExecutor_{nullptr};

    bool LoadDynamicSymbols()
    {
        dlerror();

        rtHandle_ = dlopen("libruntime.so", RTLD_NOW);
        if (!rtHandle_) {
            PTO_HOST_LOG_ERROR("SDMA", "dlopen libruntime.so failed: " << dlerror());
            return false;
        }

        pRtStreamGetSqid_ = reinterpret_cast<detail::RtStreamGetSqidFn>(dlsym(rtHandle_, "rtStreamGetSqid"));
        pRtStreamGetCqid_ = reinterpret_cast<detail::RtStreamGetCqidFn>(dlsym(rtHandle_, "rtStreamGetCqid"));
        pRtGetDeviceInfo_ = reinterpret_cast<detail::RtGetDeviceInfoFn>(dlsym(rtHandle_, "rtGetDeviceInfo"));
        if (!pRtStreamGetSqid_ || !pRtStreamGetCqid_ || !pRtGetDeviceInfo_) {
            PTO_HOST_LOG_ERROR("SDMA", "Failed to resolve runtime symbols: " << dlerror());
            return false;
        }

        opapiHandle_ = dlopen("libopapi.so", RTLD_NOW);
        if (!opapiHandle_) {
            PTO_HOST_LOG_ERROR("SDMA", "dlopen libopapi.so failed: " << dlerror());
            return false;
        }

        pAclnnGetWsSize_ =
            reinterpret_cast<detail::AclnnGetWsSizeFn>(dlsym(opapiHandle_, "aclnnShmemSdmaStarsQueryGetWorkspaceSize"));
        pAclnnExec_ = reinterpret_cast<detail::AclnnExecFn>(dlsym(opapiHandle_, "aclnnShmemSdmaStarsQuery"));
        if (!pAclnnGetWsSize_ || !pAclnnExec_) {
            PTO_HOST_LOG_ERROR("SDMA", "Failed to resolve opapi symbols: " << dlerror());
            return false;
        }

        pAclSetExecutorRepeatable_ =
            reinterpret_cast<detail::AclExecutorFn>(dlsym(opapiHandle_, "aclSetAclOpExecutorRepeatable"));
        if (!pAclSetExecutorRepeatable_) {
            PTO_HOST_LOG_ERROR("SDMA", "Failed to resolve aclSetAclOpExecutorRepeatable: " << dlerror());
            return false;
        }
        pAclDestroyExecutor_ = reinterpret_cast<detail::AclExecutorFn>(dlsym(opapiHandle_, "aclDestroyAclOpExecutor"));
        if (!pAclDestroyExecutor_) {
            PTO_HOST_LOG_ERROR("SDMA", "Failed to resolve aclDestroyAclOpExecutor: " << dlerror());
            return false;
        }

        return true;
    }

    void CloseDynamicLibs()
    {
        pRtStreamGetSqid_ = nullptr;
        pRtStreamGetCqid_ = nullptr;
        pRtGetDeviceInfo_ = nullptr;
        pAclnnGetWsSize_ = nullptr;
        pAclnnExec_ = nullptr;
        pAclSetExecutorRepeatable_ = nullptr;
        pAclDestroyExecutor_ = nullptr;

        if (opapiHandle_) {
            dlclose(opapiHandle_);
            opapiHandle_ = nullptr;
        }
        if (rtHandle_) {
            dlclose(rtHandle_);
            rtHandle_ = nullptr;
        }
    }

    bool InitOneStarsStream(detail::HostStreamInfo& info, int32_t channelIdx, int32_t dieId)
    {
        info.stream_ = 0;

        void* stream = nullptr;
        if (const auto ret =
                aclrtCreateStreamWithConfig(reinterpret_cast<aclrtStream*>(&stream), 0, ACL_STREAM_DEVICE_USE_ONLY);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtCreateStreamWithConfig channel " << channelIdx << " failed, ret=" << ret);
            return false;
        }
        info.stream_ = reinterpret_cast<uint64_t>(stream);

        int32_t streamId = 0;
        if (const auto ret = aclrtStreamGetId(reinterpret_cast<aclrtStream>(stream), &streamId); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtStreamGetId channel " << channelIdx << " failed, ret=" << ret);
            return false;
        }

        uint32_t sqId = 0;
        if (const auto ret = pRtStreamGetSqid_(stream, &sqId); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "rtStreamGetSqid channel " << channelIdx << " failed, ret=" << ret);
            return false;
        }

        uint32_t cqId = 0, logicCqId = 0;
        if (const auto ret = pRtStreamGetCqid_(stream, &cqId, &logicCqId); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "rtStreamGetCqid channel " << channelIdx << " failed, ret=" << ret);
            return false;
        }

        void* ctx = nullptr;
        if (const auto ret = aclrtGetCurrentContext(&ctx); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtGetCurrentContext channel " << channelIdx << " failed, ret=" << ret);
            return false;
        }

        info.ctx_ = reinterpret_cast<uint64_t>(ctx);
        info.stream_id = streamId;
        info.sq_id = sqId;
        info.cq_id = cqId;
        info.logic_cq_id = logicCqId;
        info.dev_id = dieId;
        return true;
    }

    bool CreateStarsStreams(int32_t channelNum)
    {
        int32_t deviceId = -1;
        if (const auto ret = aclrtGetDevice(&deviceId); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtGetDevice failed, ret=" << ret);
            return false;
        }

        int64_t dieId = -1;
        constexpr int32_t kInfoTypePhyDieId = 19;
        if (const auto ret = pRtGetDeviceInfo_(static_cast<uint32_t>(deviceId), 0, kInfoTypePhyDieId, &dieId);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "rtGetDeviceInfo (die_id) failed, ret=" << ret);
            return false;
        }

        streams_.resize(channelNum);
        for (int32_t i = 0; i < channelNum; ++i) {
            if (!InitOneStarsStream(streams_[i], i, static_cast<int32_t>(dieId))) {
                return false;
            }
        }

        PTO_HOST_LOG_INFO("SDMA", "Created " << channelNum << " STARS streams OK");
        return true;
    }

    bool BindWorkspace(void* workspaceAddr)
    {
        opResInfo_.workspace_addr = reinterpret_cast<uint64_t>(workspaceAddr);
        streamsDevicePtr_ = static_cast<uint8_t*>(workspaceAddr) + detail::kSdmaStreamsOffset;
        opResDevicePtr_ = static_cast<uint8_t*>(workspaceAddr) + detail::kSdmaOpResOffset;
        if (const auto ret = aclrtMemset(workspaceAddr, kSdmaWorkspaceBytes, 0, kSdmaWorkspaceBytes); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtMemset workspace failed, ret=" << ret);
            return false;
        }
        return true;
    }

    bool CopyOpResToDevice()
    {
        size_t streamsBytes = streams_.size() * sizeof(detail::HostStreamInfo);
        if (const auto ret =
                aclrtMemcpy(streamsDevicePtr_, streamsBytes, streams_.data(), streamsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "CopyOpResToDevice streams memcpy failed, ret=" << ret);
            return false;
        }

        opResInfo_.size = streams_.size();
        opResInfo_.streams_addr = reinterpret_cast<uint64_t>(streamsDevicePtr_);

        size_t opResSize = sizeof(opResInfo_);
        if (const auto ret = aclrtMemset(opResDevicePtr_, opResSize, 0, opResSize); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtMemset(opRes) failed, ret=" << ret);
            return false;
        }
        if (const auto ret = aclrtMemcpy(opResDevicePtr_, opResSize, &opResInfo_, opResSize, ACL_MEMCPY_HOST_TO_DEVICE);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtMemcpy(opRes) failed, ret=" << ret);
            return false;
        }
        return true;
    }

    struct TensorGuard {
        ::aclTensor* tensor{nullptr};
        void Release()
        {
            if (tensor) {
                ::aclDestroyTensor(tensor);
                tensor = nullptr;
            }
        }
    };

    bool CalculateElementCount(const std::vector<int64_t>& shape, uint64_t& elemCount)
    {
        if (shape.empty()) {
            PTO_HOST_LOG_ERROR("SDMA", "CalculateElementCount empty shape");
            return false;
        }
        elemCount = 1;
        for (int64_t dim : shape) {
            if (dim <= 0) {
                PTO_HOST_LOG_ERROR("SDMA", "CalculateElementCount invalid dim: " << dim);
                return false;
            }
            const uint64_t uDim = static_cast<uint64_t>(dim);
            if (elemCount > std::numeric_limits<uint64_t>::max() / uDim) {
                PTO_HOST_LOG_ERROR("SDMA", "CalculateElementCount shape overflow");
                return false;
            }
            elemCount *= uDim;
        }
        return true;
    }

    bool CreateAclTensor(
        const std::vector<uint64_t>& hostData, const std::vector<int64_t>& shape, void* deviceAddr, TensorGuard& guard)
    {
        uint64_t elemCount = 0;
        if (!CalculateElementCount(shape, elemCount)) {
            return false;
        }
        if (elemCount != hostData.size()) {
            PTO_HOST_LOG_ERROR(
                "SDMA", "CreateAclTensor hostData size mismatch, elemCount=" << elemCount
                                                                             << ", hostData.size=" << hostData.size());
            return false;
        }
        if (elemCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
            PTO_HOST_LOG_ERROR("SDMA", "CreateAclTensor totalBytes overflow");
            return false;
        }
        size_t totalBytes = static_cast<size_t>(elemCount * sizeof(uint64_t));

        if (const auto ret =
                aclrtMemcpy(deviceAddr, totalBytes, hostData.data(), totalBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "CreateAclTensor aclrtMemcpy failed, ret=" << ret);
            guard.Release();
            return false;
        }

        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        guard.tensor = ::aclCreateTensor(
            shape.data(), shape.size(), aclDataType::ACL_UINT64, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), deviceAddr);
        if (!guard.tensor) {
            PTO_HOST_LOG_ERROR("SDMA", "aclCreateTensor failed");
            guard.Release();
            return false;
        }
        return true;
    }

    bool DestroyQueryExecutor(aclOpExecutor* executor)
    {
        if (executor == nullptr) {
            return true;
        }
        const auto repeatable = pAclSetExecutorRepeatable_(executor);
        const auto destroyed = pAclDestroyExecutor_(executor);
        if (repeatable != 0 || destroyed != 0) {
            PTO_HOST_LOG_ERROR(
                "SDMA", "query executor cleanup failed, repeatable=" << repeatable << ", destroy=" << destroyed);
            return false;
        }
        return true;
    }

    bool QueryScratchSize(uint64_t& bytes)
    {
        const int64_t inputShape[] = {2};
        const int64_t outputShape[] = {1};
        const int64_t stride[] = {1};
        TensorGuard inputGuard, outputGuard;
        inputGuard.tensor =
            ::aclCreateTensor(inputShape, 1, ACL_UINT64, stride, 0, ACL_FORMAT_ND, inputShape, 1, nullptr);
        outputGuard.tensor =
            ::aclCreateTensor(outputShape, 1, ACL_UINT64, stride, 0, ACL_FORMAT_ND, outputShape, 1, nullptr);
        if (inputGuard.tensor == nullptr || outputGuard.tensor == nullptr) {
            PTO_HOST_LOG_ERROR("SDMA", "aclCreateTensor for workspace size query failed");
            inputGuard.Release();
            outputGuard.Release();
            return false;
        }
        bytes = 0;
        aclOpExecutor* executor = nullptr;
        const auto ret = pAclnnGetWsSize_(inputGuard.tensor, outputGuard.tensor, &bytes, &executor);
        const bool queried = ret == 0 && executor != nullptr;
        const bool released = DestroyQueryExecutor(executor);
        inputGuard.Release();
        outputGuard.Release();
        if (!queried) {
            PTO_HOST_LOG_ERROR("SDMA", "workspace size query failed, ret=" << ret);
        }
        return queried && released;
    }

    bool ExecuteAicpuQuery(aclrtStream aicpuStream, uint64_t streamsAddr, uint64_t workspaceAddr)
    {
        TensorGuard inputGuard, outputGuard;
        std::vector<int64_t> inShape = {2};
        std::vector<int64_t> outShape = {1};
        std::vector<uint64_t> inData = {streamsAddr, workspaceAddr};
        std::vector<uint64_t> outData = {0};

        auto* workspaceBase = reinterpret_cast<uint8_t*>(workspaceAddr);
        if (!CreateAclTensor(inData, inShape, workspaceBase + detail::kSdmaInputOffset, inputGuard))
            return false;
        if (!CreateAclTensor(outData, outShape, workspaceBase + detail::kSdmaOutputOffset, outputGuard))
            return false;

        uint64_t aclnnWsSize = 0;
        aclOpExecutor* executor = nullptr;
        if (const auto ret = pAclnnGetWsSize_(inputGuard.tensor, outputGuard.tensor, &aclnnWsSize, &executor);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclnnShmemSdmaStarsQueryGetWorkspaceSize failed, ret=" << ret);
            return false;
        }

        if (aclnnWsSize > workspaceBytes_ - detail::kSdmaScratchOffset) {
            PTO_HOST_LOG_ERROR("SDMA", "aclnn scratch size exceeds the available workspace capacity");
            (void)DestroyQueryExecutor(executor);
            inputGuard.Release();
            outputGuard.Release();
            return false;
        }
        workspaceBytes_ = detail::kSdmaScratchOffset + aclnnWsSize;
        void* aclnnWs = aclnnWsSize == 0 ? nullptr : workspaceBase + detail::kSdmaScratchOffset;

        bool ok = false;
        if (const auto ret = pAclnnExec_(aclnnWs, aclnnWsSize, executor, aicpuStream); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclnnShmemSdmaStarsQuery exec failed, ret=" << ret);
        } else if (const auto ret = aclrtSynchronizeStream(aicpuStream); ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtSynchronizeStream (aicpu) failed, ret=" << ret);
        } else {
            ok = true;
        }

        inputGuard.Release();
        outputGuard.Release();
        return ok;
    }

    bool LaunchAicpuKernel(uint64_t streamsAddr, uint64_t workspaceAddr)
    {
        aclrtStream aicpuStream = nullptr;
        if (const auto ret =
                aclrtCreateStreamWithConfig(&aicpuStream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
            ret != 0) {
            PTO_HOST_LOG_ERROR("SDMA", "aclrtCreateStreamWithConfig (aicpu) failed, ret=" << ret);
            return false;
        }

        aclrtStreamAttrValue value;
        value.failureMode = 1;
        aclrtSetStreamAttribute(aicpuStream, ACL_STREAM_ATTR_FAILURE_MODE, &value);

        bool ok = ExecuteAicpuQuery(aicpuStream, streamsAddr, workspaceAddr);
        aclrtDestroyStream(aicpuStream);

        if (ok) {
            PTO_HOST_LOG_INFO("SDMA", "STARS query AICPU kernel completed OK");
        }
        return ok;
    }
};

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_WORKSPACE_MANAGER_HPP
