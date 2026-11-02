/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_COMM_CONTEXT_DATA_HPP
#define PTO_COMM_DOMAIN_HOST_COMM_CONTEXT_DATA_HPP

// ============================================================================
// Internal data header — do NOT include directly.
// Use communication/host/comm_context.hpp instead, which provides Reset() /
// BuildComm() definitions required by ~CommContext().
// ============================================================================

#if defined(__CCE_KT_TEST__)
#error "communication/host/comm_context_data.hpp is host-only"
#endif

#include <cstdint>
#include <memory>

#include "pto/common/arch_macro.hpp"
#include "communication/comm_device_context.hpp"
#include "communication/comm_domain_types.hpp"
#include "communication/host/detail/rt_fwd.hpp"

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

// Host builds do not define __gm__ (a CCE address-space attribute). Stub it so
// that headers declaring __gm__-qualified pointers (SdmaWorkspaceManager, etc.)
// compile under g++. The macro remains defined for the rest of this host-only TU
// which is intentional — all domain host headers expect __gm__ to be empty.
#ifndef __gm__
#define __gm__
#define PTO_DOMAIN_DEFINED_GM_STUB 1
#endif

#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"

// Host builds often omit __NPU_ARCH__; enable URMA host path unless A2/A3.
#if defined(PTO_URMA_SUPPORTED) || (!defined(PTO_NPU_ARCH_A2A3) && !defined(__CCE_KT_TEST__))
#define PTO_DOMAIN_URMA_HOST 1
#endif

#ifdef PTO_DOMAIN_URMA_HOST
#include "pto/comm/async/urma/urma_workspace_manager.hpp"
#endif

namespace pto {
namespace comm {
namespace domain {

// ============================================================================
// CommContext: host-side communication domain state (non-copyable, movable).
// Own resources are released by Reset() / DestroyComm() / destructor.
// ============================================================================
struct CommContext {
    CommContext() = default;
    CommContext(const CommContext&) = delete;
    CommContext& operator=(const CommContext&) = delete;

    CommContext(CommContext&& other) noexcept { MoveFrom(other); }
    CommContext& operator=(CommContext&& other) noexcept
    {
        if (this != &other) {
            Reset();
            MoveFrom(other);
        }
        return *this;
    }

    ~CommContext() { Reset(); }

    uint32_t backends = 0;
    Bootstrap bootstrap = Bootstrap::Mpi;
    int deviceId = -1;
    HcclComm comm = nullptr;
    rtStream_t stream = nullptr;
    bool ownsComm = false;

    CommDeviceContext winHostCtx{};
    CommDeviceContext* winDevCtx = nullptr;
    void* winBase = nullptr;
    __gm__ uint8_t* sdmaWs = nullptr;
    std::unique_ptr<sdma::SdmaWorkspaceManager> sdmaMgr;

#ifdef PTO_DOMAIN_URMA_HOST
    CommDeviceContext urmaHostCtx{};
    CommDeviceContext* urmaDevCtx = nullptr;
    void* urmaDevBuf = nullptr;
    __gm__ uint8_t* urmaWs = nullptr;
    std::unique_ptr<urma::UrmaWorkspaceManager> urmaMgr;
#endif

    void Reset();

private:
    void MoveFrom(CommContext& other) noexcept;
};

inline void CommContext::MoveFrom(CommContext& other) noexcept
{
    backends = other.backends;
    bootstrap = other.bootstrap;
    deviceId = other.deviceId;
    comm = other.comm;
    stream = other.stream;
    ownsComm = other.ownsComm;
    winHostCtx = other.winHostCtx;
    winDevCtx = other.winDevCtx;
    winBase = other.winBase;
    sdmaWs = other.sdmaWs;
    sdmaMgr = std::move(other.sdmaMgr);
#ifdef PTO_DOMAIN_URMA_HOST
    urmaHostCtx = other.urmaHostCtx;
    urmaDevCtx = other.urmaDevCtx;
    urmaDevBuf = other.urmaDevBuf;
    urmaWs = other.urmaWs;
    urmaMgr = std::move(other.urmaMgr);
#endif
    other.backends = 0;
    other.comm = nullptr;
    other.stream = nullptr;
    other.ownsComm = false;
    other.winHostCtx = {};
    other.winDevCtx = nullptr;
    other.winBase = nullptr;
    other.sdmaWs = nullptr;
#ifdef PTO_DOMAIN_URMA_HOST
    other.urmaHostCtx = {};
    other.urmaDevCtx = nullptr;
    other.urmaDevBuf = nullptr;
    other.urmaWs = nullptr;
#endif
}

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_COMM_CONTEXT_DATA_HPP
