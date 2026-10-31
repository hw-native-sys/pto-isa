/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-safe RDMA workspace layout shared by the Host control plane and the
// device dispatch path. Device-only request types live in rdma_types.hpp.

#ifndef PTO_COMM_ASYNC_RDMA_WORKSPACE_TYPES_HPP
#define PTO_COMM_ASYNC_RDMA_WORKSPACE_TYPES_HPP

#include <cstdint>

#include "pto/comm/rdma_backend.hpp"

namespace pto {
namespace comm {
namespace rdma {

constexpr uint32_t kRdmaWorkspaceMagic = 0x52444d41U; // "RDMA"
constexpr uint32_t kRdmaWorkspaceVersion = 3;

enum class RdmaDbMode : int32_t {
    INVALID_DB = -1,
    HW_DB = 0,
    SW_DB,
};

// Device workspace header. Queue arrays referenced below have layouts owned by
// the selected backend. A process/session selects exactly one backend.
struct RdmaInfo {
    uint32_t magic;
    uint32_t version;
    RdmaBackend backend;
    uint32_t qpNum;
    uint32_t rankCount;
    uint32_t reserved;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
};

// Verbs-style memory metadata. Each backend registration owns its own keys even
// when several engines register the same device virtual-address range.
struct RdmaMemInfo {
    uint64_t size;
    uint64_t addr;
    uint32_t lkey;
    uint32_t rkey;
};

// RDMA dispatch errors encoded in an AsyncEvent handle. Backend completion
// errors retain their own status ranges.
constexpr uint32_t kRdmaBackendUnavailableError = 0x21000;
constexpr uint32_t kRdmaInvalidWorkspaceError = 0x21001;
constexpr uint32_t kRdmaBackendMismatchError = 0x21002;
constexpr uint32_t kRdmaSessionBuildError = 0x21003;
constexpr uint32_t kRdmaUnsupportedOperationError = 0x21004;

static_assert(sizeof(RdmaInfo) == 64, "unexpected RDMA workspace-header layout");
static_assert(sizeof(RdmaMemInfo) == 24, "unexpected RDMA memory-info layout");

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_WORKSPACE_TYPES_HPP
