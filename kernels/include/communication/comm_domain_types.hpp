/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_COMM_DOMAIN_TYPES_HPP
#define PTO_COMM_DOMAIN_COMM_DOMAIN_TYPES_HPP

#include <cstdint>

namespace pto {
namespace comm {
namespace domain {

// ============================================================================
// CommBackend: capability bitmask for BuildComm (combinable).
// ============================================================================
enum CommBackend : uint32_t {
    MTE = 1u << 0,  // Sync TPUT/TGET; Window family; no transport workspace
    SDMA = 1u << 1, // Async; Window family; SDMA workspace
    URMA = 1u << 2, // Async; URMA family; UrmaInfo workspace
    // CCU = 1u << 3, // reserved
};

// ============================================================================
// AddrFamily: addressing family for GetDeviceContext / GetSymmetricBase.
// ============================================================================
enum class AddrFamily : uint32_t {
    Window = 0, // MTE + SDMA shared
    Urma = 1,   // URMA-owned (windowsIn backfilled)
};

// ============================================================================
// Bootstrap: how HcclRootInfo / barrier is exchanged.
// ============================================================================
enum class Bootstrap { Mpi, External };

// ============================================================================
// UrmaJettyLayout: mirrors urma::UrmaLayout so this header stays free of the
// URMA includes, which only host-side URMA builds pull in. The values match.
//
// PerPeer:    one jetty per peer. Many AIVs are fine while each owns a distinct
//             peer; two AIVs aimed at one peer share an SQ and corrupt it.
// SharedPool: one run of jetties per AIV, every jetty reaching every peer. Needed
//             when several AIVs must reach the same peer, or when the AIV-to-peer
//             split is not known when the domain is built.
// ============================================================================
enum class UrmaJettyLayout : uint32_t { PerPeer = 0, SharedPool = 1 };

inline constexpr uint32_t kUrmaAutoAivCount = 0xFFFFFFFFu;

// ============================================================================
// CommConfig: host-facing build config.
// Hccl handles are opaque void* so this header stays free of HCCL includes
// (cast to HcclComm / HcclRootInfo* in host-only code).
// ============================================================================
struct CommConfig {
    int rankId = -1;
    int rankNum = 0;
    int deviceId = -1; // default: rankId % nDevices + first
    uint32_t backends = CommBackend::MTE;
    uint64_t symBytes = 0;
    Bootstrap bootstrap = Bootstrap::Mpi;

    void* existingComm = nullptr;   // HcclComm when External
    const void* rootInfo = nullptr; // const HcclRootInfo* when External

    UrmaJettyLayout urmaLayout = UrmaJettyLayout::PerPeer;
    uint32_t urmaAivCount = kUrmaAutoAivCount;
    uint32_t urmaJettiesPerCore = 1;
};

// The caller queries these requirements before reserving device storage.
// Reserve conversion storage before runtime topology is known; Array layouts borrow HCCL's context.
struct CommWorkspaceSizes {
    uint64_t sdma{0};
    uint64_t urma{0};
    uint64_t windowDeviceContext{0};
    uint64_t urmaDeviceContext{0};
    uint64_t urmaSymmetric{0};
};

// Each nonzero requirement needs a distinct, aligned external memory region.
// All regions remain caller-owned through both successful and failed builds.
struct CommWorkspaceAddresses {
    void* sdma{nullptr};
    void* urma{nullptr};
    void* windowDeviceContext{nullptr};
    void* urmaDeviceContext{nullptr};
    void* urmaSymmetric{nullptr};
};

inline constexpr uint32_t kSignalPrefixBytes = 64u * sizeof(int32_t);

inline bool HasBackend(uint32_t backends, CommBackend bit) { return (backends & static_cast<uint32_t>(bit)) != 0u; }

inline bool NeedsWindowFamily(uint32_t backends)
{
    return HasBackend(backends, CommBackend::MTE) || HasBackend(backends, CommBackend::SDMA);
}

inline bool NeedsUrmaFamily(uint32_t backends) { return HasBackend(backends, CommBackend::URMA); }

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_COMM_DOMAIN_TYPES_HPP
