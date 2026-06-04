/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP
#define PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_mesh_common.hpp is a host-only header and cannot be included in device code."
#endif

// Shared scaffolding for the 1D-mesh CCU collective kernels (reduce / gather /
// scatter / broadcast).  These kernels differ only in their data path
// (DoReduce / DoGather / ...); the argument-loading, notify barriers and
// GeneArgs packing are identical and live here once.

#include <cstdint>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"

namespace pto {
namespace comm {
namespace ccu {
namespace detail {

class CcuMeshKernelBase : public hcomm::CcuKernel {
public:
    using hcomm::CcuKernel::CcuKernel;

protected:
    // Load every rank's input/output/token (packed by GeneArgs) then length.
    // Mirrors the GeneArgs layout [input_0..N-1, output_0..N-1, token_0..N-1, length].
    template <typename VarVec>
    inline void LoadPeerArgs(const VarVec &input, const VarVec &output, const VarVec &token,
                             const hcomm::CcuRep::Variable &length)
    {
        for (const auto &v : input) {
            Load(v);
        }
        for (const auto &v : output) {
            Load(v);
        }
        for (const auto &v : token) {
            Load(v);
        }
        Load(length);
    }

    // Symmetric notify barrier across all channels on a single notify bit.
    // Used for both the readiness PreSync and the data-landing PostSync; the
    // two are kept distinct by passing different syncBit values.
    template <typename ChannelVec>
    inline void NotifyBarrier(const ChannelVec &channels, uint32_t notifyIdx, uint32_t syncBit)
    {
        for (const auto &ch : channels) {
            (void)NotifyRecord(ch, notifyIdx, 1u << syncBit);
        }
        for (const auto &ch : channels) {
            (void)NotifyWait(ch, notifyIdx, 1u << syncBit);
        }
    }
};

// Pack all ranks' peer addresses into the GeneArgs uint64 vector:
// [input_0..N-1, output_0..N-1, token_0..N-1, length].  Addresses are
// exchanged on the host (MPI AllGather) and shipped to the CCU microcode
// through this packing, so no runtime address-exchange PreSync is needed.
inline std::vector<uint64_t> PackPeerArgs(uint32_t rankSize, const uint64_t *peerInput, const uint64_t *peerOutput,
                                          const uint64_t *peerToken, uint64_t length)
{
    std::vector<uint64_t> args;
    args.reserve(3 * rankSize + 1);
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerInput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerOutput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerToken[i]);
    }
    args.push_back(length);
    return args;
}

} // namespace detail
} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP
