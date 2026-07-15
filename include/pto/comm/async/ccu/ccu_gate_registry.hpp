/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_GATE_REGISTRY_HPP
#define PTO_COMM_ASYNC_CCU_CCU_GATE_REGISTRY_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_gate_registry.hpp is a host-only header and cannot be included in device code."
#endif

// Process-local descriptor registry for PTO gated CCU kernels.  Header-only.
//
// Producer (CCU kernel GeneArgs): calls Publish(rankId, dieId, ckeId, mask)
// Consumer (host / ST main):     calls TryGet(rankId, &desc) to retrieve it

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "pto/comm/async/ccu/ccu_types.hpp"

namespace pto {
namespace comm {
namespace ccu {

namespace detail {

inline std::mutex& GateMapMutex()
{
    static std::mutex m;
    return m;
}

inline std::unordered_map<uint32_t, CcuGateDescriptor>& GateMap()
{
    static std::unordered_map<uint32_t, CcuGateDescriptor> m;
    return m;
}

} // namespace detail

inline bool IsCcuGateEnabledFromEnv()
{
    const char* v = std::getenv(CCU_GATE_ENV);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

inline void Publish(uint32_t rankId, uint32_t dieId, uint32_t ckeId, uint32_t mask)
{
    CcuGateDescriptor desc{};
    desc.dieId = dieId;
    desc.ckeId = ckeId;
    desc.mask = mask;
    desc.mmioAddr = 0;

    std::lock_guard<std::mutex> lk(detail::GateMapMutex());
    detail::GateMap()[rankId] = desc;
}

inline bool TryGet(uint32_t rankId, CcuGateDescriptor& out)
{
    std::lock_guard<std::mutex> lk(detail::GateMapMutex());
    auto it = detail::GateMap().find(rankId);
    if (it == detail::GateMap().end()) {
        return false;
    }
    out = it->second;
    return true;
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_GATE_REGISTRY_HPP
