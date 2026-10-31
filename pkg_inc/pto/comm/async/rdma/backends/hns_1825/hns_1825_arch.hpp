/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_ARCH_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_ARCH_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_arch.hpp is a host-only header and cannot be included in device code."
#endif

#include <dlfcn.h>

#include <cstdint>
#include <string>

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

struct ArchitectureProbe {
    std::string runtimeSocName;
    uint32_t compiledNpuArch{0};
    uint8_t runtimeResolved{0};
    uint8_t supported{0};
    uint8_t reserved[2]{0};
};

// Accept HNS1825 only when every available signal agrees with 3510. Host-only
// translation units may not define __NPU_ARCH__, so the runtime SoC remains
// authoritative in that case.
inline ArchitectureProbe ProbeArchitecture()
{
    ArchitectureProbe result{};
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    result.compiledNpuArch = 3510;
#elif defined(__NPU_ARCH__)
    result.compiledNpuArch = static_cast<uint32_t>(__NPU_ARCH__);
#endif

    using GetSocNameFn = const char* (*)();
    auto getSocName = reinterpret_cast<GetSocNameFn>(dlsym(RTLD_DEFAULT, "aclrtGetSocName"));
    if (getSocName != nullptr) {
        const char* name = getSocName();
        if (name != nullptr && name[0] != '\0') {
            result.runtimeSocName = name;
            result.runtimeResolved = 1;
            const bool runtimeIs3510 = result.runtimeSocName.find("Ascend950") != std::string::npos ||
                                       result.runtimeSocName.find("dav-3510") != std::string::npos ||
                                       result.runtimeSocName.find("3510") != std::string::npos;
            result.supported = runtimeIs3510 && (result.compiledNpuArch == 0 || result.compiledNpuArch == 3510);
            return result;
        }
    }

    result.supported = result.compiledNpuArch == 3510;
    return result;
}

inline std::string DescribeArchitecture(const ArchitectureProbe& probe)
{
    if (probe.runtimeResolved) {
        if (probe.compiledNpuArch != 0) {
            return probe.runtimeSocName + " (build NPU_ARCH " + std::to_string(probe.compiledNpuArch) + ')';
        }
        return probe.runtimeSocName + " (build NPU_ARCH unavailable)";
    }
    if (probe.compiledNpuArch != 0) {
        return std::string("NPU_ARCH ") + std::to_string(probe.compiledNpuArch) + " (runtime SoC unavailable)";
    }
    return "unknown (runtime SoC and NPU_ARCH unavailable)";
}

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_ARCH_HPP
