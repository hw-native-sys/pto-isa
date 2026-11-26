/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TESTS_NPU_A5_COMM_ST_TESTCASE_CCU_TEST_MAIN_HPP
#define PTO_TESTS_NPU_A5_COMM_ST_TESTCASE_CCU_TEST_MAIN_HPP

// Shared GTest entry point for the four CCU collective ST drivers
// (treduce_ccu / tscatter_ccu / tbroadcast_ccu / tgather_ccu).  Each
// driver passes its file-local CleanupEnv() so per-test device buffers
// and HCCL handles are released before aclFinalize() runs.

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <utility>

#include "acl/acl.h"
#include <gtest/gtest.h>

#include "pto/comm/async/ccu/ccu_types.hpp"

#include "comm_mpi.h"

namespace pto {
namespace comm {
namespace ccu {
namespace st {

// Resolve the CKE slot used by the host driver before it launches the AIV trigger.
namespace detail {

struct RtDevResInfo {
    uint32_t dieId;
    int procType;
    int resType;
    uint32_t resId;
    uint32_t flag;
};

struct RtDevResAddrInfo {
    uint64_t* resAddress;
    uint32_t* len;
};

using RtGetDevResAddressFn = int (*)(RtDevResInfo*, RtDevResAddrInfo*);

inline RtGetDevResAddressFn GetRtAddressQuery()
{
    using RuntimeHandle = std::unique_ptr<void, int (*)(void*)>;
    static std::mutex mutex;
    static RuntimeHandle runtime(nullptr, dlclose);
    static RtGetDevResAddressFn query = nullptr;

    std::lock_guard<std::mutex> lock(mutex);
    if (query != nullptr) {
        return query;
    }

    RuntimeHandle candidate(dlopen("libruntime.so", RTLD_NOW | RTLD_GLOBAL), dlclose);
    if (!candidate) {
        std::fprintf(stderr, "dlopen libruntime.so: %s\n", dlerror());
        return nullptr;
    }
    auto symbol = reinterpret_cast<RtGetDevResAddressFn>(dlsym(candidate.get(), "rtGetDevResAddress"));
    if (symbol == nullptr) {
        std::fprintf(stderr, "dlsym rtGetDevResAddress: %s\n", dlerror());
        return nullptr;
    }

    // Keep the symbol valid for this ST process. Failed lookups release the candidate and can be retried.
    runtime = std::move(candidate);
    query = symbol;
    return query;
}

} // namespace detail

inline uint64_t ResolveCkeSlotVa(const CcuGateDescriptor& gate, const char* testName, int rankId)
{
    if (gate.mmioAddr != 0) {
        return gate.mmioAddr;
    }

    const auto query = detail::GetRtAddressQuery();
    if (query == nullptr) {
        return 0;
    }

    constexpr int CCU_RT_PROCESS_CP1 = 0;
    constexpr int CCU_RT_RES_TYPE_CKE = 3;
    detail::RtDevResInfo in{gate.dieId, CCU_RT_PROCESS_CP1, CCU_RT_RES_TYPE_CKE, gate.ckeId, 0};
    uint64_t ckeSlotVA = 0;
    uint32_t len = 0;
    detail::RtDevResAddrInfo out{&ckeSlotVA, &len};
    const int rc = query(&in, &out);
    if (rc != 0 || ckeSlotVA == 0) {
        std::fprintf(stderr, "[%s] rank=%d rtGetDevResAddress FAIL rc=%d\n", testName, rankId, rc);
        return 0;
    }
    return ckeSlotVA;
}

inline int RunCcuStMain(int argc, char** argv, void (*cleanup)())
{
    setvbuf(stderr, nullptr, _IONBF, 0);
    CommMpiInit(&argc, &argv);
    aclError ar = aclInit(nullptr);
    if (ar != ACL_SUCCESS && static_cast<int>(ar) != 100002) {
        CommMpiFinalize();
        return 2;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    if (cleanup != nullptr)
        cleanup();
    aclFinalize();
    CommMpiBarrier();
    CommMpiFinalize();
    return ret;
}

} // namespace st
} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_TESTS_NPU_A5_COMM_ST_TESTCASE_CCU_TEST_MAIN_HPP
