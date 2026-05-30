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
// and HCCL handles are released before aclFinalize() runs.  Extracting
// this avoids a 19-line copy-paste block flagged by codecheck.

#include <cstdio>

#include "acl/acl.h"
#include <gtest/gtest.h>

#include "comm_mpi.h"

namespace pto {
namespace comm {
namespace ccu {
namespace st {

inline int RunCcuStMain(int argc, char **argv, void (*cleanup)())
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
