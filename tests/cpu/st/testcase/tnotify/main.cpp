/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TNOTIFY : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int op_type>
void LaunchTNotify(int32_t* out, int32_t* src, int32_t* signal, void* stream);

template <int op_type>
void test_tnotify()
{
    size_t fileSize = sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *dstHost, *src0Host, *src1Host;
    int32_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/sharedMemory.bin", fileSize, src0Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/signal.bin", fileSize, src1Host, fileSize));

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTNotify<op_type>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> golden(fileSize);
    std::vector<int32_t> devFinal(fileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TNOTIFY, case1) { test_tnotify<1>(); }
TEST_F(TNOTIFY, case2) { test_tnotify<2>(); }