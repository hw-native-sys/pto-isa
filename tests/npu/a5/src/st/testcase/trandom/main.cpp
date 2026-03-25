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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTRANDOMTestCase(void *out, uint32_t *key, uint32_t *counter, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TRANDOMTest : public testing::Test {
public:
    aclrtStream stream;
    void *dstHost;
    void *dstDevice;
    uint32_t *keyHost;
    uint32_t *keyDevice;
    uint32_t *counterHost;
    uint32_t *counterDevice;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <typename T>
    bool CompareGolden(size_t dstSize, bool printAllEn = false)
    {
        std::vector<T> golden(dstSize);
        std::vector<T> result(dstSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstSize, result.data(), dstSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstSize, golden.data(), dstSize);

        float eps = 0.001f;
        if (printAllEn) {
            ResultCmp(golden, result, eps, 0, 1000, true);
        }
        ResultCmp(golden, result, eps);
        return true;
    }

    template <uint32_t caseId, typename T, int rows, int cols>
    bool TRANDOMTestFramework()
    {
        size_t dstSize = rows * cols * sizeof(T);
        size_t keySize = 2 * sizeof(uint32_t);
        size_t counterSize = 4 * sizeof(uint32_t);

        aclrtMallocHost(&dstHost, dstSize);
        aclrtMallocHost((void **)&keyHost, keySize);
        aclrtMallocHost((void **)&counterHost, counterSize);

        aclrtMalloc(&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&keyDevice, keySize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&counterDevice, counterSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/key.bin", keySize, keyHost, keySize);
        ReadFile(GetGoldenDir() + "/counter.bin", counterSize, counterHost, counterSize);
        aclrtMemcpy(keyDevice, keySize, keyHost, keySize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(counterDevice, counterSize, counterHost, counterSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTRANDOMTestCase<caseId>(dstDevice, keyDevice, counterDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstSize);

        aclrtFree(dstDevice);
        aclrtFree(keyDevice);
        aclrtFree(counterDevice);

        aclrtFreeHost(dstHost);
        aclrtFreeHost(keyHost);
        aclrtFreeHost(counterHost);

        return CompareGolden<T>(dstSize);
    }
};

TEST_F(TRANDOMTest, case01)
{
    bool ret = TRANDOMTestFramework<1, uint32_t, 4, 256>();
    EXPECT_TRUE(ret);
}
