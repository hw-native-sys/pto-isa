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

template <int32_t tilingKey>
void launchTRANDOM_demo(uint8_t *out, uint8_t *src, void *stream);

class TRANDOMTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T>
void CheckResults(const std::string &goldenDir, size_t dstSize)
{
    std::vector<T> output(dstSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/output.bin", dstSize, output.data(), dstSize);

    bool allZero = true;
    for (const auto &v : output) {
        if (v != 0) {
            allZero = false;
            break;
        }
    }
    EXPECT_FALSE(allZero) << "TRANDOM output is all zeros";

    bool allSame = true;
    for (size_t i = 1; i < output.size(); ++i) {
        if (output[i] != output[0]) {
            allSame = false;
            break;
        }
    }
    EXPECT_FALSE(allSame) << "TRANDOM output is constant";
}

template <typename T, int rows, int cols>
void LaunchTRandom(T *out, uint32_t *key, uint32_t *counter, void *stream);

template <typename T, int rows, int cols>
void test_trandom()
{
    size_t dstSize = rows * cols * sizeof(T);
    size_t keySize = 2 * sizeof(uint32_t);
    size_t counterSize = 4 * sizeof(uint32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost;
    uint32_t *keyHost;
    uint32_t *counterHost;
    T *dstDevice;
    uint32_t *keyDevice;
    uint32_t *counterDevice;

    aclrtMallocHost((void **)(&dstHost), dstSize);
    aclrtMallocHost((void **)(&keyHost), keySize);
    aclrtMallocHost((void **)(&counterHost), counterSize);

    aclrtMalloc((void **)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&keyDevice, keySize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&counterDevice, counterSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/key.bin", keySize, keyHost, keySize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/counter.bin", counterSize, counterHost, counterSize));

    aclrtMemcpy(keyDevice, keySize, keyHost, keySize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(counterDevice, counterSize, counterHost, counterSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTRandom<T, rows, cols>(dstDevice, keyDevice, counterDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstSize);

    aclrtFree(dstDevice);
    aclrtFree(keyDevice);
    aclrtFree(counterDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(keyHost);
    aclrtFreeHost(counterHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    CheckResults<T>(GetGoldenDir(), dstSize);
}

const int ROWS = 4;
const int COLS = 256;

TEST_F(TRANDOMTest, case_uint32_4x256_4x256)
{
    test_trandom<uint32_t, ROWS, COLS>();
}
