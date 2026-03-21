/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

namespace TRowSumTest {
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTROWSUMTest(T *out, T *src, aclrtStream stream);

class TROWSUMTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
bool TRowSumTest()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t inputFileSize = fileSize;
    size_t outputFileSize = fileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;

    aclrtMallocHost((void **)&dstHost, outputFileSize);
    aclrtMallocHost((void **)&srcHost, inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", inputFileSize, srcHost, inputFileSize);

    aclrtMemcpy(srcDevice, inputFileSize, srcHost, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWSUMTest<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    return ResultCmp(golden, devFinal, 0.001f);
}

constexpr int smallSize = 16;

// FIXME: The tile size inside trowsum_kernel.cpp is way larger than the UB size (192KB for A2 and 256KB for A5),
// which should return an error from aclrtSynchronizeStream and fail.
constexpr int bigSize666 = 666;
constexpr int bigSizeAligned = 672;
constexpr int bigSize64 = 64;

TEST_F(TROWSUMTest, test1)
{
    bool res = TRowSumTest<float, smallSize, smallSize, smallSize, smallSize>();
    EXPECT_TRUE(res);
}

TEST_F(TROWSUMTest, test2)
{
    bool res = TRowSumTest<uint16_t, smallSize, smallSize, smallSize, smallSize>();
    EXPECT_TRUE(res);
}

TEST_F(TROWSUMTest, test3)
{
    bool res = TRowSumTest<float, smallSize, bigSize666, smallSize, bigSizeAligned>();
    EXPECT_TRUE(res);
}

// int32 test cases
TEST_F(TROWSUMTest, test4)
{
    bool res = TRowSumTest<int32_t, smallSize, smallSize, smallSize, smallSize>();
    EXPECT_TRUE(res);
}

TEST_F(TROWSUMTest, test5)
{
    bool res = TRowSumTest<int32_t, bigSize64, bigSize64, bigSize64, bigSize64>();
    EXPECT_TRUE(res);
}

// int16 test cases
TEST_F(TROWSUMTest, test6)
{
    bool res = TRowSumTest<int16_t, smallSize, smallSize, smallSize, smallSize>();
    EXPECT_TRUE(res);
}

TEST_F(TROWSUMTest, test7)
{
    bool res = TRowSumTest<int16_t, bigSize64, bigSize64, bigSize64, bigSize64>();
    EXPECT_TRUE(res);
}
} // namespace TRowSumTest
