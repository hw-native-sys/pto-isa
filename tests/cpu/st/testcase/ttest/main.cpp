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
#include "pto/common/type.hpp"
using namespace std;
using namespace PtoTestCommon;

class TTESTTest : public testing::Test {
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
bool LaunchTTest(T *src, int32_t cmpValue, pto::comm::WaitCmp cmp, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_ttest()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *srcHost;
    T *srcDevice;

    aclrtMallocHost((void **)(&srcHost), fileSize);

    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", fileSize, srcHost, fileSize));

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    int32_t cmpValue;
    std::string cmpFileName = GetGoldenDir() + "/cmp_file.bin";
    std::ifstream cmpFile(cmpFileName, std::ios::binary);

    cmpFile.read(reinterpret_cast<char *>(&cmpValue), 4);
    cmpFile.close();

    bool outputTest =
        LaunchTTest<T, kGRows_, kGCols_, kTRows_, kTCols_>(srcDevice, cmpValue, pto::comm::WaitCmp::LE, stream);

    aclrtSynchronizeStream(stream);

    aclrtFree(srcDevice);

    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool goldenValue;
    std::string goldenFileName = GetGoldenDir() + "/golden.bin";
    std::ifstream goldenFile(goldenFileName, std::ios::binary);

    goldenFile.read(reinterpret_cast<char *>(&goldenValue), 1);
    goldenFile.close();

    bool ret = goldenValue == outputTest;

    EXPECT_TRUE(ret);
}

TEST_F(TTESTTest, case1)
{
    test_ttest<int32_t, 64, 64, 64, 64>();
}
TEST_F(TTESTTest, case2)
{
    test_ttest<int32_t, 64, 64, 64, 64>();
}
TEST_F(TTESTTest, case3)
{
    test_ttest<int32_t, 64, 64, 64, 64>();
}
TEST_F(TTESTTest, case4)
{
    test_ttest<int32_t, 16, 256, 16, 256>();
}