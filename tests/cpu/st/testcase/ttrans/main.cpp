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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTTRANS(__gm__ T __out__* out, __gm__ T __in__* src);

class TTRANSTest : public testing::Test {
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

template <typename DType>
void LaunchTest()
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;

    size_t srcFileSize = M * N * sizeof(DType);
    size_t dstFileSize = M * N * sizeof(DType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    auto readResult = ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize);

    if (!readResult && std::string(testing::UnitTest::GetInstance()->current_test_info()->name()) == "case_hifloat8") {
        GTEST_SKIP() << "Skipping case hifloat8: "
                     << "Source data was not generated, probably because en_dtypes package is missing.";
        return;
    }

    CHECK_RESULT_GTEST(readResult);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    runTTRANS<DType, M, N, M, N>(
        reinterpret_cast<__gm__ DType*>(dstDevice), reinterpret_cast<__gm__ DType*>(srcDevice));

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DType> golden(dstFileSize);
    std::vector<DType> devFinal(dstFileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_z.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSTest, case_float32) { LaunchTest<float>(); }

TEST_F(TTRANSTest, case_hifloat8) { LaunchTest<hifloat8_t>(); }
