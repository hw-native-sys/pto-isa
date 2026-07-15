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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TLOGTest : public testing::Test {
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

template <
    typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace = false,
    bool highPrecision = false>
void LaunchTLog(T* out, T* src, void* stream);

template <
    typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace = false,
    bool highPrecision = false>
void test_tlog()
{
    size_t dstSize = dstRow * dstCol * sizeof(T);
    size_t srcSize = srcRow * srcCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstSize);
    aclrtMallocHost((void**)(&srcHost), srcSize);

    aclrtMalloc((void**)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcSize, srcHost, srcSize);

    aclrtMemcpy(srcDevice, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTLog<T, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace, highPrecision>(
        dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstSize);
    std::vector<T> devFinal(dstSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstSize, golden.data(), dstSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstSize, devFinal.data(), dstSize);

    float eps = 0.0f;
    if constexpr (highPrecision) {
        eps = 0.000001f;
    } else if constexpr (std::is_same_v<T, float>) {
        eps = 0.0001f;
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        eps = 0.001f;
    }
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TLOGTest, case_float_64x64_64x64_64x64_inPlace) { test_tlog<float, 64, 64, 64, 64, 64, 64, true>(); }
TEST_F(TLOGTest, case_float_64x64_64x64_64x64) { test_tlog<float, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TLOGTest, case_half_64x64_64x64_64x64_inPlace) { test_tlog<aclFloat16, 64, 64, 64, 64, 64, 64, true>(); }
TEST_F(TLOGTest, case_half_64x64_64x64_64x64) { test_tlog<aclFloat16, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TLOGTest, case_float_hp_64x64_64x64_64x64) { test_tlog<float, 64, 64, 64, 64, 64, 64, false, true>(); }
TEST_F(TLOGTest, case_half_hp_64x64_64x64_64x64) { test_tlog<aclFloat16, 64, 64, 64, 64, 64, 64, false, true>(); }
