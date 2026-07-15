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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int dstDC1HW, int dstN1, int dstN0, int dstC0, int srcN, int srcC, int srcD, int srcH, int srcW>
void LaunchTTRANS3D(T* out, T* src, void* stream);

class TTRANS3DTest : public testing::Test {
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

template <typename T, int dstDC1HW, int dstN1, int dstN0, int dstC0, int srcN, int srcC, int srcD, int srcH, int srcW>
void test_ttrans_3d()
{
    size_t srcFileSize = static_cast<size_t>(srcN) * srcC * srcD * srcH * srcW * sizeof(T);
    size_t dstFileSize = static_cast<size_t>(dstDC1HW) * dstN1 * dstN0 * dstC0 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANS3D<T, dstDC1HW, dstN1, dstN0, dstC0, srcN, srcC, srcD, srcH, srcW>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANS3DTest, case1_float32_2_4_2_2_2) { test_ttrans_3d<float, 8, 1, 16, 8, 2, 4, 2, 2, 2>(); }

TEST_F(TTRANS3DTest, case2_float32_4_5_2_2_4) { test_ttrans_3d<float, 16, 1, 16, 8, 4, 5, 2, 2, 4>(); }

TEST_F(TTRANS3DTest, case3_int32_17_3_3_2_2) { test_ttrans_3d<int32_t, 12, 2, 16, 8, 17, 3, 3, 2, 2>(); }

TEST_F(TTRANS3DTest, case4_half_5_6_2_2_4) { test_ttrans_3d<aclFloat16, 16, 1, 16, 16, 5, 6, 2, 2, 4>(); }

TEST_F(TTRANS3DTest, case5_half_19_14_2_4_2) { test_ttrans_3d<aclFloat16, 16, 2, 16, 16, 19, 14, 2, 4, 2>(); }

TEST_F(TTRANS3DTest, case6_int16_8_13_2_3_4) { test_ttrans_3d<int16_t, 24, 1, 16, 16, 8, 13, 2, 3, 4>(); }

TEST_F(TTRANS3DTest, case7_uint16_4_8_2_2_3) { test_ttrans_3d<uint16_t, 12, 1, 16, 16, 4, 8, 2, 2, 3>(); }

TEST_F(TTRANS3DTest, case8_int8_7_28_2_3_4) { test_ttrans_3d<int8_t, 24, 1, 16, 32, 7, 28, 2, 3, 4>(); }

TEST_F(TTRANS3DTest, case9_int8_16_26_3_3_4) { test_ttrans_3d<int8_t, 36, 1, 16, 32, 16, 26, 3, 3, 4>(); }

TEST_F(TTRANS3DTest, case10_uint8_9_18_2_2_4) { test_ttrans_3d<uint8_t, 16, 1, 16, 32, 9, 18, 2, 2, 4>(); }
