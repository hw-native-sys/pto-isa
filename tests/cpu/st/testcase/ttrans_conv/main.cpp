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

template <typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
          int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5, int groupN>
void LaunchTTRANSConv(T *out, T *src, void *stream);

template <typename MXType, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
          int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5, int groupN>
void LaunchTTRANSConvMX(uint8_t *out, uint8_t *src, void *stream);

class TTRANSConvTest : public testing::Test {
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

template <typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
          int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5 = 1, int groupN = 1>
void test_ttrans()
{
    size_t srcFileSize = srcShape0 * srcShape1 * srcShape2 * srcShape3 * srcShape4 * groupN * sizeof(T);
    size_t dstFileSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstShape4 * dstShape5 * groupN * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConv<T, format, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2,
                     dstShape3, dstShape4, dstShape5, groupN>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename MXType, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
          int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5 = 1, int groupN = 1>
void test_ttrans_MX()
{
    size_t srcFileSize = srcShape0 * srcShape1 * srcShape2 * srcShape3 * srcShape4 * groupN * sizeof(uint8_t);
    size_t dstFileSize =
        dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstShape4 * dstShape5 * groupN * sizeof(uint8_t);
    if (isTwinType<MXType>()) {
        srcFileSize /= 2;
        dstFileSize /= 2;
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize * 2);
    aclrtMallocHost((void **)(&srcHost), srcFileSize * 2);

    aclrtMalloc((void **)&dstDevice, dstFileSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize * 2, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize * 2, srcHost, srcFileSize * 2, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTTRANSConvMX<MXType, format, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1,
                       dstShape2, dstShape3, dstShape4, dstShape5, groupN>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize * 2, dstDevice, dstFileSize * 2, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(dstFileSize / sizeof(uint8_t), 0);
    std::vector<uint8_t> result(dstFileSize / sizeof(uint8_t), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<uint8_t>(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_1)
{
    test_ttrans<float, 0, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_2)
{
    test_ttrans<int32_t, 0, 5, 14, 13, 16, 1, 5, 2, 13, 16, 8>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_3)
{
    test_ttrans<uint16_t, 0, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_4)
{
    test_ttrans<int32_t, 0, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_5)
{
    test_ttrans<int8_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32>();
}

TEST_F(TTRANSConvTest, NCHW2NC1HWC0_MX_e8m0)
{
    test_ttrans_MX<float8_e8m0_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32>();
}

/*------------------------------------------------*/

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_1)
{
    test_ttrans<float, 1, 25, 4, 3, 8, 8, 4, 3, 8, 2, 16, 8>();
}

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_2)
{
    test_ttrans<int32_t, 1, 15, 2, 3, 16, 8, 2, 3, 16, 2, 8, 8>();
}

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_3)
{
    test_ttrans<uint16_t, 1, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16>();
}

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_4)
{
    test_ttrans<int32_t, 1, 4, 32, 3, 7, 8, 32, 3, 7, 1, 4, 8>();
}

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_5)
{
    test_ttrans<int8_t, 1, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32>();
}

TEST_F(TTRANSConvTest, NC1HWC02C1HWN1N0C0_MX_e4m3)
{
    test_ttrans_MX<float8_e4m3_t, 2, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32>();
}

/*------------------------------------------------*/

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_1)
{
    test_ttrans<float, 2, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1, 4>();
}

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_2)
{
    test_ttrans<int32_t, 2, 5, 14, 13, 8, 1, 5, 2, 13, 8, 8, 1, 2>();
}

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_3)
{
    test_ttrans<uint16_t, 2, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1, 3>();
}

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_4)
{
    test_ttrans<int32_t, 2, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1, 1>();
}

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_5)
{
    test_ttrans<int8_t, 2, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 3>();
}

// TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_MX_e2m1)
// {
//     test_ttrans_MX<float4_e1m2x2_t, 3, 4, 64, 3, 7, 1, 4, 1, 3, 7, 64, 1, 3>();
// }

TEST_F(TTRANSConvTest, GNCHW2GNC1HWC0_MXFP4_e2m1)
{
    test_ttrans_MX<float4_e2m1x2_t, 3, 4, 64, 3, 14, 1, 4, 1, 3, 14, 64, 1, 1>();
}

/*------------------------------------------*/

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_1)
{
    test_ttrans<float, 3, 25, 4, 3, 4, 8, 4, 3, 4, 2, 16, 8, 2>();
}

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_2)
{
    test_ttrans<int32_t, 3, 15, 2, 3, 4, 8, 2, 3, 4, 2, 8, 8, 3>();
}

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_3)
{
    test_ttrans<uint16_t, 3, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16, 2>();
}

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_4)
{
    test_ttrans<int32_t, 3, 4, 8, 3, 7, 8, 8, 3, 7, 1, 4, 8, 3>();
}

TEST_F(TTRANSConvTest, GNC1HWC02C1HWN1N0C0_5)
{
    test_ttrans<int8_t, 3, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1>();
}