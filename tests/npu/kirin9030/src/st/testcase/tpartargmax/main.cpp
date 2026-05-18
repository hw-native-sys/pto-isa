/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

template <typename T, typename U, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR,
          int dstTC, int src0TR, int src0TC, int src1TR, int src1TC, bool isHalf>
void LaunchTPartArgMax(T *outVal, T *src0Val, T *src1Val, U *outIdx, U *src0Idx, U *src1Idx, void *stream);

class TPARTARGMAXTest : public testing::Test {
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

template <typename T, typename U, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR,
          int dstTC, int src0TR, int src0TC, int src1TR, int src1TC, bool isHalf = false>
void test_tpartargmax()
{
    size_t src0ValFileSize = src0TR * src0TC * sizeof(T);
    size_t src1ValFileSize = src1TR * src1TC * sizeof(T);
    size_t dstValFileSize = dstTR * dstTC * sizeof(T);
    size_t src0IdxFileSize = src0TR * src0TC * sizeof(U);
    size_t src1IdxFileSize = src1TR * src1TC * sizeof(U);
    size_t dstIdxFileSize = dstTR * dstTC * sizeof(U);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;
    U *dstIdxHost, *src0IdxHost, *src1IdxHost;
    U *dstIdxDevice, *src0IdxDevice, *src1IdxDevice;

    aclrtMallocHost((void **)(&dstHost), dstValFileSize);
    aclrtMallocHost((void **)(&src0Host), src0ValFileSize);
    aclrtMallocHost((void **)(&src1Host), src1ValFileSize);
    aclrtMallocHost((void **)(&dstIdxHost), dstIdxFileSize);
    aclrtMallocHost((void **)(&src0IdxHost), src0IdxFileSize);
    aclrtMallocHost((void **)(&src1IdxHost), src1IdxFileSize);

    aclrtMalloc((void **)&dstDevice, dstValFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0ValFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1ValFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstIdxDevice, dstIdxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0IdxDevice, src0IdxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1IdxDevice, src1IdxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0_val.bin", src0ValFileSize, src0Host, src0ValFileSize);
    ReadFile(GetGoldenDir() + "/input1_val.bin", src1ValFileSize, src1Host, src1ValFileSize);
    ReadFile(GetGoldenDir() + "/input0_idx.bin", src0IdxFileSize, src0IdxHost, src0IdxFileSize);
    ReadFile(GetGoldenDir() + "/input1_idx.bin", src1IdxFileSize, src1IdxHost, src1IdxFileSize);

    aclrtMemcpy(src0Device, src0ValFileSize, src0Host, src0ValFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1ValFileSize, src1Host, src1ValFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0IdxDevice, src0IdxFileSize, src0IdxHost, src0IdxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1IdxDevice, src1IdxFileSize, src1IdxHost, src1IdxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPartArgMax<T, U, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR, src1TC,
                      isHalf>(dstDevice, src0Device, src1Device, dstIdxDevice, src0IdxDevice, src1IdxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstValFileSize, dstDevice, dstValFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstIdxHost, dstIdxFileSize, dstIdxDevice, dstIdxFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_val.bin", dstHost, dstValFileSize);
    WriteFile(GetGoldenDir() + "/output_idx.bin", dstIdxHost, dstIdxFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(dstIdxDevice);
    aclrtFree(src0IdxDevice);
    aclrtFree(src1IdxDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(dstIdxHost);
    aclrtFreeHost(src0IdxHost);
    aclrtFreeHost(src1IdxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden_val(dstValFileSize);
    std::vector<T> devFinal_val(dstValFileSize);
    std::vector<U> golden_idx(dstIdxFileSize);
    std::vector<U> devFinal_idx(dstIdxFileSize);
    ReadFile(GetGoldenDir() + "/golden_val.bin", dstValFileSize, golden_val.data(), dstValFileSize);
    ReadFile(GetGoldenDir() + "/output_val.bin", dstValFileSize, devFinal_val.data(), dstValFileSize);
    ReadFile(GetGoldenDir() + "/golden_idx.bin", dstIdxFileSize, golden_idx.data(), dstIdxFileSize);
    ReadFile(GetGoldenDir() + "/output_idx.bin", dstIdxFileSize, devFinal_idx.data(), dstIdxFileSize);

    bool ret = ResultCmp<T>(golden_val, devFinal_val, 0.001f);
    EXPECT_TRUE(ret);
    ret = ResultCmp<U>(golden_idx, devFinal_idx, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TPARTARGMAXTest, case_fp32_64x64_64x64_64x64)
{
    test_tpartargmax<float, uint32_t, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TPARTARGMAXTest, case_fp32_2x24_2x24_2x8)
{
    test_tpartargmax<float, int32_t, 2, 24, 2, 24, 2, 8, 4, 32, 3, 24, 2, 16>();
}
TEST_F(TPARTARGMAXTest, case_fp32_12x63_12x63_6x60)
{
    test_tpartargmax<float, int32_t, 12, 63, 12, 63, 6, 60, 12, 64, 12, 64, 6, 64>();
}
TEST_F(TPARTARGMAXTest, case_fp16_10x31_8x16_10x31)
{
    test_tpartargmax<aclFloat16, int16_t, 10, 31, 8, 16, 10, 31, 10, 32, 8, 32, 12, 32, true>();
}
TEST_F(TPARTARGMAXTest, case_fp16_5x33_5x33_5x30)
{
    test_tpartargmax<aclFloat16, uint16_t, 5, 33, 5, 33, 5, 30, 8, 48, 5, 48, 6, 48, true>();
}