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
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopDirBoth(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcD, uint8_t* srcF, uint8_t* fifoMem,
    void* stream);

class TPushPopDirBothTest : public testing::Test {
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

template <typename T, int32_t key>
void TPushPopDirBothTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(T);
    size_t bFileSize = M * K * sizeof(T);
    size_t dFileSize = K * N * sizeof(T);
    size_t fFileSize = M * N * sizeof(T);
    size_t outFileSize = M * N * sizeof(T);
    size_t fifoFileSize = 2 * M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *outHost, *srcAHost, *srcBHost, *srcDHost, *srcFHost;
    uint8_t *outDevice, *srcADevice, *srcBDevice, *srcDDevice, *srcFDevice, *fifoMemDevice;

    aclrtMallocHost((void**)(&outHost), outFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);
    aclrtMallocHost((void**)(&srcDHost), dFileSize);
    aclrtMallocHost((void**)(&srcFHost), fFileSize);

    aclrtMalloc((void**)&outDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDDevice, dFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcFDevice, fFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/srcA_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/srcB_gm.bin", bFileSize, srcBHost, bFileSize);
    ReadFile(GetGoldenDir() + "/srcD_gm.bin", dFileSize, srcDHost, dFileSize);
    ReadFile(GetGoldenDir() + "/srcF_gm.bin", fFileSize, srcFHost, fFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDDevice, dFileSize, srcDHost, dFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcFDevice, fFileSize, srcFHost, fFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopDirBoth<key>(
        (uint8_t*)ffts, outDevice, srcADevice, srcBDevice, srcDDevice, srcFDevice, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outFileSize, outDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, outFileSize);

    aclrtFree(outDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcDDevice);
    aclrtFree(srcFDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(outHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcDHost);
    aclrtFreeHost(srcFHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outFileSize);
    std::vector<T> devFinal(outFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outFileSize, golden.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TPushPopDirBothTest, case1_float_dir_both) { TPushPopDirBothTestFunc<float, 1>(128, 64, 128); }

TEST_F(TPushPopDirBothTest, case2_float_dir_both_left_right) { TPushPopDirBothTestFunc<float, 2>(128, 64, 128); }
