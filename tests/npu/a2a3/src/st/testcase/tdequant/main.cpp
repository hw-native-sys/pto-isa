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

#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

class TDEQUANTTest : public testing::Test {
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

template <typename dstDType, typename srcDType, int dstRows, int dstCols, int srcRows, int srcCols, int dstValidRows,
          int dstValidCols, int paraRows, int paraCols>
void LaunchTDequant(dstDType *out, srcDType *src, dstDType *scale, dstDType *offset, void *stream);

template <typename dstDType, typename srcDType, int dstRows, int dstCols, int srcRows, int srcCols, int dstValidRows,
          int dstValidCols, int paraRows, int paraCols>
void test_tdequant()
{
    size_t dstFileSize = dstRows * dstCols * sizeof(dstDType);
    size_t srcFileSize = srcRows * srcCols * sizeof(srcDType);
    size_t paraFileSize = paraRows * paraCols * sizeof(dstDType);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    srcDType *srcHost;
    srcDType *srcDevice;
    dstDType *dstHost, *scaleHost, *offsetHost;
    dstDType *dstDevice, *scaleDevice, *offsetDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&scaleHost), paraFileSize);
    aclrtMallocHost((void **)(&offsetHost), paraFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDevice, paraFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&offsetDevice, paraFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/srcInput.bin", srcFileSize, srcHost, srcFileSize);
    ReadFile(GetGoldenDir() + "/scaleInput.bin", paraFileSize, scaleHost, paraFileSize);
    ReadFile(GetGoldenDir() + "/offsetInput.bin", paraFileSize, offsetHost, paraFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDevice, paraFileSize, scaleHost, paraFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, paraFileSize, offsetHost, paraFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTDequant<dstDType, srcDType, dstRows, dstCols, srcRows, srcCols, dstValidRows, dstValidCols, paraRows,
                   paraCols>(dstDevice, srcDevice, scaleDevice, offsetDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(scaleDevice);
    aclrtFree(offsetDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(scaleHost);
    aclrtFreeHost(offsetHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dstDType> golden(dstFileSize);
    std::vector<dstDType> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<dstDType>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TDEQUANTTest, case1)
{
    test_tdequant<float, int8_t, 32, 32, 32, 32, 32, 32, 32, 32>();
}

TEST_F(TDEQUANTTest, case2)
{
    test_tdequant<float, int16_t, 32, 32, 32, 32, 32, 32, 32, 32>();
}

TEST_F(TDEQUANTTest, case3)
{
    test_tdequant<float, int8_t, 64, 64, 32, 64, 31, 31, 48, 32>();
}

TEST_F(TDEQUANTTest, case4)
{
    test_tdequant<float, int16_t, 32, 32, 16, 32, 15, 15, 24, 16>();
}

TEST_F(TDEQUANTTest, case5)
{
    test_tdequant<float, int8_t, 64, 128, 32, 128, 31, 62, 48, 32>();
}

TEST_F(TDEQUANTTest, case6)
{
    test_tdequant<float, int16_t, 4, 256, 4, 256, 4, 255, 4, 16>();
}