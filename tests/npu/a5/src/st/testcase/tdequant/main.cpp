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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <typename dstType, typename srcType, int kTRows_, int kTCols_, int vRows, int vCols>
void launchTDequant(dstType* out, srcType* src, dstType* scale, dstType* offset, void* stream);

class TDEQUANTTest : public testing::Test {
public:
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

template <typename dstType, typename srcType, int kTRows_, int kTCols_, int vRows, int vCols>
void test_tdequant()
{
    size_t dstFileSize = kTRows_ * kTCols_ * sizeof(dstType);
    size_t srcFileSize = kTRows_ * kTCols_ * sizeof(srcType);
    size_t paraFileSize = kTRows_ * 1 * sizeof(dstType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    dstType *dstHost, *scaleHost, *offsetHost;
    dstType *dstDevice, *scaleDevice, *offsetDevice;
    srcType* srcHost;
    srcType* srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&scaleHost), paraFileSize);
    aclrtMallocHost((void**)(&offsetHost), paraFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scaleDevice, paraFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&offsetDevice, paraFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    ReadFile(GetGoldenDir() + "/scale.bin", paraFileSize, scaleHost, paraFileSize);
    ReadFile(GetGoldenDir() + "/offset.bin", paraFileSize, offsetHost, paraFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDevice, paraFileSize, scaleHost, paraFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, paraFileSize, offsetHost, paraFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTDequant<dstType, srcType, kTRows_, kTCols_, vRows, vCols>(
        dstDevice, srcDevice, scaleDevice, offsetDevice, stream);

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

    std::vector<dstType> golden(dstFileSize);
    std::vector<dstType> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<dstType>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TDEQUANTTest, case1) { test_tdequant<float, int16_t, 64, 64, 64, 64>(); }

TEST_F(TDEQUANTTest, case2) { test_tdequant<float, int16_t, 128, 128, 64, 64>(); }

TEST_F(TDEQUANTTest, case3) { test_tdequant<float, int16_t, 128, 128, 63, 63>(); }

TEST_F(TDEQUANTTest, case4) { test_tdequant<float, int8_t, 64, 64, 64, 64>(); }

TEST_F(TDEQUANTTest, case5) { test_tdequant<float, int8_t, 128, 128, 64, 64>(); }

TEST_F(TDEQUANTTest, case6) { test_tdequant<float, int8_t, 128, 128, 63, 63>(); }

TEST_F(TDEQUANTTest, case7) { test_tdequant<float, int16_t, 51, 112, 51, 112>(); }