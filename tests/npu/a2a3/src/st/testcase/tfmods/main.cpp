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
#include <acl/acl.h>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTFMODSTestCase(void *out, void *src, float scalar, aclrtStream stream);

class TFMODSTest : public testing::Test {
public:
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

template <uint32_t caseId, typename T, int dstTileRow, int dstTileCol, int row, int vaildRow, int col, int srcVaildCol>
void TFMODSTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;
    float scalar;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char *>(&scalar), 4);
    file.close();
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTFMODSTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize);
    std::vector<T> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f, 0, 1000, false, true);
    EXPECT_TRUE(ret);
}

TEST_F(TFMODSTest, case1)
{
    TFMODSTestFramework<1, float, 32, 64, 32, 32, 64, 64>();
}

TEST_F(TFMODSTest, case5)
{
    TFMODSTestFramework<5, float, 7, 448, 7, 7, 448, 448>();
}

TEST_F(TFMODSTest, case6)
{
    TFMODSTestFramework<6, float, 256, 16, 256, 256, 16, 16>();
}

TEST_F(TFMODSTest, case7)
{
    TFMODSTestFramework<7, float, 32, 128, 32, 32, 64, 64>();
}

TEST_F(TFMODSTest, case11)
{
    TFMODSTestFramework<11, float, 7, 512, 7, 7, 448, 448>();
}

TEST_F(TFMODSTest, case12)
{
    TFMODSTestFramework<12, float, 256, 32, 256, 256, 16, 16>();
}

TEST_F(TFMODSTest, case16)
{
    TFMODSTestFramework<16, float, 1, 8192, 1, 1, 8192, 8192>();
}