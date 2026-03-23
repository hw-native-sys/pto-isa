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
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTADDSTestCase(void *out, void *src, float scalar, aclrtStream stream);

class TADDSTest : public testing::Test {
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

template <uint32_t caseId, typename T, int row, int vaildRow, int col, int srcVaildCol, float profiling, float accuracy>
bool TAddSTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = row * col * sizeof(T);
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

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTADDSTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return true;
}

TEST_F(TADDSTest, case1)
{
    bool ret = TAddSTestFramework<1, float, 32, 32, 64, 64, 46.0f, 1.0f>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDSTest, case2)
{
    bool ret = TAddSTestFramework<2, aclFloat16, 63, 63, 64, 64, 77.0f, 1.0f>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDSTest, case3)
{
    bool ret = TAddSTestFramework<3, int32_t, 31, 31, 128, 128, 76.0f, 1.0f>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDSTest, case4)
{
    bool ret = TAddSTestFramework<4, int16_t, 15, 15, 192, 192, 44.0f, 1.0f>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDSTest, case5)
{
    bool ret = TAddSTestFramework<5, float, 7, 7, 448, 448, 63.0f, 1.0f>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDSTest, case6)
{
    bool ret = TAddSTestFramework<6, float, 256, 256, 16, 16, 270.0f, 1.0f>();
    EXPECT_TRUE(ret);
}
