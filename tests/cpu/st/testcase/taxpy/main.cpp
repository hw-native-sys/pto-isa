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
void launchTAXPYTestCase(void *out, void *src, float scalar, aclrtStream stream);

class TAXPYTest : public testing::Test {
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

template <typename T, int oRow, int oCol>
inline void InitDstDevice(T *dstDevice)
{
    constexpr int size = oRow * oCol;
    for (int k = 0; k < size; k++) {
        dstDevice[k] = T{0};
    }
}

template <uint32_t caseId, typename T, int validRow, int validCol, int iRow = validRow, int iCol = validCol,
          int oRow = validRow, int oCol = validCol>
bool TAxpyTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = oRow * oCol * sizeof(T);
    size_t srcByteSize = iRow * iCol * sizeof(T);
    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;
    float scalar;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_dst.bin", dstByteSize, dstHost, dstByteSize);
    ReadFile(GetGoldenDir() + "/input_src.bin", srcByteSize, srcHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char *>(&scalar), 4);
    file.close();
    aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTAXPYTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
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

    return ResultCmp<T>(golden, devFinal, 0.001f);
}

TEST_F(TAXPYTest, case1)
{
    bool ret = TAxpyTestFramework<1, float, 32, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TAXPYTest, case2)
{
    bool ret = TAxpyTestFramework<2, aclFloat16, 63, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TAXPYTest, case3)
{
    bool ret = TAxpyTestFramework<3, float, 7, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TAXPYTest, case4)
{
    bool ret = TAxpyTestFramework<4, float, 256, 16>();
    EXPECT_TRUE(ret);
}

TEST_F(TAXPYTest, case5)
{
    bool ret = TAxpyTestFramework<5, float, 16, 16, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}
