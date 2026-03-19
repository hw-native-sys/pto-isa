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

template <uint32_t caseId>
void launchTROWPRODTestCase(void *out, void *src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TROWPRODTest : public testing::Test {
public:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <typename T>
    bool CompareGolden(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<T> golden(dstByteSize);
        std::vector<T> result(dstByteSize);
        float eps = sizeof(T) == 4 ? 0.001f : 0.005f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <uint32_t caseId, typename T, int dstRow, int srcRow, int validRow, int srcCol, int srcValidCol>
    void TRowProdTestFramework()
    {
        size_t dstByteSize = dstRow * sizeof(T);
        size_t srcByteSize = srcRow * srcCol * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTROWPRODTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        bool ret = CompareGolden<T>(dstByteSize);
        EXPECT_TRUE(ret);
    }
};

TEST_F(TROWPRODTest, case1)
{
    TRowProdTestFramework<1, float, 8, 1, 1, 8, 8>();
}

TEST_F(TROWPRODTest, case2)
{
    TRowProdTestFramework<2, float, 8, 1, 1, 16, 16>();
}

TEST_F(TROWPRODTest, case3)
{
    TRowProdTestFramework<3, float, 8, 1, 1, 128, 128>();
}

TEST_F(TROWPRODTest, case4)
{
    TRowProdTestFramework<4, float, 8, 1, 1, 8, 5>();
}

TEST_F(TROWPRODTest, case5)
{
    TRowProdTestFramework<5, float, 8, 1, 1, 16, 11>();
}

TEST_F(TROWPRODTest, case6)
{
    TRowProdTestFramework<6, float, 8, 3, 2, 8, 8>();
}

TEST_F(TROWPRODTest, case7)
{
    TRowProdTestFramework<7, float, 8, 3, 2, 24, 16>();
}

TEST_F(TROWPRODTest, case8)
{
    TRowProdTestFramework<8, float, 8, 4, 3, 16, 9>();
}

TEST_F(TROWPRODTest, case9)
{
    TRowProdTestFramework<9, __fp16, 16, 1, 1, 16, 16>();
}

TEST_F(TROWPRODTest, case10)
{
    TRowProdTestFramework<10, __fp16, 32, 26, 19, 32, 26>();
}

TEST_F(TROWPRODTest, case11)
{
    TRowProdTestFramework<11, int32_t, 8, 1, 1, 8, 8>();
}

TEST_F(TROWPRODTest, case12)
{
    TRowProdTestFramework<12, int32_t, 8, 1, 1, 16, 16>();
}

TEST_F(TROWPRODTest, case13)
{
    TRowProdTestFramework<13, int32_t, 8, 1, 1, 128, 128>();
}

TEST_F(TROWPRODTest, case14)
{
    TRowProdTestFramework<14, int32_t, 8, 1, 1, 8, 5>();
}

TEST_F(TROWPRODTest, case15)
{
    TRowProdTestFramework<15, int32_t, 8, 3, 2, 24, 16>();
}

TEST_F(TROWPRODTest, case16)
{
    TRowProdTestFramework<16, int16_t, 16, 1, 1, 16, 16>();
}

TEST_F(TROWPRODTest, case17)
{
    TRowProdTestFramework<17, int16_t, 32, 26, 19, 32, 32>();
}

TEST_F(TROWPRODTest, case18)
{
    TRowProdTestFramework<18, int16_t, 16, 1, 1, 16, 16>();
}
