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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision = false>
void LaunchTFModS(T *out, T *src, T scalar, void *stream);

template <int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision = false>
void LaunchTFModSHalf(aclFloat16 *out, aclFloat16 *src, aclFloat16 scalar, void *stream);

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

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool isHalf = false, bool highPrecision = false>
inline void TFModSTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = srcTileRow * srcTileCol * sizeof(T);
    size_t scalarByteSize = sizeof(T);
    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;
    T scalar;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/divider.bin", scalarByteSize, &scalar, scalarByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (isHalf) {
        LaunchTFModSHalf<dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>(
            dstDevice, srcDevice, scalar, stream);
    } else {
        LaunchTFModS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>(
            dstDevice, srcDevice, scalar, stream);
    }
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

    bool res = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(res);
}

TEST_F(TFMODSTest, case1)
{
    TFModSTestFramework<float, 32, 128, 32, 128, 32, 64>();
}

TEST_F(TFMODSTest, case2)
{
    TFModSTestFramework<aclFloat16, 63, 128, 63, 128, 63, 64, true>();
}

TEST_F(TFMODSTest, case3)
{
    TFModSTestFramework<int32_t, 31, 256, 31, 256, 31, 128>();
}

TEST_F(TFMODSTest, case4)
{
    TFModSTestFramework<int16_t, 15, 192, 15, 192, 15, 192>();
}

TEST_F(TFMODSTest, case5)
{
    TFModSTestFramework<float, 7, 512, 7, 512, 7, 448>();
}

TEST_F(TFMODSTest, case6)
{
    TFModSTestFramework<float, 256, 32, 256, 32, 256, 31>();
}

TEST_F(TFMODSTest, caseHP1)
{
    TFModSTestFramework<float, 64, 64, 64, 64, 64, 64, false, true>();
}

TEST_F(TFMODSTest, caseHP2)
{
    TFModSTestFramework<float, 64, 64, 64, 64, 64, 61, false, true>();
}
