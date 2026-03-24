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

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTExtractNDVec(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

template <int32_t testKey>
void launchTExtractNDVecScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

class TExtractNDVecTest : public testing::Test {
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

using NdVecLaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);

template <typename dType>
void runTExtractNDVecTest(size_t srcByteSize, size_t dstByteSize, NdVecLaunchFn launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *outHost, *srcHost, *dstInitHost;
    uint8_t *outDevice, *srcDevice, *dstInitDevice;

    aclrtMallocHost((void **)(&outHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMallocHost((void **)(&dstInitHost), dstByteSize);

    aclrtMalloc((void **)&outDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstInitDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src_input.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/dst_init.bin", dstByteSize, dstInitHost, dstByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstInitDevice, dstByteSize, dstInitHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(outDevice, srcDevice, dstInitDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, dstByteSize, outDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, dstByteSize);

    aclrtFree(outDevice);
    aclrtFree(srcDevice);
    aclrtFree(dstInitDevice);
    aclrtFreeHost(outHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(dstInitHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    bool ret = ResultCmp(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

template <int32_t testKey, typename dType>
void testTExtractNDVec(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runTExtractNDVecTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                                launchTExtractNDVec<testKey>);
}

template <int32_t testKey, typename dType>
void testTExtractNDVecScalar(int32_t srcRows, int32_t srcCols)
{
    constexpr size_t minAlignedCols = 32 / sizeof(dType);
    runTExtractNDVecTest<dType>(srcRows * srcCols * sizeof(dType), 1 * minAlignedCols * sizeof(dType),
                                launchTExtractNDVecScalar<testKey>);
}

TEST_F(TExtractNDVecTest, case_nd_vec_1)
{
    testTExtractNDVec<1, float>(16, 16, 8, 8);
}

TEST_F(TExtractNDVecTest, case_nd_vec_2)
{
    testTExtractNDVec<2, float>(16, 16, 8, 8);
}

TEST_F(TExtractNDVecTest, case_nd_vec_3)
{
    testTExtractNDVec<3, uint16_t>(32, 32, 16, 16);
}

TEST_F(TExtractNDVecTest, case_nd_vec_4)
{
    testTExtractNDVec<4, int8_t>(64, 64, 32, 32);
}

TEST_F(TExtractNDVecTest, case_nd_vec_5)
{
    testTExtractNDVec<5, uint16_t>(32, 48, 16, 16);
}

TEST_F(TExtractNDVecTest, case_nd_vec_6)
{
    testTExtractNDVec<6, float>(16, 24, 8, 8);
}

TEST_F(TExtractNDVecTest, case_nd_vec_7)
{
    testTExtractNDVec<7, float>(16, 24, 8, 8);
}

TEST_F(TExtractNDVecTest, case_nd_vec_8)
{
    testTExtractNDVec<8, uint16_t>(16, 48, 8, 16);
}

TEST_F(TExtractNDVecTest, case_nd_vec_9)
{
    testTExtractNDVec<9, int8_t>(64, 64, 32, 32);
}

TEST_F(TExtractNDVecTest, case_nd_vec_10)
{
    testTExtractNDVecScalar<1, float>(16, 16);
}

TEST_F(TExtractNDVecTest, case_nd_vec_11)
{
    testTExtractNDVecScalar<2, uint16_t>(32, 32);
}

TEST_F(TExtractNDVecTest, case_nd_vec_12)
{
    testTExtractNDVecScalar<3, int8_t>(64, 64);
}
