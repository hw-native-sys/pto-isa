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
void launchTExtractVecND(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

template <int32_t testKey>
void launchTExtractVecNDScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

template <int32_t testKey>
void launchTExtractVecNZ(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

template <int32_t testKey>
void launchTExtractVecNZScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream);

class TExtractVecTest : public testing::Test {
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
    return "../" + suiteName + "." + caseName;
}

using NdLaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);
using NzLaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);

template <typename dType>
void runNDTest(size_t srcByteSize, size_t dstByteSize, NdLaunchFn launch)
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

template <typename dType>
void runNZTest(size_t srcByteSize, size_t dstByteSize, NzLaunchFn launch)
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

template <int32_t TestKey, typename dType>
void testND(int32_t srcRows, int32_t srcCols, int32_t dstStaticRows, int32_t dstStaticCols)
{
    runNDTest<dType>(srcRows * srcCols * sizeof(dType), dstStaticRows * dstStaticCols * sizeof(dType),
                     launchTExtractVecND<TestKey>);
}

template <int32_t TestKey, typename dType>
void testNDScalar(int32_t srcRows, int32_t srcCols)
{
    constexpr size_t MinAligned = 32 / sizeof(dType);
    runNDTest<dType>(srcRows * srcCols * sizeof(dType), 1 * MinAligned * sizeof(dType),
                     launchTExtractVecNDScalar<TestKey>);
}

template <int32_t TestKey, typename dType>
void testNZ(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runNZTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                     launchTExtractVecNZ<TestKey>);
}

template <int32_t TestKey, typename dType>
void testNZScalar(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runNDTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                     launchTExtractVecNZScalar<TestKey>);
}

TEST_F(TExtractVecTest, case_nd_aligned_1)
{
    testND<1, float>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_aligned_2)
{
    testND<2, float>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_aligned_3)
{
    testND<3, uint16_t>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_4)
{
    testND<4, uint16_t>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_5)
{
    testND<5, int32_t>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_aligned_6)
{
    testND<6, int8_t>(64, 64, 32, 32);
}

TEST_F(TExtractVecTest, case_nd_unaligned_validcol_1)
{
    testND<7, float>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_unaligned_validcol_2)
{
    testND<8, uint16_t>(16, 32, 8, 16);
}

TEST_F(TExtractVecTest, case_nd_unaligned_indexcol_1)
{
    testND<9, float>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_unaligned_indexcol_2)
{
    testND<10, uint16_t>(16, 48, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unaligned_indexcol_3)
{
    testND<11, int8_t>(64, 64, 32, 32);
}
TEST_F(TExtractVecTest, case_nd_unaligned_validcol_3)
{
    testND<12, int8_t>(64, 64, 32, 32);
}

TEST_F(TExtractVecTest, case_nd_aligned_hif8)
{
    testND<13, uint8_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_aligned_fp8_e4m3)
{
    testND<14, uint8_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_aligned_fp8_e5m2)
{
    testND<15, uint8_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_partial_validrow)
{
    testND<16, uint16_t>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_fp4_e2m1)
{
    testND<17, uint8_t>(16, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_aligned_fp4_e1m2)
{
    testND<18, uint8_t>(16, 64, 16, 32);
}

TEST_F(TExtractVecTest, case_nd_scalar_1)
{
    testNDScalar<1, float>(16, 16);
}
TEST_F(TExtractVecTest, case_nd_scalar_2)
{
    testNDScalar<2, uint16_t>(32, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_3)
{
    testNDScalar<3, uint16_t>(32, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_4)
{
    testNDScalar<4, int8_t>(64, 64);
}
TEST_F(TExtractVecTest, case_nd_scalar_5)
{
    testNDScalar<5, int32_t>(16, 16);
}
TEST_F(TExtractVecTest, case_nd_scalar_fp4_e2m1)
{
    testNDScalar<6, uint8_t>(16, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_fp4_e1m2)
{
    testNDScalar<7, uint8_t>(16, 32);
}

TEST_F(TExtractVecTest, case_nz_1)
{
    testNZ<1, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_2)
{
    testNZ<2, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_3)
{
    testNZ<3, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_4)
{
    testNZ<4, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_5)
{
    testNZ<5, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_6)
{
    testNZ<6, int8_t>(32, 64, 16, 64);
}

TEST_F(TExtractVecTest, case_nz_indexcol_nonzero)
{
    testNZ<7, int8_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_partial_valid)
{
    testNZ<8, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_multi_fractal_dst)
{
    testNZ<9, uint16_t>(64, 32, 32, 32);
}
TEST_F(TExtractVecTest, case_nz_hif8)
{
    testNZ<10, uint8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_fp8_e4m3)
{
    testNZ<11, uint8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_fp8_e5m2)
{
    testNZ<12, uint8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_int32)
{
    testNZ<13, int32_t>(32, 16, 16, 8);
}

TEST_F(TExtractVecTest, case_nz_scalar_1)
{
    testNZScalar<1, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_2)
{
    testNZScalar<2, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_3)
{
    testNZScalar<3, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_4)
{
    testNZScalar<4, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_scalar_5)
{
    testNZScalar<5, int32_t>(32, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_scalar_fp4_e2m1)
{
    testNZScalar<6, uint8_t>(16, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_scalar_fp4_e1m2)
{
    testNZScalar<7, uint8_t>(16, 64, 16, 64);
}
