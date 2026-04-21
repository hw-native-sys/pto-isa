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

using LaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);

template <typename dType>
void runVecTest(size_t srcByteSize, size_t dstByteSize, LaunchFn launch)
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
    runVecTest<dType>(srcRows * srcCols * sizeof(dType), dstStaticRows * dstStaticCols * sizeof(dType),
                      launchTExtractVecND<TestKey>);
}

template <int32_t TestKey, typename dType>
void testNDScalar(int32_t srcRows, int32_t srcCols)
{
    constexpr size_t MinAligned = 32 / sizeof(dType);
    runVecTest<dType>(srcRows * srcCols * sizeof(dType), 1 * MinAligned * sizeof(dType),
                      launchTExtractVecNDScalar<TestKey>);
}

template <int32_t TestKey, typename dType>
void testNZ(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runVecTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                      launchTExtractVecNZ<TestKey>);
}

template <int32_t TestKey, typename dType>
void testNZScalar(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runVecTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
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
TEST_F(TExtractVecTest, case_nd_aligned_3_half)
{
    testND<3, aclFloat16>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_4_bf16)
{
    testND<4, uint16_t>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_5_int32)
{
    testND<5, int32_t>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_aligned_6_int8)
{
    testND<6, int8_t>(64, 64, 32, 32);
}
TEST_F(TExtractVecTest, case_nd_partial_validrow)
{
    testND<7, aclFloat16>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_unaligned_validcol_full_row)
{
    testND<8, float>(16, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_bf16_2)
{
    testND<9, uint16_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_aligned_uint8)
{
    testND<10, uint8_t>(64, 64, 32, 32);
}
TEST_F(TExtractVecTest, case_nd_aligned_int16)
{
    testND<11, int16_t>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_uint16)
{
    testND<12, uint16_t>(32, 32, 16, 16);
}
TEST_F(TExtractVecTest, case_nd_aligned_uint32)
{
    testND<13, uint32_t>(16, 16, 8, 8);
}
TEST_F(TExtractVecTest, case_nd_partial_validboth_float)
{
    testND<14, float>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_aligned_int8_strided)
{
    testND<15, int8_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_partial_validrowonly_half)
{
    testND<16, aclFloat16>(64, 64, 32, 32);
}
TEST_F(TExtractVecTest, case_nd_nonpow2_float)
{
    testND<17, float>(24, 48, 12, 32);
}
TEST_F(TExtractVecTest, case_nd_nonpow2_half)
{
    testND<18, aclFloat16>(30, 80, 14, 48);
}
TEST_F(TExtractVecTest, case_nd_nonpow2_int8)
{
    testND<19, int8_t>(48, 96, 24, 64);
}
TEST_F(TExtractVecTest, case_nd_nonpow2_partial_bf16)
{
    testND<20, uint16_t>(24, 48, 12, 32);
}
TEST_F(TExtractVecTest, case_nd_nonpow2_int32)
{
    testND<21, int32_t>(20, 16, 10, 8);
}
TEST_F(TExtractVecTest, case_nd_nonpow2_partial_float_unaligned_idxrow)
{
    testND<22, float>(24, 48, 12, 32);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_float)
{
    testND<23, float>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_half)
{
    testND<24, aclFloat16>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_bf16)
{
    testND<25, uint16_t>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_int16)
{
    testND<26, int16_t>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_int32)
{
    testND<27, int32_t>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_uint16_strided)
{
    testND<28, uint16_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_int8_even)
{
    testND<29, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_float_with_idx)
{
    testND<30, float>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_uint16_taillarge)
{
    testND<31, uint16_t>(32, 32, 8, 16);
}
TEST_F(TExtractVecTest, case_nd_unalignedvalid_float_smallthan32B)
{
    testND<32, float>(32, 32, 8, 16);
}

TEST_F(TExtractVecTest, case_nd_scalar_1_float)
{
    testNDScalar<1, float>(16, 16);
}
TEST_F(TExtractVecTest, case_nd_scalar_2_half)
{
    testNDScalar<2, aclFloat16>(32, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_3_bf16)
{
    testNDScalar<3, uint16_t>(32, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_4_int8)
{
    testNDScalar<4, int8_t>(64, 64);
}
TEST_F(TExtractVecTest, case_nd_scalar_5_int32)
{
    testNDScalar<5, int32_t>(16, 16);
}
TEST_F(TExtractVecTest, case_nd_scalar_6_int16)
{
    testNDScalar<6, int16_t>(32, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_7_uint16)
{
    testNDScalar<7, uint16_t>(32, 32);
}
TEST_F(TExtractVecTest, case_nd_scalar_8_uint32)
{
    testNDScalar<8, uint32_t>(16, 16);
}
TEST_F(TExtractVecTest, case_nd_scalar_nonpow2_half)
{
    testNDScalar<9, aclFloat16>(30, 80);
}
TEST_F(TExtractVecTest, case_nd_scalar_nonpow2_int8)
{
    testNDScalar<10, int8_t>(48, 96);
}
TEST_F(TExtractVecTest, case_nd_scalar_nonpow2_float)
{
    testNDScalar<11, float>(24, 48);
}

TEST_F(TExtractVecTest, case_nz_1)
{
    testNZ<1, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_2)
{
    testNZ<2, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_3_half)
{
    testNZ<3, aclFloat16>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_4_bf16)
{
    testNZ<4, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_5_int8)
{
    testNZ<5, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_6_int8)
{
    testNZ<6, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_multi_fractal_dst)
{
    testNZ<7, aclFloat16>(64, 32, 32, 32);
}
TEST_F(TExtractVecTest, case_nz_partial_int8)
{
    testNZ<8, int8_t>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_partial_bf16)
{
    testNZ<9, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_uint8)
{
    testNZ<10, uint8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_int32_partial)
{
    testNZ<11, int32_t>(32, 16, 16, 8);
}
TEST_F(TExtractVecTest, case_nz_int16)
{
    testNZ<12, int16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_uint16)
{
    testNZ<13, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_uint32)
{
    testNZ<14, uint32_t>(32, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_half_large)
{
    testNZ<15, aclFloat16>(64, 64, 32, 64);
}
TEST_F(TExtractVecTest, case_nz_partial_float_unaligned_idxcol)
{
    testNZ<16, float>(64, 32, 32, 16);
}
TEST_F(TExtractVecTest, case_nz_bf16_large)
{
    testNZ<17, uint16_t>(64, 64, 32, 32);
}
TEST_F(TExtractVecTest, case_nz_nonpow2_half)
{
    testNZ<18, aclFloat16>(48, 32, 32, 32);
}
TEST_F(TExtractVecTest, case_nz_nonpow2_float)
{
    testNZ<19, float>(48, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_nonpow2_partial_bf16)
{
    testNZ<20, uint16_t>(48, 48, 32, 32);
}
TEST_F(TExtractVecTest, case_nz_nonpow2_partial_int8)
{
    testNZ<21, int8_t>(48, 96, 32, 64);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_float)
{
    testNZ<22, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_half)
{
    testNZ<23, aclFloat16>(32, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_bf16)
{
    testNZ<24, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_int16)
{
    testNZ<25, int16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_int32)
{
    testNZ<26, int32_t>(32, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_int8_even)
{
    testNZ<27, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_validrow_half)
{
    testNZ<28, aclFloat16>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_both_float)
{
    testNZ<29, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_validcol_with_idx_half)
{
    testNZ<30, aclFloat16>(64, 64, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_unalignedvalid_both_with_idx_bf16)
{
    testNZ<31, uint16_t>(32, 32, 16, 32);
}

TEST_F(TExtractVecTest, case_nz_scalar_1_float)
{
    testNZScalar<1, float>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_2_half)
{
    testNZScalar<2, aclFloat16>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_3_bf16)
{
    testNZScalar<3, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_4_int8)
{
    testNZScalar<4, int8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_scalar_5_int32)
{
    testNZScalar<5, int32_t>(32, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_scalar_6_int16)
{
    testNZScalar<6, int16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_7_uint16)
{
    testNZScalar<7, uint16_t>(32, 32, 16, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_8_uint32)
{
    testNZScalar<8, uint32_t>(32, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_scalar_9_uint8_edge)
{
    testNZScalar<9, uint8_t>(32, 64, 16, 64);
}
TEST_F(TExtractVecTest, case_nz_scalar_nonpow2_half)
{
    testNZScalar<10, aclFloat16>(48, 32, 32, 32);
}
TEST_F(TExtractVecTest, case_nz_scalar_nonpow2_float)
{
    testNZScalar<11, float>(48, 16, 16, 16);
}
TEST_F(TExtractVecTest, case_nz_scalar_nonpow2_int8)
{
    testNZScalar<12, int8_t>(48, 96, 32, 64);
}
