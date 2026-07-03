/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TQuantDNTest {

template <int M, int N, int N_pad>
void LaunchTQuantDN(uint16_t *src, int8_t *fp8_nd, uint8_t *e8_dn, int8_t *fp8_nz, uint8_t *e8_zz, uint16_t *max_dn,
                    void *stream);

template <int M, int N, int N_pad>
void LaunchTQuantDN_fp32(uint32_t *src, int8_t *fp8_nd, uint8_t *e8_dn, int8_t *fp8_nz, uint8_t *e8_zz,
                         uint32_t *max_dn, void *stream);

template <int M, int N, int N_pad>
void LaunchTQuantDN_MXFP4_bf16(uint16_t *src, uint8_t *fp4_nd, uint8_t *e8_dn, uint8_t *fp4_nz, uint16_t *max_dn,
                               void *stream);

template <int M, int N, int N_pad>
void LaunchTQuantDN_MXFP4_fp16(uint16_t *src, uint8_t *fp4_nd, uint8_t *e8_dn, uint8_t *fp4_nz, uint16_t *max_dn,
                               void *stream);

} // namespace TQuantDNTest

class TQUANTDNTest : public testing::Test {
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
    const std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T>
void ExpectGoldenMatch(const char *stageName, const char *tensorName, const std::vector<T> &golden,
                       const std::vector<T> &output)
{
    SCOPED_TRACE(stageName);
    ASSERT_EQ(golden.size(), output.size()) << tensorName << " size mismatch";
    EXPECT_TRUE(ResultCmp<T>(golden, output, 0.0f)) << stageName << ": " << tensorName << " mismatch vs golden";
}

template <int M, int N, int N_pad>
void test_tquant_dn_bf16()
{
    constexpr int grpSize = 32;
    constexpr int hatM = M / grpSize;
    constexpr int paddedCols = N_pad;
    constexpr int paddedRows16 = ((M + 15) / 16) * 16;
    constexpr int virtualRow = paddedRows16 + 1;
    constexpr int groupedCols = paddedCols / 32;
    constexpr int numGroupsFlat = M * groupedCols;
    constexpr int numGroupsFlatAligned = ((numGroupsFlat + 31) / 32) * 32;
    size_t srcFileSize = M * paddedCols * sizeof(uint16_t);
    size_t fp8NDFileSize = M * paddedCols * sizeof(int8_t);
    size_t e8DNFileSize = hatM * paddedCols * sizeof(uint8_t);
    size_t fp8NZFileSize = M * paddedCols * sizeof(int8_t);
    size_t e8ZZFileSize = numGroupsFlatAligned * sizeof(uint8_t);
    size_t maxDNFileSize = hatM * paddedCols * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost;
    uint8_t *fp8NDHost;
    uint8_t *e8DNHost;
    uint8_t *fp8NZHost;
    uint8_t *e8ZZHost;
    uint16_t *maxDNHost;
    uint16_t *srcDevice;
    int8_t *fp8NDDevice;
    uint8_t *e8DNDevice;
    int8_t *fp8NZDevice;
    uint8_t *e8ZZDevice;
    uint16_t *maxDNDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&fp8NDHost), fp8NDFileSize);
    aclrtMallocHost((void **)(&e8DNHost), e8DNFileSize);
    aclrtMallocHost((void **)(&fp8NZHost), fp8NZFileSize);
    aclrtMallocHost((void **)(&e8ZZHost), e8ZZFileSize);
    aclrtMallocHost((void **)(&maxDNHost), maxDNFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp8NDDevice, fp8NDFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8DNDevice, e8DNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp8NZDevice, fp8NZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8ZZDevice, e8ZZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maxDNDevice, maxDNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    const std::string goldenDir = GetGoldenDir();

    // Full DN pipeline: TQUANT(DN) + TMOV(ND->NZ) + TMOV<0>(DN->ZZ).
    TQuantDNTest::LaunchTQuantDN<M, N, N_pad>(srcDevice, fp8NDDevice, e8DNDevice, fp8NZDevice, e8ZZDevice, maxDNDevice,
                                              stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "DN pipeline sync failed: " << aclGetRecentErrMsg();

    aclrtMemcpy(fp8NDHost, fp8NDFileSize, fp8NDDevice, fp8NDFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8DNHost, e8DNFileSize, e8DNDevice, e8DNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(fp8NZHost, fp8NZFileSize, fp8NZDevice, fp8NZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8ZZHost, e8ZZFileSize, e8ZZDevice, e8ZZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxDNHost, maxDNFileSize, maxDNDevice, maxDNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output_fp8_nd.bin", fp8NDHost, fp8NDFileSize);
    WriteFile(goldenDir + "/output_e8_dn.bin", e8DNHost, e8DNFileSize);
    WriteFile(goldenDir + "/output_fp8_nz.bin", fp8NZHost, fp8NZFileSize);
    WriteFile(goldenDir + "/output_e8_zz.bin", e8ZZHost, e8ZZFileSize);
    WriteFile(goldenDir + "/output_group_max.bin", maxDNHost, maxDNFileSize);

    std::vector<uint8_t> goldenFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> goldenE8Dn(e8DNFileSize);
    std::vector<uint8_t> goldenFp8Nz(fp8NZFileSize);
    std::vector<uint8_t> goldenE8Zz(e8ZZFileSize);
    std::vector<uint16_t> goldenGroupMax(maxDNFileSize / sizeof(uint16_t));
    std::vector<uint8_t> outFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> outE8Dn(e8DNFileSize);
    std::vector<uint8_t> outFp8Nz(fp8NZFileSize);
    std::vector<uint8_t> outE8Zz(e8ZZFileSize);
    std::vector<uint16_t> outGroupMax(maxDNFileSize / sizeof(uint16_t));
    ReadFile(goldenDir + "/golden_fp8_nd.bin", fp8NDFileSize, goldenFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/golden_e8_dn.bin", e8DNFileSize, goldenE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/golden_fp8_nz.bin", fp8NZFileSize, goldenFp8Nz.data(), fp8NZFileSize);
    ReadFile(goldenDir + "/golden_e8_zz.bin", e8ZZFileSize, goldenE8Zz.data(), e8ZZFileSize);
    ReadFile(goldenDir + "/golden_group_max.bin", maxDNFileSize, goldenGroupMax.data(), maxDNFileSize);
    ReadFile(goldenDir + "/output_fp8_nd.bin", fp8NDFileSize, outFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/output_e8_dn.bin", e8DNFileSize, outE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/output_fp8_nz.bin", fp8NZFileSize, outFp8Nz.data(), fp8NZFileSize);
    ReadFile(goldenDir + "/output_e8_zz.bin", e8ZZFileSize, outE8Zz.data(), e8ZZFileSize);
    ReadFile(goldenDir + "/output_group_max.bin", maxDNFileSize, outGroupMax.data(), maxDNFileSize);
    ExpectGoldenMatch("DN_Pipeline", "fp8_nd", goldenFp8Nd, outFp8Nd);
    ExpectGoldenMatch("DN_Pipeline", "e8_dn (exponents)", goldenE8Dn, outE8Dn);
    ExpectGoldenMatch("DN_Pipeline", "fp8_nz", goldenFp8Nz, outFp8Nz);
    ExpectGoldenMatch("DN_Pipeline", "e8_zz", goldenE8Zz, outE8Zz);
    ExpectGoldenMatch("DN_Pipeline", "group_max", goldenGroupMax, outGroupMax);

    aclrtFree(srcDevice);
    aclrtFree(fp8NDDevice);
    aclrtFree(e8DNDevice);
    aclrtFree(fp8NZDevice);
    aclrtFree(e8ZZDevice);
    aclrtFree(maxDNDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(fp8NDHost);
    aclrtFreeHost(e8DNHost);
    aclrtFreeHost(fp8NZHost);
    aclrtFreeHost(e8ZZHost);
    aclrtFreeHost(maxDNHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TQUANTDNTest, case_bf16_64x64)
{
    test_tquant_dn_bf16<64, 64, 64>();
}
TEST_F(TQUANTDNTest, case_bf16_128x64)
{
    test_tquant_dn_bf16<128, 64, 64>();
}
TEST_F(TQUANTDNTest, case_bf16_64x128)
{
    test_tquant_dn_bf16<64, 128, 128>();
}
TEST_F(TQUANTDNTest, case_bf16_128x128)
{
    test_tquant_dn_bf16<128, 128, 128>();
}
TEST_F(TQUANTDNTest, case_bf16_64x256)
{
    test_tquant_dn_bf16<64, 256, 256>();
}
TEST_F(TQUANTDNTest, case_bf16_128x256)
{
    test_tquant_dn_bf16<128, 256, 256>();
}
TEST_F(TQUANTDNTest, case_bf16_256x64)
{
    test_tquant_dn_bf16<256, 64, 64>();
}
TEST_F(TQUANTDNTest, case_bf16_256x128)
{
    test_tquant_dn_bf16<256, 128, 128>();
}

template <int M, int N, int N_pad>
void test_tquant_dn_fp32()
{
    constexpr int grpSize = 32;
    constexpr int hatM = M / grpSize;
    constexpr int paddedCols = N_pad;
    constexpr int paddedRows16 = ((M + 15) / 16) * 16;
    constexpr int virtualRow = paddedRows16 + 1;
    constexpr int groupedCols = paddedCols / 32;
    constexpr int numGroupsFlat = M * groupedCols;
    constexpr int numGroupsFlatAligned = ((numGroupsFlat + 31) / 32) * 32;
    size_t srcFileSize = M * paddedCols * sizeof(uint32_t);
    size_t fp8NDFileSize = M * paddedCols * sizeof(int8_t);
    size_t e8DNFileSize = hatM * paddedCols * sizeof(uint8_t);
    size_t fp8NZFileSize = M * paddedCols * sizeof(int8_t);
    size_t e8ZZFileSize = numGroupsFlatAligned * sizeof(uint8_t);
    size_t maxDNFileSize = hatM * paddedCols * sizeof(uint32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost;
    uint8_t *fp8NDHost;
    uint8_t *e8DNHost;
    uint8_t *fp8NZHost;
    uint8_t *e8ZZHost;
    uint32_t *maxDNHost;
    uint32_t *srcDevice;
    int8_t *fp8NDDevice;
    uint8_t *e8DNDevice;
    int8_t *fp8NZDevice;
    uint8_t *e8ZZDevice;
    uint32_t *maxDNDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&fp8NDHost), fp8NDFileSize);
    aclrtMallocHost((void **)(&e8DNHost), e8DNFileSize);
    aclrtMallocHost((void **)(&fp8NZHost), fp8NZFileSize);
    aclrtMallocHost((void **)(&e8ZZHost), e8ZZFileSize);
    aclrtMallocHost((void **)(&maxDNHost), maxDNFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp8NDDevice, fp8NDFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8DNDevice, e8DNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp8NZDevice, fp8NZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8ZZDevice, e8ZZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maxDNDevice, maxDNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    const std::string goldenDir = GetGoldenDir();

    // Full DN pipeline: TQUANT(DN) + TMOV(ND->NZ) + TMOV<0>(DN->ZZ).
    TQuantDNTest::LaunchTQuantDN_fp32<M, N, N_pad>(srcDevice, fp8NDDevice, e8DNDevice, fp8NZDevice, e8ZZDevice,
                                                   maxDNDevice, stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "DN pipeline fp32 sync failed: " << aclGetRecentErrMsg();

    aclrtMemcpy(fp8NDHost, fp8NDFileSize, fp8NDDevice, fp8NDFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8DNHost, e8DNFileSize, e8DNDevice, e8DNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(fp8NZHost, fp8NZFileSize, fp8NZDevice, fp8NZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8ZZHost, e8ZZFileSize, e8ZZDevice, e8ZZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxDNHost, maxDNFileSize, maxDNDevice, maxDNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output_fp8_nd.bin", fp8NDHost, fp8NDFileSize);
    WriteFile(goldenDir + "/output_e8_dn.bin", e8DNHost, e8DNFileSize);
    WriteFile(goldenDir + "/output_fp8_nz.bin", fp8NZHost, fp8NZFileSize);
    WriteFile(goldenDir + "/output_e8_zz.bin", e8ZZHost, e8ZZFileSize);
    WriteFile(goldenDir + "/output_group_max.bin", maxDNHost, maxDNFileSize);

    std::vector<uint8_t> goldenFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> goldenE8Dn(e8DNFileSize);
    std::vector<uint8_t> goldenFp8Nz(fp8NZFileSize);
    std::vector<uint8_t> goldenE8Zz(e8ZZFileSize);
    std::vector<uint32_t> goldenGroupMax(maxDNFileSize / sizeof(uint32_t));
    std::vector<uint8_t> outFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> outE8Dn(e8DNFileSize);
    std::vector<uint8_t> outFp8Nz(fp8NZFileSize);
    std::vector<uint8_t> outE8Zz(e8ZZFileSize);
    std::vector<uint32_t> outGroupMax(maxDNFileSize / sizeof(uint32_t));
    ReadFile(goldenDir + "/golden_fp8_nd.bin", fp8NDFileSize, goldenFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/golden_e8_dn.bin", e8DNFileSize, goldenE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/golden_fp8_nz.bin", fp8NZFileSize, goldenFp8Nz.data(), fp8NZFileSize);
    ReadFile(goldenDir + "/golden_e8_zz.bin", e8ZZFileSize, goldenE8Zz.data(), e8ZZFileSize);
    ReadFile(goldenDir + "/golden_group_max.bin", maxDNFileSize, goldenGroupMax.data(), maxDNFileSize);
    ReadFile(goldenDir + "/output_fp8_nd.bin", fp8NDFileSize, outFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/output_e8_dn.bin", e8DNFileSize, outE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/output_fp8_nz.bin", fp8NZFileSize, outFp8Nz.data(), fp8NZFileSize);
    ReadFile(goldenDir + "/output_e8_zz.bin", e8ZZFileSize, outE8Zz.data(), e8ZZFileSize);
    ReadFile(goldenDir + "/output_group_max.bin", maxDNFileSize, outGroupMax.data(), maxDNFileSize);
    ExpectGoldenMatch("DN_Pipeline_fp32", "fp8_nd", goldenFp8Nd, outFp8Nd);
    ExpectGoldenMatch("DN_Pipeline_fp32", "e8_dn (exponents)", goldenE8Dn, outE8Dn);
    ExpectGoldenMatch("DN_Pipeline_fp32", "fp8_nz", goldenFp8Nz, outFp8Nz);
    ExpectGoldenMatch("DN_Pipeline_fp32", "e8_zz", goldenE8Zz, outE8Zz);
    ExpectGoldenMatch("DN_Pipeline_fp32", "group_max", goldenGroupMax, outGroupMax);

    aclrtFree(srcDevice);
    aclrtFree(fp8NDDevice);
    aclrtFree(e8DNDevice);
    aclrtFree(fp8NZDevice);
    aclrtFree(e8ZZDevice);
    aclrtFree(maxDNDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(fp8NDHost);
    aclrtFreeHost(e8DNHost);
    aclrtFreeHost(fp8NZHost);
    aclrtFreeHost(e8ZZHost);
    aclrtFreeHost(maxDNHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TQUANTDNTest, case_fp32_64x128)
{
    test_tquant_dn_fp32<64, 128, 128>();
}
TEST_F(TQUANTDNTest, case_fp32_128x128)
{
    test_tquant_dn_fp32<128, 128, 128>();
}
TEST_F(TQUANTDNTest, case_fp32_64x256)
{
    test_tquant_dn_fp32<64, 256, 256>();
}

// MXFP4 (E2M1) DN tests. Both bf16 and fp16 sources share the same UB/GM shape:
//   fp4_nd : M * packedCols bytes       (packedCols = paddedCols/2)
//   fp4_nz : M * packedCols bytes       (ND->NZ packed FP4, block = 32B = 64 FP4 values)
//   e8_dn  : hatM * paddedCols bytes
//   max_dn : hatM * paddedCols * sizeof(b16) bytes
template <int M, int N, int N_pad>
void test_tquant_dn_mxfp4_bf16()
{
    constexpr int grpSize = 32;
    constexpr int hatM = M / grpSize;
    constexpr int paddedCols = N_pad;
    constexpr int packedCols = paddedCols / 2;
    size_t srcFileSize = M * paddedCols * sizeof(uint16_t);
    size_t fp4NDFileSize = M * packedCols * sizeof(uint8_t);
    size_t fp4NZFileSize = M * packedCols * sizeof(uint8_t);
    size_t e8DNFileSize = hatM * paddedCols * sizeof(uint8_t);
    size_t maxDNFileSize = hatM * paddedCols * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost;
    uint8_t *fp4NDHost;
    uint8_t *fp4NZHost;
    uint8_t *e8DNHost;
    uint16_t *maxDNHost;
    uint8_t *srcDevice;
    uint8_t *fp4NDDevice;
    uint8_t *fp4NZDevice;
    uint8_t *e8DNDevice;
    uint16_t *maxDNDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&fp4NDHost), fp4NDFileSize);
    aclrtMallocHost((void **)(&fp4NZHost), fp4NZFileSize);
    aclrtMallocHost((void **)(&e8DNHost), e8DNFileSize);
    aclrtMallocHost((void **)(&maxDNHost), maxDNFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp4NDDevice, fp4NDFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp4NZDevice, fp4NZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8DNDevice, e8DNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maxDNDevice, maxDNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    const std::string goldenDir = GetGoldenDir();

    TQuantDNTest::LaunchTQuantDN_MXFP4_bf16<M, N, N_pad>((uint16_t *)srcDevice, fp4NDDevice, e8DNDevice, fp4NZDevice,
                                                         maxDNDevice, stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "MXFP4 bf16 DN sync failed: " << aclGetRecentErrMsg();

    aclrtMemcpy(fp4NDHost, fp4NDFileSize, fp4NDDevice, fp4NDFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(fp4NZHost, fp4NZFileSize, fp4NZDevice, fp4NZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8DNHost, e8DNFileSize, e8DNDevice, e8DNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxDNHost, maxDNFileSize, maxDNDevice, maxDNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output_fp4_nd.bin", fp4NDHost, fp4NDFileSize);
    WriteFile(goldenDir + "/output_fp4_nz.bin", fp4NZHost, fp4NZFileSize);
    WriteFile(goldenDir + "/output_e8_dn.bin", e8DNHost, e8DNFileSize);
    WriteFile(goldenDir + "/output_group_max.bin", maxDNHost, maxDNFileSize);

    std::vector<uint8_t> goldenFp4Nd(fp4NDFileSize);
    std::vector<uint8_t> goldenFp4Nz(fp4NZFileSize);
    std::vector<uint8_t> goldenE8Dn(e8DNFileSize);
    std::vector<uint16_t> goldenGroupMax(maxDNFileSize / sizeof(uint16_t));
    std::vector<uint8_t> outFp4Nd(fp4NDFileSize);
    std::vector<uint8_t> outFp4Nz(fp4NZFileSize);
    std::vector<uint8_t> outE8Dn(e8DNFileSize);
    std::vector<uint16_t> outGroupMax(maxDNFileSize / sizeof(uint16_t));
    ReadFile(goldenDir + "/golden_fp4_nd.bin", fp4NDFileSize, goldenFp4Nd.data(), fp4NDFileSize);
    ReadFile(goldenDir + "/golden_fp4_nz.bin", fp4NZFileSize, goldenFp4Nz.data(), fp4NZFileSize);
    ReadFile(goldenDir + "/golden_e8_dn.bin", e8DNFileSize, goldenE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/golden_group_max.bin", maxDNFileSize, goldenGroupMax.data(), maxDNFileSize);
    ReadFile(goldenDir + "/output_fp4_nd.bin", fp4NDFileSize, outFp4Nd.data(), fp4NDFileSize);
    ReadFile(goldenDir + "/output_fp4_nz.bin", fp4NZFileSize, outFp4Nz.data(), fp4NZFileSize);
    ReadFile(goldenDir + "/output_e8_dn.bin", e8DNFileSize, outE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/output_group_max.bin", maxDNFileSize, outGroupMax.data(), maxDNFileSize);
    ExpectGoldenMatch("MXFP4_BF16_DN", "fp4_nd", goldenFp4Nd, outFp4Nd);
    ExpectGoldenMatch("MXFP4_BF16_DN", "fp4_nz", goldenFp4Nz, outFp4Nz);
    ExpectGoldenMatch("MXFP4_BF16_DN", "e8_dn (exponents)", goldenE8Dn, outE8Dn);
    ExpectGoldenMatch("MXFP4_BF16_DN", "group_max", goldenGroupMax, outGroupMax);

    aclrtFree(srcDevice);
    aclrtFree(fp4NDDevice);
    aclrtFree(fp4NZDevice);
    aclrtFree(e8DNDevice);
    aclrtFree(maxDNDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(fp4NDHost);
    aclrtFreeHost(fp4NZHost);
    aclrtFreeHost(e8DNHost);
    aclrtFreeHost(maxDNHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TQUANTDNTest, case_mxfp4_bf16_64x128)
{
    test_tquant_dn_mxfp4_bf16<64, 128, 128>();
}
TEST_F(TQUANTDNTest, case_mxfp4_bf16_128x128)
{
    test_tquant_dn_mxfp4_bf16<128, 128, 128>();
}
TEST_F(TQUANTDNTest, case_mxfp4_bf16_64x256)
{
    test_tquant_dn_mxfp4_bf16<64, 256, 256>();
}

template <int M, int N, int N_pad>
void test_tquant_dn_mxfp4_fp16()
{
    constexpr int grpSize = 32;
    constexpr int hatM = M / grpSize;
    constexpr int paddedCols = N_pad;
    constexpr int packedCols = paddedCols / 2;
    size_t srcFileSize = M * paddedCols * sizeof(uint16_t);
    size_t fp4NDFileSize = M * packedCols * sizeof(uint8_t);
    size_t fp4NZFileSize = M * packedCols * sizeof(uint8_t);
    size_t e8DNFileSize = hatM * paddedCols * sizeof(uint8_t);
    size_t maxDNFileSize = hatM * paddedCols * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost;
    uint8_t *fp4NDHost;
    uint8_t *fp4NZHost;
    uint8_t *e8DNHost;
    uint16_t *maxDNHost;
    uint8_t *srcDevice;
    uint8_t *fp4NDDevice;
    uint8_t *fp4NZDevice;
    uint8_t *e8DNDevice;
    uint16_t *maxDNDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&fp4NDHost), fp4NDFileSize);
    aclrtMallocHost((void **)(&fp4NZHost), fp4NZFileSize);
    aclrtMallocHost((void **)(&e8DNHost), e8DNFileSize);
    aclrtMallocHost((void **)(&maxDNHost), maxDNFileSize);

    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp4NDDevice, fp4NDFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fp4NZDevice, fp4NZFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&e8DNDevice, e8DNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maxDNDevice, maxDNFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    const std::string goldenDir = GetGoldenDir();

    TQuantDNTest::LaunchTQuantDN_MXFP4_fp16<M, N, N_pad>((uint16_t *)srcDevice, fp4NDDevice, e8DNDevice, fp4NZDevice,
                                                         maxDNDevice, stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "MXFP4 fp16 DN sync failed: " << aclGetRecentErrMsg();

    aclrtMemcpy(fp4NDHost, fp4NDFileSize, fp4NDDevice, fp4NDFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(fp4NZHost, fp4NZFileSize, fp4NZDevice, fp4NZFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8DNHost, e8DNFileSize, e8DNDevice, e8DNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxDNHost, maxDNFileSize, maxDNDevice, maxDNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output_fp4_nd.bin", fp4NDHost, fp4NDFileSize);
    WriteFile(goldenDir + "/output_fp4_nz.bin", fp4NZHost, fp4NZFileSize);
    WriteFile(goldenDir + "/output_e8_dn.bin", e8DNHost, e8DNFileSize);
    WriteFile(goldenDir + "/output_group_max.bin", maxDNHost, maxDNFileSize);

    std::vector<uint8_t> goldenFp4Nd(fp4NDFileSize);
    std::vector<uint8_t> goldenFp4Nz(fp4NZFileSize);
    std::vector<uint8_t> goldenE8Dn(e8DNFileSize);
    std::vector<uint16_t> goldenGroupMax(maxDNFileSize / sizeof(uint16_t));
    std::vector<uint8_t> outFp4Nd(fp4NDFileSize);
    std::vector<uint8_t> outFp4Nz(fp4NZFileSize);
    std::vector<uint8_t> outE8Dn(e8DNFileSize);
    std::vector<uint16_t> outGroupMax(maxDNFileSize / sizeof(uint16_t));
    ReadFile(goldenDir + "/golden_fp4_nd.bin", fp4NDFileSize, goldenFp4Nd.data(), fp4NDFileSize);
    ReadFile(goldenDir + "/golden_fp4_nz.bin", fp4NZFileSize, goldenFp4Nz.data(), fp4NZFileSize);
    ReadFile(goldenDir + "/golden_e8_dn.bin", e8DNFileSize, goldenE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/golden_group_max.bin", maxDNFileSize, goldenGroupMax.data(), maxDNFileSize);
    ReadFile(goldenDir + "/output_fp4_nd.bin", fp4NDFileSize, outFp4Nd.data(), fp4NDFileSize);
    ReadFile(goldenDir + "/output_fp4_nz.bin", fp4NZFileSize, outFp4Nz.data(), fp4NZFileSize);
    ReadFile(goldenDir + "/output_e8_dn.bin", e8DNFileSize, outE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/output_group_max.bin", maxDNFileSize, outGroupMax.data(), maxDNFileSize);
    ExpectGoldenMatch("MXFP4_FP16_DN", "fp4_nd", goldenFp4Nd, outFp4Nd);
    ExpectGoldenMatch("MXFP4_FP16_DN", "fp4_nz", goldenFp4Nz, outFp4Nz);
    ExpectGoldenMatch("MXFP4_FP16_DN", "e8_dn (exponents)", goldenE8Dn, outE8Dn);
    ExpectGoldenMatch("MXFP4_FP16_DN", "group_max", goldenGroupMax, outGroupMax);

    aclrtFree(srcDevice);
    aclrtFree(fp4NDDevice);
    aclrtFree(fp4NZDevice);
    aclrtFree(e8DNDevice);
    aclrtFree(maxDNDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(fp4NDHost);
    aclrtFreeHost(fp4NZHost);
    aclrtFreeHost(e8DNHost);
    aclrtFreeHost(maxDNHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TQUANTDNTest, case_mxfp4_fp16_64x128)
{
    test_tquant_dn_mxfp4_fp16<64, 128, 128>();
}
TEST_F(TQUANTDNTest, case_mxfp4_fp16_128x128)
{
    test_tquant_dn_mxfp4_fp16<128, 128, 128>();
}
TEST_F(TQUANTDNTest, case_mxfp4_fp16_64x256)
{
    test_tquant_dn_mxfp4_fp16<64, 256, 256>();
}
