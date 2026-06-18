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

template <int Stage, int M, int N, int N_pad>
void LaunchTQuantDN(uint16_t *src, int8_t *fp8_nd, uint8_t *e8_dn, int8_t *fp8_nz, uint8_t *e8_zz, uint16_t *max_dn,
                    void *stream);

template <int Stage, int M, int N, int N_pad>
void LaunchTQuantDN_fp32(uint32_t *src, int8_t *fp8_nd, uint8_t *e8_dn, int8_t *fp8_nz, uint8_t *e8_zz,
                         uint32_t *max_dn, void *stream);

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
    size_t fp8NZFileSize = virtualRow * paddedCols * sizeof(int8_t);
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

    // Stage 1: after TQUANT — FP8 ND + E8M0 DN + per-group max
    TQuantDNTest::LaunchTQuantDN<1, M, N, N_pad>(srcDevice, fp8NDDevice, e8DNDevice, nullptr, nullptr, maxDNDevice,
                                                 stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "Stage1 sync failed: " << aclGetRecentErrMsg();

    aclrtMemcpy(fp8NDHost, fp8NDFileSize, fp8NDDevice, fp8NDFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8DNHost, e8DNFileSize, e8DNDevice, e8DNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxDNHost, maxDNFileSize, maxDNDevice, maxDNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output_fp8_nd.bin", fp8NDHost, fp8NDFileSize);
    WriteFile(goldenDir + "/output_e8_dn.bin", e8DNHost, e8DNFileSize);
    WriteFile(goldenDir + "/output_group_max.bin", maxDNHost, maxDNFileSize);

    std::vector<uint8_t> goldenFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> goldenE8Dn(e8DNFileSize);
    std::vector<uint16_t> goldenGroupMax(maxDNFileSize / sizeof(uint16_t));
    std::vector<uint8_t> outFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> outE8Dn(e8DNFileSize);
    std::vector<uint16_t> outGroupMax(maxDNFileSize / sizeof(uint16_t));
    ReadFile(goldenDir + "/golden_fp8_nd.bin", fp8NDFileSize, goldenFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/golden_e8_dn.bin", e8DNFileSize, goldenE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/golden_group_max.bin", maxDNFileSize, goldenGroupMax.data(), maxDNFileSize);
    ReadFile(goldenDir + "/output_fp8_nd.bin", fp8NDFileSize, outFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/output_e8_dn.bin", e8DNFileSize, outE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/output_group_max.bin", maxDNFileSize, outGroupMax.data(), maxDNFileSize);
    ExpectGoldenMatch("Stage1_AfterTQuant", "fp8_nd", goldenFp8Nd, outFp8Nd);
    ExpectGoldenMatch("Stage1_AfterTQuant", "e8_dn (exponents)", goldenE8Dn, outE8Dn);
    ExpectGoldenMatch("Stage1_AfterTQuant", "group_max", goldenGroupMax, outGroupMax);

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
    size_t fp8NZFileSize = virtualRow * paddedCols * sizeof(int8_t);
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

    TQuantDNTest::LaunchTQuantDN_fp32<1, M, N, N_pad>(srcDevice, fp8NDDevice, e8DNDevice, nullptr, nullptr, maxDNDevice,
                                                      stream);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "Stage1 sync failed: " << aclGetRecentErrMsg();

    aclrtMemcpy(fp8NDHost, fp8NDFileSize, fp8NDDevice, fp8NDFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(e8DNHost, e8DNFileSize, e8DNDevice, e8DNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(maxDNHost, maxDNFileSize, maxDNDevice, maxDNFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(goldenDir + "/output_fp8_nd.bin", fp8NDHost, fp8NDFileSize);
    WriteFile(goldenDir + "/output_e8_dn.bin", e8DNHost, e8DNFileSize);
    WriteFile(goldenDir + "/output_group_max.bin", maxDNHost, maxDNFileSize);

    std::vector<uint8_t> goldenFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> goldenE8Dn(e8DNFileSize);
    std::vector<uint32_t> goldenGroupMax(maxDNFileSize / sizeof(uint32_t));
    std::vector<uint8_t> outFp8Nd(fp8NDFileSize);
    std::vector<uint8_t> outE8Dn(e8DNFileSize);
    std::vector<uint32_t> outGroupMax(maxDNFileSize / sizeof(uint32_t));
    ReadFile(goldenDir + "/golden_fp8_nd.bin", fp8NDFileSize, goldenFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/golden_e8_dn.bin", e8DNFileSize, goldenE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/golden_group_max.bin", maxDNFileSize, goldenGroupMax.data(), maxDNFileSize);
    ReadFile(goldenDir + "/output_fp8_nd.bin", fp8NDFileSize, outFp8Nd.data(), fp8NDFileSize);
    ReadFile(goldenDir + "/output_e8_dn.bin", e8DNFileSize, outE8Dn.data(), e8DNFileSize);
    ReadFile(goldenDir + "/output_group_max.bin", maxDNFileSize, outGroupMax.data(), maxDNFileSize);
    ExpectGoldenMatch("Stage1_AfterTQuant", "fp8_nd", goldenFp8Nd, outFp8Nd);
    ExpectGoldenMatch("Stage1_AfterTQuant", "e8_dn (exponents)", goldenE8Dn, outE8Dn);
    ExpectGoldenMatch("Stage1_AfterTQuant", "group_max", goldenGroupMax, outGroupMax);

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
