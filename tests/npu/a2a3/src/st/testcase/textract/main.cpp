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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTMOV(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template <int32_t tilingKey>
void launchTEXTRACT(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template <int32_t tilingKey>
void launchTEXTRACT_COMPACT(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TMOVTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};
class TEXTRACTTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class TEXTRACT_Compact_Test : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int32_t key, typename T, typename U, typename S>
void tmov_test(uint32_t M, uint32_t N, uint32_t K, uint32_t baseM = 0, uint32_t baseN = 0, uint32_t baseK = 0)
{
    baseM = (baseM == 0) ? M : baseM;
    baseN = (baseN == 0) ? N : baseN;
    baseK = (baseK == 0) ? K : baseK;

    size_t aFileSize = baseM * baseK * sizeof(U);
    size_t bFileSize = baseK * baseN * sizeof(U);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case1_half_0_1_param) { tmov_test<1, float, uint16_t, uint16_t>(64, 32, 80); }
TEST_F(TMOVTest, case2_int8_0_1_param) { tmov_test<2, int32_t, int8_t, int8_t>(128, 64, 128); }
TEST_F(TMOVTest, case3_float_0_1_param) { tmov_test<3, float, float, float>(128, 48, 64); }
TEST_F(TMOVTest, case4_bfloat16_0_1_param) { tmov_test<4, float, uint16_t, uint16_t>(64, 48, 96); }
TEST_F(TMOVTest, case11_half_1_0_param) { tmov_test<11, float, uint16_t, uint16_t>(128, 64, 128); }
TEST_F(TMOVTest, case12_int8_1_0_param) { tmov_test<12, int32_t, int8_t, int8_t>(64, 64, 128); }
TEST_F(TMOVTest, case13_float_1_0_param) { tmov_test<13, float, float, float>(64, 32, 96); }
TEST_F(TMOVTest, case14_bfloat16_1_0_param) { tmov_test<14, float, uint16_t, uint16_t>(96, 80, 96); }
TEST_F(TMOVTest, case21_float_0_0_29_29_44_param) { tmov_test<21, float, float, float>(29, 29, 44, 32, 32, 48); }
TEST_F(TMOVTest, case22_float_0_0_29_29_36_param) { tmov_test<22, float, float, float>(29, 29, 36, 32, 32, 48); }
TEST_F(TMOVTest, case23_int8_0_0_65_66_40_param) { tmov_test<23, int32_t, int8_t, int8_t>(65, 66, 40, 80, 96, 64); }
TEST_F(TMOVTest, case24_int8_0_0_65_82_40_param) { tmov_test<24, int32_t, int8_t, int8_t>(65, 82, 40, 80, 96, 64); }
TEST_F(TMOVTest, case25_bfloat16_0_0_44_39_39_param)
{
    tmov_test<25, float, uint16_t, uint16_t>(44, 39, 39, 48, 48, 48);
}
TEST_F(TMOVTest, case31_float_1_1_29_29_44_param) { tmov_test<31, float, float, float>(29, 29, 44, 32, 32, 48); }
TEST_F(TMOVTest, case32_float_1_1_29_29_36_param) { tmov_test<32, float, float, float>(29, 29, 36, 32, 32, 48); }
TEST_F(TMOVTest, case33_int8_1_1_65_66_40_param) { tmov_test<33, int32_t, int8_t, int8_t>(65, 66, 40, 96, 80, 64); }
TEST_F(TMOVTest, case34_int8_1_1_65_82_40_param) { tmov_test<34, int32_t, int8_t, int8_t>(65, 82, 40, 96, 96, 64); }
TEST_F(TMOVTest, case35_bfloat16_1_1_44_39_39_param)
{
    tmov_test<35, float, uint16_t, uint16_t>(44, 39, 39, 48, 48, 48);
}

template <int32_t key, typename T, typename U, typename S>
void textract_test(
    uint32_t M, uint32_t N, uint32_t K, uint16_t indexM, uint16_t indexN, uint16_t indexK, uint32_t baseM = 0,
    uint32_t baseN = 0, uint32_t baseK = 0)
{
    baseM = (baseM == 0) ? M : baseM;
    baseN = (baseN == 0) ? N : baseN;
    baseK = (baseK == 0) ? K : baseK;

    uint32_t M1 = M - indexM;
    uint32_t N1 = N - indexN;

    size_t aFileSize = baseM * baseK * sizeof(U);
    size_t bFileSize = baseK * baseN * sizeof(U);
    size_t cFileSize = M1 * N1 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTEXTRACT<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case1_half_0_1_16_16_32_param)
{
    textract_test<1, float, uint16_t, uint16_t>(64, 32, 80, 16, 16, 32);
}
TEST_F(TEXTRACTTest, case2_int8_0_1_48_32_64_param)
{
    textract_test<2, int32_t, int8_t, int8_t>(128, 64, 128, 48, 32, 64);
}
TEST_F(TEXTRACTTest, case3_float_0_1_32_16_48_param) { textract_test<3, float, float, float>(96, 48, 64, 32, 16, 48); }
TEST_F(TEXTRACTTest, case4_bfloat16_0_1_32_32_16_param)
{
    textract_test<4, float, uint16_t, uint16_t>(64, 48, 96, 32, 32, 16);
}
TEST_F(TEXTRACTTest, case11_half_1_0_96_0_64_param)
{
    textract_test<11, float, uint16_t, uint16_t>(128, 64, 128, 96, 32, 64);
}
TEST_F(TEXTRACTTest, case12_int8_1_0_32_0_32_param)
{
    textract_test<12, int32_t, int8_t, int8_t>(64, 64, 128, 32, 32, 32);
}
TEST_F(TEXTRACTTest, case13_float_1_0_32_0_16_param) { textract_test<13, float, float, float>(64, 32, 96, 32, 16, 16); }
TEST_F(TEXTRACTTest, case14_bfloat16_1_0_32_0_48_param)
{
    textract_test<14, float, uint16_t, uint16_t>(96, 80, 96, 32, 64, 48);
}
TEST_F(TEXTRACTTest, case21_float_0_0_29_29_36_param)
{
    textract_test<21, float, float, float>(29, 29, 36, 16, 16, 32, 32, 32, 48);
}
TEST_F(TEXTRACTTest, case22_int8_0_0_65_66_40_param)
{
    textract_test<22, int32_t, int8_t, int8_t>(65, 66, 40, 32, 64, 32, 80, 96, 64);
}
TEST_F(TEXTRACTTest, case23_bfloat16_0_0_44_39_39_param)
{
    textract_test<23, float, uint16_t, uint16_t>(44, 39, 39, 32, 16, 32, 48, 48, 48);
}
TEST_F(TEXTRACTTest, case31_float_1_1_29_29_36_param)
{
    textract_test<31, float, float, float>(29, 29, 36, 16, 16, 32, 32, 32, 48);
}
TEST_F(TEXTRACTTest, case32_int8_1_1_65_66_40_param)
{
    textract_test<32, int32_t, int8_t, int8_t>(65, 66, 40, 32, 64, 32, 96, 80, 64);
}
TEST_F(TEXTRACTTest, case33_bfloat16_1_1_44_39_39_param)
{
    textract_test<33, float, uint16_t, uint16_t>(44, 39, 39, 32, 16, 32, 48, 48, 48);
}
TEST_F(TEXTRACTTest, case41_dynamic_half_0_1_16_0_32_param)
{
    textract_test<41, float, uint16_t, uint16_t>(64, 32, 80, 16, 0, 32);
}
TEST_F(TEXTRACTTest, case42_dynamic_int8_1_1_32_0_32_param)
{
    textract_test<42, int32_t, int8_t, int8_t>(64, 64, 128, 32, 0, 32);
}

template <int32_t key, typename T, typename U, typename S>
void textract_compact_test(
    uint32_t M, uint32_t N, uint32_t K, uint16_t indexM, uint16_t indexN, uint16_t indexK, uint32_t baseM = 0,
    uint32_t baseN = 0, uint32_t baseK = 0)
{
    baseM = (baseM == 0) ? M : baseM;
    baseN = (baseN == 0) ? N : baseN;
    baseK = (baseK == 0) ? K : baseK;

    uint32_t M1 = M - indexM;
    uint32_t N1 = N - indexN;

    size_t aFileSize = baseM * baseK * sizeof(U);
    size_t bFileSize = baseK * baseN * sizeof(U);
    size_t cFileSize = M1 * N1 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTEXTRACT_COMPACT<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACT_Compact_Test, case1_float_1_0_param)
{
    textract_compact_test<1, float, float, float>(20, 215, 22, 0, 0, 0, 128, 256, 128);
}
TEST_F(TEXTRACT_Compact_Test, case2_int8_1_0_param)
{
    textract_compact_test<2, int32_t, int8_t, int8_t>(46, 36, 203, 0, 0, 0, 128, 128, 256);
}
TEST_F(TEXTRACT_Compact_Test, case3_bfloat16_1_0_param)
{
    textract_compact_test<3, float, uint16_t, uint16_t>(220, 25, 30, 0, 0, 0, 256, 128, 128);
}
TEST_F(TEXTRACT_Compact_Test, case11_float_0_1_param)
{
    textract_compact_test<11, float, float, float>(20, 215, 22, 0, 0, 0, 128, 256, 128);
}
TEST_F(TEXTRACT_Compact_Test, case12_int8_0_1_param)
{
    textract_compact_test<12, int32_t, int8_t, int8_t>(46, 36, 203, 0, 0, 0, 128, 128, 256);
}
TEST_F(TEXTRACT_Compact_Test, case13_bfloat16_0_1_param)
{
    textract_compact_test<13, float, uint16_t, uint16_t>(220, 25, 30, 0, 0, 0, 256, 128, 128);
}
TEST_F(TEXTRACT_Compact_Test, case21_float_0_0_param)
{
    textract_compact_test<21, float, float, float>(36, 215, 22, 16, 16, 16, 128, 256, 128);
}
TEST_F(TEXTRACT_Compact_Test, case22_int8_0_0_param)
{
    textract_compact_test<22, int32_t, int8_t, int8_t>(46, 36, 203, 32, 32, 32, 128, 128, 256);
}
TEST_F(TEXTRACT_Compact_Test, case23_bfloat16_0_0_param)
{
    textract_compact_test<23, float, uint16_t, uint16_t>(220, 25, 30, 16, 16, 16, 256, 128, 128);
}
TEST_F(TEXTRACT_Compact_Test, case31_float_1_1_param)
{
    textract_compact_test<31, float, float, float>(20, 215, 22, 16, 16, 16, 128, 256, 128);
}
TEST_F(TEXTRACT_Compact_Test, case32_int8_1_1_param)
{
    textract_compact_test<32, int32_t, int8_t, int8_t>(46, 36, 203, 32, 32, 32, 128, 128, 256);
}
TEST_F(TEXTRACT_Compact_Test, case33_bfloat16_1_1_param)
{
    textract_compact_test<33, float, uint16_t, uint16_t>(220, 25, 30, 16, 16, 16, 256, 128, 128);
}
