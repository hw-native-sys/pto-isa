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

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TMATMULTest : public testing::Test {
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

template <typename OutType, typename AType, typename BType, int32_t key>
void tmatmul_test(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(BType);
    size_t cFileSize = m * n * sizeof(OutType);

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

    LaunchTMATMUL<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<OutType> golden(cFileSize / sizeof(OutType));
    std::vector<OutType> devFinal(cFileSize / sizeof(OutType));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

#define DEFINE_TMATMUL_CASE(CASE_NAME, OUT_T, A_T, B_T, KEY, M, K, N) \
    TEST_F(TMATMULTest, CASE_NAME) { tmatmul_test<OUT_T, A_T, B_T, KEY>(M, K, N); }

DEFINE_TMATMUL_CASE(case_fp16_fp16_to_fp32_31x96x47, float, uint16_t, uint16_t, 1, 31, 96, 47)
DEFINE_TMATMUL_CASE(case_int8_int8_to_int32_65x90x89, int32_t, int8_t, int8_t, 2, 65, 90, 89)
DEFINE_TMATMUL_CASE(case_fp32_fp32_to_fp32_16x32x64, float, float, float, 3, 16, 32, 64)
DEFINE_TMATMUL_CASE(case_fp16_fp16_to_fp32_1x256x64, float, uint16_t, uint16_t, 4, 1, 256, 64)
DEFINE_TMATMUL_CASE(case_nd_fp16_fp16_to_fp32_64x64x64, float, uint16_t, uint16_t, 5, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_nd_int8_int8_to_int32_96x128x65, int32_t, int8_t, int8_t, 6, 96, 128, 65)
DEFINE_TMATMUL_CASE(case_nd_fp32_fp32_to_fp32_33x63x31, float, float, float, 7, 33, 63, 31)
DEFINE_TMATMUL_CASE(case_nd_fp16_fp16_to_fp32_2x80x48, float, uint16_t, uint16_t, 8, 2, 80, 48)
DEFINE_TMATMUL_CASE(case_fp16_fp16_to_fp32_127x33x95, float, uint16_t, uint16_t, 9, 127, 33, 95)
DEFINE_TMATMUL_CASE(case_int8_int8_to_int32_17x33x31, int32_t, int8_t, int8_t, 10, 17, 33, 31)
DEFINE_TMATMUL_CASE(case_fp32_fp32_to_fp32_63x31x15, float, float, float, 11, 63, 31, 15)
DEFINE_TMATMUL_CASE(case_nd_fp16_fp16_to_fp32_95x33x79, float, uint16_t, uint16_t, 12, 95, 33, 79)
DEFINE_TMATMUL_CASE(case_nd_int8_int8_to_int32_129x95x33, int32_t, int8_t, int8_t, 13, 129, 95, 33)
DEFINE_TMATMUL_CASE(case_nd_fp32_fp32_to_fp32_47x29x25, float, float, float, 14, 47, 29, 25)
DEFINE_TMATMUL_CASE(case_mmad_s8s4_nd_64x64x64, int32_t, int8_t, pto::int4b_t, 15, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_s8s4_nd_96x128x65, int32_t, int8_t, pto::int4b_t, 16, 96, 128, 65)
DEFINE_TMATMUL_CASE(case_mmad_s8s4_nd_129x95x33, int32_t, int8_t, pto::int4b_t, 17, 129, 95, 33)
DEFINE_TMATMUL_CASE(case_mmad_s8s4_nd_17x33x31, int32_t, int8_t, pto::int4b_t, 18, 17, 33, 31)
DEFINE_TMATMUL_CASE(case_mmad_s8s4_nd_2x80x48, int32_t, int8_t, pto::int4b_t, 19, 2, 80, 48)
DEFINE_TMATMUL_CASE(case_mmad_f16s8_nd_64x64x64, float, uint16_t, int8_t, 20, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_f16s8_nd_96x128x89, float, uint16_t, int8_t, 21, 96, 128, 89)
DEFINE_TMATMUL_CASE(case_mmad_f16s8_nd_129x95x63, float, uint16_t, int8_t, 22, 129, 95, 63)
DEFINE_TMATMUL_CASE(case_mmad_f16s8_dn_65x90x89, float, uint16_t, int8_t, 23, 65, 90, 89)
DEFINE_TMATMUL_CASE(case_mmad_f16s8_nd_2x90x31, float, uint16_t, int8_t, 24, 2, 90, 31)
DEFINE_TMATMUL_CASE(case_mmad_f16f32_nd_64x64x64, float, uint16_t, uint16_t, 25, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_f16f32_nd_95x33x79, float, uint16_t, uint16_t, 26, 95, 33, 79)
DEFINE_TMATMUL_CASE(case_mmad_f16f32_dn_127x33x95, float, uint16_t, uint16_t, 27, 127, 33, 95)
DEFINE_TMATMUL_CASE(case_mmad_f16e4m3_nd_64x64x64, float, uint16_t, uint8_t, 28, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_f16e4m3_dn_127x64x95, float, uint16_t, uint8_t, 29, 127, 64, 95)
DEFINE_TMATMUL_CASE(case_mmad_bf16e4m3_nd_64x64x64, float, uint16_t, uint8_t, 30, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_bf16e4m3_dn_127x64x95, float, uint16_t, uint8_t, 31, 127, 64, 95)
DEFINE_TMATMUL_CASE(case_mmad_bf16s8_nd_64x64x64, float, uint16_t, int8_t, 32, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_bf16s8_dn_65x90x89, float, uint16_t, int8_t, 33, 65, 90, 89)
DEFINE_TMATMUL_CASE(case_mmad_f16s4_nd_64x64x64, float, uint16_t, pto::int4b_t, 34, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_f16s4_nd_65x90x89, float, uint16_t, pto::int4b_t, 35, 65, 90, 89)
DEFINE_TMATMUL_CASE(case_mmad_f16s4_nd_96x128x89, float, uint16_t, pto::int4b_t, 36, 96, 128, 89)
DEFINE_TMATMUL_CASE(case_mmad_f16s4_nd_129x95x63, float, uint16_t, pto::int4b_t, 37, 129, 95, 63)
DEFINE_TMATMUL_CASE(case_mmad_f16s4_nd_16x64x32, float, uint16_t, pto::int4b_t, 38, 16, 64, 32)
DEFINE_TMATMUL_CASE(case_mmad_f16s4_nd_128x128x128, float, uint16_t, pto::int4b_t, 39, 128, 128, 128)
DEFINE_TMATMUL_CASE(case_mmad_bf16s4_nd_64x64x64, float, uint16_t, pto::int4b_t, 40, 64, 64, 64)
DEFINE_TMATMUL_CASE(case_mmad_bf16s4_nd_65x90x89, float, uint16_t, pto::int4b_t, 41, 65, 90, 89)
DEFINE_TMATMUL_CASE(case_mmad_bf16s4_nd_96x128x89, float, uint16_t, pto::int4b_t, 42, 96, 128, 89)
DEFINE_TMATMUL_CASE(case_mmad_bf16s4_nd_129x95x63, float, uint16_t, pto::int4b_t, 43, 129, 95, 63)
DEFINE_TMATMUL_CASE(case_mmad_bf16s4_nd_16x64x32, float, uint16_t, pto::int4b_t, 44, 16, 64, 32)
DEFINE_TMATMUL_CASE(case_mmad_bf16s4_nd_128x128x128, float, uint16_t, pto::int4b_t, 45, 128, 128, 128)
DEFINE_TMATMUL_CASE(case_mmad_bf16s8_nd_96x128x89, float, uint16_t, int8_t, 46, 96, 128, 89)
DEFINE_TMATMUL_CASE(case_mmad_bf16s8_nd_129x95x63, float, uint16_t, int8_t, 47, 129, 95, 63)
DEFINE_TMATMUL_CASE(case_mmad_bf16s8_nd_2x90x31, float, uint16_t, int8_t, 48, 2, 90, 31)
DEFINE_TMATMUL_CASE(case_mmad_bf16e4m3_nd_95x64x95, float, uint16_t, uint8_t, 49, 95, 64, 95)
DEFINE_TMATMUL_CASE(case_mmad_bf16e4m3_nd_2x64x31, float, uint16_t, uint8_t, 50, 2, 64, 31)
DEFINE_TMATMUL_CASE(case_mmad_f16f32_nd_2x80x48, float, uint16_t, uint16_t, 51, 2, 80, 48)
DEFINE_TMATMUL_CASE(case_mmad_f16f32_nd_128x128x128, float, uint16_t, uint16_t, 52, 128, 128, 128)

#undef DEFINE_TMATMUL_CASE
