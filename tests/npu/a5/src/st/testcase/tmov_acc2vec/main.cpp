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
void LaunchTMOVAcc2VecNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2NZ(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecFBQuantNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void LaunchTMOVAcc2VecSCQuantNZ2DN(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

class TMOVTest : public testing::Test {
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

template <int32_t funcKey, typename CType, typename AType, typename BType, int32_t key>
void tmov_acc2vec_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (funcKey == 1) {
        LaunchTMOVAcc2VecNZ2ND<key>(dstDevice, src0Device, src1Device, stream);
    } else if constexpr (funcKey == 2) {
        LaunchTMOVAcc2VecNZ2NZ<key>(dstDevice, src0Device, src1Device, stream);
    } else if constexpr (funcKey == 3) {
        LaunchTMOVAcc2VecNZ2DN<key>(dstDevice, src0Device, src1Device, stream);
    } else if constexpr (funcKey == 4) {
        LaunchTMOVAcc2VecSCQuantNZ2ND<key>(dstDevice, src0Device, src1Device, stream);
    } else if constexpr (funcKey == 5) {
        LaunchTMOVAcc2VecSCQuantNZ2NZ<key>(dstDevice, src0Device, src1Device, stream);
    } else if constexpr (funcKey == 6) {
        LaunchTMOVAcc2VecSCQuantNZ2DN<key>(dstDevice, src0Device, src1Device, stream);
    }

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

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <int32_t funcKey, typename CType, typename AType, typename BType, typename QuantType, int32_t key>
void tmov_acc2vec_fb_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = M * N * sizeof(CType);
    size_t FBQuantFileSize = N * sizeof(QuantType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), FBQuantFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, FBQuantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", FBQuantFileSize, src2Host, FBQuantFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, FBQuantFileSize, src2Host, FBQuantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (funcKey == 1) {
        LaunchTMOVAcc2VecFBQuantNZ2ND<key>(dstDevice, src0Device, src1Device, src2Device, stream);
    } else if constexpr (funcKey == 2) {
        LaunchTMOVAcc2VecFBQuantNZ2NZ<key>(dstDevice, src0Device, src1Device, src2Device, stream);
    } else if constexpr (funcKey == 3) {
        LaunchTMOVAcc2VecFBQuantNZ2DN<key>(dstDevice, src0Device, src1Device, src2Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case_nz2nd_1)
{
    tmov_acc2vec_test<1, uint32_t, uint16_t, uint16_t, 1>(60, 127, 120);
}

TEST_F(TMOVTest, case_nz2nd_2)
{
    tmov_acc2vec_test<1, uint16_t, uint16_t, uint16_t, 2>(110, 100, 80);
}

TEST_F(TMOVTest, case_nz2nd_3)
{
    tmov_acc2vec_test<1, uint32_t, uint16_t, uint16_t, 3>(6, 7, 8);
}

TEST_F(TMOVTest, case_nz2nd_4)
{
    tmov_acc2vec_test<1, uint16_t, uint16_t, uint16_t, 4>(111, 47, 96);
}

TEST_F(TMOVTest, case_nz2nd_split_1)
{
    tmov_acc2vec_test<1, uint32_t, uint16_t, uint16_t, 5>(96, 32, 48);
}

TEST_F(TMOVTest, case_nz2nd_split_2)
{
    tmov_acc2vec_test<1, uint32_t, uint16_t, uint16_t, 6>(48, 32, 128);
}

TEST_F(TMOVTest, case_nz2nd_uf_partial)
{
    tmov_acc2vec_test<1, uint32_t, uint16_t, uint16_t, 7>(6, 7, 8);
}

TEST_F(TMOVTest, case_nz2nd_uf_final_mode)
{
    tmov_acc2vec_test<1, uint32_t, uint16_t, uint16_t, 8>(6, 7, 8);
}

TEST_F(TMOVTest, case_nz2nz_1)
{
    tmov_acc2vec_test<2, uint16_t, uint16_t, uint16_t, 1>(96, 80, 112);
}

TEST_F(TMOVTest, case_nz2nz_2)
{
    tmov_acc2vec_test<2, uint32_t, uint16_t, uint16_t, 2>(80, 112, 96);
}

TEST_F(TMOVTest, case_nz2nz_3)
{
    tmov_acc2vec_test<2, uint32_t, uint32_t, uint32_t, 3>(16, 16, 16);
}

TEST_F(TMOVTest, case_nz2nz_4)
{
    tmov_acc2vec_test<2, uint16_t, uint16_t, uint16_t, 4>(48, 112, 48);
}

TEST_F(TMOVTest, case_nz2nz_split_1)
{
    tmov_acc2vec_test<2, uint32_t, uint32_t, uint32_t, 5>(48, 80, 128);
}

TEST_F(TMOVTest, case_nz2nz_split_2)
{
    tmov_acc2vec_test<2, uint32_t, uint16_t, uint16_t, 6>(80, 16, 96);
}

TEST_F(TMOVTest, case_nz2nz_split_3)
{
    tmov_acc2vec_test<2, uint32_t, uint16_t, uint16_t, 7>(112, 48, 80);
}

TEST_F(TMOVTest, case_nz2nz_split_4)
{
    tmov_acc2vec_test<2, uint32_t, uint32_t, uint32_t, 8>(16, 112, 112);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_1)
{
    tmov_acc2vec_fb_quant_test<2, int8_t, int8_t, int8_t, uint64_t, 1>(128, 48, 128);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_2)
{
    tmov_acc2vec_fb_quant_test<2, uint16_t, int8_t, int8_t, uint64_t, 2>(64, 80, 96);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_3)
{
    tmov_acc2vec_fb_quant_test<2, int8_t, uint32_t, uint32_t, uint64_t, 3>(128, 32, 96);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_4)
{
    tmov_acc2vec_fb_quant_test<2, uint16_t, uint32_t, uint32_t, uint64_t, 4>(80, 16, 112);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_1)
{
    tmov_acc2vec_test<5, uint16_t, uint32_t, uint32_t, 1>(48, 32, 80);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_2)
{
    tmov_acc2vec_test<5, uint16_t, int8_t, int8_t, 2>(96, 48, 128);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_3)
{
    tmov_acc2vec_test<5, int8_t, int8_t, int8_t, 3>(128, 64, 128);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_4)
{
    tmov_acc2vec_test<5, int8_t, uint32_t, uint32_t, 4>(64, 80, 96);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_1)
{
    tmov_acc2vec_fb_quant_test<1, int8_t, int8_t, int8_t, uint64_t, 1>(30, 48, 64);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_2)
{
    tmov_acc2vec_fb_quant_test<1, uint16_t, int8_t, uint8_t, uint64_t, 2>(60, 128, 32);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_3)
{
    tmov_acc2vec_fb_quant_test<1, uint16_t, int8_t, uint8_t, uint64_t, 3>(128, 64, 96);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_4)
{
    tmov_acc2vec_fb_quant_test<1, int8_t, uint32_t, uint32_t, uint64_t, 4>(60, 128, 64);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_5)
{
    tmov_acc2vec_fb_quant_test<1, uint16_t, uint32_t, uint32_t, uint64_t, 5>(31, 128, 128);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_uf_partial)
{
    tmov_acc2vec_fb_quant_test<1, uint16_t, uint32_t, uint32_t, uint64_t, 6>(31, 128, 128);
}

TEST_F(TMOVTest, case_nz2nd_fb_quant_uf_final_mode)
{
    tmov_acc2vec_fb_quant_test<1, int8_t, uint32_t, uint32_t, uint64_t, 7>(60, 128, 64);
}

TEST_F(TMOVTest, case_nz2nd_sc_quant_1)
{
    tmov_acc2vec_test<4, uint16_t, uint32_t, uint32_t, 1>(128, 48, 96);
}

TEST_F(TMOVTest, case_nz2nd_sc_quant_2)
{
    tmov_acc2vec_test<4, int8_t, uint32_t, uint32_t, 2>(60, 128, 64);
}

TEST_F(TMOVTest, case_nz2nd_sc_quant_3)
{
    tmov_acc2vec_test<4, uint16_t, int8_t, int8_t, 3>(30, 48, 64);
}

TEST_F(TMOVTest, case_nz2nd_sc_quant_4)
{
    tmov_acc2vec_test<4, int8_t, int8_t, int8_t, 4>(60, 128, 32);
}

TEST_F(TMOVTest, case_nz2nd_sc_quant_uf_partial)
{
    tmov_acc2vec_test<4, uint16_t, uint32_t, uint32_t, 5>(128, 48, 96);
}

TEST_F(TMOVTest, case_nz2nd_sc_quant_uf_final_mode)
{
    tmov_acc2vec_test<4, int8_t, uint32_t, uint32_t, 6>(60, 128, 64);
}

TEST_F(TMOVTest, case_nz2dn_1)
{
    tmov_acc2vec_test<3, uint32_t, uint32_t, uint32_t, 1>(8, 7, 6);
}

TEST_F(TMOVTest, case_nz2dn_2)
{
    tmov_acc2vec_test<3, uint16_t, uint16_t, uint16_t, 2>(112, 48, 95);
}

TEST_F(TMOVTest, case_nz2dn_3)
{
    tmov_acc2vec_test<3, uint16_t, uint16_t, uint16_t, 3>(48, 31, 31);
}

TEST_F(TMOVTest, case_nz2dn_4)
{
    tmov_acc2vec_test<3, uint32_t, uint16_t, uint16_t, 4>(88, 48, 95);
}

TEST_F(TMOVTest, case_nz2dn_fb_quant_1)
{
    tmov_acc2vec_fb_quant_test<3, int8_t, int8_t, int8_t, uint64_t, 1>(96, 128, 60);
}

TEST_F(TMOVTest, case_nz2dn_fb_quant_2)
{
    tmov_acc2vec_fb_quant_test<3, uint16_t, int8_t, int8_t, uint64_t, 2>(32, 48, 64);
}

TEST_F(TMOVTest, case_nz2dn_fb_quant_3)
{
    tmov_acc2vec_fb_quant_test<3, int8_t, uint16_t, uint16_t, uint64_t, 3>(32, 128, 60);
}

TEST_F(TMOVTest, case_nz2dn_fb_quant_4)
{
    tmov_acc2vec_fb_quant_test<3, uint16_t, uint16_t, uint16_t, uint64_t, 4>(64, 64, 90);
}

TEST_F(TMOVTest, case_nz2dn_sc_quant_1)
{
    tmov_acc2vec_test<6, uint16_t, float, float, 1>(80, 40, 66);
}

TEST_F(TMOVTest, case_nz2dn_sc_quant_2)
{
    tmov_acc2vec_test<6, int8_t, float, float, 2>(96, 128, 60);
}

TEST_F(TMOVTest, case_nz2dn_sc_quant_3)
{
    tmov_acc2vec_test<6, uint16_t, int8_t, int8_t, 3>(32, 128, 64);
}

TEST_F(TMOVTest, case_nz2dn_sc_quant_4)
{
    tmov_acc2vec_test<6, int8_t, int8_t, int8_t, 4>(64, 64, 90);
}
