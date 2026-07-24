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
void launchTMOVAcc2MatNZ2NZ(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

template <int32_t tilingKey>
void launchTMOVAcc2MatFBQuantNz(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);

template <int32_t tilingKey>
void launchTMOVAcc2MatSCQuantNz(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

class TMOVTest : public testing::Test {
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

template <
    typename CType, typename AType, typename BType, int32_t key, uint32_t IdxRow = 0, uint32_t IdxCol = 0,
    bool isInsert = false, uint32_t DstRow = 0, uint32_t DstCol = 0>
void tmov_acc2mat_nz2nz_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = (M - IdxRow) * (N - IdxCol) * sizeof(CType);
    if (isInsert) {
        cFileSize = DstRow * DstCol * sizeof(CType);
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src2Host), cFileSize);

    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    if (isInsert) {
        ReadFile(GetGoldenDir() + "/dst.bin", cFileSize, src2Host, cFileSize);
    }

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (isInsert) {
        aclrtMemcpy(src2Device, cFileSize, src2Host, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    launchTMOVAcc2MatNZ2NZ<key>(dstDevice, src0Device, src1Device, src2Device, stream);

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
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <
    typename CType, typename AType, typename BType, typename QuantType, int32_t key, uint32_t IdxRow = 0,
    uint32_t IdxCol = 0, bool isInsert = false, uint32_t DstRow = 0, uint32_t DstCol = 0>
void tmov_acc2mat_nz2nz_fb_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = (M - IdxRow) * (N - IdxCol) * sizeof(CType);
    if (isInsert) {
        cFileSize = DstRow * DstCol * sizeof(CType);
    }
    size_t FBQuantFileSize = N * sizeof(QuantType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *src3Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), FBQuantFileSize);
    aclrtMallocHost((void**)(&src3Host), cFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, FBQuantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src3Device, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", FBQuantFileSize, src2Host, FBQuantFileSize);
    if (isInsert) {
        ReadFile(GetGoldenDir() + "/dst.bin", cFileSize, src3Host, cFileSize);
    }

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, FBQuantFileSize, src2Host, FBQuantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (isInsert) {
        aclrtMemcpy(src3Device, cFileSize, src3Host, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    launchTMOVAcc2MatFBQuantNz<key>(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);
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

template <
    typename CType, typename AType, typename BType, int32_t key, uint32_t IdxRow = 0, uint32_t IdxCol = 0,
    bool isInsert = false, uint32_t DstRow = 0, uint32_t DstCol = 0>
void tmov_acc2mat_nz2nz_sc_quant_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = (M - IdxRow) * (N - IdxCol) * sizeof(CType);
    if (isInsert) {
        cFileSize = DstRow * DstCol * sizeof(CType);
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), cFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    if (isInsert) {
        ReadFile(GetGoldenDir() + "/dst.bin", cFileSize, src2Host, cFileSize);
    }

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (isInsert) {
        aclrtMemcpy(src2Device, cFileSize, src2Host, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    launchTMOVAcc2MatSCQuantNz<key>(dstDevice, src0Device, src1Device, src2Device, stream);

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
TEST_F(TMOVTest, case_nz2nz_1)
{
    uint32_t M = 64;
    uint32_t K = 128;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_test<uint16_t, uint16_t, uint16_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_2)
{
    uint32_t M = 48;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_test<uint16_t, uint16_t, uint16_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_3)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<uint16_t, int8_t, int8_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_4)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<uint16_t, int8_t, int8_t, uint64_t, 1>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_5)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<int8_t, uint16_t, uint16_t, 2>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_6)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<int8_t, uint16_t, uint16_t, uint64_t, 2>(M, K, N);
}
TEST_F(TMOVTest, case_nz2nz_sc_quant_7)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<int8_t, int8_t, int8_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_8)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<int8_t, int8_t, int8_t, uint64_t, 3>(M, K, N);
}
TEST_F(TMOVTest, case_nz2nz_sc_quant_9)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<uint8_t, int8_t, int8_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_10)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<uint8_t, int8_t, int8_t, uint64_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_11)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<int16_t, int8_t, int8_t, 5>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_12)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<int16_t, int8_t, int8_t, uint64_t, 5>(M, K, N);
}
TEST_F(TMOVTest, case_nz2nz_21)
{
    uint32_t M = 16;
    uint32_t K = 16;
    uint32_t N = 16;

    tmov_acc2mat_nz2nz_test<uint16_t, uint16_t, uint16_t, 3>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_22)
{
    uint32_t M = 48;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_test<uint16_t, uint16_t, uint16_t, 4>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_23)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<uint16_t, int8_t, int8_t, 6>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_24)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<uint16_t, int8_t, int8_t, uint64_t, 6>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_25)
{
    uint32_t M = 48;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_sc_quant_test<int8_t, uint16_t, uint16_t, 7>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_26)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<int8_t, uint16_t, uint16_t, uint64_t, 7>(M, K, N);
}
TEST_F(TMOVTest, case_nz2nz_sc_quant_27)
{
    uint32_t M = 16;
    uint32_t K = 32;
    uint32_t N = 32;

    tmov_acc2mat_nz2nz_sc_quant_test<int8_t, int8_t, int8_t, 8>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_28)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<int8_t, int8_t, int8_t, uint64_t, 8>(M, K, N);
}
TEST_F(TMOVTest, case_nz2nz_sc_quant_29)
{
    uint32_t M = 16;
    uint32_t K = 32;
    uint32_t N = 32;

    tmov_acc2mat_nz2nz_sc_quant_test<uint8_t, int8_t, int8_t, 9>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_30)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<uint8_t, int8_t, int8_t, uint64_t, 9>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_31)
{
    uint32_t M = 16;
    uint32_t K = 32;
    uint32_t N = 32;

    tmov_acc2mat_nz2nz_sc_quant_test<int16_t, int8_t, int8_t, 10>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_32)
{
    uint32_t M = 80;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_fb_quant_test<int16_t, int8_t, int8_t, uint64_t, 10>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_extract)
{
    uint32_t M = 64;
    uint32_t K = 64;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_test<uint16_t, uint16_t, uint16_t, 5, 16, 16>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_extract)
{
    uint32_t M = 96;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_sc_quant_test<uint16_t, int8_t, int8_t, 11, 48, 48>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_extract)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_fb_quant_test<int8_t, uint16_t, uint16_t, uint64_t, 11, 32, 32>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_insert)
{
    uint32_t M = 32;
    uint32_t K = 32;
    uint32_t N = 32;

    tmov_acc2mat_nz2nz_test<uint16_t, uint16_t, uint16_t, 6, 32, 32, true, 128, 128>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_sc_quant_insert)
{
    uint32_t M = 96;
    uint32_t K = 128;
    uint32_t N = 64;

    tmov_acc2mat_nz2nz_sc_quant_test<uint16_t, int8_t, int8_t, 12, 48, 48, true, 256, 256>(M, K, N);
}

TEST_F(TMOVTest, case_nz2nz_fb_quant_insert)
{
    uint32_t M = 128;
    uint32_t K = 64;
    uint32_t N = 128;

    tmov_acc2mat_nz2nz_fb_quant_test<int8_t, uint16_t, uint16_t, uint64_t, 12, 32, 32, true, 256, 256>(M, K, N);
}
