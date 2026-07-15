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
void launchTEXTRACT(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void launchTMOV(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void launchTEXTRACTMX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* srcMx0, uint8_t* srcMx1, void* stream);

template <int32_t tilingKey>
void launchTMOVMX(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* srcMx0, uint8_t* srcMx1, void* stream);

class TEXTRACTTest : public testing::Test {
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
void textract_test(uint32_t M, uint32_t K, uint32_t N, uint16_t indexM, uint16_t indexK, uint16_t indexN)
{
    uint32_t mValid = M - indexM;
    uint32_t nValid = N - indexN;
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(U);
    size_t cFileSize = mValid * nValid * sizeof(T);

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

template <int32_t key, typename T, typename U, typename S>
void textract_mx_test(uint32_t M, uint32_t K, uint32_t N, uint16_t indexM, uint16_t indexK, uint16_t indexN)
{
    uint32_t mValid = M - indexM;
    uint32_t nValid = N - indexN;
    size_t aFileSize = M * K * sizeof(U) / 2;
    size_t bFileSize = K * N * sizeof(U) / 2;
    size_t amxFileSize = M * K / 32 * sizeof(int8_t);
    size_t bmxFileSize = K / 32 * N * sizeof(int8_t);
    size_t cFileSize = mValid * nValid * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *srcMx0Host, *srcMx1Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *srcMx0Device, *srcMx1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&srcMx0Host), amxFileSize);
    aclrtMallocHost((void**)(&srcMx1Host), bmxFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx0Device, amxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx1Device, bmxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx_gm.bin", amxFileSize, srcMx0Host, amxFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx_gm.bin", bmxFileSize, srcMx1Host, bmxFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx0Device, amxFileSize, srcMx0Host, amxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx1Device, bmxFileSize, srcMx1Host, bmxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTEXTRACTMX<key>(dstDevice, src0Device, src1Device, srcMx0Device, srcMx1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(srcMx0Device);
    aclrtFree(srcMx1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(srcMx0Host);
    aclrtFreeHost(srcMx1Host);
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

TEST_F(TEXTRACTTest, case1) { textract_test<1, float, uint16_t, uint16_t>(32, 96, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case2) { textract_test<2, float, float, float>(128, 48, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case3) { textract_test<3, int32_t, int8_t, int8_t>(128, 128, 64, 0, 0, 0); }

TEST_F(TEXTRACTTest, case4) { textract_test<4, float, uint16_t, uint16_t>(64, 96, 64, 32, 16, 16); }

TEST_F(TEXTRACTTest, case5) { textract_test<5, float, float, float>(64, 128, 64, 32, 32, 16); }

TEST_F(TEXTRACTTest, case6) { textract_test<6, int32_t, int8_t, int8_t>(128, 128, 64, 32, 64, 32); }

TEST_F(TEXTRACTTest, case7) { textract_test<7, float, uint16_t, uint16_t>(64, 128, 64, 0, 64, 0); }

TEST_F(TEXTRACTTest, case8) { textract_test<8, float, float, float>(64, 64, 128, 0, 0, 32); }

TEST_F(TEXTRACTTest, case9) { textract_test<9, int32_t, int8_t, int8_t>(128, 64, 128, 32, 0, 0); }

TEST_F(TEXTRACTTest, case10) { textract_test<10, float, uint16_t, uint16_t>(64, 128, 64, 16, 0, 0); }

TEST_F(TEXTRACTTest, case11) { textract_test<11, float, int8_t, int8_t>(64, 128, 64, 0, 32, 0); }

TEST_F(TEXTRACTTest, case12) { textract_test<12, float, int8_t, int8_t>(64, 128, 64, 0, 0, 32); }

TEST_F(TEXTRACTTest, case13) { textract_test<13, float, int8_t, int8_t>(64, 128, 64, 0, 32, 0); }

TEST_F(TEXTRACTTest, case14) { textract_test<14, float, int8_t, int8_t>(64, 96, 32, 32, 0, 0); }

TEST_F(TEXTRACTTest, case15) { textract_test<15, float, uint16_t, uint16_t>(64, 48, 96, 16, 16, 0); }

TEST_F(TEXTRACTTest, case16) { textract_test<16, float, float, float>(32, 96, 48, 0, 32, 16); }

TEST_F(TEXTRACTTest, case17) { textract_mx_test<17, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case18) { textract_mx_test<18, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case19) { textract_mx_test<19, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

TEST_F(TEXTRACTTest, case20) { textract_mx_test<20, float, int8_t, int8_t>(256, 128, 256, 128, 64, 128); }

class TMOVTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <int32_t key, typename T, typename U, typename S>
void tmov_test(uint32_t M, uint32_t K, uint32_t N, uint32_t targetM = 0, uint32_t targetK = 0, uint32_t targetN = 0)
{
    if (targetM == 0)
        targetM = M;
    if (targetN == 0)
        targetN = N;
    if (targetK == 0)
        targetK = K;
    if (targetM < M || targetN < N || targetK < K) {
        printf("Error: targetM targetN targetK should large than M N K");
        return;
    }
    size_t aFileSize = targetM * targetK * sizeof(U);
    size_t bFileSize = targetK * targetN * sizeof(U);
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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <int32_t key, typename T, typename U, typename S>
void tmov_mx_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U) / 2;
    size_t bFileSize = K * N * sizeof(U) / 2;
    size_t amxFileSize = M * K / 32 * sizeof(int8_t);
    size_t bmxFileSize = K / 32 * N * sizeof(int8_t);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *srcMx0Host, *srcMx1Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *srcMx0Device, *srcMx1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&srcMx0Host), amxFileSize);
    aclrtMallocHost((void**)(&srcMx1Host), bmxFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx0Device, amxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcMx1Device, bmxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/x1_mx_gm.bin", amxFileSize, srcMx0Host, amxFileSize);
    ReadFile(GetGoldenDir() + "/x2_mx_gm.bin", bmxFileSize, srcMx1Host, bmxFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx0Device, amxFileSize, srcMx0Host, amxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcMx1Device, bmxFileSize, srcMx1Host, bmxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOVMX<key>(dstDevice, src0Device, src1Device, srcMx0Device, srcMx1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(srcMx0Device);
    aclrtFree(srcMx1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(srcMx0Host);
    aclrtFreeHost(srcMx1Host);
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

TEST_F(TMOVTest, case1) { tmov_test<1, float, uint16_t, uint16_t>(32, 96, 64); }

TEST_F(TMOVTest, case2) { tmov_test<2, float, float, float>(128, 48, 64); }

TEST_F(TMOVTest, case3) { tmov_test<3, int32_t, int8_t, int8_t>(128, 128, 64); }

TEST_F(TMOVTest, case4) { tmov_test<4, float, uint16_t, uint16_t>(64, 128, 64); }

TEST_F(TMOVTest, case5) { tmov_test<5, float, int8_t, int8_t>(64, 96, 64); }

TEST_F(TMOVTest, case6) { tmov_test<6, float, int8_t, int8_t>(64, 128, 64); }

TEST_F(TMOVTest, case7) { tmov_test<7, float, int8_t, int8_t>(128, 128, 64); }

TEST_F(TMOVTest, case8) { tmov_test<8, float, int8_t, int8_t>(64, 96, 64); }

TEST_F(TMOVTest, case9) { tmov_test<9, float, uint16_t, uint16_t>(64, 128, 64); }

TEST_F(TMOVTest, case10) { tmov_test<10, float, float, float>(64, 128, 64); }

TEST_F(TMOVTest, case11) { tmov_test<11, int32_t, int8_t, int8_t>(65, 40, 66, 96, 64, 96); }

TEST_F(TMOVTest, case12) { tmov_test<12, float, uint16_t, uint16_t>(65, 40, 66, 80, 48, 80); }

TEST_F(TMOVTest, case13) { tmov_test<13, float, float, float>(65, 40, 66, 80, 48, 80); }

TEST_F(TMOVTest, case14) { tmov_mx_test<14, float, int8_t, int8_t>(128, 64, 128); }

TEST_F(TMOVTest, case15) { tmov_mx_test<15, float, int8_t, int8_t>(128, 64, 128); }

TEST_F(TMOVTest, case16) { tmov_mx_test<16, float, int8_t, int8_t>(128, 64, 128); }

TEST_F(TMOVTest, case17) { tmov_mx_test<17, float, int8_t, int8_t>(128, 64, 128); }