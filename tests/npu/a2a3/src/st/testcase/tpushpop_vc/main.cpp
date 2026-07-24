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
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopVCMatmul(
    uint8_t* ffts, uint8_t* out, uint8_t* srcA, uint8_t* quantB, uint8_t* scale, uint8_t* offset, uint8_t* fifoMem,
    void* stream);

class TPushPopVCTest : public testing::Test {
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

template <typename QuantT, typename InT, typename OutT, int32_t key>
void TPushPopVCMatmulTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(InT);
    size_t quantBFileSize = K * N * sizeof(QuantT);
    size_t scaleFileSize = K * sizeof(OutT);
    size_t offsetFileSize = K * sizeof(OutT);
    size_t cFileSize = M * N * sizeof(OutT);
    size_t fifoFileSize = 2 * 64 * N * sizeof(OutT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcAHost, *quantBHost, *scaleHost, *offsetHost;
    uint8_t *dstDevice, *srcADevice, *quantBDevice, *scaleDevice, *offsetDevice, *fifoMemDevice;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&quantBHost), quantBFileSize);
    aclrtMallocHost((void**)(&scaleHost), scaleFileSize);
    aclrtMallocHost((void**)(&offsetHost), offsetFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&quantBDevice, quantBFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scaleDevice, scaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&offsetDevice, offsetFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/quant_b_gm.bin", quantBFileSize, quantBHost, quantBFileSize);
    ReadFile(GetGoldenDir() + "/scale_gm.bin", scaleFileSize, scaleHost, scaleFileSize);
    ReadFile(GetGoldenDir() + "/offset_gm.bin", offsetFileSize, offsetHost, offsetFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantBDevice, quantBFileSize, quantBHost, quantBFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDevice, scaleFileSize, scaleHost, scaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, offsetFileSize, offsetHost, offsetFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopVCMatmul<key>(
        (uint8_t*)ffts, dstDevice, srcADevice, quantBDevice, scaleDevice, offsetDevice, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(quantBDevice);
    aclrtFree(scaleDevice);
    aclrtFree(offsetDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(quantBHost);
    aclrtFreeHost(scaleHost);
    aclrtFreeHost(offsetHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<OutT> golden(cFileSize);
    std::vector<OutT> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// TILE_UP_DOWN: vector cores split quantB along K rows
TEST_F(TPushPopVCTest, case1_int8_single_k_tile) { TPushPopVCMatmulTestFunc<int8_t, float, float, 1>(16, 64, 32); }

TEST_F(TPushPopVCTest, case2_int8_two_k_tiles) { TPushPopVCMatmulTestFunc<int8_t, float, float, 2>(16, 128, 32); }

TEST_F(TPushPopVCTest, case3_int8_four_k_tiles) { TPushPopVCMatmulTestFunc<int8_t, float, float, 3>(16, 256, 32); }

TEST_F(TPushPopVCTest, case4_int16_single_k_tile) { TPushPopVCMatmulTestFunc<int16_t, float, float, 4>(16, 64, 32); }

TEST_F(TPushPopVCTest, case5_int16_two_k_tiles) { TPushPopVCMatmulTestFunc<int16_t, float, float, 5>(16, 128, 32); }

TEST_F(TPushPopVCTest, case6_int16_four_k_tiles) { TPushPopVCMatmulTestFunc<int16_t, float, float, 6>(16, 256, 32); }

// TILE_LEFT_RIGHT: vector cores split quantB along N columns
// int8_t uses N=64 so PROD_N=32 satisfies the 32-byte alignment for QuantTile
TEST_F(TPushPopVCTest, case7_int8_single_k_tile_left_right)
{
    TPushPopVCMatmulTestFunc<int8_t, float, float, 7>(16, 64, 64);
}

TEST_F(TPushPopVCTest, case8_int8_two_k_tiles_left_right)
{
    TPushPopVCMatmulTestFunc<int8_t, float, float, 8>(16, 128, 64);
}

TEST_F(TPushPopVCTest, case9_int8_four_k_tiles_left_right)
{
    TPushPopVCMatmulTestFunc<int8_t, float, float, 9>(16, 256, 64);
}

TEST_F(TPushPopVCTest, case10_int16_single_k_tile_left_right)
{
    TPushPopVCMatmulTestFunc<int16_t, float, float, 10>(16, 64, 32);
}

TEST_F(TPushPopVCTest, case11_int16_two_k_tiles_left_right)
{
    TPushPopVCMatmulTestFunc<int16_t, float, float, 11>(16, 128, 32);
}

TEST_F(TPushPopVCTest, case12_int16_four_k_tiles_left_right)
{
    TPushPopVCMatmulTestFunc<int16_t, float, float, 12>(16, 256, 32);
}

template <int32_t tilingKey>
void LaunchTPushPopVCSubBlockId(
    uint8_t* ffts, uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* fifoMem, void* stream);

// TPUSH with explicit subBlockId: Vec TADD + push, Cube matmul + store
TEST_F(TPushPopVCTest, case13_float_subblock_id_up_down)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 64;

    size_t src0FileSize = M * K * sizeof(float);
    size_t src1FileSize = M * K * sizeof(float);
    size_t src2FileSize = K * N * sizeof(float);
    size_t outFileSize = M * N * sizeof(float);
    size_t fifoFileSize = 2 * M * K * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *outHost, *src0Host, *src1Host, *src2Host;
    uint8_t *outDevice, *src0Device, *src1Device, *src2Device, *fifoMemDevice;

    aclrtMallocHost((void**)(&outHost), outFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);
    aclrtMallocHost((void**)(&src2Host), src2FileSize);

    aclrtMalloc((void**)&outDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, src2FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0_gm.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1_gm.bin", src1FileSize, src1Host, src1FileSize);
    ReadFile(GetGoldenDir() + "/src2_gm.bin", src2FileSize, src2Host, src2FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, src2FileSize, src2Host, src2FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopVCSubBlockId<1>((uint8_t*)ffts, outDevice, src0Device, src1Device, src2Device, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outFileSize, outDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, outFileSize);

    aclrtFree(outDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(outHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outFileSize);
    std::vector<float> devFinal(outFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outFileSize, golden.data(), outFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}
