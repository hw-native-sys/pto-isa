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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TROWEXPANDOPTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <typename T, int rows, int cols>
inline void InitDstDevice(T* dst)
{
    constexpr int size = rows * cols;
    for (int k = 0; k < size; k++) {
        dst[k] = T{0};
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDDIV(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDMUL(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDSUB(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDADD(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDMAX(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDMIN(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDEXPDIF(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols,
    typename LaunchFn>
void run_vec_op(LaunchFn fn)
{
    const size_t iMatSize = iRow * iCol;
    const size_t oMatSize = oRow * oCol;
    size_t iMatSizeFile = iMatSize * sizeof(T);
    size_t oMatSizeFile = oMatSize * sizeof(T);
    size_t vecSizeFile = iRow * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), oMatSizeFile);
    aclrtMallocHost((void**)(&src0Host), iMatSizeFile);
    aclrtMallocHost((void**)(&src1Host), vecSizeFile);

    aclrtMalloc((void**)&dstDevice, oMatSizeFile, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, iMatSizeFile, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, iMatSizeFile, ACL_MEM_MALLOC_HUGE_FIRST);

    InitDstDevice<T, oRow, oCol>(dstDevice);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", iMatSizeFile, src0Host, iMatSizeFile));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", vecSizeFile, src1Host, vecSizeFile));
    aclrtMemcpy(src0Device, iMatSizeFile, src0Host, iMatSizeFile, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, vecSizeFile, src1Host, vecSizeFile, ACL_MEMCPY_HOST_TO_DEVICE);

    fn(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, oMatSizeFile, dstDevice, oMatSizeFile, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, oMatSizeFile);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(oMatSize);
    std::vector<T> devFinal(oMatSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", oMatSizeFile, golden.data(), oMatSizeFile));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", oMatSizeFile, devFinal.data(), oMatSizeFile));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDOPTest, case_div_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDDIV<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDDIV<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDMUL<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDMUL<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDSUB<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDSUB<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDADD<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDADD<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDMAX<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDMAX<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDMIN<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDMIN<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_expdif_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDEXPDIF<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_expdif_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTROWEXPANDEXPDIF<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDDIV<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDMUL<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDSUB<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDADD<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDMIN<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDMAX<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_expdif_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTROWEXPANDEXPDIF<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTROWEXPANDDIV<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTROWEXPANDMUL<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTROWEXPANDSUB<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTROWEXPANDADD<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTROWEXPANDMAX<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTROWEXPANDMIN<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTROWEXPANDDIV<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTROWEXPANDMUL<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTROWEXPANDSUB<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTROWEXPANDADD<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTROWEXPANDMAX<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTROWEXPANDMIN<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTROWEXPANDDIV<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTROWEXPANDMUL<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTROWEXPANDSUB<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTROWEXPANDADD<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTROWEXPANDMAX<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTROWEXPANDMIN<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_div_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTROWEXPANDDIV<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTROWEXPANDMUL<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_mul_uint8_64x64_64x64_64x64)
{
    run_vec_op<uint8_t, 64, 64>([](uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream) {
        LaunchTROWEXPANDMUL<uint8_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_sub_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTROWEXPANDSUB<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_add_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTROWEXPANDADD<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_max_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTROWEXPANDMAX<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TROWEXPANDOPTest, case_min_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTROWEXPANDMIN<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}
