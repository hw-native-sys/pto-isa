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

class TCOLEXPANDOPTest : public testing::Test {
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

template <typename T, int oRow, int oCol>
inline void InitDstDevice(T* dstDevice)
{
    constexpr int size = oRow * oCol;
    for (int k = 0; k < size; k++) {
        dstDevice[k] = T{0};
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDDIV(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDMUL(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDSUB(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDADD(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDMAX(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDMIN(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDEXPDIF(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols,
    typename LaunchFn>
void run_vec_op(LaunchFn fn)
{
    const size_t iMatSize = iRow * iCol;
    const size_t oMatSize = oRow * oCol;
    size_t iMatFileSize = iMatSize * sizeof(T);
    size_t oMatFileSize = oMatSize * sizeof(T);
    size_t vecFileSize = iCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), oMatFileSize);
    aclrtMallocHost((void**)(&src0Host), iMatFileSize);
    aclrtMallocHost((void**)(&src1Host), vecFileSize);

    aclrtMalloc((void**)&dstDevice, oMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, iMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, iMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    InitDstDevice<T, oRow, oCol>(dstDevice);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", iMatFileSize, src0Host, iMatFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", vecFileSize, src1Host, vecFileSize));
    aclrtMemcpy(src0Device, iMatFileSize, src0Host, iMatFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, vecFileSize, src1Host, vecFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    fn(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, oMatFileSize, dstDevice, oMatFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, oMatFileSize);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(oMatSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", oMatFileSize, golden.data(), oMatFileSize));
    std::vector<T> devFinal(oMatSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", oMatFileSize, devFinal.data(), oMatFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDOPTest, case_div_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDDIV<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_div_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDDIV<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDMUL<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDMUL<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDSUB<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDSUB<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDADD<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDADD<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDMAX<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDMAX<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDMIN<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDMIN<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_expdif_float_64x64_64x64_64x64)
{
    run_vec_op<float, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDEXPDIF<float, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_expdif_half_16x256_16x256_16x256)
{
    run_vec_op<aclFloat16, 16, 256>([](aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream) {
        LaunchTCOLEXPANDEXPDIF<aclFloat16, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_div_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDDIV<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDMUL<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDSUB<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDADD<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDMIN<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDMAX<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_expdif_float_16x16_32x32_64x64)
{
    run_vec_op<float, 16, 16, 32, 32, 64, 64>([](float* out, float* src0, float* src1, void* stream) {
        LaunchTCOLEXPANDEXPDIF<float, 16, 16, 32, 32, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_div_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTCOLEXPANDDIV<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTCOLEXPANDMUL<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTCOLEXPANDSUB<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTCOLEXPANDADD<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTCOLEXPANDMAX<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_int16_16x256_16x256_16x256)
{
    run_vec_op<int16_t, 16, 256>([](int16_t* out, int16_t* src0, int16_t* src1, void* stream) {
        LaunchTCOLEXPANDMIN<int16_t, 16, 256>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_div_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTCOLEXPANDDIV<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTCOLEXPANDMUL<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTCOLEXPANDSUB<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTCOLEXPANDADD<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTCOLEXPANDMAX<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_int32_64x64_64x64_64x64)
{
    run_vec_op<int32_t, 64, 64>([](int32_t* out, int32_t* src0, int32_t* src1, void* stream) {
        LaunchTCOLEXPANDMIN<int32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_div_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTCOLEXPANDDIV<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTCOLEXPANDMUL<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTCOLEXPANDSUB<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTCOLEXPANDADD<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTCOLEXPANDMAX<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_uint16_64x64_64x64_64x64)
{
    run_vec_op<uint16_t, 64, 64>([](uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream) {
        LaunchTCOLEXPANDMIN<uint16_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_div_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTCOLEXPANDDIV<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_mul_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTCOLEXPANDMUL<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_sub_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTCOLEXPANDSUB<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_add_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTCOLEXPANDADD<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_max_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTCOLEXPANDMAX<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}

TEST_F(TCOLEXPANDOPTest, case_min_uint32_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, 64, 64>([](uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream) {
        LaunchTCOLEXPANDMIN<uint32_t, 64, 64>(out, src0, src1, stream);
    });
}
