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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TARGREDUCEOPTest : public testing::Test {
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
    return "../" + suiteName + "." + caseName;
}

template <typename T, int rows, int cols>
inline void InitDstDevice(T *dst)
{
    constexpr int size = rows * cols;
    for (int k = 0; k < size; k++) {
        dst[k] = T{0};
    }
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMAX(TIdx *out, T *src, void *stream);

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMIN(TIdx *out, T *src, void *stream);

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oIdxRow = kTRows,
          int oIdxCol = kTCols, int oValRow = kTRows, int oValCol = kTCols>
void LaunchTROWARGMAX(T *outVal, TIdx *outIdx, T *src, void *stream);

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oIdxRow = kTRows,
          int oIdxCol = kTCols, int oValRow = kTRows, int oValCol = kTCols>
void LaunchTROWARGMIN(T *outVal, TIdx *outIdx, T *src, void *stream);

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMAX(TIdx *out, T *src, void *stream);

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMIN(TIdx *out, T *src, void *stream);

template <typename T>
inline void CheckOutput(size_t vecSize, size_t &fileSize, const std::string &outputFile, const std::string &goldenFile)
{
    std::vector<T> golden(vecSize);
    std::vector<T> devFinal(vecSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + outputFile, fileSize, devFinal.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + goldenFile, fileSize, golden.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <typename TIdx, typename T, int kTRows, int kTCols, bool checkVal = false, int iRow = kTRows,
          int iCol = kTCols, int oRow = kTRows, int oCol = kTCols, int oVRow = oRow, int oVCol = oCol,
          typename LaunchFn>
void run_vec_op(LaunchFn fn)
{
    constexpr size_t iMatSize = iRow * iCol;
    constexpr size_t oMatSize = oRow * oCol;
    constexpr size_t oMatValSize = oVRow * oVCol;
    size_t iMatFileSize = iMatSize * sizeof(T);
    size_t oMatFileSize = oMatSize * sizeof(TIdx);
    size_t oMatValFileSize = oMatValSize * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    TIdx *dstHost;
    TIdx *dstDevice;

    T *dstValHost, *srcHost;
    T *dstValDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), oMatFileSize);
    aclrtMallocHost((void **)(&dstValHost), oMatValFileSize);
    aclrtMallocHost((void **)(&srcHost), iMatFileSize);

    aclrtMalloc((void **)&dstDevice, oMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstValDevice, oMatValFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, iMatFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    InitDstDevice<TIdx, oRow, oCol>(dstDevice);
    InitDstDevice<T, oVRow, oVCol>(dstValDevice);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", iMatFileSize, srcHost, iMatFileSize));
    aclrtMemcpy(srcDevice, iMatFileSize, srcHost, iMatFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    fn(dstValDevice, dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, oMatFileSize, dstDevice, oMatFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstValHost, oMatValFileSize, dstValDevice, oMatValFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, oMatFileSize);
    WriteFile(GetGoldenDir() + "/output_val.bin", dstValHost, oMatValFileSize);

    aclrtFree(dstDevice);
    aclrtFree(dstValDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(dstValHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    CheckOutput<TIdx>(oMatSize, oMatFileSize, "/output.bin", "/golden.bin");

    if constexpr (checkVal) {
        CheckOutput<T>(oMatValSize, oMatValFileSize, "/output_val.bin", "/golden_val.bin");
    }
}

TEST_F(TARGREDUCEOPTest, case_row_max_uint32_float_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, float, 64, 64>([](float *outVal, uint32_t *out, float *src, void *stream) {
        LaunchTROWARGMAX<uint32_t, float, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_max_int32_half_16x256_16x256_16x256)
{
    run_vec_op<int32_t, aclFloat16, 16, 256>([](aclFloat16 *outVal, int32_t *out, aclFloat16 *src, void *stream) {
        LaunchTROWARGMAX<int32_t, aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_max_uint32_float_16x16_32x32_64x64)
{
    run_vec_op<uint32_t, float, 16, 16, false, 32, 32, 64, 64>(
        [](float *outVal, uint32_t *out, float *src, void *stream) {
            LaunchTROWARGMAX<uint32_t, float, 16, 16, 32, 32, 64, 64>(out, src, stream);
        });
}

TEST_F(TARGREDUCEOPTest, case_row_min_uint32_float_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, float, 64, 64>([](float *outVal, uint32_t *out, float *src, void *stream) {
        LaunchTROWARGMIN<uint32_t, float, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_min_int32_half_16x256_16x256_16x256)
{
    run_vec_op<int32_t, aclFloat16, 16, 256>([](aclFloat16 *outVal, int32_t *out, aclFloat16 *src, void *stream) {
        LaunchTROWARGMIN<int32_t, aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_min_uint32_float_16x16_32x32_64x64)
{
    run_vec_op<uint32_t, float, 16, 16, false, 32, 32, 64, 64>(
        [](float *outVal, uint32_t *out, float *src, void *stream) {
            LaunchTROWARGMIN<uint32_t, float, 16, 16, 32, 32, 64, 64>(out, src, stream);
        });
}

TEST_F(TARGREDUCEOPTest, case_col_max_uint32_float_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, float, 64, 64>([](float *outVal, uint32_t *out, float *src, void *stream) {
        LaunchTCOLARGMAX<uint32_t, float, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_max_int32_half_16x256_16x256_16x256)
{
    run_vec_op<int32_t, aclFloat16, 16, 256>([](aclFloat16 *outVal, int32_t *out, aclFloat16 *src, void *stream) {
        LaunchTCOLARGMAX<int32_t, aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_max_uint32_float_16x16_32x32_64x64)
{
    run_vec_op<uint32_t, float, 16, 16, false, 32, 32, 64, 64>(
        [](float *outVal, uint32_t *out, float *src, void *stream) {
            LaunchTCOLARGMAX<uint32_t, float, 16, 16, 32, 32, 64, 64>(out, src, stream);
        });
}

TEST_F(TARGREDUCEOPTest, case_col_min_uint32_float_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, float, 64, 64>([](float *outVal, uint32_t *out, float *src, void *stream) {
        LaunchTCOLARGMIN<uint32_t, float, 64, 64>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_min_int32_half_16x256_16x256_16x256)
{
    run_vec_op<int32_t, aclFloat16, 16, 256>([](aclFloat16 *outVal, int32_t *out, aclFloat16 *src, void *stream) {
        LaunchTCOLARGMIN<int32_t, aclFloat16, 16, 256>(out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_col_min_uint32_float_16x16_32x32_64x64)
{
    run_vec_op<uint32_t, float, 16, 16, false, 32, 32, 64, 64>(
        [](float *outVal, uint32_t *out, float *src, void *stream) {
            LaunchTCOLARGMIN<uint32_t, float, 16, 16, 32, 32, 64, 64>(out, src, stream);
        });
}

TEST_F(TARGREDUCEOPTest, case_row_val_max_uint32_float_64x64_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, float, 64, 64, true>([](float *outVal, uint32_t *out, float *src, void *stream) {
        LaunchTROWARGMAX<uint32_t, float, 64, 64>(outVal, out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_val_max_int32_half_16x256_16x256_16x256_16x256)
{
    run_vec_op<int32_t, aclFloat16, 16, 256, true>([](aclFloat16 *outVal, int32_t *out, aclFloat16 *src, void *stream) {
        LaunchTROWARGMAX<int32_t, aclFloat16, 16, 256>(outVal, out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_val_max_uint32_float_16x16_32x32_64x64_32x32)
{
    run_vec_op<uint32_t, float, 16, 16, true, 32, 32, 64, 64, 32, 32>(
        [](float *outVal, uint32_t *out, float *src, void *stream) {
            LaunchTROWARGMAX<uint32_t, float, 16, 16, 32, 32, 64, 64, 32, 32>(outVal, out, src, stream);
        });
}

TEST_F(TARGREDUCEOPTest, case_row_val_min_uint32_float_64x64_64x64_64x64_64x64)
{
    run_vec_op<uint32_t, float, 64, 64, true>([](float *outVal, uint32_t *out, float *src, void *stream) {
        LaunchTROWARGMIN<uint32_t, float, 64, 64>(outVal, out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_val_min_int32_half_16x256_16x256_16x256_16x256)
{
    run_vec_op<int32_t, aclFloat16, 16, 256, true>([](aclFloat16 *outVal, int32_t *out, aclFloat16 *src, void *stream) {
        LaunchTROWARGMIN<int32_t, aclFloat16, 16, 256>(outVal, out, src, stream);
    });
}

TEST_F(TARGREDUCEOPTest, case_row_val_min_uint32_float_16x16_32x32_64x64_32x32)
{
    run_vec_op<uint32_t, float, 16, 16, true, 32, 32, 64, 64, 32, 32>(
        [](float *outVal, uint32_t *out, float *src, void *stream) {
            LaunchTROWARGMIN<uint32_t, float, 16, 16, 32, 32, 64, 64, 32, 32>(outVal, out, src, stream);
        });
}
