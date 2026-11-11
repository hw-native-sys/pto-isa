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
#include <cstring>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

namespace TRowExpandMulTest {
template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandMul(T* out, T* src0, T* src1, void* stream);

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandMul2(T* out, T* src0, T* src1, void* stream);

class TRowExpandMulTest : public testing::Test {
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
    typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst, bool isRowMajor>
void test_trowexpandmul()
{
    size_t inputFileSize = src1Row * src1Col * sizeof(T);
    size_t outputFileSize = dstRow * dstCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), outputFileSize);
    aclrtMallocHost((void**)(&src1Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputFileSize, src0Host, outputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, outputFileSize, src0Host, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (isRowMajor) {
        launchTRowExpandMul2<T, dstRow, dstCol, src1Row, src1Col, src0eqdst>(dstDevice, src0Device, src1Device, stream);
    } else {
        launchTRowExpandMul<T, dstRow, dstCol, src1Row, src1Col, src0eqdst>(dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        std::vector<T> golden64(outputFileSize / sizeof(T));
        std::vector<T> devFinal64(outputFileSize / sizeof(T));
        ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden64.data(), outputFileSize);
        ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal64.data(), outputFileSize);
        EXPECT_TRUE(ResultCmpExact(golden64, devFinal64.data()));
        return;
    }

    std::vector<float> golden(outputFileSize / sizeof(float));
    std::vector<float> devFinal(outputFileSize / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TRowExpandMulTest, case_fp32_16_32) { test_trowexpandmul<float, 16, 32, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_56_128) { test_trowexpandmul<float, 56, 128, 56, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp16_48_64) { test_trowexpandmul<aclFloat16, 48, 64, 48, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp16_16_128) { test_trowexpandmul<aclFloat16, 16, 128, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_24_64) { test_trowexpandmul<float, 24, 64, 24, 8, true, true>(); }
TEST_F(TRowExpandMulTest, case_fp16_32_64) { test_trowexpandmul<aclFloat16, 32, 64, 32, 1, false, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_20_64) { test_trowexpandmul<float, 20, 64, 20, 8, false, true>(); }
TEST_F(TRowExpandMulTest, case_int32_16_32) { test_trowexpandmul<int32_t, 16, 32, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_int16_16_64) { test_trowexpandmul<int16_t, 16, 64, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_uint32_16_32) { test_trowexpandmul<uint32_t, 16, 32, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_uint16_16_64) { test_trowexpandmul<uint16_t, 16, 64, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_int64_16_32) { test_trowexpandmul<int64_t, 16, 32, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_int64_24_64) { test_trowexpandmul<int64_t, 24, 64, 24, 4, true, true>(); }
TEST_F(TRowExpandMulTest, case_uint64_16_32) { test_trowexpandmul<uint64_t, 16, 32, 16, 1, true, false>(); }

// issue #828: bf16 was never exercised on A5 (vmul bf16 is native; regression coverage)
template <uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandMulBf16(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);

template <uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
void test_trowexpandmul_bf16()
{
    size_t inputFileSize = src1Row * src1Col * sizeof(uint16_t);
    size_t outputFileSize = dstRow * dstCol * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *dstHost, *src0Host, *src1Host;
    uint16_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), outputFileSize);
    aclrtMallocHost((void**)(&src1Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputFileSize, src0Host, outputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, outputFileSize, src0Host, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTRowExpandMulBf16<dstRow, dstCol, src1Row, src1Col, true>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // widen bf16 bit patterns to f32 (exact) and compare with the usual tolerance
    std::vector<uint16_t> golden16(outputFileSize / sizeof(uint16_t));
    std::vector<uint16_t> devFinal16(outputFileSize / sizeof(uint16_t));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden16.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal16.data(), outputFileSize);
    std::vector<float> golden(golden16.size());
    std::vector<float> devFinal(devFinal16.size());
    for (size_t i = 0; i < golden16.size(); i++) {
        uint32_t bits = static_cast<uint32_t>(golden16[i]) << 16;
        std::memcpy(&golden[i], &bits, sizeof(float));
        uint32_t devBits = static_cast<uint32_t>(devFinal16[i]) << 16;
        std::memcpy(&devFinal[i], &devBits, sizeof(float));
    }
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TRowExpandMulTest, case_bf16_16_32) { test_trowexpandmul_bf16<16, 32, 16, 1>(); }
// Short-row cases (issue #406) with dynamic valid extents: these tiles use DYNAMIC (-1) valid
// dimensions, so isContiguous is false and the dispatch keeps the pre-existing per-row 2D path;
// they act as regression coverage for it. dst aliases src0 in all cases (in-place), scales and
// values are nonuniform; bigscale uses f32 scales beyond the fp16 range.
TEST_F(TRowExpandMulTest, case_fp32_8_8) { test_trowexpandmul<float, 8, 8, 8, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_16_16) { test_trowexpandmul<float, 16, 16, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_16_16_bigscale) { test_trowexpandmul<float, 16, 16, 16, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp16_32_16) { test_trowexpandmul<aclFloat16, 32, 16, 32, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_1024_16) { test_trowexpandmul<float, 1024, 16, 1024, 1, true, false>(); }
TEST_F(TRowExpandMulTest, case_fp32_512_32) { test_trowexpandmul<float, 512, 32, 512, 1, true, false>(); }

template <typename T, uint32_t physRow, uint32_t physCol, uint32_t validRow>
void launchTRowExpandMulValid(T* out, T* src0, T* src1, void* stream);

template <typename T, uint32_t physRow, uint32_t physCol, uint32_t validRow>
void test_trowexpandmul_tail()
{
    size_t inputFileSize = validRow * sizeof(T);
    size_t outputFileSize = validRow * physCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), outputFileSize);
    aclrtMallocHost((void**)(&src1Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputFileSize, src0Host, outputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, outputFileSize, src0Host, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTRowExpandMulValid<T, physRow, physCol, validRow>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize / sizeof(float));
    std::vector<float> devFinal(outputFileSize / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TRowExpandMulTest, case_fp32_21_16_tail) { test_trowexpandmul_tail<float, 24, 16, 21>(); }

template <typename T, uint32_t dstRow, uint32_t dstCol>
void launchTRowExpandMulStatic(T* out, T* src0, T* src1, void* stream);

template <typename T, uint32_t dstRow, uint32_t dstCol>
void test_trowexpandmul_static()
{
    size_t inputFileSize = dstRow * sizeof(T);
    size_t outputFileSize = dstRow * dstCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), outputFileSize);
    aclrtMallocHost((void**)(&src1Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputFileSize, src0Host, outputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, outputFileSize, src0Host, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTRowExpandMulStatic<T, dstRow, dstCol>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize / sizeof(float));
    std::vector<float> devFinal(outputFileSize / sizeof(float));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// Static-valid-extent cases (issue #406): these tiles omit the DYNAMIC valid dimensions, so the
// dispatch enters the packed short-row fast path; they cover groupRows = 2/4/8 with per-row
// nonuniform scales (16x32 -> 2, 16x16 -> 4, 8x8 -> 8, half 16x16 -> 8 at elementsPerRepeat 128)
// and f32 scales beyond the fp16 range (bigscale).
TEST_F(TRowExpandMulTest, case_fp32_16_32_static) { test_trowexpandmul_static<float, 16, 32>(); }
TEST_F(TRowExpandMulTest, case_fp32_16_16_static) { test_trowexpandmul_static<float, 16, 16>(); }
TEST_F(TRowExpandMulTest, case_fp32_8_8_static) { test_trowexpandmul_static<float, 8, 8>(); }
TEST_F(TRowExpandMulTest, case_fp32_16_16_bigscale_static) { test_trowexpandmul_static<float, 16, 16>(); }
TEST_F(TRowExpandMulTest, case_fp16_16_16_static) { test_trowexpandmul_static<aclFloat16, 16, 16>(); }
} // namespace TRowExpandMulTest
