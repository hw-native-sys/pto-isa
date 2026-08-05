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
#include "tgather_common.h"

using namespace std;
using namespace PtoTestCommon;

template <
    typename src0T, typename src1T, typename dstT, uint32_t SRCROW, uint32_t SRCCOL, uint32_t DSTROW, uint32_t DSTCOL,
    bool isF8E4M3 = false, bool isF8E5M2 = false>
void launchTGATHER_demo(src0T* src0, src1T* src1, dstT* out, void* stream);

constexpr int HALF_SIZE = 2;
constexpr int QUARTER_SIZE = 4;

template <
    typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, pto::MaskPattern maskPattern>
void LaunchTGATHER(dstT* out, srcT* src, void* stream);

template <typename T, int staticRows, int staticCols, int validRows, int validCols, pto::MaskPattern maskPattern>
void LaunchTGATHERDynamic(T* out, T* src, void* stream);

template <
    typename srcT, typename src1T, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int K,
    pto::CmpMode cmpMode>
void LaunchTGATHER_CMP(srcT* src, src1T* src1, dstT* out, uint32_t offset, void* stream);

class TGATHERTest : public testing::Test {
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
    typename src0T, typename src1T, typename dstT, uint32_t SRCROW, uint32_t SRCCOL, uint32_t DSTROW, uint32_t DSTCOL,
    bool isF8E4M3 = false, bool isF8E5M2 = false>
void test_gather_index()
{
    size_t src0FileSize = SRCROW * SRCCOL * sizeof(src0T);
    size_t src1FileSize = DSTROW * DSTCOL * sizeof(src1T);
    size_t dstFileSize = DSTROW * DSTCOL * sizeof(dstT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    src0T* src0Host;
    src1T* src1Host;
    dstT* dstHost;
    src0T* src0Device;
    src1T* src1Device;
    dstT* dstDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER_demo<src0T, src1T, dstT, SRCROW, SRCCOL, DSTROW, DSTCOL, isF8E4M3, isF8E5M2>(
        src0Device, src1Device, dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dstT> golden(DSTROW * DSTCOL);
    std::vector<dstT> devFinal(DSTROW * DSTCOL);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret;
    if constexpr (std::is_same_v<dstT, int64_t> || std::is_same_v<dstT, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp(golden, devFinal, 0.001f);
    }

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case1_float_32x1024_16x64) { test_gather_index<float, int32_t, float, 32, 1024, 16, 64>(); }

TEST_F(TGATHERTest, case2_int32_32x512_16x256) { test_gather_index<int32_t, int32_t, int32_t, 32, 512, 16, 256>(); }

TEST_F(TGATHERTest, case3_half_16x1024_16x128) { test_gather_index<int16_t, int16_t, int16_t, 16, 1024, 16, 128>(); }

TEST_F(TGATHERTest, case4_int16_32x256_32x64) { test_gather_index<int16_t, int16_t, int16_t, 32, 256, 32, 64>(); }

TEST_F(TGATHERTest, case5_f8e4m3_i16_16x128_16x64)
{
    test_gather_index<int8_t, int16_t, int8_t, 16, 128, 16, 64, true, false>();
}

TEST_F(TGATHERTest, case6_f8e5m2_i16_16x128_16x64)
{
    test_gather_index<int8_t, int16_t, int8_t, 16, 128, 16, 64, false, true>();
}

TEST_F(TGATHERTest, case7_i8_u16_16x128_16x64) { test_gather_index<int8_t, uint16_t, int8_t, 16, 128, 16, 64>(); }

TEST_F(TGATHERTest, case8_u8_u16_16x128_16x64) { test_gather_index<uint8_t, uint16_t, uint8_t, 16, 128, 16, 64>(); }

TEST_F(TGATHERTest, case9_int64_u32_4x16_4x16) { test_gather_index<int64_t, uint32_t, int64_t, 4, 16, 4, 16>(); }

TEST_F(TGATHERTest, case10_uint64_u32_4x16_4x16) { test_gather_index<uint64_t, uint32_t, uint64_t, 4, 16, 4, 16>(); }

template <typename T, pto::MaskPattern PATTERN, uint32_t ROW, uint32_t COL, typename dstT = T>
void test_gather()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t size = ROW * COL * sizeof(T);
    size_t dstsize = 0;
    if constexpr (PATTERN == pto::MaskPattern::P1111) {
        dstsize = size;
    } else if constexpr (PATTERN == pto::MaskPattern::P0101 || PATTERN == pto::MaskPattern::P1010) {
        dstsize = size / HALF_SIZE;
    } else {
        dstsize = size / QUARTER_SIZE;
    }
    T *src0Host, *src0Device;
    dstT *dstHost, *dstDevice;

    aclrtMallocHost((void**)(&dstHost), dstsize);
    aclrtMallocHost((void**)(&src0Host), size);
    aclrtMalloc((void**)&dstDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, size, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", size, src0Host, size);

    aclrtMemcpy(src0Device, size, src0Host, size, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTGATHER<T, dstT, ROW, COL, ROW, COL, PATTERN>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstsize, dstDevice, dstsize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstsize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstsize);
    std::vector<T> devFinal(dstsize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstsize, golden.data(), dstsize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstsize, devFinal.data(), dstsize);

    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp(golden, devFinal, 0.001f);
    }

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case1_float_P0101)
{
    test_gather<float, pto::MaskPattern::P0101, FLOAT_P0101_ROW, FLOAT_P0101_COL>();
}

TEST_F(TGATHERTest, case1_float_P1010)
{
    test_gather<float, pto::MaskPattern::P1010, FLOAT_P1010_ROW, FLOAT_P1010_COL>();
}

TEST_F(TGATHERTest, case1_float_P0001)
{
    test_gather<float, pto::MaskPattern::P0001, FLOAT_P0001_ROW, FLOAT_P0001_COL>();
}

TEST_F(TGATHERTest, case1_float_P0010)
{
    test_gather<float, pto::MaskPattern::P0010, FLOAT_P0010_ROW, FLOAT_P0010_COL>();
}

TEST_F(TGATHERTest, case1_float_P0100)
{
    test_gather<float, pto::MaskPattern::P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL>();
}

TEST_F(TGATHERTest, case1_float_P1000)
{
    test_gather<float, pto::MaskPattern::P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL>();
}

TEST_F(TGATHERTest, case1_float_P1111)
{
    test_gather<float, pto::MaskPattern::P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL>();
}

TEST_F(TGATHERTest, case1_float_int_P1010)
{
    test_gather<float, pto::MaskPattern::P1010, FLOAT_P1010_ROW, FLOAT_P1010_COL, int32_t>();
}

TEST_F(TGATHERTest, case1_half_P0101)
{
    test_gather<uint16_t, pto::MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL>();
}

TEST_F(TGATHERTest, case1_half_P1010)
{
    test_gather<uint16_t, pto::MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL>();
}

TEST_F(TGATHERTest, case1_half_P0001)
{
    test_gather<uint16_t, pto::MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL>();
}

TEST_F(TGATHERTest, case1_half_P0010)
{
    test_gather<uint16_t, pto::MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL>();
}

TEST_F(TGATHERTest, case1_half_P0100)
{
    test_gather<uint16_t, pto::MaskPattern::P0100, HALF_P0100_ROW, HALF_P0100_COL>();
}

TEST_F(TGATHERTest, case1_half_P1000)
{
    test_gather<uint16_t, pto::MaskPattern::P1000, HALF_P1000_ROW, HALF_P1000_COL>();
}

TEST_F(TGATHERTest, case1_half_P1111)
{
    test_gather<uint16_t, pto::MaskPattern::P1111, HALF_P1111_ROW, HALF_P1111_COL>();
}

TEST_F(TGATHERTest, case1_U16_P0101)
{
    test_gather<uint16_t, pto::MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL>();
}

TEST_F(TGATHERTest, case1_U16_P1010)
{
    test_gather<uint16_t, pto::MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL>();
}

TEST_F(TGATHERTest, case1_I16_P0001)
{
    test_gather<int16_t, pto::MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL>();
}

TEST_F(TGATHERTest, case1_I16_P0010)
{
    test_gather<int16_t, pto::MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL>();
}

TEST_F(TGATHERTest, case1_U32_P0100)
{
    test_gather<uint32_t, pto::MaskPattern::P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL>();
}

TEST_F(TGATHERTest, case1_I32_P1000)
{
    test_gather<int32_t, pto::MaskPattern::P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL>();
}

TEST_F(TGATHERTest, case1_I32_P1111)
{
    test_gather<int32_t, pto::MaskPattern::P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL>();
}

TEST_F(TGATHERTest, case_int64_4x16_P1010) { test_gather<int64_t, pto::MaskPattern::P1010, 4, 16>(); }

TEST_F(TGATHERTest, case_uint64_4x16_P0001) { test_gather<uint64_t, pto::MaskPattern::P0001, 4, 16>(); }

template <typename T>
void test_gather_dynamic_b64()
{
    constexpr size_t srcElements = 3 * 15;
    constexpr size_t dstElements = 3 * 7;
    constexpr size_t srcBytes = srcElements * sizeof(T);
    constexpr size_t dstBytes = dstElements * sizeof(T);
    size_t srcFileSize = srcBytes;
    size_t dstFileSize = dstBytes;
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    T *srcHost, *dstHost, *srcDevice, *dstDevice;
    aclrtMallocHost((void**)&srcHost, srcBytes);
    aclrtMallocHost((void**)&dstHost, dstBytes);
    aclrtMalloc((void**)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcBytes);
    aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTGATHERDynamic<T, 4, 24, 3, 15, pto::MaskPattern::P1010>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<T> golden(dstElements);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstBytes);
    EXPECT_TRUE(ResultCmpExact(golden, dstHost));
    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TGATHERTest, case_int64_dynamic_3x15_static_4x24_P1010) { test_gather_dynamic_b64<int64_t>(); }
TEST_F(TGATHERTest, case_uint64_dynamic_3x15_static_4x24_P1010) { test_gather_dynamic_b64<uint64_t>(); }

TEST_F(TGATHERTest, case1_b8_P0101) { test_gather<int8_t, pto::MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL>(); }

TEST_F(TGATHERTest, case1_b8_P1010) { test_gather<uint8_t, pto::MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL>(); }

TEST_F(TGATHERTest, case1_b8_P0001) { test_gather<int8_t, pto::MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL>(); }

TEST_F(TGATHERTest, case1_b8_P0010) { test_gather<uint8_t, pto::MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL>(); }

TEST_F(TGATHERTest, case1_b8_P0100) { test_gather<int8_t, pto::MaskPattern::P0100, HALF_P0100_ROW, HALF_P0100_COL>(); }

TEST_F(TGATHERTest, case1_b8_P1000) { test_gather<uint8_t, pto::MaskPattern::P1000, HALF_P1000_ROW, HALF_P1000_COL>(); }

TEST_F(TGATHERTest, case1_b8_P1111) { test_gather<int8_t, pto::MaskPattern::P1111, HALF_P1111_ROW, HALF_P1111_COL>(); }

template <typename srcT, typename src1T, typename dstT, uint32_t ROW, uint32_t COL, uint32_t K, pto::CmpMode cmpMode>
void test_gather_cmp()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t size = ROW * COL * sizeof(srcT);
    size_t dstsize = ROW * K * sizeof(dstT);
    size_t scalarSize = ROW * sizeof(src1T);

    uint32_t offset = 0;

    srcT *srcHost, *srcDevice;
    src1T *src1Host, *src1Device;
    dstT *dstHost, *dstDevice;

    aclrtMallocHost((void**)(&dstHost), dstsize);
    aclrtMallocHost((void**)(&srcHost), size);
    aclrtMallocHost((void**)(&src1Host), scalarSize);
    aclrtMalloc((void**)&dstDevice, dstsize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src.bin", size, srcHost, size);
    ReadFile(GetGoldenDir() + "/src1.bin", scalarSize, src1Host, scalarSize);

    aclrtMemcpy(srcDevice, size, srcHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, scalarSize, src1Host, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(dstDevice, dstsize, 0, dstsize);
    LaunchTGATHER_CMP<srcT, src1T, dstT, ROW, COL, ROW, COL, K, cmpMode>(
        srcDevice, src1Device, dstDevice, offset, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstsize, dstDevice, dstsize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstsize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(dstsize);
    std::vector<float> devFinal(dstsize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstsize, golden.data(), dstsize);
    ReadFile(GetGoldenDir() + "/output.bin", dstsize, devFinal.data(), dstsize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case1_float_topk) { test_gather_cmp<float, uint32_t, uint32_t, 16, 64, 32, pto::CmpMode::GT>(); }

TEST_F(TGATHERTest, case2_u32_topk) { test_gather_cmp<uint32_t, uint32_t, uint32_t, 8, 128, 64, pto::CmpMode::GT>(); }

TEST_F(TGATHERTest, case3_float_topk) { test_gather_cmp<float, uint32_t, uint32_t, 4, 256, 64, pto::CmpMode::EQ>(); }

TEST_F(TGATHERTest, case4_s16_topk) { test_gather_cmp<int16_t, uint16_t, uint32_t, 16, 128, 32, pto::CmpMode::GT>(); }

TEST_F(TGATHERTest, case5_s16_topk) { test_gather_cmp<int16_t, uint16_t, uint32_t, 4, 64, 32, pto::CmpMode::EQ>(); }

TEST_F(TGATHERTest, case6_half_topk)
{
    test_gather_cmp<aclFloat16, uint16_t, uint32_t, 2, 256, 32, pto::CmpMode::GT>();
}

TEST_F(TGATHERTest, case7_half_topk)
{
    test_gather_cmp<aclFloat16, uint16_t, uint32_t, 8, 128, 32, pto::CmpMode::EQ>();
}

TEST_F(TGATHERTest, case8_i8_topk) { test_gather_cmp<int8_t, uint16_t, uint32_t, 16, 128, 32, pto::CmpMode::GT>(); }

TEST_F(TGATHERTest, case9_i8_topk) { test_gather_cmp<int8_t, uint16_t, uint32_t, 16, 128, 32, pto::CmpMode::EQ>(); }

TEST_F(TGATHERTest, case10_u8_topk) { test_gather_cmp<uint8_t, uint16_t, uint32_t, 16, 128, 32, pto::CmpMode::GT>(); }
