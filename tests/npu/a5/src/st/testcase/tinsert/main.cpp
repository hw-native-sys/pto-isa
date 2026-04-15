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

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t testKey>
void launchTInsertNZ(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertND(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNDVec(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

template <int32_t testKey>
void launchTInsertNDVecScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

template <int32_t testKey>
void launchTInsertNDVecValidShape(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

template <int32_t testKey>
void launchTInsertNZUnaligned(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZTwoInsert(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);

template <int32_t testKey>
void launchTInsertNZOverwrite(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);

template <int32_t testKey>
void launchTInsertNZVecToVec(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZSplitCustom(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZTwoInput(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZDoubleInput(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZFp4Offset(uint64_t *out, uint64_t *src, void *stream);

class TInsertTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + std::string(testInfo->name());
}

template <int32_t testKey, typename AType, typename CType>
void testTInsertAcc2Mat(int32_t m, int32_t k, int32_t n)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(AType);
    size_t cFileSize = m * n * sizeof(CType);
    uint8_t *outHost, *src0Host, *src1Host, *outDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&outHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMalloc((void **)&outDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertAcc2Mat<testKey>(outDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, cFileSize, outDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", outHost, cFileSize);

    aclrtFree(outDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(outHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize / sizeof(CType));
    std::vector<CType> devFinal(cFileSize / sizeof(CType));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

TEST_F(TInsertTest, case_acc2mat_1)
{
    testTInsertAcc2Mat<1, uint16_t, float>(16, 16, 16);
}
TEST_F(TInsertTest, case_acc2mat_2)
{
    testTInsertAcc2Mat<2, uint16_t, float>(32, 32, 32);
}

using LaunchFn2 = void (*)(uint64_t *, uint64_t *, void *);

template <typename dType>
void testSingleSrc(size_t srcByteSize, size_t dstByteSize, LaunchFn2 launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;
    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstByteSize, devFinal.data(), dstByteSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

// NZ basic tests
TEST_F(TInsertTest, case_nz_1)
{
    testSingleSrc<float>(16 * 32 * 4, 16 * 32 * 4, launchTInsertNZ<1>);
}
TEST_F(TInsertTest, case_nz_2)
{
    testSingleSrc<float>(16 * 32 * 4, 16 * 32 * 4, launchTInsertNZ<2>);
}
TEST_F(TInsertTest, case_nz_3)
{
    testSingleSrc<float>(32 * 64 * 4, 32 * 64 * 4, launchTInsertNZ<3>);
}
TEST_F(TInsertTest, case_nz_4)
{
    testSingleSrc<int32_t>(32 * 32 * 4, 32 * 32 * 4, launchTInsertNZ<4>);
}
TEST_F(TInsertTest, case_nz_5)
{
    testSingleSrc<float>(32 * 32 * 4, 32 * 32 * 4, launchTInsertNZ<5>);
}
TEST_F(TInsertTest, case_nz_6)
{
    testSingleSrc<float>(32 * 32 * 4, 32 * 32 * 4, launchTInsertNZ<6>);
}
TEST_F(TInsertTest, case_nz_7)
{
    testSingleSrc<float>(64 * 64 * 4, 64 * 64 * 4, launchTInsertNZ<7>);
}

// ND tests (output.bin instead of output_z.bin)
template <int32_t testKey, typename dType>
void testTInsertND(int32_t rows, int32_t cols)
{
    size_t byteSize = rows * cols * sizeof(dType);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;
    aclrtMallocHost((void **)(&dstHost), byteSize);
    aclrtMallocHost((void **)(&srcHost), byteSize);
    aclrtMalloc((void **)&dstDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", byteSize, srcHost, byteSize);
    aclrtMemcpy(srcDevice, byteSize, srcHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertND<testKey>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, byteSize, dstDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, byteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(byteSize);
    std::vector<dType> devFinal(byteSize);
    ReadFile(GetGoldenDir() + "/golden_output.bin", byteSize, golden.data(), byteSize);
    ReadFile(GetGoldenDir() + "/output.bin", byteSize, devFinal.data(), byteSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

TEST_F(TInsertTest, case_nd_1)
{
    testTInsertND<1, int8_t>(64, 32);
}
TEST_F(TInsertTest, case_nd_2)
{
    testTInsertND<2, int8_t>(128, 64);
}

using NdVecLaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);

template <typename dType>
void testNDVec(size_t srcByteSize, size_t dstByteSize, NdVecLaunchFn launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *outHost, *srcHost, *dstInitHost;
    uint8_t *outDevice, *srcDevice, *dstInitDevice;

    aclrtMallocHost((void **)(&outHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMallocHost((void **)(&dstInitHost), dstByteSize);
    aclrtMalloc((void **)&outDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstInitDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src_input.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/dst_init.bin", dstByteSize, dstInitHost, dstByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstInitDevice, dstByteSize, dstInitHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(outDevice, srcDevice, dstInitDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, dstByteSize, outDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, dstByteSize);

    aclrtFree(outDevice);
    aclrtFree(srcDevice);
    aclrtFree(dstInitDevice);
    aclrtFreeHost(outHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(dstInitHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.0f));
}

TEST_F(TInsertTest, case_nd_vec_1)
{
    testNDVec<float>(8 * 8 * 4, 16 * 16 * 4, launchTInsertNDVec<1>);
}
TEST_F(TInsertTest, case_nd_vec_2)
{
    testNDVec<float>(8 * 8 * 4, 16 * 16 * 4, launchTInsertNDVec<2>);
}
TEST_F(TInsertTest, case_nd_vec_3)
{
    testNDVec<uint16_t>(16 * 16 * 2, 32 * 32 * 2, launchTInsertNDVec<3>);
}
TEST_F(TInsertTest, case_nd_vec_4)
{
    testNDVec<int8_t>(32 * 32 * 1, 64 * 64 * 1, launchTInsertNDVec<4>);
}
TEST_F(TInsertTest, case_nd_vec_5)
{
    testNDVec<uint16_t>(16 * 16 * 2, 32 * 48 * 2, launchTInsertNDVec<5>);
}
TEST_F(TInsertTest, case_nd_vec_6)
{
    testNDVec<float>(8 * 8 * 4, 16 * 24 * 4, launchTInsertNDVec<6>);
}
TEST_F(TInsertTest, case_nd_vec_7)
{
    testNDVec<float>(8 * 8 * 4, 16 * 24 * 4, launchTInsertNDVec<7>);
}
TEST_F(TInsertTest, case_nd_vec_8)
{
    testNDVec<uint16_t>(8 * 16 * 2, 16 * 48 * 2, launchTInsertNDVec<8>);
}
TEST_F(TInsertTest, case_nd_vec_9)
{
    testNDVec<int8_t>(32 * 32, 64 * 64, launchTInsertNDVec<9>);
}
TEST_F(TInsertTest, case_nd_vec_10)
{
    testNDVec<float>(1 * 8 * 4, 16 * 16 * 4, launchTInsertNDVecScalar<1>);
}
TEST_F(TInsertTest, case_nd_vec_11)
{
    testNDVec<uint16_t>(1 * 16 * 2, 32 * 32 * 2, launchTInsertNDVecScalar<2>);
}
TEST_F(TInsertTest, case_nd_vec_12)
{
    testNDVec<int8_t>(1 * 32, 64 * 64, launchTInsertNDVecScalar<3>);
}
TEST_F(TInsertTest, case_nd_vec_13)
{
    testNDVec<float>(4 * 8 * 4, 16 * 16 * 4, launchTInsertNDVecValidShape<1>);
}
TEST_F(TInsertTest, case_nd_vec_14)
{
    testNDVec<uint16_t>(8 * 16 * 2, 16 * 32 * 2, launchTInsertNDVecValidShape<2>);
}
TEST_F(TInsertTest, case_nd_vec_15)
{
    testNDVec<int8_t>(16 * 32, 32 * 64, launchTInsertNDVecValidShape<3>);
}
TEST_F(TInsertTest, case_nd_vec_16)
{
    testNDVec<float>(4 * 8 * 4, 16 * 16 * 4, launchTInsertNDVecValidShape<4>);
}
TEST_F(TInsertTest, case_nd_vec_17)
{
    testNDVec<uint16_t>(8 * 16 * 2, 16 * 32 * 2, launchTInsertNDVecValidShape<5>);
}
TEST_F(TInsertTest, case_nd_vec_18)
{
    testNDVec<int8_t>(16 * 32, 32 * 64, launchTInsertNDVecValidShape<6>);
}
TEST_F(TInsertTest, case_nd_vec_19)
{
    testNDVec<uint16_t>(4 * 128 * 2, 8 * 144 * 2, launchTInsertNDVec<10>);
}
TEST_F(TInsertTest, case_nd_vec_20)
{
    testNDVec<uint16_t>(4 * 144 * 2, 8 * 160 * 2, launchTInsertNDVec<11>);
}

// NZ unaligned tests
TEST_F(TInsertTest, case_nz_8)
{
    testSingleSrc<float>(15 * 32 * 4, 16 * 32 * 4, launchTInsertNZUnaligned<1>);
}
TEST_F(TInsertTest, case_nz_9)
{
    testSingleSrc<float>(10 * 32 * 4, 32 * 32 * 4, launchTInsertNZUnaligned<2>);
}
TEST_F(TInsertTest, case_nz_11)
{
    testSingleSrc<float>(10 * 32 * 4, 32 * 32 * 4, launchTInsertNZUnaligned<3>);
}

using LaunchFn3 = void (*)(uint64_t *, uint64_t *, uint64_t *, void *);

template <typename dType>
void testTwoSrc(size_t src1Bytes, size_t src2Bytes, size_t dstBytes, LaunchFn3 launch)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint64_t *dstHost, *src1Host, *src2Host, *dstDevice, *src1Device, *src2Device;
    aclrtMallocHost((void **)(&dstHost), dstBytes);
    aclrtMallocHost((void **)(&src1Host), src1Bytes);
    aclrtMallocHost((void **)(&src2Host), src2Bytes);
    aclrtMalloc((void **)&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1Bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, src2Bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src1_input.bin", src1Bytes, src1Host, src1Bytes);
    ReadFile(GetGoldenDir() + "/src2_input.bin", src2Bytes, src2Host, src2Bytes);
    aclrtMemcpy(src1Device, src1Bytes, src1Host, src1Bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, src2Bytes, src2Host, src2Bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstBytes);

    aclrtFree(dstDevice);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstBytes / sizeof(dType));
    std::vector<dType> devFinal(dstBytes / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstBytes, golden.data(), dstBytes);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstBytes, devFinal.data(), dstBytes);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.001f));
}

TEST_F(TInsertTest, case_nz_10)
{
    testTwoSrc<float>(15 * 32 * 4, 10 * 32 * 4, 32 * 32 * 4, launchTInsertNZTwoInsert<1>);
}
TEST_F(TInsertTest, case_nz_12)
{
    testTwoSrc<float>(32 * 32 * 4, 10 * 32 * 4, 32 * 32 * 4, launchTInsertNZOverwrite<1>);
}
TEST_F(TInsertTest, case_nz_13)
{
    testTwoSrc<float>(8 * 256 * 4, 8 * 256 * 4, 16 * 256 * 4, launchTInsertNZTwoInsert<2>);
}

// NZ large tile tests
TEST_F(TInsertTest, case_nz_14)
{
    testSingleSrc<float>(32 * 32 * 4, 32 * 32 * 4, launchTInsertNZ<8>);
}
TEST_F(TInsertTest, case_nz_15)
{
    testSingleSrc<float>(32 * 32 * 4, 32 * 32 * 4, launchTInsertNZ<9>);
}

// NZ Vec→Vec tests
TEST_F(TInsertTest, case_nz_vec_1)
{
    testSingleSrc<float>(16 * 32 * 4, 16 * 32 * 4, launchTInsertNZVecToVec<1>);
}
TEST_F(TInsertTest, case_nz_vec_2)
{
    testSingleSrc<float>(16 * 32 * 4, 16 * 32 * 4, launchTInsertNZVecToVec<2>);
}
TEST_F(TInsertTest, case_nz_vec_3)
{
    testSingleSrc<float>(16 * 32 * 4, 32 * 32 * 4, launchTInsertNZVecToVec<3>);
}
TEST_F(TInsertTest, case_nz_vec_4)
{
    testSingleSrc<uint16_t>(16 * 32 * 2, 16 * 32 * 2, launchTInsertNZVecToVec<4>);
}
TEST_F(TInsertTest, case_nz_vec_5)
{
    testSingleSrc<uint16_t>(16 * 32 * 2, 16 * 32 * 2, launchTInsertNZVecToVec<5>);
}
TEST_F(TInsertTest, case_nz_vec_6)
{
    testSingleSrc<uint8_t>(16 * 64, 16 * 64, launchTInsertNZVecToVec<6>);
}
TEST_F(TInsertTest, case_nz_vec_7)
{
    testSingleSrc<uint8_t>(16 * 64, 16 * 64, launchTInsertNZVecToVec<7>);
}

// NZ Split custom tests
TEST_F(TInsertTest, case_nz_split_1)
{
    testSingleSrc<float>(8 * 256 * 4, 16 * 256 * 4, launchTInsertNZSplitCustom<1>);
}
TEST_F(TInsertTest, case_nz_split_2)
{
    testSingleSrc<float>(8 * 256 * 4, 16 * 256 * 4, launchTInsertNZSplitCustom<2>);
}
TEST_F(TInsertTest, case_nz_split_3)
{
    testSingleSrc<float>(128 * 128 * 4, 128 * 128 * 4, launchTInsertNZSplitCustom<3>);
}
TEST_F(TInsertTest, case_nz_split_4)
{
    testSingleSrc<float>(128 * 128 * 4, 128 * 128 * 4, launchTInsertNZSplitCustom<4>);
}

// NZ hif8 tests
TEST_F(TInsertTest, case_nz_hif8_1)
{
    testSingleSrc<uint8_t>(16 * 64, 16 * 64, launchTInsertNZ<10>);
}
TEST_F(TInsertTest, case_nz_hif8_2)
{
    testSingleSrc<uint8_t>(16 * 64, 16 * 64, launchTInsertNZ<11>);
}
TEST_F(TInsertTest, case_nz_hif8_3)
{
    testSingleSrc<uint8_t>(16 * 128, 16 * 128, launchTInsertNZ<12>);
}

template <int32_t testKey, typename dType, int32_t ValidRows, int32_t DstRows, int32_t Cols>
void testTInsertNZTwoInput()
{
    constexpr int32_t nzRow = 16;
    constexpr int32_t AlignedRow = ((ValidRows + nzRow - 1) / nzRow) * nzRow;
    constexpr int32_t c0Size = 32 / static_cast<int32_t>(sizeof(dType));
    constexpr int32_t burstNum = Cols / c0Size;
    constexpr size_t nz1TotalBytes = static_cast<size_t>(burstNum) * (AlignedRow + 1) * c0Size * sizeof(dType);
    constexpr size_t zeroBytes = static_cast<size_t>(DstRows) * Cols * sizeof(dType);

    testSingleSrc<dType>(zeroBytes + nz1TotalBytes, zeroBytes, launchTInsertNZTwoInput<testKey>);
}

TEST_F(TInsertTest, case_nz_twoinput_fp16_1)
{
    testTInsertNZTwoInput<1, uint16_t, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_bf16_1)
{
    testTInsertNZTwoInput<2, uint16_t, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp32_1)
{
    testTInsertNZTwoInput<3, float, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_int8_1)
{
    testTInsertNZTwoInput<4, uint8_t, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp8e5_1)
{
    testTInsertNZTwoInput<5, uint8_t, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp8e4_1)
{
    testTInsertNZTwoInput<6, uint8_t, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_hif8_1)
{
    testTInsertNZTwoInput<7, uint8_t, 8, 16, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp16_2)
{
    testTInsertNZTwoInput<8, uint16_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_bf16_2)
{
    testTInsertNZTwoInput<9, uint16_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp32_2)
{
    testTInsertNZTwoInput<10, float, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_int8_2)
{
    testTInsertNZTwoInput<11, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp8e5_2)
{
    testTInsertNZTwoInput<12, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp8e4_2)
{
    testTInsertNZTwoInput<13, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_hif8_2)
{
    testTInsertNZTwoInput<14, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp4e2m1_1)
{
    testTInsertNZTwoInput<15, uint8_t, 8, 16, 64>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp4e1m2_1)
{
    testTInsertNZTwoInput<16, uint8_t, 8, 16, 64>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp4e2m1_2)
{
    testTInsertNZTwoInput<17, uint8_t, 129, 256, 128>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp4e1m2_2)
{
    testTInsertNZTwoInput<18, uint8_t, 129, 256, 128>();
}

template <int32_t testKey, typename dType, int32_t TileRows, int32_t DstRows, int32_t Cols>
void testTInsertNZDoubleInput()
{
    constexpr int32_t c0Size = 32 / static_cast<int32_t>(sizeof(dType));
    constexpr int32_t burstNum = Cols / c0Size;
    constexpr size_t nz1Bytes = static_cast<size_t>(burstNum) * TileRows * c0Size * sizeof(dType);
    constexpr size_t zeroBytes = static_cast<size_t>(DstRows) * Cols * sizeof(dType);
    testSingleSrc<dType>(zeroBytes + 2 * nz1Bytes, zeroBytes, launchTInsertNZDoubleInput<testKey>);
}

TEST_F(TInsertTest, case_nz_dblinput_fp4e2m1_1)
{
    testTInsertNZDoubleInput<1, uint8_t, 17, 16, 64>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp4e1m2_1)
{
    testTInsertNZDoubleInput<2, uint8_t, 17, 16, 64>();
}
TEST_F(TInsertTest, case_nz_dblinput_hif8_1)
{
    testTInsertNZDoubleInput<3, uint8_t, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp4e2m1_2)
{
    testTInsertNZDoubleInput<4, uint8_t, 129, 256, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp4e1m2_2)
{
    testTInsertNZDoubleInput<5, uint8_t, 129, 256, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_hif8_2)
{
    testTInsertNZDoubleInput<6, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp16_1)
{
    testTInsertNZDoubleInput<7, uint16_t, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_bf16_1)
{
    testTInsertNZDoubleInput<8, uint16_t, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp32_1)
{
    testTInsertNZDoubleInput<9, float, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_int8_1)
{
    testTInsertNZDoubleInput<10, uint8_t, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp8e5_1)
{
    testTInsertNZDoubleInput<11, uint8_t, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp8e4_1)
{
    testTInsertNZDoubleInput<12, uint8_t, 17, 16, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp16_2)
{
    testTInsertNZDoubleInput<13, uint16_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_dblinput_bf16_2)
{
    testTInsertNZDoubleInput<14, uint16_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp32_2)
{
    testTInsertNZDoubleInput<15, float, 129, 256, 128>();
}
TEST_F(TInsertTest, case_nz_dblinput_int8_2)
{
    testTInsertNZDoubleInput<16, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp8e5_2)
{
    testTInsertNZDoubleInput<17, uint8_t, 129, 256, 256>();
}
TEST_F(TInsertTest, case_nz_dblinput_fp8e4_2)
{
    testTInsertNZDoubleInput<18, uint8_t, 129, 256, 256>();
}

// FP4 split remainder tests (3 NZ c0 blocks with SPLIT2 → partBurstNum=1, lastBurstNum=2)
TEST_F(TInsertTest, case_nz_twoinput_fp4e2m1_3)
{
    testTInsertNZTwoInput<19, uint8_t, 8, 16, 96>();
}
TEST_F(TInsertTest, case_nz_twoinput_fp4e1m2_3)
{
    testTInsertNZTwoInput<20, uint8_t, 8, 16, 96>();
}

// FP4 non-zero indexCol tests
template <int32_t testKey, int32_t SrcRows, int32_t SrcByteCols, int32_t DstRows, int32_t DstByteCols>
void testTInsertNZFp4Offset()
{
    constexpr size_t srcNzBytes = static_cast<size_t>(SrcRows) * SrcByteCols;
    constexpr size_t dstBytes = static_cast<size_t>(DstRows) * DstByteCols;
    testSingleSrc<uint8_t>(dstBytes + srcNzBytes, dstBytes, launchTInsertNZFp4Offset<testKey>);
}

TEST_F(TInsertTest, case_nz_fp4_offset_e2m1_col)
{
    testTInsertNZFp4Offset<1, 16, 32, 16, 128>();
}
TEST_F(TInsertTest, case_nz_fp4_offset_e1m2_col)
{
    testTInsertNZFp4Offset<2, 16, 32, 16, 128>();
}
TEST_F(TInsertTest, case_nz_fp4_offset_e2m1_rowcol)
{
    testTInsertNZFp4Offset<3, 16, 32, 16, 128>();
}
TEST_F(TInsertTest, case_nz_fp4_offset_e1m2_rowcol)
{
    testTInsertNZFp4Offset<4, 16, 32, 16, 128>();
}
