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

template <int32_t testKey>
void launchTInsertAcc2Mat(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t testKey>
void launchTInsertNZ(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertND(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZUnaligned(uint64_t *out, uint64_t *src, void *stream);

template <int32_t testKey>
void launchTInsertNZTwoInsert(uint64_t *out, uint64_t *src1, uint64_t *src2, void *stream);

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
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <int32_t testKey, typename AType, typename CType>
void testTInsertAcc2Mat(int32_t M, int32_t K, int32_t N)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(AType);
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

    launchTInsertAcc2Mat<testKey>(dstDevice, src0Device, src1Device, stream);

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

    std::vector<CType> golden(cFileSize / sizeof(CType));
    std::vector<CType> devFinal(cFileSize / sizeof(CType));
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TInsertTest, case_acc2mat_1)
{
    testTInsertAcc2Mat<1, uint16_t, float>(16, 16, 16);
}

TEST_F(TInsertTest, case_acc2mat_2)
{
    testTInsertAcc2Mat<2, uint16_t, float>(32, 32, 32);
}

template <int32_t testKey, typename dType>
void testTInsertNZ(int32_t rows, int32_t cols)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = rows * cols * sizeof(dType);
    size_t dstByteSize = rows * cols * sizeof(dType);
    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertNZ<testKey>(dstDevice, srcDevice, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TInsertTest, case_nz_1)
{
    testTInsertNZ<1, float>(16, 32);
}

TEST_F(TInsertTest, case_nz_2)
{
    testTInsertNZ<2, float>(16, 32);
}

TEST_F(TInsertTest, case_nz_3)
{
    testTInsertNZ<3, float>(32, 64);
}

TEST_F(TInsertTest, case_nz_4)
{
    testTInsertNZ<4, int32_t>(32, 32);
}

TEST_F(TInsertTest, case_nz_5)
{
    testTInsertNZ<5, float>(32, 32);
}

TEST_F(TInsertTest, case_nz_6)
{
    testTInsertNZ<6, float>(32, 32);
}

TEST_F(TInsertTest, case_nz_7)
{
    testTInsertNZ<7, float>(64, 64);
}

template <int32_t testKey, typename dType>
void testTInsertND(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = srcRows * srcCols * sizeof(dType);
    size_t dstByteSize = dstRows * dstCols * sizeof(dType);
    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertND<testKey>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize);
    std::vector<dType> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TInsertTest, case_nd_1)
{
    testTInsertND<1, int8_t>(64, 32, 64, 32);
}

TEST_F(TInsertTest, case_nd_2)
{
    testTInsertND<2, int8_t>(128, 64, 128, 64);
}

template <int32_t testKey>
void launchTInsertNDVec(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

template <int32_t testKey>
void launchTInsertNDVecScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

template <int32_t testKey>
void launchTInsertNDVecValidShape(uint8_t *out, uint8_t *srcIn, uint8_t *dstIn, void *stream);

using NdVecLaunchFn = void (*)(uint8_t *, uint8_t *, uint8_t *, void *);

template <typename dType>
void runTInsertNDVecTest(size_t srcByteSize, size_t dstByteSize, NdVecLaunchFn launch)
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
    bool ret = ResultCmp(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

template <int32_t testKey, typename dType>
void testTInsertNDVec(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runTInsertNDVecTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                               launchTInsertNDVec<testKey>);
}

TEST_F(TInsertTest, case_nd_vec_1)
{
    testTInsertNDVec<1, float>(8, 8, 16, 16);
}

TEST_F(TInsertTest, case_nd_vec_2)
{
    testTInsertNDVec<2, float>(8, 8, 16, 16);
}

TEST_F(TInsertTest, case_nd_vec_3)
{
    testTInsertNDVec<3, uint16_t>(16, 16, 32, 32);
}

TEST_F(TInsertTest, case_nd_vec_4)
{
    testTInsertNDVec<4, int8_t>(32, 32, 64, 64);
}

TEST_F(TInsertTest, case_nd_vec_5)
{
    testTInsertNDVec<5, uint16_t>(16, 16, 32, 48);
}

TEST_F(TInsertTest, case_nd_vec_6)
{
    testTInsertNDVec<6, float>(8, 8, 16, 24);
}

TEST_F(TInsertTest, case_nd_vec_7)
{
    testTInsertNDVec<7, float>(8, 8, 16, 24);
}

TEST_F(TInsertTest, case_nd_vec_8)
{
    testTInsertNDVec<8, uint16_t>(8, 16, 16, 48);
}

TEST_F(TInsertTest, case_nd_vec_9)
{
    testTInsertNDVec<9, int8_t>(32, 32, 64, 64);
}

template <int32_t testKey, typename dType>
void testTInsertNDVecScalar(int32_t dstRows, int32_t dstCols)
{
    constexpr size_t minAlignedCols = 32 / sizeof(dType);
    runTInsertNDVecTest<dType>(1 * minAlignedCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                               launchTInsertNDVecScalar<testKey>);
}

template <int32_t testKey, typename dType>
void testTInsertNDVecValidShape(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    runTInsertNDVecTest<dType>(srcRows * srcCols * sizeof(dType), dstRows * dstCols * sizeof(dType),
                               launchTInsertNDVecValidShape<testKey>);
}

TEST_F(TInsertTest, case_nd_vec_10)
{
    testTInsertNDVecScalar<1, float>(16, 16);
}

TEST_F(TInsertTest, case_nd_vec_11)
{
    testTInsertNDVecScalar<2, uint16_t>(32, 32);
}

TEST_F(TInsertTest, case_nd_vec_12)
{
    testTInsertNDVecScalar<3, int8_t>(64, 64);
}

TEST_F(TInsertTest, case_nd_vec_13)
{
    testTInsertNDVecValidShape<1, float>(4, 8, 16, 16);
}

TEST_F(TInsertTest, case_nd_vec_14)
{
    testTInsertNDVecValidShape<2, uint16_t>(8, 16, 16, 32);
}

TEST_F(TInsertTest, case_nd_vec_15)
{
    testTInsertNDVecValidShape<3, int8_t>(16, 32, 32, 64);
}

TEST_F(TInsertTest, case_nd_vec_16)
{
    testTInsertNDVecValidShape<4, float>(4, 8, 16, 16);
}

TEST_F(TInsertTest, case_nd_vec_17)
{
    testTInsertNDVecValidShape<5, uint16_t>(8, 16, 16, 32);
}

TEST_F(TInsertTest, case_nd_vec_18)
{
    testTInsertNDVecValidShape<6, int8_t>(16, 32, 32, 64);
}

TEST_F(TInsertTest, case_nd_vec_19)
{
    testTInsertNDVec<10, uint16_t>(4, 128, 8, 144);
}

TEST_F(TInsertTest, case_nd_vec_20)
{
    testTInsertNDVec<11, uint16_t>(4, 144, 8, 160);
}

template <int32_t testKey, typename dType>
void testTInsertNZUnaligned(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = srcRows * srcCols * sizeof(dType);
    size_t dstByteSize = dstRows * dstCols * sizeof(dType);
    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertNZUnaligned<testKey>(dstDevice, srcDevice, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TInsertTest, case_nz_8)
{
    testTInsertNZUnaligned<1, float>(15, 32, 16, 32);
}

TEST_F(TInsertTest, case_nz_9)
{
    testTInsertNZUnaligned<2, float>(10, 32, 32, 32);
}

template <int32_t testKey, typename dType>
void testTInsertNZTwoInsert(int32_t srcRows1, int32_t srcRows2, int32_t cols, int32_t dstRows)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t src1ByteSize = srcRows1 * cols * sizeof(dType);
    size_t src2ByteSize = srcRows2 * cols * sizeof(dType);
    size_t dstByteSize = dstRows * cols * sizeof(dType);
    uint64_t *dstHost, *src1Host, *src2Host, *dstDevice, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&src1Host), src1ByteSize);
    aclrtMallocHost((void **)(&src2Host), src2ByteSize);
    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1ByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, src2ByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src1_input.bin", src1ByteSize, src1Host, src1ByteSize);
    ReadFile(GetGoldenDir() + "/src2_input.bin", src2ByteSize, src2Host, src2ByteSize);
    aclrtMemcpy(src1Device, src1ByteSize, src1Host, src1ByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, src2ByteSize, src2Host, src2ByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTInsertNZTwoInsert<testKey>(dstDevice, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TInsertTest, case_nz_10)
{
    testTInsertNZTwoInsert<1, float>(15, 10, 32, 32);
}
