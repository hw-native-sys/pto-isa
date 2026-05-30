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
void launchTLOAD(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);

class TLOADTest : public testing::Test {
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

#define LOGSIZE 128
#define PRINTLOG 4
#define DEBUGLOG
#define MAXBLOCK 64

template <int32_t testKey, typename T, int32_t kBlock>
void tload_test()
{
    uint32_t M = 1024;
    uint32_t N = 1024;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int in_byteSize = M * N * sizeof(float);
    int out_byteSize = M * N * sizeof(float);

    void *dstHost, *srcHost, *goldHost;
    void *dstDevice, *srcDevice;
    void *logDevice;

    aclrtMallocHost((void **)(&srcHost), in_byteSize);
    aclrtMallocHost((void **)(&dstHost), out_byteSize);
    aclrtMallocHost((void **)(&goldHost), out_byteSize);

    aclrtMalloc((void **)&dstDevice, in_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, out_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Read pre-generated test data from files
    std::string goldenDir = GetGoldenDir();
    std::ifstream inFile(goldenDir + "/input.bin", std::ios::binary | std::ios::in);
    std::ifstream goldFile(goldenDir + "/golden.bin", std::ios::binary | std::ios::in);

    // Check if files exist and provide helpful error message
    if (!inFile.is_open()) {
        std::cerr << "Error: Cannot open input file: " << goldenDir << "/input.bin" << std::endl;
        std::cerr << "Please run 'python3 gen_data.py' in the test directory to generate test data first." << std::endl;
        FAIL() << "Input file not found. Run gen_data.py to generate test data.";
    }
    if (!goldFile.is_open()) {
        std::cerr << "Error: Cannot open golden file: " << goldenDir << "/golden.bin" << std::endl;
        std::cerr << "Please run 'python3 gen_data.py' in the test directory to generate test data first." << std::endl;
        FAIL() << "Golden file not found. Run gen_data.py to generate test data.";
    }

    // Get file sizes
    inFile.seekg(0, std::ios::end);
    size_t in_file_size = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    goldFile.seekg(0, std::ios::end);
    int actual_out_byteSize = goldFile.tellg();
    goldFile.seekg(0, std::ios::beg);

    // Read data from files
    inFile.read((char *)srcHost, in_file_size);
    goldFile.read((char *)goldHost, actual_out_byteSize);

    inFile.close();
    goldFile.close();

    std::fill((uint8_t *)dstHost, ((uint8_t *)(dstHost)) + out_byteSize, 0);

    aclrtMemcpy(srcDevice, in_byteSize, srcHost, in_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, out_byteSize, dstHost, out_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef DEBUGLOG
    uint64_t logHost[MAXBLOCK][LOGSIZE];
    std::fill((uint8_t *)logHost, ((uint8_t *)(logHost)) + sizeof(logHost), 0);
    aclrtMalloc((void **)&logDevice, sizeof(logHost), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(logDevice, sizeof(logHost), logHost, sizeof(logHost), ACL_MEMCPY_HOST_TO_DEVICE);
#endif

    launchTLOAD<testKey>((uint8_t *)dstDevice, (uint8_t *)srcDevice, (uint64_t *)logDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, out_byteSize, dstDevice, out_byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
#ifdef DEBUGLOG
    aclrtMemcpy(logHost, sizeof(logHost), logDevice, sizeof(logHost), ACL_MEMCPY_DEVICE_TO_HOST);
#endif

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, actual_out_byteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
#ifdef DEBUGLOG
    aclrtFree(logDevice);
#endif

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    int elements = actual_out_byteSize / sizeof(T);
    std::vector<T> golden(elements);
    std::vector<T> devFinal(elements);
    size_t oFileSize = actual_out_byteSize;
    ReadFile(GetGoldenDir() + "/golden.bin", oFileSize, golden.data(), oFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", oFileSize, devFinal.data(), oFileSize);

    bool ret = ResultCmp(golden, devFinal, 0);

#ifdef DEBUGLOG
    for (int b = 0; b < kBlock; b++) {
        cout << "Block: " << setw(2) << b << " ";
        for (int l = 0; l < sizeof(logHost[0]) / sizeof(logHost[0][0]) && l < PRINTLOG; l++) {
            cout << hex << setfill('0') << setw(16) << logHost[b][l] << " ";
        }
        cout << dec << endl;
    }
#endif

    EXPECT_TRUE(ret);
}

TEST_F(TLOADTest, case_float_GT_128_128_VT_128_128_BLK1)
{
    tload_test<1, float, 1>();
}

TEST_F(TLOADTest, case_float_GT_2_2_2_256_64_VT_256_64_BLK8)
{
    tload_test<2, float, 8>();
}

TEST_F(TLOADTest, case_float_GT_128_127_VT_128_128_BLK1_PADMAX)
{
    tload_test<3, float, 1>();
}

TEST_F(TLOADTest, case_s16_GT_128_127_VT_128_128_BLK1_PADMAX)
{
    tload_test<4, int16_t, 1>();
}

TEST_F(TLOADTest, case_u8_GT_128_127_VT_128_128_BLK1_PADMIN)
{
    tload_test<5, uint8_t, 1>();
}

TEST_F(TLOADTest, case_float_GT_8_64_128_VT_64_128_BLK8_DYN)
{
    tload_test<6, int16_t, 8>();
}

TEST_F(TLOADTest, case_float_GT_8_64_128_VT_64_128_BLK8_STC)
{
    tload_test<7, int16_t, 8>();
}

TEST_F(TLOADTest, case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX)
{
    tload_test<8, float, 8>();
}

TEST_F(TLOADTest, case_int64_GT_128_128_VT_128_128_BLK1)
{
    tload_test<9, int64_t, 1>();
}

TEST_F(TLOADTest, case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO)
{
    tload_test<10, uint64_t, 1>();
}

TEST_F(TLOADTest, case_int64_GT_2_2_2_256_62_VT_256_64_BLK8_PADZERO)
{
    tload_test<11, int64_t, 8>();
}

TEST_F(TLOADTest, case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8)
{
    tload_test<12, uint64_t, 8>();
}