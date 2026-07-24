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
#include <cstdint>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void launchTLOADMX(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TLOADMXTest : public testing::Test {
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
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void TLOADMXFUNC()
{
    size_t aFileSize = WN1 * WN2 * WN3 * WN4 * WN5 * sizeof(T);
    size_t bFileSize = N4 * N5 * sizeof(T);
    size_t cFileSize = BASEM * BASEK * sizeof(T);

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
    launchTLOADMX<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
        dstDevice, src0Device, src1Device, stream);

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

// format 0:AND2ZZ
TEST_F(TLOADMXTest, 1_1_1_16_4_uint8_AND2ZZ) { TLOADMXFUNC<uint8_t, 0, 1, 1, 1, 16, 4, 1, 1, 1, 16, 4, 16, 4>(); }
TEST_F(TLOADMXTest, 1_1_1_16_64_uint8_AND2ZZ) { TLOADMXFUNC<uint8_t, 0, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 32, 158>(); }
TEST_F(TLOADMXTest, 1_1_1_32_128_uint8_AND2ZZ)
{
    TLOADMXFUNC<uint8_t, 0, 1, 1, 1, 32, 128, 1, 1, 1, 160, 128, 64, 1008>();
}
TEST_F(TLOADMXTest, 1_1_1_128_128_uint8_AND2ZZ)
{
    TLOADMXFUNC<uint8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADMXTest, 1_1_1_64_128_uint8_AND2ZZ)
{
    TLOADMXFUNC<uint8_t, 0, 1, 1, 1, 31, 118, 1, 1, 1, 34, 126, 64, 128>();
}

// format 1:ADN2ZZ
TEST_F(TLOADMXTest, 1_1_1_16_4_uint8_ADN2ZZ) { TLOADMXFUNC<uint8_t, 1, 1, 1, 1, 1, 2, 1, 1, 1, 65534, 4, 16, 8>(); }
TEST_F(TLOADMXTest, 1_1_1_16_64_uint8_ADN2ZZ) { TLOADMXFUNC<uint8_t, 1, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 16, 64>(); }
TEST_F(TLOADMXTest, 1_1_1_32_128_uint8_ADN2ZZ)
{
    TLOADMXFUNC<uint8_t, 1, 1, 1, 1, 32, 128, 1, 1, 1, 32, 128, 32, 128>();
}
TEST_F(TLOADMXTest, 1_1_1_128_128_uint8_ADN2ZZ)
{
    TLOADMXFUNC<uint8_t, 1, 1, 1, 1, 27, 126, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADMXTest, 1_1_1_64_128_uint8_ADN2ZZ)
{
    TLOADMXFUNC<uint8_t, 1, 1, 1, 1, 31, 118, 1, 1, 1, 34, 126, 64, 128>();
}

// format 2:BND2NN
TEST_F(TLOADMXTest, 1_1_1_4_64_uint8_BND2NN) { TLOADMXFUNC<uint8_t, 2, 1, 1, 1, 4, 64, 1, 1, 1, 4, 64, 4, 64>(); }
TEST_F(TLOADMXTest, 1_1_1_16_64_uint8_BND2NN) { TLOADMXFUNC<uint8_t, 2, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 16, 64>(); }
TEST_F(TLOADMXTest, 1_1_1_32_128_uint8_BND2NN)
{
    TLOADMXFUNC<uint8_t, 2, 1, 1, 1, 32, 127, 1, 1, 1, 32, 128, 32, 256>();
}
TEST_F(TLOADMXTest, 1_1_1_128_128_uint8_BND2NN)
{
    TLOADMXFUNC<uint8_t, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADMXTest, 1_1_1_128_64_uint8_BND2NN)
{
    TLOADMXFUNC<uint8_t, 2, 1, 1, 1, 116, 34, 1, 1, 1, 130, 60, 128, 64>();
}

// format 3:BDN2NN
TEST_F(TLOADMXTest, 1_1_1_4_64_uint8_BDN2NN) { TLOADMXFUNC<uint8_t, 3, 1, 1, 1, 4, 64, 1, 1, 1, 4, 64, 4, 64>(); }
TEST_F(TLOADMXTest, 1_1_1_16_64_uint8_BDN2NN) { TLOADMXFUNC<uint8_t, 3, 1, 1, 1, 16, 64, 1, 1, 1, 16, 64, 16, 64>(); }
TEST_F(TLOADMXTest, 1_1_1_32_128_uint8_BDN2NN)
{
    TLOADMXFUNC<uint8_t, 3, 1, 1, 1, 2, 128, 1, 1, 1, 32, 128, 4, 1088>();
}
TEST_F(TLOADMXTest, 1_1_1_128_128_uint8_BDN2NN)
{
    TLOADMXFUNC<uint8_t, 3, 1, 1, 1, 30, 127, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADMXTest, 1_1_1_128_64_uint8_BDN2NN)
{
    TLOADMXFUNC<uint8_t, 3, 1, 1, 1, 116, 34, 1, 1, 1, 130, 60, 128, 64>();
}
