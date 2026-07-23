/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <vector>
#include <pto/common/type.hpp>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

// Forward declarations — match the launch wrappers in tquant_kernel.cpp
namespace TQuantHif4A6 {
template <int validRows, int validCols>
void LaunchTQuantHif4A6(
    uint16_t* src, uint16_t* max4, uint16_t* max8, uint8_t* ea, uint8_t* eb, uint8_t* ec, uint8_t* exp_dst,
    uint8_t* fp4, uint16_t* scale, void* stream);
} // namespace TQuantHif4A6

namespace TQuantVaddA6 {
template <int validRows, int validCols>
void LaunchVaddA6(uint16_t* dst, uint16_t* src0, uint16_t* src1, void* stream);
} // namespace TQuantVaddA6

class TQUANT_HIF4_A6_TEST : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

// ---- VADD stub: basic connectivity probe ----
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_128x128_vadd_nd)
{
    constexpr int validRows = 128;
    constexpr int validCols = 128;
    size_t srcFileSize = validRows * validCols * sizeof(uint16_t);
    size_t dstFileSize = srcFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t* src0Host = nullptr;
    uint16_t* src1Host = nullptr;
    uint16_t* dstHost = nullptr;
    uint16_t* src0Device = nullptr;
    uint16_t* src1Device = nullptr;
    uint16_t* dstDevice = nullptr;
    aclrtMallocHost((void**)&src0Host, srcFileSize);
    aclrtMallocHost((void**)&src1Host, srcFileSize);
    aclrtMallocHost((void**)&dstHost, dstFileSize);
    aclrtMalloc((void**)&src0Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", srcFileSize, src0Host, srcFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", srcFileSize, src1Host, srcFileSize);
    aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, srcFileSize, src1Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    TQuantVaddA6::LaunchVaddA6<validRows, validCols>(dstDevice, src0Device, src1Device, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src1Device);
    aclrtFree(src0Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint16_t> golden(dstFileSize / sizeof(uint16_t));
    std::vector<uint16_t> dev(dstFileSize / sizeof(uint16_t));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, dev.data(), dstFileSize);
    EXPECT_TRUE(ResultCmp<uint16_t>(golden, dev, 0.0f));
}

// ---- HiF4 quantization helper ----
template <int validRows, int validCols>
void RunHif4Case(const std::string& goldenDir)
{
    constexpr int totalElem = validRows * validCols;
    size_t srcFileSize = totalElem * sizeof(uint16_t);
    size_t max4Size = (totalElem / 4) * sizeof(uint16_t);
    size_t max8Size = (totalElem / 8) * sizeof(uint16_t);
    size_t eaSize = (totalElem / 64) * 2;
    size_t ebSize = ((totalElem / 8) / 8) * 2;
    size_t ecSize = (totalElem / 4) / 8;
    size_t expDstSize = (totalElem / 64) * 4;
    size_t fp4Size = totalElem / 2;
    size_t scaleSize = (totalElem / 2) * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // Host buffers
    uint16_t* srcHost = nullptr;
    aclrtMallocHost((void**)&srcHost, srcFileSize);
    ReadFile(goldenDir + "/input1.bin", srcFileSize, srcHost, srcFileSize);

    // Device buffers
    uint16_t* srcDevice = nullptr;
    uint16_t *max4Dev = nullptr, *max8Dev = nullptr;
    uint8_t *eaDev = nullptr, *ebDev = nullptr, *ecDev = nullptr, *expDstDev = nullptr, *fp4Dev = nullptr;
    uint16_t* scaleDev = nullptr;
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&max4Dev, max4Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&max8Dev, max8Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&eaDev, eaSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ebDev, ebSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&ecDev, ecSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&expDstDev, expDstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&fp4Dev, fp4Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scaleDev, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    TQuantHif4A6::LaunchTQuantHif4A6<validRows, validCols>(
        srcDevice, max4Dev, max8Dev, eaDev, ebDev, ecDev, expDstDev, fp4Dev, scaleDev, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();

    // Copy outputs back and verify
    auto verify = [&](const std::string& name, void* devPtr, size_t size) {
        std::vector<uint8_t> hostBuf(size);
        std::vector<uint8_t> goldenBuf(size);
        aclrtMemcpy(hostBuf.data(), size, devPtr, size, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(goldenDir + "/" + name + ".bin", hostBuf.data(), size);
        ReadFile(goldenDir + "/golden_" + name + ".bin", size, goldenBuf.data(), size);
        EXPECT_TRUE(ResultCmp<uint8_t>(goldenBuf, hostBuf, 0.0f)) << name << " mismatch";
    };

    verify("max4", max4Dev, max4Size);
    verify("max8", max8Dev, max8Size);
    verify("ea", eaDev, eaSize);
    verify("eb", ebDev, ebSize);
    verify("ec", ecDev, ecSize);
    verify("exp_dst", expDstDev, expDstSize);
    verify("fp4", fp4Dev, fp4Size);
    verify("scale", scaleDev, scaleSize);

    aclrtFree(scaleDev);
    aclrtFree(fp4Dev);
    aclrtFree(expDstDev);
    aclrtFree(ecDev);
    aclrtFree(ebDev);
    aclrtFree(eaDev);
    aclrtFree(max8Dev);
    aclrtFree(max4Dev);
    aclrtFree(srcDevice);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

// ---- HiF4 test cases ----
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_128x128_hif4_nd) { RunHif4Case<128, 128>(GetGoldenDir()); }
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_64x128_hif4_nd) { RunHif4Case<64, 128>(GetGoldenDir()); }
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_256x128_hif4_nd) { RunHif4Case<256, 128>(GetGoldenDir()); }
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_128x256_hif4_nd) { RunHif4Case<128, 256>(GetGoldenDir()); }
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_256x256_hif4_nd) { RunHif4Case<256, 256>(GetGoldenDir()); }
TEST_F(TQUANT_HIF4_A6_TEST, case_bf16_128x512_hif4_nd) { RunHif4Case<128, 512>(GetGoldenDir()); }
