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
#include "acl_env.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTInsertZN(uint64_t* out, uint64_t* src, void* stream);

template <int32_t testKey>
void launchTInsertZNOffset(uint64_t* out, uint64_t* src, void* stream);

template <int32_t testKey>
void launchTInsertZNFp4(uint64_t* out, uint64_t* src, void* stream);

class TInsertZNTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + std::string(testInfo->name());
}

using LaunchFn = void (*)(uint64_t*, uint64_t*, void*);

template <typename dType>
void testTInsertZN(size_t srcByteSize, size_t dstByteSize, LaunchFn launch)
{
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;
    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);
    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);
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

    std::vector<dType> golden(dstByteSize / sizeof(dType));
    std::vector<dType> devFinal(dstByteSize / sizeof(dType));
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstByteSize, devFinal.data(), dstByteSize);
    EXPECT_TRUE(ResultCmp(golden, devFinal, 0.0f));
}

TEST_F(TInsertZNTest, case_zn_1) { testTInsertZN<uint16_t>(16 * 16 * 2, 16 * 16 * 2, launchTInsertZN<1>); }
TEST_F(TInsertZNTest, case_zn_2) { testTInsertZN<uint16_t>(16 * 32 * 2, 16 * 32 * 2, launchTInsertZN<2>); }
TEST_F(TInsertZNTest, case_zn_3) { testTInsertZN<float>(8 * 16 * 4, 8 * 16 * 4, launchTInsertZN<3>); }
TEST_F(TInsertZNTest, case_zn_4) { testTInsertZN<float>(16 * 32 * 4, 16 * 32 * 4, launchTInsertZN<4>); }
TEST_F(TInsertZNTest, case_zn_5) { testTInsertZN<int32_t>(8 * 16 * 4, 8 * 16 * 4, launchTInsertZN<5>); }
TEST_F(TInsertZNTest, case_zn_6) { testTInsertZN<int8_t>(32 * 32, 32 * 32, launchTInsertZN<6>); }
TEST_F(TInsertZNTest, case_zn_7) { testTInsertZN<uint16_t>(32 * 64 * 2, 32 * 64 * 2, launchTInsertZN<7>); }
TEST_F(TInsertZNTest, case_zn_8) { testTInsertZN<uint16_t>(16 * 32 * 2, 16 * 32 * 2, launchTInsertZN<8>); }
TEST_F(TInsertZNTest, case_zn_9) { testTInsertZN<uint8_t>(32 * 32, 32 * 32, launchTInsertZN<9>); }
TEST_F(TInsertZNTest, case_zn_10) { testTInsertZN<uint8_t>(32 * 32, 32 * 32, launchTInsertZN<10>); }
TEST_F(TInsertZNTest, case_zn_11) { testTInsertZN<uint8_t>(32 * 64, 32 * 64, launchTInsertZN<11>); }
TEST_F(TInsertZNTest, case_zn_12) { testTInsertZN<uint8_t>(32 * 32, 32 * 32, launchTInsertZN<12>); }
TEST_F(TInsertZNTest, case_zn_fp4_1) { testTInsertZN<uint8_t>(32 * 32, 32 * 32, launchTInsertZNFp4<1>); }
TEST_F(TInsertZNTest, case_zn_fp4_2) { testTInsertZN<uint8_t>(32 * 32, 32 * 32, launchTInsertZNFp4<2>); }

TEST_F(TInsertZNTest, case_zn_offset_1)
{
    testTInsertZN<uint16_t>((32 * 32 + 16 * 16) * 2, 32 * 32 * 2, launchTInsertZNOffset<1>);
}
TEST_F(TInsertZNTest, case_zn_offset_2)
{
    testTInsertZN<uint16_t>((32 * 32 + 16 * 16) * 2, 32 * 32 * 2, launchTInsertZNOffset<2>);
}
TEST_F(TInsertZNTest, case_zn_offset_3)
{
    testTInsertZN<float>((16 * 32 + 8 * 16) * 4, 16 * 32 * 4, launchTInsertZNOffset<3>);
}
TEST_F(TInsertZNTest, case_zn_offset_4)
{
    testTInsertZN<uint16_t>((16 * 32 + 16 * 16) * 2, 16 * 32 * 2, launchTInsertZNOffset<4>);
}
TEST_F(TInsertZNTest, case_zn_offset_5)
{
    testTInsertZN<int8_t>((64 * 64 + 32 * 32), 64 * 64, launchTInsertZNOffset<5>);
}
TEST_F(TInsertZNTest, case_zn_offset_6)
{
    testTInsertZN<uint16_t>((32 * 64 + 16 * 32) * 2, 32 * 64 * 2, launchTInsertZNOffset<6>);
}
TEST_F(TInsertZNTest, case_zn_offset_7)
{
    testTInsertZN<uint16_t>((32 * 32 + 16 * 16) * 2, 32 * 32 * 2, launchTInsertZNOffset<7>);
}
TEST_F(TInsertZNTest, case_zn_offset_8)
{
    testTInsertZN<float>((16 * 32 + 8 * 16) * 4, 16 * 32 * 4, launchTInsertZNOffset<8>);
}
TEST_F(TInsertZNTest, case_zn_offset_9)
{
    testTInsertZN<uint16_t>((64 * 64 + 32 * 32) * 2, 64 * 64 * 2, launchTInsertZNOffset<9>);
}
TEST_F(TInsertZNTest, case_zn_offset_10)
{
    testTInsertZN<uint16_t>((16 * 48 + 16 * 16) * 2, 16 * 48 * 2, launchTInsertZNOffset<10>);
}
TEST_F(TInsertZNTest, case_zn_offset_11)
{
    testTInsertZN<float>((8 * 48 + 8 * 16) * 4, 8 * 48 * 4, launchTInsertZNOffset<11>);
}
TEST_F(TInsertZNTest, case_zn_offset_12)
{
    testTInsertZN<uint16_t>((32 * 48 + 16 * 16) * 2, 32 * 48 * 2, launchTInsertZNOffset<12>);
}
TEST_F(TInsertZNTest, case_zn_offset_13)
{
    testTInsertZN<float>((8 * 48 + 8 * 16) * 4, 8 * 48 * 4, launchTInsertZNOffset<13>);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new AclEnv());
    return RUN_ALL_TESTS();
}
