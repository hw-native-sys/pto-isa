/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"
#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <
    int format, typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTStore(T* out, T* src, void* stream);

class TStoreTest : public testing::Test {
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
    int format, typename DataType, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
    int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void test_tstore()
{
    size_t dataSize = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4 * sizeof(DataType);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *srcHost;
    DataType *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dataSize);
    aclrtMallocHost((void**)(&srcHost), dataSize);

    aclrtMalloc((void**)&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::fill(dstDevice, dstDevice + (dataSize / sizeof(DataType)), 0);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", dataSize, srcHost, dataSize));

    aclrtMemcpy(srcDevice, dataSize, srcHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStore<
        format, DataType, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
        gWholeShape3, gWholeShape4>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(dataSize);
    std::vector<DataType> devFinal(dataSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dataSize, golden.data(), dataSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dataSize, devFinal.data(), dataSize));

    bool ret = ResultCmp<DataType>(golden, devFinal, 0);
    EXPECT_TRUE(ret);
}

TEST_F(TStoreTest, ND_float_1_1_1_2_128_1_1_1_2_128) { test_tstore<0, float, 1, 1, 1, 2, 128, 1, 1, 1, 2, 128>(); }

TEST_F(TStoreTest, ND_int16_t_1_2_1_23_121_3_2_2_35_125)
{
    test_tstore<0, int16_t, 1, 2, 1, 23, 121, 3, 2, 2, 35, 125>();
}

TEST_F(TStoreTest, ND_int8_t_2_2_3_23_47_3_3_4_32_50) { test_tstore<0, int8_t, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50>(); }

TEST_F(TStoreTest, DN_float_1_1_1_4_21_1_1_1_8_32) { test_tstore<1, float, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32>(); }

TEST_F(TStoreTest, DN_int16_t_3_1_1_1_124_5_1_1_2_128) { test_tstore<1, int16_t, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128>(); }

TEST_F(TStoreTest, DN_int8_t_2_1_2_32_32_3_4_3_64_35) { test_tstore<1, int8_t, 2, 1, 2, 32, 32, 3, 4, 3, 64, 35>(); }

TEST_F(TStoreTest, NZ_float_1_1_1_16_8_1_1_2_16_8) { test_tstore<2, float, 1, 1, 1, 16, 8, 1, 1, 2, 16, 8>(); }

TEST_F(TStoreTest, NZ_int16_t_2_2_2_16_16_5_3_3_16_16) { test_tstore<2, int16_t, 2, 2, 2, 16, 16, 5, 3, 3, 16, 16>(); }

TEST_F(TStoreTest, NZ_int8_t_1_2_1_16_32_2_4_2_16_32) { test_tstore<2, int8_t, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32>(); }

TEST_F(TStoreTest, ND_int64_1_1_1_2_128_1_1_1_2_128) { test_tstore<0, int64_t, 1, 1, 1, 2, 128, 1, 1, 1, 2, 128>(); }

TEST_F(TStoreTest, ND_uint64_t_1_2_1_23_121_3_2_2_35_125)
{
    test_tstore<0, uint64_t, 1, 2, 1, 23, 121, 3, 2, 2, 35, 125>();
}

TEST_F(TStoreTest, DN_int64_1_1_1_4_21_1_1_1_8_32) { test_tstore<1, int64_t, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32>(); }

TEST_F(TStoreTest, DN_uint64_t_3_1_1_1_124_5_1_1_2_128)
{
    test_tstore<1, uint64_t, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128>();
}

TEST_F(TStoreTest, FpVariantStoresTileIntoGlobalTensor)
{
    using TileData = pto::Tile<pto::TileType::Vec, float, 2, 8>;
    using FpTile = pto::Tile<pto::TileType::Vec, float, 1, 8>;
    using GlobalData = pto::GlobalTensor<float, pto::Shape<1, 1, 1, 2, 8>, pto::Stride<16, 16, 16, 8, 1>>;

    TileData src;
    FpTile fp;
    size_t addr = 0;
    CpuTileTestUtils::AssignTileStorage(addr, src, fp);

    std::vector<float> buffer(16, 0.0f);
    GlobalData dst(buffer.data());

    CpuTileTestUtils::FillLinear(src, 1.0f);
    CpuTileTestUtils::FillAll(fp, 0.5f);
    pto::TSTORE_FP(dst, src, fp);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            CpuTileTestUtils::ExpectValueEquals(buffer[static_cast<size_t>(r) * src.GetValidCol() + c],
                                                CpuTileTestUtils::GetValue(src, r, c));
        }
    }
}
