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
#include <pto/cpu/NPUMemoryModel.hpp>
#include "test_common.h"
#include <gtest/gtest.h>
#include <filesystem>

using namespace std;
using namespace PtoTestCommon;

template <int32_t testKey>
void launchTLOAD(uint8_t* out, uint8_t* src, uint64_t* gLog, void* stream);

template <int32_t testKey>
int get_input_golden(uint8_t* input, uint8_t* golden);

class TLOADTest : public testing::Test {
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

#define LOGSIZE 128
#define PRINTLOG 4
#define MAXBLOCK 64

template <int32_t testKey, typename T, int32_t kBlock>
void tload_test()
{
    pto::NPU_MEMORY_INIT();
    pto::NPU_MEMORY_CLEAR();

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
    void* logDevice;

    aclrtMallocHost((void**)(&srcHost), in_byteSize);
    aclrtMallocHost((void**)(&dstHost), out_byteSize);
    aclrtMallocHost((void**)(&goldHost), out_byteSize);

    aclrtMalloc((void**)&dstDevice, in_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, out_byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    int actual_out_byteSize = 0;
    actual_out_byteSize = get_input_golden<testKey>((uint8_t*)srcHost, (uint8_t*)goldHost);
    std::fill((uint8_t*)dstHost, ((uint8_t*)(dstHost)) + out_byteSize, 0);

    aclrtMemcpy(srcDevice, in_byteSize, srcHost, in_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, out_byteSize, dstHost, out_byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

#ifdef DEBUGLOG
    uint64_t logHost[MAXBLOCK][LOGSIZE];
    std::fill((uint8_t*)logHost, ((uint8_t*)(logHost)) + sizeof(logHost), 0);
    aclrtMalloc((void**)&logDevice, sizeof(logHost), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(logDevice, sizeof(logHost), logHost, sizeof(logHost), ACL_MEMCPY_HOST_TO_DEVICE);
#endif

    launchTLOAD<testKey>((uint8_t*)dstDevice, (uint8_t*)srcDevice, (uint64_t*)logDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, out_byteSize, dstDevice, out_byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
#ifdef DEBUGLOG
    aclrtMemcpy(logHost, sizeof(logHost), logDevice, sizeof(logHost), ACL_MEMCPY_DEVICE_TO_HOST);
#endif

    std::filesystem::create_directories(GetGoldenDir());
    std::ofstream inFile(GetGoldenDir() + "/input.bin", std::ios::binary | std::ios::out);
    std::ofstream outFile(GetGoldenDir() + "/output.bin", std::ios::binary | std::ios::out);
    std::ofstream goldFile(GetGoldenDir() + "/golden.bin", std::ios::binary | std::ios::out);
    inFile.write((const char*)srcHost, actual_out_byteSize);
    outFile.write((const char*)dstHost, actual_out_byteSize);
    goldFile.write((const char*)goldHost, actual_out_byteSize);
    inFile.close();
    outFile.close();
    goldFile.close();

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

    using VecType = std::conditional_t<IsTwinType<T>(), uint8_t, T>;
    int elements = actual_out_byteSize / sizeof(T);
    std::vector<VecType> golden(elements);
    std::vector<VecType> devFinal(elements);
    size_t oFileSize = actual_out_byteSize;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", oFileSize, golden.data(), oFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", oFileSize, devFinal.data(), oFileSize));

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

TEST_F(TLOADTest, case_float_GT_128_128_VT_128_128_BLK1) { tload_test<1, float, 1>(); }

TEST_F(TLOADTest, case_float_GT_2_2_2_256_64_VT_256_64_BLK8) { tload_test<2, float, 8>(); }

TEST_F(TLOADTest, case_float_GT_128_127_VT_128_128_BLK1_PADMAX) { tload_test<3, float, 1>(); }

TEST_F(TLOADTest, case_s16_GT_128_127_VT_128_128_BLK1_PADMAX) { tload_test<4, int16_t, 1>(); }

TEST_F(TLOADTest, case_u8_GT_128_127_VT_128_128_BLK1_PADMIN) { tload_test<5, uint8_t, 1>(); }

TEST_F(TLOADTest, case_float_GT_32_64_128_VT_64_128_BLK32_DYN) { tload_test<6, int16_t, 32>(); }

TEST_F(TLOADTest, case_float_GT_32_64_128_VT_64_128_BLK32_STC) { tload_test<7, int16_t, 32>(); }

TEST_F(TLOADTest, case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX) { tload_test<8, float, 8>(); }

TEST_F(TLOADTest, case_float_GT_32_64_128_VT_64_128_BLK32_DN) { tload_test<9, float, 32>(); }

TEST_F(TLOADTest, case_float_GT_2_2_2_255_60_VT_256_64_BLK8_DN) { tload_test<10, float, 8>(); }

TEST_F(TLOADTest, case_NZ_float_1_1_1_16_8_1_1_2_16_8) { tload_test<11, float, 1>(); }

TEST_F(TLOADTest, case_NZ_int16_t_2_2_2_16_16_5_3_3_16_16) { tload_test<12, int16_t, 1>(); }

TEST_F(TLOADTest, case_NZ_int8_t_1_2_1_16_32_2_4_2_16_32) { tload_test<13, uint8_t, 1>(); }

TEST_F(TLOADTest, case_float4_e2m1x2_GT_128_128_VT_128_128_BLK1) { tload_test<14, float4_e2m1x2_t, 1>(); }

TEST_F(TLOADTest, case_float4_e2m1x2_GT_2_2_2_256_64_VT_256_64_BLK8) { tload_test<15, float4_e2m1x2_t, 8>(); }

TEST_F(TLOADTest, case_float4_e2m1x2_GT_128_127_VT_128_128_BLK1_PADMAX) { tload_test<16, float4_e2m1x2_t, 1>(); }

TEST_F(TLOADTest, case_float4_e1m2x2_GT_128_128_VT_128_128_BLK1) { tload_test<17, float4_e1m2x2_t, 1>(); }

TEST_F(TLOADTest, case_float4_e1m2x2_GT_2_2_2_256_64_VT_256_64_BLK8) { tload_test<18, float4_e1m2x2_t, 8>(); }

TEST_F(TLOADTest, case_float4_e1m2x2_GT_128_127_VT_128_128_BLK1_PADMAX) { tload_test<19, float4_e1m2x2_t, 1>(); }

TEST_F(TLOADTest, MatNzRowViewsPreservePreviouslyLoadedRows)
{
    pto::NPU_MEMORY_INIT();
    pto::NPU_MEMORY_CLEAR();

    constexpr int kRows = 128;
    constexpr int kCols = 512;
    constexpr int kL1Offset = 32 * 1024;
    constexpr int kRowOffsetBytes = 32;

    using ParentTile = pto::Tile<
        pto::TileType::Mat, int16_t, kRows, kCols, pto::BLayout::ColMajor, kRows, kCols, pto::SLayout::RowMajor, 512>;
    using RowView = pto::Tile<
        pto::TileType::Mat, int16_t, kRows, kCols, pto::BLayout::ColMajor, 1, kCols, pto::SLayout::RowMajor, 512>;
    using SrcGlobal =
        pto::GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, kCols>, pto::Stride<kCols, kCols, kCols, kCols, 1>>;

    ParentTile parentTile;
    pto::TASSIGN(parentTile, kL1Offset);

    std::vector<int16_t> src(kCols);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            src[col] = static_cast<int16_t>((row * 17 + col) % 30000 + 1);
        }

        RowView rowView;
        pto::TASSIGN(rowView, kL1Offset + row * kRowOffsetBytes);
        SrcGlobal srcGlobal(src.data());
        pto::TLOAD(rowView, srcGlobal);
    }

    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const int16_t expected = static_cast<int16_t>((row * 17 + col) % 30000 + 1);
            EXPECT_EQ(parentTile.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, VecNdPreservesFullBlockGapsAndInactiveRows)
{
    pto::NPU_MEMORY_INIT();
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidRows = 2;
    constexpr int kValidCols = 17;
    using VecTile = pto::Tile<
        pto::TileType::Vec, int16_t, 4, 48, pto::BLayout::RowMajor, kValidRows, kValidCols, pto::SLayout::NoneBox, 512,
        pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, kValidRows, kValidCols>,
        pto::Stride<kValidRows * kValidCols, kValidRows * kValidCols, kValidRows * kValidCols, kValidCols, 1>>;

    VecTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidRows * kValidCols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            int16_t expected = kSentinel;
            if (row < kValidRows && col < kValidCols) {
                expected = src[row * kValidCols + col];
            } else if (row < kValidRows && col < 32) {
                expected = std::numeric_limits<int16_t>::max();
            }
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, VecNullPadPreservesOutsideTransferRegion)
{
    pto::NPU_MEMORY_INIT();
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidCols = 17;
    using VecTile = pto::Tile<
        pto::TileType::Vec, int16_t, 2, 32, pto::BLayout::RowMajor, 1, kValidCols, pto::SLayout::NoneBox, 512,
        pto::PadValue::Null>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, 1, kValidCols>, pto::Stride<kValidCols, kValidCols, kValidCols, kValidCols, 1>>;

    VecTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidCols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            const int16_t expected = row == 0 && col < kValidCols ? src[col] : kSentinel;
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, VecDnPreservesFullBlockGapsAndInactiveCols)
{
    pto::NPU_MEMORY_INIT();
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidRows = 17;
    constexpr int kValidCols = 2;
    using VecTile = pto::Tile<
        pto::TileType::Vec, int16_t, 48, 4, pto::BLayout::ColMajor, kValidRows, kValidCols, pto::SLayout::NoneBox, 512,
        pto::PadValue::Min>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, kValidRows, kValidCols>,
        pto::Stride<kValidRows * kValidCols, kValidRows * kValidCols, kValidRows * kValidCols, 1, kValidRows>,
        pto::Layout::DN>;

    VecTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidRows * kValidCols);
    for (int col = 0; col < kValidCols; ++col) {
        for (int row = 0; row < kValidRows; ++row) {
            src[col * kValidRows + row] = static_cast<int16_t>(col * 100 + row + 1);
        }
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            int16_t expected = kSentinel;
            if (row < kValidRows && col < kValidCols) {
                expected = src[col * kValidRows + row];
            } else if (row < 32 && col < kValidCols) {
                expected = std::numeric_limits<int16_t>::min();
            }
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, VecNzPreservesOutsideTransferRegion)
{
    pto::NPU_MEMORY_INIT();
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidRows = 16;
    constexpr int kValidCols = 16;
    using VecTile = pto::Tile<
        pto::TileType::Vec, int16_t, 32, 32, pto::BLayout::ColMajor, kValidRows, kValidCols, pto::SLayout::RowMajor,
        512, pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, kValidRows, kValidCols>,
        pto::Stride<kValidRows * kValidCols, kValidRows * kValidCols, kValidRows * kValidCols, kValidCols, 1>,
        pto::Layout::NZ>;

    VecTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidRows * kValidCols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            const int16_t expected = row < kValidRows && col < kValidCols ? src[row * kValidCols + col] : kSentinel;
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, A2A3MatNdPreservesOutsideAlignedTransferRegion)
{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A2A3);
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidRows = 2;
    constexpr int kValidCols = 16;
    using MatTile = pto::Tile<
        pto::TileType::Mat, int16_t, 4, 48, pto::BLayout::RowMajor, kValidRows, kValidCols, pto::SLayout::NoneBox, 512,
        pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, kValidRows, kValidCols>,
        pto::Stride<kValidRows * kValidCols, kValidRows * kValidCols, kValidRows * kValidCols, kValidCols, 1>>;

    MatTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidRows * kValidCols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            const int16_t expected = row < kValidRows && col < kValidCols ? src[row * kValidCols + col] : kSentinel;
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, A2A3MatNdToNzZeroPadsOnlyFinalC0Tail)
{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A2A3);
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidRows = 2;
    constexpr int kValidCols = 17;
    using MatTile = pto::Tile<
        pto::TileType::Mat, int16_t, 16, 48, pto::BLayout::ColMajor, kValidRows, kValidCols, pto::SLayout::RowMajor,
        512, pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, kValidRows, kValidCols>,
        pto::Stride<kValidRows * kValidCols, kValidRows * kValidCols, kValidRows * kValidCols, kValidCols, 1>>;

    MatTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidRows * kValidCols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            int16_t expected = kSentinel;
            if (row < kValidRows && col < kValidCols) {
                expected = src[row * kValidCols + col];
            } else if (row < kValidRows && col < 32) {
                expected = 0;
            }
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

TEST_F(TLOADTest, A5MatNdPadsOnlyFinalPartialBlock)
{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kValidRows = 2;
    constexpr int kValidCols = 17;
    using MatTile = pto::Tile<
        pto::TileType::Mat, int16_t, 4, 48, pto::BLayout::RowMajor, kValidRows, kValidCols, pto::SLayout::NoneBox, 512,
        pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, 1, kValidRows, kValidCols>,
        pto::Stride<kValidRows * kValidCols, kValidRows * kValidCols, kValidRows * kValidCols, kValidCols, 1>>;

    MatTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kValidRows * kValidCols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data());
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            int16_t expected = kSentinel;
            if (row < kValidRows && col < kValidCols) {
                expected = src[row * kValidCols + col];
            } else if (row < kValidRows && col < 32) {
                expected = std::numeric_limits<int16_t>::max();
            }
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

// GM holds an attention-style [B, S, N, G, D] tensor. One [S, G, D] slice of a single head is
// loaded into an NZ Mat tile whose rows are the (s, g) pairs, which is the layout matmul wants.
// GlobalTensor DIM_2 (S) is the nd-matrix count and maps onto the hardware ndNum loop.
TEST_F(TLOADTest, A5MatNdToNzMultiNdStacksRows)
{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kS = 8;  // nd matrix count -> ndNum
    constexpr int kG = 2;  // rows per nd matrix -> nValue
    constexpr int kD = 16; // row length -> dValue, one full C0 for int16_t
    constexpr int kN = 3;  // heads, only head kHead is loaded
    constexpr int kHead = 1;
    constexpr int kValidRows = kS * kG;
    constexpr int kSStride = kN * kG * kD;

    using MatTile = pto::Tile<
        pto::TileType::Mat, int16_t, kValidRows, kD, pto::BLayout::ColMajor, kValidRows, kD, pto::SLayout::RowMajor,
        512>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, kS, kG, kD>, pto::Stride<kS * kSStride, kS * kSStride, kSStride, kD, 1>,
        pto::Layout::ND>;

    MatTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kS * kSStride);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data() + kHead * kG * kD);
    pto::TLOAD(dst, srcGlobal);

    for (int s = 0; s < kS; ++s) {
        for (int g = 0; g < kG; ++g) {
            for (int d = 0; d < kD; ++d) {
                const int16_t expected = src[s * kSStride + kHead * kG * kD + g * kD + d];
                EXPECT_EQ(dst.GetElement(s * kG + g, d), expected) << "s=" << s << " g=" << g << " d=" << d;
            }
        }
    }
}

// Same multi-ND load with a row length that is not a whole C0. Only the final partial C0 block of
// the valid rows is zero filled; rows and full blocks outside the valid area stay untouched.
TEST_F(TLOADTest, A5MatNdToNzMultiNdZeroPadsFinalC0Tail)
{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kS = 4;
    constexpr int kG = 2;
    constexpr int kD = 20;
    constexpr int kN = 3;
    constexpr int kHead = 2;
    constexpr int kValidRows = kS * kG;
    constexpr int kSStride = kN * kG * kD;
    constexpr int kTileCols = 48;

    using MatTile = pto::Tile<
        pto::TileType::Mat, int16_t, 16, kTileCols, pto::BLayout::ColMajor, kValidRows, kD, pto::SLayout::RowMajor, 512,
        pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, kS, kG, kD>, pto::Stride<kS * kSStride, kS * kSStride, kSStride, kD, 1>,
        pto::Layout::ND>;

    MatTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kS * kSStride);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data() + kHead * kG * kD);
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            int16_t expected = kSentinel;
            if (row < kValidRows && col < kD) {
                const int s = row / kG;
                const int g = row % kG;
                expected = src[s * kSStride + kHead * kG * kD + g * kD + col];
            } else if (row < kValidRows && col < 32) {
                expected = 0;
            }
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}

// Same multi-ND contract on A2/A3, where DIM_2 maps onto the instruction's own ndNum operand
// instead of a config register. The row mapping the simulator must reproduce is identical.
TEST_F(TLOADTest, A2A3MatNdToNzMultiNdStacksRows)
{
    pto::NPU_MEMORY_INIT(pto::NPUArch::A2A3);
    pto::NPU_MEMORY_CLEAR();

    constexpr int kSentinel = 1234;
    constexpr int kS = 8;
    constexpr int kG = 3;
    constexpr int kD = 20; // not a whole C0 for int16_t, so the C0 tail must be zero filled
    constexpr int kN = 2;
    constexpr int kHead = 1;
    constexpr int kValidRows = kS * kG;
    constexpr int kSStride = kN * kG * kD;
    constexpr int kTileCols = 48;

    using MatTile = pto::Tile<
        pto::TileType::Mat, int16_t, 32, kTileCols, pto::BLayout::ColMajor, kValidRows, kD, pto::SLayout::RowMajor, 512,
        pto::PadValue::Max>;
    using SrcGlobal = pto::GlobalTensor<
        int16_t, pto::Shape<1, 1, kS, kG, kD>, pto::Stride<kS * kSStride, kS * kSStride, kSStride, kD, 1>,
        pto::Layout::ND>;

    MatTile dst;
    pto::TASSIGN(dst, 4096);
    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            dst.SetElement(row, col, kSentinel);
        }
    }

    std::vector<int16_t> src(kS * kSStride);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<int16_t>(i + 1);
    }
    SrcGlobal srcGlobal(src.data() + kHead * kG * kD);
    pto::TLOAD(dst, srcGlobal);

    for (int row = 0; row < MatTile::Rows; ++row) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            int16_t expected = kSentinel;
            if (row < kValidRows && col < kD) {
                const int s = row / kG;
                const int g = row % kG;
                expected = src[s * kSStride + kHead * kG * kD + g * kD + col];
            } else if (row < kValidRows && col < 32) {
                expected = 0;
            }
            EXPECT_EQ(dst.GetElement(row, col), expected) << "row=" << row << " col=" << col;
        }
    }
}
