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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <limits>
#include <vector>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTROWSUM_demo(uint8_t* out, uint8_t* src, void* stream);

class TROWSUMTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { pto::NPU_MEMORY_INIT(pto::NPUArch::A2A3); }
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTROWSUM(T* out, T* src, void* stream);

template <typename T, int kGSize_>
inline void init_dst(T* dstHost)
{
    for (size_t i = 0; i < kGSize_; i++) {
        dstHost[i] = 0;
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_trowsum()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);

    aclrtMalloc((void**)(&dstDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSize, 0, fileSize);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", fileSize, srcHost, fileSize));
    init_dst<T, kGRows_ * kGCols_>(dstHost);

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTROWSUM<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWSUMTest, case_float_64x64_64x64_64x64) { test_trowsum<float, 64, 64, 64, 64>(); }
TEST_F(TROWSUMTest, case_half_16x256_16x256_16x256) { test_trowsum<aclFloat16, 16, 256, 16, 256>(); }
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TROWSUMTest, case_bf16_16x256_16x256_16x256) { test_trowsum<bfloat16_t, 16, 256, 16, 256>(); }
#endif

template <int Rows, std::size_t Cols>
void ExpectFp32ReductionResult(pto::NPUArch arch, const std::array<float, Cols>& input, int validCols, float expected)
{
    SCOPED_TRACE(arch == pto::NPUArch::A5 ? "A5" : "A2A3");
    using SrcTile = pto::Tile<pto::TileType::Vec, float, Rows, Cols, pto::BLayout::RowMajor, Rows, -1>;
    using DstTile = pto::Tile<pto::TileType::Vec, float, Rows, 8, pto::BLayout::RowMajor, Rows, 1>;

    pto::NPU_MEMORY_INIT(arch);
    SrcTile src(validCols);
    SrcTile tmp(validCols);
    DstTile dst;
    pto::TASSIGN(src, 0);
    pto::TASSIGN(tmp, SrcTile::GetSizeInBytes());
    pto::TASSIGN(dst, SrcTile::GetSizeInBytes() * 2);

    for (int row = 0; row < Rows; ++row) {
        std::copy(input.begin(), input.end(), src.data() + row * Cols);
    }
    constexpr float sentinel = -123.0f;
    std::fill_n(dst.data(), Rows * DstTile::Cols, sentinel);

    pto::TROWSUM(dst, src, tmp);

    for (int row = 0; row < Rows; ++row) {
        ASSERT_EQ(std::bit_cast<uint32_t>(dst.data()[row * DstTile::Cols]), std::bit_cast<uint32_t>(expected))
            << "row=" << row << ", validCols=" << validCols;
        for (int col = 1; col < DstTile::Cols; ++col) {
            EXPECT_EQ(dst.data()[row * DstTile::Cols + col], sentinel);
        }
    }
}

template <std::size_t Cols>
std::array<float, Cols> CancellationInput(int validCols = Cols)
{
    constexpr std::array<float, 4> pattern = {100000000.0f, 1.0f, -100000000.0f, 1.0f};
    std::array<float, Cols> input;
    input.fill(std::numeric_limits<float>::quiet_NaN());
    for (int col = 0; col < validCols; ++col) {
        input[col] = pattern[col % pattern.size()];
    }
    return input;
}

std::array<float, 64> ExactFp32Input()
{
    // Every partial sum is exactly representable, including after vectorization.
    std::array<float, 64> input;
    for (std::size_t col = 0; col < input.size(); ++col) {
        input[col] = static_cast<float>(col + 1) / 4.0f;
    }
    return input;
}

TEST_F(TROWSUMTest, case_a2a3_fp32_exact_sum)
{
    ExpectFp32ReductionResult<1>(pto::NPUArch::A2A3, ExactFp32Input(), 64, 520.0f);
}

TEST_F(TROWSUMTest, case_a5_fp32_reduction_tree_exact)
{
    ExpectFp32ReductionResult<1>(pto::NPUArch::A5, CancellationInput<64>(), 64, 0.0f);
}

TEST_F(TROWSUMTest, case_fp32_arch_switch_and_parallel_threshold_exact)
{
    if (pto::cpu::get_thread_count() < 2) {
        GTEST_SKIP() << "Parallel regression requires at least two CPU threads.";
    }
    constexpr int rowsAtThreshold = (PTO_CPU_PARALLEL_THRESHOLD_ELEMS + 63) / 64;
    static_assert(rowsAtThreshold > 1);
    const auto input = CancellationInput<64>();
    ExpectFp32ReductionResult<rowsAtThreshold - 1>(pto::NPUArch::A5, input, 64, 0.0f);
    ExpectFp32ReductionResult<rowsAtThreshold>(pto::NPUArch::A5, input, 64, 0.0f);
    ExpectFp32ReductionResult<rowsAtThreshold + 1>(pto::NPUArch::A5, input, 64, 0.0f);
    ExpectFp32ReductionResult<rowsAtThreshold>(pto::NPUArch::A2A3, ExactFp32Input(), 64, 520.0f);
    ExpectFp32ReductionResult<rowsAtThreshold>(pto::NPUArch::A5, input, 64, 0.0f);
}

TEST_F(TROWSUMTest, case_a5_fp32_tail_and_multivector_exact)
{
    // Poison padding to detect reductions that include masked lanes.
    for (int validCols : {1, 63, 64, 65, 127, 128, 129, 192, 255, 256}) {
        SCOPED_TRACE(validCols);
        const auto input = CancellationInput<256>(validCols);
        const float expected = validCols % 4 == 1 ? 100000000.0f : 0.0f;
        ExpectFp32ReductionResult<1>(pto::NPUArch::A5, input, validCols, expected);
        ExpectFp32ReductionResult<130>(pto::NPUArch::A5, input, validCols, expected);
    }
    ExpectFp32ReductionResult<263>(pto::NPUArch::A5, CancellationInput<64>(63), 63, 0.0f);
    ExpectFp32ReductionResult<253>(pto::NPUArch::A5, CancellationInput<128>(65), 65, 100000000.0f);
}

TEST_F(TROWSUMTest, case_a5_fp32_group_accumulation_order_exact)
{
    // Each vector sums exactly; address-order accumulation gives 1, while a tree gives 0.
    std::array<float, 256> input{};
    input[0] = 100000000.0f;
    input[64] = 1.0f;
    input[128] = -100000000.0f;
    input[192] = 1.0f;
    ExpectFp32ReductionResult<1>(pto::NPUArch::A5, input, 256, 1.0f);
    ExpectFp32ReductionResult<65>(pto::NPUArch::A5, input, 256, 1.0f);
}

TEST_F(TROWSUMTest, case_a5_fp32_signed_zero_bits)
{
    std::array<float, 64> input;
    input.fill(-0.0f);
    // Adding the row's initial +0 to a vector sum of -0 produces +0.
    ExpectFp32ReductionResult<1>(pto::NPUArch::A5, input, 64, 0.0f);
    ExpectFp32ReductionResult<256>(pto::NPUArch::A5, input, 64, 0.0f);
    input.fill(0.0f);
    ExpectFp32ReductionResult<1>(pto::NPUArch::A5, input, 63, 0.0f);
}

template <typename T, int Rows, bool Dn = false, std::size_t Cols>
void ExpectA5TypedReduction(const std::array<T, Cols>& input, int validCols, T expected)
{
    using Src = pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor, Rows, -1>;
    constexpr int alignment = 32 / sizeof(T);
    constexpr int dstCols = Dn ? 1 : alignment;
    constexpr int dstRows = Dn ? (Rows + alignment - 1) / alignment * alignment : Rows;
    using Dst = pto::Tile<
        pto::TileType::Vec, T, dstRows, dstCols, Dn ? pto::BLayout::ColMajor : pto::BLayout::RowMajor, Rows, 1>;
    pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
    Src src(validCols);
    Src tmp(validCols);
    Dst dst;
    pto::TASSIGN(src, 0);
    pto::TASSIGN(tmp, Src::GetSizeInBytes());
    pto::TASSIGN(dst, Src::GetSizeInBytes() * 2);
    for (int row = 0; row < Rows; ++row) {
        std::copy(input.begin(), input.end(), src.data() + row * Cols);
    }
    constexpr T sentinel = static_cast<T>(123);
    std::fill_n(dst.data(), Dst::Numel, sentinel);
    pto::TROWSUM(dst, src, tmp);
    for (int row = 0; row < Rows; ++row) {
        const T actual = dst.data()[row * dstCols];
        if constexpr (std::is_same_v<T, half>) {
            ASSERT_EQ(std::bit_cast<uint16_t>(actual), std::bit_cast<uint16_t>(expected)) << "row=" << row;
        } else {
            ASSERT_EQ(actual, expected) << "row=" << row;
        }
        for (int col = 1; col < dstCols; ++col) {
            EXPECT_EQ(dst.data()[row * dstCols + col], sentinel);
        }
    }
    for (int row = Rows; row < dstRows; ++row) {
        EXPECT_EQ(dst.data()[row], sentinel);
    }
}

TEST_F(TROWSUMTest, case_a5_half_tree_rounding_and_tail)
{
    std::array<half, 256> input;
    constexpr std::array<half, 4> pattern = {4096, 1, -4096, 1};
    for (int validCols : {127, 128, 129, 255, 256}) {
        SCOPED_TRACE(validCols);
        input.fill(static_cast<half>(std::numeric_limits<float>::quiet_NaN()));
        for (int col = 0; col < validCols; ++col) {
            input[col] = pattern[col % pattern.size()];
        }
        const half expected = validCols == 129 ? 4096 : 0;
        ExpectA5TypedReduction<half, 1>(input, validCols, expected);
        ExpectA5TypedReduction<half, 130, true>(input, validCols, expected);
    }
}

TEST_F(TROWSUMTest, case_a5_half_rounds_between_vectors)
{
    std::array<half, 512> input{};
    input[0] = 4096;
    input[128] = 1;
    input[256] = -4096;
    input[384] = 1;
    // Half accumulation gives 1; widening all four group sums to float gives 2.
    ExpectA5TypedReduction<half, 1>(input, 512, static_cast<half>(1));
    ExpectA5TypedReduction<half, 33>(input, 512, static_cast<half>(1));
}

TEST_F(TROWSUMTest, case_a5_half_valid_columns_do_not_truncate)
{
    std::array<half, 65536> input{};
    input.front() = 1;
    input.back() = 2;
    ExpectA5TypedReduction<half, 1>(input, 65536, static_cast<half>(3));
}

template <int ValidRows = -1, bool Dn = false>
void ExpectWideRowCount(pto::NPUArch arch, unsigned validRows)
{
    constexpr int physicalRows = 65544;
    using Src = pto::Tile<pto::TileType::Vec, float, physicalRows, 8, pto::BLayout::RowMajor, ValidRows, 8>;
    using Dst = pto::Tile<
        pto::TileType::Vec, float, physicalRows, Dn ? 1 : 8, Dn ? pto::BLayout::ColMajor : pto::BLayout::RowMajor,
        ValidRows, 1>;
    Src src;
    Src tmp;
    Dst dst;
    if constexpr (ValidRows == -1) {
        src.SetValidRow(validRows);
        dst.SetValidRow(validRows);
    }
    constexpr float sentinel = -123.0f;
    std::vector<float> input(Src::Numel, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> output(Dst::Numel, sentinel);
    for (unsigned row = 0; row < validRows; ++row) {
        std::fill_n(input.data() + row * Src::Cols, Src::Cols, 1.0f);
        input[row * Src::Cols] += static_cast<float>(row % 17);
    }
    pto::NPU_MEMORY_INIT(arch);
    // Use host storage so row-count coverage does not depend on simulated UB capacity.
    src.data() = input.data();
    dst.data() = output.data();
    pto::TROWSUM(dst, src, tmp);
    for (unsigned row = 0; row < physicalRows; ++row) {
        const float expected = row < validRows ? 8.0f + static_cast<float>(row % 17) : sentinel;
        ASSERT_EQ(std::bit_cast<uint32_t>(output[row * Dst::Cols]), std::bit_cast<uint32_t>(expected))
            << "row=" << row << ", validRows=" << validRows << ", dn=" << Dn;
        for (int col = 1; col < Dst::Cols; ++col) {
            ASSERT_EQ(output[row * Dst::Cols + col], sentinel);
        }
    }
}

TEST_F(TROWSUMTest, case_a5_dynamic_valid_rows_do_not_truncate)
{
    for (unsigned rows : {65535u, 65536u, 65537u}) {
        ExpectWideRowCount<-1, false>(pto::NPUArch::A5, rows);
        ExpectWideRowCount<-1, true>(pto::NPUArch::A5, rows);
    }
}

TEST_F(TROWSUMTest, case_a2a3_dynamic_valid_rows_do_not_truncate)
{
    for (unsigned rows : {65535u, 65536u, 65537u}) {
        ExpectWideRowCount<-1, false>(pto::NPUArch::A2A3, rows);
        ExpectWideRowCount<-1, true>(pto::NPUArch::A2A3, rows);
    }
}

TEST_F(TROWSUMTest, case_a5_static_valid_rows_do_not_truncate)
{
    ExpectWideRowCount<65535>(pto::NPUArch::A5, 65535);
    ExpectWideRowCount<65536>(pto::NPUArch::A5, 65536);
    ExpectWideRowCount<65537>(pto::NPUArch::A5, 65537);
    ExpectWideRowCount<65535, true>(pto::NPUArch::A5, 65535);
    ExpectWideRowCount<65536, true>(pto::NPUArch::A5, 65536);
    ExpectWideRowCount<65537, true>(pto::NPUArch::A5, 65537);
}

TEST_F(TROWSUMTest, case_a2a3_static_valid_rows_do_not_truncate)
{
    ExpectWideRowCount<65535>(pto::NPUArch::A2A3, 65535);
    ExpectWideRowCount<65536>(pto::NPUArch::A2A3, 65536);
    ExpectWideRowCount<65537>(pto::NPUArch::A2A3, 65537);
    ExpectWideRowCount<65535, true>(pto::NPUArch::A2A3, 65535);
    ExpectWideRowCount<65536, true>(pto::NPUArch::A2A3, 65536);
    ExpectWideRowCount<65537, true>(pto::NPUArch::A2A3, 65537);
}

template <typename T>
void ExpectSignedIntegerOverflow()
{
    constexpr T minValue = std::numeric_limits<T>::min();
    constexpr T maxValue = std::numeric_limits<T>::max();
    std::array<T, 256> input{};
    input[0] = maxValue;
    input[128] = 1;
    input[129] = maxValue; // Must be masked out.
    ExpectA5TypedReduction<T, 1>(input, 129, minValue);
    ExpectA5TypedReduction<T, 65, true>(input, 129, minValue);
    input[0] = minValue;
    input[128] = -1;
    ExpectA5TypedReduction<T, 1, true>(input, 129, maxValue);
    input.fill(maxValue);
    // For odd n=255, n*(2^(bits-1)-1) modulo 2^bits equals maxValue-254.
    ExpectA5TypedReduction<T, 65>(input, 255, static_cast<T>(maxValue - 254));
}

TEST_F(TROWSUMTest, case_a5_int16_truncates_overflow) { ExpectSignedIntegerOverflow<int16_t>(); }
TEST_F(TROWSUMTest, case_a5_int32_wraps_overflow) { ExpectSignedIntegerOverflow<int32_t>(); }
TEST_F(TROWSUMTest, case_a5_int64_wraps_overflow) { ExpectSignedIntegerOverflow<int64_t>(); }

TEST_F(TROWSUMTest, case_a5_uint64_carry_and_wrap)
{
    std::array<uint64_t, 256> input{};
    input[0] = 0xffffffffULL;
    input[32] = 1;
    input[64] = 2;
    input[65] = 123;
    ExpectA5TypedReduction<uint64_t, 1, true>(input, 65, 0x100000002ULL);
    input.fill(std::numeric_limits<uint64_t>::max());
    ExpectA5TypedReduction<uint64_t, 65>(input, 255, std::numeric_limits<uint64_t>::max() - 254);
    ExpectA5TypedReduction<uint64_t, 65, true>(input, 255, std::numeric_limits<uint64_t>::max() - 254);
}

using RowSumValidSrc = pto::Tile<pto::TileType::Vec, float, 16, 16, pto::BLayout::RowMajor, -1, -1>;
using RowSumValidDst = pto::Tile<pto::TileType::Vec, float, 8, 8, pto::BLayout::RowMajor, -1, 1>;

template <typename Src, typename Dst>
void InvokeRowSumForValidation(Src& src, Dst& dst, pto::NPUArch arch = pto::NPUArch::A5)
{
    Src tmp;
    pto::NPU_MEMORY_INIT(arch);
    pto::TASSIGN(src, 0);
    pto::TASSIGN(tmp, Src::GetSizeInBytes());
    pto::TASSIGN(dst, Src::GetSizeInBytes() * 2);
    std::fill_n(src.data(), Src::Numel, static_cast<typename Src::DType>(1));
    std::fill_n(dst.data(), Dst::Numel, static_cast<typename Dst::DType>(123));
    pto::TROWSUM(dst, src, tmp);
}

TEST_F(TROWSUMTest, case_a5_accepts_different_physical_rows)
{
    RowSumValidSrc src(4, 16);
    RowSumValidDst dst(4);
    InvokeRowSumForValidation(src, dst);
    for (int row = 0; row < 4; ++row) {
        EXPECT_EQ(dst.data()[row * RowSumValidDst::Cols], 16.0f);
    }
    for (int row = 4; row < RowSumValidDst::Rows; ++row) {
        EXPECT_EQ(dst.data()[row * RowSumValidDst::Cols], 123.0f);
    }
}

TEST_F(TROWSUMTest, case_a5_rejects_empty_or_mismatched_valid_rows)
{
    RowSumValidSrc emptyRows(0, 16);
    RowSumValidSrc emptyCols(4, 0);
    RowSumValidSrc src(4, 16);
    RowSumValidDst dst(3);
    EXPECT_DEATH(InvokeRowSumForValidation(emptyRows, dst), "must be non-empty");
    EXPECT_DEATH(InvokeRowSumForValidation(emptyCols, dst), "must be non-empty");
    EXPECT_DEATH(InvokeRowSumForValidation(src, dst), "preserves row count");
}

TEST_F(TROWSUMTest, case_a5_rejects_mixed_types)
{
    pto::Tile<pto::TileType::Vec, half, 8, 16> srcHalf;
    pto::Tile<pto::TileType::Vec, int16_t, 8, 16> srcInt;
    pto::Tile<pto::TileType::Vec, float, 8, 8> dstFloat;
    pto::Tile<pto::TileType::Vec, int32_t, 8, 8> dstInt;
    EXPECT_DEATH(InvokeRowSumForValidation(srcHalf, dstFloat), "requires matching");
    EXPECT_DEATH(InvokeRowSumForValidation(srcInt, dstInt), "requires matching");
}

TEST_F(TROWSUMTest, case_a5_rejects_invalid_layouts_and_locations)
{
    pto::Tile<pto::TileType::Vec, float, 16, 16> nd;
    pto::Tile<pto::TileType::Vec, float, 16, 16, pto::BLayout::ColMajor> dn;
    pto::Tile<pto::TileType::Vec, float, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::RowMajor> boxed;
    pto::Tile<pto::TileType::Mat, float, 16, 16> mat;
    EXPECT_DEATH(InvokeRowSumForValidation(dn, nd), "Input tile must use ND");
    EXPECT_DEATH(InvokeRowSumForValidation(boxed, nd), "Input tile must use ND");
    EXPECT_DEATH(InvokeRowSumForValidation(nd, dn), "Output tile must use ND");
    EXPECT_DEATH(InvokeRowSumForValidation(nd, boxed), "Output tile must use ND");
    EXPECT_DEATH(InvokeRowSumForValidation(mat, nd), "only works on vector tiles");
    EXPECT_DEATH(InvokeRowSumForValidation(nd, mat), "only works on vector tiles");
}

TEST_F(TROWSUMTest, case_a5_rejects_bfloat16)
{
    if constexpr (std::is_same_v<bfloat16_t, half>) {
        GTEST_SKIP() << "Native BF16 is unavailable in this compiler.";
    } else {
        pto::Tile<pto::TileType::Vec, bfloat16_t, 8, 16> src;
        pto::Tile<pto::TileType::Vec, bfloat16_t, 8, 16> dst;
        EXPECT_DEATH(InvokeRowSumForValidation(src, dst), "requires matching");
    }
}

TEST_F(TROWSUMTest, case_a2a3_keeps_legacy_mixed_types)
{
    pto::Tile<pto::TileType::Vec, half, 8, 16> srcHalf;
    pto::Tile<pto::TileType::Vec, float, 8, 8> dstFloat;
    InvokeRowSumForValidation(srcHalf, dstFloat, pto::NPUArch::A2A3);
    EXPECT_EQ(dstFloat.data()[0], 16.0f);
    pto::Tile<pto::TileType::Vec, int16_t, 8, 16> srcInt;
    pto::Tile<pto::TileType::Vec, int32_t, 8, 8> dstInt;
    InvokeRowSumForValidation(srcInt, dstInt, pto::NPUArch::A2A3);
    EXPECT_EQ(dstInt.data()[0], 16);
}

TEST_F(TROWSUMTest, case_a2a3_rejects_a5_only_int64)
{
    pto::Tile<pto::TileType::Vec, int64_t, 8, 16> src;
    pto::Tile<pto::TileType::Vec, int64_t, 8, 4> dst;
    EXPECT_DEATH(InvokeRowSumForValidation(src, dst, pto::NPUArch::A2A3), "Not supported data type for A2A3");
}
