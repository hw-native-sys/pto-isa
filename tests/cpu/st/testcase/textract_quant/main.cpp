/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;
using namespace pto;

class TEXTRACTTest : public testing::Test {
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
    const std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

std::pair<int, int> get_closest_factor(int c)
{
    if (c <= 0)
        return {0, 0};
    int a = static_cast<int>(std::sqrt(c));
    while (a > 0) {
        if (c % a == 0) {
            int b = c / a;
            return {a, b};
        }
        a--;
    }
    return {1, c};
}

template <typename SrcT, typename DstT, size_t src_rows, size_t src_cols, size_t src_validRows, size_t src_validCols,
          size_t dst_rows, size_t dst_cols, size_t dst_validRows, size_t dst_validCols, size_t idx_row, size_t idx_col,
          bool is_v_quant, bool apply_relu, bool apply_saturation = false>
struct Params {
    using ST = SrcT;
    using DT = DstT;
    static constexpr size_t srcRows = src_rows;
    static constexpr size_t srcCols = src_cols;
    static constexpr size_t srcValidRows = src_validRows;
    static constexpr size_t srcValidCols = src_validCols;
    static constexpr size_t dstRows = dst_rows;
    static constexpr size_t dstCols = dst_cols;
    static constexpr size_t dstValidRows = dst_validRows;
    static constexpr size_t dstValidCols = dst_validCols;
    static constexpr size_t idxRow = idx_row;
    static constexpr size_t idxCol = idx_col;
    static constexpr bool isVQuant = is_v_quant;
    static constexpr bool applyRelu = apply_relu;
    static constexpr bool applySaturation = apply_saturation;
};

template <typename Conf>
void runTEXTRACT_Scalar(typename Conf::DT *dst, typename Conf::ST *src, uint64_t *quant)
{
    using ST = typename Conf::ST;
    using DT = typename Conf::DT;
    using GlobalDataDst = GlobalTensor<
        DT, pto::Shape<1, 1, 1, Conf::dstValidRows, Conf::dstValidCols>,
        pto::Stride<1 * Conf::dstValidRows * Conf::dstValidCols, 1 * Conf::dstValidRows * Conf::dstValidCols,
                    Conf::dstValidRows * Conf::dstValidCols, Conf::dstValidCols, 1>>;
    using GlobalDataSrc = GlobalTensor<
        ST, pto::Shape<1, 1, 1, Conf::srcValidRows, Conf::srcValidCols>,
        pto::Stride<1 * Conf::srcValidRows * Conf::srcValidCols, 1 * Conf::srcValidRows * Conf::srcValidCols,
                    Conf::srcValidRows * Conf::srcValidCols, Conf::srcValidCols, 1>>;

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(dst);

    using SrcTile = Tile<TileType::Mat, ST, Conf::srcRows, Conf::srcCols, BLayout::RowMajor, Conf::srcValidRows,
                         Conf::srcValidCols, SLayout::NoneBox, 512>;
    using DstTile = Tile<TileType::Mat, DT, Conf::dstRows, Conf::dstCols, BLayout::RowMajor, Conf::dstValidRows,
                         Conf::dstValidCols, SLayout::NoneBox, 512>;
    SrcTile srcTile;
    DstTile dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    uint64_t scalarQuant = quant[0];
    static constexpr ReluPreMode reluMode = Conf::applyRelu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    TEXTRACT<DstTile, SrcTile, reluMode>(dstTile, srcTile, scalarQuant, static_cast<uint16_t>(Conf::idxRow),
                                         static_cast<uint16_t>(Conf::idxCol));

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

template <typename Conf>
void runTEXTRACT_Vector(typename Conf::DT *dst, typename Conf::ST *src, uint64_t *quant)
{
    using DT = typename Conf::DT;
    using ST = typename Conf::ST;
    using GlobalDataSrc = GlobalTensor<
        ST, pto::Shape<1, 1, 1, Conf::srcValidRows, Conf::srcValidCols>,
        pto::Stride<1 * Conf::srcValidRows * Conf::srcValidCols, 1 * Conf::srcValidRows * Conf::srcValidCols,
                    Conf::srcValidRows * Conf::srcValidCols, Conf::srcValidCols, 1>>;
    using GlobalDataDst = GlobalTensor<
        DT, pto::Shape<1, 1, 1, Conf::dstValidRows, Conf::dstValidCols>,
        pto::Stride<1 * Conf::dstValidRows * Conf::dstValidCols, 1 * Conf::dstValidRows * Conf::dstValidCols,
                    Conf::dstValidRows * Conf::dstValidCols, Conf::dstValidCols, 1>>;
    using GlobalDataFp = GlobalTensor<
        uint64_t, pto::Shape<1, 1, 1, 1, Conf::dstValidCols>,
        pto::Stride<1 * Conf::dstValidCols, Conf::dstValidCols, Conf::dstValidCols, Conf::dstValidCols, 1>>;

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(dst);
    GlobalDataFp fpGlobal(quant);

    using SrcTile = Tile<TileType::Mat, ST, Conf::srcRows, Conf::srcCols, BLayout::RowMajor, Conf::srcValidRows,
                         Conf::srcValidCols, SLayout::RowMajor, 512>;
    using DstTile = Tile<TileType::Mat, DT, Conf::dstRows, Conf::dstCols, BLayout::RowMajor, Conf::dstValidRows,
                         Conf::dstValidCols, SLayout::RowMajor, 512>;
    using FbTile = Tile<TileType::Mat, uint64_t, 1, Conf::dstValidCols, BLayout::RowMajor, 1, Conf::dstValidCols,
                        SLayout::NoneBox, 512>;
    SrcTile srcTile;
    DstTile dstTile;
    FbTile fpTileLocal;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    TASSIGN(fpTileLocal, 0x20000);

    TLOAD(srcTile, srcGlobal);
    TLOAD(fpTileLocal, fpGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    static constexpr ReluPreMode reluMode = Conf::applyRelu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    TEXTRACT_FP<DstTile, SrcTile, FbTile, reluMode>(dstTile, srcTile, fpTileLocal, static_cast<uint16_t>(Conf::idxRow),
                                                    static_cast<uint16_t>(Conf::idxCol));

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

template <typename Conf>
void test_textract()
{
    using ST = typename Conf::ST;
    using DT = typename Conf::DT;

    size_t srcFileSize = Conf::srcValidRows * Conf::srcValidCols * sizeof(ST);
    size_t dstFileSize = Conf::dstValidRows * Conf::dstValidCols * sizeof(DT);
    size_t quantFileSize = Conf::isVQuant ? Conf::dstValidCols * sizeof(uint64_t) : sizeof(uint64_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    DT *dstHost, *dstDevice;
    ST *srcHost, *srcDevice;
    uint64_t *quantHost, *quantDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&quantHost), quantFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantDevice, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/quant.bin", quantFileSize, quantHost, quantFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantDevice, quantFileSize, quantHost, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    auto run = [&](uint64_t cookie) {
        pto::cpu_sim::set_task_cookie(cookie);
        if constexpr (Conf::isVQuant) {
            runTEXTRACT_Vector<Conf>(dstDevice, srcDevice, quantDevice);
        } else {
            runTEXTRACT_Scalar<Conf>(dstDevice, srcDevice, quantDevice);
        }
    };

    uint64_t applySaturation = Conf::applySaturation ? 1 : 0;
    uint64_t ctrl_bits = ((applySaturation & 0x1) << 48) & 0xFFFFFFFFFFFFFFFF;
    std::thread process(run, ctrl_bits);
    process.join();

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    std::vector<DT> golden(dstFileSize);
    std::vector<DT> devFinal(dstFileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<DT>(golden, devFinal, 0.001f);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(quantDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(quantHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case_1_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_2_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_3_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_4_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_5_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 128, 64, 128, 64, 64, 32, 64, 32, 8, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_6_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 96, 96, 96, 96, 64, 64, 64, 64, 0, 0, true, true>>();
}
TEST_F(TEXTRACTTest, case_7_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 128, 128, 128, 128, 96, 96, 96, 96, 0, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_8_int32_t_int8_t)
{
    test_textract<Params<int32_t, int8_t, 256, 64, 256, 64, 128, 32, 128, 32, 0, 0, true, true>>();
}

TEST_F(TEXTRACTTest, case_9_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_10_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_11_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_12_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_13_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 128, 64, 128, 64, 64, 32, 64, 32, 8, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_14_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 96, 96, 96, 96, 64, 64, 64, 64, 0, 0, true, true>>();
}
TEST_F(TEXTRACTTest, case_15_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 128, 128, 128, 128, 96, 96, 96, 96, 0, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_16_int32_t_uint8_t)
{
    test_textract<Params<int32_t, uint8_t, 256, 64, 256, 64, 128, 32, 128, 32, 0, 0, true, true>>();
}

TEST_F(TEXTRACTTest, case_17_int32_t_half)
{
    test_textract<Params<int32_t, half, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_18_int32_t_half)
{
    test_textract<Params<int32_t, half, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_19_int32_t_half)
{
    test_textract<Params<int32_t, half, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_20_int32_t_half)
{
    test_textract<Params<int32_t, half, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_21_int32_t_half)
{
    test_textract<Params<int32_t, half, 128, 64, 128, 64, 64, 32, 64, 32, 8, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_22_int32_t_half)
{
    test_textract<Params<int32_t, half, 96, 96, 96, 96, 64, 64, 64, 64, 0, 0, true, true>>();
}
TEST_F(TEXTRACTTest, case_23_int32_t_half)
{
    test_textract<Params<int32_t, half, 128, 128, 128, 128, 96, 96, 96, 96, 0, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_24_int32_t_half)
{
    test_textract<Params<int32_t, half, 256, 64, 256, 64, 128, 32, 128, 32, 0, 0, true, true>>();
}

TEST_F(TEXTRACTTest, case_25_float_int8_t)
{
    test_textract<Params<float, int8_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_26_float_int8_t)
{
    test_textract<Params<float, int8_t, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_27_float_int8_t)
{
    test_textract<Params<float, int8_t, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_28_float_int8_t)
{
    test_textract<Params<float, int8_t, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_29_float_int8_t)
{
    test_textract<Params<float, int8_t, 128, 64, 128, 64, 64, 32, 64, 32, 8, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_30_float_int8_t)
{
    test_textract<Params<float, int8_t, 96, 96, 96, 96, 64, 64, 64, 64, 0, 0, true, true>>();
}
TEST_F(TEXTRACTTest, case_31_float_int8_t)
{
    test_textract<Params<float, int8_t, 128, 128, 128, 128, 96, 96, 96, 96, 0, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_32_float_int8_t)
{
    test_textract<Params<float, int8_t, 256, 64, 256, 64, 128, 32, 128, 32, 0, 0, true, true>>();
}

TEST_F(TEXTRACTTest, case_33_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_34_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_35_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_36_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_37_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 128, 64, 128, 64, 64, 32, 64, 32, 8, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_38_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 96, 96, 96, 96, 64, 64, 64, 64, 0, 0, true, true>>();
}
TEST_F(TEXTRACTTest, case_39_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 128, 128, 128, 128, 96, 96, 96, 96, 0, 0, true, false>>();
}
TEST_F(TEXTRACTTest, case_40_float_uint8_t)
{
    test_textract<Params<float, uint8_t, 256, 64, 256, 64, 128, 32, 128, 32, 0, 0, true, true>>();
}

TEST_F(TEXTRACTTest, case_41_float_half)
{
    test_textract<Params<float, half, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_42_float_half)
{
    test_textract<Params<float, half, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_43_float_half)
{
    test_textract<Params<float, half, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false, true>>();
}
TEST_F(TEXTRACTTest, case_44_float_half)
{
    test_textract<Params<float, half, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true, true>>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TEXTRACTTest, case_45_float_bfloat16_t)
{
    test_textract<Params<float, bfloat16_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_46_float_bfloat16_t)
{
    test_textract<Params<float, bfloat16_t, 128, 64, 128, 64, 96, 32, 96, 32, 0, 0, false, true>>();
}
TEST_F(TEXTRACTTest, case_47_float_bfloat16_t)
{
    test_textract<Params<float, bfloat16_t, 128, 128, 128, 128, 64, 64, 64, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_48_float_bfloat16_t)
{
    test_textract<Params<float, bfloat16_t, 256, 128, 256, 128, 128, 64, 128, 64, 0, 0, false, true>>();
}
#endif

TEST_F(TEXTRACTTest, case_49_int32_t_int16_t)
{
    test_textract<Params<int32_t, int16_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}
TEST_F(TEXTRACTTest, case_50_int32_t_int16_t)
{
    test_textract<Params<int32_t, int16_t, 128, 64, 128, 64, 128, 64, 128, 64, 0, 0, false, false>>();
}