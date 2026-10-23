/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

template <int32_t Key>
void launchTMOVAcc2MatNZ2NZ(uint8_t* out, uint8_t* a, uint8_t* b, uint8_t* initial, void* stream);
template <int32_t Key>
void launchTMOVAcc2MatSCQuantNz(uint8_t* out, uint8_t* a, uint8_t* b, uint8_t* initial, void* stream);
template <int32_t Key>
void launchTMOVAcc2MatFBQuantNz(uint8_t* out, uint8_t* a, uint8_t* b, uint8_t* quant, uint8_t* initial, void* stream);

namespace {
enum class MoveKind { Plain, Scalar, Vector };

bool ReadBinary(const std::string& path, std::vector<uint8_t>& buffer)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0) {
        return false;
    }
    buffer.resize(static_cast<size_t>(input.tellg()));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    return bool(input);
}

template <MoveKind Kind, int32_t Key>
void RunCase()
{
#if !defined(CPU_SIM_BFLOAT_ENABLED)
    if constexpr (Kind == MoveKind::Plain && (Key == 2 || Key == 4)) {
        GTEST_SKIP() << "BF16 cases require tests/run_cpu.py --enable-bf16";
    }
#endif
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    const std::string directory = std::string("../") + info->test_suite_name() + "." + info->name() + "/";
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::vector<uint8_t> golden;
    std::vector<uint8_t> quant;
    std::vector<uint8_t> initial;
    ASSERT_TRUE(ReadBinary(directory + "x1_gm.bin", a)) << directory + "x1_gm.bin";
    ASSERT_TRUE(ReadBinary(directory + "x2_gm.bin", b)) << directory + "x2_gm.bin";
    ASSERT_TRUE(ReadBinary(directory + "golden.bin", golden)) << directory + "golden.bin";
    constexpr bool insert = (Kind == MoveKind::Plain && (Key == 6 || Key == 9 || Key == 12)) ||
                            (Kind == MoveKind::Scalar && Key == 12) ||
                            (Kind == MoveKind::Vector && (Key == 12 || Key == 15));
    if constexpr (insert) {
        ASSERT_TRUE(ReadBinary(directory + "dst.bin", initial)) << directory + "dst.bin";
        ASSERT_EQ(initial.size(), golden.size());
    }
    if constexpr (Kind == MoveKind::Vector) {
        ASSERT_TRUE(ReadBinary(directory + "quant_gm.bin", quant)) << directory + "quant_gm.bin";
    }
    std::vector<uint8_t> output(golden.size(), 0xA5);
    if constexpr (Kind == MoveKind::Plain) {
        launchTMOVAcc2MatNZ2NZ<Key>(output.data(), a.data(), b.data(), initial.data(), nullptr);
    } else if constexpr (Kind == MoveKind::Scalar) {
        launchTMOVAcc2MatSCQuantNz<Key>(output.data(), a.data(), b.data(), initial.data(), nullptr);
    } else {
        launchTMOVAcc2MatFBQuantNz<Key>(output.data(), a.data(), b.data(), quant.data(), initial.data(), nullptr);
    }
    const std::string actualPath = directory + "actual.bin";
    std::ofstream actual(actualPath, std::ios::binary);
    ASSERT_TRUE(actual.is_open()) << actualPath;
    actual.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(output.size()));
    ASSERT_TRUE(bool(actual)) << actualPath;
    size_t mismatch = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        mismatch += output[i] != golden[i];
    }
    std::cout << "bytes=" << output.size() << " mismatch_bytes=" << mismatch << std::endl;
    EXPECT_EQ(mismatch, 0U) << "Output differs from golden.bin; see " << actualPath;
}

class TMOVTest : public testing::Test {};

TEST_F(TMOVTest, case_nz2nz_1) { RunCase<MoveKind::Plain, 1>(); }

TEST_F(TMOVTest, case_nz2nz_2) { RunCase<MoveKind::Plain, 2>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_3) { RunCase<MoveKind::Scalar, 1>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_4) { RunCase<MoveKind::Vector, 1>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_5) { RunCase<MoveKind::Scalar, 2>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_6) { RunCase<MoveKind::Vector, 2>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_7) { RunCase<MoveKind::Scalar, 3>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_8) { RunCase<MoveKind::Vector, 3>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_9) { RunCase<MoveKind::Scalar, 4>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_10) { RunCase<MoveKind::Vector, 4>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_11) { RunCase<MoveKind::Scalar, 5>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_12) { RunCase<MoveKind::Vector, 5>(); }

TEST_F(TMOVTest, case_nz2nz_21) { RunCase<MoveKind::Plain, 3>(); }

TEST_F(TMOVTest, case_nz2nz_22) { RunCase<MoveKind::Plain, 4>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_23) { RunCase<MoveKind::Scalar, 6>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_24) { RunCase<MoveKind::Vector, 6>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_25) { RunCase<MoveKind::Scalar, 7>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_26) { RunCase<MoveKind::Vector, 7>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_27) { RunCase<MoveKind::Scalar, 8>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_28) { RunCase<MoveKind::Vector, 8>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_29) { RunCase<MoveKind::Scalar, 9>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_30) { RunCase<MoveKind::Vector, 9>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_31) { RunCase<MoveKind::Scalar, 10>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_32) { RunCase<MoveKind::Vector, 10>(); }

TEST_F(TMOVTest, case_nz2nz_extract) { RunCase<MoveKind::Plain, 5>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_extract) { RunCase<MoveKind::Scalar, 11>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_extract) { RunCase<MoveKind::Vector, 11>(); }

TEST_F(TMOVTest, case_nz2nz_insert) { RunCase<MoveKind::Plain, 6>(); }

TEST_F(TMOVTest, case_nz2nz_uf_final) { RunCase<MoveKind::Plain, 7>(); }

TEST_F(TMOVTest, case_nz2nz_extract_uf_final) { RunCase<MoveKind::Plain, 8>(); }

TEST_F(TMOVTest, case_nz2nz_insert_uf_final) { RunCase<MoveKind::Plain, 9>(); }

TEST_F(TMOVTest, case_nz2nz_uf_multi_drain) { RunCase<MoveKind::Plain, 10>(); }

TEST_F(TMOVTest, case_nz2nz_extract_uf_multi_drain) { RunCase<MoveKind::Plain, 11>(); }

TEST_F(TMOVTest, case_nz2nz_insert_uf_multi_drain) { RunCase<MoveKind::Plain, 12>(); }

TEST_F(TMOVTest, case_nz2nz_sc_quant_insert) { RunCase<MoveKind::Scalar, 12>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_insert) { RunCase<MoveKind::Vector, 12>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_uf_final) { RunCase<MoveKind::Vector, 13>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_extract_uf_final) { RunCase<MoveKind::Vector, 14>(); }

TEST_F(TMOVTest, case_nz2nz_fb_quant_insert_uf_final) { RunCase<MoveKind::Vector, 15>(); }

} // namespace
