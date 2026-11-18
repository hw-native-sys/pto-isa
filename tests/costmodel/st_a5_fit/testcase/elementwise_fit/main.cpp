/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <gtest/gtest.h>

#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

struct TabsFitCase {
    DType dtype;
    int64_t rows;
    int64_t cols;
    int64_t validRows;
    int64_t validCols;
    double profilingCycles;
};

constexpr TabsFitCase kTabsCases[] = {
    {DType::Float, 1, 64, 1, 64, 36.002},        {DType::Float, 1, 640, 1, 640, 45.002},
    {DType::Float, 1, 1280, 1, 1280, 56.001},    {DType::Float, 1, 63, 1, 63, 36.002},
    {DType::Float, 1, 1279, 1, 1279, 56.001},    {DType::Float, 2, 96, 2, 64, 37.003},
    {DType::Float, 10, 352, 10, 320, 108.032},   {DType::Float, 20, 1312, 20, 1280, 566.033},
    {DType::Float, 3, 224, 3, 192, 44.033},      {DType::Float, 3, 672, 3, 640, 66.004},
    {DType::Float, 2, 96, 2, 63, 37.003},        {DType::Float, 10, 352, 10, 319, 108.005},
    {DType::Float, 20, 1312, 20, 1279, 566.084}, {DType::Float, 3, 224, 3, 191, 44.128},
    {DType::Float, 3, 672, 3, 639, 66.074},      {DType::Half, 1, 128, 1, 128, 36.002},
    {DType::Half, 1, 1280, 1, 1280, 45.002},     {DType::Half, 1, 2560, 1, 2560, 56.001},
    {DType::Half, 1, 127, 1, 127, 36.002},       {DType::Half, 1, 2559, 1, 2559, 56.001},
    {DType::Half, 2, 144, 2, 128, 37.003},       {DType::Half, 10, 656, 10, 640, 108.005},
    {DType::Half, 20, 2576, 20, 2560, 692.031},  {DType::Half, 3, 400, 3, 384, 44.037},
    {DType::Half, 3, 1296, 3, 1280, 67.004},     {DType::Half, 2, 144, 2, 127, 38.003},
    {DType::Half, 10, 656, 10, 639, 133.005},    {DType::Half, 20, 2576, 20, 2559, 751.002},
    {DType::Half, 3, 400, 3, 383, 49.076},       {DType::Half, 3, 1296, 3, 1279, 75.005},
    {DType::Half, 4, 144, 4, 127, 42.005},       {DType::Half, 20, 144, 20, 127, 138.005},
    {DType::Half, 4, 272, 4, 255, 48.005},       {DType::Half, 17, 272, 17, 255, 137.006},
    {DType::Half, 7, 528, 7, 511, 86.005},       {DType::Half, 8, 528, 8, 511, 96.005},
    {DType::Half, 20, 528, 20, 511, 208.006},    {DType::Half, 13, 912, 13, 895, 218.005},
    {DType::Half, 17, 1552, 17, 1535, 398.005},  {DType::Half, 4, 2064, 4, 2047, 132.003},
};

TEST(A5ElementwiseFit, AbsHardwareProfiles)
{
    constexpr double kRequiredPrecisionWithHelperTolerance = 0.95;
    for (const auto& testCase : kTabsCases) {
        CostModelInput input{
            .op = PtoOpcode::TABS,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.validRows,
            .valid_cols = testCase.validCols,
        };
        pto::test::ExpectA5CycleNear("TABS", input, testCase.profilingCycles, kRequiredPrecisionWithHelperTolerance);
    }
}

struct ElementwiseFitCase {
    PtoOpcode op;
    const char* name;
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr ElementwiseFitCase kElementwiseCases[] = {
    {PtoOpcode::TDIV, "TDIV", DType::Half, 1, 16, 53.0},
    {PtoOpcode::TDIV, "TDIV", DType::Half, 1, 1024, 78.0},
    {PtoOpcode::TDIV, "TDIV", DType::Half, 24, 256, 238.0},
    {PtoOpcode::TMAX, "TMAX", DType::Half, 1, 16, 37.0},
    {PtoOpcode::TMAX, "TMAX", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TMIN, "TMIN", DType::Half, 1, 1024, 44.0},
    {PtoOpcode::TMIN, "TMIN", DType::Half, 48, 64, 306.0},
    {PtoOpcode::TADDS, "TADDS", DType::Half, 1, 16, 37.0},
    {PtoOpcode::TADDS, "TADDS", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TSUBS, "TSUBS", DType::Half, 1, 1024, 44.0},
    {PtoOpcode::TSUBS, "TSUBS", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TMULS, "TMULS", DType::Half, 1, 16, 38.0},
    {PtoOpcode::TMULS, "TMULS", DType::Half, 48, 64, 306.0},
    {PtoOpcode::TMAXS, "TMAXS", DType::Half, 1, 1024, 43.0},
    {PtoOpcode::TMAXS, "TMAXS", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TRELU, "TRELU", DType::Half, 1, 16, 36.0},
    {PtoOpcode::TRELU, "TRELU", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TLRELU, "TLRELU", DType::Half, 1, 1024, 45.0},
    {PtoOpcode::TLRELU, "TLRELU", DType::Half, 48, 64, 306.9},
    {PtoOpcode::TLOG, "TLOG", DType::Half, 1, 16, 53.0},
    {PtoOpcode::TLOG, "TLOG", DType::Half, 24, 256, 238.0},
    {PtoOpcode::TFMOD, "TFMOD", DType::Float, 1, 1024, 110.0},
    {PtoOpcode::TFMOD, "TFMOD", DType::Float, 24, 256, 379.0},
    {PtoOpcode::TFMODS, "TFMODS", DType::Float, 1, 16, 60.0},
    {PtoOpcode::TFMODS, "TFMODS", DType::Float, 64, 128, 606.0},
    {PtoOpcode::TREM, "TREM", DType::Float, 1, 64, 71.0},
    {PtoOpcode::TREM, "TREM", DType::Float, 16, 256, 467.0},
    {PtoOpcode::TREMS, "TREMS", DType::Float, 1, 1024, 111.0},
    {PtoOpcode::TREMS, "TREMS", DType::Float, 48, 64, 354.0},
    {PtoOpcode::TPOW, "TPOW", DType::Float, 1, 16, 90.0},
    {PtoOpcode::TPOW, "TPOW", DType::Float, 16, 256, 1877.0},
    {PtoOpcode::TPOWS, "TPOWS", DType::Float, 16, 16, 166.5},
    {PtoOpcode::TPOWS, "TPOWS", DType::Float, 64, 128, 3912.5},
    {PtoOpcode::TMADD, "TMADD", DType::Half, 1, 1024, 57.0},
    {PtoOpcode::TMADD, "TMADD", DType::Half, 24, 256, 234.0},
    {PtoOpcode::TMULA, "TMULA", DType::Half, 1, 16, 42.0},
    {PtoOpcode::TMULA, "TMULA", DType::Half, 48, 64, 354.0},
    {PtoOpcode::TPRELU, "TPRELU", DType::Half, 1, 512, 42.0},
    {PtoOpcode::TPRELU, "TPRELU", DType::Half, 128, 128, 786.0},
    {PtoOpcode::TAXPY, "TAXPY", DType::Half, 16, 128, 114.0},
};

TEST(A5ElementwiseFit, HardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kElementwiseCases) {
        CostModelInput input{
            .op = testCase.op,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5ElementwiseFit, Fp16SingleRepeatUsesMeasuredStartupRegion)
{
    struct SmallRepeatCase {
        PtoOpcode op;
        double cycles;
    };
    constexpr SmallRepeatCase kCases[] = {
        {PtoOpcode::TADD, 38.0}, {PtoOpcode::TSUB, 38.0},  {PtoOpcode::TMUL, 39.0},
        {PtoOpcode::TNEG, 38.0}, {PtoOpcode::TMINS, 36.0},
    };
    for (const auto& testCase : kCases) {
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(
            CostModelInput{
                .op = testCase.op,
                .dtype = DType::Half,
                .rows = 1,
                .cols = 16,
                .arch = CostModelArch::A5,
                .valid_rows = 1,
                .valid_cols = 16,
            },
            result));
        EXPECT_EQ(result.cycles, testCase.cycles);
    }
}

TEST(A5ElementwiseFit, Fp16LongSingleRowAndTwoDimensionalPaths)
{
    struct RegressionCase {
        PtoOpcode op;
        const char* name;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr RegressionCase kCases[] = {
        {PtoOpcode::TADD, "TADD/long-row", 1, 43648, 737.0}, {PtoOpcode::TMUL, "TMUL/long-row", 1, 43648, 737.0},
        {PtoOpcode::TMAX, "TMAX/long-row", 1, 43648, 736.0}, {PtoOpcode::TMIN, "TMIN/long-row", 1, 43648, 736.0},
        {PtoOpcode::TABS, "TABS/long-row", 1, 65472, 548.0}, {PtoOpcode::TABS, "TABS/2D", 128, 128, 786.0},
        {PtoOpcode::TNEG, "TNEG/2D", 16, 256, 130.0},        {PtoOpcode::TNEG, "TNEG/narrow-2D", 4092, 16, 24570.0},
    };
    constexpr double kRequiredPrecision = 0.95;
    for (const auto& testCase : kCases) {
        const CostModelInput input{
            .op = testCase.op,
            .dtype = DType::Half,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
            .a5_shape_path = testCase.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5ElementwiseFit, FloatPartMinMaxMissingShapeKeys)
{
    struct MeasuredCase {
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr MeasuredCase kMeasured[] = {
        {1, 8, 37.0},     {1, 144, 39.0},  {2, 8, 39.0},     {16, 16, 114.0}, {16, 64, 114.0},
        {16, 256, 162.0}, {32, 32, 210.0}, {64, 128, 466.0}, {148, 8, 906.0},
    };
    for (auto op : {PtoOpcode::TPARTMAX, PtoOpcode::TPARTMIN}) {
        for (const auto& testCase : kMeasured) {
            const CostModelInput input{
                .op = op,
                .dtype = DType::Float,
                .rows = testCase.rows,
                .cols = testCase.cols,
                .arch = CostModelArch::A5,
                .a5_shape_path = testCase.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D,
            };
            pto::test::ExpectA5CycleNear("TPARTMAX/MIN/fp32", input, testCase.cycles, 0.90);
        }
    }
}

TEST(A5ValidatedProfiles, IdentifiableCrossValidatedHardwareProfiles)
{
    struct Profile {
        PtoOpcode op;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {PtoOpcode::TPARTADD, DType::Float, 148, 8, 906.0},   {PtoOpcode::TPARTADD, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TPARTADD, DType::Float, 16, 16, 114.0},   {PtoOpcode::TPARTADD, DType::Float, 16, 256, 162.0},
        {PtoOpcode::TPARTADD, DType::Float, 16, 64, 114.0},   {PtoOpcode::TPARTADD, DType::Float, 1, 128, 39.0},
        {PtoOpcode::TPARTADD, DType::Float, 1, 144, 40.0},    {PtoOpcode::TPARTADD, DType::Float, 1, 8, 38.0},
        {PtoOpcode::TPARTADD, DType::Float, 2, 8, 40.0},      {PtoOpcode::TPARTADD, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TPARTADD, DType::Float, 4, 128, 46.0},    {PtoOpcode::TPARTADD, DType::Float, 4, 256, 54.0},
        {PtoOpcode::TPARTADD, DType::Float, 64, 128, 466.0},  {PtoOpcode::TPARTADD, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TPARTADD, DType::Float, 7, 96, 67.0},     {PtoOpcode::TPARTADD, DType::Int16, 16, 128, 114.0},
        {PtoOpcode::TPARTADD, DType::Int16, 16, 256, 130.0},  {PtoOpcode::TPARTADD, DType::Int16, 16, 64, 114.0},
        {PtoOpcode::TPARTADD, DType::Int16, 1, 128, 38.0},    {PtoOpcode::TPARTADD, DType::Int16, 1, 16, 38.0},
        {PtoOpcode::TPARTADD, DType::Int16, 1, 208, 39.0},    {PtoOpcode::TPARTADD, DType::Int16, 209, 16, 1272.0},
        {PtoOpcode::TPARTADD, DType::Int16, 2, 16, 40.0},     {PtoOpcode::TPARTADD, DType::Int16, 4, 128, 42.0},
        {PtoOpcode::TPARTADD, DType::Int16, 4, 256, 46.0},    {PtoOpcode::TPARTADD, DType::Int16, 7, 96, 60.0},
        {PtoOpcode::TPARTADD, DType::Int32, 148, 8, 906.0},   {PtoOpcode::TPARTADD, DType::Int32, 16, 128, 130.0},
        {PtoOpcode::TPARTADD, DType::Int32, 16, 256, 162.0},  {PtoOpcode::TPARTADD, DType::Int32, 16, 64, 114.0},
        {PtoOpcode::TPARTADD, DType::Int32, 1, 128, 39.0},    {PtoOpcode::TPARTADD, DType::Int32, 1, 144, 40.0},
        {PtoOpcode::TPARTADD, DType::Int32, 1, 8, 38.0},      {PtoOpcode::TPARTADD, DType::Int32, 2, 8, 40.0},
        {PtoOpcode::TPARTADD, DType::Int32, 4, 128, 46.0},    {PtoOpcode::TPARTADD, DType::Int32, 4, 256, 54.0},
        {PtoOpcode::TPARTADD, DType::Int32, 7, 96, 67.0},     {PtoOpcode::TPARTMAX, DType::Float, 1, 128, 38.0},
        {PtoOpcode::TPARTMAX, DType::Int16, 16, 128, 114.0},  {PtoOpcode::TPARTMAX, DType::Int16, 16, 256, 130.0},
        {PtoOpcode::TPARTMAX, DType::Int16, 16, 64, 114.0},   {PtoOpcode::TPARTMAX, DType::Int16, 1, 128, 37.0},
        {PtoOpcode::TPARTMAX, DType::Int16, 1, 16, 37.0},     {PtoOpcode::TPARTMAX, DType::Int16, 1, 208, 38.0},
        {PtoOpcode::TPARTMAX, DType::Int16, 209, 16, 1272.0}, {PtoOpcode::TPARTMAX, DType::Int16, 2, 16, 39.0},
        {PtoOpcode::TPARTMAX, DType::Int16, 4, 128, 42.0},    {PtoOpcode::TPARTMAX, DType::Int16, 4, 256, 46.0},
        {PtoOpcode::TPARTMAX, DType::Int16, 7, 96, 60.0},     {PtoOpcode::TPARTMAX, DType::Int32, 148, 8, 906.0},
        {PtoOpcode::TPARTMAX, DType::Int32, 16, 128, 130.0},  {PtoOpcode::TPARTMAX, DType::Int32, 16, 256, 162.0},
        {PtoOpcode::TPARTMAX, DType::Int32, 16, 64, 114.0},   {PtoOpcode::TPARTMAX, DType::Int32, 1, 128, 38.0},
        {PtoOpcode::TPARTMAX, DType::Int32, 1, 144, 39.0},    {PtoOpcode::TPARTMAX, DType::Int32, 1, 8, 37.0},
        {PtoOpcode::TPARTMAX, DType::Int32, 2, 8, 39.0},      {PtoOpcode::TPARTMAX, DType::Int32, 4, 128, 46.0},
        {PtoOpcode::TPARTMAX, DType::Int32, 4, 256, 54.0},    {PtoOpcode::TPARTMAX, DType::Int32, 7, 96, 67.0},
        {PtoOpcode::TPARTMAX, DType::Int8, 16, 128, 114.0},   {PtoOpcode::TPARTMAX, DType::Int8, 16, 256, 114.0},
        {PtoOpcode::TPARTMAX, DType::Int8, 16, 64, 114.0},    {PtoOpcode::TPARTMAX, DType::Int8, 1, 128, 37.0},
        {PtoOpcode::TPARTMAX, DType::Int8, 1, 288, 38.0},     {PtoOpcode::TPARTMAX, DType::Int8, 1, 32, 37.0},
        {PtoOpcode::TPARTMAX, DType::Int8, 295, 32, 1788.0},  {PtoOpcode::TPARTMAX, DType::Int8, 2, 32, 39.0},
        {PtoOpcode::TPARTMAX, DType::Int8, 4, 128, 42.0},     {PtoOpcode::TPARTMAX, DType::Int8, 4, 256, 42.0},
        {PtoOpcode::TPARTMAX, DType::Int8, 7, 96, 60.0},      {PtoOpcode::TPARTMAX, DType::Uint8, 16, 128, 114.0},
        {PtoOpcode::TPARTMAX, DType::Uint8, 16, 256, 114.0},  {PtoOpcode::TPARTMAX, DType::Uint8, 16, 64, 114.0},
        {PtoOpcode::TPARTMAX, DType::Uint8, 1, 128, 37.0},    {PtoOpcode::TPARTMAX, DType::Uint8, 1, 288, 38.0},
        {PtoOpcode::TPARTMAX, DType::Uint8, 1, 32, 37.0},     {PtoOpcode::TPARTMAX, DType::Uint8, 295, 32, 1788.0},
        {PtoOpcode::TPARTMAX, DType::Uint8, 2, 32, 39.0},     {PtoOpcode::TPARTMAX, DType::Uint8, 4, 128, 42.0},
        {PtoOpcode::TPARTMAX, DType::Uint8, 4, 256, 42.0},    {PtoOpcode::TPARTMAX, DType::Uint8, 7, 96, 60.0},
        {PtoOpcode::TPARTMIN, DType::Float, 1, 128, 38.0},    {PtoOpcode::TPARTMIN, DType::Int16, 16, 128, 114.0},
        {PtoOpcode::TPARTMIN, DType::Int16, 16, 256, 130.0},  {PtoOpcode::TPARTMIN, DType::Int16, 16, 64, 114.0},
        {PtoOpcode::TPARTMIN, DType::Int16, 1, 128, 37.0},    {PtoOpcode::TPARTMIN, DType::Int16, 1, 16, 37.0},
        {PtoOpcode::TPARTMIN, DType::Int16, 1, 208, 38.0},    {PtoOpcode::TPARTMIN, DType::Int16, 209, 16, 1272.0},
        {PtoOpcode::TPARTMIN, DType::Int16, 2, 16, 39.0},     {PtoOpcode::TPARTMIN, DType::Int16, 4, 128, 42.0},
        {PtoOpcode::TPARTMIN, DType::Int16, 4, 256, 46.0},    {PtoOpcode::TPARTMIN, DType::Int16, 7, 96, 60.0},
        {PtoOpcode::TPARTMIN, DType::Int32, 148, 8, 906.0},   {PtoOpcode::TPARTMIN, DType::Int32, 16, 128, 130.0},
        {PtoOpcode::TPARTMIN, DType::Int32, 16, 256, 162.0},  {PtoOpcode::TPARTMIN, DType::Int32, 16, 64, 114.0},
        {PtoOpcode::TPARTMIN, DType::Int32, 1, 128, 38.0},    {PtoOpcode::TPARTMIN, DType::Int32, 1, 144, 39.0},
        {PtoOpcode::TPARTMIN, DType::Int32, 1, 8, 37.0},      {PtoOpcode::TPARTMIN, DType::Int32, 2, 8, 39.0},
        {PtoOpcode::TPARTMIN, DType::Int32, 4, 128, 46.0},    {PtoOpcode::TPARTMIN, DType::Int32, 4, 256, 54.0},
        {PtoOpcode::TPARTMIN, DType::Int32, 7, 96, 67.0},     {PtoOpcode::TPARTMIN, DType::Int8, 16, 128, 114.0},
        {PtoOpcode::TPARTMIN, DType::Int8, 16, 256, 114.0},   {PtoOpcode::TPARTMIN, DType::Int8, 16, 64, 114.0},
        {PtoOpcode::TPARTMIN, DType::Int8, 1, 128, 37.0},     {PtoOpcode::TPARTMIN, DType::Int8, 1, 288, 38.0},
        {PtoOpcode::TPARTMIN, DType::Int8, 1, 32, 37.0},      {PtoOpcode::TPARTMIN, DType::Int8, 295, 32, 1788.0},
        {PtoOpcode::TPARTMIN, DType::Int8, 2, 32, 39.0},      {PtoOpcode::TPARTMIN, DType::Int8, 4, 128, 42.0},
        {PtoOpcode::TPARTMIN, DType::Int8, 4, 256, 42.0},     {PtoOpcode::TPARTMIN, DType::Int8, 7, 96, 60.0},
        {PtoOpcode::TPARTMIN, DType::Uint8, 16, 128, 114.0},  {PtoOpcode::TPARTMIN, DType::Uint8, 16, 256, 114.0},
        {PtoOpcode::TPARTMIN, DType::Uint8, 16, 64, 114.0},   {PtoOpcode::TPARTMIN, DType::Uint8, 1, 128, 37.0},
        {PtoOpcode::TPARTMIN, DType::Uint8, 1, 288, 38.0},    {PtoOpcode::TPARTMIN, DType::Uint8, 1, 32, 37.0},
        {PtoOpcode::TPARTMIN, DType::Uint8, 295, 32, 1788.0}, {PtoOpcode::TPARTMIN, DType::Uint8, 2, 32, 39.0},
        {PtoOpcode::TPARTMIN, DType::Uint8, 4, 128, 42.0},    {PtoOpcode::TPARTMIN, DType::Uint8, 4, 256, 42.0},
        {PtoOpcode::TPARTMIN, DType::Uint8, 7, 96, 60.0},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.op = profile.op;
        input.dtype = profile.dtype;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.arch = CostModelArch::A5;
        input.a5_shape_path = profile.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.op) << " " << profile.rows << "x" << profile.cols;
    }
}

TEST(A5DenseSingleRowFit, MeasuredProfiles)
{
    struct Profile {
        PtoOpcode op;
        DType dtype;
        int64_t cols;
        double cycles;
    };
    const Profile profiles[] = {
        {PtoOpcode::TAND, DType::Int16, 144, 38.0},
        {PtoOpcode::TAND, DType::Int16, 112, 37.0},
        {PtoOpcode::TAND, DType::Int16, 256, 38.0},
        {PtoOpcode::TAND, DType::Int16, 1024, 44.0},
        {PtoOpcode::TAND, DType::Int16, 384, 39.0},
        {PtoOpcode::TAND, DType::Int16, 16, 37.0},
        {PtoOpcode::TAND, DType::Int16, 128, 37.0},
        {PtoOpcode::TAND, DType::Uint16, 256, 38.0},
        {PtoOpcode::TAND, DType::Uint16, 144, 38.0},
        {PtoOpcode::TAND, DType::Uint16, 1024, 44.0},
        {PtoOpcode::TAND, DType::Uint16, 112, 37.0},
        {PtoOpcode::TAND, DType::Uint16, 384, 39.0},
        {PtoOpcode::TAND, DType::Uint16, 128, 37.0},
        {PtoOpcode::TAND, DType::Uint16, 16, 37.0},
        {PtoOpcode::TANDS, DType::Uint16, 112, 36.0},
        {PtoOpcode::TANDS, DType::Uint16, 144, 37.0},
        {PtoOpcode::TANDS, DType::Uint16, 384, 38.0},
        {PtoOpcode::TANDS, DType::Uint16, 256, 37.0},
        {PtoOpcode::TANDS, DType::Uint16, 1024, 43.0},
        {PtoOpcode::TANDS, DType::Uint16, 128, 36.0},
        {PtoOpcode::TANDS, DType::Uint16, 16, 36.0},
        {PtoOpcode::TCMP, DType::BFloat16, 16, 37.0},
        {PtoOpcode::TCMP, DType::BFloat16, 112, 37.0},
        {PtoOpcode::TCMP, DType::BFloat16, 1024, 44.0},
        {PtoOpcode::TCMP, DType::BFloat16, 256, 38.0},
        {PtoOpcode::TCMP, DType::BFloat16, 144, 38.0},
        {PtoOpcode::TCMP, DType::BFloat16, 384, 39.0},
        {PtoOpcode::TCMP, DType::BFloat16, 32, 37.0},
        {PtoOpcode::TCMP, DType::BFloat16, 128, 37.0},
        {PtoOpcode::TCMP, DType::BFloat16, 224, 38.0},
        {PtoOpcode::TCMP, DType::Float, 72, 46.0},
        {PtoOpcode::TCMP, DType::Float, 8, 46.0},
        {PtoOpcode::TCMP, DType::Float, 192, 48.0},
        {PtoOpcode::TCMP, DType::Float, 56, 46.0},
        {PtoOpcode::TCMP, DType::Float, 512, 56.0},
        {PtoOpcode::TCMP, DType::Float, 64, 46.0},
        {PtoOpcode::TCMP, DType::Float, 32, 46.0},
        {PtoOpcode::TCMP, DType::Float, 128, 46.0},
        {PtoOpcode::TCMP, DType::Int32, 8, 46.0},
        {PtoOpcode::TCMP, DType::Int32, 64, 46.0},
        {PtoOpcode::TCMP, DType::Int32, 192, 50.0},
        {PtoOpcode::TCMP, DType::Int32, 56, 46.0},
        {PtoOpcode::TCMP, DType::Int32, 72, 46.0},
        {PtoOpcode::TCMP, DType::Int32, 512, 56.0},
        {PtoOpcode::TCMP, DType::Int32, 32, 46.0},
        {PtoOpcode::TCMP, DType::Int32, 128, 46.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 16, 36.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 112, 36.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 144, 37.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 384, 38.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 1024, 43.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 256, 37.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 32, 36.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 288, 38.0},
        {PtoOpcode::TCMPS, DType::BFloat16, 128, 36.0},
        {PtoOpcode::TCMPS, DType::Float, 64, 44.0},
        {PtoOpcode::TCMPS, DType::Float, 72, 44.0},
        {PtoOpcode::TCMPS, DType::Float, 8, 44.0},
        {PtoOpcode::TCMPS, DType::Float, 192, 46.0},
        {PtoOpcode::TCMPS, DType::Float, 56, 44.0},
        {PtoOpcode::TCMPS, DType::Float, 512, 50.0},
        {PtoOpcode::TCMPS, DType::Float, 224, 46.0},
        {PtoOpcode::TCMPS, DType::Float, 32, 44.0},
        {PtoOpcode::TCMPS, DType::Float, 128, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 8, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 64, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 192, 46.0},
        {PtoOpcode::TCMPS, DType::Int32, 56, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 72, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 512, 50.0},
        {PtoOpcode::TCMPS, DType::Int32, 32, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 128, 44.0},
        {PtoOpcode::TCMPS, DType::Int32, 224, 46.0},
        {PtoOpcode::TCMPS, DType::Int8, 288, 37.0},
        {PtoOpcode::TCMPS, DType::Int8, 512, 37.0},
        {PtoOpcode::TCMPS, DType::Int8, 224, 36.0},
        {PtoOpcode::TCMPS, DType::Int8, 768, 38.0},
        {PtoOpcode::TCMPS, DType::Int8, 256, 36.0},
        {PtoOpcode::TCMPS, DType::Int8, 2048, 43.0},
        {PtoOpcode::TCMPS, DType::Int8, 352, 37.0},
        {PtoOpcode::TCMPS, DType::Int8, 32, 36.0},
        {PtoOpcode::TCMPS, DType::Int8, 128, 36.0},
        {PtoOpcode::TCMPS, DType::Uint16, 256, 37.0},
        {PtoOpcode::TCMPS, DType::Uint16, 16, 36.0},
        {PtoOpcode::TCMPS, DType::Uint16, 384, 38.0},
        {PtoOpcode::TCMPS, DType::Uint16, 112, 36.0},
        {PtoOpcode::TCMPS, DType::Uint16, 1024, 43.0},
        {PtoOpcode::TCMPS, DType::Uint16, 144, 37.0},
        {PtoOpcode::TCMPS, DType::Uint16, 288, 38.0},
        {PtoOpcode::TCMPS, DType::Uint16, 32, 36.0},
        {PtoOpcode::TCMPS, DType::Uint16, 128, 36.0},
        {PtoOpcode::TCMPS, DType::Uint32, 8, 44.0},
        {PtoOpcode::TCMPS, DType::Uint32, 192, 46.0},
        {PtoOpcode::TCMPS, DType::Uint32, 56, 44.0},
        {PtoOpcode::TCMPS, DType::Uint32, 512, 50.0},
        {PtoOpcode::TCMPS, DType::Uint32, 64, 44.0},
        {PtoOpcode::TCMPS, DType::Uint32, 72, 44.0},
        {PtoOpcode::TCMPS, DType::Uint32, 32, 44.0},
        {PtoOpcode::TCMPS, DType::Uint32, 128, 44.0},
        {PtoOpcode::TCMPS, DType::Uint32, 224, 46.0},
        {PtoOpcode::TCMPS, DType::Uint8, 768, 38.0},
        {PtoOpcode::TCMPS, DType::Uint8, 2048, 43.0},
        {PtoOpcode::TCMPS, DType::Uint8, 224, 36.0},
        {PtoOpcode::TCMPS, DType::Uint8, 256, 36.0},
        {PtoOpcode::TCMPS, DType::Uint8, 288, 37.0},
        {PtoOpcode::TCMPS, DType::Uint8, 512, 37.0},
        {PtoOpcode::TCMPS, DType::Uint8, 32, 36.0},
        {PtoOpcode::TCMPS, DType::Uint8, 128, 36.0},
        {PtoOpcode::TCMPS, DType::Uint8, 352, 37.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 8, 39.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 56, 39.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 64, 39.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 192, 41.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 512, 50.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 72, 40.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 128, 40.0},
        {PtoOpcode::TEXPANDS, DType::Float, 56, 34.0},
        {PtoOpcode::TEXPANDS, DType::Float, 64, 34.0},
        {PtoOpcode::TEXPANDS, DType::Float, 192, 36.0},
        {PtoOpcode::TEXPANDS, DType::Float, 72, 35.0},
        {PtoOpcode::TEXPANDS, DType::Float, 512, 41.0},
        {PtoOpcode::TEXPANDS, DType::Float, 128, 35.0},
        {PtoOpcode::TEXPANDS, DType::Float, 8, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 144, 35.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 1024, 41.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 384, 36.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 112, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 256, 35.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 16, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 128, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 56, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 64, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 72, 35.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 192, 36.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 512, 41.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 8, 34.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 128, 35.0},
        {PtoOpcode::TMOV, DType::Float, 8, 34.0},
        {PtoOpcode::TMOV, DType::Float, 56, 34.0},
        {PtoOpcode::TMOV, DType::Float, 64, 34.0},
        {PtoOpcode::TMOV, DType::Float, 192, 36.0},
        {PtoOpcode::TMOV, DType::Float, 512, 41.0},
        {PtoOpcode::TMOV, DType::Float, 72, 35.0},
        {PtoOpcode::TMOV, DType::Float, 128, 35.0},
        {PtoOpcode::TNOT, DType::Int16, 112, 36.0},
        {PtoOpcode::TNOT, DType::Int16, 256, 37.0},
        {PtoOpcode::TNOT, DType::Int16, 1024, 43.0},
        {PtoOpcode::TNOT, DType::Int16, 144, 37.0},
        {PtoOpcode::TNOT, DType::Int16, 384, 38.0},
        {PtoOpcode::TNOT, DType::Int16, 16, 36.0},
        {PtoOpcode::TNOT, DType::Int16, 128, 36.0},
        {PtoOpcode::TNOT, DType::Uint16, 384, 38.0},
        {PtoOpcode::TNOT, DType::Uint16, 256, 37.0},
        {PtoOpcode::TNOT, DType::Uint16, 112, 36.0},
        {PtoOpcode::TNOT, DType::Uint16, 144, 37.0},
        {PtoOpcode::TNOT, DType::Uint16, 1024, 43.0},
        {PtoOpcode::TNOT, DType::Uint16, 128, 36.0},
        {PtoOpcode::TNOT, DType::Uint16, 16, 36.0},
        {PtoOpcode::TOR, DType::Int16, 112, 37.0},
        {PtoOpcode::TOR, DType::Int16, 144, 38.0},
        {PtoOpcode::TOR, DType::Int16, 384, 39.0},
        {PtoOpcode::TOR, DType::Int16, 256, 38.0},
        {PtoOpcode::TOR, DType::Int16, 1024, 44.0},
        {PtoOpcode::TOR, DType::Int16, 16, 37.0},
        {PtoOpcode::TOR, DType::Int16, 128, 37.0},
        {PtoOpcode::TOR, DType::Uint16, 1024, 44.0},
        {PtoOpcode::TOR, DType::Uint16, 384, 39.0},
        {PtoOpcode::TOR, DType::Uint16, 256, 38.0},
        {PtoOpcode::TOR, DType::Uint16, 112, 37.0},
        {PtoOpcode::TOR, DType::Uint16, 144, 38.0},
        {PtoOpcode::TOR, DType::Uint16, 128, 37.0},
        {PtoOpcode::TOR, DType::Uint16, 16, 37.0},
        {PtoOpcode::TORS, DType::Uint16, 112, 36.0},
        {PtoOpcode::TORS, DType::Uint16, 144, 37.0},
        {PtoOpcode::TORS, DType::Uint16, 384, 38.0},
        {PtoOpcode::TORS, DType::Uint16, 256, 37.0},
        {PtoOpcode::TORS, DType::Uint16, 1024, 43.0},
        {PtoOpcode::TORS, DType::Uint16, 128, 36.0},
        {PtoOpcode::TORS, DType::Uint16, 16, 36.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 8, 41.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 64, 41.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 192, 43.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 512, 48.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 56, 41.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 72, 42.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 128, 42.0},
        {PtoOpcode::TSEL, DType::Uint16, 16, 38.0},
        {PtoOpcode::TSEL, DType::Uint16, 112, 38.0},
        {PtoOpcode::TSEL, DType::Uint16, 256, 39.0},
        {PtoOpcode::TSEL, DType::Uint16, 144, 39.0},
        {PtoOpcode::TSEL, DType::Uint16, 384, 42.0},
        {PtoOpcode::TSEL, DType::Uint16, 1024, 51.0},
        {PtoOpcode::TSEL, DType::Uint16, 32, 38.0},
        {PtoOpcode::TSEL, DType::Uint16, 160, 39.0},
        {PtoOpcode::TSEL, DType::Uint16, 128, 38.0},
        {PtoOpcode::TTRI, DType::Float, 56, 37.0},
        {PtoOpcode::TTRI, DType::Float, 72, 39.0},
        {PtoOpcode::TTRI, DType::Float, 512, 47.0},
        {PtoOpcode::TTRI, DType::Float, 64, 37.0},
        {PtoOpcode::TTRI, DType::Float, 192, 39.0},
        {PtoOpcode::TTRI, DType::Float, 256, 41.0},
        {PtoOpcode::TTRI, DType::Float, 128, 39.0},
        {PtoOpcode::TTRI, DType::Float, 8, 37.0},
        {PtoOpcode::TTRI, DType::Int32, 56, 37.0},
        {PtoOpcode::TTRI, DType::Int32, 64, 37.0},
        {PtoOpcode::TTRI, DType::Int32, 72, 39.0},
        {PtoOpcode::TTRI, DType::Int32, 192, 39.0},
        {PtoOpcode::TTRI, DType::Int32, 512, 47.0},
        {PtoOpcode::TTRI, DType::Int32, 256, 41.0},
        {PtoOpcode::TTRI, DType::Int32, 8, 37.0},
        {PtoOpcode::TTRI, DType::Int32, 128, 39.0},
        {PtoOpcode::TXOR, DType::Uint16, 112, 37.0},
        {PtoOpcode::TXOR, DType::Uint16, 256, 38.0},
        {PtoOpcode::TXOR, DType::Uint16, 1024, 44.0},
        {PtoOpcode::TXOR, DType::Uint16, 144, 38.0},
        {PtoOpcode::TXOR, DType::Uint16, 384, 39.0},
        {PtoOpcode::TXOR, DType::Uint16, 128, 37.0},
        {PtoOpcode::TXOR, DType::Uint16, 16, 37.0},
        {PtoOpcode::TXORS, DType::Uint16, 112, 36.0},
        {PtoOpcode::TXORS, DType::Uint16, 144, 37.0},
        {PtoOpcode::TXORS, DType::Uint16, 384, 38.0},
        {PtoOpcode::TXORS, DType::Uint16, 256, 37.0},
        {PtoOpcode::TXORS, DType::Uint16, 1024, 43.0},
        {PtoOpcode::TXORS, DType::Uint16, 128, 36.0},
        {PtoOpcode::TXORS, DType::Uint16, 16, 36.0},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.op = profile.op;
        input.dtype = profile.dtype;
        input.rows = 1;
        input.cols = profile.cols;
        input.data_size = profile.cols;
        input.arch = CostModelArch::A5;
        input.a5_shape_path = A5ShapePathHint::Path1D;
        input.tile_type = TransferTileType::VecTile;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.op) << " cols=" << profile.cols;
    }
}

TEST(A5DenseSingleRowFit, PreservesUncalibratedKeys)
{
    auto keyFor = [](CostModelInput input) {
        a5::fit::VfCurveKey key{};
        uint64_t loop = 0, outer = 0, inner = 0, tail = 0, rowInner = 0;
        EXPECT_TRUE(a5::BuildVfCurveKeyAndLoopCount(input, key, loop, outer, inner, tail, rowInner));
        return key.op_params;
    };
    CostModelInput input{};
    input.op = PtoOpcode::TEXPANDS;
    input.dtype = DType::Float;
    input.arch = CostModelArch::A5;
    input.rows = 1;
    input.cols = 64;
    input.a5_shape_path = A5ShapePathHint::Path1D;
    EXPECT_EQ(keyFor(input), "dense_row1");
    input.valid_cols = 32;
    EXPECT_NE(keyFor(input), "dense_row1");
    input.valid_cols = 0;
    input.cols = 576;
    EXPECT_NE(keyFor(input), "dense_row1");
    input.cols = 64;
    input.rows = 2;
    EXPECT_NE(keyFor(input), "dense_row1");
    input.rows = 1;
    input.a5_op_params = "explicit_parameter";
    EXPECT_EQ(keyFor(input), "explicit_parameter");
}

} // namespace
