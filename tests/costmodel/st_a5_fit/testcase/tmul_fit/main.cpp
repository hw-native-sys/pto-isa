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

#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using ::pto::VFImplKind;
using ::pto::mocker::lightweight::CostModelInput;
using ::pto::mocker::lightweight::PtoOpcode;

template <
    typename T, int Rows, int Cols, int ValidRows, int ValidCols,
    VFImplKind VfImplKindValue = VFImplKind::VFIMPL_DEFAULT>
CostModelInput TMulInput()
{
    return pto::test::MakeA5VfInput<T, PtoOpcode::TMUL, Rows, Cols, ValidRows, ValidCols, VfImplKindValue>();
}

template <
    typename T, int Rows, int Cols, double Profiling, double Precision,
    VFImplKind VfImplKindValue = VFImplKind::VFIMPL_DEFAULT>
void runTMul()
{
    pto::test::ExpectA5CycleNear("TMUL", TMulInput<T, Rows, Cols, 0, 0, VfImplKindValue>(), Profiling, Precision);
}

template <
    typename T, int Rows, int Cols, int ValidRows, int ValidCols, double Profiling, double Precision,
    VFImplKind VfImplKindValue = VFImplKind::VFIMPL_DEFAULT>
void runTMul()
{
    pto::test::ExpectA5CycleNear(
        "TMUL", TMulInput<T, Rows, Cols, ValidRows, ValidCols, VfImplKindValue>(), Profiling, Precision);
}

} // namespace

TEST(A5TMulFit, Fp32Default1DPostUpdateFull)
{
    runTMul<float, 1, 64, 56.0, 0.9821>();
    runTMul<float, 1, 192, 61.0, 0.9836>();
    runTMul<float, 1, 320, 63.0, 0.9841>();
    runTMul<float, 1, 640, 68.0, 1.0000>();
    runTMul<float, 1, 1280, 80.0, 1.0000>();
}

TEST(A5TMulFit, Fp32Default1DPostUpdateTail)
{
    runTMul<float, 1, 64, 1, 1, 56.0, 0.9821>();
    runTMul<float, 1, 192, 1, 129, 61.0, 0.9836>();
    runTMul<float, 1, 320, 1, 257, 63.0, 0.9841>();
    runTMul<float, 1, 640, 1, 577, 68.0, 1.0000>();
    runTMul<float, 1, 1280, 1, 1217, 80.0, 1.0000>();
}

TEST(A5TMulFit, Fp32Explicit1DNoPostUpdate)
{
    runTMul<float, 1, 640, 1, 640, 66.0, 0.9696, VFImplKind::VFIMPL_1D_NO_POST_UPDATE>();
}

TEST(A5TMulFit, Fp32Default2DNoPostUpdateFull)
{
    runTMul<float, 2, 160, 2, 128, 66.0, 0.8787>();
    runTMul<float, 2, 224, 2, 192, 67.0, 0.9253>();
    runTMul<float, 2, 672, 2, 640, 82.0, 0.9268>();
    runTMul<float, 2, 1312, 2, 1280, 118.0, 0.9322>();

    runTMul<float, 3, 160, 3, 128, 73.0, 0.8904>();
    runTMul<float, 3, 224, 3, 192, 76.0, 0.9210>();
    runTMul<float, 3, 672, 3, 640, 104.0, 0.9423>();
    runTMul<float, 3, 1312, 3, 1280, 160.0, 0.9625>();

    runTMul<float, 10, 160, 10, 128, 122.0, 0.9098>();
    runTMul<float, 10, 224, 10, 192, 132.0, 0.9848>();
    runTMul<float, 10, 672, 10, 640, 245.0, 0.9346>();
    runTMul<float, 10, 1312, 10, 1280, 449.0, 0.9977>();

    runTMul<float, 20, 160, 20, 128, 192.0, 0.9270>();
    runTMul<float, 20, 224, 20, 192, 212.0, 0.9858>();
    runTMul<float, 20, 672, 20, 640, 450.0, 0.9400>();
    runTMul<float, 20, 1312, 20, 1280, 869.0, 0.9792>();
}

TEST(A5TMulFit, Fp32Default2DNoPostUpdateTail)
{
    runTMul<float, 2, 160, 2, 65, 66.0, 0.8787>();
    runTMul<float, 2, 224, 2, 129, 67.0, 0.9253>();
    runTMul<float, 2, 672, 2, 577, 82.0, 0.9268>();
    runTMul<float, 2, 1312, 2, 1217, 118.0, 0.9322>();

    runTMul<float, 3, 160, 3, 65, 73.0, 0.8904>();
    runTMul<float, 3, 224, 3, 129, 76.0, 0.9210>();
    runTMul<float, 3, 672, 3, 577, 104.0, 0.9423>();
    runTMul<float, 3, 1312, 3, 1217, 160.0, 0.9625>();

    runTMul<float, 10, 160, 10, 65, 122.0, 0.9098>();
    runTMul<float, 10, 224, 10, 129, 132.0, 0.9848>();
    runTMul<float, 10, 672, 10, 577, 245.0, 0.9346>();
    runTMul<float, 10, 1312, 10, 1217, 449.0, 0.9977>();

    runTMul<float, 20, 160, 20, 65, 192.0, 0.9270>();
    runTMul<float, 20, 224, 20, 129, 212.0, 0.9858>();
    runTMul<float, 20, 672, 20, 577, 450.0, 0.9400>();
    runTMul<float, 20, 1312, 20, 1217, 869.0, 0.9792>();
}

TEST(A5TMulFit, Fp16ReusesFp32Curve) { runTMul<half, 1, 1280, 68.0, 1.0000>(); }
