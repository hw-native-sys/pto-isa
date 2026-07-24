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
CostModelInput TSubInput()
{
    return pto::test::MakeA5VfInput<T, PtoOpcode::TSUB, Rows, Cols, ValidRows, ValidCols, VfImplKindValue>();
}

template <
    typename T, int Rows, int Cols, double Profiling, double Precision,
    VFImplKind VfImplKindValue = VFImplKind::VFIMPL_DEFAULT>
void runTSub()
{
    pto::test::ExpectA5CycleNear("TSUB", TSubInput<T, Rows, Cols, 0, 0, VfImplKindValue>(), Profiling, Precision);
}

template <
    typename T, int Rows, int Cols, int ValidRows, int ValidCols, double Profiling, double Precision,
    VFImplKind VfImplKindValue = VFImplKind::VFIMPL_DEFAULT>
void runTSub()
{
    pto::test::ExpectA5CycleNear(
        "TSUB", TSubInput<T, Rows, Cols, ValidRows, ValidCols, VfImplKindValue>(), Profiling, Precision);
}

} // namespace

TEST(A5TSubFit, Fp32Default1DPostUpdateFull)
{
    runTSub<float, 1, 64, 55.0, 0.9818>();
    runTSub<float, 1, 192, 60.0, 0.9833>();
    runTSub<float, 1, 320, 62.0, 0.9838>();
    runTSub<float, 1, 640, 67.0, 0.9850>();
    runTSub<float, 1, 1280, 80.0, 1.0000>();
}

TEST(A5TSubFit, Fp32Default1DPostUpdateTail)
{
    runTSub<float, 1, 64, 1, 1, 55.0, 0.9818>();
    runTSub<float, 1, 192, 1, 129, 60.0, 0.9833>();
    runTSub<float, 1, 320, 1, 257, 62.0, 0.9838>();
    runTSub<float, 1, 640, 1, 577, 67.0, 0.9850>();
    runTSub<float, 1, 1280, 1, 1217, 80.0, 1.0000>();
}

TEST(A5TSubFit, Fp32Explicit1DNoPostUpdate)
{
    runTSub<float, 1, 640, 1, 640, 65.0, 0.9538, VFImplKind::VFIMPL_1D_NO_POST_UPDATE>();
}

TEST(A5TSubFit, Fp32Default2DNoPostUpdateFull)
{
    runTSub<float, 2, 160, 2, 128, 65.0, 0.8923>();
    runTSub<float, 2, 224, 2, 192, 66.0, 0.9242>();
    runTSub<float, 2, 672, 2, 640, 82.0, 0.9268>();
    runTSub<float, 2, 1312, 2, 1280, 118.0, 0.9406>();

    runTSub<float, 3, 160, 3, 128, 72.0, 0.8888>();
    runTSub<float, 3, 224, 3, 192, 75.0, 0.9333>();
    runTSub<float, 3, 672, 3, 640, 104.0, 0.9519>();
    runTSub<float, 3, 1312, 3, 1280, 160.0, 0.9687>();

    runTSub<float, 10, 160, 10, 128, 121.0, 0.9173>();
    runTSub<float, 10, 224, 10, 192, 131.0, 0.9847>();
    runTSub<float, 10, 672, 10, 640, 245.0, 0.9346>();
    runTSub<float, 10, 1312, 10, 1280, 449.0, 0.9977>();

    runTSub<float, 20, 160, 20, 128, 191.0, 0.9267>();
    runTSub<float, 20, 224, 20, 192, 211.0, 0.9857>();
    runTSub<float, 20, 672, 20, 640, 450.0, 0.9400>();
    runTSub<float, 20, 1312, 20, 1280, 869.0, 0.9804>();
}

TEST(A5TSubFit, Fp32Default2DNoPostUpdateTail)
{
    runTSub<float, 2, 160, 2, 65, 65.0, 0.8923>();
    runTSub<float, 2, 224, 2, 129, 66.0, 0.9242>();
    runTSub<float, 2, 672, 2, 577, 82.0, 0.9268>();
    runTSub<float, 2, 1312, 2, 1217, 118.0, 0.9406>();

    runTSub<float, 3, 160, 3, 65, 72.0, 0.8888>();
    runTSub<float, 3, 224, 3, 129, 75.0, 0.9333>();
    runTSub<float, 3, 672, 3, 577, 104.0, 0.9519>();
    runTSub<float, 3, 1312, 3, 1217, 160.0, 0.9687>();

    runTSub<float, 10, 160, 10, 65, 121.0, 0.9173>();
    runTSub<float, 10, 224, 10, 129, 131.0, 0.9847>();
    runTSub<float, 10, 672, 10, 577, 245.0, 0.9346>();
    runTSub<float, 10, 1312, 10, 1217, 449.0, 0.9977>();

    runTSub<float, 20, 160, 20, 65, 191.0, 0.9267>();
    runTSub<float, 20, 224, 20, 129, 211.0, 0.9857>();
    runTSub<float, 20, 672, 20, 577, 450.0, 0.9400>();
    runTSub<float, 20, 1312, 20, 1217, 869.0, 0.9804>();
}

TEST(A5TSubFit, Fp16ReusesFp32Curve) { runTSub<half, 1, 1280, 67.0, 0.9850>(); }
