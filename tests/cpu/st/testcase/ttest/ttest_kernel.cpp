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
#include <pto/common/constants.hpp>
using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE bool runTTest(__gm__ T __in__ *input, __gm__ int32_t __in__ cmpValue, __gm__ comm::WaitCmp __in__ cmp)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    pto::comm::Signal localSignal(input);
    bool testResult = comm::TTEST(localSignal, cmpValue, cmp);
    return testResult;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
bool LaunchTTest(T *src, int32_t cmpValue, comm::WaitCmp cmp, void *stream)
{
    return runTTest<T, kGRows_, kGCols_, kTRows_, kTCols_>(src, cmpValue, cmp);
}

template bool LaunchTTest<int32_t, 64, 64, 64, 64>(int32_t *src, int32_t cmpValue, comm::WaitCmp cmp, void *stream);
template bool LaunchTTest<int32_t, 16, 256, 16, 256>(int32_t *src, int32_t cmpValue, comm::WaitCmp cmp, void *stream);
