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
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTCOLEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kTRows_, kTCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using TileT = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileT srcTile(kTRows_, kTCols_);
    TileT dstTile(kTRows_, kTCols_);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(typename TileT::DType));

    TLOAD(srcTile, srcGlobal);
    TCOLEXPAND(dstTile, srcTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTCOLEXPAND(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPAND<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src));
    } else {
        runTCOLEXPAND<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src);
    }
}

template void LaunchTCOLEXPAND<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTCOLEXPAND<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTCOLEXPAND<bfloat16_t, 16, 256, 16, 256>(bfloat16_t *out, bfloat16_t *src, void *stream);
#endif
