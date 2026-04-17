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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kSubRows_, int kSubCols_>
AICORE void runTSubView(__gm__ T __out__ *out, __gm__ T __in__ *src, uint16_t rowIdx, uint16_t colIdx)
{
    using SrcDynShape = Shape<1, 1, 1, kGRows_, kGCols_>;
    using SrcDynStride = Stride<1, 1, 1, kGCols_, 1>;
    using DstDynShape = Shape<1, 1, 1, kSubRows_, kSubCols_>;
    using DstDynStride = Stride<1, 1, 1, kSubCols_, 1>;
    using GlobalDataSrc = GlobalTensor<T, SrcDynShape, SrcDynStride>;
    using GlobalDataDst = GlobalTensor<T, DstDynShape, DstDynStride>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(kTRows_, kTCols_);
    TileData subTile(kSubRows_, kSubCols_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(subTile, 0x1000);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUBVIEW(subTile, srcTile, rowIdx, colIdx);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, subTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kSubRows_, int kSubCols_>
void LaunchTSubView(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTSubView<T, kGRows_, kGCols_, kTRows_, kTCols_, kSubRows_, kSubCols_>((half *)(out), (half *)(src), 0, 0);
    else
        runTSubView<T, kGRows_, kGCols_, kTRows_, kTCols_, kSubRows_, kSubCols_>(out, src, 0, 0);
}

template void LaunchTSubView<float, 64, 64, 64, 64, 32, 32>(float *out, float *src, void *stream);
template void LaunchTSubView<int32_t, 64, 64, 64, 64, 32, 32>(int32_t *out, int32_t *src, void *stream);
template void LaunchTSubView<aclFloat16, 16, 256, 16, 256, 8, 128>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTSubView<int16_t, 64, 64, 64, 64, 32, 32>(int16_t *out, int16_t *src, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTSubView<bfloat16_t, 16, 256, 16, 256, 8, 128>(bfloat16_t *out, bfloat16_t *src, void *stream);
#endif