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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;

template <typename T, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTCVT(__gm__ T *out, __gm__ S *src)
{
    using DynShapeDim4 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim4 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src = GlobalTensor<S, DynShapeDim4, DynStridDim4>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    using TileDataSrc = Tile<TileType::Vec, S, kTRows_, kTCols_, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(S));

    GlobalData_src srcGlobal(src);

    GlobalData_dst dstGlobal(out);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(typename TileDataSrc::DType));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TCVT(dstTile, srcTile, RoundMode::CAST_RINT);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVT(D *dst, S *src, void *stream)
{
    if constexpr (std::is_same_v<D, aclFloat16>) {
        runTCVT<half, S, kGRows_, kGCols_, kTRows_, kTCols_>((half *)dst, src);
    } else if constexpr (std::is_same_v<S, aclFloat16>) {
        runTCVT<D, half, kGRows_, kGCols_, kTRows_, kTCols_>(dst, (half *)src);
    } else {
        runTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_>(dst, src);
    }
}

template void launchTCVT<int32_t, float, 128, 128, 128, 128>(int32_t *dst, float *src, void *stream);
template void launchTCVT<float, int32_t, 256, 64, 256, 64>(float *dst, int32_t *src, void *stream);
template void launchTCVT<int16_t, float, 16, 32, 16, 32>(int16_t *dst, float *src, void *stream);
template void launchTCVT<int32_t, float, 32, 512, 32, 512>(int32_t *dst, float *src, void *stream);
template void launchTCVT<int32_t, int16_t, 2, 512, 2, 512>(int32_t *dst, int16_t *src, void *stream);
template void launchTCVT<int32_t, float, 4, 4096, 4, 4096>(int32_t *dst, float *src, void *stream);
template void launchTCVT<float, int16_t, 64, 64, 64, 64>(float *dst, int16_t *src, void *stream);
template void launchTCVT<aclFloat16, float, 64, 64, 64, 64>(aclFloat16 *dst, float *src, void *stream);
template void launchTCVT<uint8_t, aclFloat16, 64, 64, 64, 64>(uint8_t *dst, aclFloat16 *src, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void launchTCVT<bfloat16_t, float, 64, 64, 64, 64>(bfloat16_t *dst, float *src, void *stream);
template void launchTCVT<float, bfloat16_t, 64, 64, 64, 64>(float *dst, bfloat16_t *src, void *stream);
#endif
