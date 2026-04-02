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

using namespace pto;

template <typename T, int isUpperOrLower, int diagonal, int kTRows_, int kTCols_, int vRows, int vCols>
AICORE void runTTRI(__gm__ T __out__ *out)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData dstTile(vRows, vCols);

    GlobalData dstGlobal(out);
    TASSIGN(dstTile, 0x0);

    TTRI<TileData, isUpperOrLower>(dstTile, diagonal);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int isUpperOrLower, int diagonal, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTTRI(T *out, void *stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTTRI<half, isUpperOrLower, diagonal, kTRows_, kTCols_, vRows, vCols>((half *)(out));
    else
        runTTRI<T, isUpperOrLower, diagonal, kTRows_, kTCols_, vRows, vCols>(out);
}

template void LaunchTTRI<float, 1, 0, 4, 8, 4, 4>(float *out, void *stream);
template void LaunchTTRI<float, 1, 0, 64, 64, 64, 64>(float *out, void *stream);
template void LaunchTTRI<int32_t, 1, 0, 64, 64, 64, 64>(int32_t *out, void *stream);
template void LaunchTTRI<int16_t, 1, 0, 64, 64, 64, 64>(int16_t *out, void *stream);
template void LaunchTTRI<aclFloat16, 1, 0, 16, 256, 16, 256>(aclFloat16 *out, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTTRI<bfloat16_t, 1, 0, 16, 256, 16, 256>(bfloat16_t *out, void *stream);
#endif
template void LaunchTTRI<float, 1, 0, 128, 128, 128, 128>(float *out, void *stream);
template void LaunchTTRI<float, 0, 0, 64, 64, 64, 64>(float *out, void *stream);
template void LaunchTTRI<int32_t, 0, 0, 64, 64, 64, 64>(int32_t *out, void *stream);
template void LaunchTTRI<int16_t, 0, 0, 64, 64, 64, 64>(int16_t *out, void *stream);
template void LaunchTTRI<aclFloat16, 0, 0, 16, 256, 16, 256>(aclFloat16 *out, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTTRI<bfloat16_t, 0, 0, 16, 256, 16, 256>(bfloat16_t *out, void *stream);
#endif
template void LaunchTTRI<float, 0, 0, 128, 128, 128, 128>(float *out, void *stream);
template void LaunchTTRI<float, 0, 0, 128, 128, 128, 125>(float *out, void *stream);
template void LaunchTTRI<uint32_t, 1, 0, 64, 64, 64, 64>(uint32_t *out, void *stream);
template void LaunchTTRI<uint32_t, 0, 0, 64, 64, 64, 64>(uint32_t *out, void *stream);

template void LaunchTTRI<float, 0, 2, 128, 128, 128, 111>(float *out, void *stream);
template void LaunchTTRI<float, 0, -2, 128, 128, 128, 111>(float *out, void *stream);
template void LaunchTTRI<float, 0, 444, 128, 128, 128, 31>(float *out, void *stream);
template void LaunchTTRI<float, 0, -444, 128, 128, 128, 31>(float *out, void *stream);
template void LaunchTTRI<float, 1, 2, 128, 128, 128, 111>(float *out, void *stream);
template void LaunchTTRI<float, 1, -2, 128, 128, 128, 111>(float *out, void *stream);
template void LaunchTTRI<float, 1, 444, 128, 128, 128, 31>(float *out, void *stream);
template void LaunchTTRI<float, 1, -444, 128, 128, 128, 31>(float *out, void *stream);
