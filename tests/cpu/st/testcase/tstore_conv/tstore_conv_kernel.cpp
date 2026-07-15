/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T, int N, int D, int C1, int H, int W, int C0>
__global__ AICORE void runTStoreConv_NDC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(C0 == 32 / sizeof(T));

    // Define the dimensions for readability
    constexpr int64_t W_dim = W;
    constexpr int64_t H_dim = H;
    constexpr int64_t C1_dim = C1;
    constexpr int64_t D_dim = D;

    using ShapeDim5 = Shape<N, D, C1, H, W>;
    using StrideDim5 = pto::Stride<
        D_dim * C1_dim * H_dim * W_dim * C0, // Stride for N (Total elements)
        C1_dim * H_dim * W_dim * C0,         // Stride for D
        H_dim * W_dim * C0,                  // Stride for C1
        W_dim * C0,                          // Stride for H (Crucial: W * C0)
        C0                                   // Stride for W (Jump one full C0 vector)
        >;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NDC1HWC0>;

    constexpr size_t srcElemNum = N * D * C1 * H * W * C0;
    constexpr size_t srcBufferSize = srcElemNum * sizeof(T);

    using SrcConvTile = ConvTile<TileType::Mat, T, srcBufferSize, Layout::NDC1HWC0, ConvTileShape<N, D, C1, H, W, C0>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 6);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

template <
    typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3, int dstShape4, int groupN>
void LaunchTStoreConv(T* out, T* src, void* stream)
{
    if constexpr (format == 2) {
        runTStoreConv_NDC1HWC0<T, groupN, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4>(out, src);
    }
}

/*-------------------6D---------------------------*/
template void LaunchTStoreConv<float, 2, 1, 1, 1, 2, 8, 1, 1, 1, 2, 8, 1>(float* out, float* src, void* stream);
template void LaunchTStoreConv<float, 2, 3, 4, 1, 7, 8, 3, 4, 1, 7, 8, 2>(float* out, float* src, void* stream);
