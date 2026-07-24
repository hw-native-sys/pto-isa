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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
inline AICORE void runTTRANS(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using DynShapeDim4 = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStridDim4 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    constexpr uint16_t aligned_Rows = ((kTRows_ * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t aligned_Cols = ((kTCols_ * sizeof(T) + 31) / 32) * (32 / sizeof(T));

    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, aligned_Cols, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;
    using TileDataTmp = Tile<TileType::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;
    TileDataDst tmpTile;

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, aligned_Cols * aligned_Rows * sizeof(typename TileDataSrc::DType));
    TASSIGN(tmpTile, 2 * aligned_Cols * aligned_Rows * sizeof(typename TileDataSrc::DType));

    int offset = 0;

    GlobalData srcGlobal(src + offset, pto::Shape(1, 1, 1, kGRows_, kGCols_), pto::Stride(1, 1, 1, kGCols_, 1));

    GlobalData dstGlobal(out + offset, pto::Shape(1, 1, 1, kGCols_, kGRows_), pto::Stride(1, 1, 1, kGRows_, 1));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TTRANS(dstTile, srcTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTTRANS_1(__gm__ uint8_t* out, __gm__ uint8_t* src)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t K = 128;
    constexpr uint32_t L = 128;
    runTTRANS<float, M, N, K, L>(reinterpret_cast<__gm__ float*>(out), reinterpret_cast<__gm__ float*>(src));
}

template <int32_t tilingKey>
void launchTTRANS(uint8_t* out, uint8_t* src, void* stream)
{
    if constexpr (tilingKey == 1) {
        launchTTRANS_1(out, src);
    }
}

template void launchTTRANS<1>(uint8_t* out, uint8_t* src, void* stream);
