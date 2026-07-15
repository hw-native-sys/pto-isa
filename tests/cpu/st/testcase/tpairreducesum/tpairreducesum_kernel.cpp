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

template <typename T, int row, int col>
__global__ AICORE void runTPAIRREDUCESUM(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using Shape = pto::Shape<1, 1, 1, row, col>;
    using Stride = pto::Stride<1, 1, 1, col, 1>;
    using GlobalData = GlobalTensor<T, Shape, Stride>;
    GlobalData dstGlobal(out);
    GlobalData srcGlobal(src);

    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor>;
    TileData dstTile;
    TileData srcTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TPAIRREDUCESUM(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int row, int col>
void LaunchTPAIRREDUCESUM(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPAIRREDUCESUM<half, row, col>((half *)(out), (half *)(src));
    } else {
        runTPAIRREDUCESUM<T, row, col>(out, src);
    }
}

template void LaunchTPAIRREDUCESUM<float, 16, 64>(float *out, float *src, void *stream);
template void LaunchTPAIRREDUCESUM<int32_t, 64, 8>(int32_t *out, int32_t *src, void *stream);
template void LaunchTPAIRREDUCESUM<aclFloat16, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTPAIRREDUCESUM<int16_t, 20, 16>(int16_t *out, int16_t *src, void *stream);
