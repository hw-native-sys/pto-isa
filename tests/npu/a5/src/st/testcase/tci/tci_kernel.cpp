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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int reverse>
__global__ AICORE void runTci(__gm__ T __out__ *out, T S)
{
    // 1. 定义两个类型
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    // 2. 声明device侧的相关变量
    TileData dstTile(kTRows_, kTCols_);

    // 3. 为devices侧的变量分配地址；
    TASSIGN(dstTile, 0x0);

    // 4. 定义host侧的变量，并分配地址；
    GlobalData dstGlobal(out);

    // 5. 加载数据  but as for Tci接口也是不存在这个事项的,因为没有源数据需要加载 pass;

    // 6. 调用指令集进行计算；
    TCI<TileData, T, reverse>(dstTile, S);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    // 7.将数据保存到host侧，并输出到gm;
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// 8. 特别注意，生成索引的数量通过
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int reverse>
void LaunchTci(T *out, T S, void *stream)
{
    runTci<T, kGRows_, kGCols_, kTRows_, kTCols_, reverse><<<1, nullptr, stream>>>((T *)(out), S);
}

template void LaunchTci<int32_t, 1, 128, 1, 128, 1>(int32_t *out, int32_t S = 100, void *stream);
template void LaunchTci<int16_t, 1, 128, 1, 128, 0>(int16_t *out, int16_t S = -1, void *stream);
template void LaunchTci<int16_t, 1, 128, 1, 128, 1>(int16_t *out, int16_t S = -1, void *stream);
template void LaunchTci<int16_t, 1, 192, 1, 192, 1>(int16_t *out, int16_t S = -1, void *stream);
template void LaunchTci<int32_t, 1, 192, 1, 192, 1>(int32_t *out, int32_t S = -1, void *stream);
template void LaunchTci<int32_t, 1, 600, 1, 600, 1>(int32_t *out, int32_t S = 0, void *stream);
template void LaunchTci<int16_t, 1, 800, 1, 800, 0>(int16_t *out, int16_t S = 0, void *stream);
template void LaunchTci<int32_t, 1, 2560, 1, 2560, 1>(int32_t *out, int32_t S = 0, void *stream);
template void LaunchTci<int32_t, 1, 3200, 1, 3200, 0>(int32_t *out, int32_t S = 0, void *stream);
template void LaunchTci<int32_t, 1, 8, 1, 8, 0>(int32_t *out, int32_t S = 0, void *stream);