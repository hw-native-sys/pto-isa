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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int kGRowsD_, int kGColsD_, int kGRowsS0_, int kGColsS0_, int kGRowsS1_, int kGColsS1_,
          int kTRowsD_, int kTColsD_, int kTRowsS0_, int kTColsS0_, int kTRowsS1_, int kTColsS1_>
__global__ AICORE void runTPartMin(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim4 = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStridDim4 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim4, DynStridDim4>;
    using dstTileData = Tile<TileType::Vec, T, kTRowsD_, kTColsD_, BLayout::RowMajor, -1, -1>;
    using src0TileData = Tile<TileType::Vec, T, kTRowsS0_, kTColsS0_, BLayout::RowMajor, -1, -1>;
    using src1TileData = Tile<TileType::Vec, T, kTRowsS1_, kTColsS1_, BLayout::RowMajor, -1, -1>;
    dstTileData dstTile(kGRowsD_, kGColsD_);
    src0TileData src0Tile(kGRowsS0_, kGColsS0_);
    src1TileData src1Tile(kGRowsS1_, kGColsS1_);
    constexpr unsigned src0Offset = kTRowsD_ * kTColsD_;
    constexpr unsigned src1Offset = src0Offset + kTRowsS0_ * kTColsS0_;
    constexpr unsigned blockDataElems = src1Offset + kTRowsS1_ * kTColsS1_;
    TASSIGN(dstTile, (blockDataElems * block_idx) * sizeof(T));
    TASSIGN(src0Tile, (blockDataElems * block_idx + src0Offset) * sizeof(T));
    TASSIGN(src1Tile, (blockDataElems * block_idx + src1Offset) * sizeof(T));
    GlobalData dstGlobal(out + kTRowsD_ * kTColsD_ * block_idx, Shape(1, 1, 1, kGRowsD_, kGColsD_),
                         pto::Stride(1, 1, kGRowsD_, kGColsD_, 1));
    GlobalData src0Global(src0 + kTRowsS0_ * kTColsS0_ * block_idx, Shape(1, 1, 1, kGRowsS0_, kGColsS0_),
                          pto::Stride(1, 1, kGRowsS0_, kGColsS0_, 1));
    GlobalData src1Global(src1 + kTRowsS1_ * kTColsS1_ * block_idx, Shape(1, 1, 1, kGRowsS1_, kGColsS1_),
                          pto::Stride(1, 1, kGRowsS1_, kGColsS1_, 1));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TPARTMIN<dstTileData, src0TileData, src1TileData>(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRowsD_, int kGColsD_, int kGRowsS0_, int kGColsS0_, int kGRowsS1_, int kGColsS1_,
          int kTRowsD_, int kTColsD_, int kTRowsS0_, int kTColsS0_, int kTRowsS1_, int kTColsS1_>
void LaunchTPartMin(T *out, T *src0, T *src1, aclrtStream stream)
{
    runTPartMin<T, kGRowsD_, kGColsD_, kGRowsS0_, kGColsS0_, kGRowsS1_, kGColsS1_, kTRowsD_, kTColsD_, kTRowsS0_,
                kTColsS0_, kTRowsS1_, kTColsS1_><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTPartMin<float, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32>(float *out, float *src0,
                                                                                    float *src1, aclrtStream stream);
template void LaunchTPartMin<float, 22, 32, 22, 32, 16, 32, 22, 32, 22, 32, 16, 32>(float *out, float *src0,
                                                                                    float *src1, aclrtStream stream);
template void LaunchTPartMin<float, 22, 40, 22, 40, 22, 32, 22, 40, 22, 40, 22, 32>(float *out, float *src0,
                                                                                    float *src1, aclrtStream stream);
template void LaunchTPartMin<float, 22, 40, 22, 40, 8, 40, 22, 40, 22, 40, 8, 40>(float *out, float *src0, float *src1,
                                                                                  aclrtStream stream);
template void LaunchTPartMin<float, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128>(float *out, float *src0,
                                                                                          float *src1,
                                                                                          aclrtStream stream);
template void LaunchTPartMin<float, 16, 32, 16, 0, 16, 32, 16, 32, 16, 8, 16, 32>(float *out, float *src0, float *src1,
                                                                                  aclrtStream stream);
template void LaunchTPartMin<float, 16, 32, 0, 32, 16, 32, 16, 32, 8, 32, 16, 32>(float *out, float *src0, float *src1,
                                                                                  aclrtStream stream);
template void LaunchTPartMin<float, 16, 32, 16, 32, 16, 0, 16, 32, 16, 32, 16, 8>(float *out, float *src0, float *src1,
                                                                                  aclrtStream stream);
template void LaunchTPartMin<float, 16, 32, 16, 32, 0, 32, 16, 32, 16, 32, 8, 32>(float *out, float *src0, float *src1,
                                                                                  aclrtStream stream);
