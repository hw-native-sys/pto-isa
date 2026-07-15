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
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTPrelu(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    constexpr unsigned tmpRow = kTRows_ + 1;
    constexpr unsigned tmpCol = (((kTCols_ + 7) / 8 + 31) / 32) * 32; // 除以8后向上取整，然后向上取32B对齐
    // tmp的vaild row/col在运算中不生效，不需要打印出来的话不用在意
    constexpr unsigned tmpVRow = vRows;
    constexpr unsigned tmpVCol = (vCols + 7) / 8;

    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileDataTmp = Tile<TileType::Vec, uint8_t, tmpRow, tmpCol, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TileDataTmp tmpTile(tmpVRow, tmpVCol);
    size_t size = kTRows_ * kTCols_ * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, size * 2);
    TASSIGN(tmpTile, size * 3);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TPRELU> event0;
    Event<Op::TPRELU, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TPRELU(dstTile, src0Tile, src1Tile, tmpTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTPrelu(T* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPrelu<half, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    } else {
        runTPrelu<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTPrelu<aclFloat16, 64, 64, 64, 64>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPrelu<aclFloat16, 64, 64, 63, 63>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPrelu<aclFloat16, 1, 16384, 1, 16384>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPrelu<aclFloat16, 1024, 16, 1024, 16>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPrelu<float, 64, 64, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPrelu<float, 64, 64, 63, 63>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPrelu<float, 1, 8192, 1, 8192>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPrelu<float, 1024, 8, 1024, 8>(float* out, float* src0, float* src1, void* stream);