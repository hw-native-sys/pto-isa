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

template <typename T, int kRows, int kCols, int kValidRows1, int kValidCols1>
AICORE inline void runTPARTADD(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStridDim5 = Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using GlobalData1 = GlobalTensor<T, Shape<1, 1, 1, kValidRows1, kValidCols1>, DynStridDim5>;

    using TileT = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileT src0Tile(kRows, kCols);
    TileT src1Tile(kValidRows1, kValidCols1);
    TileT dstTile(kRows, kCols);

    GlobalData src0Global(src0);
    GlobalData1 src1Global(src1);
    GlobalData dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, kRows * kCols * sizeof(typename TileT::DType));
    TASSIGN(dstTile, 2 * kRows * kCols * sizeof(typename TileT::DType));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TPARTADD(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTADD(T* out, T* src0, T* src1, void* stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPARTADD<half, kRows, kCols, kValidRows1, kValidCols1>((half*)(out), (half*)src0, (half*)src1);
    } else {
        runTPARTADD<T, kRows, kCols, kValidRows1, kValidCols1>(out, src0, src1);
    }
}

template void LaunchTPARTADD<float, 64, 64, 32, 32>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPARTADD<int32_t, 64, 64, 32, 32>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTPARTADD<int64_t, 64, 64, 32, 32>(int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTPARTADD<uint64_t, 64, 64, 32, 32>(uint64_t* out, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTPARTADD<int16_t, 64, 64, 32, 32>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTPARTADD<uint16_t, 64, 64, 32, 32>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTPARTADD<uint32_t, 64, 64, 32, 32>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTPARTADD<bfloat16_t, 64, 64, 32, 32>(
    bfloat16_t* out, bfloat16_t* src0, bfloat16_t* src1, void* stream);
#endif
