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

template <typename T, typename TDst, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTCmp(__gm__ TDst __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1, pto::CmpMode mode)
{
    constexpr int kBitsPerDst = sizeof(TDst) * 8;
    constexpr int kPackedCols = (kTCols_ + kBitsPerDst - 1) / kBitsPerDst;

    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using DstShape = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DstStride = Stride<1, 1, 1, kGCols_, 1>;
    using DstGlobal = GlobalTensor<TDst, DstShape, DstStride>;

    using SrcTile = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, TDst, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    SrcTile src0Tile(kTRows_, kTCols_);
    SrcTile src1Tile(kTRows_, kTCols_);
    DstTile dstTile(kTRows_, kPackedCols);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    DstGlobal dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, kTRows_ * kTCols_ * sizeof(T));
    TASSIGN(dstTile, 2 * kTRows_ * kTCols_ * sizeof(T));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TCMP(dstTile, src0Tile, src1Tile, mode);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, typename TDst, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTCmp(TDst* out, T* src0, T* src1, pto::CmpMode mode, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTCmp<half, TDst, kGRows_, kGCols_, kTRows_, kTCols_>(out, (half*)(src0), (half*)(src1), mode);
    else
        runTCmp<T, TDst, kGRows_, kGCols_, kTRows_, kTCols_>(out, src0, src1, mode);
}

template void LaunchTCmp<float, uint8_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint8_t* out, float* src0, float* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<int32_t, uint8_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint8_t* out, int32_t* src0, int32_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<aclFloat16, uint8_t, NUM_16, NUM_256, NUM_16, NUM_256>(
    uint8_t* out, aclFloat16* src0, aclFloat16* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<uint32_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, uint32_t* src0, uint32_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<int32_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, int32_t* src0, int32_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<uint16_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, uint16_t* src0, uint16_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<int16_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, int16_t* src0, int16_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<uint8_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, uint8_t* src0, uint8_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<int8_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, int8_t* src0, int8_t* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<float, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    uint32_t* out, float* src0, float* src1, pto::CmpMode mode, void* stream);
template void LaunchTCmp<aclFloat16, uint32_t, NUM_16, NUM_256, NUM_16, NUM_256>(
    uint32_t* out, aclFloat16* src0, aclFloat16* src1, pto::CmpMode mode, void* stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTCmp<bfloat16_t, uint32_t, NUM_16, NUM_256, NUM_16, NUM_256>(
    uint32_t* out, bfloat16_t* src0, bfloat16_t* src1, pto::CmpMode mode, void* stream);
#endif
