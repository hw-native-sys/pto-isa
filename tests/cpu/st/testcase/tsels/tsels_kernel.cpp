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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__aicore__ void runTSelS(__gm__ T* out, T scalar, __gm__ uint32_t* src0, __gm__ T* src1)
{
    constexpr int totalElements = kGRows_ * kGCols_;
    constexpr int maskWords = (totalElements + 31) / 32;
    constexpr int maskCols = (maskWords + kTRows_ - 1) / kTRows_;

    // Data shapes maintain 5D structural mappings matching your matrix topology
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    // Flatted 1D representation for the packed bitmask to prevent step-stride truncation errors
    using MaskShape1D = Shape<1, 1, 1, kGRows_, maskCols>;
    using MaskStride1D = Stride<1, 1, 1, maskCols, 1>;
    using GlobalMask = GlobalTensor<uint32_t, MaskShape1D, MaskStride1D>;

    // Keep Tile allocations matching hardware vector processing layouts
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileMask = Tile<TileType::Vec, uint32_t, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    TileMask src0Tile(kTRows_, maskCols);
    TileData src1Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TmpTile tmpTile(1, 32);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);
    TASSIGN(tmpTile, 0x12000);

    GlobalMask src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // TSELS handles bitwise extraction dynamically under the hood based on your hardware pipeline
    TSELS(dstTile, src0Tile, src1Tile, tmpTile, scalar);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTSelS(T* out, T scalar, uint32_t* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTSelS<half, kGRows_, kGCols_, kTRows_, kTCols_>((half*)(out), (half)(scalar), src0, (half*)(src1));
    else
        runTSelS<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, scalar, src0, src1);
}

template void LaunchTSelS<float, 64, 64, 64, 64>(float* out, float scalar, uint32_t* src0, float* src1, void* stream);
template void LaunchTSelS<int32_t, 64, 64, 64, 64>(
    int32_t* out, int32_t scalar, uint32_t* src0, int32_t* src1, void* stream);
template void LaunchTSelS<aclFloat16, 16, 256, 16, 256>(
    aclFloat16* out, aclFloat16 scalar, uint32_t* src0, aclFloat16* src1, void* stream);
template void LaunchTSelS<int16_t, 64, 64, 64, 64>(
    int16_t* out, int16_t scalar, uint32_t* src0, int16_t* src1, void* stream);
template void LaunchTSelS<uint16_t, 64, 64, 64, 64>(
    uint16_t* out, uint16_t scalar, uint32_t* src0, uint16_t* src1, void* stream);
template void LaunchTSelS<uint32_t, 64, 64, 64, 64>(
    uint32_t* out, uint32_t scalar, uint32_t* src0, uint32_t* src1, void* stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTSelS<bfloat16_t, 16, 256, 16, 256>(
    bfloat16_t* out, bfloat16_t scalar, uint32_t* src0, bfloat16_t* src1, void* stream);
#endif
