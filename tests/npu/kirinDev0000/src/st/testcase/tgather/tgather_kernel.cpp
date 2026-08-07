/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include "tgather_common.h"

using namespace std;
using namespace pto;

#ifdef __CCE_AICORE__
template <int rows, int cols>
using StrideDim2 = pto::Stride<rows * cols, rows * cols, rows * cols, cols, 1>;
#endif
template <typename Tsrc0, typename Tsrc1, int kGRows0_, int kGCols0_, int kGRows1_, int kGCols1_>
inline AICORE void runTGather(__gm__ Tsrc0* out, __gm__ Tsrc0* src0, __gm__ Tsrc1* src1)
{
    using DynShapeDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = StrideDim2<kGRows0_, kGCols0_>;
    using GlobalData_src0 = GlobalTensor<Tsrc0, DynShapeDim5_src0, DynStridDim5_src0>;
    using DynShapeDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = StrideDim2<kGRows1_, kGCols1_>;
    using GlobalData_src1 = GlobalTensor<Tsrc1, DynShapeDim5_src1, DynStridDim5_src1>;
    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_dst = StrideDim2<kGRows1_, kGCols1_>;
    using GlobalData_dst = GlobalTensor<Tsrc0, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_;
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows1_;
    constexpr int dst_col = kGCols1_;

    using TileData_src0 = Tile<TileType::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<TileType::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<TileType::Vec, Tsrc0, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col);
    TileData_dst dstTile(dst_row, dst_col);
    TileData_dst tmpTile(dst_row, dst_col);

    TASSIGN<0x0>(src0Tile);
    TASSIGN<TileData_src0::Numel * sizeof(Tsrc0)>(src1Tile);
    TASSIGN<TileData_src0::Numel * sizeof(Tsrc0) + TileData_src1::Numel * sizeof(Tsrc1)>(dstTile);
    TASSIGN<(TileData_src0::Numel + TileData_dst::Numel) * sizeof(Tsrc0) + TileData_src1::Numel * sizeof(Tsrc1)>(
        tmpTile);

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER(dstTile, src0Tile, src1Tile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void test_tgather_float(__gm__ float* out, __gm__ float* src0, __gm__ int32_t* src1)
{
    runTGather<float, int32_t, 32, 864, 16, 64>(out, src0, src1);
}

extern "C" __global__ AICORE void test_tgather_int32(__gm__ int32_t* out, __gm__ int32_t* src0, __gm__ int32_t* src1)
{
    runTGather<int32_t, int32_t, 32, 512, 16, 256>(out, src0, src1);
}

extern "C" __global__ AICORE void test_tgather_half(__gm__ int16_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTGather<int16_t, int16_t, 16, 1024, 16, 128>(out, src0, src1);
}

extern "C" __global__ AICORE void test_tgather_int16(__gm__ int16_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTGather<int16_t, int16_t, 32, 256, 32, 64>(out, src0, src1);
}

void launchTGATHER_demo_float(float* out, float* src0, int32_t* src1, aclrtStream stream)
{
    test_tgather_float<<<1, nullptr, stream>>>(out, src0, src1);
}

void launchTGATHER_demo_int32(int32_t* out, int32_t* src0, int32_t* src1, aclrtStream stream)
{
    test_tgather_int32<<<1, nullptr, stream>>>(out, src0, src1);
}

void launchTGATHER_demo_half(int16_t* out, int16_t* src0, int16_t* src1, aclrtStream stream)
{
    test_tgather_half<<<1, nullptr, stream>>>(out, src0, src1);
}

void launchTGATHER_demo_int16(int16_t* out, int16_t* src0, int16_t* src1, aclrtStream stream)
{
    test_tgather_int16<<<1, nullptr, stream>>>(out, src0, src1);
}

template <typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
__global__ AICORE void runTGATHER(__gm__ dstT* out, __gm__ srcT* src)
{
    using DynShapeDim5 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = StrideDim2<kGRows_, kGCols_>;
    using SrcGlobalData = GlobalTensor<srcT, DynShapeDim5, DynStridDim5>;
    using DstGlobalData = GlobalTensor<dstT, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, srcT, (kTRows_ + 5), (kTCols_ + 32), BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, dstT, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, kTCols_);
    TASSIGN<0x0>(srcTile);
    TASSIGN<TileData::Numel * sizeof(srcT)>(dstTile);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, maskPattern>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
void LaunchTGATHER(dstT* out, srcT* src, void* stream)
{
    if constexpr (std::is_same_v<srcT, uint16_t>) {
        runTGATHER<half, half, kGRows_, kGCols_, kTRows_, kTCols_, maskPattern>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), reinterpret_cast<half*>(src));
    } else {
        runTGATHER<srcT, dstT, kGRows_, kGCols_, kTRows_, kTCols_, maskPattern><<<1, nullptr, stream>>>(out, src);
    }
}

template void
LaunchTGATHER<int8_t, int8_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
    int8_t* out, int8_t* src, void* stream);

template void
LaunchTGATHER<uint8_t, uint8_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
    uint8_t* out, uint8_t* src, void* stream);

template void
LaunchTGATHER<int8_t, int8_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
    int8_t* out, int8_t* src, void* stream);

template void
LaunchTGATHER<uint8_t, uint8_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
    uint8_t* out, uint8_t* src, void* stream);

template void
LaunchTGATHER<int8_t, int8_t, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
    int8_t* out, int8_t* src, void* stream);

template void
LaunchTGATHER<uint8_t, uint8_t, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
    uint8_t* out, uint8_t* src, void* stream);

template void
LaunchTGATHER<int8_t, int8_t, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL, MaskPattern::P1111>(
    int8_t* out, int8_t* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<int16_t, int16_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
    int16_t* out, int16_t* src, void* stream);

template void
LaunchTGATHER<int16_t, int16_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
    int16_t* out, int16_t* src, void* stream);

template void LaunchTGATHER<
    uint32_t, uint32_t, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL, MaskPattern::P0100>(
    uint32_t* out, uint32_t* src, void* stream);

template void
LaunchTGATHER<int32_t, int32_t, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL, MaskPattern::P1000>(
    int32_t* out, int32_t* src, void* stream);

template void
LaunchTGATHER<int32_t, int32_t, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
    int32_t* out, int32_t* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<half, half, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL, MaskPattern::P1111>(
    half* out, half* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<uint16_t, uint16_t, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL, MaskPattern::P1111>(
    uint16_t* out, uint16_t* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P0101_ROW, FLOAT_P0101_COL, FLOAT_P0101_ROW, FLOAT_P0101_COL, MaskPattern::P0101>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL, MaskPattern::P1010>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P0001_ROW, FLOAT_P0001_COL, FLOAT_P0001_ROW, FLOAT_P0001_COL, MaskPattern::P0001>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P0010_ROW, FLOAT_P0010_COL, FLOAT_P0010_ROW, FLOAT_P0010_COL, MaskPattern::P0010>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL, MaskPattern::P0100>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL, MaskPattern::P1000>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, float, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
    float* out, float* src, void* stream);

template void
LaunchTGATHER<float, int32_t, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL, MaskPattern::P1010>(
    int32_t* out, float* src, void* stream);
